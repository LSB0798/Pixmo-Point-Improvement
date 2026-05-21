#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抓取规划器：推理 + 指标 + 绘图 一体化脚本
适配新规则：恒为 select（无 state 字段），输出为 target_bbox_2d + side_flag + stance_flag

用法示例（保持与原脚本类似）：
python eval_only_select_all.py all \
  --model_path /path/to/model \
  --gt_set    /path/to/a.jsonl /path/to/b.jsonl ... \
  --out_dir   /path/to/out_dir \
  --iou_thr   0.5 \
  --batch_size 256
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from transformers import AutoProcessor
from vllm import LLM
from tqdm import tqdm
from PIL import Image
from vllm import SamplingParams

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 兜底用的默认 prompt（如果 dataset 里没有 messages 时使用）
# DEFAULT_SYSTEM_PROMPT = (
#     "你是充电柜指示灯状态识别助手。任务：根据提供的充电柜图片，识别两个充电仓口上方的指示灯颜色状态，按「从上到下」的仓口顺序输出，仅返回指示灯颜色，无需任何解释。注意：两个指示灯尺寸一致。请不要受到其他非仓口指示灯影响；可能会存在指示灯反光问题，因此可能存在多条亮光。只要识别指示灯本身状态，注意区分反光造成的亮光。"
#     "输出格式：{\"upper_indicator_light\": \"green|yellow|red|off|unkown\",\"lower_indicator_light\": \"green|yellow|red|off|unkown\"}。只返回一个 JSON，不要其他文字。"
# )
DEFAULT_SYSTEM_PROMPT = """
你是工业视觉质检助手。你的任务是从图片中识别充电柜黑色背板区域内上指示灯（upper_indicator_light）和下指示灯（lower_indicator_light）状态。
只允许输出 JSON，禁止输出多余文字或解释。
状态枚举：
  - "green"：指示灯发出绿光
  - "yellow"：指示灯发出黄光或者橙光
  - "red"：指示灯发出红光
  - "off"：指示灯无任何发光状态
  - "unkown"：指示灯被遮挡、模糊无法识别
规则：只判断“黑色背板区域、且位于每个电池进出口正上方”的指示灯；忽略柜体顶部的发光边缘、环境反光、屏幕/UI 叠字、以及其他非指示灯灯带的任何亮光。
若看不清/过曝/遮挡/无法定位指示灯，输出 "unkown"。
"""
DEFAULT_USER_TEXT = """
识别充电柜黑色背板区域内两条指示灯状态，仅输出 JSON：
{"upper_indicator_light": "green|yellow|red|off|unkown", "lower_indicator_light": "green|yellow|red|off|unkown"}
"""


# ====================== 基础工具 ======================
def extract_json(text: str) -> Optional[str]:
    """从文本中粗暴提取第一个合法的 JSON 对象字符串。"""
    if not isinstance(text, str):
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        cand = m.group(0).strip()
        try:
            json.loads(cand)
            return cand
        except Exception:
            pass
    cleaned = text.strip().strip("`").strip()
    m2 = re.search(r"\{[\s\S]*\}", cleaned)
    if m2:
        cand2 = m2.group(0).strip()
        try:
            json.loads(cand2)
            return cand2
        except Exception:
            pass
    return None

def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def normalize_light(s: Any) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    if s in {"green", "yellow", "red", "off", "unkown"}:
        return s
    return None
# ====================== GT JSONL 解析 ======================
def gt_image_path_from_dialog(gt_obj: Dict[str, Any]) -> Optional[str]:
    """
    从 GT JSONL 的对象中取出图片路径。
    优先 images[0]，兼容旧字段 image / img。
    """
    images = gt_obj.get("images")
    if isinstance(images, list) and images and isinstance(images[0], str):
        return images[0]
    if isinstance(gt_obj.get("image"), str):
        return gt_obj["image"]
    if isinstance(gt_obj.get("img"), str):
        return gt_obj["img"]
    return None

def parse_gt_schema_or_dialog(gt_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    尝试从 GT JSONL 的对象中解析出 GT 结构：
      - 新格式：assistant 中是 {"target_bbox_2d": [...], "side_flag": "...", "stance_flag": "..."}
      - 兼容：只要有 target_bbox_2d 就认为是有效 GT。
    """
    if not isinstance(gt_obj, dict):
        return None

    # 1) 顶层就是 schema
    if isinstance(gt_obj.get("upper_indicator_light"), list):
        return gt_obj

    # 2) 对话式：从最后一个 assistant 里抽 JSON
    msgs = gt_obj.get("messages")
    if isinstance(msgs, list) and msgs:
        for m in reversed(msgs):
            if not (isinstance(m, dict) and m.get("role") == "assistant"):
                continue
            content = m.get("content")

            if isinstance(content, str):
                js = extract_json(content)
                if js:
                    try:
                        obj = json.loads(js)
                        if isinstance(obj.get("upper_indicator_light"), str):
                            return obj
                    except Exception:
                        pass

            if isinstance(content, list):
                joined = " ".join(str(x) for x in content)
                js = extract_json(joined)
                if js:
                    try:
                        obj = json.loads(js)
                        if isinstance(obj.get("upper_indicator_light"), str):
                            return obj
                    except Exception:
                        pass

    return None

def make_sample_id(img_path: str, keep_parts: int = 5) -> str:
    """
    从完整图片路径生成一个可读且基本唯一的 sample_id，
    例如：
    /.../2x2/yungu_0_1110/1494731249/current/000025_color_1497..._1280x800.jpg
    -> 2x2_yungu_0_1110_1494731249_current_000025_color_1497..._1280x800
    """
    p = Path(os.path.normpath(img_path))
    parts = [x for x in p.parts if x not in (os.sep, "")]
    if len(parts) >= keep_parts:
        tail = parts[-keep_parts:]
    else:
        tail = parts

    if tail:
        # 最后一个用 stem（去掉扩展名）
        tail = list(tail[:-1]) + [p.stem]
    else:
        tail = [p.stem]

    # 合成，并把奇怪字符都清理掉
    s = "_".join(tail)
    s = s.replace(":", "")
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or p.stem

def load_gt_sets(gt_paths):
    """
    读取多个 jsonl，构建一个以 sample_id 为 key 的 gt_map：
      gt_map[sample_id] = {
        "img_path": <图片路径字符串>,
        "gt": <只包含 target_bbox_2d/side_flag/stance_flag 的结构>,
        "raw": <原始 json 对象>
      }

    去重规则：
      - 按完整 image_path 去重（同一张图多次出现在不同 jsonl 里只保留第一次）
      - sample_id 冲突时自动加 __dupX 后缀，避免覆盖
    """
    gt_map: Dict[str, Dict[str, Any]] = {}
    seen_img_keys = set()

    for p in gt_paths:
        with open(p, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                raw = json.loads(line)

                # 1) 拿图片路径：优先 raw["image"]，兜底用 gt_image_path_from_dialog
                img_path = raw.get("image")
                if not isinstance(img_path, str):
                    img_path = gt_image_path_from_dialog(raw)

                if not isinstance(img_path, str) or not img_path:
                    print(f"[WARN] 行 {ln} 缺少 image 字段，跳过: {p}")
                    continue

                # 用完整路径做去重 key
                img_key = os.path.normpath(img_path)
                if img_key in seen_img_keys:
                    # 真正意义上的“同一张图在多个 jsonl 里重复了”
                    print(f"[WARN] image_path 冲突，保持首次记录: {img_key} ({p}:{ln})")
                    continue
                seen_img_keys.add(img_key)

                # 2) 生成一个可读的 sample_id，用来当 key 和文件名前缀
                sample_id = make_sample_id(img_path)

                # 万一 tail 一样导致 sample_id 冲突，就自动加后缀避免覆盖
                if sample_id in gt_map:
                    base = sample_id
                    k = 2
                    while sample_id in gt_map:
                        sample_id = f"{base}__dup{k}"
                        k += 1
                    print(f"[WARN] sample_id 冲突，重命名为: {sample_id}")

                # 3) 解析 GT 结构（只要能拿到 target_bbox_2d 就行）
                gt_struct = parse_gt_schema_or_dialog(raw)
                if gt_struct is None:
                    gt_struct = {}

                gt_map[sample_id] = {
                    "img_path": img_path,
                    "gt": gt_struct,
                    "raw": raw,
                }
    print(gt_struct)
    print(f"[GT] 已加载样本数(去重后按 image_path): {len(gt_map)}")
    return gt_map

# ====================== 推理部分（从 GT 集推理） ======================
def _load_vllm_and_processor(
    model_path: str,
    max_model_len: int,
    max_num_seqs: int,
    max_images_per_prompt: int,
    gpu_memory_utilization: float = 0.80,
    dtype: str = "bfloat16",
    enforce_eager: bool = False,
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    vLLM 离线推理加载：返回 (processor, llm)
    """
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype=dtype,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        limit_mm_per_prompt={"image": max_images_per_prompt},
        mm_processor_kwargs=mm_processor_kwargs or {},
    )
    return processor, llm

def _build_fewshot_bank():
    """
    只加载一次 few-shot 示例图，避免每条样本重复 IO。
    """
    
    example_image_paths = [
        "./light_data/lightVideo/green_image/20251219-175455/frame_000150.jpg",
        "./light_data/lightVideo/go_image/A5_green_25/frame_000000.jpg",
        "./light_data/lightVideo/og_image/A5_3/frame_000000.jpg",
        "./light_data/lightVideo/orange_image/A5_orange_25/frame_000000.jpg",
        "./light_data/lightVideo/og_image_val/A5_8/frame_000000.jpg",
        "./light_data/train/off_off/20251224-114316/frame_001110.jpg",
        "./light_data/train/off_unkown/20251224-102019/frame_001110.jpg",
        "./light_data/train/unkown_red/20251224-150357/frame_000675.jpg",
    ]
    example_image_gt = [
        ["green", "green"],
        ["green", "yellow"],
        ["yellow", "green"],
        ["yellow", "yellow"],
        ["yellow", "green"],
        ["off", "off"],
        ["off", "unkown"],
        ["unkown", "red"],
    ]

    bank = []
    for i, p in enumerate(example_image_paths):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"[WARN] few-shot 示例图加载失败: {p} err={e}")
            continue
        bank.append({
            "path": p,
            "img": img,
            "gt_up": example_image_gt[i][0],
            "gt_low": example_image_gt[i][1],
            "uuid": f"fewshot-{i+1}",  # 稳定 id（可用于 multimodal cache）
        })
    return bank

def _infer_batch_from_gt_records_vllm(
    processor,
    llm,
    records: List[Tuple[str, str, Dict[str, Any]]],  # (stem, img_path, raw)
    fewshot_bank: Optional[List[Dict[str, Any]]] = None,
    use_fewshot: bool = True,
    max_new_tokens: int = 128,
) -> List[Dict[str, Any]]:
    """
    vLLM 批量推理：每条样本 -> 一个 llm_input dict
    """

    if not records:
        return []

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    final_results: List[Dict[str, Any]] = [{"_error": "not_run"} for _ in records]
    llm_inputs = []
    index_map = []  # llm_outputs 的 idx -> records idx

    fewshot_bank = fewshot_bank or []

    for ridx, (stem, img_path, raw) in enumerate(records):
        try:
            query_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            final_results[ridx] = {"_error": f"failed_to_load_image: {e}"}
            continue

        # 构造 interleaved 内容：image/text/image/text/.../image/text
        content = []
        if use_fewshot and fewshot_bank:
            for k, ex in enumerate(fewshot_bank, 1):
                content.append({"type": "image", "image": ex["path"]})
                content.append({
                    "type": "text",
                    "text": (
                        f"\n示例{k} 标注输出："
                        f'{{"upper_indicator_light":"{ex["gt_up"]}","lower_indicator_light":"{ex["gt_low"]}"}}\n'
                    )
                })

        content.append({"type": "image", "image": img_path})
        content.append({
            "type": "text",
            "text": DEFAULT_USER_TEXT
        })

        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # multi_modal_data 的 images 顺序必须和 prompt 中 image placeholder 出现顺序一致
        images = []
        uuids = []
        if use_fewshot and fewshot_bank:
            for ex in fewshot_bank:
                images.append(ex["img"])
                uuids.append(ex["uuid"])
        images.append(query_img)
        uuids.append(stem)  # 每条样本独立

        llm_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": images},
            "multi_modal_uuids": {"image": uuids},  # 可减少重复 hashing 开销:contentReference[oaicite:3]{index=3}
        })
        index_map.append(ridx)

    if not llm_inputs:
        return final_results

    try:
        outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
    except Exception as e:
        # 整个 batch 失败，回填错误
        for ridx in index_map:
            final_results[ridx] = {"_error": f"vllm_generate_failed: {e}"}
        return final_results

    # 逐条解析输出
    for out, ridx in zip(outputs, index_map):
        text_out = (out.outputs[0].text or "").strip()
        js = extract_json(text_out)
        if js is None:
            final_results[ridx] = {"_raw": text_out, "_error": "no_json_found"}
        else:
            try:
                final_results[ridx] = json.loads(js)
            except Exception as e:
                final_results[ridx] = {"_raw": text_out, "_error": f"json_load_error: {e}"}

    return final_results

def run_infer_from_gtset(
    model_path: str,
    gt_map: Dict[str, Dict[str, Any]],
    out_dir: str,
    max_new_tokens: int = 128,
    batch_size: int = 32,
):

    os.makedirs(out_dir, exist_ok=True)

    # few-shot 只加载一次
    fewshot_bank = _build_fewshot_bank()
    max_images_per_prompt = (len(fewshot_bank) + 5) if fewshot_bank else 1  # 关键！:contentReference[oaicite:4]{index=4}

    processor, llm = _load_vllm_and_processor(
        model_path=model_path,
        max_model_len=20000,
        max_num_seqs=max(1, batch_size),
        max_images_per_prompt=max_images_per_prompt,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
        enforce_eager=False,
        mm_processor_kwargs={"max_dynamic_patch": 4},  # 常见的提速配置（可按需调）
    )

    samples: List[Tuple[str, str, Dict[str, Any]]] = []
    for stem, rec in sorted(gt_map.items()):
        img_path = rec.get("img_path")
        raw = rec.get("raw") if isinstance(rec.get("raw"), dict) else {}
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            print(f"[WARN] 跳过（找不到图像）: {img_path}")
            continue
        samples.append((stem, img_path, raw))

    with tqdm(total=len(samples), desc="Infer[vLLM]") as pbar:
        for i in range(0, len(samples), batch_size):
            batch_records = samples[i:i + batch_size]
            batch_stems = [r[0] for r in batch_records]

            results = _infer_batch_from_gt_records_vllm(
                processor=processor,
                llm=llm,
                records=batch_records,
                fewshot_bank=fewshot_bank,
                use_fewshot=bool(fewshot_bank),
                max_new_tokens=max_new_tokens,
            )

            for stem, result in zip(batch_stems, results):
                out_path = os.path.join(out_dir, f"{stem}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

            pbar.update(len(batch_records))

    print(f"[Infer] Done -> {out_dir}")

# ====================== 评估部分（bbox + side + stance） ======================
def eval_with_gtset(pred_dir: Path, gt_map: Dict[str, Dict[str, Any]], iou_thr: float = 0.5):
    n_total = 0
    n_valid_gt = 0
    invalid_gt = 0
    invalid_pred = 0

    up_correct = 0      
    low_correct = 0        
    all_correct = 0    

    per_image_rows: List[Dict[str, Any]] = []

    json_parse_ok_count = 0
    json_parse_fail = 0

    for stem, rec in sorted(gt_map.items()):
        n_total += 1
        gt = rec.get("gt")
        img_path = rec.get("img_path")

        row: Dict[str, Any] = {
            "stem": stem,
            "img_path": img_path,
        }

        pred = read_json(pred_dir / f"{stem}.json")

        # JSON 解析健康度
        json_parse_ok = isinstance(pred, dict)
        row["json_parse_ok"] = json_parse_ok
        if not json_parse_ok:
            json_parse_fail += 1
            row["err"] = "json_parse_fail"
            per_image_rows.append(row)
            continue
        else:
            json_parse_ok_count += 1


        gt_status = None
        if isinstance(gt, dict):
            gb = gt.get("upper_indicator_light")
            if gb is not None:
                gb = gt.get("lower_indicator_light")
                if gb is not None:
                    gt_status = True


        pr_status = None
        if isinstance(pred, dict):
            pb = pred.get("upper_indicator_light")
            if pb is not None:
                pb = pred.get("lower_indicator_light")
                if pb is not None:
                    pr_status = True


        if gt_status is None:
            invalid_gt += 1
            row["err"] = "invalid_gt"
            per_image_rows.append(row)
            continue

        if pr_status is None:
            invalid_pred += 1
            row["err"] = "invalid_pred"
            per_image_rows.append(row)
            continue

        # 到这里 GT / Pred 都有 bbox，计入有效样本
        n_valid_gt += 1

        gt_up = normalize_light(gt.get("upper_indicator_light") if isinstance(gt, dict) else None)
        gt_low = normalize_light(gt.get("lower_indicator_light") if isinstance(gt, dict) else None)
        pr_up = normalize_light(pred.get("upper_indicator_light"))
        pr_low = normalize_light(pred.get("lower_indicator_light"))

        # row["gt_up"] = gt_side
        # row["pred_up"] = pr_side
        # row["gt_lower"] = gt_stance
        # row["pred_lower"] = pr_stance

        up_ok = False
        low_ok = False
        all_ok = False


        if gt_up is not None and pr_up is not None and gt_up == pr_up:
            up_ok = True
            up_correct += 1
        if gt_low is not None and pr_low is not None and gt_low == pr_low:
            low_ok = True
            low_correct += 1
        if up_ok and low_ok:
            all_ok = True
            all_correct += 1

        row["up_correct"] = up_ok
        row["low_correct"] = low_ok
        row["all_correct"] = all_ok

        per_image_rows.append(row)

    denom = n_valid_gt if n_valid_gt > 0 else 1

    select_up_acc = up_correct / denom
    select_low_acc = low_correct / denom
    select_all_acc = all_correct / denom

    metrics = {
        "总样本数": n_total,
        "有效样本数(有GT/Pred bbox)": n_valid_gt,
        "无效GT数": invalid_gt,
        "无效预测数": invalid_pred,

        "上指示灯准确率": round(select_up_acc, 4),
        "下指示灯准确率": round(select_low_acc, 4),
        "总准确率": round(select_all_acc, 4),

    }

    metrics.update({
        "JSON可解析样本数": json_parse_ok_count,
        "JSON解析失败样本数": json_parse_fail,
        "JSON解析成功率": round(json_parse_ok_count / (n_total if n_total else 1), 4)
    })

    return metrics, per_image_rows

def save_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    import csv
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ====================== 可视化 ======================
def try_import_cv2():
    try:
        import cv2
        return cv2
    except Exception:
        return None


def draw_box_cv2(img, x1, y1, x2, y2, color, label: Optional[str] = None):
    cv2 = try_import_cv2()
    if cv2 is None:
        return img
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    if label:
        cv2.putText(
            img,
            label,
            (int(x1), max(0, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return img

def draw_gt_pred(img, gt, pred):
    cv2 = try_import_cv2()
    if cv2 is None:
        return img
    gt_color = (0, 255, 0)    # GT - 绿色
    pred_color = (0, 0, 255)  # Pred - 红色
    if gt:
        cv2.putText(
            img,
            f"Gt   :[ up_light: {gt[0]}, low_light: {gt[1]} ]",
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            gt_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"Pred :[ up_light: {pred[0]}, low_light: {pred[1]} ]",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            pred_color,
            2,
            cv2.LINE_AA,
        )
    return img


def load_image_cv2(path: str):
    cv2 = try_import_cv2()
    if cv2 is None:
        return None
    return cv2.imread(path, cv2.IMREAD_COLOR)


def save_image_cv2(path: str, img):
    cv2 = try_import_cv2()
    if cv2 is None:
        return False
    return cv2.imwrite(path, img)


def visualize_annotations_from_gtset(
    pred_dir: str,
    gt_map: Dict[str, Dict[str, Any]],
    save_dir: str,
    iou_thr: float = 0.5,
):
    cv2 = try_import_cv2()
    if cv2 is None:
        print("[WARN] 未安装 OpenCV，跳过可视化。")
        return

    pred_p = Path(pred_dir)
    save_p = Path(save_dir)
    true_p = save_p / "true"
    false_p = save_p / "false"
    true_p.mkdir(parents=True, exist_ok=True)
    false_p.mkdir(parents=True, exist_ok=True)

    for stem, rec in sorted(gt_map.items()):
        img_path = rec.get("img_path")
        gt = rec.get("gt")
        pred = read_json(pred_p / f"{stem}.json")

        if not isinstance(img_path, str) or not os.path.exists(img_path):
            print(f"[WARN] 找不到图像: {img_path} (stem={stem})")
            continue

        img = load_image_cv2(img_path)
        if img is None:
            print(f"[WARN] 图像读取失败: {img_path}")
            continue

        # 先假设是 false，稍后如果判断三者都对再改为 true 目录
        out_dir_this = false_p


        gt_up = normalize_light(gt.get("upper_indicator_light") if isinstance(gt, dict) else None)
        gt_low = normalize_light(gt.get("lower_indicator_light") if isinstance(gt, dict) else None)
        pr_up = normalize_light(pred.get("upper_indicator_light") if isinstance(pred, dict) else None)
        pr_low = normalize_light(pred.get("lower_indicator_light") if isinstance(pred, dict) else None)

        img = draw_gt_pred(img, (gt_up, gt_low), (pr_up, pr_low))
        # 判定 true / false：
        all_correct = False
        if (
            gt_up is not None
            and gt_low is not None
            and pr_up is not None
            and pr_low is not None
            and gt_up == pr_up
            and gt_low == pr_low
        ):
            all_correct = True

        out_dir_this = true_p if all_correct else false_p

        out_path = out_dir_this / f"{stem}_vis.png"
        save_image_cv2(str(out_path), img)
        print(f"[VIS] {'TRUE ' if all_correct else 'FALSE'}保存: {out_path}")


# ====================== CLI ======================
def main():
    parser = argparse.ArgumentParser(
        description="抓取规划器：推理+评估+可视化 一体化脚本（适配新规则：bbox+side+stance）"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # infer：仅推理（从 GT JSONL 中读图片路径和 prompt）
    p_infer = sub.add_parser("infer", help="仅推理：从 --gt_set 输入，输出到 --out_dir")
    p_infer.add_argument("--gt_set", type=str, nargs="+", required=True, help="一个或多个 GT JSONL（从中读取图片路径与prompt）")
    p_infer.add_argument("--model_path", type=str, required=True)
    p_infer.add_argument("--out_dir", type=str, required=True, help="预测 JSON 输出目录")
    p_infer.add_argument("--max_new_tokens", type=int, default=128)
    p_infer.add_argument("--batch_size", type=int, default=1, help="批量推理的批大小（默认1）")

    # eval：只评估（需要已经有 pred_dir）
    p_eval = sub.add_parser("eval", help="仅评估：需要 --gt_set；指标写到 --out_dir/metrics")
    p_eval.add_argument("--pred_dir", type=str, required=True, help="预测 JSON 目录")
    p_eval.add_argument("--gt_set", type=str, nargs="+", required=True, help="一个或多个 GT JSONL")
    p_eval.add_argument("--out_dir", type=str, required=True, help="基准输出目录（将创建 metrics 子目录）")
    p_eval.add_argument("--iou_thr", type=float, default=0.5)

    # vis：只可视化（基于 GT + Pred），图写到 --out_dir/vis/{true,false}
    p_vis = sub.add_parser("vis", help="仅可视化：需要 --gt_set；图写到 --out_dir/vis/{true,false}")
    p_vis.add_argument("--pred_dir", type=str, required=True, help="预测 JSON 目录")
    p_vis.add_argument("--gt_set", type=str, nargs="+", required=True, help="一个或多个 GT JSONL")
    p_vis.add_argument("--out_dir", type=str, required=True, help="基准输出目录（将创建 vis 子目录）")
    p_vis.add_argument("--iou_thr", type=float, default=0.5)

    # all：推理(从GT)->评估->可视化
    p_all = sub.add_parser("all", help="全流程：推理(从GT)->评估->可视化。全部输出放在 --out_dir 下的子目录。")
    p_all.add_argument("--model_path", type=str, required=True)
    p_all.add_argument("--gt_set", type=str, nargs="+", required=True, help="一个或多个 GT JSONL（用于推理、评估、可视化）")
    p_all.add_argument("--out_dir", type=str, required=True, help="运行输出根目录（将创建 preds/metrics/vis）")
    p_all.add_argument("--max_new_tokens", type=int, default=128)
    p_all.add_argument("--iou_thr", type=float, default=0.5)
    p_all.add_argument("--batch_size", type=int, default=1, help="批量推理的批大小（默认1）")

    args = parser.parse_args()

    if args.cmd == "infer":
        gt_map = load_gt_sets(args.gt_set)
        if not gt_map:
            raise SystemExit("未能从 --gt_set 读取到任何样本")
        run_infer_from_gtset(
            args.model_path,
            gt_map,
            args.out_dir,
            args.max_new_tokens,
            args.batch_size,
        )

    elif args.cmd == "eval":
        gt_map = load_gt_sets(args.gt_set)
        if not gt_map:
            raise SystemExit("未能从 --gt_set 读取到任何样本")
        metrics, per_image_rows = eval_with_gtset(pred_dir=Path(args.pred_dir), gt_map=gt_map, iou_thr=args.iou_thr)
        mdir = Path(args.out_dir) / "metrics"
        mdir.mkdir(parents=True, exist_ok=True)
        save_json(metrics, mdir / "metrics.json")
        save_csv(per_image_rows, mdir / "per_image.csv")
        print(f"[OK] 指标已保存到: {mdir}")
        print("\n=== 评估结果 ===")
        print(f"上指示灯准确率: {metrics['上指示灯准确率']:.2%}")
        print(f"下指示灯准确率: {metrics['下指示灯准确率']:.2%}")
        print(f"总准确率: {metrics['总准确率']:.2%}")

    elif args.cmd == "vis":
        gt_map = load_gt_sets(args.gt_set)
        if not gt_map:
            raise SystemExit("未能从 --gt_set 读取到任何样本")
        vdir = Path(args.out_dir) / "vis"
        vdir.mkdir(parents=True, exist_ok=True)
        visualize_annotations_from_gtset(args.pred_dir, gt_map, str(vdir), iou_thr=args.iou_thr)
        print(f"[OK] 可视化已保存到: {vdir}")

    elif args.cmd == "all":
        out_root = Path(args.out_dir)
        preds_dir = out_root / "preds"
        metrics_dir = out_root / "metrics"
        vis_dir = out_root / "vis"
        preds_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

        # 1) 推理
        gt_map = load_gt_sets(args.gt_set)
        if not gt_map:
            raise SystemExit("未能从 --gt_set 读取到任何样本")
        run_infer_from_gtset(
            args.model_path,
            gt_map,
            str(preds_dir),
            args.max_new_tokens,
            args.batch_size,
        )

        # 2) 评估
        metrics, per_image_rows = eval_with_gtset(pred_dir=preds_dir, gt_map=gt_map, iou_thr=args.iou_thr)
        save_json(metrics, metrics_dir / "metrics.json")
        save_csv(per_image_rows, metrics_dir / "per_image.csv")
        print(f"[ALL] 指标已保存到: {metrics_dir}")
        print("\n=== 评估结果 ===")
        print(f"上指示灯准确率: {metrics['上指示灯准确率']:.2%}")
        print(f"下指示灯准确率: {metrics['下指示灯准确率']:.2%}")
        print(f"总准确率: {metrics['总准确率']:.2%}")

        # 3) 可视化
        visualize_annotations_from_gtset(str(preds_dir), gt_map, str(vis_dir), iou_thr=args.iou_thr)
        print(f"[ALL] 可视化已保存到: {vis_dir}")


if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=7 python light_sh/light_eval_example_yjp.py all \
# --model_path /ogi-data/pzb-data/model/Qwen3-VL-4B-Instruct \
# --gt_set  ./light_data/labels/green_green_val.jsonl \
# ./light_data/labels/green_yellow_val.jsonl \
# ./light_data/labels/yellow_yellow_val.jsonl \
# ./light_data/labels/yellow_green_val.jsonl \
# ./light_data/labels1225/off_off_val.jsonl \
# ./light_data/labels1225/off_unkown_val.jsonl \
# ./light_data/labels1225/red_red_val.jsonl \
# ./light_data/labels1225/unkown_off_val.jsonl \
# ./light_data/labels1225/unkown_red_val.jsonl \
# ./light_data/labels1225/unkown_unkown_val.jsonl \
# ./light_data/labels1225/unkown_yellow_val.jsonl \
# ./light_data/labels1225/yellow_unkown_val.jsonl \
# ./light_data/labels1225/yellow_yellow_val.jsonl \
# --out_dir    light_results_yjp/basemodel-260115-imageprompt \
# --iou_thr    0.5 \
# --batch_size 32