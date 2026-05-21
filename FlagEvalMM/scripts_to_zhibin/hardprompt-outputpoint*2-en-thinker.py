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
import ast
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from transformers import AutoProcessor
from vllm import LLM
from tqdm import tqdm
from PIL import Image
from vllm import SamplingParams
from collections import Counter

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_SYSTEM_PROMPT = """Please point out the location of the indicator lights in the image. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should range from 0 to 1000, representing the relative pixel positions of the points in the image.
"""

PROMPT_UP = """Please point out the upper indicator lights on the black center back panel. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should range from 0 to 1000, representing the relative pixel positions of the points in the image."""

PROMPT_LOW = """Please point out the lower indicator lights on the black center back panel. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should range from 0 to 1000, representing the relative pixel positions of the points in the image."""

PROMPT_BOTH = """
A) Prefer those that best match “about 30 cm long and about 4 cm wide”, and located directly above the bay openings / battery inlet-outlets;
- Ignore light strips that are clearly longer, thinner, intermittent, or deviating from the core area of the backboard.
B) For an occluded indicator light, return E. unknown.
C) The following items must be excluded as targets no matter how salient they are, and must NOT be treated as indicator lights:
- QR codes / barcodes / text labels (usually square or with grid/text textures)
- Red buttons / emergency stops / round knobs (circular or small-area blocks)
- Charging ports / jacks / USB ports / round holes / rectangular holes (hole-like shapes)
D) Do not recognize reflections as indicator lights. Reflections usually appear as localized glare/flare or are offset from the light body; only count it as lit when the light body itself shows a stable, visible luminous area.
"""

STATES = ["green", "yellow", "red", "off", "unknown"]
ON_COLORS = {"green", "yellow", "red"}

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
    if s in {"green", "yellow", "red", "off", "unknown"}:
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
      - 新格式：assistant 中是 {"upper_indicator_light": [...], "lower_indicator_light": "..."}
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
        "gt": <只包含 green yellow red off unknown 的结构>,
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

def _normalize_label(x: Any) -> str:
    """把模型输出尽量归一到 {'green','yellow','red','off','unknown'}"""
    if x is None:
        return "unknown"
    s = str(x).strip().lower()

    # 常见同义/中英文映射
    if s in {"unknown", "unkown", "?", "n/a", "na", "none", "null", "uncertain", "unclear", "不确定", "未知", "看不清"}:
        return "unknown"
    if s in {"green", "g", "绿", "绿色"}:
        return "green"
    if s in {"yellow", "amber", "y", "黄", "黄色"}:
        return "yellow"
    if s in {"red", "r", "红", "红色"}:
        return "red"
    if s in {"off", "dark", "black", "灭", "熄灭", "不亮", "无灯"}:
        return "off"

    # 兜底：如果刚好是允许集合就返回，否则 unknown
    return s if s in STATES else "unknown"

# ====================== 推理部分（从 GT 集推理） ======================
def _load_vllm_and_processor(
    model_path: str,
    max_model_len: int,
    max_num_seqs: int,
    max_images_per_prompt: int,
    gpu_memory_utilization: float = 0.50,
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
        limit_mm_per_prompt={"image": max_images_per_prompt, "video": 0},
        mm_processor_kwargs=mm_processor_kwargs or {},
    )
    return processor, llm

def _build_fewshot_bank():
    """
    只加载一次 few-shot 示例图，避免每条样本重复 IO。
    """
    
    example_image_paths = [
        # "./light_data/few_shot2/ex_green_green.png",
        # "./light_data/few_shot2/ex_green_yellow.png",
        # "./light_data/few_shot2/ex_yellow_green.png",
        # "./light_data/few_shot2/ex_yellow_yellow.png",
        # "./light_data/few_shot2/ex_yellow_green2.png",
        # "./light_data/few_shot2/ex_off_off.png",
        # "./light_data/few_shot2/ex_off_unkown.png",
        # "./light_data/few_shot2/ex_unkown_red.png",
    ]
    example_image_gt = [
        # ["green", "green"],
        # ["green", "yellow"],
        # ["yellow", "green"],
        # ["yellow", "yellow"],
        # ["yellow", "green"],
        # ["off", "off"],
        # ["off", "unknown"],
        # ["unknown", "red"],
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

def _collect_images_and_uuids_from_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Image.Image], List[Optional[str]]]:
    """按照 messages 中 image 出现顺序收集 PIL.Image 和 uuid（uuid 可选）"""
    images: List[Image.Image] = []
    uuids: List[Optional[str]] = []

    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    img = item.get("image")
                    uid = item.get("uuid")

                    if isinstance(img, Image.Image):
                        images.append(img)
                        uuids.append(uid if isinstance(uid, str) else None)
                    elif isinstance(img, str):
                        # 允许传路径（不推荐：会产生IO）
                        try:
                            pil = Image.open(img).convert("RGB")
                            images.append(pil)
                            uuids.append(uid if isinstance(uid, str) else None)
                        except Exception:
                            # 路径打不开就跳过（但这会导致 prompt 里的 image token 数和 images 数不一致 -> 可能报错）
                            raise RuntimeError(f"Failed to open image path in messages: {img}")

    return images, uuids

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
    metas: List[Dict[str, Any]] = []
    index_map = []  # llm_outputs 的 idx -> records idx

    fewshot_bank = fewshot_bank or []

    for ridx, (stem, img_path, raw) in enumerate(records):
        try:
            query_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            final_results[ridx] = {"_error": f"failed_to_load_image: {e}"}
            continue

        for which, user_text in (("up", PROMPT_UP), ("low", PROMPT_LOW)):
            # 3) 组装 messages（few-shot + query）
            messages: List[Dict[str, Any]] = []

            query_content = [{"type": "image", "image": query_img, "uuid": f"{stem}-main"}]
            query_content.append({
                "type": "text",
                "text": user_text + PROMPT_BOTH + "\nYour answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should range from 0 to 1000, representing the relative pixel positions of the points in the image.",
            })
            messages.append({"role": "user", "content": query_content})

            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            mm_images, mm_uuids = _collect_images_and_uuids_from_messages(messages)

            llm_inp: Dict[str, Any] = {
                "prompt": prompt,
                "multi_modal_data": {"image": mm_images},
            }

            llm_inputs.append(llm_inp)
            metas.append({
                "ridx": ridx,      # << 新增：回填用
                "stem": stem,
                "img_path": img_path,
                "n_images": len(mm_images),
                "which": which,    # << 新增：up/low
            })


    # 过滤掉打开失败占位 None，但要保持顺序对齐：这里用 index 映射回填
    valid_pairs = [(idx, inp) for idx, inp in enumerate(llm_inputs) if isinstance(inp, dict)]
    valid_indices = [p[0] for p in valid_pairs]
    valid_inputs = [p[1] for p in valid_pairs]

    outputs_text: List[Optional[str]] = [None] * len(llm_inputs)
    if valid_inputs:
        outputs = llm.generate(valid_inputs, sampling_params=sampling_params)
        # outputs 的顺序与 valid_inputs 对齐
        for local_i, out in enumerate(outputs):
            try:
                outputs_text[valid_indices[local_i]] = out.outputs[0].text
            except Exception:
                outputs_text[valid_indices[local_i]] = ""

    # 解析输出（每次推理一条）
    partials: List[Dict[str, Any]] = []

    for meta, text in zip(metas, outputs_text):
        raw_text = (text or "").strip()

        obj_str = extract_json_any(raw_text)
        obj = json.loads(obj_str) if obj_str else None

        pred_points: List[List[float]] = []
        if obj is not None:
            pred_points = get_points_from_any_obj(obj)
        if not pred_points:
            pred_points = extract_points_from_text(raw_text)

        partials.append({
            **meta,
            "raw_text": raw_text,
            "pred_points": pred_points,
            "point_parse_ok": bool(pred_points),
        })

    # 聚合：两次推理合成一条（回填到 final_results，保持 records 顺序 & 保留加载失败）
    by_ridx: Dict[int, Dict[str, Any]] = {}

    for r in partials:
        ridx = r["ridx"]
        agg = by_ridx.get(ridx)
        if agg is None:
            agg = {
                "stem": r.get("stem"),
                "img_path": r.get("img_path"),
                "n_images": r.get("n_images"),

                "pred_points_up": [],
                "pred_points_low": [],
                "raw_text_up": "",
                "raw_text_low": "",
                "point_parse_ok_up": False,
                "point_parse_ok_low": False,
            }
            by_ridx[ridx] = agg

        if r.get("which") == "up":
            agg["pred_points_up"] = r.get("pred_points", [])
            agg["raw_text_up"] = r.get("raw_text", "")
            agg["point_parse_ok_up"] = bool(r.get("point_parse_ok"))
        elif r.get("which") == "low":
            agg["pred_points_low"] = r.get("pred_points", [])
            agg["raw_text_low"] = r.get("raw_text", "")
            agg["point_parse_ok_low"] = bool(r.get("point_parse_ok"))

    # 兼容你原来的可视化：提供 pred_points = up + low
    for ridx, agg in by_ridx.items():
        agg["pred_points"] = (agg["pred_points_up"] or []) + (agg["pred_points_low"] or [])
        agg["point_parse_ok"] = agg["point_parse_ok_up"] and agg["point_parse_ok_low"]
        final_results[ridx] = agg

    return final_results


def run_infer_from_gtset(
    model_path: str,
    gt_map: Dict[str, Dict[str, Any]],
    out_dir: str,
    max_new_tokens: int = 128,
    batch_size: int = 32,
    gpu_memory_utilization: float = 0.50,
):

    os.makedirs(out_dir, exist_ok=True)

    # few-shot 只加载一次
    fewshot_bank = _build_fewshot_bank()
    max_images_per_prompt = (len(fewshot_bank) + 5) if fewshot_bank else 5

    processor, llm = _load_vllm_and_processor(
        model_path=model_path,
        max_model_len=20000,
        max_num_seqs=max(1, batch_size),
        max_images_per_prompt=max_images_per_prompt,
        gpu_memory_utilization=gpu_memory_utilization,
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

# ====================== 评估部分======================
def classify_error(gt: Optional[str], pr: Optional[str]) -> str:
    if gt is None or pr is None:
        return "invalid_label"
    if gt == pr:
        return "correct"
    if gt in ON_COLORS and pr in ON_COLORS:
        return "color_mismatch"          # 绿/黄/红互相错
    if gt == "off" and pr in ON_COLORS:
        return "off_as_on"               # off -> 有灯
    if gt == "unknown" and pr in ON_COLORS:
        return "unknown_as_on"            # unknown -> 有灯
    if gt in ON_COLORS and pr == "off":
        return "on_as_off"               # 有灯 -> off
    if gt in ON_COLORS and pr == "unknown":
        return "on_as_unknown"            # 有灯 -> unknown
    if gt == "off" and pr == "unknown":
        return "off_as_unknown"
    if gt == "unknown" and pr == "off":
        return "unknown_as_off"
    return f"{gt}_as_{pr}"

def init_confusion() -> Dict[str, Dict[str, int]]:
    return {g: {p: 0 for p in STATES} for g in STATES}

def eval_with_gtset(pred_dir: Path, gt_map: Dict[str, Dict[str, Any]], iou_thr: float = 0.5):
    n_total = 0
    n_valid = 0
    invalid_gt = 0
    invalid_pred = 0

    up_correct = 0
    low_correct = 0
    all_correct = 0

    per_image_rows: List[Dict[str, Any]] = []

    json_parse_ok_count = 0
    json_parse_fail = 0

    err_upper = Counter()
    err_lower = Counter()
    err_overall = Counter()
    conf_upper = init_confusion()
    conf_lower = init_confusion()

    for stem, rec in sorted(gt_map.items()):
        n_total += 1
        gt = rec.get("gt")
        img_path = rec.get("img_path")

        row: Dict[str, Any] = {
            "stem": stem,
            "img_path": img_path,
        }

        pred = read_json(pred_dir / f"{stem}.json")

        json_parse_ok = isinstance(pred, dict)
        row["json_parse_ok"] = json_parse_ok
        if not json_parse_ok:
            json_parse_fail += 1
            row["err"] = "json_parse_fail"
            per_image_rows.append(row)
            continue
        json_parse_ok_count += 1

        # 字段是否存在
        gt_ok = isinstance(gt, dict) and ("upper_indicator_light" in gt) and ("lower_indicator_light" in gt)
        pr_ok = isinstance(pred, dict) and ("pred_up" in pred) and ("pred_low" in pred)

        if not gt_ok:
            invalid_gt += 1
            row["err"] = "invalid_gt"
            per_image_rows.append(row)
            continue
        if not pr_ok:
            invalid_pred += 1
            row["err"] = "invalid_pred"
            per_image_rows.append(row)
            continue

        n_valid += 1

        gt_up = normalize_light(gt.get("upper_indicator_light"))
        gt_low = normalize_light(gt.get("lower_indicator_light"))
        pr_up = normalize_light(pred.get("pred_up"))
        pr_low = normalize_light(pred.get("pred_low"))

        row["gt_up"] = gt_up
        row["gt_low"] = gt_low
        row["pred_up"] = pr_up
        row["pred_low"] = pr_low

        # confusion（仅统计合法 label）
        if gt_up in STATES and pr_up in STATES:
            conf_upper[gt_up][pr_up] += 1
        if gt_low in STATES and pr_low in STATES:
            conf_lower[gt_low][pr_low] += 1

        # error type
        eu = classify_error(gt_up, pr_up)
        el = classify_error(gt_low, pr_low)
        row["err_up_type"] = eu
        row["err_low_type"] = el
        err_upper[eu] += 1
        err_lower[el] += 1
        err_overall[eu] += 1
        err_overall[el] += 1

        up_ok = (gt_up is not None and pr_up is not None and gt_up == pr_up)
        low_ok = (gt_low is not None and pr_low is not None and gt_low == pr_low)

        if up_ok:
            up_correct += 1
        if low_ok:
            low_correct += 1
        if up_ok and low_ok:
            all_correct += 1

        row["up_correct"] = up_ok
        row["low_correct"] = low_ok
        row["all_correct"] = (up_ok and low_ok)

        per_image_rows.append(row)

    denom = n_valid if n_valid > 0 else 1

    metrics = {
        "总样本数": n_total,
        "有效样本数(有GT/Pred字段)": n_valid,
        "无效GT数": invalid_gt,
        "无效预测数": invalid_pred,

        "上指示灯准确率": round(up_correct / denom, 4),
        "下指示灯准确率": round(low_correct / denom, 4),
        "总准确率": round(all_correct / denom, 4),

        "JSON可解析样本数": json_parse_ok_count,
        "JSON解析失败样本数": json_parse_fail,
        "JSON解析成功率": round(json_parse_ok_count / (n_total if n_total else 1), 4),

        "错误统计": {
            "upper": dict(err_upper),
            "lower": dict(err_lower),
            "overall(upper+lower累计)": dict(err_overall),
        },
        "混淆矩阵": {
            "upper": conf_upper,
            "lower": conf_lower,
        },
    }

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

    gt_color = (0, 255, 0)    # GT - 绿色 (BGR)
    pred_color = (0, 0, 255)  # Pred - 红色 (BGR)

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

        h, w = img.shape[:2]

        gt_points = get_points_from_any_obj(gt)

        pred_points: List[List[float]] = []
        if isinstance(pred, dict):
            # (a) 优先用 pred_points
            if isinstance(pred.get("pred_points"), list):
                pred_points = _as_point_list(pred.get("pred_points"))  # 归一化一下
            # (b) 没有就从 pred dict 里抽
            if not pred_points:
                pred_points = get_points_from_any_obj(pred)
            # (c) 还没有就从 raw_text 抽（兼容 JSON 或 [(x,y),...]）
            if not pred_points and isinstance(pred.get("raw_text"), str):
                pred_points = extract_points_from_text(pred["raw_text"])

        # ---------------------------
        # 3) 画框：GT 绿，Pred 红
        # ---------------------------
        for i, pt in enumerate(gt_points):
            try:
                x, y = point_to_pixel_xy(pt, w, h)
                img = draw_point_cv2(img, x, y, gt_color, label=f"GT[{i}]")
            except Exception as e:
                print(f"[WARN] GT point 绘制失败: stem={stem} pt={pt} err={e}")

        for i, pt in enumerate(pred_points):
            try:
                x, y = point_to_pixel_xy(pt, w, h)
                img = draw_point_cv2(img, x, y, pred_color, label=f"Pred[{i}]")
            except Exception as e:
                print(f"[WARN] Pred point 绘制失败: stem={stem} pt={pt} err={e}")


        # ---------------------------
        # 4) 可选：保留你原来的文字 overlay（如果还需要）
        #    注意：你原代码这里读 pred 的 key 写错了（用的是 upper_indicator_light），
        #    我这里给你按 pred_up/pred_low 修正一下（有则显示，无则略过）
        # ---------------------------
        if isinstance(gt, dict) and isinstance(pred, dict):
            gt_up = normalize_light(gt.get("upper_indicator_light"))
            gt_low = normalize_light(gt.get("lower_indicator_light"))
            pr_up = normalize_light(pred.get("pred_up"))
            pr_low = normalize_light(pred.get("pred_low"))
            img = draw_gt_pred(img, (gt_up, gt_low), (pr_up, pr_low))

        # ---------------------------
        # 5) true/false 分桶：如果你现在主要看 bbox，就先按“是否存在 pred bbox”粗分
        #    你后续如果要按 IoU 分 true/false，再把这里改成 IoU 判定即可
        # ---------------------------
        all_correct = bool(pred_points)  # 先粗暴：有点就算 true（你可以改成 IoU/匹配逻辑）
        out_dir_this = true_p if all_correct else false_p

        out_path = out_dir_this / f"{stem}_vis.png"
        save_image_cv2(str(out_path), img)
        print(f"[VIS] {'TRUE ' if all_correct else 'FALSE'}保存: {out_path}")


def extract_json_any(text: str) -> Optional[str]:
    """
    从文本中粗暴提取第一个合法 JSON（支持 {} 或 []）。
    """
    if not isinstance(text, str):
        return None

    # 先试图找列表 [...]
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        cand = m.group(0).strip()
        try:
            json.loads(cand)
            return cand
        except Exception:
            pass

    # 再找对象 {...}
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        cand = m.group(0).strip()
        try:
            json.loads(cand)
            return cand
        except Exception:
            pass

    cleaned = text.strip().strip("`").strip()
    m2 = re.search(r"\[[\s\S]*\]", cleaned)
    if m2:
        cand2 = m2.group(0).strip()
        try:
            json.loads(cand2)
            return cand2
        except Exception:
            pass

    m3 = re.search(r"\{[\s\S]*\}", cleaned)
    if m3:
        cand3 = m3.group(0).strip()
        try:
            json.loads(cand3)
            return cand3
        except Exception:
            pass

    return None

def _as_point_list(v: Any) -> List[List[float]]:
    """
    统一成 List[[x,y], ...]
    支持：
      - [x,y]
      - [[x,y],[x,y]]
      - [{"point_2d":[x,y]}, ...]
      - [{"center_2d":[x,y]}, ...]
    """
    out: List[List[float]] = []
    if isinstance(v, list):
        # [x,y]
        if len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            return [[float(v[0]), float(v[1])]]

        # [[x,y], ...] or [dict, ...]
        for item in v:
            if isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(x, (int, float)) for x in item):
                out.append([float(item[0]), float(item[1])])
            elif isinstance(item, dict):
                pt = item.get("point_2d") or item.get("center_2d") or item.get("point") or item.get("center")
                if isinstance(pt, list) and len(pt) == 2 and all(isinstance(x, (int, float)) for x in pt):
                    out.append([float(pt[0]), float(pt[1])])
    return out

def get_points_from_any_obj(obj: Any) -> List[List[float]]:
    """
    从任意 obj 里尽量抽取 point 列表。
    优先找：point_2d / points / keypoints / coords
    兼容：如果只有 bbox，就取 bbox 中心点（这样老 GT/老输出也能画点）
    """
    if obj is None:
        return []

    # obj 本身就是 list（可能是 [[x,y],...] 或 [{"point_2d":...},...]）
    if isinstance(obj, list):
        pts = _as_point_list(obj)
        if pts:
            return pts
        return []

    if not isinstance(obj, dict):
        return []

    for k in ("point_2d", "points", "keypoints", "coords", "centers", "center_2d"):
        if k in obj:
            pts = _as_point_list(obj.get(k))
            if pts:
                return pts

    # 有些人会把结果放在 pred / output / result 里面
    for k in ("pred", "output", "result", "prediction"):
        if k in obj:
            pts = get_points_from_any_obj(obj.get(k))
            if pts:
                return pts
    return []

def point_to_pixel_xy(pt: List[float], img_w: int, img_h: int) -> Tuple[int, int]:
    """
    自动判断坐标系：
      - max<=1.5  -> 0~1 归一化
      - max<=1005 -> 0~1000 坐标系
      - 否则      -> 像素
    """
    x, y = float(pt[0]), float(pt[1])
    m = max(abs(x), abs(y))

    if m <= 1.5:
        x, y = x * img_w, y * img_h
    elif m <= 1005:
        x, y = x / 1000.0 * img_w, y / 1000.0 * img_h

    x = max(0, min(int(round(x)), img_w - 1))
    y = max(0, min(int(round(y)), img_h - 1))
    return x, y

def draw_point_cv2(img, x, y, color, label: Optional[str] = None, r: int = 6):
    cv2 = try_import_cv2()
    if cv2 is None:
        return img
    cv2.circle(img, (int(x), int(y)), r, color, 2)
    if label:
        cv2.putText(
            img,
            label,
            (int(x) + 6, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return img

def extract_points_from_text(text: str) -> List[List[float]]:
    """
    兼容：
      - JSON: [{"point_2d":[x,y]}, ...] / [[x,y],[x,y]]
      - Python repr: [(x,y),(x,y)]
    """
    if not isinstance(text, str):
        return []

    raw = text.strip().strip("`").strip()

    # 1) 先尝试 JSON（优先取 [] 或 {}）
    js = extract_json_any(raw)
    if js:
        try:
            obj = json.loads(js)
            pts = get_points_from_any_obj(obj)
            if pts:
                return pts
        except Exception:
            pass

    # 2) 再尝试 python literal
    try:
        obj = ast.literal_eval(raw)
        pts = get_points_from_any_obj(obj)
        return pts
    except Exception:
        return []


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
    p_infer.add_argument("--gpu_memory_utilization", type=float, default=0.5, help="vLLM GPU 内存利用率（默认0.5）")

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
    p_all.add_argument("--max_new_tokens", type=int, default=512)
    p_all.add_argument("--iou_thr", type=float, default=0.5)
    p_all.add_argument("--batch_size", type=int, default=1, help="批量推理的批大小（默认1）")
    p_all.add_argument("--gpu_memory_utilization", type=float, default=0.5, help="vLLM GPU 内存利用率（默认0.5）")

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
        for k, v in metrics.items():
            if k in {"错误统计", "混淆矩阵"}:
                print(f"{k}:")
                for sub_k, sub_v in v.items():
                    print(f"  {sub_k}: {sub_v}")
            else:
                print(f"{k}: {v}")

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
            args.gpu_memory_utilization,
        )

        # 2) 评估
        metrics, per_image_rows = eval_with_gtset(pred_dir=preds_dir, gt_map=gt_map, iou_thr=args.iou_thr)
        save_json(metrics, metrics_dir / "metrics.json")
        save_csv(per_image_rows, metrics_dir / "per_image.csv")
        print(f"[ALL] 指标已保存到: {metrics_dir}")
        print("\n=== 评估结果 ===")
        for k, v in metrics.items():
            if k in {"错误统计", "混淆矩阵"}:
                print(f"{k}:")
                for sub_k, sub_v in v.items():
                    print(f"  {sub_k}: {sub_v}")
            else:
                print(f"{k}: {v}")

        # 3) 可视化
        visualize_annotations_from_gtset(str(preds_dir), gt_map, str(vis_dir), iou_thr=args.iou_thr)
        print(f"[ALL] 可视化已保存到: {vis_dir}")


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=1 python light_sh_yjp/hardprompt-outputpoint*2-en-thinker.py all \
# --model_path /ogi-data/pzb-data/model/VLABench-main/weight/qwen3vl_1215 \
# --gt_set  ./light_data/labels/green_green_val.jsonl \
# ./light_data/labels/green_yellow_val.jsonl \
# ./light_data/labels/yellow_yellow_val.jsonl \
# ./light_data/labels/yellow_green_val.jsonl \
# ./light_data/labels1225/off_off_val.jsonl \
# ./light_data/labels1225/off_unknown_val.jsonl \
# ./light_data/labels1225/red_red_val.jsonl \
# ./light_data/labels1225/unknown_off_val.jsonl \
# ./light_data/labels1225/unknown_red_val.jsonl \
# ./light_data/labels1225/unknown_unknown_val.jsonl \
# ./light_data/labels1225/unknown_yellow_val.jsonl \
# ./light_data/labels1225/yellow_unknown_val.jsonl \
# ./light_data/labels1225/yellow_yellow_val.jsonl \
# --out_dir    light_results_yjp/hardprompt-outputpoint*2-en-thinker \
# --iou_thr    0.5 \
# --batch_size 32 \
# --gpu_memory_utilization 0.5

# CUDA_VISIBLE_DEVICES=2 python light_sh_yjp/light_eval_example_hardprompt-0shot.py eval \
# --pred_dir ./light_results_yjp/basemodel-260115-hardprompt-0shot/preds \
# --gt_set ./light_data/labels/green_green_val.jsonl \
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
# --out_dir    light_results_yjp/basemodel-260115-easyprompt-0shot \
# --iou_thr    0.5