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
from collections import Counter

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_SYSTEM_PROMPT = """
你是一个专业的充电柜指示灯状态识别助手。你的唯一任务：根据用户提供的充电柜图片，识别黑色背板区域内两条指示灯的颜色状态。

【输出硬规则】
1) 只输出一个 JSON。
2) 只识别两条目标指示灯：upper_indicator_light（上指示灯）、lower_indicator_light（下指示灯）。

【目标指示灯的定义】
1) 位置：目标指示灯必须位于充电柜中间的黑色背板区域内。
2) 数量与排列：目标最多只有两条，且位于对应仓口开口/电池进出口的正上方（两个指示灯间距约 30 厘米）。
3) 颜色状态：指示灯可能的状态包括：
- "green"：指示灯呈绿色；
- "yellow"：指示灯呈黄色或者橙色；
- "red"：指示灯呈红色；
- "off"：指示灯实体清晰，但没有任何明显发光/高亮，呈白灰色；
- "unknown"：指示灯被遮挡、模糊无法识别；
4) 尺寸：目标灯条长度约 30 厘米，宽度约 4 厘米；明显更长的灯带/氛围灯/装饰灯一律忽略。
5) 相对关系（可选锚点）：若图片中能看到对应仓口开口/电池进出口，则目标指示灯位于其正上方。

【注意事项】
1) 若候选多于两条：
- 优先选择最符合“长度约 30cm，宽度约 4 厘米且两条尺寸最一致，均处于仓口开口/电池进出口正上方的光条；
- 忽略明显更长、更细、断续、偏离背板核心区域的光条。
2) 若候选少于两条，指示灯可能被遮挡，被遮挡的指示灯对应列表为空，只返回可见的指示灯识别结果。
3) 若指示灯未发光（呈无光白色），状态为 off。
4) 指示灯可能偏暖色（黄/橙），亮度比较低，状态为 yellow。
5) 以下目标无论多显眼都必须排除，不要当成指示灯：
- 二维码/条码/文字标签（通常是方形或带网格/文字纹理）
- 红色按钮/急停/圆形旋钮（圆形或小面积块）
- 充电口/插孔/USB口/圆孔矩形孔（孔洞形态）
6) 严禁基于常识猜测；看不清就识别为 unknown。

【最终输出格式】
{
  "upper_indicator_light": "green|yellow|red|off|unknown",
  "lower_indicator_light": "green|yellow|red|off|unknown"
}
"""


DEFAULT_USER_TEXT = """
识别黑色背板区域内两条指示灯的颜色状态，只输出一个 JSON：
{
  "upper_indicator_light": "green|yellow|red|off|unknown",
  "lower_indicator_light": "green|yellow|red|off|unknown"
}
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

def _normalize_label(x: Any) -> str:
    """把模型输出尽量归一到 {'green','yellow','red','off','unknown'}"""
    if x is None:
        return "unknown"
    s = str(x).strip().lower()

    # 常见同义/中英文映射
    if s in {"unknown", "unknown", "?", "n/a", "na", "none", "null", "uncertain", "unclear", "不确定", "未知", "看不清"}:
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
        # "./light_data/lightVideo/green_image/20251219-175455/frame_000150.jpg",
        # "./light_data/lightVideo/go_image/A5_green_25/frame_000000.jpg",
        # "./light_data/lightVideo/og_image/A5_3/frame_000000.jpg",
        # "./light_data/lightVideo/orange_image/A5_orange_25/frame_000000.jpg",
        # "./light_data/lightVideo/og_image_val/A5_8/frame_000000.jpg",
        # "./light_data/train/off_off/20251224-114316/frame_001110.jpg",
        # "./light_data/train/off_unknown/20251224-102019/frame_001110.jpg",
        # "./light_data/train/unknown_red/20251224-150357/frame_000675.jpg",
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

        # 3) 组装 messages（few-shot + query）
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
        ]

        if use_fewshot and fewshot_bank:
            # 每个 few-shot：user(图片+指令) -> assistant(JSON答案)
            for ex in fewshot_bank:
                ex_img = ex["img"]
                ex_up = ex.get("gt_up", "unknown")
                ex_low = ex.get("gt_low", "unknown")
                ex_uuid = ex.get("uuid")

                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ex_img, "uuid": ex_uuid},
                        {"type": "text", "text": DEFAULT_USER_TEXT}
                    ],
                })
                messages.append({
                    "role": "assistant",
                    "content": json.dumps({"upper_indicator_light": ex_up, "lower_indicator_light": ex_low}, ensure_ascii=False),
                })

        # query：主图+ 指令
        query_content = [{"type": "image", "image": query_img, "uuid": f"{stem}-main"}]
        query_content.append({
            "type": "text",
            "text": DEFAULT_USER_TEXT,
        })
        messages.append({"role": "user", "content": query_content})

        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 收集 images（顺序必须与 prompt 中的 image token 对齐）
        mm_images, mm_uuids = _collect_images_and_uuids_from_messages(messages)

        llm_inp: Dict[str, Any] = {
            "prompt": prompt,
            "multi_modal_data": {"image": mm_images},
        }

        # # vLLM 支持 multi_modal_uuids 做稳定缓存（可选）。:contentReference[oaicite:7]{index=7}
        # if mm_uuids and any(u is not None for u in mm_uuids):
        #     llm_inp["multi_modal_uuids"] = {"image": mm_uuids}

        llm_inputs.append(llm_inp)
        metas.append({
            "stem": stem,
            "img_path": img_path,
            "n_images": len(mm_images),
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

    # 解析输出，组装结果
    results: List[Dict[str, Any]] = []
    for meta, text in zip(metas, outputs_text):
        if meta.get("error"):
            results.append({
                **meta,
                "pred_up": "unknown",
                "pred_low": "unknown",
                "raw_text": "",
                "parse_ok": False,
            })
            continue

        raw_text = (text or "").strip()
        obj_str = extract_json(raw_text)
        obj = json.loads(obj_str) if obj_str else None

        pred_up = "unknown"
        pred_low = "unknown"
        parse_ok = False

        if isinstance(obj, dict):
            up = obj.get("up") or obj.get("upper") or obj.get("top") or obj.get("upper_indicator_light")
            low = obj.get("low") or obj.get("lower") or obj.get("bottom") or obj.get("lower_indicator_light")
            pred_up = _normalize_label(up)
            pred_low = _normalize_label(low)
            parse_ok = True

        # 也兼容模型输出 ["green","red"] 这种
        if not parse_ok:
            try:
                arr = json.loads(raw_text)
                if isinstance(arr, list) and len(arr) >= 2:
                    pred_up = _normalize_label(arr[0])
                    pred_low = _normalize_label(arr[1])
                    parse_ok = True
            except Exception:
                pass
        results.append({
            **meta,
            "pred_up": pred_up,
            "pred_low": pred_low,
            "raw_text": raw_text,
            "parse_ok": parse_ok,
        })

    return results

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
    p_infer.add_argument("--gpu_memory_utilization", type=float, default=0.50, help="vLLM GPU 内存利用率（默认0.50）")

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
    p_all.add_argument("--gpu_memory_utilization", type=float, default=0.50, help="vLLM GPU 内存利用率（默认0.50）")

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
        # visualize_annotations_from_gtset(str(preds_dir), gt_map, str(vis_dir), iou_thr=args.iou_thr)
        # print(f"[ALL] 可视化已保存到: {vis_dir}")


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=2 python light_sh_yjp/hardprompt2-outputjson-cn-qwen3vl.py all \
# --model_path /ogi-data/pzb-data/model/Qwen3-VL-4B-Instruct \
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
# --out_dir    light_results_yjp/260122-hardprompt2-outputjson-cn-qwen3vl \
# --iou_thr    0.5 \
# --batch_size 32

# CUDA_VISIBLE_DEVICES=2 python light_sh_yjp/light_eval_example_hardprompt-0shot.py eval \
# --pred_dir ./light_results_yjp/basemodel-260115-hardprompt-0shot/preds \
# --gt_set ./light_data/labels/green_green_val.jsonl \
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
# --out_dir    light_results_yjp/basemodel-260115-easyprompt-0shot \
# --iou_thr    0.5