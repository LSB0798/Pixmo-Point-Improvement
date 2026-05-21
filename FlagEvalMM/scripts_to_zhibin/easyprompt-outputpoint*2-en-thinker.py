#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# =========================
# 配置
# =========================
STATES = {"green", "yellow", "red", "off", "unknown"}

PROMPT_UP = """Please point out the upper indicator lights on the black center back panel. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should range from 0 to 1000, representing the relative pixel positions of the points in the image."""

PROMPT_LOW = """Please point out the lower indicator lights on the black center back panel. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should range from 0 to 1000, representing the relative pixel positions of the points in the image."""

PROMPT_UP2 = """Please point out the upper long indicator lights on the black center back panel. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should range from 0 to 1000, representing the relative pixel positions of the points in the image."""

PROMPT_LOW2 = """Please point out the lower long indicator lights on the black center back panel. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should range from 0 to 1000, representing the relative pixel positions of the points in the image."""

OUTCOMES = [
    "correct",                    # 正确（命中 bbox 或 unknown->空）
    "no_predicted_points",        # 该侧模型没输出点（空）
    "points_outside_bbox",        # 输出了点，但都不在 bbox 内
    "unknown_but_predicted_points",  # GT=unknown，但模型输出了点（误报）
    "missing_gt_bbox",            # GT 需要 bbox，但标注缺失
]

def init_confusion_by_state() -> Dict[str, Dict[str, int]]:
    return {s: {o: 0 for o in OUTCOMES} for s in sorted(STATES)}

def outcome_from_reason(reason: str) -> str:
    """
    如果你也把 eval_one_side 的 reason 改成下面那些直观名字，
    这里就基本等同于 return reason（做个兜底避免意外）
    """
    if reason in OUTCOMES:
        return reason

    legacy = {
        "hit": "correct",
        "unknown_and_empty": "correct",
        "missing_pred_point": "no_predicted_points",
        "all_points_miss": "points_outside_bbox",
        "unknown_but_has_points": "unknown_but_predicted_points",
        "missing_gt_bbox": "missing_gt_bbox",
    }
    return legacy.get(reason, "points_outside_bbox")

# =========================
# 工具
# =========================
def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_json_any(text: str) -> Optional[str]:
    """从文本里粗暴提取第一个合法 JSON（支持 [] 或 {}）。"""
    if not isinstance(text, str):
        return None

    for pat in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
        m = re.search(pat, text)
        if m:
            cand = m.group(0).strip()
            try:
                json.loads(cand)
                return cand
            except Exception:
                pass

    cleaned = text.strip().strip("`").strip()
    for pat in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
        m = re.search(pat, cleaned)
        if m:
            cand = m.group(0).strip()
            try:
                json.loads(cand)
                return cand
            except Exception:
                pass

    return None


def _as_point_list(v: Any) -> List[List[float]]:
    """统一成 List[[x,y], ...]，兼容 [(x,y)], [[x,y]], [{"point_2d":[x,y]}] 等。"""
    out: List[List[float]] = []
    if isinstance(v, list):
        if len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            return [[float(v[0]), float(v[1])]]

        for item in v:
            if isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(x, (int, float)) for x in item):
                out.append([float(item[0]), float(item[1])])
            elif isinstance(item, dict):
                pt = item.get("point_2d") or item.get("center_2d") or item.get("point") or item.get("center")
                if isinstance(pt, list) and len(pt) == 2 and all(isinstance(x, (int, float)) for x in pt):
                    out.append([float(pt[0]), float(pt[1])])
    return out


def get_points_from_any_obj(obj: Any) -> List[List[float]]:
    """从任意 obj 尽量抽取点列表。"""
    if obj is None:
        return []
    if isinstance(obj, list):
        return _as_point_list(obj)
    if not isinstance(obj, dict):
        return []

    for k in ("point_2d", "points", "keypoints", "coords", "centers", "center_2d"):
        if k in obj:
            pts = _as_point_list(obj.get(k))
            if pts:
                return pts

    for k in ("pred", "output", "result", "prediction"):
        if k in obj:
            pts = get_points_from_any_obj(obj.get(k))
            if pts:
                return pts
    return []


def extract_points_from_text(text: str) -> List[List[float]]:
    """
    兼容：
      - JSON: [[x,y],[x,y]] / [{"point_2d":[x,y]}, ...]
      - Python repr: [(x,y),(x,y)]
    """
    if not isinstance(text, str):
        return []

    raw = text.strip().strip("`").strip()

    js = extract_json_any(raw)
    if js:
        try:
            obj = json.loads(js)
            pts = get_points_from_any_obj(obj)
            if pts:
                return pts
        except Exception:
            pass

    try:
        obj = ast.literal_eval(raw)
        return get_points_from_any_obj(obj)
    except Exception:
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


def make_sample_id(img_path: str, keep_parts: int = 6) -> str:
    """用路径末尾几段拼出稳定且可读的 id。"""
    p = Path(os.path.normpath(img_path))
    parts = [x for x in p.parts if x not in (os.sep, "")]
    tail = parts[-keep_parts:] if len(parts) >= keep_parts else parts
    if tail:
        tail = list(tail[:-1]) + [p.stem]
    s = "_".join(tail)
    s = s.replace(":", "")
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or p.stem


def normalize_state(s: str) -> str:
    s = (s or "").strip().lower()
    if s == "unkown":
        s = "unknown"
    return s if s in STATES else "unknown"


def parse_states_from_jsonl_name(jsonl_path: str) -> Tuple[str, str]:
    """
    由文件名解析上下灯状态：
      green_unknown_val.jsonl -> ("green","unknown")
    兼容多余下划线：red_unknown__val.jsonl
    """
    name = Path(jsonl_path).name
    base = name
    for suf in ("_val.jsonl", ".jsonl"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    toks = [t for t in base.split("_") if t]  # 去掉空 token
    up = normalize_state(toks[0]) if len(toks) >= 1 else "unknown"
    low = normalize_state(toks[1]) if len(toks) >= 2 else "unknown"
    return up, low


# =========================
# 标注读取：labelme points -> bbox
# =========================
def points_to_bbox(points: Any) -> Optional[Tuple[float, float, float, float]]:
    """
    points:
      - 4 点(左上,右上,右下,左下) 或任意 polygon
      - 2 点(左上,右下)
    返回 (x1,y1,x2,y2)
    """
    if not isinstance(points, list) or len(points) < 2:
        return None
    xs, ys = [], []
    for p in points:
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            continue
        xs.append(float(p[0]))
        ys.append(float(p[1]))
    if len(xs) < 2:
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return (x1, y1, x2, y2)


def load_labelme_bboxes(
    img_path: str,
    up_state: str,
    low_state: str,
) -> Dict[str, Any]:
    """
    读取与图片同名的 labelme json：
      .../xxx.jpg -> .../xxx.json

    重要：unknown_unknown 时允许标注 json 不存在（GT 为空框）。
    """
    ann_path = str(Path(img_path).with_suffix(".json"))
    ann = read_json(Path(ann_path))

    out = {
        "ann_path": ann_path,
        "img_w": None,
        "img_h": None,
        "bbox_up": None,
        "bbox_low": None,
        "ann_ok": False,
    }

    # 允许 unknown_unknown 没有标注文件：GT 为空
    if ann is None:
        if up_state == "unknown" and low_state == "unknown":
            out["ann_ok"] = True
        return out

    if not isinstance(ann, dict):
        # 同样：unknown_unknown 允许
        if up_state == "unknown" and low_state == "unknown":
            out["ann_ok"] = True
        return out

    img_w = ann.get("imageWidth")
    img_h = ann.get("imageHeight")
    if isinstance(img_w, int) and isinstance(img_h, int):
        out["img_w"] = img_w
        out["img_h"] = img_h

    shapes = ann.get("shapes")
    if not isinstance(shapes, list):
        # unknown_unknown 允许 shapes 不存在
        if up_state == "unknown" and low_state == "unknown":
            out["ann_ok"] = True
        return out

    bboxes: List[Tuple[float, float, float, float]] = []
    for sh in shapes:
        if not isinstance(sh, dict):
            continue
        bbox = points_to_bbox(sh.get("points"))
        if bbox is None:
            continue
        bboxes.append(bbox)

    # 允许 unknown_unknown shapes 为空（GT 无 bbox）
    if not bboxes:
        out["ann_ok"] = True
        return out

    # 计算 center_y 并排序：上(小) -> 下(大)
    def cy(b):
        return (b[1] + b[3]) / 2.0

    bboxes_sorted = sorted(bboxes, key=cy)

    if len(bboxes_sorted) >= 2:
        out["bbox_up"] = bboxes_sorted[0]
        out["bbox_low"] = bboxes_sorted[-1]
        out["ann_ok"] = True
        return out

    # 只有 1 个 bbox：按“哪个不是 unknown”分配
    only = bboxes_sorted[0]
    if up_state != "unknown" and low_state == "unknown":
        out["bbox_up"] = only
        out["ann_ok"] = True
        return out
    if low_state != "unknown" and up_state == "unknown":
        out["bbox_low"] = only
        out["ann_ok"] = True
        return out

    # 两个都不是 unknown 但只给了 1 个框：保底猜（不严谨）
    if out["img_h"] is not None:
        if cy(only) < out["img_h"] / 2.0:
            out["bbox_up"] = only
        else:
            out["bbox_low"] = only
    else:
        out["bbox_up"] = only

    out["ann_ok"] = True
    return out


# =========================
# GT 构建：jsonl -> gt_map（以 image 为单位）
# =========================
def get_img_path_from_jsonl_obj(obj: Dict[str, Any]) -> Optional[str]:
    images = obj.get("images")
    if isinstance(images, list) and images and isinstance(images[0], str):
        return images[0]
    if isinstance(obj.get("image"), str):
        return obj["image"]
    return None


def load_gt_sets(gt_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    按“图片”去重，构造 gt_map：
      gt_map[stem] = {
        "stem": ...,
        "img_path": ...,
        "up_state": ...,
        "low_state": ...,
        "img_w": ...,
        "img_h": ...,
        "bbox_up": ...,
        "bbox_low": ...,
        "ann_path": ...,
      }
    """
    gt_map: Dict[str, Dict[str, Any]] = {}
    seen_img = set()

    for jsonl_path in gt_paths:
        up_state, low_state = parse_states_from_jsonl_name(jsonl_path)

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    print(f"[WARN] JSONL 解析失败: {jsonl_path}:{ln}")
                    continue

                img_path = get_img_path_from_jsonl_obj(obj)
                if not isinstance(img_path, str) or not img_path:
                    print(f"[WARN] 缺少 images[0]: {jsonl_path}:{ln}")
                    continue

                img_key = os.path.normpath(img_path)
                if img_key in seen_img:
                    continue
                seen_img.add(img_key)

                stem = make_sample_id(img_path)

                ann_info = load_labelme_bboxes(img_path, up_state, low_state)

                gt_map[stem] = {
                    "stem": stem,
                    "img_path": img_path,
                    "up_state": up_state,
                    "low_state": low_state,
                    **ann_info,
                }

    print(f"[GT] Loaded images: {len(gt_map)}")
    return gt_map


# =========================
# 推理：Qwen3-VL + vLLM（每图两次：upper/lower）
# =========================
def load_vllm(model_path: str, max_num_seqs: int, gpu_memory_utilization: float) -> Tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=20000,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False,
        limit_mm_per_prompt={"image": 5, "video": 0},
    )
    return processor, llm


def build_one_prompt(processor, img: Image.Image, text: str) -> Tuple[str, List[Image.Image]]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt, [img]


def infer_batch(
    processor,
    llm,
    batch: List[Tuple[str, str]],
    max_new_tokens: int,
) -> Dict[str, Dict[str, Any]]:
    """
    batch: [(stem, img_path), ...]
    return:
      results[stem] = {
        "pred_points_up": [...],
        "pred_points_low": [...],
        "raw_text_up": "...",
        "raw_text_low": "...",
      }
    """
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    llm_inputs = []
    metas = []  # 每条 input 对应 (stem, which)

    for stem, img_path in batch:
        img = Image.open(img_path).convert("RGB")

        p_up, mm_up = build_one_prompt(processor, img, PROMPT_UP2)
        llm_inputs.append({"prompt": p_up, "multi_modal_data": {"image": mm_up}})
        metas.append((stem, "up"))

        p_low, mm_low = build_one_prompt(processor, img, PROMPT_LOW2)
        llm_inputs.append({"prompt": p_low, "multi_modal_data": {"image": mm_low}})
        metas.append((stem, "low"))

    outputs = llm.generate(llm_inputs, sampling_params=sampling_params)

    results: Dict[str, Dict[str, Any]] = {}
    for (stem, which), out in zip(metas, outputs):
        text = ""
        try:
            text = out.outputs[0].text.strip()
        except Exception:
            text = ""

        pts = extract_points_from_text(text)

        if stem not in results:
            results[stem] = {
                "pred_points_up": [],
                "pred_points_low": [],
                "raw_text_up": "",
                "raw_text_low": "",
            }

        if which == "up":
            results[stem]["pred_points_up"] = pts
            results[stem]["raw_text_up"] = text
        else:
            results[stem]["pred_points_low"] = pts
            results[stem]["raw_text_low"] = text

    return results


def run_infer_from_gtset(
    model_path: str,
    gt_map: Dict[str, Dict[str, Any]],
    out_dir: str,
    max_new_tokens: int,
    batch_size: int,
    gpu_memory_utilization: float,
):
    os.makedirs(out_dir, exist_ok=True)

    # 注意：每张图两条请求，所以 max_num_seqs 建议 >= 2*batch_size
    processor, llm = load_vllm(
        model_path=model_path,
        max_num_seqs=max(1, batch_size * 2),
        gpu_memory_utilization=gpu_memory_utilization,
    )

    samples = []
    for stem, rec in sorted(gt_map.items()):
        img_path = rec.get("img_path")
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            print(f"[WARN] image not found, skip: {img_path}")
            continue
        samples.append((stem, img_path))

    with tqdm(total=len(samples), desc="Infer[vLLM]") as pbar:
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            batch_results = infer_batch(processor, llm, batch, max_new_tokens=max_new_tokens)

            for stem, img_path in batch:
                rec = gt_map[stem]
                pred = batch_results.get(stem, {
                    "pred_points_up": [],
                    "pred_points_low": [],
                    "raw_text_up": "",
                    "raw_text_low": "",
                })

                out = {
                    "stem": stem,
                    "img_path": img_path,
                    "up_state": rec.get("up_state"),
                    "low_state": rec.get("low_state"),
                    "pred_points_up": pred["pred_points_up"],
                    "pred_points_low": pred["pred_points_low"],
                    "raw_text_up": pred["raw_text_up"],
                    "raw_text_low": pred["raw_text_low"],
                }

                with open(os.path.join(out_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)

            pbar.update(len(batch))

    print(f"[Infer] Done -> {out_dir}")


# =========================
# 评估：point in bbox / unknown -> []
# =========================
def point_in_bbox(px: int, py: int, bbox: Tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = bbox
    return (x1 <= px <= x2) and (y1 <= py <= y2)


def eval_one_side(
    gt_state: str,
    gt_bbox: Optional[Tuple[float, float, float, float]],
    pred_points: List[List[float]],
    img_w: int,
    img_h: int,
) -> Tuple[bool, str]:
    """
    return: (ok, reason)
    reason 使用直观名字，方便 per_image.csv 和 confusion matrix 直接看：
      - correct
      - no_predicted_points
      - points_outside_bbox
      - unknown_but_predicted_points
      - missing_gt_bbox
    """
    if gt_state == "unknown":
        if not pred_points:
            return True, "correct"
        return False, "unknown_but_predicted_points"

    if gt_bbox is None:
        return False, "missing_gt_bbox"

    if not pred_points:
        return False, "no_predicted_points"

    # 任意一个点落入 bbox 就算对
    for pt in pred_points:
        px, py = point_to_pixel_xy(pt, img_w, img_h)
        if point_in_bbox(px, py, gt_bbox):
            return True, "correct"

    return False, "points_outside_bbox"

def eval_with_gtset(pred_dir: Path, gt_map: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n_total = 0

    upper_eval = 0
    lower_eval = 0
    both_eval = 0

    upper_correct = 0
    lower_correct = 0
    both_correct = 0

    invalid_pred_file = 0
    invalid_ann = 0

    rows: List[Dict[str, Any]] = []
    conf_upper = init_confusion_by_state()
    conf_lower = init_confusion_by_state()


    for stem, rec in sorted(gt_map.items()):
        n_total += 1
        img_path = rec.get("img_path")
        up_state = rec.get("up_state")
        low_state = rec.get("low_state")
        ann_ok = bool(rec.get("ann_ok"))

        row: Dict[str, Any] = {
            "stem": stem,
            "img_path": img_path,
            "up_state": up_state,
            "low_state": low_state,
            "ann_path": rec.get("ann_path"),
            "ann_ok": ann_ok,
        }

        if not ann_ok:
            invalid_ann += 1
            row["err"] = "ann_not_ok"
            rows.append(row)
            continue

        img_w = rec.get("img_w")
        img_h = rec.get("img_h")
        if not isinstance(img_w, int) or not isinstance(img_h, int):
            # 尺寸缺失：兜底从图片读（评估阶段才读，避免无脑 IO）
            try:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception:
                invalid_ann += 1
                row["err"] = "missing_image_size"
                rows.append(row)
                continue

        pred = read_json(pred_dir / f"{stem}.json")
        if not isinstance(pred, dict):
            invalid_pred_file += 1
            row["err"] = "missing_pred_json"
            rows.append(row)
            continue

        pred_up = pred.get("pred_points_up") or []
        pred_low = pred.get("pred_points_low") or []
        pred_up = _as_point_list(pred_up) if isinstance(pred_up, list) else []
        pred_low = _as_point_list(pred_low) if isinstance(pred_low, list) else []

        # unknown_unknown：无需 bbox/尺寸，直接按“必须输出空”评估
        if up_state == "unknown" and low_state == "unknown":
            up_ok = (len(pred_up) == 0)
            low_ok = (len(pred_low) == 0)
            row.update({
                "bbox_up": None,
                "bbox_low": None,
                "pred_points_up": pred_up,
                "pred_points_low": pred_low,
                "up_ok": up_ok,
                "low_ok": low_ok,
                "up_reason": "unknown_and_empty" if up_ok else "unknown_but_has_points",
                "low_reason": "unknown_and_empty" if low_ok else "unknown_but_has_points",
                "both_ok": bool(up_ok and low_ok),
            })

            upper_eval += 1
            lower_eval += 1
            both_eval += 1
            if up_ok: upper_correct += 1
            if low_ok: lower_correct += 1
            if up_ok and low_ok: both_correct += 1
            # 更新 confusion
            conf_upper[up_state][outcome_from_reason(row["up_reason"])] += 1
            conf_lower[low_state][outcome_from_reason(row["low_reason"])] += 1

            rows.append(row)
            continue

        bbox_up = rec.get("bbox_up")
        bbox_low = rec.get("bbox_low")

        # upper
        up_ok, up_reason = eval_one_side(up_state, bbox_up, pred_up, img_w, img_h)
        # lower
        low_ok, low_reason = eval_one_side(low_state, bbox_low, pred_low, img_w, img_h)

        row.update({
            "bbox_up": bbox_up,
            "bbox_low": bbox_low,
            "pred_points_up": pred_up,
            "pred_points_low": pred_low,
            "up_ok": up_ok,
            "low_ok": low_ok,
            "up_reason": up_reason,
            "low_reason": low_reason,
            "both_ok": bool(up_ok and low_ok),
        })

        # 计分时：unknown 也属于可评估（因为规则明确：必须输出空）
        upper_eval += 1
        lower_eval += 1
        both_eval += 1

        if up_ok:
            upper_correct += 1
        if low_ok:
            lower_correct += 1
        if up_ok and low_ok:
            both_correct += 1
        # 更新 confusion
        conf_upper[up_state][outcome_from_reason(up_reason)] += 1
        conf_lower[low_state][outcome_from_reason(low_reason)] += 1

        rows.append(row)

    def safe_div(a, b):
        return round(a / b, 6) if b else 0.0

    metrics = {
        "total_images": n_total,
        "invalid_ann": invalid_ann,
        "invalid_pred_file": invalid_pred_file,
        "upper_acc": safe_div(upper_correct, upper_eval),
        "lower_acc": safe_div(lower_correct, lower_eval),
        "both_acc": safe_div(both_correct, both_eval),
        "upper_eval": upper_eval,
        "lower_eval": lower_eval,
        "both_eval": both_eval,
        "upper_correct": upper_correct,
        "lower_correct": lower_correct,
        "both_correct": both_correct,
        "confusion": {
            "upper": conf_upper,
            "lower": conf_lower,
        },
    }
    return metrics, rows


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# =========================
# 可视化（可选）
# =========================
def try_import_cv2():
    try:
        import cv2
        return cv2
    except Exception:
        return None


def draw_bbox_cv2(img, bbox, color, label: str):
    cv2 = try_import_cv2()
    if cv2 is None or bbox is None:
        return img
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(img, label, (int(x1), max(0, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def draw_points_cv2(img, pts, img_w, img_h, color, prefix: str):
    cv2 = try_import_cv2()
    if cv2 is None:
        return img
    for i, pt in enumerate(pts or []):
        px, py = point_to_pixel_xy(pt, img_w, img_h)
        cv2.circle(img, (px, py), 6, color, 2)
        cv2.putText(img, f"{prefix}{i}", (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def visualize(pred_dir: str, gt_map: Dict[str, Dict[str, Any]], out_dir: str):
    cv2 = try_import_cv2()
    if cv2 is None:
        print("[WARN] OpenCV not installed, skip visualization.")
        return

    pred_p = Path(pred_dir)
    out_p = Path(out_dir)
    true_p = out_p / "true"
    false_p = out_p / "false"
    true_p.mkdir(parents=True, exist_ok=True)
    false_p.mkdir(parents=True, exist_ok=True)

    for stem, rec in sorted(gt_map.items()):
        img_path = rec.get("img_path")
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            continue

        pred = read_json(pred_p / f"{stem}.json")
        if not isinstance(pred, dict):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        bbox_up = rec.get("bbox_up")
        bbox_low = rec.get("bbox_low")

        pred_up = _as_point_list(pred.get("pred_points_up")) if isinstance(pred.get("pred_points_up"), list) else []
        pred_low = _as_point_list(pred.get("pred_points_low")) if isinstance(pred.get("pred_points_low"), list) else []

        # GT bbox：绿色
        img = draw_bbox_cv2(img, bbox_up, (0, 255, 0), "GT_UP")
        img = draw_bbox_cv2(img, bbox_low, (0, 255, 0), "GT_LOW")

        # Pred 点：红色
        img = draw_points_cv2(img, pred_up, img_w, img_h, (0, 0, 255), "PUP")
        img = draw_points_cv2(img, pred_low, img_w, img_h, (0, 0, 255), "PLO")

        # 重新算一下是否 both_ok，用于分桶
        up_ok, _ = eval_one_side(rec.get("up_state"), bbox_up, pred_up, img_w, img_h)
        low_ok, _ = eval_one_side(rec.get("low_state"), bbox_low, pred_low, img_w, img_h)
        both_ok = bool(up_ok and low_ok)

        out_path = (true_p if both_ok else false_p) / f"{stem}_vis.png"
        cv2.imwrite(str(out_path), img)

    print(f"[VIS] saved to: {out_p}")


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser("Light point eval (point-in-bbox, unknown->[])")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_infer = sub.add_parser("infer", help="infer only -> out_dir")
    p_infer.add_argument("--gt_set", type=str, nargs="+", required=True)
    p_infer.add_argument("--model_path", type=str, required=True)
    p_infer.add_argument("--out_dir", type=str, required=True)
    p_infer.add_argument("--max_new_tokens", type=int, default=128)
    p_infer.add_argument("--batch_size", type=int, default=32)
    p_infer.add_argument("--gpu_memory_utilization", type=float, default=0.3)

    p_eval = sub.add_parser("eval", help="eval only -> out_dir/metrics")
    p_eval.add_argument("--pred_dir", type=str, required=True)
    p_eval.add_argument("--gt_set", type=str, nargs="+", required=True)
    p_eval.add_argument("--out_dir", type=str, required=True)
    # 兼容你旧命令：新逻辑不使用 iou_thr，但保留参数不报错
    p_eval.add_argument("--iou_thr", type=float, default=0.5)

    p_vis = sub.add_parser("vis", help="visualize -> out_dir/vis/{true,false}")
    p_vis.add_argument("--pred_dir", type=str, required=True)
    p_vis.add_argument("--gt_set", type=str, nargs="+", required=True)
    p_vis.add_argument("--out_dir", type=str, required=True)
    p_vis.add_argument("--iou_thr", type=float, default=0.5)

    p_all = sub.add_parser("all", help="infer -> eval -> vis")
    p_all.add_argument("--model_path", type=str, required=True)
    p_all.add_argument("--gt_set", type=str, nargs="+", required=True)
    p_all.add_argument("--out_dir", type=str, required=True)
    p_all.add_argument("--max_new_tokens", type=int, default=128)
    p_all.add_argument("--batch_size", type=int, default=32)
    p_all.add_argument("--gpu_memory_utilization", type=float, default=0.3)
    p_all.add_argument("--iou_thr", type=float, default=0.5)

    args = parser.parse_args()

    if args.cmd == "infer":
        gt_map = load_gt_sets(args.gt_set)
        run_infer_from_gtset(
            model_path=args.model_path,
            gt_map=gt_map,
            out_dir=args.out_dir,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    elif args.cmd == "eval":
        gt_map = load_gt_sets(args.gt_set)
        metrics, rows = eval_with_gtset(Path(args.pred_dir), gt_map)
        out_root = Path(args.out_dir) / "metrics"
        save_json(metrics, out_root / "metrics.json")
        save_csv(rows, out_root / "per_image.csv")
        print("[EVAL] metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print(f"[EVAL] saved to: {out_root}")

    elif args.cmd == "vis":
        gt_map = load_gt_sets(args.gt_set)
        vis_dir = Path(args.out_dir) / "vis"
        visualize(args.pred_dir, gt_map, str(vis_dir))

    elif args.cmd == "all":
        out_root = Path(args.out_dir)
        preds_dir = out_root / "preds"
        metrics_dir = out_root / "metrics"
        vis_dir = out_root / "vis"
        preds_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

        gt_map = load_gt_sets(args.gt_set)

        # 1) infer
        run_infer_from_gtset(
            model_path=args.model_path,
            gt_map=gt_map,
            out_dir=str(preds_dir),
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        # 2) eval
        metrics, rows = eval_with_gtset(preds_dir, gt_map)
        save_json(metrics, metrics_dir / "metrics.json")
        save_csv(rows, metrics_dir / "per_image.csv")
        print("[ALL] metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print(f"[ALL] metrics saved to: {metrics_dir}")

        # 3) vis
        visualize(str(preds_dir), gt_map, str(vis_dir))
        print(f"[ALL] vis saved to: {vis_dir}")


if __name__ == "__main__":
    main()

# /ogi-data/pzb-data/model/VLABench-main/weight/qwen3vl_1215
# /ogi-data/pzb-data/light_output/t260130-rank8/v0-20260130-190415/checkpoint-2900-merged: 未加thinker数据
# /ogi-data/pzb-data/light_output/t260201-rank8/v9-20260202-213924/checkpoint-1100-merged：加thinker数据
# /ogi-data/pzb-data/light_output_yjp/20260206-onlyvisualgd/v5-20260208-181742/checkpoint-900-merged.  sft

# CUDA_VISIBLE_DEVICES=0 python light_sh_yjp/easyprompt-outputpoint*2-en-thinker.py all \
# --model_path /ogi-data/pzb-data/light_output_yjp/20260206-onlyvisualgd/v5-20260208-181742/checkpoint-900-merged \
# --gt_set  ./light_data/bc_light/val_jsonl/green_green_val.jsonl \
# ./light_data/bc_light/val_jsonl/green_unknown_val.jsonl \
# ./light_data/bc_light/val_jsonl/green_yellow_val.jsonl \
# ./light_data/bc_light/val_jsonl/off_off_val.jsonl \
# ./light_data/bc_light/val_jsonl/off_unknown_val.jsonl \
# ./light_data/bc_light/val_jsonl/red_red_val.jsonl \
# ./light_data/bc_light/val_jsonl/red_unknown_val.jsonl \
# ./light_data/bc_light/val_jsonl/unknown_green_val.jsonl \
# ./light_data/bc_light/val_jsonl/unknown_off_val.jsonl \
# ./light_data/bc_light/val_jsonl/unknown_red_val.jsonl \
# ./light_data/bc_light/val_jsonl/unknown_unknown_val.jsonl \
# ./light_data/bc_light/val_jsonl/unknown_yellow_val.jsonl \
# ./light_data/bc_light/val_jsonl/yellow_green_val.jsonl \
# ./light_data/bc_light/val_jsonl/yellow_unknown_val.jsonl \
# ./light_data/bc_light/val_jsonl/yellow_yellow_val.jsonl \
# --out_dir    light_results_yjp/easyprompt2-outputpoint*2-en-thinker-v5-20260208-181742-checkpoint-900 \
# --batch_size 32 \
# --gpu_memory_utilization 0.3
