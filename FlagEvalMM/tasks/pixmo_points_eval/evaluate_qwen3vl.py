from typing import Dict, List, Tuple
from PIL import Image, ImageDraw
import numpy as np
import re
from collections import defaultdict
import os.path as osp
import os
import json


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def _to_pixel_xy(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    """
    支持两类输入：
    - 0~1000 坐标（你的 prompt 常用）
    - 0~1 归一化坐标（有的模型会输出）
    """
    # 0~1
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        px = int(round(x * width))
        py = int(round(y * height))
    else:
        # 默认 0~1000
        px = int(round((x / 1000.0) * width))
        py = int(round((y / 1000.0) * height))

    px = _clamp(px, 0, width - 1)
    py = _clamp(py, 0, height - 1)

    # px, py = int(x), int(y)
    return px, py

def _legacy_text2pts_tuple_format(text, width=640, height=480):
    """
    Backward-compatible parser for your old format:
    [(x1, y1), (x2, y2), ...] or rectangles (x0,y0,x1,y1)
    (kept as fallback in case model output is messy)
    """
    text = text.strip().split("\n")[-1]
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    points = []
    for match in matches:
        vector = [float(num) if "." in num else int(num) for num in match.split(",")]
        if len(vector) == 2:
            x, y = vector
            x, y = _to_pixel_xy(x, y, width, height)
            points.append((x, y))
        elif len(vector) == 4:
            x0, y0, x1, y1 = vector
            x0, y0 = _to_pixel_xy(x0, y0, width, height)
            x1, y1 = _to_pixel_xy(x1, y1, width, height)
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = 1
            yy, xx = np.where(mask)
            points.extend(list(np.stack([xx, yy], axis=1)))
    return points

def text2pts(text, width=640, height=480):
    """
    New parser for JSON format:
    [
      {"point_2d": [x, y], "label": "point_1"},
      ...
    ]
    Returns: List[(x_px, y_px), ...]
    """
    raw = text.strip()
    if not raw:
        return []

    # 1) Try to extract JSON from ```json ...``` fence if present
    candidates = []
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, flags=re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())

    # 2) Try bracketed JSON array substrings (handles extra chatter before/after)
    #    We collect multiple candidates and attempt json.loads sequentially.
    for m in re.finditer(r"\[[\s\S]*\]", raw):
        candidates.append(m.group(0))

    # 3) Finally, try the whole raw text
    candidates.append(raw)

    parsed = None
    for cand in candidates:
        # small cleanup: remove trailing commas like {...,}
        cand2 = re.sub(r",\s*([\]}])", r"\1", cand)
        try:
            parsed = json.loads(cand2)
            break
        except Exception:
            continue

    if parsed is None:
        # fallback: old tuple format
        return _legacy_text2pts_tuple_format(raw, width=width, height=height)

    # Normalize to list
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    points = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if "point_2d" not in item:
            continue
        pt = item["point_2d"]
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            continue

        x, y = pt[0], pt[1]
        try:
            x_px, y_px = _to_pixel_xy(x, y, width, height)
            points.append((x_px, y_px))
        except Exception:
            continue

    return points


def draw_result(gt: Dict, mask_img: Image.Image, score: float, points: List[Tuple[int, int]]):
    """
    Debug 可视化：如果原图存在则叠加 mask+points；否则用 mask 作为底图画 points。
    输出到 output/imgs/{question_id}.png
    """
    output_dir = "output/imgs"
    os.makedirs(output_dir, exist_ok=True)

    data_root = gt.get("data_root", "")
    img_path = gt.get("img_path", None)

    base_img = None
    if img_path:
        abs_img = osp.join(data_root, img_path) if data_root else img_path
        if osp.exists(abs_img):
            try:
                base_img = Image.open(abs_img).convert("RGBA")
            except Exception:
                base_img = None

    if base_img is None:
        # 没有原图：用 mask 作为底图
        base_img = mask_img.convert("L").convert("RGBA")

    # mask overlay
    mask_array = np.array(mask_img)
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]
    mask_norm = mask_array / 255.0

    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    overlay_arr = np.array(overlay)

    mask_indices = mask_norm > 0.5
    overlay_arr[mask_indices] = [0, 255, 0, 100]  # green alpha
    overlay = Image.fromarray(overlay_arr)

    img = Image.alpha_composite(base_img, overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # points: red
    for (x, y) in points:
        r = 3
        draw.ellipse([x - r, y - r, x + r, y + r], fill="red", outline="darkred")

    # score text
    score_text = f"Score: {score:.3f}"
    bbox = draw.textbbox((0, 0), score_text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx, ty = 10, 10
    draw.rectangle([tx - 5, ty - 5, tx + tw + 5, ty + th + 5], fill="white", outline="black")
    draw.text((tx, ty), score_text, fill="black")

    img.save(osp.join(output_dir, f"{gt['question_id']}.png"))


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    results = defaultdict(lambda: {"num": 0, "score": 0.0})

    total_seen = 0
    missing_image = 0
    evaluated = 0

    for pred in predictions:
        question_id = str(pred.get("question_id"))
        if question_id not in annotations:
            continue
        gt = annotations[question_id]
        total_seen += 1

        # ---- 1) 如果 img_path 为空/None：图片缺失，跳过评测但计数 ----
        img_path = gt.get("img_path", None)
        if img_path is None or (isinstance(img_path, str) and img_path.strip() == ""):
            missing_image += 1
            # 可选：给 pred 打个标记，方便你后处理
            pred["skipped"] = True
            pred["skip_reason"] = "missing_image"
            continue

        # ---- 2) 必须有 mask 才能评测 ----
        data_root = gt.get("data_root", "")
        mask_path = gt.get("mask_path") or gt.get("answer")
        if not mask_path:
            pred["skipped"] = True
            pred["skip_reason"] = "missing_mask_path"
            continue

        abs_mask = osp.join(data_root, mask_path) if data_root else mask_path
        if not osp.exists(abs_mask):
            pred["skipped"] = True
            pred["skip_reason"] = "mask_not_found"
            continue

        try:
            mask_img = Image.open(abs_mask)
        except Exception:
            pred["skipped"] = True
            pred["skip_reason"] = "mask_open_failed"
            continue

        # width/height：优先用 gt 中的；否则用 mask 尺寸
        width = gt.get("image_width", None)
        height = gt.get("image_height", None)
        if width is None or height is None:
            width, height = mask_img.size

        # ---- 3) parse points ----
        try:
            pred["raw_answer"] = pred.get("answer", "")
            points = text2pts(pred.get("answer", ""), width=width, height=height)
            pred["answer"] = str(points)
            points_array = (
                np.array(points, dtype=np.int32)
                if len(points) > 0
                else np.zeros((0, 2), dtype=np.int32)
            )
        except Exception:
            pred["skipped"] = True
            pred["skip_reason"] = "parse_failed"
            continue

        # ---- 4) compute score ----
        mask = np.array(mask_img)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask / 255.0).astype(np.float32)

        acc = 0.0
        if len(points) > 0:
            in_range = (
                (points_array[:, 0] >= 0)
                & (points_array[:, 0] < mask.shape[1])
                & (points_array[:, 1] >= 0)
                & (points_array[:, 1] < mask.shape[0])
            )
            hit = (
                mask[points_array[in_range, 1], points_array[in_range, 0]]
                if in_range.any()
                else np.array([], dtype=np.float32)
            )
            miss = np.zeros(points_array.shape[0] - int(in_range.sum()), dtype=np.float32)
            acc = float(np.concatenate([hit, miss]).mean()) if points_array.shape[0] > 0 else 0.0

        pred["score"] = acc
        pred["label"] = mask_path

        # ---- 5) 聚合：只对“有图片”的样本计入 num/score ----
        evaluated += 1
        results["avg"]["num"] += 1
        results["avg"]["score"] += acc

        group = gt.get("sub_task", "pixmo_points_eval")
        results[group]["num"] += 1
        results[group]["score"] += acc

        # debug draw
        try:
            draw_result(gt, mask_img, acc, points)
        except Exception:
            pass

        print(f"score for {question_id}: {acc:.4f}")
        print(f"running_avg: {results['avg']['score'] / max(1, results['avg']['num']):.4f}")

    # finalize accuracy
    for k, v in results.items():
        if v["num"] > 0:
            v["accuracy"] = round(v["score"] / v["num"] * 100, 2)
        else:
            v["accuracy"] = 0.0

    results["accuracy"] = results["avg"]["accuracy"] if "avg" in results else 0.0

    # ---- 6) 写入缺失统计 ----
    results["missing_image"] = {
        "num": missing_image,
        "ratio": round((missing_image / total_seen * 100.0), 2) if total_seen > 0 else 0.0,
    }
    results["evaluated"] = {"num": evaluated}
    results["total_seen"] = {"num": total_seen}

    return results
