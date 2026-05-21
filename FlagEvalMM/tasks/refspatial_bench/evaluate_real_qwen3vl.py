from typing import Dict, List, Tuple
from PIL import Image, ImageDraw
import json
import numpy as np
import re
from collections import defaultdict
import os.path as osp
import os


import json
import re
import numpy as np

def _to_pixel_xy(x, y, width, height):
    """
    Convert (x, y) to pixel coordinates with robust handling:
    - if x,y in [0,1]   => treat as normalized ratio
    - elif x,y in [0,1000] => treat as 0~1000 relative coords (your original setting)
    - else => treat as already in pixels
    """
    x = float(x)
    y = float(y)

    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        x = x * width
        y = y * height
    elif 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0:
        x = (x / 1000.0) * width
        y = (y / 1000.0) * height
    # else: assume pixels already

    x = int(round(x))
    y = int(round(y))
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    return x, y


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

def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    results = defaultdict(lambda: {"num": 0, "score": 0})
    for pred in predictions:
        question_id = str(pred["question_id"])
        gt = annotations[question_id]
        try:
            pred["raw_answer"] = pred["answer"]
            points = text2pts(
                pred.get("answer", ""),
                width=gt["image_width"],
                height=gt["image_height"],
            )
            pred["answer"] = str(points)
            points_array = np.array(points)
        except Exception:
            continue
        mask_img = Image.open(osp.join(gt["data_root"], gt["mask_path"]))

        mask = np.array(mask_img) / 255.0
        acc = 0
        if len(points) > 0:
            in_range = (
                (points_array[:, 0] >= 0)
                & (points_array[:, 0] < mask.shape[1])
                & (points_array[:, 1] >= 0)
                & (points_array[:, 1] < mask.shape[0])
            )
            acc = float(
                np.concatenate(
                    [
                        mask[points_array[in_range, 1], points_array[in_range, 0]],
                        np.zeros(points_array.shape[0] - in_range.sum()),
                    ]
                ).mean()
            )

        pred["score"] = acc
        pred["label"] = gt["mask_path"]
        results["avg"]["num"] += 1
        results["avg"]["score"] += acc
        question_type = gt.get("sub_task")
        results[question_type]["num"] += 1
        results[question_type]["score"] += acc
    for question_type, result in results.items():
        if result["num"]:
            result["accuracy"] = round(result["score"] / result["num"] * 100, 2)
        else:
            result["accuracy"] = 0.0
    results["accuracy"] = results["avg"]["accuracy"]
    return results
