from typing import Dict, List, Tuple
from PIL import Image, ImageDraw
import numpy as np
import re
import json
from collections import defaultdict
import os.path as osp
import os

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


def draw_result(gt: Dict, mask_img: Image, score: float, points: List[Tuple[int, int]]):
    """
    Draws the result of a prediction on an image, including a mask overlay, points, and a score.
    Parameters:
        gt (Dict): Ground truth data containing metadata such as the image path and question ID.
        mask_img (Image): Binary mask image indicating regions of interest.
        score (float): Prediction score to display on the image.
        points (List[Tuple[int, int]]): List of (x, y) coordinates to mark on the image.
    Side Effects:
        Saves the resulting image with overlays and annotations to the 'output/imgs' directory.
    """
    # For debug
    # Load the original image
    img = Image.open(osp.join(gt["data_root"], gt["img_path"]))
    img = img.convert("RGBA")

    # Convert mask to numpy array and create overlay
    mask_array = np.array(mask_img)
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]  # Take first channel if RGB
    mask_array = mask_array / 255.0  # Normalize to 0-1

    # Create semi-transparent green overlay for mask
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_array = np.array(overlay)

    # Apply green color where mask is positive
    mask_indices = mask_array > 0.5
    overlay_array[mask_indices] = [0, 255, 0, 100]  # Semi-transparent green

    overlay = Image.fromarray(overlay_array)
    img = Image.alpha_composite(img, overlay)

    # Convert back to RGB for drawing
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw points as red circles
    points_array = np.array(points)
    for point in points_array:
        x, y = int(point[0]), int(point[1])
        radius = 3
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill="red",
            outline="darkred",
        )

    score_text = f"Score: {score:.3f}"
    text_bbox = draw.textbbox((0, 0), score_text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Position text in top-left corner with padding
    text_x, text_y = 10, 10

    # Draw text background
    draw.rectangle(
        [text_x - 5, text_y - 5, text_x + text_width + 5, text_y + text_height + 5],
        fill="white",
        outline="black",
    )

    # Draw text
    draw.text((text_x, text_y), score_text, fill="black")

    output_dir = "output/imgs"
    os.makedirs(output_dir, exist_ok=True)
    img.save(osp.join(output_dir, f"{gt['question_id']}.png"))


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
        draw_result(gt, mask_img, acc, points)
        print(f"score for {question_id}: {acc}")
        print(f"avg_accuracy:: {results['avg']['score']}")

    for question_type, result in results.items():
        if result["num"]:
            result["accuracy"] = round(result["score"] / result["num"] * 100, 2)
        else:
            result["accuracy"] = 0.0
    results["accuracy"] = results["avg"]["accuracy"]
    return results
