import json
from typing import Dict, List, Tuple
import re
from collections import defaultdict
import ast
from flagevalmm.evaluator.pre_process import normalize_string
import numpy as np
import os.path as osp
import os
from PIL import Image, ImageDraw

def _load_json_from_text(text: str):
    """
    Try best-effort to extract and json.loads an array/dict from a messy LLM output.
    Supports:
      - ```json ... ``` fenced blocks
      - substring from first '[' to last ']'
      - whole text
    Also removes trailing commas like {...,} / [...,]
    Returns parsed object or None.
    """
    s = (text or "").strip()
    if not s:
        return None

    candidates = []

    # fenced block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    if m:
        candidates.append(m.group(1).strip())

    # first [ ... last ]
    lb, rb = s.find("["), s.rfind("]")
    if lb != -1 and rb != -1 and rb > lb:
        candidates.append(s[lb:rb + 1])

    # whole text
    candidates.append(s)

    for cand in candidates:
        # remove trailing commas before ] or }
        cand2 = re.sub(r",\s*([\]}])", r"\1", cand)
        try:
            return json.loads(cand2)
        except Exception:
            continue
    return None


# From the official evaluation code of RoboSpatial-Home: https://github.com/chanhee-luke/RoboSpatial-Eval/blob/master/evaluation.py
def point_in_polygon(x, y, poly):
    """
    Check if the point (x, y) lies within the polygon defined by a list of (x, y) tuples.
    Uses the ray-casting algorithm.
    """
    num = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, num + 1):
        p2x, p2y = poly[i % num]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                else:
                    xinters = p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def evaluate_answer(ground_truth, generated_answer, width=None, height=None):
    """
    Evaluates if the generated answer is correct based on the ground truth.
    Returns (is_correct, is_binary_answer, parsed_answer, is_parsable).

    If width & height provided:
      - if x,y in [0,1] => scale by (x*width, y*height)
      - elif x,y in [0,1000] => scale by (x/1000*width, y/1000*height)
      - else => treat as pixels
    """
    def _maybe_scale(x, y):
        if width is not None and height is not None:
            if 0 <= x <= 1 and 0 <= y <= 1:
                return x * width, y * height
            if 0 <= x <= 1000 and 0 <= y <= 1000:
                return (x / 1000.0) * width, (y / 1000.0) * height
        return x, y

    gen_answer = (generated_answer or "").strip().lower()
    gt_lower = (ground_truth or "").strip().lower()

    # Binary yes/no
    if gt_lower in ["yes", "no"]:
        is_binary = True
        is_parsable = len(gen_answer) > 0
        correct = gen_answer.startswith(gt_lower)
        return correct, is_binary, gen_answer, is_parsable

    # Numeric: ground_truth is polygon
    is_binary = False
    parsed_answer = None
    is_parsable = False

    try:
        gt_polygon = ast.literal_eval(ground_truth)
        if not isinstance(gt_polygon, list) or len(gt_polygon) < 3:
            return False, is_binary, parsed_answer, is_parsable

        # ---- NEW: try JSON first ----
        parsed_json = _load_json_from_text(generated_answer)
        if parsed_json is not None:
            items = [parsed_json] if isinstance(parsed_json, dict) else parsed_json
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and "point_2d" in item:
                        pt = item["point_2d"]
                        if isinstance(pt, (list, tuple)) and len(pt) == 2:
                            try:
                                x = float(pt[0]); y = float(pt[1])
                                x, y = _maybe_scale(x, y)
                                parsed_answer = (x, y)
                                is_parsable = True
                                correct = point_in_polygon(x, y, gt_polygon)
                                return correct, is_binary, parsed_answer, is_parsable
                            except Exception:
                                pass
        # ---- END JSON ----

        # Old tuple format (x, y)
        tuple_match = re.search(r"\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)", generated_answer)
        if tuple_match:
            try:
                x = float(tuple_match.group(1))
                y = float(tuple_match.group(2))
                x, y = _maybe_scale(x, y)
                parsed_answer = (x, y)
                is_parsable = True
                correct = point_in_polygon(x, y, gt_polygon)
                return correct, is_binary, parsed_answer, is_parsable
            except (ValueError, TypeError):
                pass

        # Old list format [x, y]
        list_match = re.search(r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]", generated_answer)
        if list_match:
            try:
                x = float(list_match.group(1))
                y = float(list_match.group(2))
                x, y = _maybe_scale(x, y)
                parsed_answer = (x, y)
                is_parsable = True
                correct = point_in_polygon(x, y, gt_polygon)
                return correct, is_binary, parsed_answer, is_parsable
            except (ValueError, TypeError):
                pass

        # Fallback: original bracket parsing
        match = re.search(r"\[(.*?)\]", generated_answer, re.DOTALL)
        if match is None:
            return False, is_binary, parsed_answer, is_parsable

        list_content = match.group(1)
        list_content = re.sub(r",(\S)", r", \1", list_content)

        list_content = list_content.strip()
        if list_content.endswith(","):
            list_content = list_content[:-1]

        list_str = "[" + list_content + "]"

        try:
            gen_val = ast.literal_eval(list_str)
        except (SyntaxError, ValueError):
            tuple_match = re.search(r"\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)", list_content)
            if tuple_match:
                x = float(tuple_match.group(1))
                y = float(tuple_match.group(2))
                x, y = _maybe_scale(x, y)
                parsed_answer = (x, y)
                is_parsable = True
                correct = point_in_polygon(x, y, gt_polygon)
                return correct, is_binary, parsed_answer, is_parsable
            return False, is_binary, parsed_answer, is_parsable

        if isinstance(gen_val, list):
            if len(gen_val) == 0:
                return False, is_binary, parsed_answer, is_parsable
            if len(gen_val) == 2 and all(isinstance(v, (int, float)) for v in gen_val):
                gen_point = tuple(gen_val)
            elif isinstance(gen_val[0], tuple):
                gen_point = gen_val[0]
            elif isinstance(gen_val[0], list) and len(gen_val[0]) == 2:
                gen_point = tuple(gen_val[0])
            else:
                return False, is_binary, parsed_answer, is_parsable
        elif isinstance(gen_val, tuple):
            gen_point = gen_val
        else:
            return False, is_binary, parsed_answer, is_parsable

        if not (isinstance(gen_point, tuple) and len(gen_point) == 2):
            return False, is_binary, parsed_answer, is_parsable

        x, y = float(gen_point[0]), float(gen_point[1])
        x, y = _maybe_scale(x, y)
        parsed_answer = (x, y)
        is_parsable = True
        correct = point_in_polygon(x, y, gt_polygon)
        return correct, is_binary, parsed_answer, is_parsable

    except Exception as e:
        print(f"Error evaluating answer: {e}")
        return False, is_binary, parsed_answer, is_parsable

def text2pts(text, width=640, height=480):
    def _to_pixel(x, y):
        x = float(x); y = float(y)
        if 0 <= x <= 1 and 0 <= y <= 1:
            x = x * width
            y = y * height
        elif 0 <= x <= 1000 and 0 <= y <= 1000:
            x = (x / 1000.0) * width
            y = (y / 1000.0) * height
        x = int(round(x)); y = int(round(y))
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        return x, y

    # ---- NEW: try JSON first (do NOT only take last line) ----
    parsed_json = _load_json_from_text(text)
    if parsed_json is not None:
        items = [parsed_json] if isinstance(parsed_json, dict) else parsed_json
        points = []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and "point_2d" in item:
                    pt = item["point_2d"]
                    if isinstance(pt, (list, tuple)) and len(pt) == 2:
                        try:
                            x, y = _to_pixel(pt[0], pt[1])
                            points.append((x, y))
                        except Exception:
                            pass
        if points:
            return points
    # ---- END JSON ----

    # Fallback: old behavior (last line)
    text_last = (text or "").strip().split("\n")[-1]
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text_last)
    if matches == []:
        pattern = re.compile(
            r'<points\b(?=[^>]*\bx1\s*=\s*"(-?\d+(?:\.\d+)?)")(?=[^>]*\by1\s*=\s*"(-?\d+(?:\.\d+)?)")[^>]*>',
            re.IGNORECASE
        )
        matches = [f"{x},{y}" for x, y in pattern.findall(text_last)]

    points = []
    for match in matches:
        vector = [float(num) if "." in num else int(num) for num in match.split(",")]
        if len(vector) == 2:
            x, y = _to_pixel(vector[0], vector[1])
            points.append((x, y))
        elif len(vector) == 4:
            x0, y0 = _to_pixel(vector[0], vector[1])
            x1, y1 = _to_pixel(vector[2], vector[3])
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = True
            y_coords, x_coords = np.where(mask)
            points.extend(list(np.stack([x_coords, y_coords], axis=1)))
    return points


def evaluate_answer_v2(gt, normed_answer):
    # use where2place's evaluation method
    gt_lower = gt["answer"].strip().lower()

    # Check if this is a binary yes/no question
    if gt_lower in ["yes", "no"]:
        return normed_answer.lower().startswith(gt_lower)

    mask_img = Image.open(osp.join(gt["data_root"], gt["mask_path"]))

    points = text2pts(normed_answer, width=gt["image_width"], height=gt["image_height"])
    points_array = np.array(points)
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
    draw_result(gt, mask_img, acc, points)
    return acc


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    results = defaultdict(
        lambda: {"num_correct": 0, "total": 0, "illformed_responses": 0}
    )
    results_v2 = defaultdict(lambda: {"num_correct": 0, "total": 0})
    num_correct = 0
    num_correct_v2 = 0
    for pred in predictions:
        question_id = str(pred["question_id"])
        gt = annotations[question_id]
        pred["raw_answer"] = pred["answer"]
        raw_pred = (pred["answer"] or "").strip()
        # 用 raw_pred 做解析（JSON 不能 normalize）
        correct, is_binary, parsed_answer, is_parsable = evaluate_answer(
            gt["answer"], raw_pred, gt.get("image_width"), gt.get("image_height")
        )

        category = gt["category"]
        if not is_parsable:
            results[category]["illformed_responses"] += 1
        results[category]["total"] += 1
        if correct:
            num_correct += 1
            results[category]["num_correct"] += 1

        pred["label"] = gt["answer"]
        pred["answer"] = raw_pred  # 存回原始答案（或你也可以存提取后的 JSON）

        # v2 同样传 raw_pred
        correct_v2 = evaluate_answer_v2(gt, raw_pred)
        pred["correct"] = True if correct_v2 > 0.5 else False
        num_correct_v2 += correct_v2
        results_v2[category]["total"] += 1
        results_v2[category]["num_correct"] += correct_v2

    for category, result in results.items():
        result["accuracy"] = round(result["num_correct"] / result["total"] * 100, 4)
    for category, result in results_v2.items():
        result["accuracy"] = round(result["num_correct"] / result["total"] * 100, 4)
    final_results = {}
    final_results["accuracy_ori"] = round(num_correct_v2 / len(predictions) * 100, 4)
    final_results["accuracy"] = round(num_correct / len(predictions) * 100, 4)
    final_results["results_ori"] = results
    final_results["results_v2"] = results_v2
    return final_results


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

