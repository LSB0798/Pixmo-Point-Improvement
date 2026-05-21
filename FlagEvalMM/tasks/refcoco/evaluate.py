import re
import json
from typing import Dict, List, Any
from collections import defaultdict


COCO_REC_METRICS = [
    "sIoU",        # main metric = mean IoU (sIoU)
    "IoU",
    "ACC@0.1",
    "ACC@0.3",
    "ACC@0.5",
    "ACC@0.7",
    "ACC@0.9",
    "Center_ACC",
]

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def extract_answer(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def parse_float_sequence_within(input_str: str) -> List[float]:
    if not isinstance(input_str, str):
        return [0, 0, 0, 0]
    pattern = r"[\(\[]?\s*(-?\d+(?:\.\d+)?)\s*,?\s*(-?\d+(?:\.\d+)?)\s*,?\s*(-?\d+(?:\.\d+)?)\s*,?\s*(-?\d+(?:\.\d+)?)\s*[\)\]]?"
    match = re.search(pattern, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    return [0, 0, 0, 0]


def _as_float4(x: Any) -> List[float]:
    try:
        if isinstance(x, (list, tuple)) and len(x) == 4:
            return [float(v) for v in x]
    except Exception:
        pass
    return [0.0, 0.0, 0.0, 0.0]


def extract_bbox_raw(ans: Any) -> List[float]:
    """
    Extract bbox [x1,y1,x2,y2] from:
      - list/tuple
      - dict {"boxes":[...]} or {"box":[...]} or {"bbox":[...]}
      - string: <answer>JSON</answer>, JSON, or free text containing 4 numbers
    Returns raw numbers without any scale conversion.
    """
    if isinstance(ans, (list, tuple)) and len(ans) == 4:
        return _as_float4(ans)

    if isinstance(ans, dict):
        for key in ("boxes", "box", "bbox"):
            b = ans.get(key)
            if isinstance(b, (list, tuple)) and len(b) == 4:
                return _as_float4(b)
        return _as_float4(parse_float_sequence_within(json.dumps(ans, ensure_ascii=False)))

    if isinstance(ans, str):
        s = extract_answer(ans)
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                for key in ("boxes", "box", "bbox"):
                    b = obj.get(key)
                    if isinstance(b, list) and len(b) == 4:
                        return _as_float4(b)
            if isinstance(obj, list) and len(obj) == 4:
                return _as_float4(obj)
        except Exception:
            pass
        return _as_float4(parse_float_sequence_within(s))

    return [0.0, 0.0, 0.0, 0.0]


def gt_1000_to_pixel(gt_box_1000: List[float], image_width: int, image_height: int) -> List[float]:
    """
    Convert GT box from 0~1000 coordinate system to absolute pixel coords.
    Assumption (common in Qwen VL datasets):
      x in [0,1000] maps to [0, image_width]
      y in [0,1000] maps to [0, image_height]
    """
    b = _as_float4(gt_box_1000)
    w = float(image_width) if image_width else 0.0
    h = float(image_height) if image_height else 0.0
    if w <= 0 or h <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [b[0] * w / 1000.0, b[1] * h / 1000.0, b[2] * w / 1000.0, b[3] * h / 1000.0]


def compute_iou(box1, box2, eps=1e-12) -> float:
    """
    IoU aligned with vLLM script style:
      - no swapping/repairing
      - negative widths/heights become 0 via max(0, ...)
    """
    if not isinstance(box1, (list, tuple)) or not isinstance(box2, (list, tuple)):
        return 0.0
    if len(box1) != 4 or len(box2) != 4:
        return 0.0

    x1, y1, x2, y2 = map(float, box1)
    X1, Y1, X2, Y2 = map(float, box2)

    inter_x1, inter_y1 = max(x1, X1), max(y1, Y1)
    inter_x2, inter_y2 = min(x2, X2), min(y2, Y2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, X2 - X1) * max(0.0, Y2 - Y1)
    union = area1 + area2 - inter_area

    return inter_area / union if union > eps else 0.0


def compute_siou(gt_box_pixel, pred_box_pixel) -> float:
    return compute_iou(gt_box_pixel, pred_box_pixel)


def compute_accuracy(gt_box_pixel, pred_box_pixel, threshold=0.5) -> bool:
    return compute_iou(gt_box_pixel, pred_box_pixel) >= threshold


def compute_center_accuracy(gt_box_pixel, pred_box_pixel) -> bool:
    center_x = (pred_box_pixel[0] + pred_box_pixel[2]) / 2
    center_y = (pred_box_pixel[1] + pred_box_pixel[3]) / 2
    return (
        gt_box_pixel[0] <= center_x <= gt_box_pixel[2]
        and gt_box_pixel[1] <= center_y <= gt_box_pixel[3]
    )


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    results = defaultdict(lambda: {"num": 0, "correct": 0})

    scorers = {
        "ACC@0.1": lambda g, p: compute_accuracy(g, p, 0.1),
        "ACC@0.3": lambda g, p: compute_accuracy(g, p, 0.3),
        "ACC@0.5": lambda g, p: compute_accuracy(g, p, 0.5),
        "ACC@0.7": lambda g, p: compute_accuracy(g, p, 0.7),
        "ACC@0.9": lambda g, p: compute_accuracy(g, p, 0.9),
        "Center_ACC": compute_center_accuracy,
    }

    siou_sum = 0.0
    siou_n = 0

    for pred in predictions:
        question_id = str(pred["question_id"])
        gt = annotations[question_id]

        # ---- GT: parse 0~1000 boxes from string, then convert to pixel coords ----
        gt_box_1000 = extract_bbox_raw(gt["answer"])
        gt_box = gt_1000_to_pixel(gt_box_1000, gt.get("image_width", 0), gt.get("image_height", 0))

        # ---- Pred: model outputs absolute pixel coords (keep as-is) ----
        pred["raw_answer"] = pred.get("answer")
        pred_box = extract_bbox_raw(pred.get("answer"))

        pred["img_path"] = gt.get("img_path")
        pred["label"] = gt_box
        pred["answer_box"] = pred_box

        # threshold metrics
        for scorer_name, scorer in scorers.items():
            ok = scorer(gt_box, pred_box)
            results[scorer_name]["num"] += 1
            results[scorer_name]["correct"] += int(ok)

        # sIoU / IoU per sample
        pred["sIoU"] = compute_siou(gt_box, pred_box)
        pred["IOU"] = pred["sIoU"]
        pred["correct"] = pred["sIoU"] >= 0.5

        siou_sum += float(pred["sIoU"])
        siou_n += 1

        # keep old avg bucket (acc@0.5)
        results["avg"]["num"] += 1
        results["avg"]["correct"] += int(pred["sIoU"] >= 0.5)

    # finalize threshold metrics (%)
    for metric_name, r in results.items():
        if "num" in r and r["num"] > 0 and "correct" in r:
            r["accuracy"] = round(r["correct"] / r["num"] * 100, 2)

    # main metric: mean sIoU (%)
    mean_siou = siou_sum / siou_n if siou_n > 0 else 0.0
    results["sIoU"] = {"num": siou_n, "mean": mean_siou, "accuracy": round(mean_siou * 100, 2)}
    results["accuracy"] = results["sIoU"]["accuracy"]

    return results
