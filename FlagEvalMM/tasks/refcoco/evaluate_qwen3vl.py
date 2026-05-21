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

import re
import json
from typing import Any, List, Optional

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def extract_answer(text: Any) -> Any:
    print('-8-' * 10)
    if not isinstance(text, str):
        return text
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else text.strip()

def _as_float4(x: Any) -> List[float]:
    print('-9-' * 10)
    try:
        if isinstance(x, (list, tuple)) and len(x) == 4:
            return [float(v) for v in x]
    except Exception:
        pass
    return [0.0, 0.0, 0.0, 0.0]

def _points2_to_bbox4(x: Any) -> Optional[List[float]]:
    print('-10-' * 10)
    """
    Convert [[x1,y1],[x2,y2]] or [(x1,y1),(x2,y2)] to [x1,y1,x2,y2]
    """
    if isinstance(x, (list, tuple)) and len(x) == 2:
        p1, p2 = x[0], x[1]
        if isinstance(p1, (list, tuple)) and isinstance(p2, (list, tuple)) and len(p1) == 2 and len(p2) == 2:
            try:
                return [float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])]
            except Exception:
                return None
    return None

def parse_bbox_from_text(s: str) -> List[float]:
    print('-11-' * 10)
    """
    Robustly parse bbox from text like:
      - "[(587, 183), (796, 695)]"
      - "[587, 183, 796, 695]"
      - "x1=587 y1=183 x2=796 y2=695"
    Strategy:
      1) Prefer bracket blocks [...] that contain >=4 numbers (ignore [0,1000] which has 2 nums)
      2) Fallback: all numbers in whole string
    """
    if not isinstance(s, str):
        return [0.0, 0.0, 0.0, 0.0]

    # candidate blocks like "[ ... ]" (non-nested)
    blocks = re.findall(r"\[[^\[\]]*\]", s)
    candidates = []
    for b in blocks:
        nums = NUM_RE.findall(b)
        if len(nums) >= 4:
            candidates.append(nums)

    if candidates:
        # Prefer the candidate with the fewest numbers (usually exactly 4)
        nums = min(candidates, key=len)
    else:
        nums = NUM_RE.findall(s)

    if len(nums) >= 4:
        return [float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])]
    return [0.0, 0.0, 0.0, 0.0]

def extract_bbox_raw(ans: Any) -> List[float]:
    print('-12-' * 10)
    """
    Extract bbox [x1,y1,x2,y2] from:
      - [x1,y1,x2,y2]
      - [[x1,y1],[x2,y2]]
      - dict {"boxes"/"box"/"bbox": ...} and nested common variants
      - string: <answer>...</answer>, JSON, or free text
    """
    # list/tuple direct bbox
    if isinstance(ans, (list, tuple)):
        b4 = _as_float4(ans)
        if b4 != [0.0, 0.0, 0.0, 0.0]:
            return b4
        # list of 2 points
        p = _points2_to_bbox4(ans)
        if p is not None:
            return p
        # list of boxes e.g. [[x1,y1,x2,y2], ...] or [[[x1,y1],[x2,y2]], ...]
        if len(ans) > 0:
            first = ans[0]
            b4 = _as_float4(first)
            if b4 != [0.0, 0.0, 0.0, 0.0]:
                return b4
            p = _points2_to_bbox4(first)
            if p is not None:
                return p
        return [0.0, 0.0, 0.0, 0.0]

    # dict
    if isinstance(ans, dict):
        for key in ("boxes", "box", "bbox"):
            if key in ans:
                b = ans.get(key)
                # direct bbox
                b4 = _as_float4(b)
                if b4 != [0.0, 0.0, 0.0, 0.0]:
                    return b4
                # 2-point bbox
                p = _points2_to_bbox4(b)
                if p is not None:
                    return p
                # list of boxes
                if isinstance(b, list) and len(b) > 0:
                    first = b[0]
                    b4 = _as_float4(first)
                    if b4 != [0.0, 0.0, 0.0, 0.0]:
                        return b4
                    p = _points2_to_bbox4(first)
                    if p is not None:
                        return p

        # fallback: parse from dumped json text
        return parse_bbox_from_text(json.dumps(ans, ensure_ascii=False))

    # string
    if isinstance(ans, str):
        s = extract_answer(ans)

        # try json parse
        try:
            obj = json.loads(s)
            # recurse once: obj could be list/dict shapes above
            return extract_bbox_raw(obj)
        except Exception:
            pass

        # regex-based robust parse
        return parse_bbox_from_text(s)

    return [0.0, 0.0, 0.0, 0.0]

def sanitize_bbox(box: List[float], clamp_min=0.0, clamp_max=1000.0) -> List[float]:
    print('-13-' * 10)
    """
    Optional but recommended:
    - ensure x1<=x2, y1<=y2
    - clamp to [0,1000]
    """
    b = _as_float4(box)
    x1, y1, x2, y2 = b
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1 = max(clamp_min, min(clamp_max, x1))
    y1 = max(clamp_min, min(clamp_max, y1))
    x2 = max(clamp_min, min(clamp_max, x2))
    y2 = max(clamp_min, min(clamp_max, y2))
    return [x1, y1, x2, y2]



def normalize_bbox_to_1000(box: List[float], eps: float = 1e-6) -> List[float]:
    print('-14-' * 10)
    """
    Auto-handle scale:
      - If box looks like 0~1 (all coords in [-eps, 1+eps]) => multiply by 1000
      - Else treat as already 0~1000 => keep unchanged
    """
    b = _as_float4(box)
    mx = max(b)
    mn = min(b)

    # Heuristic: all in [~0,~1] => normalized coords
    if mn >= -eps and mx <= 1.0 + eps:
        # avoid turning an all-zero garbage box into "valid": but 0 stays 0 anyway
        return [v * 1000.0 for v in b]
    return b


def compute_iou(box1, box2, eps=1e-12) -> float:
    print('-15-' * 10)
    """
    IoU in vLLM-script style: no "swap fix", but clamp negative widths/heights to 0.
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


def compute_siou(gt_box, pred_box) -> float:
    print('-16-' * 10)
    return compute_iou(gt_box, pred_box)


def compute_accuracy(gt_box, pred_box, threshold=0.5) -> bool:
    print('-17-' * 10)
    return compute_iou(gt_box, pred_box) >= threshold


def compute_center_accuracy(gt_box, pred_box) -> bool:
    print('-18-' * 10)
    center_x = (pred_box[0] + pred_box[2]) / 2
    center_y = (pred_box[1] + pred_box[3]) / 2
    return gt_box[0] <= center_x <= gt_box[2] and gt_box[1] <= center_y <= gt_box[3]


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    print('-19-' * 10)
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

        # --- GT bbox: extract then normalize to 0~1000 ---
        gt_raw = extract_bbox_raw(gt["answer"])
        gt_box = sanitize_bbox(normalize_bbox_to_1000(gt_raw))

        # --- Pred bbox: may be list or json or <answer>json</answer>; extract then normalize to 0~1000 ---
        pred["raw_answer"] = pred.get("answer")
        pred_raw = extract_bbox_raw(pred.get("answer"))
        pred_box = sanitize_bbox(normalize_bbox_to_1000(pred_raw))

        pred["img_path"] = gt.get("img_path")
        pred["label"] = gt_box
        pred["answer_box"] = pred_box

        # threshold metrics
        for scorer_name, scorer in scorers.items():
            ok = scorer(gt_box, pred_box)
            results[scorer_name]["num"] += 1
            results[scorer_name]["correct"] += int(ok)

        # sIoU / IoU
        pred["sIoU"] = compute_siou(gt_box, pred_box)
        pred["IOU"] = pred["sIoU"]
        pred["correct"] = pred["sIoU"] >= 0.5

        siou_sum += float(pred["sIoU"])
        siou_n += 1

        # keep old avg bucket (acc@0.5)
        results["avg"]["num"] += 1
        results["avg"]["correct"] += int(pred["sIoU"] >= 0.5)

    # percent for threshold metrics
    for metric_name, r in results.items():
        if "num" in r and r["num"] > 0 and "correct" in r:
            r["accuracy"] = round(r["correct"] / r["num"] * 100, 2)

    # main metric: mean sIoU (%)
    # mean_siou = siou_sum / siou_n if siou_n > 0 else 0.0
    # results["sIoU"] = {"num": siou_n, "mean": mean_siou, "accuracy": round(mean_siou * 100, 2)}
    # results["accuracy"] = results["sIoU"]["accuracy"]
    
    # main metric: ACC@0.5
    results["accuracy"] = results["ACC@0.5"]["accuracy"]

    return results

