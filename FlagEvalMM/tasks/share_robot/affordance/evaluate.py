from ast import literal_eval
from typing import Dict, List


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: A list / dict / string representing the first box [x1, y1, x2, y2]
        box2: A list / dict / string representing the second box [x1, y1, x2, y2]

    Returns:
        float: IoU value between 0.0 and 1.0
    """
    if isinstance(box1, str):
        box1 = literal_eval(box1)
    if isinstance(box2, str):
        box2 = literal_eval(box2)

    if isinstance(box1, dict):
        box1 = box1.get("bbox_2d", [])
    if isinstance(box2, dict):
        box2 = box2.get("bbox_2d", [])

    if len(box1) != 4 or len(box2) != 4:
        raise ValueError("Bounding boxes must be in [x1, y1, x2, y2] format")

    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    # No intersection (or just touching edge)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0

    return intersection / union


def calculate_metrics(data):
    """
    All scores are returned in 0-100 scale.
    """
    iou_thresholds = [0.5, 0.75, 0.9]

    metrics = {
        "average_iou": 0.0,  # 0-100
        "iou_at_thresholds": {str(thresh): 0.0 for thresh in iou_thresholds},  # 0-100
        "average_precision": 0.0,  # 0-100 (still "mean IoU" in your simplified definition)
        "valid_items": 0,
    }

    for item in data:
        try:
            pred_box = item["answer"]
            gt_box = item["gt"]

            iou = calculate_iou(pred_box, gt_box)  # 0-1

            # store per-sample IoU in 0-100
            item["iou"] = round(iou * 100, 2)

            metrics["average_iou"] += iou
            metrics["average_precision"] += iou  # still simplified (mean IoU)

            for thresh in iou_thresholds:
                metrics["iou_at_thresholds"][str(thresh)] += (iou >= thresh)

            metrics["valid_items"] += 1

        except (ValueError, KeyError, SyntaxError) as e:
            print(f"Skipping invalid item: {e}")

    if metrics["valid_items"] > 0:
        n = metrics["valid_items"]
        metrics["average_iou"] = round((metrics["average_iou"] / n) * 100, 2)
        metrics["average_precision"] = round((metrics["average_precision"] / n) * 100, 2)

        for thresh in iou_thresholds:
            metrics["iou_at_thresholds"][str(thresh)] = round(
                (metrics["iou_at_thresholds"][str(thresh)] / n) * 100, 2
            )

    return metrics


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    result = []
    for pred in predictions:
        question_id = str(pred["question_id"])
        gt_data = annotations[question_id]
        pred["gt"] = gt_data["answer"]
        pred["width"] = gt_data["width"]
        pred["height"] = gt_data["height"]
        result.append(pred)

    return calculate_metrics(result)

