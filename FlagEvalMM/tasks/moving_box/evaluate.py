import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------- 工具函数（与原脚本精神一致） -----------------
def extract_json(text: str) -> Optional[str]:
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


def normalize_stance(s: Any) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    return s if s in {"left", "middle", "right"} else None


def normalize_side(s: Any) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    return s if s in {"current", "left", "right", "opposite"} else None


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


def _parse_pred_obj(answer: Any) -> Optional[Dict[str, Any]]:
    """预测 answer 可能是 dict，也可能是包含 JSON 的字符串。"""
    if isinstance(answer, dict):
        return answer
    if isinstance(answer, str):
        js = extract_json(answer)
        if not js:
            return None
        try:
            obj = json.loads(js)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _to_abs_xyxy(box: List[Any], w: int, h: int) -> Optional[List[int]]:
    """
    尽量鲁棒地把 bbox 转成像素坐标：
    - 若值都在 [0,1] 附近 -> 视为归一化
    - 若 max<=1000 -> 视为 0~1000 标尺
    - 否则视为像素
    """
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        vals = [float(x) for x in box]
    except Exception:
        return None

    maxv = max(vals)
    minv = min(vals)

    # normalized
    if minv >= -1e-6 and maxv <= 1:
        x1 = vals[0] * w
        y1 = vals[1] * h
        x2 = vals[2] * w
        y2 = vals[3] * h
    # 0~1000 scale (Qwen3VL style)
    elif minv >= -1e-6 and maxv <= 1000 + 1e-6:
        x1 = vals[0] / 1000.0 * w
        y1 = vals[1] / 1000.0 * h
        x2 = vals[2] / 1000.0 * w
        y2 = vals[3] / 1000.0 * h
    # pixel
    else:
        x1, y1, x2, y2 = vals

    # clamp
    x1 = max(0, min(int(round(x1)), w))
    x2 = max(0, min(int(round(x2)), w))
    y1 = max(0, min(int(round(y1)), h))
    y2 = max(0, min(int(round(y2)), h))

    return [x1, y1, x2, y2]


# ----------------- FlagEvalMM 标准入口 -----------------
def get_result(annotations: Dict[str, Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    annotations: {question_id: gt_item}
      gt_item 至少包含:
        - answer: {target_bbox_2d, side_flag, stance_flag}
        - image_width, image_height
        - img_path
    predictions: [{question_id, answer, ...}, ...]
    """
    iou_thr = 0.5

    n_total = 0
    n_valid = 0
    json_ok = 0
    json_fail = 0

    select_loc_correct = 0
    select_side_correct = 0
    select_stance_correct = 0
    select_all_correct = 0

    iou_sum = 0.0
    iou_count = 0
    print(f'annotations size: {len(annotations)}, predictions size: {len(predictions)}')
    print(f'annnotations: {list(annotations.items())[:2]}')
    print(f'predictions: {predictions[:2]}')

    # 建立 pred_map 方便对齐
    pred_map = {}
    for p in predictions:
        qid = str(p.get("question_id"))
        if qid:
            pred_map[qid] = p

    for qid, gt in annotations.items():
        n_total += 1
        pred = pred_map.get(str(qid))
        if pred is None:
            json_fail += 1
            continue

        gt_answer = gt.get("answer") if isinstance(gt, dict) else None
        if not isinstance(gt_answer, dict):
            json_fail += 1
            continue

        w = int(gt.get("image_width") or 0)
        h = int(gt.get("image_height") or 0)
        if w <= 0 or h <= 0:
            json_fail += 1
            continue

        pred_obj = _parse_pred_obj(pred.get("answer"))
        if pred_obj is None:
            json_fail += 1
            continue
        json_ok += 1

        gt_box_raw = gt_answer.get("target_bbox_2d")
        pr_box_raw = pred_obj.get("target_bbox_2d")

        gt_box = _to_abs_xyxy(gt_box_raw, w, h) if isinstance(gt_box_raw, list) else None
        pr_box = _to_abs_xyxy(pr_box_raw, w, h) if isinstance(pr_box_raw, list) else None

        if gt_box is None or pr_box is None:
            continue

        n_valid += 1

        iou_val = iou_xyxy(tuple(gt_box), tuple(pr_box))
        iou_sum += iou_val
        iou_count += 1

        gt_side = normalize_side(gt_answer.get("side_flag"))
        gt_stance = normalize_stance(gt_answer.get("stance_flag"))
        pr_side = normalize_side(pred_obj.get("side_flag"))
        pr_stance = normalize_stance(pred_obj.get("stance_flag"))

        if iou_val >= iou_thr:
            select_loc_correct += 1

            side_ok = gt_side is not None and pr_side is not None and gt_side == pr_side
            stance_ok = gt_stance is not None and pr_stance is not None and gt_stance == pr_stance

            if side_ok:
                select_side_correct += 1
            if stance_ok:
                select_stance_correct += 1
            if side_ok and stance_ok:
                select_all_correct += 1

    denom = n_valid if n_valid > 0 else 1

    select_loc_acc = select_loc_correct / denom
    select_side_acc = select_side_correct / denom
    select_stance_acc = select_stance_correct / denom
    select_all_acc = select_all_correct / denom
    mean_iou = iou_sum / iou_count if iou_count else 0.0

    # 按 FlagEvalMM 常见风格返回
    results = {
        "Select_Loc_ACC@0.5": {
            "num": denom,
            "correct": select_loc_correct,
            "accuracy": round(select_loc_acc * 100, 2),
        },
        "Select_Side_ACC": {
            "num": denom,
            "correct": select_side_correct,
            "accuracy": round(select_side_acc * 100, 2),
        },
        "Select_Stance_ACC": {
            "num": denom,
            "correct": select_stance_correct,
            "accuracy": round(select_stance_acc * 100, 2),
        },
        "Select_All_ACC": {
            "num": denom,
            "correct": select_all_correct,
            "accuracy": round(select_all_acc * 100, 2),
        },
        "Mean_IoU": round(mean_iou, 4),
        "JSON_Parse_Success_Rate": round((json_ok / (n_total if n_total else 1)) * 100, 2),
    }

    # 主指标
    results["accuracy"] = results["Select_All_ACC"]["accuracy"]
    return results


# ----------------- 下面是可选：保留原脚本的“目录式可视化能力” -----------------
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


def visualize_from_processed(
    processed_data_json: str,
    pred_dir: str,
    save_dir: str,
    iou_thr: float = 0.5,
):
    """
    基于 FlagEvalMM 处理后的 data.json + 你保存的逐 question_id JSON 预测文件
    输出 true/false 可视化
    """
    cv2 = try_import_cv2()
    if cv2 is None:
        print("[WARN] 未安装 OpenCV，跳过可视化。")
        return

    with open(processed_data_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    save_p = Path(save_dir)
    true_p = save_p / "true"
    false_p = save_p / "false"
    true_p.mkdir(parents=True, exist_ok=True)
    false_p.mkdir(parents=True, exist_ok=True)

    pred_p = Path(pred_dir)

    for item in data:
        qid = str(item["question_id"])
        img_rel = item["img_path"]
        img_abs = str(Path(processed_data_json).parent / img_rel)

        if not os.path.exists(img_abs):
            print(f"[WARN] 找不到图像: {img_abs}")
            continue

        gt_ans = item.get("answer", {})
        w = int(item.get("image_width") or 0)
        h = int(item.get("image_height") or 0)

        pred_file = pred_p / f"{qid}.json"
        if not pred_file.exists():
            out_dir_this = false_p
            pred_obj = None
        else:
            try:
                pred_obj = json.load(open(pred_file, "r", encoding="utf-8"))
            except Exception:
                pred_obj = None

        img = cv2.imread(img_abs, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] 图像读取失败: {img_abs}")
            continue

        gt_box = _to_abs_xyxy(gt_ans.get("target_bbox_2d"), w, h) if isinstance(gt_ans, dict) else None
        pr_box = None
        pr_side = pr_stance = None

        if isinstance(pred_obj, dict):
            pr_box = _to_abs_xyxy(pred_obj.get("target_bbox_2d"), w, h)
            pr_side = normalize_side(pred_obj.get("side_flag"))
            pr_stance = normalize_stance(pred_obj.get("stance_flag"))

        gt_side = normalize_side(gt_ans.get("side_flag")) if isinstance(gt_ans, dict) else None
        gt_stance = normalize_stance(gt_ans.get("stance_flag")) if isinstance(gt_ans, dict) else None

        # draw GT
        if gt_box:
            draw_box_cv2(img, *gt_box, (0, 255, 0), f"GT {gt_side}/{gt_stance}")

        iou_val = None
        if gt_box and pr_box:
            iou_val = iou_xyxy(tuple(gt_box), tuple(pr_box))

        # draw Pred
        if pr_box:
            label = f"Pred {pr_side}/{pr_stance}"
            if iou_val is not None:
                label += f" (IoU={iou_val:.2f})"
            draw_box_cv2(img, *pr_box, (0, 0, 255), label)

        all_correct = (
            gt_box is not None
            and pr_box is not None
            and iou_val is not None
            and iou_val >= iou_thr
            and gt_side is not None and pr_side is not None and gt_side == pr_side
            and gt_stance is not None and pr_stance is not None and gt_stance == pr_stance
        )

        out_dir_this = true_p if all_correct else false_p
        out_path = out_dir_this / f"{qid}_vis.png"
        cv2.imwrite(str(out_path), img)


# visualize_from_processed(
#     processed_data_json="/path/to/processed/val/data.json",
#     pred_dir="/path/to/preds",
#     save_dir="/path/to/vis_out",
#     iou_thr=0.5,
# )
