# evaluate.py
import json
from ast import literal_eval
from typing import Any, Dict, List


def normalize_state(s: Any) -> str | None:
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    if s in {"select", "approach"}:
        return s
    return None


def normalize_stance(s: Any) -> str | None:
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    if s in {"left", "middle", "right"}:
        return s
    return None


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


def extract_json(text: str):
    """简化版，从模型输出里抠出第一个 JSON 对象"""
    if not isinstance(text, str):
        return None
    import re

    text = text.strip()
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


def _parse_pred_answer(raw: Any) -> Dict[str, Any] | None:
    """
    将模型输出统一解析成 dict：
        {"state": "...", "target_bbox_2d": [...], "stance_flag": "..."}
    尽量兼容几种常见情况：
        - 已经是 dict
        - 纯列表 [x1,y1,x2,y2] -> 视为 state=select, 没有 stance
        - 字符串 JSON / 字符串里嵌 JSON / 字符串形式的 list/dict
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        if len(raw) == 4:
            return {"state": "select", "target_bbox_2d": raw}
        return None
    if not isinstance(raw, str):
        return None

    raw = raw.strip()

    # 1) 直接 JSON
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) 从中提取 JSON 子串
    js = extract_json(raw)
    if js:
        try:
            return json.loads(js)
        except Exception:
            pass

    # 3) literal_eval 兜底（比如 "[0, 0, 10, 10]"）
    try:
        v = literal_eval(raw)
        if isinstance(v, dict):
            return v
        if isinstance(v, list) and len(v) == 4:
            return {"state": "select", "target_bbox_2d": v}
    except Exception:
        pass

    return None


def get_result(
    annotations: Dict[str, Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    iou_thr: float = 0.5,
) -> Dict[str, Any]:
    """
    FlagEvalMM 的评测入口，复现 eval_unova.py 的简化评测逻辑：
    - 只关注 GT state == "select" 的样本
    - 统计：
        1) Select 定位准确率 (IoU >= iou_thr)
        2) Select+Stance 准确率 (IoU >= iou_thr 且 stance 一致)
        3) 平均 IoU (仅 GT=select 且 Pred=select)
    """
    # question_id -> prediction 映射
    pred_map: Dict[str, Dict[str, Any]] = {
        str(p["question_id"]): p for p in predictions
    }

    n_total = 0
    n_gt_select = 0
    invalid_gt = 0
    invalid_pred = 0

    select_localization_correct = 0  # IoU >= thr
    select_with_stance_correct = 0   # IoU >= thr 且 stance 正确

    iou_sum = 0.0
    iou_count = 0

    for qid, gt_data in annotations.items():
        n_total += 1

        gt = gt_data.get("answer")
        gt_state = (
            normalize_state(gt.get("state")) if isinstance(gt, dict) else None
        )

        pred_rec = pred_map.get(str(qid))
        pred_obj = _parse_pred_answer(pred_rec.get("answer")) if pred_rec else None
        pred_state = (
            normalize_state(pred_obj.get("state"))
            if isinstance(pred_obj, dict)
            else None
        )

        # === 只评估 GT=select 的样本 ===
        if gt_state != "select":
            continue

        n_gt_select += 1

        if gt_state is None:
            invalid_gt += 1
            continue

        if pred_obj is None or pred_state is None:
            invalid_pred += 1
            continue

        if pred_state == "select":
            gt_box = gt.get("target_bbox_2d") if isinstance(gt, dict) else None
            pr_box = (
                pred_obj.get("target_bbox_2d")
                if isinstance(pred_obj, dict)
                else None
            )

            if (
                isinstance(gt_box, (list, tuple))
                and len(gt_box) == 4
                and isinstance(pr_box, (list, tuple))
                and len(pr_box) == 4
            ):
                gx1, gy1, gx2, gy2 = map(int, gt_box)
                px1, py1, px2, py2 = map(int, pr_box)
                iou_val = iou_xyxy(
                    (gx1, gy1, gx2, gy2), (px1, py1, px2, py2)
                )

                iou_sum += iou_val
                iou_count += 1

                if iou_val >= iou_thr:
                    # 定位正确
                    select_localization_correct += 1

                    # 再看 stance
                    gt_stance = normalize_stance(
                        gt.get("stance_flag") if isinstance(gt, dict) else None
                    )
                    pr_stance = normalize_stance(
                        pred_obj.get("stance_flag")
                        if isinstance(pred_obj, dict)
                        else None
                    )
                    if gt_stance and pr_stance and gt_stance == pr_stance:
                        select_with_stance_correct += 1
        # pred_state == "approach" 的情况，视作定位失败（什么都不加）

    select_loc_acc = (
        select_localization_correct / n_gt_select if n_gt_select > 0 else 0.0
    )
    select_stance_acc = (
        select_with_stance_correct / n_gt_select if n_gt_select > 0 else 0.0
    )
    mean_iou = iou_sum / iou_count if iou_count > 0 else 0.0

    metrics = {
        "总样本数": n_total,
        "GT为select的样本数": n_gt_select,
        "无效GT数": invalid_gt,
        "无效预测数": invalid_pred,
        f"Select定位准确率(IoU>={iou_thr:.2f})": round(select_loc_acc, 4),
        "Select+Stance准确率": round(select_stance_acc, 4),
        "平均IoU(仅GT=select且Pred=select)": round(mean_iou, 4),
        "IoU统计样本数": iou_count,
        "配置": {"iou_threshold": iou_thr},
    }
    return metrics
