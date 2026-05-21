import re
import math
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import ast

import numpy as np

try:
    from pycocotools import mask as cocomask
except Exception as e:
    raise RuntimeError(
        "缺少依赖 pycocotools（用于 RLE 解码 / polygon->mask）。请先安装：pip install pycocotools\n"
        f"原始错误：{repr(e)}"
    )

# ============================================================
# Answer parsing: model outputs polygons as
# [[(x1,y1),(x2,y2),...], [...]], coords in 0~1000 (relative)
# Text may contain <answer>...</answer>
# ============================================================
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def extract_answer(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else text.strip()

def _pairwise(nums: List[float]) -> List[Tuple[float, float]]:
    pts = []
    for i in range(0, len(nums) - 1, 2):
        pts.append((nums[i], nums[i + 1]))
    return pts

def parse_polygons_from_text(ans: Any) -> Optional[List[List[Tuple[float, float]]]]:
    """
    Return: List[polygon], each polygon is List[(x,y)] in 0~1000 (float).
    Robust strategy for your prompt-controlled output:

    Accept:
      - string containing: [[(x,y),(x,y)...], [...]]
      - also tolerates [[x,y,x,y,...], ...] / [x,y,x,y,...] by pairing numbers
      - also tolerates JSON-ish list (but you said it's tuples; we treat as plain text anyway)
      - if multiple bracket blocks exist, pick the block that yields the most points
    """
    if ans is None:
        return None

    # direct list object (already parsed upstream)
    if isinstance(ans, list):
        # could be [[(x,y)...], ...] or [[x,y,...], ...] or [x,y,...]
        if len(ans) == 0:
            return None

        # [x,y,x,y,...]
        if all(isinstance(v, (int, float)) for v in ans) and len(ans) >= 6:
            pts = _pairwise([float(v) for v in ans])
            return [pts] if len(pts) >= 3 else None

        polys: List[List[Tuple[float, float]]] = []
        for item in ans:
            if isinstance(item, list):
                # [(x,y)...] as list of pairs OR [x,y,...]
                if len(item) >= 3 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in item):
                    pts = [(float(p[0]), float(p[1])) for p in item]
                    if len(pts) >= 3:
                        polys.append(pts)
                elif all(isinstance(v, (int, float)) for v in item) and len(item) >= 6:
                    pts = _pairwise([float(v) for v in item])
                    if len(pts) >= 3:
                        polys.append(pts)
        return polys if polys else None

    # dict: try common keys
    if isinstance(ans, dict):
        for k in ("polygons", "polygon", "segmentation_polygon", "answer"):
            if k in ans:
                return parse_polygons_from_text(ans[k])
        return None

    # string
    if not isinstance(ans, str):
        return None

    s = extract_answer(ans)
    # 先尝试把 "[[(x,y)...], [(x,y)...]]" 直接变成 Python list
    try:
        obj = ast.literal_eval(s)
        # 如果成功解析成 list，就走你已经写好的 list 分支（稳定支持多 polygon）
        if isinstance(obj, list):
            return parse_polygons_from_text(obj)
    except Exception:
        pass

    # Find top-level bracket blocks like "[...]" (non-nested heuristic)
    # We'll just scan all and pick the one with most numbers -> most likely the polygon list.
    blocks = re.findall(r"\[[\s\S]*?\]", s)
    cand_strs = blocks if blocks else [s]

    best: Optional[List[List[Tuple[float, float]]]] = None
    best_pts = 0

    for cs in cand_strs:
        nums = [float(x) for x in NUM_RE.findall(cs)]
        if len(nums) < 6:
            continue

        # Heuristic: split into polygons by detecting "][" boundaries if present,
        # else treat as single polygon.
        # Because regex blocks may already isolate, we do a lightweight parse:
        # - if cs contains "],", try to split into sublists
        polys: List[List[Tuple[float, float]]] = []

        # try split on "],"
        parts = re.split(r"\]\s*,\s*\[", cs.strip())
        if len(parts) > 1:
            for p in parts:
                nn = [float(x) for x in NUM_RE.findall(p)]
                pts = _pairwise(nn)
                if len(pts) >= 3:
                    polys.append(pts)
        else:
            pts = _pairwise(nums)
            if len(pts) >= 3:
                polys = [pts]

        npts = sum(len(p) for p in polys)
        if npts > best_pts:
            best_pts = npts
            best = polys

    return best


# ============================================================
# Mask conversions: GT(RLE/polygon) and Pred(polygon->mask)
# ============================================================
def _squeeze_mask(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m)
    if m.ndim == 3:
        if m.shape[0] == 1:
            m = m[0]
        elif m.shape[-1] == 1:
            m = m[..., 0]
    return (m > 0).astype(np.uint8)

def decode_coco_rle_to_mask(rle_obj: Dict[str, Any]) -> np.ndarray:
    counts = rle_obj["counts"]
    size = rle_obj["size"]
    if isinstance(counts, list):
        rle = {"counts": counts, "size": size}
    else:
        rle = {"counts": counts.encode("utf-8"), "size": size}
    m = cocomask.decode(rle)
    return _squeeze_mask(m)

def polygons_1000_to_polygons_px(polys_1000: List[List[Tuple[float, float]]], W: int, H: int) -> List[List[float]]:
    """
    Convert [(x,y)] in 0~1000 to COCO polygon format [x1,y1,x2,y2,...] in PIXELS.
    """
    out: List[List[float]] = []
    for poly in polys_1000:
        if not poly or len(poly) < 3:
            continue
        flat: List[float] = []
        for x1000, y1000 in poly:
            x = float(x1000) * W / 1000.0
            y = float(y1000) * H / 1000.0
            # clamp a bit
            x = max(0.0, min(float(W), x))
            y = max(0.0, min(float(H), y))
            flat.extend([x, y])
        if len(flat) >= 6:
            out.append(flat)
    return out

def polygons_px_to_mask(polys_px: List[List[float]], H: int, W: int) -> np.ndarray:
    if not polys_px:
        return np.zeros((H, W), dtype=np.uint8)
    rles = cocomask.frPyObjects(polys_px, H, W)
    rle = rles if isinstance(rles, dict) else cocomask.merge(rles)
    m = cocomask.decode(rle)
    return _squeeze_mask(m)

def build_gt_masks_both(sample: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int]:
    """
    OneThinker里评测实际用 RLE 解码（像素级最准确）。
    Return: (gt_mask_rle, gt_mask_poly, H, W)
      - gt_mask_rle: from seg_out['rle'] (string counts)
      - gt_mask_poly: from seg_out['polygon'] (list of polygons)
    Either can be None if not available.
    """
    seg_out = sample.get("answer", {}) or {}
    W = int(sample.get("image_width", 0))
    H = int(sample.get("image_height", 0))

    gt_rle = None
    gt_poly = None

    rle = seg_out.get("rle", None)
    if isinstance(rle, str) and len(rle) > 0:
        rle_obj = {"counts": rle, "size": [H, W]}
        gt_rle = decode_coco_rle_to_mask(rle_obj)

    poly = seg_out.get("polygon", None)
    if isinstance(poly, list) and len(poly) > 0:
        polys_px = []
        for p in poly:
            if isinstance(p, list) and len(p) >= 6:
                polys_px.append([float(v) for v in p])
        if polys_px:
            gt_poly = polygons_px_to_mask(polys_px, H, W)

    return gt_rle, gt_poly, H, W


# ============================================================
# Metrics: EXACT same as OneThinker seg evaluator snippet
# ============================================================
def compute_inter_union(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[int, int]:
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return inter, union

def compute_iou_mask(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter, union = compute_inter_union(pred_mask, gt_mask)
    return float(inter) / float(union) if union > 0 else 0.0

def _neighbors_sum(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape
    s = np.zeros_like(binary, dtype=np.uint16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            y0 = max(0, dy); y1 = h + min(0, dy)
            x0 = max(0, dx); x1 = w + min(0, dx)
            s[y0:y1, x0:x1] += binary[y0-dy:y1-dy, x0-dx:x1-dx]
    return s

def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    neigh = _neighbors_sum(m)
    boundary = (m == 1) & (neigh < 8)
    return boundary.astype(np.uint8)

def _dilate(binary: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return (binary > 0).astype(np.uint8)
    h, w = binary.shape
    out = np.zeros_like(binary, dtype=np.uint8)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            y0 = max(0, dy); y1 = h + min(0, dy)
            x0 = max(0, dx); x1 = w + min(0, dx)
            out[y0:y1, x0:x1] |= binary[y0-dy:y1-dy, x0-dx:x1-dx]
    return out

def boundary_fscore(pred_mask: np.ndarray, gt_mask: np.ndarray, tau_ratio: float = 0.0075) -> float:
    pm = (pred_mask > 0).astype(np.uint8)
    gm = (gt_mask   > 0).astype(np.uint8)
    if pm.sum() == 0 and gm.sum() == 0:
        return 1.0
    if pm.sum() == 0 or gm.sum() == 0:
        return 0.0
    h, w = pm.shape
    r = max(1, int(round(tau_ratio * math.hypot(h, w))))
    pb = _extract_boundary(pm)
    gb = _extract_boundary(gm)
    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0
    pb_d = _dilate(pb, r)
    gb_d = _dilate(gb, r)
    tp_p = (pb & gb_d).sum()
    tp_g = (gb & pb_d).sum()
    prec = float(tp_p) / float(pb.sum()) if pb.sum() > 0 else 0.0
    rec  = float(tp_g) / float(gb.sum()) if gb.sum() > 0 else 0.0
    if (prec + rec) == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


# ============================================================
# get_result_seg: style aligned with your bbox get_result()
# - writes per-pred fields: raw_answer, reward, status
# - returns dict with avg_rewards (OneThinker-consistent keys)
# ============================================================
def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    """
    annotations: mapping[question_id(str)] -> GT sample dict (OneThinker seg json item)
    predictions: list of dicts containing:
      - question_id
      - answer (string)
        where answer is polygon list of tuples in 0~1000 coords

    This function mutates predictions to store:
      - raw_answer
      - reward
      - status
    Returns:
      {
        "avg_rewards": {...},
        "ok": ...,
        "total": ...,
        "accuracy": ...,
      }
    """
    ok_items = []
    video_J, video_F, video_JF = [], [], []
    image_IoU = []
    total_inter, total_union = 0, 0

    for pred in predictions:
        question_id = str(pred["question_id"])
        gt = annotations.get(question_id)
        pred["raw_answer"] = pred.get("answer")

        if gt is None:
            pred["status"] = "missing-gt"
            pred["reward"] = None
            continue

        data_type = gt.get("data_type", "")

        try:
            if data_type == "image":
                gt_rle_mask, gt_poly_mask, H, W = build_gt_masks_both(gt)

                pred_polys_1000 = parse_polygons_from_text(pred.get("answer"))
                if pred_polys_1000 is None:
                    pred_mask = np.zeros((H, W), dtype=np.uint8)
                else:
                    polys_px = polygons_1000_to_polygons_px(pred_polys_1000, W, H)
                    pred_mask = polygons_px_to_mask(polys_px, H, W)

                # ---- compute metrics for both GT types ----
                reward = {}

                # RLE-GT metrics
                if gt_rle_mask is not None:
                    inter_rle, union_rle = compute_inter_union(pred_mask, gt_rle_mask)
                    iou_rle = float(inter_rle) / float(union_rle) if union_rle > 0 else 0.0
                    reward["IoU_rle"] = iou_rle
                    reward["inter_rle"] = int(inter_rle)
                    reward["union_rle"] = int(union_rle)
                else:
                    reward["IoU_rle"] = None
                    reward["inter_rle"] = 0
                    reward["union_rle"] = 0

                # Polygon-GT metrics
                if gt_poly_mask is not None:
                    inter_poly, union_poly = compute_inter_union(pred_mask, gt_poly_mask)
                    iou_poly = float(inter_poly) / float(union_poly) if union_poly > 0 else 0.0
                    reward["IoU_poly"] = iou_poly
                    reward["inter_poly"] = int(inter_poly)
                    reward["union_poly"] = int(union_poly)
                else:
                    reward["IoU_poly"] = None
                    reward["inter_poly"] = 0
                    reward["union_poly"] = 0

                # ---- choose the "main" IoU for aggregation (OneThinker: prefer RLE) ----
                if gt_rle_mask is not None:
                    reward["IoU"] = reward["IoU_rle"]
                    reward["inter"] = reward["inter_rle"]
                    reward["union"] = reward["union_rle"]
                elif gt_poly_mask is not None:
                    reward["IoU"] = reward["IoU_poly"]
                    reward["inter"] = reward["inter_poly"]
                    reward["union"] = reward["union_poly"]
                else:
                    reward["IoU"] = 0.0
                    reward["inter"] = 0
                    reward["union"] = 0

                pred["status"] = "ok"
                pred["reward"] = reward
                ok_items.append(pred)

                # ---- aggregation uses the "main" IoU (keeps OneThinker consistency) ----
                image_IoU.append(float(reward["IoU"]))
                total_inter += int(reward["inter"])
                total_union += int(reward["union"])

            elif data_type == "video":
                seg_out = gt.get("segmentation_output", {}) or {}
                frames_list: List[str] = seg_out.get("frames", []) or []
                rle_map: Dict[str, Any] = seg_out.get("segmentation_rle", {}) or {}

                if not frames_list:
                    pred["status"] = "no-frames-list"
                    pred["reward"] = None
                    continue

                # determine H,W (prefer rle size)
                reso = gt.get("resolution", {}) or {}
                W = int(reso.get("width", 0))
                H = int(reso.get("height", 0))
                if rle_map:
                    any_rle = next(iter(rle_map.values()))
                    if isinstance(any_rle, dict) and "size" in any_rle and len(any_rle["size"]) == 2:
                        H, W = int(any_rle["size"][0]), int(any_rle["size"][1])

                # Parse per-frame prediction.
                # Expected simplest: dict keyed by frame_key -> polygons (same format as image answer).
                # If user gives a single polygon answer, we will broadcast to all frames as fallback.
                pred_ans = pred.get("answer")
                pred_obj = pred_ans
                if isinstance(pred_ans, str):
                    s = extract_answer(pred_ans)
                    pred_obj = None
                    # Try literal eval-ish by number extraction fallback:
                    # We'll just keep string and parse_polygons_from_text(frame-specific) later if needed.
                    pred_obj = s

                per_frame_polys_1000: Dict[int, Optional[List[List[Tuple[float, float]]]]] = {}

                if isinstance(pred_obj, dict):
                    key2idx = {k: i for i, k in enumerate(frames_list)}
                    for fk, v in pred_obj.items():
                        if fk in key2idx:
                            per_frame_polys_1000[key2idx[fk]] = parse_polygons_from_text(v)
                elif isinstance(pred_obj, list) and len(pred_obj) == len(frames_list):
                    for i, v in enumerate(pred_obj):
                        per_frame_polys_1000[i] = parse_polygons_from_text(v)
                else:
                    # broadcast single polygon text to all frames
                    one = parse_polygons_from_text(pred.get("answer"))
                    for i in range(len(frames_list)):
                        per_frame_polys_1000[i] = one

                Js, Fs = [], []
                for order, frame_key in enumerate(frames_list):
                    rle = rle_map.get(frame_key, None)
                    if not rle:
                        continue
                    gt_mask = decode_coco_rle_to_mask(rle)

                    polys_1000 = per_frame_polys_1000.get(order, None)
                    if polys_1000 is None:
                        pred_mask = np.zeros((H, W), dtype=np.uint8)
                    else:
                        polys_px = polygons_1000_to_polygons_px(polys_1000, W, H)
                        pred_mask = polygons_px_to_mask(polys_px, H, W)

                    j = compute_iou_mask(pred_mask, gt_mask)
                    f = boundary_fscore(pred_mask, gt_mask, tau_ratio=0.0075)
                    Js.append(j)
                    Fs.append(f)

                if Js:
                    J_mean = float(np.mean(Js))
                    F_mean = float(np.mean(Fs))
                    JF_mean = float((J_mean + F_mean) / 2.0)
                else:
                    J_mean = F_mean = JF_mean = 0.0

                reward = {"J": J_mean, "F": F_mean, "J&F": JF_mean}
                pred["status"] = "ok"
                pred["reward"] = reward
                ok_items.append(pred)

                video_J.append(J_mean)
                video_F.append(F_mean)
                video_JF.append(JF_mean)

            else:
                pred["status"] = "skip-unknown-data_type"
                pred["reward"] = None

        except Exception as e:
            pred["status"] = f"error: {repr(e)}"
            pred["reward"] = None

    avg_rewards: Dict[str, float] = {}
    if video_J:
        avg_rewards["video/J"] = float(np.mean(video_J))
        avg_rewards["video/F"] = float(np.mean(video_F))
        avg_rewards["video/J&F"] = float(np.mean(video_JF))
    if image_IoU:
        giou_val = float(np.mean(image_IoU))
        avg_rewards["image/IoU"] = giou_val
        avg_rewards["image/gIoU"] = giou_val
        avg_rewards["image/cIoU"] = (float(total_inter) / float(total_union)) if total_union > 0 else 0.0

    return {
        "avg_rewards": avg_rewards,
        "ok": len(ok_items),
        "total": len(predictions),
        "accuracy": avg_rewards["image/cIoU"]
    }
