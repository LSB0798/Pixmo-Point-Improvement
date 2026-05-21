from typing import Dict, List, Tuple
from PIL import Image, ImageDraw
import numpy as np
import re
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


def _bbox_points(x0: int, y0: int, x1: int, y1: int, max_points: int = 2000) -> List[Tuple[int, int]]:
    """
    从 bbox 区域采样点，避免 bbox 很大时枚举所有像素。
    """
    if x1 <= x0 or y1 <= y0:
        return []
    area = (x1 - x0) * (y1 - y0)
    if area <= max_points:
        pts = [(x, y) for y in range(y0, y1) for x in range(x0, x1)]
        return pts

    # 采样：按步长网格取点
    stride = int(np.ceil(np.sqrt(area / max_points)))
    stride = max(1, stride)
    pts = []
    for y in range(y0, y1, stride):
        for x in range(x0, x1, stride):
            pts.append((x, y))
            if len(pts) >= max_points:
                return pts
    return pts


def text2pts(text: str, width: int, height: int) -> List[Tuple[int, int]]:
    """
    从模型输出里抽取点坐标，返回像素坐标 (x,y)
    支持：
      1) (x, y) / (x0, y0, x1, y1)
      2) <point x=".." y="..">
    """
    if text is None:
        return []

    # Answer 通常在最后一行
    text = text.strip().split("\n")[-1]

    points: List[Tuple[int, int]] = []

    # 1) <point x=".." y="..">
    # 允许单引号/双引号/无引号
    p_pat = r"<point[^>]*x\s*=\s*['\"]?([0-9]+(?:\.[0-9]+)?)['\"]?[^>]*y\s*=\s*['\"]?([0-9]+(?:\.[0-9]+)?)['\"]?"
    for mx, my in re.findall(p_pat, text, flags=re.IGNORECASE):
        x = float(mx)
        y = float(my)
        px, py = _to_pixel_xy(x, y, width, height)
        points.append((px, py))

    # 2) ( ... ) 括号坐标：支持 2 或 4 个数
    pattern = r"\(([-+]?\d+\.?\d*(?:\s*,\s*[-+]?\d+\.?\d*){1,3})\)"
    matches = re.findall(pattern, text)
    for match in matches:
        nums = [float(s.strip()) for s in match.split(",")]
        if len(nums) == 2:
            x, y = nums
            px, py = _to_pixel_xy(x, y, width, height)
            points.append((px, py))
        elif len(nums) == 4:
            x0, y0, x1, y1 = nums
            px0, py0 = _to_pixel_xy(x0, y0, width, height)
            px1, py1 = _to_pixel_xy(x1, y1, width, height)
            # 规范化 bbox 方向
            bx0, bx1 = sorted([px0, px1])
            by0, by1 = sorted([py0, py1])
            points.extend(_bbox_points(bx0, by0, bx1, by1))
        # 其他长度忽略

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

