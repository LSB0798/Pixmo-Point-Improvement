#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# 12 edges for 8 corners (same corner ordering as your draw_3d.py)
EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def color_for_idx(i: int) -> Tuple[int, int, int]:
    # BGR
    palette = [
        (0, 255, 0), (0, 128, 255), (255, 0, 0), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (200, 200, 255), (255, 200, 200),
    ]
    return palette[i % len(palette)]


def safe_relpath(image_path: str, root_dir: Optional[str]) -> str:
    p = Path(image_path)
    if root_dir:
        try:
            return str(p.resolve().relative_to(Path(root_dir).resolve()))
        except Exception:
            pass
    return p.name


def draw_text_block(img: np.ndarray, x: int, y: int, lines: List[str]) -> int:
    """Draw multi-line text with background, return next y."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    pad = 4
    line_h = 16

    widths = []
    for ln in lines:
        (tw, _), _ = cv2.getTextSize(ln, font, scale, thickness)
        widths.append(tw)
    w = (max(widths) if widths else 0) + pad * 2
    h = line_h * len(lines) + pad * 2

    x0, y0 = x, y
    x1, y1 = x0 + w, y0 + h
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), -1)

    ty = y0 + pad + 12
    for ln in lines:
        cv2.putText(img, ln, (x0 + pad, ty), font, scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)
        ty += line_h

    return y1 + 6

def generate_camera_params(image_path, fx=None, fy=None, cx=None, cy=None, fov=60):
    with Image.open(image_path) as image:
        w, h = image.size

    if fx is None or fy is None:
        fx = round(w / (2 * np.tan(np.deg2rad(fov) / 2)), 2)
        fy = round(h / (2 * np.tan(np.deg2rad(fov) / 2)), 2)

    if cx is None or cy is None:
        cx = round(w / 2, 2)
        cy = round(h / 2, 2)

    return {'fx': float(fx), 'fy': float(fy), 'cx': float(cx), 'cy': float(cy)}


def build_R_from_bbox3d_angles_like_draw3d(bbox_3d: List[float]) -> np.ndarray:
    """
    与你的 draw_3d.py 行为对齐：

    你原逻辑：
      bbox_3d[-3:] *= 180
      convert_3dbbox: ... pitch, yaw, roll = point[-3:]
      rotate_xyz: Rx(pitch) -> Ry(yaw) -> Rz(roll)
      deg2rad(pitch/yaw/roll)

    等价于（注意顺序映射）：
      ax = bbox_3d[6] * pi   (作为 pitch 绕 X)
      ay = bbox_3d[7] * pi   (作为 yaw   绕 Y)
      az = bbox_3d[8] * pi   (作为 roll  绕 Z)
      R  = Rz(az) @ Ry(ay) @ Rx(ax)
    """
    ax = float(bbox_3d[6]) * math.pi
    ay = float(bbox_3d[7]) * math.pi
    az = float(bbox_3d[8]) * math.pi

    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)

    Rx = np.array([[1, 0,  0],
                   [0, cx, -sx],
                   [0, sx, cx]], dtype=np.float32)

    Ry = np.array([[cy, 0, sy],
                   [0,  1, 0],
                   [-sy, 0, cy]], dtype=np.float32)

    Rz = np.array([[cz, -sz, 0],
                   [sz, cz,  0],
                   [0,  0,   1]], dtype=np.float32)

    return (Rz @ Ry @ Rx).astype(np.float32)


def corners_from_bbox3d(bbox_3d: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    输入 bbox_3d = [x,y,z,sx,sy,sz, a1,a2,a3]
    输出:
      center (3,)
      R (3,3)
      corners_cam (8,3)
    """
    x, y, z, sx, sy, sz = map(float, bbox_3d[:6])
    center = np.array([x, y, z], dtype=np.float32)

    R = build_R_from_bbox3d_angles_like_draw3d(bbox_3d)

    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    local = np.array([
        [ hx,  hy,  hz],
        [ hx,  hy, -hz],
        [ hx, -hy,  hz],
        [ hx, -hy, -hz],
        [-hx,  hy,  hz],
        [-hx,  hy, -hz],
        [-hx, -hy,  hz],
        [-hx, -hy, -hz],
    ], dtype=np.float32)

    corners_cam = (local @ R.T) + center[None, :]
    return center, R, corners_cam


def project_points(XYZ: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> Tuple[np.ndarray, np.ndarray]:
    """Project Nx3 to Nx2, return (uv, visible_mask_by_Z)."""
    X = XYZ[:, 0]
    Y = XYZ[:, 1]
    Z = XYZ[:, 2]
    vis = Z > 1e-6
    Zsafe = np.maximum(Z, 1e-6)
    u = fx * (X / Zsafe) + cx
    v = fy * (Y / Zsafe) + cy
    uv = np.stack([u, v], axis=1).astype(np.float32)
    return uv, vis


def load_jsonl_group_by_image(in_jsonl: Path) -> Dict[str, List[Dict[str, Any]]]:
    per_image: Dict[str, List[Dict[str, Any]]] = {}
    with in_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not obj.get("ok", False):
                continue
            ip = obj.get("image_path")
            js = obj.get("json", [])
            if not ip or not isinstance(js, list):
                continue
            boxes = [b for b in js if isinstance(b, dict) and isinstance(b.get("bbox_3d"), list)]
            if boxes:
                per_image.setdefault(ip, []).extend(boxes)
    return per_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="推理输出 jsonl（包含 ok/json/image_path）")
    ap.add_argument("--out_dir", required=True, help="可视化输出目录")
    ap.add_argument("--root_dir", default=None, help="用于保持相对目录结构（可选）")
    ap.add_argument("--max_images", type=int, default=-1, help="最多处理多少张图（-1=全部）")
    ap.add_argument("--random_sample", action="store_true",help="随机抽取 max_images 张图片（不是按原顺序）")
    ap.add_argument("--seed", type=int, default=0,help="随机种子（用于可复现随机抽样）")
    ap.add_argument("--skip_existing", action="store_true", help="若输出已存在则跳过")
    ap.add_argument("--draw_corner_index", action="store_true", help="角点旁标注 0~7")

    # Fixed intrinsics (default = your provided values)
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--fov", type=float, default=60.0, help="当 fx/fy 未指定时，用 fov 估算伪内参")

    ap.add_argument("--log", default=None, help="log 文件（默认 out_dir/vis.log）")
    args = ap.parse_args()

    in_jsonl = Path(args.in_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else (out_dir / "vis.log")

    def log(msg: str):
        print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    per_image = load_jsonl_group_by_image(in_jsonl)
    items = list(per_image.items())

    if args.random_sample:
        rng = random.Random(args.seed)
        rng.shuffle(items)

    if args.max_images and args.max_images > 0:
        items = items[: args.max_images]


    log(f"[START] images={len(items)} fx={args.fx} fy={args.fy} cx={args.cx} cy={args.cy} fov={args.fov}")

    processed_images = 0
    missing_images = 0
    total_boxes = 0
    drawn_boxes = 0

    iterator = tqdm(items, desc="Visualize", unit="img") if tqdm else items

    for image_path, boxes in iterator:
        ip = Path(image_path)
        if not ip.is_file():
            missing_images += 1
            log(f"[WARN] image not found: {image_path}")
            continue

        rel = safe_relpath(image_path, args.root_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and out_path.exists():
            continue

        img = cv2.imread(str(ip))
        if img is None:
            missing_images += 1
            log(f"[WARN] cv2 read failed: {image_path}")
            continue

        # per-image text y cursor
        y_cursor = 10

        for bi, b in enumerate(boxes):
            bbox_3d = b.get("bbox_3d")
            if not isinstance(bbox_3d, list) or len(bbox_3d) != 9:
                continue

            total_boxes += 1
            try:
                center, R, corners_cam = corners_from_bbox3d(bbox_3d)
            except Exception:
                continue

            # project corners + center
            cam = generate_camera_params(
                str(ip),
                fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                fov=args.fov
            )

            uv, vis = project_points(corners_cam, cam["fx"], cam["fy"], cam["cx"], cam["cy"])
            center_uv, center_vis = project_points(center[None, :], cam["fx"], cam["fy"], cam["cx"], cam["cy"])


            col = color_for_idx(bi)

            # draw edges
            for s, e in EDGES:
                if not (vis[s] and vis[e]):
                    continue
                p1 = (int(round(uv[s, 0])), int(round(uv[s, 1])))
                p2 = (int(round(uv[e, 0])), int(round(uv[e, 1])))
                cv2.line(img, p1, p2, col, 2, cv2.LINE_AA)

            # draw corners
            for ci in range(8):
                if not vis[ci]:
                    continue
                p = (int(round(uv[ci, 0])), int(round(uv[ci, 1])))
                cv2.circle(img, p, 3, col, -1, cv2.LINE_AA)
                if args.draw_corner_index:
                    cv2.putText(img, str(ci), (p[0] + 4, p[1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

            # draw center (cross) + xyz text near it
            if bool(center_vis[0]):
                cu, cvv = int(round(center_uv[0, 0])), int(round(center_uv[0, 1]))
                cv2.drawMarker(img, (cu, cvv), col, markerType=cv2.MARKER_CROSS,
                               markerSize=14, thickness=2, line_type=cv2.LINE_AA)
                cv2.putText(
                    img,
                    f"c=({center[0]:+.2f},{center[1]:+.2f},{center[2]:+.2f})",
                    (cu + 6, cvv - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA
                )

            # draw R as a text block (top-left stack)
            Rn = R.astype(np.float64)
            lines = [
                f"box#{bi} label={b.get('label','box')}",
                f"center=({center[0]:+.3f},{center[1]:+.3f},{center[2]:+.3f})",
                f"R=[{Rn[0,0]:+.3f} {Rn[0,1]:+.3f} {Rn[0,2]:+.3f}]",
                f"  [{Rn[1,0]:+.3f} {Rn[1,1]:+.3f} {Rn[1,2]:+.3f}]",
                f"  [{Rn[2,0]:+.3f} {Rn[2,1]:+.3f} {Rn[2,2]:+.3f}]",
            ]
            y_cursor = draw_text_block(img, 10, y_cursor, lines)

            drawn_boxes += 1

        cv2.imwrite(str(out_path), img)
        processed_images += 1

    log(f"[DONE] processed_images={processed_images} missing_images={missing_images} "
        f"total_boxes={total_boxes} drawn_boxes={drawn_boxes} out_dir={out_dir} log={log_path}")


if __name__ == "__main__":
    main()


#   --in_jsonl to_zhuming/iter_0007800.jsonl \
#   --out_dir to_zhuming/vis_iter_0007800 \
#   --root_dir /code1/data/robobrain2-benchmark/testset_2w_960x576 \

# python scripts_to_zhuming/visualize_centerR_corner.py \
#   --in_jsonl to_zhuming/qwen3_vl_true.jsonl \
#   --out_dir to_zhuming/qwen3_vl_true \
#   --root_dir /code1/data/robobrain2-benchmark/moving_box/val/img \
#   --draw_corner_index \
#   --max_images 100 \
#   --random_sample \
#   --seed 42 \
#   --fov 60


#   --fx 433.33678549056947 \
#   --fy 433.33678549056947 \
#   --cx 480.0 \
#   --cy 216.0

# scp -r to_zhuming ubt@10.10.22.57:/media/ubt/04d101c2-7eb0-4765-afd4-85d6f4b201a7/workspace/1230