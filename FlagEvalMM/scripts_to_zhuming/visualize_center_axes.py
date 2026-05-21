#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def project_point(pt3: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> Optional[Tuple[int, int]]:
    """Pinhole projection. pt3: (3,) in camera coord. Return (u,v) int or None if behind camera."""
    x, y, z = float(pt3[0]), float(pt3[1]), float(pt3[2])
    if z <= 1e-6:
        return None
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return int(round(u)), int(round(v))


def safe_relpath(image_path: str, root_dir: Optional[str]) -> str:
    p = Path(image_path)
    if root_dir:
        try:
            return str(p.resolve().relative_to(Path(root_dir).resolve()))
        except Exception:
            pass
    return p.name


def load_centerR_jsonl(in_jsonl: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Group records by image_path."""
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

            ip = obj.get("image_path")
            center = obj.get("center_xyz")
            R = obj.get("R")
            if not ip or center is None or R is None:
                continue

            per_image.setdefault(ip, []).append(obj)

    # sort by det_id if exists
    for ip in per_image:
        per_image[ip].sort(key=lambda x: int(x.get("det_id", 0)))
    return per_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="extract_center_R.py 生成的 *_centerR.jsonl")
    ap.add_argument("--out_dir", required=True, help="可视化输出目录")
    ap.add_argument("--root_dir", default=None, help="用于保持相对目录结构（可选）")
    ap.add_argument("--skip_existing", action="store_true", help="若输出图片已存在则跳过")
    ap.add_argument("--max_images", type=int, default=-1, help="最多处理多少张图（-1=全部）")

    # camera intrinsics (defaults = your provided values)
    ap.add_argument("--fx", type=float, default=433.33678549056947)
    ap.add_argument("--fy", type=float, default=433.33678549056947)
    ap.add_argument("--cx", type=float, default=480.0)
    ap.add_argument("--cy", type=float, default=216.0)

    # visualization params
    ap.add_argument("--axis_len", type=float, default=0.25, help="坐标轴长度（与center同单位，通常米）")
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--show_center_text", action="store_true", help="在中心点旁显示(x,y,z)")
    ap.add_argument("--draw_id", action="store_true", help="在中心点旁标 det_id")

    args = ap.parse_args()

    in_jsonl = Path(args.in_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_image = load_centerR_jsonl(in_jsonl)
    items = list(per_image.items())
    if args.max_images and args.max_images > 0:
        items = items[: args.max_images]

    iterator = tqdm(items, desc="Visualize", unit="img") if tqdm else items

    # OpenCV BGR colors:
    # X: Red, Y: Green, Z: Blue
    color_x = (0, 0, 255)
    color_y = (0, 255, 0)
    color_z = (255, 0, 0)

    for image_path, dets in iterator:
        ip = Path(image_path)
        if not ip.is_file():
            continue

        rel = safe_relpath(image_path, args.root_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and out_path.exists():
            continue

        img = cv2.imread(str(ip))
        if img is None:
            continue

        for rec in dets:
            center = np.array(rec["center_xyz"], dtype=np.float32).reshape(3)
            R = np.array(rec["R"], dtype=np.float32).reshape(3, 3)

            # Axis directions in camera frame: columns of R
            ex = R[:, 0]
            ey = R[:, 1]
            ez = R[:, 2]

            p0 = project_point(center, args.fx, args.fy, args.cx, args.cy)
            if p0 is None:
                continue

            px = project_point(center + args.axis_len * ex, args.fx, args.fy, args.cx, args.cy)
            py = project_point(center + args.axis_len * ey, args.fx, args.fy, args.cx, args.cy)
            pz = project_point(center + args.axis_len * ez, args.fx, args.fy, args.cx, args.cy)

            # draw center
            cv2.circle(img, p0, 4, (255, 255, 255), -1, cv2.LINE_AA)

            det_id = rec.get("det_id", None)
            label = rec.get("label", "box")

            # draw axes
            if px is not None:
                cv2.arrowedLine(img, p0, px, color_x, args.thickness, cv2.LINE_AA, tipLength=0.15)
                cv2.putText(img, "X", (px[0] + 3, px[1] + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_x, 1, cv2.LINE_AA)
            if py is not None:
                cv2.arrowedLine(img, p0, py, color_y, args.thickness, cv2.LINE_AA, tipLength=0.15)
                cv2.putText(img, "Y", (py[0] + 3, py[1] + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_y, 1, cv2.LINE_AA)
            if pz is not None:
                cv2.arrowedLine(img, p0, pz, color_z, args.thickness, cv2.LINE_AA, tipLength=0.15)
                cv2.putText(img, "Z", (pz[0] + 3, pz[1] + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_z, 1, cv2.LINE_AA)

            # optional text near center
            text_lines = []
            if args.draw_id and det_id is not None:
                text_lines.append(f"id={det_id} {label}")
            if args.show_center_text:
                text_lines.append(f"c=({center[0]:+.2f},{center[1]:+.2f},{center[2]:+.2f})")

            y0 = p0[1] - 8
            for t in text_lines:
                cv2.putText(img, t, (p0[0] + 6, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                y0 -= 14

        cv2.imwrite(str(out_path), img)

    print(f"[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()


# python scripts_to_zhuming/visualize_center_axes.py \
#   --in_jsonl to_zhuming/iter_0007800_centerR_test.jsonl \
#   --out_dir to_zhuming/vis_center_axes \
#   --root_dir /code1/data/robobrain2-benchmark/testset_2w_960x576 \
#   --show_center_text \
#   --draw_id \
#   --max_images 100
