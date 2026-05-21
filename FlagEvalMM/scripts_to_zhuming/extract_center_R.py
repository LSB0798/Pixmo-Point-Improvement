#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math
import numpy as np

def bbox3d_to_center_R(bbox_3d):
    x, y, z, sx, sy, sz, roll_raw, pitch_raw, yaw_raw = bbox_3d
    # roll  = roll_raw  * math.pi
    # pitch = pitch_raw * math.pi
    # yaw   = yaw_raw   * math.pi

    # 复刻旧代码的错位：pitch<-roll_raw, yaw<-pitch_raw, roll<-yaw_raw
    pitch = roll_raw  * math.pi   # Rx
    yaw   = pitch_raw * math.pi   # Ry
    roll  = yaw_raw   * math.pi   # Rz

    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy_ = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)

    Rx = np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]], float)
    Ry = np.array([[cy,0,sy_],[0,1,0],[-sy_,0,cy]], float)
    Rz = np.array([[cr,-sr,0],[sr,cr,0],[0,0,1]], float)

    R = (Rz @ Ry @ Rx).tolist()
    center = [float(x), float(y), float(z)]
    return center, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    total_samples = 0          # jsonl 行数
    ok_samples = 0             # ok=true 的行数
    total_boxes = 0            # 解析到的 box 数
    ok_boxes = 0               # center/R 生成成功的 box 数

    log_fp = open(args.log, "a", encoding="utf-8") if args.log else None
    def log(msg):
        print(msg)
        if log_fp:
            log_fp.write(msg + "\n")
            log_fp.flush()

    with open(args.in_jsonl, "r", encoding="utf-8") as fin, \
         open(args.out_jsonl, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total_samples += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if not obj.get("ok", False):
                continue
            ok_samples += 1

            boxes = obj.get("json", [])
            if not isinstance(boxes, list):
                continue

            for i, b in enumerate(boxes):
                total_boxes += 1
                bbox_3d = b.get("bbox_3d") if isinstance(b, dict) else None
                if not bbox_3d or len(bbox_3d) != 9:
                    continue

                center, R = bbox3d_to_center_R(bbox_3d)
                ok_boxes += 1

                out = {
                    "box_dir": obj.get("box_dir"),
                    "image_path": obj.get("image_path"),
                    "det_id": i,
                    "center_xyz": center,   # 相机坐标系 (x,y,z)
                    "R": R,                 # 3x3 旋转矩阵
                    "label": b.get("label", "box") if isinstance(b, dict) else "box",
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    log(f"[STAT] total_samples={total_samples}, ok_samples={ok_samples}, "
        f"total_boxes={total_boxes}, ok_boxes={ok_boxes}, "
        f"ok_box_rate={(ok_boxes/total_boxes if total_boxes else 0):.4f}")

    if log_fp:
        log_fp.close()

if __name__ == "__main__":
    main()


# python scripts_to_zhuming/extract_center_R.py \
#   --in_jsonl to_zhuming/iter_0007800.jsonl \
#   --out_jsonl to_zhuming/iter_0007800_centerR.jsonl \
#   --log to_zhuming/iter_0007800_centerR.log