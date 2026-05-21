import os
import re
import json
import base64
import random
import ast
import textwrap
from io import BytesIO

from PIL import Image
import matplotlib.pyplot as plt


DATASET_ROOT = "/code1/data/robobrain2-benchmark/RoboSpatial-Home/all"
DATA_JSON = os.path.join(DATASET_ROOT, "data.json")


def load_json_flexible(path: str):
    """兼容：标准 JSON(list/dict) 或 JSON Lines(一行一个json)"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for _, v in data.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return v
            return [data]
    except json.JSONDecodeError:
        pass

    items = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def load_data_map(data_json_path: str):
    """data.json 是 list[dict]"""
    data = load_json_flexible(data_json_path)
    mp = {}
    for item in data:
        qid = item.get("question_id")
        if qid:
            mp[qid] = item
    return mp


def is_false(v) -> bool:
    """兼容 correct: false/False/'false'/'False'/0"""
    if isinstance(v, bool):
        return v is False
    if isinstance(v, (int, float)):
        return v == 0
    if isinstance(v, str):
        return v.strip().lower() in {"false", "0", "no"}
    return False


def sanitize_dirname(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "unknown"
    name = name.lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9._-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unknown"


def extract_question_text(sample: dict) -> str:
    """优先从 evaluated.json 的 question 里取 type=='text'"""
    q = sample.get("question", [])
    if not isinstance(q, list):
        return ""
    texts = []
    for item in q:
        if isinstance(item, dict) and item.get("type") == "text":
            t = item.get("text", "")
            if t:
                texts.append(t.strip())
    return "\n".join(texts).strip()


def extract_rgb_from_base64_if_present(sample: dict):
    """如果 evaluated.json 里有 base64 image_url，就直接解码成 PIL；否则返回 None"""
    q = sample.get("question", [])
    if not isinstance(q, list):
        return None

    data_url = None
    for item in q:
        if isinstance(item, dict) and item.get("type") == "image_url":
            data_url = item.get("image_url", {}).get("url")
            break

    if not data_url:
        return None
    if "base64," not in data_url:
        return None

    b64 = data_url.split("base64,", 1)[1]
    img_bytes = base64.b64decode(b64)
    return Image.open(BytesIO(img_bytes)).convert("RGB")


def parse_points(val):
    """解析 '[(x,y), ...]' / list -> list[(x,y)]"""
    if val is None:
        return []
    if isinstance(val, list):
        pts = []
        for p in val:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append((float(p[0]), float(p[1])))
        return pts
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            obj = ast.literal_eval(s)
            pts = []
            if isinstance(obj, (list, tuple)):
                for p in obj:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        pts.append((float(p[0]), float(p[1])))
            return pts
        except Exception:
            return []
    return []


def scale_points(points, width, height, mode: str):
    """
    mode:
      - "rel1000": x,y in [0,1000] -> pixel
      - "norm01": x,y in [0,1] -> pixel
      - "pixel": already pixel
    """
    out = []
    for x, y in points:
        if mode == "rel1000":
            out.append((x / 1000.0 * width, y / 1000.0 * height))
        elif mode == "norm01":
            out.append((x * width, y * height))
        else:
            out.append((x, y))
    return out


def infer_gt_mode(gt_points):
    """GT 可能是 0~1 或 0~1000 或 已经像素；做个简单判断"""
    if not gt_points:
        return "pixel"
    xs = [p[0] for p in gt_points]
    ys = [p[1] for p in gt_points]
    mx = max(xs)
    my = max(ys)
    if mx <= 1.5 and my <= 1.5:
        return "norm01"
    if mx <= 1000.0 and my <= 1000.0:
        return "rel1000"
    return "pixel"


def visualize_robo_spatial_home_badcases(
    evaluated_json_path: str,
    out_dir: str = "robo_spatial_home_badcases_vis",
    n: int = 20,
    seed: int = 0,
    wrap_width: int = 140,
):
    os.makedirs(out_dir, exist_ok=True)

    data_map = load_data_map(DATA_JSON)
    eval_data = load_json_flexible(evaluated_json_path)

    bad = [s for s in eval_data if is_false(s.get("correct"))]
    if not bad:
        print("没有找到 correct==false 的样本。")
        return

    random.seed(seed)
    picked = bad if len(bad) <= n else random.sample(bad, n)

    per_type_counter = {}

    for s in picked:
        qid = s.get("question_id")
        if not qid:
            # 如果你们 evaluated.json 的 key 名不是 question_id，改这里即可
            print("跳过：样本缺少 question_id")
            continue

        meta = data_map.get(qid)
        if not meta:
            print(f"跳过：data.json 找不到 question_id={qid}")
            continue

        qtype_raw = meta.get("question_type", "") or s.get("question_type", "")
        qtype_dir = sanitize_dirname(qtype_raw)
        type_out_dir = os.path.join(out_dir, qtype_dir)
        os.makedirs(type_out_dir, exist_ok=True)

        per_type_counter.setdefault(qtype_dir, 0)
        idx = per_type_counter[qtype_dir]
        per_type_counter[qtype_dir] += 1

        # 图像：优先 evaluated 里 base64；否则从 dataset 路径读
        rgb = extract_rgb_from_base64_if_present(s)
        if rgb is None:
            img_path = meta.get("img_path", "")
            if not img_path:
                print(f"[{qid}] 跳过：没有 img_path 且 evaluated 也无 base64")
                continue
            rgb_path = os.path.join(DATASET_ROOT, img_path)
            if not os.path.exists(rgb_path):
                print(f"[{qid}] 跳过：RGB 不存在 {rgb_path}")
                continue
            rgb = Image.open(rgb_path).convert("RGB")

        width = int(meta.get("image_width", rgb.size[0]))
        height = int(meta.get("image_height", rgb.size[1]))

        # 文本：优先 evaluated 的 question text；没有就用 data.json 的 question
        q_text = extract_question_text(s) or (meta.get("question") or "")
        q_text = q_text.strip() if q_text else "(no question text)"
        q_text_wrapped = "\n".join(textwrap.wrap(q_text, width=wrap_width))

        pred = s.get("answer", "")
        # yes/no 任务的 GT 直接用 evaluated label；point 任务 GT 必须从 data.json
        gt_yesno = s.get("label", "")
        gt_from_data = meta.get("answer", "")

        # 画图布局：
        # - point: RGB + Mask + 底部文本
        # - 其他: RGB + (右侧空) + 底部文本
        is_point = (str(qtype_raw).lower() == "point")

        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.30], hspace=0.12, wspace=0.05)
        ax_l = fig.add_subplot(gs[0, 0])
        ax_r = fig.add_subplot(gs[0, 1])
        ax_t = fig.add_subplot(gs[1, :])

        # 左：RGB
        ax_l.imshow(rgb)
        ax_l.axis("off")
        ax_l.set_title("RGB")

        # 右：Mask（仅 point）
        if is_point:
            mask_path = meta.get("mask_path", "")
            mask = None
            if mask_path:
                full_mask_path = os.path.join(DATASET_ROOT, mask_path)
                if os.path.exists(full_mask_path):
                    mask = Image.open(full_mask_path)
                    if mask.size != rgb.size:
                        mask = mask.resize(rgb.size, resample=Image.NEAREST)
            if mask is not None:
                ax_r.imshow(mask)
                ax_r.set_title("Mask")
            else:
                ax_r.text(0.5, 0.5, "Mask not found", ha="center", va="center")
                ax_r.set_title("Mask")
            ax_r.axis("off")

            # 叠加 pred 点（answer 是 0~1000 相对坐标）
            pred_pts = parse_points(pred)
            pred_px = scale_points(pred_pts, width, height, mode="rel1000")
            if pred_px:
                ax_l.scatter([p[0] for p in pred_px], [p[1] for p in pred_px], s=55, marker="x")
                ax_r.scatter([p[0] for p in pred_px], [p[1] for p in pred_px], s=55, marker="x")

            # 叠加 GT 点（从 data.json）
            gt_pts = parse_points(gt_from_data)
            gt_mode = infer_gt_mode(gt_pts)
            gt_px = scale_points(gt_pts, width, height, mode=gt_mode)
            if gt_px:
                ax_l.scatter([p[0] for p in gt_px], [p[1] for p in gt_px], s=25, marker="o")
                ax_r.scatter([p[0] for p in gt_px], [p[1] for p in gt_px], s=25, marker="o")

            gt_show = gt_from_data
        else:
            ax_r.axis("off")
            ax_r.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax_r.set_title("Mask")
            gt_show = gt_yesno

        # 底部文本（不挡图）
        ax_t.axis("off")
        info = (
            f"question_id: {qid}\n"
            f"question_type: {qtype_raw}\n"
            f"Model answer: {pred}\n"
            f"GT label: {gt_show}\n\n"
            f"Question:\n{q_text_wrapped}"
        )
        ax_t.text(0.01, 0.98, info, ha="left", va="top", fontsize=10, wrap=True)

        fig.suptitle("badcase | correct=False", fontsize=12)

        out_path = os.path.join(type_out_dir, f"{qid}_{idx:03d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    total = sum(per_type_counter.values())
    print(f"完成：共保存 {total} 张到 {os.path.abspath(out_dir)}")
    print("按 question_type 目录分布：")
    for k, v in sorted(per_type_counter.items(), key=lambda x: x[0]):
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    visualize_robo_spatial_home_badcases(
        evaluated_json_path="./results/v164-20251211-152343-alignment/iter_0008500/robo_spatial_home_all/robo_spatial_home_all_evaluated.json",  # 改成你的实际路径
        out_dir="badcases_vis/robo_spatial_home",
        n=40,
        seed=0,
        wrap_width=140,
    )
    