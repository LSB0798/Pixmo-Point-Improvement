import os
import json
import base64
import random
import ast
import textwrap
from io import BytesIO

from PIL import Image
import matplotlib.pyplot as plt


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


def extract_rgb_pil(sample: dict) -> Image.Image:
    """从 sample['question'] 里解析 base64 data-url 得到 RGB PIL 图"""
    q = sample.get("question", [])
    if not isinstance(q, list):
        raise ValueError("sample['question'] 不是 list")

    data_url = None
    for item in q:
        if isinstance(item, dict) and item.get("type") == "image_url":
            data_url = item.get("image_url", {}).get("url")
            break

    if not data_url or "base64," not in data_url:
        raise ValueError("未找到 base64 image_url")

    b64 = data_url.split("base64,", 1)[1]
    img_bytes = base64.b64decode(b64)
    return Image.open(BytesIO(img_bytes)).convert("RGB")


def extract_question_text(sample: dict) -> str:
    """拿到 question 里 type=='text' 的内容（可能有多段就拼起来）"""
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


def parse_points(val) -> list[tuple[int, int]]:
    """解析 answer: 可能是字符串 '[(x,y)]' 或 list"""
    if val is None:
        return []
    if isinstance(val, list):
        pts = []
        for p in val:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append((int(p[0]), int(p[1])))
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
                        pts.append((int(p[0]), int(p[1])))
            return pts
        except Exception:
            return []
    return []


def resolve_mask_path(mask_root: str, label: str) -> str:
    """
    label 可能是:
      - 'mask/xxx.png'
      - 'xxx.png'
    统一映射到 mask_root/文件名
    """
    if not label:
        return ""
    filename = os.path.normpath(label).split(os.sep)[-1]
    return os.path.join(mask_root, filename)


def visualize_pixmo_point_badcases(
    json_path: str,
    mask_root: str,
    out_dir: str = "pixmo_point_badcases_vis",
    n: int = 20,
    seed: int = 0,
    wrap_width: int = 140,
):
    os.makedirs(out_dir, exist_ok=True)
    data = load_json_flexible(json_path)

    bad = [s for s in data if float(s.get("score", 0.0)) == 0.0]
    if not bad:
        print("没有找到 score==0 的样本。")
        return

    random.seed(seed)
    picked = bad if len(bad) <= n else random.sample(bad, n)

    for i, s in enumerate(picked):
        # RGB
        try:
            rgb = extract_rgb_pil(s)
        except Exception as e:
            print(f"[{i}] RGB 解析失败: {e}")
            continue

        # Mask
        label = s.get("label", "")
        mask_path = resolve_mask_path(mask_root, label)
        mask = None
        if mask_path and os.path.exists(mask_path):
            try:
                mask = Image.open(mask_path)
            except Exception as e:
                print(f"[{i}] mask 读取失败: {mask_path} | {e}")
        else:
            print(f"[{i}] mask 不存在: {mask_path}")

        # 若 mask 尺寸不同，resize 到 RGB 尺寸方便对齐
        if mask is not None and mask.size != rgb.size:
            mask = mask.resize(rgb.size, resample=Image.NEAREST)

        # 文本
        q_text = extract_question_text(s) or "(no question text)"
        q_text_wrapped = "\n".join(textwrap.wrap(q_text, width=wrap_width))

        # 点：answer 是你缩放后的像素坐标，直接用
        pts = parse_points(s.get("answer"))
        score = s.get("score", None)

        # 三块布局：RGB | Mask，上面；下面文本（不挡图）
        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.28], hspace=0.15, wspace=0.05)

        ax_rgb = fig.add_subplot(gs[0, 0])
        ax_mask = fig.add_subplot(gs[0, 1])
        ax_text = fig.add_subplot(gs[1, :])

        # RGB
        ax_rgb.imshow(rgb)
        ax_rgb.set_title("RGB + answer (pixel coords)")
        ax_rgb.axis("off")
        if pts:
            ax_rgb.scatter([p[0] for p in pts], [p[1] for p in pts], s=55, marker="x")

        # Mask
        if mask is not None:
            ax_mask.imshow(mask)
        else:
            ax_mask.text(0.5, 0.5, "Mask not found", ha="center", va="center")
        ax_mask.set_title(f"Mask (label={label})")
        ax_mask.axis("off")
        if pts and mask is not None:
            ax_mask.scatter([p[0] for p in pts], [p[1] for p in pts], s=55, marker="x")

        # Text
        ax_text.axis("off")
        info = (
            f"score: {score}\n"
            f"answer (pixel): {pts}\n"
            f"label(mask): {label}\n\n"
            f"Question:\n{q_text_wrapped}"
        )
        ax_text.text(0.01, 0.98, info, ha="left", va="top", fontsize=10, wrap=True)

        fig.suptitle("pixmo-point-eval badcase | score=0", fontsize=12)

        out_path = os.path.join(out_dir, f"badcase_{i:03d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"完成：保存 {len(picked)} 张到 {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    visualize_pixmo_point_badcases(
        json_path="./results/iter_0008500-vllm-1/pixmo_points_eval/pixmo_points_eval_evaluated.json",  # 改成你的实际路径
        mask_root="/code1/data/robobrain2-benchmark/pixmo-points-eval/test/mask",                        # 改成你的 mask 根目录（或绝对路径）
        out_dir="badcases_vis/pixmo_point_eval",
        n=40,
        seed=0,
        wrap_width=140,
    )
