import os
import re
import json
import base64
import random
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
    """从 sample['question'] 中解析 image_url（data:image/...;base64,...）"""
    q = sample.get("question", [])
    if not isinstance(q, list):
        raise ValueError("sample['question'] 不是 list")

    data_url = None
    for item in q:
        if isinstance(item, dict) and item.get("type") == "image_url":
            data_url = item.get("image_url", {}).get("url")
            break

    if not data_url:
        raise ValueError("未找到 image_url")
    if "base64," not in data_url:
        raise ValueError("image_url 不是 base64 data-url（没有 'base64,'）")

    b64 = data_url.split("base64,", 1)[1]
    img_bytes = base64.b64decode(b64)
    return Image.open(BytesIO(img_bytes)).convert("RGB")


def extract_question_text(sample: dict) -> str:
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
    """把 question_type 变成安全的目录名"""
    name = (name or "").strip()
    if not name:
        return "unknown"
    name = name.lower()
    # 将空白变成下划线，并移除奇怪字符
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9._-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unknown"


def visualize_blink_badcases(
    json_path: str,
    out_dir: str = "blink_badcases_vis",
    n: int = 20,
    seed: int = 0,
    wrap_width: int = 140,
):
    os.makedirs(out_dir, exist_ok=True)
    data = load_json_flexible(json_path)

    bad = [s for s in data if is_false(s.get("correct"))]
    if not bad:
        print("没有找到 correct==false 的样本。")
        return

    random.seed(seed)
    picked = bad if len(bad) <= n else random.sample(bad, n)

    # 每个 question_type 单独计数，避免重名覆盖
    per_type_counter = {}

    for _, s in enumerate(picked):
        qtype_raw = s.get("question_type", "")
        qtype_dir = sanitize_dirname(qtype_raw)
        type_out_dir = os.path.join(out_dir, qtype_dir)
        os.makedirs(type_out_dir, exist_ok=True)

        per_type_counter.setdefault(qtype_dir, 0)
        idx = per_type_counter[qtype_dir]
        per_type_counter[qtype_dir] += 1

        # RGB
        try:
            rgb = extract_rgb_pil(s)
        except Exception as e:
            print(f"[{qtype_dir} #{idx}] RGB 解析失败: {e}")
            continue

        q_text = extract_question_text(s) or "(no question text)"
        q_text_wrapped = "\n".join(textwrap.wrap(q_text, width=wrap_width))

        pred = s.get("answer", "")
        gt = s.get("label", "")

        # 上图像，下文字（不遮挡）
        fig = plt.figure(figsize=(10, 7))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.30], hspace=0.08)

        ax_img = fig.add_subplot(gs[0, 0])
        ax_txt = fig.add_subplot(gs[1, 0])

        ax_img.imshow(rgb)
        ax_img.axis("off")
        ax_img.set_title("RGB")

        ax_txt.axis("off")
        info = (
            f"Question type: {qtype_raw}\n"
            f"Model answer: {pred}\n"
            f"GT label: {gt}\n\n"
            f"Question:\n{q_text_wrapped}"
        )
        ax_txt.text(0.01, 0.98, info, ha="left", va="top", fontsize=10, wrap=True)

        fig.suptitle("badcase | correct=False", fontsize=12)

        out_path = os.path.join(type_out_dir, f"badcase_{idx:03d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"完成：共保存 {sum(per_type_counter.values())} 张到 {os.path.abspath(out_dir)}")
    print("按 question_type 目录分布：")
    for k, v in sorted(per_type_counter.items(), key=lambda x: x[0]):
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    visualize_blink_badcases(
        json_path="./results/v164-20251211-152343-alignment/iter_0008500/blink_val/blink_val_evaluated.json",  # 改成你的实际路径
        out_dir="badcases_vis/blink_val",
        n=40,
        seed=0,
        wrap_width=140,
    )
