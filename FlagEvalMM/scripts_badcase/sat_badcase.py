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
    """question 里所有 type=='text' 的内容拼起来"""
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


def extract_images_from_question(sample: dict):
    """提取 question 中所有 image_url（base64 data-url）为 PIL 列表"""
    q = sample.get("question", [])
    if not isinstance(q, list):
        raise ValueError("sample['question'] 不是 list")

    imgs = []
    for item in q:
        if isinstance(item, dict) and item.get("type") == "image_url":
            data_url = item.get("image_url", {}).get("url")
            if not data_url:
                continue
            if "base64," not in data_url:
                # 如果 SAT 有 http/https URL，这里需要联网下载；
                # 按你给的示例默认都是 base64。
                raise ValueError("image_url 不是 base64 data-url（没有 'base64,'）")
            b64 = data_url.split("base64,", 1)[1]
            img_bytes = base64.b64decode(b64)
            imgs.append(Image.open(BytesIO(img_bytes)).convert("RGB"))

    if not imgs:
        raise ValueError("未找到任何 image_url")
    return imgs


def visualize_sat_badcases(
    json_path: str,
    out_dir: str = "sat_badcases_vis",
    n: int = 20,
    seed: int = 0,
    wrap_width: int = 140,
    max_imgs_per_row: int = 3,
):
    os.makedirs(out_dir, exist_ok=True)
    data = load_json_flexible(json_path)

    bad = [s for s in data if is_false(s.get("correct"))]
    if not bad:
        print("没有找到 correct==false 的样本。")
        return

    random.seed(seed)
    picked = bad if len(bad) <= n else random.sample(bad, n)

    per_type_counter = {}

    for s in picked:
        qtype_raw = s.get("question_type", "") or "unknown"
        qtype_dir = sanitize_dirname(qtype_raw)
        type_out_dir = os.path.join(out_dir, qtype_dir)
        os.makedirs(type_out_dir, exist_ok=True)

        per_type_counter.setdefault(qtype_dir, 0)
        idx = per_type_counter[qtype_dir]
        per_type_counter[qtype_dir] += 1

        # images (可能多张)
        try:
            imgs = extract_images_from_question(s)
        except Exception as e:
            print(f"[{qtype_dir} #{idx}] 图像解析失败: {e}")
            continue

        # text + labels
        q_text = extract_question_text(s) or "(no question text)"
        q_text_wrapped = "\n".join(textwrap.wrap(q_text, width=wrap_width))

        pred = s.get("answer", "")
        gt = s.get("label", "")

        # 多图网格：每行最多 max_imgs_per_row 张
        n_imgs = len(imgs)
        cols = min(max_imgs_per_row, n_imgs)
        rows = (n_imgs + cols - 1) // cols

        # 总布局：上面 rows x cols 图，下面一条文本区
        fig = plt.figure(figsize=(4.5 * cols, 3.8 * rows + 2.2))
        gs = fig.add_gridspec(rows + 1, cols, height_ratios=[1.0] * rows + [0.40], hspace=0.12, wspace=0.05)

        # 画图像
        for i_img in range(rows * cols):
            r = i_img // cols
            c = i_img % cols
            ax = fig.add_subplot(gs[r, c])
            ax.axis("off")
            if i_img < n_imgs:
                ax.imshow(imgs[i_img])
                ax.set_title(f"Frame {i_img+1}", fontsize=11)
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center")

        # 底部文本区（横跨所有列）
        ax_t = fig.add_subplot(gs[rows, :])
        ax_t.axis("off")
        info = (
            f"Question type: {qtype_raw}\n"
            f"Model answer: {pred}\n"
            f"GT label: {gt}\n\n"
            f"Question:\n{q_text_wrapped}"
        )
        ax_t.text(0.01, 0.98, info, ha="left", va="top", fontsize=11, wrap=True)

        fig.suptitle("SAT badcase | correct=False", fontsize=13)

        out_path = os.path.join(type_out_dir, f"badcase_{idx:03d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    total = sum(per_type_counter.values())
    print(f"完成：共保存 {total} 张到 {os.path.abspath(out_dir)}")
    print("按 question_type 目录分布：")
    for k, v in sorted(per_type_counter.items(), key=lambda x: x[0]):
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    visualize_sat_badcases(
        json_path="./results/v164-20251211-152343-alignment/iter_0008500/SAT/SAT_evaluated.json",  # 改成你的实际路径
        out_dir="badcases_vis/sat",
        n=40,
        seed=0,
        wrap_width=140,
        max_imgs_per_row=3,
    )
