# process.py
import json
import os
import os.path as osp
from pathlib import Path

from PIL import Image

# 直接复用你给的 eval_unova.py 里的工具
from eval_unova import load_gt_sets


def process(cfg):
    """
    把 UNOVA 的 GT JSONL 转成 FlagEvalMM 需要的 data.json
    每条样本格式：
        {
            "question_id": <stem>,
            "img_path": <图像绝对路径>,
            "question": "",              # 问题文本（这里留空，prompt 全在 post_prompt 里）
            "answer": <gt_schema_dict>,  # eval_unova.parse_gt_schema_or_dialog 的结果
            "question_type": "short-answer",
            "width": <图像宽>,
            "height": <图像高>,
        }
    """
    data_src = cfg.dataset_path
    split = getattr(cfg, "split", "test")
    output_root = cfg.processed_dataset_path
    output_dir = osp.join(output_root, split)
    os.makedirs(output_dir, exist_ok=True)

    # 支持三种写法：
    #   1) dataset_path = "/path/to/test.jsonl"
    #   2) dataset_path = ["/path/to/a.jsonl", "/path/to/b.jsonl"]
    #   3) dataset_path = "/dir/contains/jsonl_files"
    if isinstance(data_src, (list, tuple)):
        jsonl_list = [str(p) for p in data_src]
    else:
        data_src = str(data_src)
        if osp.isdir(data_src):
            import glob

            jsonl_list = sorted(glob.glob(osp.join(data_src, "*.jsonl")))
        else:
            jsonl_list = [data_src]

    gt_map = load_gt_sets(jsonl_list)
    print(f"[process] Loaded {len(gt_map)} samples from {jsonl_list}")

    processed_data = []
    for stem, rec in sorted(gt_map.items()):
        img_path = rec.get("img_path")
        gt = rec.get("gt")

        if not isinstance(img_path, str) or not osp.exists(img_path):
            print(f"[process][WARN] skip {stem}: missing image {img_path}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            width, height = image.size
        except Exception as e:
            print(f"[process][WARN] skip {stem}: load image failed: {e}")
            continue

        item = {
            # 这里直接用图片文件名的 stem 当 id，和 eval_unova 保持一致
            "question_id": stem,
            "img_path": img_path,
            # 真正的 prompt 全放在 unova.py 的 post_prompt 里，这里留空
            "question": "",
            "answer": gt,  # 直接存 dict，在 evaluate.py 里使用
            "question_type": "short-answer",
            "width": width,
            "height": height,
        }
        processed_data.append(item)

    os.makedirs(output_dir, exist_ok=True)
    out_path = osp.join(output_dir, "data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"[process] Saved {len(processed_data)} samples to {out_path}")


if __name__ == "__main__":
    # 简单本地测试用
    class Cfg:
        dataset_path = ["/data/llm_team/planning/unova/labels/only_select_rightfirst/val/stereo_0915_val.jsonl", 
                        "/data/llm_team/planning/unova/labels/only_select_rightfirst/val/stereo_0925_unova0108_val.jsonl",
                        "/data/llm_team/planning/unova/labels/only_select_rightfirst/val/stereo_0925_unova0175_val.jsonl",
                        "/data/llm_team/planning/unova/labels/only_select_rightfirst/val/stereo_0926_unova0218_val.jsonl"]
        split = "test"
        processed_dataset_path = "/data/llm_team/planning/unova/"

    process(Cfg)
