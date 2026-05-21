import json
import os
import os.path as osp

import tqdm
from datasets import load_dataset
from PIL import Image


def process(cfg):

    data_dir, split = cfg.dataset_path, cfg.split
    output_root = cfg.processed_dataset_path
    output_dir = osp.join(output_root, split)
    os.makedirs(output_dir, exist_ok=True)

    # cmd = f"huggingface-cli download --repo-type dataset --resume-download {data_dir} --local-dir {os.path.dirname(os.path.dirname(output_root))}"
    # os.system(cmd)
    # dataset = load_dataset(os.path.dirname(output_root), split=split)

    dataset = load_dataset(
        "json", 
        data_files=osp.join(output_root, f"{split}.json"),
        split="train"   # 单文件 JSON 默认叫 "train"
    )

    print(f"Loaded {len(dataset)} samples from {data_dir} {split}")

    processed_data = []
    for i, item in tqdm.tqdm(enumerate(dataset)):
        conversations = item["conversations"]
        question, answer = "", ""
        for conv in conversations:
            if conv["from"] == "human":
                question = conv["value"]
            if conv["from"] == "gpt":
                answer = conv["value"]
        if not question or not answer:
            continue
        image_path = osp.join(os.path.dirname(output_root), item["image"])
        image = Image.open(image_path)
        width, height = image.size

        processed_item = {
            "question_id": item["id"],
            "img_path": image_path,
            "question": question.replace("<image>", ""),
            "answer": answer,
            "question_type": "short-answer",
            "width": width,
            "height": height,
        }
        processed_data.append(processed_item)

    with open(osp.join(output_dir, "data.json"), "w") as f:
        json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    class Cfg:  # 简单模拟 cfg 对象
        dataset_path = "/data/nlp/nlp_team1/data/ShareRobot-Bench"
        split = "test"
        processed_dataset_path = "/data/nlp/nlp_team1/data/ShareRobot-Bench/affordance/"
    process(Cfg)
