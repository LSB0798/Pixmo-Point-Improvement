import json
import os
import os.path as osp
from datasets import load_dataset
from glob import glob


def save_image(full_image_path, image_obj):
    image_obj.save(full_image_path)

def load_json_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]
    
def video_to_img_path(video_path: str) -> str:
    # 取出 video_id
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    
    # 拆分路径
    parts = video_path.split(os.sep)
    
    # 找到 "videos" 目录的位置
    if "videos" not in parts:
        raise ValueError("路径中未找到 'videos' 目录")
    
    idx = parts.index("videos")
    
    # 前缀路径（直到 videos 之前）
    prefix = os.sep.join(parts[:idx])
    
    # 拼接新的 img_path
    img_path = os.path.join(
        prefix,
        "cot_images",
        f"{video_id}_last.png"
    )
    
    return img_path

def process(cfg):
    """Process the dataset and save it in a standard format"""
    data_dir, split = cfg.dataset_path, cfg.split
    name = ""

    # build output path
    output_dir = osp.join(cfg.processed_dataset_path, name, "val_my")

    content = []

    # load dataset
    data = []
    print("Loading all JSON files...")
    for json_path in sorted(glob(os.path.join(data_dir, "*.json"))):
        data.extend(load_json_lines(json_path))

    # process each item
    for index, annotation in enumerate(data):
        video_path = annotation["videos"][0]
        if ". is it" in annotation["messages"][0]["content"]:
            annotation["messages"][0]["content"] += " You must respond only with 'yes' or 'no'."
        question = annotation["messages"][0]["content"]
        gt_answer = annotation["messages"][1]["content"]
        # build information dictionary
        info = {
            "question_type": annotation["task_type"],
            "question_id": index,
            "question": question,
            "gt_answer": gt_answer,
            "video_path": [video_path, video_to_img_path(video_path)],
        }
        content.append(info)

    # save data
    output_file = osp.join(output_dir, "data.json")
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} items. Data saved to {output_file}")

if __name__ == "__main__":
    class Cfg:  # 简单模拟 cfg 对象
        # dataset_path = "/data/nlp/nlp_team1/data/robovqa/convert_data/val"
        dataset_path = "/data/robovqa/convert_data/val"
        split = "val"
        # processed_dataset_path = "/data/nlp/nlp_team1/data/robovqa/convert_data/"
        processed_dataset_path = "/data/robovqa/convert_data/"
    process(Cfg)

