import json
import os
import os.path as osp
from datasets import load_dataset
from glob import glob

QA_template = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
Considering the progress shown in the video and my current observation in the last frame, what action should I take next in order to {}?
A. {}
B. {}
C. {}
D. {}
"""

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
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
        "convert_data",
        "CoT_images",
        f"{video_id}_last.png"
    )
    
    return img_path

def process(cfg):
    """Process the dataset and save it in a standard format"""
    data_dir, split = cfg.dataset_path, cfg.split

    # build output path
    output_dir = osp.join(cfg.processed_dataset_path, split)
    content = []

    # load dataset
    data = load_json(osp.join(data_dir, "EgoPlan-Bench2_with_video_type.json"))
    print("Loading all JSON files...")

    # process each item
    for index, annotation in enumerate(data):
        # video_path = f"/code1/7TB/v2/egoplan2/clips/{annotation['sample_id']}.mp4"
        video_path = f"/data/lishuaibing/test_process/clips/{annotation['sample_id']}.mp4"
        last_frame_path = osp.join(data_dir, "cot_images", f"{annotation['sample_id']}_last.png")
        question = QA_template.format(
            annotation['task_goal'],
            annotation['choice_a'],
            annotation['choice_b'],
            annotation['choice_c'],
            annotation['choice_d']
        )
        gt_answer = annotation["golden_choice_idx"]
        gt_answer_text = annotation["answer"]
        domain = annotation["domain"]
        video_type = annotation["video_type"]
        clip_duration_seconds = annotation["clip_duration_seconds"]

        # build information dictionary
        info = {
            "question_type": "multiple_choice",
            "question_id": index,
            "question": question,
            "gt_answer": gt_answer,
            "gt_answer_text": gt_answer_text,
            "video_path": [video_path, last_frame_path],
            "domain": domain,
            "video_type": video_type,
            "clip_duration_seconds": clip_duration_seconds,
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
        dataset_path = "/data/lishuaibing/test_process/robobrain2-benchmark/EgoPlan-Bench2"
        # dataset_path = "/data/robovqa/convert_data/val"
        split = "test"
        processed_dataset_path = "/data/lishuaibing/test_process/robobrain2-benchmark/EgoPlan-Bench2"
        # processed_dataset_path = "/data/robovqa/convert_data/"
    process(Cfg)

