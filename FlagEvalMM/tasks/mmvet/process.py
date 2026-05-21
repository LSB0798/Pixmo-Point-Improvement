import os.path as osp
import json


def parse_question(question):
    splits = question.split("<IMG>")
    image_paths = []
    new_question = ""
    cnt = 0
    for split in splits:
        if split.endswith(".png") or split.endswith(".jpg"):
            image_paths.append(f"images/{split}")
            new_question += f"<image {cnt + 1}>"
            cnt += 1
        else:
            new_question += " " + split
    return new_question.strip(), image_paths


def process(cfg):
    data_dir = cfg.dataset_path
    output_dir = cfg.processed_dataset_path
    data = json.load(open(osp.join(data_dir, "mm-vet-v2.json")))
    data_reformat = []
    id_st = set()
    for i, d in data.items():
        question, image_path = parse_question(d["question"])
        if i in id_st:
            print(f"Duplicate id: {i}")

        id_st.add(i)
        new_item = {
            "question_id": i,
            "question": question,
            "question_type": "open",
            "answer": d["answer"],
            "capability": d["capability"],
            "added_in": d["added_in"],
            "img_path": image_path,
        }
        data_reformat.append(new_item)
    with open(osp.join(output_dir, "data.json"), "w") as f:
        json.dump(data_reformat, f, indent=2)
