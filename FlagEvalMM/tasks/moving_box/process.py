import json
import os
import os.path as osp
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from PIL import Image


ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_SYSTEM_PROMPT = (
    "你是抓取规划器。任务：从堆垛中选择要抓取的目标箱，并给出 bbox、side_flag 和 stance_flag。"
    "输出格式：{\"target_bbox_2d\":[x1,y1,x2,y2],\"side_flag\":\"current|left|right|opposite\","
    "\"stance_flag\":\"left|middle|right\"}。只返回一个 JSON，不要其他文字。"
)
DEFAULT_USER_TEXT = "输出当前的最优拆垛信息"


# ----------------- 基础解析工具 -----------------
def extract_json(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        cand = m.group(0).strip()
        try:
            json.loads(cand)
            return cand
        except Exception:
            pass
    cleaned = text.strip().strip("`").strip()
    m2 = re.search(r"\{[\s\S]*\}", cleaned)
    if m2:
        cand2 = m2.group(0).strip()
        try:
            json.loads(cand2)
            return cand2
        except Exception:
            pass
    return None


def gt_image_paths_from_dialog(gt_obj: Dict[str, Any]) -> List[str]:
    """
    从 GT JSONL 的对象中取出“所有”图片路径，支持多图。
    兼容：
      - images: ["xxx.jpg", ...]
      - images: [{"image_path"/"path"/"file_name"/"image": "..."}]
      - 旧字段 image / img
    """
    paths: List[str] = []

    images = gt_obj.get("images")
    if isinstance(images, list):
        for it in images:
            if isinstance(it, str):
                paths.append(it)
            elif isinstance(it, dict):
                for key in ["image_path", "path", "file_name", "image"]:
                    v = it.get(key)
                    if isinstance(v, str):
                        paths.append(v)
                        break

    # 兜底：老格式只一张图
    if not paths:
        if isinstance(gt_obj.get("image"), str):
            paths.append(gt_obj["image"])
        elif isinstance(gt_obj.get("img"), str):
            paths.append(gt_obj["img"])

    # 去重、去空
    clean = []
    for p in paths:
        if isinstance(p, str) and p.strip() and p not in clean:
            clean.append(p)
    return clean


def parse_gt_schema_or_dialog(gt_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(gt_obj, dict):
        return None

    # 1) 顶层就是 schema
    if isinstance(gt_obj.get("target_bbox_2d"), list):
        return gt_obj

    # 2) 对话式：从最后一个 assistant 里抽 JSON
    msgs = gt_obj.get("messages")
    if isinstance(msgs, list) and msgs:
        for m in reversed(msgs):
            if not (isinstance(m, dict) and m.get("role") == "assistant"):
                continue
            content = m.get("content")

            if isinstance(content, str):
                js = extract_json(content)
                if js:
                    try:
                        obj = json.loads(js)
                        if isinstance(obj.get("target_bbox_2d"), list):
                            return obj
                    except Exception:
                        pass

            if isinstance(content, list):
                joined = " ".join(str(x) for x in content)
                js = extract_json(joined)
                if js:
                    try:
                        obj = json.loads(js)
                        if isinstance(obj.get("target_bbox_2d"), list):
                            return obj
                    except Exception:
                        pass

    return None


def _get_prompts_from_raw(raw: Dict[str, Any]) -> Tuple[str, str]:
    system_prompt = DEFAULT_SYSTEM_PROMPT
    user_text = DEFAULT_USER_TEXT

    msgs = raw.get("messages")
    if isinstance(msgs, list) and msgs:
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "system":
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    system_prompt = c.strip()
                break

        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    user_text = c.strip()
                break

    return system_prompt, user_text


def make_sample_id(img_path: str, keep_parts: int = 5) -> str:
    p = Path(os.path.normpath(img_path))
    parts = [x for x in p.parts if x not in (os.sep, "")]
    tail = parts[-keep_parts:] if len(parts) >= keep_parts else parts
    if tail:
        tail = list(tail[:-1]) + [p.stem]
    else:
        tail = [p.stem]

    s = "_".join(tail)
    s = s.replace(":", "")
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or p.stem


def _resolve_image_path(img_path: str, dataset_root: str) -> str:
    """jsonl 里可能是绝对路径，也可能是相对路径。这里做兜底拼接。"""
    img_path = img_path.replace("/ogi-code/transport_planner_data/real", "/code1/data/robobrain2-benchmark/moving_box/all_data")
    if osp.exists(img_path):
        return img_path
    cand = osp.join(dataset_root, img_path)
    if osp.exists(cand):
        return cand
    return img_path  # 最后兜底，后续会报错但不至于崩


# ----------------- 主处理函数 -----------------
def process(cfg):
    """
    读取 labels 中文件名含 yungu_1 的 jsonl
    输出标准 data.json + 拷贝“所有 images”到 processed 目录
    img_path 写成 list，FlagEvalMM 可以多图输入。
    """
    data_dir = cfg.dataset_path
    split = getattr(cfg, "split", "val")
    name = getattr(cfg, "name", "")

    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    img_dir = osp.join(output_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    labels_dir = osp.join(data_dir, "labels")
    if not osp.isdir(labels_dir):
        raise FileNotFoundError(f"labels dir not found: {labels_dir}")

    jsonl_files = sorted(
        [
            osp.join(labels_dir, fn)
            for fn in os.listdir(labels_dir)
            if fn.endswith(".jsonl") and "yungu_1" in fn
        ]
    )

    if not jsonl_files:
        print(f"[WARN] 未找到包含 yungu_1 的 jsonl: {labels_dir}")

    content: List[Dict[str, Any]] = []
    question_id_set = set()
    seen_primary_img_keys = set()

    for jp in jsonl_files:
        with open(jp, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    raw = json.loads(line)
                except Exception as e:
                    print(f"[WARN] JSONL 解析失败 {jp}:{ln} -> {e}")
                    continue

                # 1) 拿到所有 images
                img_paths = gt_image_paths_from_dialog(raw)
                if not img_paths:
                    print(f"[WARN] 缺少 images/image/img 字段，跳过: {jp}:{ln}")
                    continue

                # 2) 选第一张作为“主图”（GT bbox 所在图）
                primary_path = _resolve_image_path(img_paths[0], data_dir)
                primary_key = osp.normpath(primary_path)
                if primary_key in seen_primary_img_keys:
                    # 同一主图在多个 jsonl 里重复，直接跳过重复样本
                    continue
                seen_primary_img_keys.add(primary_key)

                sample_id = make_sample_id(primary_path)

                # question_id 去重
                if sample_id in question_id_set:
                    k = 2
                    new_id = f"{sample_id}__dup{k}"
                    while new_id in question_id_set:
                        k += 1
                        new_id = f"{sample_id}__dup{k}"
                    sample_id = new_id
                question_id_set.add(sample_id)

                # 3) 解析 GT
                gt_struct = parse_gt_schema_or_dialog(raw) or {}

                # 4) 加载主图，拿尺寸
                try:
                    im0 = Image.open(primary_path).convert("RGB")
                    image_width, image_height = im0.width, im0.height
                except Exception as e:
                    print(f"[WARN] 主图读取失败 {primary_path} -> {e}")
                    continue

                # 5) 拷贝所有 images 到 processed 目录，并生成相对路径 list
                rel_img_paths: List[str] = []
                for idx, p in enumerate(img_paths):
                    abs_p = _resolve_image_path(p, data_dir)
                    try:
                        im = Image.open(abs_p).convert("RGB")
                    except Exception as e:
                        print(f"[WARN] 子图读取失败 {abs_p} -> {e}")
                        continue

                    ext = Path(abs_p).suffix.lower()
                    if ext not in ALLOWED_EXTS:
                        ext = ".jpg"
                    img_name = f"{sample_id}_{idx+1}{ext}"
                    rel_path = f"img/{img_name}"
                    save_path = osp.join(output_dir, rel_path)
                    try:
                        im.save(save_path)
                    except Exception as e:
                        print(f"[WARN] 图片保存失败 {abs_p} -> {e}")
                        continue
                    rel_img_paths.append(rel_path)

                if not rel_img_paths:
                    print(f"[WARN] 所有子图保存失败，跳过样本 {sample_id}")
                    continue

                system_prompt, user_text = _get_prompts_from_raw(raw)
                # 是否在 question 里显式标明多图，这里保持简单，交给 PromptTemplate 处理；
                # 如果你想要 <image 1> <image 2> 这种占位，也可以改成：
                # image_tokens = " ".join(f"<image {i+1}>" for i in range(len(rel_img_paths)))
                # question = f"{image_tokens} {system_prompt}\n{user_text}"
                # question = f"{system_prompt}\n{user_text}"

                info = {
                    "system_prompt": system_prompt,
                    "question": user_text,
                    # 直接存 GT dict，方便 evaluate 同时评 bbox + side + stance
                    "answer": {
                        "target_bbox_2d": gt_struct.get("target_bbox_2d"),
                        "side_flag": gt_struct.get("side_flag"),
                        "stance_flag": gt_struct.get("stance_flag"),
                    },
                    "question_id": sample_id,
                    "img_path": rel_img_paths,   # 这里是 list
                    "image_width": image_width,  # 以主图尺寸为准
                    "image_height": image_height,
                    "question_type": "bbox",
                    "sub_task": "yungu diantiting",
                }
                content.append(info)
                print(f"Processed {sample_id} with {len(rel_img_paths)} images.")

    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(content)} items. Data saved to {output_file}")


if __name__ == "__main__":
    class config:
        dataset_path = "/code1/data/robobrain2-benchmark/moving_box/all_data"
        split = "val"
        processed_dataset_path = "/code1/data/robobrain2-benchmark/moving_box"
        processor = "process.py"

    process(config)

