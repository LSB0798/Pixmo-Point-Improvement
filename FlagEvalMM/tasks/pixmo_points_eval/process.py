import json
import os
import os.path as osp
from io import BytesIO
import glob
import numpy as np

from datasets import load_dataset
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_session():
    s = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.6,
        status_forcelist=[403, 408, 429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def guess_ext_from_content_type(ct: str) -> str:
    ct = (ct or "").lower()
    if "png" in ct:
        return ".png"
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "webp" in ct:
        return ".webp"
    return ".jpg"


def download_image(session: requests.Session, url: str, timeout=(8, 25)):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }

    resp = session.get(url, headers=headers, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}")

    ct = resp.headers.get("Content-Type", "")
    if "image" not in ct.lower():
        raise RuntimeError(f"Non-image content-type: {ct}")

    data = resp.content
    Image.open(BytesIO(data)).verify()
    img = Image.open(BytesIO(data)).convert("RGB")
    ext = guess_ext_from_content_type(ct)
    return img, ext


def find_existing_img(img_dir: str, idx: int):
    """查找 img/{idx:06d}.* 任意后缀"""
    pattern = osp.join(img_dir, f"{idx:06d}.*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None, None, None
    abs_path = matches[0]
    rel_path = f"img/{osp.basename(abs_path)}"
    try:
        with Image.open(abs_path) as im:
            w, h = im.size
        return rel_path, w, h
    except Exception:
        return None, None, None


def safe_label(label: str) -> str:
    return " ".join((label or "").strip().split())


def merge_masks_to_one_png(masks, out_abs_path: str):
    """
    masks: list of 2D bool/0-1 arrays (H,W)
    输出: 单张合并 mask（logical OR），保存为 PNG（0/255）
    """
    if masks is None or len(masks) == 0:
        return False

    merged = None
    for m in masks:
        arr = np.asarray(m, dtype=np.uint8)
        if merged is None:
            merged = (arr > 0).astype(np.uint8)
        else:
            merged |= (arr > 0).astype(np.uint8)

    merged = (merged * 255).astype(np.uint8)

    im = Image.fromarray(merged)   # <- 不传 mode
    if im.mode != "L":
        im = im.convert("L")

    im.save(out_abs_path, format="PNG", optimize=True, compress_level=9)
    return True



def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    name = ""  # 保持你原逻辑

    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    img_dir = osp.join(output_dir, "img")
    mask_dir = osp.join(output_dir, "mask")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    data = load_dataset(data_dir, split=split)
    session = build_session()

    content = []
    failures = []

    prompt_tmpl = getattr(
        cfg,
        "prompt_template",
        "Please point out all the {label}.",
    )
    sub_task = getattr(cfg, "sub_task", "pixmo_points_eval")

    for idx, ann in enumerate(data):
        image_url = ann["image_url"]
        label = safe_label(ann["label"])
        points = ann["points"]  # 保留（你需要的话）
        masks = ann["masks"]

        # ---- 图片：已存在则跳过下载 ----
        img_rel_path, image_width, image_height = find_existing_img(img_dir, idx)
        if getattr(cfg, "download_images", True) and img_rel_path is None:
            print(f"Downloading image {idx} from {image_url}")
            try:
                img, ext = download_image(session, image_url)
                image_width, image_height = img.size
                img_filename = f"{idx:06d}{ext}"
                img_rel_path = f"img/{img_filename}"
                img.save(osp.join(output_dir, img_rel_path))
            except Exception as e:
                failures.append({"idx": idx, "image_url": image_url, "error": str(e)})

        # ---- 合并所有 masks 成一张 mask PNG ----
        question_id = f"{sub_task}_{idx}"
        mask_rel = f"mask/{question_id}.png"
        mask_abs = osp.join(output_dir, mask_rel)

        if not osp.exists(mask_abs) or osp.getsize(mask_abs) == 0:
            ok = merge_masks_to_one_png(masks, mask_abs)
            if not ok:
                # 没有 masks 的样本：这里选择跳过（更干净）
                print(f"[{idx}] 无有效 masks，跳过样本。")
                continue

        question = prompt_tmpl.format(label=label)

        info = {
            "question_id": question_id,
            "question": question,
            "sub_task": sub_task,
            "answer": mask_rel,          # 仍是单个字符串
            "question_type": "point",
            "img_path": img_rel_path,
            "image_width": image_width,
            "image_height": image_height,
            "mask_path": mask_rel,       # 仍是单个字符串
            # 可选：如果你评测要用 points，也可以保留
            # "points": points,
        }
        content.append(info)

    out_json = osp.join(output_dir, "data.json")
    with open(out_json, "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    fail_json = osp.join(output_dir, "download_failures.json")
    with open(fail_json, "w") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} items. Data saved to {out_json}")
    print(f"Download failures: {len(failures)}. Saved to {fail_json}")


if __name__ == "__main__":
    class config:
        dataset_path = "/work/llm_team/planning/pixmo-points-eval"
        split = "test"
        processed_dataset_path = "/work/llm_team/planning/pixmo-points-eval"
        download_images = True

        # 你想要的 prompt，在这里改
        prompt_template = "Please point out the {label}."

        # 类似 placement / counting / pointing 这种子任务名
        sub_task = "pixmo_points_eval"

    process(config)

