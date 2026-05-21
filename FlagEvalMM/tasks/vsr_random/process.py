import io
import json
import os
import os.path as osp
import time
import threading
from typing import Any, Dict, List, Tuple

import requests
from requests.exceptions import SSLError
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ---------- JSONL ----------
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"JSONL parse error at line {ln}: {e}")
    return items


# ---------- Thread-local session (faster than creating per request) ----------
_tls = threading.local()

def _get_session() -> requests.Session:
    if getattr(_tls, "session", None) is None:
        s = requests.Session()
        # optional: you can tune retries via adapters, but keep simple here
        _tls.session = s
    return _tls.session


# ---------- Download with SSL fallback ----------
def _download_image(url: str, save_path: str, max_retries: int = 3, timeout: int = 30) -> Tuple[int, int]:
    """
    Download image from url and save to save_path as RGB jpg.
    Return (width, height).
    """
    os.makedirs(osp.dirname(save_path), exist_ok=True)

    # If already exists, open and return
    if osp.exists(save_path):
        try:
            img = Image.open(save_path).convert("RGB")
            return img.width, img.height
        except Exception:
            pass  # corrupted -> re-download

    candidates = [url]
    if url.startswith("https://"):
        candidates.append("http://" + url[len("https://"):])

    last_err = None
    sess = _get_session()

    for attempt in range(max_retries):
        for u in candidates:
            # 1) verify=True
            try:
                r = sess.get(u, timeout=timeout, allow_redirects=True, verify=True)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                img.save(save_path, format="JPEG", quality=95)
                return img.width, img.height
            except SSLError as e:
                last_err = e
                # 2) SSL error -> try verify=False for https
                if u.startswith("https://"):
                    try:
                        r = sess.get(u, timeout=timeout, allow_redirects=True, verify=False)
                        r.raise_for_status()
                        img = Image.open(io.BytesIO(r.content)).convert("RGB")
                        img.save(save_path, format="JPEG", quality=95)
                        return img.width, img.height
                    except Exception as e2:
                        last_err = e2
                        continue
                continue
            except Exception as e:
                last_err = e
                continue

        time.sleep(0.5 * (attempt + 1))

    raise RuntimeError(f"Failed to download image: {url}\nLast error: {last_err}")


def _parallel_download(jobs: List[Tuple[int, str, str]], max_workers: int = 16):
    """
    jobs: [(index, url, abs_path), ...]
    return: dict index -> (w,h)
    """
    results: Dict[int, Tuple[int, int]] = {}
    errors: List[Tuple[int, str]] = []

    iterator = jobs
    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=len(jobs), desc="Downloading images", ncols=100)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_download_image, url, path): idx for idx, url, path in iterator}
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                errors.append((idx, str(e)))
            finally:
                if pbar is not None:
                    pbar.update(1)

    if pbar is not None:
        pbar.close()

    if errors:
        # 也可以选择直接 raise，让流程中断；这里给出汇总，便于你定位失败样本
        print(f"[WARN] {len(errors)} images failed to download. Showing first 10:")
        for idx, msg in errors[:10]:
            print(f"  - idx={idx}: {msg}")

    return results


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    name = ""

    jsonl_path = osp.join(data_dir, f"{split}.jsonl")
    if not osp.exists(jsonl_path):
        raise FileNotFoundError(f"Cannot find split file: {jsonl_path}")

    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    img_dir = osp.join(output_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    data = _read_jsonl(jsonl_path)

    # 先构建 content + 并行下载任务
    content: List[Dict[str, Any]] = []
    jobs: List[Tuple[int, str, str]] = []

    for index, item in enumerate(data):
        question_id = f"{split}_{index}"
        image_rel_path = f"img/{question_id}.jpg"
        image_abs_path = osp.join(output_dir, image_rel_path)

        image_url = item.get("image_link", "")
        if not image_url:
            raise KeyError(f"Missing 'image_link' in item {index}. Keys={list(item.keys())}")

        lab = int(item["label"])
        answer = "yes" if lab == 1 else "no"

        info = {
            "question_id": question_id,
            "question": (item.get("caption") or "").strip(),
            "category": (item.get("relation") or "").strip(),
            "answer": answer,
            "question_type": "yes-no",
            "img_path": image_rel_path,
            "image_width": 0,   # 下载后回填
            "image_height": 0,  # 下载后回填
            "source_image": item.get("image", ""),
            "source_image_link": image_url,
        }
        content.append(info)
        jobs.append((index, image_url, image_abs_path))

    # 并行下载并回填宽高
    max_workers = int(getattr(cfg, "num_workers", 16))  # 你可以在 cfg 里加 num_workers
    wh_map = _parallel_download(jobs, max_workers=max_workers)

    for idx, (w, h) in wh_map.items():
        content[idx]["image_width"] = w
        content[idx]["image_height"] = h

    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} items. Data saved to {output_file}")


if __name__ == "__main__":
    class config:
        # dataset_path = "/work/llm_team/planning/vsr_random"
        dataset_path = "/code1/data/robobrain2-benchmark/vsr_random"
        split = "test"
        # processed_dataset_path = "/work/llm_team/planning/vsr_random"
        processed_dataset_path = "/code1/data/robobrain2-benchmark/vsr_random"
        processor = "process.py"

    process(config)