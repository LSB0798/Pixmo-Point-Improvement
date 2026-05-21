#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
import logging
from pathlib import Path
from typing import Any, Optional, Set, Tuple

from PIL import Image
from vllm import LLM, SamplingParams

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # fallback


INSTRUCTION = (
    "Detect the box in the image and predict the 3D box. "
    'Output JSON: [{"bbox_3d":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw],'
    '"label":"category"}]'
)

SYSTEM_PROMPT = (
    "You are a vision-language perception model. "
    "Return ONLY valid JSON. No extra text, no markdown."
)


def extract_json(text: str) -> Optional[Any]:
    s = text.strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    l = s.find("[")
    r = s.rfind("]")
    if 0 <= l < r:
        try:
            return json.loads(s[l : r + 1])
        except Exception:
            pass

    l = s.find("{")
    r = s.rfind("}")
    if 0 <= l < r:
        try:
            return json.loads(s[l : r + 1])
        except Exception:
            pass

    return None


def iter_image_paths(root_dir: Path):
    # for box_dir in sorted([p for p in root_dir.iterdir() if p.is_dir() and p.name.startswith("box_")]):
    for box_dir in sorted([p for p in root_dir.iterdir() if p.is_dir() and p.name.startswith("img")]):
        # color_dir = box_dir / "color"
        color_dir = Path(box_dir)
        if not color_dir.is_dir():
            continue
        # png 或者 jpg
        for img_path in sorted(color_dir.glob("*.png")) + sorted(color_dir.glob("*.jpg")):
            yield box_dir.name, img_path


def batched(iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_done_set_and_truncate_if_needed(
    out_path: Path, redo_failed: bool
) -> Tuple[Set[str], Set[str], int, int]:
    """
    返回:
      done_abs: 已处理过的绝对路径集合
      done_rel: 已处理过的相对路径集合
      done_total: 输出文件中已完整记录的条数
      done_ok:    输出文件中 ok=true 的条数
    同时：若末尾存在不完整 JSONL 行，会截断到最后一条完整行。
    """
    done_abs: Set[str] = set()
    done_rel: Set[str] = set()
    done_total = 0
    done_ok = 0

    if not out_path.exists() or out_path.stat().st_size == 0:
        return done_abs, done_rel, done_total, done_ok

    last_good_pos = 0
    with out_path.open("rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                last_good_pos = pos
                break
            try:
                obj = json.loads(line.decode("utf-8"))
                done_total += 1
                ok = bool(obj.get("ok", True))
                if ok:
                    done_ok += 1

                # redo_failed=True 时，失败的不加入 done 集合（允许重跑）
                if redo_failed and (not ok):
                    last_good_pos = f.tell()
                    continue

                ip = obj.get("image_path")
                rip = obj.get("rel_image_path")
                if ip:
                    done_abs.add(str(Path(ip).resolve()))
                if rip:
                    done_rel.add(rip)

                last_good_pos = f.tell()
            except Exception:
                # 这一行可能不完整（crash/kill），截断到上一条完整记录
                break

    file_size = out_path.stat().st_size
    if last_good_pos < file_size:
        with out_path.open("r+b") as f:
            f.truncate(last_good_pos)

    return done_abs, done_rel, done_total, done_ok


def make_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("infer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, default=".", help="包含 box_* 文件夹的根目录")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--out", type=str, default="qwen3_vl_4b_preds.jsonl")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)

    # 断点与实时写入
    ap.add_argument("--resume", action="store_true", help="若 out 已存在，则读取已完成记录并跳过")
    ap.add_argument("--redo_failed", action="store_true", help="配合 --resume：重跑 ok=false 的记录")
    ap.add_argument("--flush_every", type=int, default=1, help="每写入多少条 flush 一次（默认每条）")
    ap.add_argument("--fsync", action="store_true", help="每次 flush 后执行 os.fsync（更稳但更慢）")

    # logging
    ap.add_argument("--log", type=str, default=None, help="log 文件路径（默认 out 同名 .log）")
    ap.add_argument("--log_every", type=int, default=200, help="每处理多少张图写一次统计到 log（0=只写开头/结尾）")

    args = ap.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log).expanduser().resolve() if args.log else out_path.with_suffix(out_path.suffix + ".log")
    logger = make_logger(log_path)

    # 断点：读取已完成集合（并处理不完整尾行）
    done_abs: Set[str] = set()
    done_rel: Set[str] = set()
    done_total = 0
    done_ok = 0
    if args.resume:
        done_abs, done_rel, done_total, done_ok = load_done_set_and_truncate_if_needed(
            out_path, redo_failed=args.redo_failed
        )
        logger.info(f"[RESUME] existing_records={done_total} existing_ok={done_ok} out={out_path}")

    # 扫描全部图片
    all_items = list(iter_image_paths(root_dir))
    if not all_items:
        logger.warning(f"[WARN] 在 {root_dir} 下没有找到 box_*/color/*.png")
        return

    # 过滤掉已完成
    filtered_items = []
    skipped = 0
    for box_name, img_path in all_items:
        abs_key = str(img_path.resolve())
        rel_key = None
        try:
            rel_key = str(img_path.resolve().relative_to(root_dir))
        except Exception:
            rel_key = None

        if abs_key in done_abs or (rel_key and rel_key in done_rel):
            skipped += 1
            continue
        filtered_items.append((box_name, img_path))

    total_pending = len(filtered_items)
    total_all = len(all_items)

    logger.info(
        f"[INFO] total_all={total_all} skipped_by_resume={skipped} pending={total_pending} "
        f"batch_size={args.batch_size} max_model_len={args.max_model_len} max_tokens={args.max_tokens}"
    )
    logger.info(f"[INFO] model={args.model}")

    # 初始化 vLLM（离线推理，不起 server）
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=str(root_dir),  # 允许读取 root_dir 下的 file://...
    )

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # 全局统计（本次运行新增）
    new_total = 0
    new_ok = 0

    # 全局进度条
    use_pbar = (tqdm is not None)
    pbar = tqdm(total=total_pending, dynamic_ncols=True, desc="Infer", unit="img") if use_pbar else None

    t0 = time.time()

    # 追加写（实时落盘）
    with out_path.open("a", encoding="utf-8") as f:
        for batch in batched(filtered_items, args.batch_size):
            conversations = []
            metas = []

            for box_name, img_path in batch:
                abs_path = str(img_path.resolve())
                try:
                    rel_path = str(img_path.resolve().relative_to(root_dir))
                except Exception:
                    rel_path = None

                # 关键：每张图自己生成 file:// URL
                img_url = Path(abs_path).resolve().as_uri()

                conv = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": img_url}},
                            {"type": "text", "text": INSTRUCTION},
                        ],
                    },
                ]
                conversations.append(conv)
                metas.append({"box_dir": box_name, "image_path": abs_path, "rel_image_path": rel_path})

            outputs = llm.chat(
                messages=conversations,
                sampling_params=sampling,
                use_tqdm=False,
            )

            for meta, out in zip(metas, outputs):
                text = out.outputs[0].text.strip()
                parsed = extract_json(text)
                ok = (parsed is not None)

                record = {**meta, "raw": text, "json": parsed, "ok": ok}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                new_total += 1
                if ok:
                    new_ok += 1

                if args.flush_every > 0 and (new_total % args.flush_every == 0):
                    f.flush()
                    if args.fsync:
                        os.fsync(f.fileno())

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(ok=f"{new_ok}/{new_total}")

                if args.log_every and (new_total % args.log_every == 0):
                    elapsed = time.time() - t0
                    speed = new_total / elapsed if elapsed > 0 else 0.0
                    rate = (new_ok / new_total) if new_total > 0 else 0.0
                    logger.info(
                        f"[STAT] new_total={new_total} new_ok={new_ok} ok_rate={rate:.4f} "
                        f"speed={speed:.2f} img/s latest={meta.get('rel_image_path') or meta['image_path']}"
                    )

        f.flush()
        if args.fsync:
            os.fsync(f.fileno())


    if pbar is not None:
        pbar.close()

    elapsed = time.time() - t0
    speed = new_total / elapsed if elapsed > 0 else 0.0
    rate = (new_ok / new_total) if new_total > 0 else 0.0

    # 总体（含 resume 旧记录）
    total_records = done_total + new_total
    total_ok = done_ok + new_ok
    total_rate = (total_ok / total_records) if total_records > 0 else 0.0

    logger.info(
        f"[DONE] new_total={new_total} new_ok={new_ok} new_ok_rate={rate:.4f} "
        f"elapsed={elapsed:.1f}s speed={speed:.2f} img/s"
    )
    logger.info(
        f"[DONE] overall_records={total_records} overall_ok={total_ok} overall_ok_rate={total_rate:.4f} "
        f"out={out_path} log={log_path}"
    )


if __name__ == "__main__":
    main()


# /code1/data/robobrain2-benchmark/testset_2w_960x576
# /code1/data/robobrain2-benchmark/moving_box/val
# /code1/train_logs/merge_lora/thinker/v82-20251112-104714-alignment/iter_0007800 \
# export CUDA_VISIBLE_DEVICES=3
# python scripts_to_zhuming/run_qwen3_vl_vllm.py \
#     --root_dir /code1/data/robobrain2-benchmark/moving_box/val \
#     --resume \
#     --batch_size 16 \
#     --gpu_mem_util 0.5 \
#     --max_tokens 2048 \
#     --max_model_len 32768 \
#     --temperature 0.6 \
#     --top_p 0.9 \
#     --tensor_parallel_size 1 \
#     --model /code1/train_logs/merge_lora/thinker/v82-20251112-104714-alignment/iter_0007800 \
#     --out to_zhuming/qwen3_vl_true.jsonl \
#     --log to_zhuming/qwen3_vl_true.log