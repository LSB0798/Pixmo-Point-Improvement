#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指示灯状态选项选择评估脚本
VLM 接受的 image 提前在 indicator lights 上打上了蓝色圆圈
模型输出选项字母（A-E），与 GT 对比计算准确率和混淆矩阵
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from transformers import AutoProcessor
from vllm import LLM
from tqdm import tqdm
from PIL import Image
from vllm import SamplingParams
from collections import defaultdict

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =========================
# 配置
# =========================
# 状态定义
STATES = {"green", "yellow", "red", "off", "unknown"}

# 选项映射：字母 -> 状态
OPTION_TO_STATE = {
    "A": "green",
    "B": "yellow", 
    "C": "red",
    "D": "off",
    "E": "unknown",
}

STATE_TO_OPTION = {v: k for k, v in OPTION_TO_STATE.items()}

# Prompt 模板 - 上灯
PROMPT_UP = """The indicator light in the image is marked with a blue circle. Please identify the status (color) of this indicator light. Choose from the following options:
A. Green
B. Yellow
C. Red
D. off (indicator light is off)
E. unknown (blocked or hard to distinguish)
Answer with the option's letter from the given choices directly."""

# Prompt 模板 - 下灯
PROMPT_LOW = """The indicator light in the image is marked with a blue circle. Please identify the status (color) of this indicator light. Choose from the following options:
A. Green
B. Yellow
C. Red
D. off (indicator light is off)
E. unknown (blocked or hard to distinguish)
Answer with the option's letter from the given choices directly."""


def normalize_state(s: str) -> str:
    """标准化状态名"""
    s = (s or "").strip().lower()
    if s == "unkown":
        s = "unknown"
    return s if s in STATES else "unknown"


def parse_states_from_jsonl_name(jsonl_path: str) -> Tuple[str, str]:
    """
    由文件名解析上下灯状态：
      green_unknown_val.jsonl -> ("green","unknown")
    兼容多余下划线：red_unknown__val.jsonl
    """
    name = Path(jsonl_path).name
    base = name
    for suf in ("_val.jsonl", ".jsonl"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    toks = [t for t in base.split("_") if t]  # 去掉空 token
    up = normalize_state(toks[0]) if len(toks) >= 1 else "unknown"
    low = normalize_state(toks[1]) if len(toks) >= 2 else "unknown"
    return up, low


def make_sample_id(img_path: str, keep_parts: int = 6) -> str:
    """用路径末尾几段拼出稳定且可读的 id。"""
    p = Path(os.path.normpath(img_path))
    parts = [x for x in p.parts if x not in (os.sep, "")]
    tail = parts[-keep_parts:] if len(parts) >= keep_parts else parts
    if tail:
        tail = list(tail[:-1]) + [p.stem]
    s = "_".join(tail)
    s = s.replace(":", "")
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or p.stem


# =========================
# GT 构建：jsonl -> gt_map（以 image 为单位）
# =========================
def get_img_path_from_jsonl_obj(obj: Dict[str, Any]) -> Optional[str]:
    images = obj.get("images")
    if isinstance(images, list) and images and isinstance(images[0], str):
        return images[0]
    if isinstance(obj.get("image"), str):
        return obj["image"]
    return None


def load_gt_sets(gt_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    按"图片"去重，构造 gt_map：
      gt_map[stem] = {
        "stem": ...,
        "img_path": ...,
        "up_state": ...,
        "low_state": ...,
      }
    """
    gt_map: Dict[str, Dict[str, Any]] = {}
    seen_img = set()

    for jsonl_path in gt_paths:
        up_state, low_state = parse_states_from_jsonl_name(jsonl_path)

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    print(f"[WARN] JSONL 解析失败: {jsonl_path}:{ln}")
                    continue

                img_path = get_img_path_from_jsonl_obj(obj)
                if not isinstance(img_path, str) or not img_path:
                    print(f"[WARN] 缺少 images[0]: {jsonl_path}:{ln}")
                    continue

                img_key = os.path.normpath(img_path)
                if img_key in seen_img:
                    continue
                seen_img.add(img_key)

                stem = make_sample_id(img_path)

                gt_map[stem] = {
                    "stem": stem,
                    "img_path": img_path,
                    "up_state": up_state,
                    "low_state": low_state,
                }

    print(f"[GT] Loaded images: {len(gt_map)}")
    return gt_map


# =========================
# 推理：Qwen3-VL + vLLM（每图两次：upper/lower）
# =========================
def load_vllm(model_path: str, max_num_seqs: int, gpu_memory_utilization: float) -> Tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=20000,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False,
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_processor_kwargs={"max_dynamic_patch": 4},
    )
    return processor, llm


def build_one_prompt(processor, img: Image.Image, text: str) -> Tuple[str, List[Image.Image]]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt, [img]


def extract_choice_from_text(text: str) -> Optional[str]:
    """
    从模型输出文本中提取选项字母 A-E
    优先匹配单个字母，然后是字母加括号等形式
    """
    if not isinstance(text, str):
        return None
    
    text = text.strip()
    if not text:
        return None
    
    # 直接就是单个字母
    if len(text) == 1 and text.upper() in OPTION_TO_STATE:
        return text.upper()
    
    # 匹配 "A"、"A."、"(A)"、"[A]" 等形式
    patterns = [
        r'^\s*([A-Ea-e])\s*$',  # 纯字母
        r'^\s*\(?([A-Ea-e])\)?[.:\)]?\s*',  # (A) 或 A. 开头
        r'["\']?([A-Ea-e])["\']?',  # "A" 或 'A'
        r'answer\s*[=:]?\s*["\']?([A-Ea-e])["\']?',  # answer: A
        r'option\s*[=:]?\s*["\']?([A-Ea-e])["\']?',  # option: A
        r'\b([A-Ea-e])\b',  # 单词边界匹配
    ]
    
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    
    # 兜底：查找任何 A-E 字母
    for char in text.upper():
        if char in OPTION_TO_STATE:
            return char
    
    return None


def infer_batch(
    processor,
    llm,
    batch: List[Tuple[str, str]],
    max_new_tokens: int,
) -> Dict[str, Dict[str, Any]]:
    """
    batch: [(stem, img_path), ...]
    return:
      results[stem] = {
        "pred_choice_up": "A" / "B" / "C" / "D" / "E" / None,
        "pred_choice_low": "A" / "B" / "C" / "D" / "E" / None,
        "raw_text_up": "...",
        "raw_text_low": "...",
      }
    """
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    llm_inputs = []
    metas = []  # 每条 input 对应 (stem, which)

    for stem, img_path in batch:
        img = Image.open(img_path).convert("RGB")

        p_up, mm_up = build_one_prompt(processor, img, PROMPT_UP)
        llm_inputs.append({"prompt": p_up, "multi_modal_data": {"image": mm_up}})
        metas.append((stem, "up"))

        p_low, mm_low = build_one_prompt(processor, img, PROMPT_LOW)
        llm_inputs.append({"prompt": p_low, "multi_modal_data": {"image": mm_low}})
        metas.append((stem, "low"))

    outputs = llm.generate(llm_inputs, sampling_params=sampling_params)

    results: Dict[str, Dict[str, Any]] = {}
    for (stem, which), out in zip(metas, outputs):
        text = ""
        try:
            text = out.outputs[0].text.strip()
        except Exception:
            text = ""

        choice = extract_choice_from_text(text)

        if stem not in results:
            results[stem] = {
                "pred_choice_up": None,
                "pred_choice_low": None,
                "raw_text_up": "",
                "raw_text_low": "",
            }

        if which == "up":
            results[stem]["pred_choice_up"] = choice
            results[stem]["raw_text_up"] = text
        else:
            results[stem]["pred_choice_low"] = choice
            results[stem]["raw_text_low"] = text

    return results


def run_infer_from_gtset(
    model_path: str,
    gt_map: Dict[str, Dict[str, Any]],
    out_dir: str,
    max_new_tokens: int,
    batch_size: int,
    gpu_memory_utilization: float,
):
    os.makedirs(out_dir, exist_ok=True)

    # 注意：每张图两条请求，所以 max_num_seqs 建议 >= 2*batch_size
    processor, llm = load_vllm(
        model_path=model_path,
        max_num_seqs=max(1, batch_size * 2),
        gpu_memory_utilization=gpu_memory_utilization,
    )

    samples = []
    for stem, rec in sorted(gt_map.items()):
        img_path = rec.get("img_path")
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            print(f"[WARN] image not found, skip: {img_path}")
            continue
        samples.append((stem, img_path))

    with tqdm(total=len(samples), desc="Infer[vLLM]") as pbar:
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            batch_results = infer_batch(processor, llm, batch, max_new_tokens=max_new_tokens)

            for stem, img_path in batch:
                rec = gt_map[stem]
                pred = batch_results.get(stem, {
                    "pred_choice_up": None,
                    "pred_choice_low": None,
                    "raw_text_up": "",
                    "raw_text_low": "",
                })

                out = {
                    "stem": stem,
                    "img_path": img_path,
                    "up_state": rec.get("up_state"),
                    "low_state": rec.get("low_state"),
                    "pred_choice_up": pred["pred_choice_up"],
                    "pred_choice_low": pred["pred_choice_low"],
                    "raw_text_up": pred["raw_text_up"],
                    "raw_text_low": pred["raw_text_low"],
                }

                with open(os.path.join(out_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)

            pbar.update(len(batch))

    print(f"[Infer] Done -> {out_dir}")


# =========================
# 评估：选项对比
# =========================
def eval_with_gtset(pred_dir: Path, gt_map: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    评估模型预测的选项与 GT 的对比
    输出准确率和混淆矩阵
    """
    n_total = 0
    
    # 统计计数
    upper_eval = 0
    lower_eval = 0
    both_eval = 0
    
    upper_correct = 0
    lower_correct = 0
    both_correct = 0
    
    invalid_pred_file = 0
    
    rows: List[Dict[str, Any]] = []
    
    # 混淆矩阵：真实状态 -> 预测状态 的计数
    # conf_upper[gt_state][pred_state] = count
    conf_upper: Dict[str, Dict[str, int]] = {s: {p: 0 for p in STATES} for s in STATES}
    conf_lower: Dict[str, Dict[str, int]] = {s: {p: 0 for p in STATES} for s in STATES}
    
    # 预测为无效选项的计数
    invalid_choice_up = 0
    invalid_choice_low = 0

    for stem, rec in sorted(gt_map.items()):
        n_total += 1
        img_path = rec.get("img_path")
        up_state = rec.get("up_state")
        low_state = rec.get("low_state")

        row: Dict[str, Any] = {
            "stem": stem,
            "img_path": img_path,
            "up_state": up_state,
            "low_state": low_state,
        }

        pred = read_json(pred_dir / f"{stem}.json")
        if not isinstance(pred, dict):
            invalid_pred_file += 1
            row["err"] = "missing_pred_json"
            rows.append(row)
            continue

        pred_choice_up = pred.get("pred_choice_up")
        pred_choice_low = pred.get("pred_choice_low")
        
        # 转换选项为状态
        pred_state_up = OPTION_TO_STATE.get(pred_choice_up) if pred_choice_up else None
        pred_state_low = OPTION_TO_STATE.get(pred_choice_low) if pred_choice_low else None
        
        row.update({
            "pred_choice_up": pred_choice_up,
            "pred_choice_low": pred_choice_low,
            "pred_state_up": pred_state_up,
            "pred_state_low": pred_state_low,
        })

        # 评估上灯
        up_ok = False
        if pred_state_up is not None:
            upper_eval += 1
            up_ok = (pred_state_up == up_state)
            if up_ok:
                upper_correct += 1
            # 更新混淆矩阵
            conf_upper[up_state][pred_state_up] += 1
        else:
            invalid_choice_up += 1
            row["up_invalid"] = True

        # 评估下灯
        low_ok = False
        if pred_state_low is not None:
            lower_eval += 1
            low_ok = (pred_state_low == low_state)
            if low_ok:
                lower_correct += 1
            # 更新混淆矩阵
            conf_lower[low_state][pred_state_low] += 1
        else:
            invalid_choice_low += 1
            row["low_invalid"] = True

        # 双边评估（两个都有效时才计算）
        both_ok = False
        if pred_state_up is not None and pred_state_low is not None:
            both_eval += 1
            both_ok = up_ok and low_ok
            if both_ok:
                both_correct += 1

        row.update({
            "up_ok": up_ok,
            "low_ok": low_ok,
            "both_ok": both_ok,
        })

        rows.append(row)

    def safe_div(a, b):
        return round(a / b, 6) if b else 0.0

    # 计算每个状态的准确率
    def calc_per_state_acc(conf_mat: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        """计算每个真实状态的准确率"""
        per_state_acc = {}
        for gt_state in STATES:
            total = sum(conf_mat[gt_state].values())
            correct = conf_mat[gt_state].get(gt_state, 0)
            per_state_acc[gt_state] = safe_div(correct, total)
        return per_state_acc

    metrics = {
        "total_images": n_total,
        "invalid_pred_file": invalid_pred_file,
        "invalid_choice_up": invalid_choice_up,
        "invalid_choice_low": invalid_choice_low,
        "upper_acc": safe_div(upper_correct, upper_eval),
        "lower_acc": safe_div(lower_correct, lower_eval),
        "both_acc": safe_div(both_correct, both_eval),
        "upper_eval": upper_eval,
        "lower_eval": lower_eval,
        "both_eval": both_eval,
        "upper_correct": upper_correct,
        "lower_correct": lower_correct,
        "both_correct": both_correct,
        "per_state_acc_upper": calc_per_state_acc(conf_upper),
        "per_state_acc_lower": calc_per_state_acc(conf_lower),
        "confusion": {
            "upper": conf_upper,
            "lower": conf_lower,
        },
    }
    return metrics, rows


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser("Light choice eval (option A-E selection)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_infer = sub.add_parser("infer", help="infer only -> out_dir")
    p_infer.add_argument("--gt_set", type=str, nargs="+", required=True)
    p_infer.add_argument("--model_path", type=str, required=True)
    p_infer.add_argument("--out_dir", type=str, required=True)
    p_infer.add_argument("--max_new_tokens", type=int, default=16)
    p_infer.add_argument("--batch_size", type=int, default=32)
    p_infer.add_argument("--gpu_memory_utilization", type=float, default=0.5)

    p_eval = sub.add_parser("eval", help="eval only -> out_dir/metrics")
    p_eval.add_argument("--pred_dir", type=str, required=True)
    p_eval.add_argument("--gt_set", type=str, nargs="+", required=True)
    p_eval.add_argument("--out_dir", type=str, required=True)

    p_all = sub.add_parser("all", help="infer -> eval")
    p_all.add_argument("--model_path", type=str, required=True)
    p_all.add_argument("--gt_set", type=str, nargs="+", required=True)
    p_all.add_argument("--out_dir", type=str, required=True)
    p_all.add_argument("--max_new_tokens", type=int, default=16)
    p_all.add_argument("--batch_size", type=int, default=32)
    p_all.add_argument("--gpu_memory_utilization", type=float, default=0.5)

    args = parser.parse_args()

    if args.cmd == "infer":
        gt_map = load_gt_sets(args.gt_set)
        run_infer_from_gtset(
            model_path=args.model_path,
            gt_map=gt_map,
            out_dir=args.out_dir,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    elif args.cmd == "eval":
        gt_map = load_gt_sets(args.gt_set)
        metrics, rows = eval_with_gtset(Path(args.pred_dir), gt_map)
        out_root = Path(args.out_dir) / "metrics"
        save_json(metrics, out_root / "metrics.json")
        save_csv(rows, out_root / "per_image.csv")
        print("[EVAL] metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print(f"[EVAL] saved to: {out_root}")

    elif args.cmd == "all":
        out_root = Path(args.out_dir)
        preds_dir = out_root / "preds"
        metrics_dir = out_root / "metrics"
        preds_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        gt_map = load_gt_sets(args.gt_set)

        # 1) infer
        run_infer_from_gtset(
            model_path=args.model_path,
            gt_map=gt_map,
            out_dir=str(preds_dir),
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        # 2) eval
        metrics, rows = eval_with_gtset(preds_dir, gt_map)
        save_json(metrics, metrics_dir / "metrics.json")
        save_csv(rows, metrics_dir / "per_image.csv")
        print("[ALL] metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print(f"[ALL] metrics saved to: {metrics_dir}")


if __name__ == "__main__":
    main()

# /ogi-data/pzb-data/model/VLABench-main/weight/qwen3vl_1215
# /ogi-data/pzb-data/light_output/t260130-rank8/v0-20260130-190415/checkpoint-2900-merged: 未加thinker数据
# /ogi-data/pzb-data/light_output/t260201-rank8/v9-20260202-213924/checkpoint-1100-merged：加thinker数据

# CUDA_VISIBLE_DEVICES=6 python light_sh_yjp/easyprompt-outputpoint*2-en-thinker.py all \
# --model_path /ogi-data/pzb-data/light_output/t260130-rank8/v0-20260130-190415/checkpoint-2900-merged \
# --gt_set  ./light_data/bc_light/val_jsonl/green_green_val.jsonl \
# ./light_data/bc_light/val_jsonl/green_unknown_val.jsonl \
# ./light_data/bc_light/val_jsonl/green_yellow_val.jsonl \
# ./light_data/bc_light/val_jsonl/off_off_val.jsonl \
# ./light_data/bc_light/val_jsonl/off_unknown_val.jsonl \
# ./light_data/bc_light/val_jsonl/red_red_val.jsonl \
# ./light_data/bc_light/val_jsonl/red_unknown_val.jsonl \
# ./light_data/bc_light/val_jsonl/unknown_green_val.jsonl \
# ./light_data/bc_light/val_jsonl/unknown_off_val.jsonl \
# ./light_data/bc_light/val_jsonl/unknown_red_val.jsonl \
# ./light_data/bc_light/val_jsonl/unknown_unknown_val.jsonl \
# ./light_data/bc_light/val_jsonl/unknown_yellow_val.jsonl \
# ./light_data/bc_light/val_jsonl/yellow_green_val.jsonl \
# ./light_data/bc_light/val_jsonl/yellow_unknown_val.jsonl \
# ./light_data/bc_light/val_jsonl/yellow_yellow_val.jsonl \
# --out_dir    light_results_yjp/easyprompt2-outputpoint*2-en-thinker-noupstream \
# --batch_size 32 \
# --gpu_memory_utilization 0.3