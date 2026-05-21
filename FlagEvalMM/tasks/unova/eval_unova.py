#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抓取规划器：推理 + 指标 + 绘图 一体化脚本（GT JSONL 优先驱动）- 批量推理版本
------------------------------------------------
简化版：只关注 select 状态的两个指标
1. Select定位准确率（IoU>=阈值）
2. Select+Stance准确率（定位准确且stance正确）
"""
import os, re, json, glob, math, argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------- 基础工具 ----------------------
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


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def normalize_state(s: Any) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    if s in {"select", "approach"}:
        return s
    return None


def normalize_stance(s: Any) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    if s in {"left", "middle", "right"}:
        return s
    return None


def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


# ---------------------- GT JSONL ----------------------
def gt_image_path_from_dialog(gt_obj: Dict[str, Any]) -> Optional[str]:
    images = gt_obj.get("images")
    if isinstance(images, list) and images and isinstance(images[0], str):
        return images[0]
    return None


def parse_gt_schema_or_dialog(gt_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(gt_obj, dict):
        return None
    if normalize_state(gt_obj.get("state")) is not None:
        return gt_obj
    msgs = gt_obj.get("messages")
    if isinstance(msgs, list) and msgs:
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "assistant":
                content = m.get("content")
                if isinstance(content, str):
                    js = extract_json(content)
                    if js:
                        try:
                            obj = json.loads(js)
                            if normalize_state(obj.get("state")) is not None:
                                return obj
                        except Exception:
                            pass
                if isinstance(content, list):
                    joined = " ".join([str(x) for x in content])
                    js = extract_json(joined)
                    if js:
                        try:
                            obj = json.loads(js)
                            if normalize_state(obj.get("state")) is not None:
                                return obj
                        except Exception:
                            pass
    return None


def load_gt_sets(jsonl_list: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in jsonl_list:
        p = Path(path)
        if not p.exists():
            print(f"[WARN] GT JSONL 不存在: {p}")
            continue
        with p.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(f"[WARN] JSONL 解析失败 {p}:{ln}: {e}")
                    continue
                img_path = gt_image_path_from_dialog(obj) or obj.get("image") or obj.get("img")
                if not isinstance(img_path, str):
                    print(f"[WARN] 缺少 images 路径 {p}:{ln}")
                    continue
                stem = Path(img_path).stem
                parsed = parse_gt_schema_or_dialog(obj)
                if stem in out:
                    print(f"[WARN] stem 冲突，保持首次记录: {stem} ({p}:{ln})")
                    continue
                out[stem] = {"gt": parsed, "img_path": img_path, "raw": obj}
    if not out:
        print("[WARN] 未从任何 GT JSONL 读到样本")
    else:
        print(f"[GT] 已加载样本数(去重后按 stem): {len(out)}")
    return out


# ---------------------- 推理 ----------------------
SYSTEM_PROMPT = (
    "你是抓取规划器。任务：从「托盘（栈板）上的箱子堆垛」中选择并规划抓取；"
    "若距离堆垛过远先给出靠近点，若已接近则给出抓取目标与站位。 "
    "对象与范围：箱子：塑料/周转/物流收纳箱等（颜色不限）。仅考虑「位于托盘上的堆垛」的箱子；"
    "忽略地面/桌面/货架等非托盘上的箱子。 "
    "选箱规则：1) 最上层优先。2) 若同层，选最右（以 bbox 中心 x 最大为准）。 "
    "站位规则：目标左侧有同层邻箱 → \"right\"；目标右侧有同层邻箱 → \"left\"；两侧都无 → \"middle\"。 "
    "远/近与靠近点：无法可靠给出目标框时视为「远」，返回堆垛正面约 1m 的落脚点投影（图像像素点）。 "
    "输出格式（只输出其一）：远：{\"state\":\"approach\",\"approach_point_2d\":[x,y]} "
    "近：{\"state\":\"select\",\"target_bbox_2d\":[x1,y1,x2,y2],\"stance_flag\":\"left|middle|right\"} "
    "约束：坐标为像素整型，bbox 为左上-右下且在图像范围内。只返回一个 JSON，不要任何其他文字。"
)
USER_TEXT = "输出当前的最优拆垛信息"


def _load_model_and_processor(model_path: str):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import torch
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    pad_id = processor.tokenizer.pad_token_id
    eos_id = processor.tokenizer.eos_token_id
    if pad_id is None and eos_id is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.eval()
    return processor, model


def _infer_batch_images(processor, model, image_paths: List[str], max_new_tokens: int = 128) -> List[Dict[str, Any]]:
    """批量推理多张图片"""
    from PIL import Image
    import torch

    if not image_paths:
        return []

    images = []
    valid_indices = []
    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            valid_indices.append(i)
        except Exception as e:
            print(f"[WARN] 图片加载失败 {img_path}: {e}")

    if not images:
        return [{"_error": "failed to load image"}] * len(image_paths)

    texts = []
    for img in images:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": USER_TEXT},
            ]},
        ]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(prompt_text)

    try:
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, temperature=0.0, top_p=1.0, repetition_penalty=1.0,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        new_tokens = gen_ids[:, inputs["input_ids"].shape[1]:]
        text_outs = processor.batch_decode(new_tokens, skip_special_tokens=True)

        results = []
        for text_out in text_outs:
            text_out = text_out.strip()
            js = extract_json(text_out)
            results.append({"_raw": text_out} if js is None else json.loads(js))
    except Exception as e:
        print(f"[ERROR] 批量推理失败: {e}")
        results = [{"_error": str(e)}] * len(images)

    final_results = []
    valid_idx = 0
    for i in range(len(image_paths)):
        if i in valid_indices:
            final_results.append(results[valid_idx])
            valid_idx += 1
        else:
            final_results.append({"_error": "failed to load image"})

    return final_results


def run_infer_from_imagedir(model_path: str, image_dir: str, out_dir: str, max_new_tokens: int = 128,
                            batch_size: int = 1):
    from tqdm import tqdm
    processor, model = _load_model_and_processor(model_path)
    os.makedirs(out_dir, exist_ok=True)
    img_paths = sorted(
        p for p in glob.glob(os.path.join(image_dir, "**", "*"), recursive=True)
        if os.path.splitext(p)[1].lower() in ALLOWED_EXTS
    )

    with tqdm(total=len(img_paths), desc="Infer[image_dir]") as pbar:
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i:i + batch_size]
            batch_stems = [Path(p).stem for p in batch_paths]

            results = _infer_batch_images(processor, model, batch_paths, max_new_tokens=max_new_tokens)

            for stem, result in zip(batch_stems, results):
                with open(os.path.join(out_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

            pbar.update(len(batch_paths))

    print(f"[Infer] Done -> {out_dir}")


def run_infer_from_gtset(model_path: str, gt_map: Dict[str, Dict[str, Any]], out_dir: str, max_new_tokens: int = 128,
                         batch_size: int = 1):
    from tqdm import tqdm
    processor, model = _load_model_and_processor(model_path)
    os.makedirs(out_dir, exist_ok=True)

    samples = []
    for stem, rec in sorted(gt_map.items()):
        img_path = rec.get("img_path")
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            print(f"[WARN] 跳过（找不到图像）: {img_path}")
            continue
        samples.append((stem, img_path))

    with tqdm(total=len(samples), desc="Infer[gt_set]") as pbar:
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]
            batch_stems = [s[0] for s in batch_samples]
            batch_paths = [s[1] for s in batch_samples]

            results = _infer_batch_images(processor, model, batch_paths, max_new_tokens=max_new_tokens)

            for stem, result in zip(batch_stems, results):
                with open(os.path.join(out_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

            pbar.update(len(batch_samples))

    print(f"[Infer] Done -> {out_dir}")


# ---------------------- 评估（简化版：只看select） ----------------------
def eval_with_gtset(pred_dir: Path, gt_map: Dict[str, Dict[str, Any]], iou_thr: float = 0.5):
    """
    简化评估：只关注 GT 为 select 的样本
    指标：
    1. Select定位准确率 = IoU达标数 / GT为select总数
    2. Select+Stance准确率 = (IoU达标且stance正确数) / GT为select总数
    """
    n_total = 0
    n_gt_select = 0
    invalid_gt = 0
    invalid_pred = 0

    # Select相关统计
    select_localization_correct = 0  # IoU达标
    select_with_stance_correct = 0  # IoU达标且stance正确

    iou_sum = 0.0
    iou_count = 0
    per_image_rows: List[Dict[str, Any]] = []

    for stem, rec in sorted(gt_map.items()):
        n_total += 1
        gt = rec.get("gt")
        img_path = rec.get("img_path")
        pred = read_json(pred_dir / f"{stem}.json")

        gt_state = normalize_state(gt.get("state") if isinstance(gt, dict) else None) if gt else None
        pred_state = normalize_state(pred.get("state") if isinstance(pred, dict) else None) if pred else None

        row: Dict[str, Any] = {
            "stem": stem,
            "img_path": img_path,
            "gt_state": gt_state,
            "pred_state": pred_state
        }

        # 只关注GT为select的样本
        if gt_state != "select":
            per_image_rows.append(row)
            continue

        n_gt_select += 1

        if gt_state is None:
            invalid_gt += 1
            row["err"] = "invalid_gt"
            per_image_rows.append(row)
            continue

        if pred is None or pred_state is None:
            invalid_pred += 1
            row["err"] = "invalid_pred"
            per_image_rows.append(row)
            continue

        # GT是select，检查预测
        if pred_state == "select":
            gt_box = gt.get("target_bbox_2d")
            pr_box = pred.get("target_bbox_2d")

            if isinstance(gt_box, list) and len(gt_box) == 4 and isinstance(pr_box, list) and len(pr_box) == 4:
                gx1, gy1, gx2, gy2 = map(int, gt_box)
                px1, py1, px2, py2 = map(int, pr_box)
                iou_val = iou_xyxy((gx1, gy1, gx2, gy2), (px1, py1, px2, py2))
                iou_sum += iou_val
                iou_count += 1

                row["iou"] = round(iou_val, 4)
                row["localization_correct"] = bool(iou_val >= iou_thr)

                if iou_val >= iou_thr:
                    # Select定位正确
                    select_localization_correct += 1

                    # 检查stance
                    gt_stance = normalize_stance(gt.get("stance_flag"))
                    pr_stance = normalize_stance(pred.get("stance_flag"))
                    row["gt_stance"] = gt_stance
                    row["pred_stance"] = pr_stance

                    if gt_stance is not None and pr_stance is not None:
                        stance_match = (gt_stance == pr_stance)
                        row["stance_correct"] = stance_match
                        if stance_match:
                            select_with_stance_correct += 1
                    else:
                        row["err"] = "missing_stance"
            else:
                row["err"] = "missing_bbox"
        else:
            # 预测为approach，定位失败
            row["localization_correct"] = False

        per_image_rows.append(row)

    # 计算最终指标
    select_localization_accuracy = (select_localization_correct / n_gt_select) if n_gt_select > 0 else 0.0
    select_with_stance_accuracy = (select_with_stance_correct / n_gt_select) if n_gt_select > 0 else 0.0
    mean_iou = (iou_sum / iou_count) if iou_count > 0 else 0.0

    metrics = {
        "总样本数": n_total,
        "GT为select的样本数": n_gt_select,
        "无效GT数": invalid_gt,
        "无效预测数": invalid_pred,

        "Select定位准确率(IoU>=%.2f)" % iou_thr: round(select_localization_accuracy, 4),
        "Select+Stance准确率": round(select_with_stance_accuracy, 4),

        "平均IoU(仅GT=select且Pred=select)": round(mean_iou, 4),
        "IoU统计样本数": iou_count,

        "配置": {
            "iou_threshold": iou_thr,
        }
    }

    return metrics, per_image_rows


def save_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    import csv
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------- 可视化 ----------------------
def try_import_cv2():
    try:
        import cv2
        return cv2
    except Exception:
        return None


def draw_box_cv2(img, x1, y1, x2, y2, color, label: Optional[str] = None):
    cv2 = try_import_cv2()
    if cv2 is None:
        return img
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    if label:
        cv2.putText(img, label, (int(x1), max(0, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


def load_image_cv2(path: str):
    cv2 = try_import_cv2()
    if cv2 is None:
        return None
    return cv2.imread(path, cv2.IMREAD_COLOR)


def save_image_cv2(path: str, img):
    cv2 = try_import_cv2()
    if cv2 is None:
        return False
    return cv2.imwrite(path, img)


def visualize_annotations_from_gtset(pred_dir: str, gt_map: Dict[str, Dict[str, Any]], save_dir: str,
                                     iou_thr: float = 0.5):
    cv2 = try_import_cv2()
    if cv2 is None:
        print("[WARN] 未安装 OpenCV，跳过可视化。")
        return
    pred_p = Path(pred_dir)
    save_p = Path(save_dir)
    save_p.mkdir(parents=True, exist_ok=True)

    for stem, rec in sorted(gt_map.items()):
        img_path = rec.get("img_path")
        gt = rec.get("gt")
        pred = read_json(pred_p / f"{stem}.json")

        if not isinstance(img_path, str) or not os.path.exists(img_path):
            print(f"[WARN] 找不到图像: {img_path} (stem={stem})")
            continue
        img = load_image_cv2(img_path)
        if img is None:
            print(f"[WARN] 图像读取失败: {img_path}")
            continue

        # GT (只画select)
        if isinstance(gt, dict):
            gst = normalize_state(gt.get("state"))
            if gst == "select":
                bx = gt.get("target_bbox_2d")
                if isinstance(bx, list) and len(bx) == 4:
                    draw_box_cv2(img, bx[0], bx[1], bx[2], bx[3], (0, 255, 0),
                                 f"GT {gt.get('stance_flag', '')}")

        # Pred (只画select)
        if isinstance(pred, dict):
            pst = normalize_state(pred.get("state"))
            if pst == "select":
                bx = pred.get("target_bbox_2d")
                if isinstance(bx, list) and len(bx) == 4:
                    label = f"Pred {pred.get('stance_flag', '')}"
                    if isinstance(gt, dict) and normalize_state(gt.get("state")) == "select":
                        gb = gt.get("target_bbox_2d")
                        if isinstance(gb, list) and len(gb) == 4:
                            iouv = iou_xyxy((int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])),
                                            (int(gb[0]), int(gb[1]), int(gb[2]), int(gb[3])))
                            label += f" (IoU={iouv:.2f}{'✓' if iouv >= iou_thr else '✗'})"
                    draw_box_cv2(img, bx[0], bx[1], bx[2], bx[3], (0, 0, 255), label)

        out_path = save_p / f"{stem}_vis.png"
        save_image_cv2(str(out_path), img)
        print(f"[VIS] 保存: {out_path}")


def visualize_annotations_from_imgdir(img_dir: str, pred_dir: str, save_dir: str):
    cv2 = try_import_cv2()
    if cv2 is None:
        print("[WARN] 未安装 OpenCV，跳过可视化。")
        return
    img_dir_p = Path(img_dir)
    pred_dir_p = Path(pred_dir)
    save_p = Path(save_dir)
    save_p.mkdir(parents=True, exist_ok=True)

    for pf in sorted(pred_dir_p.glob("*.json")):
        stem = pf.stem
        pred = read_json(pf)

        img_path = None
        for ext in ALLOWED_EXTS:
            cand = img_dir_p / f"{stem}{ext}"
            if cand.exists():
                img_path = str(cand)
                break
        if img_path is None:
            print(f"[WARN] 找不到图像: stem={stem}")
            continue
        img = load_image_cv2(img_path)
        if img is None:
            print(f"[WARN] 图像读取失败: {img_path}")
            continue

        if isinstance(pred, dict):
            pst = normalize_state(pred.get("state"))
            if pst == "select":
                bx = pred.get("target_bbox_2d")
                if isinstance(bx, list) and len(bx) == 4:
                    draw_box_cv2(img, bx[0], bx[1], bx[2], bx[3], (0, 0, 255),
                                 f"Pred {pred.get('stance_flag', '')}")

        out_path = save_p / f"{stem}_vis.png"
        save_image_cv2(str(out_path), img)
        print(f"[VIS] 保存: {out_path}")


# ---------------------- CLI ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="抓取规划器：推理+评估+可视化 一体化脚本（简化版：只关注select状态）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # infer
    p_infer = sub.add_parser("infer", help="仅推理：从 --image_dir 或 --gt_set 输入，输出到 --out_dir")
    src = p_infer.add_mutually_exclusive_group(required=True)
    src.add_argument("--image_dir", type=str, help="图片目录")
    src.add_argument("--gt_set", type=str, nargs="+", help="一个或多个 GT JSONL（从中读取图片路径）")
    p_infer.add_argument("--model_path", type=str, required=True)
    p_infer.add_argument("--out_dir", type=str, required=True, help="预测 JSON 输出目录")
    p_infer.add_argument("--max_new_tokens", type=int, default=128)
    p_infer.add_argument("--batch_size", type=int, default=1, help="批量推理的批大小（默认1）")

    # eval
    p_eval = sub.add_parser("eval", help="仅评估：需要 --gt_set；指标写到 --out_dir/metrics")
    p_eval.add_argument("--pred_dir", type=str, required=True, help="预测 JSON 目录")
    p_eval.add_argument("--gt_set", type=str, nargs="+", required=True, help="一个或多个 GT JSONL")
    p_eval.add_argument("--out_dir", type=str, required=True, help="基准输出目录（将创建 metrics 子目录）")
    p_eval.add_argument("--iou_thr", type=float, default=0.5)

    # vis
    p_vis = sub.add_parser("vis", help="仅可视化：有 GT 用 --gt_set（优先），否则用 --img_dir；图写到 --out_dir/vis")
    p_vis.add_argument("--pred_dir", type=str, required=True, help="预测 JSON 目录")
    p_vis.add_argument("--out_dir", type=str, required=True, help="基准输出目录（将创建 vis 子目录）")
    src2 = p_vis.add_mutually_exclusive_group(required=False)
    src2.add_argument("--gt_set", type=str, nargs="+", help="一个或多个 GT JSONL（从中读取图片路径）")
    src2.add_argument("--img_dir", type=str, help="图片目录（当未提供 gt_set 时使用）")
    p_vis.add_argument("--iou_thr", type=float, default=0.5)

    # all
    p_all = sub.add_parser("all", help="全流程：推理(从GT)->评估->可视化。全部输出放在 --out_dir 下的子目录。")
    p_all.add_argument("--model_path", type=str, required=True)
    p_all.add_argument("--gt_set", type=str, nargs="+", required=True, help="一个或多个 GT JSONL（用于推理、评估、可视化）")
    p_all.add_argument("--out_dir", type=str, required=True, help="运行输出根目录（将创建 preds/metrics/vis）")
    p_all.add_argument("--max_new_tokens", type=int, default=128)
    p_all.add_argument("--iou_thr", type=float, default=0.5)
    p_all.add_argument("--batch_size", type=int, default=1, help="批量推理的批大小（默认1）")

    args = parser.parse_args()

    if args.cmd == "infer":
        if args.gt_set:
            gt_map = load_gt_sets(args.gt_set)
            if not gt_map:
                raise SystemExit("未能从 --gt_set 读取到任何样本")
            run_infer_from_gtset(args.model_path, gt_map, args.out_dir, args.max_new_tokens, args.batch_size)
        else:
            run_infer_from_imagedir(args.model_path, args.image_dir, args.out_dir, args.max_new_tokens, args.batch_size)

    elif args.cmd == "eval":
        gt_map = load_gt_sets(args.gt_set)
        if not gt_map:
            raise SystemExit("未能从 --gt_set 读取到任何样本")
        metrics, per_image_rows = eval_with_gtset(pred_dir=Path(args.pred_dir), gt_map=gt_map, iou_thr=args.iou_thr)
        mdir = Path(args.out_dir) / "metrics"
        mdir.mkdir(parents=True, exist_ok=True)
        save_json(metrics, mdir / "metrics.json")
        save_csv(per_image_rows, mdir / "per_image.csv")
        print(f"[OK] 指标已保存到: {mdir}")
        print("\n=== 评估结果 ===")
        print(f"Select定位准确率: {metrics['Select定位准确率(IoU>=%.2f)' % args.iou_thr]:.2%}")
        print(f"Select+Stance准确率: {metrics['Select+Stance准确率']:.2%}")

    elif args.cmd == "vis":
        if args.gt_set:
            gt_map = load_gt_sets(args.gt_set)
            if not gt_map:
                raise SystemExit("未能从 --gt_set 读取到任何样本")
            vdir = Path(args.out_dir) / "vis"
            vdir.mkdir(parents=True, exist_ok=True)
            visualize_annotations_from_gtset(args.pred_dir, gt_map, str(vdir), iou_thr=args.iou_thr)
        else:
            if not args.img_dir:
                raise SystemExit("未提供 --gt_set 时，vis 需要 --img_dir")
            vdir = Path(args.out_dir) / "vis"
            vdir.mkdir(parents=True, exist_ok=True)
            visualize_annotations_from_imgdir(args.img_dir, args.pred_dir, str(vdir))

    elif args.cmd == "all":
        out_root = Path(args.out_dir)
        preds_dir = out_root / "preds"
        metrics_dir = out_root / "metrics"
        vis_dir = out_root / "vis"
        preds_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

        # 1) 推理
        gt_map = load_gt_sets(args.gt_set)
        if not gt_map:
            raise SystemExit("未能从 --gt_set 读取到任何样本")
        run_infer_from_gtset(args.model_path, gt_map, str(preds_dir), args.max_new_tokens, args.batch_size)

        # 2) 评估
        metrics, per_image_rows = eval_with_gtset(pred_dir=preds_dir, gt_map=gt_map, iou_thr=args.iou_thr)
        save_json(metrics, metrics_dir / "metrics.json")
        save_csv(per_image_rows, metrics_dir / "per_image.csv")
        print(f"[ALL] 指标已保存到: {metrics_dir}")
        print("\n=== 评估结果 ===")
        print(f"Select定位准确率: {metrics['Select定位准确率(IoU>=%.2f)' % args.iou_thr]:.2%}")
        print(f"Select+Stance准确率: {metrics['Select+Stance准确率']:.2%}")

        # 3) 可视化
        visualize_annotations_from_gtset(str(preds_dir), gt_map, str(vis_dir), iou_thr=args.iou_thr)
        print(f"[ALL] 可视化已保存到: {vis_dir}")


if __name__ == "__main__":
    main()


# python /ogi-code/scrpit/eval_only_select_qwen2_5.py all \
#   --model_path /ogi-code/result/fwz_111502/v0-20251115-084653/checkpoint-300-merged \
#   --gt_set  /ogi-code/data/labels/only_select_rightfirst/val/stereo_0915_val.jsonl \
#   /ogi-code/data/labels/only_select_rightfirst/val/stereo_0925_unova0108_val.jsonl \
#   /ogi-code/data/labels/only_select_rightfirst/val/stereo_0925_unova0175_val.jsonl \
#   /ogi-code/data/labels/only_select_rightfirst/val/stereo_0926_unova0218_val.jsonl \
#   --out_dir    /ogi-code/result/fwz_111502/v0-20251115-084653/checkpoint-300-merged/run_onlyselct/ \
#   --iou_thr    0.5 \
#   --batch_size 256