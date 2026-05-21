#!/usr/bin/env python3
"""
Babysit a LoRA training folder and automatically:
1) Run `swift export` to merge the LoRA into full weights.
2) Run `flagevalmm` to evaluate the merged model (now supports MULTIPLE tasks).
3) Parse evaluation results JSON, append to per-task history JSON, and draw per-task global trend charts.

Global artifacts (auto-updated on each successful eval) will be stored under:
./results/<model_name>/<task_name>/
- STATE.json  (shared across tasks, stored at ./results/<model_name>/STATE.json)
- _summary.json (per task)
- _summary.png  (per task)
"""

import argparse
import json
import os
import sys
import re
import numpy as np
import time
import hashlib
import shutil
import subprocess
import math
from pathlib import Path
from typing import Dict, Set, Optional, List

# Headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

_num_pat = re.compile(r"^(iter_|checkpoint-)(\d+)$")

# -------------------- task_name -> {sota_model_name -> score} --------------------
SOTA_BY_TASK = {
    # "taskA": {"GPT-4o": 0.812, "Claude-3.5": 0.825},
    # "taskB": {"Qwen2.5-72B": 0.901},
    "share_robot_affordance": {},
    "refspatial_bench": {"Qwen3-VL-32B-Instruct": 58.66},
    "where2place": {"RoboBrain-2.0-32B": 74.25, "RoboBrain2.0-7B": 64.51},
    "robovqa": {"MiMo-Embodied-7B": 61.99},
    "egoplan2": {"Qwen3-VL-32B-Instruct": 58.66},
    "blink_val": {"Robix-7B": 87.6, "Qwen3-VL-4B": 85.02},
    "cv_bench_test": {"MiMo-Embodied-7B": 88.11, "Robix-7B": 86.5},
    "embspatial_bench": {"Qwen3-VL-32B-Instruct": 81.02, "Qwen3-VL-4B": 79.6},
    "robo_spatial_home_all": {"Qwen3-VL-8B-Instruct": 66.9, "Qwen3-VL-32B-Instruct": 65.98},
    "sat": {"RoboBrain-2.0-32B": 86.67, "Cosmos-Reason1-7B": 82.67},
    "vsi_bench_tiny": {"Qwen3-VL-32B-Instruct": 66.4, "Qwen3-VL-8B-Instruct": 63.34, "Pelican-72B": 60.99},
    "vsi_bench_test": {"Qwen3-VL-8B-Instruct": 63.39},
    "erqa": {"Qwen3-VL-32B-Instruct": 48.0, "MiMo-Embodied-7B": 46.75, "Pelican-72B": 44.25},
    "moving_box_val": {},
    "refcoco_val": {},
    "refcoco_g_val": {},
    "refcoco_plus_val": {},
    "seg_refcoco": {},
    "seg_refcocog": {},
    "seg_refcocop": {},
    "vsr_random": {"Pelican-72B": 86.83, "Qwen3-VL-32B-Instruct": 86.19, "RoboBrain-2.0-32B": 84.0, "RoboBrain2.0-7B": 84.0},
    "pixmo_points_eval": {"Robix-32B": 47.3, "Qwen3-VL-32B-Instruct": 47.17, "RoboBrain-2.0-32B": 46.88},
}

# -------------------- Global Artifact Paths --------------------

def get_global_dirs(model_name: str, task_name: str):
    base = Path("./results") / model_name / task_name
    print(f"[INFO] Global artifacts base dir (task={task_name}): {base}")
    return {
        "base": base,
        "state": (Path("./results") / model_name / "STATE.json"),  # shared across tasks
        "history": base / "_summary.json",
        "png": base / "_summary.png",
        "baselines": base / "_baselines.json",  # <-- per-task manual baselines (SOTA, iter0000000)
    }

# -------------------- State Management --------------------

def load_state(state_path: Path) -> Dict[str, List[str]]:
    if state_path.exists():
        try:
            obj = json.loads(state_path.read_text())
            if isinstance(obj, dict) and "processed" in obj:
                return obj
        except Exception:
            return {"processed": []}
    return {"processed": []}


def save_state(state_path: Path, state: Dict[str, Set[str]]):
    state["processed"] = sorted(list(set(state.get("processed", []))))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def list_immediate_subdirs(p: Path) -> Set[str]:
    filters = set()
    if not p.exists():
        return filters
    for d in p.iterdir():
        if d.is_dir() and (d.name.startswith("checkpoint-") or d.name.startswith("iter_")):
            if d.name.endswith("merged"):
                filters.add(d.name)
    return filters

def ckpt_key(name: str) -> int:
    m = re.search(r"(?:^checkpoint-|^iter_)(\d+)", name)
    if m:
        return int(m.group(1))
    return -1

def dir_is_stable(path: Path, window_sec: int = 30) -> bool:
    latest_mtime = 0.0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                mtime = (Path(root) / f).stat().st_mtime
                latest_mtime = max(latest_mtime, mtime)
            except FileNotFoundError:
                return False
    return (time.time() - latest_mtime) >= window_sec

# -------------------- Run Helpers --------------------

def run_and_wait(
    cmd: str,
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
    timeout: Optional[int] = None,   # 秒；None 表示不设超时
    shell: bool = True,
) -> int:
    """
    同步执行命令：实时将 stdout 写到控制台和日志文件；
    等到命令退出后返回其退出码。支持可选超时。
    """
    print(f"\n[CMD] {cmd}\n", flush=True)
    log_f = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_f = log_path.open("a", encoding="utf-8")
            log_f.write(f"===== RUN: {cmd}\n")
            log_f.flush()

        proc = subprocess.Popen(
            cmd,
            env=env or os.environ.copy(),
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None

        # 实时读取并输出
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_f:
                log_f.write(line)

        try:
            # 显式阻塞等待（可超时）
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # 超时处理：杀掉进程并返回 124（常见的超时代码）
            proc.kill()
            if log_f:
                log_f.write("===== TIMEOUT: process killed\n")
            print("[ERROR] Command timed out and was killed.")
            return 124

        rc = proc.returncode or 0
        if log_f:
            log_f.write(f"===== EXIT CODE: {rc}\n")
        return rc
    finally:
        if log_f:
            log_f.close()


# -------------------- Results Parsing & History --------------------

def case_insensitive_path(base: Path, task_name: str) -> Optional[Path]:
    if not base.exists():
        return None
    for child in base.iterdir():
        if child.name.lower() == task_name.lower():
            return child
    return None


def find_result_json(results_dir: Path) -> Optional[Path]:
    if results_dir is None or (not results_dir.exists()):
        return None
    candidates = sorted(results_dir.glob("*.json"))
    for c in candidates:
        if c.name.endswith("result.json"):
            return c
    # Fallback: first JSON if *_result.json* not found    return None


def extract_metrics(data: dict, task_name: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    if task_name == "share_robot_affordance":
        v = data.get("average_iou")
        metrics["average_iou"] = float(v) if isinstance(v, (int, float)) else float("nan")

    elif task_name == "refspatial_bench":
        def acc(sec: str) -> float:
            v = data.get(sec, {})
            return float(v.get("accuracy")) if isinstance(v, dict) and "accuracy" in v else float("nan")
        metrics.update({
            "avg": acc("avg"),
            "location": acc("location"),
            "placement": acc("placement"),
            "unseen": acc("unseen"),
        })

    elif task_name == "where2place":
        def acc(sec: str) -> float:
            v = data.get(sec, {})
            return float(v.get("accuracy")) if isinstance(v, dict) and "accuracy" in v else float("nan")
        metrics.update({
            "avg": acc("avg"),
            "seen": acc("seen"),
            "unseen": acc("unseen"),
        })

    elif task_name == "robovqa":
        over = data.get('overall', {})
        for k in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "BLEU-avg"]:
            v = over.get(k, None)
            if isinstance(v, (int, float)):
                metrics[k] = float(v)

    elif task_name == "egoplan2":
        v = data.get('overall', {}).get('accuracy', float('nan'))
        metrics["accuracy"] = float(v) if isinstance(v, (int, float)) else float("nan")

    elif task_name == "erqa":
        v = data.get('accuracy', float('nan'))
        metrics["accuracy"] = float(v) if isinstance(v, (int, float)) else float("nan")

    elif task_name == "moving_box_val":
        def acc(sec: str) -> float:
            v = data.get(sec, {})
            return float(v.get("accuracy")) if isinstance(v, dict) and "accuracy" in v else float("nan")
        metrics.update({
            "Select_Loc_ACC@0.5": acc("Select_Loc_ACC@0.5"),
            "Select_Side_ACC": acc("Select_Side_ACC"),
            "Select_Stance_ACC": acc("Select_Stance_ACC"),
            "Select_All_ACC": acc("Select_All_ACC"),
            "Mean_IoU": data.get("Mean_IoU", float("nan")),
            "JSON_Parse_Success_Rate": data.get("JSON_Parse_Success_Rate", float("nan")),
            "accuracy": data.get("accuracy", float("nan")),
        })

    elif task_name in ["refcoco_val", "refcoco_g_val", "refcoco_plus_val", "refcoco_testA", "refcoco_g_test", "refcoco_plus_testA", "refcoco_testB", "refcoco_plus_testB", "refcoco_val_dq1", "refcoco_g_val_dq1", "refcoco_plus_val_dq1", "refcoco_testA_dq1", "refcoco_g_test_dq1", "refcoco_plus_testA_dq1", "refcoco_testB_dq1", "refcoco_plus_testB_dq1"]:
        v = data.get('accuracy', float('nan'))
        metrics["accuracy"] = float(v) if isinstance(v, (int, float)) else float("nan")

    elif task_name == "vsr_random":
        v = data.get('accuracy', float('nan'))
        metrics["accuracy"] = float(v) if isinstance(v, (int, float)) else float("nan")
    
    elif task_name == "pixmo_points_eval":
        v = data.get('accuracy', float('nan'))
        metrics["accuracy"] = float(v) if isinstance(v, (int, float)) else float("nan")

    elif task_name in ["seg_refcoco", "seg_refcocog", "seg_refcocop"]:
        over = data.get('avg_rewards', {})
        for k in ["image/IoU", "image/gIoU", "image/cIoU"]:
            v = over.get(k, None)
            if isinstance(v, (int, float)):
                metrics[k] = float(v)

    else:
        # ---- 通用兜底：尽量从常见结构里捞数值 ----
        over = data.get("overall")
        if isinstance(over, dict):
            for k, v in over.items():
                if isinstance(v, (int, float)):
                    metrics[str(k)] = float(v)
        if not metrics:
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    metrics[str(k)] = float(v)
                elif isinstance(v, dict) and "accuracy" in v and isinstance(v["accuracy"], (int, float)):
                    metrics[str(k)] = float(v["accuracy"])
    return metrics

def load_history(path: Path) -> Dict[str, Dict[str, float]]:
    if path.exists():
        try:
            obj = json.loads(path.read_text())
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def save_history(path: Path, data: Dict[str, Dict[str, float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def update_history(path: Path, name: str, metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    hist = load_history(path)
    hist[name] = metrics
    save_history(path, hist)
    return hist

# -------------------- Plotting --------------------

def _coerce_metrics_dict(metrics: dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(metrics, dict):
        return out
    for k, v in metrics.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            out[str(k)] = float("nan")
    return out


def load_baselines(path: Path) -> Dict[str, Dict[str, float]]:
    """读取“扁平结构”的基线文件：{"SOTA": {"accuracy": 0.85}, "iter0000000": {...}}"""
    data: Dict[str, Dict[str, float]] = {}
    try:
        if path.exists():
            obj = json.loads(path.read_text())
            if isinstance(obj, dict):
                # 如果是全局结构（包含 task 名），这里先不处理，交给 load_task_baselines
                looks_task_map = any(isinstance(v, dict) and any(isinstance(x, dict) for x in v.values()) for v in obj.values())
                if not looks_task_map:
                    for run_name, metrics in obj.items():
                        data[str(run_name)] = _coerce_metrics_dict(metrics)
    except Exception as e:
        print(f"[BASELINES] Failed to load {path}: {e}")
    return data


def load_task_baselines(global_path: Path, per_task_path: Path, task_name: str) -> Dict[str, Dict[str, float]]:
    """优先级：per-task 覆盖 global。同一键（SOTA/iter0000000）下的指标按 per-task 覆盖。
    支持两种全局 JSON 结构：
    1) 按任务划分：
       {
         "blink_val": {"SOTA": {"accuracy": 0.8}, "iter0000000": {"accuracy": 0.1}},
         "robovqa":   {"SOTA": {"BLEU-avg": 0.35}}
       }
       可选 `_default` 节点作为兜底：
       {"_default": {"SOTA": {"accuracy": 0.75}, "iter0000000": {"accuracy": 0.05}}}
    2) 扁平通用：适用于所有任务：
       {"SOTA": {"accuracy": 0.75}, "iter0000000": {"accuracy": 0.05}}
    """
    result: Dict[str, Dict[str, float]] = {}

    # 先读取扁平 per-task 文件（如果存在）
    result.update(load_baselines(per_task_path))

    # 读取全局文件
    try:
        if global_path and global_path.exists():
            obj = json.loads(global_path.read_text())
            if isinstance(obj, dict):
                # 结构 1：任务映射
                if task_name in obj and isinstance(obj[task_name], dict):
                    for run_name, metrics in obj[task_name].items():
                        rn = str(run_name)
                        merged = _coerce_metrics_dict(metrics)
                        # 若 per-task 已定义则保持 per-task 值
                        if rn not in result:
                            result[rn] = merged
                        else:
                            # 合并字段，per-task 优先
                            for k, v in merged.items():
                                result[rn].setdefault(k, v)
                # `_default` 兜底
                if "_default" in obj and isinstance(obj["_default"], dict):
                    for run_name, metrics in obj["_default"].items():
                        rn = str(run_name)
                        merged = _coerce_metrics_dict(metrics)
                        if rn not in result:
                            result[rn] = merged
                        else:
                            for k, v in merged.items():
                                result[rn].setdefault(k, v)
                # 结构 2：全局扁平（对所有任务生效）
                if not result:
                    # 如果当前还没装到任何 baseline，再尝试把整个文件按扁平结构解析
                    flat = load_baselines(global_path)
                    for k, v in flat.items():
                        result.setdefault(k, v)
    except Exception as e:
        print(f"[BASELINES] Failed to load global {global_path}: {e}")

    return result


def merge_hist_with_baselines(history: Dict[str, Dict[str, float]], baselines: Dict[str, Dict[str, float]]):
    merged = {}
    merged.update(baselines or {})
    merged.update(history or {})
    return merged


def sort_labels_with_baselines(labels: List[str], baseline_order: List[str] = None) -> List[str]:
    if baseline_order is None:
        baseline_order = ["SOTA", "iter0000000"]
    labels_set = set(labels)
    ordered = [b for b in baseline_order if b in labels_set]
    rest = [l for l in labels if l not in baseline_order]
    rest_sorted = sorted(rest, key=ckpt_key)
    return ordered + rest_sorted


def plot_global_lines(history: Dict[str, Dict[str, float]], out_path: Path, task_name: str):
    if not history:
        print("[PLOT] History is empty; skip global plot.")
        return

    labels = sort_labels_with_baselines(list(history.keys()))

    # --- 为不同任务定义绘图配置；不在白名单里的任务，使用通用逻辑 ---
    if task_name == "share_robot_affordance":
        cats = [c for c in ["average_iou"] if any(c in history[k] for k in labels)]
        title = "ShareRobot Affordance IoU Trend"
        ylabel = "IoU"
        highlight_metric = "average_iou"
        data_type = "Affordance"

    elif task_name == "refspatial_bench":
        cats = [c for c in ["avg"] if any(c in history[k] for k in labels)]
        title = "RefSpatial Bench Accuracy Trend"
        ylabel = "Accuracy (%)"
        highlight_metric = "avg"
        data_type = "Spatial Positioning"

    elif task_name == "where2place":
        cats = [c for c in ["avg"] if any(c in history[k] for k in labels)]
        title = "Where2Place Accuracy Trend"
        ylabel = "Accuracy (%)"
        highlight_metric = "avg"
        data_type = "Spatial Positioning"

    elif task_name == "robovqa":
        cats = [c for c in ["BLEU-avg"] if any(c in history[k] for k in labels)]
        title = "RoboVQA BLEU Score Trend"
        ylabel = "BLEU Score"
        highlight_metric = "BLEU-avg"
        data_type = "Planning"

    elif task_name == "egoplan2":
        cats = [c for c in ["accuracy"] if any(c in history[k] for k in labels)]
        title = "EgoPlan2 Accuracy Trend"
        ylabel = "Accuracy (%)"
        highlight_metric = "accuracy"
        data_type = "Planning Mcq"

    elif task_name == "blink_val":
        cats = [c for c in ["accuracy"] if any(c in history[k] for k in labels)]
        title = "BLiNK Accuracy Trend"
        ylabel = "Accuracy (%)"
        highlight_metric = "accuracy"
        data_type = "Spatial Box Mcq"
    
    elif task_name == "cv_bench_test":
        cats = [c for c in ["accuracy"] if any(c in history[k] for k in labels)]
        title = "CV Bench Accuracy Trend"
        ylabel = "Accuracy (%)"
        highlight_metric = "accuracy"
        data_type = "Spatial QA & Box Mcq"

    elif task_name == "embspatial_bench":
        cats = [c for c in ["accuracy"] if any(c in history[k] for k in labels)]
        title = "EmbSpatial Bench Accuracy Trend"
        ylabel = "Accuracy (%)"
        highlight_metric = "accuracy"
        data_type = "Spatial Reasoning Mcq"

    elif task_name == "robo_spatial_home_all":
        cats = [c for c in ["accuracy"] if any(c in history[k] for k in labels)]
        title = "RoboSpatial Home Accuracy Trend"
        ylabel = "Accuracy (%)"
        highlight_metric = "accuracy"
        data_type = "Spatial Positioning & YesNo"

    elif task_name == "sat":
        cats = [c for c in ["accuracy"] if any(c in history[k] for k in labels)]
        title = "SAT Accuracy Trend"
        ylabel = "Accuracy (%)"
        highlight_metric = "accuracy"
        data_type = "Spatial Reasoning Mcq"

    elif task_name == "vsi_bench_tiny":
        cats = [c for c in ["accuracy"] if any(c in history[k] for k in labels)]
        title = "VSI Bench Tiny Accuracy Trend"
        ylabel = "Accuracy (%)"
        highlight_metric = "accuracy"
        data_type = "Spatial Reasoning Mcq & QA"

    elif task_name == "vsi_bench_test":
        cats = [c for c in ["accuracy"] if any(c in history[k] for k in labels)]
        title = "VSI Bench Test Accuracy Trend"
        ylabel = "Accuracy (%)"
        highlight_metric = "accuracy"
        data_type = "Spatial Reasoning Mcq & QA"

    elif task_name == "erqa":
        cats = [c for c in ["accuracy"] if any(c in history[k] for k in labels)]
        title = "ERQA Accuracy Trend"
        ylabel = "Accuracy (%)"
        highlight_metric = "accuracy"
        data_type = "Embodied Reasoning MCQ"

    elif task_name == "moving_box_val":
        cats = [c for c in ["accuracy"] if any(c in history[k] for k in labels)]
        title = "Moving Box Validation Select_All_ACC Trend"
        ylabel = "Accuracy (%)"
        highlight_metric = "accuracy"
        data_type = "Moving Box Val"

    elif task_name in ["refcoco_val", "refcoco_g_val", "refcoco_plus_val", "refcoco_testA", "refcoco_g_test", "refcoco_plus_testA", "refcoco_testB", "refcoco_plus_testB", "refcoco_val_dq1", "refcoco_g_val_dq1", "refcoco_plus_val_dq1", "refcoco_testA_dq1", "refcoco_g_test_dq1", "refcoco_plus_testA_dq1", "refcoco_testB_dq1", "refcoco_plus_testB_dq1"]:
        cats = [c for c in ["accuracy"] if any(c in history[k] for k in labels)]
        title = f"{task_name} Accuracy Trend"
        ylabel = "Accuracy (%)"
        highlight_metric = "accuracy"
        data_type = f"{task_name} IoU"

    elif task_name == "vsr_random":
        cats = [c for c in ["accuracy"] if any(c in history[k] for k in labels)]
        title = "VSR Random Accuracy Trend"
        ylabel = "Accuracy (%)"
        highlight_metric = "accuracy"
        data_type = "VSR Random yes-no"

    elif task_name == "pixmo_points_eval":
        cats = [c for c in ["accuracy"] if any(c in history[k] for k in labels)]
        title = "PixMo Points Eval Accuracy Trend"
        ylabel = "Accuracy (%)"
        highlight_metric = "accuracy"
        data_type = "PixMo Points Evaluation"

    elif task_name in ["seg_refcoco", "seg_refcocog", "seg_refcocop"]:
        cats = [c for c in ["image/cIoU"] if any(c in history[k] for k in labels)]
        title = f"{task_name} Segmentation cIoU Trend"
        ylabel = "cIoU"
        highlight_metric = "image/cIoU"
        data_type = f"{task_name} Segmentation"

    else:
        # ---- 通用：聚合所有出现过的数值 key ----
        keyset = []
        for k in labels:
            for m in history[k].keys():
                if m not in keyset:
                    keyset.append(m)
        cats = keyset
        title = f"{task_name} Metric Trend"
        ylabel = "Score"
        highlight_metric = cats[0] if cats else None

    if not cats:
        print("[PLOT] No common metrics found across runs; skip global plot.")
        return

    # 收集所有曲线的 y 值到一个矩阵 (num_cats x num_labels)
    series = []
    for c in cats:
        ys = [history[k].get(c, np.nan) for k in labels]
        series.append(np.array(ys, dtype=float))
    stack = np.vstack(series)  # shape: (C, N)

    # 全部都是 NaN 就不画
    if np.all(np.isnan(stack)):
        print("[PLOT] All values are NaN; skip global plot.")
        return

    # --- 选定需要高亮的指标 ---
    max_idx = None
    if highlight_metric is not None and highlight_metric in cats:
        h_idx = cats.index(highlight_metric)
        h_vals = stack[h_idx, :]
        if not np.all(np.isnan(h_vals)):
            max_idx = int(np.nanargmax(h_vals))

    # --- 画图 ---
    plt.figure(figsize=(max(8, len(labels) * 0.5), 5))
    for i, c in enumerate(cats):
        plt.plot(labels, stack[i, :], marker="o", label=c)

    ax = plt.gca()
    # ---- SOTA 横线：不同颜色 + 在虚线上标注数值 ----
    sota_items = list(SOTA_BY_TASK.get(task_name, {}).items())
    if sota_items:
        # 让顺序稳定一点（可选）：按分数从高到低
        sota_items.sort(key=lambda kv: float(kv[1]), reverse=True)

        # 使用 matplotlib 默认颜色循环
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        n_colors = max(1, len(color_cycle))

        for j, (sota_model, sota_score) in enumerate(sota_items):
            if sota_score is None:
                continue
            y = float(sota_score)
            if np.isnan(y):
                continue

            color = color_cycle[j % n_colors] if color_cycle else None

            # 虚线本体
            ax.axhline(
                y,
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
                color=color,
                label=str(sota_model),  # 图例名仍然是模型名
            )

            # 在右侧标注数值（x 用轴坐标，y 用数据坐标）
            ax.text(
                0.99, y, f"{y:.2f}",
                transform=ax.get_yaxis_transform(),
                ha="right", va="bottom",
                fontsize=8,
                color=color,
            )

    # --- 在每个点上标注具体数值，避免重叠做微小错位 ---
    global_min = np.nanmin(stack)
    global_max = np.nanmax(stack)
    y_range = global_max - global_min
    if not np.isfinite(y_range) or y_range == 0:
        base = np.nanmax(np.abs(stack))
        y_range = base if np.isfinite(base) and base > 0 else 1.0

    step_offset = 0.02 * y_range  # 每条线之间的垂直间距

    for col_idx in range(len(labels)):
        col_vals = stack[:, col_idx]  # 当前 x 位置所有类别的取值 (C,)
        valid_mask = ~np.isnan(col_vals)
        if not np.any(valid_mask):
            continue
        order = np.argsort(np.where(np.isnan(col_vals), -np.inf, col_vals))
        k_center = (np.sum(valid_mask) - 1) / 2.0
        for j, i in enumerate(order):
            y = col_vals[i]
            if np.isnan(y):
                continue
            y_disp = y + (j - k_center) * step_offset
            plt.text(
                labels[col_idx], y_disp, f"{y:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    if max_idx is not None:
        xticks = ax.get_xticklabels()
        if max_idx < len(xticks):
            xticks[max_idx].set_color("red")

    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.xlabel("Run / Weight Folder")
    # plt.title(title + (f"  | highlight: {highlight_metric}" if max_idx is not None else ""))
    plt.title(title + (f"  | type: {data_type}" if max_idx is not None else ""))
    plt.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(loc="best", fontsize=8, frameon=True, borderaxespad=0.0)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"[PLOT] Saved global line chart to {out_path}")


def choose_primary_metric(task_name: str, history: Dict[str, Dict[str, float]]) -> Optional[str]:
    """为每个任务选择一条“主指标”曲线用于全局折线图。
    规则：
    1) 若命中预设映射，优先使用。
    2) 否则按常见优先级查找：accuracy > avg > BLEU-avg > average_iou。
    3) 否则选择在该任务历史中出现次数最多的数值键。
    """
    preferred = {
        "share_robot_affordance": "average_iou,Affordance",
        "refspatial_bench": "avg,Spatial Positioning",
        "where2place": "avg,Spatial Positioning",
        "robovqa": "BLEU-avg,Planning",
        "egoplan2": "accuracy,Planning Mcq",
        "blink_val": "accuracy,Spatial Box Mcq",
        "cv_bench_test": "accuracy,Spatial QA & Box Mcq",
        "embspatial_bench": "accuracy,Spatial Reasoning Mcq",
        "robo_spatial_home_all": "accuracy,Spatial Positioning & YesNo",
        "sat": "accuracy,Spatial Reasoning Mcq",
        "vsi_bench_tiny": "accuracy,Spatial Reasoning Mcq & QA",
        "vsi_bench_test": "accuracy,Spatial Reasoning Mcq & QA",
        "erqa": "accuracy,Embodied Reasoning MCQ",
        "moving_box_val": "accuracy,Moving Box Val",
        "refcoco_val": "accuracy,refcoco_val IoU",
        "refcoco_testA": "accuracy,refcoco_testA IoU",
        "refcoco_testB": "accuracy,refcoco_testB IoU",
        "refcoco_g_val": "accuracy,refcoco_g_val IoU",
        "refcoco_g_test": "accuracy,refcoco_g_test IoU",
        "refcoco_plus_val": "accuracy,refcoco_plus_val IoU",
        "refcoco_plus_testA": "accuracy,refcoco_plus_testA IoU",
        "refcoco_plus_testB": "accuracy,refcoco_plus_testB IoU",
        "refcoco_val_dq1": "accuracy,refcoco_val_dq1 IoU",
        "refcoco_testA_dq1": "accuracy,refcoco_testA_dq1 IoU",
        "refcoco_testB_dq1": "accuracy,refcoco_testB_dq1 IoU",
        "refcoco_g_val_dq1": "accuracy,refcoco_g_val_dq1 IoU",
        "refcoco_g_test_dq1": "accuracy,refcoco_g_test_dq1 IoU",
        "refcoco_plus_val_dq1": "accuracy,refcoco_plus_val_dq1 IoU",
        "refcoco_plus_testA_dq1": "accuracy,refcoco_plus_testA_dq1 IoU",
        "refcoco_plus_testB_dq1": "accuracy,refcoco_plus_testB_dq1 IoU",
        "vsr_random": "accuracy,VSR Random yes-no",
        "pixmo_points_eval": "accuracy,PixMo Points Evaluation",
        "seg_refcoco": "image/cIoU,seg_refcoco Segmentation",
        "seg_refcocog": "image/cIoU,seg_refcocog Segmentation",
        "seg_refcocop": "image/cIoU,seg_refcocop Segmentation",
    }.get(task_name)

    # 统计各 metric 在历史中的覆盖次数
    counts: Dict[str, int] = {}
    for run in history.values():
        for k, v in run.items():
            try:
                val = float(v)
                if math.isfinite(val):
                    counts[k] = counts.get(k, 0) + 1
            except Exception:
                continue

    if not counts:
        return None

    if preferred:
        return preferred

    for key in ["accuracy", "avg", "BLEU-avg", "average_iou"]:
        if key in counts:
            return key

    # fallback: 选择出现次数最多的键
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def plot_global_all_tasks(hist_by_task: Dict[str, Dict[str, Dict[str, float]]], out_path: Path):
    """绘制跨任务的全局结果图：横轴为 checkpoint（run 目录名），纵轴为所选“主指标”，每个任务 1 条曲线。
    hist_by_task: {task_name: {run_name: {metric: value}}}
    """
    # 汇总所有 checkpoint 标签
    labels = sort_labels_with_baselines(list({lbl for hist in hist_by_task.values() for lbl in hist.keys()}))
    if not labels:
        print("[PLOT-ALL] No checkpoints to plot.")
        return

    task_order = list(hist_by_task.keys())  # 遵循传入顺序
    series = []
    names = []

    for tn in task_order:
        hist = hist_by_task[tn]
        label_name = choose_primary_metric(tn, hist)
        metric = label_name.split(",")[0]  # 只取指标名部分
        if not metric:
            continue
        ys = [float(hist.get(lbl, {}).get(metric, np.nan)) for lbl in labels]
        arr = np.array(ys, dtype=float)
        if np.all(np.isnan(arr)):
            continue
        series.append(arr)
        names.append(f"{tn}:{label_name}")

    if not series:
        print("[PLOT-ALL] No valid series to plot.")
        return

    stack = np.vstack(series)
    plt.figure(figsize=(max(10, len(labels) * 0.6), 6))
    for i, nm in enumerate(names):
        plt.plot(labels, stack[i, :], marker="o", label=nm)
        # y_last = stack[i, -1]
        # if np.isfinite(y_last):
        #     plt.text(labels[-1], y_last, f"{y_last:.3f}", ha="left", va="center")

    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Score")
    plt.xlabel("Run / Weight Folder")
    plt.title("All Tasks Primary Metric Trend")
    plt.grid(True, linestyle=":", linewidth=0.6)
    ax = plt.gca()
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True)
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"[PLOT-ALL] Saved to {out_path}")

def discover_all_task_histories(model_name: str, global_baselines_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """扫描 ./results/<model_name>/ 下所有任务目录，收集其 _summary.json + _baselines.json 合并后的历史。"""
    base = Path("./results") / model_name
    hist_by_task: Dict[str, Dict[str, Dict[str, float]]] = {}
    if not base.exists():
        return hist_by_task
    for d in base.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("_"):
            continue  # 跳过 _ALL_TASKS 等聚合目录
        # if d.name.startswith("share_robot"):
        #     continue
        summary = d / "_summary.json"
        if not summary.exists():
            continue
        h = load_history(summary)
        b = load_task_baselines(global_baselines_path, d / "_baselines.json", d.name)
        hist_by_task[d.name] = merge_hist_with_baselines(h, b)
    return hist_by_task

def discover_all_task_summary_pngs(model_name: str) -> Dict[str, Path]:
    """扫描 ./results/<model_name>/ 下的所有任务目录，收集其 _summary.png 路径。"""
    base = Path("./results") / model_name
    mapping: Dict[str, Path] = {}
    if not base.exists():
        return mapping
    for d in base.iterdir():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        p = d / "_summary.png"
        if p.exists():
            mapping[d.name] = p
    return mapping

def plot_all_tasks_panels_from_pngs(task_pngs: Dict[str, Path], out_path: Path, cols: int = 3):
    """将每个任务各自的 _summary.png 合并到一个多子图画布上。"""
    tasks = list(task_pngs.keys())
    if not tasks:
        print("[PLOT-ALL2] No task PNGs to compose.")
        return

    n = len(tasks)
    cols = max(1, min(cols, n))
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), squeeze=False)
    for idx, tn in enumerate(tasks):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        try:
            img = mpimg.imread(str(task_pngs[tn]))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(tn)
        except Exception as e:
            ax.text(0.5, 0.5, f"Failed to load\n{task_pngs[tn].name}\n{e}", ha="center", va="center")
            ax.axis("off")
    # 隐藏多余子图
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].axis("off")

    fig.suptitle("All Tasks Panels (composed from _summary.png)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)
    print(f"[PLOT-ALL2] Saved to {out_path}")


def filter_bad_iters(data: dict, threshold: float = 10.0, only_accuracy: bool = False) -> dict:
    """
    从 flagevalmm 的结果 JSON 中删除“低分 iter”：
      - 只处理 key 形如 'iter_XXXX' 的条目
      - 如果该 iter 里有任意一个分数 < threshold，则整个 iter 删除

    如果 only_accuracy=True，则只看名字里包含 'acc' 或 'accuracy' 的字段。
    """
    cleaned = {}
    for key, value in data.items():
        # 非 iter_* 的东西（比如 meta 信息）原样保留
        if not key.startswith("iter_") or not isinstance(value, dict):
            cleaned[key] = value
            continue

        # 找出所有需要判定的 numeric 值
        scores = []
        for k, v in value.items():
            if not isinstance(v, (int, float)):
                continue
            if only_accuracy:
                lk = k.lower()
                if "acc" not in lk and "accuracy" not in lk:
                    continue
            scores.append(float(v))

        # 没有可用分数就保留（根据你需求也可以选择直接丢掉）
        if not scores:
            cleaned[key] = value
            continue

        # 只要有一个 < threshold 就丢掉整 iter
        if any(s < threshold for s in scores):
            print(f"[WARN] Drop iter {key} due to low metrics: {scores}")
            continue

        cleaned[key] = value

    return cleaned


# -------------------- Main Logic --------------------

def have_all_results(results_dir: Path, task_names: List[str]) -> bool:
    """检查所有 task 的结果是否都存在（至少找到一个 *_result.json）。"""
    for tn in task_names:
        tdir = case_insensitive_path(results_dir, tn)
        r = find_result_json(tdir) if tdir else None
        if r is None:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Babysit LoRA export + evaluation + history plotting (multi-task)")
    parser.add_argument("--watch-dir", required=True, type=Path)
    parser.add_argument("--output-base", required=True, type=Path)
    # 接受 1 个或多个 task 脚本路径
    parser.add_argument("--tasks", nargs="+", required=True, help="One or more task script paths, e.g. tasks/a.py tasks/b.py")
    parser.add_argument("--exec", dest="exec_path", default="model_zoo/vlm/qwen_vl/model_adapter.py")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--model-type", default="http", help="模型类型，传递给 --model-type 参数")
    parser.add_argument("--cfg", default="", help="配置参数，传递给模型")
    parser.add_argument("--is-lora", default="true", choices=["true", "false"],
                        help="是否需要先合并 LoRA 再评测；true=先swift export再评测，false=直接用LoRA权重目录评测")
    parser.add_argument("--is-mcore", default="false", choices=["true", "false"],
                        help="是否为mcore模型（影响--swift export参数）；true=是mcore模型，false=普通模型")
    parser.add_argument("--swift-bin", default="swift")
    parser.add_argument("--flagevalmm-bin", default="flagevalmm")
    parser.add_argument("--extra-args", default=(""), help="传递给评测脚本的额外参数字符串")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--stability-window", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--state-id", default="",
                    help="可选后缀，用于区分运行中的处理状态. "
                         "如果为空，将使用 --tasks 的稳定哈希值")
    parser.add_argument("--all-tasks-scope", choices=["suite", "model", "off"], default="suite",
                        help="全局all-tasks画图: "
                            "suite -> _all_tasks.<suite>.png; "
                            "model -> _all_tasks.ALL.png (and legacy _all_tasks.png); "
                            "off -> disable.")
    parser.add_argument("--scp-dest",
                        default="",
                        help="可选：scp 目标，形如 ubt@10.10.22.168:/path/to/dir/；为空则不自动传输")

    args = parser.parse_args()

    watch_dir: Path = args.watch_dir
    output_base: Path = args.output_base

    model_name = output_base.name
    global_baselines_path = Path("./results") / model_name / "_baselines.json"  # 单文件全局基线（可选）
    # 任务名列表（去重但保序）
    task_paths: List[str] = list(dict.fromkeys(args.tasks))
    suffixes = ["_thinker", "_qwen3vl", "_pilecan", "_cosmos", "_mimo", "_robix", "_robobrain2_5"]
    task_names: List[str] = []
    for t in task_paths:
        name = Path(t).stem  # 等价于 name.split(".")[0]，更稳
        for s in suffixes:
            name = name.replace(s, "")
        task_names.append(name)

    # 为每个 task 准备全局 artifact 路径（State 共享）
    global_by_task = {tn: get_global_dirs(model_name, tn) for tn in task_names}

    def _compute_suite_id(paths: List[str]) -> str:
        # 任务列表排序后拼接，生成稳定 hash（和顺序无关）
        norm = " ".join(sorted(str(Path(p)) for p in paths))
        return hashlib.md5(norm.encode()).hexdigest()[:8]

    suite_id = (args.state_id.strip() or _compute_suite_id(task_paths)).replace(os.sep, "_")
    shared_state_path = Path("./results") / model_name / f"STATE.{suite_id}.json"
    print(f"[INFO] Using state file: {shared_state_path}")

    state = load_state(shared_state_path)
    processed: Set[str] = set(state.get("processed", []))


    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    # 查找当前 watch_dir 下的所有一级子目录
    def pending_targets() -> Set[str]:
        current = list_immediate_subdirs(watch_dir)
        print(f"[INFO] Current candidate dirs: {sorted(current, key=ckpt_key)}")
        print(f"[INFO] Already processed dirs: {sorted(processed, key=ckpt_key)}")
        print(f"--------------------------------------------")
        return {d for d in current if d not in processed}
    
    # 自动发送 _ALL_TASKS 目录到远程服务器
    def scp_all_tasks(model_name: str, run_name: str, scp_dest: str):
        """
        将 ./results/<model_name>/_ALL_TASKS 整个目录 scp 到 scp_dest。
        scp_dest 形如:  ubt@10.10.22.168:/some/dir/
        """
        if not scp_dest:
            return

        all_base = Path("./results") / model_name / "_ALL_TASKS"
        if not all_base.exists():
            print(f"[WARN] _ALL_TASKS not found: {all_base}, skip scp.")
            return

        cmd = [
            "scp", "-r",
            "-o", "StrictHostKeyChecking=accept-new",
            str(all_base),
            scp_dest
        ]

        try:
            r = subprocess.run(cmd, text=True)
            if r.returncode != 0:
                print(f"[WARN] scp _ALL_TASKS failed for '{run_name}', rc={r.returncode}.")
            else:
                print(f"[INFO] scp _ALL_TASKS done for '{run_name}'.")
        except Exception as e:
            print(f"[WARN] scp exception for '{run_name}': {e}")

    def process_one(name: str):
        src_dir = Path(f"{watch_dir}/{name}")
        if not src_dir.is_dir():
            return
        if not dir_is_stable(src_dir, window_sec=args.stability_window):
            print(f"[INFO] Skipping '{name}' for now; folder not yet stable.")
            return

        out_dir = output_base / name
        logs_dir = Path("./logs") / output_base.name / name
        logs_dir.mkdir(parents=True, exist_ok=True)
        export_log = logs_dir / "export.log"
        eval_log = logs_dir / "eval.log"

        is_lora = (args.is_lora.lower() == "true")
        is_mcore = (args.is_mcore.lower() == "true")
        print(f"[INFO] Processing '{name}': src={src_dir}, out={out_dir}, is_lora={is_lora}, is_mcore={is_mcore}")
        # 给 mcore 导出指定 iter（通过 latest_checkpointed_iteration.txt）
        def _try_update_latest_iter_from_name():
            try:
                match = re.search(r"\d+", name)
                if not match:
                    raise ValueError(f"Invalid name format: {name}")
                iteration_num = int(match.group())
                latest_ckpt_file = watch_dir / "latest_checkpointed_iteration.txt"
                with open(latest_ckpt_file, "w") as f:
                    f.write(str(iteration_num))
                print(f"[INFO] Updated {latest_ckpt_file} to {iteration_num}")
            except Exception as e:
                print(f"[ERROR] Failed to update latest_checkpointed_iteration.txt: {e}")

        if is_lora:
            merge_exists = False
            if out_dir.exists() and any(out_dir.iterdir()):
                print(f"[INFO] Output dir {out_dir} already exists and non-empty; skipping export.")
                merge_exists = True

            if not merge_exists:
                if is_mcore:
                    print(f"[INFO] Detected mcore model; using mcore export parameters.")
                    _try_update_latest_iter_from_name()

                    export_cmd = (
                        f"CUDA_VISIBLE_DEVICES={args.cuda} "
                        f"{args.swift_bin} export "
                        f"--mcore_adapters {str(watch_dir)} "
                        f"--to_hf true "
                        f"--torch_dtype bfloat16 "
                        f"--output_dir {str(out_dir)}"
                    )
                else:
                    export_cmd = (
                        f"CUDA_VISIBLE_DEVICES={args.cuda} "
                        f"{args.swift_bin} export "
                        f"--adapters {str(src_dir)} "
                        f"--merge_lora {args.is_lora} "
                        f"--output_dir {str(out_dir)}"
                    )
                rc = run_and_wait(export_cmd, log_path=export_log)  # 阻塞直至导出结束
                if rc != 0:
                    print(f"[ERROR] swift export failed for {name}.")
                    return

            model_path = str(out_dir)
        else:
            # 处理全参训练的 full 形式 mcore
            if is_mcore:
                # mcore full checkpoints -> 先转 HF 再评测
                if out_dir.exists() and any(out_dir.iterdir()):
                    print(f"[INFO] Output dir {out_dir} already exists and non-empty; skipping mcore->hf export.")
                else:
                    print(f"[INFO] Detected mcore full; converting mcore -> HF for evaluation.")
                    _try_update_latest_iter_from_name()

                    export_cmd = (
                        f"CUDA_VISIBLE_DEVICES={args.cuda} "
                        f"{args.swift_bin} export "
                        f"--mcore_model {str(watch_dir)} "
                        f"--to_hf true "
                        f"--torch_dtype bfloat16 "
                        f"--output_dir {str(out_dir)}"
                    )
                    rc = run_and_wait(export_cmd, log_path=export_log)
                    if rc != 0:
                        print(f"[ERROR] swift export (mcore full -> hf) failed for {name}.")
                        return

                model_path = str(out_dir)
                print(f"[INFO] Evaluate on converted HF model: {model_path}")

            else:
                # 不是LoRA权重，且不是mcore：使用正常全参模型目录做评测
                model_path = str(src_dir)
                print(f"[INFO] Skip merge. Evaluate directly on {model_path}")

        results_root = Path("./results") / model_name.split("/")[-1] / name
        # 如果任何一个 task 的结果缺失，则触发 eval；否则跳过
        need_eval = not have_all_results(results_root, task_names)

        if need_eval:
            tasks_joined = " ".join(task_paths)

            eval_cmd = [
                args.flagevalmm_bin,
                "--tasks", *tasks_joined.split(),
                "--exec", args.exec_path,
                "--model", model_path,
                "--model-type", args.model_type,
                "--num-workers", str(args.num_workers),
                "--output-dir", str(results_root),
                "--backend", args.backend,
                "--cfg", args.cfg,
                "--extra-args", args.extra_args,  # 保留外层那一整串，原封不动
            ]
            rc = run_and_wait(eval_cmd, env=env, log_path=eval_log, shell=False)  # 阻塞直至评测结束
            if rc != 0:
                print(f"[ERROR] flagevalmm failed for {name}.")
                return
        else:
            print(f"[INFO] All task results already exist for run '{name}', skip eval.")

        # ---- 逐 task 解析结果、更新历史、绘图 ----
        for tn in task_names:
            task_paths_obj = global_by_task[tn]
            results_json_dir = case_insensitive_path(results_root, tn)
            results_json = find_result_json(results_json_dir)
            if results_json is None:
                print(f"[WARN] No results JSON found for task '{tn}' under {results_json_dir}")
                continue

            # 单次 run 的简单折线图（保存在该 task 目录）
            # plot_single_run(results_json, results_json_dir / "accuracy.png", title=f"{name} - {tn}", task_name=tn)

            # 更新该 task 的全局历史并绘制趋势
            try:
                data_raw = json.loads(results_json.read_text())
                # 先过滤掉坏 iter（可以用 only_accuracy=True，只看 accuracy 指标）
                data = filter_bad_iters(data_raw, threshold=10.0, only_accuracy=True)
                # 把清洗结果写回去，这样以后再跑也不会把坏 iter 画出来
                results_json.write_text(json.dumps(data, indent=2, ensure_ascii=False))
                metrics = extract_metrics(data, tn)
                hist = update_history(task_paths_obj["history"], name, metrics)

                baselines = load_task_baselines(global_baselines_path, task_paths_obj["baselines"], tn)
                hist_merged = merge_hist_with_baselines(hist, baselines)
                plot_global_lines(hist_merged, task_paths_obj["png"], task_name=tn)
            except Exception as e:
                print(f"[WARN] Failed to update history for task '{tn}' from {results_json}: {e}")

        # ---- 全局跨任务趋势图（可按套件或全模型） ----
        try:
            all_base = Path("./results") / model_name / "_ALL_TASKS"
            if args.all_tasks_scope == "suite":
                # 仅使用当前 babysitter 的任务集合，避免与另一套件冲突
                hist_by_task = {}
                for tn in task_names:
                    _h = load_history(global_by_task[tn]["history"])
                    _b = load_task_baselines(global_baselines_path, global_by_task[tn]["baselines"], tn)
                    hist_by_task[tn] = merge_hist_with_baselines(_h, _b)
                all_png = all_base / f"_all_tasks.{suite_id}.png"
                plot_global_all_tasks(hist_by_task, all_png)

            elif args.all_tasks_scope == "model":
                # 扫描所有任务目录，画真正“全模型”总图
                hist_by_task = discover_all_task_histories(model_name, global_baselines_path)
                all_png = all_base / "_all_tasks.ALL.png"
                plot_global_all_tasks(hist_by_task, all_png)
                # 为兼容旧路径，复制一份
                legacy = all_base / "_all_tasks.png"
                try:
                    shutil.copyfile(all_png, legacy)
                except Exception:
                    pass
                # 追加合成 ALL2 大面板
                task_pngs = discover_all_task_summary_pngs(model_name)
                all_png2 = all_base / "_all_tasks.ALL2.png"
                plot_all_tasks_panels_from_pngs(task_pngs, all_png2, cols=3)
            else:
                # off: 不画全局跨任务图
                pass
        except Exception as e:
            print(f"[WARN] Failed to draw global all-tasks chart: {e}")

        # ---- 每个 iter 画完图后自动 scp _ALL_TASKS ----
        try:
            scp_all_tasks(model_name, name, args.scp_dest)
        except Exception as e:
            print(f"[WARN] SCP step failed unexpectedly for '{name}': {e}")

        processed.add(name)
        save_state(shared_state_path, {"processed": list(processed)})
        print(f"[DONE] {name} processed successfully.")

    while True:
        for name in sorted(pending_targets(), key=ckpt_key):
            try:
                # if name != "iter_0013000":
                #     print(f"[SKIP] Currently only process 'iter_0013000' for demo; skip '{name}'.")
                #     continue
                process_one(name)
            except Exception as e:
                print(f"[EXCEPTION] While processing '{name}': {e}")
        if args.once:
            break
        time.sleep(args.poll_interval)

if __name__ == "__main__":
    main()
