#!/usr/bin/env python3
"""
Babysit a LoRA training folder and automatically:
1) Run `swift export` to merge the LoRA into full weights.
2) Run `flagevalmm` to evaluate the merged model.
3) Parse evaluation results JSON, append to history JSON, and draw global line chart.

Global artifacts (auto-updated on each successful eval) will be stored under:
./results/<model_name>/<task_name>/
- STATE.json
- _summary.json
- _summary.png
"""

import argparse
import json
import os
import sys
import re
import numpy as np
import time
import subprocess
import math
from pathlib import Path
from typing import Dict, Set, Optional

# Headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- Global Artifact Paths --------------------

def get_global_dirs(model_name: str, task_name: str):
    base = Path("./results") / model_name / task_name
    print(f"[INFO] Global artifacts base dir: {base}")
    return {
        "base": base,
        "state": base / "STATE.json",
        "history": base / "_summary.json",
        "png": base / "_summary.png",
    }

# -------------------- State Management --------------------

def load_state(state_path: Path) -> Dict[str, str]:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except Exception:
            return {"processed": []}
    return {"processed": []}


def save_state(state_path: Path, state: Dict[str, Set[str]]):
    state["processed"] = sorted(list(set(state.get("processed", []))))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def list_immediate_subdirs(p: Path) -> Set[str]:
    filters = set()
    for d in p.iterdir():
        if d.is_dir() and (d.name.startswith("checkpoint-") or d.name.startswith("iter_")):
            filters.add(d.name)

    return filters

def ckpt_key(name: str) -> int:
    # 匹配末尾数字
    m = re.search(r"(\d+)$", name)
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

def run(cmd: str, env: Optional[Dict[str, str]] = None, log_path: Optional[Path] = None) -> int:
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
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_f:
                log_f.write(line)
        proc.wait()
        rc = proc.returncode or 0
        if log_f:
            log_f.write(f"===== EXIT CODE: {rc}\n")
        return rc
    finally:
        if log_f:
            log_f.close()


def run_and_wait(
    cmd: str,
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
    timeout: Optional[int] = None,   # 秒；None 表示不设超时
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
            shell=True,
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
    for child in base.iterdir():
        if child.name.lower() == task_name.lower():
            return child
    return None


def find_result_json(results_dir: Path) -> Optional[Path]:
    candidates = sorted(results_dir.glob("*.json"))
    for c in candidates:
        if c.name.endswith("result.json"):
            return c
    return None


def extract_metrics(data: dict, task_name: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    if task_name == "share_robot_affordance":
        # 主指标：average_iou
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
        bleu1 = data.get('overall', {}).get('BLEU-1', float('nan'))
        bleu2 = data.get('overall', {}).get('BLEU-2', float('nan'))
        bleu3 = data.get('overall', {}).get('BLEU-3', float('nan'))
        bleu4 = data.get('overall', {}).get('BLEU-4', float('nan'))
        blue_avg = data.get('overall', {}).get('BLEU-avg', float('nan'))
        metrics.update({
            "BLEU-1": float(bleu1) if isinstance(bleu1, (int, float)) else float("nan"),
            "BLEU-2": float(bleu2) if isinstance(bleu2, (int, float)) else float("nan"),
            "BLEU-3": float(bleu3) if isinstance(bleu3, (int, float)) else float("nan"),
            "BLEU-4": float(bleu4) if isinstance(bleu4, (int, float)) else float("nan"),
            "BLEU-avg": float(blue_avg) if isinstance(blue_avg, (int, float)) else float("nan"),
        })

    elif task_name == "egoplan2":
        accuracy = data.get('overall', {}).get('accuracy', float('nan'))
        metrics.update({
            "accuracy": float(accuracy) if isinstance(accuracy, (int, float)) else float("nan"),
        })

    else:
        # 未知任务名：返回空或仅放一个 NaN 占位，按需选择
        # 这里返回空字典，调用方自行处理
        pass

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

def plot_global_lines(history: Dict[str, Dict[str, float]], out_path: Path, task_name: str):
    if not history:
        print("[PLOT] History is empty; skip global plot.")
        return

    labels = sorted(history.keys(), key=ckpt_key)

    if task_name == "share_robot_affordance":
        cats = [c for c in ["average_iou"] if any(c in history[k] for k in labels)]
        title = "ShareRobot Affordance IoU Trend"
        ylabel = "IoU"
        highlight_metric = "average_iou"

    elif task_name == "refspatial_bench":
        cats = [c for c in ["avg", "location", "placement", "unseen"]
                if any(c in history[k] for k in labels)]
        title = "RefSpatial Bench Accuracy Trend"
        ylabel = "Accuracy (%)"
        highlight_metric = "avg"

    elif task_name == "where2place":
        cats = [c for c in ["avg", "seen", "unseen"]
                if any(c in history[k] for k in labels)]
        title = "Where2Place Accuracy Trend"
        ylabel = "Accuracy (%)"
        highlight_metric = "avg"
    
    elif task_name == "robovqa":
        cats = [c for c in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "BLEU-avg"]
                if any(c in history[k] for k in labels)]
        title = "RoboVQA BLEU Score Trend"
        ylabel = "BLEU Score"
        highlight_metric = "BLEU-avg"

    elif task_name == "egoplan2":
        cats = [c for c in ["accuracy"]
                if any(c in history[k] for k in labels)]
        title = "EgoPlan2 Accuracy Trend"
        ylabel = "Accuracy (%)"
        highlight_metric = "accuracy"

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
    if highlight_metric is None or highlight_metric not in cats:
        print(f"[PLOT] Highlight metric '{highlight_metric}' not found in categories.")

    h_idx = cats.index(highlight_metric)
    h_vals = stack[h_idx, :]

    if np.all(np.isnan(h_vals)):
        print(f"[PLOT] All values are NaN for highlight metric '{highlight_metric}'; skip highlighting.")
        max_idx = None
    else:
        max_idx = int(np.nanargmax(h_vals))
        max_label = labels[max_idx]
        max_val = h_vals[max_idx]

    # --- 画图 ---
    plt.figure(figsize=(max(8, len(labels) * 0.5), 5))
    for i, c in enumerate(cats):
        plt.plot(labels, stack[i, :], marker="o", label=c)

    ax = plt.gca()

    # # --- 标注：在高亮横坐标位置，把所有曲线的 y 值都标上数字，并把该刻度标红 ---
    # if max_idx is not None:
    #     # 把该 x 位置所有 y 取出
    #     col_vals = stack[:, max_idx]  # shape: (C,)

    #     # 计算一个合适的竖直偏移，避免重叠
    #     global_min = np.nanmin(stack)
    #     global_max = np.nanmax(stack)
    #     y_range = global_max - global_min
    #     if not np.isfinite(y_range) or y_range == 0:
    #         # 回退：用当前列的跨度或值大小的一个比例
    #         col_span = np.nanmax(col_vals) - np.nanmin(col_vals)
    #         if not np.isfinite(col_span) or col_span == 0:
    #             base = np.nanmax(np.abs(col_vals))
    #             y_range = base if np.isfinite(base) and base > 0 else 1.0
    #         else:
    #             y_range = col_span

    #     step_offset = 0.02 * y_range  # 每条线之间的垂直间距

    #     # 为了更清晰，按 y 值从小到大排一下，低的往下偏移，高的往上偏移
    #     order = np.argsort(np.where(np.isnan(col_vals), -np.inf, col_vals))
    #     for j, i in enumerate(order):
    #         y = col_vals[i]
    #         if np.isnan(y):
    #             continue
    #         # 把中心放在原值附近，上下均匀错开
    #         k = j - (np.sum(~np.isnan(col_vals)) - 1) / 2.0
    #         y_disp = y + k * step_offset
    #         plt.text(labels[max_idx], y_disp, f"{y:.6f}",
    #                  ha="center", va="bottom", fontsize=9, fontweight="bold")

    #     # 标红对应的 xtick
    #     xticks = ax.get_xticklabels()
    #     if max_idx < len(xticks):
    #         xticks[max_idx].set_color("red")


    # --- 标注：对每个横坐标位置（每个 checkpoint），把所有曲线的 y 值都标上数字 ---
    # 先计算一个全局 y_range，便于控制错位间隔
    global_min = np.nanmin(stack)
    global_max = np.nanmax(stack)
    y_range = global_max - global_min
    if not np.isfinite(y_range) or y_range == 0:
        # 回退：用所有值的绝对最大值或 1.0
        base = np.nanmax(np.abs(stack))
        y_range = base if np.isfinite(base) and base > 0 else 1.0

    step_offset = 0.02 * y_range  # 每条线之间的垂直间距

    for col_idx in range(len(labels)):
        col_vals = stack[:, col_idx]  # 当前 x 位置所有类别的取值 (C,)

        # 为了更清晰，按 y 值从小到大排一下，低的往下偏移，高的往上偏移
        valid_mask = ~np.isnan(col_vals)
        if not np.any(valid_mask):
            continue
        order = np.argsort(np.where(np.isnan(col_vals), -np.inf, col_vals))
        # 使偏移围绕中心对称
        k_center = (np.sum(valid_mask) - 1) / 2.0

        for j, i in enumerate(order):
            y = col_vals[i]
            if np.isnan(y):
                continue
            y_disp = y + (j - k_center) * step_offset
            plt.text(
                labels[col_idx],
                y_disp,
                f"{y:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # 如果还想保留“高亮最佳点并把该 xtick 设为红色”的效果，下面这几行可以保留
    if max_idx is not None:
        xticks = ax.get_xticklabels()
        if max_idx < len(xticks):
            xticks[max_idx].set_color("red")


    # 轴与装饰
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.xlabel("Run / Weight Folder")
    plt.title(title + (f"  | highlight: {highlight_metric}" if max_idx is not None else ""))
    plt.grid(True, linestyle=":", linewidth=0.6)
    # 图例放在右上角（轴外）
    ax = plt.gca()
    legend = ax.legend(
        loc="upper left",           # 图例自身的左上角
        bbox_to_anchor=(1.02, 1.0), # 锚到坐标轴右上角的外侧
        borderaxespad=0.0,
        frameon=True,
    )

    # 为图例预留右侧空白（比如 18%），避免被裁剪或压住曲线
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"[PLOT] Saved global line chart to {out_path}")


def plot_single_run(results_json: Path, out_path: Path, title: str, task_name: str):
    try:
        data = json.loads(results_json.read_text())
    except Exception as e:
        print(f"[WARN] Could not parse results JSON {results_json}: {e}")
        return

    labels, values = [], []

    if task_name == "share_robot_affordance":
        ylabel = "IoU"
        labels.append("average_iou")
        values.append(float(data["average_iou"]))

    elif task_name == "refspatial_bench":
        ylabel = "Accuracy (%)"
        metrics = ["avg", "location", "placement", "unseen"]
        for m in metrics:
            if m in data and isinstance(data[m], dict) and ("accuracy" in data[m]):
                labels.append(m)
                values.append(float(data[m]["accuracy"]))
            else:
                print(f"[WARN] accuracy for '{m}' not found.")
    
    elif task_name == "where2place":
        ylabel = "Accuracy (%)"
        metrics = ["avg", "seen", "unseen"]
        for m in metrics:
            if m in data and isinstance(data[m], dict) and ("accuracy" in data[m]):
                labels.append(m)
                values.append(float(data[m]["accuracy"]))
            else:
                print(f"[WARN] accuracy for '{m}' not found.")

    elif task_name == "robovqa":
        ylabel = "BLEU Score"
        metrics = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "BLEU-avg"]
        for m in metrics:
            v = data.get('overall', {}).get(m, None)
            if v is not None:
                labels.append(m)
                values.append(float(v))
            else:
                print(f"[WARN] {m} not found.")

    elif task_name == "egoplan2":
        ylabel = "Accuracy (%)"
        metrics = ["accuracy"]
        for m in metrics:
            v = data.get('overall', {}).get(m, None)
            if v is not None:
                labels.append(m)
                values.append(float(v))
            else:
                print(f"[WARN] {m} not found.")

    plt.figure(figsize=(6, 4))
    plt.plot(labels, values, marker="o")
    plt.ylabel(ylabel)
    plt.xlabel("Metric")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"[PLOT] Saved per-run lineplot to {out_path}")

# -------------------- Main Logic --------------------

def main():
    parser = argparse.ArgumentParser(description="Babysit LoRA export + evaluation + history plotting")
    parser.add_argument("--watch-dir", required=True, type=Path)
    parser.add_argument("--output-base", required=True, type=Path)
    parser.add_argument("--tasks", default="tasks/refspatial_bench/refspatial_bench.py")
    parser.add_argument("--exec", dest="exec_path", default="model_zoo/vlm/qwen_vl/model_adapter.py")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--is-lora", default="true", choices=["true", "false"],
                        help="是否需要先合并 LoRA 再评测；true=先swift export再评测，false=直接用LoRA权重目录评测")
    parser.add_argument("--is-mcore", default="false", choices=["true", "false"],
                        help="是否为mcore模型（影响--swift export参数）；true=是mcore模型，false=普通模型")
    parser.add_argument("--swift-bin", default="swift")
    parser.add_argument("--flagevalmm-bin", default="flagevalmm")
    parser.add_argument("--extra-args", default=(
        "--limit-mm-per-prompt '{\"image\": 8, \"video\": 1}' --media-io-kwargs '{\"video\": {\"num_frames\": 8} }' --tensor-parallel-size 8 "
        "--max-model-len 40000 --trust-remote-code --gpu-memory-utilization 0.8}'"
    ))
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--stability-window", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    watch_dir: Path = args.watch_dir
    output_base: Path = args.output_base

    model_name = output_base.name
    task_name = Path(args.tasks).name.split(".")[0]
    global_paths = get_global_dirs(model_name, task_name)

    state = load_state(global_paths["state"])
    processed: Set[str] = set(state.get("processed", []))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    def pending_targets() -> Set[str]:
        current = list_immediate_subdirs(watch_dir)
        print(f"[INFO] Current candidate dirs: {sorted(current)}")
        print(f"[INFO] Already processed dirs: {sorted(processed)}")
        print(f"--------------------------------------------")
        return {d for d in current if d not in processed}

    def process_one(name: str):
        src_dir = Path(f"{watch_dir}/{name}")
        if not src_dir.is_dir():
            return
        if not dir_is_stable(src_dir, window_sec=args.stability_window):
            print(f"[INFO] Skipping '{name}' for now; folder not yet stable.")
            return

        out_dir = output_base / name
        logs_dir = Path("./logs") / name
        export_log = logs_dir / "export.log"
        eval_log = logs_dir / "eval.log"

        is_lora = (args.is_lora.lower() == "true")
        is_mcore = (args.is_mcore.lower() == "true")
        print(f"[INFO] Processing '{name}': src={src_dir}, out={out_dir}, is_lora={is_lora}, is_mcore={is_mcore}")

        if is_lora:
            merge_exists = False
            if out_dir.exists() and any(out_dir.iterdir()):
                print(f"[INFO] Output dir {out_dir} already exists and non-empty; skipping export.")
                merge_exists = True

            if not merge_exists:
                if is_mcore:
                    print(f"[INFO] Detected mcore model; using mcore export parameters.")
                    try:
                        # 直接匹配字符串中的数字
                        match = re.search(r"\d+", name)
                        if not match:
                            raise ValueError(f"Invalid name format: {name}")

                        iteration_num = int(match.group())  # 自动去掉前导零
                        latest_ckpt_file = watch_dir / "latest_checkpointed_iteration.txt"

                        with open(latest_ckpt_file, "w") as f:
                            f.write(str(iteration_num))

                        print(f"[INFO] Updated {latest_ckpt_file} to {iteration_num}")

                    except Exception as e:
                        print(f"[ERROR] Failed to update latest_checkpointed_iteration.txt: {e}")
                        
                    export_cmd = (
                        f"{args.swift_bin} export "
                        f"--mcore_adapters {str(watch_dir)} "
                        f"--to_hf true "
                        f"--torch_dtype bfloat16 "
                        f"--output_dir {str(out_dir)} "
                    )
                else:
                    export_cmd = (
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
            # 不是LoRA权重，使用正常全参模型目录做评测
            model_path = str(src_dir)
            print(f"[INFO] Skip merge. Evaluate directly on {model_path}")

        results_dir = Path("./results") / model_name.split("/")[-1] / name
        if not results_dir.exists():
            eval_cmd = (
                f"{args.flagevalmm_bin} --tasks {args.tasks} "
                f"--exec {args.exec_path} "
                f"--model {model_path} "
                f"--num-workers {args.num_workers} "
                f"--output-dir {str(results_dir)} "
                f"--backend {args.backend} "
                f"--extra-args \"{args.extra_args}\""
            )
            rc = run_and_wait(eval_cmd, env=env, log_path=eval_log)  # 阻塞直至评测结束
            if rc != 0:
                print(f"[ERROR] flagevalmm failed for {name}.")
                return

        results_json_path = case_insensitive_path(results_dir, task_name)
        results_json = find_result_json(results_json_path)
        if results_json is None:
            print(f"[WARN] No results JSON found in {results_json_path}")
        else:
            plot_single_run(results_json, results_json_path / "accuracy.png", title=name, task_name=task_name)
            try:
                data = json.loads(results_json.read_text())
                metrics = extract_metrics(data, task_name)
                hist = update_history(global_paths["history"], name, metrics)
                plot_global_lines(hist, global_paths["png"], task_name=task_name)
            except Exception as e:
                print(f"[WARN] Failed to update history from {results_json}: {e}")

        processed.add(name)
        save_state(global_paths["state"], {"processed": list(processed)})
        print(f"[DONE] {name} processed successfully.")

    while True:
        for name in sorted(pending_targets(), key=ckpt_key):
            try:
                process_one(name)
            except Exception as e:
                print(f"[EXCEPTION] While processing '{name}': {e}")
        if args.once:
            break
        time.sleep(args.poll_interval)

if __name__ == "__main__":
    main()
