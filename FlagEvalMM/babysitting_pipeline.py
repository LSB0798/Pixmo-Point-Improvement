#!/usr/bin/env python3
"""
Babysit LoRA training folders and automatically:
1) Run `swift export` to merge the LoRA into full weights.
2) Run `flagevalmm` to evaluate the merged model (supports MULTIPLE tasks).
3) Parse evaluation results JSON, append to per-task history JSON, and draw per-task global trend charts.

NOW SUPPORTS MULTI TARGETS (multi watch-dir + output-base):
- Timeslicing scheduler: whichever target produces stable weights earlier gets evaluated earlier.
- Results and plots remain isolated under:
  ./results/<model_name>/<task_name>/
  ./results/<model_name>/STATE.<suite_id>.json

Global artifacts (auto-updated on each successful eval) stored under:
./results/<model_name>/<task_name>/
- _summary.json (per task)
- _summary.png  (per task)

Baseline support (_baselines.json) has been removed; if you want a baseline,
copy the base model as a run such as iter_000000 and evaluate it like others.
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
from typing import Dict, Set, Optional, List, Union, Tuple
from dataclasses import dataclass, field

# Headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# -------------------- Global Artifact Paths --------------------

def get_global_dirs(model_name: str, task_name: str):
    base = Path("./results") / model_name / task_name
    print(f"[INFO] Global artifacts base dir (task={task_name}): {base}")
    return {
        "base": base,
        "history": base / "_summary.json",
        "png": base / "_summary.png",
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
            filters.add(d.name)
    return filters


def ckpt_key(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    if m:
        return int(m.group(1))
    return -1


def get_latest_mtime(path: Path) -> float:
    """返回路径下所有文件中修改时间最新的文件。如果为空，则返回目录修改时间"""
    latest = 0.0
    if not path.exists():
        return latest
    try:
        latest = max(latest, path.stat().st_mtime)
    except Exception:
        pass
    for root, _, files in os.walk(path):
        for f in files:
            try:
                mtime = (Path(root) / f).stat().st_mtime
                latest = max(latest, mtime)
            except FileNotFoundError:
                continue
            except Exception:
                continue
    return latest


def dir_is_stable(path: Path, window_sec: int = 30) -> bool:
    latest_mtime = 0.0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                mtime = (Path(root) / f).stat().st_mtime
                latest_mtime = max(latest_mtime, mtime)
            except FileNotFoundError:
                return False
    if latest_mtime == 0.0:
        try:
            latest_mtime = path.stat().st_mtime
        except Exception:
            latest_mtime = time.time()
    return (time.time() - latest_mtime) >= window_sec

# -------------------- Run Helpers --------------------

def run_and_wait(
    cmd: Union[str, List[str]],
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
    timeout: Optional[int] = None,
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

        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_f:
                log_f.write(line)

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
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
    return candidates[0] if candidates else None


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

    else:
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

def sort_labels_with_baselines(labels: List[str], baseline_order: List[str] = None) -> List[str]:
    """
    Only used for label ordering; does NOT read any baselines file.
    Keeps 'SOTA' and 'iter0000000' (if present) at the front, then sort by numeric suffix.
    """
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

    elif task_name == "erqa":
        cats = [c for c in ["accuracy"] if any(c in history[k] for k in labels)]
        title = "ERQA Accuracy Trend"
        ylabel = "Accuracy (%)"
        highlight_metric = "accuracy"
        data_type = "Embodied Reasoning MCQ"

    else:
        keyset = []
        for k in labels:
            for m in history[k].keys():
                if m not in keyset:
                    keyset.append(m)
        cats = keyset
        title = f"{task_name} Metric Trend"
        ylabel = "Score"
        highlight_metric = cats[0] if cats else None
        data_type = ""

    if not cats:
        print("[PLOT] No common metrics found across runs; skip global plot.")
        return

    series = []
    for c in cats:
        ys = [history[k].get(c, np.nan) for k in labels]
        series.append(np.array(ys, dtype=float))
    stack = np.vstack(series)

    if np.all(np.isnan(stack)):
        print("[PLOT] All values are NaN; skip global plot.")
        return

    max_idx = None
    if highlight_metric is not None and highlight_metric in cats:
        h_idx = cats.index(highlight_metric)
        h_vals = stack[h_idx, :]
        if not np.all(np.isnan(h_vals)):
            max_idx = int(np.nanargmax(h_vals))

    plt.figure(figsize=(max(8, len(labels) * 0.5), 5))
    for i, c in enumerate(cats):
        plt.plot(labels, stack[i, :], marker="o", label=c)

    ax = plt.gca()

    global_min = np.nanmin(stack)
    global_max = np.nanmax(stack)
    y_range = global_max - global_min
    if not np.isfinite(y_range) or y_range == 0:
        base = np.nanmax(np.abs(stack))
        y_range = base if np.isfinite(base) and base > 0 else 1.0

    step_offset = 0.02 * y_range

    for col_idx in range(len(labels)):
        col_vals = stack[:, col_idx]
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
    plt.title(title + (f"  | type: {data_type}" if max_idx is not None and data_type else ""))
    plt.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(loc="best", fontsize=8, frameon=True, borderaxespad=0.0)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"[PLOT] Saved global line chart to {out_path}")


def choose_primary_metric(task_name: str, history: Dict[str, Dict[str, float]]) -> Optional[str]:
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
        "erqa": "accuracy,Embodied Reasoning MCQ",
    }.get(task_name)

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

    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def plot_global_all_tasks(hist_by_task: Dict[str, Dict[str, Dict[str, float]]], out_path: Path):
    labels = sort_labels_with_baselines(list({lbl for hist in hist_by_task.values() for lbl in hist.keys()}))
    if not labels:
        print("[PLOT-ALL] No checkpoints to plot.")
        return

    task_order = list(hist_by_task.keys())
    series = []
    names = []

    for tn in task_order:
        hist = hist_by_task[tn]
        label_name = choose_primary_metric(tn, hist)
        if not label_name:
            continue
        metric = label_name.split(",")[0]
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


def discover_all_task_histories(model_name: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    扫描 ./results/<模型名称>/ 目录下的所有任务目录，并收集它们的 _summary.json 历史记录。
    Baselines 不进行合并；仅使用实际运行数据。
    """
    base = Path("./results") / model_name
    hist_by_task: Dict[str, Dict[str, Dict[str, float]]] = {}
    if not base.exists():
        return hist_by_task
    for d in base.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("_"):
            continue
        summary = d / "_summary.json"
        if not summary.exists():
            continue
        h = load_history(summary)
        hist_by_task[d.name] = h
    return hist_by_task

def discover_all_task_summary_pngs(model_name: str) -> Dict[str, Path]:
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

    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].axis("off")

    fig.suptitle("All Tasks Panels (composed from _summary.png)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)
    print(f"[PLOT-ALL2] Saved to {out_path}")

# -------------------- Main Logic Helpers --------------------

def have_all_results(results_dir: Path, task_names: List[str]) -> bool:
    for tn in task_names:
        tdir = case_insensitive_path(results_dir, tn)
        r = find_result_json(tdir) if tdir else None
        if r is None:
            return False
    return True


def _compute_suite_id(paths: List[str]) -> str:
    norm = " ".join(sorted(str(Path(p)) for p in paths))
    return hashlib.md5(norm.encode()).hexdigest()[:8]


def _parse_target_string(s: str) -> Tuple[Path, Path, Optional[str]]:
    """
    解析目标规范字符串。
    支持的格式：
    1) watch:out
    2) watch::out
    3) watch:out:model_name（可选）
    4) watch::out::model_name
    """
    if "::" in s:
        parts = s.split("::")
    else:
        parts = s.split(":")

    parts = [p for p in parts if p != ""]
    if len(parts) < 2:
        raise ValueError(f"Invalid target spec: {s}")
    watch = Path(parts[0])
    out = Path(parts[1])
    mname = parts[2] if len(parts) >= 3 else None
    return watch, out, mname


def load_targets_from_config(path: Path) -> List[Dict[str, str]]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, list):
        raise ValueError("targets-config must be a JSON list")
    out = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        wd = item.get("watch_dir") or item.get("watch") or item.get("watch-dir")
        ob = item.get("output_base") or item.get("output") or item.get("output-base")
        mn = item.get("model_name") or item.get("model")
        if not wd or not ob:
            continue
        out.append({"watch_dir": str(wd), "output_base": str(ob), "model_name": str(mn) if mn else ""})
    return out


def derive_unique_model_names(targets: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    确保模型名称在所有目标中唯一。
    默认model_name = output_base.name。
    如果重复，则附加 watch_dir + output_base 的短哈希值。
    """
    derived = []
    for t in targets:
        watch = Path(t["watch_dir"])
        out = Path(t["output_base"])
        mn = (t.get("model_name") or "").strip()
        if not mn:
            mn = out.name
        derived.append({**t, "model_name": mn})

    counts: Dict[str, int] = {}
    for t in derived:
        counts[t["model_name"]] = counts.get(t["model_name"], 0) + 1

    final = []
    for t in derived:
        mn = t["model_name"]
        if counts.get(mn, 0) > 1:
            key = f"{Path(t['watch_dir']).resolve()}|{Path(t['output_base']).resolve()}"
            suffix = hashlib.md5(key.encode()).hexdigest()[:6]
            mn = f"{mn}_{suffix}"
        final.append({**t, "model_name": mn})
    return final


@dataclass
class TargetContext:
    watch_dir: Path
    output_base: Path
    model_name: str
    task_paths: List[str]
    task_names: List[str]
    suite_id: str
    global_by_task: Dict[str, Dict[str, Path]]
    shared_state_path: Path
    processed: Set[str] = field(default_factory=set)

    def pending_targets(self) -> Set[str]:
        current = list_immediate_subdirs(self.watch_dir)
        print(f"[INFO][{self.model_name}] Current candidate dirs: {sorted(current)}")
        print(f"[INFO][{self.model_name}] Already processed dirs: {sorted(self.processed)}")
        print(f"--------------------------------------------")
        return {d for d in current if d not in self.processed}


def build_context(target: Dict[str, str], task_paths: List[str], task_names: List[str], suite_id: str) -> TargetContext:
    watch_dir = Path(target["watch_dir"])
    output_base = Path(target["output_base"])
    model_name = target["model_name"]

    global_by_task = {tn: get_global_dirs(model_name, tn) for tn in task_names}

    shared_state_path = Path("./results") / model_name / f"STATE.{suite_id}.json"
    print(f"[INFO][{model_name}] Using state file: {shared_state_path}")

    state = load_state(shared_state_path)
    processed: Set[str] = set(state.get("processed", []))

    return TargetContext(
        watch_dir=watch_dir,
        output_base=output_base,
        model_name=model_name,
        task_paths=task_paths,
        task_names=task_names,
        suite_id=suite_id,
        global_by_task=global_by_task,
        shared_state_path=shared_state_path,
        processed=processed,
    )

# -------------------- SCP Helper --------------------

def scp_all_tasks(model_name: str, run_name: str, scp_dest: str):
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

# -------------------- Core Processing --------------------

def process_one(ctx: TargetContext, args, name: str, env: Dict[str, str]):
    src_dir = ctx.watch_dir / name
    if not src_dir.is_dir():
        return

    if not dir_is_stable(src_dir, window_sec=args.stability_window):
        print(f"[INFO][{ctx.model_name}] Skipping '{name}' for now; folder not yet stable.")
        return

    out_dir = ctx.output_base / name
    logs_dir = Path("./logs") / ctx.model_name / name
    export_log = logs_dir / "export.log"
    eval_log = logs_dir / "eval.log"

    is_lora = (args.is_lora.lower() == "true")
    is_mcore = (args.is_mcore.lower() == "true")
    print(f"[INFO][{ctx.model_name}] Processing '{name}': src={src_dir}, out={out_dir}, is_lora={is_lora}, is_mcore={is_mcore}")

    if is_lora:
        merge_exists = False
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[INFO][{ctx.model_name}] Output dir {out_dir} already exists and non-empty; skipping export.")
            merge_exists = True

        if not merge_exists:
            if is_mcore:
                print(f"[INFO][{ctx.model_name}] Detected mcore model; using mcore export parameters.")
                try:
                    match = re.search(r"\d+", name)
                    if not match:
                        raise ValueError(f"Invalid name format: {name}")
                    iteration_num = int(match.group())
                    latest_ckpt_file = ctx.watch_dir / "latest_checkpointed_iteration.txt"
                    with open(latest_ckpt_file, "w") as f:
                        f.write(str(iteration_num))
                    print(f"[INFO][{ctx.model_name}] Updated {latest_ckpt_file} to {iteration_num}")
                except Exception as e:
                    print(f"[ERROR][{ctx.model_name}] Failed to update latest_checkpointed_iteration.txt: {e}")

                export_cmd = (
                    f"CUDA_VISIBLE_DEVICES={args.cuda} "
                    f"{args.swift_bin} export "
                    f"--mcore_adapters {str(ctx.watch_dir)} "
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

            rc = run_and_wait(export_cmd, log_path=export_log)
            if rc != 0:
                print(f"[ERROR][{ctx.model_name}] swift export failed for {name}.")
                return

        model_path = str(out_dir)
    else:
        model_path = str(src_dir)
        print(f"[INFO][{ctx.model_name}] Skip merge. Evaluate directly on {model_path}")

    results_root = Path("./results") / ctx.model_name / name

    need_eval = not have_all_results(results_root, ctx.task_names)
    if need_eval:
        eval_cmd = [
            args.flagevalmm_bin,
            "--tasks", *ctx.task_paths,
            "--exec", args.exec_path,
            "--model", model_path,
            "--num-workers", str(args.num_workers),
            "--output-dir", str(results_root),
            "--backend", args.backend,
            "--extra-args", args.extra_args,
        ]
        rc = run_and_wait(eval_cmd, env=env, log_path=eval_log, shell=False)
        if rc != 0:
            print(f"[ERROR][{ctx.model_name}] flagevalmm failed for {name}.")
            return
    else:
        print(f"[INFO][{ctx.model_name}] All task results already exist for run '{name}', skip eval.")

    # ---- 低分过滤 metrics < 10, 标记为已处理并跳过 ----
    try:
        flat_scores: List[float] = []
        for tn in ctx.task_names:
            results_json_dir = case_insensitive_path(results_root, tn)
            results_json = find_result_json(results_json_dir) if results_json_dir else None
            if results_json is None:
                continue

            data = json.loads(results_json.read_text())
            metrics = extract_metrics(data, tn)
            for v in (metrics or {}).values():
                if isinstance(v, (int, float)):
                    flat_scores.append(float(v))

        if flat_scores and max(flat_scores) < 10:
            print(f"[WARN][{ctx.model_name}] Suspicious low scores (<10) detected for '{name}'. "
                  f"Mark as processed and skip aggregation/plotting.")

            ctx.processed.add(name)
            save_state(ctx.shared_state_path, {"processed": list(ctx.processed)})
            return

    except Exception as e:
        print(f"[WARN][{ctx.model_name}] Low-score gate check failed for '{name}': {e}")

    # ---- 逐 task 解析结果、更新历史、绘图 ----
    for tn in ctx.task_names:
        task_paths_obj = ctx.global_by_task[tn]
        results_json_dir = case_insensitive_path(results_root, tn)
        results_json = find_result_json(results_json_dir) if results_json_dir else None
        if results_json is None:
            print(f"[WARN][{ctx.model_name}] No results JSON found for task '{tn}' under {results_json_dir}")
            continue

        try:
            data = json.loads(results_json.read_text())
            metrics = extract_metrics(data, tn)
            hist = update_history(task_paths_obj["history"], name, metrics)
            # No baselines; plot directly from history
            plot_global_lines(hist, task_paths_obj["png"], task_name=tn)
        except Exception as e:
            print(f"[WARN][{ctx.model_name}] Failed to update history for task '{tn}' from {results_json}: {e}")

    # ---- 全局跨任务趋势图（可按套件或全模型） ----
    try:
        all_base = Path("./results") / ctx.model_name / "_ALL_TASKS"
        if args.all_tasks_scope == "suite":
            hist_by_task = {}
            for tn in ctx.task_names:
                _h = load_history(ctx.global_by_task[tn]["history"])
                hist_by_task[tn] = _h

            all_png = all_base / f"_all_tasks.{ctx.suite_id}.png"
            plot_global_all_tasks(hist_by_task, all_png)

        elif args.all_tasks_scope == "model":
            hist_by_task = discover_all_task_histories(ctx.model_name)
            all_png = all_base / "_all_tasks.ALL.png"
            plot_global_all_tasks(hist_by_task, all_png)

            legacy = all_base / "_all_tasks.png"
            try:
                shutil.copyfile(all_png, legacy)
            except Exception:
                pass

            task_pngs = discover_all_task_summary_pngs(ctx.model_name)
            all_png2 = all_base / "_all_tasks.ALL2.png"
            plot_all_tasks_panels_from_pngs(task_pngs, all_png2, cols=3)

        else:
            pass
    except Exception as e:
        print(f"[WARN][{ctx.model_name}] Failed to draw global all-tasks chart: {e}")

    # ---- 每个 iter 画完图后自动 scp _ALL_TASKS ----
    try:
        scp_all_tasks(ctx.model_name, name, args.scp_dest)
    except Exception as e:
        print(f"[WARN][{ctx.model_name}] SCP step failed unexpectedly for '{name}': {e}")

    ctx.processed.add(name)
    save_state(ctx.shared_state_path, {"processed": list(ctx.processed)})
    print(f"[DONE][{ctx.model_name}] {name} processed successfully.")

# -------------------- Multi-Target Scheduler --------------------

@dataclass
class Candidate:
    ctx: TargetContext
    name: str
    latest_mtime: float

def collect_candidates(contexts: List[TargetContext], stability_window: int) -> List[Candidate]:
    cands: List[Candidate] = []
    for ctx in contexts:
        try:
            pend = ctx.pending_targets()
            for name in pend:
                src = ctx.watch_dir / name
                if not src.is_dir():
                    continue
                if not dir_is_stable(src, window_sec=stability_window):
                    continue
                lm = get_latest_mtime(src)
                cands.append(Candidate(ctx=ctx, name=name, latest_mtime=lm))
        except Exception as e:
            print(f"[WARN][{ctx.model_name}] Failed to collect candidates: {e}")
    # 谁先产生（更早稳定）谁优先 => 按 latest_mtime 升序
    cands.sort(key=lambda c: (c.latest_mtime, ckpt_key(c.name), c.ctx.model_name))
    return cands

# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(description="Babysit LoRA export + evaluation + history plotting (multi-task, multi-target, no baselines)")

    # ---- New multi-target options ----
    parser.add_argument("--targets-config", type=Path, default=None,
                        help="JSON list of targets: [{'watch_dir':..., 'output_base':..., 'model_name': optional}, ...]")
    parser.add_argument("--targets", nargs="+", default=None,
                        help="One or more target specs. Format: watch:output[:model_name] or watch::output[::model_name]")

    # ---- Backward-compatible single target options ----
    parser.add_argument("--watch-dir", type=Path, default=None)
    parser.add_argument("--output-base", type=Path, default=None)

    # ---- Task & runtime options ----
    parser.add_argument("--tasks", nargs="+", required=True,
                        help="One or more task script paths, e.g. tasks/a.py tasks/b.py")
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
    parser.add_argument("--extra-args", default=(""), help="传递给评测脚本的额外参数字符串")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--stability-window", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--state-id", default="",
                        help="可选后缀，用于区分运行中的处理状态. 如果为空，将使用 --tasks 的稳定哈希值")
    parser.add_argument("--all-tasks-scope", choices=["suite", "model", "off"], default="suite",
                        help="全局all-tasks画图: suite -> _all_tasks.<suite>.png; model -> _all_tasks.ALL.png; off -> disable.")
    parser.add_argument("--scp-dest", default="",
                        help="可选：scp 目标，形如 ubt@10.10.22.168:/path/to/dir/；为空则不自动传输")

    args = parser.parse_args()

    # ---- Build task lists ----
    task_paths: List[str] = list(dict.fromkeys(args.tasks))
    task_names: List[str] = [Path(t).name.split(".")[0].replace("_qwen3vl", "") for t in task_paths]

    suite_id = (args.state_id.strip() or _compute_suite_id(task_paths)).replace(os.sep, "_")

    # ---- Collect targets ----
    targets: List[Dict[str, str]] = []

    if args.targets_config is not None:
        cfg_targets = load_targets_from_config(args.targets_config)
        targets.extend(cfg_targets)

    if args.targets:
        for s in args.targets:
            watch, out, mname = _parse_target_string(s)
            targets.append({
                "watch_dir": str(watch),
                "output_base": str(out),
                "model_name": str(mname) if mname else "",
            })

    if not targets:
        # fallback to legacy single target
        if args.watch_dir is None or args.output_base is None:
            print("[ERROR] Must provide either --targets-config, --targets, or both --watch-dir and --output-base.")
            sys.exit(2)
        targets.append({
            "watch_dir": str(args.watch_dir),
            "output_base": str(args.output_base),
            "model_name": "",  # will derive
        })

    targets = derive_unique_model_names(targets)

    # ---- Build contexts ----
    contexts: List[TargetContext] = []
    for t in targets:
        ctx = build_context(t, task_paths, task_names, suite_id)
        contexts.append(ctx)

    # ---- Shared env ----
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    # ---- Scheduler loop ----
    while True:
        candidates = collect_candidates(contexts, args.stability_window)

        if candidates:
            print(f"[SCHED] Found {len(candidates)} stable pending candidates across {len(contexts)} targets.")
        else:
            print(f"[SCHED] No stable pending candidates across {len(contexts)} targets.")

        for cand in candidates:
            try:
                process_one(cand.ctx, args, cand.name, env=env)
            except Exception as e:
                print(f"[EXCEPTION][{cand.ctx.model_name}] While processing '{cand.name}': {e}")

        if args.once:
            break

        time.sleep(args.poll_interval)

if __name__ == "__main__":
    main()
