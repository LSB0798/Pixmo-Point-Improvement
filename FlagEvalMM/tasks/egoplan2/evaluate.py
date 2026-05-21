from typing import Dict, List, Tuple
from PIL import Image, ImageDraw
import numpy as np
import re
import string
from rouge_score import rouge_scorer
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os.path as osp
import os

def strip_answer(answer):
    answer = re.sub("The", "", answer)
    answer = re.sub("If", "", answer)
    answer = re.sub("[INST]", "", answer)
    answer = re.sub("[/INST]", "", answer)
    answer = re.sub("<Img>", "", answer)
    answer = re.sub("</Img>", "", answer)
    answer = re.sub("<RichMediaReference>", "", answer)
    answer = re.sub("</RichMediaReference>", "", answer)
    answer = re.sub("<richmediareference>", "", answer)
    answer = re.sub("</richmediareference>", "", answer)
    answer = re.sub("<Answer>", "", answer)
    answer = re.sub("</Answer>", "", answer)
    answer = re.sub("<answer>", "", answer)
    answer = re.sub("</answer>", "", answer)
    answer = answer.strip()
    return answer

def extract_option(answer: str):
    """提取回答中的第一个大写字母选项"""
    answer = strip_answer(answer)
    answer = answer.split("\n")[-1]
    answer = answer.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
    ]
    for answer_prefix in answer_prefixes:
        answer = answer.replace(answer_prefix, "")

    matches = re.search(r'[ABCD]', answer)
    if matches is None:
        return None #无法提取选项
    return matches[0]


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    """
    评测 MCQ（多项选择）准确率，按 domain / video_type 聚合，并给出 overall。
    - annotations: Dict[str(question_id)] -> {
          "gt_answer": "A"/"B"/... 或其他可解析文本,
          "domain": "xxx",
          "video_type": "short"/"long"/...
      }
    - predictions: List[{
          "question_id": ...,
          "answer": "模型输出文本（可含 <answer> 标签、解释等）"
      }]

    返回：
    {
      "per_domain": {
        dom: {"correct": int, "total": int, "skipped": int, "accuracy": float or None}, ...
      },
      "per_video_type": {
        vt:  {"correct": int, "total": int, "skipped": int, "accuracy": float or None}, ...
      },
      "overall": {"correct": int, "total": int, "skipped": int, "accuracy": float}
    }
    """
    per_domain = defaultdict(lambda: {"correct": 0, "total": 0, "skipped": 0, "accuracy": 0.0})
    per_vtype  = defaultdict(lambda: {"correct": 0, "total": 0, "skipped": 0, "accuracy": 0.0})

    total_correct = 0
    total_total   = 0
    total_skipped = 0

    for pred in predictions:
        qid = str(pred["question_id"])
        if qid not in annotations:
            # 若某些预测没有标注，可选择跳过
            total_skipped += 1
            continue

        ann = annotations[qid]
        dom = ann.get("domain", "Unknown")
        vt  = ann.get("video_type", "Unknown")

        # 解析 gt/pred 选项（均返回 A/B/C...）
        gt_raw = ann.get("gt_answer")
        gt_opt   = extract_option(gt_raw)

        pred_raw = pred.get("answer", "")
        pred_opt = extract_option(pred_raw)

        # 对不可解析的预测，计入 skipped
        if pred_opt is None or gt_opt is None:
            per_domain[dom]["skipped"] += 1
            per_vtype[vt]["skipped"]   += 1
            total_skipped += 1
            # 不中断，继续下一个样本
            continue

        # 可评样本总数
        per_domain[dom]["total"] += 1
        per_vtype[vt]["total"]   += 1
        total_total += 1

        if pred_opt == gt_opt:
            per_domain[dom]["correct"] += 1
            per_vtype[vt]["correct"]   += 1
            total_correct += 1

    # 聚合准确率
    def _finalize(acc_dict: Dict[str, Dict]):
        for k, s in acc_dict.items():
            if s["total"] > 0:
                s["accuracy"] = s["correct"] / s["total"]
            else:
                s["accuracy"] = None

    _finalize(per_domain)
    _finalize(per_vtype)

    overall_acc = (total_correct / total_total) if total_total > 0 else 0.0
    overall_acc = round(overall_acc * 100, 3)  # 转为百分比
    overall = {
        "correct": total_correct,
        "total": total_total,
        "skipped": total_skipped,
        "accuracy": overall_acc
    }

    result = {
        "per_domain": dict(per_domain),
        "per_video_type": dict(per_vtype),
        "overall": overall
    }

    return result
