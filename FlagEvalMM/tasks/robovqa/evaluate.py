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

def extract_answer(text: str) -> str:
    """
    从 <answer>...</answer> 标签中提取内容。
    如果不存在标签，则返回原始字符串。
    """
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def _normalize_text_for_robovqa(s: str) -> str:
    s = extract_answer(s)
    # 与 robovqa_process_results 一致：去换行、小写、规范 yes/no
    s = extract_answer(s)
    # 去换行、首尾空格、小写
    s = s.replace("\n", " ").strip().lower()
    # 折叠多余空格
    s = re.sub(r"\s+", " ", s)
    # 去常见句末标点
    s = s.strip(string.punctuation + " ")
    # 严格匹配独立的 'yes' / 'no'（不再误伤 yesterday）
    if re.search(r"\byes\b", s):
        return "yes"
    if re.search(r"\bno\b", s):
        return "no"
    return s

def get_bleu_score(prediction, target):
    bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
    candidate = list(prediction.split(" "))
    reference = [list(target.split(" "))]
    if target is not None:
        if len(reference[0]) <= 1:
            bleu1 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
            bleu2 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
            bleu3 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
            bleu4 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
        elif len(reference[0]) == 2:
            bleu1 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
            bleu2 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00))
            bleu3 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00))
            bleu4 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00))
        elif len(reference[0]) == 3:
            bleu1 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
            bleu2 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00))
            bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0.00))
            bleu4 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0.00))
        else:
            bleu1 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
            bleu2 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00))
            bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0.00))
            bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
    score = (bleu1 + bleu2 + bleu3 + bleu4) / 4
    return score, bleu1, bleu2, bleu3, bleu4



def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    per_cat_raw = defaultdict(lambda: {
        "bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0,
        "rougeL": 0, "cnt": 0
    })

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # ---------- 单样本累积 ----------
    for pred in predictions:
        qid = str(pred["question_id"])
        # if qid not in annotations:
        #     continue
        gt_info = annotations[qid]
        gt   = _normalize_text_for_robovqa(gt_info["gt_answer"])
        pred_text = _normalize_text_for_robovqa(pred["answer"])
        print(f"GT: {gt}\nPRED: {pred_text}\n---")
        cat  = gt_info.get("category", "default")

        score, b1, b2, b3, b4 = get_bleu_score(pred_text, gt)
        rL = rouge.score(gt, pred_text)['rougeL'].fmeasure

        per_cat_raw[cat]["bleu1"] += b1
        per_cat_raw[cat]["bleu2"] += b2
        per_cat_raw[cat]["bleu3"] += b3
        per_cat_raw[cat]["bleu4"] += b4
        per_cat_raw[cat]["rougeL"] += rL
        per_cat_raw[cat]["cnt"] += 1

    # ---------- 汇总 per-category ----------
    per_category = {}
    total = {"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"rougeL":0,"cnt":0}

    for cat, r in per_cat_raw.items():
        if r["cnt"] == 0: 
            continue
        bleu1 = r["bleu1"]/r["cnt"]; bleu2 = r["bleu2"]/r["cnt"]
        bleu3 = r["bleu3"]/r["cnt"]; bleu4 = r["bleu4"]/r["cnt"]
        rougeL= r["rougeL"]/r["cnt"]

        per_category[cat] = {
            "BLEU-1": bleu1, "BLEU-2": bleu2,
            "BLEU-3": bleu3, "BLEU-4": bleu4,
            "ROUGE-L": rougeL,
            "cnt": r["cnt"]
        }

        for k in ("bleu1","bleu2","bleu3","bleu4","rougeL","cnt"):
            total[k] += r[k]

    # ---------- 汇总 overall ----------
    overall = {}
    if total["cnt"]:
        overall = {
            "BLEU-1": total["bleu1"]/total["cnt"],
            "BLEU-2": total["bleu2"]/total["cnt"],
            "BLEU-3": total["bleu3"]/total["cnt"],
            "BLEU-4": total["bleu4"]/total["cnt"],
            "ROUGE-L": total["rougeL"]/total["cnt"]
        }
        overall["BLEU-avg"] = (
            overall["BLEU-1"] + overall["BLEU-2"] +
            overall["BLEU-3"] + overall["BLEU-4"]
        ) / 4.0
        overall["BLEU-avg"] = round(overall["BLEU-avg"] * 100, 3)
    results = {
        "per_category": per_category,
        "overall": overall
    }
    return results

