from typing import Dict, List, Optional
from collections import defaultdict

from flagevalmm.evaluator.pre_process import normalize_string


def _parse_yesno(text: str) -> Optional[str]:
    """
    Parse model output into 'yes'/'no'.
    Return None if cannot parse.
    """
    if text is None:
        return None

    # follow your convention: answer is in the last line
    s = text.strip()
    if not s:
        return None
    s = s.split("\n")[-1].strip()
    s = normalize_string(s)
    s = s.strip().lower()

    # common wrappers
    if s.startswith("answer:"):
        s = s[len("answer:"):].strip()

    # strict yes/no first
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"

    # optional tolerant mapping (kept conservative)
    if s in {"true", "correct"}:
        return "yes"
    if s in {"false", "incorrect"}:
        return "no"
    if s == "1":
        return "yes"
    if s == "0":
        return "no"

    return None


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    """
    annotations: dict[question_id] -> gt dict (must include 'answer', 'category')
    predictions: list of dict (must include 'question_id', 'answer')
    """
    results = defaultdict(lambda: {"num_correct": 0, "total": 0, "illformed_responses": 0})
    num_correct = 0

    for pred in predictions:
        question_id = str(pred["question_id"])
        gt = annotations[question_id]

        pred["raw_answer"] = pred.get("answer", "")
        gt_ans = (gt.get("answer", "") or "").strip().lower()
        category = gt.get("category", "default")

        parsed = _parse_yesno(pred["raw_answer"])
        is_parsable = parsed in {"yes", "no"}

        results[category]["total"] += 1
        if not is_parsable:
            results[category]["illformed_responses"] += 1
            correct = False
        else:
            correct = (parsed == gt_ans)

        if correct:
            num_correct += 1
            results[category]["num_correct"] += 1

        # align with your usual prediction record format
        pred["label"] = gt.get("answer", "")
        pred["answer"] = normalize_string(pred["raw_answer"].strip().split("\n")[-1]) if pred["raw_answer"] else ""
        pred["correct"] = bool(correct)

    # compute per-category accuracy
    for _, r in results.items():
        r["accuracy"] = round((r["num_correct"] / r["total"]) * 100, 4) if r["total"] > 0 else 0.0

    final_results = {
        "accuracy": round((num_correct / len(predictions)) * 100, 4) if predictions else 0.0,
        "results": results,
    }
    return final_results
