import re
from typing import Dict, Tuple, Any, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from flagevalmm.registry import EVALUATORS
from flagevalmm.evaluator import BaseEvaluator
from flagevalmm.models import HttpClient
from flagevalmm.models.api_response import ApiResponse
from flagevalmm.common.logger import get_logger
from flagevalmm.evaluator.pre_process import strip_special_tags_for_predictions

logger = get_logger(__name__)

GRADER_TEMPLATE = """
You are an AI assistant tasked with evaluating whether a response matches the correct answer to a given question.

Evaluation Rules
(1) Output 1 if the response matches the answer exactly or with synonymous/equivalent wording.
- Synonyms, paraphrases, or different surface forms of the same meaning count as matches.
- Minor wording differences (e.g., “put tomato into fridge” vs. “the person is putting a tomato in the fridge”) count as matches.

(2) Output 0 if the response is incorrect, contradictory, or refers to a different entity, object, or attribute.
- If the answer and response describe different objects, actions, or states, mark as 0.
- If the response introduces additional details that change the meaning of the answer, mark as 0.

Special Cases
- Similar meaning: Output 1 if the response conveys essentially the same meaning as the answer and does not omit or add critical information (e.g., answer:“put meat on the table”, response:“The person moved meat from the fridge to the counter.”).
- Partial matches: If the response overlaps but misses or alters essential details (e.g., answer:“put meat and tomato on the table” vs. response:“put meat on the table”), output 0.
- Granularity differences: If the response is more specific but still semantically equivalent (e.g., answer:“woman”, response:“Jessica”), output 1.
- Yes/No questions: Only output 1 if the polarity matches (yes <-> yes, no <-> no). Any mismatch outputs 0, regardless of explanation.
- Ambiguity: If the response cannot be reasonably interpreted as equivalent to the answer, output 0.

Examples
Example 1 
Question: Did the attribute of plant changed because of the action getting something from something? 
Answer: yes 
Response: Yes, the attribute of plant got watered from no to yes after the action getting something from something. 
Your output: 1

Example 2 
Question: what status of fork changed while the person do the first action did before he/she put something to something? 
Answer: cleanliness 
Response: fork was in drawer before the person put fork to sink. 
Your output: 0

Example 3 
Question: What is the person doing before he/she close something? 
Answer: Put tomato to fridge 
Response: The person is putting a tomato in the fridge. 
Your output: 1

Example 4 
Question: What is the first action the person did in the video? 
Answer: Work on sofa 
Response: The person pulled out a chair. 
Your output: 0

Example 5 
Question: How did the person changed the spatial relationships of meat? 
Answer: Put meat to table 
Response: The person moved meat from the fridge to the counter. 
Your output: 1

Example 6 
Question: what status of fridge changed while the person do the first action did after he/she point to something? 
Answer: openess 
Response: The fridge was closed before the person point to something, and after that the fridge changed to open. 
Your output: 1

Example 7 
Question: which object changed its status when the person do the last action in the video? 
Answer: fork 
Response: spoon 
Your output: 0

Example 8 
Question: What is the action that just happened? 
Answer: Place can in the tray 
Response: The person puts the can on the table. 
Your output: 0

Example 9 
Question: current goal is: Please place the fruits in the bowl then place the kitchen supplies into the holder. last 20 steps: 1. put white packet in the bowl 2. put white packet in the bowl 3. put yellow packet in the bowl 4. put blue packet in the bowl 5. put blue packet in the bowl 6. put blue packet in the bowl 7. put yellow packet in the bowl. What’s the immediate next step? 
Answer: Put duster in the black stand 
Response: put brush in the holder 
Your output: 0

Your Turn: 
Question: {question} 
Answer: {answer} 
Response: {prediction}
Your output:
""".strip()


@EVALUATORS.register_module()
class RoboVQAGPTEvaluator(BaseEvaluator):
    """RoboVQA: GPT judge -> output 0/1, then compute accuracy."""

    def __init__(
        self,
        eval_model_name: str,
        use_llm_evaluator: bool = True,
        num_threads: int = 8,
        pred_keys=("answer", "response", "raw_answer", "prediction", "pred"),
        **kwargs,
    ) -> None:
        super().__init__(use_llm_evaluator=False, eval_func=None, **kwargs)

        assert use_llm_evaluator, "use_llm_evaluator must be True"
        self.model_name = eval_model_name
        self.num_threads = num_threads
        self.pred_keys = pred_keys

        self.base_url = kwargs.pop("base_url", "http://localhost:8000/v1/chat/completions")
        self.api_key = kwargs.pop("api_key", None)

        self.llm_evaluator = HttpClient(  # type: ignore
            model_name=self.model_name,
            url=self.base_url,
            api_key=self.api_key,
        )

    def _get_pred_text(self, pred: Dict[str, Any]) -> str:
        for k in self.pred_keys:
            v = pred.get(k, None)
            if v is not None and str(v).strip() != "":
                return str(v)
        return ""

    def _parse_01(self, s: str) -> int:
        # 允许模型偶尔输出 "Output: 1" / "1\n" 之类
        m = re.search(r"\b([01])\b", s)
        return int(m.group(1)) if m else 0

    def _grade_one(self, gt: Dict[str, Any], pred: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        pred_result = pred.copy()

        # label json 里 gt 是 gt_answer
        gt_answer = gt.get("gt_answer", "")
        pred_result["label"] = gt_answer

        predicted_text = self._get_pred_text(pred_result)
        predicted_text = strip_special_tags_for_predictions(predicted_text)

        prompt = GRADER_TEMPLATE.format(
            question=gt.get("question", ""),
            answer=gt_answer,
            response=predicted_text,
        )

        try:
            msg = self.llm_evaluator.build_message(query=prompt)
            resp = self.llm_evaluator.infer(chat_messages=msg, temperature=0, top_p=1, seed=42)
            assert isinstance(resp, ApiResponse), f"response is not an ApiResponse: {resp}"

            raw = resp.content.strip()
            score = self._parse_01(raw)

            pred_result["judge_raw"] = raw
            pred_result["correct"] = score
            pred_result["eval_method"] = "gpt_judge_01"
            pred_result["used_pred_text_key_order"] = list(self.pred_keys)
            return pred_result, score

        except Exception as e:
            logger.error(f"Error in RoboVQA grading: {e}")
            pred_result["judge_raw"] = "[FAILED]"
            pred_result["correct"] = 0
            pred_result["eval_method"] = "gpt_judge_01"
            return pred_result, 0

    def cal_accuracy(self, annotations: Dict, predictions: list, *args, **kwargs) -> Dict:
        right = 0
        processed = []

        with ThreadPoolExecutor(max_workers=self.num_threads) as ex:
            futures = {
                ex.submit(self._grade_one, annotations[str(p["question_id"])], p): p
                for p in predictions
            }
            for fut in as_completed(futures):
                pred_result, ok = fut.result()
                processed.append(pred_result)
                right += ok

        predictions.clear()
        predictions.extend(processed)
        return {"accuracy": round(right / len(predictions) * 100, 2)}