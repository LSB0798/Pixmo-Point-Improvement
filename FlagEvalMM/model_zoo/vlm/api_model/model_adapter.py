import re
import os
import os.path as osp
import json
from typing import Dict, Any, Optional, Union, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import atexit
import signal
import time
import copy
from importlib.metadata import version, PackageNotFoundError

from flagevalmm.server import ServerDataset
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.models import HttpClient, Claude, Gemini, GPT, Hunyuan, HttpClient_RoboBrain
from flagevalmm.models.api_response import ApiResponse, ProcessResult
from flagevalmm.server.model_server import ModelServer
from flagevalmm.server.utils import get_random_port
from flagevalmm.common.logger import get_logger
from flagevalmm.server.utils import parse_args

logger = get_logger(__name__)


class ModelAdapter(BaseModelAdapter):
    def __init__(
        self,
        server_ip: str,
        server_port: int,
        timeout: int,
        model_type: Optional[str] = None,
        extra_cfg: Optional[Union[str, Dict]] = None,
        local_mode: bool = False,
        task_names: List[str] = None,
        **kwargs,
    ):
        self.model_type = model_type
        super().__init__(
            server_ip=server_ip,
            server_port=server_port,
            timeout=timeout,
            extra_cfg=extra_cfg,
            local_mode=local_mode,
            task_names=task_names,
            **kwargs,
        )

        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print('-0-' * 10)
        logger.info(f"Received signal {signum}, cleaning up...")
        self.cleanup()
        os._exit(0)

    def model_init(self, task_info: Dict):
        print('-1-' * 10)
        if task_info.get("backend", None):
            self.model_server = self.launch_model(task_info)

        model_config_keys = [
            "model_name",
            "url",
            "base_url",
            "api_key",
            "use_cache",
            "max_image_size",
            "min_short_side",
            "max_long_side",
            "max_tokens",
            "temperature",
            "chat_name",
            "max_num_frames",
            "stream",
            "system_prompt",
            "num_infers",
            "reasoning",
            "thinking",
            "provider",
            "retry_time",
        ]
        print(f"task_info: {task_info}")
        model_config = {k: task_info[k] for k in model_config_keys if k in task_info}
        print(f"model_config: {model_config}")

        model_type_map = {
            "http": HttpClient,
            "http_robobrain": HttpClient_RoboBrain,
            "claude": Claude,
            "gemini": Gemini,
            "gpt": GPT,
            "hunyuan": Hunyuan,
        }
        model_type = self.model_type or task_info.get("model_type", "http")
        self.model = model_type_map[model_type](**model_config)

    def launch_model(self, task_info: Dict):
        print('-2-' * 10)
        if task_info.get("server_port"):
            port = task_info.get("server_port")
        else:
            port = get_random_port()
        # replace port in url
        url = re.sub(
            r":(\d+)/",
            f":{port}/",
            task_info.get("url", "http://localhost:8000/v1/chat/completions"),
        )
        task_info["url"] = url

        model_name = task_info["model_name"]
        backend = task_info.get("backend", "vllm")
        model_server = ModelServer(
            model_name,
            port=port,
            backend=backend,
            extra_args=task_info.get("extra_args", None),
        )
        task_info["execute_cmd"] = model_server.execute_cmd
        important_packages = [backend, "transformers", "torch"]
        task_info["important_packages"] = []
        for package in important_packages:
            try:
                version_pkg = version(package)
                task_info["important_packages"].append(f"{package}=={version_pkg}")
            except PackageNotFoundError:
                task_info["important_packages"].append(f"{package} not installed")
        return model_server

    def _process_single_result(self, single_result: ApiResponse) -> Dict[str, Any]:
        print('-3-' * 10)
        """
        Process a single inference result and extract content, reason, and usage.

        Args:
            single_result: Single inference result (string or ApiResponse)

        Returns:
            Dictionary containing processed content, reason, and usage
        """
        usage_info = None
        reason = ""

        # Extract content and usage from ApiResponse
        content = single_result.content
        if single_result.usage:
            usage_info = single_result.usage.to_dict()

        # Split reasoning and answer if present
        if "</think>" in content:
            reason, answer = content.split("</think>", 1)
            reason += "</think>"
        else:
            answer = content

        return {"answer": answer, "reason": reason, "usage": usage_info}

    def process_single_item(self, i, inter_results_dir, double_q: bool = False):
        print('-4-' * 10)
        question_id, multi_modal_data, qs, system_prompt = self.dataset[i]
        inter_results_file = osp.join(inter_results_dir, f"{question_id}.json")
        if osp.exists(inter_results_file):
            logger.info(f"Skipping {question_id} because it already exists")
            with open(inter_results_file, "r") as f:
                data = json.load(f)
                reason = data.get("reason", "")
                result = data.get("answer", "")
                usage_info = data.get("usage", None)
                return ProcessResult(
                    question_id=question_id,
                    question=qs,
                    answer=result,
                    reason=reason,
                    usage=usage_info,
                )
        logger.info(f"Processing {question_id}")
        # logger.info(qs)
        if double_q:
            qs = qs + qs
            print(f"Double question for {question_id}: {qs}")
        if system_prompt:
            messages = self.model.build_message(qs, system_prompt=system_prompt, multi_modal_data=multi_modal_data)
        else:
            messages = self.model.build_message(qs, multi_modal_data=multi_modal_data)
        messages_info = copy.deepcopy(messages)
        messages_info = messages_info[-1]['content']
        for item in messages_info:
            if isinstance(item, dict):
                item.pop("image_url", None)
        logger.info(f"Built messages: {messages_info}")
        # is_fixed = False
        # is_*** = False
        images_path, videos_path, all_path = [], [], []
        if isinstance(multi_modal_data, dict):
            for k in ("image", "images"):
                if k in multi_modal_data and multi_modal_data[k]:
                    v = multi_modal_data[k]
                    images_path.extend(v if isinstance(v, list) else [v])
            for k in ("video", "videos"):
                if k in multi_modal_data and multi_modal_data[k]:
                    v = multi_modal_data[k]
                    videos_path.extend(v if isinstance(v, list) else [v])
        all_path = images_path + videos_path
        
        is_fixed = False
        is_vsi = False
        is_erqa = False
        is_refspatial = False
        is_where2place = False
        is_robo_spatial_home = False

        for path in all_path:
            # 转成字符串以防 path 是 Path 对象之类的
            if "vsi" in str(path).lower():
                is_vsi = True
                break
            if "erqa" in str(path).lower():
                is_erqa = True
                break
            if "refspatial" in str(path).lower():
                is_refspatial = True
                break
            if "where2place" in str(path).lower():
                is_where2place = True
                break
            if "robospatial" in str(path).lower():
                is_robo_spatial_home = True
                break

        try:
            if is_vsi:
                result = self.model.infer(messages, temperature=0.0, max_tokens=self.task_info.get("max_tokens", 2048), top_p=1.0, top_k=-1, seed=3407, do_sample=False)
                print('Fix vsi dataset inference randomness!')
            elif is_erqa:
            #     print('11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
                print('Fix erqa dataset inference repeating!')
                result = self.model.infer(messages, temperature=0.6, max_tokens=self.task_info.get("max_tokens", 2048), top_p = 0.9, presence_penalty=0.3, frequency_penalty=0.8, repetition_penalty=1.1)
            #elif is_refspatial:
            #    print('11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
            #    print('Fix refspatial dataset inference randomness!')
            #    result = self.model.infer(messages, temperature=0.0, max_tokens=2048, top_p = 1.0, top_k = -1, seed=3407, do_sample=False)
            #elif is_where2place:
            #    print('11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
            #    print('Fix where2palce dataset inference randomness!')
            #    result = self.model.infer(messages, temperature=0.0, max_tokens=2048, top_p = 1.0, top_k = -1, seed=3407, do_sample=False)
            elif is_robo_spatial_home:
                print('Fix robo_spatial_home dataset inference randomness!')
                result = self.model.infer(messages, temperature=0.0, max_tokens=self.task_info.get("max_tokens", 2048), top_p = 1.0, top_k = -1, seed=3407, do_sample=True)
            else:
                print('RUN THIS WAY')
                result = self.model.infer(messages, max_tokens=self.task_info.get("max_tokens", 2048))
                # result = self.model.infer(messages, temperature=0.8, top_p=0.9, top_k=50, max_tokens=self.task_info.get("max_tokens", 2048))
            # result = self.model.infer(messages, extra_body={"mm_processor_kwargs": {"fps": 2, "min_frames": 16, "max_frames": 256}})
            # result = self.model.infer(messages, extra_body={"mm_processor_kwargs": {"nframes": 256}})
            # import pdb; pdb.set_trace()

            if isinstance(result, list):
                # Multiple inferences case
                inference_answers = {}
                reasons = []
                usages = []

                for i, single_result in enumerate(result):
                    processed = self._process_single_result(single_result)
                    inference_answers[f"inference_{i}"] = processed["answer"]
                    reasons.append(processed["reason"])
                    if processed["usage"]:
                        usages.append(processed["usage"])

                final_result = inference_answers
                final_reason = reasons  # Store all reasons as list
                final_usage = usages if usages else None  # Store all usages as list

                logger.info(
                    f"Multiple inferences completed. Got {len(inference_answers)} results."
                )
            else:
                # Single inference case
                processed = self._process_single_result(result)
                final_result = processed["answer"]
                final_reason = processed["reason"]
                final_usage = processed["usage"]

            if "!!!!!!!!!!!!!!!!!" in str(final_result) or "！！！！！！！！！！！！" in str(final_result):
                # 出现错误标记，终止这个iter的评测
                raise RuntimeError("AbortIter")

        except Exception as e:
            if str(e) == "AbortIter":
                raise  # 继续抛给外层 run_one_task
            final_result = "Error code " + str(e)
            final_reason = ""
            final_usage = None

        # Create ProcessResult object
        process_result = ProcessResult(
            question_id=question_id,
            question=messages[-1]["content"],
            answer=final_result,
            reason=final_reason,
            usage=final_usage,
        )

        return process_result

    def cleanup(self):
        print('-5-' * 10)
        if hasattr(self, "model_server") and self.model_server is not None:
            try:
                self.model_server.stop()
                self.model_server = None
            except Exception as e:
                logger.error(f"Error shutting down model server: {e}")

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        print('-6-' * 10)
        self.dataset = ServerDataset(
            task_name,
            task_type=meta_info["type"],
            task_manager=self.task_manager,
        )

        results = []
        num_workers = self.task_info.get("num_workers", 8)
        print(f"Using {num_workers} workers for task {task_name}")
        inter_results_dir = osp.join(meta_info["output_dir"], "items")
        os.makedirs(inter_results_dir, exist_ok=True)
        double_q = self.task_info.get("double_q", False)

        executor = ThreadPoolExecutor(max_workers=num_workers)
        future_to_item = {
            executor.submit(self.process_single_item, i, inter_results_dir, double_q): i
            for i in range(len(self.dataset))
        }

        try:
            for future in as_completed(future_to_item):
                result = future.result()
                results.append(result)
                if isinstance(result.answer, str) and result.answer.startswith("Error code"):
                    continue
                else:
                    self.save_item(result, result.question_id, meta_info)

        except RuntimeError as e:
            if str(e) == "AbortIter":
                logger.info(f"[ABORT] Detect abnormal response, stop eval for task {task_name}")

                # 1. 尝试取消所有还没开始/排队的任务
                for f in future_to_item.keys():
                    f.cancel()

                # 2. 立即关闭线程池，不等待未完成任务
                executor.shutdown(wait=False, cancel_futures=True)

                raise  # 往上传，让 BaseModelAdapter.run() 中断后续 tasks

        finally:
            # 正常情况/其他异常时，还是要正常关掉线程池
            executor.shutdown(wait=True)

        self.save_result(results, meta_info)
        # # 主动关闭 vLLM & 日志线程，避免悬挂
        # time.sleep(2)
        # self.cleanup()
        # # 明确退出子进程，避免被后台非守护线程牵制
        # import sys
        # sys.exit(0)


if __name__ == "__main__":
    print('-7-' * 10)
    args = parse_args()
    model_adapter = ModelAdapter(
        server_ip=args.server_ip,
        server_port=args.server_port,
        timeout=args.timeout,
        model_type=args.model_type,
        extra_cfg=args.cfg,
        local_mode=args.local_mode,
        task_names=args.tasks,
        output_dir=args.output_dir,
        model_path=args.model,
        debug=args.debug,
        quiet=args.quiet,
    )
    model_adapter.run()
