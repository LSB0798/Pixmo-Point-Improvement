import torch
from typing import Dict, Any
import time
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from flagevalmm.server import ServerDataset
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.server.utils import parse_args, process_images_symbol
from qwen_vl_utils import process_vision_info
import os
from typing import List, Optional, Tuple
from accelerate import Accelerator, infer_auto_device_map, load_checkpoint_and_dispatch
import decord

MIN_PIXELS_IMAGE = 256 * 32 * 32
MAX_PIXELS_IMAGE = 512 * 32 * 32
MAX_FRAMES_VIDEO = 8
MIN_PIXELS_VIDEO = 256 * 32 * 32
MAX_PIXELS_VIDEO = 512 * 32 * 32

def _is_image(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in {
        ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"
    }

def _is_video(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in {
        ".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".m4v"
    }

def collate_keep_list(batch):
    # batch: List[(question_id, img_path_idx, qs)]
    qids, img_paths, qss = zip(*batch)
    return list(qids), list(img_paths), list(qss)

class CustomDataset(ServerDataset):
    def __getitem__(self, index):
        data = self.get_data(index)
        question_id = data["question_id"]
        if self.task_type == "video_qa":
            img_path = data["video_path"]
        else:
            img_path = data["img_path"]
        qs = data["question"]
        qs, idx = process_images_symbol(qs)
        qs = qs.strip()
        return question_id, img_path, qs


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict):
        ckpt_path = task_info["model_path"]
        torch.set_grad_enabled(False)
        
        # 以免 prepare_model 搬回单卡
        self.accelerator = Accelerator(device_placement=False)

        # max_memory = {0: "10GB", 1: "25GB", 2: "25GB", 3: "25GB", 4: "25GB", 5: "25GB", 6: "25GB", 7: "25GB", "cpu": "1024GiB"}

        with self.accelerator.main_process_first():
            tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
            if "qwen2-" in ckpt_path.lower():
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    ckpt_path,
                    device_map="auto",
                    dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                )
            else:
                model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                    ckpt_path,
                    device_map="auto",
                    dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",

                )

        model = self.accelerator.prepare_model(model, evaluation_mode=True)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if hasattr(model, "module"):
            model = model.module
        self.model = model
        self.processor = AutoProcessor.from_pretrained(
            ckpt_path, min_pixels=MIN_PIXELS_IMAGE, max_pixels=MAX_PIXELS_IMAGE, use_fast=True, 
        )

        # 关键：左填充 + pad_token
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token  # 常见做法：用 eos 作为 pad
        model.config.pad_token_id = tok.pad_token_id
        self.processor.tokenizer = tok  # 重新挂回 processor

    def build_message(
        self,
        query: str,
        image_paths=[],
    ) -> str:
        """
        规则：
        1) 只有图像或只有视频：先添加对应媒体，再添加文本。
        2) 如果既有视频又有图像，且列表最后一个是图像（作为 ending frame）：
            顺序：video -> image -> text
            文本内容：According to video <video> and ending frame <image>. <原始文本>
        3) 其它混合情况：按输入顺序把媒体逐个加入，然后再添加原始文本。
        """
        messages = []
        messages.append(
            {
                "role": "user",
                "content": [],
            },
        )
        # 分类
        images = [p for p in image_paths if _is_image(p)]
        videos = [p for p in image_paths if _is_video(p)]

        has_images = len(images) > 0
        has_videos = len(videos) > 0
        last_is_image = (len(image_paths) > 0 and _is_image(image_paths[-1]))

        # 情况 1：只有图像或只有视频 → 先媒体后文本
        if has_images and not has_videos:
            for p in images:
                messages[-1]["content"].append({"type": "image", "image": p, "max_pixels": MAX_PIXELS_IMAGE})
                # messages[-1]["content"].append({"type": "image", "image": p})
            messages[-1]["content"].append({"type": "text", "text": query})
            return messages

        if has_videos and not has_images:
            for p in videos:
                messages[-1]["content"].append({"type": "video", "video": p, "max_frames": MAX_FRAMES_VIDEO, "max_pixels": MAX_PIXELS_VIDEO})
            messages[-1]["content"].append({"type": "text", "text": query})
            return messages

        # 情况 2：特殊的“视频 + 最后一张图像(ending frame)”
        if has_images and has_videos and last_is_image:
            last_image  = next(p for p in reversed(image_paths) if _is_image(p))

            messages[-1]["content"].append({"type": "text", "text": "According to video "})
            for p in videos:
                messages[-1]["content"].append({"type": "video", "video": p, "max_frames": MAX_FRAMES_VIDEO, "max_pixels": MAX_PIXELS_VIDEO})
            messages[-1]["content"].append({"type": "text", "text": " and ending frame "})
            messages[-1]["content"].append({"type": "image", "image": last_image})
            messages[-1]["content"].append({"type": "text", "text": ". " + query})
            return messages
        
        # 情况 3：其它混合（既有图像又有视频，但不符合“最后一张图像”为 ending frame 的特判）
        # 按输入顺序依次加入媒体，再加原始文本
        for p in image_paths:
            if _is_image(p):
                messages[-1]["content"].append({"type": "image", "image": p})
            elif _is_video(p):
                messages[-1]["content"].append({"type": "video", "video": p})
            else:
                # 忽略未知类型，或根据需要 raise / log
                pass
        messages[-1]["content"].append({"type": "text", "text": query})
        return messages


    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        results = []
        cnt = 0
        batch_size = 16
        if task_name in ["vsi_bench_tiny", "robovqa", "egoplan2"]:
            batch_size = 8
        print(f"Running task {task_name} of type {meta_info['type']} with batch size {batch_size}")

        data_loader = self.create_data_loader(
            CustomDataset,
            task_name,
            batch_size=batch_size,
            num_workers=32,
            task_type=meta_info["type"],
            collate_fn=collate_keep_list,
        )
        for question_ids, img_path, qss in data_loader:
            if cnt == 0:
                start_time = time.perf_counter()
            cnt += len(question_ids)
            messages = []
            for qid, paths, qs in zip(question_ids, img_path, qss):
                if paths is None:
                    paths = []
                elif isinstance(paths, str):
                    paths = [paths]
                message = self.build_message(qs, image_paths=paths)
                messages.append(message)
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages
            ]
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True, return_video_metadata=True)
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
            else:
                video_metadatas = None

            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadatas,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )

            # inputs = inputs.to("cuda")
            # inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            # Inference
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            responses = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for response, question_id, qs in zip(responses, question_ids, qss):
                self.accelerator.print(f"{qs}\n{response}\n\n")
                results.append(
                    # {"question_id": question_id, "answer": response.strip(), "prompt": qs}
                    {"question_id": question_id, "question": qs, "answer": response.strip(), "reason": "", "usage": None}
                )
        rank = self.accelerator.state.local_process_index

        self.save_result(results, meta_info, rank=rank)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            correct_num = self.collect_results_and_save(meta_info)
            total_time = time.perf_counter() - start_time
            print(
                f"Total time: {total_time}\nAverage time:{total_time / cnt}\nResults_collect number: {correct_num}"
            )

        print("rank", rank, "finished")


if __name__ == "__main__":
    args = parse_args()
    model_adapter = ModelAdapter(
        server_ip=args.server_ip,
        server_port=args.server_port,
        timeout=args.timeout,
        extra_cfg=args.cfg,
        local_mode=args.local_mode,
        task_names=args.tasks,
        output_dir=args.output_dir,
        model_path=args.model,
        debug=args.debug,
        quiet=args.quiet,
    )
    model_adapter.run()
