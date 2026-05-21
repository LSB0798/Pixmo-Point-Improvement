import torch
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModel
from flagevalmm.server import ServerDataset
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.server.utils import parse_args, process_images_symbol, load_pil_image, default_collate_fn


class CustomDataset(ServerDataset):
    def __getitem__(self, index):
        data = self.get_data(index)
        question_id = data["question_id"]
        qs = data["question"]
        
        # 处理图像符号
        qs, idx = process_images_symbol(qs, dst_pattern="")
        image_list, _ = load_pil_image(
            data["img_path"], idx, reduplicate=True, reqiures_img=True
        )
        
        return question_id, qs, image_list


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict):
        ckpt_path = task_info["model_path"]
        torch.set_grad_enabled(False)
        
        with self.accelerator.main_process_first():
            # 加载 GLM-4.5V 模型和分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                ckpt_path, 
                trust_remote_code=True
            )
            
            model = AutoModel.from_pretrained(
                ckpt_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            ).eval()

        model = self.accelerator.prepare_model(model, evaluation_mode=True)
        if hasattr(model, "module"):
            model = model.module
        self.model = model

    def build_message(self, query: str, image_paths=[]):
        """构建 GLM-4.5V 格式的消息"""
        messages = [{"role": "user", "content": []}]
        
        # 添加图像
        for img_path in image_paths:
            messages[0]["content"].append({
                "type": "image",
                "image": img_path
            })
        
        # 添加文本问题
        messages[0]["content"].append({
            "type": "text", 
            "text": query
        })
        
        return messages

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        results = []
        
        data_loader = self.create_data_loader(
            CustomDataset,
            task_name,
            collate_fn=default_collate_fn,
            batch_size=1,
            num_workers=2,
        )
        
        for question_id, batch_question, batch_images in data_loader:
            for qid, question, images in zip(question_id, batch_question, batch_images):
                try:
                    # 构建消息
                    messages = self.build_message(question, images)
                    
                    # 生成回答
                    response = self.model.chat(
                        self.tokenizer,
                        messages,
                        max_new_tokens=1024,
                        do_sample=False,
                        temperature=0.0,
                    )
                    
                    print(f"{question}\n{response}\n\n")
                    
                    results.append({
                        "question_id": qid,
                        "answer": response,
                        "prompt": question,
                    })
                    
                except Exception as e:
                    print(f"Error processing {qid}: {e}")
                    results.append({
                        "question_id": qid,
                        "answer": "Error during inference",
                        "prompt": question,
                    })
        
        self.save_result(results, meta_info)


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