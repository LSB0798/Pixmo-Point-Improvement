# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
# swift export \
#     --adapters output/vx-xxx/checkpoint-xxx \
#     --merge_lora true

# 合并 LoRA 权重到基础模型
CUDA_VISIBLE_DEVICES=0,1 \
swift merge-lora \
    --adapters /data/lishuaibing/Qwen3-VL/ms-swift-main/output/Qwen3-VL-4B-Instruct/v11-20260301-105508/checkpoint-40 \
    --merge_device_map auto \
    --torch_dtype bfloat16 \
    --output_dir /data/lishuaibing/Qwen3-VL-4B-Instruct-merged/checkpoint-40