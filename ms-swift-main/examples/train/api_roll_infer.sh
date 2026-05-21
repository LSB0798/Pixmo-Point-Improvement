# If it's full parameter training, use `--model xxx` instead of `--adapters xxx`.
# If you are using the validation set for inference, add the parameter `--load_data_args true`.
CUDA_VISIBLE_DEVICES=0 \
swift rollout \
    --model /data/models/Qwen3-VL-2B-Instruct \
    --vllm_data_parallel_size 1