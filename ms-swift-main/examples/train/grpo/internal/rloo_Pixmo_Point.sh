export NCCL_ASYNC_ERROR_HANDLING=1  # 启用异步错误处理
export NCCL_BLOCKING_WAIT=0         # 禁用阻塞等待，改用超时机制
export NCCL_TIMEOUT=3600000          # 将超时时间设置得长一些（单位毫秒，例如30分钟）
export NCCL_RETRY_TIMEOUT=3600       # 设置连接重试的超时时间

export SWANLAB_API_KEY='*****'
export WANDB_API_KEY='*****'

export FPS=1
export FPS_MIN_FRAMES=2
export FPS_MAX_FRAMES=16

export TORCH_SYMMETRIC_MEMORY=0

# 分布式训练端口设置 - 解决端口冲突
export MASTER_PORT=29501
export MASTER_ADDR=localhost
export WORLD_SIZE=2
export RANK=0
export LOCAL_RANK=0

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export VLLM_USE_CUMEM_ALLOCATOR=0
export VLLM_MAX_NUM_SEQS=8

E2B_API_KEY='*****' \
SWANLAB_API_KEY='*****' \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift rlhf \
    --rlhf_type grpo \
    --advantage_estimator rloo \
    --kl_in_reward true \
    --model /data/models/Qwen3-VL-4B-Instruct \
    --reward_funcs pixmo_point_reward \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_engine_kwargs '{"swap_space": 2}' \
    --vllm_tensor_parallel_size 2 \
    --vllm_max_model_len 16384 \
    --max_length 12288 \
    --train_type lora \
    --dataset '/data/check_test_250k_Pixmo_Point.json' \
    --overlong_filter true \
    --truncation_strategy delete \
    --epsilon 3e-2 \
    --epsilon_high 4e-2 \
    --max_completion_length 4096 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-7 \
    --warmup_steps 2000 \
    --lr_scheduler_type cosine \
    --bf16 true \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing true \
    --eval_steps 1000 \
    --save_steps 100 \
    --save_total_limit 20 \
    --sleep_level 2 \
    --offload_model true \
    --offload_optimizer false \
    --logging_steps 1 \
    --dataloader_num_workers 8 \
    --num_generations 4 \
    --temperature 0.7 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero2 \
    --log_completions true \
    --report_to tensorboard swanlab \
    --num_iterations 1 \
    --async_generate false \
    --beta 0.01 \
    --attn_impl flash_attention_2 \
    --padding_free true \
    --loss_type grpo \
    --report_to swanlab \
    > train.log 2>&1 &
