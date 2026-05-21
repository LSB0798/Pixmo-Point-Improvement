#!/bin/bash
# 如果预处理失败/root/.cache/flagevalmm/datasets缓存要清

export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
# export TOKENIZERS_PARALLELISM=false
# export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
# export NPROC_PER_NODE=8

export NCCL_P2P_DISABLE=1          # 强制禁用P2P通信（关键！）
export NCCL_SHM_DISABLE=1          # 禁用共享内存传输
export NCCL_CUMEM_HOST_ENABLE=0    # 明确禁用cuMem
export TMPDIR=''

export FPS_MAX_FRAMES=256
export FORCE_QWENVL_VIDEO_READER=decord # torchcodec暂时有问题
export MEGATRON_LM_PATH=''
export CUDA_VISIBLE_DEVICES=0,1

export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0 # 还不行再加


# MODEL_PATH=/data/Qwen3-VL-2B-Instruct/
# MODEL_PATH=/data/Qwen3-VL-4B-Instruct/
# MODEL_PATH=/data/Qwen3-VL-8B-Instruct/

MODEL_PATH=/data/Qwen3-VL/ms-swift-main/output1/iter_0001000/checkpoint-12200/
# MODEL_PATH=/data/Qwen3-VL-4B-Instruct/

# thinker
export CUDA_VISIBLE_DEVICES=0,1
flagevalmm --tasks tasks/pixmo_points_eval/pixmo_points_eval_thinker.py \
        --exec model_zoo/vlm/api_model/model_adapter.py \
        --model $MODEL_PATH \
        --num-workers 8 \
        --backend vllm \
        --output-dir ./results/$(basename $MODEL_PATH)-vllm-1 \
        --extra-args "--limit-mm-per-prompt '{\"image\": 512, \"video\": 2}' \
                      --media-io-kwargs '{\"video\": {\"num_frames\": 32} }' \
                      --mm-processor-kwargs '{\"min_pixels\": 784, \"max_pixels\": 524288, \"fps\": 1, \"do_sample_frames\": false, \"do_resize\": true}' \
                      --tensor-parallel-size 2 --data-parallel-size 1 --seed 3407 --max-model-len 8192 --mm-encoder-tp-mode data --trust-remote-code --gpu-memory-utilization 0.3 --allowed-local-media-path /data --interleave-mm-strings"

