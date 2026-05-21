set -e  # 一旦命令出错就退出脚本

cd /code1/llm_team/junpeng.yang/FlagEvalMM
echo "当前目录是：$(pwd)"

#tasks/robovqa/robovqa.py tasks/egoplan2/egoplan2.py

BASEMODEL=/code1/data/Qwen3-VL-4B-Instruct
MODEL=v42-20260209-161031
READ_MODEL_PATH=/code1/train_logs/lora/jp_thinker/$MODEL
SAVE_MODEL_PATH=/code1/train_logs/merge_lora/jp_thinker/$MODEL

# 创建目标目录
mkdir -p /code1/train_logs/lora/jp_thinker/$MODEL/iter_0000000
mkdir -p /code1/train_logs/merge_lora/jp_thinker/$MODEL/iter_0000000
cp -r $BASEMODEL/* /code1/train_logs/merge_lora/jp_thinker/$MODEL/iter_0000000/
echo "✅  已完成复制：$BASEMODEL → /code1/train_logs/merge_lora/jp_thinker/$MODEL/iter_0000000/"

#PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
#export CUDA_LAUNCH_BLOCKING=1

# 用 ED25519 算法生成一对 SSH 密钥（公钥 + 私钥），发到本地主机用于无密码scp结果图  
# ssh-keygen -t ed25519
# ssh-copy-id ubt@10.10.22.92

# export MEGATRON_LM_PATH='/mnt/workspace/.cache/modelscope/hub/_github/Megatron-LM'
export MEGATRON_LM_PATH='/root/.cache/modelscope/_github/Megatron-LM'
export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
python babysitting_multi.py \
  --watch-dir $READ_MODEL_PATH \
  --output-base $SAVE_MODEL_PATH \
  --tasks tasks/refcoco/seg_refcoco_thinker.py tasks/refcoco/seg_refcocog_thinker.py tasks/refcoco/seg_refcocop_thinker.py \
  --cuda 1,3 \
  --exec model_zoo/vlm/api_model/model_adapter.py \
  --state-id yjp_refcocoseg \
  --all-tasks-scope model \
  --num-workers 8 \
  --backend vllm \
  --extra-args "--limit-mm-per-prompt '{\"image\": 512, \"video\": 2}' \
                --media-io-kwargs '{\"video\": {\"num_frames\": 256} }' \
                --mm-processor-kwargs '{\"min_pixels\": 784, \"max_pixels\": 2097152, \"fps\": 2, \"do_sample_frames\": false}' \
                --data-parallel-size 2 --seed 3407 --max-model-len 50000 --mm-encoder-tp-mode data --trust-remote-code --gpu-memory-utilization 0.3 --allowed-local-media-path /code1 --interleave-mm-strings"  \
  --is-lora true \
  --is-mcore true \
  --scp-dest ubt@10.10.22.92:/media/ubt/04d101c2-7eb0-4765-afd4-85d6f4b201a7/workspace/260230/refcocoseg/v42-20260209-161031/