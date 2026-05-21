set -e  # 一旦命令出错就退出脚本

cd /code1/llm_team/junpeng.yang/FlagEvalMM
echo "当前目录是：$(pwd)"


#tasks/robovqa/robovqa.py tasks/egoplan2/egoplan2.py

BASEMODEL=/code1/train_logs/merge_lora/thinker/v82-20251112-104714-alignment/iter_0007000
MODEL=v2-20251124-102613
READ_MODEL_PATH=/code1/train_logs/lora/jp_thinker/$MODEL
SAVE_MODEL_PATH=/code1/train_logs/merge_lora/jp_thinker/$MODEL

# 创建目标目录
mkdir -p /code1/train_logs/lora/jp_thinker/$MODEL/iter_0000000
mkdir -p /code1/train_logs/merge_lora/jp_thinker/$MODEL/iter_0000000
cp -r $BASEMODEL/* /code1/train_logs/merge_lora/jp_thinker/$MODEL/iter_0000000/
echo "✅  已完成复制：$BASEMODEL → /code1/train_logs/merge_lora/jp_thinker/$MODEL/iter_0000000/"


#PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \

#export CUDA_LAUNCH_BLOCKING=1
export MEGATRON_LM_PATH='/mnt/workspace/.cache/modelscope/hub/_github/Megatron-LM'
export TOKENIZERS_PARALLELISM=false
NPROC_PER_NODE=2 \
python babysitting_multi.py \
  --watch-dir $READ_MODEL_PATH \
  --output-base $SAVE_MODEL_PATH \
  --tasks tasks/egoplan2/egoplan2.py \
  --cuda 0,1 \
  --exec model_zoo/vlm/api_model/model_adapter.py \
  --state-id rsft_egoplan2 \
  --all-tasks-scope model \
  --num-workers 16 \
  --backend vllm \
  --extra-args "--limit-mm-per-prompt '{\"image\": 512, \"video\": 2}' \
                --media-io-kwargs '{\"video\": {\"num_frames\": 32} }' \
                --mm-processor-kwargs '{\"min_pixels\": 784, \"max_pixels\": 2097152, \"fps\": 2, \"do_sample_frames\": false}' \
                --data-parallel-size 2 --seed 3407 --max-model-len 60000 --mm-encoder-tp-mode data --trust-remote-code --gpu-memory-utilization 0.8 --allowed-local-media-path /code1 --interleave-mm-strings"  \
  --is-lora true \
  --is-mcore true