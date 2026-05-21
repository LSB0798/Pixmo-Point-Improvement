set -e  # 一旦命令出错就退出脚本

cd /code1/llm_team/junpeng.yang/FlagEvalMM
echo "当前目录是：$(pwd)"


#tasks/robovqa/robovqa.py tasks/egoplan2/egoplan2.py

# BASEMODEL=/code1/train_logs/merge_lora/thinker/v164-20251211-152343-alignment/iter_0008500
# BASEMODEL=/code1/train_logs/merge_lora/thinker/v300-20251225-024907-alignment/iter_0001000
BASEMODEL=/code1/data/Qwen3-VL-32B-Instruct
MODEL=eval-10-qwen3vl32b-20251230-1600
READ_MODEL_PATH=/code1/train_logs/merge_lora/jp_thinker/$MODEL
SAVE_MODEL_PATH=/code1/train_logs/merge_lora/jp_thinker/$MODEL

# 创建目标目录
mkdir -p "$SAVE_MODEL_PATH"
# 如果 1-10 都存在，则整体跳过
all_exist=true
for i in $(seq 1 5); do
  dst="$(printf "%s/iter_%07d" "$SAVE_MODEL_PATH" "$i")"
  if [ ! -d "$dst" ]; then
    all_exist=false
    break
  fi
done

if $all_exist; then
  echo "[SKIP-ALL] iter_0000001-iter_000005 already exist in: $SAVE_MODEL_PATH"
else
  # 否则只复制缺失的 iter
  for i in $(seq 1 5); do
    dst="$(printf "%s/iter_%07d" "$SAVE_MODEL_PATH" "$i")"
    if [ -d "$dst" ]; then
      echo "[SKIP] exists: $dst"
      continue
    elif [ -e "$dst" ]; then
      echo "[SKIP] path exists but not a dir: $dst"
      continue
    fi
    echo "[COPY] $BASEMODEL -> $dst"
    cp -a "$BASEMODEL" "$dst"
  done
fi


#PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \

export MEGATRON_LM_PATH='/mnt/workspace/.cache/modelscope/hub/_github/Megatron-LM'
export TOKENIZERS_PARALLELISM=false
NPROC_PER_NODE=1 \
python babysitting_multi.py \
  --watch-dir $READ_MODEL_PATH \
  --output-base $SAVE_MODEL_PATH \
  --tasks tasks/robovqa/robovqa.py tasks/egoplan2/egoplan2.py \
  --cuda 2 \
  --exec model_zoo/vlm/api_model/model_adapter.py \
  --state-id ldq_robovqa \
  --all-tasks-scope model \
  --num-workers 8 \
  --backend vllm \
  --extra-args "--limit-mm-per-prompt '{\"image\": 512, \"video\": 2}' \
                --media-io-kwargs '{\"video\": {\"num_frames\": 32} }' \
                --mm-processor-kwargs '{\"min_pixels\": 784, \"max_pixels\": 2097152, \"fps\": 2, \"do_sample_frames\": false}' \
                --data-parallel-size 1 --seed 3407 --max-model-len 50000 --mm-encoder-tp-mode data --trust-remote-code --gpu-memory-utilization 0.85 --allowed-local-media-path /code1 --interleave-mm-strings"  \
  --is-lora false \
  --is-mcore false \
  --scp-dest ubt@10.10.22.57:/media/ubt/04d101c2-7eb0-4765-afd4-85d6f4b201a7/workspace/1230/qwen3vl32b/