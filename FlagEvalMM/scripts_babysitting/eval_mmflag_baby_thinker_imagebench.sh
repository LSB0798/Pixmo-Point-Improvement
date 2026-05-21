set -e  # 一旦命令出错就退出脚本

cd /code1/llm_team/junpeng.yang/FlagEvalMM
echo "当前目录是：$(pwd)"

#tasks/robovqa/robovqa.py tasks/egoplan2/egoplan2.py

BASEMODEL=/code1/data/Qwen3-VL-8B-Instruct
MODEL=v300-20251225-024907-alignment
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
# ssh-copy-id ubt@10.10.22.57

export MEGATRON_LM_PATH='/mnt/workspace/.cache/modelscope/hub/_github/Megatron-LM'
export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
python babysitting_multi.py \
  --watch-dir $READ_MODEL_PATH \
  --output-base $SAVE_MODEL_PATH \
  --tasks  tasks/blink/blink_val_qwen3vl.py tasks/cv_bench/cv_bench_test.py tasks/embspatial_bench/embspatial_bench.py tasks/robo_spatial_home/robo_spatial_home_all_qwen3vl.py tasks/sat/sat_qwen3vl.py tasks/refspatial_bench/refspatial_bench_qwen3vl.py tasks/where2place/where2place_qwen3vl.py tasks/vsr_random/vsr_random.py tasks/refcoco/refcoco_val_qwen3vl.py tasks/refcoco/refcoco_g_val_qwen3vl.py tasks/refcoco/refcoco_plus_val_qwen3vl.py tasks/erqa/erqa_pelican.py \
  --cuda 5,6 \
  --exec model_zoo/vlm/api_model/model_adapter.py \
  --state-id ldq_imagebench \
  --all-tasks-scope model \
  --num-workers 8 \
  --backend vllm \
  --extra-args "--limit-mm-per-prompt '{\"image\": 512, \"video\": 2}' \
                --media-io-kwargs '{\"video\": {\"num_frames\": 256} }' \
                --mm-processor-kwargs '{\"min_pixels\": 784, \"max_pixels\": 2097152, \"fps\": 2, \"do_sample_frames\": false}' \
                --data-parallel-size 2 --seed 3407 --max-model-len 50000 --mm-encoder-tp-mode data --trust-remote-code --gpu-memory-utilization 0.2 --allowed-local-media-path /code1 --interleave-mm-strings"  \
  --is-lora true \
  --is-mcore true \
  --scp-dest ubt@10.10.22.57:/media/ubt/04d101c2-7eb0-4765-afd4-85d6f4b201a7/workspace/1230/test/