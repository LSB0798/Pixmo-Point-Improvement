set -e  # 一旦命令出错就退出脚本

cd /code1/llm_team/junpeng.yang/FlagEvalMM
echo "当前目录是：$(pwd)"


#tasks/robovqa/robovqa.py tasks/egoplan2/egoplan2.py
# --tasks tasks/blink/blink_val.py tasks/cv_bench/cv_bench_test.py tasks/embspatial_bench/embspatial_bench.py tasks/robo_spatial_home/robo_spatial_home_all_qwen3vl.py tasks/refspatial_bench/refspatial_bench_qwen3vl.py tasks/sat/sat.py tasks/vsi_bench/vsi_bench_tiny.py tasks/where2place/where2place_qwen3vl.py tasks/robovqa/robovqa.py tasks/egoplan2/egoplan2.py \

BASEMODEL=Qwen3-VL-235B-A22B-Instruct
MODEL=v52-20260102-171324
READ_MODEL_PATH=/data/train_logs/lora/thinker/$MODEL
SAVE_MODEL_PATH=/code1/train_logs/merge_lora/thinker/$MODEL-alignment


# 创建目标目录
#mkdir -p /code1/train_logs/lora/thinker/$MODEL/iter_0000000
#mkdir -p /code1/train_logs/merge_lora/thinker/$MODEL-alignment/iter_0000000
#cp -r /data/$BASEMODEL/* /code1/train_logs/merge_lora/thinker/$MODEL-alignment/iter_0000000/
#echo "✅  已完成复制：/data/Qwen3-VL-4B-Instruct → /code1/train_logs/merge_lora/thinker/$xxx/iter_0000000/"

#tasks/vsi_bench/vsi_bench_tiny.py
#tasks/blink/blink_val.py tasks/cv_bench/cv_bench_test.py tasks/embspatial_bench/embspatial_bench.py tasks/robo_spatial_home/robo_spatial_home_all_qwen3vl.py tasks/refspatial_bench/refspatial_bench_qwen3vl.py tasks/sat/sat.py tasks/vsi_bench/vsi_bench_tiny.py tasks/where2place/where2place_qwen3vl.py tasks/robovqa/robovqa.py tasks/egoplan2/egoplan2.py

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MEGATRON_LM_PATH='/mnt/workspace/.cache/modelscope/hub/_github/Megatron-LM'
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export TOKENIZERS_PARALLELISM=false
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
export VLLM_USE_DEEP_GEMM=0

NPROC_PER_NODE=8 \
python babysitting_multi.py \
  --watch-dir $READ_MODEL_PATH \
  --output-base $SAVE_MODEL_PATH \
  --tasks tasks/robovqa/robovqa.py tasks/blink/blink_val_qwen3vl.py tasks/cv_bench/cv_bench_test.py tasks/embspatial_bench/embspatial_bench.py tasks/robo_spatial_home/robo_spatial_home_all_qwen3vl.py tasks/sat/sat_qwen3vl.py tasks/erqa/erqa_qwen3vl.py tasks/refspatial_bench/refspatial_bench_qwen3vl.py tasks/where2place/where2place_qwen3vl.py tasks/pixmo_points_eval/pixmo_points_eval_qwen3vl.py \
  --cuda 0,1,2,3,4,5,6,7 \
  --exec model_zoo/vlm/api_model/model_adapter.py \
  --state-id ldq_robovqa_VG_235b \
  --all-tasks-scope model \
  --num-workers 16 \
  --backend vllm \
  --extra-args "--limit-mm-per-prompt '{\"image\": 512, \"video\": 2}' \
                --media-io-kwargs '{\"video\": {\"num_frames\": 32} }' \
                --mm-processor-kwargs '{\"min_pixels\": 784, \"max_pixels\": 2097152, \"fps\": 2, \"do_sample_frames\": false}' \
                --tensor-parallel-size 8 --seed 3407 --max-model-len 50000 --mm-encoder-tp-mode data --trust-remote-code --gpu-memory-utilization 0.6 --allowed-local-media-path /code1 --interleave-mm-strings --enable-expert-parallel --distributed-executor-backend mp --enforce-eager" \
  --is-lora true \
  --is-mcore true \