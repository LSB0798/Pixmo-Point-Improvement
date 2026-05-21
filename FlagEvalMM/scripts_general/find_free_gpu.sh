#!/bin/bash
# monitor_gpu_and_run.sh
SCRIPT="bash scripts/eval.sh "
THRESHOLD=40000   # 阈值: 30G 显存 (MB)

while true; do
  # 遍历所有 GPU
  for gpu in $(nvidia-smi --query-gpu=index --format=csv,noheader,nounits); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu)
    if [ "$free" -ge "$THRESHOLD" ]; then
      echo "✅ GPU $gpu 有 $free MB 可用显存，开始运行脚本..."
      CUDA_VISIBLE_DEVICES=$gpu $SCRIPT
      exit 0   # 执行一次后退出（想要循环就删掉这一行）
    fi
  done
  sleep 30   # 每 30 秒检查一次
done
