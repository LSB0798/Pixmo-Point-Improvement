#!/bin/bash
python scripts_badcase/blink_badcase.py &
python scripts_badcase/cvbench_badcase.py &
python scripts_badcase/erqa_badcase.py &
python scripts_badcase/robo_spatial_home_badcase.py &
python scripts_badcase/sat_badcase.py &
python scripts_badcase/where2place_badcase.py &

# 等待所有后台进程完成
wait
echo "所有Python文件已运行完成！"