#!/bin/bash

docker run --rm --shm-size=64g -it --net host --gpus all --name junpeng.yang-flagevalmm \
  -v /data/data/NLP/nlp_team1:/data/nlp/nlp_team1 \
  -v /data/ldq:/data/ldq \
  -v /home/daqin.luo:/home/daqin.luo \
  -v /data/llm_team:/work/llm_team \
  -v /home/junpeng.yang/workspace:/code1/llm_team/junpeng.yang \
  -v /data/data/NLP/nlp_team1/data/hanlp:/root/.hanlp \
  ubhub.ubtrobot.com/ogi/junpeng-flagevalmm:v3.3 \
  bash -c "cd /code1/llm_team/junpeng.yang/FlagEvalMM && \
           pip install -e . && \
           bash"
