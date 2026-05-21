#!/usr/bin/env bash
set -euo pipefail

BASE="/code1/llm_team/junpeng.yang/FlagEvalMM/results/v329-20260109-165104-alignment"

# DRYRUN=1 只打印将要删除的路径；DRYRUN=0 才真正删除
DRYRUN="${DRYRUN:-1}"

die() { echo "[ERR] $*" >&2; exit 1; }
run_rm() {
  if [[ "$DRYRUN" == "1" ]]; then
    echo rm -rf -- "$@"
  else
    rm -rf -- "$@"
  fi
}

[[ -d "$BASE" ]] || die "BASE not found: $BASE"

echo "[INFO] BASE=$BASE  DRYRUN=$DRYRUN"

echo "[STEP1] Remove top-level entries except iter_* ..."
while IFS= read -r -d '' p; do
  run_rm "$p"
done < <(find "$BASE" -mindepth 1 -maxdepth 1 ! -name 'iter_*' -print0)

echo "[STEP2] Clean each iter_*/<task>/ keep items/, *_config.json, task_info.json ..."
for iter_dir in "$BASE"/iter_*; do
  [[ -d "$iter_dir" ]] || continue

  for task_dir in "$iter_dir"/*; do
    [[ -d "$task_dir" ]] || continue

    # 删除 task_dir 下（深度=1）除 items/、*_config.json、task_info.json 之外的一切
    while IFS= read -r -d '' victim; do
      run_rm "$victim"
    done < <(
      find "$task_dir" -mindepth 1 -maxdepth 1 \
        \( -name 'items' -o -name '*_config.json' -o -name 'task_info.json' \) -prune \
        -o -print0
    )
  done
done

echo "[DONE] If output looks correct, run: DRYRUN=0 bash this_script.sh"


# 默认先看删除哪些文件
# bash cleanup_results.sh
# 真正删除多余文件
# DRYRUN=0 bash cleanup_results.sh
