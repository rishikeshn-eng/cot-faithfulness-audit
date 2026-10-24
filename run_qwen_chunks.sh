#!/bin/bash
# Qwen2.5-1.5B over the whole benchmark, one GPU-lock acquisition per 500-question chunk so
# other projects' GPU jobs can run in between. Resumable: finished generations are cached.
cd "$(dirname "$0")/src"
export HF_HOME=$PWD/../data/hf_home HF_HUB_OFFLINE=1
LOCK=~/ai-safety-portfolio/_shared/gpulock.sh
for n in $(seq 500 500 ${1:-4000}); do
  $LOCK ../.venv/bin/python -u run_qwen.py --limit-questions $n --batch ${BATCH:-64}
done
echo QWEN_DONE
