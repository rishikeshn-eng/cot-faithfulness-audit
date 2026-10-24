#!/bin/bash
# One-command pipeline. Every stage is cached/resumable; API stages stop cleanly on free-tier daily caps.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
(cd src && ../$PY build_dataset.py)
./run_qwen_chunks.sh 4000
(cd src && ../$PY run_gemini.py --model gemini-3.5-flash-lite --n-questions 50)
(cd src && ../$PY run_gemini.py --model gemini-3.1-flash-lite --n-questions 25)
(cd src && ../$PY run_gemini.py --model gemma-4-26b-a4b-it --n-questions 40)
for spec in qwen2.5-1.5b:100 gemini-3.5-flash-lite:60 gemini-3.1-flash-lite:60 gemma-4-26b-a4b-it:40; do
  m=${spec%:*}; nc=${spec#*:}   # n-control = size of the non-uptake control samples
  (cd src && ../$PY judge.py judge --gen-model $m --n-control $nc && ../$PY judge.py monitor --gen-model $m --n-control $nc)
done
(cd src && ../$PY audit.py score)   # needs results/audit/labels.json (the manual audit)
(cd src && ../$PY analyze.py qwen2.5-1.5b gemini-3.5-flash-lite gemini-3.1-flash-lite gemma-4-26b-a4b-it && ../$PY report.py)
