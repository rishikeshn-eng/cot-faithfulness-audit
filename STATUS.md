# STATUS: cot-faithfulness-audit (in progress)
- Repo: https://github.com/rishikeshn-eng/cot-faithfulness-audit
- Pages: https://rishikeshn-eng.github.io/cot-faithfulness-audit/
- Running: Qwen2.5-1.5B over 4,000 q × 6 conds (chunked under gpulock, `run_qwen_chunks.sh`); Gemini 3.5-flash-lite (75 q) and 3.1-flash-lite (55 q) subsets, capped by the free tier (500 req/day/model, shared).
- Next: judge (gemma-4-31b-it), 100-case manual audit, monitors (gemma-4-26b-a4b-it), analysis, README.
