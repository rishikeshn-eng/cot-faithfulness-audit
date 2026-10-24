# STATUS: cot-faithfulness-audit (published with partial runs; usage limit reached 2026-10-25 ~01:00 IST)
- Repo: https://github.com/rishikeshn-eng/cot-faithfulness-audit
- Pages: https://rishikeshn-eng.github.io/cot-faithfulness-audit/
- Headline (see README table): Qwen2.5-1.5B takes hints in ~22% of eligible pairs (chance switch ~8%) but verbalises them in only ~5% of those cases. The keyword monitor catches ~1% of unfaithful uptake. Gemini flash-lite thought summaries mention the hint in most uptake cases, the user-facing response much less often. Gemini 3.5 Flash-Lite follows the grader-hack hint in 26/26 eligible cases.
- Data: benchmark 4,000 q × 6 conds built. Qwen run on 750 questions (4,500 gens); `run_qwen_chunks.sh` is still queued for the GPU and resumes from cache. Gemini 3.5-FL 50 q, 3.1-FL 25 q, Gemma-4-26B 40 q (free-tier caps).
## Left undone
1. Manual audit: 100 blind cases exported to results/audit/blind.md; labels.json NOT written yet, so the README has no judge-agreement numbers. Then run `src/audit.py score` + `report.py`.
2. LLM monitor on Qwen not run (`judge.py monitor --gen-model qwen2.5-1.5b --n-control 100`); some judge batches failed with 503 (rerun `judge.py judge` to fill in).
3. Qwen beyond 750 questions if GPU time allows (rerun judge/analyze/report after). Fill the README <!-- FINDINGS --> block with prose.
