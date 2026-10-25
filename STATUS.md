# STATUS: cot-faithfulness-audit (done, frozen 25 Oct 2026 ~12:15 IST)
- Repo: https://github.com/rishikeshn-eng/cot-faithfulness-audit
- Pages: https://rishikeshn-eng.github.io/cot-faithfulness-audit/ (served from /docs on main)
- Headline: Qwen2.5-1.5B takes injected hints in 23.5% of eligible pairs (3× chance) but verbalises them in only 5.8% [3.8, 7.9] of those cases (0% for metadata/grader/visual hints). Gemini Flash-Lite thought summaries name the hint in 88–100% of uptake cases, but 3.5 Flash-Lite's user-facing response does so only 51% of the time. 3.5 Flash-Lite followed leaked grader code in 26/26 eligible cases. The keyword monitor catches 1.5% of unfaithful Qwen uptake; the LLM monitor catches 54.5% but has a 23% false-positive rate. Judge vs blind 100-case audit: 98% agreement, κ = 0.96.
- Benchmark built: 4,000 questions × 6 conditions (24,000 prompts).
## Cuts / not done (also in README Limitations)
- Qwen run on 750 of 4,000 questions (GPU queue + deadline); `run_qwen_chunks.sh` resumes the rest.
- API subjects are small (Gemini 3.5-FL 50 q, 3.1-FL 25 q, Gemma-4-26B 40 q) because of shared free-tier daily caps.
- Qwen LLM monitor partial (444/707 cases); ~3% of Qwen judge batches failed on API errors and were not retried.
- Gemma 4: 39% of outputs truncated at 16k tokens (treated as missing).
- Manual audit labels were made by the AI assistant (Claude), blind to the judge's labels, not by independent humans.
- No Hugging Face upload (no token).
