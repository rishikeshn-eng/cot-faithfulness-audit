# CoT Faithfulness Audit

**Status: experiments running. Results, figures and the full write-up land here when the runs finish.**

A reproducible hint-injection benchmark for chain-of-thought faithfulness. It covers 4,000 questions (3,000 MMLU, 500 ARC-Challenge, 500 GSM8K as 4-way MC) × 5 hint types (sycophancy, metadata, grader hack, visual pattern, unethical-access). It is run on Gemini flash-lite thinking models (with thought summaries) and on Qwen2.5-1.5B-Instruct locally. The question: when a model switches to the hinted answer, does its visible reasoning admit it?

Code: `src/`. Site: `docs/` (GitHub Pages).
