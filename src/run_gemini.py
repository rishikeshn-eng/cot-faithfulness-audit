"""Query a Gemini thinking model with thought summaries on a stratified subset of questions. Cached + resumable.

usage: python src/run_gemini.py --model gemini-3.5-flash-lite --n-questions 300 [--budget 2048] [--rpm 6]
"""
import argparse
import asyncio

from google.genai import types

from common import CACHE, DATA, append_jsonl, parse_answer, read_jsonl
from gemini_client import DailyQuotaExceeded, Gemini


def subset_ids(n):
    """First n questions of a fixed interleaving (3 MMLU : 1 ARC : 1 GSM8K), so any prefix is stratified
    and every model sees the same questions."""
    qs = read_jsonl(DATA / "questions.jsonl")
    by = {s: [q["id"] for q in qs if q["source"] == s] for s in ("mmlu", "arc_challenge", "gsm8k")}
    order = []
    for i in range(len(by["arc_challenge"])):
        order += by["mmlu"][3 * i:3 * i + 3] + [by["arc_challenge"][i], by["gsm8k"][i]]
    order += by["mmlu"][3 * len(by["arc_challenge"]):]  # remaining MMLU questions last
    return order[:n]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--n-questions", type=int, default=300)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--rpm", type=float, default=6)
    a = ap.parse_args()
    out = CACHE / f"gen_{a.model}.jsonl"
    done = {(r["qid"], r["cond"]) for r in read_jsonl(out) if "error" not in r}
    keep = subset_ids(a.n_questions)
    order = {q: i for i, q in enumerate(keep)}
    prompts = [p for p in read_jsonl(DATA / "prompts.jsonl") if p["qid"] in order]
    prompts.sort(key=lambda p: order[p["qid"]])  # all 6 conditions of a question run together
    todo = [p for p in prompts if (p["qid"], p["cond"]) not in done]
    print(f"{a.model}: {len(keep)} questions, {len(done)} cached, {len(todo)} to run", flush=True)
    g = Gemini(rpm=a.rpm)
    # Gemma 4 rejects thinking_budget; it thinks by default and returns its thoughts as thought parts.
    tc = (types.ThinkingConfig(include_thoughts=True) if a.model.startswith("gemma") else
          types.ThinkingConfig(include_thoughts=True, thinking_budget=a.budget))
    cfg = types.GenerateContentConfig(temperature=0.0, max_output_tokens=16384 if a.model.startswith("gemma") else a.budget + 2048, thinking_config=tc)

    async def one(p):
        try:
            r = await g.generate(a.model, p["prompt"], cfg)
        except DailyQuotaExceeded:
            raise
        except Exception as e:
            append_jsonl(out, dict(qid=p["qid"], cond=p["cond"], model=a.model, error=str(e)[:500]))
            return
        thoughts, answer = [], []
        cand = r.candidates[0] if r.candidates else None
        for part in (cand.content.parts or []) if cand and cand.content else []:
            if part.text:
                (thoughts if part.thought else answer).append(part.text)
        u = r.usage_metadata
        row = dict(qid=p["qid"], cond=p["cond"], model=a.model, thoughts="\n".join(thoughts),
                   response="\n".join(answer), thought_tokens=getattr(u, "thoughts_token_count", None),
                   output_tokens=getattr(u, "candidates_token_count", None),
                   finish=str(cand.finish_reason) if cand else None)
        # Final answer from the user-facing response only; a truncated (MAX_TOKENS) response stays unparsed.
        row["pred"] = parse_answer(row["response"])
        append_jsonl(out, row)

    try:
        for i in range(0, len(todo), 60):
            await asyncio.gather(*[one(p) for p in todo[i:i + 60]])
            print(f"  {min(i + 60, len(todo))}/{len(todo)} (429s so far: {g.n429})", flush=True)
    except DailyQuotaExceeded as e:
        print("DAILY QUOTA EXHAUSTED, stopping; rerun tomorrow to resume.", e, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
