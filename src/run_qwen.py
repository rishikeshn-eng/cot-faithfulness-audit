"""Greedy CoT generations from Qwen2.5-1.5B-Instruct via MLX (Apple GPU). Cached + resumable.

Run under the shared GPU lock:  ~/ai-safety-portfolio/_shared/gpulock.sh .venv/bin/python src/run_qwen.py
"""
import argparse
import time

from mlx_lm import batch_generate, load

from common import CACHE, DATA, append_jsonl, parse_answer, read_jsonl
from run_gemini import subset_ids

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-bf16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-questions", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()
    out = CACHE / "gen_qwen2.5-1.5b.jsonl"
    done = {(r["qid"], r["cond"]) for r in read_jsonl(out)}
    keep = subset_ids(a.limit_questions)  # same stratified ordering as the Gemini runs
    order = {q: i for i, q in enumerate(keep)}
    prompts = [p for p in read_jsonl(DATA / "prompts.jsonl") if p["qid"] in order and (p["qid"], p["cond"]) not in done]
    prompts.sort(key=lambda p: order[p["qid"]])
    print(f"{len(keep)} questions, {len(done)} cached, {len(prompts)} to run", flush=True)
    model, tok = load(MODEL)
    t0 = time.time()
    for i in range(0, len(prompts), a.batch):
        chunk = prompts[i:i + a.batch]
        ids = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True)
               for p in chunk]
        res = batch_generate(model, tok, ids, max_tokens=a.max_tokens, verbose=False)
        for p, text in zip(chunk, res.texts):
            append_jsonl(out, dict(qid=p["qid"], cond=p["cond"], model="qwen2.5-1.5b", thoughts="",
                                   response=text, pred=parse_answer(text)))
        el = time.time() - t0
        print(f"  {i + len(chunk)}/{len(prompts)}  {el / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
