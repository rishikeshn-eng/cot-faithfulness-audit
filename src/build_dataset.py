"""Build the question pool and the hint-injected prompts.

Output: data/questions.jsonl (one row per question) and data/prompts.jsonl
(one row per question x condition, condition in {none} + HINT_TYPES).
Everything is seeded so the benchmark is byte-for-byte reproducible.
"""
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

from hints import HINT_TYPES, LETTERS, render_prompt

SEED = 1234
N_MMLU, N_ARC, N_GSM = 3000, 500, 500
DATA = Path(__file__).resolve().parent.parent / "data"

MMLU_GROUPS = {
    "stem": "abstract_algebra anatomy astronomy college_biology college_chemistry college_computer_science "
    "college_mathematics college_physics computer_security conceptual_physics electrical_engineering "
    "elementary_mathematics high_school_biology high_school_chemistry high_school_computer_science "
    "high_school_mathematics high_school_physics high_school_statistics machine_learning",
    "humanities": "formal_logic high_school_european_history high_school_us_history high_school_world_history "
    "international_law jurisprudence logical_fallacies moral_disputes moral_scenarios philosophy prehistory "
    "professional_law world_religions",
    "social": "econometrics high_school_geography high_school_government_and_politics high_school_macroeconomics "
    "high_school_microeconomics high_school_psychology human_sexuality professional_psychology public_relations "
    "security_studies sociology us_foreign_policy",
}
SUBJ2GROUP = {s: g for g, ss in MMLU_GROUPS.items() for s in ss.split()}


def mmlu(rng):
    ds = load_dataset("cais/mmlu", "all", split="test", cache_dir=str(DATA / "hf"))
    idx = rng.sample(range(len(ds)), N_MMLU)
    for i in idx:
        r = ds[i]
        yield dict(id=f"mmlu-{i}", source="mmlu", subject=r["subject"],
                   group=SUBJ2GROUP.get(r["subject"], "other"), question=r["question"].strip(),
                   options=[c.strip() for c in r["choices"]], answer=LETTERS[r["answer"]])


def arc(rng):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", cache_dir=str(DATA / "hf"))
    ok = [i for i in range(len(ds)) if len(ds[i]["choices"]["text"]) == 4 and ds[i]["answerKey"] in "ABCD1234"]
    for i in rng.sample(ok, N_ARC):
        r = ds[i]
        labels = r["choices"]["label"]
        yield dict(id=f"arc-{r['id']}", source="arc_challenge", subject="science", group="arc",
                   question=r["question"].strip(), options=list(r["choices"]["text"]),
                   answer=LETTERS[labels.index(r["answerKey"])])


def gsm_distractors(g, rng):
    cands = [g + 1, g - 1, g * 2, g + 10, g - 10, g + 2, round(g * 1.5), g // 2 if g % 2 == 0 else g + 5,
             g * 10, g + 100, g - 2, g + 3]
    cands = [c for c in dict.fromkeys(cands) if c != g and c >= 0]
    rng.shuffle(cands)
    return cands[:3]


def gsm8k(rng):
    """GSM8K converted to 4-way multiple choice so all hint types apply uniformly."""
    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=str(DATA / "hf"))
    ok = []
    for i in range(len(ds)):
        m = re.search(r"####\s*(-?[\d,]+)\s*$", ds[i]["answer"])
        if m:
            ok.append((i, int(m.group(1).replace(",", ""))))
    for i, g in rng.sample(ok, N_GSM):
        opts = gsm_distractors(g, rng) + [g]
        rng.shuffle(opts)
        yield dict(id=f"gsm8k-{i}", source="gsm8k", subject="math_word_problem", group="gsm8k",
                   question=ds[i]["question"].strip(), options=[str(o) for o in opts],
                   answer=LETTERS[opts.index(g)])


def main():
    DATA.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    qs = list(mmlu(rng)) + list(arc(rng)) + list(gsm8k(rng))
    for q in qs:
        # One hint target per question, shared by all hint types: correct half the time.
        q["hint_correct"] = rng.random() < 0.5
        wrong = [l for l in LETTERS if l != q["answer"]]
        q["hint_target"] = q["answer"] if q["hint_correct"] else rng.choice(wrong)
    with open(DATA / "questions.jsonl", "w") as f:
        for q in qs:
            f.write(json.dumps(q) + "\n")
    n = 0
    with open(DATA / "prompts.jsonl", "w") as f:
        for q in qs:
            for cond in ["none"] + HINT_TYPES:
                f.write(json.dumps(dict(qid=q["id"], cond=cond, prompt=render_prompt(q, cond))) + "\n")
                n += 1
    print(f"{len(qs)} questions, {n} prompts")
    from collections import Counter
    print(Counter(q["source"] for q in qs), Counter(q["hint_correct"] for q in qs))


if __name__ == "__main__":
    main()
