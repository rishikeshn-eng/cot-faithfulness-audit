"""Manual audit of the LLM judge.

  python src/audit.py export   # writes results/audit/blind.md: 100 stratified uptake cases, judge labels hidden
  python src/audit.py score    # compares results/audit/labels.json (human labels) with the judge

labels.json: {"<case no>": {"thoughts": "<label>", "response": "<label>"}, ...} using the judge's label set.
"""
import json
import random
import sys

from common import CACHE, RESULTS
from hints import HINT_DESCRIPTIONS
from judge import combined_label, load_pairs

QUOTA = {"qwen2.5-1.5b": 45, "gemini-3.5-flash-lite": 25, "gemini-3.1-flash-lite": 12, "gemma-4-26b-a4b-it": 18}
AUD = RESULTS / "audit"


def judged_uptake(model):
    j = {(r["qid"], r["cond"]): r for r in map(json.loads, open(CACHE / f"judge_{model}.jsonl")) if "error" not in r}
    return [(r, j[(r["qid"], r["cond"])]) for r in load_pairs(model)
            if r["eligible"] and r["uptake"] and (r["qid"], r["cond"]) in j]


def export():
    AUD.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    cases, short = [], 0
    for m, k in reversed(list(QUOTA.items())):  # Qwen last: it absorbs the shortfall of the small API pools
        pool = judged_uptake(m)
        rng.shuffle(pool)
        k = k + short if m == "qwen2.5-1.5b" else k
        cases += [(m, r, j) for r, j in pool[:k]]
        short += max(0, k - len(pool))
    rng.shuffle(cases)
    key, md = [], []
    for i, (m, r, j) in enumerate(cases):
        key.append(dict(no=i, model=m, qid=r["qid"], cond=r["cond"]))
        md.append(f"## Case {i}\nHint: {HINT_DESCRIPTIONS[r['cond']]} Pointed to ({r['target']}). Model answered {r['gen']['pred']}.\n"
                  f"<thought_summary>\n{(r['gen'].get('thoughts') or '')[-5000:]}\n</thought_summary>\n"
                  f"<final_response>\n{(r['gen'].get('response') or '')[-3000:]}\n</final_response>\n")
    (AUD / "blind.md").write_text("\n".join(md))
    json.dump(key, open(AUD / "key.json", "w"), indent=0)
    print(f"exported {len(cases)} cases")


def score():
    key = json.load(open(AUD / "key.json"))
    human = json.load(open(AUD / "labels.json"))
    judges = {}
    for m in QUOTA:
        judges.update({(m, r["qid"], r["cond"]): r for r in map(json.loads, open(CACHE / f"judge_{m}.jsonl"))})
    rows = []
    for k in key:
        h = human.get(str(k["no"]))
        if not h:
            continue
        j = judges[(k["model"], k["qid"], k["cond"])]
        hc = combined_label({"thoughts_label": h["thoughts"], "response_label": h["response"]})
        rows.append(dict(**k, human=hc, judge=combined_label(j), human_resp=h["response"], judge_resp=j.get("response_label")))
    n = len(rows)
    agree3 = sum(r["human"] == r["judge"] for r in rows) / n
    agree_bin = sum((r["human"] == "verbalized") == (r["judge"] == "verbalized") for r in rows) / n
    agree_resp = sum((r["human_resp"] == "verbalized") == (r["judge_resp"] == "verbalized") for r in rows) / n
    tp = sum(r["human"] == "verbalized" and r["judge"] == "verbalized" for r in rows)
    fp = sum(r["human"] != "verbalized" and r["judge"] == "verbalized" for r in rows)
    fn = sum(r["human"] == "verbalized" and r["judge"] != "verbalized" for r in rows)
    # Cohen's kappa on the binary verbalized label
    ph = sum(r["human"] == "verbalized" for r in rows) / n
    pj = sum(r["judge"] == "verbalized" for r in rows) / n
    pe = ph * pj + (1 - ph) * (1 - pj)
    kappa = (agree_bin - pe) / (1 - pe) if pe < 1 else 1.0
    out = dict(n=n, agreement_3class=agree3, agreement_verbalized=agree_bin, agreement_response_verbalized=agree_resp,
               judge_precision=tp / max(tp + fp, 1), judge_recall=tp / max(tp + fn, 1), kappa_verbalized=kappa,
               human_verbalized_rate=ph, judge_verbalized_rate=pj,
               disagreements=[r for r in rows if r["human"] != r["judge"]])
    json.dump(out, open(AUD / "agreement.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "disagreements"}, indent=1))


if __name__ == "__main__":
    {"export": export, "score": score}[sys.argv[1]]()
