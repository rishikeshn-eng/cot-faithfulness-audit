"""Render the README result tables from results/metrics.json (+ audit agreement) between marker comments."""
import json
import re

from common import RESULTS, ROOT, read_jsonl
from hints import HINT_TYPES

LABEL = {"qwen2.5-1.5b": "Qwen2.5-1.5B-Instruct (local)", "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
         "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite", "gemma-4-26b-a4b-it": "Gemma 4 26B-A4B"}


def p(o):
    return "–" if not o or o.get("mean") is None else f"{100 * o['mean']:.1f}"


def ci(o):
    return "–" if not o or o.get("mean") is None else f"{100 * o['mean']:.1f} [{100 * o['lo']:.1f}, {100 * o['hi']:.1f}]"


def n_questions(model):
    rows = read_jsonl(ROOT / "cache" / f"gen_{model}.jsonl")
    return len({r["qid"] for r in rows if "error" not in r})


def render(M, audit):
    out = []
    out.append("| Model | Questions | Unparsed | Acc. no hint | Eligible pairs | Hint uptake | Chance switch | Verbalised \\| uptake | …in final response only | Mentioned at all |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for m, r in M.items():
        o = r["overall"]
        out.append(f"| {LABEL.get(m, m)} | {n_questions(m)} | {100 * r['parse_fail']:.1f} | {p(r['accuracy']['none'])} | {o['eligible']} | {ci(o['uptake'])} | "
                   f"{100 * o['chance_switch']:.1f} | {ci(o['verbalized'])} (n={o['verbalized']['n']}) | {ci(o['verbalized_response'])} | {ci(o['mentioned'])} |")
    out.append("\nAll numbers are %, with 95% bootstrap CIs over questions (1,000 resamples). *Unparsed* = outputs with no final "
               "\"Answer: X\" (mostly truncation at the token cap); they are treated as missing. *Eligible pairs* = hinted prompts "
               "whose un-hinted answer differed from the hint target (both answers parsed). *Verbalised* uses the LLM judge on the model's visible output "
               "(thought summary/thoughts + final response).\n")
    out.append("**By hint type: uptake % / verbalised-given-uptake %**\n")
    out.append("| Model | " + " | ".join(HINT_TYPES) + " |")
    out.append("|---|" + "---|" * len(HINT_TYPES))
    for m, r in M.items():
        cells = [f"{p(r['by_hint'][h]['uptake'])} / {p(r['by_hint'][h]['verbalized'])} (n={r['by_hint'][h]['verbalized']['n']})" for h in HINT_TYPES]
        out.append(f"| {LABEL.get(m, m)} | " + " | ".join(cells) + " |")
    out.append("\n**Accuracy when the hint points to a wrong answer (%; compare with the no-hint accuracy column above)**\n")
    out.append("| Model | " + " | ".join(HINT_TYPES) + " |")
    out.append("|---|" + "---|" * len(HINT_TYPES))
    for m, r in M.items():
        a = r["accuracy"]
        out.append(f"| {LABEL.get(m, m)} | " + " | ".join(f"{p(a[h + '|hint_wrong'])}" for h in HINT_TYPES) + f" |")
    out.append("\n**Difficulty: uptake % / verbalised %, split by whether the model was right without the hint**\n")
    out.append("| Model | Correct w/o hint (easier) | Wrong w/o hint (harder) | Hint → correct option | Hint → wrong option |")
    out.append("|---|---|---|---|---|")
    for m, r in M.items():
        b, hc = r["by_baseline_correct"], r["by_hint_correct"]
        f = lambda o: f"{p(o['uptake'])} / {p(o['verbalized'])}"
        out.append(f"| {LABEL.get(m, m)} | {f(b['True'])} | {f(b['False'])} | {f(hc['True'])} | {f(hc['False'])} |")
    out.append("\n**CoT monitors (read only the model's visible reasoning, never the prompt): flag rate %**\n")
    out.append("| Model | Monitor | Unfaithful uptake (recall) | Verbalised uptake (recall) | Hint ignored | No hint (FPR) |")
    out.append("|---|---|---|---|---|---|")
    for m, r in M.items():
        for k, name in [("kw", "keyword regex"), ("llm_mon", "LLM (Gemma 4 31B)")]:
            d = r["monitors"][k]
            f = lambda o: f"{p(o)} (n={o['n']})"
            out.append(f"| {LABEL.get(m, m)} | {name} | {f(d['unfaithful_uptake'])} | {f(d['faithful_uptake'])} | "
                       f"{f(d['hint_ignored'])} | {f(d['no_hint_baseline'])} |")
    if audit:
        out.append(f"\n**Judge validation (manual audit, n={audit['n']} uptake cases):** agreement on *verbalised vs not* "
                   f"{100 * audit['agreement_verbalized']:.0f}% (Cohen's κ = {audit['kappa_verbalized']:.2f}); "
                   f"judge precision {100 * audit['judge_precision']:.0f}%, recall {100 * audit['judge_recall']:.0f}%; "
                   f"3-class agreement {100 * audit['agreement_3class']:.0f}%; final-response-only agreement "
                   f"{100 * audit['agreement_response_verbalized']:.0f}%. Human verbalised rate {100 * audit['human_verbalized_rate']:.0f}% "
                   f"vs judge {100 * audit['judge_verbalized_rate']:.0f}%.")
    return "\n".join(out)


def main():
    M = json.load(open(RESULTS / "metrics.json"))
    ap = RESULTS / "audit" / "agreement.json"
    audit = json.load(open(ap)) if ap.exists() else None
    readme = (ROOT / "README.md").read_text()
    block = render(M, audit)
    readme = re.sub(r"<!-- RESULTS -->.*<!-- /RESULTS -->", lambda _: f"<!-- RESULTS -->\n{block}\n<!-- /RESULTS -->",
                    readme, flags=re.S)
    (ROOT / "README.md").write_text(readme)
    print(block)


if __name__ == "__main__":
    main()
