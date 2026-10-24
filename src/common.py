import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA, CACHE, RESULTS = ROOT / "data", ROOT / "cache", ROOT / "results"

_ANS = re.compile(r"answer\s*(?:is)?\s*[:：]?\s*\**\s*\(?([ABCD])\)?\b", re.I)


def parse_answer(text):
    """Last 'Answer: X' in the text; falls back to a lone trailing (X)."""
    if not text:
        return None
    m = _ANS.findall(text)
    if m:
        return m[-1].upper()
    m = re.findall(r"\(([ABCD])\)", text[-200:])
    return m[-1] if m else None


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(p, row):
    with open(p, "a") as f:
        f.write(json.dumps(row) + "\n")
