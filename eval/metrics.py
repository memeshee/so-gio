"""One matching rule, shared by every extraction arm — including arms scored elsewhere.

`claims_match` is the single definition of "this claim and this gold item are the same fact".
The local arms in `eval/run_eval.py` and any externally scored arm (e.g. the Tinker arms written
to data/eval/tinker_*.json) must import this function rather than re-deriving it, or the table
becomes incomparable.

Why the rule looks like this: for a `gio_date` claim the quote is only the citation — the day and
month are the answer, so a model that quotes the whole sentence around the date is still correct,
and the quote is not compared at all. For a text claim (offering, instruction, person) the quote
*is* the answer, so containment in either direction or 60% syllable overlap accepts a longer
verbatim span but refuses a paraphrase.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import textnorm  # noqa: E402


def claims_match(model_claim: dict, gold: dict, overlap: float = 0.6) -> bool:
    if model_claim["kind"] != gold["kind"]:
        return False
    if gold["kind"] == "gio_date" and gold.get("lunar_day") is not None:
        return (model_claim.get("lunar_day") == gold.get("lunar_day")
                and model_claim.get("lunar_month") == gold.get("lunar_month"))
    a = textnorm.normalise(model_claim["quote"])
    b = textnorm.normalise(gold["quote"])
    if not a or not b:
        return False
    if b in a or a in b:
        return True
    gold_syl = set(b.split())
    return bool(gold_syl) and len(gold_syl & set(a.split())) / len(gold_syl) >= overlap


def score_claims(claims: list[dict], gold: list[dict]) -> dict:
    """Greedy one-to-one matching: each gold item is consumed at most once."""
    used = [False] * len(gold)
    tp = 0
    for claim in claims:
        for i, g in enumerate(gold):
            if used[i]:
                continue
            if claims_match(claim, g):
                used[i] = True
                tp += 1
                break
    precision = tp / len(claims) if claims else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "predicted": len(claims), "gold": len(gold),
            "precision": precision, "recall": recall, "f1": f1}
