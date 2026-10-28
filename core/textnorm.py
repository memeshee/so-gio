"""Vietnamese text normalisation and word-level comparison.

Vietnamese is written in syllables separated by spaces, so a "word" diff over syllables is both what a
reader perceives and what the speech models disagree about. Everything here is deliberately plain:
NFC-normalise, casefold, drop punctuation, collapse whitespace, keep `đ` (it is not a `d` with a mark).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Canonical form used for every comparison and every quote check."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    text = _PUNCT.sub(" ", text)
    text = text.casefold()
    return _WS.sub(" ", text).strip()


def syllables(text: str) -> list[str]:
    return normalise(text).split()


def contained(needle: str, haystack: str) -> bool:
    """Is `needle` literally present in `haystack` after normalisation? The grounding guard uses this."""
    n = normalise(needle)
    return bool(n) and n in normalise(haystack)


def edit_distance(a: list[str] | str, b: list[str] | str) -> int:
    """Levenshtein distance over syllables (or characters when strings are passed)."""
    if isinstance(a, str):
        a = list(a)
    if isinstance(b, str):
        b = list(b)
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate over syllables, against a reference transcript."""
    ref, hyp = syllables(reference), syllables(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


@dataclass
class DiffResult:
    agreed: list[str] = field(default_factory=list)
    disputed: list[tuple[list[str], list[str]]] = field(default_factory=list)
    only_a: list[str] = field(default_factory=list)
    only_b: list[str] = field(default_factory=list)

    @property
    def agreed_text(self) -> str:
        return " ".join(self.agreed)

    @property
    def agreement_rate(self) -> float:
        total = len(self.agreed) + len(self.disputed) + len(self.only_a) + len(self.only_b)
        return len(self.agreed) / total if total else 0.0


def diff_transcripts(text_a: str, text_b: str) -> DiffResult:
    """Word-level diff of two transcripts of the same audio.

    Where the models agree, the words are safe to quote. Where they disagree, the honest output is both
    readings, not a winner — that is the product's whole claim.
    """
    import difflib

    a, b = syllables(text_a), syllables(text_b)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out = DiffResult()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.agreed.extend(a[i1:i2])
        elif tag == "replace":
            out.disputed.append((a[i1:i2], b[j1:j2]))
        elif tag == "delete":
            out.only_a.extend(a[i1:i2])
        else:
            out.only_b.extend(b[j1:j2])
    return out
