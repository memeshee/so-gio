"""The layer that decides. Nothing reaches the family unless it survives this file.

Three rules, applied in order, and each one is a refusal rather than a correction:

1. **The quote must exist.** The claim's words have to be a normalised substring of the text that both
   speech models agreed on. A sentence the model composed is dropped, not fixed.
2. **The quote must have a timestamp.** We locate it in a segment; without a moment in the recording there
   is nothing for the family to check the claim against, so it is dropped.
3. **A date must be computable.** A lunar day/month that the calendar cannot resolve (day 31, month 13, a
   leap month that does not exist that year) is dropped rather than rendered as a plausible-looking date.

Every refusal is kept and counted, because "we invented nothing" is only credible next to the list of things
we refused to invent.
"""
from __future__ import annotations

from dataclasses import dataclass

from core import textnorm, lunar
from core.extract import Claim


@dataclass
class Verdict:
    kept: list[Claim]
    refused: list[Claim]

    @property
    def uncited_kept(self) -> int:
        return sum(1 for c in self.kept if c.t_start is None or not c.quote)

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        for c in self.kept:
            kinds[c.kind] = kinds.get(c.kind, 0) + 1
        return {
            "kept": len(self.kept),
            "refused": len(self.refused),
            "uncited_kept": self.uncited_kept,
            "kinds": kinds,
            "refused_reasons": sorted({c.reason for c in self.refused}),
        }


def _locate(quote: str, pieces: list[tuple[float, float, str]]) -> tuple[float, float] | None:
    """Find the moment a quote was said. `pieces` is [(start, end, text)]."""
    needle = textnorm.normalise(quote)
    if not needle:
        return None
    # Prefer a single segment containing the whole quote; fall back to the run of consecutive segments
    # that together contain it, which is what a sentence spanning two ASR chunks looks like.
    for start, end, text in pieces:
        if needle in textnorm.normalise(text):
            return start, end
    joined, spans = [], []
    for start, end, text in pieces:
        joined.append(textnorm.normalise(text))
        spans.append((start, end))
        if needle in " ".join(joined):
            return spans[0][0], spans[-1][1]
    return None


def review(claims: list[Claim], pieces: list[tuple[float, float, str]], today: tuple[int, int, int],
           grounded_text: str | None = None) -> Verdict:
    """Keep what can be cited and computed; refuse the rest, with a reason."""
    grounded = grounded_text if grounded_text is not None else " ".join(p[2] for p in pieces)
    kept: list[Claim] = []
    refused: list[Claim] = []
    seen: set[str] = set()

    for claim in claims:
        key = (claim.kind, textnorm.normalise(claim.quote), claim.lunar_day, claim.lunar_month)
        if str(key) in seen:
            claim.status, claim.reason = "refused", "duplicate"
            refused.append(claim)
            continue
        seen.add(str(key))

        if not textnorm.contained(claim.quote, grounded):
            claim.status, claim.reason = "refused", "not in the recording"
            refused.append(claim)
            continue

        where = _locate(claim.quote, pieces)
        if where is None:
            claim.status, claim.reason = "refused", "no timestamp for the quote"
            refused.append(claim)
            continue
        claim.t_start, claim.t_end = where

        if claim.kind == "gio_date":
            day, month = claim.lunar_day, claim.lunar_month
            if day is None or month is None:
                claim.status, claim.reason = "refused", "date named without a lunar day and month"
                refused.append(claim)
                continue
            if not (1 <= day <= 30 and 1 <= month <= 12):
                claim.status, claim.reason = "refused", f"impossible lunar date {day}/{month}"
                refused.append(claim)
                continue
            try:
                nxt = lunar.next_occurrence(month, day, today, years_ahead=1, leap=claim.leap)
            except ValueError as exc:
                claim.status, claim.reason = "refused", f"calendar cannot resolve it: {exc}"
                refused.append(claim)
                continue
            if not nxt:
                claim.status, claim.reason = "refused", "no upcoming occurrence found"
                refused.append(claim)
                continue
            y, m, d = nxt[0]
            claim.solar_next = f"{y:04d}-{m:02d}-{d:02d}"

        claim.status = "kept"
        kept.append(claim)

    return Verdict(kept=kept, refused=refused)
