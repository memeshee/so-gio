"""Baselines, so the model arms have something to lose to.

A published improvement is only meaningful next to a baseline that a sceptic would accept. Two are used:

- **`regex`** — pull Vietnamese lunar dates out of the transcript with a hand-written pattern for the
  spoken numbers a grandmother actually uses ("mười hai tháng bảy", "ngày rằm", "hai mươi tháng chạp").
  It cites perfectly and generalises not at all, which is the right shape for a baseline.
- **`all_dates`** — the degenerate arm: treat every lunar-looking phrase as a giỗ. It will have high
  recall and no judgement, and it is what "just use a regex" collapses to in the hands of a family.

Both run through the same guard as the model arms, so the comparison is like for like: every arm's output
must survive `core.guard` before it counts.
"""
from __future__ import annotations

import re

from core.extract import Claim

UNITS = {
    "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "lăm": 5,
    "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10,
}
SPECIAL_DAYS = {"rằm": 15, "mồng một": 1, "mùng một": 1, "mồng 1": 1, "ba mươi": 30, "ba muoi": 30}
MONTHS_LUNAR = re.compile(r"tháng\s+(mười\s+hai|mười\s+một|mười|một|hai|ba|bốn|tư|năm|sáu|bảy|tám|chín|\d{1,2})")
DAYS_LUNAR = re.compile(r"(?:ngày\s+)?(rằm|mồng\s+một|mùng\s+một|mồng\s+\w+|mùng\s+\w+|"
                        r"(?:hai\s+mươi|ba\s+mươi)(?:\s+\w+)?|mười\s+\w+|"
                        r"một|hai|ba|bốn|tư|năm|sáu|bảy|tám|chín|\d{1,2})")


def _parse_number(phrase: str) -> int | None:
    phrase = phrase.strip().lower()
    if phrase.isdigit():
        return int(phrase)
    if phrase in SPECIAL_DAYS:
        return SPECIAL_DAYS[phrase]
    words = phrase.split()
    if not words:
        return None
    if words[0] in ("hai", "ba") and len(words) > 1 and words[1] == "mươi":
        tens = UNITS.get(words[0], 0) * 10
        if len(words) > 2:
            return tens + UNITS.get(words[2], 0)
        return tens
    if words[0] == "mười":
        return 10 if len(words) == 1 else 10 + UNITS.get(words[1], 0)
    if len(words) == 1:
        return UNITS.get(words[0])
    return None


def regex_baseline(pieces: list[tuple[float, float, str]], all_dates: bool = False) -> list[Claim]:
    """Extract lunar dates with a pattern, no model.

    `all_dates=False` requires a month to be present (a giỗ is a date, not a day of the week).
    `all_dates=True` emits a claim for every date-shaped phrase it sees, which is the degenerate arm.
    """
    claims: list[Claim] = []
    for start, end, text in pieces:
        low = text.lower()
        for m in MONTHS_LUNAR.finditer(low):
            month = _parse_number(m.group(1))
            window = low[max(0, m.start() - 40): m.start()]
            days = list(DAYS_LUNAR.finditer(window))
            day = _parse_number(days[-1].group(1)) if days else None
            if month is None or day is None:
                continue
            quote = text[max(0, m.start() - 40): m.end()].strip()
            claims.append(Claim(kind="gio_date", quote=quote, lunar_day=day, lunar_month=month))
        if all_dates:
            for d in DAYS_LUNAR.finditer(low):
                day = _parse_number(d.group(1))
                if day is None:
                    continue
                claims.append(Claim(kind="gio_date", quote=text.strip(),
                                    lunar_day=day, lunar_month=1))
    return claims
