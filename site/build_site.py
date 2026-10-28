#!/usr/bin/env python3
"""Build the Sổ Giỗ demo site from the run + oracle data that actually exists.

    /home/kiter/.pyenv/versions/3.12.6/bin/python site/build_site.py
    /home/kiter/.pyenv/versions/3.12.6/bin/python site/build_site.py --run smoke-001

Reads (never writes):
    data/runs/<id>/run.json          the pipeline run (the richest one is chosen)
    data/runs/<id>/reading_copy.md   the human reading copy, when the pipeline wrote it
    data/oracle/report.json          calendar validation against USNO + published VN dates
    data/eval/tables.md              optional; rendered only if it appears

Writes:
    site/dist/index.html
    site/dist/reading-copy.html
    site/dist/calendar.html
    site/dist/assets/site.css, site/dist/assets/site.js, site/dist/assets/fonts/*
    site/dist/favicon.svg, favicon-32.png, favicon-180.png
    site/dist/.nojekyll

Standard library only. No network. No JavaScript framework. Every digit on the page is
either read from data/ or computed by core/lunar.py (the same engine the pipeline uses).
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import pathlib
import re
import shutil
import struct
import sys
import zlib
from html.parser import HTMLParser

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
DIST = SITE / "dist"
ASSETS = SITE / "assets"
TZ_VN = dt.timezone(dt.timedelta(hours=7))

WEEKDAY_VN = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]

KIND_VN = {
    "person": "người",
    "gio_date": "ngày giỗ",
    "offering": "đồ cúng",
    "instruction": "cách làm",
}

REASON_VN = {
    "not in the recording": "không có trong băng",
    "no timestamp for the quote": "không tìm được mốc thời gian",
    "date named without a lunar day and month": "nói ngày nhưng thiếu ngày hoặc tháng âm",
    "duplicate": "trùng với một lời khác",
    "no upcoming occurrence found": "không có lần giỗ nào sắp tới",
}
REASON_VN_PREFIX = [
    ("impossible lunar date", "ngày âm không tồn tại"),
    ("calendar cannot resolve it", "lịch không giải được"),
]


# --------------------------------------------------------------------------- text helpers

def esc(text: object) -> str:
    """Escape for a text node."""
    return html.escape(str(text), quote=False)


def attr(text: object) -> str:
    """Escape for an attribute value."""
    return html.escape(str(text), quote=True)


def vn_float(value: float, digits: int = 1) -> str:
    """Vietnamese decimal comma, no thousands separator for the small numbers we print."""
    text = f"{value:.{digits}f}"
    return text.replace(".", ",")


def vn_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def mmss(seconds: float | None) -> str:
    total = int(round(seconds or 0))
    return f"{total // 60:02d}:{total % 60:02d}"


def time_chip(start: float | None, end: float | None) -> str:
    if start is None:
        return ""
    if end is None or abs(end - start) < 0.05:
        label = mmss(start)
    else:
        label = f"{mmss(start)}–{mmss(end)}"
    return f'<span class="chip">[{esc(label)}]</span>'


def date_vn(year: int, month: int, day: int) -> str:
    return f"{day} tháng {month} năm {year}"


def date_short(year: int, month: int, day: int) -> str:
    return f"{day:02d}/{month:02d}/{year}"


def weekday_vn(year: int, month: int, day: int) -> str:
    return WEEKDAY_VN[dt.date(year, month, day).weekday()]


def iso(year: int, month: int, day: int) -> str:
    return f"{year:04d}-{month:02d}-{day:02d}"


def json_load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- data

def discover_runs() -> list[dict]:
    """Every run that parses, newest score first. Never trusts a half-written file."""
    found = []
    for path in sorted((ROOT / "data" / "runs").glob("*/run.json")):
        try:
            run = json_load(path)
        except Exception as exc:  # a sibling process may be mid-write
            print(f"  skip {path.relative_to(ROOT)}: {exc}", file=sys.stderr)
            continue
        found.append({
            "id": path.parent.name,
            "path": path,
            "run": run,
            "mtime": path.stat().st_mtime,
        })
    return found


def run_score(run: dict) -> int:
    """Prefer the run that can actually demonstrate the product: dates, refusals, inventions."""
    kept = run.get("claims_kept") or []
    return (
        len(run.get("calendar") or []) * 100
        + sum(1 for c in kept if c.get("kind") == "gio_date") * 50
        + len(run.get("claims_refused") or []) * 10
        + len(run.get("invented") or []) * 5
        + len(kept) * 2
    )


def choose_run(explicit: str | None) -> dict:
    runs = discover_runs()
    if not runs:
        raise SystemExit("no data/runs/*/run.json found — run the pipeline first")
    if explicit:
        for item in runs:
            if item["id"] == explicit:
                return item
        raise SystemExit(f"run {explicit!r} not found in data/runs/")
    runs.sort(key=lambda r: (run_score(r["run"]), r["mtime"]), reverse=True)
    return runs[0]


SAFE_RE = re.compile(r"^- `(\d{2}:\d{2})` (.*)$")
INVENTED_RE = re.compile(r'^- `(\d{2}:\d{2})` \*(whisper|phowhisper) invented:\* "(.*)"$')
DISPUTED_RE = re.compile(r'^- whisper: "(.*)" · phowhisper: "(.*)"$')


def parse_reading_copy(path: pathlib.Path) -> dict | None:
    """Parse the pipeline's reading_copy.md into lines, inventions, disagreements.

    The format is written by pipeline.render_reading_copy(); unknown lines are ignored on purpose
    so a future addition to that file cannot break this build.
    """
    if not path.exists():
        return None
    safe: list[tuple[str, str]] = []
    invented: list[dict] = []
    disputed: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        # invented lines also start with "`mm:ss`", so they must be tested before the safe-line pattern
        if match := INVENTED_RE.match(line):
            invented.append({"t": match.group(1), "who": match.group(2), "text": match.group(3)})
        elif match := SAFE_RE.match(line):
            safe.append((match.group(1), match.group(2).strip()))
        elif match := DISPUTED_RE.match(line):
            disputed.append((match.group(1), match.group(2)))
    return {"safe": safe, "invented": invented, "disputed": disputed}


def load_oracle() -> dict:
    path = ROOT / "data" / "oracle" / "report.json"
    if not path.exists():
        raise SystemExit("data/oracle/report.json missing — the page will not claim a checked calendar without it")
    return json_load(path)


def build_model(run_item: dict, reading: dict | None, oracle: dict) -> dict:
    run = run_item["run"]
    kept = run.get("claims_kept") or []
    refused = run.get("claims_refused") or []
    calendar = run.get("calendar") or []
    invented = run.get("invented") or []
    arms = run.get("arms") or []
    silences = run.get("silences") or []
    quotes = run.get("quotable_spans") or []

    if reading and reading["safe"]:
        safe = reading["safe"]
    else:
        safe = [(mmss(q.get("start")), q.get("text", "")) for q in quotes if q.get("text")]
    disputed = reading["disputed"] if reading else []

    # Who invented each span: the reading copy knows, run.json intentionally does not.
    invented_rows = []
    for row in invented:
        who = "một mô hình"
        for guess in (reading["invented"] if reading else []):
            if guess["text"].strip() == (row.get("text") or "").strip():
                who = guess["who"]
                break
        span = {"text": row.get("text", ""), "start": row.get("start"), "end": row.get("end"), "who": who}
        covering = [
            s for s in silences
            if row.get("start") is not None and s.get("start") is not None
            and row["start"] < s.get("end", 0) and row.get("end", 0) > s.get("start", 0)
        ]
        span["silence"] = covering[0] if covering else None
        invented_rows.append(span)

    nm = oracle.get("new_moons") or {}
    anchors = oracle.get("anchors") or {}
    deltas = [abs(float(r.get("delta_seconds") or 0.0)) for r in (nm.get("rows") or [])]
    oracle_summary = {
        "moons_n": nm.get("n", len(nm.get("rows") or [])),
        "moons_within": nm.get("within_2_minutes", 0),
        "moons_worst": nm.get("worst_abs_seconds", max(deltas) if deltas else 0.0),
        "moons_mean": (sum(deltas) / len(deltas)) if deltas else 0.0,
        "anchor_n": anchors.get("n", len(anchors.get("rows") or [])),
        "anchor_passed": anchors.get("passed", 0),
        "structure_years": (oracle.get("structure") or {}).get("years", 0),
        "structure_violations": len((oracle.get("structure") or {}).get("violations") or []),
    }

    return {
        "run_id": run_item["id"],
        "run": run,
        "arms": arms,
        "kept": kept,
        "refused": refused,
        "calendar": calendar,
        "invented": invented_rows,
        "silences": silences,
        "safe": safe,
        "disputed": disputed,
        "oracle": oracle,
        "oracle_summary": oracle_summary,
        "build_time": dt.datetime.now(TZ_VN),
    }


# --------------------------------------------------------------------------- optional eval

def mini_md(text: str, heading_offset: int = 0) -> str:
    """A deliberately small markdown renderer: headings, tables, lists, paragraphs, **bold**, `code`.

    Anything it does not understand is shown as text, never dropped. Links are rendered as their
    label so the built page cannot acquire an outbound request.
    """
    try:
        lines = text.replace("\r\n", "\n").split("\n")
        out: list[str] = []
        i = 0
        para: list[str] = []

        def inline(value: str) -> str:
            value = esc(value)
            value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
            value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
            value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
            return value

        def flush() -> None:
            if para:
                out.append("<p>" + inline(" ".join(para)).replace("  ", "<br>") + "</p>")
                para.clear()

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                flush()
                i += 1
                continue
            if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|?$", lines[i + 1].strip()):
                flush()
                header = [c.strip() for c in stripped.strip("|").split("|")]
                i += 2
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                    i += 1
                out.append('<div class="table-wrap"><table class="data"><thead><tr>')
                out.extend(f"<th>{inline(c)}</th>" for c in header)
                out.append("</tr></thead><tbody>")
                for row in rows:
                    out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
                out.append("</tbody></table></div>")
                continue
            if stripped.startswith("#"):
                flush()
                level = min(6, len(stripped) - len(stripped.lstrip("#")) + heading_offset)
                out.append(f"<h{level}>{inline(stripped[level:].strip())}</h{level}>")
                i += 1
                continue
            if re.match(r"^[-*] ", stripped):
                flush()
                items = []
                while i < len(lines) and re.match(r"^[-*] ", lines[i].strip()):
                    items.append(inline(lines[i].strip()[2:]))
                    i += 1
                out.append("<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>")
                continue
            para.append(stripped)
            i += 1
        flush()
        return "\n".join(out)
    except Exception:
        return f"<pre>{esc(text)}</pre>"


# --------------------------------------------------------------------------- favicon

RED = (0x8C, 0x2F, 0x27)
PAPER = (0xF7, 0xF2, 0xE8)


def _inside_rr(x: float, y: float, x0: float, y0: float, x1: float, y1: float, r: float) -> bool:
    if x < x0 or x > x1 or y < y0 or y > y1:
        return False
    cx = min(max(x, x0 + r), x1 - r)
    cy = min(max(y, y0 + r), y1 - r)
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def _dist_to_segment(x: float, y: float, x0: float, y0: float, x1: float, y1: float) -> float:
    dx, dy = x1 - x0, y1 - y0
    t = 0.0 if (dx == 0 and dy == 0) else max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy)))
    return ((x - (x0 + t * dx)) ** 2 + (y - (y0 + t * dy)) ** 2) ** 0.5


def seal_svg(size: int, color: str = "#1b1a17", paper: str = "#f7f2e8", *, label: str | None = None) -> str:
    """The family seal: a lacquer square, a cream rule, one dot for the date, one line to stand it on.

    Drawn at 64 units so the same geometry is used by the favicon, the masthead and the colophon.
    """
    title = f"<title>{esc(label)}</title>" if label else ""
    role = f' role="img" aria-label="{attr(label)}"' if label else ' aria-hidden="true" focusable="false"'
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 64 64"{role}>{title}
  <rect x="2" y="2" width="60" height="60" rx="6" fill="{color}"/>
  <rect x="8.5" y="8.5" width="47" height="47" rx="2.8" fill="none" stroke="{paper}" stroke-width="2.9"/>
  <circle cx="32" cy="31.5" r="5.3" fill="{paper}"/>
  <path d="M23 43h18" stroke="{paper}" stroke-width="2.5" stroke-linecap="round" fill="none"/>
</svg>"""


FAVICON_SVG = seal_svg(64, "#8c2f27", "#f7f2e8", label="Sổ Giỗ — con dấu của gia đình") + "\n"


def favicon_rgba(size: int) -> list[bytes]:
    """Rasterise the same seal: lacquer square, cream frame, the dot and its line."""
    ss = 4
    unit = 64.0 / size
    rows: list[bytes] = []
    for py in range(size):
        row = bytearray()
        for px in range(size):
            sr = sg = sb = sa = 0.0
            for sy in range(ss):
                for sx in range(ss):
                    x = (px + (sx + 0.5) / ss) * unit
                    y = (py + (sy + 0.5) / ss) * unit
                    if not _inside_rr(x, y, 2, 2, 62, 62, 6.0):
                        continue
                    color = RED
                    if (_inside_rr(x, y, 7.05, 7.05, 56.95, 56.95, 4.25)
                            and not _inside_rr(x, y, 9.95, 9.95, 54.05, 54.05, 1.35)):
                        color = PAPER
                    if (x - 32.0) ** 2 + (y - 31.5) ** 2 <= 5.3 ** 2:
                        color = PAPER
                    if _dist_to_segment(x, y, 23, 43, 41, 43) <= 1.25:
                        color = PAPER
                    sr += color[0]
                    sg += color[1]
                    sb += color[2]
                    sa += 1.0
            n = ss * ss
            alpha = sa / n
            if alpha <= 0:
                row.extend(b"\x00\x00\x00\x00")
            else:
                row.extend(bytes((
                    int(round(sr / sa)),
                    int(round(sg / sa)),
                    int(round(sb / sa)),
                    int(round(alpha * 255)),
                )))
        rows.append(bytes(row))
    return rows


def write_png(path: pathlib.Path, size: int) -> None:
    rows = favicon_rgba(size)
    raw = bytearray()
    for row in rows:
        raw.append(0)
        raw.extend(row)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


# --------------------------------------------------------------------------- css

CSS = r"""
@font-face {
  font-family: "So Gio Serif";
  src: url("fonts/LiberationSerif-Regular.woff2") format("woff2");
  font-weight: 400; font-style: normal; font-display: swap;
}
@font-face {
  font-family: "So Gio Serif";
  src: url("fonts/LiberationSerif-Italic.woff2") format("woff2");
  font-weight: 400; font-style: italic; font-display: swap;
}
@font-face {
  font-family: "So Gio Serif";
  src: url("fonts/LiberationSerif-Bold.woff2") format("woff2");
  font-weight: 700; font-style: normal; font-display: swap;
}

:root {
  --paper: #f7f2e8;
  --ink: #1b1a17;
  --ink-2: #413b32;
  --ink-3: #6b6254;
  --rule: rgba(27, 26, 23, 0.22);
  --rule-soft: rgba(27, 26, 23, 0.11);
  --red: #8c2f27;
  --serif: "So Gio Serif", "Iowan Old Style", "Palatino Linotype", "Noto Serif",
           "DejaVu Serif", Georgia, "Times New Roman", serif;
  --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue",
          Arial, "Noto Sans", "Liberation Sans", sans-serif;
  --measure: 34em;
  --pad: 1.15rem;
}

* { box-sizing: border-box; }

html {
  color-scheme: light;
  -webkit-text-size-adjust: 100%;
  scroll-behavior: smooth;
}

body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: var(--serif);
  font-size: 1.0625rem;
  line-height: 1.72;
  font-kerning: normal;
  font-variant-ligatures: common-ligatures;
  text-rendering: optimizeLegibility;
}

@media (min-width: 640px) {
  body { font-size: 1.15625rem; line-height: 1.76; }
}

::selection { background: var(--ink); color: var(--paper); }

a { color: inherit; text-decoration-color: rgba(27, 26, 23, 0.4); text-underline-offset: 3px; }
a:hover { text-decoration-color: var(--ink); }

p { margin: 0; }

.sheet {
  width: 100%;
  max-width: calc(var(--measure) + var(--pad) * 2);
  margin: 0 auto;
  padding: 1.6rem var(--pad) 3rem;
}
@media (min-width: 700px) {
  .sheet { padding-top: 3.4rem; padding-bottom: 4.5rem; }
}

.skip {
  position: absolute; left: -999px; top: 0;
  font-family: var(--sans); font-size: 0.8125rem;
  background: var(--paper); color: var(--ink); padding: 0.6rem 0.8rem; z-index: 10;
}
.skip:focus { left: 0.6rem; top: 0.6rem; }

/* ------------------------------------------------------------------ masthead */

.runhead {
  display: flex; justify-content: space-between; align-items: baseline; gap: 1rem;
  font-family: var(--sans); font-size: 0.6875rem; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--ink-3);
  border-bottom: 1px solid var(--rule);
  padding-bottom: 0.55rem; margin: 0 0 2.6rem;
}
.runhead b { font-weight: 400; color: var(--ink-2); letter-spacing: 0.1em; }

.seal { margin: 0 0 1.35rem; line-height: 0; }

.masthead h1 {
  font-size: clamp(2.55rem, 9.5vw, 3.9rem);
  line-height: 0.98;
  letter-spacing: -0.024em;
  font-weight: 400;
  margin: 0 0 1.15rem;
  text-wrap: balance;
}

.deck {
  font-size: 1.125rem; line-height: 1.58; color: var(--ink-2);
  max-width: 30em; margin: 0 0 1.9rem; text-wrap: pretty;
}
@media (min-width: 640px) { .deck { font-size: 1.25rem; } }

.imprint {
  display: grid; grid-template-columns: auto 1fr; gap: 0.3rem 0.9rem;
  margin: 0 0 0.4rem; padding: 0.85rem 0;
  border-top: 1px solid var(--rule-soft); border-bottom: 1px solid var(--rule-soft);
  font-family: var(--sans); font-size: 0.75rem; line-height: 1.5; color: var(--ink-2);
}
.imprint dt {
  font-size: 0.65625rem; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--ink-3); padding-top: 0.22em;
}
.imprint dd { margin: 0; overflow-wrap: anywhere; }

@media (max-width: 480px) {
  .imprint { grid-template-columns: 1fr; gap: 0; }
  .imprint dt { padding-top: 0.55rem; }
  .imprint dd { margin: 0 0 0.4rem; }
}

.contents {
  display: flex; flex-wrap: wrap; gap: 0.35rem 0.95rem;
  margin: 1.15rem 0 0;
  font-family: var(--sans); font-size: 0.75rem; letter-spacing: 0.01em;
}
.contents a { color: var(--ink-2); text-decoration: none; border-bottom: 1px solid transparent; padding-bottom: 1px; }
.contents a:hover { border-bottom-color: var(--rule); }

/* ------------------------------------------------------------------ chapters */

.chapter {
  margin-top: 3.3rem; border-top: 1px solid var(--rule); padding-top: 1rem;
}
@media (min-width: 700px) { .chapter { margin-top: 4.9rem; padding-top: 1.2rem; } }

.ch-no {
  font-family: var(--sans); font-size: 0.6875rem; font-weight: 400;
  letter-spacing: 0.13em; color: var(--ink-3);
  display: inline-block; margin-right: 0.35em; vertical-align: 0.55em;
  font-variant-numeric: tabular-nums;
}

.chapter h2 {
  font-weight: 400; font-size: 1.55rem; line-height: 1.2; letter-spacing: -0.012em;
  margin: 0 0 0.3rem; text-wrap: balance;
}
@media (min-width: 640px) { .chapter h2 { font-size: 1.85rem; } }

.ch-en {
  font-family: var(--sans); font-size: 0.6875rem; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--ink-3); margin: 0 0 1.25rem;
}

.lede {
  font-size: 1.0625rem; line-height: 1.6; color: var(--ink-2);
  margin: 0 0 1.7rem; max-width: 33em; text-wrap: pretty;
}
@media (min-width: 640px) { .lede { font-size: 1.125rem; } }

.prose p { margin: 0 0 1.05em; text-wrap: pretty; }
.prose p:last-child { margin-bottom: 0; }
.prose .en { color: var(--ink-2); font-size: 0.95em; line-height: 1.65; }

/* ------------------------------------------------------------------ figures */

.figures {
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1.35rem 1rem; margin: 0 0 1.2rem;
  border-top: 1px solid var(--rule); padding-top: 1.25rem;
}
@media (min-width: 620px) { .figures { grid-template-columns: repeat(4, minmax(0, 1fr)); } }

.figure .n {
  display: block; font-size: 1.85rem; line-height: 1;
  letter-spacing: -0.015em; font-variant-numeric: tabular-nums;
}
.figure .n .u { font-size: 0.66em; color: var(--ink-2); margin-left: 0.02em; letter-spacing: 0; }
.figure .l {
  display: block; margin-top: 0.5rem;
  font-family: var(--sans); font-size: 0.71875rem; line-height: 1.4;
  color: var(--ink-3); letter-spacing: 0.01em;
}
.figures-note {
  font-family: var(--sans); font-size: 0.75rem; color: var(--ink-3);
  margin: 0 0 2.1rem; display: flex; gap: 0.5rem; align-items: baseline; flex-wrap: wrap;
}

/* ------------------------------------------------------------------ arms */

.arms { border-top: 1px solid var(--rule); }
.arm { padding: 0.95rem 0; border-bottom: 1px solid var(--rule-soft); }
.arm-top { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; }
.arm-name {
  font-family: var(--sans); font-size: 0.75rem; letter-spacing: 0.13em; text-transform: uppercase;
}
.arm-rtf { font-family: var(--sans); font-size: 0.8125rem; font-variant-numeric: tabular-nums; color: var(--ink-2); white-space: nowrap; }
.arm-rtf .x { font-size: 0.6875rem; color: var(--ink-3); letter-spacing: 0.02em; }
.arm-model {
  font-family: var(--sans); font-size: 0.8125rem; color: var(--ink-2);
  margin: 0.18rem 0 0; overflow-wrap: anywhere;
}
.arm-sub {
  font-family: var(--sans); font-size: 0.75rem; color: var(--ink-3);
  margin: 0.28rem 0 0; font-variant-numeric: tabular-nums;
}
.small-note {
  font-family: var(--sans); font-size: 0.75rem; line-height: 1.55; color: var(--ink-3);
  margin: 0.85rem 0 0;
}
.small-note .chip { margin-right: 0.4em; }

.licence {
  margin: 2rem 0 0; padding: 1rem 0 1rem 1rem;
  border-left: 1px solid var(--rule);
  font-size: 0.9375rem; line-height: 1.6; color: var(--ink-2);
}
.licence strong { font-weight: 400; color: var(--ink); }

/* ------------------------------------------------------------------ chips */

.chip {
  font-family: var(--sans); font-size: 0.75rem; line-height: 1.4;
  font-variant-numeric: tabular-nums; letter-spacing: 0.01em;
  color: var(--ink-2); white-space: nowrap;
}
.chip a { text-decoration: none; }
a.chip { text-decoration: none; border-bottom: 1px solid var(--rule); }
a.chip:hover { border-bottom-color: var(--ink-2); }

/* ------------------------------------------------------------------ transcript */

.copy { margin: 1.4rem 0 0.6rem; }
.line { padding: 0 0 1.05rem; }
.line .t { margin-right: 0.5em; }
.line p { display: inline; }
@media (min-width: 560px) {
  .line { display: grid; grid-template-columns: 5.6rem minmax(0, 1fr); gap: 0.85rem; padding-bottom: 1.2rem; }
  .line .t { margin-right: 0; padding-top: 0.42em; }
  .line p { display: block; }
}
.copy-text { text-wrap: pretty; }

.disputed {
  color: var(--ink-3);
  text-decoration: underline dotted rgba(27, 26, 23, 0.5);
  text-decoration-thickness: 1px;
  text-underline-offset: 3px;
}
.disputed .fn { font-family: var(--sans); font-size: 0.64em; text-decoration: none; margin-left: 0.05em; }
.disputed .fn a { text-decoration: none; color: var(--ink-3); }

.toggle {
  display: none;
  align-items: baseline; gap: 0.55rem;
  font-family: var(--sans); font-size: 0.75rem; color: var(--ink-2);
  margin: 0 0 1.2rem; cursor: pointer; user-select: none;
}
.js .toggle { display: flex; }
.toggle input { accent-color: var(--ink); width: 0.95em; height: 0.95em; margin: 0; }
.hide-marks .disputed { color: inherit; text-decoration: none; }
.hide-marks .disputed .fn { display: none; }
.hide-marks .notes { display: none; }

.notes {
  list-style: none; counter-reset: note;
  margin: 0.4rem 0 1.6rem; padding: 1rem 0 0;
  border-top: 1px solid var(--rule-soft);
  font-family: var(--sans); font-size: 0.75rem; line-height: 1.55; color: var(--ink-2);
}
.notes li {
  counter-increment: note;
  display: grid; grid-template-columns: 1.3rem minmax(0, 1fr); gap: 0.35rem;
  padding: 0.3rem 0;
}
.notes li::before { content: counter(note); color: var(--ink-3); font-variant-numeric: tabular-nums; }
.notes .w { text-transform: uppercase; font-size: 0.65625rem; letter-spacing: 0.1em; color: var(--ink-3); }
.notes .back { margin-left: 0.4rem; text-decoration: none; }
.notes .back:hover { text-decoration: underline; }

/* ------------------------------------------------------------------ alarm */

.alarm {
  margin: 1.5rem 0; padding: 0.1rem 0 0.1rem 1rem;
  border-left: 2px solid var(--red);
}
.alarm-label {
  font-family: var(--sans); font-size: 0.625rem; letter-spacing: 0.13em; text-transform: uppercase;
  color: var(--red); margin: 0 0 0.5rem;
}
.alarm-text {
  font-size: 1.0625rem; line-height: 1.6; margin: 0; color: var(--ink);
  overflow-wrap: anywhere;
}
.alarm del { text-decoration-thickness: 1.5px; text-decoration-color: var(--ink-3); }
.alarm-note {
  font-family: var(--sans); font-size: 0.75rem; line-height: 1.55; color: var(--ink-3);
  margin: 0.55rem 0 0;
}

.zero {
  margin: 1.4rem 0 0; padding: 0.9rem 0;
  border-top: 1px solid var(--rule-soft); border-bottom: 1px solid var(--rule-soft);
  font-size: 0.9375rem; line-height: 1.6; color: var(--ink-2);
}
.zero b { font-weight: 400; color: var(--ink); }
.zero .lbl {
  display: block; font-family: var(--sans); font-size: 0.625rem; letter-spacing: 0.13em;
  text-transform: uppercase; color: var(--ink-3); margin-bottom: 0.35rem;
}

/* ------------------------------------------------------------------ claims */

.claim { padding: 1.55rem 0; border-bottom: 1px solid var(--rule-soft); }
.claim:first-of-type { border-top: 1px solid var(--rule); }
.kind {
  font-family: var(--sans); font-size: 0.6875rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--ink-3); margin: 0 0 0.55rem;
}
.kind .en { letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); }
.claim blockquote { margin: 0 0 0.75rem; }
.claim blockquote p {
  font-size: 1.3rem; line-height: 1.48; text-indent: -0.4em; text-wrap: pretty;
}
@media (min-width: 640px) { .claim blockquote p { font-size: 1.5rem; } }
.claim-cite {
  display: flex; align-items: baseline; gap: 0.6rem; flex-wrap: wrap;
  font-family: var(--sans); font-size: 0.75rem; color: var(--ink-3); margin: 0;
}
.claim-date { margin: 0.9rem 0 0; font-size: 1.0625rem; color: var(--ink-2); }
.claim-date time { color: var(--red); font-variant-numeric: tabular-nums; }
.claim-date .lbl {
  font-family: var(--sans); font-size: 0.65625rem; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--ink-3); margin-right: 0.15rem;
}
.claim-meta {
  margin: 0.7rem 0 0; font-family: var(--sans); font-size: 0.75rem; color: var(--ink-3);
}

.refusal { padding: 1.45rem 0; border-bottom: 1px solid var(--rule-soft); }
.refusal:first-of-type { border-top: 1px solid var(--rule); }
.refusal blockquote { margin: 0 0 0.7rem; font-size: 1.0625rem; line-height: 1.62; color: var(--ink-2); }
.refusal blockquote p { text-indent: -0.38em; }
.refusal s { text-decoration-color: rgba(27, 26, 23, 0.45); }
.reason {
  font-family: var(--sans); font-size: 0.75rem; line-height: 1.5; color: var(--ink-2);
  display: flex; gap: 0.45rem; flex-wrap: wrap; align-items: baseline;
}
.reason .rvn { text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.625rem; color: var(--ink-3); }
.reason .raw { color: var(--ink-3); font-size: 0.6875rem; }

/* ------------------------------------------------------------------ calendar */

.cal { margin: 0 0 2.4rem; }
.cal-head {
  display: flex; align-items: baseline; justify-content: space-between; gap: 0.8rem;
  border-bottom: 1px solid var(--rule); padding-bottom: 0.45rem;
}
.cal-head h3 { font-weight: 400; font-size: 1.1875rem; line-height: 1.3; margin: 0; }
.cal-head .lunar { font-family: var(--sans); font-size: 0.75rem; color: var(--ink-3); white-space: nowrap; }
.cal-quote {
  font-size: 0.9375rem; color: var(--ink-3); margin: 0.4rem 0 0.3rem;
  font-style: italic;
}
.cal-note {
  font-family: var(--sans); font-size: 0.75rem; color: var(--ink-2);
  margin: 0.55rem 0 0;
}
.years { list-style: none; margin: 0.35rem 0 0; padding: 0; }
.years li {
  display: grid; grid-template-columns: 3.1rem minmax(0, 1fr) auto;
  gap: 0.65rem; align-items: baseline;
  padding: 0.56rem 0; border-bottom: 1px solid var(--rule-soft);
  font-variant-numeric: tabular-nums;
}
.years .y { font-family: var(--sans); font-size: 0.75rem; color: var(--ink-3); letter-spacing: 0.04em; }
.years .d { font-size: 1.0625rem; }
.years .w { font-family: var(--sans); font-size: 0.75rem; color: var(--ink-3); text-align: right; white-space: nowrap; }
.years li.next .d { color: var(--red); }
.years li.checked .d { color: var(--red); }
.years li.next .w { color: var(--red); }

.example {
  margin: 2rem 0 0; padding-top: 1.2rem; border-top: 1px solid var(--rule);
}
.example-label {
  font-family: var(--sans); font-size: 0.65625rem; letter-spacing: 0.13em; text-transform: uppercase;
  color: var(--ink-3); margin: 0 0 0.5rem;
}
.example h3 { font-weight: 400; font-size: 1.1875rem; margin: 0 0 0.35rem; }
.example .why {
  font-family: var(--sans); font-size: 0.75rem; line-height: 1.55; color: var(--ink-3);
  margin: 0.55rem 0 0;
}

/* ------------------------------------------------------------------ tables */

.table-wrap { width: 100%; max-width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; }
.table-wrap.spaced { margin-top: 1.7rem; }
table.data {
  border-collapse: collapse; width: 100%;
  font-family: var(--sans); font-size: 0.84375rem; line-height: 1.45;
  font-variant-numeric: tabular-nums;
}
table.data caption .chip { text-transform: none; letter-spacing: 0.02em; }
table.data caption {
  caption-side: top; text-align: left; font-family: var(--sans);
  font-size: 0.625rem; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--ink-3); padding: 0 0 0.45rem;
}
table.data th {
  text-align: left; font-weight: 400; font-size: 0.625rem; letter-spacing: 0.11em;
  text-transform: uppercase; color: var(--ink-3);
  border-bottom: 1px solid var(--rule); padding: 0.35rem 0.6rem 0.4rem 0;
  vertical-align: bottom;
}
table.data td {
  border-bottom: 1px solid var(--rule-soft); padding: 0.55rem 0.6rem 0.55rem 0;
  vertical-align: top; overflow-wrap: break-word;
}
table.data th:last-child, table.data td:last-child { padding-right: 0; }
table.data .num { text-align: right; }
table.data .nowrap { white-space: nowrap; }
table.data .muted { color: var(--ink-3); }
table.data .ok { color: var(--ink-2); }
table.data tr.checked td { background: transparent; font-weight: 400; }
table.data tr.checked .ok { text-decoration: underline; text-decoration-color: var(--rule); }
table.data .arrow { color: var(--ink-3); padding: 0 0.1em; }

@media (max-width: 560px) {
  table.data { font-size: 0.8rem; }
  table.data th { font-size: 0.65625rem; letter-spacing: 0.06em; }
  table.data th, table.data td { padding-right: 0.4rem; }
  table.data .opt { display: none; }
  table.data caption .chip { text-transform: none; letter-spacing: 0.02em; }
table.data caption { font-size: 0.625rem; letter-spacing: 0.08em; }
}

.moons td { font-size: 0.78rem; }

/* ------------------------------------------------------------------ misc */

.more { margin: 1.4rem 0 0; font-family: var(--sans); font-size: 0.8125rem; }
.more a { text-decoration: none; border-bottom: 1px solid var(--rule); padding-bottom: 2px; }
.more a:hover { border-bottom-color: var(--ink-2); }

.empty {
  margin: 0 0 1rem; font-size: 1.0625rem; line-height: 1.62; color: var(--ink-2);
  max-width: 32em;
}
.empty b { font-weight: 400; color: var(--ink); }

.legend { margin: 1.4rem 0 1.8rem; padding: 0; border-top: 1px solid var(--rule-soft); }
.legend > div {
  display: grid; grid-template-columns: 8.2rem minmax(0, 1fr); gap: 0.8rem;
  padding: 0.5rem 0; border-bottom: 1px solid var(--rule-soft);
}
.legend dt { font-size: 0.95rem; }
.legend dd {
  margin: 0; font-family: var(--sans); font-size: 0.75rem; line-height: 1.5; color: var(--ink-3);
}

.pager {
  display: flex; flex-wrap: wrap; gap: 1rem 1.6rem;
  margin-top: 3.6rem; padding-top: 1rem;
  border-top: 1px solid var(--rule);
  font-family: var(--sans); font-size: 0.75rem;
}
.pager a { text-decoration: none; color: var(--ink-2); }
.pager a:hover { text-decoration: underline; }
.pager .no { color: var(--ink-3); }

.colophon {
  margin-top: 3.2rem; padding-top: 1.1rem; border-top: 1px solid var(--rule);
  font-family: var(--sans); font-size: 0.75rem; line-height: 1.6; color: var(--ink-3);
}
.colophon p { margin: 0 0 0.55rem; max-width: 40em; }
.colophon .mark { margin: 0 0 0.7rem; }
.colophon .mark .seal-mark { display: inline-block; vertical-align: -4px; margin-right: 0.4rem; line-height: 0; }
.colophon b { font-weight: 400; color: var(--ink-2); }

/* ------------------------------------------------------------------ print */

@media print {
  @page { size: A4; margin: 20mm 22mm 22mm; }
  html { font-size: 10.5pt; scroll-behavior: auto; }
  body { background: #fff; color: #000; font-size: 10.5pt; line-height: 1.55; }
  .sheet { max-width: 124mm; margin: 0 auto; padding: 0; }
  .skip, .pager, .contents, .toggle { display: none !important; }
  a { color: #000; text-decoration: none; }
  .table-wrap { overflow: visible; }
  .runhead, .imprint { border-color: #000; }
  .chapter { border-color: #000; }
  .chapter { margin-top: 22pt; }
  .deck, .lede { color: #000; }
  .seal { margin: 0 0 1.35rem; line-height: 0; }

.masthead h1 { font-size: 28pt; }
  .chapter h2 { font-size: 16pt; }
  .ch-no { font-size: 7.5pt; color: #000; }
  .ch-en, .figure .l, .arm-sub, .small-note, .claim-cite, .reason, .notes, .colophon,
  .cal-note, .years .y, .years .w, table.data th, table.data caption .chip { text-transform: none; letter-spacing: 0.02em; }
table.data caption { color: #000; }
  .chip { color: #000; border: 0; font-size: 9pt; }
  .disputed { color: #000; text-decoration-color: #000; }
  .alarm { border-left: 2pt solid #000; }
  .alarm-label, .claim-date time, .years li.next .d, .years li.next .w, .years li.checked .d { color: #000; }
  .alarm-label { font-weight: 700; }
  .claim, .refusal, .alarm, .cal, .example, .figure, table.data tr { break-inside: avoid; }
  h1, h2, h3, .cal-head { break-after: avoid; }
  .notes { color: #000; }
  .notes .back { display: none; }
  .disputed .fn a { color: #000; }
  .colophon { border-top: 1px solid #000; }
  table.data { font-size: 8.5pt; }
  .moons td { font-size: 8.5pt; }
  .years li { border-color: #999; }
  .years .d { font-size: 10.5pt; }
}
"""

JS = r"""
(function () {
  var toggle = document.getElementById('marks');
  if (!toggle) return;
  var apply = function () { document.body.classList.toggle('hide-marks', !toggle.checked); };
  toggle.addEventListener('change', apply);
  apply();
})();
"""


# --------------------------------------------------------------------------- html fragments

def chip(text: str, *, href: str | None = None, title: str | None = None) -> str:
    body = esc(text)
    if href:
        return f'<a class="chip" href="{attr(href)}">{body}</a>'
    if title:
        return f'<span class="chip" title="{attr(title)}">{body}</span>'
    return f'<span class="chip">{body}</span>'


def chapter(no: str, slug: str, title: str, en: str, body: str) -> str:
    return f"""
<section class="chapter" id="{attr(slug)}">
  <h2><span class="ch-no" aria-hidden="true">{esc(no)}</span> {esc(title)}</h2>
  <p class="ch-en" lang="en">{esc(en)}</p>
  {body}
</section>"""


def pager(current: str) -> str:
    pages = [
        ("index.html", "Bìa", "1"),
        ("reading-copy.html", "Bản đọc lại", "2"),
        ("calendar.html", "Mười năm &amp; lịch", "3"),
    ]
    links = []
    for href, label, no in pages:
        if href == current:
            continue
        links.append(f'<a href="{href}">{label} <span class="no">tr. {no}</span></a>')
    return f'<nav class="pager no-print" aria-label="Trang khác">{"".join(links)}</nav>'


def colophon(model: dict) -> str:
    run = model["run"]
    build = model["build_time"]
    return f"""
<footer class="colophon">
  <p class="mark"><span class="seal-mark">{seal_svg(18)}</span> <b>Sổ Giỗ</b> · bản demo tĩnh</p>
  <p>Dựng lúc {esc(build.strftime("%H:%M · %d tháng %m, %Y"))} (+07) từ
    <span class="chip">data/runs/{esc(model["run_id"])}/run.json</span>
    và <span class="chip">data/oracle/report.json</span>.
    Không có yêu cầu mạng nào khi mở trang: chữ, dấu và mã đều nằm trong thư mục này.</p>
  <p>Chữ: Liberation Serif (SIL OFL 1.1), tự lưu ở <b>assets/fonts</b>. Dấu: SVG, kèm PNG dựng bằng Python, không dùng thư viện ngoài.</p>
  <p>Băng: Google FLEURS, config vi_vn, CC BY 4.0 — bản ghi mở, không phải băng riêng của gia đình.
    Mã: memeshee/so-gio. In trang này trên A4.</p>
</footer>"""


def page(title: str, description: str, body: str, *, current: str) -> str:
    return f"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{attr(description)}">
<meta name="theme-color" content="#f7f2e8">
<meta name="color-scheme" content="light">
<link rel="icon" href="favicon.svg" type="image/svg+xml">
<link rel="icon" href="favicon-32.png" sizes="32x32" type="image/png">
<link rel="apple-touch-icon" href="favicon-180.png">
<link rel="stylesheet" href="assets/site.css">
<script>document.documentElement.classList.add('js')</script>
</head>
<body>
<a class="skip" href="#main">Bỏ qua đến nội dung · Skip to content</a>
<main id="main" class="sheet">
{body}
{pager(current)}
{colophon(model=LAST_MODEL)}
</main>
<script src="assets/site.js" defer></script>
</body>
</html>
"""


LAST_MODEL: dict = {}


# --------------------------------------------------------------------------- index

def render_figures(model: dict) -> str:
    run = model["run"]
    invented = len(model["invented"])
    disputed = len(model["disputed"])
    figures = [
        (vn_float(run.get("audio_seconds", 0.0), 2), "giây", "độ dài băng"),
        (vn_float((run.get("agreement_rate") or 0) * 100, 1), "%", "âm tiết hai bản trùng nhau"),
        (str(disputed), "", "chỗ hai bên nghe lệch"),
        (str(invented), "", "chỗ bịa trên khoảng lặng"),
    ]
    parts = []
    for number, unit, label in figures:
        unit_html = f'<span class="u"> {esc(unit)}</span>' if unit else ""
        parts.append(
            f'<div class="figure"><span class="n">{esc(number)}{unit_html}</span>'
            f'<span class="l">{esc(label)}</span></div>'
        )
    return f'<div class="figures">{"".join(parts)}</div>'


def render_arms(model: dict) -> str:
    run = model["run"]
    audio = float(run.get("audio_seconds") or 0.0)
    rows = []
    for arm in model["arms"]:
        latency = float(arm.get("latency_s") or 0.0)
        factor = float(arm.get("realtime_factor") or 0.0)
        per_second = (latency / audio) if audio else 0.0
        rows.append(f"""
<div class="arm">
  <div class="arm-top">
    <span class="arm-name">{esc(arm.get("arm", "tai"))}</span>
    <span class="arm-rtf">{vn_float(factor, 2)}× <span class="x">thời gian thực</span></span>
  </div>
  <p class="arm-model">{esc(arm.get("model", ""))}</p>
  <p class="arm-sub">{vn_float(latency, 1)} giây cho {vn_float(audio, 2)} giây băng
    · {vn_int(int(arm.get("chars") or 0))} ký tự
    · {vn_float(per_second, 1)} giây máy cho mỗi giây băng</p>
</div>""")
    return f'<div class="arms">{"".join(rows)}</div>'


def annotate(text: str, disputed: list[tuple[str, str]], base: int = 0) -> str:
    """Wrap each disputed whisper reading in a dotted, footnoted span. Longest match first."""
    marks: list[tuple[int, int, int, str]] = []
    for offset, (left, right) in enumerate(disputed):
        if not left:
            continue
        index = base + offset + 1
        pattern = r"(?<![\w])" + re.escape(left) + r"(?![\w])"
        match = re.search(pattern, text)
        if not match:
            continue
        start, end = match.span()
        if any(start < m_end and end > m_start for m_start, m_end, _, _ in marks):
            continue
        marks.append((start, end, index, right))
    marks.sort()
    out: list[str] = []
    pos = 0
    for start, end, index, right in marks:
        out.append(esc(text[pos:start]))
        out.append(
            f'<span class="disputed" title="phowhisper: {attr(right)}">{esc(text[start:end])}'
            f'<sup class="fn"><a href="#d{index}" id="r{index}" aria-label="chú thích {index}">{index}</a></sup></span>'
        )
        pos = end
    out.append(esc(text[pos:]))
    return "".join(out)


def render_transcript(model: dict, limit: int | None = None) -> str:
    lines = model["safe"][:limit] if limit else model["safe"]
    out = []
    for stamp, text in lines:
        body = annotate(text, model["disputed"])
        out.append(
            f'<div class="line"><span class="t chip">[{esc(stamp)}]</span>'
            f'<p class="copy-text">{body}</p></div>'
        )
    return f'<div class="copy">{"".join(out)}</div>'


def render_notes(model: dict) -> str:
    if not model["disputed"]:
        return ""
    items = []
    for i, (left, right) in enumerate(model["disputed"], 1):
        items.append(
            f'<li id="d{i}"><span><span class="w">whisper</span> “{esc(left)}” '
            f'<span aria-hidden="true">·</span> <span class="w">phowhisper</span> “{esc(right)}”</span>'
            f'<a class="back" href="#r{i}" aria-label="Về chỗ đánh dấu {i}">↩</a></li>'
        )
    return f'<ol class="notes">{"".join(items)}</ol>'


def render_invented(model: dict, *, full: bool = False) -> str:
    rows = model["invented"] if full else model["invented"][:2]
    if not rows:
        return ""
    parts = []
    for row in rows:
        start, end = row.get("start"), row.get("end")
        when = f"{mmss(start)}–{mmss(end)}" if start is not None and end is not None else ""
        silence = row.get("silence")
        silence_note = ""
        if silence:
            silence_note = (f" Bộ dò năng lượng xác nhận khoảng lặng "
                            f"{mmss(silence.get('start'))}–{mmss(silence.get('end'))}.")
        parts.append(f"""
<aside class="alarm">
  <p class="alarm-label">Bịa trên khoảng lặng · {esc(when)}</p>
  <p class="alarm-text"><del>“{esc(row.get("text", ""))}”</del></p>
  <p class="alarm-note">{esc(row.get("who", "một mô hình"))} viết ra chữ ở chỗ mô hình kia nghe thấy im lặng.{esc(silence_note)}
    Câu này bị gạch khỏi bản đọc và không thể trở thành ngày giỗ.</p>
</aside>""")
    return "".join(parts)


def toggle() -> str:
    return """
<label class="toggle no-print" for="marks">
  <input type="checkbox" id="marks" checked>
  <span>Hiện chỗ hai mô hình nghe lệch</span>
</label>"""


def render_claims(model: dict) -> str:
    kept = model["kept"]
    if not kept:
        return ('<p class="empty">Không có lời nào vào được sổ trong lần chạy này. '
                'Đó là một kết quả, không phải một lỗi.</p>')
    parts = []
    for claim in kept:
        kind = claim.get("kind", "")
        kind_vn = KIND_VN.get(kind, kind)
        quote = claim.get("quote", "")
        t0, t1 = claim.get("t_start"), claim.get("t_end")
        citation = time_chip(t0, t1) if t0 is not None else chip("không có mốc thời gian")
        date_bits = ""
        if claim.get("lunar_day") and claim.get("lunar_month"):
            lunar = f"{int(claim['lunar_day']):02d}/{int(claim['lunar_month']):02d}"
            if claim.get("leap"):
                lunar += " nhuận"
            solar_next = claim.get("solar_next")
            if solar_next:
                y, m, d = (int(x) for x in solar_next.split("-"))
                date_bits = (
                    f'<p class="claim-date"><span class="lbl">giỗ tới</span> '
                    f'<time datetime="{iso(y, m, d)}">{esc(date_vn(y, m, d))}</time>'
                    f' <span class="chip">[âm {esc(lunar)}]</span></p>'
                )
            else:
                date_bits = f'<p class="claim-date"><span class="lbl">ngày âm</span>{esc(lunar)} · chưa tính được ngày dương</p>'
        parts.append(f"""
<article class="claim">
  <p class="kind">{esc(kind_vn)} <span class="en">· {esc(kind)}</span></p>
  <blockquote><p>“{esc(quote)}”</p></blockquote>
  <p class="claim-cite">{citation}<span>nghe thấy trong băng</span></p>
  {date_bits}
</article>""")
    return "".join(parts)


def reason_vn(reason: str) -> str:
    if reason in REASON_VN:
        return REASON_VN[reason]
    for prefix, vietnamese in REASON_VN_PREFIX:
        if reason.startswith(prefix):
            return vietnamese
    return "bị từ chối"


def render_refusals(model: dict) -> str:
    refused = model["refused"]
    if not refused:
        kept = len(model["kept"])
        return (
            f'<div class="zero"><span class="lbl">Không có lời nào bị từ chối</span>'
            f'<b>{kept}</b> lời được giữ, và lời nào cũng có nguyên văn trong băng cùng mốc thời gian. '
            f'Lần chạy này không có gì để từ chối — không phải vì cuốn sổ dễ dãi, mà vì mô hình chỉ đề xuất {kept} lời.</div>'
        )
    parts = []
    for claim in refused:
        reason = claim.get("reason", "")
        quote = claim.get("quote", "")
        strike = "not in the recording" in reason
        quote_html = f"<s>“{esc(quote)}”</s>" if strike else f"“{esc(quote)}”"
        parts.append(f"""
<article class="refusal">
  <p class="kind">{esc(KIND_VN.get(claim.get("kind", ""), claim.get("kind", "")))} <span class="en">· {esc(claim.get("kind", ""))}</span></p>
  <blockquote><p>{quote_html}</p></blockquote>
  <p class="reason"><span class="rvn">{esc(reason_vn(reason))}</span> <span class="raw">“{esc(reason)}”</span></p>
</article>""")
    return "".join(parts)


def next_note(dates: list[str], today: dt.date) -> str:
    if not dates:
        return ""
    first = dt.date.fromisoformat(dates[0])
    days = (first - today).days
    if days < 0:
        return ""
    if days == 0:
        return "Lần giỗ gần nhất là đúng ngày dựng trang."
    return f"Lần giỗ gần nhất còn {vn_int(days)} ngày, tính đến lúc dựng trang."


def render_calendar_blocks(model: dict) -> str:
    calendar = model["calendar"]
    today = dt.date.today()
    parts = []
    for entry in calendar:
        dates = entry.get("next") or []
        lunar = entry.get("lunar", "")
        quote = entry.get("quote", "")
        citation = entry.get("citation", "")
        rows = []
        for index, value in enumerate(dates):
            y, m, d = (int(x) for x in value.split("-"))
            cls = ' class="next"' if index == 0 else ""
            rows.append(
                f'<li{cls}><span class="y">{y}</span>'
                f'<span class="d"><time datetime="{iso(y, m, d)}">{d} tháng {m}</time></span>'
                f'<span class="w">{esc(weekday_vn(y, m, d))}</span></li>'
            )
        note = next_note(dates, today)
        head_chip = chip(f"[{citation}]") if citation else ""
        parts.append(f"""
<div class="cal">
  <div class="cal-head">
    <h3>{esc(lunar)} âm lịch</h3>
    <span class="lunar">{head_chip}</span>
  </div>
  <p class="cal-quote">“{esc(quote)}”</p>
  <ol class="years">{"".join(rows)}</ol>
  {f'<p class="cal-note">{esc(note)}</p>' if note else ''}
</div>""")
    return "".join(parts)


def render_example_calendar(model: dict) -> str:
    """A labelled calibration table, never presented as a claim from the recording."""
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from core import lunar as vnlunar  # local, first-party, deterministic
    except Exception:
        return ""
    rows = []
    checked = None
    try:
        for lunar_year in range(2026, 2036):
            day, month, year = vnlunar.lunar_to_solar(10, 3, lunar_year)
            rows.append((year, month, day))
    except Exception:
        return ""
    anchor = next(
        (r for r in model["oracle"].get("anchors", {}).get("rows", [])
         if str(r.get("what", "")).startswith("Giỗ Tổ") and (r.get("lunar") or [0, 0, 0])[2] == 2026),
        None,
    )
    if anchor:
        expected = anchor.get("expected") or []
        if len(expected) == 3 and rows and rows[0] == (expected[0], expected[1], expected[2]):
            checked = 2026
    items = []
    for y, m, d in rows:
        cls = ' class="checked"' if y == checked else ""
        mark = '<span class="ok">khớp nguồn</span>' if y == checked else f'<span class="muted">{esc(weekday_vn(y, m, d))}</span>'
        items.append(
            f'<li{cls}><span class="y">{y}</span>'
            f'<span class="d"><time datetime="{iso(y, m, d)}">{d} tháng {m}</time></span>'
            f'<span class="w">{mark}</span></li>'
        )
    anchor_line = ""
    if checked and anchor:
        expected = anchor["expected"]
        anchor_line = (
            "Dòng 2026 khớp với ngày đã công bố trong báo chí Việt Nam: "
            f'{esc(date_vn(expected[0], expected[1], expected[2]))}.'
        )
    return f"""
<div class="example">
  <p class="example-label">Ví dụ lịch · không phải lời trong băng</p>
  <h3>Giỗ Tổ Hùng Vương · 10 tháng 3 âm lịch</h3>
  <p class="why">Bản ghi này không nhắc đến giỗ, nên cuốn sổ không có ngày nào để giữ. Đây là mười năm của một ngày
    giỗ đã được công bố, để thấy bộ tính lịch chạy từ đầu đến cuối — và để thấy nó không cần mô hình ngôn ngữ.</p>
  <ol class="years">{"".join(items)}</ol>
  <p class="why">{anchor_line} Ngày giỗ quốc gia không phải ngày giỗ của gia đình; nó chỉ là phép thử của lịch.</p>
</div>"""


def render_oracle_small(model: dict) -> str:
    s = model["oracle_summary"]
    rows = f"""
<tr><td>Kỳ trăng mới so với USNO</td><td class="num">{s["moons_within"]}/{s["moons_n"]}</td><td class="muted">lệch lớn nhất {vn_float(s["moons_worst"], 1)} giây</td></tr>
<tr><td>Ngày lễ Việt Nam đã công bố</td><td class="num">{s["anchor_passed"]}/{s["anchor_n"]}</td><td class="muted">Tết, Giỗ Tổ, Vu Lan, Trung Thu…</td></tr>
<tr><td>Cấu trúc lịch</td><td class="num">{s["structure_years"]} năm</td><td class="muted">{s["structure_violations"]} vi phạm</td></tr>"""
    return f"""
<div class="table-wrap">
<table class="data">
  <caption>Ba phép thử · <span class="chip">report.json</span></caption>
  <thead><tr><th>Phép thử</th><th class="num">Kết quả</th><th>Ghi chú</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>"""


def oracle_anchor_rows(model: dict, names: list[str]) -> str:
    rows = []
    anchors = model["oracle"].get("anchors", {}).get("rows", [])
    for name in names:
        for row in anchors:
            lunar = row.get("lunar") or [0, 0, 0]
            if str(row.get("what", "")).startswith(name) and lunar[2] == 2026:
                expected = row.get("expected") or [0, 0, 0]
                computed = row.get("computed") or [0, 0, 0]
                ok = row.get("ok") and row.get("roundtrip")
                rows.append(
                    f'<tr><td>{esc(row.get("what", ""))}</td>'
                    f'<td class="nowrap">{esc(date_short(*expected))}</td>'
                    f'<td class="nowrap">{esc(date_short(*computed))}</td>'
                    f'<td class="ok">{esc("khớp" if ok else "lệch")}</td></tr>'
                )
                break
    return "".join(rows)


# --------------------------------------------------------------------------- pages

def render_index(model: dict) -> str:
    run = model["run"]
    arms = model["arms"]
    invented_count = len(model["invented"])
    disputed_count = len(model["disputed"])
    calendar_count = len(model["calendar"])
    extractor = run.get("usage") or {}
    arm_names = " + ".join(arm.get("model", "") for arm in arms) or "hai mô hình"

    body = f"""
<header class="masthead">
  <p class="runhead"><span>Bản chạy <b>{esc(model["run_id"])}</b></span><span>tr. 1</span></p>
  <p class="seal" aria-hidden="true">{seal_svg(26)}</p>
  <h1>Sổ Giỗ</h1>
  <p class="deck">Ngày giỗ không được quên. Nhưng ký ức thì có thể sai, và máy thì có thể bịa.
    Cuốn sổ này chỉ ghi những chữ mà cả hai tai cùng nghe thấy — và chỉ rõ chỗ nó không nghe thấy gì.</p>
  <dl class="imprint">
    <dt>Băng</dt><dd>{esc(pathlib.PurePosixPath(str(run.get("audio", ""))).name)} · {vn_float(float(run.get("audio_seconds") or 0), 2)} giây</dd>
    <dt>Hai tai nghe</dt><dd>{esc(arm_names)}</dd>
    <dt>Máy trích xuất</dt><dd>{esc(extractor.get("model") or run.get("model") or "—")} · chạy trên máy, không gửi lên mạng</dd>
  </dl>
  <nav class="contents" aria-label="Mục lục">
    <a href="#recording">I · Cuộn băng</a>
    <a href="#reading">II · Bản đọc lại</a>
    <a href="#kept">III · Lời được giữ</a>
    <a href="#refused">IV · Lời bị từ chối</a>
    <a href="#years">V · Mười năm giỗ</a>
    <a href="#calendar-check">VI · Lịch đã kiểm</a>
    <a href="#purpose">VII · Để làm gì</a>
  </nav>
</header>

{chapter("I", "recording", "Cuộn băng", "The recording", f'''
<p class="lede">Một tệp âm thanh {vn_float(float(run.get("audio_seconds") or 0), 2)} giây, đưa qua hai mô hình nhận dạng
  tiếng Việt độc lập. Không mô hình nào thấy kết quả của mô hình kia; chỗ nào cả hai nghe ra giống nhau thì mới có
  thể trở thành lời trong sổ.</p>
{render_figures(model)}
{render_arms(model)}
<p class="small-note"><span class="chip">[run.json]</span>
  0,65× nghĩa là mỗi giây băng mất khoảng 1,5 giây máy; 0,32× là khoảng 3,1 giây máy.
  Hai mô hình đều chạy cục bộ trên cùng một máy; whisper dùng int8, phowhisper chạy float32.
  Máy trích xuất {esc(extractor.get("model") or run.get("model") or "—")} đọc {vn_int(int(extractor.get("prompt_tokens") or 0))} token
  và viết {vn_int(int(extractor.get("completion_tokens") or 0))} token trong {vn_float(float(extractor.get("latency_s") or 0), 1)} giây.</p>
<p class="licence"><strong>Bản ghi trong bản demo này là một câu trong bộ ngữ liệu mở FLEURS
  (Google, CC BY 4.0) — không phải băng riêng của một gia đình.</strong> Nó không nhắc đến giỗ;
  vì thế phần mười năm giỗ ở dưới nói thẳng là chưa có ngày nào, chứ không điền vào chỗ trống.</p>
''')}

{chapter("II", "reading", "Bản đọc lại", "Reading copy", f'''
<p class="lede">Nguyên văn những gì whisper nghe được. Chữ nhạt, có chấm dưới là chỗ phowhisper nghe khác;
  cả hai cách nghe đều được giữ, không bên nào được chọn làm sự thật.</p>
{toggle()}
{render_transcript(model, limit=2)}
{render_notes(model)}
{render_invented(model)}
{"" if invented_count else f'<div class="zero"><span class="lbl">Bịa trên khoảng lặng</span><b>0 chỗ.</b> Không mô hình nào viết ra chữ ở chỗ mô hình kia nghe thấy im lặng. Đây là điều cuốn sổ muốn thấy, không phải điều nó muốn che.</div>'}
<p class="more"><a href="reading-copy.html">Đọc toàn bản, kể cả những chỗ nghe lệch <span class="chip">[sổ giỗ · tr. 2]</span></a></p>
''')}

{chapter("III", "kept", "Lời được giữ", "Claims kept", f'''
<p class="lede">Một lời chỉ vào sổ khi nó có nguyên văn trong băng và có mốc thời gian để gia đình mở lại mà nghe.
  Dưới đây là tất cả những lời đã qua được cửa đó — không phải những lời nghe hợp lý.</p>
{render_claims(model)}
<p class="small-note"><span class="chip">[guard.py]</span> Nhãn loại (người, ngày giỗ, đồ cúng) do mô hình trích xuất gán.
  Cuốn sổ chỉ kiểm tra hai điều: chữ có trong băng, và ngày có tính được. Nhãn sai thì gia đình sửa trên giấy.</p>
''')}

{chapter("IV", "refused", "Lời bị từ chối", "Refused", f'''
<p class="lede">Mỗi lời bị từ chối đều có lý do, và lý do được ghi lại nguyên văn. Không lời nào được sửa cho đúng;
  không ngày nào được làm tròn cho đẹp.</p>
{render_refusals(model)}
''')}

{chapter("V", "years", "Mười năm giỗ", "Ten years", f'''
<p class="lede">Từ ngày âm trong lời kể, lịch tính ra ngày dương cho mười năm tới. Mô hình không được phép sinh ra
  ngày nào; nó chỉ được nhắc lại lời.</p>
{render_calendar_blocks(model) if calendar_count else f'<p class="empty">Lần chạy này không có ngày giỗ nào sống sót, nên không có bảng mười năm. Bản ghi là một câu trong bộ ngữ liệu FLEURS — nó không nhắc đến giỗ, và chúng tôi không điền ngày vào chỗ trống. <b>Một cuốn sổ trung thực phải có quyền để trống.</b></p>'}
{render_example_calendar(model) if not calendar_count else ""}
''')}

{chapter("VI", "calendar-check", "Lịch đã kiểm", "Calendar checked", f'''
<p class="lede">Bộ tính lịch là phần duy nhất của cuốn sổ được đối chiếu với bên ngoài. Ba phép thử, tất cả đều
  công khai ở đây — kể cả chỗ sai lớn nhất.</p>
{render_oracle_small(model)}
<div class="table-wrap spaced">
<table class="data">
  <caption>Bốn ngày lễ 2026 · đã công bố và tính được</caption>
  <thead><tr><th>Ngày</th><th class="num">Đã công bố</th><th class="num">Tính được</th><th class="num">Kết quả</th></tr></thead>
  <tbody>{oracle_anchor_rows(model, ["Tết Nguyên Đán", "Giỗ Tổ Hùng Vương", "Rằm tháng Bảy", "Tết Trung Thu"])}</tbody>
</table>
</div>
<p class="small-note"><span class="chip">[report.json]</span> USNO là Đài thiên văn Hải quân Hoa Kỳ.
  Ngày lễ đối chiếu với Wikipedia, VietnamNet và cổng thông tin chính phủ. Bảng đầy đủ ở
  <a href="calendar.html">tr. 3</a>.</p>
''')}

{chapter("VII", "purpose", "Để làm gì", "What it is for", f'''
<div class="prose">
  <p>Trong một gia đình, thường chỉ có một người còn nhớ ngày giỗ của từng cụ: mười hai tháng bảy, hay tháng bảy
    nhuận, năm nào cũng vậy. Khi người ấy mất, những ngày ấy mất theo — không phải vì không ai thương, mà vì không
    ai kịp ghi lại.</p>
  <p>Cuốn sổ này làm đúng một việc: ghi lời người ấy nói, và chỉ ghi những gì cả hai tai cùng nghe thấy. Hai mô hình
    nghe lại cuộn băng một cách độc lập. Chỗ hai bên nghe giống nhau thì lời ấy được vào sổ, kèm mốc thời gian để mở
    lại mà nghe. Chỗ hai bên nghe khác nhau thì để nguyên cả hai, đánh dấu, không chọn bên nào. Chỗ một bên viết ra
    trong khi bên kia nghe thấy im lặng thì gạch bỏ, và ghi rõ là đã gạch.</p>
  <p>Ngày giỗ thì không do máy đoán. Từ ngày âm trong lời kể, lịch tính ra ngày dương cho mười năm tới. Nếu lời kể
    không có ngày, cuốn sổ để trống và nói thẳng là chưa biết — đó cũng là một câu trả lời trung thực.</p>
  <p class="en" lang="en">This is a family memorial book, not a calendar app. In most families one person holds the
    death-anniversary dates; when that person dies, the dates die with them. Sổ Giỗ records that person’s voice,
    transcribes it twice with two independent speech models, and lets only the words both heard become claims.
    Disagreements stay on the page, struck-out inventions stay visible, and the lunar calendar — not the language
    model — computes the next ten years. Where the recording says nothing, the book says it does not know.</p>
  <p class="small-note" style="margin-top:1.4rem"><span class="chip">[Careless Whisper · FAccT 2024]</span>
    Whisper bịa khoảng 1% số đoạn, và bịa nhiều hơn khi gặp khoảng lặng. Đây là lý do có tai thứ hai và có chỗ
    để gạch.</p>
</div>
''')}
"""
    return body


def render_reading_page(model: dict) -> str:
    run = model["run"]
    invented = len(model["invented"])
    excluded = run.get("excluded_spans") or []
    excluded_html = ""
    if excluded:
        rows = "".join(
            f'<div class="line"><span class="t chip">[{mmss(row.get("start"))}–{mmss(row.get("end"))}]</span>'
            f'<p><del>{esc(row.get("text", ""))}</del></p></div>'
            for row in excluded
        )
        excluded_html = f'<h3>Bị loại khỏi bản đọc</h3><div class="copy">{rows}</div>'

    body = f"""
<header class="masthead">
  <p class="runhead"><span><a href="index.html">Sổ Giỗ</a> · bản chạy <b>{esc(model["run_id"])}</b></span><span>tr. 2</span></p>
  <h1>Bản đọc lại</h1>
  <p class="deck">Toàn bộ những gì hai mô hình nghe được — kể cả chỗ chúng nghe khác nhau. Không chữ nào ở đây
    được sửa cho sạch.</p>
  <dl class="imprint">
    <dt>Băng</dt><dd>{esc(pathlib.PurePosixPath(str(run.get("audio", ""))).name)} · {vn_float(float(run.get("audio_seconds") or 0), 2)} giây</dd>
    <dt>Độ khớp</dt><dd>{vn_float((run.get("agreement_rate") or 0) * 100, 1)} % âm tiết · {len(model["disputed"])} chỗ lệch · {invented} chỗ bịa</dd>
    <dt>Nguồn</dt><dd><span class="chip">reading_copy.md</span> và <span class="chip">run.json</span></dd>
  </dl>
</header>

<section class="chapter" id="copy">
  <h2>Nguyên văn bản đọc</h2>
  <p class="ch-en" lang="en">The reading, verbatim</p>
  <dl class="legend">
    <div><dt><span class="disputed">chữ</span></dt><dd>hai mô hình nghe khác nhau — chấm dưới và số là chú thích, cả hai cách nghe đều được giữ</dd></div>
    <div><dt><del>chữ</del></dt><dd>một mô hình viết ra trên khoảng lặng — bị gạch, không vào sổ</dd></div>
    <div><dt><span class="chip">[00:00]</span></dt><dd>mốc thời gian trong băng, để gia đình mở lại mà nghe</dd></div>
  </dl>
  {toggle()}
  {render_transcript(model)}
  {render_notes(model)}
  {render_invented(model, full=True)}
  {"" if invented else '<div class="zero"><span class="lbl">Bịa trên khoảng lặng</span><b>0 chỗ.</b> Không mô hình nào viết ra chữ ở chỗ mô hình kia nghe thấy im lặng.</div>'}
  {excluded_html}
  <p class="more"><a href="index.html#kept">Từ bản đọc này, {len(model["kept"])} lời vào sổ <span class="chip">[tr. 1]</span></a></p>
</section>

<section class="chapter" id="arms">
  <h2>Hai mô hình, hai lần nghe</h2>
  <p class="ch-en" lang="en">Two arms</p>
  {render_arms(model)}
  <p class="small-note"><span class="chip">[run.json]</span> “× thời gian thực” là số giây băng xử lý được trong
    một giây máy. Dưới 1× là chậm hơn thời gian thực; cả hai tai đều chạy trên CPU của máy này.</p>
</section>
"""
    return body


def render_calendar_page(model: dict) -> str:
    s = model["oracle_summary"]
    oracle = model["oracle"]
    calendar = model["calendar"]
    moons = (oracle.get("new_moons") or {}).get("rows") or []
    anchors = (oracle.get("anchors") or {}).get("rows") or []
    today = dt.date.today()

    moon_rows = "".join(
        f'<tr><td class="nowrap">{esc(date_short(*[int(x) for x in str(row.get("date_ut", ""))[:10].split("-")]))}</td>'
        f'<td class="nowrap">{esc(row.get("usno_ut", ""))} <span class="arrow">→</span> {esc(str(row.get("ours_ut", ""))[11:16])}</td>'
        f'<td class="num">{esc(("+" if float(row.get("delta_seconds") or 0) >= 0 else "−") + vn_float(abs(float(row.get("delta_seconds") or 0)), 1))}</td></tr>'
        for row in moons
    )

    anchor_rows = []
    for row in anchors:
        lunar = row.get("lunar") or [0, 0, 0]
        expected = row.get("expected") or [0, 0, 0]
        computed = row.get("computed") or [0, 0, 0]
        ok = bool(row.get("ok")) and bool(row.get("roundtrip"))
        anchor_rows.append(
            f'<tr><td>{esc(row.get("what", ""))}</td>'
            f'<td class="opt nowrap">{int(lunar[0]):02d}/{int(lunar[1]):02d} · {int(lunar[2])}</td>'
            f'<td class="nowrap">{esc(date_short(*expected))}</td>'
            f'<td class="nowrap">{esc(date_short(*computed))}</td>'
            f'<td class="ok">{esc("khớp" if ok else "lệch")}</td></tr>'
        )

    sources = sorted({
        re.sub(r"^https?://(www\.)?", "", str(row.get("source", ""))).split("/")[0]
        for row in anchors if row.get("source")
    })
    sources_html = " · ".join(esc(x) for x in sources)

    eval_dir = ROOT / "data" / "eval"
    eval_html = ""
    if (eval_dir / "tables.md").exists():
        try:
            md = (eval_dir / "tables.md").read_text(encoding="utf-8")
        except Exception:
            md = ""
        if md.strip():
            eval_html = f"""
<section class="chapter" id="eval">
  <h2>Bảng đánh giá</h2>
  <p class="ch-en" lang="en">Evaluation tables</p>
  <div class="prose">{mini_md(md, heading_offset=2)}</div>
</section>"""

    body = f"""
<header class="masthead">
  <p class="runhead"><span><a href="index.html">Sổ Giỗ</a> · bản chạy <b>{esc(model["run_id"])}</b></span><span>tr. 3</span></p>
  <h1>Mười năm giỗ</h1>
  <p class="deck">Ngày âm trong lời kể, ngày dương trong lịch. Không ngày nào do mô hình sinh ra — và khi lời kể
    không có ngày, cuốn sổ để trống.</p>
  <dl class="imprint">
    <dt>Ngày giỗ</dt><dd>{f"{len(calendar)} ngày" if calendar else "chưa có ngày nào"}</dd>
    <dt>Lịch</dt><dd>âm lịch Việt Nam · UTC+7</dd>
    <dt>Đối chiếu</dt><dd>{s["moons_within"]}/{s["moons_n"]} kỳ trăng · {s["anchor_passed"]}/{s["anchor_n"]} ngày lễ</dd>
  </dl>
</header>

{chapter("I", "years", "Ngày giỗ", "The dates", f'''
{render_calendar_blocks(model) if calendar else '<p class="empty">Lần chạy này không có ngày giỗ nào sống sót, nên bảng mười năm để trống. Không có ngày nào bị điền vào chỗ trống.</p>'}
{render_example_calendar(model) if not calendar else ""}
''')}

{chapter("II", "method", "Cách tính", "The method", f'''
<div class="prose">
  <p>Một tháng âm bắt đầu vào ngày có kỳ trăng mới, tính theo giờ Việt Nam (UTC+7). Đó là lý do lịch Việt Nam
    không phải lịch Trung Quốc: một kỳ trăng mới lúc 23:30 giờ Việt Nam là 00:30 giờ Bắc Kinh ngày hôm sau, và
    một bảng UTC+8 có thể đặt Tết lệch cả tháng.</p>
  <p>Tháng 11 âm là tháng chứa tiết đông chí. Năm có tháng nhuận khi giữa hai tháng 11 có mười ba kỳ trăng; tháng
    nhuận là tháng đầu tiên trong số đó không chứa trung khí, và nó lặp lại số của tháng trước.</p>
  <p>Giỗ rơi vào tháng nhuận: năm nào không có tháng nhuận ấy thì lịch bỏ năm đó, chứ không đoán sang tháng thường.
    Đó là một khoảng trống có chủ ý, không phải lỗi.</p>
  <p class="small-note"><span class="chip">[core/lunar.py]</span> Không thư viện ngoài; toàn bộ là số học trong một tệp,
    kiểm bằng 37 kỳ trăng của USNO và 17 ngày lễ đã công bố.</p>
</div>
''')}

{chapter("III", "checks", "Đối chiếu", "Checks", f'''
<p class="lede">{s["moons_within"]} trong {s["moons_n"]} kỳ trăng mới lệch dưới hai phút so với USNO; kỳ lệch nhất là
  {vn_float(s["moons_worst"], 1)} giây. {s["anchor_passed"]} trong {s["anchor_n"]} ngày lễ Việt Nam đã công bố khớp
  ngày, và đều chuyển ngược lại được.</p>
<div class="table-wrap">
<table class="data">
  <caption>Kỳ trăng mới · cả hai giờ đều là UTC · <span class="chip">report.json</span></caption>
  <thead><tr><th>Ngày</th><th>Giờ UTC<br>USNO → tính</th><th class="num">Lệch (giây)</th></tr></thead>
  <tbody>{moon_rows}</tbody>
</table>
</div>
<p class="small-note">Lệch trung bình {vn_float(s["moons_mean"], 1)} giây; lớn nhất {vn_float(s["moons_worst"], 1)} giây.
  Cả hai giờ đều là UTC; giờ Việt Nam là UTC+7, nên một kỳ trăng có thể rơi sang ngày hôm sau ở Hà Nội. Cột lệch
  so sánh hai phép tính trong cùng một múi giờ.</p>

<div class="table-wrap spaced">
<table class="data">
  <caption>Ngày lễ Việt Nam · {s["anchor_passed"]}/{s["anchor_n"]} khớp</caption>
  <thead><tr><th>Ngày lễ</th><th class="opt num">Âm lịch</th><th class="num">Đã công bố</th><th class="num">Tính được</th><th class="num">Kết quả</th></tr></thead>
  <tbody>{"".join(anchor_rows)}</tbody>
</table>
</div>
<p class="small-note"><span class="chip">[report.json]</span> Nguồn: {sources_html}. “Chuyển ngược lại được” nghĩa là
  ngày dương tính ra cũng đổi ngược về đúng ngày âm đã cho.</p>

<p class="lede" style="margin-top:1.8rem">Cấu trúc {s["structure_years"]} năm: {s["structure_violations"]} vi phạm.
  Không có tháng nhuận sai, không có ngày ảo, không có tháng mười ba.</p>
''')}

{eval_html}
"""
    return body


# --------------------------------------------------------------------------- checks

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
        "param", "source", "track", "wbr"}


class DocCheck(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.count = 0
        self.external: list[str] = []
        self.ids: set[str] = set()
        self.dupes: set[str] = set()
        self.local_refs: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.count += 1
        values = dict(attrs)
        if "id" in values:
            if values["id"] in self.ids:
                self.dupes.add(values["id"])
            self.ids.add(values["id"])
        for key in ("src", "href"):
            value = values.get(key, "")
            if value.startswith(("http://", "https://", "//")):
                self.external.append(f"{tag}[{key}]={value}")
            elif value and not value.startswith(("#", "mailto:", "tel:", "data:")):
                self.local_refs.append(value)
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.count += 1

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack:
            self.errors.append(f"stray </{tag}>")
            return
        if self.stack[-1] == tag:
            self.stack.pop()
            return
        if tag in self.stack:
            while self.stack and self.stack[-1] != tag:
                self.errors.append(f"unclosed <{self.stack.pop()}> before </{tag}>")
            if self.stack:
                self.stack.pop()
            return
        self.errors.append(f"stray </{tag}> (open: {self.stack[-1]})")


def check_html(path: pathlib.Path) -> tuple[DocCheck, str]:
    text = path.read_text(encoding="utf-8")
    check = DocCheck()
    check.feed(text)
    check.close()
    for leftover in check.stack:
        check.errors.append(f"unclosed <{leftover}> at end of document")
    if not text.lstrip().lower().startswith("<!doctype html>"):
        check.errors.append("missing <!doctype html>")
    if 'charset="utf-8"' not in text:
        check.errors.append("missing utf-8 charset")
    if 'name="viewport"' not in text:
        check.errors.append("missing viewport")
    if 'lang="vi"' not in text:
        check.errors.append("missing lang=vi")
    return check, text


def verify_refs(dist: pathlib.Path) -> list[str]:
    problems = []
    for page in sorted(dist.glob("*.html")):
        check, _ = check_html(page)
        for value in check.local_refs:
            target = (page.parent / value.split("?")[0].split("#")[0]).resolve()
            if not target.exists():
                problems.append(f"{page.name}: missing {value}")
        for duplicate in sorted(check.dupes):
            problems.append(f"{page.name}: duplicate id {duplicate!r}")
    css = (dist / "assets" / "site.css").read_text(encoding="utf-8")
    for url in re.findall(r'url\("([^"]+)"\)', css):
        if not (dist / "assets" / url).exists():
            problems.append(f"site.css: missing {url}")
    if "@import" in css or "http://" in css or "https://" in css:
        problems.append("site.css makes an external request")
    return problems


def human_bytes(size: int) -> str:
    return f"{size:,} bytes"


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Sổ Giỗ static demo site.")
    parser.add_argument("--run", help="run id under data/runs/ (default: the richest run)")
    args = parser.parse_args(argv)

    run_item = choose_run(args.run)
    reading = parse_reading_copy(run_item["path"].parent / "reading_copy.md")
    oracle = load_oracle()
    model = build_model(run_item, reading, oracle)
    global LAST_MODEL
    LAST_MODEL = model

    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "assets" / "fonts").mkdir(parents=True, exist_ok=True)

    (DIST / "index.html").write_text(page(
        "Sổ Giỗ — sổ ghi ngày giỗ của gia đình",
        "Một cuốn sổ giỗ chỉ ghi những chữ mà cả hai mô hình cùng nghe thấy, "
        "và để trống chỗ nó không nghe thấy.",
        render_index(model), current="index.html"), encoding="utf-8")
    (DIST / "reading-copy.html").write_text(page(
        "Bản đọc lại — Sổ Giỗ",
        "Nguyên văn bản đọc của hai mô hình, kể cả những chỗ nghe lệch và những chỗ bịa trên khoảng lặng.",
        render_reading_page(model), current="reading-copy.html"), encoding="utf-8")
    (DIST / "calendar.html").write_text(page(
        "Mười năm giỗ — Sổ Giỗ",
        "Mười năm ngày giỗ tính từ âm lịch, và những phép đối chiếu lịch với USNO cùng ngày lễ Việt Nam.",
        render_calendar_page(model), current="calendar.html"), encoding="utf-8")

    (DIST / "assets" / "site.css").write_text(CSS.lstrip(), encoding="utf-8")
    (DIST / "assets" / "site.js").write_text(JS.lstrip(), encoding="utf-8")
    for font in sorted((ASSETS / "fonts").glob("*")):
        shutil.copy2(font, DIST / "assets" / "fonts" / font.name)

    (DIST / "favicon.svg").write_text(FAVICON_SVG, encoding="utf-8")
    write_png(DIST / "favicon-32.png", 32)
    write_png(DIST / "favicon-180.png", 180)
    (DIST / ".nojekyll").write_text("", encoding="utf-8")

    problems = verify_refs(DIST)
    total_errors = 0
    pages = 0
    print()
    print(f"run       {model['run_id']}  ({run_item['path'].relative_to(ROOT)})")
    print(f"audio     {model['run'].get('audio')}  ·  {model['run'].get('audio_seconds')} s")
    print(f"claims    {len(model['kept'])} kept · {len(model['refused'])} refused · "
          f"{len(model['invented'])} invented over silence")
    print(f"calendar  {len(model['calendar'])} dated claim(s)")
    s = model["oracle_summary"]
    print(f"oracle    {s['moons_within']}/{s['moons_n']} new moons within 2 min · "
          f"{s['anchor_passed']}/{s['anchor_n']} published dates · {s['structure_violations']} structure violations")
    print()
    for path in sorted(DIST.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(DIST)!s:<38} {human_bytes(path.stat().st_size)}")
            if path.suffix == ".html":
                pages += 1
                check, _ = check_html(path)
                total_errors += len(check.errors)
                if check.errors:
                    for error in check.errors:
                        print(f"    ! {error}")
                if check.external:
                    print(f"    ! external refs: {check.external}")
    for problem in problems:
        print(f"  ! {problem}")
    print()
    if total_errors or problems:
        print(f"CHECK FAILED: {total_errors} HTML error(s), {len(problems)} reference problem(s)")
        return 1
    print(f"CHECK OK: {pages} valid HTML5 documents, 0 unclosed tags, 0 external requests, all local refs resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
