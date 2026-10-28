#!/usr/bin/env python3
"""Render data/eval/report.json into pasteable markdown tables.

`python eval/tables.py` regenerates data/eval/tables.md from the last report without
re-running any model. `run_eval.py` calls `render()` at the end of every run, so the
two files can never drift.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval"


def _f(value, digits=3, dash="—") -> str:
    if value is None:
        return dash
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _lat(seconds) -> str:
    if seconds is None:
        return "—"
    if seconds < 0.01:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f} s"


def _num(summary: dict, *names):
    """First numeric field present under any of `names` — external arms name their keys differently."""
    for n in names:
        v = summary.get(n)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
    return None


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _quote(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def render(report: dict) -> str:
    env = report.get("environment", {})
    corpus = report.get("corpus", {})
    parts: list[str] = []

    parts.append("# Sổ Giỗ — evaluation tables\n")
    parts.append(
        f"Every number below was produced by `python eval/run_eval.py` on "
        f"{env.get('run_at', 'n/a')}. Machine: {env.get('cpu_count', '?')} vCPU "
        f"({env.get('cpu_model', 'unknown')}), {env.get('machine', '')}, Python {env.get('python', '?')}. "
        f"ASR: `{env.get('whisper_model', '')}` and `{env.get('phowhisper_model', '')}`, CPU only, "
        f"models warm-loaded once before timing. Audio: {corpus.get('name', '')} "
        f"({corpus.get('licence', '')}).\n")

    # ---------------------------------------------------------------- 1
    asr = report.get("asr")
    if asr:
        rows = []
        for key in ("whisper", "phowhisper"):
            s = asr["per_arm"][key]
            rows.append([f"`{key}`", f"`{s['model']}`", s["n_clips"],
                         _f(s["audio_seconds"], 1), _f(s["mean_wer"], 3),
                         _f(s["micro_wer"], 3),
                         f"{s['realtime_factor_aggregate']:.2f}×",
                         f"{s['realtime_factor_median']:.2f}×"])
        parts.append("## 1. ASR accuracy per arm\n")
        parts.append(_table(
            ["Arm", "Model", "Clips (n)", "Audio (s)", "Mean WER", "Micro WER",
             "Realtime factor", "Median RTF"], rows))
        parts.append(
            f"\n*Syllable-level word error rate against all {asr['per_arm']['whisper']['n_clips']} published "
            f"FLEURS vi_vn reference transcripts ({_f(asr['per_arm']['whisper']['audio_seconds'], 1)} s of audio). "
            f"Mean WER is the unweighted mean of per-clip WER; micro WER pools all edits over all reference "
            f"syllables. Realtime factor is total audio seconds ÷ total inference seconds on this CPU, "
            f"higher is faster; median RTF is the per-clip median.*\n")

    # ---------------------------------------------------------------- 2
    inv = report.get("invention")
    if inv:
        rows = []
        for key in ("whisper", "phowhisper"):
            s = inv["per_arm"][key]
            rows.append([f"`{key}`", s["silences_with_invention"], s["invented_syllables"],
                         _f(s["silence_seconds"], 1), _f(s["wpm_of_silence"], 1),
                         s["invented_syllables_full_silent_window"]])
        parts.append("## 2. Invention over inserted silence\n")
        parts.append(_table(
            ["Arm", "Silences with invention", "Invented syllables",
             "Inserted silence (s)", "Invented words/min of silence",
             "Syllables in full silent window"], rows))
        parts.append(
            f"\n*Syllables each arm emitted with timestamps inside all {inv['per_arm']['whisper']['silences']} "
            f"digitally spliced silences ({_f(inv['inserted_silence_seconds'], 1)} s total), where the reference "
            f"transcript says nothing at all. A segment counts when its timestamp midpoint falls inside the "
            f"splice. The last column counts inside the wider acoustically silent window (the splice plus the "
            f"natural pause it was inserted into), so it is the loosest reading. Vietnamese is written in "
            f"syllables, so syllables ≈ words.*\n")

        examples = []
        for key in ("whisper", "phowhisper"):
            for ex in inv["per_arm"][key]["worst_examples"]:
                examples.append([f"`{key}`", f"`{ex['audio'].split('/')[-1]}`", ex["timestamp"],
                                 ex["window"], ex["syllables"], f"“{_quote(ex['text'])}”"])
        if examples:
            parts.append("### 2b. Verbatim worst cases\n")
            parts.append(_table(["Arm", "File", "Detected at", "Inserted silence", "Syllables",
                                 "Model output (verbatim)"], examples))
            parts.append(
                "\n*The two longest stretches each arm produced inside a silent splice, quoted exactly as the "
                "model returned them, with the segment timestamps the model itself assigned. Each row is a "
                "sentence that exists nowhere in the recording and nowhere in the reference transcript.*\n")

    # ---------------------------------------------------------------- 3
    catch = report.get("catch")
    if catch:
        fp = catch["false_positive"]
        rows = [[ "All arms", catch["silences_with_invention"], catch["flagged"],
                  _pct(catch["catch_rate"]) ]]
        for key in ("whisper", "phowhisper"):
            s = catch["per_arm"][key]
            rows.append([f"`{key}`", s["silences_with_invention"], s["flagged"], _pct(s["catch_rate"])])
        parts.append("## 3. Does the two-model diff catch it?\n")
        parts.append(_table(["Arm", "Silences with invention", "Flagged as invented", "Catch rate"], rows))
        parts.append(
            f"\n*A “silence with invention” is an (arm, spliced silence) pair where that arm emitted at least "
            f"one syllable inside the splice; it is “flagged” when `core.asr` produced an `invented` span for "
            f"that arm overlapping the splice (both conditions: the segment is ≥50% inside detected non-speech "
            f"and the other arm has no segment there). Catch rate is the share of such pairs the pipeline "
            f"actually marks.{_leak_note(catch)}*\n")

        parts.append("## 3b. False positives on clean audio\n")
        parts.append(_table(
            ["Clips (n)", "Clips with an invented flag", "False-positive rate (clips)",
             "Invented spans", "Invented syllables",
             "Spans quoting text that is in the reference"],
            [[fp["n_clean_clips"], fp["clips_with_invented_flag"],
              _pct(fp["clip_false_positive_rate"]), fp["invented_spans"], fp["invented_syllables"],
              fp["spans_quoting_reference_text"]]]))
        parts.append(
            f"\n*The same invented-span logic run over all {fp['n_clean_clips']} clean clips "
            f"({_f(fp['clean_audio_seconds'], 1)} s), which contain no spliced silence — only the pauses the "
            f"speakers actually left. These clips do have natural silence, so a flag here is not automatically "
            f"wrong; the last column is the honest error count: spans whose text is literally present in the "
            f"published reference transcript, i.e. words that were really spoken and were labelled invented. "
            f"{fp['syllables_quoting_reference_text']} of {fp['invented_syllables']} flagged syllables are in "
            f"the reference.*\n")

    # ---------------------------------------------------------------- 4
    ext = report.get("extraction")
    if ext and ext.get("status") == "run":
        rows = []
        for name, a in ext["arms"].items():
            if a.get("guard_bug"):
                rows.append([f"`{name}`", "—", "—", "—", f"0/{a['predicted']}", a["gold"],
                             a["refusals"], f"**{a['uncited_kept']}**", _lat(a["mean_latency_s"])])
                continue
            rows.append([f"`{name}`", _f(a["precision"], 3), _f(a["recall"], 3), _f(a["f1"], 3),
                         f"{a['tp']}/{a['predicted']}", a["gold"], a["refusals"],
                         a["uncited_kept"], _lat(a["mean_latency_s"])])
        remote = []
        for name, t in (report.get("tinker") or {}).items():
            s = t.get("summary") or {}
            rows.append([f"`{name}` †", _f(_num(s, "precision"), 3), _f(_num(s, "recall"), 3),
                         _f(_num(s, "f1"), 3),
                         f"{_num(s, 'tp') if _num(s, 'tp') is not None else '?'}/"
                         f"{_num(s, 'predicted', 'kept', 'n_predicted') if _num(s, 'predicted', 'kept', 'n_predicted') is not None else '?'}",
                         _num(s, "gold", "n_gold", "gold_total") if _num(s, "gold", "n_gold", "gold_total") is not None else ext["gold_total"],
                         _num(s, "refusals") if _num(s, "refusals") is not None else "—",
                         _num(s, "uncited_kept") if _num(s, "uncited_kept") is not None else "—",
                         "remote"])
            remote.append(name)
        parts.append("## 4. Extraction arms\n")
        parts.append(_table(["Arm", "Precision", "Recall", "F1", "TP / kept", "Gold claims",
                             "Refusals", "Uncited kept", "Mean latency"], rows))
        tail = ""
        if remote:
            tail = (" † Scored separately, on the same gold set and with the same "
                    "`eval/metrics.claims_match` rule, by the Tinker training process; a remote model on a "
                    "different latency tier, so its latency is not comparable to the local arms and is not "
                    "quoted.")
        parts.append(
            f"\n*{ext['n_items']} held-out recordings, {ext['gold_total']} gold claims. A claim counts only "
            f"after `core.guard.review` keeps it — the quote must be verbatim in the transcript, carry a "
            f"timestamp, and (for a date) resolve in the lunar calendar. A true positive requires the same "
            f"kind, and then either the same lunar day/month (for a date, where the quote is only the "
            f"citation) or a quote that contains the gold quote, or is contained by it, or shares ≥60% of "
            f"its syllables (for a text claim, where the quote is the answer). Matching is greedy one-to-one, "
            f"so each gold claim can be found once. Precision and recall are pooled over all recordings. "
            f"Uncited kept claims must be 0 for every arm; if any arm broke that rule the row shows the count "
            f"in bold and the arm is left unscored, because that is a bug in the guard, not a metric.{tail}*\n")
        if ext.get("guard_failures"):
            parts.append("\n> **Guard failure:** " + "; ".join(ext["guard_failures"]) +
                         ". A kept claim without a citation is a `core/guard.py` bug — "
                         "`eval/run_eval.py` exits 2 rather than publish a score over it.\n")
    elif ext:
        parts.append("## 4. Extraction arms — pending\n")
        parts.append(f"*Not run: {ext.get('reason', 'held-out set missing')}. "
                     f"The harness (`eval/run_eval.py`) runs this table as soon as the file exists.*\n")

    return "\n".join(parts).rstrip() + "\n"


def _leak_note(catch: dict) -> str:
    leaked = sum(u["leaked_into_quotable_syllables"] for u in catch.get("units", []))
    if not leaked:
        return (" No invented syllable leaked into `agreed_spans`, so nothing fabricated became quotable on "
                "these clips.")
    return (f" Separately, {leaked} syllable(s) of invented text also appeared inside a span "
            f"`agreed_spans` marked quotable — the failure mode the guard exists to prevent.")


def main() -> int:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    (OUT / "tables.md").write_text(render(report), encoding="utf-8")
    print(f"wrote {(OUT / 'tables.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
