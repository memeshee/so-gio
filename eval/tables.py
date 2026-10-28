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


def mmss(sec: float | None) -> str:
    if sec is None:
        return "—"
    m, s = divmod(int(round(sec)), 60)
    return f"{m:02d}:{s:02d}"


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
            for kind, label in (("splice", "spliced pause"), ("tail", "appended tail")):
                b = s.get("by_probe", {}).get(kind)
                if not b:
                    continue
                rows.append([f"`{key}`", label, b["probes"], _f(b["silence_seconds"], 1),
                             b["invented_syllables"], _f(b["wpm_of_silence"], 1),
                             f"{b['probes_with_invention']}/{b['probes']}", s["timestamp_usable_probes"]])
        parts.append("## 2. Invention over silence\n")
        parts.append(_table(
            ["Arm", "Probe", "Probes (n)", "Silence (s)", "Invented syllables",
             "Invented words/min of silence", "Probes with invention", "Timestamps usable (probes)"], rows))
        parts.append(
            f"\n*Syllables each arm produced **only because silence was added**, across "
            f"{inv['probe_count']['splice']} clipped-pause probes and {inv['probe_count']['tail']} "
            f"appended-tail probes ({_f(inv['silence_seconds_total'], 1)} s of dead air in total, where the "
            f"reference transcript says nothing at all). Invention is measured as a content delta: the "
            f"syllables in the arm's transcript of the probed file that have no counterpart in the same arm's "
            f"transcript of the unmodified source — a definition that needs no timestamps and also catches a "
            f"model that repeats real words into the silence. Vietnamese is written in syllables, so "
            f"syllables ≈ words. The last column says on how many probes the arm's own timestamps are fine "
            f"enough to place a syllable; counting by timestamp instead would credit PhoWhisper with "
            f"{inv['per_arm']['phowhisper']['syllables_in_window_by_timestamp']} syllables on the "
            f"spliced-pause probes alone, and every one of them is the clip's real speech carried by a single "
            f"segment spanning the whole 25 s window — the reason the content delta is the published number.*\n")
        if not inv["per_arm"]["whisper"]["worst_examples"] and not inv["per_arm"]["phowhisper"]["worst_examples"]:
            tail_ev = [e for e in inv["events"] if e["probe"] == "tail" and e["arm"] == "whisper"]
            closed = [e for e in tail_ev if e["prev_segment_end"] is not None]
            gaps = sorted(e["window_start"] - e["prev_segment_end"] for e in closed)
            gap_text = ""
            if gaps:
                gap_text = (f" On the {len(tail_ev)} appended-tail probes the model's last segment ends "
                            f"{gaps[0]:.2f}–{gaps[-1]:.2f} s before the appended silence on {len(closed)} of "
                            f"them, and it puts nothing after that segment.")
            parts.append(
                "**No arm produced a single syllable inside a silent window, on any of the "
                f"{inv['probe_count']['splice'] + inv['probe_count']['tail']} probes.** There is therefore no "
                "fabricated sentence to quote. What can be shown instead is where each model chose to stop "
                "and start: on the three spliced-pause probes, Whisper's own segment boundaries bracket the "
                f"digital silence — it heard the hole.{gap_text}\n")
            rows = []
            for e in inv["events"]:
                if e["probe"] != "splice" or e["arm"] != "whisper":
                    continue
                rows.append([f"`{e['clip_id']}`",
                             f"{mmss(e['window_start'])}–{mmss(e['window_end'])}",
                             _f(e["prev_segment_end"], 2), _f(e["next_segment_start"], 2),
                             _f((e["next_segment_start"] or 0) - (e["prev_segment_end"] or 0), 2),
                             e["syllables_added"]])
            if rows:
                parts.append(_table(["Probe", "Digital silence", "Previous segment ends (s)",
                                     "Next segment starts (s)", "Gap the model left (s)",
                                     "Syllables in the gap"], rows))
                parts.append(
                    "\n*Whisper's own timestamps, not ours: the model closed one segment at or before the "
                    "start of the digital silence and opened the next at or after its end, leaving the hole "
                    "empty. A model that fabricated over the silence would have put a segment in that gap.*\n")
        else:
            examples = []
            for key in ("whisper", "phowhisper"):
                for ex in inv["per_arm"][key]["worst_examples"]:
                    examples.append([f"`{key}`", f"`{ex['audio'].split('/')[-1]}`", ex["timestamp"],
                                     ex["window"], ex["syllables"], f"“{_quote(ex['text'])}”"])
            parts.append("### 2b. Verbatim worst cases\n")
            parts.append(_table(["Arm", "File", "Detected at", "Silent window", "Syllables",
                                 "Model output (verbatim)"], examples))
            parts.append(
                "\n*The longest text each arm produced inside a silent window, quoted exactly as the model "
                "returned it. Where an arm has no per-word timestamps the quote is the surplus syllables "
                "themselves and the timestamp is marked n/a.*\n")

    # ---------------------------------------------------------------- 3
    catch = report.get("catch")
    if catch:
        fp = catch["false_positive"]
        rows = [["All arms", catch["probe_arm_pairs"], catch["silences_with_invention"],
                 catch["flagged"], _pct(catch["catch_rate"]), "", ""]]
        for key in ("whisper", "phowhisper"):
            s = catch["per_arm"][key]
            rows.append([f"`{key}`", s["pairs"], s["pairs_with_invention"], s["flagged"],
                         _pct(s["catch_rate"]), s["candidates"], s["suppressed_by_other_arm"]])
        parts.append("## 3. Does the two-model diff catch it?\n")
        parts.append(_table(
            ["Arm", "Probe×arm pairs", "Pairs with invention", "Flagged as invented", "Catch rate",
             "Segments ≥50% inside detected silence", "…of those, suppressed by the other arm"], rows))
        parts.append(
            f"\n*An (arm, silence) pair “has invention” when that arm added syllables because the silence was "
            f"there; it is “flagged” when `core.asr` emitted an `invented` span for that arm overlapping the "
            f"silent window (both conditions: the segment is ≥50% inside detected non-speech **and** the other "
            f"arm has no segment there). With {catch['silences_with_invention']} pairs of invention the catch "
            f"rate is undefined rather than 0/0-favourable. The last two columns are the mechanism, not an "
            f"excuse: they count the segments each arm placed ≥50% inside detected non-speech — the raw "
            f"material for an invented flag — and how many of those the other arm's segments overlapped, which "
            f"suppresses the flag by construction. `invented_spans` fired "
            f"{catch['spans_fired_anywhere']} time(s) across all {catch['recordings_checked']} recordings in "
            f"this evaluation.{_leak_note(catch)}*\n")

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
            f"({_f(fp['clean_audio_seconds'], 1)} s), which contain no added silence — only the pauses the "
            f"speakers actually left. A flag here is not automatically wrong, because clean speech has natural "
            f"silence in it; the last column is the honest error count: spans whose text is literally present "
            f"in the published reference transcript, i.e. words that were really spoken and were labelled "
            f"invented. {fp['syllables_quoting_reference_text']} of {fp['invented_syllables']} flagged "
            f"syllables are in the reference. A detector that never fires has a false-positive rate of zero "
            f"and a recall of zero, and only the pair of numbers is honest.*\n")

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
