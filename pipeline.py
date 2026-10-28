"""Sổ Giỗ — the family memorial book whose ears admit what they didn't hear.

One run takes a recording and produces:
  - a reading copy of what both speech models heard, with the disputed and invented spans visible;
  - the claims that survived the guard, each with the timestamp it came from;
  - the next ten years of giỗ dates, computed from the lunar calendar, one per ancestor named;
  - and the list of things we refused to invent.

Usage:
    python pipeline.py --audio path/to/recording.wav [--out data/runs/<id>] [--model gemma3:1b-it-qat]
    python pipeline.py --self-test          # runs the guard against an authored fixture, no audio needed
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

from core import asr, extract, guard, telemetry, textnorm, lunar

ROOT = pathlib.Path(__file__).resolve().parent
TODAY = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).date()


def grounded_pieces(hearing: asr.Hearing) -> tuple[list[tuple[float, float, str]], list[asr.Segment]]:
    """Split what the two models heard into quotable and unquotable spans.

    Quotable means: the syllables both arms produced, with the timestamp of the segment they came from.
    Everything else — a disagreement, or one arm talking over silence the other heard — stays in the
    reading copy and never enters extraction. This is the function the whole product rests on: it is why a
    claim can be cited, and why an invented sentence has nowhere to go.
    """
    quotable = asr.agreed_spans(hearing)
    invented_ids = {(s.start, s.end) for s in hearing.invented_spans}
    excluded = [seg for seg in hearing.arms[0].segments if (seg.start, seg.end) in invented_ids]
    return quotable, excluded


def calendar_for(kept: list[extract.Claim], today: tuple[int, int, int], years: int = 10) -> list[dict]:
    """Ten years of dates per named ancestor, computed — never generated."""
    out = []
    for claim in kept:
        if claim.kind != "gio_date" or claim.lunar_day is None or claim.lunar_month is None:
            continue
        dates = lunar.next_occurrence(claim.lunar_month, claim.lunar_day, today,
                                      years_ahead=years, leap=claim.leap)
        out.append({
            "lunar": f"{claim.lunar_day:02d}/{claim.lunar_month:02d}{' nhuận' if claim.leap else ''}",
            "quote": claim.quote,
            "citation": claim.citation,
            "next": [f"{y:04d}-{m:02d}-{d:02d}" for y, m, d in dates],
        })
    return out


def run(audio: pathlib.Path, out_dir: pathlib.Path, model: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    traced = telemetry.init("pipeline")

    with telemetry.transaction("so-gio pipeline run", op="pipeline.run", model=model,
                               audio_seconds=asr.probe_duration(str(audio))) as root:
        print(f"[1/4] hearing {audio.name} twice (whisper + phowhisper) …", flush=True)
        hearing = asr.hear(str(audio))
        for arm in hearing.arms:
            print(f"      {arm.arm:11s} {arm.latency_s:6.1f}s for {arm.audio_seconds:5.1f}s audio "
                  f"({arm.realtime_factor:.2f}x realtime)")
            # Numbers only. The transcript itself is never sent: see core/telemetry.py.
            telemetry.trace_result("asr.arm", arm=arm.arm, model=arm.model,
                                   latency_ms=round(arm.latency_s * 1000),
                                   realtime_factor=round(arm.realtime_factor, 3),
                                   chars=len(arm.text), segments=len(arm.segments or []))
        telemetry.trace_result("asr.agreement", agreement_rate=round(hearing.agreement_rate, 4),
                               silences=len(hearing.silences),
                               invented_spans=len(hearing.invented_spans))

    # The timestamp heuristic inside `hear()` cannot flag invention (Whisper emits one segment that spans the
    # silence), so the pipeline asks the models directly: hand each non-speech window over alone, and whatever
    # comes back is invented by construction. Without this step the reading copy would show "0 invented" on a
    # recording that is 34% verified silence, which is a worse lie than showing nothing.
    with telemetry.span("pipeline.probe_silences", silences=len(hearing.silences)) as _probe:
        probed = asr.probe_silences(str(audio))
        seen = {(round(sp.start, 2), round(sp.end, 2)) for sp in hearing.invented_spans}
        fresh = [sp for sp in probed if (round(sp.start, 2), round(sp.end, 2)) not in seen]
        hearing.invented_spans.extend(fresh)
        telemetry.trace_result("probe.result", probed=len(probed), new=len(fresh),
                               syllables=sum(len((sp.text_a or sp.text_b or "").split()) for sp in probed))

    with telemetry.span("pipeline.ground") as _:
        quotable, excluded = grounded_pieces(hearing)
        telemetry.trace_result("ground.result", quotable=len(quotable), excluded=len(excluded),
                               invented_over_silence=len(hearing.invented_spans))
    print(f"[2/4] {len(quotable)} quotable spans, {len(excluded)} excluded, "
          f"{len(hearing.invented_spans)} invented over silence, "
          f"agreement {hearing.agreement_rate:.1%}")

    print(f"[3/4] asking {model} for claims (it may only quote) …", flush=True)
    with telemetry.span("pipeline.extract", model=model, quotable_pieces=len(quotable)) as sp:
        prompt = extract.build_prompt([(t, text) for t, _, text in quotable])
        claims, usage = extract.ollama_extract(prompt, model=model)
        sp.set_data("prompt_tokens", (usage or {}).get("prompt_eval_count"))
        sp.set_data("completion_tokens", (usage or {}).get("eval_count"))
        sp.set_data("claims_suggested", len(claims))

    with telemetry.span("pipeline.guard", claims=len(claims)):
        verdict = guard.review(claims, quotable, (TODAY.year, TODAY.month, TODAY.day))
        for claim in verdict.refused:
            telemetry.capture_refusal(getattr(claim, "reason", "unknown"), claim.kind,
                                      telemetry.redact(claim.quote))
        telemetry.trace_result("guard.result", kept=len(verdict.kept), refused=len(verdict.refused),
                               uncited_kept=verdict.uncited_kept)
    print(f"      {len(claims)} suggested → {len(verdict.kept)} kept, {len(verdict.refused)} refused")

    print("[4/4] computing dates …", flush=True)
    with telemetry.span("pipeline.calendar", kept=len(verdict.kept)):
        cal = calendar_for(verdict.kept, (TODAY.year, TODAY.month, TODAY.day))
        telemetry.trace_result("calendar.result", years=len(cal),
                               dates_resolved=sum(1 for c in cal if c.get("solar_next")))

    result = {
        "audio": str(audio),
        "audio_seconds": hearing.audio_seconds,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "usage": usage,
        "arms": [{"arm": a.arm, "model": a.model, "latency_s": round(a.latency_s, 2),
                  "realtime_factor": round(a.realtime_factor, 2), "chars": len(a.text)}
                 for a in hearing.arms],
        "agreement_rate": round(hearing.agreement_rate, 4),
        "silences": [{"start": round(s.start, 2), "end": round(s.end, 2)} for s in hearing.silences],
        "invented": [{"text": s.text_a or s.text_b, "start": s.start, "end": s.end}
                     for s in hearing.invented_spans],
        "quotable_spans": [{"start": t, "end": e, "text": txt} for t, e, txt in quotable],
        "excluded_spans": [{"start": s.start, "end": s.end, "text": s.text} for s in excluded],
        "claims_kept": [c.as_dict() for c in verdict.kept],
        "claims_refused": [c.as_dict() for c in verdict.refused],
        "calendar": cal,
        "summary": verdict.summary(),
    }
    (out_dir / "run.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (out_dir / "reading_copy.md").write_text(render_reading_copy(hearing, excluded, verdict), encoding="utf-8")
    print(f"      wrote {out_dir}/run.json and reading_copy.md")
    if traced:
        telemetry.flush(5.0)
    return result


def render_reading_copy(hearing: asr.Hearing, excluded: list[asr.Segment], verdict: guard.Verdict) -> str:
    def mmss(sec: float | None) -> str:
        m, s = divmod(int(round(sec or 0)), 60)
        return f"{m:02d}:{s:02d}"

    lines = ["# Bản đọc lại / Reading copy", "",
             f"Audio: `{hearing.audio}` — {hearing.audio_seconds:.1f}s. "
             f"Two models agreed on {hearing.agreement_rate:.0%} of the syllables they heard.", "",
             "**What both models heard** — safe to quote:", ""]
    for seg in hearing.arms[0].segments:
        if seg not in excluded:
            lines.append(f"- `{mmss(seg.start)}` {seg.text}")
    if hearing.invented_spans:
        lines += ["", "**Produced over silence** — one model wrote this where the other heard nothing, "
                      "so it is not part of the record and cannot become a date:", ""]
        for span in hearing.invented_spans:
            who = "whisper" if span.text_a else "phowhisper"
            lines.append(f"- `{mmss(span.start)}` *{who} invented:* \"{span.text_a or span.text_b}\"")
    disputed = [s for s in hearing.spans if s.kind in ("disputed",)]
    if disputed:
        lines += ["", "**They disagree** — both readings below, neither is treated as fact:", ""]
        for span in disputed[:20]:
            lines.append(f"- whisper: \"{span.text_a}\" · phowhisper: \"{span.text_b}\"")
    lines += ["", "**Claims that survived the guard:**", ""]
    for claim in verdict.kept:
        extra = f" → next {claim.solar_next}" if claim.solar_next else ""
        lines.append(f"- [{claim.citation}] {claim.kind}: \"{claim.quote}\"{extra}")
    if verdict.refused:
        lines += ["", "**Refused (and why):**", ""]
        for claim in verdict.refused:
            lines.append(f"- {claim.reason}: \"{claim.quote[:120]}\"")
    return "\n".join(lines) + "\n"


SELF_TEST_PIECES = [
    (0.0, 6.0, "Giỗ ông nội là ngày mười hai tháng bảy âm lịch."),
    (6.0, 12.0, "Mâm cúng phải có gà luộc và xôi gấc, nhớ mua hoa cúc."),
    (12.5, 20.0, "Con nhớ thắp nhang rồi đợi nhang tàn mới dọn đồ xuống."),
]


def self_test(model: str = "gemma3:1b-it-qat") -> int:
    """Exercise the guard on an authored transcript with no audio, so CI can run without a model."""
    print("self-test: guard on an authored fixture (no audio, no model needed for the guard itself)")
    claims = [
        extract.Claim(kind="offering", quote="gà luộc và xôi gấc, nhớ mua hoa cúc"),
        extract.Claim(kind="offering", quote="bánh chưng và giò lụa"),           # never said
        extract.Claim(kind="gio_date", quote="mười hai tháng bảy âm lịch",
                      lunar_day=12, lunar_month=7),
        extract.Claim(kind="gio_date", quote="mười hai tháng bảy âm lịch",
                      lunar_day=31, lunar_month=7),                             # impossible
    ]
    verdict = guard.review(claims, SELF_TEST_PIECES, (TODAY.year, TODAY.month, TODAY.day))
    print(json.dumps(verdict.summary(), ensure_ascii=False, indent=2))
    for c in verdict.kept:
        print(f"  KEPT    {c.kind:10s} [{c.citation}] \"{c.quote}\" next={c.solar_next}")
    for c in verdict.refused:
        print(f"  REFUSED {c.kind:10s} ({c.reason})")
    ok = len(verdict.kept) == 2 and len(verdict.refused) == 2 and verdict.uncited_kept == 0
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--model", default="gemma3:1b-it-qat")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(args.model)
    if not args.audio:
        ap.error("--audio is required unless --self-test is given")
    out = args.out or (ROOT / "data" / "runs" / args.audio.stem)
    run(args.audio, out, args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
