#!/usr/bin/env python3
"""The harness behind every number this project publishes.

One command, idempotent:

    python eval/run_eval.py            # run everything that can be run
    python eval/run_eval.py --stage asr      # only measurements 1-3 (all cached, seconds)
    python eval/run_eval.py --stage extract  # only measurement 4
    python eval/run_eval.py --force    # ignore the ASR cache

Four measurements, each written to data/eval/report.json and rendered into
data/eval/tables.md:

1. syllable WER per ASR arm on the 18 FLEURS clips that have a published reference
   transcript, plus each arm's realtime factor on this machine;
2. syllables each arm emits *inside* the digitally spliced silences, where the
   reference for that window is nothing;
3. whether the two-model diff (`asr.Hearing.invented_spans`, `asr.agreed_spans`)
   actually catches measurement 2 — and how often it cries wolf on clean audio;
4. four extraction arms (two local LLMs, regex, degenerate regex) scored against a
   held-out gold set, every arm passed through `core.guard.review`.

ASR output is cached per file under data/eval/cache/, keyed by the file's sha256 and
the model names, so a re-run costs nothing and a crash never loses completed work.
Nothing here writes outside eval/ and data/eval/.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import asr, baselines, extract, guard, textnorm  # noqa: E402

CACHE = ROOT / "data" / "eval" / "cache"
OUT = ROOT / "data" / "eval"
TZ7 = datetime.timezone(datetime.timedelta(hours=7))
TODAY = datetime.datetime.now(TZ7).date()

CACHE_VERSION = "asr-cache-v1"


# --------------------------------------------------------------------------- helpers

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def hearing_to_dict(h: asr.Hearing) -> dict:
    from dataclasses import asdict

    return asdict(h)


def hearing_from_dict(d: dict) -> asr.Hearing:
    h = asr.Hearing(audio=d["audio"], audio_seconds=d["audio_seconds"])
    h.arms = [asr.ArmResult(a["arm"], a["model"], a["text"],
                            [asr.Segment(**s) for s in a["segments"]],
                            a["latency_s"], a["audio_seconds"]) for a in d["arms"]]
    h.silences = [asr.Silence(**s) for s in d["silences"]]
    h.spans = [asr.Span(**s) for s in d["spans"]]
    h.invented_spans = [asr.Span(**s) for s in d["invented_spans"]]
    h.agreement_rate = d.get("agreement_rate", 0.0)
    return h


class Runner:
    """Runs `asr.hear` once per file, caches the whole Hearing, reuses it forever."""

    def __init__(self, force: bool = False):
        self.force = force
        self.whisper_model = None
        self.models_loaded = False
        self.cache_dir = CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, audio: Path, digest: str) -> Path:
        return self.cache_dir / f"{audio.stem}.{digest[:12]}.json"

    def hearing(self, rel_path: str) -> dict:
        audio = ROOT / rel_path
        digest = sha256(audio)
        target = self._key(audio, digest)
        if target.exists() and not self.force:
            d = load_json(target)
            if d.get("cache_version") == CACHE_VERSION:
                return d
        t0 = time.time()
        self._warm()
        h = asr.hear(str(audio), self.whisper_model)
        d = hearing_to_dict(h)
        d.update({
            "cache_version": CACHE_VERSION,
            "audio": rel_path,
            "audio_sha256": digest,
            "measured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "wall_s": round(time.time() - t0, 2),
            "models": {"whisper": asr.WHISPER_SIZE, "phowhisper": asr.PHO_MODEL},
        })
        write_json(target, d)
        return d

    def _warm(self):
        """Load both models before the timed loop so per-clip latency is inference only."""
        if self.models_loaded:
            return
        from faster_whisper import WhisperModel

        self.whisper_model = WhisperModel(asr.WHISPER_SIZE, device="cpu", compute_type="int8",
                                          download_root=str(ROOT / "data" / "models" / "fw"))
        # PhoWhisper lazy-loads inside run_phowhisper; touching it here makes the first
        # timed clip comparable to the rest.
        asr.run_phowhisper(__import__("numpy").zeros(1600, dtype="float32"), "")
        self.models_loaded = True


def arm_of(h: dict) -> dict[str, dict]:
    return {a["arm"]: a for a in h["arms"]}


def syllables_in(arm: dict, start: float, end: float, rule: str = "midpoint") -> tuple[int, list[dict]]:
    """Syllables this arm emitted inside [start, end].

    rule="midpoint": a segment counts when its timestamp midpoint lies inside the window —
    the plain reading of "words whose timestamps fall inside the inserted silence".
    rule="overlap50": a segment counts when >=50% of its duration is inside the window,
    which is the test `asr._in_silence` uses.
    """
    total = 0
    segs: list[dict] = []
    for seg in arm["segments"]:
        dur = max(seg["end"] - seg["start"], 1e-6)
        if rule == "midpoint":
            inside = start <= (seg["start"] + seg["end"]) / 2.0 <= end
        else:
            lo, hi = max(seg["start"], start), min(seg["end"], end)
            inside = (hi - lo) / dur >= 0.5
        if not inside:
            continue
        n = len(textnorm.syllables(seg["text"]))
        if n:
            total += n
            segs.append({"start": round(seg["start"], 2), "end": round(seg["end"], 2),
                         "text": seg["text"], "syllables": n})
    return total, segs


def mmss(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m:02d}:{s:02d}"


# --------------------------------------------------------------------------- 1. ASR accuracy

def measure_asr(runner: Runner, manifest: dict) -> dict:
    rows, per_arm = [], {"whisper": [], "phowhisper": []}
    for clip in manifest["clips"]:
        h = runner.hearing(clip["audio_path"])
        arms = arm_of(h)
        ref = clip["reference_transcript"]
        row = {"id": clip["id"], "audio_path": clip["audio_path"],
               "audio_seconds": h["audio_seconds"],
               "reference_syllables": len(textnorm.syllables(ref))}
        for name in ("whisper", "phowhisper"):
            a = arms[name]
            w = textnorm.wer(ref, a["text"])
            row[name] = {
                "wer": round(w, 4),
                "hypothesis_syllables": len(textnorm.syllables(a["text"])),
                "latency_s": round(a["latency_s"], 2),
                "realtime_factor": round(a["audio_seconds"] / a["latency_s"], 3) if a["latency_s"] else 0.0,
                "text": a["text"],
            }
            per_arm[name].append(row[name] | {"audio_seconds": h["audio_seconds"],
                                              "ref_syllables": row["reference_syllables"]})
        rows.append(row)

    summary = {}
    for name, rs in per_arm.items():
        total_audio = sum(r["audio_seconds"] for r in rs)
        total_latency = sum(r["latency_s"] for r in rs)
        total_ref = sum(r["ref_syllables"] for r in rs)
        # micro WER: total edits / total reference syllables, weighted like a corpus score
        edits = sum(round(r["wer"] * r["ref_syllables"]) for r in rs)
        summary[name] = {
            "model": asr.WHISPER_SIZE if name == "whisper" else asr.PHO_MODEL,
            "n_clips": len(rs),
            "audio_seconds": round(total_audio, 2),
            "mean_wer": round(sum(r["wer"] for r in rs) / len(rs), 4),
            "median_wer": round(sorted(r["wer"] for r in rs)[len(rs) // 2], 4),
            "min_wer": round(min(r["wer"] for r in rs), 4),
            "max_wer": round(max(r["wer"] for r in rs), 4),
            "micro_wer": round(edits / total_ref, 4) if total_ref else 0.0,
            "total_latency_s": round(total_latency, 1),
            "realtime_factor_aggregate": round(total_audio / total_latency, 3) if total_latency else 0.0,
            "realtime_factor_median": round(sorted(r["realtime_factor"] for r in rs)[len(rs) // 2], 3),
            "realtime_factor_min": round(min(r["realtime_factor"] for r in rs), 3),
        }
    return {"per_arm": summary, "clips": rows}


# --------------------------------------------------------------------------- 2. invention over silence

TAIL_SECONDS = 5.0
PROBE_DIR = ROOT / "data" / "eval" / "probe"
# Every third clean clip, in manifest order: six probes spanning 6.9 s to 13.9 s, none of them
# costing an extra clean-audio pass that is not already cached. Enough to say something real on
# this 4-core box without spending an hour of CPU on a probe whose expected rate is ~1%.
TAIL_STRIDE = 3


def build_probes(silence_manifest: dict, manifest: dict) -> list[dict]:
    """Two probe families, both with a window where the reference transcript is empty.

    `splice` — digital silence spliced into a real mid-utterance pause (from the corpus manifest).
    `tail`   — 5 s of digital silence appended after the last word, which is what a family
               recording actually looks like when someone forgets to stop the phone. Built here,
               deterministically, from the same clean clips.
    """
    probes = []
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    for clip in silence_manifest["clips"]:
        for ins in clip["insertions"]:
            probes.append({
                "probe": "splice", "clip_id": clip["id"], "source_clip_id": clip["source_clip_id"],
                "audio_path": clip["output_audio"], "clean_audio": clip["source_audio"],
                "window_start": round(ins["offset_seconds"], 3),
                "window_end": round(ins["offset_seconds"] + ins["duration_seconds"], 3),
                "note": "digital silence spliced into a detected mid-utterance pause",
            })
        # Third condition: hand the verified non-vocal window to each model on its own. There is no
        # speech in the input at all, so anything that comes back is fabrication by construction —
        # no timestamps, no second opinion, no threshold to tune.
        for i, win in enumerate(clip.get("silent_windows", [])):
            out = PROBE_DIR / f"iso_{clip['id']}_{i}.wav"
            if not out.exists():
                import subprocess
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(ROOT / clip["output_audio"]),
                                "-ss", f"{win['start_seconds']:.3f}",
                                "-t", f"{win['end_seconds'] - win['start_seconds']:.3f}",
                                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)], check=True)
            probes.append({
                "probe": "isolated", "clip_id": f"iso_{clip['id']}", "source_clip_id": clip["source_clip_id"],
                "audio_path": str(out.relative_to(ROOT)), "clean_audio": None,
                "window_start": 0.0,
                "window_end": round(win["end_seconds"] - win["start_seconds"], 3),
                "note": "verified non-vocal window fed to the model alone; the reference is nothing",
            })
    for clip in manifest["clips"][::TAIL_STRIDE]:
        out = PROBE_DIR / f"tail_{clip['id']}.wav"
        if not out.exists():
            import subprocess
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(ROOT / clip["audio_path"]),
                            "-af", f"apad=pad_dur={TAIL_SECONDS}", "-c:a", "pcm_s16le", str(out)],
                           check=True)
        probes.append({
            "probe": "tail", "clip_id": f"tail_{clip['id']}", "source_clip_id": clip["id"],
            "audio_path": str(out.relative_to(ROOT)), "clean_audio": clip["audio_path"],
            "window_start": round(clip["duration_seconds"], 3),
            "window_end": round(clip["duration_seconds"] + TAIL_SECONDS, 3),
            "note": f"{TAIL_SECONDS:.0f} s of digital silence appended after the last word",
        })
    return probes


def content_delta(clean_text: str, probe_text: str) -> tuple[int, str]:
    """Syllables the model produced on the probed file that have no counterpart in its own
    transcript of the unmodified source — the timestamp-free definition of invention.

    Needed because one arm (PhoWhisper) returns a single segment per 25 s window, so its
    timestamps cannot say *where* inside the window a syllable was spoken. Comparing content is
    the only fair test for both arms, and it also catches the classic failure of repeating real
    words into the silence, which a timestamp test would miss.
    """
    d = textnorm.diff_transcripts(clean_text, probe_text)
    return len(d.only_b), " ".join(d.only_b)


def measure_invention(runner: Runner, probes: list[dict]) -> dict:
    events = []
    for p in probes:
        h = runner.hearing(p["audio_path"])
        hc = runner.hearing(p["clean_audio"]) if p["clean_audio"] else None
        arms = arm_of(h)
        clean_arms = arm_of(hc) if hc else {}
        for name in ("whisper", "phowhisper"):
            n_mid, segs_mid = syllables_in(arms[name], p["window_start"], p["window_end"], "midpoint")
            n_ov, _ = syllables_in(arms[name], p["window_start"], p["window_end"], "overlap50")
            # With no clean counterpart (isolated window) the reference is nothing, so every
            # syllable the model returns is a content delta.
            added, added_text = content_delta(
                clean_arms[name]["text"] if clean_arms else "", arms[name]["text"])
            durs = sorted(s["end"] - s["start"] for s in arms[name]["segments"]) or [0.0]
            median_seg = durs[len(durs) // 2]
            before = [s["end"] for s in arms[name]["segments"] if s["end"] <= p["window_start"]]
            after = [s["start"] for s in arms[name]["segments"] if s["start"] >= p["window_end"]]
            events.append({
                "probe": p["probe"], "clip_id": p["clip_id"], "source_clip_id": p["source_clip_id"],
                "audio_path": p["audio_path"], "arm": name,
                "window_start": p["window_start"], "window_end": p["window_end"],
                "window_seconds": round(p["window_end"] - p["window_start"], 3),
                "timestamp_usable": median_seg <= 8.0,
                "median_segment_seconds": round(median_seg, 2),
                "prev_segment_end": max(before) if before else None,
                "next_segment_start": min(after) if after else None,
                "syllables": n_mid,                      # timestamp-localised
                "syllables_overlap50": n_ov,
                "syllables_added": added,                # content delta — the published number
                "added_text": added_text,
                "wpm_of_silence": round(added / ((p["window_end"] - p["window_start"]) / 60.0), 1)
                if p["window_end"] > p["window_start"] else 0.0,
                "segments": segs_mid,
                "text_probe": arms[name]["text"],
                "text_clean": clean_arms[name]["text"] if clean_arms else "",
            })

    per_arm, by_probe = {}, {}
    for name in ("whisper", "phowhisper"):
        evs = [e for e in events if e["arm"] == name]
        total_syl = sum(e["syllables_added"] for e in evs)
        total_sec = sum(e["window_seconds"] for e in evs)
        usable = all(e["timestamp_usable"] for e in evs) if evs else False
        worst = sorted((e for e in evs if e["syllables_added"]), key=lambda e: -e["syllables_added"])[:2]
        worst_examples = []
        for e in worst:
            inside = sorted(e["segments"], key=lambda s: -s["syllables"])
            if inside:
                for seg in inside[:2]:
                    worst_examples.append({
                        "clip_id": e["clip_id"], "audio": e["audio_path"], "arm": name,
                        "timestamp": f"{mmss(seg['start'])}–{mmss(seg['end'])}",
                        "start": seg["start"], "end": seg["end"], "text": seg["text"],
                        "syllables": seg["syllables"],
                        "window": f"{mmss(e['window_start'])}–{mmss(e['window_end'])}",
                        "located": True,
                    })
            else:  # no usable timestamps: quote the surplus syllables themselves
                worst_examples.append({
                    "clip_id": e["clip_id"], "audio": e["audio_path"], "arm": name,
                    "timestamp": "n/a", "start": None, "end": None,
                    "text": e["added_text"] or "(none)", "syllables": e["syllables_added"],
                    "window": f"{mmss(e['window_start'])}–{mmss(e['window_end'])}",
                    "located": False,
                })
            if len(worst_examples) >= 2:
                break
        per_arm[name] = {
            "probes": len(evs),
            "probes_with_invention": sum(1 for e in evs if e["syllables_added"] > 0),
            "invented_syllables": total_syl,
            "syllables_in_window_by_timestamp": sum(e["syllables"] for e in evs),
            "silence_seconds": round(total_sec, 2),
            "wpm_of_silence": round(total_syl / (total_sec / 60.0), 1) if total_sec else 0.0,
            "timestamp_usable": usable,
            "timestamp_usable_probes": f"{sum(1 for e in evs if e['timestamp_usable'])}/{len(evs)}",
            "median_segment_seconds": round(sorted(e["median_segment_seconds"] for e in evs)[len(evs) // 2], 2) if evs else None,
            "worst_examples": worst_examples[:2],
        }
        for kind in ("isolated", "splice", "tail"):
            sub = [e for e in evs if e["probe"] == kind]
            sec = sum(e["window_seconds"] for e in sub)
            per_arm[name].setdefault("by_probe", {})[kind] = {
                "probes": len(sub),
                "probes_with_invention": sum(1 for e in sub if e["syllables_added"] > 0),
                "invented_syllables": sum(e["syllables_added"] for e in sub),
                "silence_seconds": round(sec, 2),
                "wpm_of_silence": round(sum(e["syllables_added"] for e in sub) / (sec / 60.0), 1) if sec else 0.0,
            }
    return {"events": events, "per_arm": per_arm,
            "silence_seconds_total": round(sum(e["window_seconds"] for e in events
                                               if e["arm"] == "whisper"), 2),
            "probe_count": {kind: sum(1 for p in probes if p["probe"] == kind)
                            for kind in ("isolated", "splice", "tail")}}


# --------------------------------------------------------------------------- 3. does the diff catch it?

def measure_catch(runner: Runner, probes: list[dict], manifest: dict, invention: dict) -> dict:
    units = []
    for p in probes:
        h = runner.hearing(p["audio_path"])
        hearing = hearing_from_dict(h)
        invented = h["invented_spans"]
        quotable = asr.agreed_spans(hearing)
        evs = {e["arm"]: e for e in invention["events"] if e["clip_id"] == p["clip_id"]}
        start, end = p["window_start"], p["window_end"]
        for name in ("whisper", "phowhisper"):
            ev = evs[name]
            flagged_spans, leaked = [], 0
            for span in invented:
                who = "whisper" if span["text_a"] else "phowhisper"
                if who != name or span["start"] is None or span["end"] is None:
                    continue
                if span["start"] < end and start < span["end"]:  # any overlap with the dead air
                    flagged_spans.append(span)
            for qs, qe, qtext in quotable:
                if not (qs < end and start < qe):
                    continue
                qn = textnorm.normalise(qtext)
                for seg in ev["segments"]:
                    sn = textnorm.normalise(seg["text"])
                    if sn and sn in qn:
                        leaked += seg["syllables"]
            units.append({
                "probe": p["probe"], "clip_id": p["clip_id"], "arm": name,
                "window": f"{mmss(start)}–{mmss(end)}",
                "invented_syllables": ev["syllables_added"],
                "invented_syllables_by_timestamp": ev["syllables"],
                "flagged": bool(flagged_spans),
                "flagged_spans": [{"start": s["start"], "end": s["end"],
                                   "text": s["text_a"] or s["text_b"]} for s in flagged_spans],
                "flagged_syllables": sum(len(textnorm.syllables(s["text_a"] or s["text_b"]))
                                         for s in flagged_spans),
                "leaked_into_quotable_syllables": leaked,
            })

    with_inv = [u for u in units if u["invented_syllables"] > 0]
    flagged = [u for u in with_inv if u["flagged"]]

    # Why the invented flag can or cannot fire: count each arm's segments that sit ≥50% inside
    # detected non-speech, and how many of those the *other* arm's segments overlapped (which
    # suppresses the flag by construction in core/asr.hear).
    cand = {name: {"candidates": 0, "suppressed_by_other_arm": 0} for name in ("whisper", "phowhisper")}
    for p in probes:
        h = runner.hearing(p["audio_path"])
        hearing = hearing_from_dict(h)
        arms = {a["arm"]: a for a in h["arms"]}
        for name, other in (("whisper", "phowhisper"), ("phowhisper", "whisper")):
            for seg in arms[name]["segments"]:
                if asr._in_silence(asr.Segment(**seg), hearing.silences, 0.5):
                    cand[name]["candidates"] += 1
                    if any(asr._overlaps(seg["start"], seg["end"], o["start"], o["end"])
                           for o in arms[other]["segments"]):
                        cand[name]["suppressed_by_other_arm"] += 1

    per_arm = {}
    for name in ("whisper", "phowhisper"):
        us = [u for u in with_inv if u["arm"] == name]
        per_arm[name] = {"pairs": len([u for u in units if u["arm"] == name]),
                         "pairs_with_invention": len(us),
                         "flagged": sum(1 for u in us if u["flagged"]),
                         "catch_rate": round(sum(1 for u in us if u["flagged"]) / len(us), 3) if us else None,
                         "invented_syllables": sum(u["invented_syllables"] for u in us),
                         "flagged_syllables": sum(u["flagged_syllables"] for u in us),
                         "leaked_into_quotable_syllables": sum(u["leaked_into_quotable_syllables"] for u in us),
                         **cand[name]}

    # ---- false positives: the pipeline's invented flag on clips with no added silence
    fp_rows = []
    clean_flagged_clips = 0
    total_spans = 0
    total_syl = 0
    in_reference_spans = 0
    in_reference_syl = 0
    spans_anywhere = 0
    for clip in manifest["clips"]:
        h = runner.hearing(clip["audio_path"])
        spans = h["invented_spans"]
        spans_anywhere += len(spans)
        syls = sum(len(textnorm.syllables(s["text_a"] or s["text_b"])) for s in spans)
        row = {"id": clip["id"], "invented_spans": len(spans), "invented_syllables": syls,
               "audio_seconds": h["audio_seconds"], "spans": []}
        if spans:
            clean_flagged_clips += 1
        total_spans += len(spans)
        total_syl += syls
        for s in spans:
            text = s["text_a"] or s["text_b"]
            words = textnorm.syllables(text)
            hit = textnorm.contained(text, clip["reference_transcript"])
            hit_syl = sum(1 for w in words if w in textnorm.syllables(clip["reference_transcript"]))
            row["spans"].append({"start": s["start"], "end": s["end"], "text": text,
                                 "syllables": len(words), "in_reference": hit,
                                 "syllables_in_reference": hit_syl})
            if hit:
                in_reference_spans += 1
                in_reference_syl += len(words)
        fp_rows.append(row)

    for p in probes:
        spans_anywhere += len(runner.hearing(p["audio_path"])["invented_spans"])

    n_clean = len(manifest["clips"])
    return {
        "units": units,
        "probe_arm_pairs": len(units),
        "silences_with_invention": len(with_inv),
        "flagged": len(flagged),
        "catch_rate": round(len(flagged) / len(with_inv), 3) if with_inv else None,
        "per_arm": per_arm,
        "spans_fired_anywhere": spans_anywhere,
        "recordings_checked": n_clean + len(probes),
        "false_positive": {
            "n_clean_clips": n_clean,
            "clips_with_invented_flag": clean_flagged_clips,
            "clip_false_positive_rate": round(clean_flagged_clips / n_clean, 3) if n_clean else None,
            "invented_spans": total_spans,
            "invented_syllables": total_syl,
            "clean_audio_seconds": round(sum(r["audio_seconds"] for r in fp_rows), 2),
            "spans_quoting_reference_text": in_reference_spans,
            "syllables_quoting_reference_text": in_reference_syl,
            "per_clip": fp_rows,
        },
    }



# --------------------------------------------------------------------------- 4. extraction arms
#
# The matching rule lives in exactly one place, `eval/metrics.claims_match`, because the Tinker
# arms are scored by a different process against the same gold set. Nothing here re-derives it.

from metrics import claims_match, score_claims  # noqa: E402


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def measure_extraction(heldout: Path) -> dict:
    if not heldout.exists():
        return {"status": "pending",
                "reason": f"{rel(heldout)} does not exist yet; measurement 4 not run",
                "path": rel(heldout)}

    rows = [json.loads(line) for line in heldout.read_text(encoding="utf-8").splitlines() if line.strip()]
    arms: dict[str, dict] = {}

    def register(name: str, model: str):
        arms[name] = {"arm": name, "model": model, "kept": [], "refused": 0,
                      "latencies": [], "uncited_kept": 0, "errors": [], "raw": [],
                      "tp": 0, "predicted": 0, "gold": 0}

    register("gemma3:1b-it-qat", "ollama/gemma3:1b-it-qat")
    register("qwen2.5:3b-instruct", "ollama/qwen2.5:3b-instruct")
    register("regex", "core.baselines.regex_baseline")
    register("regex_all_dates", "core.baselines.regex_baseline(all_dates=True)")

    def absorb(arm: dict, claims: list, pieces: list, item: dict, dt: float):
        verdict = guard.review(claims, pieces, (TODAY.year, TODAY.month, TODAY.day))
        kept = [c.as_dict() for c in verdict.kept]
        arm["kept"].extend(kept)
        arm["refused"] += len(verdict.refused)
        arm["uncited_kept"] += verdict.uncited_kept
        arm["latencies"].append(dt)
        # greedy one-to-one, scored per recording so a claim can only match gold from its own clip
        s = score_claims(kept, item["gold"])
        arm["tp"] += s["tp"]
        arm["predicted"] += s["predicted"]
        arm["gold"] += s["gold"]
        arm["raw"].append({"id": item["id"], "suggested": len(claims), "kept": len(kept),
                           "refused": len(verdict.refused), "tp": s["tp"],
                           "latency_s": round(dt, 3)})

    for item in rows:
        pieces = [(float(p[0]), float(p[1]), p[2]) for p in item["pieces"]]
        prompt = extract.build_prompt([(t, txt) for t, _, txt in pieces])
        for name in ("gemma3:1b-it-qat", "qwen2.5:3b-instruct"):
            arm = arms[name]
            t0 = time.time()
            try:
                claims, usage = extract.ollama_extract(prompt, model=name)
                dt = time.time() - t0
                absorb(arm, claims, pieces, item, dt)
                arm["raw"][-1]["response_excerpt"] = usage.get("raw", "")[:400]
            except Exception as exc:  # a dead ollama must not lose the other arms
                arm["errors"].append({"id": item["id"], "error": f"{type(exc).__name__}: {exc}"})
        for name, all_dates in (("regex", False), ("regex_all_dates", True)):
            t0 = time.time()
            claims = baselines.regex_baseline(pieces, all_dates=all_dates)
            absorb(arms[name], claims, pieces, item, time.time() - t0)

    out = {}
    guard_failures = []
    for name, arm in arms.items():
        lat = arm["latencies"]
        tp, pred, gold = arm["tp"], arm["predicted"], arm["gold"]
        prec = tp / pred if pred else 0.0
        rec = tp / gold if gold else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        bug = arm["uncited_kept"] != 0
        out[name] = {
            "model": arm["model"], "n_items": len(rows),
            "guard_bug": bug,
            "precision": None if bug else round(prec, 4),
            "recall": None if bug else round(rec, 4),
            "f1": None if bug else round(f1, 4),
            "tp": tp, "predicted": pred, "gold": gold,
            "refusals": arm["refused"],
            "uncited_kept": arm["uncited_kept"],
            "mean_latency_s": round(sum(lat) / len(lat), 3) if lat else None,
            "total_latency_s": round(sum(lat), 2),
            "errors": arm["errors"],
            "per_item": arm["raw"],
        }
        if bug:
            guard_failures.append(f"{name}: {arm['uncited_kept']} kept claims without a citation")
            print(f"      !! GUARD BUG — {name} kept {arm['uncited_kept']} uncited claim(s); "
                  f"arm NOT scored", flush=True)

    return {"status": "run", "path": rel(heldout), "n_items": len(rows),
            "arms": out, "guard_failures": guard_failures,
            "gold_total": sum(len(i["gold"]) for i in rows)}


TINKER_FILES = {"tinker_base": "data/eval/tinker_base.json",
                "tinker_tuned": "data/eval/tinker_tuned.json"}


def read_tinker_arms() -> dict:
    """Optional fifth/sixth arm: scored by another process, rendered not recomputed.

    Each file is {"summary": {...}, "items": [...]}. We copy the summary through untouched —
    re-deriving someone else's numbers here would defeat the point of a shared matching rule.
    """
    found = {}
    for name, rel_path in TINKER_FILES.items():
        path = ROOT / rel_path
        if not path.exists():
            continue
        try:
            data = load_json(path)
        except Exception as exc:
            found[name] = {"scored_separately": True, "error": f"{type(exc).__name__}: {exc}",
                           "source": rel_path}
            continue
        summary = data.get("summary") if isinstance(data, dict) else None
        found[name] = {"scored_separately": True,
                       "remote": True,
                       "source": rel_path,
                       "summary": summary if isinstance(summary, dict) else {},
                       "n_items": len(data.get("items") or []) if isinstance(data, dict) else None}
    return found



# --------------------------------------------------------------------------- main

def environment() -> dict:
    import multiprocessing

    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    try:
        import torch
        torch_threads = torch.get_num_threads()
    except Exception:
        torch_threads = None
    return {
        "machine": platform.platform(),
        "cpu_model": next((l.split(":", 1)[1].strip()
                           for l in Path("/proc/cpuinfo").read_text().splitlines()
                           if l.startswith("model name")), "unknown"),
        "cpu_count": multiprocessing.cpu_count(),
        "cpu_threads_physical": os.cpu_count(),
        "torch_threads": torch_threads,
        "python": platform.python_version(),
        "faster_whisper": ver("faster_whisper"),
        "transformers": ver("transformers"),
        "torch": ver("torch"),
        "whisper_model": f"Systran/faster-whisper-{asr.WHISPER_SIZE} (CTranslate2 int8, cpu)",
        "phowhisper_model": asr.PHO_MODEL,
        "run_at": datetime.datetime.now(TZ7).isoformat(timespec="seconds"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="ignore the ASR cache")
    ap.add_argument("--stage", default="all", choices=["all", "asr", "extract"],
                    help="asr = measurements 1-3, extract = measurement 4")
    args = ap.parse_args(argv)

    manifest = load_json(ROOT / "data" / "corpus" / "manifest.json")
    silence_manifest = load_json(ROOT / "data" / "corpus" / "silence_manifest.json")
    OUT.mkdir(parents=True, exist_ok=True)

    report_path = OUT / "report.json"
    report = load_json(report_path) if report_path.exists() else {}
    report["environment"] = environment()
    report["corpus"] = {
        "name": manifest["corpus"],
        "clips_with_reference": len(manifest["clips"]),
        "audio_seconds": manifest["totals"]["audio_seconds"],
        "licence": manifest["licence"],
        "silence_clips": len(silence_manifest["clips"]),
        "inserted_silence_seconds": round(sum(i["duration_seconds"]
                                             for c in silence_manifest["clips"]
                                             for i in c["insertions"]), 2),
        "silence_licence": silence_manifest["clips"][0]["licence"],
        "probes": {
            "isolated": sum(len(c.get("silent_windows", [])) for c in silence_manifest["clips"]),
            "splice": len(silence_manifest["clips"]),
            "tail": len(manifest["clips"][::TAIL_STRIDE]),
            "tail_seconds": TAIL_SECONDS,
            "silence_seconds": round(sum(w["end_seconds"] - w["start_seconds"]
                                         for c in silence_manifest["clips"]
                                         for w in c.get("silent_windows", []))
                                     + TAIL_SECONDS * len(manifest["clips"][::TAIL_STRIDE]), 2),
        },
    }

    if args.stage in ("all", "asr"):
        runner = Runner(force=args.force)
        probes = build_probes(silence_manifest, manifest)
        t0 = time.time()
        print(f"[1/4] ASR accuracy: {len(manifest['clips'])} clips × 2 arms", flush=True)
        report["asr"] = measure_asr(runner, manifest)
        for name, s in report["asr"]["per_arm"].items():
            print(f"      {name:11s} mean WER {s['mean_wer']:.3f}  RTF {s['realtime_factor_aggregate']:.2f}x", flush=True)

        print(f"[2/4] invention over silence: {len(probes)} probes × 2 arms", flush=True)
        report["invention"] = measure_invention(runner, probes)
        for name, s in report["invention"]["per_arm"].items():
            print(f"      {name:11s} {s['invented_syllables']} syllables over {s['silence_seconds']}s "
                  f"({s['wpm_of_silence']} wpm); timestamps usable: {s['timestamp_usable']}", flush=True)

        print("[3/4] does the two-model diff catch it?", flush=True)
        report["catch"] = measure_catch(runner, probes, manifest, report["invention"])
        print(f"      pairs with invention {report['catch']['silences_with_invention']}"
              f"/{report['catch']['probe_arm_pairs']}, flagged {report['catch']['flagged']}; "
              f"FP on {report['catch']['false_positive']['clips_with_invented_flag']}"
              f"/{report['catch']['false_positive']['n_clean_clips']} clean clips; "
              f"spans fired anywhere: {report['catch']['spans_fired_anywhere']}", flush=True)
        print(f"      asr stages took {time.time() - t0:.0f}s wall", flush=True)
        write_json(report_path, report)

    if args.stage in ("all", "extract"):
        print("[4/4] extraction arms", flush=True)
        report["extraction"] = measure_extraction(ROOT / "data" / "extract" / "heldout.jsonl")
        if report["extraction"]["status"] == "pending":
            print(f"      PENDING: {report['extraction']['reason']}", flush=True)
        else:
            for name, a in report["extraction"]["arms"].items():
                if a.get("guard_bug"):
                    print(f"      {name:20s} GUARD BUG: {a['uncited_kept']} uncited kept — NOT SCORED",
                          flush=True)
                    continue
                print(f"      {name:20s} P {a['precision']:.3f} R {a['recall']:.3f} "
                      f"F1 {a['f1']:.3f} refusals {a['refusals']}", flush=True)
        report["tinker"] = read_tinker_arms()
        for name, t in report["tinker"].items():
            print(f"      {name:20s} scored separately ({t.get('source')})", flush=True)

    write_json(report_path, report)

    from tables import render  # eval/tables.py
    table_md = render(report)
    (OUT / "tables.md").write_text(table_md, encoding="utf-8")
    print(f"wrote {report_path.relative_to(ROOT)} and data/eval/tables.md")

    failures = (report.get("extraction") or {}).get("guard_failures") or []
    if failures:
        print("GUARD FAILURE — a kept claim lost its citation, which is a core/ bug, not a metric:",
              flush=True)
        for f in failures:
            print(f"  !! {f}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
