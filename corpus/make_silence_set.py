#!/usr/bin/env python3
"""Build the "invention over silence" probe set from the FLEURS clips.

Run from the repo root, after `python corpus/fetch_corpus.py`::

    python corpus/make_silence_set.py

For three source clips we lengthen a real internal pause to 4.0 / 5.0 / 6.0 s and
write the result to corpus/audio/silence/. The reference transcript for the
inserted stretch is NOTHING (empty string): whatever a model emits there was not
said. This is the set Careless Whisper (FAccT 2024) predicts Whisper will
hallucinate on, because hallucination rises with non-vocal duration.

Why a pause and not a hard cut: FLEURS has utterance-level transcripts only, no
word timings, so we cannot compute a sentence timestamp. We therefore find pauses
acoustically with ffmpeg `silencedetect` (threshold = clip peak - 22 dB, minimum
0.20 s) and grow the longest internal one. The pre-insertion stretch is verified
to be below that threshold, so the claim we make is exactly the claim we can
support: nothing is spoken in the silent window before or after the edit.

Every derived file is re-probed: duration must equal source + inserted seconds,
the inserted window must measure as digital silence (<= -90 dBFS), and
`silencedetect` must still find >= the inserted duration at that offset.

Writes data/corpus/silence_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
AUDIO_DIR = BASE / "corpus" / "audio"
OUT_DIR = AUDIO_DIR / "silence"
MANIFEST_IN = BASE / "data" / "corpus" / "manifest.json"
MANIFEST_OUT = BASE / "data" / "corpus" / "silence_manifest.json"

SAMPLE_RATE = 16_000
PEAK_MARGIN_DB = 22.0      # pause threshold sits this far below the clip peak
MIN_PAUSE_SECONDS = 0.20
MIN_CLIP_SECONDS = 8.0
INSERTION_SECONDS = [4.0, 5.0, 6.0]   # 3-6 s, one per derived clip
N_CLIPS = 3


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def stderr_of(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stderr


def peak_dbfs(path: Path) -> float:
    err = stderr_of([
        "ffmpeg", "-nostdin", "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-",
    ])
    m = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB", err)
    if not m:
        raise SystemExit(f"could not measure peak of {path}")
    return float(m.group(1))


def detect_pauses(path: Path, threshold_db: float) -> list[dict]:
    err = stderr_of([
        "ffmpeg", "-nostdin", "-hide_banner", "-i", str(path),
        "-af", f"silencedetect=noise={threshold_db}dB:d={MIN_PAUSE_SECONDS}",
        "-f", "null", "-",
    ])
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?\d+(?:\.\d+)?)", err)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*(-?\d+(?:\.\d+)?)", err)]
    return [
        {"start_seconds": round(s, 3), "end_seconds": round(e, 3), "duration_seconds": round(e - s, 3)}
        for s, e in zip(starts, ends)
    ]


def probe_duration(path: Path) -> float:
    out = run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_entries", "format=duration", "-show_entries", "stream=sample_rate,channels,codec_name",
        str(path),
    ])
    info = json.loads(out)
    stream = info["streams"][0]
    assert int(stream["sample_rate"]) == SAMPLE_RATE and int(stream["channels"]) == 1
    return round(float(info["format"]["duration"]), 3)


def insert_silence(src: Path, dst: Path, offset: float, seconds: float) -> None:
    graph = (
        f"[0:a]atrim=start=0:end={offset:.3f},asetpts=PTS-STARTPTS[p1];"
        f"anullsrc=r={SAMPLE_RATE}:cl=mono:d={seconds:.3f}[sil];"
        f"[0:a]atrim=start={offset:.3f},asetpts=PTS-STARTPTS[p2];"
        f"[p1][sil][p2]concat=n=3:v=0:a=1[out]"
    )
    run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src), "-filter_complex", graph, "-map", "[out]",
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst),
    ])


def window_peak_dbfs(path: Path, start: float, end: float) -> float:
    err = stderr_of([
        "ffmpeg", "-nostdin", "-hide_banner", "-i", str(path),
        "-af", f"atrim=start={start:.3f}:end={end:.3f},volumedetect", "-f", "null", "-",
    ])
    m = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?|-inf) dB", err)
    if not m:
        raise SystemExit(f"could not measure window {start}-{end} of {path}")
    return float("-inf") if m.group(1) == "-inf" else float(m.group(1))


def longest_internal_pause(pauses: list[dict], duration: float) -> dict | None:
    internal = [
        p for p in pauses
        if p["start_seconds"] >= 0.30 and p["end_seconds"] <= duration - 0.30
    ]
    if not internal:
        return None
    return max(internal, key=lambda p: (p["duration_seconds"], -p["start_seconds"]))


def count_sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--clips", nargs="*", help="explicit source clip ids (default: 3 longest)")
    args = ap.parse_args(argv)

    if not MANIFEST_IN.exists():
        raise SystemExit("run corpus/fetch_corpus.py first")
    manifest = json.loads(MANIFEST_IN.read_text())
    clips = {c["id"]: c for c in manifest["clips"]}

    # Measure every candidate: peak, pauses, longest internal pause.
    analysed = []
    for clip in manifest["clips"]:
        path = BASE / clip["audio_path"]
        peak = peak_dbfs(path)
        thr = round(peak - PEAK_MARGIN_DB, 1)
        pauses = detect_pauses(path, thr)
        best = longest_internal_pause(pauses, clip["duration_seconds"])
        analysed.append({
            "clip": clip, "path": path, "peak_dbfs": peak, "threshold_db": thr,
            "pauses": pauses, "best_pause": best,
        })

    if args.clips:
        chosen = [a for cid in args.clips for a in analysed if a["clip"]["id"] == cid]
        if len(chosen) != len(args.clips):
            raise SystemExit(f"unknown clip id(s) in {args.clips}")
    else:
        usable = [
            a for a in analysed
            if a["best_pause"] and a["clip"]["duration_seconds"] >= MIN_CLIP_SECONDS
        ]
        usable.sort(key=lambda a: (
            -a["best_pause"]["duration_seconds"], -a["clip"]["duration_seconds"], a["clip"]["id"],
        ))
        chosen = usable[:N_CLIPS]

    if len(chosen) != N_CLIPS:
        raise SystemExit(f"need {N_CLIPS} clips with an internal pause, found {len(chosen)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    derived = []
    for idx, (item, insert) in enumerate(zip(chosen, INSERTION_SECONDS), start=1):
        clip = item["clip"]
        src = item["path"]
        pause = item["best_pause"]
        # Insert in the middle of the existing pause: the flanks stay non-vocal.
        offset = round((pause["start_seconds"] + pause["end_seconds"]) / 2.0, 3)
        out_id = f"silence_{idx:02d}"
        dst = OUT_DIR / f"{out_id}.wav"

        insert_silence(src, dst, offset, insert)

        src_dur = clip["duration_seconds"]
        out_dur = probe_duration(dst)
        if abs(out_dur - (src_dur + insert)) >= 0.05:
            raise SystemExit(f"{out_id}: duration {out_dur} != {src_dur} + {insert}")

        # The inserted stretch must be digital silence, not merely quiet.
        win_peak = window_peak_dbfs(dst, offset + 0.05, offset + insert - 0.05)
        if win_peak > -90.0:
            raise SystemExit(f"{out_id}: inserted window peaks at {win_peak} dBFS, not digital silence")

        # ffmpeg must still find a silence at least as long as what we inserted,
        # starting where the original pause started.
        out_pauses = detect_pauses(dst, item["threshold_db"])
        covering = [
            p for p in out_pauses
            if p["start_seconds"] <= offset + 0.10 and p["end_seconds"] >= offset + insert - 0.10
        ]
        if not covering:
            raise SystemExit(f"{out_id}: no silence re-detected around {offset}-{offset+insert}s")
        measured = max(covering, key=lambda p: p["duration_seconds"])

        total_silence = round(pause["duration_seconds"] + insert, 3)
        sentences = count_sentences(clip["reference_transcript"])
        derived.append({
            "id": out_id,
            "probe": "invention_over_silence",
            "source_clip_id": clip["id"],
            "source_audio": clip["audio_path"],
            "output_audio": str(dst.relative_to(BASE)),
            "licence": clip["licence"],
            "licence_url": clip["licence_url"],
            "attribution": clip["attribution"],
            "source_duration_seconds": src_dur,
            "duration_seconds": out_dur,
            "insertions": [{
                "offset_seconds": offset,
                "duration_seconds": insert,
                "kind": "digital silence splice inside a detected speech pause",
                "detected_pause": pause,
                "detected_pause_threshold_db": item["threshold_db"],
                "silent_window_seconds": total_silence,
                "silent_window_start_seconds": pause["start_seconds"],
                "silent_window_end_seconds": round(pause["end_seconds"] + insert, 3),
            }],
            "silent_windows": [{
                "start_seconds": pause["start_seconds"],
                "end_seconds": round(pause["end_seconds"] + insert, 3),
                "reference_text": "",
                "reference_text_note": "nothing is spoken in this window; any emitted token is an invention",
            }],
            "reference_transcript": clip["reference_transcript"],
            "reference_transcript_normalized": clip["reference_transcript_normalized"],
            "transcript_sentences": sentences,
            "boundary_note": (
                "FLEURS publishes utterance-level transcripts only, so no sentence-level timestamps exist. "
                "The splice sits inside the longest acoustically detected internal pause "
                f"({pause['start_seconds']}-{pause['end_seconds']} s), verified non-vocal before the edit; "
                f"the transcript has {sentences} sentence(s), so this pause is a phrase/sentence boundary "
                "but is not claimed to be a specific word boundary."
            ),
            "verification": {
                "inserted_window_peak_dbfs": win_peak,
                "remeasured_silence_seconds": measured["duration_seconds"],
                "remeasured_silence_start_seconds": measured["start_seconds"],
                "duration_delta_seconds": round(out_dur - src_dur, 3),
            },
            "expected_behavior": "no tokens inside silent_windows; transcript elsewhere unchanged",
        })

    silence_total = sum(d["insertions"][0]["duration_seconds"] for d in derived)
    manifest_out = {
        "set": "so-gio-invention-over-silence",
        "purpose": (
            "Measure Whisper's documented fabrication over non-vocal audio (Careless Whisper, FAccT 2024) "
            "on Vietnamese speech with published reference transcripts."
        ),
        "source_manifest": str(MANIFEST_IN.relative_to(BASE)),
        "insertion_rule": (
            f"longest internal pause per clip (silencedetect, peak-{PEAK_MARGIN_DB:.0f} dB, "
            f">= {MIN_PAUSE_SECONDS} s), grown by {INSERTION_SECONDS} s of digital silence at the pause midpoint"
        ),
        "clips": derived,
        "totals": {
            "clips": len(derived),
            "source_seconds": round(sum(d["source_duration_seconds"] for d in derived), 3),
            "output_seconds": round(sum(d["duration_seconds"] for d in derived), 3),
            "inserted_silence_seconds": round(silence_total, 3),
            "silent_window_seconds": round(sum(d["insertions"][0]["silent_window_seconds"] for d in derived), 3),
            "silent_fraction_of_output": round(
                sum(d["insertions"][0]["silent_window_seconds"] for d in derived)
                / sum(d["duration_seconds"] for d in derived), 4),
        },
    }
    MANIFEST_OUT.write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for d in derived:
        i = d["insertions"][0]
        print(f"{d['id']}: {d['source_clip_id']} {d['source_duration_seconds']:.2f}s -> {d['duration_seconds']:.2f}s "
              f"| +{i['duration_seconds']}s at {i['offset_seconds']}s "
              f"(silent window {i['silent_window_seconds']}s, peak {d['verification']['inserted_window_peak_dbfs']} dB)")
    t = manifest_out["totals"]
    print(f"wrote {t['clips']} derived clips, {t['inserted_silence_seconds']}s inserted "
          f"({t['silent_fraction_of_output']*100:.1f}% of {t['output_seconds']}s) -> {MANIFEST_OUT.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
