"""Invention over silence, measured against ground truth rather than against a detector.

The three derived clips in `data/corpus/silence_manifest.json` carry the exact windows where digital silence
was spliced in (verified peak -91 dBFS), and the reference text for every one of those windows is empty. So
this is not an estimate: any syllable a model emits whose timestamp lands inside one of those windows is an
invention, by construction.

Run: python eval/measure_invention.py
Writes: data/eval/invention_groundtruth.json
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import asr, textnorm   # noqa: E402

MANIFEST = ROOT / "data" / "corpus" / "silence_manifest.json"
OUT = ROOT / "data" / "eval" / "invention_groundtruth.json"


def slice_audio(src: str, start: float, end: float, dst: str) -> bool:
    proc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ss", f"{start:.3f}",
                           "-t", f"{end - start:.3f}", "-ac", "1", "-ar", "16000", dst],
                          capture_output=True)
    return proc.returncode == 0


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    clips = manifest.get("clips") or manifest.get("items")
    results = []
    totals: dict[str, int] = {}
    for clip in clips:
        audio = str(ROOT / clip["output_audio"]) if not str(clip["output_audio"]).startswith("/") \
            else clip["output_audio"]
        for window in clip["silent_windows"]:
            start, end = window["start_seconds"], window["end_seconds"]
            with tempfile.TemporaryDirectory() as tmp:
                dst = f"{tmp}/w.wav"
                if not slice_audio(audio, start, end, dst):
                    continue
                pcm = asr.load_pcm(dst)
                arms = [asr.run_whisper(pcm, dst), asr.run_phowhisper(pcm, dst)]
            row = {"clip": clip["id"], "window": [start, end],
                   "window_seconds": round(end - start, 2), "arms": {}}
            print(f"\n=== {clip['id']}  silent window {start:.2f}–{end:.2f}s "
                  f"({end - start:.1f}s, reference says nothing)")
            for arm in arms:
                syl = textnorm.syllables(arm.text)
                totals[arm.arm] = totals.get(arm.arm, 0) + len(syl)
                row["arms"][arm.arm] = {"model": arm.model, "syllables": len(syl),
                                        "text": arm.text.strip(),
                                        "words_per_minute_of_silence":
                                            round(len(syl) / ((end - start) / 60.0), 1)}
                flag = "INVENTED" if syl else "silent (correct)"
                print(f"    {arm.arm:11s} {len(syl):3d} syllables  {flag}")
                if syl:
                    print(f"      \"{arm.text.strip()[:220]}\"")
            results.append(row)
    total_window = sum(r["window_seconds"] for r in results)
    summary = {
        "windows": len(results), "silence_seconds": round(total_window, 2),
        "invented_syllables": totals,
        "per_arm_wpm": {arm: round(n / (total_window / 60.0), 1) for arm, n in totals.items()},
        "arms_that_invented": sum(1 for v in totals.values() if v),
        "arms_measured": len(totals),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "clips": results}, ensure_ascii=False, indent=2))
    print(f"\nSUMMARY {json.dumps(summary)}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
