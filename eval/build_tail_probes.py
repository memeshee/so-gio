"""Build trailing-silence probes (5 s appended) for every clean clip and warm the ASR cache."""
import json, subprocess, sys
from pathlib import Path

ROOT = Path("/home/kiter/hackathon/so-gio")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
import run_eval

PROBE = ROOT / "data/eval/probe"
PROBE.mkdir(parents=True, exist_ok=True)
manifest = json.load(open(ROOT / "data/corpus/manifest.json"))
runner = run_eval.Runner()

for clip in manifest["clips"]:
    src = ROOT / clip["audio_path"]
    out = PROBE / f"tail_{clip['id']}.wav"
    if not out.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                        "-af", f"apad=pad_dur={run_eval.TAIL_SECONDS}",
                        "-c:a", "pcm_s16le", str(out)], check=True)
    rel = str(out.relative_to(ROOT))
    d = runner.hearing(rel)
    print(f"{clip['id']}: {d['audio_seconds']:.2f}s -> "
          f"{[(a['arm'], round(a['latency_s'], 1), len(a['segments'])) for a in d['arms']]}", flush=True)
print("done")
