#!/usr/bin/env python3
"""Fetch the So Gio Vietnamese speech corpus (FLEURS vi_vn, CC-BY-4.0).

Run from the repo root::

    python corpus/fetch_corpus.py

Why streaming instead of a plain download: the FLEURS `vi_vn` dev audio tarball
is 214,500,592 bytes, which alone breaks our 200 MB download budget. The tarball
is a gzip stream, so we read it sequentially over HTTPS and stop as soon as the
selected members are in hand. Only the first few megabytes are ever transferred
(clips are selected in archive order, see `select_clips`).

Outputs (never outside `corpus/` and `data/corpus/`):
  corpus/audio/*.wav              16 kHz mono PCM s16le, one file per clip
  corpus/cache/                   10 kB of metadata (tsv copies, fetch log)
  data/corpus/manifest.json       transcript + licence + duration + path per clip

Verification: every written wav is probed with ffprobe (16000 Hz, 1 channel,
duration within 0.25 s of the official FLEURS `num_samples`) and every
transcript is asserted non-empty before it enters the manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:  # optional: the URL helper is a convenience, not a hard dependency
    from huggingface_hub import hf_hub_url
except ModuleNotFoundError:  # pragma: no cover - stdlib-only interpreter
    hf_hub_url = None

REPO_ID = "google/fleurs"
SPLIT = "dev"
LANG = "vi_vn"

SOURCE_NAME = "Google FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech), config vi_vn, split dev"
SOURCE_URL = f"https://huggingface.co/datasets/{REPO_ID}"
LICENCE = "CC-BY-4.0"
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"

SAMPLE_RATE = 16_000
MIN_SECONDS = 3.0
MAX_SECONDS = 30.0
DEFAULT_TARGET = 18
# Cap the transfer even if something goes wrong upstream.
DEFAULT_MAX_DOWNLOAD_MB = 150.0

BASE = Path(__file__).resolve().parent.parent
AUDIO_DIR = BASE / "corpus" / "audio"
CACHE_DIR = BASE / "corpus" / "cache"
DATA_DIR = BASE / "data" / "corpus"
MANIFEST = DATA_DIR / "manifest.json"


@dataclass
class Row:
    speaker_id: str
    filename: str
    transcript: str
    transcript_normalized: str
    num_samples: int
    gender: str

    @property
    def seconds(self) -> float:
        return self.num_samples / SAMPLE_RATE


class CountingReader:
    """Wraps a stream and counts how many (compressed) bytes we pulled."""

    def __init__(self, fp):
        self._fp = fp
        self.n = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._fp.read(size)
        self.n += len(chunk)
        return chunk


def hf_url(relpath: str) -> str:
    """Resolve a dataset file URL through huggingface_hub when it is installed.

    Falls back to the public `resolve/main` URL so the script still runs under an
    interpreter that has no third-party packages (the repo's own interpreter does
    have huggingface_hub; the sandbox default `python` may not).
    """
    try:
        assert hf_hub_url is not None
        return hf_hub_url(repo_id=REPO_ID, filename=relpath, repo_type="dataset")
    except Exception:  # pragma: no cover - only hit without huggingface_hub
        return f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{relpath}"


def open_stream(url: str):
    """Open a streaming HTTP response with stdlib only."""
    req = urllib.request.Request(url, headers={"User-Agent": "so-gio-corpus/1.0"})
    return urllib.request.urlopen(req, timeout=120)


def fetch_bytes(url: str) -> bytes:
    with open_stream(url) as resp:
        return resp.read()


def load_tsv(split: str = SPLIT) -> list[Row]:
    """Download and parse the official FLEURS transcript table."""
    url = hf_url(f"data/{LANG}/{split}.tsv")
    dest = CACHE_DIR / f"{LANG}-{split}.tsv"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not dest.exists():
        dest.write_bytes(fetch_bytes(url))

    rows: list[Row] = []
    with dest.open(encoding="utf-8", newline="") as fh:
        for rec in csv.reader(fh, delimiter="\t"):
            if len(rec) != 7:
                continue  # headerless table; malformed lines are dropped, not guessed
            speaker_id, filename, raw, norm, _phones, nsamp, gender = rec
            try:
                n = int(nsamp)
            except ValueError:
                continue
            rows.append(Row(speaker_id, filename, raw, norm, n, gender))
    if not rows:
        raise SystemExit(f"no usable rows parsed from {dest}")
    return rows


def select_clips(rows: list[Row], target: int) -> list[Row]:
    """Pick clips deterministically, in tarball order.

    FLEURS names the wavs after a numeric id and writes them into the tarball in
    lexicographic filename order, so taking the first `target` eligible rows
    keeps the read window inside the first ~20 members of a 214 MB archive.
    Duration spread is enforced with a soft preference for a few long clips
    (needed by make_silence_set.py), not with a random sample.
    """
    eligible = sorted(
        (r for r in rows if MIN_SECONDS <= r.seconds <= MAX_SECONDS and r.transcript.strip()),
        key=lambda r: r.filename,
    )
    if not eligible:
        raise SystemExit("no FLEURS clip falls inside 3-30 s")

    picked: list[Row] = []
    seen_speakers: set[str] = set()
    # First pass: distinct speakers, so the corpus is not one person reading 18 lines.
    for r in eligible:
        if r.speaker_id in seen_speakers:
            continue
        picked.append(r)
        seen_speakers.add(r.speaker_id)
        if len(picked) >= target and sum(1 for x in picked if x.seconds >= 10) >= 4:
            break
    return picked


def ffmpeg_to_wav(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    info = json.loads(out)
    stream = next(s for s in info["streams"] if s.get("codec_type") == "audio")
    return {
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "duration_seconds": round(float(info["format"]["duration"]), 3),
        "bytes": int(info["format"]["size"]),
    }


def stream_extract(rows: list[Row], staged: Path, max_bytes: int) -> tuple[dict[str, Path], int]:
    """Pull only the wanted members out of the remote .tar.gz and stop early."""
    import tarfile

    url = hf_url(f"data/{LANG}/audio/{SPLIT}.tar.gz")
    wanted = {r.filename: r for r in rows}
    got: dict[str, Path] = {}
    resp = open_stream(url)
    counter = CountingReader(resp)
    tf = None
    try:
        tf = tarfile.open(fileobj=counter, mode="r|gz")  # type: ignore[arg-type]  # stream mode: read() only
        for member in tf:
            if counter.n > max_bytes:
                print(f"  stopped: read cap {max_bytes/1e6:.0f} MB reached", file=sys.stderr)
                break
            name = Path(member.name).name
            if name not in wanted or not member.isfile():
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            target = staged / name
            target.write_bytes(extracted.read())
            got[name] = target
            if len(got) == len(wanted):
                break
    finally:
        try:
            if tf is not None:
                tf.close()
        except Exception:
            pass
        resp.close()
    return got, counter.n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET, help="clips to keep (default 18)")
    ap.add_argument("--max-download-mb", type=float, default=DEFAULT_MAX_DOWNLOAD_MB)
    ap.add_argument("--force", action="store_true", help="re-fetch even if corpus/audio is populated")
    args = ap.parse_args(argv)

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg and ffprobe must be on PATH")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    staged = CACHE_DIR / "staged"
    staged.mkdir(parents=True, exist_ok=True)

    if MANIFEST.exists() and not args.force:
        prior = json.loads(MANIFEST.read_text())
        wavs = sorted(AUDIO_DIR.glob("*.wav"))
        if len(wavs) >= len(prior.get("clips", [])) > 0:
            print(f"already fetched: {len(wavs)} wavs + manifest (use --force to redo)")
            return 0

    print(f"FLEURS {LANG}/{SPLIT} - {LICENCE}")
    rows = load_tsv()
    print(f"  transcript table: {len(rows)} rows")

    selected = select_clips(rows, args.target)
    total_ref = sum(r.seconds for r in selected)
    print(f"  selected {len(selected)} clips, {total_ref:.1f} s reference audio")

    got, nbytes = stream_extract(selected, staged, int(args.max_download_mb * 1e6))
    print(f"  transferred {nbytes/1e6:.2f} MB of the 214.5 MB tarball, {len(got)}/{len(selected)} members")

    missing = [r.filename for r in selected if r.filename not in got]
    if missing:
        raise SystemExit(f"missing members after stream: {missing}")

    clips = []
    for idx, row in enumerate(selected, start=1):
        clip_id = f"fleurs_vi_{idx:03d}"
        wav = AUDIO_DIR / f"{clip_id}.wav"
        ffmpeg_to_wav(got[row.filename], wav)
        meta = probe(wav)
        if meta["sample_rate"] != SAMPLE_RATE or meta["channels"] != 1:
            raise SystemExit(f"{wav}: not 16 kHz mono ({meta})")
        drift = abs(meta["duration_seconds"] - row.seconds)
        if drift >= 0.25:
            raise SystemExit(f"{wav}: ffprobe {meta['duration_seconds']}s vs FLEURS {row.seconds:.2f}s")
        text = row.transcript.strip()
        if not text:
            raise SystemExit(f"{clip_id}: empty reference transcript")
        clips.append({
            "id": clip_id,
            "source": SOURCE_NAME,
            "source_url": SOURCE_URL,
            "source_file": f"data/{LANG}/audio/{SPLIT}.tar.gz::{SPLIT}/{row.filename}",
            "source_filename": row.filename,
            "split": SPLIT,
            "speaker_id": row.speaker_id,
            "speaker_gender_reported": row.gender,
            "licence": LICENCE,
            "licence_url": LICENCE_URL,
            "attribution": "Google FLEURS (Conneau et al., 2022), CC-BY-4.0; Vietnamese config vi_vn",
            "duration_seconds": meta["duration_seconds"],
            "reference_transcript": text,
            "reference_transcript_normalized": row.transcript_normalized.strip(),
            "audio_path": str(wav.relative_to(BASE)),
            "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
        })

    manifest = {
        "corpus": "so-gio-vn-fleurs-dev",
        "purpose": "Vietnamese ASR cross-model disagreement + Whisper-over-silence probe set",
        "source": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "licence": LICENCE,
        "licence_url": LICENCE_URL,
        "audio_format": "wav, pcm_s16le, 16 kHz, mono",
        "selection_rule": (
            f"FLEURS {LANG}/{SPLIT} rows with {MIN_SECONDS}-{MAX_SECONDS} s of audio and a non-empty "
            "raw_transcription, taken in tarball (filename) order, one per speaker, "
            f"stopping at {args.target} clips or once 4 clips >= 10 s exist."
        ),
        "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bytes_transferred_from_hf": nbytes,
        "totals": {
            "clips": len(clips),
            "audio_seconds": round(sum(c["duration_seconds"] for c in clips), 3),
            "audio_megabytes": round(sum(Path(BASE / c["audio_path"]).stat().st_size for c in clips) / 1e6, 2),
            "distinct_speakers": len({c["speaker_id"] for c in clips}),
        },
        "clips": clips,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for tmp in staged.glob("*.wav"):
        tmp.unlink()
    shutil.rmtree(staged, ignore_errors=True)

    t = manifest["totals"]
    print(f"wrote {t['clips']} clips, {t['audio_seconds']:.1f} s, {t['audio_megabytes']:.2f} MB -> {MANIFEST.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
