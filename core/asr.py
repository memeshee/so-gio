"""Two local speech-recognition models, one recording, and the disagreement is the output.

Why two models: Whisper is documented to invent fluent text over silence and non-vocal stretches
(Koene et al., *Careless Whisper*, FAccT 2024 — ~1% of segments wholly invented, 38% of those harmful,
and the rate rises with the duration of non-speech). A family recording of a tired 90-year-old is
mostly pauses, so the failure lands exactly where the record matters most. Running a second, differently
trained Vietnamese model over the same audio and diffing them at syllable level turns that failure into
something visible: where the two agree, quote it; where they disagree, show both; where a model produced
words over a stretch that the other heard and the energy detector calls silence, mark it invented.

Arm A: OpenAI Whisper `small`, multilingual, CTranslate2 int8 — the model most people would actually run.
Arm B: `vinai/PhoWhisper-small`, trained on Vietnamese (VLSP), a different architecture family and corpus.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

from core import textnorm

WHISPER_SIZE = "small"
PHO_MODEL = "vinai/PhoWhisper-small"


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class ArmResult:
    arm: str
    model: str
    text: str
    segments: list[Segment]
    latency_s: float
    audio_seconds: float

    @property
    def realtime_factor(self) -> float:
        return self.audio_seconds / self.latency_s if self.latency_s else 0.0


@dataclass
class Silence:
    start: float
    end: float


@dataclass
class Span:
    kind: str  # agree | disputed | invented | only_a | only_b
    text_a: str = ""
    text_b: str = ""
    start: float | None = None
    end: float | None = None


@dataclass
class Hearing:
    """The product's output for one recording: what was heard, what was not, and by whom."""
    audio: str
    audio_seconds: float
    arms: list[ArmResult] = field(default_factory=list)
    silences: list[Silence] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    agreement_rate: float = 0.0
    invented_spans: list[Span] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- audio energy

def detect_silence(path: str, noise_db: int = -35, min_len: float = 1.2) -> list[Silence]:
    """Silence intervals from ffmpeg's silencedetect. Independent of both models."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af",
           f"silencedetect=noise={noise_db}dB:d={min_len}", "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stderr
    starts = [float(m) for m in re.findall(r"silence_start:\s*([0-9.]+)", out)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([0-9.]+)", out)]
    sil = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else s + min_len
        if e > s:
            sil.append(Silence(s, e))
    return sil


def audio_duration(path: str) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True)
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


# --------------------------------------------------------------------------- the two arms

def run_whisper(path: str, model=None) -> ArmResult:
    from faster_whisper import WhisperModel

    if model is None:
        model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8",
                             download_root="data/models/fw")
    t0 = time.time()
    segments, _info = model.transcribe(path, language="vi", beam_size=5,
                                       vad_filter=False, condition_on_previous_text=True)
    segs = [Segment(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]
    dt = time.time() - t0
    return ArmResult("whisper", f"faster-whisper/{WHISPER_SIZE}", " ".join(s.text for s in segs),
                     segs, dt, audio_duration(path))


_PHO = {}


def run_phowhisper(path: str) -> ArmResult:
    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    if not _PHO:
        _PHO["proc"] = AutoProcessor.from_pretrained(PHO_MODEL, cache_dir="data/models/hf")
        _PHO["model"] = AutoModelForSpeechSeq2Seq.from_pretrained(PHO_MODEL, cache_dir="data/models/hf")
        _PHO["model"].eval()
    proc, model = _PHO["proc"], _PHO["model"]

    import librosa

    t0 = time.time()
    wav, _sr = librosa.load(path, sr=16000)
    # Long audio in one pass exhausts RAM on a 6 GB box; 25 s windows with 2 s overlap is plenty for a
    # transcript that only needs to agree or disagree with the other arm.
    win, hop = 25 * 16000, 23 * 16000
    segs: list[Segment] = []
    for start in range(0, max(len(wav), 1), hop):
        chunk = wav[start:start + win]
        if len(chunk) < 4000:
            break
        feats = proc(chunk, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(feats, max_new_tokens=225)
        text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if text:
            segs.append(Segment(start / 16000, min(len(wav), start + win) / 16000, text))
        if start + win >= len(wav):
            break
    dt = time.time() - t0
    return ArmResult("phowhisper", PHO_MODEL, " ".join(s.text for s in segs), segs, dt, audio_duration(path))


# --------------------------------------------------------------------------- fusion

def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def _in_silence(seg: Segment, silences: list[Silence], min_overlap: float = 0.5) -> bool:
    """True when most of a segment's duration sits inside detected non-speech."""
    dur = max(seg.end - seg.start, 1e-6)
    covered = 0.0
    for s in silences:
        lo, hi = max(seg.start, s.start), min(seg.end, s.end)
        if hi > lo:
            covered += hi - lo
    return covered / dur >= min_overlap


def hear(path: str, whisper_model=None) -> Hearing:
    """Run both arms, detect silence, and classify the result into quotable and unquotable spans."""
    h = Hearing(audio=path, audio_seconds=audio_duration(path))
    h.arms = [run_whisper(path, whisper_model), run_phowhisper(path)]
    h.silences = detect_silence(path)

    a, b = h.arms[0], h.arms[1]
    d = textnorm.diff_transcripts(a.text, b.text)
    h.agreement_rate = d.agreement_rate

    for words_a, words_b in d.disputed:
        h.spans.append(Span("disputed", " ".join(words_a), " ".join(words_b)))
    if d.only_a:
        h.spans.append(Span("only_a", " ".join(d.only_a), ""))
    if d.only_b:
        h.spans.append(Span("only_b", "", " ".join(d.only_b)))

    # Invented: words one arm produced over a stretch the other heard as silence AND the detector
    # calls non-speech. Both conditions, because one model's silence is not evidence of silence.
    for arm, other in ((a, b), (b, a)):
        for seg in arm.segments:
            if not _in_silence(seg, h.silences, 0.5):
                continue
            if any(_overlaps(seg.start, seg.end, o.start, o.end) for o in other.segments):
                continue
            span = Span("invented", seg.text if arm.arm == "whisper" else "",
                        seg.text if arm.arm == "phowhisper" else "", seg.start, seg.end)
            h.spans.append(span)
            h.invented_spans.append(span)
    return h
