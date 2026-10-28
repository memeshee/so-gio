# eval/ — the numbers this project publishes

One command, idempotent, CPU-only:

```bash
python eval/run_eval.py            # all four measurements
python eval/run_eval.py --stage asr      # measurements 1-3 (ASR, cached)
python eval/run_eval.py --stage extract  # measurement 4 only
python eval/run_eval.py --force          # ignore the ASR cache
```

Outputs: `data/eval/report.json` (every number plus per-clip detail) and `data/eval/tables.md`
(one pasteable markdown table per measurement, with n and a caption). ASR output is cached per
file under `data/eval/cache/`, keyed by the audio file's sha256 and the model names, so a re-run
costs nothing and a crash never loses completed work.

| File | Role |
|---|---|
| `run_eval.py` | all four measurements, caching, report + table emission |
| `tables.py` | `render(report) -> tables.md`; also runnable alone |
| `metrics.py` | **the** claim-matching rule (`claims_match`) shared with any arm scored elsewhere |
| `build_tail_probes.py` | warms the cache for the appended-silence probes (run_eval builds them too) |

Measurement 4 is not cached: it re-runs the two local models on every invocation (temperature 0,
fixed seed), which is why `--stage extract` exists. Measurements 1–3 come from the cache in seconds.

## What each measurement means

**1. ASR accuracy.** Syllable-level WER (`core.textnorm.wer`) of `faster-whisper small` int8 and
`vinai/PhoWhisper-small` against the published FLEURS vi_vn reference transcripts, plus each arm's
realtime factor (audio seconds ÷ inference seconds) on this box. Both models are warm-loaded once
before the timed loop, so per-clip latency is inference only.

**2. Invention over silence.** Three probe families where the reference transcript for a window is
empty by construction:

- `isolated` — the verified non-vocal window fed to each model **on its own** (3 windows, 19.25 s). No speech in the input at all, so anything returned is fabrication; this is the condition that reproduces the published Whisper-over-silence failure, and it is what the shipped `asr.probe_silences` does;
- `splice` — digital silence inserted into a real mid-utterance pause (`data/corpus/silence_manifest.json`, 3 clips, 4–6 s each);
- `tail` — 5 s of digital silence appended after the last word, every third clean clip (6 clips), which is what a family recording looks like when nobody stops the phone.

Invention is measured as a **content delta**: the syllables in the arm's transcript of the probed
file that have no counterpart in the same arm's transcript of the unmodified source. Timestamps are
*not* used as the primary test, because PhoWhisper returns one segment per 25 s window and therefore
cannot say where inside the window a syllable was spoken; a timestamp test on that arm reports the
clip's real words as inventions. The content delta is fair to both arms and also catches a model that
repeats real words into the silence. Timestamp-localised counts are still stored per event
(`syllables`, `timestamp_usable`, `median_segment_seconds`) for anyone who wants them.

**3. Does the two-model diff catch it?** For every (arm, probe) pair, whether `core.asr` produced an
`invented` span overlapping the silent window, plus the mechanism: how many segments each arm placed
≥50% inside detected non-speech, and how many of those the other arm's segments overlapped (which
suppresses the flag by construction in `asr.hear`). The false-positive side is measured on the 18
clean clips, including the honest error count: flagged spans whose text is literally in the published
reference transcript.

**4. Extraction arms.** `gemma3:1b-it-qat` and `qwen2.5:3b-instruct` via `core.extract.ollama_extract`,
`core.baselines.regex_baseline`, and the degenerate `regex_baseline(all_dates=True)`, every arm passed
through `core.guard.review` before scoring. Matching is `metrics.claims_match` — greedy one-to-one,
identical for every arm including any scored remotely (the Tinker arms in `data/eval/tinker_*.json`
are rendered, not recomputed). If any arm keeps an uncited claim the row is left unscored and
`run_eval.py` exits 2: that is a `core/guard.py` bug, not a metric.

## Known limits, stated where the numbers are

- 3 spliced-pause probes cannot resolve a phenomenon documented at ~1% of segments; the 6 tail probes
  are there to give the null result some weight. Report the n with the number.
- `detect_silence` uses a fixed −35 dB threshold. On one spliced clip the whole file measures as
  silence, which tells you the source is quiet, not that nobody spoke.
- `invented_spans` requires that the *other* arm has no overlapping segment. With a window-level
  second arm that condition is almost never true, so the flag is close to inert; measurement 3
  reports the count rather than assuming it works.
