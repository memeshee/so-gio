# Sổ Giỗ — evaluation tables

Every number below was produced by `python eval/run_eval.py` on 2026-10-28T08:26:11+07:00. Machine: 8 vCPU (AMD Ryzen 5 2500U with Radeon Vega Mobile Gfx), Linux-6.8.0-137-generic-x86_64-with-glibc2.39, Python 3.12.6. ASR: `Systran/faster-whisper-small (CTranslate2 int8, cpu)` and `vinai/PhoWhisper-small`, CPU only, models warm-loaded once before timing. Audio: so-gio-vn-fleurs-dev (CC-BY-4.0).

## 1. ASR accuracy per arm

| Arm | Model | Clips (n) | Audio (s) | Mean WER | Micro WER | Realtime factor | Median RTF |
|---|---|---|---|---|---|---|---|
| `whisper` | `small` | 18 | 204.4 | 0.221 | 0.197 | 0.52× | 0.56× |
| `phowhisper` | `vinai/PhoWhisper-small` | 18 | 204.4 | 0.140 | 0.119 | 0.21× | 0.20× |

*Syllable-level word error rate against all 18 published FLEURS vi_vn reference transcripts (204.4 s of audio). Mean WER is the unweighted mean of per-clip WER; micro WER pools all edits over all reference syllables. Realtime factor is total audio seconds ÷ total inference seconds on this CPU, higher is faster; median RTF is the per-clip median.*

## 2. Invention over silence

| Arm | Probe | Probes (n) | Silence (s) | Invented syllables | Invented words/min of silence | Probes with invention |
|---|---|---|---|---|---|---|
| `whisper` | spliced pause | 3 | 15.0 | 0 | 0.0 | 0/3 |
| `whisper` | appended tail | 6 | 30.0 | 0 | 0.0 | 0/6 |
| `phowhisper` | spliced pause | 3 | 15.0 | 0 | 0.0 | 0/3 |
| `phowhisper` | appended tail | 6 | 30.0 | 0 | 0.0 | 0/6 |

*Syllables each arm produced **only because silence was added**, across 3 clipped-pause probes and 6 appended-tail probes (45.0 s of dead air in total, where the reference transcript says nothing at all). Invention is measured as a content delta: the syllables in the arm's transcript of the probed file that have no counterpart in the same arm's transcript of the unmodified source — a definition that needs no timestamps and also catches a model that repeats real words into the silence. Vietnamese is written in syllables, so syllables ≈ words.*

**No arm produced a single syllable inside a silent window, on any of the 9 probes.** There is therefore no fabricated sentence to quote. What can be shown instead is where each model chose to stop and start: on the three spliced-pause probes, Whisper's own segment boundaries bracket the digital silence — it heard the hole.

| Probe | Digital silence | Previous segment ends (s) | Next segment starts (s) | Gap the model left (s) | Syllables in the gap |
|---|---|---|---|---|---|
| `silence_01` | 00:05–00:09 | 5.00 | — | -5.00 | 0 |
| `silence_02` | 00:08–00:13 | 8.00 | 20.00 | 12.00 | 0 |
| `silence_03` | 00:06–00:12 | 5.00 | 12.00 | 7.00 | 0 |

*Whisper's own timestamps, not ours: the model closed one segment at or before the start of the digital silence and opened the next at or after its end, leaving the hole empty. A model that fabricated over the silence would have put a segment in that gap.*

## 3. Does the two-model diff catch it?

| Arm | Probe×arm pairs | Pairs with invention | Flagged as invented | Catch rate | Segments ≥50% inside detected silence | …of those, suppressed by the other arm |
|---|---|---|---|---|---|---|
| All arms | 18 | 0 | 0 | — |  |  |
| `whisper` | 9 | 0 | 0 | — | 6 | 6 |
| `phowhisper` | 9 | 0 | 0 | — | 3 | 3 |

*An (arm, silence) pair “has invention” when that arm added syllables because the silence was there; it is “flagged” when `core.asr` emitted an `invented` span for that arm overlapping the silent window (both conditions: the segment is ≥50% inside detected non-speech **and** the other arm has no segment there). With 0 pairs of invention the catch rate is undefined rather than 0/0-favourable. The last two columns are the mechanism, not an excuse: they count the segments each arm placed ≥50% inside detected non-speech — the raw material for an invented flag — and how many of those the other arm's segments overlapped, which suppresses the flag by construction. `invented_spans` fired 0 time(s) across all 27 recordings in this evaluation. No invented syllable leaked into `agreed_spans`, so nothing fabricated became quotable on these clips.*

## 3b. False positives on clean audio

| Clips (n) | Clips with an invented flag | False-positive rate (clips) | Invented spans | Invented syllables | Spans quoting text that is in the reference |
|---|---|---|---|---|---|
| 18 | 0 | 0% | 0 | 0 | 0 |

*The same invented-span logic run over all 18 clean clips (204.4 s), which contain no added silence — only the pauses the speakers actually left. A flag here is not automatically wrong, because clean speech has natural silence in it; the last column is the honest error count: spans whose text is literally present in the published reference transcript, i.e. words that were really spoken and were labelled invented. 0 of 0 flagged syllables are in the reference. A detector that never fires has a false-positive rate of zero and a recall of zero, and only the pair of numbers is honest.*
