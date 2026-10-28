# Sổ Giỗ — evaluation tables

Every number below was produced by `python eval/run_eval.py` on 2026-10-28T09:02:42+07:00. Machine: AMD Ryzen 5 2500U with Radeon Vega Mobile Gfx, 8 vCPU over 4 physical cores, Linux-6.8.0-137-generic-x86_64-with-glibc2.39, Python 3.12.6. ASR: `Systran/faster-whisper-small (CTranslate2 int8, cpu)` and `vinai/PhoWhisper-small`, CPU only, models warm-loaded once before timing. Audio: so-gio-vn-fleurs-dev (CC-BY-4.0).

## 1. ASR accuracy per arm

| Arm | Model | Clips (n) | Audio (s) | Mean WER | Micro WER | Realtime factor | Median RTF |
|---|---|---|---|---|---|---|---|
| `whisper` | `faster-whisper/small` | 18 | 204.4 | 0.221 | 0.197 | 0.52× | 0.56× |
| `phowhisper` | `vinai/PhoWhisper-small` | 18 | 204.4 | 0.140 | 0.119 | 0.21× | 0.20× |

*Syllable-level word error rate against all 18 published FLEURS vi_vn reference transcripts (204.4 s of audio). Mean WER is the unweighted mean of per-clip WER; micro WER pools all edits over all reference syllables. Realtime factor is total audio seconds ÷ total inference seconds on this CPU, higher is faster; median RTF is the per-clip median.*

## 2. Invention over silence

| Arm | Probe | Probes (n) | Silence (s) | Invented syllables | Invented words/min of silence | Probes with invention | Timestamps usable (probes) |
|---|---|---|---|---|---|---|---|
| `whisper` | isolated window, fed alone | 3 | 19.2 | 31 | 96.6 | 2/3 | 8/12 |
| `whisper` | spliced pause | 3 | 15.0 | 0 | 0.0 | 0/3 | 8/12 |
| `whisper` | appended tail | 6 | 30.0 | 0 | 0.0 | 0/6 | 8/12 |
| `phowhisper` | isolated window, fed alone | 3 | 19.2 | 42 | 130.9 | 3/3 | 3/12 |
| `phowhisper` | spliced pause | 3 | 15.0 | 0 | 0.0 | 0/3 | 3/12 |
| `phowhisper` | appended tail | 6 | 30.0 | 0 | 0.0 | 0/6 | 3/12 |

*Syllables each arm produced **only because silence was added or isolated**, across three conditions: 3 verified non-vocal windows fed to the model alone, 3 spliced mid-utterance pauses and 6 appended tails (64.2 s of dead air in total, where the reference transcript says nothing at all). Invention is measured as a content delta: the syllables in the arm's transcript of the probed audio that have no counterpart in the same arm's transcript of the unmodified source — a definition that needs no timestamps and also catches a model that repeats real words into the silence. The isolated condition is the one that reproduces the published Whisper-over-silence failure; the splice and tail conditions are what a carelessly recorded family tape looks like, and they are reported separately because the stock pipelines do not fail there. Vietnamese is written in syllables, so syllables ≈ words. The last column says on how many probes the arm's own timestamps are fine enough to place a syllable; counting by timestamp across all twelve probes would credit PhoWhisper with 138 syllables, and 96 of those sit on the six probes where the silence was bolted into real speech — every one of them the clip's own words, carried by a single segment spanning the whole 25 s window.*

### 2b. Verbatim worst cases

| Arm | File | Detected at | Silent window | Syllables | Model output (verbatim) |
|---|---|---|---|---|---|
| `whisper` | `iso_silence_01_0.wav` | 00:00–00:06 | 00:00–00:06 | 16 | “Cảm ơn các bạn đã chào tạm biệt và hẹn gặp lại trong video tiếp theo.” |
| `whisper` | `iso_silence_02_0.wav` | 00:00–00:06 | 00:00–00:06 | 15 | “Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn” |
| `phowhisper` | `iso_silence_03_0.wav` | 00:00–00:07 | 00:00–00:07 | 17 | “một trong những thay tòa soạn nhất là các tàu ngầm vũ trang sức và tính mạng.” |
| `phowhisper` | `iso_silence_02_0.wav` | 00:00–00:06 | 00:00–00:06 | 14 | “một trong những thay đổi đều được xem là một tài liệu sản phẩm.” |

*The longest text each arm produced inside a silent window, quoted exactly as the model returned it. The timestamp is the segment the model itself put the text in; for PhoWhisper that segment is the whole window, so it says where the audio began, not where a word was spoken.*

## 3. Does the two-model diff catch it?

| Arm | Probe×arm pairs | Pairs with invention | Flagged as invented | Catch rate | Segments ≥50% inside detected silence | …of those, suppressed by the other arm |
|---|---|---|---|---|---|---|
| All arms | 24 | 5 | 1 | 20% | — | — |
| `whisper` | 12 | 2 | 0 | 0% | 8 | 8 |
| `phowhisper` | 12 | 3 | 1 | 33% | 6 | 5 |

*An (arm, silence) pair “has invention” when that arm added syllables because the silence was there; it is “flagged” when `core.asr` emitted an `invented` span for that arm overlapping the silent window (both conditions: the segment is ≥50% inside detected non-speech **and** the other arm has no segment there). It caught 1 of 5. The last two columns are the mechanism, not an excuse: they count the segments each arm placed ≥50% inside detected non-speech — the raw material for an invented flag — and how many of those the other arm's segments overlapped, which suppresses the flag by construction. `invented_spans` fired 1 time(s) across all 30 recordings in this evaluation. No invented syllable leaked into `agreed_spans`, so nothing fabricated became quotable on these clips.*

## 3b. False positives on clean audio

| Clips (n) | Clips with an invented flag | False-positive rate (clips) | Invented spans | Invented syllables | Spans quoting text that is in the reference |
|---|---|---|---|---|---|
| 18 | 0 | 0% | 0 | 0 | 0 |

*The same invented-span logic run over all 18 clean clips (204.4 s), which contain no added silence — only the pauses the speakers actually left. A flag here is not automatically wrong, because clean speech has natural silence in it; the last column is the honest error count: spans whose text is literally present in the published reference transcript, i.e. words that were really spoken and were labelled invented. There were no flagged syllables to check. A detector that never fires has a false-positive rate of zero and a recall of zero, and only the pair of numbers is honest.*

## 4. Extraction arms

| Arm | Precision | Recall | F1 | TP / kept | Gold claims | Refusals | Uncited kept | Mean latency |
|---|---|---|---|---|---|---|---|---|
| `gemma3:1b-it-qat` | 0.000 | 0.000 | 0.000 | 0/8 | 21 | 0 | 0 | 36.26 s |
| `qwen2.5:3b-instruct` | 0.625 | 0.476 | 0.540 | 10/16 | 21 | 6 | 0 | 92.32 s |
| `regex` | 0.857 | 0.286 | 0.429 | 6/7 | 21 | 0 | 0 | 1.0 ms |
| `regex_all_dates` | 0.194 | 0.286 | 0.231 | 6/31 | 21 | 3 | 0 | 1.0 ms |
| `tinker_base` † | 0.802 | 0.667 | 0.728 | — | 21 | — | — | remote |
| `tinker_tuned` † | 0.958 | 0.875 | 0.915 | — | 21 | — | — | remote |

*8 held-out recordings, 21 gold claims. A claim counts only after `core.guard.review` keeps it — the quote must be verbatim in the transcript, carry a timestamp, and (for a date) resolve in the lunar calendar. A true positive requires the same kind, and then either the same lunar day/month (for a date, where the quote is only the citation) or a quote that contains the gold quote, or is contained by it, or shares ≥60% of its syllables (for a text claim, where the quote is the answer). Matching is greedy one-to-one, so each gold claim can be found once. Precision and recall are pooled over all recordings. Uncited kept claims must be 0 for every arm; if any arm broke that rule the row shows the count in bold and the arm is left unscored, because that is a bug in the guard, not a metric. † Scored separately, on the same gold set and with the same `eval/metrics.claims_match` rule, by the Tinker training process; a remote model on a different latency tier, so its latency is not comparable to the local arms and is not quoted.*

> **A model that quotes and still loses:** `gemma3:1b-it-qat` kept 8 claims and matched none. Every quote was verbatim in the transcript, so the guard passed it — the extractor picked the wrong thing to quote, which is the failure the guard cannot catch and the gold set can.
