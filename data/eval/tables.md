# Sổ Giỗ — evaluation tables

Every number below was produced by `python eval/run_eval.py` on 2026-10-28T07:45:02+07:00. Machine: 8 vCPU (AMD Ryzen 5 2500U with Radeon Vega Mobile Gfx), Linux-6.8.0-137-generic-x86_64-with-glibc2.39, Python 3.12.6. ASR: `Systran/faster-whisper-small (CTranslate2 int8, cpu)` and `vinai/PhoWhisper-small`, CPU only, models warm-loaded once before timing. Audio: so-gio-vn-fleurs-dev (CC-BY-4.0).

## 1. ASR accuracy per arm

| Arm | Model | Clips (n) | Audio (s) | Mean WER | Micro WER | Realtime factor | Median RTF |
|---|---|---|---|---|---|---|---|
| `whisper` | `small` | 18 | 204.4 | 0.221 | 0.197 | 0.52× | 0.56× |
| `phowhisper` | `vinai/PhoWhisper-small` | 18 | 204.4 | 0.140 | 0.119 | 0.21× | 0.20× |

*Syllable-level word error rate against all 18 published FLEURS vi_vn reference transcripts (204.4 s of audio). Mean WER is the unweighted mean of per-clip WER; micro WER pools all edits over all reference syllables. Realtime factor is total audio seconds ÷ total inference seconds on this CPU, higher is faster; median RTF is the per-clip median.*

## 2. Invention over inserted silence

| Arm | Silences with invention | Invented syllables | Inserted silence (s) | Invented words/min of silence | Syllables in full silent window |
|---|---|---|---|---|---|
| `whisper` | 0 | 0 | 15.0 | 0.0 | 0 |
| `phowhisper` | 3 | 96 | 15.0 | 384.0 | 96 |

*Syllables each arm emitted with timestamps inside all 3 digitally spliced silences (15.0 s total), where the reference transcript says nothing at all. A segment counts when its timestamp midpoint falls inside the splice. The last column counts inside the wider acoustically silent window (the splice plus the natural pause it was inserted into), so it is the loosest reading. Vietnamese is written in syllables, so syllables ≈ words.*

### 2b. Verbatim worst cases

| Arm | File | Detected at | Inserted silence | Syllables | Model output (verbatim) |
|---|---|---|---|---|---|
| `phowhisper` | `silence_02.wav` | 00:00–00:23 | 00:08–00:13 | 37 | “tôi bị lạc em gái và bạn của cô ấy và khi đang trên đường.có hai người khuyết tật ngồi xe lăn mọi người cứ trầm lên và xô đẩy họ a khman vơ say nói.” |
| `phowhisper` | `silence_01.wav` | 00:00–00:17 | 00:05–00:09 | 30 | “bản báo cáo mở đầu bằng lời biện hộ cho cuộc tranh luận mở và tạo lập sự đồng thuận của hoa kỳ về chính sách đối với trung đông.” |

*The two longest stretches each arm produced inside a silent splice, quoted exactly as the model returned them, with the segment timestamps the model itself assigned. Each row is a sentence that exists nowhere in the recording and nowhere in the reference transcript.*

## 3. Does the two-model diff catch it?

| Arm | Silences with invention | Flagged as invented | Catch rate |
|---|---|---|---|
| All arms | 3 | 0 | 0% |
| `whisper` | 0 | 0 | — |
| `phowhisper` | 3 | 0 | 0% |

*A “silence with invention” is an (arm, spliced silence) pair where that arm emitted at least one syllable inside the splice; it is “flagged” when `core.asr` produced an `invented` span for that arm overlapping the splice (both conditions: the segment is ≥50% inside detected non-speech and the other arm has no segment there). Catch rate is the share of such pairs the pipeline actually marks. No invented syllable leaked into `agreed_spans`, so nothing fabricated became quotable on these clips.*

## 3b. False positives on clean audio

| Clips (n) | Clips with an invented flag | False-positive rate (clips) | Invented spans | Invented syllables | Spans quoting text that is in the reference |
|---|---|---|---|---|---|
| 18 | 0 | 0% | 0 | 0 | 0 |

*The same invented-span logic run over all 18 clean clips (204.4 s), which contain no spliced silence — only the pauses the speakers actually left. These clips do have natural silence, so a flag here is not automatically wrong; the last column is the honest error count: spans whose text is literally present in the published reference transcript, i.e. words that were really spoken and were labelled invented. 0 of 0 flagged syllables are in the reference.*
