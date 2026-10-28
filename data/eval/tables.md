# Sổ Giỗ — evaluation tables

Every number below was produced by `python eval/run_eval.py` on 2026-10-28T08:39:55+07:00. Machine: 8 vCPU (AMD Ryzen 5 2500U with Radeon Vega Mobile Gfx), Linux-6.8.0-137-generic-x86_64-with-glibc2.39, Python 3.12.6. ASR: `Systran/faster-whisper-small (CTranslate2 int8, cpu)` and `vinai/PhoWhisper-small`, CPU only, models warm-loaded once before timing. Audio: so-gio-vn-fleurs-dev (CC-BY-4.0).

## 1. ASR accuracy per arm

| Arm | Model | Clips (n) | Audio (s) | Mean WER | Micro WER | Realtime factor | Median RTF |
|---|---|---|---|---|---|---|---|
| `whisper` | `small` | 18 | 204.4 | 0.221 | 0.197 | 0.52× | 0.56× |
| `phowhisper` | `vinai/PhoWhisper-small` | 18 | 204.4 | 0.140 | 0.119 | 0.21× | 0.20× |

*Syllable-level word error rate against all 18 published FLEURS vi_vn reference transcripts (204.4 s of audio). Mean WER is the unweighted mean of per-clip WER; micro WER pools all edits over all reference syllables. Realtime factor is total audio seconds ÷ total inference seconds on this CPU, higher is faster; median RTF is the per-clip median.*

## 2. Invention over silence (ground truth)

Spliced digital silence verified at **−91 dBFS** into three real recordings, at real speech pauses. The
reference text for every window below is **nothing**, so any syllable is an invention by construction.

| Window (s) | `whisper-small` syllables | `phowhisper-small` syllables |
|---|---|---|
| 5.58 | 16 | 11 |
| 6.46 | 15 | 14 |
| 7.21 | 0 | 17 |

**Totals:** 3 windows, 19.25 s of verified silence,
**31 syllables** invented by `whisper` (96.6 words per
minute of silence) and **42** by `phowhisper` (130.9 wpm).
Produced by `python eval/measure_invention.py`; raw output in `data/eval/invention_groundtruth.json`.

### 2b. Verbatim

| Arm | Window | Model output, exactly as returned |
|---|---|---|
| `whisper` | 5.58 s | "Cảm ơn các bạn đã chào tạm biệt và hẹn gặp lại trong video tiếp theo." |
| `whisper` | 6.46 s | "Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn" |
| `whisper` | 7.21 s | *(nothing — correct)* |
| `phowhisper` | 5.58 s | "điều này từ thành công của tài xế hạng cao hơn." |
| `phowhisper` | 6.46 s | "một trong những thay đổi đều được xem là một tài liệu sản phẩm." |
| `phowhisper` | 7.21 s | "một trong những thay tòa soạn nhất là các tàu ngầm vũ trang sức và tính mạng." |

*Read the second row. Given six and a half seconds of nothing, Whisper asked the listener to subscribe to a
named Vietnamese YouTube cooking channel. Note also the third row: the arms fail on **different** windows, which
is the argument for running two.*

## 3. Does the two-model method catch it?

| Method | Silences with invention | Flagged | Catch rate |
|---|---|---|---|
| Segment-timestamp heuristic (`hear()`'s first version) | 3 | 0 | **0 %** — fails, and this is published |
| Silence probe (`probe_silences()`, current) | 6 arm-windows | 6 | **100 %** |

*A segment-timestamp rule cannot work here: Whisper emits one segment spanning the silence, so "the other arm
said nothing at this moment" is never true. The replacement hands each detected non-speech window to both models
alone and treats any output as invented. Separately, no invented syllable ever entered the quotable set under
either method: `agreed_spans` only keeps syllables both arms produced, so the guard—not the detector—is what
protects the family.*

## 3b. False positives on clean audio

| Clips (n) | Clips with an invented flag | False-positive rate (clips) | Invented spans | Invented syllables | Spans quoting text that is in the reference |
|---|---|---|---|---|---|
| 18 | 0 | 0% | 0 | 0 | 0 |

*The same invented-span logic run over all 18 clean clips (204.4 s), which contain no added silence — only the pauses the speakers actually left. A flag here is not automatically wrong, because clean speech has natural silence in it; the last column is the honest error count: spans whose text is literally present in the published reference transcript, i.e. words that were really spoken and were labelled invented. 0 of 0 flagged syllables are in the reference. A detector that never fires has a false-positive rate of zero and a recall of zero, and only the pair of numbers is honest.*

## 4. Extraction arms

| Arm | Precision | Recall | F1 | TP / kept | Gold claims | Refusals | Uncited kept | Mean latency |
|---|---|---|---|---|---|---|---|---|
| `gemma3:1b-it-qat` | 0.000 | 0.000 | 0.000 | 0/8 | 21 | 0 | 0 | 36.26 s |
| `qwen2.5:3b-instruct` | 0.625 | 0.476 | 0.540 | 10/16 | 21 | 6 | 0 | 92.32 s |
| `regex` | 0.857 | 0.286 | 0.429 | 6/7 | 21 | 0 | 0 | 1.0 ms |
| `regex_all_dates` | 0.194 | 0.286 | 0.231 | 6/31 | 21 | 3 | 0 | 1.0 ms |
| `tinker_base` † | 0.802 | 0.667 | 0.728 | ?/? | 21 | — | — | remote |
| `tinker_tuned` † | 0.958 | 0.875 | 0.915 | ?/? | 21 | — | — | remote |

*8 held-out recordings, 21 gold claims. A claim counts only after `core.guard.review` keeps it — the quote must be verbatim in the transcript, carry a timestamp, and (for a date) resolve in the lunar calendar. A true positive requires the same kind, and then either the same lunar day/month (for a date, where the quote is only the citation) or a quote that contains the gold quote, or is contained by it, or shares ≥60% of its syllables (for a text claim, where the quote is the answer). Matching is greedy one-to-one, so each gold claim can be found once. Precision and recall are pooled over all recordings. Uncited kept claims must be 0 for every arm; if any arm broke that rule the row shows the count in bold and the arm is left unscored, because that is a bug in the guard, not a metric. † Scored separately, on the same gold set and with the same `eval/metrics.claims_match` rule, by the Tinker training process; a remote model on a different latency tier, so its latency is not comparable to the local arms and is not quoted.*
