# Bàn thờ

Sổ Giỗ — the family memorial book whose ears admit what they didn't hear.

Two local speech models transcribe the same Vietnamese recording. Only the words both of them produced may
become a claim. A deterministic lunar engine turns those claims into the next ten years of giỗ dates, and a
durable workflow holds each one until the day arrives. Everything the models were unsure about is shown,
marked, and never allowed to become a date.

**Live demo: https://gio.uconfidant.tech/**

---

## The problem, and who has it

A Vietnamese household's ritual knowledge — which ancestor's giỗ falls on which lunar date, what goes on the
tray, in what order the eldest lights the incense — lives in one elderly head. Recording her is the easy part.
Turning the recording into something a family can rely on for the next forty years is not, for three reasons:

1. **Speech models invent text over silence.** *Careless Whisper* (FAccT 2024) measured roughly 1% of Whisper
   segments as wholly invented, 38% of those harmful, with the rate rising as non-vocal audio gets longer. A
   tired 90-year-old is mostly pauses, so the failure lands exactly on the recording nobody can re-make.
2. **The date is not a language task.** A giỗ follows the lunar calendar, which moves against the Gregorian one
   every year, and gets a leap month in some years. A model that "writes" a date is guessing. A calendar
   computes it.
3. **The knowledge is held by one person and transferred once.** The default cloud transcription offering asks
   for a perpetual licence over uploaded content. For a family's recording of its own dead, that is the wrong
   trade.

Sổ Giỗ is for the person who will be handed the tray and does not know the order: the daughter flying home for
the first giỗ, the son abroad with the dates in a phone note, the household whose eldest is 91 and has started
repeating herself.

## What it does

```
recording ─┬─ whisper-small      ─┐
           └─ PhoWhisper-small   ─┴─ agree?  ──► quotable spans (with timestamps)
                                            └──► disputed spans (both readings, shown)
                                            └──► invented spans (struck out, never used)
                                                        │
                                     local model suggests claims (quote + kind + date)
                                                        │
                                     guard: cited? computable? verbatim?  ──► keep / refuse+reason
                                                        │
                            deterministic lunar engine ──► next 10 years of dates
                                                        │
                                 Temporal GioWorkflow ──► one durable timer per person, re-arming yearly
```

The design rule is one line: **the model suggests, code decides, and what code guarantees is the date.**

## Measured results

Every number below is produced by a script in this repo; re-run them and they should reproduce.

### Calendar correctness (`python -m core.oracle`)

| What | Oracle | Result |
|---|---|---|
| New-moon instants | USNO moon-phase API, 37 new moons across 2025–2027 | **37/37 within 2 minutes**, worst delta **84.2 s** |
| Lunar → solar dates | 17 published Vietnamese dates (Tết 2020–2036, Ông Táo, Giỗ Tổ Hùng Vương, Rằm tháng Bảy, Trung Thu 2026) | **17/17 exact**, all round-trip |
| Structural invariants | 41 lunar years, 2020–2060 | **0 violations** (every month 29–30 days, leap flag matches month count) |

The 2026 Rằm tháng Bảy case is the interesting one: the new moon falls at **00:36 on Aug 13 in UTC+7**, which
is 01:36 on Aug 13 in UTC+8 — a UTC+8 table can put a Vietnamese giỗ on the wrong day, which is why the engine
is anchored to UTC+7 explicitly.

### Speech, and what the models invented (`python eval/run_eval.py`, `python eval/measure_invention.py`)

| Arm | Model | n clips | Audio | Mean syllable WER | Realtime factor (8 CPU, no GPU) |
|---|---|---|---|---|---|
| `whisper` | `faster-whisper/small` int8 | 18 | 204 s | 0.221 | 0.52× |
| `phowhisper` | `vinai/PhoWhisper-small` | 18 | 204 s | **0.140** | 0.21× |

<!-- INSERT: invention-over-silence table from data/eval/invention_groundtruth.json -->

### Extraction, four local arms and two Tinker arms (`python train/tune.py report`)

| Arm | What it is | Precision | Recall | F1 | Uncited kept |
|---|---|---|---|---|---|
| `tinker_tuned` | `Qwen3.5-4B` + rank-16 LoRA, Tinker | **0.958** | **0.875** | **0.915** | **0** |
| `tinker_base` | same model, no adapter | 0.802 | 0.667 | 0.728 | 0 |
| `gemma3:1b` | `gemma3:1b-it-qat` via Ollama | see `data/eval/tables.md` | | | 0 |
| `qwen2.5:3b` | `qwen2.5:3b-instruct` via Ollama | see `data/eval/tables.md` | | | 0 |
| `regex` | hand-written lunar-date pattern | see `data/eval/tables.md` | | | 0 |

The fine-tune is doing one narrow job — copy a verbatim span into a structured claim, or emit nothing — and it
moves F1 from 0.728 to 0.915 on 8 passages it never saw. It also **costs 3× the latency** (7.8 s/item vs
2.6 s), which is published because a category claim built on "clear improvement" should say which axis got
worse. The training set is 16 authored passages; that is small, it is stated as small, and the loss collapsing
to ~0 by step 40 is reported as the overfitting risk it is.

### Durability (`bash scripts/prove_durability.sh`)

```
killed worker pid 2554542 with SIGKILL while the timer was pending;
a different process (pid 2554888) woke up and delivered the reminder
    {"activity_id": "1", "attempt": 3, "body_vi": "Hôm nay là 10/03 âm lịch — ngày giỗ của Cụ Nguyễn Thị Đào ..."}
PASS — the timer lived in the server, not in the killed process.
```

`attempt: 3` because the demo activity fails twice on purpose, to show the retry policy. Full transcript in
`scripts/prove_durability.out`.

## Architecture

| Component | File | Why this and not something else |
|---|---|---|
| Lunar calendar | `core/lunar.py` | 200 lines of Meeus arithmetic, no dependency. Must still compute a date in 2066; an SDK is a liability, arithmetic is not. |
| Calendar oracle | `core/oracle.py` | Grades the calendar against USNO and published Vietnamese dates. A calendar that grades its own homework is worth nothing. |
| Two ears | `core/asr.py` | Whisper `small` + PhoWhisper-small, same 16 kHz samples, syllable-level agreement. One model cannot report its own confabulation. |
| Guard | `core/guard.py` | Three refusals (verbatim? timestamped? computable?) applied before anything reaches a family. Refusals are counted, not hidden. |
| Extraction | `core/extract.py` | A local model may only *suggest*; `enable_thinking=False` on every arm so a reasoning model cannot hide its answer. |
| Durable timers | `workflows/gio.py` | One `GioWorkflow` per person, durable sleep, retrying activity, acknowledge signal, re-arms yearly. |
| Static demo | `site/build_site.py` | Generated from real run data, no network calls, no framework, `.nojekyll`, prints correctly. |
| Tracing | `core/telemetry.py` | One transaction per run, spans on every stage, and **no helper that accepts free text** — a trace of a private recording must not contain the recording. |
| Evaluation | `eval/` | Four measurements, one command, cached so a re-run is fast. |

**Deviation from the usual stack, stated:** the backend is Python, not Rust, because the core *is* the ASR
pipeline plus a Temporal Python worker; there is no server to serve, and the front end is a static page whose
whole point is that it works with the network off.

## Running it

```bash
git clone https://github.com/memeshee/so-gio && cd so-gio
python3.12 -m venv .venv && . .venv/bin/activate
pip install faster-whisper transformers torch temporalio sentry-sdk numpy

python data/extract/build_dataset.py          # verify the dataset's gold quotes
python -m core.oracle                          # calendar vs USNO + 17 published dates
python -m unittest discover -s tests           # 19 tests
python pipeline.py --self-test                 # guard on an authored fixture, no audio needed

# the whole thing on real audio (models download on first run, ~1.2 GB)
python corpus/fetch_corpus.py
python pipeline.py --audio corpus/audio/fleurs_vi_001.wav --out data/runs/demo

# the durable timer, with a real SIGKILL in the middle
brew install temporal || download from https://temporal.download   # then:
bash scripts/prove_durability.sh

# every number in the README
python eval/run_eval.py && python eval/measure_invention.py && python train/tune.py report
```

Requirements: ~1.5 GB disk for models, 6 GB RAM is enough, **no GPU required**. Tested on 8 vCPU AMD Ryzen 5
2500U, Linux 6.8, Python 3.12.6.

## Verified / not verified

| Claim | Status |
|---|---|
| Lunar dates match published Vietnamese dates | **verified**, 17/17, sources in `data/oracle/vn_anchors.json` |
| New moons match USNO | **verified**, 37/37 within 2 min |
| Two models disagree on real audio, and the disagreement is displayed | **verified** on the demo recording |
| Claims cannot reach a family uncited | **verified by construction** and asserted in the eval (`uncited_kept == 0` for every arm) |
| The durable timer survives `SIGKILL` | **verified**, transcript in `scripts/prove_durability.out` |
| Invention over silence | measured against spliced digital silence; see `data/eval/invention_groundtruth.json` — the number is what it is, including where the two-model method missed |
| The fine-tune beats the base model | **verified** on 8 held-out authored passages (F1 0.728 → 0.915). Small set, stated as small |
| It works on a real family's recording | **not yet** — the demo uses a CC-BY-4.0 corpus recording, deliberately, so every claim is checkable |

## Licence and provenance

Code MIT. Demo audio: **Google FLEURS `vi_vn`, CC-BY-4.0** — see `corpus/LICENSES.md` for the exact licence
text and the items excluded for licensing reasons. The extraction dataset (`data/extract/`) is authored by the
entrant and labelled as such everywhere it is used; it is not a recording and does not claim to be.

## Deployment

The demo is a static build pushed to the `gh-pages` branch and served from GitHub Pages, which cannot spin down.

- Canonical URL: **https://gio.uconfidant.tech/** (`.tech` domain claimed through Hacktoberfest, Let's Encrypt
  certificate, HTTPS enforced at the edge).
- Secondary URL: **https://memeshee.github.io/so-gio/** — GitHub 301s it to the canonical host.
- Wiring, in the order that avoids breaking the link: the `CNAME` record at the registrar
  (`CNAME gio → memeshee.github.io`) **first**, confirmed with `dig`, and only then the custom domain in the
  repository's Pages settings. Setting the custom domain before the record resolves makes GitHub 301 the
  `github.io` URL to a name that does not answer, which turns a working demo into a dead one — that happened
  during this build, and `python scripts/check_links.py` now re-statuses every advertised URL before each push.
