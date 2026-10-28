*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 4: Ghost in the Machine](https://dev.to/challenges/hacktoberfest-week4-2026-10-26)*

<!-- PERSONAL HOOK — 3 sentences, first person, from kiter. Exactly what is needed:
     (1) the relative who holds the giỗ dates in your family, by kinship term only ("bà ngoại", "mẹ");
     (2) one thing you personally do not know that she knows (a date, a dish, the order of the tray);
     (3) what happens to it when she cannot be asked any more.
     Do not invent this. Everything else in this post is measured or cited. -->

My grandmother is the only one who knows the dates.

## What I Built

**Sổ Giỗ** is a family memorial book that computes itself, and admits what it did not hear.

You record the one person in the family who knows the death anniversaries — in Vietnamese, on a phone, at a
kitchen table, for as long as they are willing to talk. Two different open speech models, both running on your
own laptop with the network cable out, transcribe that recording independently. Where they agree, the words
become quotable. Where they disagree, the page shows both readings and marks the span. Where one of them
produced words over a stretch the other heard as silence, the page strikes the sentence out and labels it
**invented** — and that sentence is never allowed to become a date.

From the words that survive, a small open model (also local) proposes claims: who the giỗ is for, on which
lunar day and month, what goes on the tray, in which order. Every claim must carry the words that produced it
and the second in the recording they were said at, or it is refused. Then a deterministic Vietnamese lunar
engine — 200 lines, no dependencies — turns the surviving lunar dates into the next ten years of giỗ, and a
durable workflow holds each one until the day arrives, a year at a time, surviving restarts.

One sentence: **the model is allowed to suggest, and only code decides — and the thing code guarantees is the
date, not the prose.**

{% embed https://github.com/memeshee/so-gio %}

## Demo

[memeshee.github.io/so-gio](https://memeshee.github.io/so-gio/)

What to do, in order:

1. Open the page. The top shows the recording it read — duration, both model names, and how fast each one ran
   on an 8-core CPU with no GPU.
2. Scroll to **what both models heard**. Each line has a timestamp chip. Those are the only lines allowed to
   become claims.
3. Look for the struck-through lines. Those are words one model produced over silence the other heard. There
   is at least one; the audio is public, so you can listen and decide for yourself.
4. Then the claims that survived, each with its chip, and the ones the guard refused, each with its reason.
5. Then ten years of dates, computed, with the same quote behind each one.

The demo audio is a **Google FLEURS `vi_vn` recording (CC-BY-4.0)**, not a private family tape — deliberately,
so that you can check every claim against the same public audio I used. The demo runs on a static page because
a page about a family's private recording should not need a server to be alive in order to be read.

## Code

MIT. Python 3.12, no GPU, one command per stage:

```bash
git clone https://github.com/memeshee/so-gio && cd so-gio
python data/extract/build_dataset.py        # 24 authored passages, 65 gold claims, every quote verified
python -m core.oracle                        # calendar vs USNO + 17 published Vietnamese dates
python pipeline.py --audio corpus/audio/fleurs_vi_001.wav   # the whole thing, end to end
python -m unittest discover -s tests         # 19 tests on the parts that decide what a family is told
python eval/run_eval.py                      # every number in this post
```

`data/models/` and the audio are gitignored; the models are downloaded on first run
(`faster-whisper/small` and `vinai/PhoWhisper-small`, ~1.2 GB together).

## How I Built It

Four layers, and the interesting decisions are all about which one is allowed to speak.

**1. Two ears, not one.** Arm A is OpenAI Whisper `small` through CTranslate2 (int8), the model most people
would actually run. Arm B is `vinai/PhoWhisper-small`, trained on Vietnamese (VLSP), a different architecture
family and a different corpus. Both get the *identical* 16 kHz float32 samples, decoded once by ffmpeg, so any
disagreement is attributable to the models and not to two different audio decoders.

Why two: *Careless Whisper* (FAccT 2024) measured roughly **1% of Whisper segments wholly invented, 38% of
those harmful, and the rate rising with the duration of non-vocal audio**. A tired 90-year-old is mostly
pauses, so the failure lands on the recording you cannot re-make. One model cannot tell you when it did this;
two models disagreeing can.

**2. The guard is the product.** A claim survives only if (a) its quote is a normalised substring of what both
arms produced, (b) that quote can be located at a timestamp, and (c) any lunar date in it is resolvable by the
calendar. Failing any of those, the claim is *refused with a reason* rather than corrected. The refusals are
printed on the page and counted in the post, because "we invented nothing" is only credible next to the list
of things we refused to invent. In the demo run, **zero kept claims were uncited**, by construction.

**3. The calendar is not a model.** `core/lunar.py` implements the definitions rather than copying a code
path: a month begins on the astronomical new moon in **UTC+7**; lunar month 11 is the one containing the
December solstice; a year has a leap month when 13 new moons fall between two month-11s, and the leap month is
the first with no major solar term (a solar longitude that is a multiple of 30°), repeating the previous
month's number. New moons come from Meeus chapter 49 with the full periodic series and a ΔT conversion to UT.

**4. Durability is Temporal, and the ritual repeats.** One `GioWorkflow` per person sleeps on a durable timer
until the next giỗ, sends the reminder through an activity with a retry policy, waits for an acknowledgement,
and re-arms for the following year. A giỗ is not a notification; it is an annual obligation, and the only
honest way to show software will hold it is to kill the process holding it.

```
=== 14. verdict ===
killed worker pid 2554542 with SIGKILL while the timer was pending;
a different process (pid 2554888) woke up and delivered the reminder:
    {"activity_id": "1", "attempt": 3, "solar_date": "2027-04-16", "worker_pid": 2554888,
     "body_vi": "Hôm nay là 10/03 âm lịch — ngày giỗ của Cụ Nguyễn Thị Đào (2027-04-16).
                 Con cháu chuẩn bị hương, hoa và mâm cơm cúng."}
PASS — the timer lived in the server, not in the killed process.
```

`attempt: 3` because the activity fails twice on purpose, to show the retry policy earning its place. The full
transcript, including the workflow input and every state transition, is in `scripts/prove_durability.out`.

## What I Measured

**The calendar is right.** Every giỗ date in this post comes from `core/lunar.py`, and I graded it against two
external oracles rather than against itself:

| What | Oracle | Result |
|---|---|---|
| New-moon instants | USNO moon-phase API, 37 new moons across 2025–2027 | **37/37 within 2 minutes**, worst delta **84.2 s** |
| Lunar → solar dates | 17 published Vietnamese dates (Tết 2020–2036, Ông Táo, Giỗ Tổ Hùng Vương, Rằm tháng Bảy, Trung Thu 2026) | **17/17 exact**, all round-trip |
| Structural invariants | 41 lunar years, 2020–2060 | **0 violations** |

The Rằm tháng Bảy case is the one that matters: in 2026 that new moon falls at **00:36 on 13 August in UTC+7** —
36 minutes past midnight. Anchor the calendar to UTC+8 instead and the same margin lands on the wrong day the
first time a new moon falls a few minutes earlier.

**The two models are not equally good, which is why there are two.** Syllable-level WER against the published
FLEURS reference transcripts, 18 clips, 204 s of audio, CPU only:

| Arm | Model | Mean WER | Realtime factor |
|---|---|---|---|
| `whisper` | `faster-whisper/small` int8 | 0.221 | 0.52× |
| `phowhisper` | `vinai/PhoWhisper-small` | **0.140** | 0.21× |

The Vietnamese-specific model is 36% more accurate and 2.5× slower. Neither number decides anything; what
decides is that they make *different* mistakes, and the difference is the product.

**What they invented in silence.** I spliced verified digital silence (peak **-91 dBFS**) into real recordings
at real speech pauses and asked each model to transcribe only the silent window. The reference for every one of
those windows is nothing at all. Model output, quoted exactly:

| Window | `whisper-small` said | `phowhisper-small` said |
|---|---|---|
| 5.6 s of silence | *"Cảm ơn các bạn đã chào tạm biệt và hẹn gặp lại trong video tiếp theo."* — 16 syllables | *"điều này từ thành công của tài xế hạng cao hơn."* — 11 syllables |
| 6.5 s of silence | *"Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn"* — 15 syllables | *"một trong những thay đổi đều được xem là một tài liệu sản phẩm."* — 14 syllables |
| 7.2 s of silence | **nothing** — correct | *"một trong những thay tòa soạn nhất là các tàu ngầm vũ trang sức và tính mạng."* — 17 syllables |

Nineteen and a quarter seconds of verified nothing; **73 syllables of invented Vietnamese**, at 96.6 invented
words per minute from one arm and 130.9 from the other. Both arms do it, on different windows, which is the whole
argument for running two: sample one silence and you would conclude that Whisper is safe and PhoWhisper is not —
the third window says the opposite of the first two.

The narrowing matters as much as the finding. I ran two more conditions: silences spliced **into** continuous
speech at 15 s total, and 5 s tails appended after the last word at 30 s total. Both arms produced **zero**
invented syllables on all of them. So this is not "Whisper invents whenever it hears quiet" — it invents when a
silent stretch arrives as a clean block with nothing around it, which is precisely what a long pause at the end
of a recording, a gap where the tape was stopped, or a trimmed file looks like. The measurement that would have
flattered the product (one big scary percentage) is not the one I am reporting.

Read the second row again. Given six and a half seconds of *nothing*, Whisper asked the listener to subscribe to
a named Vietnamese YouTube cooking channel. This is the failure the product exists to catch, and it is not
subtle: it is a commercial playing in the pause where a grandmother stopped talking.

**Can the two-model method catch it?** Here I have to publish a loss. The catch rule compares segment
timestamps: if one arm placed words where the other had no segment, flag it invented. Across 24 (arm × silence)
pairs, five had real invention and it flagged **one — 20%**. The mechanism is not mysterious: 14 segments sat
more than half inside detected non-speech, and **13 of those were suppressed** because the other arm had emitted
one long segment spanning the whole window, so "nothing was said here" was never true.
The fix was to stop inferring invention from timestamps and start *probing* it: take each detected non-speech
window, hand the models that window alone, and treat whatever comes back as invented, because there is no speech
in the input. That is `probe_silences()`, and it is now wired into the pipeline rather than sitting in an
evaluation script, which it was for one embarrassing afternoon. It cannot miss by construction, which is exactly
what the in-transcript rule cannot say. And no invented syllable ever reached the quotable set under either
method — `agreed_spans` only keeps syllables both arms produced — which is why the guard, not the detector, is
the load-bearing part.

**Extraction, six arms.** 8 held-out passages, 65 gold claims, every arm scored through the same guard:

| Arm | What it is | Precision | Recall | F1 | Uncited kept | s/item |
|---|---|---|---|---|---|---|
| `tinker_tuned` | `Qwen3.5-4B` + rank-16 LoRA, Tinker | **0.958** | **0.875** | **0.915** | **0** | 7.8 |
| `tinker_base` | same model, no adapter | 0.802 | 0.667 | 0.728 | 0 | 2.6 |
| `qwen2.5:3b-instruct` | local, Ollama, 3B | 0.625 | 0.476 | 0.540 | 0 | 92.3 |
| `regex` | hand-written lunar-date pattern | 0.857 | 0.286 | 0.429 | 0 | 0.001 |
| `regex_all_dates` | the degenerate "every date is a giỗ" arm | 0.194 | 0.286 | 0.231 | 0 | 0.001 |
| `gemma3:1b-it-qat` | local, Ollama, 1B | **0.000** | **0.000** | **0.000** | 0 | 36.3 |

Two rows deserve their own sentence. **A 1B model scores zero** — not badly, zero: it produces well-formed
claims that are not the ones in the text, so "just run a small model locally" is the assumption this table
kills. And a regex gets 0.857 precision at 0.286 recall, because a pattern is never wrong about a date it can
see and blind to every date phrased differently. What wins is neither; it is a model trained for this one narrow
job. Two things I would rather not print: the fine-tune costs **3× the latency**, and 16 training passages is
small enough that a loss collapsing to ~0.00002 by step 40 is memorisation as much as learning.

## What Went Wrong

Three failures worth publishing, because each one would have shipped silently.

**A reasoning model hid its own answer.** `Qwen3.5-4B` given an ordinary generation prompt emits
`Thinking Process:` and reasons in prose before any JSON. The strict parser found no claims, and the first
scoring run read as a total failure of the model arm. The fix is one argument — render *both* arms through the
chat template with `enable_thinking=False` — and it is not cosmetic: without it the tuned arm is scored against
a baseline that was never allowed to answer.

**Segment-level grounding threw away everything.** My first version of "only the words both models agree on"
required a whole ASR segment to be contained in the other arm's transcript. The two arms agree on ~84% of
syllables and disagree somewhere in almost every sentence, so the filter returned **zero quotable spans** on a
clean recording. The fix was to align the two transcripts syllable by syllable and keep the runs they agree on.
Same principle, one level finer — and the difference between a product and a demo that outputs nothing.

**The leap month was numbered like the month after it.** My first version gave `tháng 4 nhuận` the number 5, and
it still round-tripped its own tests — consistently wrong in both directions. Only the 17 *published* dates
caught it. That is the whole argument for an external oracle.

## Why Does Open Innovation Matter?

Because this artefact has to outlive every company in this post.

A giỗ calendar is a document about dead people that a household will consult for forty years. If it lives in a
subscription, it has a churn date. If the lunar conversion is a call to somebody's API, the family's ritual
calendar is now a dependency of a vendor's pricing decision. And if the only copy of grandmother's recording is
on a server that grants itself a perpetual licence over uploaded content — which is what the default cloud
transcription offering asks for — then the family has handed over the one thing that cannot be re-recorded.

Concretely, three things were only possible because the weights are open:

- **Diffing two models requires owning two models.** You cannot run a disagreement check against two rented
  endpoints you are not allowed to inspect, at a price a household will pay per recording.
- **The failure mode is the product feature.** I could point a camera at the invention because both models sit
  on the same disk as the audio and the whole pipeline runs with `nmcli networking off`. A closed pipeline
  gives you a transcript and no way to ask what it heard that wasn't there.
- **The engine is inspectable and forkable.** Vietnamese lunar practice differs by region and by family; some
  households hold the giỗ on the solar date, some use the leap month, some move it when it lands badly. A tool
  for this must be readable, editable, and runnable in 2066 by someone who has never heard of me. Two hundred
  lines of arithmetic and a text file travel better than an SDK.

And one honest concession: a hosted frontier model writes a warmer remembrance paragraph than a 4B model on a
six-year-old laptop. Where a bigger model genuinely wins in this build is prose, not dates — which is exactly
why the architecture puts the guarantee in the code and treats the writing as decoration the family can
replace.

## My Agent Session

This build was done with a coding agent (Hermes, running the model this IDE was talking to) driving the repo,
the tests, and the calendar validation, with the human making the calls that mattered: which product, which
category claims, what to refuse. DevRelay's transcript parser does not cover this harness, so rather than
embed a session it cannot read, here is what the agent actually did: wrote 19 tests and the oracle before the
features, ran the calendar against USNO until the worst new-moon delta was under two minutes, caught the
reasoning-model trap by reading a probe rather than the docs, and produced every table in this post with a
script that can be re-run.

## The Spirit of It

An ancestor who is fed is not a ghost that haunts.

That line is not mine. It is the closing thought of a Vietnamese-American physician writing about her
grandmother's death feast, and it is the best sentence about AI I have read this year:

> "Throughout the year, at the family altar, the spirit is offered fruit, rice, water and tea. It's not
> uncommon to see a can or two of Heineken. … Without the offerings, the spirit becomes a 'hungry ghost,' that
> haunts the family."
> — Christine Nguyen, MD

The ritual is a maintenance contract between the living and the dead, renewed annually, with food. Nothing in it
requires a machine to speak as anybody. The ghost in the machine, in this build, is confabulation — and the
whole product is a way of pointing at it rather than pretending it is a feature.

What landed, for me, was smaller and more uncomfortable. The demo recording is public, so you can do what I did:
listen to the six seconds of silence where one model asked you to subscribe to a cooking channel. Then imagine
that silence is not a corpus clip. Imagine it is the pause after your grandmother says she is tired and would
like to stop now. Every family has six seconds like that, somewhere in the middle of the recording that was
supposed to keep her.

<!-- PERSONAL MOMENT — optional 1-2 sentences here if you have a real one. Leave blank otherwise. -->

## Prize Categories

Five categories, each one carrying a real part of the build rather than a logo.

- **Best Use of Tinker** — `Qwen/Qwen3.5-4B` with a rank-16 LoRA trained on Tinker for one task: copy a
  verbatim span into a structured claim, or emit nothing. Scored on 8 held-out passages against the same base
  model, same prompt, same `enable_thinking=False` position, plus a regex baseline and a degenerate
  always-a-date arm. The comparison table is in *What I Measured*, including the run where it lost.
- **Best Use of Gemma** — `gemma3:1b-it-qat` through Ollama is the *on-device, no-fine-tune* arm: what a family
  gets with zero training and zero setup. It is the weakest arm on extraction, which is the point of printing
  it.
- **Best Use of Temporal** — one `GioWorkflow` per deceased relative, sleeping on a durable timer until the next
  giỗ, sending the reminder through an activity with a retry policy, waiting for an acknowledgement, then
  re-arming for the next year. The proof is a transcript, not a claim: the worker was killed with `SIGKILL`
  mid-timer and a different operating-system process woke up and delivered the reminder. Temporal's brief asks
  for software that "survives failures, retries flaky tool calls, and picks up where it left off" — this is that,
  at a one-year scale, which is the scale a ritual actually runs on.
- **Best Use of Sentry Agent Tracing** — one root transaction per pipeline run (`core/telemetry.py`), with a
  span for each ASR arm carrying latency, realtime factor and segment count; a span for the extraction call with
  its prompt and completion tokens; a span for the guard carrying kept and refused counts plus **one span per
  refusal** with the rule that fired; and a per-attempt span in the Temporal activity, so the delivery that
  finally succeeded shows up as three attempts with the retry backoff visible between them.

  The design decision is the part worth judging. Tracing a product whose entire premise is that a private
  recording never leaves the house means the trace must not contain the recording. So the module exposes no
  helper that accepts free text — every call site passes a number, a duration or an enum. `send_default_pii` is
  forced off, request bodies are never attached, and any string that could have come from audio goes through
  `redact()`, which keeps length, syllable count and a fingerprint and throws away the words. A contributor who
  wants to log a transcript has to edit that file, which is exactly the kind of change a reviewer sees. Verified
  by watching an envelope leave the process: `Sending envelope [envelope with 1 items (transaction)]
  project:4512199446757376 host:o1068235.ingest.us.sentry.io`.

  And the honest limit: tracing did not find my bug. The bug that mattered — an invention detector that caught
  **one of five** hallucinations — was found by measuring against ground truth, and it is in *What I Measured*
  with the numbers attached. Sentry's contribution here is the refusal ledger and the retry trail: the thing that
  makes "we invented nothing" and "it fired after the crash" auditable after the fact rather than asserted.

- **Best Use of a .tech Domain** — `gio.uconfidant.tech`, claimed through Hacktoberfest. The demo behind it is a
  static build, on purpose: a page about a private recording must not depend on anybody's free tier staying
  awake. Of the 36 live demo links among this series' winners, **five are already dead** — every one of them a
  suspended free instance. A prebuilt HTML page with a `.nojekyll` costs nothing to serve, loads in one request,
  prints correctly, and the link a judge clicks three weeks after the deadline is the same bytes I tested today.
  Live at [gio.uconfidant.tech](https://gio.uconfidant.tech/) behind a Let's Encrypt certificate issued this
  afternoon, with the `.github.io` hostname as the fallback GitHub redirects to it automatically.

---

*Every number above was produced by a script in the repository, and every published date is quoted with the
source I checked it against. If you find a wrong giỗ date in here, open an issue — the calendar is the one part
of this that has to be right.*
