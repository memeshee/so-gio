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

**https://memeshee.github.io/so-gio/**

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
python -m unittest discover -s tests         # 14 tests on the parts that decide what a family is told
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

Why two: Whisper is documented to invent fluent text over non-speech. *Careless Whisper* (FAccT 2024) measured
roughly **1% of segments wholly invented, 38% of those harmful, and the rate rising with the duration of
non-vocal audio**. A tired 90-year-old is mostly pauses. The failure therefore lands precisely on the recording
you cannot re-make. A single model cannot tell you when it did this; two models disagreeing can.

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

<!-- INSERT: durability transcript, verbatim from scripts/prove_durability.out -->

## What I Measured

<!-- INSERT: data/eval/tables.md — ASR WER table, invention-over-silence table, catch-rate table,
     extraction-arms table (gemma3:1b, qwen2.5:3b, regex, always-dates, Tinker base, Tinker tuned) -->

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

**The leap month was numbered like the month after it.** My first implementation assigned `tháng 4 nhuận` the
number 5. The calendar still round-tripped its own tests, because it was consistently wrong in both directions.
It was only caught by checking against *published* dates: 17 of them, from Wikipedia's Tết table and Vietnamese
government and press sources. That is the whole argument for an external oracle, and it is why every calendar
number in this post is quoted with the source a reader can open.

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
embed a session it cannot read, here is what the agent actually did: wrote 14 tests and the oracle before the
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

What landed, for me, was smaller and more uncomfortable: the demo recording is public, so I could listen to the
silence where one model wrote a sentence nobody said. It is three seconds long. Every family has three seconds
like that somewhere in the middle of the story.

<!-- PERSONAL MOMENT — optional 1-2 sentences here if you have a real one. Leave blank otherwise. -->

## Prize Categories

Six categories, each one carrying a real part of the build rather than a logo.

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
- **Best Use of Sentry Agent Tracing** — every stage is traced: each ASR arm's latency and realtime factor, the
  extraction call's prompt and completion tokens, the guard's keep/refuse decisions with reasons, and the
  workflow's delivery attempts. The debugging story this produced is in *What Went Wrong* — the trace is how the
  zero-quotable-spans bug was found, not a screenshot stapled on afterwards.
- **Best Use of a .tech Domain** — the demo is served on a `.tech` host pointed at a static build, so the page
  answers with the network off and cannot rot when a free dyno sleeps.
- **Best Use of Ultramarine** — the remembrance paragraph is the one place a stronger hosted model genuinely
  earns its keep, while extraction and the date stay local. The post states the run's actual cost, and states
  the limit plainly: prose is not the guarantee; the guarantee is local.

---

*Every number above was produced by a script in the repository, and every published date is quoted with the
source I checked it against. If you find a wrong giỗ date in here, open an issue — the calendar is the one part
of this that has to be right.*
