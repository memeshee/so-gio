# Corpus licences

Everything shipped in `corpus/audio/` and indexed by `data/corpus/manifest.json` is
**CC BY 4.0**. Nothing non-redistributable is included; the sources we rejected are
listed below with the sentence that disqualified them.

Reproduce: `python corpus/fetch_corpus.py` (then `python corpus/make_silence_set.py`).

## Included

| Source | Config / split | Clips | Audio | Licence | Redistributable in a public repo? |
|---|---|---|---|---|---|
| Google FLEURS | `vi_vn` / `dev` | 18 | 204.42 s (6.54 MB) | CC BY 4.0 | **Yes**, with attribution |

### Google FLEURS — ACCEPTED

- URL: https://huggingface.co/datasets/google/fleurs (Vietnamese config `vi_vn`; files
  `data/vi_vn/dev.tsv` and `data/vi_vn/audio/dev.tar.gz`)
- Licence name and version: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
- Licence field on the page (dataset card YAML front matter, read 2026-10-28):

  > `license:`
  > `- cc-by-4.0`

- Licence sentence in the card body, section *Additional Information* (quoted verbatim,
  markdown link included):

  > "All datasets are licensed under the [Creative Commons license (CC-BY)](https://creativecommons.org/licenses/)."

- Canonical licence text: https://creativecommons.org/licenses/by/4.0/
- Redistribution: **allowed**, including commercially, provided attribution is given and
  modifications are indicated. We resample to 16 kHz mono and (for the silence set) insert
  digital silence, so the manifest records both the source and the edit.
- Caveat recorded honestly: the card states the licence only as the `cc-by-4.0` metadata
  tag plus the single sentence above; there is no per-language licence section, and the
  FLEURS paper (arXiv:2205.12446) does not restate a licence. We treat the card's metadata
  tag as authoritative because it is what Hugging Face gates the download on.
- Attribution string shipped with the clips:

  > Google FLEURS (Conneau et al., 2022), CC BY 4.0 — Vietnamese config `vi_vn`, split `dev`.
  > Audio resampled to 16 kHz mono; three clips have digital silence inserted (see
  > `data/corpus/silence_manifest.json`).

## Excluded

### VIVOS — EXCLUDED (non-commercial, share-alike)

- URL: https://ailab.hcmus.edu.vn/vivos (page read via the 2023 Wayback snapshot
  https://web.archive.org/web/2023/https://ailab.hcmus.edu.vn/vivos/ ; the live host
  refused the connection from this machine)
- Licence name and version: **CC BY-NC-SA 4.0**
- Quoted verbatim:

  > "The corpus should only be used for academic purposes."

  > "**License** Creative Commons Attribution NonCommercial ShareAlike v4.0 (CC BY-NC-SA 4.0)"

- Why excluded: the `NC` term forbids the commercial use a public product repo invites, and
  `SA` would force this repository's own code and docs under CC BY-SA. 15 h of clean
  read speech — the exact shape we wanted — but not shippable.

### Bud500 (`linhtran92/viet_bud500`) — EXCLUDED (non-commercial, gated)

- URL: https://huggingface.co/datasets/linhtran92/viet_bud500
- Licence name and version: **CC BY-NC-SA 4.0** (Hugging Face metadata field `license:
  cc-by-nc-sa-4.0`, repo flag `gated: auto`)
- Quoted verbatim (what an unauthenticated fetch returns):

  > "Access to dataset linhtran92/viet_bud500 is restricted. You must have access to it and be authenticated to access it. Please log in."

- Why excluded: `NC` again, plus the download needs an authenticated account, so
  `python corpus/fetch_corpus.py` could never be reproducible for a judge or CI.

### Mozilla Common Voice (Vietnamese) — EXCLUDED (no longer downloadable)

- URL: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
- Licence name and version: not restated on the page any more; Common Voice has historically
  been released as **CC0 1.0**. We cannot quote a licence sentence from the page today, so we
  do not rely on it.
- Quoted verbatim (page banner, read 2026-10-28):

  > "Effective October 2025, Mozilla Common Voice datasets are now exclusively available through Mozilla Data Collective. You can learn more about this change here."

  The repository now reports "Total file size: 2.88 kB" — the audio is gone.
- Why excluded: nothing to download at that URL, and the replacement needs an account. Even
  though CC0 would have been ideal, an unavailable corpus breaks the one-command promise.

### VietMed — EXCLUDED (non-commercial, and it re-crawls other corpora)

- URL: https://arxiv.org/html/2404.05659v3 (dataset paper)
- Licence name and version: no open licence; stated as **non-commercial research use only**
- Quoted verbatim, from the paper's ethics/limitations discussion:

  > "(1) Obviously we crawled the data and published only for non-commercial and research purposes."

- Why excluded: the `NC` restriction plus an unclear chain of title over crawled material.

### Wikimedia Commons "spoken Wikipedia" (Vietnamese) — EXCLUDED (does not exist)

- URL checked: https://commons.wikimedia.org (API search `incategory:"Spoken Wikipedia"`
  restricted to Vietnamese, namespace 6, read 2026-10-28)
- Result: **0 files**. There is no Vietnamese spoken-Wikipedia audio on Commons to license.
  Had any existed it would have been CC BY-SA 4.0, which would have been acceptable with
  attribution, but there is nothing to download.

### VLSP 2020 — NOT USED (no open mirror)

- Hugging Face repo `vlsp-2020-vlsp2020` returns `401 RepositoryNotFoundError`; VLSP corpora
  are distributed after registration with terms we could not read, so no licence sentence
  could be quoted and nothing was downloaded.

## Data handling notes

- Every clip is `wav / pcm_s16le / 16 kHz / mono`, produced with ffmpeg from the FLEURS
  release and verified with ffprobe (sample rate, channel count, and duration within 0.25 s
  of the official `num_samples` column).
- Reference transcripts are FLEURS `raw_transcription` verbatim (punctuation and casing
  preserved). A normalised copy (`reference_transcript_normalized`, the official lowercased,
  punctuation-free column) is stored alongside for WER scoring.
- FLEURS clips are **not** gain-normalised anywhere: measured peak level across the 18 clips
  spans **-35.6 dBFS to -11.5 dBFS** (median -24.2 dBFS). Two clips are quiet enough that a
  fixed -35 dB silence threshold marks the whole file as silence, which is why
  `make_silence_set.py` derives its threshold from each clip's own peak. Do not assume a
  single global threshold when building a VAD gate on this data.
- The official `gender` column reads `MALE` for every row of `vi_vn` dev (and test), so the
  corpus makes **no** claim of speaker-gender balance; 18 distinct speakers is what the
  selection rule guarantees and what the manifest records.
- Repo note: the root `.gitignore` currently ignores `corpus/audio/` and `*.wav`. The
  manifest, this file and both scripts are committable as-is; the wavs need an explicit
  `git add -f corpus/audio` (or a `.gitignore` exception) if the corpus is meant to live in
  the public repo.
