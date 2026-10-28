# Sổ Giỗ — static demo site

A one-column document, generated from the pipeline's own output. No network requests, no
JavaScript framework, no build tooling beyond Python's standard library.

## Regenerate

From the repository root:

```bash
/home/kiter/.pyenv/versions/3.12.6/bin/python site/build_site.py
# or pin a run:
/home/kiter/.pyenv/versions/3.12.6/bin/python site/build_site.py --run smoke-001
```

The script prints every file it writes with its byte size, then parses all four pages and fails
if a tag is unclosed, an id is duplicated, a local reference is missing, or a page points at
anything outside the folder.

## Inputs (read only)

| File | Used for |
|---|---|
| `data/runs/<id>/run.json` | arms, agreement rate, kept/refused claims, invented spans, calendar |
| `data/runs/<id>/reading_copy.md` | transcript lines, disputed pairs, which arm invented what |
| `data/oracle/report.json` | 37 new moons vs USNO, 17 published Vietnamese dates, structure check |
| `data/eval/report.json` + `tables.md` | optional fourth page; omitted cleanly when absent |

When more than one run exists, the richest one is chosen (dated claims, then refusals, then
invented spans), so a later demo run automatically replaces the smoke run.

## Output: `site/dist/`

- `index.html` — the recording, reading copy, kept and refused claims, ten years, calendar checks
- `reading-copy.html` — full transcript with disputed footnotes and struck inventions
- `calendar.html` — ten-year tables, lunar method, all 37 new moons and 17 anchors
- `eval.html` — WER, invention over spliced silence, and the detector's miss (only when eval data exists)
- `assets/site.css`, `assets/site.js`, `assets/fonts/*` — So Gio Serif, a subset of Liberation Serif (SIL OFL 1.1), self-hosted
- `favicon.svg`, `favicon-32.png`, `favicon-180.png` — the family seal, drawn by the script
- `.nojekyll` — for GitHub Pages

## Honesty rules this build follows

- Every number is read from `data/` or computed by `core/lunar.py`. Nothing is typed in by hand.
- An empty run stays empty: the ten-year section says so and offers a clearly labelled calendar
  example instead of inventing a family date.
- The demo audio is Google FLEURS `vi_vn`, CC BY 4.0 — stated on the page, not a private tape.
- The evaluation page publishes the detector's failure (0/3 invented spans flagged) rather than
  hiding it behind the demo run's zero.

## Deploy

Publish `site/dist/` as the site root (GitHub Pages branch folder, or upload the folder as an
artifact). All paths are relative, so it works from a project subpath such as
`https://<user>.github.io/so-gio/` without changes.
