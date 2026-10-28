# Verified facts this build stands on

Every line here was checked before it was written. Commands that produced each line are in
`docs/probe-log.md`.

## The failure this product is built against
- *Careless Whisper* (FAccT 2024): ~1% of Whisper segments are wholly invented; 38% of those are
  harmful; hallucination **rises with non-vocal duration** — the model fabricates over silence.
- Vietnamese ASR floor: PhoWhisper-large reports **26.68 WER** on VLSP Task-2.
- CPU reality: `whisper.cpp medium.en` needs ~20 s per 30 s window on a 6-core laptop. This machine has
  8 cores / 6 GB / no GPU, so both arms must be `small`-class and int8.

## The oracle
- Vietnam publishes dual dates for its lunar holidays. **Giỗ Tổ Hùng Vương 2026 = 26 April 2026** (10/3 lunar),
  corroborated by timeanddate.com's Vietnam 2026 calendar, VOV reporting ("Sáng 26-4, tức 10-3 âm lịch") and the
  Hong Kong Observatory Gregorian–Lunar conversion table for 2026.
- The conversion uses astronomical new moons at **UTC+7**; a UTC+8 table (Chinese/HKO) disagrees on some months.

## Competition facts (measured, not assumed)
- 57 winning posts in this series: median **2,289 words**, median 8 H2s.
- 52/52 winners ship a `Demo` and a `Prize Categories` section; 51/52 argue "Why Does Open Innovation Matter?".
- 41/57 open on a person, not a product. 32/57 publish a named failure. 56/57 carry a verbatim quote.
- Of 36 live demo links among winners, **only 29 still returned 200**; five were suspended Render free
  instances. The demo therefore goes on GitHub Pages behind a custom domain.
