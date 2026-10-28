"""External validation of the calendar. A calendar that grades its own homework is worth nothing.

Two independent oracles, both public, both checkable by a reader:

1. **USNO moon phases** (`aa.usno.navy.mil/api/moon/phases/year`). We compare our computed new-moon
   instants against the United States Naval Observatory's for every new moon in the cached years. This
   validates the astronomy on its own, before any calendar rule is applied.
2. **Published Vietnamese dates** (`data/oracle/vn_anchors.json`): Tết for 13 years from Wikipedia's
   table, plus Ông Táo, Giỗ Tổ Hùng Vương, Rằm tháng Bảy (Vu Lan) and Trung Thu for 2026 from Vietnamese
   government and press sources. This validates the rules — the UTC+7 time zone, the month-11 anchoring
   and the leap-month choice — which the astronomy alone cannot.

Run: `python -m core.oracle` prints the table and writes `data/oracle/report.json`.
"""
from __future__ import annotations

import datetime
import json
import math
import pathlib
import sys

from core import lunar

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "oracle" / "raw"
ANCHORS = ROOT / "data" / "oracle" / "vn_anchors.json"
REPORT = ROOT / "data" / "oracle" / "report.json"

UTC = datetime.timezone.utc


def _jd_to_utc(jd: float) -> datetime.datetime:
    return datetime.datetime.fromtimestamp((jd - 2440587.5) * 86400.0, UTC)


def check_new_moons(years: tuple[int, ...] = (2025, 2026, 2027)) -> dict:
    """Compare every new moon of the given years against USNO."""
    rows, worst = [], 0.0
    for year in years:
        path = RAW / f"usno-{year}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for ph in data.get("phasedata", []):
            if ph.get("phase") != "New Moon":
                continue
            hh, mm = (int(x) for x in ph["time"].split(":"))
            usno = datetime.datetime(ph["year"], ph["month"], ph["day"], hh, mm, tzinfo=UTC)
            target_jd = (usno.timestamp() / 86400.0) + 2440587.5
            k = lunar._k_near(int(math.floor(target_jd)))
            ours = lunar.new_moon_jd(k)
            delta_s = (ours - (usno.timestamp() / 86400.0 + 2440587.5)) * 86400.0
            worst = max(worst, abs(delta_s))
            rows.append({
                "date_ut": usno.strftime("%Y-%m-%d %H:%M"),
                "usno_ut": usno.strftime("%H:%M"),
                "ours_ut": _jd_to_utc(ours).strftime("%Y-%m-%d %H:%M"),
                "delta_seconds": round(delta_s, 1),
                "vn_local": _jd_to_utc(ours).astimezone(
                    datetime.timezone(datetime.timedelta(hours=7))).strftime("%Y-%m-%d %H:%M"),
            })
    within_2min = sum(1 for r in rows if abs(r["delta_seconds"]) <= 120)
    return {"rows": rows, "n": len(rows), "worst_abs_seconds": round(worst, 1),
            "within_2_minutes": within_2min}


def check_anchors() -> dict:
    anchors = json.loads(ANCHORS.read_text())["anchors"]
    rows, misses = [], []
    for a in anchors:
        ld, lm, ly = a["lunar"]
        ey, em, ed = a["solar"]
        try:
            d, m, y = lunar.lunar_to_solar(ld, lm, ly)
        except ValueError as exc:
            rows.append({**a, "computed": None, "ok": False, "error": str(exc)})
            misses.append(a["what"])
            continue
        ok = (y, m, d) == (ey, em, ed)
        back = lunar.solar_to_lunar(ed, em, ey)
        roundtrip = (back.day, back.month, back.year) == (ld, lm, ly)
        rows.append({"what": a["what"], "lunar": a["lunar"], "expected": a["solar"],
                     "computed": [y, m, d], "ok": ok, "roundtrip": roundtrip,
                     "source": a["source"]})
        if not ok:
            misses.append(a["what"])
    return {"rows": rows, "n": len(rows), "passed": sum(1 for r in rows if r.get("ok")),
            "misses": misses}


def check_structure(start: int = 2020, end: int = 2060) -> dict:
    """Invariants that must hold for every lunar year, independent of any oracle."""
    violations = []
    for y in range(start, end + 1):
        months = lunar.months_from_month11(y)
        if len(months) not in (12, 13):
            violations.append(f"{y}: {len(months)} months")
        if (len(months) == 13) != any(m["leap"] for m in months):
            violations.append(f"{y}: leap flag does not match month count")
        for m in months:
            length = m["end"] - m["start"] + 1
            if length not in (29, 30):
                violations.append(f"{y}: month of {length} days")
        leap_months = [m for m in months if m["leap"]]
        if len(leap_months) > 1:
            violations.append(f"{y}: {len(leap_months)} leap months")
    return {"years": end - start + 1, "violations": violations}


def main() -> int:
    print("== 1. new-moon instants vs USNO ==")
    nm = check_new_moons()
    for r in nm["rows"]:
        print(f"  USNO {r['date_ut']} | ours {r['ours_ut']} | "
              f"delta {r['delta_seconds']:+.0f}s | VN local {r['vn_local']}")
    print(f"  {nm['within_2_minutes']}/{nm['n']} within 2 minutes, worst {nm['worst_abs_seconds']}s")

    print("\n== 2. published Vietnamese dates ==")
    an = check_anchors()
    for r in an["rows"]:
        c = r.get("computed")
        print(f"  {'OK  ' if r.get('ok') else 'MISS'} {r['what'][:34]:34s} "
              f"{r['lunar'][0]:2d}/{r['lunar'][1]:2d}/{r['lunar'][2]} -> "
              f"{c[0]}-{c[1]:02d}-{c[2]:02d} (want {r['expected'][0]}-{r['expected'][1]:02d}-{r['expected'][2]:02d})")
    print(f"  {an['passed']}/{an['n']} exact" + (f", misses: {an['misses']}" if an["misses"] else ""))

    print("\n== 3. structural invariants ==")
    st = check_structure()
    print(f"  {st['years']} lunar years checked, violations: "
          f"{st['violations'] if st['violations'] else 'none'}")

    report = {"new_moons": nm, "anchors": an, "structure": st,
              "generated_utc": datetime.datetime.now(UTC).isoformat(timespec="seconds")}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {REPORT.relative_to(ROOT)}")
    ok = (not an["misses"]) and (not st["violations"]) and nm["within_2_minutes"] == nm["n"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
