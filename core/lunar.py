"""Vietnamese lunisolar calendar.

The rules implemented here are the *definitions*, not a transcription of someone else's code path,
because the definitions are what a reader can check:

- A lunar month begins on the day of the astronomical new moon, in **UTC+7**. That time zone is the
  whole reason a Vietnamese calendar is not a Chinese one: a new moon at 23:30 UTC+7 is 00:30 UTC+8
  the next day, so a UTC+8 table can place Tết a month away.
- Lunar month 11 is the month that contains the December solstice.
- A lunar year has a leap month (tháng nhuận) when 13 new moons fall between one month 11 and the next.
  The leap month is the first of those months that contains no major solar term (trung khí, a solar
  longitude that is a multiple of 30 degrees), and it repeats the number of the month before it.

No third-party dependency: 200 lines of arithmetic a household can keep running after every vendor in
this space has shut down. Validation against published dates lives in `core/oracle.py` and is not
optional.

Reference for the astronomical series: Jean Meeus, *Astronomical Algorithms*, ch. 25 (solar longitude)
and ch. 49 (new moon).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

PI = math.pi
TAU = 2.0 * PI
TZ_VN = 7.0
JD_EPOCH_NEW_MOON = 2451550.09766  # JDE of the k=0 new moon, 2000-01-06
SYNODIC = 29.530588853


# --------------------------------------------------------------------------- Julian day helpers

def jd_from_date(d: int, m: int, y: int) -> int:
    a = (14 - m) // 12
    y2 = y + 4800 - a
    m2 = m + 12 * a - 3
    return d + (153 * m2 + 2) // 5 + 365 * y2 + y2 // 4 - y2 // 100 + y2 // 400 - 32045


def date_from_jd(jd: int) -> tuple[int, int, int]:
    a = jd + 32044
    b = (4 * a + 3) // 146097
    c = a - (146097 * b) // 4
    d = (4 * c + 3) // 1461
    e = c - (1461 * d) // 4
    m = (5 * e + 2) // 153
    day = e - (153 * m + 2) // 5 + 1
    month = m + 3 - 12 * (m // 10)
    year = 100 * b + d - 4800 + m // 10
    return day, month, year


# --------------------------------------------------------------------------- astronomy

def _delta_t_seconds(year: float) -> float:
    """Espenak & Meeus polynomial for TD - UT, seconds. Good to a few seconds from 2005 to 2050."""
    t = year - 2000.0
    if 2005 <= year <= 2050:
        return 62.92 + 0.32217 * t + 0.005589 * t * t
    # Fallback: Meeus' long-term approximation, adequate outside the window above.
    return 63.86 + 0.3345 * t - 0.060374 * t * t + 0.0017275 * t ** 3


def new_moon_jd(k: int) -> float:
    """Julian day (UT) of the k-th new moon, k=0 at the new moon of 2000-01-06.

    Meeus, *Astronomical Algorithms* 2nd ed., ch. 49: the JDE series plus the periodic corrections,
    converted from dynamical to universal time with Delta-T. Verified against USNO moon-phase data
    in core/oracle.py — a reader can check the same numbers at aa.usno.navy.mil.
    """
    t = k / 1236.85
    t2, t3, t4 = t * t, t * t * t, t * t * t * t
    jde = (2451550.09766 + SYNODIC * k + 0.00015437 * t2
           - 0.000000150 * t3 + 0.00000000073 * t4)
    e = 1.0 - 0.002516 * t - 0.0000074 * t2
    m = math.radians(2.5534 + 29.10535670 * k - 0.0000014 * t2 - 0.00000011 * t3)
    mp = math.radians(201.5643 + 385.81693528 * k + 0.0107582 * t2
                      + 0.00001238 * t3 - 0.000000058 * t4)
    f = math.radians(160.7108 + 390.67050284 * k - 0.0016118 * t2
                     - 0.00000227 * t3 + 0.000000011 * t4)
    om = math.radians(124.7746 - 1.56375588 * k + 0.0020672 * t2 + 0.00000215 * t3)

    c = -0.40720 * math.sin(mp)
    c += 0.17241 * e * math.sin(m)
    c += 0.01608 * math.sin(2 * mp)
    c += 0.01039 * math.sin(2 * f)
    c += 0.00739 * e * math.sin(mp - m)
    c -= 0.00514 * e * math.sin(mp + m)
    c += 0.00208 * e * e * math.sin(2 * m)
    c -= 0.00111 * math.sin(mp - 2 * f)
    c -= 0.00057 * math.sin(mp + 2 * f)
    c += 0.00056 * e * math.sin(2 * mp + m)
    c -= 0.00042 * math.sin(3 * mp)
    c += 0.00042 * e * math.sin(m + 2 * f)
    c += 0.00038 * e * math.sin(m - 2 * f)
    c -= 0.00024 * e * math.sin(2 * mp - m)
    c -= 0.00017 * math.sin(om)
    c -= 0.00007 * math.sin(mp + 2 * m)
    c += 0.00004 * math.sin(2 * mp - 2 * f)
    c += 0.00004 * math.sin(3 * m)
    c += 0.00003 * math.sin(mp + m - 2 * f)
    c += 0.00003 * math.sin(2 * mp + 2 * f)
    c -= 0.00003 * math.sin(mp + m + 2 * f)
    c += 0.00003 * math.sin(mp - m + 2 * f)
    c -= 0.00002 * math.sin(mp - m - 2 * f)
    c -= 0.00002 * math.sin(3 * mp + m)
    c += 0.00002 * math.sin(4 * mp)

    jde += c
    year = 2000.0 + k / 12.3685
    return jde - _delta_t_seconds(year) / 86400.0


def new_moon_day(k: int, tz: float = TZ_VN) -> int:
    """Local day number on which the k-th new moon falls (the first day of that lunar month)."""
    return int(math.floor(new_moon_jd(k) + 0.5 + tz / 24.0))


def sun_longitude(day_number: int, tz: float = TZ_VN) -> float:
    """Apparent solar longitude, radians in [0, 2*pi), at local midnight of `day_number`."""
    jdn = day_number - 0.5 - tz / 24.0
    t = (jdn - 2451545.0) / 36525.0
    t2, t3 = t * t, t * t * t
    m = math.radians(357.52910 + 35999.05030 * t - 0.0001559 * t2 - 0.00000048 * t * t2)
    l0 = 280.46645 + 36000.76983 * t + 0.0003032 * t2
    c = (1.914600 - 0.004817 * t - 0.000014 * t2) * math.sin(m)
    c += (0.019993 - 0.000101 * t) * math.sin(2 * m) + 0.000290 * math.sin(3 * m)
    lon = math.radians(l0 + c)
    return lon - TAU * math.floor(lon / TAU)


def _k_near(jd: int) -> int:
    return int(round((jd - JD_EPOCH_NEW_MOON) / SYNODIC))


def month_start_for_day(day_number: int) -> tuple[int, int]:
    """(k, start_day) of the lunar month containing `day_number`."""
    k = _k_near(day_number)
    while new_moon_day(k) > day_number:
        k -= 1
    while new_moon_day(k + 1) <= day_number:
        k += 1
    return k, new_moon_day(k)


def month11_start(solar_year: int, tz: float = TZ_VN) -> int:
    """First day of the lunar month that contains the December solstice of `solar_year`."""
    # The solstice falls on 20-22 December; one-day resolution is enough because a new moon within
    # hours of the solstice is a genuine ambiguity, not a bug, and is reported by `ambiguities()`.
    best, best_err = None, 9.9
    for d in range(17, 25):
        jd = jd_from_date(d, 12, solar_year)
        err = abs((sun_longitude(jd) - 1.5 * PI + PI) % TAU - PI)
        if err < best_err:
            best, best_err = jd, err
    solstice = best
    k = _k_near(solstice)
    while new_moon_day(k, tz) > solstice:
        k -= 1
    while new_moon_day(k + 1, tz) <= solstice:
        k += 1
    return new_moon_day(k, tz)


def _has_major_solar_term(start: int, end: int, tz: float = TZ_VN) -> bool:
    """True if the sun crosses a multiple of 30 degrees between `start` and `end`."""
    a = math.degrees(sun_longitude(start, tz))
    b = math.degrees(sun_longitude(end, tz))
    if b < a:
        b += 360.0
    return int(a // 30) != int(b // 30)


# --------------------------------------------------------------------------- public API

@dataclass(frozen=True)
class LunarDate:
    day: int
    month: int
    year: int
    leap: bool = False

    @property
    def label_vi(self) -> str:
        return f"ngày {self.day} tháng {self.month}{' nhuận' if self.leap else ''} năm {self.year}"

    def __str__(self) -> str:
        return f"{self.day:02d}/{self.month:02d}{'N' if self.leap else ''}/{self.year}"


def months_from_month11(solar_year: int, tz: float = TZ_VN) -> list[dict]:
    """The 12 or 13 lunar months starting at month 11 of `solar_year`, numbered and leap-resolved.

    Numbering follows the tradition: the leap month repeats the number of the month before it
    (tháng 4 nhuận follows tháng 4), and numbering continues from there.
    """
    start = month11_start(solar_year, tz)
    nxt = month11_start(solar_year + 1, tz)
    k = _k_near(start)
    while new_moon_day(k, tz) > start:
        k -= 1
    while new_moon_day(k, tz) < start:
        k += 1
    starts = []
    while new_moon_day(k, tz) < nxt:
        starts.append({"k": k, "start": new_moon_day(k, tz), "end": new_moon_day(k + 1, tz) - 1})
        k += 1
    # A month has a trung khi when the sun's 30-degree sector advances across it.
    sectors = [int(math.degrees(sun_longitude(m["start"], tz)) // 30) % 12 for m in starts] + [None]
    for i, m in enumerate(starts):
        nxt_sec = sectors[i + 1]
        m["trung_khi"] = (nxt_sec is not None and nxt_sec != sectors[i]) or (nxt_sec is None)
    leap_i = None
    if len(starts) == 13:
        leap_i = next((i for i, m in enumerate(starts) if not m["trung_khi"] and i > 0), None)

    out: list[dict] = []
    month_no, year_no = 11, solar_year
    for i, m in enumerate(starts):
        if i == leap_i:
            prev_no = out[-1]["month"]
            out.append({**m, "month": prev_no, "year": out[-1]["year"], "leap": True})
            continue
        out.append({**m, "month": month_no, "year": year_no, "leap": False})
        month_no += 1
        if month_no > 12:
            month_no = 1
            year_no += 1
    return out


def solar_to_lunar(d: int, m: int, y: int, tz: float = TZ_VN) -> LunarDate:
    day_number = jd_from_date(d, m, y)
    for sy in (y - 1, y, y + 1):
        for entry in months_from_month11(sy, tz):
            if entry["start"] <= day_number <= entry["end"]:
                return LunarDate(day=day_number - entry["start"] + 1,
                                 month=entry["month"], year=entry["year"], leap=entry["leap"])
    raise ValueError(f"no lunar month found for {y}-{m:02d}-{d:02d}")


def lunar_to_solar(day: int, month: int, year: int, leap: bool = False, tz: float = TZ_VN) -> tuple[int, int, int]:
    for sy in (year - 2, year - 1, year):
        for entry in months_from_month11(sy, tz):
            if entry["month"] == month and entry["year"] == year and entry["leap"] == leap:
                return date_from_jd(entry["start"] + day - 1)
    raise ValueError(
        f"{day}/{month}{' leap' if leap else ''}/{year} not found "
        f"(leap month of {year} is {leap_month_of(year, tz)})"
    )


def leap_month_of(year: int, tz: float = TZ_VN) -> int | None:
    """Which lunar month is doubled in `year`, or None."""
    for entry in months_from_month11(year - 1, tz):
        if entry["leap"] and entry["year"] == year:
            return entry["month"]
    return None


def is_leap_year(year: int, tz: float = TZ_VN) -> bool:
    return leap_month_of(year, tz) is not None


def next_occurrence(month: int, day: int, after: tuple[int, int, int], years_ahead: int = 10,
                    leap: bool = False, tz: float = TZ_VN) -> list[tuple[int, int, int]]:
    """Solar dates of a repeating lunar date (a giỗ) for the next `years_ahead` lunar years.

    `after` is a solar date (year, month, day); occurrences on or before it are skipped, so a giỗ that
    has already passed this year starts at next year's date.
    """
    ay, am, ad = after
    start_jd = jd_from_date(ad, am, ay)
    out: list[tuple[int, int, int]] = []
    for year in range(ay - 1, ay + years_ahead + 2):
        try:
            d, m, y = lunar_to_solar(day, month, year, leap, tz)
        except ValueError:
            continue  # this lunar year has no such month, which a leap-month year can cause
        if jd_from_date(d, m, y) > start_jd:
            out.append((y, m, d))   # returned as (year, month, day): ISO order, like every other surface
        if len(out) >= years_ahead:
            break
    return out
