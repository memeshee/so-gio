"""Tests for the parts that decide whether a family is told the truth.

Run: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import guard, lunar, textnorm          # noqa: E402
from core.extract import Claim                   # noqa: E402

TODAY = (2026, 10, 28)

# A fixture, written by hand, used only to exercise the pipeline. Not demo data.
PIECES = [
    (0.0, 6.0, "Giỗ ông nội là ngày mười hai tháng bảy âm lịch."),
    (6.0, 12.0, "Mâm cúng phải có gà luộc và xôi gấc, nhớ mua hoa cúc."),
    (12.5, 20.0, "Con nhớ thắp nhang rồi đợi nhang tàn mới dọn đồ xuống."),
]
GROUNDED = " ".join(p[2] for p in PIECES)


class TestTextNorm(unittest.TestCase):
    def test_normalise_is_diacritic_preserving(self):
        self.assertEqual(textnorm.normalise("Giỗ Ông NỘI!"), "giỗ ông nội")
        # đ is a letter, not a decorated d: normalising must not turn "đợi" into "dợi"
        self.assertIn("đợi", textnorm.normalise("Đợi nhang tàn"))

    def test_contained_is_a_substring_test_after_normalisation(self):
        self.assertTrue(textnorm.contained("gà luộc và xôi gấc", GROUNDED))
        self.assertTrue(textnorm.contained("GÀ LUỘC", GROUNDED))
        self.assertFalse(textnorm.contained("bánh chưng", GROUNDED))

    def test_wer(self):
        self.assertAlmostEqual(textnorm.wer("một hai ba", "một hai ba"), 0.0)
        self.assertAlmostEqual(textnorm.wer("một hai ba", "một hai"), 1 / 3, places=6)


class TestCalendarAnchors(unittest.TestCase):
    """A handful of the published dates, as a guard against regressions in the engine."""

    CASES = [
        ((1, 1, 2026), (2026, 2, 17), "Tết 2026"),
        ((10, 3, 2026), (2026, 4, 26), "Giỗ Tổ Hùng Vương 2026"),
        ((15, 7, 2026), (2026, 8, 27), "Rằm tháng Bảy 2026"),
        ((15, 8, 2026), (2026, 9, 25), "Trung Thu 2026"),
        ((23, 12, 2025), (2026, 2, 10), "Ông Táo 2026"),
    ]

    def test_published_dates(self):
        for (ld, lm, ly), (y, m, d), label in self.CASES:
            with self.subTest(label=label):
                got = lunar.lunar_to_solar(ld, lm, ly)
                self.assertEqual(got, (d, m, y), label)

    def test_roundtrip(self):
        for solar in [(2026, 2, 17), (2026, 4, 26), (2033, 1, 31), (2025, 1, 29)]:
            y, m, d = solar
            back = lunar.solar_to_lunar(d, m, y)
            self.assertEqual(lunar.lunar_to_solar(back.day, back.month, back.year, back.leap), (d, m, y))

    def test_leap_months_are_known_values(self):
        self.assertEqual(lunar.leap_month_of(2020), 4)
        self.assertEqual(lunar.leap_month_of(2023), 2)
        self.assertEqual(lunar.leap_month_of(2025), 6)
        self.assertIsNone(lunar.leap_month_of(2026))

    def test_month_lengths_are_29_or_30(self):
        for year in range(2020, 2041):
            for month in lunar.months_from_month11(year):
                self.assertIn(month["end"] - month["start"] + 1, (29, 30))

    def test_next_occurrence_skips_today_and_returns_years(self):
        got = lunar.next_occurrence(7, 12, TODAY, years_ahead=3)
        self.assertEqual(len(got), 3)
        self.assertTrue(all(g > (2026, 10, 28) for g in got) or all(g[0] > 2026 for g in got))


class TestGuard(unittest.TestCase):
    def test_keeps_a_quoted_claim_and_cites_it(self):
        claims = [Claim(kind="offering", quote="gà luộc và xôi gấc, nhớ mua hoa cúc")]
        verdict = guard.review(claims, PIECES, TODAY)
        self.assertEqual(len(verdict.kept), 1)
        self.assertEqual(verdict.uncited_kept, 0)
        self.assertEqual(verdict.kept[0].t_start, 6.0)

    def test_refuses_text_that_was_never_said(self):
        claims = [Claim(kind="offering", quote="bánh chưng và giò lụa")]
        verdict = guard.review(claims, PIECES, TODAY)
        self.assertEqual(len(verdict.kept), 0)
        self.assertEqual(verdict.refused[0].reason, "not in the recording")

    def test_refuses_an_impossible_lunar_date(self):
        claims = [Claim(kind="gio_date", quote="mười hai tháng bảy âm lịch", lunar_day=31, lunar_month=7)]
        verdict = guard.review(claims, PIECES, TODAY)
        self.assertEqual(len(verdict.kept), 0)
        self.assertIn("impossible", verdict.refused[0].reason)

    def test_computes_the_next_solar_date_for_a_real_giỗ(self):
        claims = [Claim(kind="gio_date", quote="mười hai tháng bảy âm lịch", lunar_day=12, lunar_month=7)]
        verdict = guard.review(claims, PIECES, TODAY)
        self.assertEqual(len(verdict.kept), 1)
        self.assertRegex(str(verdict.kept[0].solar_next), r"^\d{4}-\d{2}-\d{2}$")

    def test_refuses_a_date_without_day_and_month(self):
        claims = [Claim(kind="gio_date", quote="mười hai tháng bảy âm lịch")]
        verdict = guard.review(claims, PIECES, TODAY)
        self.assertEqual(len(verdict.kept), 0)

    def test_deduplicates(self):
        c = lambda: Claim(kind="offering", quote="gà luộc")   # noqa: E731
        verdict = guard.review([c(), c()], PIECES, TODAY)
        self.assertEqual(len(verdict.kept), 1)
        self.assertEqual(len(verdict.refused), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
