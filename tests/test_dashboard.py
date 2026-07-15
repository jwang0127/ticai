import unittest
import json
from collections import Counter
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from src.generate_dashboard import next_draw, pl3_group_candidates

TZ = ZoneInfo("Asia/Shanghai")


class NextDrawTests(unittest.TestCase):
    def test_dlt_same_day_before_draw(self):
        now = datetime(2026, 7, 15, 11, 0, tzinfo=TZ)  # Wednesday
        self.assertEqual(next_draw(now, [0, 2, 5], time(21, 25)).isoformat(), "2026-07-15T21:25:00+08:00")

    def test_dlt_after_draw_moves_to_saturday(self):
        now = datetime(2026, 7, 15, 22, 0, tzinfo=TZ)
        self.assertEqual(next_draw(now, [0, 2, 5], time(21, 25)).isoformat(), "2026-07-18T21:25:00+08:00")

    def test_daily_game_after_draw_moves_one_day(self):
        now = datetime(2026, 7, 15, 22, 0, tzinfo=TZ)
        self.assertEqual(next_draw(now, list(range(7)), time(21, 25)).isoformat(), "2026-07-16T21:25:00+08:00")


class DetailPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.rows = json.loads((root / "data/processed/draws.json").read_text(encoding="utf-8"))["draws"]["pl3"]

    def test_group3_candidates_are_unique_and_valid(self):
        draw_at = datetime(2026, 7, 15, 21, 25, tzinfo=TZ)
        candidates = pl3_group_candidates(self.rows, "group3", "26186", draw_at)
        self.assertEqual(len({item["number"] for item in candidates}), 3)
        for item in candidates:
            self.assertEqual(sorted(Counter(item["number"]).values()), [1, 2])
            self.assertIn("排列3 组选3", item["copy_text"])

    def test_group6_candidates_are_unique_and_valid(self):
        draw_at = datetime(2026, 7, 15, 21, 25, tzinfo=TZ)
        candidates = pl3_group_candidates(self.rows, "group6", "26186", draw_at)
        self.assertEqual(len({item["number"] for item in candidates}), 3)
        self.assertTrue(all(len(set(item["number"])) == 3 for item in candidates))


if __name__ == "__main__":
    unittest.main()
