import unittest
import json
from collections import Counter
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from src.generate_dashboard import generate_digit_profile, next_draw, three_digit_group_candidates

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
        draws = json.loads((root / "data/processed/draws.json").read_text(encoding="utf-8"))["draws"]
        cls.rows = draws["pl3"]
        cls.fc3d_rows = draws["fc3d"]

    def test_group3_candidates_are_unique_and_valid(self):
        draw_at = datetime(2026, 7, 15, 21, 25, tzinfo=TZ)
        candidates = three_digit_group_candidates("排列3", self.rows, "group3", "26186", draw_at)
        self.assertEqual(len({item["number"] for item in candidates}), 3)
        for item in candidates:
            self.assertEqual(sorted(Counter(item["number"]).values()), [1, 2])
            self.assertIn("排列3 组选3", item["copy_text"])

    def test_group6_candidates_are_unique_and_valid(self):
        draw_at = datetime(2026, 7, 15, 21, 25, tzinfo=TZ)
        candidates = three_digit_group_candidates("排列3", self.rows, "group6", "26186", draw_at)
        self.assertEqual(len({item["number"] for item in candidates}), 3)
        self.assertTrue(all(len(set(item["number"])) == 3 for item in candidates))

    def test_fc3d_official_history_and_groups(self):
        self.assertEqual(len(self.fc3d_rows), 100)
        self.assertEqual(self.fc3d_rows[0]["issue"], "2026186")
        draw_at = datetime(2026, 7, 15, 21, 15, tzinfo=TZ)
        candidates = three_digit_group_candidates("福彩3D", self.fc3d_rows, "group3", "2026186", draw_at)
        self.assertEqual(len(candidates), 3)
        self.assertTrue(all("福彩3D 组选3" in item["copy_text"] for item in candidates))

    def test_hot_and_cold_profiles_are_separate(self):
        hot = generate_digit_profile(self.fc3d_rows, 3, "hot", 3)
        cold = generate_digit_profile(self.fc3d_rows, 3, "cold", 3)
        self.assertEqual(len(hot), 3)
        self.assertEqual(len(cold), 3)
        self.assertTrue(all(item[2] > 0.25 for item in hot))
        self.assertTrue(all(item[2] < -0.25 for item in cold))
        self.assertFalse({item[0] for item in hot} & {item[0] for item in cold})


if __name__ == "__main__":
    unittest.main()
