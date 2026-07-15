import unittest
from datetime import datetime, time
from zoneinfo import ZoneInfo

from src.generate_dashboard import next_draw

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


if __name__ == "__main__":
    unittest.main()

