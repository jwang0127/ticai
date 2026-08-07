from __future__ import annotations

import json
import unittest
from datetime import datetime
from pathlib import Path

from src.fetch_draws import TZ, expected_latest_draw_date, require_fresh_results
from src.optimize_models import review_feedback


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/games.json").read_text(encoding="utf-8"))


class FreshResultGateTests(unittest.TestCase):
    def test_expected_date_uses_beijing_schedule(self):
        now = datetime(2026, 7, 30, 22, 30, tzinfo=TZ)
        self.assertEqual(
            expected_latest_draw_date("pl3", CONFIG["games"]["pl3"], CONFIG["draw_time"], now),
            "2026-07-30",
        )
        self.assertEqual(
            expected_latest_draw_date("dlt", CONFIG["games"]["dlt"], CONFIG["draw_time"], now),
            "2026-07-29",
        )

    def test_gate_rejects_cached_history(self):
        now = datetime(2026, 7, 30, 22, 30, tzinfo=TZ)
        draws = {"pl3": [{"issue": "26199", "draw_date": "2026-07-29", "numbers": ["1", "2", "3"]}]}
        with self.assertRaises(SystemExit):
            require_fresh_results(["pl3"], draws, CONFIG, now)

    def test_gate_accepts_current_history(self):
        now = datetime(2026, 7, 30, 22, 30, tzinfo=TZ)
        draws = {"pl3": [{"issue": "26201", "draw_date": "2026-07-30", "numbers": ["1", "2", "3"]}]}
        require_fresh_results(["pl3"], draws, CONFIG, now)

    def test_review_feedback_marks_only_uncovered_positions_for_exploration(self):
        feedback = review_feedback("pl3", {
            "games": {"pl3": {"model_review": {
                "issue": "26207",
                "position_diagnostics": [
                    {"pool_hit": False}, {"pool_hit": True}, {"pool_hit": False}
                ],
            }}}
        })
        self.assertEqual(feedback["source_issue"], "26207")
        self.assertEqual(feedback["exploration_positions"], [0, 2])
        self.assertIn("不把上一期开奖号码直接升权", feedback["reason"])


if __name__ == "__main__":
    unittest.main()
