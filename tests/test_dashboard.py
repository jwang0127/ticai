import unittest
import json
from collections import Counter
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from src.generate_dashboard import (
    DIGIT_MODELS,
    build_analysis,
    digit_confidences,
    generate_composite_recommendations,
    generate_dlt,
    generate_digit_profile,
    generate_kl8,
    generate_pl5_from_pl3,
    generate_qxc,
    generate_ssq,
    next_draw,
)

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
        cls.dlt_rows = draws["dlt"]
        cls.pl5_rows = draws["pl5"]
        cls.fc3d_rows = draws["fc3d"]
        cls.qxc_rows = draws["qxc"]
        cls.ssq_rows = draws["ssq"]
        cls.kl8_rows = draws["kl8"]

    def test_each_game_family_has_its_own_model(self):
        self.assertNotEqual(DIGIT_MODELS["pl35"], DIGIT_MODELS["fc3d"])
        qxc, _ = generate_qxc(self.qxc_rows)
        ssq, _ = generate_ssq(self.ssq_rows, "2026083")
        self.assertEqual(len(qxc), 8)
        self.assertTrue(all(len(item["number"]) == 7 for item in qxc))
        self.assertEqual(len(ssq), 5)
        for item in ssq:
            self.assertEqual(len(item["red"]), 6)
            self.assertEqual(item["red"], sorted(set(item["red"])))
            self.assertEqual(len(item["blue"]), 1)

    def test_dlt_candidates_are_diversified(self):
        candidates, scores = generate_dlt(self.dlt_rows, "26082")
        self.assertEqual(len(candidates), 5)
        self.assertEqual(scores, sorted(scores, reverse=True))
        front_union = set().union(*(set(item["front"]) for item in candidates))
        back_union = set().union(*(set(item["back"]) for item in candidates))
        self.assertGreaterEqual(len(front_union), 10)
        self.assertGreaterEqual(len(back_union), 5)

    def test_position_analysis_is_explicit_for_direct_digit_games(self):
        expected = {
            "pl3": ["百位", "十位", "个位"],
            "pl5": ["万位", "千位", "百位", "十位", "个位"],
            "fc3d": ["百位", "十位", "个位"],
        }
        rows = {"pl3": self.rows, "pl5": self.pl5_rows, "fc3d": self.fc3d_rows}
        for game, labels in expected.items():
            analysis = build_analysis(game, rows[game])
            self.assertEqual([item["position"] for item in analysis["position_analysis"]], labels)
            self.assertTrue(all(len(item["hot_digits"]) == 3 for item in analysis["position_analysis"]))

    def test_kl8_pick_five_model_outputs_five_valid_groups(self):
        candidates, scores = generate_kl8(self.kl8_rows)
        self.assertEqual(len(candidates), 5)
        self.assertEqual(scores, sorted(scores, reverse=True))
        groups = []
        for item in candidates:
            numbers = item["numbers"]
            self.assertEqual(len(numbers), 5)
            self.assertEqual(numbers, sorted(set(numbers)))
            self.assertTrue(all(1 <= number <= 80 for number in numbers))
            groups.append(set(numbers))
        self.assertTrue(all(len(left & right) <= 3 for index, left in enumerate(groups) for right in groups[index + 1:]))
        for pick_count in range(6, 11):
            candidates, _ = generate_kl8(self.kl8_rows, pick_count)
            self.assertEqual(len(candidates), 5)
            self.assertTrue(all(len(item["numbers"]) == pick_count for item in candidates))
            self.assertTrue(all(item["numbers"] == sorted(set(item["numbers"])) for item in candidates))

    def test_fc3d_official_history(self):
        self.assertEqual(len(self.fc3d_rows), 100)
        self.assertEqual(self.fc3d_rows[0]["issue"], "2026192")

    def test_generated_output_has_only_direct_lists(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "docs/assets/data/dashboard.json").read_text(encoding="utf-8")
        games = json.loads(text)["games"]
        self.assertNotIn("play_types", games["pl3"])
        self.assertNotIn("play_types", games["fc3d"])
        for suffix in ("3", "6"):
            self.assertNotIn("组选" + suffix, text)

    def test_hot_and_cold_profiles_are_separate(self):
        hot = generate_digit_profile(self.fc3d_rows, 3, "hot", 5)
        cold = generate_digit_profile(self.fc3d_rows, 3, "cold", 5)
        self.assertEqual(len(hot), 5)
        self.assertEqual(len(cold), 5)
        self.assertTrue(all(item[2] > 0.25 for item in hot))
        self.assertTrue(all(item[2] < -0.25 for item in cold))
        self.assertFalse({item[0] for item in hot} & {item[0] for item in cold})

        hot_scores = digit_confidences(self.fc3d_rows, 3, [item[0] for item in hot])
        cold_scores = digit_confidences(self.fc3d_rows, 3, [item[0] for item in cold])
        self.assertGreater(min(hot_scores), max(cold_scores))

    def test_global_top_is_positionally_diverse(self):
        for rows in (self.rows, self.fc3d_rows):
            numbers = [item[0] for item in generate_digit_profile(rows, 3, "global", 5)]
            self.assertEqual(len(numbers), 5)
            for left_index, left in enumerate(numbers):
                for right in numbers[left_index + 1:]:
                    self.assertLessEqual(sum(a == b for a, b in zip(left, right)), 1)

    def test_generated_composite_lists_replace_hot_cold_zones(self):
        root = Path(__file__).resolve().parents[1]
        games = json.loads(
            (root / "docs/assets/data/dashboard.json").read_text(encoding="utf-8")
        )["games"]
        expected = {
            "pl3": (8, {"global": 5, "cold": 2, "hot": 1}),
            "pl5": (6, {"global": 4, "cold": 1, "hot": 1}),
            "fc3d": (8, {"global": 5, "cold": 2, "hot": 1}),
        }
        for key, (count, source_counts) in expected.items():
            game = games[key]
            candidates = game["top_candidates"]
            self.assertEqual(len(candidates), count)
            self.assertEqual(len({item["number"] for item in candidates}), count)
            self.assertNotIn("strategy_zones", game)
            self.assertEqual(Counter(item["source"] for item in candidates), source_counts)
            scores = [item["confidence"] for item in candidates]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_composite_generator_uses_requested_quotas(self):
        pl3, _ = generate_composite_recommendations("pl3", self.rows, self.rows)
        pl5, _ = generate_composite_recommendations("pl5", self.pl5_rows, self.rows)
        fc3d, _ = generate_composite_recommendations("fc3d", self.fc3d_rows, self.rows)
        self.assertEqual(Counter(item["source"] for item in pl3), {"global": 5, "cold": 2, "hot": 1})
        self.assertEqual(Counter(item["source"] for item in pl5), {"global": 4, "cold": 1, "hot": 1})
        self.assertEqual(Counter(item["source"] for item in fc3d), {"global": 5, "cold": 2, "hot": 1})

    def test_pl5_is_built_from_matching_pl3_prefixes(self):
        for profile in ("global", "hot", "cold"):
            pl3 = generate_digit_profile(self.rows, 3, profile, 5)
            pl5 = generate_pl5_from_pl3(self.rows, self.pl5_rows, profile, 5)
            self.assertEqual({item[0] for item in pl3}, {item[0][:3] for item in pl5})
            self.assertTrue(all(len(item[0]) == 5 for item in pl5))

    def test_copy_text_contains_only_name_and_number(self):
        root = Path(__file__).resolve().parents[1]
        games = json.loads(
            (root / "docs/assets/data/dashboard.json").read_text(encoding="utf-8")
        )["games"]

        def check(items, prefix):
            for item in items:
                self.assertTrue(item["copy_text"].startswith(f"{prefix} "))
                self.assertNotIn("｜", item["copy_text"])
                self.assertNotIn("期", item["copy_text"])
                self.assertNotIn("%", item["copy_text"])
                self.assertNotIn("开奖", item["copy_text"])

        for game in games.values():
            check(game["top_candidates"], game["name"])
            for play in game.get("play_types", {}).values():
                check(play["candidates"], f"{game['name']} {play['name']}")
            for zone in game.get("strategy_zones", {}).values():
                check(zone["candidates"], f"{game['name']} {zone['name']}")

    def test_homepage_has_all_game_navigation_buttons(self):
        root = Path(__file__).resolve().parents[1]
        homepage = (root / "docs/index.html").read_text(encoding="utf-8")
        for path, name in (("dlt", "超级大乐透"), ("pl3", "排列3"), ("pl5", "排列5"), ("fc3d", "福彩3D"), ("qxc", "体彩7星彩"), ("ssq", "福彩双色球"), ("kl8", "福彩快乐8")):
            self.assertIn(f'href="./{path}/"', homepage)
            self.assertIn(name, homepage)

    def test_daily_results_are_date_bound_and_copy_ready(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "docs/assets/data/dashboard.json").read_text(encoding="utf-8"))
        self.assertRegex(payload["daily_results_date"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertEqual(len(payload["daily_results"]), 7)
        for item in payload["daily_results"]:
            self.assertEqual(len(item["results"]), 3)
            self.assertEqual(item["result"], "；".join(scheme["result"] for scheme in item["results"]))
            self.assertEqual(item["copy_text"], f"{item['name']} {item['result']}")
            if item["game"] == "kl8":
                for scheme in item["results"]:
                    values = scheme["result"].split()
                    self.assertEqual(len(values), 10)
                    self.assertEqual(len(set(values)), 10)
                    self.assertTrue(all(len(value) == 2 and 0 <= int(value) <= 99 for value in values))
        homepage_script = (root / "docs/assets/js/app.js").read_text(encoding="utf-8")
        self.assertIn('id="daily-results-list"', (root / "docs/index.html").read_text(encoding="utf-8"))
        self.assertIn("data-daily-copy", homepage_script)

    def test_next_draw_time_is_rendered_on_home_and_detail_pages(self):
        root = Path(__file__).resolve().parents[1]
        homepage = (root / "docs/index.html").read_text(encoding="utf-8")
        homepage_script = (root / "docs/assets/js/app.js").read_text(encoding="utf-8")
        detail_script = (root / "docs/assets/js/detail.js").read_text(encoding="utf-8")
        self.assertIn('id="draw-board"', homepage)
        self.assertIn('class="draw-board-table"', homepage)
        for heading in ("玩法", "目标期号", "下一期开奖时间", "开奖安排"):
            self.assertIn(heading, homepage)
        self.assertIn("NEXT DRAW BOARD", homepage)
        self.assertIn('$("#draw-board").innerHTML', homepage_script)
        self.assertIn('cache: "no-store"', homepage_script)
        self.assertIn("game.target_issue", homepage_script)
        self.assertIn("const candidates = game.top_candidates || game.candidates;", homepage_script)
        self.assertIn("candidates.length", homepage_script)
        self.assertNotIn("strategyZonesHtml", detail_script)
        self.assertIn("game.top_candidates.length", detail_script)
        for script in (homepage_script, detail_script):
            self.assertIn("下一期开奖时间", script)
            self.assertIn("game.next_draw_display", script)
            self.assertIn("game.next_draw_at", script)


if __name__ == "__main__":
    unittest.main()
