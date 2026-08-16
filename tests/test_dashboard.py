import math
import unittest
import json
from collections import Counter
from itertools import combinations
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.generate_dashboard import (
    DIGIT_MODELS,
    _digit_sequence,
    _sliding_top3_hits,
    ensure_position_pool_coverage,
    build_analysis,
    calibrate_digit_model,
    calibrate_set_model,
    digit_confidences,
    direct_structure_score,
    direct_number_metrics,
    direct_prediction_summary,
    generate_composite_recommendations,
    generate_positional_ensemble,
    generate_dlt,
    generate_digit_profile,
    generate_kl8,
    generate_kl8_play_types,
    generate_pl5_from_pl3,
    generate_qxc,
    generate_ssq,
    next_draw,
    order_statistic_top_mass,
    rolling_pool_backtest,
    rolling_region_backtest,
    ensure_repeat_shape_coverage,
)
from src.fetch_draws import today_games
from src.v3_production import kl8_cross_draw_scores

try:
    TZ = ZoneInfo("Asia/Shanghai")
except ZoneInfoNotFoundError:
    TZ = timezone(timedelta(hours=8))


class NextDrawTests(unittest.TestCase):
    def test_today_games_uses_beijing_weekday_schedule(self):
        config = {"games": {
            "daily": {"draw_weekdays": list(range(7))},
            "tuesday": {"draw_weekdays": [1]},
            "monday": {"draw_weekdays": [0]},
        }}
        now = datetime(2026, 7, 28, 22, 0, tzinfo=TZ)  # Tuesday
        self.assertEqual(today_games(config, now), ["daily", "tuesday"])

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
        self.assertNotEqual(DIGIT_MODELS["pl3"], DIGIT_MODELS["fc3d"])
        self.assertNotEqual(DIGIT_MODELS["pl5"], DIGIT_MODELS["fc3d"])
        qxc, _ = generate_qxc(self.qxc_rows)
        ssq, _ = generate_ssq(self.ssq_rows, "2026083")
        self.assertEqual(len(qxc), 5)
        self.assertTrue(all(len(item["number"]) == 7 for item in qxc))
        self.assertEqual(len(ssq), 5)
        for item in ssq:
            self.assertEqual(len(item["red"]), 6)
            self.assertEqual(item["red"], sorted(set(item["red"])))
            self.assertEqual(len(item["blue"]), 1)
        self.assertGreater(direct_structure_score([4, 7, 3]), direct_structure_score([4, 4, 4]))
        self.assertGreater(direct_structure_score([4, 7, 3]), direct_structure_score([0, 0, 0]))
        blue_values = {item["blue"][0] for item in ssq}
        self.assertGreaterEqual(len(blue_values), 4)
        self.assertGreaterEqual(len({(value - 1) // 4 for value in blue_values}), 4)

    def test_dlt_candidates_are_diversified(self):
        candidates, scores = generate_dlt(self.dlt_rows, "26082")
        self.assertEqual(len(candidates), 5)
        self.assertEqual(scores, sorted(scores, reverse=True))
        front_union = set().union(*(set(item["front"]) for item in candidates))
        back_union = set().union(*(set(item["back"]) for item in candidates))
        self.assertGreaterEqual(len(front_union), 15)
        self.assertGreaterEqual(len(back_union), 6)
        self.assertEqual(len({tuple(item["back"]) for item in candidates}), 5)
        self.assertGreaterEqual(len(front_union), 20)
        self.assertGreaterEqual(len(back_union), 8)

    def test_incremental_backtest_matches_naive_reference(self):
        rows = self.rows[:400]
        for decay in (8, 18, 45):
            for window in (100, 300):
                for position in range(3):
                    sequence = _digit_sequence(rows, position)
                    fast = _sliding_top3_hits(sequence, decay, window, 60)
                    limit = min(60, max(0, len(rows) - window))
                    hits = 0
                    for index in range(1, limit + 1):
                        counts = Counter()
                        for age, row in enumerate(rows[index:index + window]):
                            counts[int(row["numbers"][position])] += math.exp(-age / decay)
                        top3 = sorted(range(10), key=lambda digit: (-counts[digit], digit))[:3]
                        hits += int(int(rows[index - 1]["numbers"][position]) in top3)
                    self.assertEqual(fast, (hits, limit), (decay, window, position))

    def test_backtests_report_uncertainty_and_fair_baseline(self):
        digit_calibration = calibrate_digit_model("pl3", self.rows, 3)
        for cell in digit_calibration["backtest"].values():
            self.assertIn("se", cell)
            self.assertEqual(cell["baseline"], 0.3)
        set_calibration = calibrate_set_model("dlt", self.dlt_rows)
        for cell in set_calibration["backtest"].values():
            self.assertIn("se", cell)
            self.assertGreater(cell["baseline"], 0.3)
        # Sorted positions of a fair draw concentrate, so the honest baseline
        # for "first of five sorted numbers from 1-35 in a top-10 pool" is far
        # above the naive 10/35.
        self.assertGreater(order_statistic_top_mass(35, 5, 0, 10), 0.8)
        # Exhaustive check against the definition on a small board: the mass
        # of the 3 most likely values for the middle of three draws from 1-8.
        exact = Counter()
        for combo in combinations(range(1, 9), 3):
            exact[sorted(combo)[1]] += 1
        expected = sum(sorted(exact.values(), reverse=True)[:3]) / math.comb(8, 3)
        self.assertAlmostEqual(order_statistic_top_mass(8, 3, 1, 3), expected, places=12)
        region = rolling_region_backtest("ssq", self.ssq_rows, 300, "red")
        self.assertAlmostEqual(region["baseline"], round(18 / 33, 4))
        pool = rolling_pool_backtest("kl8", self.kl8_rows, 300)
        self.assertAlmostEqual(pool["baseline"], 0.5)

    def test_positional_ensemble_covers_full_backtested_pools(self):
        rows = {"pl3": self.rows, "pl5": self.pl5_rows, "fc3d": self.fc3d_rows}
        for game, game_rows in rows.items():
            candidates, scores = generate_positional_ensemble(game, game_rows)
            self.assertEqual(scores[:3], sorted(scores[:3], reverse=True))
            self.assertEqual(scores[3:], sorted(scores[3:], reverse=True))
            digits = len(candidates[0]["number"])
            for position in range(digits):
                distinct = {item["number"][position] for item in candidates}
                self.assertGreaterEqual(len(distinct), 3, (game, position, distinct))

    def test_pl3_uses_four_digit_pool_after_rolling_coverage_review(self):
        candidates, _ = generate_positional_ensemble("pl3", self.rows)
        for position in range(3):
            distinct = {item["number"][position] for item in candidates}
            self.assertGreaterEqual(len(distinct), 4, (position, distinct))

    def test_fc3d_five_ticket_pool_covers_five_digits_per_position(self):
        candidates, _ = generate_positional_ensemble("fc3d", self.fc3d_rows)
        for position in range(3):
            distinct = {item["number"][position] for item in candidates}
            self.assertEqual(len(distinct), 5, (position, distinct))

    def test_qxc_candidates_cover_three_digits_per_position(self):
        candidates, _ = generate_qxc(self.qxc_rows)
        for position in range(7):
            distinct = {item["number"][position] for item in candidates}
            self.assertGreaterEqual(len(distinct), 3, (position, distinct))

    def test_ssq_red_union_spans_the_board(self):
        candidates, _ = generate_ssq(self.ssq_rows, "2026083")
        red_union = set().union(*(set(item["red"]) for item in candidates))
        self.assertGreaterEqual(len(red_union), 26)
        for index, left in enumerate(candidates):
            for right in candidates[index + 1:]:
                self.assertLessEqual(len(set(left["red"]) & set(right["red"])), 1)

    def test_coverage_repair_spreads_pool_digits_directly(self):
        # The end-to-end assertions above pass even without the repair,
        # because diversified_rank usually spreads the pools on its own.
        # Exercise the repair itself on a list that has collapsed.
        collapsed = [(f"11{index}", -float(index), 0.0) for index in range(5)]
        repaired = ensure_position_pool_coverage(
            [item for item in collapsed],
            [[1, 2, 3], [1, 4, 5], [0, 1, 2]],
            lambda text: -sum(int(value) for value in text),
        )
        self.assertEqual(len(repaired), 5)
        self.assertEqual(len({item[0] for item in repaired}), 5)
        for position, pool in enumerate([[1, 2, 3], [1, 4, 5], [0, 1, 2]]):
            present = {int(item[0][position]) for item in repaired}
            self.assertTrue(set(pool) <= present, (position, pool, present))

    def test_fc3d_repeat_shape_coverage_is_bounded(self):
        rows = [{"numbers": ["7", "5", "5"]}] * 30
        scored = [
            ("925", -1.0, 0.0),
            ("438", -1.1, 0.0),
            ("217", -1.2, 0.0),
            ("495", -1.3, 0.0),
            ("728", -1.4, 0.0),
            ("922", -1.5, 0.0),
        ]
        selected = scored[:5]
        repaired = ensure_repeat_shape_coverage(selected, scored, rows)
        self.assertEqual(len(repaired), 5)
        self.assertTrue(any(len(set(item[0])) == 2 for item in repaired))
        for position in range(3):
            self.assertGreaterEqual(
                len({item[0][position] for item in repaired}),
                min(3, len({item[0][position] for item in selected})),
            )

    def test_pl3_repeat_shape_coverage_is_bounded(self):
        rows = [{"numbers": ["7", "5", "5"]}] * 30
        scored = [
            ("925", -1.0, 0.0),
            ("438", -1.1, 0.0),
            ("217", -1.2, 0.0),
            ("495", -1.3, 0.0),
            ("728", -1.4, 0.0),
            ("722", -1.5, 0.0),
        ]
        selected = scored[:5]
        repaired = ensure_repeat_shape_coverage(selected, scored, rows)
        self.assertEqual(len(repaired), 5)
        self.assertTrue(any(len(set(item[0])) == 2 for item in repaired))

    def test_kl8_groups_are_pairwise_disjoint(self):
        for pick_count in range(5, 11):
            candidates, _ = generate_kl8(self.kl8_rows, pick_count)
            union = set().union(*(set(item["numbers"]) for item in candidates))
            self.assertEqual(len(union), 5 * pick_count, pick_count)

    def test_kl8_cross_draw_penalty_is_soft_not_hard_exclusion(self):
        scores = kl8_cross_draw_scores({1: 0.50, 2: 0.50, 3: 0.10}, {1, 3}, penalty=0.08)
        self.assertAlmostEqual(scores[1], 0.42)
        self.assertAlmostEqual(scores[2], 0.50)
        self.assertAlmostEqual(scores[3], 0.02)
        self.assertIn(1, scores)
        with self.assertRaises(ValueError):
            kl8_cross_draw_scores({1: 0.5}, {1}, penalty=-0.01)

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
            self.assertTrue(all(len(item["cold_digits"]) == 3 for item in analysis["position_analysis"]))

    def test_direct_predictions_expose_structure_metrics(self):
        metrics = direct_number_metrics([4, 7, 3])
        self.assertEqual(metrics, {"sum": 14, "span": 4, "odd_even": "2:1", "distinct": 3, "shape": "三位不同"})
        payload = json.loads((Path(__file__).resolve().parents[1] / "docs/assets/data/dashboard.json").read_text(encoding="utf-8"))
        for game in ("pl3", "fc3d"):
            self.assertEqual(set(payload["games"][game]["prediction_summary"]), {"sum", "span", "odd_even"})
            self.assertEqual(len(payload["games"][game]["prediction_summary"]["sum"]["values"]), 3)
            summary = payload["games"][game]["prediction_summary"]
            for item in payload["games"][game]["top_candidates"]:
                self.assertEqual(set(item["prediction_metrics"]), {"sum", "span", "odd_even", "distinct", "shape"})
                self.assertIsInstance(item["prediction_metrics"]["sum"], int)
                self.assertIsInstance(item["prediction_metrics"]["span"], int)
                self.assertIn(item["prediction_metrics"]["odd_even"], {"0:3", "1:2", "2:1", "3:0"})

    def test_direct_hot_cold_template_exposes_focus_and_five_lists(self):
        payload = json.loads((Path(__file__).resolve().parents[1] / "docs/assets/data/dashboard.json").read_text(encoding="utf-8"))
        for game in ("pl3", "fc3d"):
            item = payload["games"][game]
            self.assertEqual(len(item["hot_candidates"]), 5)
            self.assertEqual(len(item["cold_candidates"]), 5)
            for position in item["analysis"]["position_analysis"]:
                self.assertEqual(len(position["hot_focus_digits"]), 2)
                self.assertEqual(len(position["cold_focus_digits"]), 2)
                self.assertEqual(len(position["hot_occurrences"]), 3)
                self.assertEqual(len(position["cold_occurrences"]), 3)

    def test_structure_forecast_excludes_latest_settled_draw(self):
        rows = [{"numbers": ["9", "6", "1"]}] + [{"numbers": ["0", "0", "0"]} for _ in range(5)]
        summary = direct_prediction_summary(rows)
        self.assertNotIn(16, summary["sum"]["values"])
        self.assertEqual(summary["sum"]["values"][0], 0)

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

    def test_kl8_play_types_diversify_lead_lines_across_pick_counts(self):
        play_types = generate_kl8_play_types(self.kl8_rows, [2, 3, 4])
        lead_groups = [set(play["candidates"][0][0]["numbers"]) for play in play_types.values()]
        pairwise_overlaps = [
            len(left & right)
            for index, left in enumerate(lead_groups)
            for right in lead_groups[index + 1:]
        ]
        self.assertLessEqual(max(pairwise_overlaps), 4)
        self.assertGreaterEqual(len(set().union(*lead_groups)), 9)

    def test_fc3d_official_history(self):
        self.assertGreaterEqual(len(self.fc3d_rows), 100)
        self.assertEqual(self.fc3d_rows[0]["issue"], json.loads(
            (Path(__file__).resolve().parents[1] / "data/processed/draws.json").read_text(encoding="utf-8")
        )["draws"]["fc3d"][0]["issue"])

    def test_generated_output_has_only_direct_lists(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "docs/assets/data/dashboard.json").read_text(encoding="utf-8")
        games = json.loads(text)["games"]
        self.assertNotIn("play_types", games["pl3"])
        self.assertNotIn("play_types", games["fc3d"])
        self.assertNotIn("play_types", games["kl8"])
        self.assertEqual(len(games["kl8"]["top_candidates"]), 5)
        self.assertEqual(
            {item["pick_count"] for item in games["kl8"]["top_candidates"]},
            {2, 3, 4},
        )
        self.assertTrue(all(
            sum(item["pick_count"] == pick_count for item in games["kl8"]["top_candidates"]) >= 1
            for pick_count in (2, 3, 4)
        ))
        union = set().union(*(set(item["numbers"]) for item in games["kl8"]["top_candidates"]))
        self.assertGreaterEqual(len(union), 12)
        for suffix in ("3", "6"):
            self.assertNotIn("组选" + suffix, text)

    def test_direct_digit_reviews_explain_positions_and_next_day_form(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "docs/assets/data/dashboard.json").read_text(encoding="utf-8"))
        for game in ("pl3", "fc3d"):
            review = payload["games"][game]["model_review"]
            self.assertEqual([item["position"] for item in review["position_diagnostics"]], ["百位", "十位", "个位"])
            self.assertTrue(all(item["reason"] for item in review["position_diagnostics"]))
            self.assertEqual(len(review["next_day_advice"]), 5)
            self.assertTrue(review["next_day_advice_text"])
            self.assertEqual(len(payload["games"][game]["analysis"]["selected_position_parameters"]), 3)
            self.assertTrue(payload["games"][game]["top_candidates"][0]["purchase_suggestion"])
            self.assertEqual(
                [(item["number"], item["suggestion"]) for item in review["next_day_advice"]],
                [(item["number"], item["purchase_suggestion"]) for item in payload["games"][game]["top_candidates"]],
            )
            for position in range(3):
                self.assertLessEqual(
                    len({item["number"][position] for item in payload["games"][game]["top_candidates"][:3]}),
                    3,
                )

    def test_kl8_review_reports_union_pool_coverage_and_attribution(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "docs/assets/data/dashboard.json").read_text(encoding="utf-8"))
        review = payload["games"]["kl8"]["model_review"]
        covered, total = review["number_pool_coverage"].split(" / ")
        self.assertEqual(total, "20")
        self.assertGreaterEqual(int(covered), 0)
        self.assertTrue(review["missed_numbers"])
        self.assertTrue(review["error_attribution"])
        self.assertEqual(len(review["model_adjustments"]), 2)

    def test_set_game_reviews_are_deep_and_region_calibrated(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "docs/assets/data/dashboard.json").read_text(encoding="utf-8"))
        for game, regions, positions in (("dlt", {"前区", "后区"}, 7), ("ssq", {"红球", "蓝球"}, 7), ("qxc", set(), 7)):
            review = payload["games"][game]["model_review"]
            self.assertEqual(len(review["position_diagnostics"]), positions)
            self.assertTrue(review["error_attribution"])
            self.assertEqual(len(review["model_adjustments"]), 2)
            if regions:
                self.assertEqual({item["region"] for item in review["region_diagnostics"]}, regions)
        self.assertEqual(set(payload["games"]["dlt"]["analysis"]["selected_region_windows"]), {"front", "back"})
        self.assertEqual(set(payload["games"]["ssq"]["analysis"]["selected_region_windows"]), {"red", "blue"})
        self.assertEqual(len(payload["games"]["qxc"]["analysis"]["selected_position_windows"]), 7)

    def test_kl8_cards_wrap_numbers_without_overflow(self):
        root = Path(__file__).resolve().parents[1]
        css = (root / "docs/assets/css/detail.css").read_text(encoding="utf-8")
        self.assertIn('body[data-game="kl8"] .pick-number', css)
        self.assertIn("white-space: normal", css)
        self.assertIn("overflow: hidden", css)

    def test_hot_and_cold_profiles_are_separate(self):
        hot = generate_digit_profile(self.fc3d_rows, 3, "hot", 5, "fc3d")
        cold = generate_digit_profile(self.fc3d_rows, 3, "cold", 5, "fc3d")
        self.assertEqual(len(hot), 5)
        self.assertEqual(len(cold), 5)
        self.assertTrue(all(item[2] > 0.25 for item in hot))
        self.assertTrue(all(item[2] < -0.25 for item in cold))
        self.assertFalse({item[0] for item in hot} & {item[0] for item in cold})

        hot_scores = digit_confidences(self.fc3d_rows, 3, [item[0] for item in hot], "fc3d")
        cold_scores = digit_confidences(self.fc3d_rows, 3, [item[0] for item in cold], "fc3d")
        self.assertGreater(min(hot_scores), max(cold_scores))

    def test_global_top_is_positionally_diverse(self):
        for game, rows in (("pl3", self.rows), ("fc3d", self.fc3d_rows)):
            numbers = [item[0] for item in generate_digit_profile(rows, 3, "global", 5, game)]
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
            "pl3": 5,
            "pl5": 5,
            "fc3d": 5,
        }
        for key, count in expected.items():
            game = games[key]
            candidates = game["top_candidates"]
            self.assertEqual(len(candidates), count)
            self.assertEqual(len({item["number"] for item in candidates}), count)
            self.assertNotIn("strategy_zones", game)
            scores = [item["confidence"] for item in candidates]
            self.assertEqual(scores[:3], sorted(scores[:3], reverse=True))
            self.assertEqual(scores[3:], sorted(scores[3:], reverse=True))

    def test_positional_ensemble_uses_one_ranked_pool(self):
        pl3, _ = generate_positional_ensemble("pl3", self.rows)
        pl5, _ = generate_positional_ensemble("pl5", self.pl5_rows)
        fc3d, _ = generate_positional_ensemble("fc3d", self.fc3d_rows)
        self.assertEqual(len(pl3), 5)
        self.assertEqual(len(pl5), 5)
        self.assertEqual(len(fc3d), 5)
        self.assertEqual(Counter(item["source"] for item in pl3), {"hot_position_pool": 5})
        for candidates in (pl5,):
            self.assertEqual(Counter(item["source"] for item in candidates), {"hot_position_pool": len(candidates)})
        self.assertEqual(Counter(item["source"] for item in fc3d), {"hot_position_pool": 5})

    def test_pl5_uses_its_own_five_positions(self):
        candidates, _ = generate_positional_ensemble("pl5", self.pl5_rows)
        self.assertTrue(all(len(item["number"]) == 5 for item in candidates))

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
            self.assertEqual(
                item["copy_text"],
                "\n".join(f"{item['name']} {scheme['result']}" for scheme in item["results"]),
            )
            if item["game"] == "kl8":
                for scheme in item["results"]:
                    values = scheme["result"].split()
                    self.assertEqual(len(values), 10)
                    self.assertEqual(len(set(values)), 10)
                    self.assertTrue(all(len(value) == 2 and 1 <= int(value) <= 80 for value in values))
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
        self.assertIn("draw-board-legend", homepage)
        self.assertIn("legend-today", homepage)
        for heading in ("玩法", "目标期号", "下一期开奖时间", "开奖安排"):
            self.assertIn(heading, homepage)
        self.assertIn("NEXT DRAW BOARD", homepage)
        self.assertIn('$("#draw-board").innerHTML', homepage_script)
        self.assertIn("isTodayDraw(game.next_draw_at)", homepage_script)
        self.assertIn("今日开奖", homepage_script)
        self.assertIn("draw-today-badge", homepage_script)
        self.assertIn("draw-sector-badge", homepage_script)
        self.assertIn("sectorName(key, game)", homepage_script)
        dashboard = json.loads((root / "docs/assets/data/dashboard.json").read_text(encoding="utf-8"))
        for game in dashboard["games"].values():
            self.assertIn(game["sector"], ("fucai", "ticai"))
            self.assertIn(game["sector_name"], ("福彩", "体彩"))
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
