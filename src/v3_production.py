"""Production adapters for the supplied v3 models.

The handoff models use chronological records and richer report structures;
the dashboard uses newest-first records and a stable five-candidate contract.
This module is the only translation boundary between those two formats.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

try:
    from vendor_models_v3 import dlt_model_v3, kl8_model_v3, ssq_model_v3
except ModuleNotFoundError:
    from src.vendor_models_v3 import dlt_model_v3, kl8_model_v3, ssq_model_v3


# A previous draw is not a hard exclusion: five repeats is the theoretical
# average for two independent 20-from-80 draws. This soft adjustment reduces
# sticky hot-number portfolios without forcing an artificial zero overlap.
KL8_CROSS_DRAW_PENALTY = 0.08


def _chronological(rows: list[dict]) -> list[dict]:
    return [dict(row) for row in reversed(rows)]


def _score_combo(values: tuple[int, ...], scores: dict[int, float]) -> float:
    return sum(scores.get(value, 0.0) for value in values)


def kl8_cross_draw_scores(
    fused_scores: dict[int, float],
    previous_numbers: set[int],
    penalty: float = KL8_CROSS_DRAW_PENALTY,
) -> dict[int, float]:
    """Apply a soft penalty to numbers drawn in the immediately prior issue."""
    if penalty < 0:
        raise ValueError("KL8 cross-draw penalty must be non-negative")
    return {
        int(number): float(score) - (penalty if int(number) in previous_numbers else 0.0)
        for number, score in fused_scores.items()
    }


def generate_dlt_v3(rows: list[dict], issue: str) -> tuple[list[dict], list[float]]:
    records = [
        {"front": [int(value) for value in row["numbers"][:5]],
         "back": [int(value) for value in row["numbers"][5:]],
         "period": str(row["issue"])}
        for row in _chronological(rows)
    ]
    report = dlt_model_v3.generate_dlt_v3(records, issue)
    fused = {int(key): float(value) for key, value in report.get("fused_scores", {}).items()}
    front_pool = sorted(range(1, 36), key=lambda value: fused.get(value, 0.0), reverse=True)[:25]
    fronts = list(combinations(front_pool, 5))
    back_info = report.get("back_info", {})
    back_freq = {int(key): float(value) for key, value in back_info.get("freq", {}).items()}
    back_pairs = sorted(
        combinations(range(1, 13), 2),
        key=lambda pair: (sum(back_freq.get(value, 0.0) for value in pair), pair),
        reverse=True,
    )
    selected: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    used_back: set[tuple[int, int]] = set()
    used_back_values: set[int] = set()
    remaining_fronts = list(fronts)
    for _ in range(5):
        chosen_front = max(
            remaining_fronts,
            key=lambda front: _score_combo(front, fused) - 0.9 * max(
                (len(set(front) & set(old_front)) for old_front, _ in selected), default=0
            ),
        )
        front_tuple = tuple(sorted(chosen_front))
        back = min(
            (pair for pair in back_pairs if pair not in used_back),
            key=lambda pair: (len(set(pair) & used_back_values), -sum(back_freq.get(value, 0.0) for value in pair)),
        )
        selected.append((front_tuple, tuple(back)))
        used_back.add(tuple(back))
        used_back_values.update(back)
        remaining_fronts.remove(chosen_front)
    selected.sort(key=lambda item: _score_combo(item[0], fused) + _score_combo(item[1], back_freq), reverse=True)
    candidates = [{"front": list(front), "back": list(back), "source": "v3_ensemble"} for front, back in selected[:5]]
    scores = [_score_combo(tuple(item["front"]), fused) + _score_combo(tuple(item["back"]), back_freq) for item in candidates]
    return candidates, scores


def generate_ssq_v3(rows: list[dict], issue: str) -> tuple[list[dict], list[float]]:
    records = [
        {"red": [int(value) for value in row["numbers"][:6]],
         "blue": int(row["numbers"][6]), "period": str(row["issue"])}
        for row in _chronological(rows)
    ]
    report = ssq_model_v3.generate_ssq_v3(records, issue)
    fused = {int(key): float(value) for key, value in report.get("fused_scores", {}).items()}
    pool = sorted(set(int(value) for value in report.get("pool", [])), key=lambda value: fused.get(value, 0.0), reverse=True)
    red_pool = sorted(range(1, 34), key=lambda value: fused.get(value, 0.0), reverse=True)[:26]
    reds = list(combinations(red_pool, 6))
    selected: list[tuple[int, ...]] = []
    for _ in range(5):
        choice = min(
            reds,
            key=lambda combo: (
                max((len(set(combo) & set(old)) for old in selected), default=0) > 1,
                max((len(set(combo) & set(old)) for old in selected), default=0),
                -_score_combo(combo, fused),
            ),
        )
        selected.append(choice)
        reds.remove(choice)
    blue_scores = {int(value): float(score) for value, score in report.get("blue_pred", [])}
    blue_bands = ((1, 4), (5, 8), (9, 12), (13, 16))
    blues = [
        max(range(low, high + 1), key=lambda value: blue_scores.get(value, 0.0))
        for low, high in blue_bands
    ]
    blues.append(max((value for value in range(1, 17) if value not in blues), key=lambda value: blue_scores.get(value, 0.0)))
    candidates = [{"red": sorted(red), "blue": [blues[index]], "source": "v3_ensemble"} for index, red in enumerate(selected)]
    scores = [_score_combo(tuple(item["red"]), fused) + blue_scores.get(item["blue"][0], 0.0) for item in candidates]
    order = sorted(range(len(candidates)), key=lambda index: scores[index], reverse=True)
    return [candidates[index] for index in order], [scores[index] for index in order]


def generate_kl8_v3(rows: list[dict], pick_count: int) -> tuple[list[dict], list[float]]:
    records = [{"numbers": [int(value) for value in row["numbers"]], "period": str(row["issue"])} for row in _chronological(rows)]
    report = kl8_model_v3.generate_kl8_v3(records, pick=pick_count, next_period="NEXT")
    data = report.get("results_by_pick", {}).get(pick_count, {})
    fused = {int(key): float(value) for key, value in report.get("fused_scores", {}).items()}
    previous_numbers = {int(value) for value in records[-1]["numbers"]}
    adjusted_scores = kl8_cross_draw_scores(fused, previous_numbers)
    ranked_numbers = sorted(range(1, 81), key=lambda value: adjusted_scores.get(value, 0.0), reverse=True)
    selected = [tuple(sorted(ranked_numbers[index * pick_count:(index + 1) * pick_count])) for index in range(5)]
    scores = [_score_combo(combo, adjusted_scores) for combo in selected]
    order = sorted(range(5), key=lambda index: scores[index], reverse=True)
    return [{"numbers": list(selected[index]), "mix_label": f"v3选{pick_count}跨期软惩罚", "source": "v3_ensemble_cross_draw_soft"} for index in order], [scores[index] for index in order]
