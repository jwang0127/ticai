from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
from collections import Counter
from itertools import combinations, product
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
try:
    from v3_production import generate_dlt_v3, generate_kl8_v3, generate_ssq_v3
except ModuleNotFoundError:
    from src.v3_production import generate_dlt_v3, generate_kl8_v3, generate_ssq_v3

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "games.json"
DATA_PATH = ROOT / "data" / "processed" / "draws.json"
OUTPUT_PATH = ROOT / "docs" / "assets" / "data" / "dashboard.json"
REVIEWS_PATH = ROOT / "data" / "processed" / "model_reviews.json"
GAME_SECTORS = {
    "fc3d": ("fucai", "福彩"),
    "ssq": ("fucai", "福彩"),
    "kl8": ("fucai", "福彩"),
    "dlt": ("ticai", "体彩"),
    "pl3": ("ticai", "体彩"),
    "pl5": ("ticai", "体彩"),
    "qxc": ("ticai", "体彩"),
}
try:
    TZ = ZoneInfo("Asia/Shanghai")
except ZoneInfoNotFoundError:
    TZ = timezone(timedelta(hours=8))
RECENCY_DECAY = 18
# Stronger line-level dispersion keeps five tickets from inheriting the same
# sorted front positions. This only changes the portfolio mix, not per-number
# probabilities; the backtest compares the wider union against the old floor.
DLT_DIVERSITY_PENALTY = 0.12
DLT_FRONT_UNION_FLOOR = 20
DLT_BACK_UNION_FLOOR = 8
SSQ_RED_OVERLAP_LIMIT = 1
SSQ_RED_UNION_FLOOR = 26
WINDOW_BLEND = ((50, 0.15), (100, 0.20), (300, 0.20), (500, 0.20), (1000, 0.15), (1500, 0.10))

# 排列3/排列5共享位置模型；福彩3D使用独立参数。7星彩和双色球另有专用生成器。
DIGIT_MODELS = {
    # Selected from rolling top-3 positional coverage on each game's own
    # history. PL3 and FC3D are deliberately calibrated separately.
    "pl3": {"decay": 13, "frequency": 0.58, "omission": 0.050, "crowding": 0.18,
            "sum": 0.0, "unique": 0.0, "global_omission": 0.045,
            "global_unique": 0.0, "diversity": 1.6, "structure_weight": 0.06},
    "pl5": {"decay": 18, "frequency": 0.55, "omission": 0.045, "crowding": 0.16,
            "sum": 0.0, "unique": 0.0, "global_omission": 0.040,
            "global_unique": 0.0, "diversity": 1.45},
    "fc3d": {"decay": 18, "frequency": 0.66, "omission": 0.045, "crowding": 0.30,
             "sum": 0.0, "unique": 0.0, "global_omission": 0.035,
             "global_unique": 0.0, "diversity": 1.85},
}

BACKTEST_DECAYS = (8, 13, 18, 24, 30, 45, 60)
BACKTEST_WINDOWS = (100, 300, 500, 1000, 1500)
DIGIT_BACKTEST_FOLDS = 300
POOL_BACKTEST_FOLDS = 240
UNIFORM_TOP3_BASELINE = 0.3

# Calibration reruns the same rolling grid for one immutable history snapshot
# from several call sites (generation, analysis, confidence scaling); memoize
# on the snapshot identity so each grid is evaluated once per process.
_CALIBRATION_CACHE: dict[tuple, dict] = {}


def _history_key(game: str, rows: list[dict]) -> tuple:
    if not rows:
        return (game, 0, "", "")
    return (game, len(rows), str(rows[0]["issue"]), str(rows[-1]["issue"]))


def binomial_se(rate: float, observations: int) -> float:
    if observations <= 0:
        return 0.0
    return math.sqrt(max(rate * (1.0 - rate), 0.0) / observations)


def _digit_sequence(rows: list[dict], position: int) -> list[int]:
    return [int(row["numbers"][position]) for row in rows]


def _sliding_top3_hits(sequence: list[int], decay: int, window_size: int, folds: int) -> tuple[int, int]:
    """Count held-out top-3 hits with an O(1)-per-fold sliding decayed window.

    Fold at index i trains on sequence[i:i+window] (age 0 = newest) and tests
    sequence[i-1]. Sliding the window one draw forward rescales every weight
    by exp(-1/decay), drops the stale tail row, and admits the new head row at
    weight 1. This matches rebuilding the decayed Counter per fold up to two
    Two details keep this an exact match for the rebuild rather than an
    approximation of it. Membership is decided by an exact integer occupancy
    count, not by the decayed weight: repeated rescaling leaves floating-point
    residue on a digit that has fully left the window, and that residue would
    otherwise rank it above a digit the window never contained (which a freshly
    built counter holds no key for at all). Ties among equal weights resolve by
    digit value, so each fold is a pure function of its own training window
    rather than of how many folds preceded it.
    """
    limit = min(folds, max(0, len(sequence) - window_size))
    if limit <= 0:
        return 0, 0
    fade = math.exp(-1.0 / decay)
    tail_weight = math.exp(-window_size / decay)
    counts = [0.0] * 10
    occupancy = [0] * 10
    for age in range(window_size):
        counts[sequence[limit + age]] += math.exp(-age / decay)
        occupancy[sequence[limit + age]] += 1
    hits = 0
    index = limit
    while True:
        top3 = sorted(
            (digit for digit in range(10) if occupancy[digit]),
            key=lambda digit: (-counts[digit], digit),
        )[:3]
        hits += int(sequence[index - 1] in top3)
        index -= 1
        if index < 1:
            break
        for digit in range(10):
            counts[digit] *= fade
        departing = sequence[index + window_size]
        counts[departing] -= tail_weight
        occupancy[departing] -= 1
        counts[sequence[index]] += 1.0
        occupancy[sequence[index]] += 1
    return hits, limit


def rolling_digit_backtest(rows: list[dict], digits: int, decay: int, window_size: int, folds: int = DIGIT_BACKTEST_FOLDS) -> dict:
    """Evaluate positional top-three coverage without using the held-out draw."""
    hits = total = 0
    limit = 0
    for position in range(digits):
        position_hits, limit = _sliding_top3_hits(_digit_sequence(rows, position), decay, window_size, folds)
        hits += position_hits
        total += limit
    rate = hits / total if total else 0.0
    return {
        "folds": limit,
        "positional_top3_hit_rate": round(rate, 4),
        "se": round(binomial_se(rate, total), 4),
        "baseline": UNIFORM_TOP3_BASELINE,
    }


def rolling_digit_position_backtest(
    rows: list[dict], position: int, decay: int, window_size: int, folds: int = DIGIT_BACKTEST_FOLDS
) -> dict:
    """Evaluate one digit position without using the held-out draw."""
    hits, limit = _sliding_top3_hits(_digit_sequence(rows, position), decay, window_size, folds)
    rate = hits / limit if limit else 0.0
    return {
        "folds": limit,
        "top3_hit_rate": round(rate, 4),
        "se": round(binomial_se(rate, limit), 4),
        "baseline": UNIFORM_TOP3_BASELINE,
    }


def _one_se_choice(cells: list[tuple[int, int]], scores: dict[str, dict], rate_key: str, observations_per_fold: int = 1) -> tuple[int, int]:
    """Pick parameters with a one-standard-error rule instead of the raw max.

    For fair draws every cell shares the same true hit rate, so the observed
    grid maximum is mostly selection noise. Among cells statistically tied with
    the best (within one binomial SE), prefer the largest window and the
    slowest decay: the most stable, least noise-chasing estimator.
    """
    def rate(cell: tuple[int, int]) -> float:
        return scores[f"{cell[0]}@{cell[1]}"][rate_key]

    def observations(cell: tuple[int, int]) -> int:
        return scores[f"{cell[0]}@{cell[1]}"]["folds"] * observations_per_fold

    best = max(cells, key=rate)
    threshold = rate(best) - binomial_se(rate(best), observations(best))
    eligible = [cell for cell in cells if rate(cell) >= threshold]
    return max(eligible, key=lambda cell: (cell[1], cell[0], rate(cell)))


def calibrate_digit_model(game: str, rows: list[dict], digits: int) -> dict:
    """Return the isolated model plus rolling-backtest evidence for the game."""
    cache_key = ("digit", _history_key(game, rows), digits)
    cached = _CALIBRATION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    base = dict(digit_model(game))
    windows = [window for window in BACKTEST_WINDOWS if window < len(rows)] or [max(1, len(rows) - 1)]
    cells = [(decay, window) for decay in BACKTEST_DECAYS for window in windows]
    scores = {
        f"{decay}@{window}": rolling_digit_backtest(rows, digits, decay, window)
        for decay, window in cells
    }
    chosen_decay, chosen_window = _one_se_choice(cells, scores, "positional_top3_hit_rate", digits)
    position_parameters = []
    position_backtest = []
    for position in range(digits):
        position_scores = {
            f"{decay}@{window}": rolling_digit_position_backtest(rows, position, decay, window)
            for decay, window in cells
        }
        position_decay, position_window = _one_se_choice(cells, position_scores, "top3_hit_rate")
        position_parameters.append({"position": position, "decay": position_decay, "window_size": position_window})
        position_backtest.append({"position": position, "scores": position_scores})
    base["decay"] = chosen_decay
    base["window_size"] = chosen_window
    base["position_parameters"] = position_parameters
    result = {
        "parameters": base,
        "backtest": scores,
        "position_backtest": position_backtest,
        "selected_decay": chosen_decay,
        "selected_window": chosen_window,
    }
    _CALIBRATION_CACHE[cache_key] = result
    return result


def order_statistic_top_mass(total: int, drawn: int, position: int, top_k: int) -> float:
    """Best possible top-k hit rate for one sorted position of a fair draw.

    Sorted positions concentrate (the first of five numbers from 1-35 is small
    far more often than not), so a fair-lottery oracle already scores well on
    them. Reporting this exact mass as the baseline keeps positional coverage
    numbers honest instead of implying the frequency model found a signal.
    """
    denominator = math.comb(total, drawn)
    probabilities = [
        math.comb(value - 1, position) * math.comb(total - value, drawn - 1 - position) / denominator
        for value in range(1, total + 1)
    ]
    return sum(sorted(probabilities, reverse=True)[:top_k])


def _pool_backtest_baseline(game: str, digit_count: int) -> float:
    if game == "dlt":
        front = [order_statistic_top_mass(35, 5, position, 10) for position in range(5)]
        back = [order_statistic_top_mass(12, 2, position, 4) for position in range(2)]
        return sum(front + back) / 7
    if game == "ssq":
        return (6 * (18 / 33) + (5 / 16)) / 7
    if game == "kl8":
        return 40 / 80
    return UNIFORM_TOP3_BASELINE


def _positive_top_values(counter: Counter[int], top_k: int) -> set[int]:
    """Deterministic top-k pool: count descending, then value ascending.

    Integer count ties at the k-th slot are pervasive in unweighted windows,
    and Counter.most_common breaks them by insertion order - which for a
    long-lived sliding counter is an artifact of the traversal path (it even
    depends on how many folds ran before this one). Ordering ties by value
    makes every fold's pool a pure function of its own training window.
    Zero-count keys left behind by sliding removal are excluded, matching a
    freshly built counter."""
    ranked = sorted(
        ((value, count) for value, count in counter.items() if count > 0),
        key=lambda item: (-item[1], item[0]),
    )
    return {value for value, _ in ranked[:top_k]}


def rolling_pool_backtest(game: str, rows: list[dict], window_size: int, folds: int = POOL_BACKTEST_FOLDS) -> dict:
    """Evaluate position/set coverage with an O(1)-per-fold sliding window."""
    limit = min(folds, max(0, len(rows) - window_size))
    digit_count = len(rows[0]["numbers"]) if rows else 0
    baseline = _pool_backtest_baseline(game, digit_count)
    if limit <= 0:
        return {"folds": 0, "pool_coverage": 0.0, "se": 0.0, "baseline": round(baseline, 4)}
    values = [[int(value) for value in row["numbers"]] for row in rows]
    if game == "dlt":
        specs = [("pos", pos, 10, None) for pos in range(5)] + [("pos", pos, 4, None) for pos in (5, 6)]
    elif game == "ssq":
        specs = [("set", (0, 6), 18, 33), ("set", (6, 7), 5, 16)]
    elif game == "kl8":
        specs = [("set", (0, 20), 40, 80)]
    else:
        specs = [("pos", pos, 3, None) for pos in range(digit_count)]
    counters: list[Counter[int]] = [Counter() for _ in specs]

    def apply(row_values: list[int], sign: int) -> None:
        for counter, (kind, argument, _, _) in zip(counters, specs):
            if kind == "pos":
                counter[row_values[argument]] += sign
            else:
                for value in row_values[argument[0]:argument[1]]:
                    counter[value] += sign

    for offset in range(window_size):
        apply(values[limit + offset], 1)
    hits = total = 0
    index = limit
    while True:
        actual = values[index - 1]
        for counter, (kind, argument, top_k, _) in zip(counters, specs):
            ranked = _positive_top_values(counter, top_k)
            if kind == "pos":
                hits += int(actual[argument] in ranked)
                total += 1
            else:
                targets = actual[argument[0]:argument[1]]
                hits += len(set(targets) & ranked)
                total += len(targets)
        index -= 1
        if index < 1:
            break
        apply(values[index], 1)
        apply(values[index + window_size], -1)
    rate = hits / total if total else 0.0
    # Within one fold a set spec's n covered numbers are drawn without
    # replacement, so per-fold hit variance is hypergeometric: (N-n)/(N-1)
    # times binomial. Positional specs are single draws (factor 1); the
    # cross-position correlation of DLT's sorted positions is second-order
    # and left uncorrected.
    weight_total = variance_weight = 0.0
    for kind, argument, _, population in specs:
        drawn = (argument[1] - argument[0]) if kind == "set" else 1
        factor = (population - drawn) / (population - 1) if kind == "set" else 1.0
        variance_weight += drawn * factor
        weight_total += drawn
    se = binomial_se(rate, total) * math.sqrt(variance_weight / weight_total) if total else 0.0
    return {
        "folds": limit,
        "pool_coverage": round(rate, 4),
        "se": round(se, 4),
        "baseline": round(baseline, 4),
    }


REGION_SPECS = {
    ("dlt", "front"): (0, 5, 18, 35),
    ("dlt", "back"): (5, 7, 6, 12),
    ("ssq", "red"): (0, 6, 18, 33),
    ("ssq", "blue"): (6, 7, 5, 16),
}


def rolling_region_backtest(game: str, rows: list[dict], window_size: int, region: str, folds: int = POOL_BACKTEST_FOLDS) -> dict:
    """Backtest the independent regions used by DLT and SSQ generators."""
    if (game, region) not in REGION_SPECS:
        raise ValueError(f"unsupported region calibration: {game}/{region}")
    start, end, top_k, pool_size = REGION_SPECS[(game, region)]
    baseline = top_k / pool_size
    limit = min(folds, max(0, len(rows) - window_size))
    if limit <= 0:
        return {"folds": 0, "pool_coverage": 0.0, "se": 0.0, "baseline": round(baseline, 4)}
    values = [[int(value) for value in row["numbers"][start:end]] for row in rows]
    counts: Counter[int] = Counter()
    for offset in range(window_size):
        counts.update(values[limit + offset])
    hits = total = 0
    index = limit
    while True:
        actual = set(values[index - 1])
        hits += len(actual & _positive_top_values(counts, top_k))
        total += len(actual)
        index -= 1
        if index < 1:
            break
        counts.update(values[index])
        counts.subtract(values[index + window_size])
    rate = hits / total if total else 0.0
    # Hypergeometric correction: the region's numbers are drawn without
    # replacement against a fixed top-k pool, so binomial SE overstates the
    # spread and would loosen the one-SE eligibility threshold.
    drawn = end - start
    se = binomial_se(rate, total) * math.sqrt((pool_size - drawn) / (pool_size - 1)) if total else 0.0
    return {
        "folds": limit,
        "pool_coverage": round(rate, 4),
        "se": round(se, 4),
        "baseline": round(baseline, 4),
    }


def _one_se_window(windows: list[int], scores: dict[str, dict], rate_key: str, observations_per_fold: int = 1) -> int:
    """Window analogue of the one-standard-error rule: statistically tied
    windows resolve to the largest, most stable one."""
    def rate(window: int) -> float:
        return scores[str(window)][rate_key]

    def observations(window: int) -> int:
        return scores[str(window)]["folds"] * observations_per_fold

    best = max(windows, key=rate)
    threshold = rate(best) - binomial_se(rate(best), observations(best))
    return max(window for window in windows if rate(window) >= threshold)


def calibrate_set_model(game: str, rows: list[dict]) -> dict:
    cache_key = ("set", _history_key(game, rows))
    cached = _CALIBRATION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    windows = [window for window in BACKTEST_WINDOWS if window < len(rows)] or [max(1, len(rows) - 1)]
    pool_observations = {"dlt": 7, "ssq": 7, "kl8": 20}.get(game, len(rows[0]["numbers"]) if rows else 1)
    scores = {str(window): rolling_pool_backtest(game, rows, window) for window in windows}
    chosen = _one_se_window(windows, scores, "pool_coverage", pool_observations)
    regions = {"dlt": ("front", "back"), "ssq": ("red", "blue")}.get(game, ())
    region_windows = {}
    region_backtest = {}
    for region in regions:
        region_scores = {
            str(window): rolling_region_backtest(game, rows, window, region)
            for window in windows
        }
        region_observations = REGION_SPECS[(game, region)][1] - REGION_SPECS[(game, region)][0]
        region_windows[region] = _one_se_window(windows, region_scores, "pool_coverage", region_observations)
        region_backtest[region] = region_scores
    position_windows = {}
    position_backtest = {}
    if game == "qxc":
        for position in range(7):
            position_scores = {
                str(window): rolling_digit_position_backtest(rows, position, 27, window)
                for window in windows
            }
            position_windows[str(position)] = _one_se_window(windows, position_scores, "top3_hit_rate")
            position_backtest[str(position)] = position_scores
    result = {
        "selected_window": chosen,
        "backtest": scores,
        "region_windows": region_windows,
        "region_backtest": region_backtest,
        "position_windows": position_windows,
        "position_backtest": position_backtest,
    }
    _CALIBRATION_CACHE[cache_key] = result
    return result


def digit_model(game: str) -> dict:
    return DIGIT_MODELS[game if game in ("pl3", "pl5", "fc3d") else "pl3"]


def next_draw(now: datetime, weekdays: list[int], draw_clock: time) -> datetime:
    for offset in range(8):
        date = (now + timedelta(days=offset)).date()
        candidate = datetime.combine(date, draw_clock, tzinfo=TZ)
        if candidate.weekday() in weekdays and candidate > now:
            return candidate
    raise RuntimeError("无法计算下一期开奖时间")


def weighted_counts(rows: list[dict], position: int, decay: float = RECENCY_DECAY) -> Counter[int]:
    counts: Counter[int] = Counter()
    for index, row in enumerate(rows):
        if position >= len(row["numbers"]):
            continue
        counts[int(row["numbers"][position])] += math.exp(-index / decay)
    return counts


def blended_position_counts(rows: list[dict], position: int, decay: float, max_window: int | None = None, value_range: range | None = None) -> Counter[int]:
    """Blend short, medium, and full available windows within cached history."""
    result: Counter[int] = Counter()
    active = [(size, share) for size, share in WINDOW_BLEND if max_window is None or size <= max_window]
    total_share = sum(share for _, share in active) or 1.0
    for size, share in active:
        share /= total_share
        counts = weighted_counts(rows[: min(size, len(rows))], position, decay)
        total = sum(counts.values()) or 1.0
        for digit in value_range or range(10):
            result[digit] += share * counts[digit] / total
    return result


def relative_confidences(scores: list[float]) -> list[int]:
    """Scale comparable raw scores within one candidate pool."""
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if math.isclose(low, high):
        return [64] * len(scores)
    return [round(52 + (score - low) / (high - low) * 25, 1) for score in scores]


def stable_rng(game: str, issue: str) -> random.Random:
    seed = hashlib.sha256(f"{game}:{issue}:v2".encode()).hexdigest()
    return random.Random(int(seed[:16], 16))


def score_digit(number: int, counts: Counter[int], total: float) -> float:
    return (counts[number] + 0.65) / (total + 6.5)


def position_omissions(rows: list[dict], position: int) -> dict[int, int]:
    result: dict[int, int] = {}
    for digit in range(10):
        result[digit] = len(rows)
        for index, row in enumerate(rows):
            if position < len(row["numbers"]) and int(row["numbers"][position]) == digit:
                result[digit] = index
                break
    return result


def mixed_digit_components(rows: list[dict], digits: int, model: dict | None = None) -> tuple[list[Counter[int]], list[float], list[dict[int, int]], list[float], list[float]]:
    """Build hot/cold signals without treating either as predictive certainty."""
    model = model or DIGIT_MODELS["pl3"]
    position_parameters = model.get("position_parameters", [])
    position_counts = [
        blended_position_counts(
            rows,
            position,
            position_parameters[position].get("decay", model["decay"])
            if position < len(position_parameters) else model["decay"],
            position_parameters[position].get("window_size", model.get("window_size"))
            if position < len(position_parameters) else model.get("window_size"),
        )
        for position in range(digits)
    ]
    totals = [sum(counter.values()) for counter in position_counts]
    omissions = [position_omissions(rows, position) for position in range(digits)]
    means: list[float] = []
    scales: list[float] = []
    for pos, counter in enumerate(position_counts):
        probabilities = [score_digit(n, counter, totals[pos]) for n in range(10)]
        mean = sum(probabilities) / 10
        scale = math.sqrt(sum((value - mean) ** 2 for value in probabilities) / 10) or 1.0
        means.append(mean)
        scales.append(scale)
    return position_counts, totals, omissions, means, scales


def mixed_number_score(values: list[int], components: tuple, model: dict | None = None) -> tuple[float, float]:
    model = model or DIGIT_MODELS["pl3"]
    position_counts, totals, omissions, means, scales = components
    score = 0.0
    heat_values = []
    for pos, value in enumerate(values):
        probability = score_digit(value, position_counts[pos], totals[pos])
        heat_z = (probability - means[pos]) / scales[pos]
        heat_values.append(heat_z)
        # Keep a majority hot-frequency signal, cap omission compensation, and
        # only punish the most crowded tail. This is a mixed model, not all-cold.
        score += model["frequency"] * math.log(probability)
        omission_distance = abs(min(omissions[pos][value], 20) - 9.0) / 9.0
        score -= model["omission"] * omission_distance
        score -= model["crowding"] * max(0.0, heat_z - 0.85) ** 2
    return score, sum(heat_values) / len(heat_values)


def digit_support_score(values: list[int], components: tuple) -> float:
    """Common cross-zone support scale based only on positional frequency."""
    position_counts, totals, _, _, _ = components
    return sum(
        math.log(score_digit(value, position_counts[pos], totals[pos]))
        for pos, value in enumerate(values)
    ) / len(values)


def direct_structure_score(values: list[int]) -> float:
    """Score the small, non-positional structure seen in the 473 review.

    This is deliberately a weak tie-breaker, not a 473 literal or a claim that
    sums and spans predict a fair draw.  It rewards a three-different-digit
    shape, a middle sum, a moderate span, and a 2:1 odd/even split.
    """
    if len(values) != 3:
        return 0.0
    total = sum(values)
    span = max(values) - min(values)
    odd_count = sum(value % 2 for value in values)
    unique = len(set(values))
    sum_fit = max(0.0, 1.0 - abs(total - 14) / 14)
    span_fit = max(0.0, 1.0 - abs(span - 4) / 9)
    parity_fit = 1.0 if odd_count == 2 else 0.5 if odd_count in (1, 3) else 0.0
    unique_fit = 1.0 if unique == 3 else 0.0
    return 0.35 * unique_fit + 0.25 * sum_fit + 0.20 * span_fit + 0.20 * parity_fit


def hybrid_structure_profile(rows: list[dict], window: int = 300) -> dict[str, Counter]:
    """Build the ZIP model's structural signals without hard filters."""
    recent = rows[:window] if rows else rows
    profile = {
        "sum": Counter(),
        "span": Counter(),
        "odd_even": Counter(),
        "pair_sum": Counter(),
        "pair_diff": Counter(),
    }
    for row in recent:
        values = [int(value) for value in row["numbers"]]
        profile["sum"][sum(values)] += 1
        profile["span"][max(values) - min(values)] += 1
        odd = sum(value % 2 for value in values)
        profile["odd_even"][f"{odd}:{len(values) - odd}"] += 1
        if len(values) == 3:
            a, b, c = values
            for value in ((a + b) % 10, (a + c) % 10, (b + c) % 10):
                profile["pair_sum"][value] += 1
            for value in (abs(a - b), abs(a - c), abs(b - c)):
                profile["pair_diff"][value] += 1
    return profile


def hybrid_structure_score(values: list[int], profile: dict[str, Counter]) -> float:
    """ZIP-inspired soft structure score; current positional score stays primary."""
    def log_probability(counter: Counter, value: int, cardinality: int) -> float:
        total = sum(counter.values())
        return math.log((counter[value] + 1.0) / (total + cardinality))

    odd = sum(value % 2 for value in values)
    parity = f"{odd}:{len(values) - odd}"
    score = 0.35 * log_probability(profile["sum"], sum(values), 28)
    score += 0.25 * log_probability(profile["span"], max(values) - min(values), 10)
    score += 0.15 * log_probability(profile["odd_even"], parity, 4)
    if len(values) == 3:
        a, b, c = values
        pair_sums = ((a + b) % 10, (a + c) % 10, (b + c) % 10)
        pair_diffs = (abs(a - b), abs(a - c), abs(b - c))
        score += 0.15 * sum(log_probability(profile["pair_sum"], value, 10) for value in pair_sums) / 3
        score += 0.10 * sum(log_probability(profile["pair_diff"], value, 10) for value in pair_diffs) / 3
    return score


def direct_number_metrics(values: list[int]) -> dict[str, str | int]:
    """Return the readable structure fields shown beside direct picks."""
    total = sum(values)
    span = max(values) - min(values)
    odd = sum(value % 2 for value in values)
    return {
        "sum": total,
        "span": span,
        "odd_even": f"{odd}:{len(values) - odd}",
        "distinct": len(set(values)),
        "shape": digit_shape(values),
    }


def direct_prediction_summary(rows: list[dict]) -> dict[str, object]:
    """Forecast structure first, directly from a rolling historical window."""
    # rows[0] is the just-settled draw. It belongs in review, never in the
    # feature window for the next draw, otherwise the forecast leaks today's
    # answer into tomorrow's structure prior.
    recent = rows[1:301] if len(rows) > 1 else rows
    historical = {
        "sum": Counter(sum(int(value) for value in row["numbers"]) for row in recent),
        "span": Counter(max(map(int, row["numbers"])) - min(map(int, row["numbers"])) for row in recent),
        "odd_even": Counter(
            f"{sum(int(value) % 2 for value in row['numbers'])}:{len(row['numbers']) - sum(int(value) % 2 for value in row['numbers'])}"
            for row in recent
        ),
    }
    result = {}
    for key in ("sum", "span", "odd_even"):
        historical_top = [value for value, _ in historical[key].most_common(3)]
        result[key] = {
            "values": historical_top,
            "range": [min(historical[key]), max(historical[key])],
        }
    return result


def direct_structure_filter(rows: list[dict]) -> dict[str, set[int | str]]:
    """Return the first-stage structure pools used before positional ranking."""
    summary = direct_prediction_summary(rows)
    return {key: set(item["values"]) for key, item in summary.items()}


def direct_structure_match(values: list[int], pools: dict[str, set[int | str]]) -> tuple[bool, int]:
    metrics = direct_number_metrics(values)
    matches = sum([
        metrics["sum"] in pools["sum"],
        metrics["span"] in pools["span"],
        metrics["odd_even"] in pools["odd_even"],
    ])
    return matches == 3, matches


def global_candidate_score(values: list[int], components: tuple, model: dict | None = None) -> float:
    """Score direct numbers by position; no unordered-set or total-sum terms."""
    model = model or DIGIT_MODELS["pl3"]
    position_counts, totals, omissions, _, _ = components
    score = sum(
        math.log(score_digit(value, position_counts[pos], totals[pos]))
            - model["global_omission"] * abs(min(omissions[pos][value], 20) - 9.0) / 9.0
        for pos, value in enumerate(values)
    )
    if len(values) == 3:
        score += model.get("structure_weight", 0.0) * (direct_structure_score(values) - 0.5)
    return score


def digit_confidences(rows: list[dict], digits: int, numbers: list[str], game: str = "pl3") -> list[float]:
    """Map every displayed digit candidate onto the same global percentile scale."""
    components = mixed_digit_components(rows, digits, calibrate_digit_model(game, rows, digits)["parameters"])
    population = sorted(
        digit_support_score([int(value) for value in f"{number:0{digits}d}"], components)
        for number in range(10 ** digits)
    )
    result = []
    for number in numbers:
        support = digit_support_score([int(value) for value in number], components)
        percentile = bisect.bisect_right(population, support) / len(population)
        result.append(round(35 + 43 * percentile, 1))
    return result


def diversified_rank(
    scored: list[tuple[str, float, float]], limit: int, diversity_penalty: float = 0.34
) -> list[tuple[str, float, float]]:
    """Greedily avoid returning near-duplicates while preserving model rank."""
    remaining = sorted(scored, key=lambda item: item[1], reverse=True)
    selected: list[tuple[str, float, float]] = []
    while remaining and len(selected) < limit:
        best = max(
            remaining,
            key=lambda item: item[1] - diversity_penalty * max(
                (sum(a == b for a, b in zip(item[0], chosen[0])) for chosen in selected),
                default=0,
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return sorted(selected, key=lambda item: item[1], reverse=True)


def ensure_position_pool_coverage(
    selected: list[tuple[str, float, float]],
    pools: list[list[int]],
    rescore,
) -> list[tuple[str, float, float]]:
    """Spread every backtested pool digit across the candidate list.

    On a fair draw each ticket's exact-hit chance does not depend on which pool
    digit it uses, but the pool-coverage chance of the five-ticket list does:
    a position collapsed onto one digit covers 1/10 of draws while a position
    using all three pool digits covers 3/10. Mutating redundant digits on the
    weakest tickets is therefore a free expected-coverage gain.
    """
    for position, pool in enumerate(pools):
        for _ in range(6):
            present = {item[0][position] for item in selected}
            missing = [str(digit) for digit in pool if str(digit) not in present]
            if not missing:
                break
            counts = Counter(item[0][position] for item in selected)
            victims = sorted(
                (index for index, item in enumerate(selected) if counts[item[0][position]] > 1),
                key=lambda index: selected[index][1],
            )
            existing = {item[0] for item in selected}
            mutation = None
            for victim in victims:
                text = selected[victim][0]
                candidate = text[:position] + missing[0] + text[position + 1:]
                if candidate not in existing:
                    mutation = (victim, candidate)
                    break
            if mutation is None:
                break
            victim, candidate = mutation
            selected[victim] = (candidate, rescore(candidate), 0.0)
    return sorted(selected, key=lambda item: item[1], reverse=True)


def generate_digit_profile(
    rows: list[dict], digits: int, profile: str, limit: int = 5, game: str = "pl3"
) -> list[tuple[str, float, float]]:
    model = calibrate_digit_model(game, rows, digits)["parameters"]
    components = mixed_digit_components(rows, digits, model)
    scored = []
    for number in range(10 ** digits):
        text = f"{number:0{digits}d}"
        score, heat = mixed_number_score([int(value) for value in text], components, model)
        support = digit_support_score([int(value) for value in text], components)
        if profile == "hot" and heat <= 0.25:
            continue
        if profile == "cold" and heat >= -0.25:
            continue
        if profile == "balanced" and not (-0.25 <= heat <= 0.25):
            continue
        # The main pool uses a rolling-backtest score and stronger diversity;
        # hot and cold remain descriptive strategy zones on the common scale.
        profile_score = (
            global_candidate_score([int(value) for value in text], components, model)
            if profile == "global"
            else score if profile in ("cold", "balanced") else support
        )
        scored.append((text, profile_score, heat))
    diversity_penalty = model["diversity"] if profile == "global" else 0.34
    return diversified_rank(scored, limit, diversity_penalty)


def generate_position_two_digit_predictions(rows: list[dict], game: str) -> list[dict]:
    """Generate two digits independently for each direct position.

    The 473 review is only a capped tie-breaker over independent positional
    pools; it is never treated as a fixed three-digit pick.
    """
    digits = len(rows[0]["numbers"])
    calibration = calibrate_digit_model(game, rows, digits)
    model = calibration["parameters"]
    components = mixed_digit_components(rows, digits, model)
    structure_profile = hybrid_structure_profile(rows) if digits == 3 else None
    pools = []
    for position in range(digits):
        counter = components[0][position]
        total = components[1][position]
        ranked = sorted(range(10), key=lambda value: score_digit(value, counter, total), reverse=True)
        pools.append(ranked[:4])

    scored_numbers = []
    for values in product(*pools):
        score = global_candidate_score(list(values), components, model)
        if structure_profile is not None:
            score += 0.08 * hybrid_structure_score(list(values), structure_profile)
        scored_numbers.append((values, score))

    labels = ["百位", "十位", "个位"] if digits == 3 else [f"第{i + 1}位" for i in range(digits)]
    predictions = []
    for position, label in enumerate(labels):
        position_scores: dict[int, float] = {}
        for values, score in scored_numbers:
            value = values[position]
            position_scores[value] = max(position_scores.get(value, float("-inf")), score)
        ranked = sorted(position_scores, key=position_scores.get, reverse=True)[:2]
        predictions.append({
            "position": label,
            "digits": [str(value) for value in ranked],
            "scores": [round(position_scores[value], 4) for value in ranked],
            "source": "position_ensemble_473_tiebreak",
        })
    return predictions


def generate_pl5_from_pl3(
    pl3_rows: list[dict], pl5_rows: list[dict], profile: str, limit: int = 5
) -> list[tuple[str, float, float]]:
    """Legacy helper retained for compatibility; production PL5 never calls it."""
    prefixes = generate_digit_profile(pl3_rows, 3, profile, limit)
    components = mixed_digit_components(pl5_rows, 5)
    selected: list[tuple[str, float, float]] = []
    used_tails: list[str] = []
    for prefix, _, _ in prefixes:
        choices = []
        for tail_number in range(100):
            tail = f"{tail_number:02d}"
            number = prefix + tail
            values = [int(value) for value in number]
            mixed, heat = mixed_number_score(values, components)
            support = digit_support_score(values, components)
            if profile == "hot" and heat <= 0.25:
                continue
            if profile == "cold" and heat >= -0.25:
                continue
            selection = mixed if profile == "cold" else support
            # Encourage tail coverage without breaking the PL3-prefix contract.
            selection -= 0.10 * max(
                (sum(a == b for a, b in zip(tail, chosen)) for chosen in used_tails),
                default=0,
            )
            choices.append((number, selection, heat, tail, support))
        if not choices:
            raise RuntimeError(f"排列5无法为排列3前缀 {prefix} 生成 {profile} 尾号")
        best = max(choices, key=lambda item: item[1])
        selected.append((best[0], best[4] if profile != "cold" else best[1], best[2]))
        used_tails.append(best[3])
    return selected


def generate_positional_ensemble(game: str, rows: list[dict]) -> tuple[list[dict], list[float]]:
    """Direct-digit output from one hot positional pool.

    ``rows`` is ordered newest-first.  The latest row is already an official
    result when this function is called for the next draw, so it belongs in
    the training sample.  The previous implementation dropped ``rows[0]``;
    that made every refreshed forecast lag one draw behind the verified data.
    Calibration still uses rolling held-out folds, so including the latest
    row here does not leak the future target draw.
    """
    forecast_rows = rows
    digits = len(forecast_rows[0]["numbers"])
    if game in ("pl3", "pl5", "fc3d"):
        model = calibrate_digit_model(game, forecast_rows, digits)["parameters"]
        components = mixed_digit_components(forecast_rows, digits, model)
        position_counts, totals, _, _, _ = components
        # A useful pool should be selective: three independently backtested
        # digits per position, then five combinations from their product.
        pools = [
            sorted(
                range(10),
                key=lambda value: score_digit(value, position_counts[position], totals[position]),
                reverse=True,
            )[:3]
            for position in range(digits)
        ]
        all_values = list(product(*pools))
        scored = []
        structure_profile = hybrid_structure_profile(forecast_rows) if game in ("pl3", "fc3d") else None
        for values in all_values:
            number = "".join(str(value) for value in values)
            score = global_candidate_score(list(values), components, model)
            if structure_profile is not None:
                # ZIP signals are deliberately capped as a tie-breaker.
                score += 0.08 * hybrid_structure_score(list(values), structure_profile)
            scored.append((number, score, 0.0))
        ranked = diversified_rank(scored, 5, model["diversity"])
        ranked = ensure_position_pool_coverage(
            ranked,
            pools,
            lambda text: global_candidate_score([int(value) for value in text], components, model)
            + (0.08 * hybrid_structure_score([int(value) for value in text], structure_profile)
               if structure_profile is not None else 0.0),
        )
        if len({number for number, _, _ in ranked}) != 5:
            raise RuntimeError(f"{game} hot pool did not generate five distinct direct picks")
    else:
        ranked = generate_digit_profile(rows, digits, "global", 5, game)
    # Every displayed line is selected from the same hottest three-digit pool
    # at each position. Report how many positions use the strongest pool digit.
    candidates = []
    for index, (number, _, _) in enumerate(ranked):
        if game in ("pl3", "pl5", "fc3d"):
            leading = sum(int(number[position]) == pools[position][0] for position in range(digits))
            label = f"热门池组合 · {leading}/{digits} 位取最高权重数字"
        else:
            label = "位置综合"
        source = "hot_position_pool" if game in ("pl3", "pl5", "fc3d") else "position_ensemble"
        candidates.append({"number": number, "mix_label": label, "source": source})
    return candidates, [score for _, score, _ in ranked]


def generate_hybrid_cold_profile(game: str, rows: list[dict], limit: int = 5) -> tuple[list[dict], list[float]]:
    """Generate cold direct picks with ZIP structure as a capped tie-breaker."""
    # The latest verified result is valid training data for the next draw.
    # Keep the same sample boundary as the primary positional model.
    forecast_rows = rows
    digits = len(forecast_rows[0]["numbers"])
    model = calibrate_digit_model(game, forecast_rows, digits)["parameters"]
    components = mixed_digit_components(forecast_rows, digits, model)
    structure_profile = hybrid_structure_profile(forecast_rows)
    scored = []
    for number in range(10 ** digits):
        text = f"{number:0{digits}d}"
        values = [int(value) for value in text]
        _, heat = mixed_number_score(values, components, model)
        if heat >= -0.25:
            continue
        score, _ = mixed_number_score(values, components, model)
        score += 0.08 * hybrid_structure_score(values, structure_profile)
        scored.append((text, score, heat))
    ranked = diversified_rank(scored, limit, model["diversity"])
    candidates = [
        {"number": number, "mix_label": "ZIP结构辅助冷门池", "source": "cold_hybrid_pool"}
        for number, _, _ in ranked
    ]
    return candidates, [score for _, score, _ in ranked]


def generate_composite_recommendations(
    game: str, rows: list[dict], pl3_rows: list[dict] | None = None
) -> tuple[list[dict], list[float]]:
    """Compatibility wrapper; every direct-digit game now uses its own rows."""
    del pl3_rows
    return generate_positional_ensemble(game, rows)


def weighted_number_counts(rows: list[dict], start: int, end: int, decay: float = 24) -> Counter[int]:
    counts: Counter[int] = Counter()
    for index, row in enumerate(rows):
        weight = math.exp(-index / decay)
        for value in row["numbers"][start:end]:
            counts[int(value)] += weight
    return counts


def generate_dlt(rows: list[dict], issue: str) -> tuple[list[dict], list[float]]:
    return generate_dlt_v3(rows, issue)
    # Legacy generator retained below for comparison and emergency rollback.
    rng = stable_rng("dlt", issue)
    calibration = calibrate_set_model("dlt", rows)
    window = calibration["selected_window"]
    front_window = calibration.get("region_windows", {}).get("front", window)
    back_window = calibration.get("region_windows", {}).get("back", window)
    front_counts = blended_number_counts(rows, 0, 5, 24, front_window)
    back_counts = blended_number_counts(rows, 5, 7, 24, back_window)
    front_rank_counts = [blended_position_counts(rows, pos, 24, front_window, range(1, 36)) for pos in range(5)]
    back_rank_counts = [blended_position_counts(rows, pos + 5, 24, back_window, range(1, 13)) for pos in range(2)]
    front_total, back_total = sum(front_counts.values()), sum(back_counts.values())
    pool: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}

    for _ in range(8000):
        front = sorted(rng.sample(range(1, 36), 5))
        back = sorted(rng.sample(range(1, 13), 2))
        score = sum(math.log((front_counts[n] + 0.8) / (front_total + 28)) for n in front)
        score += sum(math.log((back_counts[n] + 0.8) / (back_total + 9.6)) for n in back)
        score += sum(math.log((front_rank_counts[pos][n] + 0.08) / (sum(front_rank_counts[pos].values()) + 0.8)) for pos, n in enumerate(front))
        score += sum(math.log((back_rank_counts[pos][n] + 0.08) / (sum(back_rank_counts[pos].values()) + 0.8)) for pos, n in enumerate(back))
        score -= 0.003 * abs(sum(front) - 90)
        score += 0.025 if 1 <= sum(n % 2 for n in front) <= 4 else -0.025
        pool[(tuple(front), tuple(back))] = score

    # Select from a broad high-score pool instead of returning five near-copies.
    # A 60-draw rolling comparison improved both front/back union coverage at
    # this modest penalty while preserving the underlying frequency score.
    remaining = sorted(pool.items(), key=lambda pair: pair[1], reverse=True)[:1000]
    selected: list[tuple[tuple[tuple[int, ...], tuple[int, ...]], float]] = []
    while remaining and len(selected) < 5:
        best = max(
            remaining,
            key=lambda item: item[1]
            - DLT_DIVERSITY_PENALTY
            * max(
                (
                    len(set(item[0][0]) & set(chosen[0][0]))
                    + 2.0 * len(set(item[0][1]) & set(chosen[0][1]))
                    + 6.0 * int(item[0][1] == chosen[0][1])
                    for chosen in selected
                ),
                default=0.0,
            ),
        )
        selected.append(best)
        remaining.remove(best)

    # Coverage repair with lexicographic floors: five distinct back pairs, a
    # back union of six numbers, then a front union of fifteen. On a fair draw
    # every line keeps the same per-line expectation whichever numbers it uses,
    # so widening the union raises the five-line coverage chance for free. A
    # swap must improve the first unmet floor without regressing a met one.
    def line_metrics(picks: list) -> tuple[int, int, int]:
        front_union = set().union(*(set(key[0]) for key, _ in picks))
        back_union = set().union(*(set(key[1]) for key, _ in picks))
        return len(front_union), len(back_union), len({key[1] for key, _ in picks})

    coverage_floors = (
        (lambda front, back, pairs: pairs, 5),
        (lambda front, back, pairs: back, DLT_BACK_UNION_FLOOR),
        (lambda front, back, pairs: front, DLT_FRONT_UNION_FLOOR),
    )
    for metric, floor in coverage_floors:
        for _ in range(10):
            current = line_metrics(selected)
            if metric(*current) >= floor:
                break
            swap = None
            for victim_index in sorted(range(len(selected)), key=lambda index: selected[index][1]):
                others = selected[:victim_index] + selected[victim_index + 1:]
                for item in remaining:
                    trial = line_metrics(others + [item])
                    if metric(*trial) <= metric(*current):
                        continue
                    if any(
                        other_metric(*trial) < min(other_metric(*current), other_floor)
                        for other_metric, other_floor in coverage_floors
                        if (other_metric, other_floor) != (metric, floor)
                    ):
                        continue
                    swap = (victim_index, item)
                    break
                if swap:
                    break
            if swap is None:
                break
            victim_index, item = swap
            remaining.remove(item)
            remaining.append(selected[victim_index])
            selected[victim_index] = item

    # The greedy swap only searches the sampled pool, so it is best-effort.
    # Constructing a replacement line from the strongest numbers the other four
    # lines do not use closes the remaining shortfall - the same fallback SSQ
    # and KL8 already rely on. A replacement is accepted only when it strictly
    # improves the unmet floor and regresses no floor already met.
    def strongest_unused(counts: Counter[int], value_range: range, used: set[int], needed: int) -> list[int]:
        ranked = sorted(value_range, key=lambda value: (counts[value], -value), reverse=True)
        chosen = [value for value in ranked if value not in used][:needed]
        if len(chosen) < needed:
            chosen += [value for value in ranked if value not in chosen][: needed - len(chosen)]
        return chosen

    def constructed_replacement(others: list) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], float]:
        used_front = set().union(*(set(key[0]) for key, _ in others))
        used_back = set().union(*(set(key[1]) for key, _ in others))
        used_pairs = {key[1] for key, _ in others}
        front = tuple(sorted(strongest_unused(front_counts, range(1, 36), used_front, 5)))
        back = tuple(sorted(strongest_unused(back_counts, range(1, 13), used_back, 2)))
        if back in used_pairs:
            ranked_back = sorted(range(1, 13), key=lambda value: back_counts[value], reverse=True)
            back = next(
                (pair for pair in combinations(ranked_back, 2) if tuple(sorted(pair)) not in used_pairs),
                back,
            )
            back = tuple(sorted(back))
        score = sum(math.log((front_counts[n] + 0.8) / (front_total + 28)) for n in front)
        score += sum(math.log((back_counts[n] + 0.8) / (back_total + 9.6)) for n in back)
        return (front, back), score

    for metric, floor in coverage_floors:
        for _ in range(len(selected)):
            current = line_metrics(selected)
            if metric(*current) >= floor:
                break
            improved = False
            for victim in sorted(range(len(selected)), key=lambda index: selected[index][1]):
                others = selected[:victim] + selected[victim + 1:]
                replacement = constructed_replacement(others)
                trial = line_metrics(others + [replacement])
                if metric(*trial) <= metric(*current):
                    continue
                if any(
                    other_metric(*trial) < min(other_metric(*current), other_floor)
                    for other_metric, other_floor in coverage_floors
                    if (other_metric, other_floor) != (metric, floor)
                ):
                    continue
                selected[victim] = replacement
                improved = True
                break
            if not improved:
                break
    selected.sort(key=lambda pair: pair[1], reverse=True)
    candidates = [{"front": list(key[0]), "back": list(key[1])} for key, _ in selected]
    return candidates, [score for _, score in selected]


QXC_FREQUENCY_WEIGHT = 0.78
# An overdue digit is no more likely on a fair wheel, so omission may only
# break near-ties in frequency. The full bonus spans 10 draws, so it can
# reorder two digits whose frequencies differ by less than
# exp(10 * weight / QXC_FREQUENCY_WEIGHT); at this weight that is under 2%.
QXC_OMISSION_WEIGHT = 0.0015
QXC_REPEAT_PENALTY = 0.08
QXC_SUM_WEIGHT = 0.006


def generate_qxc(rows: list[dict]) -> tuple[list[dict], list[float]]:
    """7星彩专用七位置束搜索，避免套用三位数枚举模型。"""
    decay = 27
    calibration = calibrate_set_model("qxc", rows)
    window = calibration["selected_window"]
    position_windows = calibration.get("position_windows", {})
    counters = [blended_position_counts(rows, pos, decay, position_windows.get(str(pos), window)) for pos in range(7)]
    totals = [sum(counter.values()) for counter in counters]
    omissions = [position_omissions(rows, pos) for pos in range(7)]

    def digit_position_score(pos: int, digit: int, previous_digit: int | None) -> float:
        score = QXC_FREQUENCY_WEIGHT * math.log(score_digit(digit, counters[pos], totals[pos]))
        score += QXC_OMISSION_WEIGHT * min(omissions[pos][digit], 10)
        if previous_digit == digit:
            score -= QXC_REPEAT_PENALTY
        return score

    def rescore(number: str) -> float:
        score = sum(
            digit_position_score(pos, int(ch), int(number[pos - 1]) if pos else None)
            for pos, ch in enumerate(number)
        )
        return score - QXC_SUM_WEIGHT * abs(sum(map(int, number)) - 31.5)

    beam: list[tuple[str, float]] = [("", 0.0)]
    for pos in range(7):
        expanded = []
        for prefix, prefix_score in beam:
            previous_digit = int(prefix[-1]) if prefix else None
            for digit in range(10):
                expanded.append((prefix + str(digit), prefix_score + digit_position_score(pos, digit, previous_digit)))
        beam = sorted(expanded, key=lambda item: item[1], reverse=True)[:350]
    scored = [(number, score - QXC_SUM_WEIGHT * abs(sum(map(int, number)) - 31.5), 0.0) for number, score in beam]
    ranked = diversified_rank(scored, 5, 1.15)
    pools = [
        sorted(range(10), key=lambda digit: score_digit(digit, counters[pos], totals[pos]), reverse=True)[:3]
        for pos in range(7)
    ]
    ranked = ensure_position_pool_coverage(ranked, pools, rescore)
    candidates = [
        {"number": number, "mix_label": "七位独立位置模型", "source": "qxc_position"}
        for number, _, _ in ranked
    ]
    return candidates, [score for _, score, _ in ranked]


def weighted_pair_counts(rows: list[dict], end: int, decay: float) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for index, row in enumerate(rows):
        weight = math.exp(-index / decay)
        values = sorted(int(value) for value in row["numbers"][:end])
        for pair in combinations(values, 2):
            counts[pair] += weight
    return counts


def blended_number_counts(rows: list[dict], start: int, end: int, decay: float = 24, max_window: int | None = None) -> Counter[int]:
    """Blend short, medium, and full cached windows for set-based games."""
    result: Counter[int] = Counter()
    active = [(size, share) for size, share in WINDOW_BLEND if max_window is None or size <= max_window]
    total_share = sum(share for _, share in active) or 1.0
    for size, share in active:
        share /= total_share
        counts = weighted_number_counts(rows[: min(size, len(rows))], start, end, decay)
        for number, value in counts.items():
            result[number] += share * value
    return result


def generate_ssq(rows: list[dict], issue: str) -> tuple[list[dict], list[float]]:
    return generate_ssq_v3(rows, issue)
    # Legacy generator retained below for comparison and emergency rollback.
    """双色球专用红蓝分区模型，加入红球共现与三区覆盖。"""
    rng = stable_rng("ssq", issue)
    calibration = calibrate_set_model("ssq", rows)
    window = calibration["selected_window"]
    red_window = calibration.get("region_windows", {}).get("red", window)
    blue_window = calibration.get("region_windows", {}).get("blue", window)
    red_counts = blended_number_counts(rows, 0, 6, 24, red_window)
    blue_counts = blended_number_counts(rows, 6, 7, 24, blue_window)
    pair_counts = weighted_pair_counts(rows, 6, 30)
    red_total, blue_total = sum(red_counts.values()), sum(blue_counts.values())

    def line_score(red: tuple[int, ...], blue: int) -> float:
        score = sum(math.log((red_counts[n] + 0.72) / (red_total + 23.76)) for n in red)
        score += 0.055 * sum(pair_counts[pair] for pair in combinations(red, 2))
        score += math.log((blue_counts[blue] + 0.70) / (blue_total + 11.2))
        zone_counts = [sum(1 for n in red if low <= n <= high) for low, high in ((1, 11), (12, 22), (23, 33))]
        score += 0.10 if all(zone_counts) else -0.30
        score -= 0.0025 * abs(sum(red) - 102)
        return score

    pool: dict[tuple[tuple[int, ...], int], float] = {}
    for _ in range(16000):
        red = tuple(sorted(rng.sample(range(1, 34), 6)))
        blue = rng.randint(1, 16)
        pool[(red, blue)] = line_score(red, blue)
    ranked_all = sorted(pool.items(), key=lambda item: item[1], reverse=True)

    # A single score sort can make all five rows inherit the same hot blue
    # ball and the same hot red cluster. Both failure modes invalidate every
    # line at once, so the composite hedges twice: one strong line per blue
    # band (fifth line = next-best unused blue), and each new line may overlap
    # the growing red union by at most one number, which targets a wider
    # 26-number red union across the five lines. When the sampled pool has no
    # such line, one is constructed from the strongest unused reds.
    def red_union_of(picks: list) -> set[int]:
        return set().union(*(set(key[0]) for key, _ in picks)) if picks else set()

    def constructed_line(band_low: int, band_high: int, union: set[int]) -> tuple[tuple[tuple[int, ...], int], float]:
        unused = [n for n in sorted(range(1, 34), key=lambda n: red_counts[n], reverse=True) if n not in union]
        red = tuple(sorted(unused[:6]))
        band_blues = [b for b in range(band_low, band_high + 1) if b not in used_blues]
        blues = band_blues or [b for b in range(1, 17) if b not in used_blues]
        blue = max(blues, key=lambda b: blue_counts[b])
        return ((red, blue), line_score(red, blue))

    blue_bands = ((1, 4), (5, 8), (9, 12), (13, 16))
    selected = []
    used_blues: set[int] = set()
    for low, high in blue_bands:
        union = red_union_of(selected)
        choice = next(
            (
                item for item in ranked_all
                if low <= item[0][1] <= high
                and item[0][1] not in used_blues
                and len(set(item[0][0]) & union) <= SSQ_RED_OVERLAP_LIMIT
            ),
            None,
        ) or constructed_line(low, high, union)
        selected.append(choice)
        used_blues.add(choice[0][1])
    union = red_union_of(selected)
    fifth = next(
        (
            item for item in ranked_all
            if item[0][1] not in used_blues and len(set(item[0][0]) & union) <= SSQ_RED_OVERLAP_LIMIT
        ),
        None,
    ) or constructed_line(1, 16, union)
    selected.append(fifth)
    ranked = sorted(selected, key=lambda item: item[1], reverse=True)
    candidates = [
        {"red": list(key[0]), "blue": [key[1]], "mix_label": "红蓝分区共现模型", "source": "ssq_zone"}
        for key, _ in ranked
    ]
    return candidates, [score for _, score in ranked]


def generate_kl8(rows: list[dict], pick_count: int = 5) -> tuple[list[dict], list[float]]:
    return generate_kl8_v3(rows, pick_count)
    # Legacy generator retained below for comparison and emergency rollback.
    """快乐8“选五”专用模型：单号强度、遗漏、共现和四区覆盖。"""
    window = calibrate_set_model("kl8", rows)["selected_window"]
    counts = blended_number_counts(rows, 0, 20, 25, window)
    total = sum(counts.values())
    pair_counts = weighted_pair_counts(rows, 20, 32)
    missed = {number: len(rows) for number in range(1, 81)}
    for number in range(1, 81):
        for index, row in enumerate(rows):
            if number in [int(value) for value in row["numbers"]]:
                missed[number] = index
                break

    individual = {
        # Omission is a display signal, not a predictor: on a fair draw an
        # overdue number is no more likely. The bonus spans 12 draws against
        # an unweighted log-frequency term, so at this weight it can only
        # reorder numbers whose frequencies differ by under 2%.
        number: math.log((counts[number] + 0.75) / (total + 60))
        + 0.0016 * min(missed[number], 12)
        for number in range(1, 81)
    }
    if not 5 <= pick_count <= 10:
        raise ValueError(f"快乐8选N仅支持5至10，收到: {pick_count}")
    # Pool sized so five pairwise-disjoint groups always fit (5 × pick_count
    # plus slack): 20 of 80 numbers hit each draw, so the union coverage of the
    # five-group list scales with how many distinct numbers the groups span,
    # while each group's own expected hits are blind to which pool numbers it
    # uses. Disjoint groups are therefore a free coverage gain.
    pool_size = min(60, max(32, 5 * pick_count + 7))
    pool = sorted(individual, key=individual.get, reverse=True)[:pool_size]
    rng = stable_rng(f"kl8-pick{pick_count}", str(len(rows)))

    def group_score(ordered: tuple[int, ...]) -> float:
        zones = {min((number - 1) // 20, 3) for number in ordered}
        odd_count = sum(number % 2 for number in ordered)
        score = sum(individual[number] for number in ordered)
        score += 0.020 * sum(pair_counts[pair] for pair in combinations(ordered, 2))
        score += 0.16 if len(zones) >= min(4, max(3, pick_count // 3)) else -0.20
        score += 0.08 if abs(odd_count - pick_count / 2) <= 1 else -0.08
        score -= 0.0018 * abs(sum(ordered) - pick_count * 40.5)
        return score

    sample_count = 9000 if pick_count >= 8 else 6000
    sampled: dict[tuple[int, ...], float] = {}
    for _ in range(sample_count):
        ordered = tuple(sorted(rng.sample(pool, pick_count)))
        if ordered not in sampled:
            sampled[ordered] = group_score(ordered)

    remaining = sorted(sampled.items(), key=lambda item: item[1], reverse=True)[:2500]
    selected: list[tuple[tuple[int, ...], float]] = [remaining.pop(0)]
    used = set(selected[0][0])
    while len(selected) < 5:
        # Lexicographic choice: least overlap with the numbers already used,
        # then score. The sampled pool rarely holds a zero-overlap group for
        # the later picks, so one is constructed from the strongest unused
        # numbers instead - the pool sizing guarantees enough remain.
        best = min(remaining, key=lambda item: (len(set(item[0]) & used), -item[1])) if remaining else None
        if best is None or set(best[0]) & used:
            unused = [number for number in pool if number not in used]
            if len(unused) >= pick_count:
                ordered = tuple(sorted(unused[:pick_count]))
                best = (ordered, group_score(ordered))
        if best in remaining:
            remaining.remove(best)
        selected.append(best)
        used |= set(best[0])
    selected.sort(key=lambda item: item[1], reverse=True)
    candidates = [
        {"numbers": list(values), "mix_label": f"选{pick_count}独立模型", "source": f"kl8_pick{pick_count}"}
        for values, _ in selected
    ]
    return candidates, [score for _, score in selected]


def generate_kl8_play_types(rows: list[dict], pick_counts: list[int]) -> dict[str, dict]:
    result = {}
    for pick_count in pick_counts:
        candidates, scores = generate_kl8(rows, pick_count)
        result[str(pick_count)] = {
            "name": f"选{pick_count}",
            "description": f"每注{pick_count}个号码，共5注",
            "candidates": (candidates, scores),
        }
    return result


def candidate_text(game: str, candidate: dict) -> str:
    if game == "dlt":
        front = " ".join(f"{value:02d}" for value in candidate["front"])
        back = " ".join(f"{value:02d}" for value in candidate["back"])
        return f"{front} + {back}"
    if game == "ssq":
        red = " ".join(f"{value:02d}" for value in candidate["red"])
        blue = " ".join(f"{value:02d}" for value in candidate["blue"])
        return f"{red} + {blue}"
    if game == "kl8":
        return " ".join(f"{value:02d}" for value in candidate["numbers"])
    return candidate["number"]


def _module_rng(draw_date: str, game: str, scheme: int) -> random.Random:
    seed = hashlib.sha256(f"daily:{draw_date}:{game}:{scheme}:v2.2:independent".encode()).hexdigest()
    return random.Random(int(seed[:16], 16))


def generate_daily_results(draw_date: str, config: dict) -> list[dict]:
    """Create the date-bound, reproducible cultural-number module output.

    The page deliberately exposes only the game name and result.  The result is
    a deterministic number mapping for the requested date, not a probability
    claim or a replacement for the statistical candidates above.
    """
    results = []
    # Keep the date-bound xuanxue module at three schemes per game.
    methods = ("date_hash", "position_map", "neutral_balance")
    for game, cfg in config["games"].items():
        values = []
        schemes = []
        for scheme, method in enumerate(methods, start=1):
            rng = _module_rng(draw_date, game, scheme)
            if game == "dlt":
                front = sorted(rng.sample(range(1, 36), 5))
                back = sorted(rng.sample(range(1, 13), 2))
                value = f"{' '.join(f'{n:02d}' for n in front)} + {' '.join(f'{n:02d}' for n in back)}"
            elif game == "ssq":
                red = sorted(rng.sample(range(1, 34), 6))
                blue = rng.randint(1, 16)
                value = f"{' '.join(f'{n:02d}' for n in red)} + {blue:02d}"
            elif game == "kl8":
                value = " ".join(f"{n:02d}" for n in sorted(rng.sample(range(1, 81), 10)))
            else:
                digits = cfg["digits"]
                value = "".join(str(rng.randrange(10)) for _ in range(digits))
            values.append(value)
            schemes.append({"result": value, "scheme": method})
        combined = "；".join(values)
        results.append({
            "game": game,
            "name": cfg["name"],
            "result": combined,
            "results": schemes,
            # Keep each scheme on its own clipboard line so the xuanxue
            # window remains readable when pasted into notes or chat.
            "copy_text": "\n".join(f"{cfg['name']} {scheme['result']}" for scheme in schemes),
        })
    return results


def digit_shape(values: list[int]) -> str:
    unique = len(set(values))
    if len(values) == 3:
        return {1: "三位相同", 2: "两位相同", 3: "三位不同"}[unique]
    return f"{unique}个不同数字"


def build_review(game: str, rows: list[dict]) -> dict:
    latest = [int(value) for value in rows[0]["numbers"]]
    previous = [int(value) for value in rows[1]["numbers"]]
    if game == "kl8":
        zones = [sum(1 for n in latest if low <= n <= high) for low, high in ((1, 20), (21, 40), (41, 60), (61, 80))]
        overlap = len(set(latest) & set(previous))
        return {
            "title": f"第{rows[0]['issue']}期结构复盘",
            "summary": f"本期开出20个号码，四区比为{' : '.join(map(str, zones))}，奇偶比{sum(n % 2 for n in latest)}:{sum(n % 2 == 0 for n in latest)}，与前一期重号{overlap}个。",
            "metrics": [
                {"label": "号码和值", "value": str(sum(latest))},
                {"label": "四区比", "value": " : ".join(map(str, zones))},
                {"label": "奇偶比", "value": f"{sum(n % 2 for n in latest)}:{sum(n % 2 == 0 for n in latest)}"},
                {"label": "与前期重号", "value": str(overlap)},
            ],
        }
    if game in ("dlt", "ssq"):
        front, back = latest[:5], latest[5:]
        if game == "ssq":
            front, back = latest[:6], latest[6:]
        previous_front = set(previous[:5])
        if game == "ssq":
            previous_front = set(previous[:6])
        return {
            "title": f"第{rows[0]['issue']}期结构复盘",
            "summary": (
                f"{'红球' if game == 'ssq' else '前区'}和值{sum(front)}、跨度{max(front) - min(front)}，"
                f"奇偶比{sum(n % 2 for n in front)}:{sum(n % 2 == 0 for n in front)}；"
                f"{'蓝球' if game == 'ssq' else '后区'}号码{'、'.join(map(str, back))}。与前一期{'红球' if game == 'ssq' else '前区'}重号{len(set(front) & previous_front)}个。"
            ),
            "metrics": [
                {"label": "红球和值" if game == "ssq" else "前区和值", "value": str(sum(front))},
                {"label": "红球跨度" if game == "ssq" else "前区跨度", "value": str(max(front) - min(front))},
                {"label": "红球奇偶" if game == "ssq" else "前区奇偶", "value": f"{sum(n % 2 for n in front)}:{sum(n % 2 == 0 for n in front)}"},
                {"label": "蓝球" if game == "ssq" else "后区和值", "value": "、".join(map(str, back)) if game == "ssq" else str(sum(back))},
            ],
        }
    return {
        "title": f"第{rows[0]['issue']}期结构复盘",
        "summary": (
            f"开奖号{''.join(map(str, latest))}，和值{sum(latest)}、跨度{max(latest) - min(latest)}，"
            f"奇偶比{sum(n % 2 for n in latest)}:{sum(n % 2 == 0 for n in latest)}。"
            + (f"三位数形态为{digit_shape(latest)}。" if game in ("pl3", "fc3d") else f"包含{len(set(latest))}个不同数字。")
        ),
        "metrics": [
            {"label": "和值", "value": str(sum(latest))},
            {"label": "跨度", "value": str(max(latest) - min(latest))},
            {"label": "奇偶", "value": f"{sum(n % 2 for n in latest)}:{sum(n % 2 == 0 for n in latest)}"},
            {"label": "不同数字", "value": str(len(set(latest)))},
        ],
    }


def build_model_review(
    game: str,
    latest: dict,
    prediction: dict,
    current_prediction: dict | None = None,
) -> dict:
    """Explain each positional hit/miss and produce the next-day play form."""
    actual = [int(value) for value in latest["numbers"]]
    candidates = prediction.get("top_candidates", prediction.get("candidates", []))

    def values(candidate: dict) -> list[int]:
        if game == "dlt":
            return [int(value) for value in candidate["front"] + candidate["back"]]
        if game == "ssq":
            return [int(value) for value in candidate["red"] + candidate["blue"]]
        if game == "kl8":
            return [int(value) for value in candidate["numbers"]]
        return [int(value) for value in candidate["number"]]

    def display(candidate: dict) -> str:
        return candidate_text(game, candidate)

    candidate_values = [values(candidate) for candidate in candidates]
    hit_counts = [len(set(item) & set(actual)) for item in candidate_values]
    exact_hits = sum(item == actual for item in candidate_values)
    review = {
        "issue": latest["issue"],
        "actual": display({
            "front": actual[:5], "back": actual[5:],
            "red": actual[:6], "blue": actual[6:],
            "numbers": actual, "number": "".join(latest["numbers"]),
        }) if game in ("dlt", "ssq", "kl8") else "".join(latest["numbers"]),
        "previous_candidates": [display(candidate) for candidate in candidates],
        "exact_hits": exact_hits,
        "best_number_hits": max(hit_counts, default=0),
        "summary": f"第{latest['issue']}期按位复盘：候选池覆盖 {sum(actual[position] in {item[position] for item in candidate_values if position < len(item)} for position in range(len(actual)))} / {len(actual)} 个位置；不按单期结果追涨参数。",
        "lesson": "按位回测继续作为主调参依据；单期命中只用于定位错误，不直接把某个数字升权。",
    }

    if game in ("pl3", "fc3d"):
        labels = ["百位", "十位", "个位"]
        analysis_positions = prediction.get("analysis", {}).get("position_analysis", [])
        diagnostics = []
        for position, target in enumerate(actual):
            pool = [item[position] for item in candidate_values if position < len(item)]
            analysis_item = analysis_positions[position] if position < len(analysis_positions) else {}
            hot_digits = [int(value) for value in analysis_item.get("hot_digits", [])]
            omitted = {int(item["digit"]): int(item["miss"]) for item in analysis_item.get("omitted_digits", [])}
            covered = target in set(pool)
            if covered:
                if target in hot_digits:
                    reason = "命中：该位进入历史位频前三，综合候选池保留了它。"
                else:
                    reason = "命中：该位不在位频前三，但多信号综合分仍把它纳入候选池。"
            elif target in hot_digits:
                reason = "未命中：该位虽在位频前三，但五组组合时被其他位置组合分和去重约束舍弃。"
            elif target in omitted and omitted[target] >= 8:
                reason = f"未命中：该位遗漏 {omitted[target]} 期，当前模型对长期遗漏只做有限补偿，未强行追冷。"
            else:
                reason = "未命中：该位的频率、遗漏和组合分均未达到五组入选阈值。"
            diagnostics.append({
                "position": labels[position],
                "actual_digit": str(target),
                "candidate_digits": sorted({str(value) for value in pool}),
                "candidate_hit_count": sum(value == target for value in pool),
                "pool_hit": covered,
                "reason": reason,
            })
        review["position_diagnostics"] = diagnostics
        review["model_adjustments"] = [
            "排列3/福彩3D改为逐位选择衰减周期和历史窗口，避免三个位数共用一个最优参数。",
            "保留位频、遗漏和组合分的约束；不因一期开出号码直接追热或追冷。",
        ]
        review["position_pool_coverage"] = sum(item["pool_hit"] for item in diagnostics)
        review["position_count"] = len(diagnostics)

        next_candidates = (current_prediction or {}).get("top_candidates", [])
        advice = build_next_day_advice(game, next_candidates)
        review["next_day_advice"] = advice
        review["next_day_advice_text"] = "；".join(item["suggestion"] for item in advice) or "暂无可用候选"
    elif game in ("dlt", "qxc", "ssq"):
        if game == "dlt":
            regions = [("前区", actual[:5], [candidate["front"] for candidate in candidates]), ("后区", actual[5:], [candidate["back"] for candidate in candidates])]
        elif game == "ssq":
            regions = [("红球", actual[:6], [candidate["red"] for candidate in candidates]), ("蓝球", actual[6:], [candidate["blue"] for candidate in candidates])]
        else:
            regions = [("七星彩位数", actual, [candidate["number"] for candidate in candidates])]
        diagnostics = []
        region_diagnostics = []
        for region_name, region_actual, region_candidates in regions:
            region_union = set(value for group in region_candidates for value in group)
            region_hits = sorted(set(region_actual) & region_union)
            region_misses = sorted(set(region_actual) - region_union)
            region_diagnostics.append({
                "region": region_name,
                "coverage": f"{len(region_hits)} / {len(region_actual)}",
                "hit_numbers": [f"{value:02d}" for value in region_hits],
                "missed_numbers": [f"{value:02d}" for value in region_misses],
            })
            for position, target in enumerate(region_actual):
                pool = [int(group[position]) for group in region_candidates if position < len(group)]
                covered = target in set(pool)
                if covered:
                    reason = f"命中：{region_name}第{position + 1}位的历史排序/分区候选池覆盖该号码。"
                elif target in region_union:
                    reason = f"未命中位置：号码进入{region_name}联合池，但没有落在第{position + 1}位的排序位置。"
                else:
                    reason = f"未命中：号码未进入{region_name}联合池，区域频率、共现和结构分未达到五组阈值。"
                diagnostics.append({
                    "position": f"{region_name}第{position + 1}位",
                    "actual_digit": f"{target:02d}",
                    "candidate_digits": sorted({f"{value:02d}" for value in pool}),
                    "candidate_hit_count": sum(value == target for value in pool),
                    "pool_hit": covered,
                    "reason": reason,
                })
        review["position_diagnostics"] = diagnostics
        review["region_diagnostics"] = region_diagnostics
        review["position_pool_coverage"] = sum(item["pool_hit"] for item in diagnostics)
        review["position_count"] = len(diagnostics)
        review["error_attribution"] = "；".join(
            f"{item['region']}覆盖 {item['coverage']}"
            for item in region_diagnostics
        ) + "。未命中项按区域频率、共现、分区/排序位置和五组多样性约束归因，不把单期结果直接变成追涨规则。"
        review["model_adjustments"] = [
            "按区域滚动回测选择历史窗口：大乐透前区/后区、双色球红球/蓝球分开校准。",
            "七星彩继续按七个位置建模，复盘同时区分号码命中与位置命中，避免把无序命中误判为有效预测。",
        ]
    elif game == "kl8":
        union = set().union(*(set(item) for item in candidate_values)) if candidate_values else set()
        hit_numbers = sorted(union & set(actual))
        missed_numbers = sorted(set(actual) - union)
        review["union_number_hits"] = len(hit_numbers)
        review["number_pool_coverage"] = f"{len(hit_numbers)} / {len(actual)}"
        review["hit_numbers"] = [f"{value:02d}" for value in hit_numbers]
        review["missed_numbers"] = [f"{value:02d}" for value in missed_numbers]
        review["error_attribution"] = (
            f"未覆盖的 {len(missed_numbers)} 个号码没有进入五组联合池，主要是单号强度、四区覆盖、共现分和组间重合惩罚共同筛选的结果；"
            "不把一次遗漏直接解释成下期必补。"
        )
        review["model_adjustments"] = [
            "快乐8继续单独使用选五模型，并以滚动联合池覆盖率选择历史窗口。",
            "保持五组之间的重合约束，优先修正长期覆盖不足的区间，不追逐单期遗漏号码。",
        ]
    else:
        review["position_candidate_hits"] = [
            sum(position < len(item) and item[position] == target for item in candidate_values)
            for position, target in enumerate(actual)
        ]
        review["position_pool_coverage"] = sum(
            any(position < len(item) and item[position] == target for item in candidate_values)
            for position, target in enumerate(actual)
        )
        review["position_count"] = len(actual)
    return review


def purchase_suggestion(game: str, candidate: dict) -> str | None:
    """Return the play form beside a direct-digit composite recommendation."""
    if game not in ("pl3", "fc3d"):
        return None
    number = candidate["number"]
    digits = list(number)
    if len(set(digits)) == 2:
        repeated = next(value for value in set(digits) if digits.count(value) == 2)
        single = next(value for value in set(digits) if digits.count(value) == 1)
        permutations = sorted({
            repeated + repeated + single,
            repeated + single + repeated,
            single + repeated + repeated,
        })
        return f"组选三参考：{number}（{'/'.join(permutations)}）"
    if len(set(digits)) == 3:
        return f"直选参考：{number}"
    return f"直选参考：{number}（豹子形态）"


def build_next_day_advice(game: str, candidates: list[dict]) -> list[dict]:
    """Build review advice from the exact candidates shown in the composite list."""
    if game not in ("pl3", "fc3d"):
        return []
    advice = []
    for candidate in candidates[:5]:
        number = candidate["number"]
        unique = len(set(number))
        if unique == 2:
            shape = "组选三"
        elif unique == 3:
            shape = "组六形态，按直选参考"
        else:
            shape = "豹子形态，按直选参考"
        advice.append({
            "number": number,
            "shape": shape,
            "suggestion": purchase_suggestion(game, candidate),
        })
    return advice


def omission(rows: list[dict], start: int, end: int, values: range) -> list[tuple[int, int]]:
    result = []
    for target in values:
        missed = len(rows)
        for index, row in enumerate(rows):
            if target in [int(value) for value in row["numbers"][start:end]]:
                missed = index
                break
        result.append((target, missed))
    return sorted(result, key=lambda pair: pair[1], reverse=True)


def build_analysis(game: str, rows: list[dict]) -> dict:
    sample = min(2000, len(rows))
    set_calibration = calibrate_set_model(game, rows) if game in ("dlt", "qxc", "ssq", "kl8") else None
    if game == "dlt":
        window = set_calibration["selected_window"]
        front = blended_number_counts(rows[:sample], 0, 5, 24, window)
        back = blended_number_counts(rows[:sample], 5, 7, 24, window)
        rank_analysis = []
        for pos in range(5):
            counter = blended_position_counts(rows[:sample], pos, 24, window, range(1, 36))
            rank_analysis.append({"position": f"前区第{pos + 1}位", "hot_numbers": [f"{n:02d}" for n, _ in counter.most_common(3)]})
        for pos in range(2):
            counter = blended_position_counts(rows[:sample], pos + 5, 24, window, range(1, 13))
            rank_analysis.append({"position": f"后区第{pos + 1}位", "hot_numbers": [f"{n:02d}" for n, _ in counter.most_common(3)]})
        hot_front = [f"{n:02d}" for n, _ in front.most_common(5)]
        hot_back = [f"{n:02d}" for n, _ in back.most_common(3)]
        omitted = [f"{n:02d}（{miss}期）" for n, miss in omission(rows[:sample], 0, 5, range(1, 36))[:5]]
        return {
            "sample": sample,
            "summary": f"最近{sample}期采用指数衰减频率，前后区分别统计；当前前区相对活跃号为{'、'.join(hot_front)}，后区为{'、'.join(hot_back)}。",
            "position_analysis": rank_analysis,
            "selected_window": window,
            "selected_region_windows": set_calibration.get("region_windows", {}),
            "region_backtest": set_calibration.get("region_backtest", {}),
            "backtest": set_calibration["backtest"],
            "signals": [
                {"label": "前区相对活跃", "value": " · ".join(hot_front)},
                {"label": "后区相对活跃", "value": " · ".join(hot_back)},
                {"label": "前区较长遗漏", "value": "、".join(omitted)},
            ],
            "method": ["最近100期指数衰减频率", "前区与后区独立建模", "和值、奇偶与跨度温和约束"],
        }

    if game == "ssq":
        window = set_calibration["selected_window"]
        red = blended_number_counts(rows[:sample], 0, 6, 24, window)
        blue = blended_number_counts(rows[:sample], 6, 7, 24, window)
        hot_red = [f"{n:02d}" for n, _ in red.most_common(6)]
        hot_blue = [f"{n:02d}" for n, _ in blue.most_common(3)]
        return {
            "sample": sample,
            "model_name": "双色球红蓝分区共现模型",
            "selected_window": window,
            "selected_region_windows": set_calibration.get("region_windows", {}),
            "region_backtest": set_calibration.get("region_backtest", {}),
            "backtest": set_calibration["backtest"],
            "summary": f"最近{sample}期将6个红球、1个蓝球完全分开建模；红球同时计算号码频率、两两共现和三区覆盖，蓝球只使用自己的1–16历史序列。",
            "signals": [
                {"label": "红球相对活跃", "value": " · ".join(hot_red)},
                {"label": "蓝球相对活跃", "value": " · ".join(hot_blue)},
                {"label": "红球结构", "value": "1–11 / 12–22 / 23–33三区覆盖"},
            ],
            "method": ["红球与蓝球独立", "红球两两共现", "三区覆盖与和值温和约束"],
        }

    if game == "kl8":
        window = set_calibration["selected_window"]
        counts = blended_number_counts(rows[:sample], 0, 20, 25, window)
        hot = [f"{number:02d}" for number, _ in counts.most_common(10)]
        omitted = [f"{number:02d}（{miss}期）" for number, miss in omission(rows[:sample], 0, 20, range(1, 81))[:6]]
        return {
            "sample": sample,
            "model_name": "快乐8选五独立模型",
            "selected_window": window,
            "backtest": set_calibration["backtest"],
            "summary": f"最近{sample}期以1–80单号衰减频率、遗漏封顶和两两共现为主，并约束每组号码覆盖至少三个区间。每个选N玩法各自生成5组互不重叠的号码（联合覆盖=5×N个）；顶部清单只展示各玩法得分最高的一组，跨玩法的组之间可能重叠。",
            "signals": [
                {"label": "相对活跃号码", "value": " · ".join(hot)},
                {"label": "较长遗漏", "value": "、".join(omitted)},
                {"label": "组合目标", "value": "选五 · 每组5个号码 · 共5组"},
            ],
            "method": ["25期单号衰减", "32期号码共现", "四区与奇偶温和约束", "五组候选分散"],
        }

    all_counts = Counter(int(value) for row in rows[:sample] for value in row["numbers"])
    hot = [str(n) for n, _ in all_counts.most_common(5)]
    omitted = [f"{n}（{miss}期）" for n, miss in omission(rows[:sample], 0, len(rows[0]["numbers"]), range(10))[:3]]
    position_hot = []
    position_analysis = []
    occurrence_window = min(100, sample)
    labels = {3: ["百位", "十位", "个位"], 5: ["万位", "千位", "百位", "十位", "个位"], 7: ["第一位", "第二位", "第三位", "第四位", "第五位", "第六位", "第七位"]}[len(rows[0]["numbers"])]
    calibration = calibrate_digit_model(game, rows, len(rows[0]["numbers"])) if game in ("pl3", "pl5", "fc3d") else set_calibration
    model = calibration["parameters"] if game in ("pl3", "pl5", "fc3d") else {"decay": 27}
    for pos in range(len(rows[0]["numbers"])):
        counter = weighted_counts(rows[:sample], pos, model["decay"])
        ranked = [str(value) for value, _ in counter.most_common(3)]
        occurrence_counts = Counter(int(row["numbers"][pos]) for row in rows[:occurrence_window])
        cold_ranked = [str(value) for value, _ in sorted(counter.items(), key=lambda item: (item[1], item[0]))[:3]]
        missed = position_omissions(rows[:sample], pos)
        longest = sorted(missed.items(), key=lambda item: item[1], reverse=True)[:2]
        position_hot.append(ranked[0])
        position_analysis.append({
            "position": labels[pos],
            "hot_digits": ranked,
            "cold_digits": cold_ranked,
            "hot_focus_digits": ranked[:2],
            "cold_focus_digits": cold_ranked[:2],
            "hot_occurrences": [{"digit": value, "count": occurrence_counts[int(value)]} for value in ranked],
            "cold_occurrences": [{"digit": value, "count": occurrence_counts[int(value)]} for value in cold_ranked],
            "omitted_digits": [{"digit": str(value), "miss": miss} for value, miss in longest],
        })
    structure_note = "排列5使用自身五个位置的独立校准与三数字候选池；" if game == "pl5" else ""
    model_name = {"pl3": "排列3独立三位置模型", "pl5": "排列5独立五位置模型", "fc3d": "福彩3D独立三位置模型", "qxc": "7星彩七位置束搜索模型"}[game]
    return {
        "sample": sample,
        "model_name": model_name,
        "summary": f"{structure_note}最近{sample}期按每一个位置分别统计，绝不把号码只当作无序集合。各位置当前热门参考为{' · '.join(position_hot)}。",
        "signals": [
            {"label": "综合活跃数字", "value": " · ".join(hot)},
            {"label": "各位置最高权重", "value": " · ".join(position_hot)},
            {"label": "较长遗漏", "value": "、".join(omitted)},
        ],
        "position_analysis": position_analysis,
        "position_two_digit_predictions": (
            generate_position_two_digit_predictions(rows, game)
            if game in ("pl3", "fc3d") else []
        ),
        "backtest": calibration["backtest"] if calibration else None,
        "selected_decay": calibration.get("selected_decay") if game in ("pl3", "pl5", "fc3d") else None,
        "selected_window": calibration.get("selected_window") if calibration else None,
        "selected_position_parameters": calibration.get("parameters", {}).get("position_parameters", []) if game in ("pl3", "pl5", "fc3d") else [],
        "selected_position_windows": calibration.get("position_windows", {}) if game == "qxc" else {},
        "position_backtest": calibration.get("position_backtest", []) if game in ("pl3", "pl5", "fc3d", "qxc") else [],
        "method": (["七位独立频率与遗漏", "逐位束搜索", "相邻重号与和值温和约束"] if game == "qxc" else ["逐位置频率与遗漏", f"{model['decay']}期衰减参数", "ZIP和值跨度奇偶与两码和差低权重辅助", "5至8注综合清单与候选分散"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成体彩数据看板")
    parser.add_argument("--games", default="dlt,pl3,pl5,fc3d,qxc,ssq,kl8", help="只刷新指定玩法，逗号分隔")
    parser.add_argument("--today", action="store_true", help="只刷新北京时间今天安排开奖的彩种")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source_data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    model_reviews = json.loads(REVIEWS_PATH.read_text(encoding="utf-8")) if REVIEWS_PATH.exists() else {}
    selected = [
        game for game, cfg in config["games"].items()
        if datetime.now(TZ).weekday() in cfg.get("draw_weekdays", list(range(7)))
    ] if args.today else [game.strip() for game in args.games.split(",") if game.strip()]
    invalid = [game for game in selected if game not in config["games"]]
    if invalid:
        raise SystemExit(f"未知玩法: {', '.join(invalid)}")
    now = datetime.now(TZ)
    try:
        previous_output = json.loads(OUTPUT_PATH.read_text(encoding="utf-8")) if OUTPUT_PATH.exists() else {}
    except json.JSONDecodeError:
        # A generated file can contain merge markers during a rebase; rebuild it
        # entirely from the verified source data instead of preserving fragments.
        previous_output = {}
    previous_games = previous_output.get("games", {})
    output = {
        "generated_at": now.isoformat(timespec="seconds"),
        "daily_results_date": now.date().isoformat(),
        "daily_model_version": "v2.3-xuanxue-date-game-schemes",
        "daily_results": generate_daily_results(now.date().isoformat(), config),
        "source_status": source_data.get("source_status", "unknown"),
        "verification": source_data.get("verification", {}),
        "draw_history": {game: source_data.get("draws", {}).get(game, [])[:400] for game in config["games"]},
        "disclaimer": "以上仅为公开信息整理后的娱乐分析，不构成任何购彩建议，请理性参考。模型相对评分仅表示本页综合候选之间的排序，不是真实中奖概率。",
        "games": dict(previous_output.get("games", {})),
        "sources": source_data.get("sources", []),
    }

    for game in selected:
        cfg = config["games"][game]
        hour, minute = map(int, cfg.get("draw_time", config["draw_time"]).split(":"))
        rows = source_data["draws"][game]
        latest = rows[0]
        target_issue = str(int(latest["issue"]) + 1)
        latest_draw_at = datetime.combine(datetime.fromisoformat(latest["draw_date"]).date(), time(hour, minute), tzinfo=TZ)
        # Once the official result exists, the target must be scheduled after it
        # even if this script is run on a machine whose wall clock is earlier.
        draw_at = next_draw(max(now, latest_draw_at), cfg["draw_weekdays"], time(hour, minute))
        if game == "dlt":
            candidates, scores = generate_dlt(rows, target_issue)
        elif game == "qxc":
            candidates, scores = generate_qxc(rows)
        elif game == "ssq":
            candidates, scores = generate_ssq(rows, target_issue)
        elif game == "kl8":
            play_types = generate_kl8_play_types(rows, cfg.get("pick_counts", [5, 6, 7, 8, 9, 10]))
            # Raw scores are not comparable across pick counts (more numbers,
            # more log terms), and rescaling each play's own list always hands
            # its top group the same ceiling - that ranking was decided by
            # dict order and silently never showed 选10. Rank plays by how far
            # the top group stands above its own play's five-group mean.
            ranked_plays = []
            for key, play in play_types.items():
                play_candidates, play_scores = play["candidates"]
                margin = play_scores[0] - sum(play_scores) / len(play_scores)
                ranked_plays.append((play_candidates[0] | {"pick_count": int(key), "play_name": play["name"]}, margin))
            ranked_plays.sort(key=lambda item: (item[1], -item[0]["pick_count"]), reverse=True)
            candidates = [candidate for candidate, _ in ranked_plays[:5]]
            scores = [margin for _, margin in ranked_plays[:5]]
        else:
            candidates, scores = generate_positional_ensemble(game, rows)

        hot_candidates, hot_scores = candidates, scores
        cold_candidates, cold_scores = [], []
        if game in ("pl3", "fc3d"):
            forecast_rows = rows[1:] if len(rows) > 1 else rows
            cold_candidates, cold_scores = generate_hybrid_cold_profile(game, rows, 5)

        # Every model exposes the same five-line contract. Direct-digit lines
        # already carry their hot positional-pool source from the generator.
        for candidate in hot_candidates + cold_candidates:
            if "source" not in candidate:
                candidate["source"] = "model_primary"

        # Main-list scores are relative to the backtested ranking model. Strategy
        # zones below keep their separate common hot/cold support scale.
        def enrich_group(raw_candidates: list[dict], raw_scores: list[float], group: str) -> list[dict]:
            confidences = relative_confidences(raw_scores)
            enriched_group = []
            for rank, (candidate, confidence) in enumerate(zip(raw_candidates, confidences), start=1):
                text_value = candidate_text(game, candidate)
                play_prefix = f"{candidate['play_name']} " if game == "kl8" else ""
                copy_text = f"{cfg['name']} {play_prefix}{text_value}"
                suggestion = purchase_suggestion(game, candidate)
                enriched_group.append({
                    **candidate,
                    "group": group,
                    "rank": rank,
                    "confidence": confidence,
                    "copy_text": copy_text,
                    **({"prediction_metrics": direct_number_metrics([int(value) for value in candidate["number"]])}
                       if game in ("pl3", "fc3d") else {}),
                    **({"purchase_suggestion": suggestion} if suggestion else {}),
                })
            return enriched_group

        enriched = enrich_group(hot_candidates, hot_scores, "hot")
        enriched_cold = enrich_group(cold_candidates, cold_scores, "cold")

        output["games"][game] = {
            "name": cfg["name"],
            "sector": GAME_SECTORS.get(game, ("ticai", "体彩"))[0],
            "sector_name": GAME_SECTORS.get(game, ("ticai", "体彩"))[1],
            "model_version": (
                "v3.4-v3-ensemble-coverage-validated"
                if game in ("dlt", "ssq", "kl8")
                else "v4.1-latest-inclusive-hybrid-positional"
            ),
            "model_reference": (
                f"src/vendor_models_v3/{game}_model_v3.py"
                if game in ("dlt", "ssq", "kl8") else None
            ),
            "history_count": len(rows),
            "model_scope": "前区/后区排序位独立" if game == "dlt" else "每一位独立评分" if game in ("pl3", "pl5", "fc3d", "qxc") else "玩法专用结构模型",
            "generated_at": now.isoformat(timespec="seconds"),
            "latest_issue": latest["issue"],
            "latest_draw_date": latest["draw_date"],
            "latest_numbers": latest["numbers"],
            "target_issue": target_issue,
            "next_draw_at": draw_at.isoformat(timespec="minutes"),
            "next_draw_display": f"{draw_at:%Y年%m月%d日 %H:%M}（北京时间）",
            "schedule_note": {
                "dlt": "每周一、三、六开奖",
                "qxc": "每周二、五、日开奖",
                "ssq": "每周二、四、日开奖",
                "kl8": "每日21:30开奖（休市日除外）",
            }.get(game, "每日开奖（休市日除外）"),
            "candidates": enriched,
            "top_candidates": enriched,
            **({"hot_candidates": enriched, "cold_candidates": enriched_cold} if game in ("pl3", "fc3d") else {}),
            **({"prediction_summary": direct_prediction_summary(rows)} if game in ("pl3", "fc3d") else {}),
            "review": build_review(game, rows),
            "analysis": build_analysis(game, rows),
            "model_review": model_reviews.get(game),
        }
        previous_game = previous_games.get(game, {})
        if previous_game.get("target_issue") == latest["issue"]:
            model_reviews[game] = build_model_review(game, latest, previous_game, {"top_candidates": enriched})
            output["games"][game]["model_review"] = model_reviews[game]
        elif (
            game in ("dlt", "pl3", "pl5", "fc3d", "qxc", "ssq", "kl8")
            and model_reviews.get(game, {}).get("issue") == latest["issue"]
            and not model_reviews[game].get("position_diagnostics")
        ):
            # Older generated dashboards kept the prior candidate text in the
            # review file but not the full candidate objects. Rehydrate that
            # small direct-digit snapshot so the detailed review can be added
            # without pretending the current pool predicted the past draw.
            saved_review = model_reviews[game]
            if game == "kl8":
                saved_candidates = [
                    {"numbers": [int(value) for value in text.split()]}
                    for text in saved_review.get("previous_candidates", [])
                ]
            elif game == "dlt":
                saved_candidates = []
                for text in saved_review.get("previous_candidates", []):
                    front_text, back_text = text.split(" + ")
                    saved_candidates.append({
                        "front": [int(value) for value in front_text.split()],
                        "back": [int(value) for value in back_text.split()],
                    })
            elif game == "ssq":
                saved_candidates = []
                for text in saved_review.get("previous_candidates", []):
                    red_text, blue_text = text.split(" + ")
                    saved_candidates.append({
                        "red": [int(value) for value in red_text.split()],
                        "blue": [int(value) for value in blue_text.split()],
                    })
            else:
                saved_candidates = [{"number": value} for value in saved_review.get("previous_candidates", [])]
            saved_prediction = {
                "top_candidates": saved_candidates,
                "analysis": output["games"][game]["analysis"],
            }
            model_reviews[game] = build_model_review(game, latest, saved_prediction, {"top_candidates": enriched})
            output["games"][game]["model_review"] = model_reviews[game]
        if model_reviews.get(game, {}).get("issue") != latest["issue"]:
            # Never present an older review as if it explained today's draw.
            # A missing snapshot is a data-pipeline issue, not a model hit or
            # miss, and must remain explicit for the next audit.
            actual_text = " ".join(latest["numbers"])
            if game in ("pl3", "pl5", "fc3d", "qxc"):
                actual_text = "".join(latest["numbers"])
            elif game in ("dlt", "ssq"):
                split_at = 5 if game == "dlt" else 6
                actual_text = (
                    " ".join(latest["numbers"][:split_at])
                    + " + "
                    + " ".join(latest["numbers"][split_at:])
                )
            model_reviews[game] = {
                "issue": latest["issue"],
                "actual": actual_text,
                "previous_candidates": [],
                "exact_hits": None,
                "best_number_hits": None,
                "summary": f"第{latest['issue']}期没有可核对的上一版预测快照，未将旧期结果冒充本期复盘。",
                "lesson": "先修复预测快照与结果期号的同步，再把命中/未命中用于模型反思；没有快照时不做伪复盘。",
                "model_adjustments": [
                    "生成器必须在结果期号与上一版 target_issue 对齐时才计算命中率。",
                    "下一期模型使用全部已核实历史，参数仍由滚动留出回测选择。",
                ],
            }
            output["games"][game]["model_review"] = model_reviews[game]
        # The historical review may already be complete while today's
        # candidates are regenerated. Keep only the forward-looking advice in
        # lockstep with the composite recommendation shown on this page.
        if game in ("pl3", "fc3d") and output["games"][game].get("model_review"):
            advice = build_next_day_advice(game, enriched)
            output["games"][game]["model_review"]["next_day_advice"] = advice
            output["games"][game]["model_review"]["next_day_advice_text"] = (
                "；".join(item["suggestion"] for item in advice) or "暂无可用候选"
            )
            model_reviews[game]["next_day_advice"] = advice
            model_reviews[game]["next_day_advice_text"] = output["games"][game]["model_review"]["next_day_advice_text"]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    REVIEWS_PATH.write_text(json.dumps(model_reviews, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 已生成 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
