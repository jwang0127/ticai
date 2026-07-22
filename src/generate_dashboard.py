from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
from collections import Counter
from itertools import combinations
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "games.json"
DATA_PATH = ROOT / "data" / "processed" / "draws.json"
OUTPUT_PATH = ROOT / "docs" / "assets" / "data" / "dashboard.json"
REVIEWS_PATH = ROOT / "data" / "processed" / "model_reviews.json"
try:
    TZ = ZoneInfo("Asia/Shanghai")
except ZoneInfoNotFoundError:
    TZ = timezone(timedelta(hours=8))
RECENCY_DECAY = 18
GLOBAL_OMISSION_WEIGHT = 0.06
GLOBAL_UNIQUE_WEIGHT = 0.06
# A 60-draw rolling comparison across PL3, PL5, and FC3D showed that 1.6
# modestly improves per-position coverage without changing the core signals.
GLOBAL_DIVERSITY_PENALTY = 1.6
DLT_DIVERSITY_PENALTY = 0.06
WINDOW_BLEND = ((50, 0.15), (100, 0.20), (300, 0.20), (500, 0.20), (1000, 0.15), (1500, 0.10))

# 排列3/排列5共享位置模型；福彩3D使用独立参数。7星彩和双色球另有专用生成器。
DIGIT_MODELS = {
    # Selected from rolling top-3 positional coverage on each game's own
    # history. PL3 and FC3D are deliberately calibrated separately.
    "pl3": {"decay": 13, "frequency": 0.58, "omission": 0.050, "crowding": 0.18,
            "sum": 0.0, "unique": 0.0, "global_omission": 0.045,
            "global_unique": 0.0, "diversity": 1.6},
    "pl5": {"decay": 18, "frequency": 0.55, "omission": 0.045, "crowding": 0.16,
            "sum": 0.0, "unique": 0.0, "global_omission": 0.040,
            "global_unique": 0.0, "diversity": 1.45},
    "fc3d": {"decay": 18, "frequency": 0.66, "omission": 0.045, "crowding": 0.30,
             "sum": 0.0, "unique": 0.0, "global_omission": 0.035,
             "global_unique": 0.0, "diversity": 1.85},
}

BACKTEST_DECAYS = (8, 13, 18, 24, 30, 45, 60)
BACKTEST_WINDOWS = (100, 300, 500, 1000, 1500)


def rolling_digit_backtest(rows: list[dict], digits: int, decay: int, window_size: int, folds: int = 120) -> dict:
    """Evaluate positional top-three coverage without using the held-out draw."""
    limit = min(folds, max(0, len(rows) - window_size))
    hits = 0
    total = 0
    for index in range(1, limit + 1):
        train = rows[index : index + window_size]
        actual = rows[index - 1]["numbers"]
        for position in range(digits):
            counts: Counter[int] = Counter()
            for age, row in enumerate(train):
                counts[int(row["numbers"][position])] += math.exp(-age / decay)
            hits += int(int(actual[position]) in {value for value, _ in counts.most_common(3)})
            total += 1
    return {"folds": limit, "positional_top3_hit_rate": round(hits / total, 4) if total else 0.0}


def calibrate_digit_model(game: str, rows: list[dict], digits: int) -> dict:
    """Return the isolated model plus rolling-backtest evidence for the game."""
    base = dict(digit_model(game))
    windows = [window for window in BACKTEST_WINDOWS if window < len(rows)] or [max(1, len(rows) - 1)]
    scores = {
        f"{decay}@{window}": rolling_digit_backtest(rows, digits, decay, window)
        for decay in BACKTEST_DECAYS for window in windows
    }
    chosen_decay, chosen_window = max(
        ((decay, window) for decay in BACKTEST_DECAYS for window in windows),
        key=lambda pair: (
            scores[f"{pair[0]}@{pair[1]}"]["positional_top3_hit_rate"],
            int(pair[1] >= 300),
            -pair[1],
            -abs(pair[0] - base["decay"]),
        ),
    )
    base["decay"] = chosen_decay
    base["window_size"] = chosen_window
    return {"parameters": base, "backtest": scores, "selected_decay": chosen_decay, "selected_window": chosen_window}


def rolling_pool_backtest(game: str, rows: list[dict], window_size: int, folds: int = 120) -> dict:
    """Evaluate position/set coverage against a simple frequency baseline."""
    limit = min(folds, max(0, len(rows) - window_size))
    hits = total = 0
    for index in range(1, limit + 1):
        train = rows[index:index + window_size]
        actual = rows[index - 1]["numbers"]
        if game == "dlt":
            specs = [(pos, 10) for pos in range(5)] + [(pos, 4) for pos in range(5, 7)]
        elif game == "ssq":
            specs = [(0, 18), (6, 5)]
        elif game == "kl8":
            specs = [(0, 40)]
        else:
            specs = [(pos, 3) for pos in range(len(actual))]
        for pos, top_k in specs:
            counts = Counter()
            if game in ("ssq", "kl8") and pos in (0, 6):
                for row in train:
                    values = row["numbers"][:6] if game == "ssq" and pos == 0 else row["numbers"][-1:] if game == "ssq" else row["numbers"]
                    for value in values:
                        counts[int(value)] += 1
                target_values = [int(value) for value in (actual[:6] if game == "ssq" and pos == 0 else actual[-1:] if game == "ssq" else actual)]
                hits += len(set(target_values) & {n for n, _ in counts.most_common(top_k)})
                total += len(target_values)
            else:
                for row in train:
                    counts[int(row["numbers"][pos])] += 1
                hits += int(int(actual[pos]) in {n for n, _ in counts.most_common(top_k)})
                total += 1
    return {"folds": limit, "pool_coverage": round(hits / total, 4) if total else 0.0}


def calibrate_set_model(game: str, rows: list[dict]) -> dict:
    windows = [window for window in BACKTEST_WINDOWS if window < len(rows)] or [max(1, len(rows) - 1)]
    scores = {str(window): rolling_pool_backtest(game, rows, window) for window in windows}
    chosen = max(
        windows,
        key=lambda window: (scores[str(window)]["pool_coverage"], int(window >= 300), -window),
    )
    return {"selected_window": chosen, "backtest": scores}


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
    position_counts = [blended_position_counts(rows, position, model["decay"], model.get("window_size")) for position in range(digits)]
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


def global_candidate_score(values: list[int], components: tuple, model: dict | None = None) -> float:
    """Score direct numbers by position; no unordered-set or total-sum terms."""
    model = model or DIGIT_MODELS["pl3"]
    position_counts, totals, omissions, _, _ = components
    score = sum(
        math.log(score_digit(value, position_counts[pos], totals[pos]))
            - model["global_omission"] * abs(min(omissions[pos][value], 20) - 9.0) / 9.0
        for pos, value in enumerate(values)
    )
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


def generate_digits(game: str, rows: list[dict], digits: int, issue: str) -> tuple[list[dict], list[float]]:
    del game, issue  # deterministic from the verified history snapshot
    ranked = generate_digit_profile(rows, digits, "global", 5, game)
    candidates = []
    for number, _, heat in ranked:
        band = "热门支撑" if heat > 0.25 else "冷门保护" if heat < -0.25 else "冷热均衡"
        candidates.append({"number": number, "mix_label": band})
    return candidates, [score for _, score, _ in ranked]


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


def generate_composite_recommendations(
    game: str, rows: list[dict], pl3_rows: list[dict]
) -> tuple[list[dict], list[float]]:
    """Merge the main, cold-protection, and hot-observation pools into one list."""
    digits = len(rows[0]["numbers"])
    quotas = (
        ("global", 4 if game == "pl5" else 5),
        ("cold", 1 if game == "pl5" else 2),
        ("hot", 1),
    )
    labels = {
        "global": "综合主榜",
        "cold": "冷门保护",
        "hot": "热门观察",
    }
    selected: list[dict] = []
    used: set[str] = set()
    for profile, quota in quotas:
        ranked = (
            generate_pl5_from_pl3(pl3_rows, rows, profile, 5)
            if game == "pl5"
            else generate_digit_profile(rows, digits, profile, 5, game)
        )
        added = 0
        for number, _, _ in ranked:
            if number in used:
                continue
            selected.append({"number": number, "mix_label": labels[profile], "source": profile})
            used.add(number)
            added += 1
            if added == quota:
                break
        if added != quota:
            raise RuntimeError(f"{game} 无法生成足够的 {labels[profile]} 候选")

    # Re-rank the merged list on one common score; source labels describe why a
    # hedge entered the list, not a separate probability claim.
    components = mixed_digit_components(rows, digits, digit_model(game))
    scored = [
        (candidate, global_candidate_score([int(value) for value in candidate["number"]], components, digit_model(game)))
        for candidate in selected
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [candidate for candidate, _ in scored], [score for _, score in scored]


def generate_positional_ensemble(game: str, rows: list[dict]) -> tuple[list[dict], list[float]]:
    """Direct-digit output from one position-only ensemble, with neutral digits allowed."""
    digits = len(rows[0]["numbers"])
    limit = 6 if game == "pl5" else 8
    ranked = generate_digit_profile(rows, digits, "global", limit, game)
    candidates = []
    for number, _, heat in ranked:
        label = "位置综合-中性" if -0.25 <= heat <= 0.25 else "位置综合-偏热" if heat > 0.25 else "位置综合-偏冷"
        candidates.append({"number": number, "mix_label": label, "source": "position_ensemble"})
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
    rng = stable_rng("dlt", issue)
    window = calibrate_set_model("dlt", rows)["selected_window"]
    front_counts = blended_number_counts(rows, 0, 5, 24, window)
    back_counts = blended_number_counts(rows, 5, 7, 24, window)
    front_rank_counts = [blended_position_counts(rows, pos, 24, window, range(1, 36)) for pos in range(5)]
    back_rank_counts = [blended_position_counts(rows, pos + 5, 24, window, range(1, 13)) for pos in range(2)]
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
                    + 0.8 * len(set(item[0][1]) & set(chosen[0][1]))
                    for chosen in selected
                ),
                default=0.0,
            ),
        )
        selected.append(best)
        remaining.remove(best)
    selected.sort(key=lambda pair: pair[1], reverse=True)
    candidates = [{"front": list(key[0]), "back": list(key[1])} for key, _ in selected]
    return candidates, [score for _, score in selected]


def generate_qxc(rows: list[dict]) -> tuple[list[dict], list[float]]:
    """7星彩专用七位置束搜索，避免套用三位数枚举模型。"""
    decay = 27
    window = calibrate_set_model("qxc", rows)["selected_window"]
    counters = [blended_position_counts(rows, pos, decay, window) for pos in range(7)]
    totals = [sum(counter.values()) for counter in counters]
    omissions = [position_omissions(rows, pos) for pos in range(7)]
    beam: list[tuple[str, float]] = [("", 0.0)]
    for pos in range(7):
        expanded = []
        for prefix, prefix_score in beam:
            for digit in range(10):
                probability = score_digit(digit, counters[pos], totals[pos])
                score = prefix_score + 0.78 * math.log(probability)
                score += 0.032 * min(omissions[pos][digit], 10)
                if prefix and int(prefix[-1]) == digit:
                    score -= 0.08
                expanded.append((prefix + str(digit), score))
        beam = sorted(expanded, key=lambda item: item[1], reverse=True)[:350]
    scored = [(number, score - 0.006 * abs(sum(map(int, number)) - 31.5), 0.0) for number, score in beam]
    ranked = diversified_rank(scored, 8, 1.15)
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
    """双色球专用红蓝分区模型，加入红球共现与三区覆盖。"""
    rng = stable_rng("ssq", issue)
    window = calibrate_set_model("ssq", rows)["selected_window"]
    red_counts = blended_number_counts(rows, 0, 6, 24, window)
    blue_counts = blended_number_counts(rows, 6, 7, 24, window)
    pair_counts = weighted_pair_counts(rows, 6, 30)
    red_total, blue_total = sum(red_counts.values()), sum(blue_counts.values())
    pool: dict[tuple[tuple[int, ...], int], float] = {}
    for _ in range(16000):
        red = sorted(rng.sample(range(1, 34), 6))
        blue = rng.randint(1, 16)
        score = sum(math.log((red_counts[n] + 0.72) / (red_total + 23.76)) for n in red)
        score += 0.055 * sum(pair_counts[pair] for pair in combinations(red, 2))
        score += math.log((blue_counts[blue] + 0.70) / (blue_total + 11.2))
        zone_counts = [sum(1 for n in red if low <= n <= high) for low, high in ((1, 11), (12, 22), (23, 33))]
        score += 0.10 if all(zone_counts) else -0.30
        score -= 0.0025 * abs(sum(red) - 102)
        pool[(tuple(red), blue)] = score
    ranked = sorted(pool.items(), key=lambda item: item[1], reverse=True)[:5]
    candidates = [
        {"red": list(key[0]), "blue": [key[1]], "mix_label": "红蓝分区共现模型", "source": "ssq_zone"}
        for key, _ in ranked
    ]
    return candidates, [score for _, score in ranked]


def generate_kl8(rows: list[dict], pick_count: int = 5) -> tuple[list[dict], list[float]]:
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
        number: math.log((counts[number] + 0.75) / (total + 60))
        + 0.024 * min(missed[number], 12)
        for number in range(1, 81)
    }
    # 先保留单号支持度最高的28个号码，再完整枚举其中的五数组合。
    if not 5 <= pick_count <= 10:
        raise ValueError(f"快乐8选N仅支持5至10，收到: {pick_count}")
    pool = sorted(individual, key=individual.get, reverse=True)[:32]
    rng = stable_rng(f"kl8-pick{pick_count}", str(len(rows)))
    scored: list[tuple[tuple[int, ...], float]] = []
    sample_count = 9000 if pick_count >= 8 else 6000
    for _ in range(sample_count):
        values = rng.sample(pool, pick_count)
        ordered = tuple(sorted(values))
        zones = {min((number - 1) // 20, 3) for number in ordered}
        odd_count = sum(number % 2 for number in ordered)
        score = sum(individual[number] for number in ordered)
        score += 0.020 * sum(pair_counts[pair] for pair in combinations(ordered, 2))
        score += 0.16 if len(zones) >= min(4, max(3, pick_count // 3)) else -0.20
        score += 0.08 if abs(odd_count - pick_count / 2) <= 1 else -0.08
        score -= 0.0018 * abs(sum(ordered) - pick_count * 40.5)
        scored.append((ordered, score))

    remaining = sorted(scored, key=lambda item: item[1], reverse=True)[:2500]
    selected: list[tuple[tuple[int, ...], float]] = []
    while remaining and len(selected) < 5:
        best = max(
            remaining,
            key=lambda item: item[1]
            - 0.52 * max((len(set(item[0]) & set(chosen[0])) for chosen in selected), default=0),
        )
        selected.append(best)
        remaining.remove(best)
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
    methods = ("日期哈希映射", "独立位置映射", "中性约束映射")
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
            "copy_text": f"{cfg['name']} {combined}",
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


def build_model_review(game: str, latest: dict, prediction: dict) -> dict:
    """Compare the predictions saved before the latest draw with its result."""
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

    hit_counts = [len(set(values(candidate)) & set(actual)) for candidate in candidates]
    exact_hits = sum(values(candidate) == actual for candidate in candidates)
    position_hits = []
    position_pool_hits = []
    for position, target in enumerate(actual):
        candidate_values = [values(candidate)[position] for candidate in candidates if position < len(values(candidate))]
        position_hits.append(sum(value == target for value in candidate_values))
        position_pool_hits.append(int(target in set(candidate_values)))
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
        "position_candidate_hits": position_hits,
        "position_pool_coverage": sum(position_pool_hits),
        "position_count": len(actual),
        "summary": (
            f"Issue {latest['issue']} actual result {display({'front': actual[:5], 'back': actual[5:], 'red': actual[:6], 'blue': actual[6:], 'numbers': actual, 'number': ''.join(latest['numbers'])}) if game in ('dlt', 'ssq', 'kl8') else ''.join(latest['numbers'])}; "
            f"the previous candidate pool reached {max(hit_counts, default=0)} matching numbers at best."
        ),
        "lesson": "The result is now included in the rolling window; keep the current model parameters and candidate diversification, without chasing a single draw shape.",
    }
    if game == "kl8":
        review["union_number_hits"] = len(set().union(*(set(values(candidate)) for candidate in candidates)) & set(actual))
    return review


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
            "summary": f"最近{sample}期以1–80单号衰减频率、遗漏封顶和两两共现为主，并约束每组5个号码覆盖至少三个区间。五组之间主动降低重合，扩大号码覆盖。",
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
    labels = {3: ["百位", "十位", "个位"], 5: ["万位", "千位", "百位", "十位", "个位"], 7: ["第一位", "第二位", "第三位", "第四位", "第五位", "第六位", "第七位"]}[len(rows[0]["numbers"])]
    calibration = calibrate_digit_model(game, rows, len(rows[0]["numbers"])) if game in ("pl3", "pl5", "fc3d") else set_calibration
    model = calibration["parameters"] if game in ("pl3", "pl5", "fc3d") else {"decay": 27}
    for pos in range(len(rows[0]["numbers"])):
        counter = weighted_counts(rows[:sample], pos, model["decay"])
        ranked = [str(value) for value, _ in counter.most_common(3)]
        missed = position_omissions(rows[:sample], pos)
        longest = sorted(missed.items(), key=lambda item: item[1], reverse=True)[:2]
        position_hot.append(ranked[0])
        position_analysis.append({
            "position": labels[pos],
            "hot_digits": ranked,
            "omitted_digits": [{"digit": str(value), "miss": miss} for value, miss in longest],
        })
    structure_note = "排列5先继承同一期排列3前三位候选，再独立计算00–99后两位；" if game == "pl5" else ""
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
        "backtest": calibration["backtest"] if calibration else None,
        "selected_decay": calibration.get("selected_decay") if game in ("pl3", "pl5", "fc3d") else None,
        "selected_window": calibration.get("selected_window") if calibration else None,
        "method": (["七位独立频率与遗漏", "逐位束搜索", "相邻重号与和值温和约束"] if game == "qxc" else ["逐位置频率与遗漏", f"{model['decay']}期衰减参数", "5至8注综合清单与候选分散"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成体彩数据看板")
    parser.add_argument("--games", default="dlt,pl3,pl5,fc3d,qxc,ssq,kl8", help="只刷新指定玩法，逗号分隔")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source_data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    model_reviews = json.loads(REVIEWS_PATH.read_text(encoding="utf-8")) if REVIEWS_PATH.exists() else {}
    selected = [game.strip() for game in args.games.split(",") if game.strip()]
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
        "daily_model_version": "v2.2-independent-date-game-schemes",
        "daily_results": generate_daily_results(now.date().isoformat(), config),
        "source_status": source_data.get("source_status", "unknown"),
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
            ranked_plays = []
            for key, play in play_types.items():
                play_candidates, play_scores = play["candidates"]
                ranked_plays.append((play_candidates[0] | {"pick_count": int(key), "play_name": play["name"]}, relative_confidences(play_scores)[0]))
            ranked_plays.sort(key=lambda item: item[1], reverse=True)
            candidates = [candidate for candidate, _ in ranked_plays[:5]]
            scores = [score for _, score in ranked_plays[:5]]
        else:
            candidates, scores = generate_positional_ensemble(game, rows)

        # Main-list scores are relative to the backtested ranking model. Strategy
        # zones below keep their separate common hot/cold support scale.
        confidences = relative_confidences(scores)
        enriched = []
        for rank, (candidate, confidence) in enumerate(zip(candidates, confidences), start=1):
            text_value = candidate_text(game, candidate)
            play_prefix = f"{candidate['play_name']} " if game == "kl8" else ""
            copy_text = f"{cfg['name']} {play_prefix}{text_value}"
            enriched.append({**candidate, "rank": rank, "confidence": confidence, "copy_text": copy_text})

        output["games"][game] = {
            "name": cfg["name"],
            "model_version": "v3.0-positional-rolling-ensemble",
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
            "review": build_review(game, rows),
            "analysis": build_analysis(game, rows),
            "model_review": model_reviews.get(game),
        }
        previous_game = previous_games.get(game, {})
        if previous_game.get("target_issue") == latest["issue"]:
            model_reviews[game] = build_model_review(game, latest, previous_game)
            output["games"][game]["model_review"] = model_reviews[game]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    REVIEWS_PATH.write_text(json.dumps(model_reviews, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 已生成 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
