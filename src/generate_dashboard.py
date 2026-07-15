from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from itertools import permutations
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "games.json"
DATA_PATH = ROOT / "data" / "processed" / "draws.json"
OUTPUT_PATH = ROOT / "docs" / "assets" / "data" / "dashboard.json"
REVIEWS_PATH = ROOT / "data" / "processed" / "model_reviews.json"
TZ = ZoneInfo("Asia/Shanghai")


def next_draw(now: datetime, weekdays: list[int], draw_clock: time) -> datetime:
    for offset in range(8):
        date = (now + timedelta(days=offset)).date()
        candidate = datetime.combine(date, draw_clock, tzinfo=TZ)
        if candidate.weekday() in weekdays and candidate > now:
            return candidate
    raise RuntimeError("无法计算下一期开奖时间")


def weighted_counts(rows: list[dict], position: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    for index, row in enumerate(rows):
        if position >= len(row["numbers"]):
            continue
        counts[int(row["numbers"][position])] += math.exp(-index / 24)
    return counts


def relative_confidences(scores: list[float]) -> list[int]:
    """Return strictly rank-based display scores; these are never draw odds."""
    if not scores:
        return []
    if len(scores) == 1:
        return [64]
    return [round(77 - index * 25 / (len(scores) - 1)) for index in range(len(scores))]


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


def mixed_digit_components(rows: list[dict], digits: int) -> tuple[list[Counter[int]], list[float], list[dict[int, int]], list[float], list[float]]:
    """Build hot/cold signals without treating either as predictive certainty."""
    position_counts = [weighted_counts(rows, position) for position in range(digits)]
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


def mixed_number_score(values: list[int], components: tuple) -> tuple[float, float]:
    position_counts, totals, omissions, means, scales = components
    score = 0.0
    heat_values = []
    for pos, value in enumerate(values):
        probability = score_digit(value, position_counts[pos], totals[pos])
        heat_z = (probability - means[pos]) / scales[pos]
        heat_values.append(heat_z)
        # Keep a majority hot-frequency signal, cap omission compensation, and
        # only punish the most crowded tail. This is a mixed model, not all-cold.
        score += 0.58 * math.log(probability)
        score += 0.075 * min(omissions[pos][value], 8)
        score -= 0.22 * max(0.0, heat_z - 0.85) ** 2
    unique_ratio = len(set(values)) / len(values)
    score += 0.09 * unique_ratio - 0.012 * abs(sum(values) - 4.5 * len(values))
    return score, sum(heat_values) / len(heat_values)


def diversified_rank(scored: list[tuple[str, float, float]], limit: int) -> list[tuple[str, float, float]]:
    """Greedily avoid returning near-duplicates while preserving model rank."""
    remaining = sorted(scored, key=lambda item: item[1], reverse=True)
    selected: list[tuple[str, float, float]] = []
    while remaining and len(selected) < limit:
        best = max(
            remaining,
            key=lambda item: item[1] - 0.34 * max(
                (sum(a == b for a, b in zip(item[0], chosen[0])) for chosen in selected),
                default=0,
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return selected


def generate_digit_profile(rows: list[dict], digits: int, profile: str, limit: int = 5) -> list[tuple[str, float, float]]:
    components = mixed_digit_components(rows, digits)
    scored = []
    for number in range(10 ** digits):
        text = f"{number:0{digits}d}"
        score, heat = mixed_number_score([int(value) for value in text], components)
        if profile == "hot" and heat <= 0.25:
            continue
        if profile == "cold" and heat >= -0.25:
            continue
        if profile == "balanced" and not (-0.25 <= heat <= 0.25):
            continue
        profile_score = score
        if profile == "balanced":
            profile_score -= 0.12 * abs(heat)
        elif profile == "hot":
            profile_score += 0.12 * min(heat, 1.5)
        else:
            profile_score += 0.10 * min(-heat, 1.5)
        scored.append((text, profile_score, heat))
    return diversified_rank(scored, limit)


def generate_digits(game: str, rows: list[dict], digits: int, issue: str) -> tuple[list[dict], list[float]]:
    del game, issue  # deterministic from the verified history snapshot
    ranked = generate_digit_profile(rows, digits, "balanced", 5)
    candidates = []
    for number, _, heat in ranked:
        candidates.append({"number": number, "mix_label": "冷热均衡"})
    return candidates, [score for _, score, _ in ranked]


def weighted_number_counts(rows: list[dict], start: int, end: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    for index, row in enumerate(rows):
        weight = math.exp(-index / 24)
        for value in row["numbers"][start:end]:
            counts[int(value)] += weight
    return counts


def generate_dlt(rows: list[dict], issue: str) -> tuple[list[dict], list[float]]:
    rng = stable_rng("dlt", issue)
    front_counts = weighted_number_counts(rows, 0, 5)
    back_counts = weighted_number_counts(rows, 5, 7)
    front_total, back_total = sum(front_counts.values()), sum(back_counts.values())
    pool: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}

    for _ in range(8000):
        front = sorted(rng.sample(range(1, 36), 5))
        back = sorted(rng.sample(range(1, 13), 2))
        score = sum(math.log((front_counts[n] + 0.8) / (front_total + 28)) for n in front)
        score += sum(math.log((back_counts[n] + 0.8) / (back_total + 9.6)) for n in back)
        score -= 0.003 * abs(sum(front) - 90)
        score += 0.025 if 1 <= sum(n % 2 for n in front) <= 4 else -0.025
        pool[(tuple(front), tuple(back))] = score

    ranked = sorted(pool.items(), key=lambda pair: pair[1], reverse=True)[:5]
    candidates = [{"front": list(key[0]), "back": list(key[1])} for key, _ in ranked]
    return candidates, [score for _, score in ranked]


def candidate_text(game: str, candidate: dict) -> str:
    if game == "dlt":
        front = " ".join(f"{value:02d}" for value in candidate["front"])
        back = " ".join(f"{value:02d}" for value in candidate["back"])
        return f"{front} + {back}"
    return candidate["number"]


def digit_shape(values: list[int]) -> str:
    unique = len(set(values))
    if len(values) == 3:
        return {1: "豹子", 2: "组选3", 3: "组选6"}[unique]
    return f"{unique}个不同数字"


def build_review(game: str, rows: list[dict]) -> dict:
    latest = [int(value) for value in rows[0]["numbers"]]
    previous = [int(value) for value in rows[1]["numbers"]]
    if game == "dlt":
        front, back = latest[:5], latest[5:]
        previous_front = set(previous[:5])
        return {
            "title": f"第{rows[0]['issue']}期结构复盘",
            "summary": (
                f"前区和值{sum(front)}、跨度{max(front) - min(front)}，"
                f"奇偶比{sum(n % 2 for n in front)}:{sum(n % 2 == 0 for n in front)}；"
                f"后区和值{sum(back)}。与前一期前区重号{len(set(front) & previous_front)}个。"
            ),
            "metrics": [
                {"label": "前区和值", "value": str(sum(front))},
                {"label": "前区跨度", "value": str(max(front) - min(front))},
                {"label": "前区奇偶", "value": f"{sum(n % 2 for n in front)}:{sum(n % 2 == 0 for n in front)}"},
                {"label": "后区和值", "value": str(sum(back))},
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
    sample = min(100, len(rows))
    if game == "dlt":
        front = weighted_number_counts(rows[:sample], 0, 5)
        back = weighted_number_counts(rows[:sample], 5, 7)
        hot_front = [f"{n:02d}" for n, _ in front.most_common(5)]
        hot_back = [f"{n:02d}" for n, _ in back.most_common(3)]
        omitted = [f"{n:02d}（{miss}期）" for n, miss in omission(rows[:sample], 0, 5, range(1, 36))[:5]]
        return {
            "sample": sample,
            "summary": f"最近{sample}期采用指数衰减频率，前后区分别统计；当前前区相对活跃号为{'、'.join(hot_front)}，后区为{'、'.join(hot_back)}。",
            "signals": [
                {"label": "前区相对活跃", "value": " · ".join(hot_front)},
                {"label": "后区相对活跃", "value": " · ".join(hot_back)},
                {"label": "前区较长遗漏", "value": "、".join(omitted)},
            ],
            "method": ["最近100期指数衰减频率", "前区与后区独立建模", "和值、奇偶与跨度温和约束"],
        }

    all_counts = Counter(int(value) for row in rows[:sample] for value in row["numbers"])
    hot = [str(n) for n, _ in all_counts.most_common(5)]
    omitted = [f"{n}（{miss}期）" for n, miss in omission(rows[:sample], 0, len(rows[0]["numbers"]), range(10))[:3]]
    position_hot = []
    for pos in range(len(rows[0]["numbers"])):
        counter = weighted_counts(rows[:sample], pos)
        position_hot.append(str(counter.most_common(1)[0][0]))
    return {
        "sample": sample,
        "summary": f"最近{sample}期按位置统计并加入时间衰减。新模型保留热门支撑，同时惩罚极端拥挤，并对中冷数字给予有上限的补偿；各位置当前热门参考为{' · '.join(position_hot)}。",
        "signals": [
            {"label": "综合活跃数字", "value": " · ".join(hot)},
            {"label": "各位置最高权重", "value": " · ".join(position_hot)},
            {"label": "较长遗漏", "value": "、".join(omitted)},
        ],
        "method": ["58%热门频率基础", "极热惩罚与封顶冷号补偿", "候选差异化与结构温和约束"],
    }


def three_digit_group_candidates(
    game_name: str, rows: list[dict], group_type: str, target_issue: str, draw_at: datetime
) -> list[dict]:
    components = mixed_digit_components(rows, 3)
    scored: dict[str, float] = {}
    for number in range(1000):
        digits = [int(value) for value in f"{number:03d}"]
        counts = sorted(Counter(digits).values())
        if group_type == "group3" and counts != [1, 2]:
            continue
        if group_type == "group6" and counts != [1, 1, 1]:
            continue
        unique_orders = set(permutations(digits))
        likelihood = 0.0
        for order in unique_orders:
            order_score, _ = mixed_number_score(list(order), components)
            likelihood += math.exp(order_score)
        canonical = "".join(map(str, sorted(digits)))
        scored[canonical] = max(scored.get(canonical, 0.0), likelihood)
    ranked = sorted(scored.items(), key=lambda pair: pair[1], reverse=True)[:5]
    confidences = relative_confidences([score for _, score in ranked])
    label = "组选3" if group_type == "group3" else "组选6"
    return [
        {
            "number": number,
            "rank": rank,
            "confidence": confidence,
            "copy_text": (
                f"{game_name} {label}｜第{target_issue}期｜候选{rank}：{number}｜"
                f"模型相对评分 {confidence}%｜下一期开奖：{draw_at:%Y-%m-%d %H:%M}（北京时间）"
            ),
        }
        for rank, ((number, _), confidence) in enumerate(zip(ranked, confidences), start=1)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成体彩数据看板")
    parser.add_argument("--games", default="dlt,pl3,pl5,fc3d", help="只刷新指定玩法，逗号分隔")
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
    output = {
        "generated_at": now.isoformat(timespec="seconds"),
        "source_status": source_data.get("source_status", "unknown"),
        "disclaimer": "以上仅为公开信息整理后的娱乐分析，不构成任何购彩建议，请理性参考。模型置信度仅表示本页 5 个候选之间的相对评分，不是真实中奖概率。",
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
        else:
            candidates, scores = generate_digits(game, rows, cfg["digits"], target_issue)

        confidences = relative_confidences(scores)
        enriched = []
        for rank, (candidate, confidence) in enumerate(zip(candidates, confidences), start=1):
            text_value = candidate_text(game, candidate)
            copy_text = (
                f"{cfg['name']} 第{target_issue}期｜候选{rank}：{text_value}｜"
                f"模型相对评分 {confidence}%｜下一期开奖：{draw_at:%Y-%m-%d %H:%M}（北京时间）"
            )
            enriched.append({**candidate, "rank": rank, "confidence": confidence, "copy_text": copy_text})

        output["games"][game] = {
            "name": cfg["name"],
            "generated_at": now.isoformat(timespec="seconds"),
            "latest_issue": latest["issue"],
            "latest_draw_date": latest["draw_date"],
            "latest_numbers": latest["numbers"],
            "target_issue": target_issue,
            "next_draw_at": draw_at.isoformat(timespec="minutes"),
            "next_draw_display": f"{draw_at:%Y年%m月%d日 %H:%M}（北京时间）",
            "schedule_note": "每周一、三、六开奖" if game == "dlt" else "每日开奖（休市日除外）",
            "candidates": enriched,
            "top_candidates": enriched[:5],
            "review": build_review(game, rows),
            "analysis": build_analysis(game, rows),
            "model_review": model_reviews.get(game),
        }
        if game in ("pl3", "fc3d"):
            direct = []
            for item in enriched[:5]:
                direct.append({**item, "copy_text": item["copy_text"].replace(f"{cfg['name']} ", f"{cfg['name']} 直选｜", 1)})
            output["games"][game]["play_types"] = {
                "direct": {"name": "直选", "description": "数字与顺序均需一致", "candidates": direct},
                "group3": {"name": "组选3", "description": "两位数字相同，顺序不限", "candidates": three_digit_group_candidates(cfg["name"], rows, "group3", target_issue, draw_at)},
                "group6": {"name": "组选6", "description": "三位数字各不相同，顺序不限", "candidates": three_digit_group_candidates(cfg["name"], rows, "group6", target_issue, draw_at)},
            }
        if game in ("pl3", "pl5", "fc3d"):
            zones = {}
            for profile, zone_name, description in (
                ("hot", "热门专区", "保留近期位置频率支撑，但已限制极端追热"),
                ("cold", "冷门专区", "选取相对低热度组合，遗漏补偿设有上限"),
            ):
                ranked_zone = generate_digit_profile(rows, cfg["digits"], profile, 5)
                zone_scores = [score for _, score, _ in ranked_zone]
                zone_confidences = relative_confidences(zone_scores)
                zone_candidates = []
                for rank, ((number, _, _), confidence) in enumerate(zip(ranked_zone, zone_confidences), start=1):
                    zone_candidates.append({
                        "number": number,
                        "rank": rank,
                        "confidence": confidence,
                        "mix_label": zone_name.replace("专区", ""),
                        "copy_text": f"{cfg['name']} {zone_name}｜第{target_issue}期｜候选{rank}：{number}｜模型相对评分 {confidence}%｜下一期开奖：{draw_at:%Y-%m-%d %H:%M}（北京时间）",
                    })
                zones[profile] = {"name": zone_name, "description": description, "candidates": zone_candidates}
            output["games"][game]["strategy_zones"] = zones

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 已生成 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
