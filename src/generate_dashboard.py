from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "games.json"
DATA_PATH = ROOT / "data" / "processed" / "draws.json"
OUTPUT_PATH = ROOT / "docs" / "assets" / "data" / "dashboard.json"
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
    low, high = min(scores), max(scores)
    if math.isclose(low, high):
        return [64] * len(scores)
    return [round(52 + (score - low) / (high - low) * 25) for score in scores]


def stable_rng(game: str, issue: str) -> random.Random:
    seed = hashlib.sha256(f"{game}:{issue}:v2".encode()).hexdigest()
    return random.Random(int(seed[:16], 16))


def score_digit(number: int, counts: Counter[int], total: float) -> float:
    return (counts[number] + 0.65) / (total + 6.5)


def generate_digits(game: str, rows: list[dict], digits: int, issue: str) -> tuple[list[dict], list[float]]:
    rng = stable_rng(game, issue)
    position_counts = [weighted_counts(rows, position) for position in range(digits)]
    totals = [sum(counter.values()) for counter in position_counts]
    pool: dict[str, float] = {}

    for _ in range(5000):
        values = []
        log_score = 0.0
        for pos in range(digits):
            weights = [score_digit(n, position_counts[pos], totals[pos]) for n in range(10)]
            value = rng.choices(range(10), weights=weights, k=1)[0]
            values.append(value)
            log_score += math.log(weights[value])
        number = "".join(map(str, values))
        # Gentle shape regularization; no historical pattern can predict a random draw.
        unique_ratio = len(set(values)) / digits
        score = log_score + 0.08 * unique_ratio - 0.01 * abs(sum(values) - 4.5 * digits)
        pool[number] = max(pool.get(number, -999), score)

    ranked = sorted(pool.items(), key=lambda pair: pair[1], reverse=True)[:5]
    return [{"number": number} for number, _ in ranked], [score for _, score in ranked]


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


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source_data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    now = datetime.now(TZ)
    hour, minute = map(int, config["draw_time"].split(":"))
    output = {
        "generated_at": now.isoformat(timespec="seconds"),
        "source_status": source_data.get("source_status", "unknown"),
        "disclaimer": "以上仅为公开信息整理后的娱乐分析，不构成任何购彩建议，请理性参考。模型置信度仅表示本页 5 个候选之间的相对评分，不是真实中奖概率。",
        "games": {},
        "sources": source_data.get("sources", []),
    }

    for game, cfg in config["games"].items():
        rows = source_data["draws"][game]
        latest = rows[0]
        target_issue = str(int(latest["issue"]) + 1)
        draw_at = next_draw(now, cfg["draw_weekdays"], time(hour, minute))
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
            "latest_issue": latest["issue"],
            "latest_draw_date": latest["draw_date"],
            "latest_numbers": latest["numbers"],
            "target_issue": target_issue,
            "next_draw_at": draw_at.isoformat(timespec="minutes"),
            "next_draw_display": f"{draw_at:%Y年%m月%d日 %H:%M}（北京时间）",
            "schedule_note": "每周一、三、六开奖" if game == "dlt" else "每日开奖（休市日除外）",
            "candidates": enriched,
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 已生成 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

