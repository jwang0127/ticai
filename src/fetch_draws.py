from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "games.json"
OUTPUT_PATH = ROOT / "data" / "processed" / "draws.json"
TZ = ZoneInfo("Asia/Shanghai")


def find_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows = [item for item in value if isinstance(item, dict)]
        if rows and any("lotteryDraw" in "".join(row.keys()) for row in rows):
            return rows
        for item in value:
            found = find_rows(item)
            if found:
                return found
    elif isinstance(value, dict):
        for item in value.values():
            found = find_rows(item)
            if found:
                return found
    return []


def pick(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if row.get(key) not in (None, ""):
            return row[key]
    return None


def tokens_from(row: dict[str, Any], game: str) -> list[str]:
    raw = pick(
        row,
        "lotteryDrawResult",
        "drawResult",
        "lotteryResult",
        "winningNumbers",
        "number",
        "numbers",
    )
    if isinstance(raw, list):
        parts = [str(value) for value in raw]
    elif isinstance(raw, str):
        parts = re.findall(r"\d+", raw)
    else:
        raise ValueError("未找到开奖号码字段")

    if game in ("dlt", "ssq"):
        return [part.zfill(2) for part in parts]
    if len(parts) == 1:
        return list(parts[0])
    return [part[-1] for part in parts]


def validate(game: str, numbers: list[str], cfg: dict[str, Any]) -> None:
    values = [int(value) for value in numbers]
    if game == "dlt":
        if len(values) != 7:
            raise ValueError(f"大乐透号码数量错误: {numbers}")
        front, back = values[:5], values[5:]
        if front != sorted(front) or back != sorted(back):
            raise ValueError(f"大乐透号码未按升序排列: {numbers}")
        if len(set(front)) != 5 or len(set(back)) != 2:
            raise ValueError(f"大乐透号码重复: {numbers}")
        if not all(cfg["front_range"][0] <= n <= cfg["front_range"][1] for n in front):
            raise ValueError(f"大乐透前区号码越界: {numbers}")
        if not all(cfg["back_range"][0] <= n <= cfg["back_range"][1] for n in back):
            raise ValueError(f"大乐透后区号码越界: {numbers}")
    elif game == "ssq":
        if len(values) != 7:
            raise ValueError(f"双色球号码数量错误: {numbers}")
        red, blue = values[:6], values[6:]
        if red != sorted(red) or len(set(red)) != 6:
            raise ValueError(f"双色球红球错误: {numbers}")
        if not all(1 <= n <= 33 for n in red) or not all(1 <= n <= 16 for n in blue):
            raise ValueError(f"双色球号码越界: {numbers}")
    elif game == "kl8":
        if len(values) != cfg["draw_count"] or len(set(values)) != cfg["draw_count"]:
            raise ValueError(f"快乐8号码数量或重复错误: {numbers}")
        if values != sorted(values) or not all(cfg["number_range"][0] <= n <= cfg["number_range"][1] for n in values):
            raise ValueError(f"快乐8号码顺序或范围错误: {numbers}")
    elif len(values) != cfg["digits"] or not all(0 <= n <= 9 for n in values):
        raise ValueError(f"{cfg['name']}号码错误: {numbers}")


def request_payload(url: str, params: dict[str, str]) -> dict[str, Any]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Referer": "https://www.lottery.gov.cn/kjxx/",
        "Origin": "https://www.lottery.gov.cn",
    }
    error: Exception | None = None
    request = Request(f"{url}?{urlencode(params)}", headers=headers)
    for attempt in range(2):
        try:
            with urlopen(request, timeout=12) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("接口未返回 JSON 对象")
            return payload
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"官方接口请求失败: {error}")


def fetch_game(game: str, cfg: dict[str, Any], api: str) -> list[dict[str, Any]]:
    payload = request_payload(
        api,
        {
            "gameNo": str(cfg["game_no"]),
            "provinceId": "0",
            "pageSize": "100",
            "pageNo": "1",
            "isVerify": "1",
        },
    )
    rows = find_rows(payload)
    if not rows:
        raise ValueError("接口响应中没有开奖列表")

    result: list[dict[str, Any]] = []
    for row in rows:
        numbers = tokens_from(row, game)
        validate(game, numbers, cfg)
        issue = pick(row, "lotteryDrawNum", "drawNum", "issue", "issueNo", "term")
        draw_date = pick(row, "lotteryDrawTime", "drawTime", "drawDate", "date")
        if not issue or not draw_date:
            raise ValueError("开奖数据缺少期号或日期")
        result.append(
            {
                "issue": str(issue),
                "draw_date": str(draw_date)[:10],
                "numbers": numbers,
            }
        )
    return sorted(result, key=lambda row: row["issue"], reverse=True)


def fetch_game_limited(game: str, cfg: dict[str, Any], api: str, history_limit: int) -> list[dict[str, Any]]:
    """Fetch a bounded official history with page-level validation."""
    result: list[dict[str, Any]] = []
    pages = (history_limit + 99) // 100
    for page_no in range(1, pages + 1):
        payload = request_payload(api, {
            "gameNo": str(cfg["game_no"]), "provinceId": "0", "pageSize": "100",
            "pageNo": str(page_no), "isVerify": "1",
        })
        rows = find_rows(payload)
        if not rows:
            break
        for row in rows:
            numbers = tokens_from(row, game)
            validate(game, numbers, cfg)
            issue = pick(row, "lotteryDrawNum", "drawNum", "issue", "issueNo", "term")
            draw_date = pick(row, "lotteryDrawTime", "drawTime", "drawDate", "date")
            if not issue or not draw_date:
                raise ValueError("official draw row is missing issue or date")
            result.append({"issue": str(issue), "draw_date": str(draw_date)[:10], "numbers": numbers})
            if len(result) >= history_limit:
                break
        if len(result) >= history_limit or len(rows) < 100:
            break
    if not result:
        raise ValueError("official endpoint returned no draw rows")
    return sorted(result[:history_limit], key=lambda row: row["issue"], reverse=True)


def fetch_welfare_game(game: str, cfg: dict[str, Any], api: str, default_results_page: str, history_limit: int = 100) -> list[dict[str, Any]]:
    welfare_name = cfg.get("welfare_name", "3d")
    results_page = cfg.get("results_page", default_results_page)
    request = Request(
        f"{api}?{urlencode({'name': welfare_name, 'issueCount': str(min(history_limit, 5000))})}",
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Referer": results_page,
        },
    )
    error: Exception | None = None
    for attempt in range(2):
        try:
            with urlopen(request, timeout=12) as response:
                payload = json.loads(response.read().decode("utf-8"))
            rows = payload.get("result", [])
            if not rows:
                raise ValueError(f"中国福彩网接口未返回{cfg['name']}开奖列表")
            result = []
            for row in rows:
                red = [value.strip() for value in str(row.get("red", "")).split(",") if value.strip()]
                blue = [value.strip() for value in str(row.get("blue", "")).split(",") if value.strip()]
                numbers = red + blue if game == "ssq" else red
                validate(game, numbers, cfg)
                issue = str(row.get("code", ""))
                draw_date = str(row.get("date", ""))[:10]
                if not issue or not draw_date:
                    raise ValueError(f"{cfg['name']}开奖数据缺少期号或日期")
                result.append({"issue": issue, "draw_date": draw_date, "numbers": numbers})
            return sorted(result, key=lambda row: row["issue"], reverse=True)
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"中国福彩网{cfg['name']}接口请求失败: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抓取体彩开奖历史")
    parser.add_argument(
        "--games",
        default="dlt,pl3,pl5,fc3d,qxc,ssq,kl8",
        help="逗号分隔的玩法代码，例如 pl3,pl5",
    )
    parser.add_argument("--history-limit", type=int, default=100, help="每个玩法抓取的历史期数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.history_limit < 1:
        raise SystemExit("--history-limit must be positive")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    selected = [game.strip() for game in args.games.split(",") if game.strip()]
    invalid = [game for game in selected if game not in config["games"]]
    if invalid:
        raise SystemExit(f"未知玩法: {', '.join(invalid)}")
    previous = json.loads(OUTPUT_PATH.read_text(encoding="utf-8")) if OUTPUT_PATH.exists() else {"draws": {}}
    draws = dict(previous.get("draws", {}))
    errors: dict[str, str] = {}

    # The three games are independent. Parallel requests keep a blocked upstream
    # from holding a scheduled GitHub Pages build for several minutes.
    with ThreadPoolExecutor(max_workers=len(selected)) as executor:
        futures = {}
        for game in selected:
            cfg = config["games"][game]
            if cfg.get("provider") == "cwl":
                future = executor.submit(
                    fetch_welfare_game,
                    game,
                    cfg,
                    config["welfare_api"],
                    config["welfare_results_page"],
                    args.history_limit,
                )
            else:
                future = executor.submit(fetch_game_limited, game, cfg, config["official_api"], args.history_limit)
            futures[future] = (game, cfg)
        for future in as_completed(futures):
            game, cfg = futures[future]
            try:
                draws[game] = future.result()
                print(f"[OK] {cfg['name']}: {draws[game][0]['issue']}")
            except Exception as exc:  # Preserve last verified data, never invent a result.
                errors[game] = str(exc)
                print(f"[KEEP] {cfg['name']}: {exc}")

    missing = [game for game in selected if not draws.get(game)]
    if missing:
        raise SystemExit(f"没有可保留的数据: {', '.join(missing)}")

    sources = list(previous.get("sources", []))
    for game in selected:
        cfg = config["games"][game]
        if cfg.get("provider") != "cwl":
            continue
        results_page = cfg.get("results_page", config["welfare_results_page"])
        if not any(source.get("url") == results_page for source in sources):
            sources.append({"name": f"中国福利彩票{cfg['name'].removeprefix('福彩')}开奖信息", "url": results_page, "role": f"{cfg['name']}官方开奖来源"})
    output = {
        **previous,
        "updated_at": datetime.now(TZ).isoformat(timespec="seconds"),
        "source_status": "official_api" if not errors else "cached_verified_data",
        "errors": errors,
        "draws": draws,
        "sources": sources,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
