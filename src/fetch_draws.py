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

    if game == "dlt":
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


def fetch_fc3d(api: str, results_page: str) -> list[dict[str, Any]]:
    request = Request(
        f"{api}?{urlencode({'name': '3d', 'issueCount': '100'})}",
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
                raise ValueError("中国福彩网接口未返回3D开奖列表")
            result = []
            for row in rows:
                numbers = [value.strip() for value in str(row.get("red", "")).split(",") if value.strip()]
                validate("fc3d", numbers, {"name": "福彩3D", "digits": 3})
                issue = str(row.get("code", ""))
                draw_date = str(row.get("date", ""))[:10]
                if not issue or not draw_date:
                    raise ValueError("福彩3D开奖数据缺少期号或日期")
                result.append({"issue": issue, "draw_date": draw_date, "numbers": numbers})
            return sorted(result, key=lambda row: row["issue"], reverse=True)
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"中国福彩网接口请求失败: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抓取体彩开奖历史")
    parser.add_argument(
        "--games",
        default="dlt,pl3,pl5,fc3d",
        help="逗号分隔的玩法代码，例如 pl3,pl5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
                future = executor.submit(fetch_fc3d, config["welfare_api"], config["welfare_results_page"])
            else:
                future = executor.submit(fetch_game, game, cfg, config["official_api"])
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
    if "fc3d" in selected and not any(source.get("url") == config["welfare_results_page"] for source in sources):
        sources.append(
            {
                "name": "中国福利彩票3D开奖信息",
                "url": config["welfare_results_page"],
                "role": "福彩3D官方开奖来源",
            }
        )
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
