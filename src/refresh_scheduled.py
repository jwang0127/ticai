"""Refresh scheduled games independently so one late source cannot block all games."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "games.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default="due")
    parser.add_argument("--history-limit", type=int, default=2000)
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.games == "due":
        from datetime import datetime

        from fetch_draws import TZ, expected_latest_draw_date

        data_path = ROOT / "data" / "processed" / "draws.json"
        try:
            cached = json.loads(data_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            cached = {"draws": {}}
        now = datetime.now(TZ)
        selected = []
        for game, game_config in config["games"].items():
            rows = cached.get("draws", {}).get(game, [])
            expected = expected_latest_draw_date(game, game_config, config["draw_time"], now)
            actual = str(rows[0].get("draw_date", ""))[:10] if rows else ""
            if actual != expected:
                selected.append(game)
        print(f"[DUE] {', '.join(selected) if selected else 'none'}", flush=True)
    else:
        selected = [item.strip() for item in args.games.split(",") if item.strip()]
    invalid = [game for game in selected if game not in config["games"]]
    if invalid:
        print(f"Unknown games: {', '.join(invalid)}", file=sys.stderr)
        return 2

    succeeded: list[str] = []
    for game in selected:
        command = [
            sys.executable,
            str(ROOT / "src" / "fetch_draws.py"),
            "--games",
            game,
            "--history-limit",
            str(args.history_limit),
            "--require-fresh-results",
        ]
        print(f"[START] independent refresh: {game}", flush=True)
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if result.stdout:
            print(result.stdout.rstrip())
        if result.returncode == 0:
            succeeded.append(game)
            print(f"[OK] independent refresh: {game}", flush=True)
        else:
            if result.stderr:
                print(result.stderr.rstrip(), file=sys.stderr)
            print(f"[KEEP] independent refresh: {game} (exit {result.returncode})", flush=True)

    value = ",".join(succeeded)
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(f"games={value}\n")
    print(f"[SUMMARY] refreshed games: {value or 'none'}")
    # A clean no-op is not a failure: it means every cached game is already
    # current for the present Beijing-time schedule.
    return 0 if succeeded or not selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
