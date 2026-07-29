"""Run rolling model calibration before the dashboard is generated."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from generate_dashboard import calibrate_digit_model, calibrate_set_model
from fetch_draws import today_games

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "games.json"
DATA_PATH = ROOT / "data" / "processed" / "draws.json"
OUTPUT_PATH = ROOT / "data" / "processed" / "model_tuning.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling backtests and save selected model parameters")
    parser.add_argument("--games", default="dlt,pl3,pl5,fc3d,qxc,ssq,kl8")
    parser.add_argument("--today", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    selected = today_games(config) if args.today else [game.strip() for game in args.games.split(",") if game.strip()]
    tuning = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "games": {}}
    for game in selected:
        rows = data["draws"][game]
        if game in ("pl3", "pl5", "fc3d"):
            result = calibrate_digit_model(game, rows, config["games"][game]["digits"])
            tuning["games"][game] = {
                "kind": "positional_digit",
                "selected_decay": result["selected_decay"],
                "selected_window": result["selected_window"],
                "selected_position_parameters": result["parameters"].get("position_parameters", []),
                "backtest": result["backtest"],
            }
        elif game in ("dlt", "ssq", "kl8"):
            result = calibrate_set_model(game, rows)
            tuning["games"][game] = {
                "kind": "set_model",
                "selected_window": result.get("selected_window"),
                "backtest": result["backtest"],
            }
        else:
            tuning["games"][game] = {"kind": "generator_internal_calibration"}
    OUTPUT_PATH.write_text(json.dumps(tuning, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] rolling calibration saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
