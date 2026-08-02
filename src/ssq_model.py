"""双色球独立模型入口。

双色球红球与蓝球分别校准历史窗口，并复用现有的红蓝分区、共现和覆盖约束。
"""
from __future__ import annotations

from generate_dashboard import calibrate_set_model, generate_ssq
from vendor_models_v3 import MODEL_VERSION


def build_ssq_model(rows: list[dict], issue: str) -> dict[str, object]:
    """生成双色球红球/蓝球独立模型结果。"""
    calibration = calibrate_set_model("ssq", rows)
    candidates, scores = generate_ssq(rows, issue)
    return {
        "game": "ssq",
        "model_version": MODEL_VERSION,
        "model_name": "双色球红球/蓝球独立模型",
        "selected_window": calibration["selected_window"],
        "selected_region_windows": calibration.get("region_windows", {}),
        "backtest": calibration["backtest"],
        "candidates": candidates,
        "scores": scores,
    }


__all__ = ["build_ssq_model", "generate_ssq"]
