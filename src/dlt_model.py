"""大乐透独立模型入口。

大乐透前区与后区分别校准历史窗口，并复用现有的前后区候选生成与多样性约束。
"""
from __future__ import annotations

from generate_dashboard import calibrate_set_model, generate_dlt
from vendor_models_v3 import MODEL_VERSION


def build_dlt_model(rows: list[dict], issue: str) -> dict[str, object]:
    """生成大乐透前区/后区独立模型结果。"""
    calibration = calibrate_set_model("dlt", rows)
    candidates, scores = generate_dlt(rows, issue)
    return {
        "game": "dlt",
        "model_version": MODEL_VERSION,
        "model_name": "大乐透前区/后区独立模型",
        "selected_window": calibration["selected_window"],
        "selected_region_windows": calibration.get("region_windows", {}),
        "backtest": calibration["backtest"],
        "candidates": candidates,
        "scores": scores,
    }


__all__ = ["build_dlt_model", "generate_dlt"]
