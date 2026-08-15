"""快乐八独立模型入口。

本文件只负责快乐八“选N”模型的对外调用；底层历史加权、遗漏、共现、
分区覆盖和滚动窗口校准逻辑仍复用 dashboard 的通用统计基础函数。
"""
from __future__ import annotations

from generate_dashboard import (
    calibrate_set_model,
    generate_kl8,
    generate_kl8_play_types,
)
from vendor_models_v3 import MODEL_VERSION


def build_kl8_model(rows: list[dict], pick_counts: list[int] | None = None) -> dict[str, dict]:
    """生成快乐八选七至选十玩法的候选结果和回测窗口信息。"""
    counts = pick_counts or [7, 8, 9, 10]
    calibration = calibrate_set_model("kl8", rows)
    play_types = generate_kl8_play_types(rows, counts)
    return {
        "game": "kl8",
        "model_version": MODEL_VERSION,
        "model_name": "快乐8选N独立模型",
        "selected_window": calibration["selected_window"],
        "backtest": calibration["backtest"],
        "play_types": play_types,
    }


__all__ = [
    "build_kl8_model",
    "generate_kl8",
    "generate_kl8_play_types",
]
