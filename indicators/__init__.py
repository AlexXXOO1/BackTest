# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Clean indicator package.

Only generic reusable indicators should be exported here.
Strategy-specific signals, scores, selected flags, renko rules, market regime,
and progressive relaxation logic should live in selection_strategies/ or a
strategy feature module, not in the global indicator cache.
"""

from .renko_basic import add_renko_basic_indicator, tdx_sma
from .basic import (
    BASE_COLUMNS,
    DEFAULT_MA_WINDOWS,
    DEFAULT_VOLUME_WINDOWS,
    add_all_indicators,
    add_kline_indicators,
    add_ma_indicators,
    add_volume_indicators,
    add_macd_indicators,
)

__all__ = [
    "BASE_COLUMNS",
    "DEFAULT_MA_WINDOWS",
    "DEFAULT_VOLUME_WINDOWS",
    "add_all_indicators",
    "add_kline_indicators",
    "add_ma_indicators",
    "add_volume_indicators",
    "add_macd_indicators",
    "add_renko_basic_indicator",
    "tdx_sma",
]
