from __future__ import annotations

import pandas as pd

from .candle_patterns import add_candle_pattern_indicators


def add_v8_risk_indicators(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Backward-compatible wrapper that adds raw risk-reference metrics only.

    Risk booleans such as prior_20d_accelerated_huge_volume_bear,
    prior_20d_shrink_limit_up, long_lower_shadow_hammer, limit_up_red_brick,
    and v8_hard_filter_pass now belong to the strategy layer.
    """
    return add_candle_pattern_indicators(df, **kwargs)
