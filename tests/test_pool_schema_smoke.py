# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd

from core.pool_schema import detect_factor_columns, validate_pool_schema


def test_pool_schema_smoke() -> None:
    df = pd.DataFrame({
        "symbol": ["000001"],
        "date": ["2026-05-11"],
        "selection_strategy": ["renko_chart_select_strategy_v0"],
        "open": [10.0],
        "high": [10.5],
        "low": [9.9],
        "close": [10.2],
        "volume": [1000],
        "amount": [10200],
        "t1_date": ["2026-05-12"],
        "t1_open": [10.3],
        "t1_close": [10.4],
        "t2_date": ["2026-05-13"],
        "t2_open": [10.4],
        "t2_close": [10.7],
        "t3_date": ["2026-05-14"],
        "t3_open": [10.8],
        "t3_close": [10.9],
        "t4_date": ["2026-05-15"],
        "t4_open": [10.9],
        "t4_close": [11.0],
        "fwd_return_pct_T1": [0.9708738],
        "fwd_return_pct_T2": [3.8834951],
        "fwd_return_pct_T3": [5.8252427],
        "fwd_return_pct_T4": [6.7961165],
        "amplitude_pct": [6.0],
    })
    report = validate_pool_schema(df, strategy_name="renko_chart_select_strategy_v0")
    assert report.schema_version == "pool_contract_v2_factor_first_t4"
    assert "amplitude_pct" in detect_factor_columns(df)
