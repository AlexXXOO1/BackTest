# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import pandas as pd

from core.path_manager import (
    PROJECT_ROOT as ROOT_DIR,
    DATA_ROOT,
    RAW_TDX_TXT_DIR,
    MARKET_CACHE_DIR,
    INDICATOR_CACHE_PATH,
    POOLS_DIR,
    OUTPUT_DIR,
)

TXT_DIR = RAW_TDX_TXT_DIR


@dataclass(frozen=True)
class BacktestConfig:
    txt_dir: Path = RAW_TDX_TXT_DIR
    market_cache_dir: Path = MARKET_CACHE_DIR
    indicator_cache_path: Path = INDICATOR_CACHE_PATH
    output_dir: Path = OUTPUT_DIR
    pools_dir: Path = POOLS_DIR
    start_date: pd.Timestamp = pd.Timestamp("2025-01-01")
    end_date: pd.Timestamp = pd.Timestamp("2025-12-31")
    n1: int = 4
    n2: int = 6
    selection_strategy: str = "renko_chart_select_strategy_v0"
    max_workers: int = max(1, (os.cpu_count() or 2) - 8)
