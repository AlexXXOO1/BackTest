from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent

# All runtime data should live outside the git project.
# Override this on another machine with:
#   $env:BACKTEST_DATA_ROOT = "D:\\BackTest Data"
DATA_ROOT = Path(
    os.environ.get(
        "BACKTEST_DATA_ROOT",
        r"C:\Users\zyf37\Desktop\BackTest Data",
    )
)

# Standard runtime paths. Keep every script using these defaults.
RAW_TDX_TXT_DIR = DATA_ROOT / "data"
MARKET_CACHE_DIR = DATA_ROOT / "market_cache" / "daily_bars_by_symbol"
INDICATOR_CACHE_PATH = DATA_ROOT / "indicator_cache" / "daily_indicators.parquet"
POOLS_DIR = DATA_ROOT / "pools"
OUTPUT_DIR = DATA_ROOT / "output"

# Compatibility aliases for older imports / docs.
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
    initial_capital: float = 20000.0
    lot_size: int = 100
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    slippage_rate: float = 0.0
    n1: int = 4
    n2: int = 6
    selection_strategy: str = "renko_chart_select_strategy_v3"
    trade_strategy: str = "renko_trade_strategy_v0"
    max_workers: int = max(1, (os.cpu_count() or 2) - 8)
