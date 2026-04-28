from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent

@dataclass(frozen=True)
class BacktestConfig:
    txt_dir: Path = Path("data")
    market_cache_dir: Path = Path("data/market_cache/daily_bars_by_symbol")
    indicator_cache_path: Path = Path("data/indicator_cache/daily_indicators.parquet")
    output_dir: Path = Path("output")
    pools_dir: Path = Path("pools")
    start_date: pd.Timestamp = pd.Timestamp("2025-01-01")
    end_date: pd.Timestamp = pd.Timestamp("2025-12-31")
    initial_capital: float = 20000.0
    lot_size: int = 100
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    slippage_rate: float = 0.0
    n1: int = 4
    n2: int = 6
    selection_strategy: str = "renko_chart_select_strategy_v0"
    trade_strategy: str = "renko_chart_select_strategy_v0_trade_v3"
    max_workers: int = max(1, (os.cpu_count() or 2) - 8)
