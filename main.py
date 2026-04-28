from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import BacktestConfig
from engine import run_backtest
from selection_strategies import SELECTION_STRATEGY_REGISTRY
from trade_strategies import TRADE_STRATEGY_REGISTRY


def parse_args():
    default = BacktestConfig()
    parser = argparse.ArgumentParser(description="Modular stock backtest entry point.")
    parser.add_argument("--txt-dir", type=Path, default=default.txt_dir)
    parser.add_argument("--market-cache-dir", type=Path, default=default.market_cache_dir)
    parser.add_argument("--indicator-cache-path", type=Path, default=default.indicator_cache_path)
    parser.add_argument("--output-dir", type=Path, default=default.output_dir)
    parser.add_argument("--pools-dir", type=Path, default=default.pools_dir)
    parser.add_argument("--start-date", type=str, default=str(default.start_date.date()))
    parser.add_argument("--end-date", type=str, default=str(default.end_date.date()))
    parser.add_argument("--selection-strategy", type=str, default=default.selection_strategy, choices=sorted(SELECTION_STRATEGY_REGISTRY))
    parser.add_argument("--trade-strategy", type=str, default=default.trade_strategy, choices=sorted(TRADE_STRATEGY_REGISTRY))
    parser.add_argument("--initial-capital", type=float, default=default.initial_capital)
    parser.add_argument("--lot-size", type=int, default=default.lot_size)
    parser.add_argument("--commission-rate", type=float, default=default.commission_rate)
    parser.add_argument("--stamp-tax-rate", type=float, default=default.stamp_tax_rate)
    parser.add_argument("--slippage-rate", type=float, default=default.slippage_rate)
    parser.add_argument("--n1", type=int, default=default.n1)
    parser.add_argument("--n2", type=int, default=default.n2)
    parser.add_argument("--max-workers", type=int, default=default.max_workers)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        BacktestConfig(),
        txt_dir=args.txt_dir,
        market_cache_dir=args.market_cache_dir,
        indicator_cache_path=args.indicator_cache_path,
        output_dir=args.output_dir,
        pools_dir=args.pools_dir,
        start_date=pd.Timestamp(args.start_date),
        end_date=pd.Timestamp(args.end_date),
        selection_strategy=args.selection_strategy,
        trade_strategy=args.trade_strategy,
        initial_capital=args.initial_capital,
        lot_size=args.lot_size,
        commission_rate=args.commission_rate,
        stamp_tax_rate=args.stamp_tax_rate,
        slippage_rate=args.slippage_rate,
        n1=args.n1,
        n2=args.n2,
        max_workers=args.max_workers,
    )
    run_backtest(config)


if __name__ == "__main__":
    main()
