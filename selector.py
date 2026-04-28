from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import BacktestConfig
from engine import run_selector
from selection_strategies import SELECTION_STRATEGY_REGISTRY


def parse_args():
    default = BacktestConfig()
    parser = argparse.ArgumentParser(description="Unified pool builder entry point.")
    parser.add_argument("--txt-dir", type=Path, default=default.txt_dir)
    parser.add_argument("--market-cache-dir", type=Path, default=default.market_cache_dir)
    parser.add_argument("--indicator-cache-path", type=Path, default=default.indicator_cache_path)
    parser.add_argument("--pools-dir", type=Path, default=default.pools_dir)
    parser.add_argument("--date", type=str, default=None, help="single selection date, for example 2026-03-05")
    parser.add_argument("--start-date", type=str, default=None, help="date")
    parser.add_argument("--end-date", type=str, default=None, help="date")
    parser.add_argument("--strategy", type=str, default=default.selection_strategy, choices=sorted(SELECTION_STRATEGY_REGISTRY))
    parser.add_argument("--n1", type=int, default=default.n1)
    parser.add_argument("--n2", type=int, default=default.n2)
    parser.add_argument("--max-workers", type=int, default=default.max_workers)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.date:
        start_date = end_date = pd.Timestamp(args.date)
    else:
        start_date = pd.Timestamp(args.start_date) if args.start_date else BacktestConfig().start_date
        end_date = pd.Timestamp(args.end_date) if args.end_date else BacktestConfig().end_date

    config = replace(
        BacktestConfig(),
        txt_dir=args.txt_dir,
        market_cache_dir=args.market_cache_dir,
        indicator_cache_path=args.indicator_cache_path,
        pools_dir=args.pools_dir,
        start_date=start_date,
        end_date=end_date,
        selection_strategy=args.strategy,
        n1=args.n1,
        n2=args.n2,
        max_workers=args.max_workers,
    )
    run_selector(config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
