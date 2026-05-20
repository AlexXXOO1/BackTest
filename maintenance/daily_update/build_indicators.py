# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import BacktestConfig
from core.indicator_store import IndicatorStore


def _parse_int_list(value: str) -> tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(int(x) for x in value)
    return tuple(int(x.strip()) for x in str(value).split(",") if x.strip())


def parse_args():
    default = BacktestConfig()
    parser = argparse.ArgumentParser(description="Build clean reusable daily indicator cache.")
    parser.add_argument("--market-cache-dir", type=Path, default=default.market_cache_dir)
    parser.add_argument("--indicator-cache-path", type=Path, default=default.indicator_cache_path)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--lookback-days", type=int, default=150)

    parser.add_argument("--n1", type=int, default=4)
    parser.add_argument("--n2", type=int, default=6)

    parser.add_argument("--ma-windows", type=_parse_int_list, default=(5, 10, 20, 60))
    parser.add_argument("--volume-windows", type=_parse_int_list, default=(5, 10))
    parser.add_argument("--macd-fast", type=int, default=12)
    parser.add_argument("--macd-slow", type=int, default=26)
    parser.add_argument("--macd-signal", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    indicator_store = IndicatorStore(
        indicator_cache_path=args.indicator_cache_path,
        market_cache_dir=args.market_cache_dir,
    )

    df = indicator_store.build(
        n1=args.n1,
        n2=args.n2,
        start_date=args.start_date,
        end_date=args.end_date,
        incremental=args.incremental,
        lookback_days=args.lookback_days,
        ma_windows=args.ma_windows,
        volume_windows=args.volume_windows,
        macd_fast=args.macd_fast,
        macd_slow=args.macd_slow,
        macd_signal=args.macd_signal,
    )

    meta_path = indicator_store.indicator_cache_path.with_suffix(indicator_store.indicator_cache_path.suffix + ".meta.json")
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    print("========== Clean indicator cache completed ==========")
    print(f"Rows: {int(meta.get('rows', len(df))):,}")
    print(f"Columns: {len(meta.get('columns', list(df.columns)))}")
    print(f"Path: {indicator_store.indicator_cache_path}")
    columns = meta.get("columns", list(df.columns))
    if columns:
        print("Column list:")
        print(columns)


if __name__ == "__main__":
    main()
