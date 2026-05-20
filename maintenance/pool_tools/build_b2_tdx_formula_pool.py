# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Compatibility entrypoint for the migrated B2 TDX formula strategy.

The formula logic now lives in:
    strategies/selection/b2_confirm_tdx_b1_v0.py

Preferred command:
    python .\\ops\\daily_update\\build_pool.py --strategy b2_confirm_tdx_b1_v0 --no-csv
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.path_manager import MARKET_CACHE_DIR, POOLS_DIR
from ops.daily_update.build_pool import build_pool_streaming_from_market_cache, parse_extra_args


STRATEGY_NAME = "b2_confirm_tdx_b1_v0"
DEFAULT_OUTPUT = POOLS_DIR / f"{STRATEGY_NAME}_pool.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper. B2 TDX formula has been migrated to "
            "strategies/selection/b2_confirm_tdx_b1_v0.py."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help=(
            "Strategy parameter. Can be repeated. "
            "Formats: key=value, key:int=10, key:float=0.75, key:bool=true"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategy_kwargs = parse_extra_args(args.param)

    print("[INFO] B2 TDX formula logic has moved to strategies/selection.", flush=True)
    print(f"[INFO] strategy: {STRATEGY_NAME}", flush=True)
    print(f"[INFO] market cache dir: {MARKET_CACHE_DIR}", flush=True)
    print("[INFO] Strategy params:", flush=True)
    print(json.dumps(strategy_kwargs, ensure_ascii=False, indent=2), flush=True)

    build_pool_streaming_from_market_cache(
        market_cache_dir=MARKET_CACHE_DIR,
        strategy_name=STRATEGY_NAME,
        strategy_kwargs=strategy_kwargs,
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.output,
        keep_all_rows=False,
        progress=True,
    )

    print(f"[DONE] output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
