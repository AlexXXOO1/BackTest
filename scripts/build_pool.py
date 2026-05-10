# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Generic pool builder.

Purpose:
- Read indicator cache.
- Dynamically load any selection strategy from selection_strategies/.
- Run strategy per symbol.
- Keep selected rows.
- Save pool parquet/csv.

Strategy file requirement:
    selection_strategies/xxx.py

Must provide either:
    def select(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ...
or:
    SELECT_FUNC = select

Strategy output should contain:
    selected

Recommended strategy output:
    selected
    selected_score_base
    score_rank_key
    score_pct
    selection_strategy
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"C:\Users\zyf37\Desktop\BackTest_Data")
DEFAULT_INDICATOR_PATH = DEFAULT_DATA_ROOT / "indicator_cache" / "daily_indicators.parquet"
DEFAULT_POOL_DIR = DEFAULT_DATA_ROOT / "pools"


def parse_extra_args(items: Optional[List[str]]) -> Dict[str, Any]:
    """
    Parse strategy kwargs from command line.

    Supported forms:
        --param key=value
        --param key:int=10
        --param key:float=0.75
        --param key:bool=true
        --param key:str=abc

    Examples:
        --param min_red_green_ratio:float=0.75
        --param target_count:int=15
        --param require_below_ma20:bool=true
    """
    if not items:
        return {}

    result: Dict[str, Any] = {}

    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --param format: {item}. Expected key=value.")

        left, value = item.split("=", 1)

        if ":" in left:
            key, typ = left.split(":", 1)
            typ = typ.lower().strip()
        else:
            key, typ = left.strip(), "str"

        key = key.strip()
        value = value.strip()

        if not key:
            raise ValueError(f"Invalid --param key: {item}")

        if typ == "int":
            result[key] = int(value)
        elif typ == "float":
            result[key] = float(value)
        elif typ == "bool":
            result[key] = value.lower() in {"1", "true", "yes", "y", "on"}
        elif typ == "str":
            result[key] = value
        else:
            raise ValueError(f"Unsupported --param type: {typ}. Use int/float/bool/str.")

    return result


def load_strategy_func(strategy_name: str):
    strategy_path = PROJECT_ROOT / "selection_strategies" / f"{strategy_name}.py"

    if not strategy_path.exists():
        raise FileNotFoundError(
            f"Strategy file not found: {strategy_path}\n"
            f"Expected: selection_strategies/{strategy_name}.py"
        )

    module_name = f"selection_strategies.{strategy_name}"

    spec = importlib.util.spec_from_file_location(module_name, strategy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load strategy spec from: {strategy_path}")

    module = importlib.util.module_from_spec(spec)

    # Make project root importable.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    spec.loader.exec_module(module)

    if hasattr(module, "SELECT_FUNC"):
        func = getattr(module, "SELECT_FUNC")
    elif hasattr(module, "select"):
        func = getattr(module, "select")
    else:
        raise AttributeError(
            f"Strategy {strategy_name} must define SELECT_FUNC or select()."
        )

    if not callable(func):
        raise TypeError(f"Strategy function for {strategy_name} is not callable.")

    return func


def normalize_symbol(x) -> str:
    if pd.isna(x):
        return ""

    s = str(x).strip().upper()
    digits = "".join(ch for ch in s if ch.isdigit())

    if len(digits) >= 6:
        return digits[-6:]

    if len(digits) > 0:
        return digits.zfill(6)

    return ""


def load_indicator_cache(indicator_path: Path) -> pd.DataFrame:
    if not indicator_path.exists():
        raise FileNotFoundError(
            f"Indicator cache not found: {indicator_path}\n"
            f"Please run scripts/build_indicators.py first."
        )

    df = pd.read_parquet(indicator_path)

    if df.empty:
        raise RuntimeError(f"Indicator cache is empty: {indicator_path}")

    required = ["symbol", "date"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Indicator cache missing required columns: {missing}\n"
            f"columns={list(df.columns)}"
        )

    df = df.copy()
    df["symbol"] = df["symbol"].map(normalize_symbol)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date"])
    df = df[df["symbol"] != ""]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    return df


def apply_date_filter(
    df: pd.DataFrame,
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    out = df

    if start_date:
        start_ts = pd.to_datetime(start_date)
        out = out[out["date"] >= start_ts]

    if end_date:
        end_ts = pd.to_datetime(end_date)
        out = out[out["date"] <= end_ts]

    return out.copy()


def ensure_output_columns(df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    out = df.copy()

    if "selected" not in out.columns:
        raise ValueError(
            f"Strategy output missing required column: selected. "
            f"Strategy={strategy_name}"
        )

    out["selected"] = pd.to_numeric(out["selected"], errors="coerce").fillna(0).astype(int)

    if "selected_score_base" not in out.columns:
        out["selected_score_base"] = out["selected"]

    if "score_rank_key" not in out.columns:
        out["score_rank_key"] = 0.0

    if "score_pct" not in out.columns:
        out["score_pct"] = 0.0

    if "selection_strategy" not in out.columns:
        out["selection_strategy"] = strategy_name

    return out


def build_pool(
    indicators: pd.DataFrame,
    strategy_name: str,
    strategy_kwargs: Dict[str, Any],
    keep_all_rows: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    select_func = load_strategy_func(strategy_name)

    parts: List[pd.DataFrame] = []
    errors: List[Dict[str, str]] = []

    grouped = indicators.groupby("symbol", sort=True)
    iterator = grouped

    if progress:
        iterator = tqdm(grouped, total=indicators["symbol"].nunique(), desc="Build pool by symbol")

    for symbol, g in iterator:
        g = g.sort_values("date").reset_index(drop=True)

        try:
            selected_df = select_func(g, **strategy_kwargs)

            if selected_df is None or not isinstance(selected_df, pd.DataFrame):
                raise TypeError(
                    f"Strategy returned {type(selected_df)}, expected pandas.DataFrame"
                )

            selected_df = ensure_output_columns(selected_df, strategy_name)

            if keep_all_rows:
                part = selected_df.copy()
            else:
                part = selected_df[selected_df["selected"] == 1].copy()

            if not part.empty:
                parts.append(part)

        except Exception as e:
            errors.append({"symbol": str(symbol), "error": repr(e)})
            print(f"[WARN] strategy failed for symbol={symbol}: {e}")

    if errors:
        print(f"[WARN] strategy failed symbols: {len(errors):,}")

    if not parts:
        print("[WARN] No selected rows generated.")
        return pd.DataFrame()

    pool = pd.concat(parts, ignore_index=True)
    pool = pool.sort_values(["date", "score_rank_key", "symbol"], ascending=[True, False, True])
    pool = pool.reset_index(drop=True)

    return pool


def save_outputs(
    pool: pd.DataFrame,
    output_dir: Path,
    strategy_name: str,
    save_csv: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{strategy_name}_pool.parquet"
    csv_path = output_dir / f"{strategy_name}_pool.csv"
    meta_path = output_dir / f"{strategy_name}_pool.meta.json"

    pool.to_parquet(parquet_path, index=False)

    if save_csv:
        pool.to_csv(csv_path, index=False, encoding="utf-8-sig")

    meta = {
        "strategy_name": strategy_name,
        "rows": int(len(pool)),
        "symbols": int(pool["symbol"].nunique()) if "symbol" in pool.columns and not pool.empty else 0,
        "date_min": str(pool["date"].min()) if "date" in pool.columns and not pool.empty else None,
        "date_max": str(pool["date"].max()) if "date" in pool.columns and not pool.empty else None,
        "columns": list(pool.columns),
    }

    if metadata:
        meta.update(metadata)

    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    paths = {
        "parquet": parquet_path,
        "meta": meta_path,
    }

    if save_csv:
        paths["csv"] = csv_path

    return paths


def print_pool_summary(pool: pd.DataFrame, strategy_name: str) -> None:
    print("\n========== POOL SUMMARY ==========")
    print(f"strategy: {strategy_name}")
    print(f"rows:     {len(pool):,}")

    if pool.empty:
        return

    if "symbol" in pool.columns:
        print(f"symbols:  {pool['symbol'].nunique():,}")

    if "date" in pool.columns:
        print(f"dates:    {pool['date'].nunique():,}")
        print(f"range:    {pool['date'].min()} -> {pool['date'].max()}")

    print("\nselected by date sample:")
    by_date = pool.groupby("date")["symbol"].count().reset_index(name="count")
    print(by_date.tail(20).to_string(index=False))

    if "score_rank_key" in pool.columns:
        print("\nscore_rank_key describe:")
        print(pool["score_rank_key"].describe().to_string())

    print("\ncolumns:")
    print(list(pool.columns))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Strategy module name under selection_strategies, without .py",
    )

    parser.add_argument(
        "--indicator-path",
        type=str,
        default=str(DEFAULT_INDICATOR_PATH),
        help="Path to daily_indicators.parquet",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_POOL_DIR),
        help="Directory to save pool outputs.",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date, e.g. 2021-01-01",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date, e.g. 2026-05-10",
    )

    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help=(
            "Strategy parameter. Can be repeated. "
            "Formats: key=value, key:int=10, key:float=0.75, key:bool=true"
        ),
    )

    parser.add_argument(
        "--keep-all-rows",
        action="store_true",
        help="Save all strategy output rows instead of selected rows only.",
    )

    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not save CSV, only parquet.",
    )

    args = parser.parse_args()

    strategy_name = args.strategy
    indicator_path = Path(args.indicator_path)
    output_dir = Path(args.output_dir)
    strategy_kwargs = parse_extra_args(args.param)

    print("[INFO] Loading indicator cache...")
    indicators = load_indicator_cache(indicator_path)

    print(f"[INFO] indicator rows: {len(indicators):,}")
    print(f"[INFO] indicator symbols: {indicators['symbol'].nunique():,}")
    print(f"[INFO] indicator date range: {indicators['date'].min()} -> {indicators['date'].max()}")

    indicators = apply_date_filter(indicators, args.start_date, args.end_date)

    if indicators.empty:
        raise RuntimeError("Indicator data is empty after date filtering.")

    print(f"[INFO] rows after date filter: {len(indicators):,}")
    print(f"[INFO] date range after filter: {indicators['date'].min()} -> {indicators['date'].max()}")

    print("[INFO] Strategy params:")
    print(json.dumps(strategy_kwargs, ensure_ascii=False, indent=2))

    print("[INFO] Building pool...")
    pool = build_pool(
        indicators=indicators,
        strategy_name=strategy_name,
        strategy_kwargs=strategy_kwargs,
        keep_all_rows=args.keep_all_rows,
        progress=True,
    )

    print_pool_summary(pool, strategy_name)

    print("[INFO] Saving outputs...")
    paths = save_outputs(
        pool=pool,
        output_dir=output_dir,
        strategy_name=strategy_name,
        save_csv=not args.no_csv,
        metadata={
            "indicator_path": str(indicator_path),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "strategy_kwargs": strategy_kwargs,
            "keep_all_rows": bool(args.keep_all_rows),
        },
    )

    print("\n========== OUTPUT FILES ==========")
    for k, p in paths.items():
        print(f"[OK] {k}: {p}")


if __name__ == "__main__":
    main()