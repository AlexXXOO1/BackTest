# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_POOL_PATH = Path(
    r"C:\Users\zyf37\Desktop\BackTest_Data\pools\renko_chart_select_strategy_v4_pool.parquet"
)


def main():
    parser = argparse.ArgumentParser(description="Inspect pool parquet/csv content.")
    parser.add_argument(
        "--pool-path",
        type=str,
        default=str(DEFAULT_POOL_PATH),
        help="Pool file path. Supports .parquet and .csv.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Optional target date, for example 2026-05-08.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=30,
        help="Number of rows to print.",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default="",
        help="Optional comma-separated columns to print.",
    )
    args = parser.parse_args()

    pool_path = Path(args.pool_path)

    if not pool_path.exists():
        raise FileNotFoundError(f"Pool file not found: {pool_path}")

    if pool_path.suffix.lower() == ".csv":
        df = pd.read_csv(pool_path)
    else:
        df = pd.read_parquet(pool_path)

    print("========== BASIC INFO ==========")
    print(f"pool_path: {pool_path}")
    print(f"rows: {len(df):,}")
    print(f"columns: {len(df.columns):,}")
    print("columns:")
    print(df.columns.tolist())

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        print(f"date range: {df['date'].min()} -> {df['date'].max()}")
        print(f"trading days: {df['date'].nunique():,}")

    if "symbol" in df.columns:
        print(f"symbols: {df['symbol'].nunique():,}")
    elif "code" in df.columns:
        print(f"codes: {df['code'].nunique():,}")

    if "selected" in df.columns:
        print("\n========== SELECTED COUNT ==========")
        print(df["selected"].value_counts(dropna=False).to_string())

    if "selection_strategy" in df.columns:
        print("\n========== STRATEGY COUNT ==========")
        print(df["selection_strategy"].value_counts(dropna=False).to_string())

    if "v4_hint_label" in df.columns:
        print("\n========== V4 HINT LABEL COUNT ==========")
        print(df["v4_hint_label"].value_counts(dropna=False).to_string())

    if "date" in df.columns:
        print("\n========== DAILY COUNT TAIL ==========")
        daily_count = df.groupby("date").size().reset_index(name="pool_count")
        print(daily_count.tail(20).to_string(index=False))

    view = df.copy()

    if args.date and "date" in view.columns:
        target_date = pd.to_datetime(args.date)
        view = view[view["date"] == target_date].copy()
        print(f"\n========== TARGET DATE: {target_date.date()} ==========")
        print(f"rows: {len(view):,}")

        if "v4_hint_label" in view.columns:
            print("\nhint label count:")
            print(view["v4_hint_label"].value_counts(dropna=False).to_string())

    default_cols = [
        "date",
        "symbol",
        "code",
        "close",
        "daily_return_pct",
        "v4_close_to_ma5",
        "v4_brk",
        "v4_crh",
        "v4_pgh",
        "v4_up_hint_score",
        "v4_risk_hint_score",
        "v4_net_hint_score",
        "v4_hint_label",
        "score_rank_key",
        "score_pct",
    ]

    if args.columns.strip():
        show_cols = [c.strip() for c in args.columns.split(",") if c.strip()]
    else:
        show_cols = [c for c in default_cols if c in view.columns]

    if not show_cols:
        show_cols = view.columns.tolist()[:20]

    if "score_rank_key" in view.columns:
        view = view.sort_values(["date", "score_rank_key"], ascending=[True, False])
    elif "v4_net_hint_score" in view.columns:
        view = view.sort_values(["date", "v4_net_hint_score"], ascending=[True, False])
    elif "date" in view.columns:
        view = view.sort_values(["date"])

    print("\n========== PREVIEW ==========")
    print(view[show_cols].head(args.head).to_string(index=False))


if __name__ == "__main__":
    main()
