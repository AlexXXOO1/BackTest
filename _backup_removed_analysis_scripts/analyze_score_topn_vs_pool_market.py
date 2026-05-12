# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze_tools.analyze_multi_pool_compare import (  # noqa: E402
    attach_returns_to_pools,
    build_market_daily,
    build_pool_daily,
    load_market_cache,
    load_pool,
    parse_horizons,
    safe_mean,
    safe_median,
    weighted_avg,
)

DEFAULT_DATA_ROOT = Path(r"C:\Users\zyf37\Desktop\BackTest_Data")
DEFAULT_POOL_PATH = DEFAULT_DATA_ROOT / "pools" / "renko_chart_select_strategy_v4_pool.parquet"
DEFAULT_MARKET_CACHE_DIR = DEFAULT_DATA_ROOT / "market_cache" / "daily_bars_by_symbol"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "output" / "score_topn_vs_pool_market"


def save_json(path: Path, obj: dict[str, Any]) -> None:
    def conv(x):
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            if math.isnan(float(x)):
                return None
            return float(x)
        if isinstance(x, pd.Timestamp):
            return x.strftime("%Y-%m-%d")
        return str(x)

    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=conv), encoding="utf-8")


def build_score_topn_pool(
    pool: pd.DataFrame,
    score_col: str,
    score_threshold: float,
    top_n: int,
) -> pd.DataFrame:
    if score_col not in pool.columns:
        raise KeyError(f"score column not found: {score_col}")

    df = pool.copy()
    df["_score_value"] = pd.to_numeric(df[score_col], errors="coerce")
    df = df[df["_score_value"].notna()].copy()
    df = df[df["_score_value"] >= float(score_threshold)].copy()

    if df.empty:
        out = pool.head(0).copy()
        out["score_col"] = score_col
        out["score_value"] = pd.Series(dtype="float64")
        out["score_rank_in_day"] = pd.Series(dtype="float64")
        return out

    sort_cols = ["date", "_score_value", "code"]
    df = df.sort_values(sort_cols, ascending=[True, False, True]).copy()
    df["score_rank_in_day"] = df.groupby("date").cumcount() + 1
    df = df[df["score_rank_in_day"] <= int(top_n)].copy()
    df["score_col"] = score_col
    df["score_value"] = df["_score_value"]
    df = df.drop(columns=["_score_value"])
    return df.reset_index(drop=True)


def _prefixed_daily(pool_daily: pd.DataFrame, pool_name: str, prefix: str) -> pd.DataFrame:
    d = pool_daily[pool_daily["pool_name"].astype(str) == str(pool_name)].copy()
    if d.empty:
        return pd.DataFrame(columns=["date", "horizon"])

    keep = [
        "date",
        "horizon",
        "signal_count",
        "valid_count",
        "pool_avg_return_pct",
        "pool_median_return_pct",
        "pool_up_count",
        "pool_up_ratio",
    ]
    keep = [c for c in keep if c in d.columns]
    d = d[keep].copy()
    d = d.rename(
        columns={
            "signal_count": f"{prefix}_count",
            "valid_count": f"{prefix}_valid_count",
            "pool_avg_return_pct": f"{prefix}_avg_return_pct",
            "pool_median_return_pct": f"{prefix}_median_return_pct",
            "pool_up_count": f"{prefix}_up_count",
            "pool_up_ratio": f"{prefix}_up_ratio",
        }
    )
    return d


def build_daily_compare(
    full_pool_fwd: pd.DataFrame,
    topn_fwd: pd.DataFrame,
    market_daily: pd.DataFrame,
    horizons: list[int],
    score_col: str,
    score_threshold: float,
    top_n: int,
) -> pd.DataFrame:
    full_pool_daily = build_pool_daily(full_pool_fwd, market_daily, horizons)
    topn_pool_daily = build_pool_daily(topn_fwd, market_daily, horizons) if not topn_fwd.empty else pd.DataFrame()

    full_d = _prefixed_daily(full_pool_daily, "full_pool", "full_pool")
    topn_d = _prefixed_daily(topn_pool_daily, "score_topn", "topn") if not topn_pool_daily.empty else pd.DataFrame(columns=["date", "horizon"])

    base_dates = pd.DataFrame({"date": sorted(full_pool_fwd["date"].dropna().unique())})
    base_h = pd.DataFrame({"horizon": [f"T{h}" for h in horizons]})
    grid = base_dates.merge(base_h, how="cross")

    out = grid.merge(full_d, on=["date", "horizon"], how="left")
    out = out.merge(topn_d, on=["date", "horizon"], how="left")
    out = out.merge(
        market_daily[
            [
                "date",
                "horizon",
                "market_count",
                "market_avg_return_pct",
                "market_median_return_pct",
                "market_up_ratio",
            ]
        ],
        on=["date", "horizon"],
        how="left",
    )

    count_defaults = {
        "full_pool_count": 0,
        "full_pool_valid_count": 0,
        "full_pool_up_count": 0,
        "topn_count": 0,
        "topn_valid_count": 0,
        "topn_up_count": 0,
    }
    value_defaults = {
        "full_pool_avg_return_pct": np.nan,
        "full_pool_median_return_pct": np.nan,
        "full_pool_up_ratio": np.nan,
        "topn_avg_return_pct": np.nan,
        "topn_median_return_pct": np.nan,
        "topn_up_ratio": np.nan,
    }

    for c, default in count_defaults.items():
        if c not in out.columns:
            out[c] = default
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(default).astype(int)

    for c, default in value_defaults.items():
        if c not in out.columns:
            out[c] = default
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["topn_active"] = out["topn_count"] > 0
    out["topn_cash_day"] = ~out["topn_active"]
    out["score_col"] = score_col
    out["score_threshold"] = float(score_threshold)
    out["top_n"] = int(top_n)

    out["topn_minus_full_pool_return_pct"] = out["topn_avg_return_pct"] - out["full_pool_avg_return_pct"]
    out["topn_minus_full_pool_up_ratio"] = out["topn_up_ratio"] - out["full_pool_up_ratio"]
    out["topn_excess_market_return_pct"] = out["topn_avg_return_pct"] - out["market_avg_return_pct"]
    out["topn_excess_market_up_ratio"] = out["topn_up_ratio"] - out["market_up_ratio"]
    out["full_pool_excess_market_return_pct"] = out["full_pool_avg_return_pct"] - out["market_avg_return_pct"]
    out["full_pool_excess_market_up_ratio"] = out["full_pool_up_ratio"] - out["market_up_ratio"]

    return out.sort_values(["date", "horizon"]).reset_index(drop=True)


def build_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for horizon, g in daily.groupby("horizon", dropna=False):
        active = g[(g["topn_count"] > 0) & pd.to_numeric(g["topn_avg_return_pct"], errors="coerce").notna()].copy()
        compare_pool = active[pd.to_numeric(active["topn_minus_full_pool_return_pct"], errors="coerce").notna()].copy()
        compare_market = active[pd.to_numeric(active["topn_excess_market_return_pct"], errors="coerce").notna()].copy()

        full_valid = g[pd.to_numeric(g["full_pool_avg_return_pct"], errors="coerce").notna()].copy()

        weighted_topn = weighted_avg(active["topn_avg_return_pct"], active["topn_valid_count"]) if not active.empty else np.nan
        weighted_full_on_active = weighted_avg(active["full_pool_avg_return_pct"], active["full_pool_valid_count"]) if not active.empty else np.nan

        rows.append(
            {
                "horizon": horizon,
                "score_col": g["score_col"].dropna().astype(str).iloc[0] if "score_col" in g.columns and g["score_col"].notna().any() else "",
                "score_threshold": float(pd.to_numeric(g["score_threshold"], errors="coerce").dropna().iloc[0]) if "score_threshold" in g.columns and pd.to_numeric(g["score_threshold"], errors="coerce").notna().any() else np.nan,
                "top_n": int(pd.to_numeric(g["top_n"], errors="coerce").dropna().iloc[0]) if "top_n" in g.columns and pd.to_numeric(g["top_n"], errors="coerce").notna().any() else np.nan,
                "pool_days": int(g["date"].nunique()),
                "topn_active_days": int((g["topn_count"] > 0).sum()),
                "topn_valid_days": int(active["date"].nunique()),
                "topn_cash_days_not_counted": int((g["topn_count"] <= 0).sum()),
                "topn_cash_day_ratio": float((g["topn_count"] <= 0).mean()) if len(g) else np.nan,
                "full_pool_valid_days": int(full_valid["date"].nunique()),
                "topn_vs_full_pool_compare_days": int(compare_pool["date"].nunique()),
                "topn_vs_market_compare_days": int(compare_market["date"].nunique()),
                "topn_total_rows": int(active["topn_valid_count"].sum()) if not active.empty else 0,
                "full_pool_total_rows_on_topn_days": int(active["full_pool_valid_count"].sum()) if not active.empty else 0,
                "topn_mean_count_on_active_days": safe_mean(active["topn_count"]),
                "topn_median_count_on_active_days": safe_median(active["topn_count"]),
                "full_pool_mean_count_on_topn_days": safe_mean(active["full_pool_count"]),
                "full_pool_mean_count_all_days": safe_mean(g["full_pool_count"]),
                "daily_mean_topn_return_pct": safe_mean(active["topn_avg_return_pct"]),
                "daily_median_topn_return_pct": safe_median(active["topn_avg_return_pct"]),
                "daily_mean_full_pool_return_on_topn_days_pct": safe_mean(active["full_pool_avg_return_pct"]),
                "daily_mean_market_return_on_topn_days_pct": safe_mean(active["market_avg_return_pct"]),
                "daily_mean_topn_minus_full_pool_return_pct": safe_mean(compare_pool["topn_minus_full_pool_return_pct"]),
                "daily_median_topn_minus_full_pool_return_pct": safe_median(compare_pool["topn_minus_full_pool_return_pct"]),
                "daily_mean_topn_excess_market_return_pct": safe_mean(compare_market["topn_excess_market_return_pct"]),
                "daily_median_topn_excess_market_return_pct": safe_median(compare_market["topn_excess_market_return_pct"]),
                "daily_mean_full_pool_excess_market_on_topn_days_pct": safe_mean(active["full_pool_excess_market_return_pct"]),
                "daily_mean_topn_up_ratio": safe_mean(active["topn_up_ratio"]),
                "daily_mean_full_pool_up_ratio_on_topn_days": safe_mean(active["full_pool_up_ratio"]),
                "daily_mean_market_up_ratio_on_topn_days": safe_mean(active["market_up_ratio"]),
                "daily_mean_topn_minus_full_pool_up_ratio": safe_mean(compare_pool["topn_minus_full_pool_up_ratio"]),
                "daily_mean_topn_excess_market_up_ratio": safe_mean(compare_market["topn_excess_market_up_ratio"]),
                "topn_win_full_pool_days": int((compare_pool["topn_minus_full_pool_return_pct"] > 0).sum()),
                "topn_win_full_pool_day_ratio": float((compare_pool["topn_minus_full_pool_return_pct"] > 0).mean()) if len(compare_pool) else np.nan,
                "topn_win_market_days": int((compare_market["topn_excess_market_return_pct"] > 0).sum()),
                "topn_win_market_day_ratio": float((compare_market["topn_excess_market_return_pct"] > 0).mean()) if len(compare_market) else np.nan,
                "topn_positive_return_days": int((active["topn_avg_return_pct"] > 0).sum()) if not active.empty else 0,
                "topn_positive_return_day_ratio": float((active["topn_avg_return_pct"] > 0).mean()) if len(active) else np.nan,
                "full_pool_overall_daily_mean_return_pct": safe_mean(full_valid["full_pool_avg_return_pct"]),
                "full_pool_overall_daily_mean_excess_market_pct": safe_mean(full_valid["full_pool_excess_market_return_pct"]),
                "weighted_topn_avg_return_pct": weighted_topn,
                "weighted_full_pool_avg_return_on_topn_days_pct": weighted_full_on_active,
                "weighted_topn_minus_full_pool_return_pct": weighted_topn - weighted_full_on_active if pd.notna(weighted_topn) and pd.notna(weighted_full_on_active) else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate daily score Top-N pool against full pool and full market.")
    parser.add_argument("--pool-path", type=Path, default=DEFAULT_POOL_PATH)
    parser.add_argument("--pool-name", default="")
    parser.add_argument("--score-col", required=True)
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--market-cache-dir", type=Path, default=DEFAULT_MARKET_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--horizons", default="1,2,3")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--selected-only", action="store_true", default=True)
    parser.add_argument("--include-unselected", action="store_false", dest="selected_only")

    args = parser.parse_args()

    if args.top_n <= 0:
        raise ValueError("top-n must be positive")

    pool_name = args.pool_name.strip() or args.pool_path.stem
    horizons = parse_horizons(args.horizons)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Score Top-N vs Full Pool vs Market")
    print("=" * 100)
    print(f"pool_name        : {pool_name}")
    print(f"pool_path        : {args.pool_path}")
    print(f"score_col        : {args.score_col}")
    print(f"score_threshold  : {args.score_threshold}")
    print(f"top_n            : {args.top_n}")
    print(f"market_cache_dir : {args.market_cache_dir}")
    print(f"output_dir       : {args.output_dir}")
    print(f"date_range       : {args.start_date} -> {args.end_date}")
    print(f"horizons         : {horizons}")
    print(f"selected_only    : {args.selected_only}")
    print("=" * 100)

    pool = load_pool(
        pool_path=args.pool_path,
        pool_name=pool_name,
        selected_only=args.selected_only,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if pool.empty:
        raise RuntimeError("pool is empty after filters")

    full_pool = pool.copy()
    full_pool["pool_name"] = "full_pool"

    topn_pool = build_score_topn_pool(
        pool=pool,
        score_col=args.score_col,
        score_threshold=args.score_threshold,
        top_n=args.top_n,
    )
    topn_pool["pool_name"] = "score_topn"

    print(f"[INFO] full_pool rows={len(full_pool):,}, days={full_pool['date'].nunique():,}, symbols={full_pool['code'].nunique():,}")
    print(f"[INFO] score_topn rows={len(topn_pool):,}, active_days={topn_pool['date'].nunique() if not topn_pool.empty else 0:,}, symbols={topn_pool['code'].nunique() if not topn_pool.empty else 0:,}")

    market = load_market_cache(
        market_cache_dir=args.market_cache_dir,
        horizons=horizons,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
    )

    full_pool_fwd = attach_returns_to_pools(full_pool, market, horizons)
    topn_fwd = attach_returns_to_pools(topn_pool, market, horizons) if not topn_pool.empty else topn_pool.copy()
    market_daily = build_market_daily(market, horizons)

    daily_compare = build_daily_compare(
        full_pool_fwd=full_pool_fwd,
        topn_fwd=topn_fwd,
        market_daily=market_daily,
        horizons=horizons,
        score_col=args.score_col,
        score_threshold=args.score_threshold,
        top_n=args.top_n,
    )
    summary = build_summary(daily_compare)

    out_paths = {
        "score_topn_rows_with_returns": args.output_dir / "0_score_topn_rows_with_returns.parquet",
        "daily_compare": args.output_dir / "1_daily_topn_vs_pool_market.csv",
        "summary": args.output_dir / "2_summary.csv",
        "diagnostics": args.output_dir / "diagnostics.json",
    }

    topn_fwd.to_parquet(out_paths["score_topn_rows_with_returns"], index=False)
    daily_compare.to_csv(out_paths["daily_compare"], index=False, encoding="utf-8-sig")
    summary.to_csv(out_paths["summary"], index=False, encoding="utf-8-sig")

    diagnostics = {
        "pool_name": pool_name,
        "pool_path": str(args.pool_path),
        "score_col": args.score_col,
        "score_threshold": args.score_threshold,
        "top_n": args.top_n,
        "market_cache_dir": str(args.market_cache_dir),
        "output_dir": str(args.output_dir),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "horizons": horizons,
        "selected_only": args.selected_only,
        "full_pool_rows": int(len(full_pool)),
        "full_pool_days": int(full_pool["date"].nunique()),
        "score_topn_rows": int(len(topn_pool)),
        "score_topn_active_days": int(topn_pool["date"].nunique()) if not topn_pool.empty else 0,
        "market_rows": int(len(market)),
        "market_symbols": int(market["code"].nunique()),
        "outputs": {k: str(v) for k, v in out_paths.items()},
    }
    save_json(out_paths["diagnostics"], diagnostics)

    print("\nDONE")
    for k, v in out_paths.items():
        print(f"{k}: {v}")

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
