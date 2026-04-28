from __future__ import annotations

r"""
Analyze whether each v1 selection indicator is positively or negatively associated
with forward returns, using the same selected signal pool as the control universe.

Recommended location:
    Put this file in your project root, for example:
    C:\Users\zyf37\Desktop\Trade Backtest v1.0.0\analyze_v1_indicator_attribution.py

Default behavior:
    1. Analyze renko_chart_select_strategy_v1 signals from 2024-01-01 to 2025-12-31.
    2. If the unified pool file does not exist, call selector.py to build it.
    3. Load all selected hits from the pool.
    4. Calculate forward returns for every hit using the same market data cache.
    5. For every indicator, compare True vs False on the exact same selected-hit universe.
    6. Also calculate a date-neutral paired comparison:
       for each signal date, mean_return(True group) - mean_return(False group),
       then average those daily differences.

Why this controls variables:
    - The universe is fixed to v1 selected hits only.
    - The holding horizon is fixed per metric.
    - The same signal dates are used for paired comparison where both True and False exist.
    - No future condition is used to decide whether a row enters the analysis sample.
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from data_store import MarketDataStore
from pool_store import PoolStore
from selection_strategies.renko_chart_select_strategy_v1 import DEFAULT_RENKO_CHART_SELECT_WEIGHTS


DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_STRATEGY = "renko_chart_select_strategy_v1"
DEFAULT_OUTPUT = "output/v1_indicator_attribution_20240101_20251231.csv"

# These are the v1 factors that have explicit weights.
WEIGHTED_INDICATORS = list(DEFAULT_RENKO_CHART_SELECT_WEIGHTS.keys())

# These are v1 hard conditions. In the selected pool, they are usually all True,
# so they often cannot be compared against a False control group.
HARD_CONDITION_INDICATORS = [
    "hard_brick_turn_strong",
    "j_momentum_or_low",
    "small_rise_long_red_brick",
]

# Extra diagnostic fields that are useful for attribution but are not all weighted.
EXTRA_ANALYSIS_INDICATORS = [
    "risk_tag_any",
    "risk_filter_pass",
    "condition6_hard_pass",
    "condition9_hard_pass",
]

RETURN_METRICS = [
    # From T0 close: useful for pure signal-forward attribution.
    "t0_close_to_t1_open_ret_pct",
    "t0_close_to_t1_close_ret_pct",
    "t0_close_to_t2_close_ret_pct",
    "t0_close_to_t3_close_ret_pct",
    # From T+1 open: closer to your executable strategy return.
    "t1_open_to_t1_close_ret_pct",
    "t1_open_to_t2_close_ret_pct",
    "t1_open_to_t3_close_ret_pct",
]


@dataclass(frozen=True)
class Args:
    start_date: str
    end_date: str
    strategy: str
    txt_dir: Path
    market_cache_dir: Path
    pools_dir: Path
    output: Path
    rebuild_pool: bool


def progress_iter(iterable, total: int | None = None, desc: str = ""):
    """Return a tqdm progress iterator when tqdm is installed; otherwise return iterable."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, ncols=100)


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Analyze v1 indicator attribution with controlled comparisons.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--txt-dir", type=Path, default=Path("data"))
    parser.add_argument("--market-cache-dir", type=Path, default=Path("data/market_cache/daily_bars_by_symbol"))
    parser.add_argument("--pools-dir", type=Path, default=Path("pools"))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--rebuild-pool", action="store_true", help="Force selector.py to rebuild the pool range.")
    ns = parser.parse_args()
    return Args(
        start_date=ns.start_date,
        end_date=ns.end_date,
        strategy=ns.strategy,
        txt_dir=ns.txt_dir,
        market_cache_dir=ns.market_cache_dir,
        pools_dir=ns.pools_dir,
        output=ns.output,
        rebuild_pool=bool(ns.rebuild_pool),
    )


def run_selector_if_needed(args: Args) -> Path:
    """Build the unified selection pool only when missing, or when --rebuild-pool is set."""
    pool_path = PoolStore(args.pools_dir).pool_path(args.strategy)

    need_build = args.rebuild_pool or (not pool_path.exists())

    if not need_build:
        try:
            pool_df = PoolStore(args.pools_dir).read(args.strategy)
            if pool_df.empty or "date" not in pool_df.columns:
                need_build = True
            else:
                dates = pd.to_datetime(pool_df["date"]).dt.normalize()
                start_ts = pd.Timestamp(args.start_date).normalize()
                end_ts = pd.Timestamp(args.end_date).normalize()
                # If there is no row inside the target range, build the pool.
                # Note: a date with zero signals cannot be distinguished from a missing date
                # by reading a selected-only pool, so use --rebuild-pool when you want certainty.
                need_build = not bool(((dates >= start_ts) & (dates <= end_ts)).any())
        except Exception:
            need_build = True

    if need_build:
        cmd = [
            sys.executable,
            "selector.py",
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--strategy",
            args.strategy,
            "--txt-dir",
            str(args.txt_dir),
            "--market-cache-dir",
            str(args.market_cache_dir),
            "--pools-dir",
            str(args.pools_dir),
        ]
        if args.rebuild_pool:
            cmd.append("--overwrite")

        print("Pool missing or rebuild requested. Running:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        print(f"Pool exists, skip selector build: {pool_path}")
        print("If you recently changed v1 code, run this script with --rebuild-pool.")

    return pool_path


def load_signal_pool(args: Args) -> pd.DataFrame:
    pool = PoolStore(args.pools_dir).read(args.strategy)
    if pool.empty:
        raise RuntimeError(f"Pool is empty: {PoolStore(args.pools_dir).pool_path(args.strategy)}")

    pool = pool.copy()
    pool["date"] = pd.to_datetime(pool["date"])
    start_ts = pd.Timestamp(args.start_date)
    end_ts = pd.Timestamp(args.end_date)
    pool = pool[(pool["date"] >= start_ts) & (pool["date"] <= end_ts)].copy()

    if "selected" in pool.columns:
        pool = pool[pool["selected"].fillna(0).astype(int) == 1].copy()

    if pool.empty:
        raise RuntimeError(f"No selected signals found from {args.start_date} to {args.end_date}.")

    pool = pool.sort_values(["date", "symbol"]).reset_index(drop=True)
    return pool


def _safe_ret(numerator: float | int | None, denominator: float | int | None) -> float:
    try:
        numerator = float(numerator)
        denominator = float(denominator)
        if denominator == 0 or np.isnan(numerator) or np.isnan(denominator):
            return np.nan
        return (numerator / denominator - 1.0) * 100.0
    except Exception:
        return np.nan


def add_forward_returns(pool: pd.DataFrame, args: Args) -> pd.DataFrame:
    """Add T+1/T+2/T+3 forward return columns for every selected signal."""
    market = MarketDataStore(args.txt_dir, args.market_cache_dir)
    symbol_cache: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []

    for _, row in progress_iter(pool.iterrows(), total=len(pool), desc="Forward returns"):
        symbol = str(row.get("symbol") or Path(str(row.get("file", ""))).stem).upper()
        if symbol not in symbol_cache:
            df = market.get_symbol_data(symbol)
            if df.empty and row.get("file"):
                df = market.get_symbol_data(str(row.get("file")))
            if not df.empty:
                df = df.copy().sort_values("date").reset_index(drop=True)
                df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            symbol_cache[symbol] = df

        df = symbol_cache[symbol]
        out_row = row.to_dict()

        if df.empty:
            out_row["forward_data_status"] = "missing_symbol_data"
            rows.append(out_row)
            continue

        signal_date = pd.Timestamp(row["date"]).normalize()
        idx_list = df.index[df["date"] == signal_date].tolist()
        if not idx_list:
            out_row["forward_data_status"] = "missing_signal_date"
            rows.append(out_row)
            continue

        i = int(idx_list[0])
        t0 = df.iloc[i]
        t1 = df.iloc[i + 1] if i + 1 < len(df) else None
        t2 = df.iloc[i + 2] if i + 2 < len(df) else None
        t3 = df.iloc[i + 3] if i + 3 < len(df) else None

        t0_close = t0.get("close", np.nan)
        t1_open = t1.get("open", np.nan) if t1 is not None else np.nan
        t1_close = t1.get("close", np.nan) if t1 is not None else np.nan
        t2_close = t2.get("close", np.nan) if t2 is not None else np.nan
        t3_close = t3.get("close", np.nan) if t3 is not None else np.nan

        out_row["t0_close"] = t0_close
        out_row["t1_date"] = t1.get("date") if t1 is not None else pd.NaT
        out_row["t1_open"] = t1_open
        out_row["t1_close"] = t1_close
        out_row["t2_date"] = t2.get("date") if t2 is not None else pd.NaT
        out_row["t2_close"] = t2_close
        out_row["t3_date"] = t3.get("date") if t3 is not None else pd.NaT
        out_row["t3_close"] = t3_close

        out_row["t0_close_to_t1_open_ret_pct"] = _safe_ret(t1_open, t0_close)
        out_row["t0_close_to_t1_close_ret_pct"] = _safe_ret(t1_close, t0_close)
        out_row["t0_close_to_t2_close_ret_pct"] = _safe_ret(t2_close, t0_close)
        out_row["t0_close_to_t3_close_ret_pct"] = _safe_ret(t3_close, t0_close)
        out_row["t1_open_to_t1_close_ret_pct"] = _safe_ret(t1_close, t1_open)
        out_row["t1_open_to_t2_close_ret_pct"] = _safe_ret(t2_close, t1_open)
        out_row["t1_open_to_t3_close_ret_pct"] = _safe_ret(t3_close, t1_open)
        out_row["forward_data_status"] = "ok"
        rows.append(out_row)

    out = pd.DataFrame(rows)
    return out


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df[col].fillna(False).astype(bool)


def expected_direction_for_indicator(indicator: str, weight: float | None) -> str:
    if weight is None:
        return "diagnostic"
    if weight > 0:
        return "positive_expected_true_should_outperform"
    if weight < 0:
        return "negative_expected_true_should_underperform"
    return "neutral"


def evaluate_direction(weight: float | None, diff: float, true_n: int, false_n: int) -> str:
    if true_n == 0 or false_n == 0 or pd.isna(diff):
        return "not_evaluable_missing_true_or_false_group"
    if weight is None or weight == 0:
        return "diagnostic_only"
    if weight > 0:
        return "matches_expected" if diff > 0 else "opposite_expected"
    return "matches_expected" if diff < 0 else "opposite_expected"


def summarize_indicator_metric(df: pd.DataFrame, indicator: str, metric: str) -> dict:
    weight = DEFAULT_RENKO_CHART_SELECT_WEIGHTS.get(indicator)
    flag = bool_series(df, indicator)
    valid = df[metric].notna()
    true_values = df.loc[valid & flag, metric]
    false_values = df.loc[valid & ~flag, metric]

    true_mean = float(true_values.mean()) if len(true_values) else np.nan
    false_mean = float(false_values.mean()) if len(false_values) else np.nan
    diff = true_mean - false_mean if len(true_values) and len(false_values) else np.nan

    true_win_rate = float((true_values > 0).mean() * 100.0) if len(true_values) else np.nan
    false_win_rate = float((false_values > 0).mean() * 100.0) if len(false_values) else np.nan
    win_rate_diff = true_win_rate - false_win_rate if len(true_values) and len(false_values) else np.nan

    # Date-neutral paired comparison: only dates where both True and False groups exist.
    tmp = df.loc[valid, ["date", metric]].copy()
    tmp["flag"] = flag.loc[valid].values
    paired_diffs = []
    paired_win_diffs = []
    for _, g in tmp.groupby(pd.to_datetime(tmp["date"]).dt.normalize()):
        t = g.loc[g["flag"], metric]
        f = g.loc[~g["flag"], metric]
        if len(t) and len(f):
            paired_diffs.append(float(t.mean() - f.mean()))
            paired_win_diffs.append(float((t > 0).mean() * 100.0 - (f > 0).mean() * 100.0))

    paired_diff_mean = float(np.mean(paired_diffs)) if paired_diffs else np.nan
    paired_win_rate_diff_mean = float(np.mean(paired_win_diffs)) if paired_win_diffs else np.nan

    return {
        "indicator": indicator,
        "weight": weight if weight is not None else "",
        "expected_direction": expected_direction_for_indicator(indicator, weight),
        "return_metric": metric,
        "sample_count_valid_return": int(valid.sum()),
        "true_count": int((valid & flag).sum()),
        "false_count": int((valid & ~flag).sum()),
        "true_mean_ret_pct": true_mean,
        "false_mean_ret_pct": false_mean,
        "diff_true_minus_false_pct": diff,
        "true_win_rate_pct": true_win_rate,
        "false_win_rate_pct": false_win_rate,
        "win_rate_diff_true_minus_false_pct": win_rate_diff,
        "paired_dates_with_both_groups": int(len(paired_diffs)),
        "paired_date_neutral_diff_pct": paired_diff_mean,
        "paired_date_neutral_win_rate_diff_pct": paired_win_rate_diff_mean,
        "basic_conclusion": evaluate_direction(weight, diff, len(true_values), len(false_values)),
        "date_neutral_conclusion": evaluate_direction(weight, paired_diff_mean, len(paired_diffs), len(paired_diffs)),
        "control_note": "same_v1_selected_pool; fixed_return_horizon; paired_result_controls_signal_date_when_both_groups_exist",
    }


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    indicators = []
    for col in WEIGHTED_INDICATORS + HARD_CONDITION_INDICATORS + EXTRA_ANALYSIS_INDICATORS:
        if col not in indicators:
            indicators.append(col)

    tasks = [(indicator, metric) for indicator in indicators for metric in RETURN_METRICS]
    rows = []
    for indicator, metric in progress_iter(tasks, total=len(tasks), desc="Indicator summary"):
        rows.append(summarize_indicator_metric(df, indicator, metric))

    summary = pd.DataFrame(rows)

    # Put the executable T+1-open-to-T+3-close metric first because it matches your usual T+1 buy logic.
    metric_order = {m: i for i, m in enumerate([
        "t1_open_to_t3_close_ret_pct",
        "t1_open_to_t2_close_ret_pct",
        "t1_open_to_t1_close_ret_pct",
        "t0_close_to_t3_close_ret_pct",
        "t0_close_to_t2_close_ret_pct",
        "t0_close_to_t1_close_ret_pct",
        "t0_close_to_t1_open_ret_pct",
    ])}
    summary["metric_order"] = summary["return_metric"].map(metric_order).fillna(999).astype(int)
    summary["indicator_order"] = summary["indicator"].apply(lambda x: indicators.index(x) if x in indicators else 999)
    summary = summary.sort_values(["metric_order", "indicator_order"]).drop(columns=["metric_order", "indicator_order"])
    return summary.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    run_selector_if_needed(args)

    print("Loading selected signal pool...")
    pool = load_signal_pool(args)
    print(f"Selected signal rows: {len(pool)}")

    print("Calculating forward returns...")
    enriched = add_forward_returns(pool, args)
    ok_count = int((enriched["forward_data_status"] == "ok").sum())
    print(f"Rows with forward data: {ok_count} / {len(enriched)}")

    print("Building controlled indicator attribution summary...")
    summary = build_summary(enriched)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved summary CSV: {args.output.resolve()}")


if __name__ == "__main__":
    main()
