from __future__ import annotations

"""
Analyze B2 Confirm V0 forward returns:
1) T0 tail buy, compare T+1/T+2/T+3 close returns
2) T+1 open buy, compare T+1/T+2/T+3 close returns

Key fix in v2:
- T+1/T+2/T+3 are found by row position in each stock's local trading calendar,
  not by calendar dates. This avoids T+2/T+3 becoming empty because of weekends/holidays.
"""

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# =============================================================================
# Helpers
# =============================================================================


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_code(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if "#" in s:
        s = s.split("#")[-1]
    s = s.replace(".SZ", "").replace(".SH", "")
    s = s.replace("SZ", "").replace("SH", "")
    s = s.strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 6:
        return digits[-6:]
    return s.zfill(6)[-6:]


def extract_code_from_path(path: Path) -> str | None:
    stem = path.stem
    parts = stem.replace(".", "#").replace("_", "#").split("#")
    for p in reversed(parts):
        p = p.strip()
        if p.isdigit() and len(p) == 6:
            return p
    digits = "".join(ch for ch in stem if ch.isdigit())
    if len(digits) >= 6:
        return digits[-6:]
    return None


def safe_float(x: Any) -> float:
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def pct_ret(sell_price: float, buy_price: float) -> float:
    if pd.isna(sell_price) or pd.isna(buy_price) or buy_price <= 0:
        return np.nan
    return (sell_price / buy_price - 1.0) * 100.0


# =============================================================================
# Data loading
# =============================================================================


def standardize_bar_df(df: pd.DataFrame, code: str) -> pd.DataFrame:
    out = df.copy()

    rename_map = {
        "日期": "date",
        "时间": "date",
        "trade_date": "date",
        "datetime": "date",
        "Date": "date",
        "DATE": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "VOLUME": "volume",
        "AMOUNT": "amount",
    }
    out = out.rename(columns={c: rename_map[c] for c in out.columns if c in rename_map})

    required = {"date", "open", "high", "low", "close"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"{code} bar data missing required columns: {sorted(missing)}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out[out["date"].notna()].copy()

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["open", "high", "low", "close"]).copy()
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    out["code"] = code
    return out


def build_market_file_map(market_cache_dir: Path) -> dict[str, Path]:
    files = sorted(market_cache_dir.glob("*.parquet"))
    mp: dict[str, Path] = {}
    for p in files:
        code = extract_code_from_path(p)
        if code:
            mp[code] = p
    return mp


def load_one_bar(code: str, file_map: dict[str, Path], cache: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    if code in cache:
        return cache[code]
    path = file_map.get(code)
    if path is None or not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df = standardize_bar_df(df, code)
        cache[code] = df
        return df
    except Exception as e:
        cache[code] = pd.DataFrame({"_load_error": [str(e)]})
        return None


def load_pool(pool_path: Path, start_date: str | None, end_date: str | None, selected_only: bool) -> pd.DataFrame:
    if not pool_path.exists():
        raise FileNotFoundError(f"Pool file not found: {pool_path}")

    df = pd.read_parquet(pool_path)
    df = df.copy()

    if "code" not in df.columns:
        # Some older files may use symbol.
        if "symbol" in df.columns:
            df = df.rename(columns={"symbol": "code"})
        else:
            raise ValueError(f"Pool file must contain code column. columns={list(df.columns)}")

    if "date" not in df.columns:
        raise ValueError(f"Pool file must contain date column. columns={list(df.columns)}")

    df["code"] = df["code"].map(normalize_code)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df[df["date"].notna() & (df["code"] != "")].copy()

    if selected_only and "selected" in df.columns:
        df = df[df["selected"].astype(bool)].copy()

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date).normalize()].copy()
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date).normalize()].copy()

    # Avoid duplicate same stock same signal date.
    df = df.sort_values(["date", "code"]).drop_duplicates(subset=["date", "code"], keep="last").reset_index(drop=True)
    return df


# =============================================================================
# Forward return calculation
# =============================================================================


def get_bar_by_offset(bars: pd.DataFrame, signal_date: pd.Timestamp, offset: int) -> pd.Series | None:
    """Return row offset trading days after signal_date by positional index."""
    dates = bars["date"].values
    matches = np.where(dates == np.datetime64(signal_date))[0]
    if len(matches) == 0:
        return None
    idx = int(matches[-1]) + offset
    if idx < 0 or idx >= len(bars):
        return None
    return bars.iloc[idx]


def calc_one_signal(
    signal_row: pd.Series,
    bars: pd.DataFrame,
    t0_slippage_pct: float,
    t1_open_slippage_pct: float,
) -> dict[str, Any]:
    code = str(signal_row["code"])
    signal_date = pd.to_datetime(signal_row["date"]).normalize()

    t0 = get_bar_by_offset(bars, signal_date, 0)
    t1 = get_bar_by_offset(bars, signal_date, 1)
    t2 = get_bar_by_offset(bars, signal_date, 2)
    t3 = get_bar_by_offset(bars, signal_date, 3)

    base: dict[str, Any] = {
        "code": code,
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "has_t0": t0 is not None,
        "has_t1": t1 is not None,
        "has_t2": t2 is not None,
        "has_t3": t3 is not None,
    }

    # Carry useful pool columns into detail output.
    for c in [
        "name", "stock_name", "score", "score_pct", "watch_score", "selected",
        "daily_return_pct", "j", "J", "volume_ratio_prev", "upper_shadow_ratio",
        "b1_days_ago_for_b2", "selection_strategy",
    ]:
        if c in signal_row.index:
            base[f"pool_{c}"] = signal_row.get(c)

    if t0 is None:
        base["skip_reason"] = "missing_t0_bar"
        return base

    t0_close = safe_float(t0.get("close"))
    t0_open = safe_float(t0.get("open"))
    t0_buy_price = t0_close * (1.0 + t0_slippage_pct / 100.0)

    base.update({
        "t0_date": t0.get("date").strftime("%Y-%m-%d"),
        "t0_open": t0_open,
        "t0_close": t0_close,
        "t0_tail_buy_price": t0_buy_price,
        "skip_reason": "",
    })

    for label, row in [("t1", t1), ("t2", t2), ("t3", t3)]:
        if row is None:
            base[f"{label}_date"] = ""
            base[f"{label}_open"] = np.nan
            base[f"{label}_close"] = np.nan
        else:
            base[f"{label}_date"] = row.get("date").strftime("%Y-%m-%d")
            base[f"{label}_open"] = safe_float(row.get("open"))
            base[f"{label}_close"] = safe_float(row.get("close"))

    # T0 tail buy -> future closes.
    base["ret_t0_tail_to_t1_close_pct"] = pct_ret(base["t1_close"], t0_buy_price)
    base["ret_t0_tail_to_t2_close_pct"] = pct_ret(base["t2_close"], t0_buy_price)
    base["ret_t0_tail_to_t3_close_pct"] = pct_ret(base["t3_close"], t0_buy_price)

    t0_rets = [
        base["ret_t0_tail_to_t1_close_pct"],
        base["ret_t0_tail_to_t2_close_pct"],
        base["ret_t0_tail_to_t3_close_pct"],
    ]
    t0_rets_valid = [x for x in t0_rets if not pd.isna(x)]
    base["ret_t0_tail_to_t1_t3_best_close_pct"] = max(t0_rets_valid) if t0_rets_valid else np.nan
    base["ret_t0_tail_to_t1_t3_worst_close_pct"] = min(t0_rets_valid) if t0_rets_valid else np.nan

    # T+1 open buy -> T+1/T+2/T+3 closes.
    if t1 is not None:
        t1_open_buy_price = base["t1_open"] * (1.0 + t1_open_slippage_pct / 100.0)
        base["t1_open_buy_price"] = t1_open_buy_price
        base["ret_t1_open_to_t1_close_pct"] = pct_ret(base["t1_close"], t1_open_buy_price)
        base["ret_t1_open_to_t2_close_pct"] = pct_ret(base["t2_close"], t1_open_buy_price)
        base["ret_t1_open_to_t3_close_pct"] = pct_ret(base["t3_close"], t1_open_buy_price)
        t1_rets = [
            base["ret_t1_open_to_t1_close_pct"],
            base["ret_t1_open_to_t2_close_pct"],
            base["ret_t1_open_to_t3_close_pct"],
        ]
        t1_rets_valid = [x for x in t1_rets if not pd.isna(x)]
        base["ret_t1_open_to_t1_t3_best_close_pct"] = max(t1_rets_valid) if t1_rets_valid else np.nan
        base["ret_t1_open_to_t1_t3_worst_close_pct"] = min(t1_rets_valid) if t1_rets_valid else np.nan
        # For users who only compare T+2/T+3 after T1 open.
        t1_t2_t3 = [base["ret_t1_open_to_t2_close_pct"], base["ret_t1_open_to_t3_close_pct"]]
        t1_t2_t3_valid = [x for x in t1_t2_t3 if not pd.isna(x)]
        base["ret_t1_open_to_t2_t3_best_close_pct"] = max(t1_t2_t3_valid) if t1_t2_t3_valid else np.nan
        base["ret_t1_open_to_t2_t3_worst_close_pct"] = min(t1_t2_t3_valid) if t1_t2_t3_valid else np.nan
    else:
        base["t1_open_buy_price"] = np.nan
        for col in [
            "ret_t1_open_to_t1_close_pct", "ret_t1_open_to_t2_close_pct", "ret_t1_open_to_t3_close_pct",
            "ret_t1_open_to_t1_t3_best_close_pct", "ret_t1_open_to_t1_t3_worst_close_pct",
            "ret_t1_open_to_t2_t3_best_close_pct", "ret_t1_open_to_t2_t3_worst_close_pct",
        ]:
            base[col] = np.nan

    return base


def calc_detail(
    pool: pd.DataFrame,
    market_cache_dir: Path,
    t0_slippage_pct: float,
    t1_open_slippage_pct: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    file_map = build_market_file_map(market_cache_dir)
    bar_cache: dict[str, pd.DataFrame] = {}

    rows: list[dict[str, Any]] = []
    stats = {
        "total_pool_rows": len(pool),
        "missing_file_rows": 0,
        "failed_file_rows": 0,
        "missing_t0_rows": 0,
        "valid_rows": 0,
    }

    iterator = pool.iterrows()
    if tqdm is not None:
        iterator = tqdm(pool.iterrows(), total=len(pool), desc="Calculating forward returns", unit="signal")

    for _, sig in iterator:
        code = str(sig["code"])
        bars = load_one_bar(code, file_map, bar_cache)
        if bars is None:
            stats["missing_file_rows"] += 1
            rows.append({
                "code": code,
                "signal_date": pd.to_datetime(sig["date"]).strftime("%Y-%m-%d"),
                "skip_reason": "missing_or_failed_market_file",
            })
            continue
        if "_load_error" in bars.columns:
            stats["failed_file_rows"] += 1
            rows.append({
                "code": code,
                "signal_date": pd.to_datetime(sig["date"]).strftime("%Y-%m-%d"),
                "skip_reason": str(bars["_load_error"].iloc[0]),
            })
            continue

        rec = calc_one_signal(sig, bars, t0_slippage_pct, t1_open_slippage_pct)
        if rec.get("skip_reason") == "missing_t0_bar":
            stats["missing_t0_rows"] += 1
        else:
            stats["valid_rows"] += 1
        rows.append(rec)

    detail = pd.DataFrame(rows)
    stats["skipped_rows"] = stats["total_pool_rows"] - stats["valid_rows"]
    return detail, stats


# =============================================================================
# Summary
# =============================================================================


def summarize_ret(detail: pd.DataFrame, ret_col: str, metric: str, meta: dict[str, Any]) -> dict[str, Any]:
    s = pd.to_numeric(detail.get(ret_col, pd.Series(dtype=float)), errors="coerce").dropna()
    row: dict[str, Any] = dict(meta)
    row["metric"] = metric
    row["ret_col"] = ret_col
    row["count"] = int(len(s))

    if len(s) == 0:
        for k in [
            "win_rate_pct", "avg_ret_pct", "median_ret_pct", "p10_ret_pct", "p25_ret_pct",
            "p75_ret_pct", "p90_ret_pct", "max_ret_pct", "min_ret_pct", "prob_ge_1pct",
            "prob_ge_2pct", "prob_ge_3pct", "prob_le_minus_1pct", "prob_le_minus_2pct", "prob_le_minus_3pct",
        ]:
            row[k] = np.nan
        return row

    row["win_rate_pct"] = float((s > 0).mean() * 100.0)
    row["avg_ret_pct"] = float(s.mean())
    row["median_ret_pct"] = float(s.median())
    row["p10_ret_pct"] = float(s.quantile(0.10))
    row["p25_ret_pct"] = float(s.quantile(0.25))
    row["p75_ret_pct"] = float(s.quantile(0.75))
    row["p90_ret_pct"] = float(s.quantile(0.90))
    row["max_ret_pct"] = float(s.max())
    row["min_ret_pct"] = float(s.min())
    row["prob_ge_1pct"] = float((s >= 1.0).mean() * 100.0)
    row["prob_ge_2pct"] = float((s >= 2.0).mean() * 100.0)
    row["prob_ge_3pct"] = float((s >= 3.0).mean() * 100.0)
    row["prob_le_minus_1pct"] = float((s <= -1.0).mean() * 100.0)
    row["prob_le_minus_2pct"] = float((s <= -2.0).mean() * 100.0)
    row["prob_le_minus_3pct"] = float((s <= -3.0).mean() * 100.0)
    return row


def build_summary(
    detail: pd.DataFrame,
    pool_path: Path,
    stats: dict[str, int],
    t0_slippage_pct: float,
    t1_open_slippage_pct: float,
) -> pd.DataFrame:
    meta = {
        "pool_path": str(pool_path),
        "total_pool_rows": stats.get("total_pool_rows", 0),
        "valid_rows": stats.get("valid_rows", 0),
        "skipped_rows": stats.get("skipped_rows", 0),
        "missing_file_rows": stats.get("missing_file_rows", 0),
        "failed_file_rows": stats.get("failed_file_rows", 0),
        "missing_t0_rows": stats.get("missing_t0_rows", 0),
        "t0_slippage_pct": t0_slippage_pct,
        "t1_open_slippage_pct": t1_open_slippage_pct,
    }

    metrics = [
        ("T0尾盘买入 -> T+1收盘", "ret_t0_tail_to_t1_close_pct"),
        ("T0尾盘买入 -> T+2收盘", "ret_t0_tail_to_t2_close_pct"),
        ("T0尾盘买入 -> T+3收盘", "ret_t0_tail_to_t3_close_pct"),
        ("T+1早盘买入 -> T+1收盘", "ret_t1_open_to_t1_close_pct"),
        ("T+1早盘买入 -> T+2收盘", "ret_t1_open_to_t2_close_pct"),
        ("T+1早盘买入 -> T+3收盘", "ret_t1_open_to_t3_close_pct"),
        ("T0尾盘买入 -> T1~T3最佳收盘", "ret_t0_tail_to_t1_t3_best_close_pct"),
        ("T0尾盘买入 -> T1~T3最差收盘", "ret_t0_tail_to_t1_t3_worst_close_pct"),
        ("T+1早盘买入 -> T1~T3最佳收盘", "ret_t1_open_to_t1_t3_best_close_pct"),
        ("T+1早盘买入 -> T1~T3最差收盘", "ret_t1_open_to_t1_t3_worst_close_pct"),
        ("T+1早盘买入 -> T2~T3最佳收盘", "ret_t1_open_to_t2_t3_best_close_pct"),
        ("T+1早盘买入 -> T2~T3最差收盘", "ret_t1_open_to_t2_t3_worst_close_pct"),
    ]
    rows = [summarize_ret(detail, col, name, meta) for name, col in metrics]
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        print("\nSummary is empty.")
        return

    # Keep text columns visible and avoid pandas errors="ignore" compatibility issues.
    out = summary.copy()
    for c in out.columns:
        if c not in {"pool_path", "metric", "ret_col"}:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    print("\n" + "=" * 120)
    print("B2 V0 T0 tail buy vs T1 open buy forward return summary")
    print("=" * 120)
    with pd.option_context(
        "display.max_rows", 200,
        "display.max_columns", 80,
        "display.width", 220,
        "display.float_format", "{:.4f}".format,
    ):
        print(out.to_string(index=False))
    print("=" * 120)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze B2 V0 T0 tail buy vs T+1 open buy forward returns."
    )
    parser.add_argument("--pool-path", type=Path, required=True)
    parser.add_argument("--market-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--t0-slippage-pct", type=float, default=0.10)
    parser.add_argument("--t1-open-slippage-pct", type=float, default=0.00)
    parser.add_argument(
        "--no-selected-filter",
        action="store_true",
        help="Do not filter pool by selected=True even if selected column exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("B2 V0 forward return comparison v2")
    print("=" * 120)
    print(f"pool_path            : {args.pool_path}")
    print(f"market_cache_dir     : {args.market_cache_dir}")
    print(f"output_dir           : {args.output_dir}")
    print(f"start_date           : {args.start_date}")
    print(f"end_date             : {args.end_date}")
    print(f"t0_slippage_pct      : {args.t0_slippage_pct}")
    print(f"t1_open_slippage_pct : {args.t1_open_slippage_pct}")
    print(f"selected_only        : {not args.no_selected_filter}")
    print("=" * 120)

    pool = load_pool(
        args.pool_path,
        start_date=args.start_date,
        end_date=args.end_date,
        selected_only=not args.no_selected_filter,
    )
    print(f"Loaded pool rows: {len(pool)}")

    detail, stats = calc_detail(
        pool=pool,
        market_cache_dir=args.market_cache_dir,
        t0_slippage_pct=args.t0_slippage_pct,
        t1_open_slippage_pct=args.t1_open_slippage_pct,
    )

    summary = build_summary(
        detail=detail,
        pool_path=args.pool_path,
        stats=stats,
        t0_slippage_pct=args.t0_slippage_pct,
        t1_open_slippage_pct=args.t1_open_slippage_pct,
    )

    ts = now_str()
    detail_path = args.output_dir / f"rk_v4_t0_tail_vs_t1_open_detail_v2_{ts}.csv"
    summary_path = args.output_dir / f"rk_v4_t0_tail_vs_t1_open_summary_v2_{ts}.csv"

    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print_summary(summary)

    print("\n" + "=" * 120)
    print("OUTPUT")
    print("=" * 120)
    print(f"detail : {detail_path}")
    print(f"summary: {summary_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()
