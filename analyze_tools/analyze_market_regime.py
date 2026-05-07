from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import MARKET_CACHE_DIR


# =============================================================================
# Helpers
# =============================================================================

def normalize_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def read_one_market_file(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"[WARN] Failed to read: {path} | {e}")
        return None

    date_col = find_col(df, ["date", "trade_date", "datetime"])
    open_col = find_col(df, ["open"])
    close_col = find_col(df, ["close"])

    if date_col is None or close_col is None:
        print(f"[WARN] Missing required columns in {path.name}")
        return None

    out = pd.DataFrame()
    out["date"] = normalize_date(df[date_col])
    out["close"] = pd.to_numeric(df[close_col], errors="coerce")

    if open_col is not None:
        out["open"] = pd.to_numeric(df[open_col], errors="coerce")
    else:
        out["open"] = np.nan

    symbol = path.stem
    if "#" in symbol:
        symbol = symbol.split("#")[-1]
    out["symbol"] = symbol

    out = out.dropna(subset=["date", "close"])
    out = out.sort_values(["symbol", "date"])
    return out


def load_market_cache(market_cache_dir: Path) -> pd.DataFrame:
    files = sorted(market_cache_dir.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files found in: {market_cache_dir}")

    frames = []
    for i, path in enumerate(files, 1):
        if i % 500 == 0:
            print(f"[LOAD] Market files: {i}/{len(files)}")

        one = read_one_market_file(path)
        if one is not None and not one.empty:
            frames.append(one)

    if not frames:
        raise RuntimeError("No valid market data loaded.")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["symbol", "date"])

    df["prev_close"] = df.groupby("symbol")["close"].shift(1)
    df["daily_ret_pct"] = (df["close"] / df["prev_close"] - 1.0) * 100.0

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def build_daily_market_stats(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["daily_ret_pct"]).copy()

    daily = (
        valid.groupby("date")
        .agg(
            stock_count=("symbol", "nunique"),
            up_count=("daily_ret_pct", lambda x: int((x > 0).sum())),
            down_count=("daily_ret_pct", lambda x: int((x < 0).sum())),
            flat_count=("daily_ret_pct", lambda x: int((x == 0).sum())),
            avg_ret_pct=("daily_ret_pct", "mean"),
            median_ret_pct=("daily_ret_pct", "median"),
            p25_ret_pct=("daily_ret_pct", lambda x: x.quantile(0.25)),
            p75_ret_pct=("daily_ret_pct", lambda x: x.quantile(0.75)),
        )
        .reset_index()
        .sort_values("date")
    )

    daily["up_ratio_pct"] = daily["up_count"] / daily["stock_count"] * 100.0
    daily["down_ratio_pct"] = daily["down_count"] / daily["stock_count"] * 100.0

    return daily


def split_trading_windows(
    dates: list[pd.Timestamp],
    window_size: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, list[pd.Timestamp]]]:
    windows = []
    for i in range(0, len(dates), window_size):
        chunk = dates[i:i + window_size]
        if not chunk:
            continue
        windows.append((chunk[0], chunk[-1], chunk))
    return windows


def classify_market(
    window_ret_pct: float,
    avg_up_ratio_pct: float,
    strong_ret_pct: float,
    weak_ret_pct: float,
    strong_up_ratio_pct: float,
    weak_up_ratio_pct: float,
) -> str:
    if window_ret_pct >= strong_ret_pct or avg_up_ratio_pct >= strong_up_ratio_pct:
        return "strong"

    if window_ret_pct <= weak_ret_pct or avg_up_ratio_pct <= weak_up_ratio_pct:
        return "weak"

    return "normal"


def build_window_market_stats(
    daily: pd.DataFrame,
    window_size: int,
    strong_ret_pct: float,
    weak_ret_pct: float,
    strong_up_ratio_pct: float,
    weak_up_ratio_pct: float,
) -> pd.DataFrame:
    dates = daily["date"].dropna().sort_values().unique().tolist()
    dates = [pd.Timestamp(x).normalize() for x in dates]

    windows = split_trading_windows(dates, window_size)

    rows = []
    for idx, (start_date, end_date, chunk_dates) in enumerate(windows, 1):
        tmp = daily[daily["date"].isin(chunk_dates)].copy()

        # 用每日平均收益复利，近似表示区间等权市场收益
        avg_daily_ret = tmp["avg_ret_pct"].fillna(0) / 100.0
        median_daily_ret = tmp["median_ret_pct"].fillna(0) / 100.0

        equal_weight_window_ret_pct = (np.prod(1.0 + avg_daily_ret) - 1.0) * 100.0
        median_window_ret_pct = (np.prod(1.0 + median_daily_ret) - 1.0) * 100.0

        avg_up_ratio_pct = tmp["up_ratio_pct"].mean()
        avg_down_ratio_pct = tmp["down_ratio_pct"].mean()

        regime = classify_market(
            window_ret_pct=equal_weight_window_ret_pct,
            avg_up_ratio_pct=avg_up_ratio_pct,
            strong_ret_pct=strong_ret_pct,
            weak_ret_pct=weak_ret_pct,
            strong_up_ratio_pct=strong_up_ratio_pct,
            weak_up_ratio_pct=weak_up_ratio_pct,
        )

        rows.append(
            {
                "window_id": idx,
                "start_date": start_date,
                "end_date": end_date,
                "trading_days": len(chunk_dates),
                "market_regime": regime,
                "stock_count_avg": tmp["stock_count"].mean(),
                "avg_up_ratio_pct": avg_up_ratio_pct,
                "avg_down_ratio_pct": avg_down_ratio_pct,
                "avg_daily_ret_pct": tmp["avg_ret_pct"].mean(),
                "median_daily_ret_pct": tmp["median_ret_pct"].mean(),
                "equal_weight_window_ret_pct": equal_weight_window_ret_pct,
                "median_window_ret_pct": median_window_ret_pct,
                "best_day": tmp.loc[tmp["avg_ret_pct"].idxmax(), "date"] if not tmp.empty else pd.NaT,
                "best_day_avg_ret_pct": tmp["avg_ret_pct"].max(),
                "worst_day": tmp.loc[tmp["avg_ret_pct"].idxmin(), "date"] if not tmp.empty else pd.NaT,
                "worst_day_avg_ret_pct": tmp["avg_ret_pct"].min(),
            }
        )

    return pd.DataFrame(rows)


def load_pool(pool_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if pool_path is None:
        return None

    if not pool_path.exists():
        raise FileNotFoundError(f"Pool file not found: {pool_path}")

    df = pd.read_parquet(pool_path)

    date_col = find_col(df, ["date", "signal_date", "trade_date"])
    code_col = find_col(df, ["symbol", "code", "stock_code", "ts_code"])

    if date_col is None:
        raise ValueError("Pool file must have one of date / signal_date / trade_date columns.")

    df = df.copy()
    df["date"] = normalize_date(df[date_col])

    if code_col is not None:
        df["symbol"] = df[code_col].astype(str)
    else:
        df["symbol"] = ""

    return df.dropna(subset=["date"])


def detect_return_columns(pool_df: pd.DataFrame) -> list[str]:
    candidates = []
    for c in pool_df.columns:
        cl = c.lower()
        if "ret" in cl and "pct" in cl:
            candidates.append(c)
        elif "return" in cl and "pct" in cl:
            candidates.append(c)
    return candidates


def attach_pool_stats(
    window_df: pd.DataFrame,
    pool_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if pool_df is None or pool_df.empty:
        return window_df

    ret_cols = detect_return_columns(pool_df)

    rows = []
    for _, row in window_df.iterrows():
        start_date = row["start_date"]
        end_date = row["end_date"]

        tmp = pool_df[
            (pool_df["date"] >= start_date)
            & (pool_df["date"] <= end_date)
        ].copy()

        extra = {
            "pool_rows": len(tmp),
            "pool_unique_symbols": tmp["symbol"].nunique() if "symbol" in tmp.columns else np.nan,
        }

        for col in ret_cols:
            s = pd.to_numeric(tmp[col], errors="coerce").dropna()
            prefix = f"pool_{col}"

            if s.empty:
                extra[f"{prefix}_count"] = 0
                extra[f"{prefix}_win_rate_pct"] = np.nan
                extra[f"{prefix}_avg_pct"] = np.nan
                extra[f"{prefix}_median_pct"] = np.nan
                extra[f"{prefix}_p25_pct"] = np.nan
                extra[f"{prefix}_p75_pct"] = np.nan
            else:
                extra[f"{prefix}_count"] = len(s)
                extra[f"{prefix}_win_rate_pct"] = (s > 0).mean() * 100.0
                extra[f"{prefix}_avg_pct"] = s.mean()
                extra[f"{prefix}_median_pct"] = s.median()
                extra[f"{prefix}_p25_pct"] = s.quantile(0.25)
                extra[f"{prefix}_p75_pct"] = s.quantile(0.75)

        rows.append({**row.to_dict(), **extra})

    return pd.DataFrame(rows)


def print_window_detail(df: pd.DataFrame, max_rows: Optional[int] = None) -> None:
    if max_rows is not None and max_rows > 0:
        show = df.head(max_rows).copy()
    else:
        show = df.copy()

    display_cols = [
        "window_id",
        "start_date",
        "end_date",
        "trading_days",
        "market_regime",
        "stock_count_avg",
        "avg_up_ratio_pct",
        "avg_down_ratio_pct",
        "avg_daily_ret_pct",
        "median_daily_ret_pct",
        "equal_weight_window_ret_pct",
        "median_window_ret_pct",
        "best_day",
        "best_day_avg_ret_pct",
        "worst_day",
        "worst_day_avg_ret_pct",
        "pool_rows",
        "pool_unique_symbols",
    ]

    display_cols = [c for c in display_cols if c in show.columns]

    out = show[display_cols].copy()

    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")

    float_cols = out.select_dtypes(include=["float", "float64", "float32"]).columns
    for c in float_cols:
        out[c] = out[c].round(4)

    print("\n" + "=" * 140)
    print("Market regime window detail")
    print("=" * 140)
    print(out.to_string(index=False))
    print("=" * 140)


def print_regime_summary(df: pd.DataFrame) -> None:
    summary = (
        df.groupby("market_regime")
        .agg(
            window_count=("window_id", "count"),
            avg_equal_weight_window_ret_pct=("equal_weight_window_ret_pct", "mean"),
            median_equal_weight_window_ret_pct=("equal_weight_window_ret_pct", "median"),
            avg_up_ratio_pct=("avg_up_ratio_pct", "mean"),
            avg_down_ratio_pct=("avg_down_ratio_pct", "mean"),
            avg_pool_rows=("pool_rows", "mean") if "pool_rows" in df.columns else ("window_id", "count"),
        )
        .reset_index()
    )

    print("\n" + "=" * 100)
    print("Market regime summary")
    print("=" * 100)
    print(summary.round(4).to_string(index=False))
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze market strong / normal / weak by fixed trading-day windows."
    )

    parser.add_argument(
        "--market-cache-dir",
        type=Path,
        default=MARKET_CACHE_DIR,
        help="Default: config.MARKET_CACHE_DIR",
    )
    parser.add_argument(
        "--pool-path",
        type=Path,
        default=None,
        help="Optional pool parquet path. If provided, pool rows will be counted by window.",
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)

    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Number of trading days per window. Default: 10",
    )

    parser.add_argument(
        "--strong-ret-pct",
        type=float,
        default=3.0,
        help="Strong market threshold by equal-weight window return pct. Default: 3.0",
    )
    parser.add_argument(
        "--weak-ret-pct",
        type=float,
        default=-3.0,
        help="Weak market threshold by equal-weight window return pct. Default: -3.0",
    )
    parser.add_argument(
        "--strong-up-ratio-pct",
        type=float,
        default=60.0,
        help="Strong market threshold by average up ratio pct. Default: 60.0",
    )
    parser.add_argument(
        "--weak-up-ratio-pct",
        type=float,
        default=40.0,
        help="Weak market threshold by average up ratio pct. Default: 40.0",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output csv path.",
    )
    parser.add_argument(
        "--max-print-rows",
        type=int,
        default=0,
        help="0 means print all rows.",
    )

    args = parser.parse_args()

    print("\n[STEP] Loading market cache...")
    market_df = load_market_cache(args.market_cache_dir)

    if args.start_date:
        start = pd.Timestamp(args.start_date).normalize()
        market_df = market_df[market_df["date"] >= start]

    if args.end_date:
        end = pd.Timestamp(args.end_date).normalize()
        market_df = market_df[market_df["date"] <= end]

    if market_df.empty:
        raise RuntimeError("No market data after date filtering.")

    print(f"[INFO] Market rows: {len(market_df):,}")
    print(f"[INFO] Symbols: {market_df['symbol'].nunique():,}")
    print(f"[INFO] Date range: {market_df['date'].min().date()} -> {market_df['date'].max().date()}")

    print("\n[STEP] Building daily market stats...")
    daily = build_daily_market_stats(market_df)

    print("\n[STEP] Building window market stats...")
    window_df = build_window_market_stats(
        daily=daily,
        window_size=args.window_size,
        strong_ret_pct=args.strong_ret_pct,
        weak_ret_pct=args.weak_ret_pct,
        strong_up_ratio_pct=args.strong_up_ratio_pct,
        weak_up_ratio_pct=args.weak_up_ratio_pct,
    )

    pool_df = load_pool(args.pool_path)
    if pool_df is not None:
        print(f"\n[INFO] Pool rows: {len(pool_df):,}")
        print(f"[INFO] Pool date range: {pool_df['date'].min().date()} -> {pool_df['date'].max().date()}")
        window_df = attach_pool_stats(window_df, pool_df)
    else:
        window_df["pool_rows"] = np.nan
        window_df["pool_unique_symbols"] = np.nan

    print_window_detail(
        window_df,
        max_rows=None if args.max_print_rows == 0 else args.max_print_rows,
    )
    print_regime_summary(window_df)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        window_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        print(f"\n[SAVED] {args.output_csv}")


if __name__ == "__main__":
    main()