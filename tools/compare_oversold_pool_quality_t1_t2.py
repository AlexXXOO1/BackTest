from __future__ import annotations

"""
Compare oversold rebound pool quality for T1 open buy -> T2 close sell.

Purpose:
    Compare:
        1. Full base pool
        2. Oversold rebound select strategy v0 pool

Trading return definition:
    T1 open buy -> T2 close sell

Return:
    t1_open_to_t2_close_ret_pct = T2 close / T1 open - 1

Output:
    1. detail csv
    2. summary csv
    3. daily summary csv

Recommended file name:
    tools/compare_oversold_pool_quality_t1_t2.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Fixed paths
# =============================================================================

BASE_POOL_PATH = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\pools\full_pool_indicator_strategy_v0_pool.parquet"
)

OVERSOLD_POOL_PATH = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\pools\oversold_rebound_select_strategy_v0_pool.parquet"
)

MARKET_CACHE_DIR = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\market_cache\daily_bars_by_symbol"
)

OUTPUT_DIR = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\output\oversold_rebound_pool_quality_t1_t2"
)


# =============================================================================
# Date range
# Set to None if you want to use all dates inside pool files.
# =============================================================================

START_DATE = None
END_DATE = None

# Example:
# START_DATE = "2025-01-01"
# END_DATE = "2025-12-31"


# =============================================================================
# Column candidates
# =============================================================================

DATE_CANDIDATES = [
    "date", "trade_date", "signal_date", "datetime", "time",
    "日期", "交易日期", "时间",
]

SYMBOL_CANDIDATES = [
    "symbol", "code", "stock_code", "ts_code",
    "证券代码", "股票代码",
]

OPEN_CANDIDATES = ["open", "开盘", "开盘价"]
HIGH_CANDIDATES = ["high", "最高", "最高价"]
LOW_CANDIDATES = ["low", "最低", "最低价"]
CLOSE_CANDIDATES = ["close", "收盘", "收盘价"]

NAME_CANDIDATES = ["name", "stock_name", "股票名称", "证券简称"]


# =============================================================================
# Helpers
# =============================================================================

def normalize_symbol(x) -> str:
    if pd.isna(x):
        return ""

    s = str(x).strip().upper()
    s = s.replace(".TXT", "")
    s = s.replace(".PARQUET", "")
    s = s.replace("\\", "/")
    s = s.split("/")[-1]

    if "#" in s:
        left, right = s.split("#", 1)
        left = left.strip()
        right = "".join(ch for ch in right if ch.isdigit())[-6:]
        if left in {"SH", "SZ"} and len(right) == 6:
            return f"{left}#{right}"

    if s.startswith("SH") and len(s) >= 8:
        return f"SH#{s[-6:]}"

    if s.startswith("SZ") and len(s) >= 8:
        return f"SZ#{s[-6:]}"

    digits = "".join(ch for ch in s if ch.isdigit())

    if len(digits) >= 6:
        code = digits[-6:]
        if code.startswith(("6", "9")):
            return f"SH#{code}"
        if code.startswith(("0", "2", "3")):
            return f"SZ#{code}"
        return code

    return s


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    exact = {str(c): c for c in df.columns}
    lower = {str(c).lower(): c for c in df.columns}

    for c in candidates:
        if c in exact:
            return exact[c]
        if c.lower() in lower:
            return lower[c.lower()]

    return None


def get_date_series(df: pd.DataFrame, source_name: str) -> pd.Series:
    col = find_col(df, DATE_CANDIDATES)

    if col is not None:
        return pd.to_datetime(df[col], errors="coerce").dt.normalize()

    idx = df.index

    if isinstance(idx, pd.DatetimeIndex):
        return pd.Series(idx, index=df.index).dt.normalize()

    idx_dt = pd.to_datetime(idx, errors="coerce")
    idx_dt_series = pd.Series(idx_dt, index=df.index)

    if idx_dt_series.notna().mean() > 0.8:
        return idx_dt_series.dt.normalize()

    raise ValueError(
        f"{source_name}: cannot find date column or parse index as date. "
        f"Columns={list(df.columns)}"
    )


def get_symbol_series(df: pd.DataFrame, fallback_symbol: str = "") -> pd.Series:
    col = find_col(df, SYMBOL_CANDIDATES)

    if col is not None:
        return df[col].map(normalize_symbol)

    if fallback_symbol:
        return pd.Series([fallback_symbol] * len(df), index=df.index)

    return pd.Series([""] * len(df), index=df.index)


def get_numeric_col(
    df: pd.DataFrame,
    candidates: list[str],
    col_name: str,
    source_name: str,
) -> pd.Series:
    col = find_col(df, candidates)

    if col is None:
        raise ValueError(
            f"{source_name}: cannot find {col_name} column. "
            f"Candidates={candidates}. Columns={list(df.columns)}"
        )

    return pd.to_numeric(df[col], errors="coerce")


def get_optional_name_series(df: pd.DataFrame) -> pd.Series:
    col = find_col(df, NAME_CANDIDATES)
    if col is None:
        return pd.Series([""] * len(df), index=df.index)
    return df[col].astype(str)


def apply_date_filter(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if START_DATE is not None:
        out = out[out["date"] >= pd.to_datetime(START_DATE)].copy()

    if END_DATE is not None:
        out = out[out["date"] <= pd.to_datetime(END_DATE)].copy()

    return out


# =============================================================================
# Load pool files
# =============================================================================

def load_pool(path: Path, pool_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{pool_name} not found: {path}")

    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError(f"{pool_name} is empty: {path}")

    out = df.copy()
    out["date"] = get_date_series(df, pool_name)
    out["symbol"] = get_symbol_series(df)
    out["stock_name"] = get_optional_name_series(df)

    out = out.dropna(subset=["date"])
    out = out[out["symbol"] != ""]
    out = out.drop_duplicates(subset=["symbol", "date"])
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    out = apply_date_filter(out)

    return out


# =============================================================================
# Load market cache
# =============================================================================

def load_one_market_file(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None

    if df.empty:
        return None

    fallback_symbol = normalize_symbol(path.stem)

    try:
        out = pd.DataFrame()
        out["symbol"] = get_symbol_series(df, fallback_symbol=fallback_symbol)
        out["date"] = get_date_series(df, path.name)
        out["open"] = get_numeric_col(df, OPEN_CANDIDATES, "open", path.name)
        out["high"] = get_numeric_col(df, HIGH_CANDIDATES, "high", path.name)
        out["low"] = get_numeric_col(df, LOW_CANDIDATES, "low", path.name)
        out["close"] = get_numeric_col(df, CLOSE_CANDIDATES, "close", path.name)

        out = out.dropna(subset=["symbol", "date", "open", "high", "low", "close"])
        out = out[out["symbol"] != ""]
        out = out.sort_values("date").drop_duplicates(subset=["symbol", "date"])

        if out.empty:
            return None

        return out

    except Exception:
        return None


def load_market_cache(market_cache_dir: Path) -> pd.DataFrame:
    if not market_cache_dir.exists():
        raise FileNotFoundError(f"Market cache dir not found: {market_cache_dir}")

    files = sorted(market_cache_dir.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files under: {market_cache_dir}")

    frames: list[pd.DataFrame] = []
    failed = 0

    print("=" * 100)
    print("Loading market cache")
    print("=" * 100)
    print(f"Market cache dir : {market_cache_dir}")
    print(f"Parquet files    : {len(files):,}")

    for i, path in enumerate(files, start=1):
        if i == 1 or i % 300 == 0 or i == len(files):
            print(f"  Loading [{i:,}/{len(files):,}] {path.name}")

        one = load_one_market_file(path)

        if one is None or one.empty:
            failed += 1
        else:
            frames.append(one)

    if not frames:
        raise ValueError(
            "No market parquet file was loaded successfully. "
            "Please check date/open/high/low/close column names."
        )

    market = pd.concat(frames, ignore_index=True)
    market = market.sort_values(["symbol", "date"]).reset_index(drop=True)

    print(f"Loaded market rows : {len(market):,}")
    print(f"Loaded symbols     : {market['symbol'].nunique():,}")
    print(f"Failed files       : {failed:,}")
    print(f"Date range         : {market['date'].min().date()} -> {market['date'].max().date()}")

    return market


# =============================================================================
# Forward returns
# =============================================================================

def add_t1_t2_returns(pool: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    m = market.copy()
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)

    g = m.groupby("symbol", sort=False)

    m["t1_date"] = g["date"].shift(-1)
    m["t1_open"] = g["open"].shift(-1)
    m["t1_high"] = g["high"].shift(-1)
    m["t1_low"] = g["low"].shift(-1)
    m["t1_close"] = g["close"].shift(-1)

    m["t2_date"] = g["date"].shift(-2)
    m["t2_open"] = g["open"].shift(-2)
    m["t2_high"] = g["high"].shift(-2)
    m["t2_low"] = g["low"].shift(-2)
    m["t2_close"] = g["close"].shift(-2)

    m = m[
        [
            "symbol", "date",
            "open", "high", "low", "close",
            "t1_date", "t1_open", "t1_high", "t1_low", "t1_close",
            "t2_date", "t2_open", "t2_high", "t2_low", "t2_close",
        ]
    ].rename(
        columns={
            "open": "t0_open",
            "high": "t0_high",
            "low": "t0_low",
            "close": "t0_close",
        }
    )

    out = pool.merge(m, on=["symbol", "date"], how="left")

    # Main trading return:
    # T1 open buy -> T2 close sell
    out["t1_open_to_t2_close_ret_pct"] = (
        out["t2_close"] / out["t1_open"] - 1.0
    ) * 100.0

    # Auxiliary statistics
    out["t1_open_to_t2_high_ret_pct"] = (
        out["t2_high"] / out["t1_open"] - 1.0
    ) * 100.0

    out["t1_open_to_t2_low_ret_pct"] = (
        out["t2_low"] / out["t1_open"] - 1.0
    ) * 100.0

    out["t1_open_to_t1_close_ret_pct"] = (
        out["t1_close"] / out["t1_open"] - 1.0
    ) * 100.0

    out["t1_open_gap_pct"] = (
        out["t1_open"] / out["t0_close"] - 1.0
    ) * 100.0

    out["has_t1_t2"] = out["t1_open"].notna() & out["t2_close"].notna()

    return out


# =============================================================================
# Summaries
# =============================================================================

def build_summary_row(df: pd.DataFrame, group_name: str) -> dict:
    valid = df.dropna(subset=["t1_open_to_t2_close_ret_pct"]).copy()

    row = {
        "group": group_name,
        "sample_count": len(df),
        "valid_count": len(valid),
        "date_count": valid["date"].nunique() if len(valid) else 0,
        "symbol_count": valid["symbol"].nunique() if len(valid) else 0,
        "missing_t0_close_count": int(df["t0_close"].isna().sum()) if "t0_close" in df.columns else np.nan,
        "missing_t1_t2_count": int((~df["has_t1_t2"]).sum()) if "has_t1_t2" in df.columns else np.nan,
    }

    if len(valid) == 0:
        return row

    ret = valid["t1_open_to_t2_close_ret_pct"]
    high_ret = valid["t1_open_to_t2_high_ret_pct"]
    low_ret = valid["t1_open_to_t2_low_ret_pct"]
    t1_ret = valid["t1_open_to_t1_close_ret_pct"]
    gap = valid["t1_open_gap_pct"]

    row.update(
        {
            "t1_t2_mean_ret_pct": ret.mean(),
            "t1_t2_median_ret_pct": ret.median(),
            "t1_t2_p10_ret_pct": ret.quantile(0.10),
            "t1_t2_p25_ret_pct": ret.quantile(0.25),
            "t1_t2_p75_ret_pct": ret.quantile(0.75),
            "t1_t2_p90_ret_pct": ret.quantile(0.90),
            "t1_t2_min_ret_pct": ret.min(),
            "t1_t2_max_ret_pct": ret.max(),
            "t1_t2_win_rate_pct": (ret > 0).mean() * 100.0,
            "t1_t2_hit_1pct_rate": (ret >= 1.0).mean() * 100.0,
            "t1_t2_hit_2pct_rate": (ret >= 2.0).mean() * 100.0,
            "t1_t2_hit_3pct_rate": (ret >= 3.0).mean() * 100.0,
            "t1_t2_hit_5pct_rate": (ret >= 5.0).mean() * 100.0,
            "t1_t2_loss_1pct_rate": (ret <= -1.0).mean() * 100.0,
            "t1_t2_loss_2pct_rate": (ret <= -2.0).mean() * 100.0,
            "t1_t2_loss_3pct_rate": (ret <= -3.0).mean() * 100.0,
            "t1_t2_loss_5pct_rate": (ret <= -5.0).mean() * 100.0,
            "t2_high_mean_ret_pct": high_ret.mean(),
            "t2_high_hit_2pct_rate": (high_ret >= 2.0).mean() * 100.0,
            "t2_high_hit_3pct_rate": (high_ret >= 3.0).mean() * 100.0,
            "t2_high_hit_5pct_rate": (high_ret >= 5.0).mean() * 100.0,
            "t2_low_mean_ret_pct": low_ret.mean(),
            "t2_low_loss_2pct_rate": (low_ret <= -2.0).mean() * 100.0,
            "t2_low_loss_3pct_rate": (low_ret <= -3.0).mean() * 100.0,
            "t2_low_loss_5pct_rate": (low_ret <= -5.0).mean() * 100.0,
            "t1_open_gap_mean_pct": gap.mean(),
            "t1_open_gap_median_pct": gap.median(),
            "t1_intraday_mean_ret_pct": t1_ret.mean(),
            "t1_intraday_win_rate_pct": (t1_ret > 0).mean() * 100.0,
        }
    )

    return row


def add_diff_vs_base(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()

    base = out[out["group"] == "BASE_FULL_POOL"]
    if base.empty:
        return out

    base_row = base.iloc[0]

    compare_cols = [
        "t1_t2_mean_ret_pct",
        "t1_t2_median_ret_pct",
        "t1_t2_win_rate_pct",
        "t1_t2_hit_1pct_rate",
        "t1_t2_hit_2pct_rate",
        "t1_t2_hit_3pct_rate",
        "t1_t2_loss_1pct_rate",
        "t1_t2_loss_2pct_rate",
        "t1_t2_loss_3pct_rate",
        "t2_high_hit_2pct_rate",
        "t2_low_loss_2pct_rate",
        "t1_open_gap_mean_pct",
    ]

    for col in compare_cols:
        if col in out.columns:
            out[f"diff_vs_base_{col}"] = out[col] - base_row[col]

    return out


def build_daily_summary(result: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for date, one_day in result.groupby("date", sort=True):
        base = one_day[one_day["is_oversold_pool"] == False].copy()
        over = one_day[one_day["is_oversold_pool"] == True].copy()

        base_ret = base["t1_open_to_t2_close_ret_pct"].dropna()
        over_ret = over["t1_open_to_t2_close_ret_pct"].dropna()

        row = {
            "date": date,
            "base_count": len(base),
            "oversold_count": len(over),
            "oversold_ratio_pct": len(over) / len(one_day) * 100.0 if len(one_day) else np.nan,
            "base_valid_count": len(base_ret),
            "oversold_valid_count": len(over_ret),
            "base_mean_ret_pct": base_ret.mean() if len(base_ret) else np.nan,
            "oversold_mean_ret_pct": over_ret.mean() if len(over_ret) else np.nan,
            "diff_oversold_minus_base_mean_ret_pct": (
                over_ret.mean() - base_ret.mean()
                if len(base_ret) and len(over_ret)
                else np.nan
            ),
            "base_win_rate_pct": (base_ret > 0).mean() * 100.0 if len(base_ret) else np.nan,
            "oversold_win_rate_pct": (over_ret > 0).mean() * 100.0 if len(over_ret) else np.nan,
            "diff_oversold_minus_base_win_rate_pct": (
                (over_ret > 0).mean() * 100.0 - (base_ret > 0).mean() * 100.0
                if len(base_ret) and len(over_ret)
                else np.nan
            ),
            "base_hit_2pct_rate": (base_ret >= 2.0).mean() * 100.0 if len(base_ret) else np.nan,
            "oversold_hit_2pct_rate": (over_ret >= 2.0).mean() * 100.0 if len(over_ret) else np.nan,
            "base_loss_2pct_rate": (base_ret <= -2.0).mean() * 100.0 if len(base_ret) else np.nan,
            "oversold_loss_2pct_rate": (over_ret <= -2.0).mean() * 100.0 if len(over_ret) else np.nan,
        }

        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame) -> None:
    show_cols = [
        "group",
        "sample_count",
        "valid_count",
        "date_count",
        "symbol_count",
        "t1_t2_mean_ret_pct",
        "t1_t2_median_ret_pct",
        "t1_t2_win_rate_pct",
        "t1_t2_hit_2pct_rate",
        "t1_t2_loss_2pct_rate",
        "t2_high_hit_2pct_rate",
        "t2_low_loss_2pct_rate",
        "diff_vs_base_t1_t2_mean_ret_pct",
        "diff_vs_base_t1_t2_median_ret_pct",
        "diff_vs_base_t1_t2_win_rate_pct",
        "diff_vs_base_t1_t2_hit_2pct_rate",
        "diff_vs_base_t1_t2_loss_2pct_rate",
    ]

    existing = [c for c in show_cols if c in summary.columns]

    print()
    print("=" * 100)
    print("POOL QUALITY SUMMARY")
    print("=" * 100)

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 260,
        "display.float_format", "{:.4f}".format,
    ):
        print(summary[existing].to_string(index=False))


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Compare oversold rebound pool quality")
    print("=" * 100)
    print(f"Base pool path     : {BASE_POOL_PATH}")
    print(f"Oversold pool path : {OVERSOLD_POOL_PATH}")
    print(f"Market cache dir   : {MARKET_CACHE_DIR}")
    print(f"Output dir         : {OUTPUT_DIR}")
    print(f"Date range         : {START_DATE} -> {END_DATE}")

    base_pool = load_pool(BASE_POOL_PATH, "BASE_FULL_POOL")
    oversold_pool = load_pool(OVERSOLD_POOL_PATH, "OVERSOLD_REBOUND_POOL")

    print()
    print("=" * 100)
    print("Loaded pools")
    print("=" * 100)
    print(f"Base rows        : {len(base_pool):,}")
    print(f"Base symbols     : {base_pool['symbol'].nunique():,}")
    print(f"Base date range  : {base_pool['date'].min().date()} -> {base_pool['date'].max().date()}")
    print(f"Oversold rows    : {len(oversold_pool):,}")
    print(f"Oversold symbols : {oversold_pool['symbol'].nunique():,}")
    print(f"Oversold dates   : {oversold_pool['date'].min().date()} -> {oversold_pool['date'].max().date()}")

    oversold_keys = oversold_pool[["symbol", "date"]].drop_duplicates().copy()
    oversold_keys["is_oversold_pool"] = True

    base = base_pool[["symbol", "date", "stock_name"]].drop_duplicates().copy()
    base = base.merge(oversold_keys, on=["symbol", "date"], how="left")
    base["is_oversold_pool"] = base["is_oversold_pool"].fillna(False).astype(bool)

    matched_oversold = int(base["is_oversold_pool"].sum())
    missing_oversold = len(oversold_keys) - matched_oversold

    print()
    print("=" * 100)
    print("Pool matching check")
    print("=" * 100)
    print(f"Matched oversold rows     : {matched_oversold:,}")
    print(f"Oversold rows not in base : {missing_oversold:,}")

    if matched_oversold == 0:
        raise ValueError(
            "Oversold pool has zero matched rows in base pool. "
            "Please check symbol/date format."
        )

    market = load_market_cache(MARKET_CACHE_DIR)

    result = add_t1_t2_returns(base, market)

    print()
    print("=" * 100)
    print("Forward return merge check")
    print("=" * 100)
    print(f"Rows without T0 close : {result['t0_close'].isna().sum():,}")
    print(f"Rows with T1/T2 data  : {result['has_t1_t2'].sum():,}")
    print(f"Rows missing T1/T2    : {(~result['has_t1_t2']).sum():,}")

    base_group = result[result["is_oversold_pool"] == False].copy()
    oversold_group = result[result["is_oversold_pool"] == True].copy()

    summary = pd.DataFrame(
        [
            build_summary_row(result, "ALL_BASE_FILE"),
            build_summary_row(base_group, "BASE_FULL_POOL"),
            build_summary_row(oversold_group, "OVERSOLD_REBOUND_POOL"),
        ]
    )

    summary = add_diff_vs_base(summary)

    daily_summary = build_daily_summary(result)

    detail_path = OUTPUT_DIR / "oversold_rebound_pool_quality_t1_t2_detail.csv"
    summary_path = OUTPUT_DIR / "oversold_rebound_pool_quality_t1_t2_summary.csv"
    daily_path = OUTPUT_DIR / "oversold_rebound_pool_quality_t1_t2_daily_summary.csv"

    result.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    daily_summary.to_csv(daily_path, index=False, encoding="utf-8-sig")

    print_summary(summary)

    print()
    print("=" * 100)
    print("Saved files")
    print("=" * 100)
    print(f"Detail CSV        : {detail_path}")
    print(f"Summary CSV       : {summary_path}")
    print(f"Daily summary CSV : {daily_path}")

    print()
    print("=" * 100)
    print("How to judge")
    print("=" * 100)
    print("Focus on OVERSOLD_REBOUND_POOL vs BASE_FULL_POOL:")
    print("1. t1_t2_mean_ret_pct       higher is better")
    print("2. t1_t2_median_ret_pct     higher is better")
    print("3. t1_t2_win_rate_pct       higher is better")
    print("4. t1_t2_hit_2pct_rate      higher is better")
    print("5. t1_t2_loss_2pct_rate     lower is better")
    print()
    print("If mean/median/win-rate/hit-2pct are higher and loss-2pct is lower,")
    print("the oversold rebound pool is better than the full base pool.")


if __name__ == "__main__":
    main()
