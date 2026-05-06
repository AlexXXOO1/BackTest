from __future__ import annotations

"""
Analyze N-shaped uptrend structure inside oversold rebound pool.

Reference idea from the B1 repair-type note:
    N-shaped uptrend structure:
        H2 > H1
        AND
        L2 > L1

    H1 = previous swing high
    H2 = latest swing high
    L1 = previous swing low
    L2 = latest swing low

    Also analyze whether T0 breaks the previous N low:
        strict version: low_T0 >= prev_n_low
        close-confirm version: close_T0 >= prev_n_low

Purpose:
    Analyze whether N-shaped structure is positive inside:
        oversold_rebound_select_strategy_v0_pool

Trading return definition:
    T1 open buy -> T2 close sell

Output:
    1. detail CSV with N-structure fields
    2. summary CSV
    3. daily summary CSV

Recommended file name:
    tools/analyze_oversold_n_structure_t1_t2.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Fixed paths
# =============================================================================

OVERSOLD_POOL_PATH = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\pools\oversold_rebound_select_strategy_v0_pool.parquet"
)

MARKET_CACHE_DIR = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\market_cache\daily_bars_by_symbol"
)

OUTPUT_DIR = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\output\oversold_rebound_n_structure_t1_t2"
)


# =============================================================================
# Date range
# Set None to use all dates in the oversold pool.
# =============================================================================

START_DATE = None
END_DATE = None

# Example:
# START_DATE = "2025-01-01"
# END_DATE = "2025-12-31"


# =============================================================================
# N-structure parameters
# =============================================================================

# A swing high/low is identified using left/right window.
# Example: SWING_WINDOW = 3 means:
#   swing high = high is the highest among previous 3 days, current day, next 3 days.
# To avoid look-ahead in signal usage, a swing point becomes "known" only after 3 more trading days.
SWING_WINDOW = 3

# Allow fake break when evaluating previous N low.
# strict_not_break_prev_n_low:
#   low_T0 >= prev_n_low
#
# close_confirm_not_break_prev_n_low:
#   close_T0 >= prev_n_low
#
# tolerant_not_break_prev_n_low:
#   low_T0 >= prev_n_low * (1 - PREV_LOW_TOLERANCE_PCT / 100)
#   AND close_T0 >= prev_n_low
PREV_LOW_TOLERANCE_PCT = 2.0


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
# Load data
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
# N structure calculation
# =============================================================================

def _is_center_swing_high(high: pd.Series, window: int) -> pd.Series:
    rolling_max = high.rolling(window=window * 2 + 1, center=True, min_periods=window * 2 + 1).max()
    return (high == rolling_max) & high.notna()


def _is_center_swing_low(low: pd.Series, window: int) -> pd.Series:
    rolling_min = low.rolling(window=window * 2 + 1, center=True, min_periods=window * 2 + 1).min()
    return (low == rolling_min) & low.notna()


def add_n_structure_for_one_symbol(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Add N-structure fields for one symbol.

    Swing points are center-confirmed:
        A swing at index i is confirmed only at index i + window.
    Therefore, when evaluating T0, only swing points with confirm_index <= current index are available.
    This avoids future leakage for signal analysis.
    """

    one = df.sort_values("date").reset_index(drop=True).copy()
    n = len(one)

    high = pd.to_numeric(one["high"], errors="coerce")
    low = pd.to_numeric(one["low"], errors="coerce")

    raw_swing_high = _is_center_swing_high(high, window)
    raw_swing_low = _is_center_swing_low(low, window)

    high_events: dict[int, list[tuple[pd.Timestamp, float]]] = {}
    low_events: dict[int, list[tuple[pd.Timestamp, float]]] = {}

    for idx in np.flatnonzero(raw_swing_high.to_numpy()):
        confirm_idx = idx + window
        if confirm_idx < n:
            high_events.setdefault(confirm_idx, []).append((one.loc[idx, "date"], float(high.iloc[idx])))

    for idx in np.flatnonzero(raw_swing_low.to_numpy()):
        confirm_idx = idx + window
        if confirm_idx < n:
            low_events.setdefault(confirm_idx, []).append((one.loc[idx, "date"], float(low.iloc[idx])))

    h1_list = []
    h2_list = []
    h1_date_list = []
    h2_date_list = []

    l1_list = []
    l2_list = []
    l1_date_list = []
    l2_date_list = []

    available_highs: list[tuple[pd.Timestamp, float]] = []
    available_lows: list[tuple[pd.Timestamp, float]] = []

    for i in range(n):
        if i in high_events:
            available_highs.extend(high_events[i])
        if i in low_events:
            available_lows.extend(low_events[i])

        if len(available_highs) >= 2:
            h1_date, h1 = available_highs[-2]
            h2_date, h2 = available_highs[-1]
        else:
            h1_date, h1 = pd.NaT, np.nan
            h2_date, h2 = pd.NaT, np.nan

        if len(available_lows) >= 2:
            l1_date, l1 = available_lows[-2]
            l2_date, l2 = available_lows[-1]
        else:
            l1_date, l1 = pd.NaT, np.nan
            l2_date, l2 = pd.NaT, np.nan

        h1_date_list.append(h1_date)
        h2_date_list.append(h2_date)
        h1_list.append(h1)
        h2_list.append(h2)

        l1_date_list.append(l1_date)
        l2_date_list.append(l2_date)
        l1_list.append(l1)
        l2_list.append(l2)

    one["n_h1_date"] = h1_date_list
    one["n_h2_date"] = h2_date_list
    one["n_h1"] = h1_list
    one["n_h2"] = h2_list

    one["n_l1_date"] = l1_date_list
    one["n_l2_date"] = l2_date_list
    one["n_l1"] = l1_list
    one["n_l2"] = l2_list

    one["n_uptrend_structure"] = (
        (one["n_h2"] > one["n_h1"])
        & (one["n_l2"] > one["n_l1"])
    ).fillna(False)

    one["prev_n_low"] = one["n_l2"]

    one["strict_not_break_prev_n_low"] = (
        pd.to_numeric(one["low"], errors="coerce") >= one["prev_n_low"]
    ).fillna(False)

    one["close_confirm_not_break_prev_n_low"] = (
        pd.to_numeric(one["close"], errors="coerce") >= one["prev_n_low"]
    ).fillna(False)

    one["tolerant_not_break_prev_n_low"] = (
        (pd.to_numeric(one["low"], errors="coerce") >= one["prev_n_low"] * (1.0 - PREV_LOW_TOLERANCE_PCT / 100.0))
        & (pd.to_numeric(one["close"], errors="coerce") >= one["prev_n_low"])
    ).fillna(False)

    one["distance_to_prev_n_low_pct"] = (
        pd.to_numeric(one["close"], errors="coerce") / one["prev_n_low"] - 1.0
    ) * 100.0

    one["has_enough_n_points"] = (
        one["n_h1"].notna()
        & one["n_h2"].notna()
        & one["n_l1"].notna()
        & one["n_l2"].notna()
    )

    one["swing_window"] = window

    return one


def add_n_structure(market: pd.DataFrame, window: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    symbols = market["symbol"].dropna().unique().tolist()

    print()
    print("=" * 100)
    print("Calculating N-shaped structure")
    print("=" * 100)
    print(f"Swing window : {window}")
    print(f"Symbols      : {len(symbols):,}")

    for i, (symbol, one) in enumerate(market.groupby("symbol", sort=False), start=1):
        if i == 1 or i % 300 == 0 or i == len(symbols):
            print(f"  N structure [{i:,}/{len(symbols):,}] {symbol}")

        frames.append(add_n_structure_for_one_symbol(one, window=window))

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

    return out


# =============================================================================
# Forward returns
# =============================================================================

def add_t1_t2_returns_and_structure(pool: pd.DataFrame, market_with_structure: pd.DataFrame) -> pd.DataFrame:
    m = market_with_structure.copy()
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

    keep_cols = [
        "symbol", "date",
        "open", "high", "low", "close",
        "n_h1_date", "n_h2_date", "n_h1", "n_h2",
        "n_l1_date", "n_l2_date", "n_l1", "n_l2",
        "n_uptrend_structure",
        "prev_n_low",
        "strict_not_break_prev_n_low",
        "close_confirm_not_break_prev_n_low",
        "tolerant_not_break_prev_n_low",
        "distance_to_prev_n_low_pct",
        "has_enough_n_points",
        "swing_window",
        "t1_date", "t1_open", "t1_high", "t1_low", "t1_close",
        "t2_date", "t2_open", "t2_high", "t2_low", "t2_close",
    ]

    m = m[keep_cols].rename(
        columns={
            "open": "t0_open",
            "high": "t0_high",
            "low": "t0_low",
            "close": "t0_close",
        }
    )

    out = pool.merge(m, on=["symbol", "date"], how="left")

    out["t1_open_gap_pct"] = (
        out["t1_open"] / out["t0_close"] - 1.0
    ) * 100.0

    out["t1_open_to_t1_close_ret_pct"] = (
        out["t1_close"] / out["t1_open"] - 1.0
    ) * 100.0

    out["t1_open_to_t2_close_ret_pct"] = (
        out["t2_close"] / out["t1_open"] - 1.0
    ) * 100.0

    out["t1_open_to_t2_high_ret_pct"] = (
        out["t2_high"] / out["t1_open"] - 1.0
    ) * 100.0

    out["t1_open_to_t2_low_ret_pct"] = (
        out["t2_low"] / out["t1_open"] - 1.0
    ) * 100.0

    out["has_t1_t2"] = out["t1_open"].notna() & out["t2_close"].notna()

    out["n_structure_state"] = np.select(
        [
            out["n_uptrend_structure"].fillna(False),
            out["has_enough_n_points"].fillna(False),
        ],
        [
            "N_UPTREND_TRUE",
            "N_UPTREND_FALSE",
        ],
        default="N_POINTS_NOT_ENOUGH",
    )

    out["n_structure_close_support_state"] = np.select(
        [
            out["n_uptrend_structure"].fillna(False) & out["close_confirm_not_break_prev_n_low"].fillna(False),
            out["n_uptrend_structure"].fillna(False) & ~out["close_confirm_not_break_prev_n_low"].fillna(False),
            ~out["n_uptrend_structure"].fillna(False) & out["has_enough_n_points"].fillna(False),
        ],
        [
            "N_TRUE_AND_CLOSE_NOT_BREAK",
            "N_TRUE_BUT_CLOSE_BREAK",
            "N_FALSE",
        ],
        default="N_POINTS_NOT_ENOUGH",
    )

    return out



def _as_bool(s: pd.Series) -> pd.Series:
    """
    Convert bool/string/numeric series to bool safely.

    This avoids errors such as:
        TypeError: Cannot perform reduction 'mean' with string dtype
    """
    if s is None:
        return pd.Series(dtype=bool)

    if s.dtype == bool:
        return s.fillna(False).astype(bool)

    text = s.astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y", "t"])


# =============================================================================
# Summaries
# =============================================================================

def build_summary_row(df: pd.DataFrame, group_col: str, bucket: str) -> dict:
    valid = df.dropna(subset=["t1_open_to_t2_close_ret_pct"]).copy()

    row = {
        "group_col": group_col,
        "bucket": bucket,
        "sample_count": len(df),
        "valid_count": len(valid),
        "date_count": valid["date"].nunique() if len(valid) else 0,
        "symbol_count": valid["symbol"].nunique() if len(valid) else 0,
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
            "avg_t1_open_gap_pct": gap.mean(),
            "median_t1_open_gap_pct": gap.median(),

            "t1_t2_mean_ret_pct": ret.mean(),
            "t1_t2_median_ret_pct": ret.median(),
            "t1_t2_p10_ret_pct": ret.quantile(0.10),
            "t1_t2_p25_ret_pct": ret.quantile(0.25),
            "t1_t2_p75_ret_pct": ret.quantile(0.75),
            "t1_t2_p90_ret_pct": ret.quantile(0.90),

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
            "t2_high_hit_1pct_rate": (high_ret >= 1.0).mean() * 100.0,
            "t2_high_hit_2pct_rate": (high_ret >= 2.0).mean() * 100.0,
            "t2_high_hit_3pct_rate": (high_ret >= 3.0).mean() * 100.0,
            "t2_high_hit_5pct_rate": (high_ret >= 5.0).mean() * 100.0,

            "t2_low_mean_ret_pct": low_ret.mean(),
            "t2_low_loss_1pct_rate": (low_ret <= -1.0).mean() * 100.0,
            "t2_low_loss_2pct_rate": (low_ret <= -2.0).mean() * 100.0,
            "t2_low_loss_3pct_rate": (low_ret <= -3.0).mean() * 100.0,
            "t2_low_loss_5pct_rate": (low_ret <= -5.0).mean() * 100.0,

            "t1_intraday_mean_ret_pct": t1_ret.mean(),
            "t1_intraday_win_rate_pct": (t1_ret > 0).mean() * 100.0,

            "n_uptrend_true_rate_pct": _as_bool(valid["n_uptrend_structure"]).mean() * 100.0,
            "close_not_break_prev_n_low_rate_pct": _as_bool(valid["close_confirm_not_break_prev_n_low"]).mean() * 100.0,
            "strict_not_break_prev_n_low_rate_pct": _as_bool(valid["strict_not_break_prev_n_low"]).mean() * 100.0,
            "tolerant_not_break_prev_n_low_rate_pct": _as_bool(valid["tolerant_not_break_prev_n_low"]).mean() * 100.0,
            "avg_distance_to_prev_n_low_pct": pd.to_numeric(valid["distance_to_prev_n_low_pct"], errors="coerce").mean(),
            "median_distance_to_prev_n_low_pct": pd.to_numeric(valid["distance_to_prev_n_low_pct"], errors="coerce").median(),
        }
    )

    optional_cols = [
        "j",
        "daily_return_pct",
        "close_to_short_trend_ratio",
    ]

    for col in optional_cols:
        if col in valid.columns:
            s = pd.to_numeric(valid[col], errors="coerce")
            row[f"avg_{col}"] = s.mean()
            row[f"median_{col}"] = s.median()

    return row


def summarize_by_col(df: pd.DataFrame, col: str, ordered_buckets: list[str] | None = None) -> pd.DataFrame:
    rows = []

    rows.append(build_summary_row(df, col, "ALL_OVERSOLD_POOL"))

    if ordered_buckets is None:
        buckets = sorted([str(x) for x in df[col].dropna().unique()])
    else:
        buckets = ordered_buckets

    for bucket in buckets:
        one = df[df[col].astype(str) == bucket].copy()
        rows.append(build_summary_row(one, col, bucket))

    out = pd.DataFrame(rows)

    order = {"ALL_OVERSOLD_POOL": -1}
    order.update({bucket: i for i, bucket in enumerate(buckets)})

    out["_order"] = out["bucket"].map(order)
    out = out.sort_values(["group_col", "_order", "bucket"]).drop(columns=["_order"])

    return out


def add_diff_vs_all(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()

    all_row = out[out["bucket"] == "ALL_OVERSOLD_POOL"]
    if all_row.empty:
        return out

    base = all_row.iloc[0]

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
        "t1_intraday_win_rate_pct",
    ]

    for col in compare_cols:
        if col in out.columns:
            out[f"diff_vs_all_{col}"] = out[col] - base[col]

    return out


def build_daily_summary(df: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    rows = []

    for (date, bucket), one in df.groupby(["date", bucket_col], observed=False, dropna=False):
        valid = one.dropna(subset=["t1_open_to_t2_close_ret_pct"]).copy()

        if len(valid) == 0:
            continue

        ret = valid["t1_open_to_t2_close_ret_pct"]

        rows.append(
            {
                "date": date,
                "bucket_col": bucket_col,
                "bucket": str(bucket),
                "valid_count": len(valid),
                "symbol_count": valid["symbol"].nunique() if "symbol" in valid.columns else np.nan,
                "mean_ret_pct": ret.mean(),
                "median_ret_pct": ret.median(),
                "win_rate_pct": (ret > 0).mean() * 100.0,
                "hit_1pct_rate": (ret >= 1.0).mean() * 100.0,
                "hit_2pct_rate": (ret >= 2.0).mean() * 100.0,
                "loss_1pct_rate": (ret <= -1.0).mean() * 100.0,
                "loss_2pct_rate": (ret <= -2.0).mean() * 100.0,
            }
        )

    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame, title: str) -> None:
    cols = [
        "group_col",
        "bucket",
        "valid_count",
        "t1_t2_mean_ret_pct",
        "t1_t2_median_ret_pct",
        "t1_t2_win_rate_pct",
        "t1_t2_hit_2pct_rate",
        "t1_t2_loss_2pct_rate",
        "t2_high_hit_2pct_rate",
        "t2_low_loss_2pct_rate",
        "avg_distance_to_prev_n_low_pct",
        "diff_vs_all_t1_t2_mean_ret_pct",
        "diff_vs_all_t1_t2_median_ret_pct",
        "diff_vs_all_t1_t2_win_rate_pct",
        "diff_vs_all_t1_t2_hit_2pct_rate",
        "diff_vs_all_t1_t2_loss_2pct_rate",
    ]

    existing = [c for c in cols if c in summary.columns]

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 320,
        "display.float_format", "{:.4f}".format,
    ):
        print(summary[existing].to_string(index=False))


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("Analyze N-shaped uptrend structure inside oversold rebound pool")
    print("=" * 120)
    print(f"Oversold pool path : {OVERSOLD_POOL_PATH}")
    print(f"Market cache dir   : {MARKET_CACHE_DIR}")
    print(f"Output dir         : {OUTPUT_DIR}")
    print(f"Date range         : {START_DATE} -> {END_DATE}")
    print(f"Swing window       : {SWING_WINDOW}")
    print(f"Prev low tolerance : {PREV_LOW_TOLERANCE_PCT}%")

    pool = load_pool(OVERSOLD_POOL_PATH, "OVERSOLD_REBOUND_POOL")

    print()
    print("=" * 100)
    print("Loaded oversold pool")
    print("=" * 100)
    print(f"Rows       : {len(pool):,}")
    print(f"Symbols    : {pool['symbol'].nunique():,}")
    print(f"Date range : {pool['date'].min().date()} -> {pool['date'].max().date()}")

    market = load_market_cache(MARKET_CACHE_DIR)
    market_with_structure = add_n_structure(market, window=SWING_WINDOW)

    result = add_t1_t2_returns_and_structure(pool, market_with_structure)

    print()
    print("=" * 100)
    print("Merge check")
    print("=" * 100)
    print(f"Rows without T0 close       : {result['t0_close'].isna().sum():,}")
    print(f"Rows with T1/T2 data        : {result['has_t1_t2'].sum():,}")
    print(f"Rows missing T1/T2          : {(~result['has_t1_t2']).sum():,}")
    print(f"Rows with enough N points   : {result['has_enough_n_points'].fillna(False).sum():,}")
    print(f"N uptrend true rows         : {result['n_uptrend_structure'].fillna(False).sum():,}")
    print(f"Close not break N low rows  : {result['close_confirm_not_break_prev_n_low'].fillna(False).sum():,}")

    structure_summary = summarize_by_col(
        result,
        col="n_structure_state",
        ordered_buckets=[
            "N_POINTS_NOT_ENOUGH",
            "N_UPTREND_FALSE",
            "N_UPTREND_TRUE",
        ],
    )
    structure_summary = add_diff_vs_all(structure_summary)

    support_summary = summarize_by_col(
        result,
        col="n_structure_close_support_state",
        ordered_buckets=[
            "N_POINTS_NOT_ENOUGH",
            "N_FALSE",
            "N_TRUE_BUT_CLOSE_BREAK",
            "N_TRUE_AND_CLOSE_NOT_BREAK",
        ],
    )
    support_summary = add_diff_vs_all(support_summary)

    bool_summaries = []

    bool_cols = [
        "has_enough_n_points",
        "n_uptrend_structure",
        "strict_not_break_prev_n_low",
        "close_confirm_not_break_prev_n_low",
        "tolerant_not_break_prev_n_low",
    ]

    for col in bool_cols:
        tmp = result.copy()

        # Do NOT overwrite original boolean columns.
        # build_summary_row still needs the original boolean fields to calculate rates.
        bucket_col = f"{col}_bucket"
        tmp[bucket_col] = _as_bool(tmp[col]).map(
            {True: f"{col}=TRUE", False: f"{col}=FALSE"}
        )

        one_summary = summarize_by_col(
            tmp,
            col=bucket_col,
            ordered_buckets=[f"{col}=FALSE", f"{col}=TRUE"],
        )
        one_summary = add_diff_vs_all(one_summary)
        bool_summaries.append(one_summary)

    bool_summary = pd.concat(bool_summaries, ignore_index=True)

    daily_structure = build_daily_summary(result, "n_structure_state")
    daily_support = build_daily_summary(result, "n_structure_close_support_state")

    detail_path = OUTPUT_DIR / "oversold_rebound_n_structure_t1_t2_detail.csv"
    structure_summary_path = OUTPUT_DIR / "oversold_rebound_n_structure_summary.csv"
    support_summary_path = OUTPUT_DIR / "oversold_rebound_n_structure_support_summary.csv"
    bool_summary_path = OUTPUT_DIR / "oversold_rebound_n_structure_bool_summary.csv"
    daily_structure_path = OUTPUT_DIR / "oversold_rebound_n_structure_daily_summary.csv"
    daily_support_path = OUTPUT_DIR / "oversold_rebound_n_structure_support_daily_summary.csv"

    result.to_csv(detail_path, index=False, encoding="utf-8-sig")
    structure_summary.to_csv(structure_summary_path, index=False, encoding="utf-8-sig")
    support_summary.to_csv(support_summary_path, index=False, encoding="utf-8-sig")
    bool_summary.to_csv(bool_summary_path, index=False, encoding="utf-8-sig")
    daily_structure.to_csv(daily_structure_path, index=False, encoding="utf-8-sig")
    daily_support.to_csv(daily_support_path, index=False, encoding="utf-8-sig")

    print_summary(structure_summary, "N STRUCTURE SUMMARY")
    print_summary(support_summary, "N STRUCTURE + SUPPORT SUMMARY")
    print_summary(bool_summary, "N STRUCTURE BOOLEAN FIELD SUMMARY")

    print()
    print("=" * 120)
    print("Saved files")
    print("=" * 120)
    print(f"Detail CSV                  : {detail_path}")
    print(f"N structure summary CSV      : {structure_summary_path}")
    print(f"N support summary CSV        : {support_summary_path}")
    print(f"Boolean summary CSV          : {bool_summary_path}")
    print(f"N daily summary CSV          : {daily_structure_path}")
    print(f"N support daily summary CSV  : {daily_support_path}")

    print()
    print("=" * 120)
    print("How to judge")
    print("=" * 120)
    print("Focus on these rows:")
    print("1. N_UPTREND_TRUE vs N_UPTREND_FALSE")
    print("2. N_TRUE_AND_CLOSE_NOT_BREAK vs N_FALSE")
    print("3. close_confirm_not_break_prev_n_low=TRUE vs FALSE")
    print()
    print("If N_UPTREND_TRUE has higher mean/median/win-rate/hit-2pct and lower loss-2pct,")
    print("then N-shaped uptrend structure is a positive factor for the oversold rebound pool.")
    print()
    print("If N_TRUE_AND_CLOSE_NOT_BREAK is better, then use:")
    print("    n_uptrend_structure AND close_confirm_not_break_prev_n_low")
    print("as a candidate hard filter or high-weight scoring factor.")
    print()
    print("If N_UPTREND_TRUE reduces sample too much or is not better,")
    print("keep it as an analysis field first, not a hard filter.")


if __name__ == "__main__":
    main()
