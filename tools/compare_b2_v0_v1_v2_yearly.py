from __future__ import annotations

r"""
Compare B2 v0 vs v1 vs v2 pools by year.

Fixed version:
- When --save-detail is enabled, this script now correctly carries strategy/pool
  feature columns into trade detail.
- It supports your current B2 pool column names, for example:
    J                         -> b2_j
    pct_change_close          -> daily_return_pct
    volume_ratio_prev         -> b2_volume_ratio
    b1_days_ago_for_b2        -> b1_days_ago
    b2_quality_score          -> b2_quality_score
- This detail file can be used directly by XGBoost training scripts.

Buy rule:
- T+1 open

Sell modes:
- T+2 open
- T+2 close
- T+3 open
- T+3 close

Default paths:
- v0 pool:
  C:/Users/zyf37/Desktop/BackTest Data/pools/b2_confirm_select_strategy_v0_pool.parquet
- v1 pool:
  C:/Users/zyf37/Desktop/BackTest Data/pools/b2_confirm_select_strategy_v1_pool.parquet
- v2 pool:
  C:/Users/zyf37/Desktop/BackTest Data/pools/b2_confirm_select_strategy_v2_pool.parquet
- TXT data:
  C:/Users/zyf37/Desktop/BackTest Data/data

Example:
python .\tools\compare_b2_v0_v1_v2_yearly.py --save-detail

Outputs:
C:/Users/zyf37/Desktop/BackTest Data/output/b2_v0_v1_v2_yearly_compare
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_POOL_V0 = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\pools\b2_confirm_select_strategy_v0_pool.parquet"
)
DEFAULT_POOL_V1 = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\pools\b2_confirm_select_strategy_v1_pool.parquet"
)
DEFAULT_POOL_V2 = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\pools\b2_confirm_select_strategy_v2_pool.parquet"
)
DEFAULT_TXT_DIR = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\data"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\output\b2_v0_v1_v2_yearly_compare"
)

DEFAULT_YEARS = [2022, 2023, 2024, 2025, 2026]


# =============================================================================
# Feature mapping from pool columns to model-friendly detail columns
# =============================================================================

POOL_FEATURE_ALIAS_MAP: dict[str, list[str]] = {
    # Core B2 features for ML.
    "b2_j": ["b2_j", "b2_j_value", "J", "j"],
    "daily_return_pct": ["daily_return_pct", "pct_change_close"],
    "b2_volume_ratio": ["b2_volume_ratio", "volume_ratio_prev"],
    "b1_days_ago": ["b1_days_ago", "b1_days_ago_for_b2"],
    "b2_quality_score": ["b2_quality_score", "score"],

    # Existing useful features.
    "upper_shadow_ratio": ["upper_shadow_ratio"],
    "position_in_range_20": ["position_in_range_20"],
    "range_width_20": ["range_width_20"],
    "distance_to_previous_n_low": ["distance_to_previous_n_low"],
    "distance_to_ma20": ["distance_to_ma20"],
    "distance_to_bbi": ["distance_to_bbi"],
    "distance_to_yellow": ["distance_to_yellow"],
    "volume_ratio_ma5": ["volume_ratio_ma5"],
    "volume_q20_20": ["volume_q20_20"],
    "volume_min_20": ["volume_min_20"],

    # Boolean tags.
    "b1_low_range_position": ["b1_low_range_position"],
    "b1_near_previous_n_low": ["b1_near_previous_n_low"],
    "b1_in_range_bottom": ["b1_in_range_bottom"],
    "b1_near_ma_support": ["b1_near_ma_support"],
    "b1_position_ok": ["b1_position_ok"],
    "b1_j_ok": ["b1_j_ok"],
    "b1_low_volume": ["b1_low_volume"],
    "b1_extreme_low_volume": ["b1_extreme_low_volume"],
    "b1_not_break_prev_low": ["b1_not_break_prev_low"],
    "b1_valid": ["b1_valid"],
    "b1_within_b2_lookback": ["b1_within_b2_lookback"],

    "b2_return_ok": ["b2_return_ok"],
    "b2_bullish_candle": ["b2_bullish_candle"],
    "b2_volume_up": ["b2_volume_up"],
    "b2_double_volume": ["b2_double_volume"],
    "b2_sky_volume": ["b2_sky_volume"],
    "b2_j_ok": ["b2_j_ok"],
    "b2_j_high_zone": ["b2_j_high_zone"],
    "b2_upper_shadow_ok": ["b2_upper_shadow_ok"],
    "b2_tiny_upper_shadow": ["b2_tiny_upper_shadow"],
    "b2_upper_shadow_warning": ["b2_upper_shadow_warning"],

    # Original indicators, useful for later experiments.
    "K": ["K", "k"],
    "D": ["D", "d"],
    "diff": ["diff"],
    "dea": ["dea"],
    "macd": ["macd"],
    "ma20": ["ma20"],
    "bbi_for_b1": ["bbi_for_b1", "bbi"],
    "yellow_for_b1": ["yellow_for_b1", "yellow_ma", "yellow_line"],
    "prev_volume": ["prev_volume", "volume_prev_1"],
    "volume_ma5": ["volume_ma5", "vol_ma5"],
    "range_high_20": ["range_high_20"],
    "range_low_20": ["range_low_20"],
    "previous_n_low": ["previous_n_low"],
}


# =============================================================================
# Robust TXT reader
# =============================================================================

ENCODINGS = (
    "utf-8-sig",
    "utf-8",
    "gb18030",
    "gbk",
    "cp936",
    "mbcs",
)


@dataclass
class MarketRow:
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float | None = None


def iter_progress(items: Iterable, total: int | None = None, desc: str = ""):
    if tqdm is not None:
        return tqdm(items, total=total, desc=desc)
    return items


def _read_text_with_fallback(path: Path) -> str:
    last_error: Exception | None = None

    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except Exception as exc:
            last_error = exc

    try:
        return path.read_text(encoding="gb18030", errors="replace")
    except Exception as exc:
        if last_error is not None:
            raise RuntimeError(f"{last_error}; fallback error: {exc}") from exc
        raise


def _split_market_line(line: str) -> list[str]:
    line = line.strip().replace("\ufeff", "")
    line = line.replace(",", " ")
    line = line.replace("，", " ")
    line = line.replace(";", " ")
    line = re.sub(r"\s+", " ", line)
    return [x for x in line.split(" ") if x]


def _parse_date_token(token: str) -> pd.Timestamp | None:
    token = token.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%d/%m/%Y", "%Y.%m.%d"):
        try:
            return pd.to_datetime(token, format=fmt, errors="raise")
        except Exception:
            pass

    dt = pd.to_datetime(token, errors="coerce")
    if pd.isna(dt):
        return None
    return dt


def _to_float(token: str) -> float | None:
    token = token.strip().replace(",", "")
    if token in {"", "-", "--", "None", "nan", "NaN"}:
        return None
    try:
        return float(token)
    except Exception:
        return None


def _parse_market_line(line: str) -> MarketRow | None:
    parts = _split_market_line(line)
    if len(parts) < 6:
        return None

    dt = _parse_date_token(parts[0])
    if dt is None:
        return None

    nums: list[float] = []
    for p in parts[1:]:
        v = _to_float(p)
        if v is not None:
            nums.append(v)

    if len(nums) < 5:
        return None

    open_, high, low, close, volume = nums[:5]
    amount = nums[5] if len(nums) >= 6 else None

    if open_ <= 0 or high <= 0 or low <= 0 or close <= 0:
        return None
    if high < low:
        return None

    return MarketRow(
        date=pd.Timestamp(dt).normalize(),
        open=float(open_),
        high=float(high),
        low=float(low),
        close=float(close),
        volume=float(volume),
        amount=float(amount) if amount is not None else None,
    )


def read_tdx_txt_robust(path: Path) -> pd.DataFrame:
    text = _read_text_with_fallback(path)

    rows: list[MarketRow] = []
    for line in text.splitlines():
        row = _parse_market_line(line)
        if row is not None:
            rows.append(row)

    if not rows:
        raise ValueError("No valid market rows parsed")

    df = pd.DataFrame([r.__dict__ for r in rows])
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    return df


# =============================================================================
# Pool / symbol helpers
# =============================================================================

def normalize_symbol(raw: str) -> str:
    s = str(raw).strip().upper()
    s = s.replace(".", "#")
    if "#" in s:
        left, right = s.split("#", 1)
        left = left.upper()
        right = re.sub(r"\D", "", right)
        if left in {"SH", "SZ"} and len(right) == 6:
            return f"{left}#{right}"

    digits = re.sub(r"\D", "", s)
    if len(digits) >= 6:
        code = digits[-6:]
        prefix = "SH" if code.startswith("6") else "SZ"
        return f"{prefix}#{code}"
    return s


def infer_pool_columns(df: pd.DataFrame) -> tuple[str, str]:
    lower_map = {str(c).lower(): c for c in df.columns}

    date_col = ""
    symbol_col = ""

    for cand in ["date", "signal_date", "trade_date", "datetime"]:
        if cand in lower_map:
            date_col = str(lower_map[cand])
            break

    for cand in ["symbol", "code", "ts_code", "stock_code", "security_code"]:
        if cand in lower_map:
            symbol_col = str(lower_map[cand])
            break

    if not date_col:
        for c in df.columns:
            if "date" in str(c).lower():
                date_col = str(c)
                break

    if not symbol_col:
        for c in df.columns:
            name = str(c).lower()
            if "symbol" in name or "code" in name:
                symbol_col = str(c)
                break

    if not date_col or not symbol_col:
        raise ValueError(f"Cannot infer date/symbol columns from pool columns: {list(df.columns)}")

    return date_col, symbol_col


def load_pool(pool_path: Path, strategy_name: str) -> pd.DataFrame:
    if not pool_path.exists():
        raise FileNotFoundError(f"Pool file not found: {pool_path}")

    df = pd.read_parquet(pool_path)
    date_col, symbol_col = infer_pool_columns(df)

    out = df.copy()
    out["_signal_date"] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out["_symbol"] = out[symbol_col].map(normalize_symbol)
    out["_strategy"] = strategy_name

    out = out.dropna(subset=["_signal_date"]).copy()
    out = out[out["_symbol"].astype(str).str.len() >= 8].copy()

    out["_year"] = out["_signal_date"].dt.year

    print(f"[POOL] {strategy_name}: {pool_path}")
    print(f"       rows={len(out):,} | date_col={date_col} | symbol_col={symbol_col}")
    if len(out) > 0:
        print(f"       min_date={out['_signal_date'].min()} | max_date={out['_signal_date'].max()}")

    return out.reset_index(drop=True)


def _to_plain_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (bool,)):
        return int(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low == "true":
            return 1
        if low == "false":
            return 0
        return value
    return value


def add_pool_features_to_row(out: dict, pool_row: pd.Series) -> None:
    """
    Copy feature columns from pool row into the detail row.

    This is the key fix for ML training.
    The old script only copied a few outdated names, which caused the generated
    detail CSV to have many all-empty feature columns.
    """
    for canonical, candidates in POOL_FEATURE_ALIAS_MAP.items():
        value = None
        found = False

        for src in candidates:
            if src in pool_row.index:
                value = _to_plain_value(pool_row[src])
                found = True
                break

        if found:
            out[canonical] = value


def print_feature_coverage(detail: pd.DataFrame) -> None:
    feature_cols = [c for c in POOL_FEATURE_ALIAS_MAP.keys() if c in detail.columns]
    if not feature_cols:
        print("[WARN] No mapped feature columns in detail.")
        return

    print("-" * 120)
    print("Mapped feature coverage in trade detail")
    print("-" * 120)

    rows = []
    for c in feature_cols:
        non_null = int(detail[c].notna().sum())
        rows.append({
            "feature": c,
            "non_null": non_null,
            "non_null_pct": round(non_null / max(len(detail), 1) * 100, 4),
        })

    cov = pd.DataFrame(rows).sort_values(["non_null", "feature"], ascending=[False, True])
    print(cov.to_string(index=False))


# =============================================================================
# Market cache
# =============================================================================

def build_market_cache(txt_dir: Path, symbols: list[str]) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    cache: dict[str, pd.DataFrame] = {}
    warnings: list[dict] = []

    for symbol in iter_progress(symbols, total=len(symbols), desc="Loading market TXT"):
        path = txt_dir / f"{symbol}.txt"
        if not path.exists():
            warnings.append({"symbol": symbol, "path": str(path), "error": "file not found"})
            continue

        try:
            cache[symbol] = read_tdx_txt_robust(path)
        except Exception as exc:
            warnings.append({"symbol": symbol, "path": str(path), "error": str(exc)})
            print(f"[WARN] Skip {path}: {exc}")

    return cache, warnings


def get_next_n_trading_rows(market: pd.DataFrame, signal_date: pd.Timestamp, n: int) -> list[pd.Series]:
    market = market.sort_values("date").reset_index(drop=True)
    future = market[market["date"] > signal_date].head(n)
    return [future.iloc[i] for i in range(len(future))]


def pct(buy: float, sell: float) -> float:
    if buy == 0:
        return 0.0
    return (sell / buy - 1.0) * 100.0


# =============================================================================
# Detail calculation
# =============================================================================

def analyze_pool(pool: pd.DataFrame, market_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    strategy = pool["_strategy"].iloc[0] if len(pool) else ""

    for _, r in iter_progress(pool.iterrows(), total=len(pool), desc=f"Analyzing {strategy}"):
        symbol = r["_symbol"]
        signal_date = pd.Timestamp(r["_signal_date"]).normalize()

        market = market_cache.get(symbol)
        if market is None or market.empty:
            continue

        future_rows = get_next_n_trading_rows(market, signal_date, n=3)
        if len(future_rows) < 3:
            continue

        t1, t2, t3 = future_rows[0], future_rows[1], future_rows[2]
        buy = float(t1["open"])

        t0_close = None
        t0_rows = market[market["date"] == signal_date]
        if len(t0_rows) > 0:
            t0_close = float(t0_rows.iloc[-1]["close"])

        t1_high = float(t1["high"])
        t1_low = float(t1["low"])
        t2_high = float(t2["high"])
        t2_low = float(t2["low"])
        t3_high = float(t3["high"])
        t3_low = float(t3["low"])

        t2_open = float(t2["open"])
        t2_close = float(t2["close"])
        t3_open = float(t3["open"])
        t3_close = float(t3["close"])

        ret_t2_open = pct(buy, t2_open)
        ret_t2_close = pct(buy, t2_close)
        ret_t3_open = pct(buy, t3_open)
        ret_t3_close = pct(buy, t3_close)

        max_high_t1_t2 = max(t1_high, t2_high)
        min_low_t1_t2 = min(t1_low, t2_low)
        max_high_t1_t3 = max(t1_high, t2_high, t3_high)
        min_low_t1_t3 = min(t1_low, t2_low, t3_low)

        out = {
            "strategy": r["_strategy"],
            "symbol": symbol,
            "signal_date": signal_date.date().isoformat(),
            "year": int(signal_date.year),

            "t1_date": pd.Timestamp(t1["date"]).date().isoformat(),
            "t2_date": pd.Timestamp(t2["date"]).date().isoformat(),
            "t3_date": pd.Timestamp(t3["date"]).date().isoformat(),

            "t0_close": t0_close,
            "buy_t1_open": buy,
            "t1_open": buy,
            "t1_close": float(t1["close"]),
            "t1_high": t1_high,
            "t1_low": t1_low,
            "t2_open": t2_open,
            "t2_close": t2_close,
            "t2_high": t2_high,
            "t2_low": t2_low,
            "t3_open": t3_open,
            "t3_close": t3_close,
            "t3_high": t3_high,
            "t3_low": t3_low,

            "ret_t1_open_to_t2_open_pct": ret_t2_open,
            "ret_t1_open_to_t2_close_pct": ret_t2_close,
            "ret_t1_open_to_t3_open_pct": ret_t3_open,
            "ret_t1_open_to_t3_close_pct": ret_t3_close,

            "win_t2_open": ret_t2_open > 0,
            "win_t2_close": ret_t2_close > 0,
            "win_t3_open": ret_t3_open > 0,
            "win_t3_close": ret_t3_close > 0,

            "max_opportunity_t1_t2_pct": pct(buy, max_high_t1_t2),
            "max_drawdown_t1_t2_pct": pct(buy, min_low_t1_t2),
            "max_opportunity_t1_t3_pct": pct(buy, max_high_t1_t3),
            "max_drawdown_t1_t3_pct": pct(buy, min_low_t1_t3),
        }

        if t0_close is not None and t0_close != 0:
            out["t1_open_gap_pct"] = (buy / t0_close - 1.0) * 100.0
        else:
            out["t1_open_gap_pct"] = None

        for x in [2, 3, 5]:
            out[f"hit_plus_{x}pct_t1_t2"] = out["max_opportunity_t1_t2_pct"] >= x
            out[f"hit_minus_{x}pct_t1_t2"] = out["max_drawdown_t1_t2_pct"] <= -x
            out[f"hit_plus_{x}pct_t1_t3"] = out["max_opportunity_t1_t3_pct"] >= x
            out[f"hit_minus_{x}pct_t1_t3"] = out["max_drawdown_t1_t3_pct"] <= -x

        # Key fix: copy mapped pool features into detail row.
        add_pool_features_to_row(out, r)

        rows.append(out)

    return pd.DataFrame(rows)


# =============================================================================
# Summary
# =============================================================================

def safe_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 0.0
    return round(float(pd.to_numeric(df[col], errors="coerce").mean()), 6)


def safe_median(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 0.0
    return round(float(pd.to_numeric(df[col], errors="coerce").median()), 6)


def safe_max(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 0.0
    return round(float(pd.to_numeric(df[col], errors="coerce").max()), 6)


def safe_min(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 0.0
    return round(float(pd.to_numeric(df[col], errors="coerce").min()), 6)


def bool_rate(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 0.0
    return round(float(pd.Series(df[col]).astype(bool).mean()) * 100.0, 6)


def summarize_detail_by_year(detail: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    rows: list[dict] = []

    modes = [
        ("t2_open", "win_t2_open", "ret_t1_open_to_t2_open_pct"),
        ("t2_close", "win_t2_close", "ret_t1_open_to_t2_close_pct"),
        ("t3_open", "win_t3_open", "ret_t1_open_to_t3_open_pct"),
        ("t3_close", "win_t3_close", "ret_t1_open_to_t3_close_pct"),
    ]

    strategies = sorted(detail["strategy"].dropna().unique().tolist())

    for strategy in strategies:
        s_df = detail[detail["strategy"] == strategy].copy()

        for year in years:
            y_df = s_df[s_df["year"] == year].copy()
            for mode, win_col, ret_col in modes:
                rows.append({
                    "strategy": strategy,
                    "period": str(year),
                    "year": year,
                    "sell_mode": mode,
                    "trade_count": int(len(y_df)),
                    "win_rate_pct": bool_rate(y_df, win_col),
                    "avg_ret_pct": safe_mean(y_df, ret_col),
                    "median_ret_pct": safe_median(y_df, ret_col),
                    "best_ret_pct": safe_max(y_df, ret_col),
                    "worst_ret_pct": safe_min(y_df, ret_col),
                    "avg_max_opportunity_t1_t3_pct": safe_mean(y_df, "max_opportunity_t1_t3_pct"),
                    "avg_max_drawdown_t1_t3_pct": safe_mean(y_df, "max_drawdown_t1_t3_pct"),
                    "hit_plus_2pct_t1_t3_rate_pct": bool_rate(y_df, "hit_plus_2pct_t1_t3"),
                    "hit_plus_5pct_t1_t3_rate_pct": bool_rate(y_df, "hit_plus_5pct_t1_t3"),
                    "hit_minus_2pct_t1_t3_rate_pct": bool_rate(y_df, "hit_minus_2pct_t1_t3"),
                    "hit_minus_5pct_t1_t3_rate_pct": bool_rate(y_df, "hit_minus_5pct_t1_t3"),
                })

        for mode, win_col, ret_col in modes:
            rows.append({
                "strategy": strategy,
                "period": "ALL",
                "year": 0,
                "sell_mode": mode,
                "trade_count": int(len(s_df)),
                "win_rate_pct": bool_rate(s_df, win_col),
                "avg_ret_pct": safe_mean(s_df, ret_col),
                "median_ret_pct": safe_median(s_df, ret_col),
                "best_ret_pct": safe_max(s_df, ret_col),
                "worst_ret_pct": safe_min(s_df, ret_col),
                "avg_max_opportunity_t1_t3_pct": safe_mean(s_df, "max_opportunity_t1_t3_pct"),
                "avg_max_drawdown_t1_t3_pct": safe_mean(s_df, "max_drawdown_t1_t3_pct"),
                "hit_plus_2pct_t1_t3_rate_pct": bool_rate(s_df, "hit_plus_2pct_t1_t3"),
                "hit_plus_5pct_t1_t3_rate_pct": bool_rate(s_df, "hit_plus_5pct_t1_t3"),
                "hit_minus_2pct_t1_t3_rate_pct": bool_rate(s_df, "hit_minus_2pct_t1_t3"),
                "hit_minus_5pct_t1_t3_rate_pct": bool_rate(s_df, "hit_minus_5pct_t1_t3"),
            })

    return pd.DataFrame(rows)


def build_pairwise_delta(summary: pd.DataFrame, base: str, compare: str) -> pd.DataFrame:
    a = summary[summary["strategy"] == base].copy()
    b = summary[summary["strategy"] == compare].copy()

    key_cols = ["period", "year", "sell_mode"]
    merged = a.merge(
        b,
        on=key_cols,
        suffixes=(f"_{base}", f"_{compare}"),
        how="outer",
    )

    out = pd.DataFrame()
    for c in key_cols:
        out[c] = merged[c]

    out["base_strategy"] = base
    out["compare_strategy"] = compare
    out["delta_name"] = f"{compare}_minus_{base}"

    out[f"trade_count_{base}"] = merged[f"trade_count_{base}"].fillna(0).astype(int)
    out[f"trade_count_{compare}"] = merged[f"trade_count_{compare}"].fillna(0).astype(int)
    out["trade_count_delta"] = out[f"trade_count_{compare}"] - out[f"trade_count_{base}"]
    out["trade_count_keep_pct"] = (
        out[f"trade_count_{compare}"] / out[f"trade_count_{base}"].replace(0, pd.NA) * 100
    ).round(6)

    metric_cols = [
        "win_rate_pct",
        "avg_ret_pct",
        "median_ret_pct",
        "best_ret_pct",
        "worst_ret_pct",
        "avg_max_opportunity_t1_t3_pct",
        "avg_max_drawdown_t1_t3_pct",
        "hit_plus_2pct_t1_t3_rate_pct",
        "hit_plus_5pct_t1_t3_rate_pct",
        "hit_minus_2pct_t1_t3_rate_pct",
        "hit_minus_5pct_t1_t3_rate_pct",
    ]

    for m in metric_cols:
        out[f"{m}_{base}"] = pd.to_numeric(merged[f"{m}_{base}"], errors="coerce")
        out[f"{m}_{compare}"] = pd.to_numeric(merged[f"{m}_{compare}"], errors="coerce")
        out[f"{m}_delta"] = (out[f"{m}_{compare}"] - out[f"{m}_{base}"]).round(6)

    out["better_by_avg_ret"] = out["avg_ret_pct_delta"] > 0
    out["better_by_win_rate"] = out["win_rate_pct_delta"] > 0
    out["better_both"] = out["better_by_avg_ret"] & out["better_by_win_rate"]

    return out.sort_values(["year", "sell_mode"]).reset_index(drop=True)


def build_best_mode_table(summary: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    rows = []
    for (strategy, period), g in summary.groupby(["strategy", "period"], dropna=False):
        if g.empty:
            continue

        best_avg = g.sort_values("avg_ret_pct", ascending=False).iloc[0]
        best_win = g.sort_values("win_rate_pct", ascending=False).iloc[0]

        rows.append({
            "strategy": strategy,
            "period": period,
            "best_avg_sell_mode": best_avg["sell_mode"],
            "best_avg_ret_pct": best_avg["avg_ret_pct"],
            "best_avg_win_rate_pct": best_avg["win_rate_pct"],
            "best_avg_trade_count": best_avg["trade_count"],
            "best_win_sell_mode": best_win["sell_mode"],
            "best_win_rate_pct": best_win["win_rate_pct"],
            "best_win_avg_ret_pct": best_win["avg_ret_pct"],
            "best_win_trade_count": best_win["trade_count"],
        })

    out = pd.DataFrame(rows)
    period_order = {str(y): y for y in years}
    period_order["ALL"] = 9999
    out["_order"] = out["period"].map(period_order).fillna(9998)
    out = out.sort_values(["_order", "strategy"]).drop(columns=["_order"])
    return out.reset_index(drop=True)


# =============================================================================
# Main
# =============================================================================

def parse_years(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare B2 v0 vs v1 vs v2 yearly quality.")
    parser.add_argument("--pool-v0", type=Path, default=DEFAULT_POOL_V0)
    parser.add_argument("--pool-v1", type=Path, default=DEFAULT_POOL_V1)
    parser.add_argument("--pool-v2", type=Path, default=DEFAULT_POOL_V2)
    parser.add_argument("--txt-dir", type=Path, default=DEFAULT_TXT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--years", default="2022,2023,2024,2025,2026")
    parser.add_argument("--save-detail", action="store_true", help="Save per-trade detail CSV.")
    parser.add_argument("--limit", type=int, default=0, help="Debug only: limit rows per pool. 0 = no limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = parse_years(args.years)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("Compare B2 v0 vs v1 vs v2 by year")
    print("=" * 120)
    print(f"Pool v0    : {args.pool_v0}")
    print(f"Pool v1    : {args.pool_v1}")
    print(f"Pool v2    : {args.pool_v2}")
    print(f"TXT dir    : {args.txt_dir}")
    print(f"Output dir : {args.output_dir}")
    print(f"Years      : {years}")
    print("-" * 120)

    pool_v0 = load_pool(args.pool_v0, "v0")
    pool_v1 = load_pool(args.pool_v1, "v1")
    pool_v2 = load_pool(args.pool_v2, "v2")

    if args.limit and args.limit > 0:
        pool_v0 = pool_v0.head(args.limit).copy()
        pool_v1 = pool_v1.head(args.limit).copy()
        pool_v2 = pool_v2.head(args.limit).copy()

    print(f"v0 pool rows: {len(pool_v0):,}")
    print(f"v1 pool rows: {len(pool_v1):,}")
    print(f"v2 pool rows: {len(pool_v2):,}")

    all_symbols = sorted(
        set(pool_v0["_symbol"].dropna().unique())
        | set(pool_v1["_symbol"].dropna().unique())
        | set(pool_v2["_symbol"].dropna().unique())
    )
    print(f"Unique symbols to load: {len(all_symbols):,}")

    market_cache, warnings = build_market_cache(args.txt_dir, all_symbols)
    print(f"Market files loaded: {len(market_cache):,}")
    print(f"Market warnings    : {len(warnings):,}")

    detail_v0 = analyze_pool(pool_v0, market_cache)
    detail_v1 = analyze_pool(pool_v1, market_cache)
    detail_v2 = analyze_pool(pool_v2, market_cache)
    detail = pd.concat([detail_v0, detail_v1, detail_v2], ignore_index=True)

    print_feature_coverage(detail)

    summary = summarize_detail_by_year(detail, years)

    delta_v1_v0 = build_pairwise_delta(summary, base="v0", compare="v1")
    delta_v2_v0 = build_pairwise_delta(summary, base="v0", compare="v2")
    delta_v2_v1 = build_pairwise_delta(summary, base="v1", compare="v2")

    delta_vs_v0 = pd.concat([delta_v1_v0, delta_v2_v0], ignore_index=True)
    delta_pairwise = pd.concat([delta_v1_v0, delta_v2_v0, delta_v2_v1], ignore_index=True)

    best_modes = build_best_mode_table(summary, years)

    summary_path = args.output_dir / "b2_v0_v1_v2_yearly_summary.csv"
    delta_vs_v0_path = args.output_dir / "b2_v0_v1_v2_yearly_delta_vs_v0.csv"
    delta_pairwise_path = args.output_dir / "b2_v0_v1_v2_yearly_delta_pairwise.csv"
    best_modes_path = args.output_dir / "b2_v0_v1_v2_best_sell_modes.csv"
    warnings_path = args.output_dir / "b2_v0_v1_v2_market_read_warnings.csv"
    detail_path = args.output_dir / "b2_v0_v1_v2_trade_detail.csv"
    feature_coverage_path = args.output_dir / "b2_v0_v1_v2_detail_feature_coverage.csv"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    delta_vs_v0.to_csv(delta_vs_v0_path, index=False, encoding="utf-8-sig")
    delta_pairwise.to_csv(delta_pairwise_path, index=False, encoding="utf-8-sig")
    best_modes.to_csv(best_modes_path, index=False, encoding="utf-8-sig")

    # Save feature coverage table.
    feature_cols = [c for c in POOL_FEATURE_ALIAS_MAP.keys() if c in detail.columns]
    coverage_rows = []
    for c in feature_cols:
        non_null = int(detail[c].notna().sum())
        coverage_rows.append({
            "feature": c,
            "non_null": non_null,
            "non_null_pct": round(non_null / max(len(detail), 1) * 100, 4),
        })
    pd.DataFrame(coverage_rows).to_csv(feature_coverage_path, index=False, encoding="utf-8-sig")

    if warnings:
        pd.DataFrame(warnings).to_csv(warnings_path, index=False, encoding="utf-8-sig")

    if args.save_detail:
        detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print("-" * 120)
    print("Best sell modes")
    print("-" * 120)
    print(best_modes.to_string(index=False))

    print("-" * 120)
    print("Delta preview: v2 - v0, T+3 close")
    print("-" * 120)
    preview = delta_v2_v0[delta_v2_v0["sell_mode"] == "t3_close"]
    preview_cols = [
        "period",
        "sell_mode",
        "trade_count_v0",
        "trade_count_v2",
        "trade_count_keep_pct",
        "win_rate_pct_delta",
        "avg_ret_pct_delta",
        "median_ret_pct_delta",
        "worst_ret_pct_delta",
        "hit_minus_5pct_t1_t3_rate_pct_delta",
    ]
    preview_cols = [c for c in preview_cols if c in preview.columns]
    print(preview[preview_cols].to_string(index=False))

    print("-" * 120)
    print("Output files")
    print("-" * 120)
    print(f"Summary         : {summary_path}")
    print(f"Delta vs v0     : {delta_vs_v0_path}")
    print(f"Delta pairwise  : {delta_pairwise_path}")
    print(f"Best modes      : {best_modes_path}")
    print(f"Feature coverage: {feature_coverage_path}")
    if warnings:
        print(f"Warnings        : {warnings_path}")
    if args.save_detail:
        print(f"Detail          : {detail_path}")
    else:
        print("Detail          : not saved. Add --save-detail to save trade detail.")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
