from __future__ import annotations

r"""
Analyze incremental quality of v1 pool vs v0 pool.

Purpose:
- Compare the part added by v1 after lowering B2 daily return threshold by 1%.
- Split candidates into:
    1. both_v0_v1  : appears in both v0 and v1
    2. v1_only     : appears only in v1, i.e. incremental candidates
    3. v0_only     : appears only in v0

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
- TXT data:
  C:/Users/zyf37/Desktop/BackTest Data/data
- Output:
  C:/Users/zyf37/Desktop/BackTest Data/output/b2_v0_v1_incremental_analysis

Run:
python .\tools\analyze_b2_v1_incremental_vs_v0.py

Save detail:
python .\tools\analyze_b2_v1_incremental_vs_v0.py --save-detail
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
except Exception:
    tqdm = None


DEFAULT_POOL_V0 = Path(r"C:\Users\zyf37\Desktop\BackTest Data\pools\b2_confirm_select_strategy_v0_pool.parquet")
DEFAULT_POOL_V1 = Path(r"C:\Users\zyf37\Desktop\BackTest Data\pools\b2_confirm_select_strategy_v1_pool.parquet")
DEFAULT_TXT_DIR = Path(r"C:\Users\zyf37\Desktop\BackTest Data\data")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\zyf37\Desktop\BackTest Data\output\b2_v0_v1_incremental_analysis")


POOL_FEATURE_ALIAS_MAP: dict[str, list[str]] = {
    "b2_j": ["b2_j", "b2_j_value", "J", "j"],
    "daily_return_pct": ["daily_return_pct", "pct_change_close"],
    "b2_volume_ratio": ["b2_volume_ratio", "volume_ratio_prev"],
    "b1_days_ago": ["b1_days_ago", "b1_days_ago_for_b2"],
    "b1_j_value": ["b1_j_value"],
    "b1_volume_value": ["b1_volume_value"],
    "b2_quality_score": ["b2_quality_score", "score"],
    "score": ["score", "b2_quality_score"],
    "score_pct": ["score_pct"],
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


ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp936", "mbcs")


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
    line = line.replace(",", " ").replace("，", " ").replace(";", " ")
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


def normalize_symbol(raw: str) -> str:
    s = str(raw).strip().upper().replace(".", "#")
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
    out["_key"] = out["_signal_date"].dt.strftime("%Y-%m-%d") + "|" + out["_symbol"]

    out = out.dropna(subset=["_signal_date"]).copy()
    out = out[out["_symbol"].astype(str).str.len() >= 8].copy()
    out["_year"] = out["_signal_date"].dt.year

    # Deduplicate by date + symbol to avoid double counting.
    before = len(out)
    out = out.drop_duplicates(subset=["_key"], keep="last").copy()
    after = len(out)

    print(f"[POOL] {strategy_name}: {pool_path}")
    print(f"       rows={after:,} | duplicate_removed={before - after:,} | date_col={date_col} | symbol_col={symbol_col}")
    if len(out) > 0:
        print(f"       min_date={out['_signal_date'].min()} | max_date={out['_signal_date'].max()}")

    return out.reset_index(drop=True)


def classify_pool_membership(pool_v0: pd.DataFrame, pool_v1: pd.DataFrame) -> pd.DataFrame:
    v0_keys = set(pool_v0["_key"])
    v1_keys = set(pool_v1["_key"])

    both_keys = v0_keys & v1_keys
    v0_only_keys = v0_keys - v1_keys
    v1_only_keys = v1_keys - v0_keys

    # Use v1 row for both/v1_only, and v0 row for v0_only.
    both = pool_v1[pool_v1["_key"].isin(both_keys)].copy()
    both["_group"] = "both_v0_v1"

    v1_only = pool_v1[pool_v1["_key"].isin(v1_only_keys)].copy()
    v1_only["_group"] = "v1_only"

    v0_only = pool_v0[pool_v0["_key"].isin(v0_only_keys)].copy()
    v0_only["_group"] = "v0_only"

    out = pd.concat([both, v1_only, v0_only], ignore_index=True)

    print("=" * 120)
    print("Pool membership split")
    print("=" * 120)
    print(f"v0 total     : {len(v0_keys):,}")
    print(f"v1 total     : {len(v1_keys):,}")
    print(f"both_v0_v1   : {len(both_keys):,}")
    print(f"v1_only      : {len(v1_only_keys):,}")
    print(f"v0_only      : {len(v0_only_keys):,}")
    print(f"v1/v0 ratio  : {len(v1_keys) / max(len(v0_keys), 1) * 100:.4f}%")
    print("-" * 120)

    return out.reset_index(drop=True)


def _to_plain_value(value):
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, bool):
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
    for canonical, candidates in POOL_FEATURE_ALIAS_MAP.items():
        for src in candidates:
            if src in pool_row.index:
                out[canonical] = _to_plain_value(pool_row[src])
                break


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


def analyze_pool(pool: pd.DataFrame, market_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []

    for _, r in iter_progress(pool.iterrows(), total=len(pool), desc="Analyzing membership groups"):
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

        t2_open = float(t2["open"])
        t2_close = float(t2["close"])
        t3_open = float(t3["open"])
        t3_close = float(t3["close"])

        ret_t2_open = pct(buy, t2_open)
        ret_t2_close = pct(buy, t2_close)
        ret_t3_open = pct(buy, t3_open)
        ret_t3_close = pct(buy, t3_close)

        t1_high, t1_low = float(t1["high"]), float(t1["low"])
        t2_high, t2_low = float(t2["high"]), float(t2["low"])
        t3_high, t3_low = float(t3["high"]), float(t3["low"])

        max_high_t1_t2 = max(t1_high, t2_high)
        min_low_t1_t2 = min(t1_low, t2_low)
        max_high_t1_t3 = max(t1_high, t2_high, t3_high)
        min_low_t1_t3 = min(t1_low, t2_low, t3_low)

        out = {
            "group": r["_group"],
            "source_strategy": r["_strategy"],
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

        for x in [1, 2, 3, 5]:
            out[f"hit_plus_{x}pct_t1_t2"] = out["max_opportunity_t1_t2_pct"] >= x
            out[f"hit_minus_{x}pct_t1_t2"] = out["max_drawdown_t1_t2_pct"] <= -x
            out[f"hit_plus_{x}pct_t1_t3"] = out["max_opportunity_t1_t3_pct"] >= x
            out[f"hit_minus_{x}pct_t1_t3"] = out["max_drawdown_t1_t3_pct"] <= -x

        add_pool_features_to_row(out, r)
        rows.append(out)

    return pd.DataFrame(rows)


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


def summarize_detail(detail: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    rows: list[dict] = []

    groups = ["both_v0_v1", "v1_only", "v0_only"]
    periods = [str(y) for y in years] + ["ALL"]

    modes = [
        ("t2_open", "win_t2_open", "ret_t1_open_to_t2_open_pct"),
        ("t2_close", "win_t2_close", "ret_t1_open_to_t2_close_pct"),
        ("t3_open", "win_t3_open", "ret_t1_open_to_t3_open_pct"),
        ("t3_close", "win_t3_close", "ret_t1_open_to_t3_close_pct"),
    ]

    for group in groups:
        g_df = detail[detail["group"] == group].copy()

        for period in periods:
            if period == "ALL":
                p_df = g_df.copy()
                year_value = 0
            else:
                year_value = int(period)
                p_df = g_df[g_df["year"] == year_value].copy()

            for mode, win_col, ret_col in modes:
                rows.append({
                    "group": group,
                    "period": period,
                    "year": year_value,
                    "sell_mode": mode,
                    "trade_count": int(len(p_df)),
                    "win_rate_pct": bool_rate(p_df, win_col),
                    "avg_ret_pct": safe_mean(p_df, ret_col),
                    "median_ret_pct": safe_median(p_df, ret_col),
                    "best_ret_pct": safe_max(p_df, ret_col),
                    "worst_ret_pct": safe_min(p_df, ret_col),
                    "avg_t1_open_gap_pct": safe_mean(p_df, "t1_open_gap_pct"),
                    "avg_max_opportunity_t1_t2_pct": safe_mean(p_df, "max_opportunity_t1_t2_pct"),
                    "avg_max_drawdown_t1_t2_pct": safe_mean(p_df, "max_drawdown_t1_t2_pct"),
                    "avg_max_opportunity_t1_t3_pct": safe_mean(p_df, "max_opportunity_t1_t3_pct"),
                    "avg_max_drawdown_t1_t3_pct": safe_mean(p_df, "max_drawdown_t1_t3_pct"),
                    "hit_plus_1pct_t1_t2_rate_pct": bool_rate(p_df, "hit_plus_1pct_t1_t2"),
                    "hit_plus_2pct_t1_t2_rate_pct": bool_rate(p_df, "hit_plus_2pct_t1_t2"),
                    "hit_plus_3pct_t1_t2_rate_pct": bool_rate(p_df, "hit_plus_3pct_t1_t2"),
                    "hit_plus_5pct_t1_t2_rate_pct": bool_rate(p_df, "hit_plus_5pct_t1_t2"),
                    "hit_minus_1pct_t1_t2_rate_pct": bool_rate(p_df, "hit_minus_1pct_t1_t2"),
                    "hit_minus_2pct_t1_t2_rate_pct": bool_rate(p_df, "hit_minus_2pct_t1_t2"),
                    "hit_minus_3pct_t1_t2_rate_pct": bool_rate(p_df, "hit_minus_3pct_t1_t2"),
                    "hit_minus_5pct_t1_t2_rate_pct": bool_rate(p_df, "hit_minus_5pct_t1_t2"),
                    "hit_plus_1pct_t1_t3_rate_pct": bool_rate(p_df, "hit_plus_1pct_t1_t3"),
                    "hit_plus_2pct_t1_t3_rate_pct": bool_rate(p_df, "hit_plus_2pct_t1_t3"),
                    "hit_plus_3pct_t1_t3_rate_pct": bool_rate(p_df, "hit_plus_3pct_t1_t3"),
                    "hit_plus_5pct_t1_t3_rate_pct": bool_rate(p_df, "hit_plus_5pct_t1_t3"),
                    "hit_minus_1pct_t1_t3_rate_pct": bool_rate(p_df, "hit_minus_1pct_t1_t3"),
                    "hit_minus_2pct_t1_t3_rate_pct": bool_rate(p_df, "hit_minus_2pct_t1_t3"),
                    "hit_minus_3pct_t1_t3_rate_pct": bool_rate(p_df, "hit_minus_3pct_t1_t3"),
                    "hit_minus_5pct_t1_t3_rate_pct": bool_rate(p_df, "hit_minus_5pct_t1_t3"),
                })

    return pd.DataFrame(rows)


def build_group_delta(summary: pd.DataFrame, base_group: str, compare_group: str) -> pd.DataFrame:
    a = summary[summary["group"] == base_group].copy()
    b = summary[summary["group"] == compare_group].copy()

    key_cols = ["period", "year", "sell_mode"]
    merged = a.merge(
        b,
        on=key_cols,
        suffixes=(f"_{base_group}", f"_{compare_group}"),
        how="outer",
    )

    out = pd.DataFrame()
    for c in key_cols:
        out[c] = merged[c]

    out["base_group"] = base_group
    out["compare_group"] = compare_group
    out["delta_name"] = f"{compare_group}_minus_{base_group}"

    out[f"trade_count_{base_group}"] = merged[f"trade_count_{base_group}"].fillna(0).astype(int)
    out[f"trade_count_{compare_group}"] = merged[f"trade_count_{compare_group}"].fillna(0).astype(int)

    metric_cols = [
        "win_rate_pct",
        "avg_ret_pct",
        "median_ret_pct",
        "worst_ret_pct",
        "avg_t1_open_gap_pct",
        "avg_max_opportunity_t1_t2_pct",
        "avg_max_drawdown_t1_t2_pct",
        "avg_max_opportunity_t1_t3_pct",
        "avg_max_drawdown_t1_t3_pct",
        "hit_plus_2pct_t1_t2_rate_pct",
        "hit_plus_5pct_t1_t2_rate_pct",
        "hit_minus_2pct_t1_t2_rate_pct",
        "hit_minus_5pct_t1_t2_rate_pct",
        "hit_plus_2pct_t1_t3_rate_pct",
        "hit_plus_5pct_t1_t3_rate_pct",
        "hit_minus_2pct_t1_t3_rate_pct",
        "hit_minus_5pct_t1_t3_rate_pct",
    ]

    for m in metric_cols:
        out[f"{m}_{base_group}"] = pd.to_numeric(merged[f"{m}_{base_group}"], errors="coerce")
        out[f"{m}_{compare_group}"] = pd.to_numeric(merged[f"{m}_{compare_group}"], errors="coerce")
        out[f"{m}_delta"] = (out[f"{m}_{compare_group}"] - out[f"{m}_{base_group}"]).round(6)

    out["better_by_avg_ret"] = out["avg_ret_pct_delta"] > 0
    out["better_by_win_rate"] = out["win_rate_pct_delta"] > 0
    out["better_both"] = out["better_by_avg_ret"] & out["better_by_win_rate"]

    return out.sort_values(["year", "sell_mode"]).reset_index(drop=True)


def build_feature_profile(detail: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    feature_cols = [
        c for c in [
            "b2_j",
            "daily_return_pct",
            "b2_volume_ratio",
            "b1_days_ago",
            "b1_j_value",
            "upper_shadow_ratio",
            "position_in_range_20",
            "distance_to_previous_n_low",
            "volume_ratio_ma5",
            "score",
            "score_pct",
            "b2_quality_score",
            "t1_open_gap_pct",
        ]
        if c in detail.columns
    ]

    rows = []
    periods = [str(y) for y in years] + ["ALL"]

    for group in sorted(detail["group"].dropna().unique()):
        g_df = detail[detail["group"] == group].copy()

        for period in periods:
            if period == "ALL":
                p_df = g_df.copy()
                year_value = 0
            else:
                year_value = int(period)
                p_df = g_df[g_df["year"] == year_value].copy()

            for c in feature_cols:
                s = pd.to_numeric(p_df[c], errors="coerce")
                rows.append({
                    "group": group,
                    "period": period,
                    "year": year_value,
                    "feature": c,
                    "count": int(s.notna().sum()),
                    "mean": round(float(s.mean()), 6) if s.notna().sum() else 0.0,
                    "median": round(float(s.median()), 6) if s.notna().sum() else 0.0,
                    "p25": round(float(s.quantile(0.25)), 6) if s.notna().sum() else 0.0,
                    "p75": round(float(s.quantile(0.75)), 6) if s.notna().sum() else 0.0,
                })

    return pd.DataFrame(rows)


def parse_years(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze v1 incremental candidates versus v0.")
    parser.add_argument("--pool-v0", type=Path, default=DEFAULT_POOL_V0)
    parser.add_argument("--pool-v1", type=Path, default=DEFAULT_POOL_V1)
    parser.add_argument("--txt-dir", type=Path, default=DEFAULT_TXT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--years", default="2022,2023,2024,2025,2026")
    parser.add_argument("--save-detail", action="store_true", help="Save per-trade detail CSV.")
    parser.add_argument("--limit", type=int, default=0, help="Debug only: limit rows after split. 0 = no limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = parse_years(args.years)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("Analyze B2 v1 incremental candidates versus v0")
    print("=" * 120)
    print(f"Pool v0    : {args.pool_v0}")
    print(f"Pool v1    : {args.pool_v1}")
    print(f"TXT dir    : {args.txt_dir}")
    print(f"Output dir : {args.output_dir}")
    print(f"Years      : {years}")
    print("-" * 120)

    pool_v0 = load_pool(args.pool_v0, "v0")
    pool_v1 = load_pool(args.pool_v1, "v1")

    split_pool = classify_pool_membership(pool_v0, pool_v1)

    if args.limit and args.limit > 0:
        split_pool = split_pool.head(args.limit).copy()

    all_symbols = sorted(set(split_pool["_symbol"].dropna().unique()))
    print(f"Unique symbols to load: {len(all_symbols):,}")

    market_cache, warnings = build_market_cache(args.txt_dir, all_symbols)
    print(f"Market files loaded: {len(market_cache):,}")
    print(f"Market warnings    : {len(warnings):,}")

    detail = analyze_pool(split_pool, market_cache)
    summary = summarize_detail(detail, years)

    delta_v1only_vs_both = build_group_delta(summary, base_group="both_v0_v1", compare_group="v1_only")
    delta_v1only_vs_v0only = build_group_delta(summary, base_group="v0_only", compare_group="v1_only")
    feature_profile = build_feature_profile(detail, years)

    summary_path = args.output_dir / "b2_v1_incremental_group_summary.csv"
    delta_both_path = args.output_dir / "b2_v1_only_minus_both_v0_v1.csv"
    delta_v0only_path = args.output_dir / "b2_v1_only_minus_v0_only.csv"
    feature_profile_path = args.output_dir / "b2_v1_incremental_feature_profile.csv"
    detail_path = args.output_dir / "b2_v1_incremental_trade_detail.csv"
    warnings_path = args.output_dir / "b2_v1_incremental_market_read_warnings.csv"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    delta_v1only_vs_both.to_csv(delta_both_path, index=False, encoding="utf-8-sig")
    delta_v1only_vs_v0only.to_csv(delta_v0only_path, index=False, encoding="utf-8-sig")
    feature_profile.to_csv(feature_profile_path, index=False, encoding="utf-8-sig")

    if warnings:
        pd.DataFrame(warnings).to_csv(warnings_path, index=False, encoding="utf-8-sig")

    if args.save_detail:
        detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print("-" * 120)
    print("Summary preview: ALL, T+2 close / T+3 close")
    print("-" * 120)
    preview = summary[
        (summary["period"] == "ALL")
        & (summary["sell_mode"].isin(["t2_close", "t3_close"]))
    ].copy()
    preview_cols = [
        "group",
        "period",
        "sell_mode",
        "trade_count",
        "win_rate_pct",
        "avg_ret_pct",
        "median_ret_pct",
        "avg_t1_open_gap_pct",
        "hit_plus_2pct_t1_t2_rate_pct",
        "hit_minus_5pct_t1_t3_rate_pct",
    ]
    preview_cols = [c for c in preview_cols if c in preview.columns]
    print(preview[preview_cols].to_string(index=False))

    print("-" * 120)
    print("v1_only - both_v0_v1 preview: T+2 close / T+3 close")
    print("-" * 120)
    delta_preview = delta_v1only_vs_both[
        delta_v1only_vs_both["sell_mode"].isin(["t2_close", "t3_close"])
    ].copy()
    delta_cols = [
        "period",
        "sell_mode",
        "trade_count_both_v0_v1",
        "trade_count_v1_only",
        "win_rate_pct_delta",
        "avg_ret_pct_delta",
        "median_ret_pct_delta",
        "hit_minus_5pct_t1_t3_rate_pct_delta",
    ]
    delta_cols = [c for c in delta_cols if c in delta_preview.columns]
    print(delta_preview[delta_cols].to_string(index=False))

    print("-" * 120)
    print("Output files")
    print("-" * 120)
    print(f"Group summary        : {summary_path}")
    print(f"v1_only - both       : {delta_both_path}")
    print(f"v1_only - v0_only    : {delta_v0only_path}")
    print(f"Feature profile      : {feature_profile_path}")
    if warnings:
        print(f"Warnings             : {warnings_path}")
    if args.save_detail:
        print(f"Trade detail         : {detail_path}")
    else:
        print("Trade detail         : not saved. Add --save-detail to save it.")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
