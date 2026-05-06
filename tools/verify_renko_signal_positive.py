from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# 路径写死在这里
# =============================================================================

FULL_POOL_PATH = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\pools\full_pool_indicator_strategy_v0_pool.parquet"
)

RENKO_SIGNAL_POOL_PATH = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\pools\renko_chart_select_strategy_v0_pool.parquet"
)

MARKET_CACHE_DIR = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\market_cache\daily_bars_by_symbol"
)

OUTPUT_DIR = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\output\renko_signal_validation"
)


# =============================================================================
# 参数
# =============================================================================

DATE_COL_CANDIDATES = ["date", "trade_date", "signal_date"]
SYMBOL_COL_CANDIDATES = ["symbol", "code", "stock_code", "ts_code"]

FORWARD_DAYS = [1, 2, 3]

HIT_THRESHOLDS = [0.0, 2.0, 5.0]


# =============================================================================
# 工具函数
# =============================================================================

def normalize_symbol(x) -> str:
    """
    统一股票代码格式。

    支持：
    - SH#600000
    - SZ#000001
    - 600000
    - 000001
    - SH600000
    - SZ000001
    """

    if pd.isna(x):
        return ""

    s = str(x).strip().upper()
    s = s.replace(".TXT", "")
    s = s.replace(".PARQUET", "")
    s = s.replace("\\", "/")
    s = s.split("/")[-1]

    if "#" in s:
        market, code = s.split("#", 1)
        code = code.strip()
        market = market.strip()
        if market in {"SH", "SZ"}:
            return f"{market}#{code}"

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


def find_col(df: pd.DataFrame, candidates: list[str], name: str) -> str:
    lower_map = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    raise ValueError(
        f"找不到{name}列。候选列={candidates}，当前文件列={list(df.columns)}"
    )


def load_pool(path: Path, pool_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{pool_name} 文件不存在: {path}")

    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError(f"{pool_name} 是空文件: {path}")

    date_col = find_col(df, DATE_COL_CANDIDATES, "日期")
    symbol_col = find_col(df, SYMBOL_COL_CANDIDATES, "股票代码")

    out = df.copy()
    out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out["symbol"] = out[symbol_col].map(normalize_symbol)

    out = out.dropna(subset=["date"])
    out = out[out["symbol"] != ""]

    out = out.drop_duplicates(subset=["date", "symbol"]).reset_index(drop=True)

    return out


def load_one_market_file(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None

    if df.empty:
        return None

    try:
        date_col = find_col(df, DATE_COL_CANDIDATES, "行情日期")
    except Exception:
        return None

    required_price_cols = ["open", "high", "low", "close"]
    lower_cols = {c.lower(): c for c in df.columns}

    missing = [c for c in required_price_cols if c not in lower_cols]
    if missing:
        return None

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    for col in required_price_cols:
        real_col = lower_cols[col]
        out[col] = pd.to_numeric(df[real_col], errors="coerce")

    if "volume" in lower_cols:
        out["volume"] = pd.to_numeric(df[lower_cols["volume"]], errors="coerce")
    elif "vol" in lower_cols:
        out["volume"] = pd.to_numeric(df[lower_cols["vol"]], errors="coerce")
    else:
        out["volume"] = np.nan

    symbol = None

    for c in SYMBOL_COL_CANDIDATES:
        if c.lower() in lower_cols:
            symbol = normalize_symbol(df[lower_cols[c.lower()]].iloc[0])
            break

    if not symbol:
        symbol = normalize_symbol(path.stem)

    out["symbol"] = symbol

    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out[out["symbol"] != ""]
    out = out.sort_values("date").drop_duplicates(subset=["symbol", "date"])

    return out


def load_market_cache(market_cache_dir: Path) -> pd.DataFrame:
    if not market_cache_dir.exists():
        raise FileNotFoundError(f"行情缓存目录不存在: {market_cache_dir}")

    files = sorted(market_cache_dir.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"行情缓存目录下没有 parquet 文件: {market_cache_dir}")

    frames: list[pd.DataFrame] = []

    print("=" * 90)
    print("Loading market cache")
    print("=" * 90)
    print(f"Market cache dir : {market_cache_dir}")
    print(f"Parquet files    : {len(files):,}")

    for i, path in enumerate(files, start=1):
        if i % 300 == 0 or i == 1 or i == len(files):
            print(f"  Loading [{i:,}/{len(files):,}] {path.name}")

        one = load_one_market_file(path)

        if one is not None and not one.empty:
            frames.append(one)

    if not frames:
        raise ValueError("没有成功读取任何行情 parquet 文件。")

    market = pd.concat(frames, ignore_index=True)
    market = market.sort_values(["symbol", "date"]).reset_index(drop=True)

    print(f"Loaded market rows: {len(market):,}")
    print(f"Loaded symbols    : {market['symbol'].nunique():,}")
    print(f"Date range        : {market['date'].min().date()} -> {market['date'].max().date()}")

    return market


def add_forward_returns(base: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """
    基于 T0 收盘价计算未来收益。

    T+1 open return = T+1 open / T0 close - 1
    T+1 close return = T+1 close / T0 close - 1
    T+2 close return = T+2 close / T0 close - 1
    T+3 close return = T+3 close / T0 close - 1

    注意：T+1/T+2/T+3 是交易日，不是自然日。
    """

    m = market.copy()
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)

    g = m.groupby("symbol", sort=False)

    for d in FORWARD_DAYS:
        m[f"t{d}_date"] = g["date"].shift(-d)
        m[f"t{d}_open"] = g["open"].shift(-d)
        m[f"t{d}_high"] = g["high"].shift(-d)
        m[f"t{d}_low"] = g["low"].shift(-d)
        m[f"t{d}_close"] = g["close"].shift(-d)

    keep_cols = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for d in FORWARD_DAYS:
        keep_cols += [
            f"t{d}_date",
            f"t{d}_open",
            f"t{d}_high",
            f"t{d}_low",
            f"t{d}_close",
        ]

    m = m[keep_cols].rename(
        columns={
            "open": "t0_open",
            "high": "t0_high",
            "low": "t0_low",
            "close": "t0_close",
            "volume": "t0_volume",
        }
    )

    out = base.merge(m, on=["symbol", "date"], how="left")

    for d in FORWARD_DAYS:
        out[f"t{d}_open_ret_pct"] = (out[f"t{d}_open"] / out["t0_close"] - 1.0) * 100.0
        out[f"t{d}_close_ret_pct"] = (out[f"t{d}_close"] / out["t0_close"] - 1.0) * 100.0
        out[f"t{d}_high_ret_pct"] = (out[f"t{d}_high"] / out["t0_close"] - 1.0) * 100.0
        out[f"t{d}_low_ret_pct"] = (out[f"t{d}_low"] / out["t0_close"] - 1.0) * 100.0

    high_cols = [f"t{d}_high_ret_pct" for d in FORWARD_DAYS]
    low_cols = [f"t{d}_low_ret_pct" for d in FORWARD_DAYS]

    out["t1_t3_max_high_ret_pct"] = out[high_cols].max(axis=1, skipna=True)
    out["t1_t3_min_low_ret_pct"] = out[low_cols].min(axis=1, skipna=True)

    out["has_forward_t1"] = out["t1_close"].notna()
    out["has_forward_t2"] = out["t2_close"].notna()
    out["has_forward_t3"] = out["t3_close"].notna()

    return out


def summarize_one_group(df: pd.DataFrame, group_name: str) -> dict:
    row: dict = {
        "group": group_name,
        "sample_count": len(df),
        "date_count": df["date"].nunique() if "date" in df.columns else np.nan,
        "symbol_count": df["symbol"].nunique() if "symbol" in df.columns else np.nan,
    }

    ret_cols = [
        "t1_open_ret_pct",
        "t1_close_ret_pct",
        "t2_close_ret_pct",
        "t3_close_ret_pct",
        "t1_t3_max_high_ret_pct",
        "t1_t3_min_low_ret_pct",
    ]

    for col in ret_cols:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce").dropna()

        row[f"{col}_valid_count"] = len(s)

        if len(s) == 0:
            row[f"{col}_mean"] = np.nan
            row[f"{col}_median"] = np.nan
            row[f"{col}_win_rate_gt_0"] = np.nan
            row[f"{col}_hit_rate_ge_2"] = np.nan
            row[f"{col}_hit_rate_ge_5"] = np.nan
            continue

        row[f"{col}_mean"] = s.mean()
        row[f"{col}_median"] = s.median()
        row[f"{col}_p25"] = s.quantile(0.25)
        row[f"{col}_p75"] = s.quantile(0.75)
        row[f"{col}_min"] = s.min()
        row[f"{col}_max"] = s.max()

        row[f"{col}_win_rate_gt_0"] = (s > 0).mean() * 100.0
        row[f"{col}_hit_rate_ge_2"] = (s >= 2.0).mean() * 100.0
        row[f"{col}_hit_rate_ge_5"] = (s >= 5.0).mean() * 100.0
        row[f"{col}_loss_rate_le_minus_2"] = (s <= -2.0).mean() * 100.0
        row[f"{col}_loss_rate_le_minus_5"] = (s <= -5.0).mean() * 100.0

    return row


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append(summarize_one_group(df, "ALL_BASE_POOL"))

    signal_df = df[df["is_renko_signal"] == True].copy()
    non_signal_df = df[df["is_renko_signal"] == False].copy()

    rows.append(summarize_one_group(signal_df, "RENKO_SIGNAL"))
    rows.append(summarize_one_group(non_signal_df, "NON_RENKO_SIGNAL"))

    summary = pd.DataFrame(rows)

    return summary


def build_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for date, one_day in df.groupby("date", sort=True):
        signal = one_day[one_day["is_renko_signal"] == True]
        non_signal = one_day[one_day["is_renko_signal"] == False]

        row = {
            "date": date,
            "all_count": len(one_day),
            "signal_count": len(signal),
            "non_signal_count": len(non_signal),
            "signal_ratio_pct": len(signal) / len(one_day) * 100.0 if len(one_day) else np.nan,
        }

        for col in [
            "t1_close_ret_pct",
            "t2_close_ret_pct",
            "t3_close_ret_pct",
            "t1_t3_max_high_ret_pct",
            "t1_t3_min_low_ret_pct",
        ]:
            s1 = pd.to_numeric(signal[col], errors="coerce").dropna()
            s0 = pd.to_numeric(non_signal[col], errors="coerce").dropna()

            row[f"signal_{col}_mean"] = s1.mean() if len(s1) else np.nan
            row[f"non_signal_{col}_mean"] = s0.mean() if len(s0) else np.nan
            row[f"diff_signal_minus_non_signal_{col}_mean"] = (
                row[f"signal_{col}_mean"] - row[f"non_signal_{col}_mean"]
                if pd.notna(row[f"signal_{col}_mean"]) and pd.notna(row[f"non_signal_{col}_mean"])
                else np.nan
            )

            row[f"signal_{col}_win_rate"] = (s1 > 0).mean() * 100.0 if len(s1) else np.nan
            row[f"non_signal_{col}_win_rate"] = (s0 > 0).mean() * 100.0 if len(s0) else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def print_key_summary(summary: pd.DataFrame) -> None:
    show_cols = [
        "group",
        "sample_count",
        "date_count",
        "symbol_count",
        "t1_close_ret_pct_mean",
        "t1_close_ret_pct_median",
        "t1_close_ret_pct_win_rate_gt_0",
        "t1_close_ret_pct_hit_rate_ge_2",
        "t2_close_ret_pct_mean",
        "t2_close_ret_pct_median",
        "t2_close_ret_pct_win_rate_gt_0",
        "t2_close_ret_pct_hit_rate_ge_2",
        "t3_close_ret_pct_mean",
        "t3_close_ret_pct_median",
        "t3_close_ret_pct_win_rate_gt_0",
        "t3_close_ret_pct_hit_rate_ge_2",
        "t1_t3_max_high_ret_pct_mean",
        "t1_t3_max_high_ret_pct_hit_rate_ge_2",
        "t1_t3_min_low_ret_pct_mean",
    ]

    existing = [c for c in show_cols if c in summary.columns]

    print()
    print("=" * 90)
    print("KEY SUMMARY")
    print("=" * 90)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.float_format",
        "{:.4f}".format,
    ):
        print(summary[existing].to_string(index=False))


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("Renko signal validation")
    print("=" * 90)
    print(f"Full base pool path      : {FULL_POOL_PATH}")
    print(f"Renko signal pool path   : {RENKO_SIGNAL_POOL_PATH}")
    print(f"Market cache dir         : {MARKET_CACHE_DIR}")
    print(f"Output dir               : {OUTPUT_DIR}")

    full_pool = load_pool(FULL_POOL_PATH, "全部票基准池")
    signal_pool = load_pool(RENKO_SIGNAL_POOL_PATH, "砖型图转强池")

    print()
    print("=" * 90)
    print("Loaded pools")
    print("=" * 90)
    print(f"Full pool rows           : {len(full_pool):,}")
    print(f"Full pool symbols        : {full_pool['symbol'].nunique():,}")
    print(f"Full pool date range     : {full_pool['date'].min().date()} -> {full_pool['date'].max().date()}")

    print(f"Renko signal rows        : {len(signal_pool):,}")
    print(f"Renko signal symbols     : {signal_pool['symbol'].nunique():,}")
    print(f"Renko signal date range  : {signal_pool['date'].min().date()} -> {signal_pool['date'].max().date()}")

    signal_keys = signal_pool[["date", "symbol"]].drop_duplicates().copy()
    signal_keys["is_renko_signal"] = True

    base = full_pool[["date", "symbol"]].drop_duplicates().copy()
    base = base.merge(signal_keys, on=["date", "symbol"], how="left")
    base["is_renko_signal"] = base["is_renko_signal"].fillna(False).astype(bool)

    matched_signal_count = int(base["is_renko_signal"].sum())
    missing_signal = len(signal_keys) - matched_signal_count

    print()
    print("=" * 90)
    print("Signal matching")
    print("=" * 90)
    print(f"Base rows                : {len(base):,}")
    print(f"Matched signal rows      : {matched_signal_count:,}")
    print(f"Signal rows not in base  : {missing_signal:,}")

    if matched_signal_count == 0:
        raise ValueError(
            "砖型图转强池在全市场基准池中一条都没有匹配到。"
            "请检查两个池子的 date/symbol 格式是否一致。"
        )

    market = load_market_cache(MARKET_CACHE_DIR)

    result = add_forward_returns(base, market)

    missing_t0 = result["t0_close"].isna().sum()
    print()
    print("=" * 90)
    print("Forward return merge check")
    print("=" * 90)
    print(f"Rows without T0 market data : {missing_t0:,}")
    print(f"Rows with T+1 close         : {result['has_forward_t1'].sum():,}")
    print(f"Rows with T+2 close         : {result['has_forward_t2'].sum():,}")
    print(f"Rows with T+3 close         : {result['has_forward_t3'].sum():,}")

    summary = build_summary(result)
    daily_summary = build_daily_summary(result)

    detail_path = OUTPUT_DIR / "renko_signal_validation_detail.csv"
    summary_path = OUTPUT_DIR / "renko_signal_validation_summary.csv"
    daily_path = OUTPUT_DIR / "renko_signal_validation_daily_summary.csv"

    result.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    daily_summary.to_csv(daily_path, index=False, encoding="utf-8-sig")

    print_key_summary(summary)

    print()
    print("=" * 90)
    print("Saved files")
    print("=" * 90)
    print(f"Detail CSV        : {detail_path}")
    print(f"Summary CSV       : {summary_path}")
    print(f"Daily summary CSV : {daily_path}")

    print()
    print("=" * 90)
    print("How to judge")
    print("=" * 90)
    print("如果 RENKO_SIGNAL 的 T+1/T+2/T+3 平均收益、胜率、涨超2%概率")
    print("持续高于 NON_RENKO_SIGNAL 或 ALL_BASE_POOL，说明砖型图转强是正向指标。")
    print("如果只提升样本涨超2%概率，但平均收益不提升，说明它可能更适合作为弹性因子。")
    print("如果平均收益、胜率都不如非信号池，说明该信号单独使用不是正向指标。")


if __name__ == "__main__":
    main()