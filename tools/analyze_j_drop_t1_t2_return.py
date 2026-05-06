from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 固定路径
# =============================================================================

POOL_PATH = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\pools\full_pool_indicator_strategy_v0_pool.parquet"
)

INDICATOR_CACHE_PATH = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\indicator_cache\daily_indicators.parquet"
)

MARKET_CACHE_DIR = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\market_cache\daily_bars_by_symbol"
)

OUTPUT_DIR = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\output\j_value_2025_t1_t2_analysis"
)


# =============================================================================
# 只验证 2025 全年
# =============================================================================

START_DATE = "2025-01-01"
END_DATE = "2025-12-31"


# =============================================================================
# J 值区间
# =============================================================================

J_VALUE_BINS = [-np.inf, 0, 20, 40, 60, 80, 100, np.inf]
J_VALUE_LABELS = [
    "J<0",
    "0<=J<20",
    "20<=J<40",
    "40<=J<60",
    "60<=J<80",
    "80<=J<100",
    "J>=100",
]


DATE_CANDIDATES = [
    "date", "trade_date", "signal_date", "datetime", "time",
    "日期", "交易日期", "时间",
]

SYMBOL_CANDIDATES = [
    "symbol", "code", "stock_code", "ts_code", "证券代码", "股票代码",
]

OPEN_CANDIDATES = ["open", "开盘", "开盘价"]
HIGH_CANDIDATES = ["high", "最高", "最高价"]
LOW_CANDIDATES = ["low", "最低", "最低价"]
CLOSE_CANDIDATES = ["close", "收盘", "收盘价"]

J_CANDIDATES = [
    "j",
    "J",
    "kdj_j",
    "KDJ_J",
    "kdj_j_value",
    "j_value",
    "J值",
    "kdj_j_tdx",
    "KDJJ",
]


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
    date_col = find_col(df, DATE_CANDIDATES)

    if date_col is not None:
        return pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    idx = df.index

    if isinstance(idx, pd.DatetimeIndex):
        return pd.Series(idx, index=df.index).dt.normalize()

    idx_dt = pd.to_datetime(idx, errors="coerce")

    if pd.Series(idx_dt).notna().mean() > 0.8:
        return pd.Series(idx_dt, index=df.index).dt.normalize()

    raise ValueError(
        f"{source_name} 找不到日期列，也无法从 index 解析日期。当前列名: {list(df.columns)}"
    )


def get_symbol_series(df: pd.DataFrame, fallback_symbol: str = "") -> pd.Series:
    symbol_col = find_col(df, SYMBOL_CANDIDATES)

    if symbol_col is not None:
        return df[symbol_col].map(normalize_symbol)

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
            f"{source_name} 找不到 {col_name} 列。候选={candidates}，当前列名={list(df.columns)}"
        )

    return pd.to_numeric(df[col], errors="coerce")


def load_pool(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"pool 文件不存在: {path}")

    df = pd.read_parquet(path)

    out = df.copy()
    out["date"] = get_date_series(df, "pool")
    out["symbol"] = get_symbol_series(df)

    j_col = find_col(out, J_CANDIDATES)

    if j_col is not None:
        out["j"] = pd.to_numeric(out[j_col], errors="coerce")
    else:
        out["j"] = np.nan

    out = out.dropna(subset=["date"])
    out = out[out["symbol"] != ""]
    out = out.drop_duplicates(subset=["symbol", "date"])
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

    start = pd.to_datetime(START_DATE)
    end = pd.to_datetime(END_DATE)

    out = out[(out["date"] >= start) & (out["date"] <= end)].copy()

    return out


def load_indicator_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"indicator cache 不存在: {path}")

    df = pd.read_parquet(path)

    out = pd.DataFrame()
    out["date"] = get_date_series(df, "indicator cache")
    out["symbol"] = get_symbol_series(df)

    j_col = find_col(df, J_CANDIDATES)

    if j_col is None:
        raise ValueError(
            f"indicator cache 中找不到 J 列。候选={J_CANDIDATES}，当前列={list(df.columns)}"
        )

    out["j_from_indicator_cache"] = pd.to_numeric(df[j_col], errors="coerce")

    out = out.dropna(subset=["date"])
    out = out[out["symbol"] != ""]
    out = out.drop_duplicates(subset=["symbol", "date"])
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

    start = pd.to_datetime(START_DATE)
    end = pd.to_datetime(END_DATE)

    out = out[(out["date"] >= start) & (out["date"] <= end)].copy()

    return out


def ensure_j(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()

    print("=" * 100)
    print("J value check")
    print("=" * 100)
    print(f"Pool J valid rows before merge: {out['j'].notna().sum():,}")

    if out["j"].notna().sum() > 0:
        return out

    indicators = load_indicator_cache(INDICATOR_CACHE_PATH)

    print("Pool 内没有有效 J，开始从 indicator_cache 合并 J...")
    print(f"Indicator rows    : {len(indicators):,}")
    print(f"Indicator J valid : {indicators['j_from_indicator_cache'].notna().sum():,}")

    out = out.drop(columns=["j"], errors="ignore")
    out = out.merge(indicators, on=["symbol", "date"], how="left")
    out = out.rename(columns={"j_from_indicator_cache": "j"})

    print(f"Pool J valid rows after merge: {out['j'].notna().sum():,}")

    if out["j"].notna().sum() == 0:
        raise ValueError("合并 indicator_cache 后 J 仍然全为空，请检查 symbol/date 是否对齐。")

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

        return out

    except Exception:
        return None


def load_market_cache(market_cache_dir: Path) -> pd.DataFrame:
    if not market_cache_dir.exists():
        raise FileNotFoundError(f"行情缓存目录不存在: {market_cache_dir}")

    files = sorted(market_cache_dir.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"行情缓存目录下没有 parquet 文件: {market_cache_dir}")

    frames = []
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
        raise ValueError("没有成功读取任何行情 parquet 文件。")

    market = pd.concat(frames, ignore_index=True)
    market = market.sort_values(["symbol", "date"]).reset_index(drop=True)

    print(f"Loaded market rows : {len(market):,}")
    print(f"Loaded symbols     : {market['symbol'].nunique():,}")
    print(f"Failed files       : {failed:,}")
    print(f"Date range         : {market['date'].min().date()} -> {market['date'].max().date()}")

    return market


def add_t1_t2_returns(pool: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    m = market.copy()
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)

    g = m.groupby("symbol", sort=False)

    m["t1_date"] = g["date"].shift(-1)
    m["t1_open"] = g["open"].shift(-1)
    m["t1_close"] = g["close"].shift(-1)

    m["t2_date"] = g["date"].shift(-2)
    m["t2_open"] = g["open"].shift(-2)
    m["t2_high"] = g["high"].shift(-2)
    m["t2_low"] = g["low"].shift(-2)
    m["t2_close"] = g["close"].shift(-2)

    m = m[
        [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "t1_date",
            "t1_open",
            "t1_close",
            "t2_date",
            "t2_open",
            "t2_high",
            "t2_low",
            "t2_close",
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

    out["t1_open_gap_pct"] = (out["t1_open"] / out["t0_close"] - 1.0) * 100.0

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

    return out


def summarize_by_j_value(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["j_value_bucket"] = pd.cut(
        out["j"],
        bins=J_VALUE_BINS,
        labels=J_VALUE_LABELS,
        right=False,
    )

    rows = []

    # ALL 总样本
    valid_all = out.dropna(subset=["t1_open_to_t2_close_ret_pct"])

    rows.append(build_summary_row(valid_all, "ALL", "ALL"))

    for bucket, one in out.groupby("j_value_bucket", observed=False, dropna=False):
        valid = one.dropna(subset=["t1_open_to_t2_close_ret_pct"])
        rows.append(build_summary_row(valid, "J_VALUE_BUCKET", str(bucket)))

    result = pd.DataFrame(rows)

    order = {"ALL": -1}
    order.update({label: i for i, label in enumerate(J_VALUE_LABELS)})
    result["_order"] = result["bucket"].map(order)

    result = result.sort_values("_order").drop(columns=["_order"])

    return result


def build_summary_row(valid: pd.DataFrame, analysis_name: str, bucket: str) -> dict:
    row = {
        "analysis_name": analysis_name,
        "bucket": bucket,
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

    row.update(
        {
            "avg_j": valid["j"].mean(),
            "median_j": valid["j"].median(),
            "t1_t2_mean_ret_pct": ret.mean(),
            "t1_t2_median_ret_pct": ret.median(),
            "t1_t2_p25_ret_pct": ret.quantile(0.25),
            "t1_t2_p75_ret_pct": ret.quantile(0.75),
            "t1_t2_win_rate_pct": (ret > 0).mean() * 100.0,
            "t1_t2_hit_1pct_rate": (ret >= 1.0).mean() * 100.0,
            "t1_t2_hit_2pct_rate": (ret >= 2.0).mean() * 100.0,
            "t1_t2_hit_3pct_rate": (ret >= 3.0).mean() * 100.0,
            "t1_t2_loss_1pct_rate": (ret <= -1.0).mean() * 100.0,
            "t1_t2_loss_2pct_rate": (ret <= -2.0).mean() * 100.0,
            "t1_t2_loss_3pct_rate": (ret <= -3.0).mean() * 100.0,
            "t2_high_mean_ret_pct": high_ret.mean(),
            "t2_high_hit_2pct_rate": (high_ret >= 2.0).mean() * 100.0,
            "t2_low_mean_ret_pct": low_ret.mean(),
            "t2_low_loss_2pct_rate": (low_ret <= -2.0).mean() * 100.0,
            "t1_open_gap_mean_pct": valid["t1_open_gap_pct"].mean(),
            "t1_intraday_mean_ret_pct": t1_ret.mean(),
            "t1_intraday_win_rate_pct": (t1_ret > 0).mean() * 100.0,
        }
    )

    return row


def add_vs_all_columns(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()

    all_row = out[out["bucket"] == "ALL"]

    if all_row.empty:
        return out

    all_row = all_row.iloc[0]

    compare_cols = [
        "t1_t2_mean_ret_pct",
        "t1_t2_median_ret_pct",
        "t1_t2_win_rate_pct",
        "t1_t2_hit_2pct_rate",
        "t1_t2_loss_2pct_rate",
        "t2_high_hit_2pct_rate",
        "t2_low_loss_2pct_rate",
    ]

    for col in compare_cols:
        if col in out.columns:
            out[f"diff_vs_all_{col}"] = out[col] - all_row[col]

    return out


def print_summary(summary: pd.DataFrame) -> None:
    show_cols = [
        "analysis_name",
        "bucket",
        "valid_count",
        "avg_j",
        "t1_t2_mean_ret_pct",
        "t1_t2_median_ret_pct",
        "t1_t2_win_rate_pct",
        "t1_t2_hit_2pct_rate",
        "t1_t2_loss_2pct_rate",
        "t2_high_hit_2pct_rate",
        "t2_low_loss_2pct_rate",
    ]

    existing = [c for c in show_cols if c in summary.columns]

    print()
    print("=" * 100)
    print("2025 J VALUE BUCKET SUMMARY")
    print("=" * 100)

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 240,
        "display.float_format", "{:.4f}".format,
    ):
        print(summary[existing].to_string(index=False))


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Analyze 2025 J value bucket for T1 open buy -> T2 close sell")
    print("=" * 100)
    print(f"Pool path       : {POOL_PATH}")
    print(f"Indicator cache : {INDICATOR_CACHE_PATH}")
    print(f"Market cache    : {MARKET_CACHE_DIR}")
    print(f"Date range      : {START_DATE} -> {END_DATE}")
    print(f"Output dir      : {OUTPUT_DIR}")

    pool = load_pool(POOL_PATH)

    print()
    print("=" * 100)
    print("Loaded pool")
    print("=" * 100)
    print(f"Rows       : {len(pool):,}")
    print(f"Symbols    : {pool['symbol'].nunique():,}")
    print(f"Date range : {pool['date'].min().date()} -> {pool['date'].max().date()}")
    print(f"J valid before merge : {pool['j'].notna().sum():,}")

    pool = ensure_j(pool)

    print()
    print("=" * 100)
    print("Loaded J")
    print("=" * 100)
    print(f"J valid rows : {pool['j'].notna().sum():,}")
    print(f"J min/max    : {pool['j'].min():.4f} / {pool['j'].max():.4f}")

    market = load_market_cache(MARKET_CACHE_DIR)

    result = add_t1_t2_returns(pool, market)

    print()
    print("=" * 100)
    print("Merge check")
    print("=" * 100)
    print(f"Rows without T0 close : {result['t0_close'].isna().sum():,}")
    print(f"Rows with T1/T2 data  : {result['has_t1_t2'].sum():,}")
    print(f"Rows missing T1/T2    : {(~result['has_t1_t2']).sum():,}")

    summary = summarize_by_j_value(result)
    summary = add_vs_all_columns(summary)

    detail_path = OUTPUT_DIR / "j_value_2025_t1_t2_detail.csv"
    summary_path = OUTPUT_DIR / "j_value_2025_t1_t2_summary.csv"

    result.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print_summary(summary)

    print()
    print("=" * 100)
    print("Saved files")
    print("=" * 100)
    print(f"Detail CSV  : {detail_path}")
    print(f"Summary CSV : {summary_path}")

    print()
    print("=" * 100)
    print("判断标准")
    print("=" * 100)
    print("如果 J<0 或 0<=J<20 的平均收益、胜率、涨超2%概率明显高于 ALL，")
    print("并且跌超2%概率低于 ALL，说明低 J 区间是正向指标。")
    print("如果 J 越高收益越差，则说明 J 低位更适合你的 T1-T2 模式。")


if __name__ == "__main__":
    main()