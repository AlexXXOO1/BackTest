from __future__ import annotations

r"""
Pool quality check for B2 confirmation strategy.

Purpose:
- Read a generated pool parquet.
- Read TongDaXin TXT market files.
- Evaluate:
    1. T+1 open buy -> T+2 open sell
    2. T+1 open buy -> T+2 close sell
    3. T+1/T+2 max opportunity
    4. T+1/T+2 min drawdown risk

This version is designed for messy TongDaXin TXT exports:
- Multiple encoding fallbacks.
- Line-by-line parser instead of strict pandas.read_csv.
- Ignores header/footer/comment/broken rows.
- Supports mixed delimiters: tab, comma, spaces.
- Does not stop the whole analysis because of one malformed TXT file.
"""

import argparse
import csv
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


# =============================================================================
# Default paths
# =============================================================================

DEFAULT_POOL_PATH = Path(r"C:\Users\zyf37\Desktop\BackTest Data\pools\b2_confirm_select_strategy_v0_pool.parquet")
DEFAULT_TXT_DIR = Path(r"C:\Users\zyf37\Desktop\BackTest Data\data")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\zyf37\Desktop\BackTest Data\output\pool_quality_b2")


# =============================================================================
# TXT parsing
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


def _read_text_with_fallback(path: Path) -> str:
    last_error: Exception | None = None

    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except Exception as exc:
            last_error = exc

    # Final fallback: never fail because of broken bytes.
    try:
        return path.read_text(encoding="gb18030", errors="replace")
    except Exception as exc:
        if last_error is not None:
            raise RuntimeError(f"{last_error}; fallback error: {exc}") from exc
        raise


def _split_market_line(line: str) -> list[str]:
    """
    Robust splitter for TongDaXin TXT rows.

    Handles:
    - tabs
    - spaces
    - comma
    - Chinese comma
    - semicolon
    """
    line = line.strip().replace("\ufeff", "")
    line = line.replace(",", " ")
    line = line.replace("，", " ")
    line = line.replace(";", " ")
    line = re.sub(r"\s+", " ", line)
    return [x for x in line.split(" ") if x]


def _parse_date_token(token: str) -> pd.Timestamp | None:
    token = token.strip()

    # Common formats:
    # 2025-01-02
    # 2025/01/02
    # 20250102
    # 02/01/2025
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

    # Need at least OHLCV.
    if len(nums) < 5:
        return None

    # Most TDX TXT rows: date open high low close volume amount...
    open_, high, low, close, volume = nums[:5]
    amount = nums[5] if len(nums) >= 6 else None

    # Basic sanity checks. Skip invalid lines.
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


def infer_symbol_columns(df: pd.DataFrame) -> tuple[str, str]:
    date_col = ""
    symbol_col = ""

    lower_map = {str(c).lower(): c for c in df.columns}

    for cand in ["date", "signal_date", "trade_date", "datetime"]:
        if cand in lower_map:
            date_col = lower_map[cand]
            break

    for cand in ["symbol", "code", "ts_code", "stock_code", "security_code"]:
        if cand in lower_map:
            symbol_col = lower_map[cand]
            break

    if not date_col:
        for c in df.columns:
            if "date" in str(c).lower():
                date_col = c
                break

    if not symbol_col:
        for c in df.columns:
            name = str(c).lower()
            if "symbol" in name or "code" in name:
                symbol_col = c
                break

    if not date_col or not symbol_col:
        raise ValueError(
            f"Cannot infer date/symbol columns from pool columns: {list(df.columns)}"
        )

    return str(date_col), str(symbol_col)


def get_next_trading_rows(market: pd.DataFrame, signal_date: pd.Timestamp) -> tuple[pd.Series | None, pd.Series | None]:
    """
    Return T+1 row and T+2 row after signal_date.
    """
    market = market.sort_values("date").reset_index(drop=True)
    future = market[market["date"] > signal_date].head(2)
    if len(future) < 2:
        return None, None
    return future.iloc[0], future.iloc[1]


def pct(a: float, b: float) -> float:
    if a == 0:
        return 0.0
    return (b / a - 1.0) * 100.0


def load_pool(pool_path: Path, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if not pool_path.exists():
        raise FileNotFoundError(f"Pool file not found: {pool_path}")

    df = pd.read_parquet(pool_path)
    date_col, symbol_col = infer_symbol_columns(df)

    df = df.copy()
    df["_signal_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df["_symbol"] = df[symbol_col].map(normalize_symbol)
    df = df.dropna(subset=["_signal_date"])
    df = df[df["_symbol"].astype(str).str.len() >= 8]

    if start_date:
        df = df[df["_signal_date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["_signal_date"] <= pd.to_datetime(end_date)]

    return df.reset_index(drop=True)


def iter_progress(items: Iterable, total: int | None = None, desc: str = ""):
    if tqdm is not None:
        return tqdm(items, total=total, desc=desc)
    return items


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


def analyze_pool(pool: pd.DataFrame, market_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []

    for _, r in iter_progress(pool.iterrows(), total=len(pool), desc="Analyzing pool"):
        symbol = r["_symbol"]
        signal_date = pd.Timestamp(r["_signal_date"]).normalize()

        market = market_cache.get(symbol)
        if market is None or market.empty:
            continue

        t1, t2 = get_next_trading_rows(market, signal_date)
        if t1 is None or t2 is None:
            continue

        buy_price = float(t1["open"])
        t1_close = float(t1["close"])
        t2_open = float(t2["open"])
        t2_close = float(t2["close"])

        t1_high = float(t1["high"])
        t1_low = float(t1["low"])
        t2_high = float(t2["high"])
        t2_low = float(t2["low"])

        max_high_t1_t2 = max(t1_high, t2_high)
        min_low_t1_t2 = min(t1_low, t2_low)

        out = {
            "symbol": symbol,
            "signal_date": signal_date.date().isoformat(),
            "t1_date": pd.Timestamp(t1["date"]).date().isoformat(),
            "t2_date": pd.Timestamp(t2["date"]).date().isoformat(),
            "buy_t1_open": buy_price,
            "t1_close": t1_close,
            "sell_t2_open": t2_open,
            "sell_t2_close": t2_close,
            "t1_high": t1_high,
            "t1_low": t1_low,
            "t2_high": t2_high,
            "t2_low": t2_low,
            "ret_t1_open_to_t2_open_pct": pct(buy_price, t2_open),
            "ret_t1_open_to_t2_close_pct": pct(buy_price, t2_close),
            "max_opportunity_t1_t2_pct": pct(buy_price, max_high_t1_t2),
            "max_drawdown_t1_t2_pct": pct(buy_price, min_low_t1_t2),
            "win_t2_open": pct(buy_price, t2_open) > 0,
            "win_t2_close": pct(buy_price, t2_close) > 0,
            "hit_plus_2pct_t1_t2": pct(buy_price, max_high_t1_t2) >= 2,
            "hit_plus_3pct_t1_t2": pct(buy_price, max_high_t1_t2) >= 3,
            "hit_plus_5pct_t1_t2": pct(buy_price, max_high_t1_t2) >= 5,
            "hit_minus_2pct_t1_t2": pct(buy_price, min_low_t1_t2) <= -2,
            "hit_minus_3pct_t1_t2": pct(buy_price, min_low_t1_t2) <= -3,
            "hit_minus_5pct_t1_t2": pct(buy_price, min_low_t1_t2) <= -5,
        }

        # Keep useful original pool columns if present.
        for c in [
            "score",
            "score_pct",
            "daily_return_pct",
            "j",
            "J",
            "kdj_j",
            "b1_days_ago",
            "b2_volume_ratio",
            "upper_shadow_ratio",
            "b1_j_value",
            "b2_j_value",
        ]:
            if c in pool.columns and c in r.index:
                out[c] = r[c]

        rows.append(out)

    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame([{"metric": "trade_count", "value": 0}])

    def safe_mean(col: str) -> float:
        return float(pd.to_numeric(detail[col], errors="coerce").mean())

    def safe_median(col: str) -> float:
        return float(pd.to_numeric(detail[col], errors="coerce").median())

    def safe_min(col: str) -> float:
        return float(pd.to_numeric(detail[col], errors="coerce").min())

    def safe_max(col: str) -> float:
        return float(pd.to_numeric(detail[col], errors="coerce").max())

    rows = [
        {"metric": "trade_count", "value": len(detail)},

        {"metric": "t2_open_win_rate_pct", "value": safe_mean("win_t2_open") * 100},
        {"metric": "t2_open_avg_ret_pct", "value": safe_mean("ret_t1_open_to_t2_open_pct")},
        {"metric": "t2_open_median_ret_pct", "value": safe_median("ret_t1_open_to_t2_open_pct")},
        {"metric": "t2_open_best_ret_pct", "value": safe_max("ret_t1_open_to_t2_open_pct")},
        {"metric": "t2_open_worst_ret_pct", "value": safe_min("ret_t1_open_to_t2_open_pct")},

        {"metric": "t2_close_win_rate_pct", "value": safe_mean("win_t2_close") * 100},
        {"metric": "t2_close_avg_ret_pct", "value": safe_mean("ret_t1_open_to_t2_close_pct")},
        {"metric": "t2_close_median_ret_pct", "value": safe_median("ret_t1_open_to_t2_close_pct")},
        {"metric": "t2_close_best_ret_pct", "value": safe_max("ret_t1_open_to_t2_close_pct")},
        {"metric": "t2_close_worst_ret_pct", "value": safe_min("ret_t1_open_to_t2_close_pct")},

        {"metric": "hit_plus_2pct_rate_pct", "value": safe_mean("hit_plus_2pct_t1_t2") * 100},
        {"metric": "hit_plus_3pct_rate_pct", "value": safe_mean("hit_plus_3pct_t1_t2") * 100},
        {"metric": "hit_plus_5pct_rate_pct", "value": safe_mean("hit_plus_5pct_t1_t2") * 100},

        {"metric": "hit_minus_2pct_rate_pct", "value": safe_mean("hit_minus_2pct_t1_t2") * 100},
        {"metric": "hit_minus_3pct_rate_pct", "value": safe_mean("hit_minus_3pct_t1_t2") * 100},
        {"metric": "hit_minus_5pct_rate_pct", "value": safe_mean("hit_minus_5pct_t1_t2") * 100},

        {"metric": "avg_max_opportunity_t1_t2_pct", "value": safe_mean("max_opportunity_t1_t2_pct")},
        {"metric": "avg_max_drawdown_t1_t2_pct", "value": safe_mean("max_drawdown_t1_t2_pct")},
    ]
    return pd.DataFrame(rows)


def summarize_by_bucket(detail: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    if detail.empty:
        return result

    bucket_cols = [
        "b1_days_ago",
        "upper_shadow_ratio",
        "b2_volume_ratio",
        "b1_j_value",
        "b2_j_value",
        "j",
        "J",
        "kdj_j",
    ]

    for col in bucket_cols:
        if col not in detail.columns:
            continue

        tmp = detail.copy()
        s = pd.to_numeric(tmp[col], errors="coerce")

        if col == "upper_shadow_ratio":
            tmp["_bucket"] = pd.cut(
                s,
                bins=[-999, 0.10, 0.20, 0.25, 999],
                labels=["<=0.10", "0.10-0.20", "0.20-0.25", ">0.25"],
            )
        elif col == "b2_volume_ratio":
            tmp["_bucket"] = pd.cut(
                s,
                bins=[-999, 1.0, 1.5, 1.9, 3.5, 999],
                labels=["<1.0", "1.0-1.5", "1.5-1.9", "1.9-3.5", ">=3.5"],
            )
        elif col in {"b1_j_value", "b2_j_value", "j", "J", "kdj_j"}:
            tmp["_bucket"] = pd.cut(
                s,
                bins=[-999, -15, -10, -5, 0, 14, 30, 45, 55, 999],
                labels=["<-15", "-15--10", "-10--5", "-5-0", "0-14", "14-30", "30-45", "45-55", ">=55"],
            )
        else:
            tmp["_bucket"] = tmp[col].astype(str)

        g = (
            tmp.groupby("_bucket", dropna=False, observed=False)
            .agg(
                trade_count=("symbol", "count"),
                t2_open_win_rate_pct=("win_t2_open", lambda x: float(pd.Series(x).mean()) * 100),
                t2_open_avg_ret_pct=("ret_t1_open_to_t2_open_pct", "mean"),
                t2_open_median_ret_pct=("ret_t1_open_to_t2_open_pct", "median"),
                t2_close_win_rate_pct=("win_t2_close", lambda x: float(pd.Series(x).mean()) * 100),
                t2_close_avg_ret_pct=("ret_t1_open_to_t2_close_pct", "mean"),
                t2_close_median_ret_pct=("ret_t1_open_to_t2_close_pct", "median"),
                hit_plus_2pct_rate_pct=("hit_plus_2pct_t1_t2", lambda x: float(pd.Series(x).mean()) * 100),
                hit_minus_2pct_rate_pct=("hit_minus_2pct_t1_t2", lambda x: float(pd.Series(x).mean()) * 100),
                avg_max_opportunity_t1_t2_pct=("max_opportunity_t1_t2_pct", "mean"),
                avg_max_drawdown_t1_t2_pct=("max_drawdown_t1_t2_pct", "mean"),
            )
            .reset_index()
            .rename(columns={"_bucket": "bucket"})
        )
        result[col] = g

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pool quality check for B2 confirmation pool.")
    parser.add_argument("--pool-path", type=Path, default=DEFAULT_POOL_PATH)
    parser.add_argument("--txt-dir", type=Path, default=DEFAULT_TXT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Debug only: limit pool rows. 0 means no limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Pool Quality Check - B2 Confirmation")
    print("=" * 88)
    print(f"Pool path : {args.pool_path}")
    print(f"TXT dir   : {args.txt_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Date range: {args.start_date or 'ALL'} -> {args.end_date or 'ALL'}")
    print("-" * 88)

    pool = load_pool(args.pool_path, args.start_date, args.end_date)
    if args.limit and args.limit > 0:
        pool = pool.head(args.limit).copy()

    print(f"Pool rows loaded: {len(pool):,}")
    if pool.empty:
        print("[WARN] Empty pool after date filter.")
        return

    symbols = sorted(pool["_symbol"].dropna().unique().tolist())
    print(f"Unique symbols: {len(symbols):,}")

    market_cache, warnings = build_market_cache(args.txt_dir, symbols)
    print(f"Market files loaded: {len(market_cache):,}")
    print(f"Market warnings    : {len(warnings):,}")

    detail = analyze_pool(pool, market_cache)
    summary = summarize(detail)
    buckets = summarize_by_bucket(detail)

    stem = args.pool_path.stem.replace("_pool", "")
    detail_path = args.output_dir / f"{stem}_quality_detail.csv"
    summary_path = args.output_dir / f"{stem}_quality_summary.csv"
    warning_path = args.output_dir / f"{stem}_market_read_warnings.csv"

    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if warnings:
        pd.DataFrame(warnings).to_csv(warning_path, index=False, encoding="utf-8-sig")

    for col, df in buckets.items():
        out_path = args.output_dir / f"{stem}_bucket_by_{col}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("-" * 88)
    print("Summary")
    print("-" * 88)
    if not summary.empty:
        for _, r in summary.iterrows():
            print(f"{str(r['metric']).ljust(36)} : {r['value']}")

    print("-" * 88)
    print("Output files")
    print("-" * 88)
    print(f"Detail : {detail_path}")
    print(f"Summary: {summary_path}")
    if warnings:
        print(f"Warnings: {warning_path}")
    for col in buckets:
        print(f"Bucket : {args.output_dir / f'{stem}_bucket_by_{col}.csv'}")

    print("=" * 88)
    print("Done")
    print("=" * 88)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
