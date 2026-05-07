from __future__ import annotations

r"""
Compare quality of N selection pool parquet files.

Purpose
-------
Given any number of pool parquet files, calculate the same forward-return
metrics for every pool and produce one summary table plus one detail table.

Typical use
-----------
python .\analyze_tools\compare_n_pools_quality.py 
  --pool-paths "C:\Users\zyf37\Desktop\BackTest Data\pools\renko_chart_select_strategy_v4_pool.parquet" 
               "C:\Users\zyf37\Desktop\BackTest Data\pools\renko_chart_select_strategy_v5_pool.parquet" 
  --market-cache-dir "C:\Users\zyf37\Desktop\BackTest Data\market_cache\daily_bars_by_symbol" 
  --output-dir "C:\Users\zyf37\Desktop\BackTest Data\output\pool_quality_compare" 
  --start-date 2024-01-01 
  --end-date 2026-04-30

Notes
-----
- Pool date is treated as T0 signal date.
- T1/T2/T3 are the next trading rows in the local market cache for that symbol.
- Returns are calculated from both T0 close and T1 open so you can judge
  T0-tail-buy vs T+1-open-buy behavior.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import MARKET_CACHE_DIR, OUTPUT_DIR

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_MARKET_CACHE_DIR = MARKET_CACHE_DIR
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "pool_quality_compare"


# =============================================================================
# Helpers
# =============================================================================

def normalize_plain_code(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    match = re.search(r"(\d{6})", text)
    return match.group(1) if match else text


def normalize_prefixed_code(value: object) -> str:
    code = normalize_plain_code(value)
    if len(code) == 6 and code.isdigit():
        return f"SH#{code}" if code.startswith("6") else f"SZ#{code}"
    return code


def infer_code_from_path(path: Path) -> str:
    return normalize_prefixed_code(path.stem)


def find_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_ohlcv(df: pd.DataFrame, fallback_code: str = "") -> pd.DataFrame:
    out = df.copy()

    rename_map = {
        "日期": "date",
        "时间": "date",
        "trade_date": "date",
        "datetime": "date",
        "signal_date": "date",
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

    date_col = find_first_col(out, ["date", "trade_date", "signal_date", "datetime", "日期"])
    if date_col is None:
        raise ValueError(f"Missing date column. Columns={list(out.columns)}")
    out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()

    code_col = find_first_col(out, ["code", "symbol", "stock_code", "ts_code", "file", "filename"])
    if code_col is not None:
        out["code"] = out[code_col].map(normalize_prefixed_code)
    else:
        out["code"] = normalize_prefixed_code(fallback_code)

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in out.columns:
            raise ValueError(f"Missing {c} column. Columns={list(out.columns)}")
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    out = out.sort_values("date").drop_duplicates(["date"], keep="last").reset_index(drop=True)
    return out


def read_pool(pool_path: Path, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if not pool_path.exists():
        raise FileNotFoundError(f"Pool not found: {pool_path}")

    df = pd.read_parquet(pool_path)
    df = ensure_ohlcv(df, fallback_code="")

    if "selected" in df.columns:
        selected = pd.to_numeric(df["selected"], errors="coerce").fillna(0).astype(int).astype(bool)
        df = df[selected].copy()

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)].copy()

    df = df.drop_duplicates(["date", "code"], keep="last").reset_index(drop=True)
    return df


def build_market_file_index(market_cache_dir: Path) -> dict[str, Path]:
    if not market_cache_dir.exists():
        raise FileNotFoundError(f"Market cache dir not found: {market_cache_dir}")

    index: dict[str, Path] = {}
    for path in sorted(market_cache_dir.glob("*.parquet")):
        code = infer_code_from_path(path)
        plain = normalize_plain_code(code)
        prefixed = normalize_prefixed_code(code)
        index[plain] = path
        index[prefixed] = path
    if not index:
        raise FileNotFoundError(f"No parquet files found in {market_cache_dir}")
    return index


class MarketCache:
    def __init__(self, market_cache_dir: Path):
        self.file_index = build_market_file_index(market_cache_dir)
        self.cache: dict[str, pd.DataFrame | None] = {}

    def get(self, code: str) -> pd.DataFrame | None:
        prefixed = normalize_prefixed_code(code)
        plain = normalize_plain_code(code)
        key = prefixed
        if key in self.cache:
            return self.cache[key]

        path = self.file_index.get(prefixed) or self.file_index.get(plain)
        if path is None:
            self.cache[key] = None
            return None

        try:
            df = pd.read_parquet(path)
            df = ensure_ohlcv(df, fallback_code=prefixed)
            self.cache[key] = df
            return df
        except Exception:
            self.cache[key] = None
            return None


def pct(a: float, b: float) -> float:
    if b is None or pd.isna(b) or b == 0 or a is None or pd.isna(a):
        return np.nan
    return (a / b - 1.0) * 100.0


def build_forward_record(pool_row: pd.Series, market: MarketCache, pool_label: str) -> dict[str, Any]:
    code = str(pool_row["code"])
    signal_date = pd.Timestamp(pool_row["date"]).normalize()

    out: dict[str, Any] = {
        "pool": pool_label,
        "date": signal_date,
        "code": code,
        "name": pool_row.get("name", pool_row.get("stock_name", "")),
        "score": pool_row.get("score", np.nan),
        "score_pct": pool_row.get("score_pct", np.nan),
    }

    m = market.get(code)
    if m is None or m.empty:
        out["has_forward"] = False
        out["missing_reason"] = "missing_market_file"
        return out

    pos_list = m.index[m["date"] == signal_date].tolist()
    if not pos_list:
        out["has_forward"] = False
        out["missing_reason"] = "signal_date_not_in_market"
        return out

    pos = int(pos_list[-1])
    if pos + 1 >= len(m):
        out["has_forward"] = False
        out["missing_reason"] = "no_t1"
        return out

    t0 = m.iloc[pos]
    t1 = m.iloc[pos + 1] if pos + 1 < len(m) else None
    t2 = m.iloc[pos + 2] if pos + 2 < len(m) else None
    t3 = m.iloc[pos + 3] if pos + 3 < len(m) else None

    t0_close = float(t0["close"])
    t1_open = float(t1["open"]) if t1 is not None else np.nan

    out.update({
        "has_forward": True,
        "missing_reason": "",
        "t0_close": t0_close,
        "t1_date": t1["date"] if t1 is not None else pd.NaT,
        "t1_open": t1_open,
        "t1_close": float(t1["close"]) if t1 is not None else np.nan,
        "t2_date": t2["date"] if t2 is not None else pd.NaT,
        "t2_open": float(t2["open"]) if t2 is not None else np.nan,
        "t2_close": float(t2["close"]) if t2 is not None else np.nan,
        "t3_date": t3["date"] if t3 is not None else pd.NaT,
        "t3_open": float(t3["open"]) if t3 is not None else np.nan,
        "t3_close": float(t3["close"]) if t3 is not None else np.nan,
    })

    out["t1_open_gap_pct"] = pct(out["t1_open"], t0_close)
    out["t1_close_ret_from_t0_close_pct"] = pct(out["t1_close"], t0_close)
    out["t2_close_ret_from_t0_close_pct"] = pct(out["t2_close"], t0_close)
    out["t3_close_ret_from_t0_close_pct"] = pct(out["t3_close"], t0_close)

    out["t1_open_to_t1_close_pct"] = pct(out["t1_close"], t1_open)
    out["t1_open_to_t2_close_pct"] = pct(out["t2_close"], t1_open)
    out["t1_open_to_t3_close_pct"] = pct(out["t3_close"], t1_open)

    lows = []
    highs = []
    for row in [t1, t2, t3]:
        if row is not None:
            lows.append(float(row["low"]))
            highs.append(float(row["high"]))
    out["t1_to_t3_max_high_from_t1_open_pct"] = pct(max(highs), t1_open) if highs else np.nan
    out["t1_to_t3_max_drawdown_from_t1_open_pct"] = pct(min(lows), t1_open) if lows else np.nan

    return out


def summarize_one(detail: pd.DataFrame, pool_label: str) -> dict[str, Any]:
    valid = detail[detail["has_forward"] == True].copy()
    row: dict[str, Any] = {
        "pool": pool_label,
        "signals": len(detail),
        "forward_valid": len(valid),
        "coverage_pct": len(valid) / len(detail) * 100 if len(detail) else 0.0,
        "unique_stocks": detail["code"].nunique() if "code" in detail.columns else 0,
        "unique_dates": detail["date"].nunique() if "date" in detail.columns else 0,
    }

    metrics = [
        "t1_open_gap_pct",
        "t1_close_ret_from_t0_close_pct",
        "t2_close_ret_from_t0_close_pct",
        "t3_close_ret_from_t0_close_pct",
        "t1_open_to_t1_close_pct",
        "t1_open_to_t2_close_pct",
        "t1_open_to_t3_close_pct",
        "t1_to_t3_max_high_from_t1_open_pct",
        "t1_to_t3_max_drawdown_from_t1_open_pct",
    ]

    for col in metrics:
        s = pd.to_numeric(valid[col], errors="coerce").dropna() if col in valid.columns else pd.Series(dtype=float)
        row[f"{col}_avg"] = s.mean() if len(s) else np.nan
        row[f"{col}_median"] = s.median() if len(s) else np.nan
        row[f"{col}_win_gt0_pct"] = (s > 0).mean() * 100 if len(s) else np.nan
        row[f"{col}_ge2_pct"] = (s >= 2).mean() * 100 if len(s) else np.nan
        row[f"{col}_le_minus2_pct"] = (s <= -2).mean() * 100 if len(s) else np.nan

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare N pool parquet files by forward return quality.")
    parser.add_argument("--pool-paths", nargs="+", type=Path, required=True, help="One or more pool parquet paths.")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels. Must match --pool-paths count.")
    parser.add_argument("--market-cache-dir", type=Path, default=DEFAULT_MARKET_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--detail-limit", type=int, default=0, help="0 means export all detail rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    labels = args.labels
    if labels and len(labels) != len(args.pool_paths):
        raise ValueError("--labels count must equal --pool-paths count.")
    if not labels:
        labels = [p.stem.replace("_pool", "") for p in args.pool_paths]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    market = MarketCache(args.market_cache_dir)

    all_details: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    print("=" * 120)
    print("Compare N pool quality")
    print(f"market_cache_dir: {args.market_cache_dir}")
    print(f"output_dir      : {args.output_dir}")
    print(f"date range      : {args.start_date} -> {args.end_date}")
    print("=" * 120)

    for pool_path, label in zip(args.pool_paths, labels):
        print(f"\n[LOAD] {label}: {pool_path}")
        pool = read_pool(pool_path, args.start_date, args.end_date)
        print(f"[INFO] pool rows after filter: {len(pool):,}")

        iterator = pool.iterrows()
        if tqdm is not None:
            iterator = tqdm(list(pool.iterrows()), desc=f"Forward {label}", unit="signal")

        rows = [build_forward_record(row, market, label) for _, row in iterator]
        detail = pd.DataFrame(rows)
        all_details.append(detail)
        summaries.append(summarize_one(detail, label))

    summary = pd.DataFrame(summaries)
    detail_all = pd.concat(all_details, ignore_index=True, sort=False) if all_details else pd.DataFrame()

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    summary_path = args.output_dir / f"compare_n_pools_summary_{timestamp}.csv"
    detail_path = args.output_dir / f"compare_n_pools_detail_{timestamp}.csv"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if args.detail_limit and args.detail_limit > 0:
        detail_all.head(args.detail_limit).to_csv(detail_path, index=False, encoding="utf-8-sig")
    else:
        detail_all.to_csv(detail_path, index=False, encoding="utf-8-sig")

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 220)

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)

    key_cols = [
        "pool",
        "signals",
        "forward_valid",
        "coverage_pct",
        "unique_stocks",
        "unique_dates",
        "t1_open_gap_pct_avg",
        "t1_open_to_t2_close_pct_avg",
        "t1_open_to_t2_close_pct_win_gt0_pct",
        "t1_open_to_t2_close_pct_ge2_pct",
        "t1_open_to_t3_close_pct_avg",
        "t1_open_to_t3_close_pct_win_gt0_pct",
        "t1_open_to_t3_close_pct_ge2_pct",
        "t1_to_t3_max_drawdown_from_t1_open_pct_avg",
    ]
    show_cols = [c for c in key_cols if c in summary.columns]
    print(summary[show_cols].to_string(index=False))

    print("\nSaved summary:", summary_path)
    print("Saved detail :", detail_path)


if __name__ == "__main__":
    main()
