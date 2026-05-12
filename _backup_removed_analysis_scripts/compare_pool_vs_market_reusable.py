# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Reusable pool-vs-market forward analysis.

Purpose:
    Verify whether a selected stock pool outperforms the same-day full market,
    instead of only benefiting from a strong market.

Features:
    1. Supports any pool parquet/csv/txt file.
    2. Compares selected pool against same-day full market.
    3. Calculates T+1/T+2/T+3/T+4/T+5 forward returns.
    4. Keeps all market-cache indicator columns for further analysis.
    5. Uses parallel parquet loading to speed up market cache loading.
    6. Outputs daily comparison, horizon summary, group summaries, and diagnostics.

Example:
python .\\analyze_tools\\compare_pool_vs_market_reusable.py `
  --pool-path "C:\\Users\\zyf37\\Desktop\\BackTest_Data\\pools\\b2_confirm_select_strategy_v0_pool.parquet" `
  --market-cache-dir "C:\\Users\\zyf37\\Desktop\\BackTest_Data\\market_cache\\daily_bars_by_symbol" `
  --output-dir "C:\\Users\\zyf37\\Desktop\\BackTest_Data\\output\\b2_v0_vs_market" `
  --start-date 2024-01-01 `
  --end-date 2026-05-08 `
  --horizons 1,2,3,4,5 `
  --max-workers 8
"""

import argparse
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


DEFAULT_DATA_ROOT = Path(r"C:\Users\zyf37\Desktop\BackTest_Data")
DEFAULT_POOL_PATH = DEFAULT_DATA_ROOT / "pools" / "b2_confirm_select_strategy_v0_pool.parquet"
DEFAULT_MARKET_CACHE_DIR = DEFAULT_DATA_ROOT / "market_cache" / "daily_bars_by_symbol"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "output" / "pool_vs_market_reusable"


def progress_iter(items, desc: str = ""):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, ncols=100)


def normalize_code(x: Any) -> str:
    if pd.isna(x):
        return ""

    s = str(x).strip().upper()
    if "#" in s:
        s = s.split("#")[-1]

    s = (
        s.replace(".SH", "")
        .replace(".SZ", "")
        .replace(".BJ", "")
        .replace("SH", "")
        .replace("SZ", "")
        .replace("BJ", "")
    )

    m = re.search(r"(\d{6})", s)
    if m:
        return m.group(1)

    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 6:
        return digits[-6:]
    if digits:
        return digits.zfill(6)[-6:]

    return ""


def infer_exchange_prefix(code: str) -> str:
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return "SH"
    if code.startswith(("000", "001", "002", "003", "300", "301")):
        return "SZ"
    if code.startswith(("8", "4", "9")):
        return "BJ"
    return ""


def normalize_symbol(x: Any) -> str:
    code = normalize_code(x)
    if not code:
        return ""

    prefix = ""
    s = str(x).strip().upper()

    if "SH" in s:
        prefix = "SH"
    elif "SZ" in s:
        prefix = "SZ"
    elif "BJ" in s:
        prefix = "BJ"
    else:
        prefix = infer_exchange_prefix(code)

    return f"{prefix}#{code}" if prefix else code


def parse_horizons(s: str) -> list[int]:
    out = []

    for part in str(s).replace(";", ",").split(","):
        part = part.strip().upper().replace("T", "")
        if not part:
            continue

        n = int(part)
        if n <= 0:
            raise ValueError(f"invalid horizon: {part}")

        out.append(n)

    if not out:
        raise ValueError("empty horizons")

    return sorted(set(out))


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix in {".csv", ".txt"}:
        for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                pass
        return pd.read_csv(path)

    raise ValueError(f"unsupported file type: {path}")


def standardize_bar_df(df: pd.DataFrame, fallback_symbol: str) -> pd.DataFrame:
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

    if "date" not in out.columns or "close" not in out.columns:
        return pd.DataFrame()

    if "symbol" not in out.columns:
        if "code" in out.columns:
            out["symbol"] = out["code"]
        else:
            out["symbol"] = fallback_symbol

    out["symbol"] = out["symbol"].map(normalize_symbol)
    out.loc[out["symbol"] == "", "symbol"] = normalize_symbol(fallback_symbol)

    out["code"] = out["symbol"].map(normalize_code)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    if "open" in out.columns:
        out["open"] = pd.to_numeric(out["open"], errors="coerce")
    else:
        out["open"] = np.nan

    out = out.dropna(subset=["date", "close"])
    out = out[(out["code"] != "") & (out["close"] > 0)].copy()
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    return out.reset_index(drop=True)


def _load_one_market_file(path: Path, horizons: list[int]) -> pd.DataFrame:
    raw = pd.read_parquet(path)

    symbol_from_file = path.stem
    bars = standardize_bar_df(raw, symbol_from_file)

    if bars.empty or len(bars) <= max(horizons):
        return pd.DataFrame()

    bars = bars.sort_values("date").reset_index(drop=True)

    for h in horizons:
        future_close = bars["close"].shift(-h)
        ret_col = f"T{h}_close_to_close_ret_pct"

        bars[f"T{h}_future_close"] = future_close
        bars[ret_col] = (future_close / bars["close"] - 1.0) * 100.0
        bars[f"T{h}_is_up"] = bars[ret_col] > 0

    return bars


def load_market_cache(
    market_cache_dir: Path,
    horizons: list[int],
    max_workers: int | None = None,
) -> pd.DataFrame:
    if not market_cache_dir.exists():
        raise FileNotFoundError(f"market cache dir not found: {market_cache_dir}")

    files = sorted(market_cache_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"no parquet files found: {market_cache_dir}")

    if max_workers is None:
        max_workers = min(12, max(2, (os.cpu_count() or 4) - 1))

    parts = []

    print(f"[INFO] Loading market cache with max_workers={max_workers}, files={len(files):,}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_load_one_market_file, path, horizons): path
            for path in files
        }

        iterator = as_completed(future_map)

        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=len(future_map),
                desc="Load market cache",
                ncols=100,
            )

        for future in iterator:
            path = future_map[future]

            try:
                bars = future.result()
                if not bars.empty:
                    parts.append(bars)
            except Exception as exc:
                print(f"[WARN] failed to read {path}: {exc}")

    if not parts:
        raise RuntimeError("market cache loaded empty")

    market = pd.concat(parts, ignore_index=True)
    market = market.drop_duplicates(subset=["date", "code"], keep="last")
    market = market.sort_values(["date", "code"]).reset_index(drop=True)

    return market


def load_pool(
    pool_path: Path,
    start_date: str | None,
    end_date: str | None,
    selected_only: bool,
) -> pd.DataFrame:
    raw = read_table(pool_path)

    if raw.empty:
        raise RuntimeError(f"pool file is empty: {pool_path}")

    if "date" not in raw.columns:
        raise KeyError(f"pool missing date column. columns={list(raw.columns)}")

    if "code" in raw.columns:
        code_col = "code"
    elif "symbol" in raw.columns:
        code_col = "symbol"
    else:
        raise KeyError(f"pool missing code/symbol column. columns={list(raw.columns)}")

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["code"] = df[code_col].map(normalize_code)
    df["symbol"] = df[code_col].map(normalize_symbol)

    df = df[df["date"].notna() & (df["code"] != "")].copy()

    if selected_only and "selected" in df.columns:
        selected_num = pd.to_numeric(df["selected"], errors="coerce")

        if selected_num.notna().any():
            df = df[selected_num.fillna(0).astype(int) == 1].copy()
        else:
            df = df[df["selected"].astype(bool)].copy()

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date).normalize()].copy()

    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date).normalize()].copy()

    df = df.sort_values(["date", "code"]).drop_duplicates(
        subset=["date", "code"],
        keep="last",
    )

    if df.empty:
        raise RuntimeError("pool is empty after filtering")

    return df.reset_index(drop=True)


def add_pool_daily_count(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()

    daily_count = out.groupby("date")["code"].nunique().rename("pool_daily_count")
    out = out.merge(daily_count, on="date", how="left")

    bins = [-1, 0, 5, 10, 20, 50, 100, 999999]
    labels = ["0", "1-5", "6-10", "11-20", "21-50", "51-100", "100+"]

    out["pool_daily_count_bin"] = pd.cut(
        out["pool_daily_count"],
        bins=bins,
        labels=labels,
    ).astype(str)

    return out


def attach_forward_returns(
    pool: pd.DataFrame,
    market_fwd: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    base_cols = ["date", "code"]

    market_cols = list(market_fwd.columns)
    attach_cols = []

    for c in market_cols:
        if c in base_cols:
            continue
        attach_cols.append(c)

    m = market_fwd[base_cols + attach_cols].copy()

    out = pool.merge(
        m,
        on=["date", "code"],
        how="left",
        suffixes=("", "_market"),
        validate="many_to_one",
    )

    return out


def safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()

    if s.empty:
        return np.nan

    return float(s.mean())


def summarize_rows(df: pd.DataFrame, ret_col: str, up_col: str) -> dict[str, Any]:
    if ret_col not in df.columns or up_col not in df.columns:
        return {
            "count": 0,
            "up_count": 0,
            "up_ratio": np.nan,
            "avg_return_pct": np.nan,
            "median_return_pct": np.nan,
            "p25_return_pct": np.nan,
            "p75_return_pct": np.nan,
        }

    valid = df[pd.to_numeric(df[ret_col], errors="coerce").notna()].copy()

    if valid.empty:
        return {
            "count": 0,
            "up_count": 0,
            "up_ratio": np.nan,
            "avg_return_pct": np.nan,
            "median_return_pct": np.nan,
            "p25_return_pct": np.nan,
            "p75_return_pct": np.nan,
        }

    ret = pd.to_numeric(valid[ret_col], errors="coerce")
    up = valid[up_col].fillna(False).astype(bool)

    return {
        "count": int(len(valid)),
        "up_count": int(up.sum()),
        "up_ratio": float(up.mean()),
        "avg_return_pct": float(ret.mean()),
        "median_return_pct": float(ret.median()),
        "p25_return_pct": float(ret.quantile(0.25)),
        "p75_return_pct": float(ret.quantile(0.75)),
    }


def build_daily_comparison(
    pool_fwd: pd.DataFrame,
    market_fwd: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    target_dates = sorted(pool_fwd["date"].dropna().unique())
    market_on_dates = market_fwd[market_fwd["date"].isin(target_dates)].copy()

    rows = []

    for d in progress_iter(target_dates, "Daily compare"):
        p_day_all = pool_fwd[pool_fwd["date"] == d]
        m_day_all = market_on_dates[market_on_dates["date"] == d]

        if p_day_all.empty or m_day_all.empty:
            continue

        for h in horizons:
            ret_col = f"T{h}_close_to_close_ret_pct"
            up_col = f"T{h}_is_up"

            ps = summarize_rows(p_day_all, ret_col, up_col)
            ms = summarize_rows(m_day_all, ret_col, up_col)

            rows.append(
                {
                    "date": pd.to_datetime(d).strftime("%Y-%m-%d"),
                    "horizon": f"T{h}",

                    "pool_count": ps["count"],
                    "pool_up_count": ps["up_count"],
                    "pool_up_ratio": ps["up_ratio"],
                    "pool_avg_return_pct": ps["avg_return_pct"],
                    "pool_median_return_pct": ps["median_return_pct"],
                    "pool_p25_return_pct": ps["p25_return_pct"],
                    "pool_p75_return_pct": ps["p75_return_pct"],

                    "market_count": ms["count"],
                    "market_up_count": ms["up_count"],
                    "market_up_ratio": ms["up_ratio"],
                    "market_avg_return_pct": ms["avg_return_pct"],
                    "market_median_return_pct": ms["median_return_pct"],
                    "market_p25_return_pct": ms["p25_return_pct"],
                    "market_p75_return_pct": ms["p75_return_pct"],

                    "excess_up_ratio": ps["up_ratio"] - ms["up_ratio"]
                    if pd.notna(ps["up_ratio"]) and pd.notna(ms["up_ratio"])
                    else np.nan,

                    "excess_avg_return_pct": ps["avg_return_pct"] - ms["avg_return_pct"]
                    if pd.notna(ps["avg_return_pct"]) and pd.notna(ms["avg_return_pct"])
                    else np.nan,

                    "excess_median_return_pct": ps["median_return_pct"] - ms["median_return_pct"]
                    if pd.notna(ps["median_return_pct"]) and pd.notna(ms["median_return_pct"])
                    else np.nan,
                }
            )

    return pd.DataFrame(rows)


def weighted_avg(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")

    mask = v.notna() & w.notna() & (w > 0)

    if not mask.any():
        return np.nan

    return float(np.average(v[mask], weights=w[mask]))


def build_horizon_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()

    rows = []

    for horizon, g in daily.groupby("horizon", sort=True):
        pool_total = int(g["pool_count"].sum())
        market_total = int(g["market_count"].sum())
        pool_up_total = int(g["pool_up_count"].sum())
        market_up_total = int(g["market_up_count"].sum())

        pool_weighted_up = pool_up_total / pool_total if pool_total else np.nan
        market_weighted_up = market_up_total / market_total if market_total else np.nan

        pool_weighted_ret = weighted_avg(g["pool_avg_return_pct"], g["pool_count"])
        market_weighted_ret = weighted_avg(g["market_avg_return_pct"], g["market_count"])

        rows.append(
            {
                "horizon": horizon,
                "trading_days": int(g["date"].nunique()),
                "pool_total_rows": pool_total,
                "market_total_rows": market_total,

                "pool_weighted_up_ratio": pool_weighted_up,
                "market_weighted_up_ratio": market_weighted_up,
                "weighted_excess_up_ratio": pool_weighted_up - market_weighted_up
                if pd.notna(pool_weighted_up) and pd.notna(market_weighted_up)
                else np.nan,

                "pool_weighted_avg_return_pct": pool_weighted_ret,
                "market_weighted_avg_return_pct": market_weighted_ret,
                "weighted_excess_avg_return_pct": pool_weighted_ret - market_weighted_ret
                if pd.notna(pool_weighted_ret) and pd.notna(market_weighted_ret)
                else np.nan,

                "pool_daily_mean_up_ratio": safe_mean(g["pool_up_ratio"]),
                "market_daily_mean_up_ratio": safe_mean(g["market_up_ratio"]),
                "daily_mean_excess_up_ratio": safe_mean(g["excess_up_ratio"]),

                "pool_daily_mean_return_pct": safe_mean(g["pool_avg_return_pct"]),
                "market_daily_mean_return_pct": safe_mean(g["market_avg_return_pct"]),
                "daily_mean_excess_return_pct": safe_mean(g["excess_avg_return_pct"]),

                "positive_excess_up_days": int((g["excess_up_ratio"] > 0).sum()),
                "positive_excess_return_days": int((g["excess_avg_return_pct"] > 0).sum()),
                "positive_excess_return_day_ratio": float((g["excess_avg_return_pct"] > 0).mean())
                if len(g)
                else np.nan,

                "worst_daily_excess_return_pct": float(g["excess_avg_return_pct"].min())
                if g["excess_avg_return_pct"].notna().any()
                else np.nan,

                "best_daily_excess_return_pct": float(g["excess_avg_return_pct"].max())
                if g["excess_avg_return_pct"].notna().any()
                else np.nan,
            }
        )

    return pd.DataFrame(rows)


def build_summary_wide(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    rows = []

    for _, r in summary.iterrows():
        h = str(r["horizon"])

        for c in summary.columns:
            if c == "horizon":
                continue

            rows.append(
                {
                    "metric": f"{h}_{c}",
                    "value": r[c],
                }
            )

    return pd.DataFrame(rows)


def build_group_summary(
    pool_fwd: pd.DataFrame,
    market_fwd: pd.DataFrame,
    horizons: list[int],
    group_col: str,
) -> pd.DataFrame:
    if group_col not in pool_fwd.columns:
        return pd.DataFrame()

    rows = []

    for group_value, p_group in pool_fwd.groupby(group_col, dropna=False):
        dates = sorted(p_group["date"].dropna().unique())
        m_same_dates = market_fwd[market_fwd["date"].isin(dates)].copy()

        if p_group.empty or m_same_dates.empty:
            continue

        for h in horizons:
            ret_col = f"T{h}_close_to_close_ret_pct"
            up_col = f"T{h}_is_up"

            ps = summarize_rows(p_group, ret_col, up_col)
            ms = summarize_rows(m_same_dates, ret_col, up_col)

            rows.append(
                {
                    group_col: group_value,
                    "horizon": f"T{h}",
                    "trading_days": int(p_group["date"].nunique()),

                    "pool_rows": ps["count"],
                    "market_rows_same_dates": ms["count"],

                    "pool_up_ratio": ps["up_ratio"],
                    "market_up_ratio": ms["up_ratio"],
                    "excess_up_ratio": ps["up_ratio"] - ms["up_ratio"]
                    if pd.notna(ps["up_ratio"]) and pd.notna(ms["up_ratio"])
                    else np.nan,

                    "pool_avg_return_pct": ps["avg_return_pct"],
                    "market_avg_return_pct": ms["avg_return_pct"],
                    "excess_avg_return_pct": ps["avg_return_pct"] - ms["avg_return_pct"]
                    if pd.notna(ps["avg_return_pct"]) and pd.notna(ms["avg_return_pct"])
                    else np.nan,

                    "pool_median_return_pct": ps["median_return_pct"],
                    "market_median_return_pct": ms["median_return_pct"],
                    "excess_median_return_pct": ps["median_return_pct"] - ms["median_return_pct"]
                    if pd.notna(ps["median_return_pct"]) and pd.notna(ms["median_return_pct"])
                    else np.nan,
                }
            )

    return pd.DataFrame(rows)


def detect_regime_col(df: pd.DataFrame) -> str | None:
    candidates = [
        "market_regime",
        "regime",
        "market_state",
        "market_env",
        "大盘状态",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    return None


def save_json(path: Path, obj: dict[str, Any]) -> None:
    def conv(x):
        if isinstance(x, np.integer):
            return int(x)

        if isinstance(x, np.floating):
            if math.isnan(float(x)):
                return None
            return float(x)

        if isinstance(x, pd.Timestamp):
            return x.strftime("%Y-%m-%d")

        return str(x)

    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=conv),
        encoding="utf-8",
    )


def save_outputs(
    output_dir: Path,
    pool_fwd: pd.DataFrame,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    wide: pd.DataFrame,
    by_count: pd.DataFrame,
    by_regime: pd.DataFrame,
    extra_group_outputs: dict[str, pd.DataFrame],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "pool_forward_rows": str(output_dir / "1_pool_forward_rows.csv"),
        "daily_pool_vs_market": str(output_dir / "2_daily_pool_vs_market.csv"),
        "horizon_summary": str(output_dir / "3_horizon_summary.csv"),
        "summary_wide": str(output_dir / "4_summary_wide.csv"),
        "by_pool_daily_count": str(output_dir / "5_by_pool_daily_count.csv"),
        "by_market_regime": str(output_dir / "6_by_market_regime.csv"),
    }

    pool_fwd.to_csv(paths["pool_forward_rows"], index=False, encoding="utf-8-sig")
    daily.to_csv(paths["daily_pool_vs_market"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["horizon_summary"], index=False, encoding="utf-8-sig")
    wide.to_csv(paths["summary_wide"], index=False, encoding="utf-8-sig")
    by_count.to_csv(paths["by_pool_daily_count"], index=False, encoding="utf-8-sig")

    if not by_regime.empty:
        by_regime.to_csv(paths["by_market_regime"], index=False, encoding="utf-8-sig")
    else:
        paths["by_market_regime"] = ""

    for col, gdf in extra_group_outputs.items():
        safe_col = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", col)
        out_path = output_dir / f"extra_group_by_{safe_col}.csv"
        gdf.to_csv(out_path, index=False, encoding="utf-8-sig")
        paths[f"extra_group_by_{col}"] = str(out_path)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare any selected stock pool against same-day full market forward returns."
    )

    parser.add_argument("--pool-path", type=Path, default=DEFAULT_POOL_PATH)
    parser.add_argument("--market-cache-dir", type=Path, default=DEFAULT_MARKET_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--horizons", default="1,2,3,4,5")

    parser.add_argument("--selected-only", action="store_true", default=True)
    parser.add_argument(
        "--no-selected-only",
        action="store_false",
        dest="selected_only",
        help="Do not filter selected==1 even if selected column exists.",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Parallel workers for loading market cache. Default: auto.",
    )

    parser.add_argument(
        "--extra-group-cols",
        default="",
        help="Optional comma-separated pool/market columns to group. Example: score_rank_key,daily_return_pct_bin",
    )

    args = parser.parse_args()

    horizons = parse_horizons(args.horizons)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Pool vs Market Reusable Analysis")
    print("=" * 100)
    print(f"pool_path        : {args.pool_path}")
    print(f"market_cache_dir : {args.market_cache_dir}")
    print(f"output_dir       : {args.output_dir}")
    print(f"start_date       : {args.start_date}")
    print(f"end_date         : {args.end_date}")
    print(f"horizons         : {horizons}")
    print(f"selected_only    : {args.selected_only}")
    print(f"max_workers      : {args.max_workers}")
    print("=" * 100)

    pool = load_pool(
        pool_path=args.pool_path,
        start_date=args.start_date,
        end_date=args.end_date,
        selected_only=args.selected_only,
    )

    pool = add_pool_daily_count(pool)

    market = load_market_cache(
        market_cache_dir=args.market_cache_dir,
        horizons=horizons,
        max_workers=args.max_workers,
    )

    pool_fwd = attach_forward_returns(
        pool=pool,
        market_fwd=market,
        horizons=horizons,
    )

    missing_market_rows = int(pool_fwd["close"].isna().sum()) if "close" in pool_fwd.columns else 0

    if missing_market_rows > 0:
        print(f"[WARN] pool rows missing market forward data: {missing_market_rows:,}")

    daily = build_daily_comparison(
        pool_fwd=pool_fwd,
        market_fwd=market,
        horizons=horizons,
    )

    summary = build_horizon_summary(daily)
    wide = build_summary_wide(summary)

    by_count = build_group_summary(
        pool_fwd=pool_fwd,
        market_fwd=market,
        horizons=horizons,
        group_col="pool_daily_count_bin",
    )

    regime_col = detect_regime_col(pool_fwd)

    if regime_col:
        by_regime = build_group_summary(
            pool_fwd=pool_fwd,
            market_fwd=market,
            horizons=horizons,
            group_col=regime_col,
        )
    else:
        by_regime = pd.DataFrame()

    extra_group_cols = [
        x.strip()
        for x in str(args.extra_group_cols).split(",")
        if x.strip()
    ]

    extra_group_outputs = {}

    for col in extra_group_cols:
        if col not in pool_fwd.columns:
            print(f"[WARN] extra group col not found, skip: {col}")
            continue

        gdf = build_group_summary(
            pool_fwd=pool_fwd,
            market_fwd=market,
            horizons=horizons,
            group_col=col,
        )

        extra_group_outputs[col] = gdf

    output_paths = save_outputs(
        output_dir=args.output_dir,
        pool_fwd=pool_fwd,
        daily=daily,
        summary=summary,
        wide=wide,
        by_count=by_count,
        by_regime=by_regime,
        extra_group_outputs=extra_group_outputs,
    )

    diagnostics = {
        "pool_path": str(args.pool_path),
        "market_cache_dir": str(args.market_cache_dir),
        "output_dir": str(args.output_dir),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "horizons": horizons,
        "selected_only": args.selected_only,
        "max_workers": args.max_workers,

        "pool_rows": int(len(pool)),
        "pool_symbols": int(pool["code"].nunique()),
        "pool_start_date": str(pool["date"].min().date()),
        "pool_end_date": str(pool["date"].max().date()),
        "pool_trading_days": int(pool["date"].nunique()),
        "pool_daily_mean_count": float(pool.groupby("date")["code"].nunique().mean()),
        "pool_daily_median_count": float(pool.groupby("date")["code"].nunique().median()),

        "market_rows": int(len(market)),
        "market_symbols": int(market["code"].nunique()),
        "market_start_date": str(market["date"].min().date()),
        "market_end_date": str(market["date"].max().date()),

        "missing_market_rows_in_pool": missing_market_rows,
        "detected_regime_col": regime_col,
        "extra_group_cols": extra_group_cols,
        "outputs": output_paths,
    }

    save_json(args.output_dir / "diagnostics.json", diagnostics)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)

    for name, path in output_paths.items():
        if path:
            print(f"Saved {name}: {path}")

    print(f"Saved diagnostics: {args.output_dir / 'diagnostics.json'}")

    if not summary.empty:
        print("\nKey table:")
        print(summary.to_string(index=False))
    else:
        print("\n[WARN] summary is empty.")


if __name__ == "__main__":
    main()