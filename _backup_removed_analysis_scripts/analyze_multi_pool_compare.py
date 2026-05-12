# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import itertools
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
DEFAULT_MARKET_CACHE_DIR = DEFAULT_DATA_ROOT / "market_cache" / "daily_bars_by_symbol"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "output" / "multi_pool_compare"


DATE_CANDIDATES = [
    "date",
    "trade_date",
    "datetime",
    "Date",
    "DATE",
    "日期",
    "时间",
]

CODE_CANDIDATES = [
    "code",
    "symbol",
    "ts_code",
    "股票代码",
    "代码",
    "证券代码",
]

CLOSE_CANDIDATES = [
    "close",
    "CLOSE",
    "收盘",
    "收盘价",
]


def progress_iter(items, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, ncols=100)


def detect_col(columns, candidates: list[str]) -> str | None:
    col_set = {str(c): c for c in columns}
    lower_map = {str(c).lower(): c for c in columns}

    for c in candidates:
        if c in col_set:
            return col_set[c]
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    return None


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


def parse_list(raw: str) -> list[str]:
    raw = str(raw).strip()

    if not raw:
        return []

    sep = ";" if ";" in raw else ","

    return [x.strip().strip('"').strip("'") for x in raw.split(sep) if x.strip()]


def parse_horizons(raw: str) -> list[int]:
    out = []

    for part in str(raw).replace(";", ",").split(","):
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


def load_pool(
    pool_path: Path,
    pool_name: str,
    selected_only: bool,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    raw = read_table(pool_path)

    if raw.empty:
        raise RuntimeError(f"{pool_name}: pool file is empty: {pool_path}")

    df = raw.copy()

    date_col = detect_col(df.columns, DATE_CANDIDATES)
    code_col = detect_col(df.columns, CODE_CANDIDATES)

    if date_col is None:
        raise KeyError(f"{pool_name}: cannot detect date column. columns={list(df.columns)}")

    if code_col is None:
        for c in df.columns:
            sample = df[c].dropna().astype(str).head(100).tolist()
            if any(normalize_code(v) for v in sample):
                code_col = c
                break

    if code_col is None:
        raise KeyError(f"{pool_name}: cannot detect code/symbol column. columns={list(df.columns)}")

    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
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

    df["pool_name"] = pool_name
    df["pool_path"] = str(pool_path)

    df = df.sort_values(["date", "code"]).drop_duplicates(
        subset=["pool_name", "date", "code"],
        keep="last",
    )

    return df.reset_index(drop=True)


def get_parquet_columns(path: Path) -> list[str] | None:
    try:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    except Exception:
        return None


def standardize_market_df(raw: pd.DataFrame, fallback_symbol: str) -> pd.DataFrame:
    df = raw.copy()

    rename_map = {
        "日期": "date",
        "时间": "date",
        "trade_date": "date",
        "datetime": "date",
        "Date": "date",
        "DATE": "date",
        "收盘": "close",
        "收盘价": "close",
        "CLOSE": "close",
    }

    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    if "date" not in df.columns:
        date_col = detect_col(df.columns, DATE_CANDIDATES)
        if date_col is not None:
            df = df.rename(columns={date_col: "date"})

    if "close" not in df.columns:
        close_col = detect_col(df.columns, CLOSE_CANDIDATES)
        if close_col is not None:
            df = df.rename(columns={close_col: "close"})

    if "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()

    if "symbol" not in df.columns:
        if "code" in df.columns:
            df["symbol"] = df["code"]
        else:
            df["symbol"] = fallback_symbol

    df["symbol"] = df["symbol"].map(normalize_symbol)
    df.loc[df["symbol"] == "", "symbol"] = normalize_symbol(fallback_symbol)

    df["code"] = df["symbol"].map(normalize_code)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["date", "close"])
    df = df[(df["code"] != "") & (df["close"] > 0)].copy()
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    return df[["date", "code", "symbol", "close"]].reset_index(drop=True)


def read_one_market_file(
    path: Path,
    horizons: list[int],
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    candidate_cols = list(dict.fromkeys(DATE_CANDIDATES + CODE_CANDIDATES + CLOSE_CANDIDATES))

    cols = get_parquet_columns(path)

    if cols:
        use_cols = [c for c in candidate_cols if c in cols]
        if len(use_cols) >= 2:
            raw = pd.read_parquet(path, columns=use_cols)
        else:
            raw = pd.read_parquet(path)
    else:
        raw = pd.read_parquet(path)

    bars = standardize_market_df(raw, path.stem)

    if bars.empty or len(bars) <= max(horizons):
        return pd.DataFrame()

    bars = bars.sort_values("date").reset_index(drop=True)

    for h in horizons:
        future_close = bars["close"].shift(-h)
        ret_col = f"T{h}_close_to_close_ret_pct"
        up_col = f"T{h}_is_up"

        bars[f"T{h}_future_close"] = future_close
        bars[ret_col] = (future_close / bars["close"] - 1.0) * 100.0
        bars[up_col] = bars[ret_col] > 0

    if start_date:
        bars = bars[bars["date"] >= pd.to_datetime(start_date).normalize()].copy()

    if end_date:
        bars = bars[bars["date"] <= pd.to_datetime(end_date).normalize()].copy()

    return bars.reset_index(drop=True)


def load_market_cache(
    market_cache_dir: Path,
    horizons: list[int],
    start_date: str | None,
    end_date: str | None,
    max_workers: int | None,
) -> pd.DataFrame:
    if not market_cache_dir.exists():
        raise FileNotFoundError(f"market cache dir not found: {market_cache_dir}")

    files = sorted(market_cache_dir.glob("*.parquet"))

    if not files:
        raise RuntimeError(f"no parquet files found: {market_cache_dir}")

    if max_workers is None:
        max_workers = min(12, max(2, (os.cpu_count() or 4) - 1))

    print(f"[INFO] Loading market cache: files={len(files):,}, max_workers={max_workers}")

    parts = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(read_one_market_file, path, horizons, start_date, end_date): path
            for path in files
        }

        iterator = as_completed(future_map)

        if tqdm is not None:
            iterator = tqdm(iterator, total=len(future_map), desc="Load market cache", ncols=100)

        for future in iterator:
            path = future_map[future]
            try:
                one = future.result()
                if not one.empty:
                    parts.append(one)
            except Exception as exc:
                print(f"[WARN] failed to read {path}: {exc}")

    if not parts:
        raise RuntimeError("market cache loaded empty")

    market = pd.concat(parts, ignore_index=True)
    market = market.drop_duplicates(subset=["date", "code"], keep="last")
    market = market.sort_values(["date", "code"]).reset_index(drop=True)

    return market


def attach_returns_to_pools(
    pools: pd.DataFrame,
    market: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    fwd_cols = ["date", "code"]

    for h in horizons:
        fwd_cols.extend(
            [
                f"T{h}_future_close",
                f"T{h}_close_to_close_ret_pct",
                f"T{h}_is_up",
            ]
        )

    fwd_cols = [c for c in fwd_cols if c in market.columns]
    m = market[fwd_cols].copy()

    out = pools.merge(
        m,
        on=["date", "code"],
        how="left",
        validate="many_to_one",
    )

    return out


def build_market_daily(market: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    rows = []

    for h in horizons:
        ret_col = f"T{h}_close_to_close_ret_pct"
        up_col = f"T{h}_is_up"

        valid = market[pd.to_numeric(market[ret_col], errors="coerce").notna()].copy()

        if valid.empty:
            continue

        g = valid.groupby("date", dropna=False)

        tmp = g.agg(
            market_count=("code", "nunique"),
            market_avg_return_pct=(ret_col, "mean"),
            market_median_return_pct=(ret_col, "median"),
            market_up_ratio=(up_col, "mean"),
        ).reset_index()

        tmp["horizon"] = f"T{h}"
        rows.append(tmp)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def build_pool_daily(
    pool_fwd: pd.DataFrame,
    market_daily: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    rows = []

    grouped = pool_fwd.groupby(["pool_name", "date"], dropna=False)

    for (pool_name, date), g in grouped:
        signal_count = int(g["code"].nunique())

        for h in horizons:
            ret_col = f"T{h}_close_to_close_ret_pct"
            up_col = f"T{h}_is_up"

            valid = g[pd.to_numeric(g[ret_col], errors="coerce").notna()].copy()
            valid_count = int(len(valid))

            if valid_count > 0:
                ret = pd.to_numeric(valid[ret_col], errors="coerce")
                up = valid[up_col].fillna(False).astype(bool)

                avg_return = float(ret.mean())
                median_return = float(ret.median())
                up_count = int(up.sum())
                up_ratio = float(up.mean())
            else:
                avg_return = np.nan
                median_return = np.nan
                up_count = 0
                up_ratio = np.nan

            rows.append(
                {
                    "pool_name": pool_name,
                    "date": date,
                    "horizon": f"T{h}",
                    "signal_count": signal_count,
                    "valid_count": valid_count,
                    "pool_avg_return_pct": avg_return,
                    "pool_median_return_pct": median_return,
                    "pool_up_count": up_count,
                    "pool_up_ratio": up_ratio,
                }
            )

    pool_daily = pd.DataFrame(rows)

    if pool_daily.empty:
        return pool_daily

    out = pool_daily.merge(
        market_daily,
        on=["date", "horizon"],
        how="left",
        validate="many_to_one",
    )

    out["excess_avg_return_pct"] = out["pool_avg_return_pct"] - out["market_avg_return_pct"]
    out["excess_median_return_pct"] = out["pool_median_return_pct"] - out["market_median_return_pct"]
    out["excess_up_ratio"] = out["pool_up_ratio"] - out["market_up_ratio"]

    out = out.sort_values(["date", "horizon", "pool_name"]).reset_index(drop=True)
    return out


def weighted_avg(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")

    mask = v.notna() & w.notna() & (w > 0)

    if not mask.any():
        return np.nan

    return float(np.average(v[mask], weights=w[mask]))


def safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(s.mean())


def safe_median(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(s.median())


def build_pool_summary(pool_daily: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (pool_name, horizon), g in pool_daily.groupby(["pool_name", "horizon"], dropna=False):
        valid_days = g[pd.to_numeric(g["pool_avg_return_pct"], errors="coerce").notna()].copy()

        total_valid_rows = int(valid_days["valid_count"].sum())
        total_up_rows = int(valid_days["pool_up_count"].sum())

        weighted_up_ratio = total_up_rows / total_valid_rows if total_valid_rows else np.nan
        weighted_avg_return = weighted_avg(valid_days["pool_avg_return_pct"], valid_days["valid_count"])
        weighted_market_return = weighted_avg(valid_days["market_avg_return_pct"], valid_days["market_count"])

        rows.append(
            {
                "pool_name": pool_name,
                "horizon": horizon,
                "trading_days": int(g["date"].nunique()),
                "valid_days": int(valid_days["date"].nunique()),
                "total_signal_rows": int(g["signal_count"].sum()),
                "total_valid_rows": total_valid_rows,
                "daily_mean_signal_count": safe_mean(g["signal_count"]),
                "daily_median_signal_count": safe_median(g["signal_count"]),

                "daily_mean_pool_return_pct": safe_mean(g["pool_avg_return_pct"]),
                "daily_median_pool_return_pct": safe_median(g["pool_avg_return_pct"]),
                "daily_mean_pool_up_ratio": safe_mean(g["pool_up_ratio"]),

                "daily_mean_market_return_pct": safe_mean(g["market_avg_return_pct"]),
                "daily_mean_market_up_ratio": safe_mean(g["market_up_ratio"]),

                "daily_mean_excess_return_pct": safe_mean(g["excess_avg_return_pct"]),
                "daily_median_excess_return_pct": safe_median(g["excess_avg_return_pct"]),
                "daily_mean_excess_up_ratio": safe_mean(g["excess_up_ratio"]),

                "positive_excess_return_days": int((g["excess_avg_return_pct"] > 0).sum()),
                "positive_excess_return_day_ratio": float((g["excess_avg_return_pct"] > 0).mean()) if len(g) else np.nan,

                "weighted_pool_avg_return_pct": weighted_avg_return,
                "weighted_market_avg_return_pct": weighted_market_return,
                "weighted_excess_avg_return_pct": weighted_avg_return - weighted_market_return
                if pd.notna(weighted_avg_return) and pd.notna(weighted_market_return)
                else np.nan,
                "weighted_pool_up_ratio": weighted_up_ratio,
            }
        )

    return pd.DataFrame(rows).sort_values(["horizon", "pool_name"]).reset_index(drop=True)


def build_pool_sets(pools: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], set[str]]:
    out: dict[tuple[str, pd.Timestamp], set[str]] = {}

    for (pool_name, date), g in pools.groupby(["pool_name", "date"], dropna=False):
        out[(str(pool_name), pd.Timestamp(date))] = set(g["code"].dropna().astype(str).tolist())

    return out


def build_daily_coverage(pools: pd.DataFrame, pool_names: list[str]) -> pd.DataFrame:
    sets = build_pool_sets(pools)
    dates = sorted(pools["date"].dropna().unique())

    rows = []

    for d in dates:
        d = pd.Timestamp(d)
        row: dict[str, Any] = {"date": d}
        date_sets = []

        for name in pool_names:
            s = sets.get((name, d), set())
            row[f"{name}_count"] = len(s)
            date_sets.append(s)

        active_sets = [s for s in date_sets if len(s) > 0]

        if active_sets:
            union_set = set().union(*active_sets)
        else:
            union_set = set()

        if len(active_sets) == len(pool_names):
            common_set = set.intersection(*active_sets)
        else:
            common_set = set()

        row["active_pool_num"] = len(active_sets)
        row["union_count"] = len(union_set)
        row["all_pool_common_count"] = len(common_set)
        row["all_pool_common_to_union_ratio"] = len(common_set) / len(union_set) if union_set else np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_pairwise_coverage_daily(pools: pd.DataFrame, pool_names: list[str]) -> pd.DataFrame:
    sets = build_pool_sets(pools)
    dates = sorted(pools["date"].dropna().unique())
    rows = []

    for a, b in itertools.combinations(pool_names, 2):
        for d in dates:
            d = pd.Timestamp(d)
            sa = sets.get((a, d), set())
            sb = sets.get((b, d), set())

            common = sa & sb
            union = sa | sb

            rows.append(
                {
                    "pair": f"{a} vs {b}",
                    "pool_a": a,
                    "pool_b": b,
                    "date": d,
                    "pool_a_count": len(sa),
                    "pool_b_count": len(sb),
                    "common_count": len(common),
                    "only_a_count": len(sa - sb),
                    "only_b_count": len(sb - sa),
                    "union_count": len(union),
                    "jaccard_ratio": len(common) / len(union) if union else np.nan,
                    "common_to_a_ratio": len(common) / len(sa) if sa else np.nan,
                    "common_to_b_ratio": len(common) / len(sb) if sb else np.nan,
                    "both_active": bool(len(sa) > 0 and len(sb) > 0),
                }
            )

    return pd.DataFrame(rows).sort_values(["date", "pair"]).reset_index(drop=True)


def build_pairwise_daily_compare(
    pool_daily: pd.DataFrame,
    pair_coverage: pd.DataFrame,
    pool_names: list[str],
    horizons: list[int],
) -> pd.DataFrame:
    daily_map = {}

    for _, r in pool_daily.iterrows():
        key = (str(r["pool_name"]), pd.Timestamp(r["date"]), str(r["horizon"]))
        daily_map[key] = r

    rows = []

    for a, b in itertools.combinations(pool_names, 2):
        cov = pair_coverage[(pair_coverage["pool_a"] == a) & (pair_coverage["pool_b"] == b)].copy()

        for _, c in cov.iterrows():
            d = pd.Timestamp(c["date"])

            for h in horizons:
                horizon = f"T{h}"

                ra = daily_map.get((a, d, horizon))
                rb = daily_map.get((b, d, horizon))

                a_ret = ra["pool_avg_return_pct"] if ra is not None else np.nan
                b_ret = rb["pool_avg_return_pct"] if rb is not None else np.nan
                a_up = ra["pool_up_ratio"] if ra is not None else np.nan
                b_up = rb["pool_up_ratio"] if rb is not None else np.nan
                a_excess = ra["excess_avg_return_pct"] if ra is not None else np.nan
                b_excess = rb["excess_avg_return_pct"] if rb is not None else np.nan

                rows.append(
                    {
                        "pair": f"{a} vs {b}",
                        "pool_a": a,
                        "pool_b": b,
                        "date": d,
                        "horizon": horizon,

                        "pool_a_count": c["pool_a_count"],
                        "pool_b_count": c["pool_b_count"],
                        "common_count": c["common_count"],
                        "only_a_count": c["only_a_count"],
                        "only_b_count": c["only_b_count"],
                        "union_count": c["union_count"],
                        "jaccard_ratio": c["jaccard_ratio"],

                        "pool_a_avg_return_pct": a_ret,
                        "pool_b_avg_return_pct": b_ret,
                        "a_minus_b_avg_return_pct": a_ret - b_ret
                        if pd.notna(a_ret) and pd.notna(b_ret)
                        else np.nan,

                        "pool_a_up_ratio": a_up,
                        "pool_b_up_ratio": b_up,
                        "a_minus_b_up_ratio": a_up - b_up
                        if pd.notna(a_up) and pd.notna(b_up)
                        else np.nan,

                        "pool_a_excess_avg_return_pct": a_excess,
                        "pool_b_excess_avg_return_pct": b_excess,
                        "a_minus_b_excess_avg_return_pct": a_excess - b_excess
                        if pd.notna(a_excess) and pd.notna(b_excess)
                        else np.nan,

                        "both_active": c["both_active"],
                    }
                )

    return pd.DataFrame(rows).sort_values(["date", "horizon", "pair"]).reset_index(drop=True)


def build_pairwise_summary(pairwise_daily: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (pair, horizon), g in pairwise_daily.groupby(["pair", "horizon"], dropna=False):
        valid = g[pd.to_numeric(g["a_minus_b_avg_return_pct"], errors="coerce").notna()].copy()

        rows.append(
            {
                "pair": pair,
                "horizon": horizon,
                "compare_days": int(valid["date"].nunique()),
                "both_active_days": int(g[g["both_active"]]["date"].nunique()),

                "pool_a_mean_count": safe_mean(valid["pool_a_count"]),
                "pool_b_mean_count": safe_mean(valid["pool_b_count"]),
                "mean_common_count": safe_mean(valid["common_count"]),
                "mean_jaccard_ratio": safe_mean(valid["jaccard_ratio"]),

                "pool_a_mean_return_pct": safe_mean(valid["pool_a_avg_return_pct"]),
                "pool_b_mean_return_pct": safe_mean(valid["pool_b_avg_return_pct"]),
                "a_minus_b_mean_return_pct": safe_mean(valid["a_minus_b_avg_return_pct"]),
                "a_minus_b_median_return_pct": safe_median(valid["a_minus_b_avg_return_pct"]),

                "pool_a_mean_up_ratio": safe_mean(valid["pool_a_up_ratio"]),
                "pool_b_mean_up_ratio": safe_mean(valid["pool_b_up_ratio"]),
                "a_minus_b_mean_up_ratio": safe_mean(valid["a_minus_b_up_ratio"]),

                "pool_a_mean_excess_pct": safe_mean(valid["pool_a_excess_avg_return_pct"]),
                "pool_b_mean_excess_pct": safe_mean(valid["pool_b_excess_avg_return_pct"]),
                "a_minus_b_mean_excess_pct": safe_mean(valid["a_minus_b_excess_avg_return_pct"]),

                "a_win_days": int((valid["a_minus_b_avg_return_pct"] > 0).sum()),
                "b_win_days": int((valid["a_minus_b_avg_return_pct"] < 0).sum()),
                "tie_days": int((valid["a_minus_b_avg_return_pct"] == 0).sum()),
                "a_win_ratio": float((valid["a_minus_b_avg_return_pct"] > 0).mean()) if len(valid) else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values(["horizon", "pair"]).reset_index(drop=True)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Reusable N-pool comparison analysis. Supports 2-5 pools.")

    parser.add_argument(
        "--pool-paths",
        required=True,
        help="Pool paths separated by semicolon. Example: path1.parquet;path2.parquet",
    )
    parser.add_argument(
        "--pool-names",
        default="",
        help="Pool names separated by semicolon. Must match pool-paths count if provided.",
    )
    parser.add_argument("--market-cache-dir", type=Path, default=DEFAULT_MARKET_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--horizons", default="1,2,3")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--selected-only", action="store_true", default=True)
    parser.add_argument("--no-selected-only", action="store_false", dest="selected_only")

    args = parser.parse_args()

    pool_paths = [Path(x) for x in parse_list(args.pool_paths)]
    pool_names = parse_list(args.pool_names)

    if not (2 <= len(pool_paths) <= 5):
        raise ValueError(f"pool count must be 2-5, got {len(pool_paths)}")

    if pool_names and len(pool_names) != len(pool_paths):
        raise ValueError("pool-names count must match pool-paths count")

    if not pool_names:
        pool_names = [p.stem for p in pool_paths]

    if len(set(pool_names)) != len(pool_names):
        raise ValueError(f"pool names must be unique: {pool_names}")

    horizons = parse_horizons(args.horizons)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("N Pool Compare Analysis")
    print("=" * 100)
    print(f"pool_names       : {pool_names}")
    print(f"pool_paths       : {[str(p) for p in pool_paths]}")
    print(f"market_cache_dir : {args.market_cache_dir}")
    print(f"output_dir       : {args.output_dir}")
    print(f"date_range       : {args.start_date} -> {args.end_date}")
    print(f"horizons         : {horizons}")
    print(f"selected_only    : {args.selected_only}")
    print("=" * 100)

    pool_parts = []

    for path, name in zip(pool_paths, pool_names):
        one = load_pool(
            pool_path=path,
            pool_name=name,
            selected_only=args.selected_only,
            start_date=args.start_date,
            end_date=args.end_date,
        )

        print(f"[INFO] loaded {name}: rows={len(one):,}, days={one['date'].nunique():,}, symbols={one['code'].nunique():,}")
        pool_parts.append(one)

    pools = pd.concat(pool_parts, ignore_index=True)

    market = load_market_cache(
        market_cache_dir=args.market_cache_dir,
        horizons=horizons,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
    )

    pool_fwd = attach_returns_to_pools(pools, market, horizons)
    market_daily = build_market_daily(market, horizons)
    pool_daily = build_pool_daily(pool_fwd, market_daily, horizons)
    pool_summary = build_pool_summary(pool_daily)

    daily_coverage = build_daily_coverage(pools, pool_names)
    pair_coverage = build_pairwise_coverage_daily(pools, pool_names)
    pairwise_daily = build_pairwise_daily_compare(pool_daily, pair_coverage, pool_names, horizons)
    pairwise_summary = build_pairwise_summary(pairwise_daily)

    out_paths = {
        "pool_rows_with_returns": args.output_dir / "0_pool_rows_with_returns.parquet",
        "pool_daily_metrics": args.output_dir / "1_pool_daily_metrics.csv",
        "pool_summary": args.output_dir / "2_pool_summary.csv",
        "pairwise_daily_compare": args.output_dir / "3_pairwise_daily_compare.csv",
        "pairwise_summary": args.output_dir / "4_pairwise_summary.csv",
        "daily_coverage": args.output_dir / "5_daily_coverage.csv",
        "pairwise_coverage_daily": args.output_dir / "6_pairwise_coverage_daily.csv",
        "diagnostics": args.output_dir / "diagnostics.json",
    }

    pool_fwd.to_parquet(out_paths["pool_rows_with_returns"], index=False)
    pool_daily.to_csv(out_paths["pool_daily_metrics"], index=False, encoding="utf-8-sig")
    pool_summary.to_csv(out_paths["pool_summary"], index=False, encoding="utf-8-sig")
    pairwise_daily.to_csv(out_paths["pairwise_daily_compare"], index=False, encoding="utf-8-sig")
    pairwise_summary.to_csv(out_paths["pairwise_summary"], index=False, encoding="utf-8-sig")
    daily_coverage.to_csv(out_paths["daily_coverage"], index=False, encoding="utf-8-sig")
    pair_coverage.to_csv(out_paths["pairwise_coverage_daily"], index=False, encoding="utf-8-sig")

    diagnostics = {
        "pool_names": pool_names,
        "pool_paths": [str(p) for p in pool_paths],
        "market_cache_dir": str(args.market_cache_dir),
        "output_dir": str(args.output_dir),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "horizons": horizons,
        "selected_only": args.selected_only,
        "pool_rows": int(len(pools)),
        "pool_trading_days": int(pools["date"].nunique()),
        "market_rows": int(len(market)),
        "market_symbols": int(market["code"].nunique()),
        "outputs": {k: str(v) for k, v in out_paths.items()},
    }

    save_json(out_paths["diagnostics"], diagnostics)

    print("\nDONE")
    for k, v in out_paths.items():
        print(f"{k}: {v}")

    print("\nPool summary:")
    print(pool_summary.to_string(index=False))

    print("\nPairwise summary:")
    print(pairwise_summary.to_string(index=False))


if __name__ == "__main__":
    main()