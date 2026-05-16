# -*- coding: utf-8 -*-
from __future__ import annotations

"""
individual_alpha_probe.py

作用：
验证“每天固定 Top10 个股选择信号”是否有横截面选股能力。

当前正式打分模型：
低波动 + 10日冷却 + MA20回踩。

当前正式参与 alpha_score 的因子：
rank_volatility_contract
rank_cooldown_10d
rank_pullback_ma20

当前额外验证因子：
rank_cooldown_3d
rank_cooldown_5d

注意：
cooldown_3d / cooldown_5d 目前只进入 FeatureIC 验证，不参与 alpha_score。
"""

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.path_manager import DATA_ROOT, INDICATOR_CACHE_PATH, MARKET_CACHE_DIR, POOLS_DIR, OUTPUT_DIR

DEFAULT_DATA_ROOT = DATA_ROOT
DEFAULT_INDICATOR_CACHE_PATH = INDICATOR_CACHE_PATH
DEFAULT_MARKET_CACHE_DIR = MARKET_CACHE_DIR
DEFAULT_POOLS_DIR = POOLS_DIR
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "individual_alpha_probe"

DEFAULT_FORMAL_POOL_PATHS = [
    DEFAULT_POOLS_DIR / "b2_confirm_select_strategy_v0_pool.parquet",
    DEFAULT_POOLS_DIR / "renko_chart_select_strategy_v4_pool.parquet",
]


def normalize_symbol(value) -> str:
    if pd.isna(value):
        return ""

    s = str(value).strip().upper()
    m = re.search(r"(\d{6})", s)
    if m:
        return m.group(1)

    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 6:
        return digits[-6:]
    if digits:
        return digits.zfill(6)

    return ""


def parse_horizons(value: str) -> list[int]:
    out = []

    for part in str(value).replace(";", ",").split(","):
        part = part.strip().upper().replace("T", "")
        if not part:
            continue

        n = int(part)
        if n <= 0:
            raise ValueError(f"Invalid horizon: {part}")

        out.append(n)

    if not out:
        raise ValueError("No horizons supplied.")

    return sorted(set(out))


def read_table(path: Path) -> pd.DataFrame:
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

    raise ValueError(f"Unsupported file type: {path}")


def symbol_from_market_file(path: Path) -> str:
    return normalize_symbol(path.stem)


def month_str(x) -> str:
    return pd.Timestamp(x).strftime("%Y-%m")


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


def cleanup_old_outputs(output_dir: Path) -> None:
    names = [
        "individual_alpha_top10.csv",
        "individual_alpha_daily.csv",
        "individual_alpha_monthly.csv",
        "individual_alpha_summary.csv",
        "individual_alpha_rank_ic.csv",
        "individual_alpha_feature_ic.csv",
    ]

    for name in names:
        path = output_dir / name
        if path.exists():
            path.unlink()


def standardize_market_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    rename_map = {
        "日期": "date",
        "时间": "date",
        "trade_date": "date",
        "datetime": "date",
        "交易日期": "date",
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

    out = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map}).copy()

    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].map(normalize_symbol)
    elif "code" in out.columns:
        out["symbol"] = out["code"].map(normalize_symbol)
    else:
        out["symbol"] = symbol

    out.loc[out["symbol"].eq(""), "symbol"] = symbol

    required = {"date", "open", "high", "low", "close"}
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"{symbol} missing columns: {sorted(missing)}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "volume" not in out.columns:
        out["volume"] = np.nan

    if "amount" not in out.columns:
        out["amount"] = np.nan

    out = out.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    out = out[out["symbol"] != ""].copy()
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    return out[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]].copy()


def add_symbol_features(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)

    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce")

    out["trade_index"] = np.arange(len(out)) + 1

    out["ret_1d_pct"] = close.pct_change(1) * 100.0
    out["ret_3d_pct"] = close.pct_change(3) * 100.0
    out["ret_5d_pct"] = close.pct_change(5) * 100.0
    out["ret_10d_pct"] = close.pct_change(10) * 100.0
    out["ret_20d_pct"] = close.pct_change(20) * 100.0

    out["ma5"] = close.rolling(5, min_periods=5).mean()
    out["ma10"] = close.rolling(10, min_periods=10).mean()
    out["ma20"] = close.rolling(20, min_periods=20).mean()
    out["ma60"] = close.rolling(60, min_periods=60).mean()

    out["close_to_ma5"] = close / out["ma5"]
    out["close_to_ma10"] = close / out["ma10"]
    out["close_to_ma20"] = close / out["ma20"]
    out["close_to_ma60"] = close / out["ma60"]

    out["high20"] = high.rolling(20, min_periods=20).max()
    out["low20"] = low.rolling(20, min_periods=20).min()
    out["close_pos20"] = (close - out["low20"]) / (out["high20"] - out["low20"]).replace(0, np.nan)

    out["volume_ma20"] = volume.rolling(20, min_periods=20).mean()
    out["volume_ratio20"] = volume / out["volume_ma20"]

    out["volatility20"] = out["ret_1d_pct"].rolling(20, min_periods=20).std()

    for h in horizons:
        out[f"future_close_T{h}"] = close.shift(-h)
        out[f"ret_T{h}"] = close.shift(-h) / close - 1.0
        out[f"ret_T{h}_pct"] = out[f"ret_T{h}"] * 100.0
        out[f"up_T{h}"] = out[f"ret_T{h}"] > 0

    return out


def load_market_panel(
    market_cache_dir: Path,
    horizons: list[int],
    *,
    only_main_board: bool,
) -> pd.DataFrame:
    if not market_cache_dir.exists():
        raise FileNotFoundError(f"Market cache dir not found: {market_cache_dir}")

    files = sorted(market_cache_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found under: {market_cache_dir}")

    parts = []

    for path in tqdm(files, desc="Load market cache"):
        symbol = symbol_from_market_file(path)

        if not symbol:
            continue

        if only_main_board and not symbol.startswith(("00", "60")):
            continue

        try:
            raw = pd.read_parquet(path)
            one = standardize_market_df(raw, symbol)

            if len(one) < max(80, max(horizons) + 30):
                continue

            one = add_symbol_features(one, horizons)
            parts.append(one)

        except Exception as exc:
            print(f"[WARN] skip {path.name}: {exc}")

    if not parts:
        raise RuntimeError("No usable market cache rows loaded.")

    panel = pd.concat(parts, ignore_index=True)
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)

    market_ret = (
        panel.groupby("date", as_index=False)["ret_1d_pct"]
        .mean()
        .rename(columns={"ret_1d_pct": "market_ret_1d_pct"})
    )

    panel = panel.merge(market_ret, on="date", how="left")
    panel["excess_1d_pct"] = panel["ret_1d_pct"] - panel["market_ret_1d_pct"]

    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    panel["excess_5d_pct"] = panel.groupby("symbol")["excess_1d_pct"].transform(
        lambda s: s.rolling(5, min_periods=5).sum()
    )
    panel["excess_20d_pct"] = panel.groupby("symbol")["excess_1d_pct"].transform(
        lambda s: s.rolling(20, min_periods=20).sum()
    )

    return panel


def read_indicator_subset(indicator_cache_path: Path) -> pd.DataFrame:
    if not indicator_cache_path.exists():
        print(f"[WARN] indicator cache not found, skip merge: {indicator_cache_path}")
        return pd.DataFrame()

    wanted = [
        "date",
        "symbol",
        "code",
        "name",
        "stock_name",
        "daily_return_pct",
        "close_to_short_trend",
        "brick_reversal_ratio",
        "short_pos_3",
        "long_pos_21",
        "short_pos",
        "long_pos",
    ]

    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(indicator_cache_path)
        available = set(pf.schema.names)
        cols = [c for c in wanted if c in available]

        if "date" not in cols:
            print("[WARN] indicator cache has no date column, skip merge.")
            return pd.DataFrame()

        if not any(c in cols for c in ["symbol", "code"]):
            print("[WARN] indicator cache has no symbol/code column, skip merge.")
            return pd.DataFrame()

        df = pd.read_parquet(indicator_cache_path, columns=cols)

    except Exception:
        df = pd.read_parquet(indicator_cache_path)
        cols = [c for c in wanted if c in df.columns]
        df = df[cols].copy()

    if df.empty:
        return df

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].map(normalize_symbol)
    elif "code" in out.columns:
        out["symbol"] = out["code"].map(normalize_symbol)

    out = out.dropna(subset=["date"]).copy()
    out = out[out["symbol"] != ""].copy()
    out = out.drop_duplicates(subset=["date", "symbol"], keep="last").reset_index(drop=True)

    return out


def merge_indicator_cache(panel: pd.DataFrame, indicator_cache_path: Path) -> pd.DataFrame:
    ind = read_indicator_subset(indicator_cache_path)

    if ind.empty:
        return panel

    duplicate_cols = [c for c in ind.columns if c in panel.columns and c not in {"date", "symbol"}]
    ind = ind.rename(columns={c: f"{c}_ind" for c in duplicate_cols})

    out = panel.merge(ind, on=["date", "symbol"], how="left")

    if "daily_return_pct_ind" in out.columns:
        out["daily_return_pct_project"] = pd.to_numeric(out["daily_return_pct_ind"], errors="coerce")
    else:
        out["daily_return_pct_project"] = out["ret_1d_pct"]

    return out


def pct_rank_by_date(df: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = out.groupby("date")[value_col].rank(pct=True, method="average")
    return out


def add_alpha_score(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()

    close_to_ma20 = pd.to_numeric(out["close_to_ma20"], errors="coerce")
    volatility20 = pd.to_numeric(out["volatility20"], errors="coerce")

    out["raw_volatility_contract"] = -volatility20
    out["raw_cooldown_10d"] = -pd.to_numeric(out["ret_10d_pct"], errors="coerce")
    out["raw_pullback_ma20"] = -np.abs(close_to_ma20 - 1.03)

    feature_specs: list[tuple[str, str, float]] = [
        ("raw_volatility_contract", "rank_volatility_contract", 3.00),
        ("raw_pullback_ma20", "rank_pullback_ma20", 1.20),
        ("raw_cooldown_10d", "rank_cooldown_10d", 0.30),
    ]

    rank_cols = []

    for raw_col, rank_col, _weight in feature_specs:
        out = pct_rank_by_date(out, raw_col, rank_col)
        rank_cols.append(rank_col)

    out["raw_cooldown_3d"] = -pd.to_numeric(out["ret_3d_pct"], errors="coerce")
    out["raw_cooldown_5d"] = -pd.to_numeric(out["ret_5d_pct"], errors="coerce")

    probe_specs: list[tuple[str, str]] = [
        ("raw_cooldown_3d", "rank_cooldown_3d"),
        ("raw_cooldown_5d", "rank_cooldown_5d"),
    ]

    for raw_col, rank_col in probe_specs:
        out = pct_rank_by_date(out, raw_col, rank_col)
        rank_cols.append(rank_col)

    out["alpha_score"] = 0.0
    out["alpha_score_weight"] = 0.0

    for _raw_col, rank_col, weight in feature_specs:
        valid = pd.to_numeric(out[rank_col], errors="coerce")
        out["alpha_score"] += valid.fillna(0.0) * weight
        out["alpha_score_weight"] += valid.notna().astype(float) * weight

    out["alpha_score"] = np.where(
        out["alpha_score_weight"] > 0,
        out["alpha_score"] / out["alpha_score_weight"] * 100.0,
        np.nan,
    )

    return out, rank_cols


def apply_universe_filter(
    df: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
    min_price: float,
    max_price: float,
    min_history_days: int,
    only_main_board: bool,
    exclude_st: bool,
) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    if start_date:
        start_ts = pd.to_datetime(start_date).normalize()
        out = out[out["date"] >= start_ts].copy()

    if end_date:
        end_ts = pd.to_datetime(end_date).normalize()
        out = out[out["date"] <= end_ts].copy()

    if only_main_board:
        out = out[out["symbol"].str.startswith(("00", "60"))].copy()

    out = out[pd.to_numeric(out["close"], errors="coerce").between(min_price, max_price)].copy()
    out = out[pd.to_numeric(out["trade_index"], errors="coerce") >= min_history_days].copy()

    if exclude_st:
        name_col = None

        for c in ["name", "name_ind", "stock_name", "stock_name_ind"]:
            if c in out.columns:
                name_col = c
                break

        if name_col:
            out = out[
                ~out[name_col].astype(str).str.contains("ST", case=False, na=False)
            ].copy()

    out = out[out["alpha_score"].notna()].copy()

    return out


def select_daily_topn(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    out = df.copy()

    out = out.sort_values(
        ["date", "alpha_score", "symbol"],
        ascending=[True, False, True],
        na_position="last",
    ).copy()

    out["alpha_rank"] = out.groupby("date").cumcount() + 1

    top = out[out["alpha_rank"] <= top_n].copy()
    return top.reset_index(drop=True)


def load_formal_pools(
    pool_paths: Iterable[Path],
    *,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    parts = []

    for path in pool_paths:
        path = Path(path)

        if not path.exists():
            print(f"[WARN] formal pool not found, skip: {path}")
            continue

        try:
            df = read_table(path)
        except Exception as exc:
            print(f"[WARN] failed to read formal pool {path}: {exc}")
            continue

        if df.empty or "date" not in df.columns:
            continue

        if "symbol" in df.columns:
            sym_col = "symbol"
        elif "code" in df.columns:
            sym_col = "code"
        else:
            continue

        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        out["symbol"] = out[sym_col].map(normalize_symbol)

        out = out.dropna(subset=["date"]).copy()
        out = out[out["symbol"] != ""].copy()

        if "selected" in out.columns:
            selected_num = pd.to_numeric(out["selected"], errors="coerce").fillna(0)
            out = out[selected_num > 0].copy()

        if start_date:
            out = out[out["date"] >= pd.to_datetime(start_date).normalize()].copy()

        if end_date:
            out = out[out["date"] <= pd.to_datetime(end_date).normalize()].copy()

        if out.empty:
            continue

        out["formal_pool_name"] = path.stem.replace("_pool", "")
        parts.append(out[["date", "symbol", "formal_pool_name"]].drop_duplicates())

    if not parts:
        return pd.DataFrame(columns=["date", "symbol", "formal_pool_name"])

    formal = pd.concat(parts, ignore_index=True).drop_duplicates()
    return formal


def summarize_return_group(df: pd.DataFrame, ret_col: str, up_col: str) -> dict:
    valid = df.dropna(subset=[ret_col]).copy()
    count = int(len(valid))

    if count == 0:
        return {
            "count": 0,
            "up_count": 0,
            "up_ratio": np.nan,
            "avg_return_pct": np.nan,
            "median_return_pct": np.nan,
        }

    up = valid[up_col].fillna(False).astype(bool)

    return {
        "count": count,
        "up_count": int(up.sum()),
        "up_ratio": float(up.mean()),
        "avg_return_pct": float(valid[ret_col].mean() * 100.0),
        "median_return_pct": float(valid[ret_col].median() * 100.0),
    }


def random_topn_stats(
    returns: pd.Series,
    *,
    top_n: int,
    trials: int,
    rng: np.random.Generator,
) -> dict:
    vals = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)

    if len(vals) == 0:
        return {
            "random_count": 0,
            "random_avg_return_pct": np.nan,
            "random_std_return_pct": np.nan,
        }

    sample_n = min(top_n, len(vals))
    means = []

    for _ in range(trials):
        sample = rng.choice(vals, size=sample_n, replace=False)
        means.append(float(np.mean(sample) * 100.0))

    return {
        "random_count": int(sample_n),
        "random_avg_return_pct": float(np.mean(means)),
        "random_std_return_pct": float(np.std(means)),
    }


def build_daily_stats(
    scored: pd.DataFrame,
    top: pd.DataFrame,
    formal: pd.DataFrame,
    *,
    horizons: list[int],
    top_n: int,
    random_trials: int,
    seed: int,
) -> pd.DataFrame:
    market = scored.copy()

    top_keys = top[["date", "symbol"]].drop_duplicates().copy()
    top_keys["in_alpha_top"] = 1

    market = market.merge(top_keys, on=["date", "symbol"], how="left")
    market["in_alpha_top"] = market["in_alpha_top"].fillna(0).astype(int)

    if formal.empty:
        market["in_formal_union"] = 0
    else:
        formal_union = formal[["date", "symbol"]].drop_duplicates().copy()
        formal_union["in_formal_union"] = 1

        market = market.merge(formal_union, on=["date", "symbol"], how="left")
        market["in_formal_union"] = market["in_formal_union"].fillna(0).astype(int)

    rows = []
    rng = np.random.default_rng(seed)

    for date, g in tqdm(market.groupby("date", sort=True), desc="Daily stats"):
        top_g = g[g["in_alpha_top"] == 1]
        formal_g = g[g["in_formal_union"] == 1]

        for h in horizons:
            ret_col = f"ret_T{h}"
            up_col = f"up_T{h}"

            top_s = summarize_return_group(top_g, ret_col, up_col)
            market_s = summarize_return_group(g, ret_col, up_col)
            formal_s = summarize_return_group(formal_g, ret_col, up_col)
            random_s = random_topn_stats(
                g[ret_col],
                top_n=top_n,
                trials=random_trials,
                rng=rng,
            )

            row = {
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "month": month_str(date),
                "horizon": f"T{h}",

                "alpha_top_count": top_s["count"],
                "alpha_top_up_count": top_s["up_count"],
                "alpha_top_up_ratio": top_s["up_ratio"],
                "alpha_top_avg_return_pct": top_s["avg_return_pct"],
                "alpha_top_median_return_pct": top_s["median_return_pct"],

                "market_count": market_s["count"],
                "market_up_count": market_s["up_count"],
                "market_up_ratio": market_s["up_ratio"],
                "market_avg_return_pct": market_s["avg_return_pct"],
                "market_median_return_pct": market_s["median_return_pct"],

                "random_count": random_s["random_count"],
                "random_avg_return_pct": random_s["random_avg_return_pct"],
                "random_std_return_pct": random_s["random_std_return_pct"],

                "formal_union_count": formal_s["count"],
                "formal_union_up_count": formal_s["up_count"],
                "formal_union_up_ratio": formal_s["up_ratio"],
                "formal_union_avg_return_pct": formal_s["avg_return_pct"],
                "formal_union_median_return_pct": formal_s["median_return_pct"],
            }

            row["excess_vs_market_pct"] = (
                row["alpha_top_avg_return_pct"] - row["market_avg_return_pct"]
                if pd.notna(row["alpha_top_avg_return_pct"]) and pd.notna(row["market_avg_return_pct"])
                else np.nan
            )
            row["excess_vs_random_pct"] = (
                row["alpha_top_avg_return_pct"] - row["random_avg_return_pct"]
                if pd.notna(row["alpha_top_avg_return_pct"]) and pd.notna(row["random_avg_return_pct"])
                else np.nan
            )
            row["excess_vs_formal_union_pct"] = (
                row["alpha_top_avg_return_pct"] - row["formal_union_avg_return_pct"]
                if pd.notna(row["alpha_top_avg_return_pct"]) and pd.notna(row["formal_union_avg_return_pct"])
                else np.nan
            )
            row["excess_up_ratio_vs_market"] = (
                row["alpha_top_up_ratio"] - row["market_up_ratio"]
                if pd.notna(row["alpha_top_up_ratio"]) and pd.notna(row["market_up_ratio"])
                else np.nan
            )

            rows.append(row)

    return pd.DataFrame(rows)


def build_monthly_stats(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (month, horizon), g in daily.groupby(["month", "horizon"], sort=True):
        valid = g.dropna(subset=["excess_vs_market_pct"]).copy()

        rows.append({
            "month": month,
            "horizon": horizon,
            "trading_days": int(len(g)),

            "alpha_top_avg_return_pct": safe_mean(g["alpha_top_avg_return_pct"]),
            "market_avg_return_pct": safe_mean(g["market_avg_return_pct"]),
            "random_avg_return_pct": safe_mean(g["random_avg_return_pct"]),
            "formal_union_avg_return_pct": safe_mean(g["formal_union_avg_return_pct"]),

            "excess_vs_market_pct": safe_mean(g["excess_vs_market_pct"]),
            "excess_vs_random_pct": safe_mean(g["excess_vs_random_pct"]),
            "excess_vs_formal_union_pct": safe_mean(g["excess_vs_formal_union_pct"]),

            "positive_excess_days": int((valid["excess_vs_market_pct"] > 0).sum()) if not valid.empty else 0,
            "valid_excess_days": int(len(valid)),
            "positive_excess_day_ratio": (
                float((valid["excess_vs_market_pct"] > 0).mean())
                if not valid.empty
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


def build_rank_ic(
    scored: pd.DataFrame,
    rank_cols: list[str],
    *,
    horizons: list[int],
    min_ic_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_rows = []
    feature_rows = []

    ic_cols = ["alpha_score"] + rank_cols

    for date, g in tqdm(scored.groupby("date", sort=True), desc="RankIC"):
        for h in horizons:
            ret_col = f"ret_T{h}"
            base = g[["alpha_score", ret_col] + rank_cols].copy()
            base = base.dropna(subset=["alpha_score", ret_col])

            if len(base) < min_ic_count:
                rank_rows.append({
                    "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "month": month_str(date),
                    "horizon": f"T{h}",
                    "count": int(len(base)),
                    "rank_ic": np.nan,
                })
                continue

            rank_ic = base["alpha_score"].corr(base[ret_col], method="spearman")

            rank_rows.append({
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "month": month_str(date),
                "horizon": f"T{h}",
                "count": int(len(base)),
                "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
            })

            for c in ic_cols:
                valid = base[[c, ret_col]].dropna()

                if len(valid) < min_ic_count:
                    ic = np.nan
                else:
                    ic = valid[c].corr(valid[ret_col], method="spearman")

                feature_rows.append({
                    "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "month": month_str(date),
                    "horizon": f"T{h}",
                    "feature": c,
                    "count": int(len(valid)),
                    "rank_ic": float(ic) if pd.notna(ic) else np.nan,
                })

    return pd.DataFrame(rank_rows), pd.DataFrame(feature_rows)


def build_summary(daily: pd.DataFrame, monthly: pd.DataFrame, rank_ic: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for horizon, g in daily.groupby("horizon", sort=True):
        ic_g = rank_ic[rank_ic["horizon"] == horizon].copy()
        m_g = monthly[monthly["horizon"] == horizon].copy()

        valid_excess = g["excess_vs_market_pct"].dropna()
        valid_month_excess = m_g["excess_vs_market_pct"].dropna()
        valid_ic = ic_g["rank_ic"].dropna()

        rows.append({
            "horizon": horizon,

            "daily_rows": int(len(g)),
            "avg_alpha_top_return_pct": safe_mean(g["alpha_top_avg_return_pct"]),
            "avg_market_return_pct": safe_mean(g["market_avg_return_pct"]),
            "avg_random_return_pct": safe_mean(g["random_avg_return_pct"]),
            "avg_formal_union_return_pct": safe_mean(g["formal_union_avg_return_pct"]),

            "avg_excess_vs_market_pct": safe_mean(g["excess_vs_market_pct"]),
            "avg_excess_vs_random_pct": safe_mean(g["excess_vs_random_pct"]),
            "avg_excess_vs_formal_union_pct": safe_mean(g["excess_vs_formal_union_pct"]),

            "positive_excess_day_ratio": (
                float((valid_excess > 0).mean()) if not valid_excess.empty else np.nan
            ),
            "positive_excess_month_ratio": (
                float((valid_month_excess > 0).mean()) if not valid_month_excess.empty else np.nan
            ),

            "avg_rank_ic": safe_mean(valid_ic),
            "median_rank_ic": safe_median(valid_ic),
            "positive_rank_ic_ratio": (
                float((valid_ic > 0).mean()) if not valid_ic.empty else np.nan
            ),
        })

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe individual stock alpha selection ability.")

    parser.add_argument("--indicator-cache-path", type=Path, default=DEFAULT_INDICATOR_CACHE_PATH)
    parser.add_argument("--market-cache-dir", type=Path, default=DEFAULT_MARKET_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default=None)

    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--horizons", default="1,2,3")

    parser.add_argument("--min-price", type=float, default=3.0)
    parser.add_argument("--max-price", type=float, default=80.0)
    parser.add_argument("--min-history-days", type=int, default=80)

    parser.add_argument("--random-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--min-ic-count", type=int, default=100)

    parser.add_argument("--include-non-main-board", action="store_true")
    parser.add_argument("--include-st", action="store_true")
    parser.add_argument("--no-indicator-cache", action="store_true")

    parser.add_argument(
        "--formal-pool-paths",
        nargs="*",
        type=Path,
        default=DEFAULT_FORMAL_POOL_PATHS,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    horizons = parse_horizons(args.horizons)
    only_main_board = not args.include_non_main_board
    exclude_st = not args.include_st

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_old_outputs(args.output_dir)

    print("=" * 100)
    print("Individual Alpha Probe")
    print("=" * 100)
    print("market_cache_dir      :", args.market_cache_dir)
    print("indicator_cache_path  :", args.indicator_cache_path)
    print("output_dir            :", args.output_dir)
    print("start_date            :", args.start_date)
    print("end_date              :", args.end_date)
    print("top_n                 :", args.top_n)
    print("horizons              :", horizons)
    print("model                 : low_volatility_cooldown_pullback")
    print("score factors         : volatility_contract / cooldown_10d / pullback_ma20")
    print("probe factors         : cooldown_3d / cooldown_5d")
    print("=" * 100)

    panel = load_market_panel(
        args.market_cache_dir,
        horizons,
        only_main_board=only_main_board,
    )

    if not args.no_indicator_cache:
        panel = merge_indicator_cache(panel, args.indicator_cache_path)

    scored, rank_cols = add_alpha_score(panel)

    scored = apply_universe_filter(
        scored,
        start_date=args.start_date,
        end_date=args.end_date,
        min_price=args.min_price,
        max_price=args.max_price,
        min_history_days=args.min_history_days,
        only_main_board=only_main_board,
        exclude_st=exclude_st,
    )

    if scored.empty:
        raise RuntimeError("No scored rows after universe filter.")

    print("[CHECK] scored date range:", scored["date"].min(), "->", scored["date"].max())
    print("[CHECK] scored rows:", len(scored))
    print("[CHECK] start_date arg:", args.start_date)
    print("[CHECK] end_date arg:", args.end_date)

    if args.start_date:
        min_allowed = pd.to_datetime(args.start_date).normalize()
        actual_min = pd.to_datetime(scored["date"].min()).normalize()
        if actual_min < min_allowed:
            raise RuntimeError(
                f"Date filter failed: actual_min={actual_min}, start_date={min_allowed}"
            )

    if args.end_date:
        max_allowed = pd.to_datetime(args.end_date).normalize()
        actual_max = pd.to_datetime(scored["date"].max()).normalize()
        if actual_max > max_allowed:
            raise RuntimeError(
                f"Date filter failed: actual_max={actual_max}, end_date={max_allowed}"
            )

    top = select_daily_topn(scored, args.top_n)

    formal = load_formal_pools(
        args.formal_pool_paths,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    daily = build_daily_stats(
        scored,
        top,
        formal,
        horizons=horizons,
        top_n=args.top_n,
        random_trials=args.random_trials,
        seed=args.seed,
    )

    monthly = build_monthly_stats(daily)

    rank_ic, feature_ic = build_rank_ic(
        scored,
        rank_cols,
        horizons=horizons,
        min_ic_count=args.min_ic_count,
    )

    summary = build_summary(daily, monthly, rank_ic)

    top_out = args.output_dir / "individual_alpha_top10.csv"
    daily_out = args.output_dir / "individual_alpha_daily.csv"
    monthly_out = args.output_dir / "individual_alpha_monthly.csv"
    summary_out = args.output_dir / "individual_alpha_summary.csv"
    rank_ic_out = args.output_dir / "individual_alpha_rank_ic.csv"
    feature_ic_out = args.output_dir / "individual_alpha_feature_ic.csv"

    keep_top_cols = [
        "date",
        "symbol",
        "alpha_rank",
        "alpha_score",
        "close",
        "ret_1d_pct",
        "ret_3d_pct",
        "ret_5d_pct",
        "ret_10d_pct",
        "ret_20d_pct",
        "excess_20d_pct",
        "close_to_ma20",
        "volatility20",
        "rank_volatility_contract",
        "rank_cooldown_10d",
        "rank_pullback_ma20",
        "rank_cooldown_3d",
        "rank_cooldown_5d",
    ]

    for h in horizons:
        keep_top_cols += [f"ret_T{h}_pct", f"up_T{h}"]

    keep_top_cols = [c for c in keep_top_cols if c in top.columns]

    top[keep_top_cols].to_csv(top_out, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_out, index=False, encoding="utf-8-sig")
    monthly.to_csv(monthly_out, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")
    rank_ic.to_csv(rank_ic_out, index=False, encoding="utf-8-sig")
    feature_ic.to_csv(feature_ic_out, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print("Top10    :", top_out)
    print("Daily    :", daily_out)
    print("Monthly  :", monthly_out)
    print("Summary  :", summary_out)
    print("RankIC   :", rank_ic_out)
    print("FeatureIC:", feature_ic_out)

    print("\nSummary:")
    if summary.empty:
        print("No summary rows.")
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()