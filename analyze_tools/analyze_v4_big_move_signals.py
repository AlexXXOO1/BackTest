# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_DATA_ROOT = Path(r"C:\Users\zyf37\Desktop\BackTest_Data")
DEFAULT_POOL_PATH = DEFAULT_DATA_ROOT / "pools" / "renko_chart_select_strategy_v4_pool.parquet"
DEFAULT_MARKET_CACHE_DIR = DEFAULT_DATA_ROOT / "market_cache" / "daily_bars_by_symbol"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "output" / "v4_big_move_signal_analysis"

HORIZONS = {
    "T1": 1,
    "T2": 2,
    "T3": 3,
}


def normalize_symbol(x) -> str:
    if pd.isna(x):
        return ""

    s = str(x).strip().upper()
    m = re.search(r"(\d{6})", s)
    if not m:
        return ""

    code = m.group(1)

    if "SH" in s:
        return f"SH#{code}"
    if "SZ" in s:
        return f"SZ#{code}"

    if code.startswith(("600", "601", "603", "605", "688")):
        return f"SH#{code}"
    if code.startswith(("000", "001", "002", "003", "300", "301")):
        return f"SZ#{code}"

    return code


def load_pool(pool_path: Path) -> pd.DataFrame:
    print("[1/5] Loading v4 pool...")

    if not pool_path.exists():
        raise FileNotFoundError(f"pool not found: {pool_path}")

    df = pd.read_parquet(pool_path) if pool_path.suffix.lower() != ".csv" else pd.read_csv(pool_path)

    if "symbol" not in df.columns:
        if "code" in df.columns:
            df["symbol"] = df["code"]
        else:
            raise ValueError("pool missing symbol/code column")

    if "date" not in df.columns:
        raise ValueError("pool missing date column")

    df = df.copy()
    df["symbol"] = df["symbol"].map(normalize_symbol)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date"])
    df = df[df["symbol"] != ""]

    if "selected" in df.columns:
        df = df[df["selected"].fillna(0).astype(int) == 1].copy()

    df = df.drop_duplicates(["date", "symbol"], keep="last")
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    print(f"    pool rows: {len(df):,}")
    print(f"    trading days: {df['date'].nunique():,}")
    print(f"    symbols: {df['symbol'].nunique():,}")
    print(f"    date range: {df['date'].min()} -> {df['date'].max()}")

    if df.empty:
        raise RuntimeError("v4 pool is empty")

    return df


def load_market(market_cache_dir: Path) -> pd.DataFrame:
    print("[2/5] Loading market cache...")

    files = sorted(glob.glob(str(market_cache_dir / "*.parquet")))
    if not files:
        raise RuntimeError(f"no market parquet found: {market_cache_dir}")

    parts = []

    for p in tqdm(files, desc="Load market", unit="file"):
        try:
            df = pd.read_parquet(p)
            if df.empty:
                continue

            if "symbol" not in df.columns:
                df["symbol"] = Path(p).stem

            if "date" not in df.columns or "close" not in df.columns:
                continue

            x = df[["symbol", "date", "close"]].copy()
            x["symbol"] = x["symbol"].map(normalize_symbol)
            x["date"] = pd.to_datetime(x["date"], errors="coerce")
            x["close"] = pd.to_numeric(x["close"], errors="coerce")
            x = x.dropna(subset=["symbol", "date", "close"])
            x = x[(x["symbol"] != "") & (x["close"] > 0)]

            if not x.empty:
                parts.append(x)

        except Exception as exc:
            print(f"[WARN] failed to read {p}: {exc}")

    if not parts:
        raise RuntimeError("market cache loaded empty")

    market = pd.concat(parts, ignore_index=True)
    market = market.drop_duplicates(["symbol", "date"], keep="last")
    market = market.sort_values(["symbol", "date"]).reset_index(drop=True)

    print(f"    market rows: {len(market):,}")
    print(f"    trading days: {market['date'].nunique():,}")
    print(f"    symbols: {market['symbol'].nunique():,}")

    return market


def add_forward_returns(market: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] Calculating T+1/T+2/T+3 returns...")

    out = market.copy()
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

    for h_name, h in tqdm(HORIZONS.items(), desc="Forward horizons", unit="horizon"):
        future_close = out.groupby("symbol")["close"].shift(-h)
        out[f"{h_name}_return_pct"] = (future_close / out["close"] - 1.0) * 100.0

    return out


def merge_pool_forward(pool: pd.DataFrame, market_fwd: pd.DataFrame) -> pd.DataFrame:
    print("[4/5] Merging pool and forward returns...")

    fwd_cols = ["date", "symbol", "close"] + [f"{h}_return_pct" for h in HORIZONS]
    fwd = market_fwd[fwd_cols].copy()

    merged = pool.merge(
        fwd,
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_mkt"),
    )

    missing = int(merged["close_mkt"].isna().sum()) if "close_mkt" in merged.columns else 0
    if missing > 0:
        print(f"[WARN] pool rows missing market forward data: {missing:,}")

    # If pool already has close column, merge creates close_mkt.
    if "close_mkt" in merged.columns:
        merged = merged.drop(columns=["close_mkt"])

    for h in HORIZONS:
        merged[f"{h}_big_up_5"] = merged[f"{h}_return_pct"] >= 5.0
        merged[f"{h}_big_down_5"] = merged[f"{h}_return_pct"] <= -5.0

    merged["any_T1_T2_T3_big_up_5"] = False
    merged["any_T1_T2_T3_big_down_5"] = False
    for h in HORIZONS:
        merged["any_T1_T2_T3_big_up_5"] |= merged[f"{h}_big_up_5"].fillna(False)
        merged["any_T1_T2_T3_big_down_5"] |= merged[f"{h}_big_down_5"].fillna(False)

    return merged


def get_signal_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "selected",
        "selected_score_base",
    }

    base_candidates = [
        "daily_return_pct",
        "intraday_return_pct",
        "amplitude_pct",
        "upper_shadow_pct",
        "lower_shadow_pct",
        "body_pct",
        "body_abs_pct",
        "ma5",
        "ma10",
        "ma20",
        "ma60",
        "volume_ratio_prev1",
        "volume_ratio_ma5",
        "volume_ratio_ma10",
        "macd_dif",
        "macd_dea",
        "macd_hist",
        "renko_value",
        "z_short_trend_line",
        "z_long_trend_line",
    ]

    cols = []

    for c in df.columns:
        if c in exclude:
            continue

        cl = str(c).lower()
        if c in base_candidates or cl.startswith(("v4", "v4a", "v4b")):
            if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
                cols.append(c)

    # Remove forward result columns from signal list
    cols = [
        c for c in cols
        if not re.match(r"^T[123]_", str(c))
        and not str(c).endswith("_return_pct")
        and "big_up" not in str(c)
        and "big_down" not in str(c)
    ]

    # Stable order, no duplicates
    return list(dict.fromkeys(cols))


def summarize_big_move_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for h in HORIZONS:
        ret_col = f"{h}_return_pct"
        rows.append({
            "scope": h,
            "rows": int(df[ret_col].notna().sum()),
            "big_up_5_count": int((df[ret_col] >= 5.0).sum()),
            "big_up_5_ratio": float((df[ret_col] >= 5.0).mean()),
            "big_down_5_count": int((df[ret_col] <= -5.0).sum()),
            "big_down_5_ratio": float((df[ret_col] <= -5.0).mean()),
            "avg_return_pct": float(df[ret_col].mean()),
            "median_return_pct": float(df[ret_col].median()),
        })

    rows.append({
        "scope": "ANY_T1_T2_T3",
        "rows": int(len(df)),
        "big_up_5_count": int(df["any_T1_T2_T3_big_up_5"].sum()),
        "big_up_5_ratio": float(df["any_T1_T2_T3_big_up_5"].mean()),
        "big_down_5_count": int(df["any_T1_T2_T3_big_down_5"].sum()),
        "big_down_5_ratio": float(df["any_T1_T2_T3_big_down_5"].mean()),
        "avg_return_pct": np.nan,
        "median_return_pct": np.nan,
    })

    return pd.DataFrame(rows)


def summarize_numeric_signals(df: pd.DataFrame, signal_cols: list[str]) -> pd.DataFrame:
    up = df[df["any_T1_T2_T3_big_up_5"]].copy()
    down = df[df["any_T1_T2_T3_big_down_5"]].copy()
    neutral = df[(~df["any_T1_T2_T3_big_up_5"]) & (~df["any_T1_T2_T3_big_down_5"])].copy()

    rows = []

    for col in signal_cols:
        s_all = pd.to_numeric(df[col], errors="coerce")
        if s_all.notna().sum() < 30:
            continue

        s_up = pd.to_numeric(up[col], errors="coerce")
        s_down = pd.to_numeric(down[col], errors="coerce")
        s_neu = pd.to_numeric(neutral[col], errors="coerce")

        all_std = float(s_all.std())
        up_mean = float(s_up.mean()) if s_up.notna().sum() else np.nan
        down_mean = float(s_down.mean()) if s_down.notna().sum() else np.nan
        neu_mean = float(s_neu.mean()) if s_neu.notna().sum() else np.nan

        rows.append({
            "signal": col,
            "all_mean": float(s_all.mean()),
            "up_mean": up_mean,
            "down_mean": down_mean,
            "neutral_mean": neu_mean,
            "up_minus_down_mean": up_mean - down_mean if pd.notna(up_mean) and pd.notna(down_mean) else np.nan,
            "up_minus_neutral_mean": up_mean - neu_mean if pd.notna(up_mean) and pd.notna(neu_mean) else np.nan,
            "down_minus_neutral_mean": down_mean - neu_mean if pd.notna(down_mean) and pd.notna(neu_mean) else np.nan,
            "all_median": float(s_all.median()),
            "up_median": float(s_up.median()) if s_up.notna().sum() else np.nan,
            "down_median": float(s_down.median()) if s_down.notna().sum() else np.nan,
            "all_std": all_std,
            "abs_up_down_mean_gap": abs(up_mean - down_mean) if pd.notna(up_mean) and pd.notna(down_mean) else np.nan,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values("abs_up_down_mean_gap", ascending=False).reset_index(drop=True)
    return out


def summarize_binary_signals(df: pd.DataFrame, signal_cols: list[str]) -> pd.DataFrame:
    rows = []

    up_mask = df["any_T1_T2_T3_big_up_5"]
    down_mask = df["any_T1_T2_T3_big_down_5"]
    neutral_mask = (~up_mask) & (~down_mask)

    for col in signal_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        uniq = set(s.dropna().unique().tolist())

        if not uniq.issubset({0, 1, 0.0, 1.0}):
            continue

        rows.append({
            "signal": col,
            "all_rate_1": float(s.mean()),
            "up_rate_1": float(pd.to_numeric(df.loc[up_mask, col], errors="coerce").mean()),
            "down_rate_1": float(pd.to_numeric(df.loc[down_mask, col], errors="coerce").mean()),
            "neutral_rate_1": float(pd.to_numeric(df.loc[neutral_mask, col], errors="coerce").mean()),
            "up_minus_down_rate": float(pd.to_numeric(df.loc[up_mask, col], errors="coerce").mean() - pd.to_numeric(df.loc[down_mask, col], errors="coerce").mean()),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["abs_up_down_rate_gap"] = out["up_minus_down_rate"].abs()
    out = out.sort_values("abs_up_down_rate_gap", ascending=False).reset_index(drop=True)
    return out


def bucket_one_signal(df: pd.DataFrame, col: str, q: int = 5) -> pd.DataFrame:
    s = pd.to_numeric(df[col], errors="coerce")
    tmp = df[[col, "any_T1_T2_T3_big_up_5", "any_T1_T2_T3_big_down_5"] + [f"{h}_return_pct" for h in HORIZONS]].copy()
    tmp[col] = s
    tmp = tmp.dropna(subset=[col])

    if tmp[col].nunique() < 4 or len(tmp) < 50:
        return pd.DataFrame()

    try:
        tmp["bucket"] = pd.qcut(tmp[col], q=q, duplicates="drop")
    except Exception:
        return pd.DataFrame()

    rows = []
    for bucket, g in tmp.groupby("bucket", observed=True):
        row = {
            "signal": col,
            "bucket": str(bucket),
            "rows": len(g),
            "big_up_any_ratio": float(g["any_T1_T2_T3_big_up_5"].mean()),
            "big_down_any_ratio": float(g["any_T1_T2_T3_big_down_5"].mean()),
        }
        for h in HORIZONS:
            row[f"{h}_avg_return_pct"] = float(g[f"{h}_return_pct"].mean())
        rows.append(row)

    return pd.DataFrame(rows)


def save_outputs(
    df: pd.DataFrame,
    overview: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    binary_summary: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cols_front = [
        "date", "symbol",
        "T1_return_pct", "T2_return_pct", "T3_return_pct",
        "any_T1_T2_T3_big_up_5", "any_T1_T2_T3_big_down_5",
    ]
    cols_front = [c for c in cols_front if c in df.columns]
    other_cols = [c for c in df.columns if c not in cols_front]

    big_up = df[df["any_T1_T2_T3_big_up_5"]].copy()
    big_down = df[df["any_T1_T2_T3_big_down_5"]].copy()

    overview.to_csv(output_dir / "big_move_overview.csv", index=False, encoding="utf-8-sig")
    numeric_summary.to_csv(output_dir / "numeric_signal_summary.csv", index=False, encoding="utf-8-sig")
    binary_summary.to_csv(output_dir / "binary_signal_summary.csv", index=False, encoding="utf-8-sig")
    bucket_summary.to_csv(output_dir / "top_signal_bucket_summary.csv", index=False, encoding="utf-8-sig")

    big_up[cols_front + other_cols].to_csv(output_dir / "big_up_any_T1_T2_T3_ge_5.csv", index=False, encoding="utf-8-sig")
    big_down[cols_front + other_cols].to_csv(output_dir / "big_down_any_T1_T2_T3_le_minus_5.csv", index=False, encoding="utf-8-sig")

    print(f"\nSaved output dir: {output_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pool-path", type=str, default=str(DEFAULT_POOL_PATH))
    parser.add_argument("--market-cache-dir", type=str, default=str(DEFAULT_MARKET_CACHE_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--no-save", action="store_true")

    args = parser.parse_args()

    pool = load_pool(Path(args.pool_path))
    market = load_market(Path(args.market_cache_dir))
    market_fwd = add_forward_returns(market)
    merged = merge_pool_forward(pool, market_fwd)

    print("[5/5] Analyzing big up/down signals...")

    signal_cols = get_signal_columns(merged)

    overview = summarize_big_move_overview(merged)
    numeric_summary = summarize_numeric_signals(merged, signal_cols)
    binary_summary = summarize_binary_signals(merged, signal_cols)

    bucket_parts = []
    for col in numeric_summary.head(max(args.top_n, 5))["signal"].tolist() if not numeric_summary.empty else []:
        b = bucket_one_signal(merged, col)
        if not b.empty:
            bucket_parts.append(b)

    bucket_summary = pd.concat(bucket_parts, ignore_index=True) if bucket_parts else pd.DataFrame()

    print("\n========== BIG MOVE OVERVIEW ==========")
    print(overview.to_string(index=False))

    print("\n========== TOP NUMERIC SIGNAL DIFFERENCES ==========")
    if numeric_summary.empty:
        print("No numeric signal summary.")
    else:
        cols = [
            "signal",
            "all_mean",
            "up_mean",
            "down_mean",
            "neutral_mean",
            "up_minus_down_mean",
            "up_median",
            "down_median",
        ]
        print(numeric_summary[cols].head(args.top_n).to_string(index=False))

    print("\n========== TOP BINARY CONDITION DIFFERENCES ==========")
    if binary_summary.empty:
        print("No binary signal summary.")
    else:
        cols = [
            "signal",
            "all_rate_1",
            "up_rate_1",
            "down_rate_1",
            "neutral_rate_1",
            "up_minus_down_rate",
        ]
        print(binary_summary[cols].head(args.top_n).to_string(index=False))

    print("\n========== TOP SIGNAL BUCKET SUMMARY ==========")
    if bucket_summary.empty:
        print("No bucket summary.")
    else:
        print(bucket_summary.head(args.top_n * 5).to_string(index=False))

    if not args.no_save:
        save_outputs(
            df=merged,
            overview=overview,
            numeric_summary=numeric_summary,
            binary_summary=binary_summary,
            bucket_summary=bucket_summary,
            output_dir=Path(args.output_dir),
        )

    print("\n========== NOTES ==========")
    print("big_up means any of T+1/T+2/T+3 return >= 5%.")
    print("big_down means any of T+1/T+2/T+3 return <= -5%.")
    print("numeric_signal_summary ranks features by mean gap between big_up and big_down groups.")
    print("bucket summary shows whether feature ranges concentrate big-up or big-down cases.")


if __name__ == "__main__":
    main()
