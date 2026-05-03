from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# User config
# =============================================================================

V2_POOL_PATH = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\pools\renko_chart_select_strategy_v2_pool.parquet"
)

V2_FORWARD_RETURNS_PATH = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\pools\cache\renko_chart_select_strategy_v2_forward_returns.parquet"
)

OUTPUT_CSV_PATH = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\pools\v2_close_to_short_trend_bucket_return_analysis.csv"
)

MAIN_RETURN_COL = "t3_close_from_t1_open_pct"


# =============================================================================
# Bucket definition
# =============================================================================

BUCKET_BINS = [
    -np.inf,
    0.90,
    0.95,
    0.98,
    1.00,
    1.02,
    1.05,
    np.inf,
]

BUCKET_LABELS = [
    "< 0.90",
    "0.90 - 0.95",
    "0.95 - 0.98",
    "0.98 - 1.00",
    "1.00 - 1.02",
    "1.02 - 1.05",
    ">= 1.05",
]


# =============================================================================
# Helpers
# =============================================================================

def drop_duplicate_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Drop duplicated column names and keep the first occurrence."""
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        print(f"\n[{name}] duplicated columns found and removed:")
        print(sorted(set(duplicated)))
    return df.loc[:, ~df.columns.duplicated()].copy()


def normalize_symbol(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace("SH#", "", regex=False)
        .str.replace("SZ#", "", regex=False)
        .str.replace(".txt", "", regex=False)
        .str.extract(r"(\d{6})", expand=False)
    )


def normalize_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}")


def win_rate_pct(ret: pd.Series) -> float:
    ret = pd.to_numeric(ret, errors="coerce").dropna()
    if len(ret) == 0:
        return np.nan
    return float((ret > 0).mean() * 100)


def avg_col(sub: pd.DataFrame, col: str) -> float:
    if col not in sub.columns:
        return np.nan
    s = sub[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce").mean()


def median_col(sub: pd.DataFrame, col: str) -> float:
    if col not in sub.columns:
        return np.nan
    s = sub[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce").median()


def get_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Safely get one column as Series even if duplicated columns exist."""
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj


def summarize_group(sub: pd.DataFrame, label: str, total_count: int) -> dict:
    ret = pd.to_numeric(get_series(sub, MAIN_RETURN_COL), errors="coerce").dropna()

    return {
        "bucket": label,
        "count": len(sub),
        "count_pct": len(sub) / total_count * 100 if total_count else np.nan,

        "valid_return_count": len(ret),
        "win_rate_pct": win_rate_pct(ret),
        "avg_return_pct": ret.mean() if len(ret) else np.nan,
        "median_return_pct": ret.median() if len(ret) else np.nan,
        "max_single_loss_pct": ret.min() if len(ret) else np.nan,
        "max_single_profit_pct": ret.max() if len(ret) else np.nan,

        "close_to_short_trend_avg": sub["close_to_short_trend"].mean(),
        "close_to_short_trend_median": sub["close_to_short_trend"].median(),

        "t1_open_from_t0_close_pct_avg": avg_col(sub, "t1_open_from_t0_close_pct"),
        "t1_open_from_t0_close_pct_median": median_col(sub, "t1_open_from_t0_close_pct"),

        "t1_close_from_t0_close_pct_avg": avg_col(sub, "t1_close_from_t0_close_pct"),
        "t1_close_from_t0_close_pct_median": median_col(sub, "t1_close_from_t0_close_pct"),

        "t2_close_from_t0_close_pct_avg": avg_col(sub, "t2_close_from_t0_close_pct"),
        "t2_close_from_t0_close_pct_median": median_col(sub, "t2_close_from_t0_close_pct"),

        "t3_close_from_t0_close_pct_avg": avg_col(sub, "t3_close_from_t0_close_pct"),
        "t3_close_from_t0_close_pct_median": median_col(sub, "t3_close_from_t0_close_pct"),

        "t1_close_from_t1_open_pct_avg": avg_col(sub, "t1_close_from_t1_open_pct"),
        "t1_close_from_t1_open_pct_median": median_col(sub, "t1_close_from_t1_open_pct"),

        "t2_close_from_t1_open_pct_avg": avg_col(sub, "t2_close_from_t1_open_pct"),
        "t2_close_from_t1_open_pct_median": median_col(sub, "t2_close_from_t1_open_pct"),

        "t3_close_from_t1_open_pct_avg": avg_col(sub, "t3_close_from_t1_open_pct"),
        "t3_close_from_t1_open_pct_median": median_col(sub, "t3_close_from_t1_open_pct"),
    }


def main() -> None:
    if not V2_POOL_PATH.exists():
        raise FileNotFoundError(f"V2 pool not found: {V2_POOL_PATH}")

    if not V2_FORWARD_RETURNS_PATH.exists():
        raise FileNotFoundError(f"V2 forward returns cache not found: {V2_FORWARD_RETURNS_PATH}")

    pool = pd.read_parquet(V2_POOL_PATH)
    fwd = pd.read_parquet(V2_FORWARD_RETURNS_PATH)

    pool = drop_duplicate_columns(pool, "pool")
    fwd = drop_duplicate_columns(fwd, "forward_cache")

    print("\n========== Loaded files ==========")
    print(f"V2 pool path: {V2_POOL_PATH}")
    print(f"V2 pool rows: {len(pool):,}")
    print(f"Forward cache path: {V2_FORWARD_RETURNS_PATH}")
    print(f"Forward cache rows: {len(fwd):,}")

    pool_date_col = find_col(pool, ["date", "signal_date", "t0_date"])
    pool_symbol_col = find_col(pool, ["symbol", "code", "stock_code"])

    fwd_date_col = find_col(fwd, ["date", "signal_date", "t0_date"])
    fwd_symbol_col = find_col(fwd, ["symbol", "code", "stock_code"])

    required_pool_cols = {"close", "short_trend"}
    missing_pool_cols = required_pool_cols - set(pool.columns)
    if missing_pool_cols:
        raise ValueError(f"V2 pool missing columns: {missing_pool_cols}")

    if MAIN_RETURN_COL not in fwd.columns:
        raise ValueError(
            f"Forward cache missing main return column: {MAIN_RETURN_COL}\n"
            f"Available columns: {list(fwd.columns)}"
        )

    pool = pool.copy()
    fwd = fwd.copy()

    pool["_date_key"] = normalize_date(pool[pool_date_col])
    pool["_symbol_key"] = normalize_symbol(pool[pool_symbol_col])

    fwd["_date_key"] = normalize_date(fwd[fwd_date_col])
    fwd["_symbol_key"] = normalize_symbol(fwd[fwd_symbol_col])

    pool = pool[pool["_date_key"].notna() & pool["_symbol_key"].notna()].copy()
    fwd = fwd[fwd["_date_key"].notna() & fwd["_symbol_key"].notna()].copy()

    return_cols_to_remove_from_pool = [
        MAIN_RETURN_COL,
        "t1_open_from_t0_close_pct",
        "t1_close_from_t0_close_pct",
        "t2_close_from_t0_close_pct",
        "t3_close_from_t0_close_pct",
        "t1_close_from_t1_open_pct",
        "t2_close_from_t1_open_pct",
        "t3_close_from_t1_open_pct",
    ]

    existing_return_cols = [c for c in return_cols_to_remove_from_pool if c in pool.columns]
    if existing_return_cols:
        print("\nRemove duplicated return columns from pool:")
        print(existing_return_cols)
        pool = pool.drop(columns=existing_return_cols)

    forward_keep_cols = [
        "_date_key",
        "_symbol_key",
        MAIN_RETURN_COL,
        "t1_open_from_t0_close_pct",
        "t1_close_from_t0_close_pct",
        "t2_close_from_t0_close_pct",
        "t3_close_from_t0_close_pct",
        "t1_close_from_t1_open_pct",
        "t2_close_from_t1_open_pct",
        "t3_close_from_t1_open_pct",
    ]
    forward_keep_cols = [c for c in forward_keep_cols if c in fwd.columns]

    fwd_small = fwd[forward_keep_cols].drop_duplicates(
        subset=["_date_key", "_symbol_key"],
        keep="first",
    )

    merged = pool.merge(
        fwd_small,
        on=["_date_key", "_symbol_key"],
        how="left",
        validate="many_to_one",
    )

    merged = drop_duplicate_columns(merged, "merged")

    main_ret = pd.to_numeric(get_series(merged, MAIN_RETURN_COL), errors="coerce")

    print("\n========== Merge summary ==========")
    print(f"Pool rows after key clean: {len(pool):,}")
    print(f"Forward rows after key clean: {len(fwd):,}")
    print(f"Forward rows after dedupe: {len(fwd_small):,}")
    print(f"Merged rows: {len(merged):,}")
    print(f"Valid main return rows: {main_ret.notna().sum():,}")
    print(f"Missing main return rows: {main_ret.isna().sum():,}")

    merged["close"] = pd.to_numeric(merged["close"], errors="coerce")
    merged["short_trend"] = pd.to_numeric(merged["short_trend"], errors="coerce")

    merged["close_to_short_trend"] = merged["close"] / merged["short_trend"]

    valid = merged[
        merged["close_to_short_trend"].notna()
        & np.isfinite(merged["close_to_short_trend"])
        & (merged["short_trend"] > 0)
    ].copy()

    valid["bucket"] = pd.cut(
        valid["close_to_short_trend"],
        bins=BUCKET_BINS,
        labels=BUCKET_LABELS,
        right=False,
    )

    total_count = len(valid)

    rows = []
    for bucket in BUCKET_LABELS:
        sub = valid[valid["bucket"] == bucket].copy()
        rows.append(summarize_group(sub, bucket, total_count))

    result = pd.DataFrame(rows)

    print("\n========== Bucket return result ==========")
    with pd.option_context(
        "display.max_rows", 100,
        "display.max_columns", 100,
        "display.width", 260,
        "display.float_format", "{:.4f}".format,
    ):
        print(result)

    valid["below_short_trend"] = valid["close"] < valid["short_trend"]

    simple_rows = []
    for below_value, sub in valid.groupby("below_short_trend"):
        label = "close < short_trend" if bool(below_value) else "close >= short_trend"
        simple_rows.append(summarize_group(sub, label, total_count))

    simple_result = pd.DataFrame(simple_rows)

    print("\n========== Simple split return result ==========")
    with pd.option_context(
        "display.max_rows", 100,
        "display.max_columns", 100,
        "display.width", 260,
        "display.float_format", "{:.4f}".format,
    ):
        print(simple_result)

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    simple_output = OUTPUT_CSV_PATH.with_name(
        OUTPUT_CSV_PATH.stem + "_simple_split.csv"
    )
    simple_result.to_csv(simple_output, index=False, encoding="utf-8-sig")

    print("\n========== Saved ==========")
    print(f"Bucket CSV: {OUTPUT_CSV_PATH}")
    print(f"Simple split CSV: {simple_output}")


if __name__ == "__main__":
    main()