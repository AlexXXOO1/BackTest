# -*- coding: utf-8 -*-
import argparse
import glob
import re
from pathlib import Path# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


DEFAULT_DATA_ROOT = Path(r"C:\Users\zyf37\Desktop\BackTest_Data")
DEFAULT_POOLS_DIR = DEFAULT_DATA_ROOT / "pools"
DEFAULT_MARKET_CACHE_DIR = DEFAULT_DATA_ROOT / "market_cache" / "daily_bars_by_symbol"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "output" / "pool_indicator_analysis"

ID_COLUMNS = {
    "date",
    "symbol",
    "code",
    "股票代码",
    "股票名",
    "name",
    "stock_name",
    "selection_strategy",
    "source_pool",
}


def progress_iter(items, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, ncols=100)


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
    result = []

    for part in str(value).replace(";", ",").split(","):
        part = part.strip().upper().replace("T", "")
        if not part:
            continue

        n = int(part)
        if n <= 0:
            raise ValueError(f"Invalid horizon: {part}")

        result.append(n)

    if not result:
        raise ValueError("No horizons supplied.")

    return sorted(set(result))


def find_symbol_col(df: pd.DataFrame) -> str:
    for col in ["symbol", "code", "股票代码"]:
        if col in df.columns:
            return col

    raise ValueError(f"Missing symbol/code column. columns={df.columns.tolist()}")


def load_pool_file(
    path: Path,
    start_date: str | None,
    end_date: str | None,
    include_unselected: bool,
) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if df.empty:
        return df

    if "date" not in df.columns:
        raise ValueError(f"{path.name} missing date column.")

    symbol_col = find_symbol_col(df)

    out = df.copy()
    out["symbol"] = out[symbol_col].map(normalize_symbol)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    out = out.dropna(subset=["date"])
    out = out[out["symbol"] != ""].copy()

    if not include_unselected and "selected" in out.columns:
        selected = pd.to_numeric(out["selected"], errors="coerce").fillna(0)
        out = out[selected == 1].copy()

    if start_date:
        out = out[out["date"] >= pd.Timestamp(start_date).normalize()].copy()

    if end_date:
        out = out[out["date"] <= pd.Timestamp(end_date).normalize()].copy()

    out = out.drop_duplicates(subset=["date", "symbol"], keep="last")
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    out["source_pool"] = path.stem

    return out


def load_pools(args) -> list[tuple[Path, pd.DataFrame]]:
    if args.pool_path:
        files = [Path(args.pool_path)]
    else:
        files = sorted(Path(args.pools_dir).glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet pool files found: {args.pools_dir}")

    result = []

    for path in files:
        try:
            df = load_pool_file(
                path=path,
                start_date=args.start_date,
                end_date=args.end_date,
                include_unselected=args.include_unselected,
            )

            if df.empty:
                print(f"[WARN] empty after filter: {path.name}")
                continue

            result.append((path, df))

            print(
                f"[POOL] {path.name}: rows={len(df):,}, "
                f"symbols={df['symbol'].nunique():,}, "
                f"date={df['date'].min().date()} -> {df['date'].max().date()}"
            )

        except Exception as exc:
            print(f"[WARN] skip {path.name}: {exc}")

    if not result:
        raise RuntimeError("No valid pool data loaded.")

    return result


def read_market_file(path: Path, fallback_symbol: str) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path, columns=["date", "close", "symbol"])
    except Exception:
        try:
            df = pd.read_parquet(path, columns=["date", "close"])
        except Exception:
            df = pd.read_parquet(path)

    if df.empty or "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()

    out = df.copy()

    if "symbol" not in out.columns:
        out["symbol"] = fallback_symbol

    out["symbol"] = out["symbol"].map(normalize_symbol)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    out = out.dropna(subset=["date", "close"])
    out = out[(out["symbol"] != "") & (out["close"] > 0)].copy()

    return out[["symbol", "date", "close"]]


def load_market_forward(
    market_cache_dir: Path,
    needed_symbols: set[str],
    horizons: list[int],
) -> pd.DataFrame:
    files = sorted(market_cache_dir.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No market parquet found: {market_cache_dir}")

    parts = []

    for path in progress_iter(files, "Load market cache"):
        file_symbol = normalize_symbol(path.stem)

        if needed_symbols and file_symbol not in needed_symbols:
            continue

        try:
            df = read_market_file(path, file_symbol)
        except Exception as exc:
            print(f"[WARN] failed to read market file {path.name}: {exc}")
            continue

        if df.empty:
            continue

        if needed_symbols:
            df = df[df["symbol"].isin(needed_symbols)].copy()

        if not df.empty:
            parts.append(df)

    if not parts:
        raise RuntimeError("Market cache loaded empty for pool symbols.")

    market = pd.concat(parts, ignore_index=True)
    market = market.drop_duplicates(subset=["symbol", "date"], keep="last")
    market = market.sort_values(["symbol", "date"]).reset_index(drop=True)

    for h in horizons:
        future_close = market.groupby("symbol")["close"].shift(-h)
        market[f"T{h}_return_pct"] = (future_close / market["close"] - 1.0) * 100.0
        market[f"T{h}_is_up"] = market[f"T{h}_return_pct"] > 0

    keep_cols = ["symbol", "date"]

    for h in horizons:
        keep_cols.extend([f"T{h}_return_pct", f"T{h}_is_up"])

    print(
        f"[MARKET] rows={len(market):,}, symbols={market['symbol'].nunique():,}, "
        f"date={market['date'].min().date()} -> {market['date'].max().date()}"
    )

    return market[keep_cols]


def is_forward_col(col: str) -> bool:
    return bool(re.fullmatch(r"T\d+_(return_pct|is_up)", str(col)))


def bool_like_series(s: pd.Series) -> bool:
    non = s.dropna()

    if non.empty:
        return False

    vals = set(non.astype(str).str.strip().str.lower().unique().tolist())

    return vals.issubset({"true", "false", "1", "0", "1.0", "0.0"})


def normalize_bool_label(value) -> str:
    if pd.isna(value):
        return "NA"

    s = str(value).strip().lower()

    if s in {"true", "1", "1.0"}:
        return "True"

    if s in {"false", "0", "0.0"}:
        return "False"

    return str(value)


def numeric_ratio(s: pd.Series) -> float:
    if s.empty:
        return 0.0

    converted = pd.to_numeric(s, errors="coerce")

    return float(converted.notna().mean())


def candidate_columns(df: pd.DataFrame) -> list[str]:
    result = []

    for col in df.columns:
        if col in ID_COLUMNS:
            continue

        if is_forward_col(col):
            continue

        if str(col).startswith("_"):
            continue

        result.append(col)

    return result


def base_metrics(df: pd.DataFrame, horizon: str) -> dict:
    h = int(horizon.replace("T", ""))
    ret = pd.to_numeric(df[f"T{h}_return_pct"], errors="coerce").dropna()

    if ret.empty:
        return {
            "base_count": 0,
            "base_avg_return_pct": np.nan,
            "base_win_rate_pct": np.nan,
        }

    return {
        "base_count": int(ret.count()),
        "base_avg_return_pct": float(ret.mean()),
        "base_win_rate_pct": float((ret > 0).mean() * 100.0),
    }


def summarize_groups(
    df: pd.DataFrame,
    indicator: str,
    group_col: str,
    indicator_type: str,
    source_pool: str,
    horizons: list[int],
) -> pd.DataFrame:
    rows = []

    for group_value, g in df.groupby(group_col, dropna=False):
        for h in horizons:
            ret = pd.to_numeric(g[f"T{h}_return_pct"], errors="coerce").dropna()

            if ret.empty:
                rows.append(
                    {
                        "source_pool": source_pool,
                        "indicator": indicator,
                        "indicator_type": indicator_type,
                        "group": str(group_value),
                        "horizon": f"T{h}",
                        "count": 0,
                        "avg_return_pct": np.nan,
                        "median_return_pct": np.nan,
                        "win_rate_pct": np.nan,
                    }
                )
                continue

            rows.append(
                {
                    "source_pool": source_pool,
                    "indicator": indicator,
                    "indicator_type": indicator_type,
                    "group": str(group_value),
                    "horizon": f"T{h}",
                    "count": int(ret.count()),
                    "avg_return_pct": float(ret.mean()),
                    "median_return_pct": float(ret.median()),
                    "win_rate_pct": float((ret > 0).mean() * 100.0),
                }
            )

    return pd.DataFrame(rows)


def make_summary_row(
    df: pd.DataFrame,
    detail: pd.DataFrame,
    indicator: str,
    indicator_type: str,
    source_pool: str,
    primary_horizon: str,
    min_group_count: int,
) -> dict | None:
    base = base_metrics(df, primary_horizon)

    row = {
        "source_pool": source_pool,
        "indicator": indicator,
        "indicator_type": indicator_type,
        "rows": len(df),
        "non_null": int(df[indicator].notna().sum()),
        "coverage_pct": float(df[indicator].notna().mean() * 100.0),
        "unique_count": int(df[indicator].nunique(dropna=True)),
        "primary_horizon": primary_horizon,
        "pearson_corr_primary": np.nan,
        "spearman_corr_primary": np.nan,
        "direction_hint": "",
        "best_group": "",
        "best_group_count": 0,
        "best_group_avg_return_pct": np.nan,
        "best_group_win_rate_pct": np.nan,
        "base_avg_return_pct": base["base_avg_return_pct"],
        "base_win_rate_pct": base["base_win_rate_pct"],
        "lift_avg_return_pct": np.nan,
        "lift_win_rate_pct": np.nan,
        "worst_group": "",
        "worst_group_avg_return_pct": np.nan,
        "filter_risk_return_gap_pct": np.nan,
        "optimization_score": np.nan,
    }

    h = int(primary_horizon.replace("T", ""))
    ret_col = f"T{h}_return_pct"

    if indicator_type == "numeric":
        x = pd.to_numeric(df[indicator], errors="coerce")
        y = pd.to_numeric(df[ret_col], errors="coerce")
        valid = x.notna() & y.notna()

        if int(valid.sum()) >= 5 and x[valid].nunique() >= 2:
            row["pearson_corr_primary"] = float(x[valid].corr(y[valid], method="pearson"))
            row["spearman_corr_primary"] = float(x[valid].corr(y[valid], method="spearman"))

            sp = row["spearman_corr_primary"]

            if pd.notna(sp):
                if sp >= 0.05:
                    row["direction_hint"] = "higher_may_be_better"
                elif sp <= -0.05:
                    row["direction_hint"] = "lower_may_be_better"
                else:
                    row["direction_hint"] = "weak_or_nonlinear"

    sub = detail[
        (detail["horizon"] == primary_horizon)
        & (detail["count"] >= min_group_count)
    ].copy()

    if sub.empty:
        return row

    sub = sub.sort_values("avg_return_pct", ascending=True)

    worst = sub.iloc[0]
    best = sub.iloc[-1]

    row["best_group"] = str(best["group"])
    row["best_group_count"] = int(best["count"])
    row["best_group_avg_return_pct"] = float(best["avg_return_pct"])
    row["best_group_win_rate_pct"] = float(best["win_rate_pct"])
    row["worst_group"] = str(worst["group"])
    row["worst_group_avg_return_pct"] = float(worst["avg_return_pct"])

    if pd.notna(row["base_avg_return_pct"]):
        row["lift_avg_return_pct"] = (
            row["best_group_avg_return_pct"] - row["base_avg_return_pct"]
        )

    if pd.notna(row["base_win_rate_pct"]):
        row["lift_win_rate_pct"] = (
            row["best_group_win_rate_pct"] - row["base_win_rate_pct"]
        )

    row["filter_risk_return_gap_pct"] = (
        row["best_group_avg_return_pct"] - row["worst_group_avg_return_pct"]
    )

    lift_avg = 0.0 if pd.isna(row["lift_avg_return_pct"]) else row["lift_avg_return_pct"]
    lift_win = 0.0 if pd.isna(row["lift_win_rate_pct"]) else row["lift_win_rate_pct"]
    count_weight = min(row["best_group_count"] / max(min_group_count * 3, 1), 1.0)

    row["optimization_score"] = float((lift_avg + 0.02 * lift_win) * count_weight)

    return row


def analyze_numeric(
    df: pd.DataFrame,
    col: str,
    source_pool: str,
    horizons: list[int],
    primary_horizon: str,
    quantiles: int,
    min_group_count: int,
) -> tuple[dict | None, pd.DataFrame]:
    x = pd.to_numeric(df[col], errors="coerce")
    valid_df = df[x.notna()].copy()
    valid_df[col] = x[x.notna()]

    if len(valid_df) < min_group_count or valid_df[col].nunique() < 2:
        return None, pd.DataFrame()

    q = min(int(quantiles), int(valid_df[col].nunique()))

    if q < 2:
        return None, pd.DataFrame()

    try:
        valid_df["_group"] = pd.qcut(valid_df[col], q=q, duplicates="drop").astype(str)
    except Exception:
        valid_df["_group"] = pd.cut(valid_df[col], bins=q, duplicates="drop").astype(str)

    detail = summarize_groups(
        df=valid_df,
        indicator=col,
        group_col="_group",
        indicator_type="numeric",
        source_pool=source_pool,
        horizons=horizons,
    )

    summary = make_summary_row(
        df=valid_df,
        detail=detail,
        indicator=col,
        indicator_type="numeric",
        source_pool=source_pool,
        primary_horizon=primary_horizon,
        min_group_count=min_group_count,
    )

    return summary, detail


def analyze_category(
    df: pd.DataFrame,
    col: str,
    source_pool: str,
    horizons: list[int],
    primary_horizon: str,
    min_group_count: int,
    max_categories: int,
    indicator_type: str,
) -> tuple[dict | None, pd.DataFrame]:
    valid_df = df[df[col].notna()].copy()

    if valid_df.empty:
        return None, pd.DataFrame()

    if indicator_type == "bool":
        valid_df["_group"] = valid_df[col].map(normalize_bool_label)
    else:
        valid_df["_group"] = valid_df[col].astype(str).str.slice(0, 120)

    unique_count = valid_df["_group"].nunique(dropna=True)

    if unique_count < 2 or unique_count > max_categories:
        return None, pd.DataFrame()

    detail = summarize_groups(
        df=valid_df,
        indicator=col,
        group_col="_group",
        indicator_type=indicator_type,
        source_pool=source_pool,
        horizons=horizons,
    )

    summary = make_summary_row(
        df=valid_df,
        detail=detail,
        indicator=col,
        indicator_type=indicator_type,
        source_pool=source_pool,
        primary_horizon=primary_horizon,
        min_group_count=min_group_count,
    )

    return summary, detail


def analyze_pool(
    path: Path,
    pool: pd.DataFrame,
    market: pd.DataFrame,
    args,
    horizons: list[int],
    primary_horizon: str,
) -> tuple[list[dict], list[pd.DataFrame], list[pd.DataFrame]]:
    df = pool.merge(
        market,
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )

    primary_h = int(primary_horizon.replace("T", ""))
    df = df[pd.to_numeric(df[f"T{primary_h}_return_pct"], errors="coerce").notna()].copy()

    if df.empty:
        print(f"[WARN] no forward return rows: {path.name}")
        return [], [], []

    summaries = []
    numeric_details = []
    categorical_details = []

    for col in progress_iter(candidate_columns(df), f"Analyze {path.stem}"):
        s = df[col]

        try:
            if bool_like_series(s):
                summary, detail = analyze_category(
                    df=df,
                    col=col,
                    source_pool=path.stem,
                    horizons=horizons,
                    primary_horizon=primary_horizon,
                    min_group_count=args.min_group_count,
                    max_categories=args.max_categories,
                    indicator_type="bool",
                )

                if summary is not None:
                    summaries.append(summary)
                    categorical_details.append(detail)

                continue

            if numeric_ratio(s) >= args.numeric_ratio:
                summary, detail = analyze_numeric(
                    df=df,
                    col=col,
                    source_pool=path.stem,
                    horizons=horizons,
                    primary_horizon=primary_horizon,
                    quantiles=args.quantiles,
                    min_group_count=args.min_group_count,
                )

                if summary is not None:
                    summaries.append(summary)
                    numeric_details.append(detail)

                continue

            if s.nunique(dropna=True) <= args.max_categories:
                summary, detail = analyze_category(
                    df=df,
                    col=col,
                    source_pool=path.stem,
                    horizons=horizons,
                    primary_horizon=primary_horizon,
                    min_group_count=args.min_group_count,
                    max_categories=args.max_categories,
                    indicator_type="category",
                )

                if summary is not None:
                    summaries.append(summary)
                    categorical_details.append(detail)

        except Exception as exc:
            print(f"[WARN] failed indicator {path.stem}.{col}: {exc}")

    return summaries, numeric_details, categorical_details


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze indicator columns inside one or all pool parquet files."
    )

    parser.add_argument("--pool-path", type=Path, default=None)
    parser.add_argument("--pools-dir", type=Path, default=DEFAULT_POOLS_DIR)
    parser.add_argument("--market-cache-dir", type=Path, default=DEFAULT_MARKET_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--horizons", default="1,2,3")
    parser.add_argument("--primary-horizon", default="T2")
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--min-group-count", type=int, default=20)
    parser.add_argument("--max-categories", type=int, default=30)
    parser.add_argument("--numeric-ratio", type=float, default=0.8)
    parser.add_argument("--include-unselected", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    horizons = parse_horizons(args.horizons)

    primary_horizon = str(args.primary_horizon).upper()

    if not primary_horizon.startswith("T"):
        primary_horizon = f"T{primary_horizon}"

    if int(primary_horizon.replace("T", "")) not in horizons:
        raise ValueError(
            f"--primary-horizon {primary_horizon} must be inside --horizons {horizons}"
        )

    pools = load_pools(args)

    needed_symbols = set()

    for _, df in pools:
        needed_symbols.update(df["symbol"].dropna().astype(str).unique().tolist())

    market = load_market_forward(
        market_cache_dir=Path(args.market_cache_dir),
        needed_symbols=needed_symbols,
        horizons=horizons,
    )

    all_summaries = []
    numeric_details = []
    categorical_details = []

    for path, pool_df in pools:
        summaries, numeric_parts, categorical_parts = analyze_pool(
            path=path,
            pool=pool_df,
            market=market,
            args=args,
            horizons=horizons,
            primary_horizon=primary_horizon,
        )

        all_summaries.extend(summaries)
        numeric_details.extend(numeric_parts)
        categorical_details.extend(categorical_parts)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(all_summaries)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["optimization_score", "lift_avg_return_pct", "lift_win_rate_pct"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    numeric_df = (
        pd.concat(numeric_details, ignore_index=True)
        if numeric_details
        else pd.DataFrame()
    )

    categorical_df = (
        pd.concat(categorical_details, ignore_index=True)
        if categorical_details
        else pd.DataFrame()
    )

    save_csv(summary_df, output_dir / "all_pool_indicator_summary.csv")
    save_csv(numeric_df, output_dir / "numeric_indicator_bins.csv")
    save_csv(categorical_df, output_dir / "categorical_indicator_groups.csv")

    if not summary_df.empty:
        top_cols = [
            "source_pool",
            "indicator",
            "indicator_type",
            "primary_horizon",
            "direction_hint",
            "best_group",
            "best_group_count",
            "best_group_avg_return_pct",
            "base_avg_return_pct",
            "lift_avg_return_pct",
            "best_group_win_rate_pct",
            "base_win_rate_pct",
            "lift_win_rate_pct",
            "filter_risk_return_gap_pct",
            "optimization_score",
        ]

        save_csv(
            summary_df[top_cols].head(100),
            output_dir / "top_indicator_candidates.csv",
        )

    print("[DONE]")


if __name__ == "__main__":
    main()

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
