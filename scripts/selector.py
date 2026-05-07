from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

import pandas as pd


# =============================================================================
# Project path
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import BacktestConfig

# =============================================================================
# Strategy registry compatibility
# =============================================================================

def load_selection_registry() -> tuple[dict[str, Callable], Optional[Callable]]:
    """
    Compatible with different selection_strategies/__init__.py versions.

    Supported forms:
    1. SELECTION_STRATEGY_REGISTRY
    2. list_selection_strategies()
    3. get_selection_strategy()
    """
    import selection_strategies as ss

    registry = getattr(ss, "SELECTION_STRATEGY_REGISTRY", None)
    get_func = getattr(ss, "get_selection_strategy", None)

    if registry is None:
        registry = {}

    if not isinstance(registry, dict):
        registry = {}

    return registry, get_func


def get_strategy_func(strategy_name: str) -> Callable:
    registry, get_func = load_selection_registry()

    if strategy_name in registry:
        return registry[strategy_name]

    if get_func is not None:
        try:
            return get_func(strategy_name)
        except Exception:
            pass

    registered = sorted(registry.keys())

    msg = [
        "",
        "[ERROR] Selection strategy is not registered.",
        f"[ERROR] Requested strategy: {strategy_name}",
        "",
        "[ERROR] Registered strategies:",
    ]

    if registered:
        msg.extend([f"  - {name}" for name in registered])
    else:
        msg.append("  - None")

    msg.extend(
        [
            "",
            "[FIX SUGGESTIONS]",
            "1. Make sure your strategy file is located in:",
            "   selection_strategies/",
            "2. Make sure the strategy file contains:",
            f'   STRATEGY_NAME = "{strategy_name}"',
            "3. Make sure the strategy file contains:",
            "   SELECT_FUNC = select_strategy",
            "4. Make sure selection_strategies/__init__.py exports the auto-discovery registry.",
            "5. Run:",
            "   python .\\scripts\\selector.py --list-strategies",
        ]
    )

    raise ValueError("\n".join(msg))


def list_registered_strategies() -> list[str]:
    registry, _ = load_selection_registry()
    return sorted(registry.keys())


# =============================================================================
# Helpers
# =============================================================================

def normalize_code(x) -> str:
    if pd.isna(x):
        return ""

    s = str(x).strip()

    if s.endswith(".txt"):
        s = s[:-4]

    if s.endswith(".parquet"):
        s = s[:-8]

    if "#" in s:
        prefix, code = s.split("#", 1)
        code = "".join(ch for ch in code if ch.isdigit())
        if len(code) == 6:
            return f"{prefix.upper()}#{code}"
        return s

    digits = "".join(ch for ch in s if ch.isdigit())

    if len(digits) == 6:
        if digits.startswith("6"):
            return f"SH#{digits}"
        return f"SZ#{digits}"

    return s


def infer_code_from_path(path: Path) -> str:
    return normalize_code(path.stem)


def find_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def ensure_date_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize date column without globally sorting the whole dataframe.

    Important:
    The indicator cache can contain millions of rows and many float columns.
    Calling sort_values("date") on the full cache creates a huge full-column copy
    and can easily trigger numpy ArrayMemoryError.

    Sorting is done later per symbol, where each group is small enough.
    """
    out = df.copy(deep=False)

    date_col = find_first_existing_col(out, ["date", "trade_date", "signal_date"])

    if date_col is None:
        raise ValueError(f"Cannot find date column. Columns: {list(out.columns)}")

    if date_col != "date":
        out["date"] = out[date_col]

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"])

    return out


def ensure_code_col(df: pd.DataFrame, fallback_code: str = "") -> pd.DataFrame:
    out = df.copy(deep=False)

    code_col = find_first_existing_col(
        out,
        ["code", "symbol", "stock_code", "ts_code", "file", "filename"],
    )

    if code_col is not None:
        out["code"] = out[code_col].map(normalize_code)
    else:
        out["code"] = normalize_code(fallback_code)

    return out


def filter_date_range(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    out = ensure_date_col(df)

    return out[
        (out["date"] >= start_date)
        & (out["date"] <= end_date)
    ].copy()


def safe_bool_selected(s: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(s, errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(bool)
    )


def read_market_file(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = ensure_date_col(df)
    df = ensure_code_col(df, fallback_code=infer_code_from_path(path))
    return df


def read_indicator_cache(indicator_cache_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(indicator_cache_path)
    df = ensure_date_col(df)
    df = ensure_code_col(df)
    return df


def get_market_files(market_cache_dir: Path) -> list[Path]:
    if not market_cache_dir.exists():
        raise FileNotFoundError(f"Market cache dir not found: {market_cache_dir}")

    files = sorted(market_cache_dir.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files found in: {market_cache_dir}")

    return files


def make_pool_path(pools_dir: Path, strategy_name: str) -> Path:
    return pools_dir / f"{strategy_name}_pool.parquet"


# =============================================================================
# Core build
# =============================================================================

def build_pool_from_indicator_cache(
    *,
    indicator_cache_path: Path,
    strategy_name: str,
    strategy_func: Callable,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n1: int,
    n2: int,
    debug_summary: bool,
) -> pd.DataFrame:
    print(f"[INFO] Loading indicator cache: {indicator_cache_path}")

    ind = read_indicator_cache(indicator_cache_path)

    # Filter date range before per-symbol processing to reduce memory pressure.
    # Indicators have already been calculated in daily_indicators.parquet, so selector
    # does not need rows outside the requested signal date range.
    ind = ind[(ind["date"] >= start_date) & (ind["date"] <= end_date)].copy()

    if debug_summary:
        print(f"[DEBUG] indicator cache rows in range: {len(ind):,}")
        print(f"[DEBUG] indicator cache columns: {len(ind.columns):,}")
        print(f"[DEBUG] indicator cache symbols in range: {ind['code'].nunique():,}")

    selected_parts: list[pd.DataFrame] = []

    grouped = ind.groupby("code", sort=True, dropna=True)

    try:
        from tqdm import tqdm
        iterator = tqdm(grouped, total=ind["code"].nunique(), desc="Build pool from indicator cache", unit="symbol")
    except Exception:
        iterator = grouped

    for code, g in iterator:
        if g.empty:
            continue

        g = g.sort_values("date").reset_index(drop=True)

        try:
            out = strategy_func(g, n1=n1, n2=n2)
        except Exception as e:
            print(f"[WARN] Strategy failed for {code}: {e}")
            continue

        if out is None or out.empty:
            continue

        out = ensure_date_col(out)
        out = ensure_code_col(out, fallback_code=code)

        if "selection_strategy" not in out.columns:
            out["selection_strategy"] = strategy_name

        if "selected" not in out.columns:
            continue

        # Date range has already been filtered before strategy execution.
        selected = out[safe_bool_selected(out["selected"])].copy()

        if selected.empty:
            continue

        selected_parts.append(selected)

    if not selected_parts:
        return pd.DataFrame()

    pool = pd.concat(selected_parts, ignore_index=True, sort=False)
    return pool


def build_pool_from_market_cache(
    *,
    market_cache_dir: Path,
    strategy_name: str,
    strategy_func: Callable,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n1: int,
    n2: int,
    debug_summary: bool,
) -> pd.DataFrame:
    files = get_market_files(market_cache_dir)

    if debug_summary:
        print(f"[DEBUG] market cache files: {len(files):,}")
        print(f"[DEBUG] market cache dir: {market_cache_dir}")

    selected_parts: list[pd.DataFrame] = []

    try:
        from tqdm import tqdm
        iterator = tqdm(files, desc="Build pool from market cache", unit="file")
    except Exception:
        iterator = files

    for path in iterator:
        code = infer_code_from_path(path)

        try:
            df = read_market_file(path)
        except Exception as e:
            print(f"[WARN] Failed to read {path.name}: {e}")
            continue

        if df.empty:
            continue

        try:
            out = strategy_func(df, n1=n1, n2=n2)
        except Exception as e:
            print(f"[WARN] Strategy failed for {code}: {e}")
            continue

        if out is None or out.empty:
            continue

        out = ensure_date_col(out)
        out = ensure_code_col(out, fallback_code=code)

        if "selection_strategy" not in out.columns:
            out["selection_strategy"] = strategy_name

        if "selected" not in out.columns:
            continue

        out_range = filter_date_range(out, start_date, end_date)
        selected = out_range[safe_bool_selected(out_range["selected"])].copy()

        if selected.empty:
            continue

        selected_parts.append(selected)

    if not selected_parts:
        return pd.DataFrame()

    pool = pd.concat(selected_parts, ignore_index=True, sort=False)
    return pool


def build_pool(
    *,
    strategy_name: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    market_cache_dir: Path,
    indicator_cache_path: Path,
    n1: int,
    n2: int,
    debug_summary: bool,
) -> pd.DataFrame:
    strategy_func = get_strategy_func(strategy_name)

    if indicator_cache_path.exists():
        pool = build_pool_from_indicator_cache(
            indicator_cache_path=indicator_cache_path,
            strategy_name=strategy_name,
            strategy_func=strategy_func,
            start_date=start_date,
            end_date=end_date,
            n1=n1,
            n2=n2,
            debug_summary=debug_summary,
        )
    else:
        print(f"[WARN] Indicator cache not found, fallback to market cache: {indicator_cache_path}")

        pool = build_pool_from_market_cache(
            market_cache_dir=market_cache_dir,
            strategy_name=strategy_name,
            strategy_func=strategy_func,
            start_date=start_date,
            end_date=end_date,
            n1=n1,
            n2=n2,
            debug_summary=debug_summary,
        )

    if pool.empty:
        return pool

    # =========================================================================
    # CRITICAL FIX:
    # Do NOT whitelist columns.
    # Keep all strategy output columns.
    #
    # This preserves:
    # - score_close_to_short_trend_below_088
    # - score_close_to_short_trend_below_086
    # - score_close_to_short_trend_below_084
    # - penalty_brick_reversal_strength_100
    # - penalty_brick_reversal_strength_120
    # - score_rank_key
    # =========================================================================

    pool = pool.copy()

    if "date" in pool.columns:
        pool["date"] = pd.to_datetime(pool["date"], errors="coerce").dt.normalize()

    if "code" in pool.columns:
        pool["code"] = pool["code"].map(normalize_code)

    if "selection_strategy" not in pool.columns:
        pool["selection_strategy"] = strategy_name

    # Move common important columns to front, but keep all other columns.
    front_cols = [
        "date",
        "code",
        "name",
        "stock_name",
        "selection_strategy",
        "selected",
        "selected_score_base",
        "score",
        "score_pct",
        "score_rank_key",
        "close_to_short_trend",
        "daily_return_pct",
        "brick_reversal_ratio",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    existing_front_cols = [c for c in front_cols if c in pool.columns]
    other_cols = [c for c in pool.columns if c not in existing_front_cols]

    pool = pool[existing_front_cols + other_cols]

    pool = pool.sort_values(
        by=[c for c in ["date", "score_pct", "score_rank_key", "code"] if c in pool.columns],
        ascending=[
            True,
            False if "score_pct" in pool.columns else True,
            False if "score_rank_key" in pool.columns else True,
            True,
        ][: len([c for c in ["date", "score_pct", "score_rank_key", "code"] if c in pool.columns])],
    ).reset_index(drop=True)

    return pool


# =============================================================================
# Save
# =============================================================================

def save_pool(
    *,
    pool: pd.DataFrame,
    pool_path: Path,
    overwrite: bool,
) -> None:
    pool_path.parent.mkdir(parents=True, exist_ok=True)

    if pool_path.exists() and not overwrite:
        old = pd.read_parquet(pool_path)

        combined = pd.concat([old, pool], ignore_index=True, sort=False)

        dedup_cols = [c for c in ["date", "code", "selection_strategy"] if c in combined.columns]

        if dedup_cols:
            combined = combined.drop_duplicates(dedup_cols, keep="last")

        combined.to_parquet(pool_path, index=False)
        print(f"[INFO] Appended pool saved: {pool_path}")
        print(f"[INFO] Rows: old={len(old):,}, new={len(pool):,}, combined={len(combined):,}")
    else:
        pool.to_parquet(pool_path, index=False)
        print(f"[INFO] Pool saved: {pool_path}")
        print(f"[INFO] Rows: {len(pool):,}")


# =============================================================================
# Debug
# =============================================================================

def print_debug_summary(
    *,
    strategy_name: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    market_cache_dir: Path,
    indicator_cache_path: Path,
    pools_dir: Path,
    pool: Optional[pd.DataFrame] = None,
) -> None:
    print()
    print("========== SELECTOR DEBUG SUMMARY ==========")
    print(f"[DEBUG] strategy: {strategy_name}")
    print(f"[DEBUG] date range: {start_date} -> {end_date}")
    print(f"[DEBUG] market_cache_dir: {market_cache_dir}")
    print(f"[DEBUG] market_cache_dir exists: {market_cache_dir.exists()}")
    print(f"[DEBUG] indicator_cache_path: {indicator_cache_path}")
    print(f"[DEBUG] indicator_cache_path exists: {indicator_cache_path.exists()}")
    print(f"[DEBUG] pools_dir: {pools_dir}")
    print(f"[DEBUG] pools_dir exists: {pools_dir.exists()}")

    if pool is not None:
        print(f"[DEBUG] pool rows: {len(pool):,}")
        print(f"[DEBUG] pool columns: {len(pool.columns):,}")

        check_cols = [
            c
            for c in pool.columns
            if (
                "088" in c
                or "086" in c
                or "084" in c
                or "55_to_7" in c
                or "penalty" in c
                or "rank" in c
                or "below_100" in c
            )
        ]

        print("[DEBUG] new v4 columns found:")
        if check_cols:
            for c in check_cols:
                print(f"  - {c}")
        else:
            print("  - None")

        if "date" in pool.columns:
            print(f"[DEBUG] pool min date: {pool['date'].min()}")
            print(f"[DEBUG] pool max date: {pool['date'].max()}")

        if "score_pct" in pool.columns:
            print("[DEBUG] score_pct describe:")
            print(pool["score_pct"].describe())

        if "selected" in pool.columns:
            print("[DEBUG] selected value counts:")
            print(pool["selected"].value_counts(dropna=False))


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    default = BacktestConfig()
    parser = argparse.ArgumentParser(description="Build selected pool for a selection strategy.")

    parser.add_argument("--start-date", default=None, help="Example: 2024-01-01")
    parser.add_argument("--end-date", default=None, help="Example: 2026-04-30")

    parser.add_argument("--strategy", default="renko_chart_select_strategy_v4")

    parser.add_argument(
        "--market-cache-dir",
        type=Path,
        default=default.market_cache_dir,
    )

    parser.add_argument(
        "--indicator-cache-path",
        type=Path,
        default=default.indicator_cache_path,
    )

    parser.add_argument(
        "--pools-dir",
        type=Path,
        default=default.pools_dir,
    )

    parser.add_argument("--n1", type=int, default=4)
    parser.add_argument("--n2", type=int, default=6)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug-summary", action="store_true")

    parser.add_argument("--list-strategies", action="store_true")

    # Keep compatibility with older commands.
    parser.add_argument("--max-workers", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_strategies:
        strategies = list_registered_strategies()
        print("========== Registered selection strategies ==========")
        if strategies:
            for name in strategies:
                print(f"  - {name}")
        else:
            print("  - None")
        return

    if args.start_date is None or args.end_date is None:
        raise ValueError("--start-date and --end-date are required unless --list-strategies is used.")

    start_date = pd.to_datetime(args.start_date, errors="raise").normalize()
    end_date = pd.to_datetime(args.end_date, errors="raise").normalize()

    if end_date < start_date:
        raise ValueError(f"end_date < start_date: {end_date} < {start_date}")

    if args.debug_summary:
        print_debug_summary(
            strategy_name=args.strategy,
            start_date=start_date,
            end_date=end_date,
            market_cache_dir=args.market_cache_dir,
            indicator_cache_path=args.indicator_cache_path,
            pools_dir=args.pools_dir,
        )

    pool = build_pool(
        strategy_name=args.strategy,
        start_date=start_date,
        end_date=end_date,
        market_cache_dir=args.market_cache_dir,
        indicator_cache_path=args.indicator_cache_path,
        n1=args.n1,
        n2=args.n2,
        debug_summary=args.debug_summary,
    )

    pool_path = make_pool_path(args.pools_dir, args.strategy)

    if pool.empty:
        print()
        print("[WARN] No selected rows generated.")
        print(f"[WARN] Pool not saved: {pool_path}")
        return

    save_pool(
        pool=pool,
        pool_path=pool_path,
        overwrite=args.overwrite,
    )

    if args.debug_summary:
        print_debug_summary(
            strategy_name=args.strategy,
            start_date=start_date,
            end_date=end_date,
            market_cache_dir=args.market_cache_dir,
            indicator_cache_path=args.indicator_cache_path,
            pools_dir=args.pools_dir,
            pool=pool,
        )


if __name__ == "__main__":
    main()