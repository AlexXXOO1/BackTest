from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_ROOT = Path(r"C:\Users\zyf37\Desktop\BackTest_Data")

DEFAULT_POOL_PATH = DATA_ROOT / "pools" / "b1_stage_low_select_strategy_v0_pool.parquet"
DEFAULT_INDICATOR_PATH = DATA_ROOT / "indicator_cache" / "daily_indicators.parquet"
DEFAULT_OUT_PATH = DATA_ROOT / "pools" / "b1_stage_low_select_strategy_v0_pool_with_structure.parquet"

WINDOW = 40
HIGH_DOWN_THRESHOLD = 0.99
LOW_DOWN_THRESHOLD = 0.97
FACTOR_COL = "b1_structure_downtrend_40"


def _load_pool(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Pool parquet not found: {path}")

    pool = pd.read_parquet(path)
    required = {"symbol", "date"}

    missing = required - set(pool.columns)
    if missing:
        raise ValueError(f"Pool missing required columns: {sorted(missing)}")

    pool = pool.copy()
    pool["symbol"] = pool["symbol"].astype(str)
    pool["date"] = pd.to_datetime(pool["date"], errors="coerce")

    return pool


def _load_daily_indicators(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Indicator parquet not found: {path}")

    daily = pd.read_parquet(path, columns=["symbol", "date", "high", "low"])

    daily = daily.copy()
    daily["symbol"] = daily["symbol"].astype(str)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["high"] = pd.to_numeric(daily["high"], errors="coerce")
    daily["low"] = pd.to_numeric(daily["low"], errors="coerce")

    daily = daily.dropna(subset=["symbol", "date", "high", "low"])
    daily = daily.sort_values(["symbol", "date"]).reset_index(drop=True)

    return daily


def _calc_structure_downtrend_for_symbol(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()

    recent_high = g["high"].rolling(WINDOW, min_periods=WINDOW).max()
    previous_high = recent_high.shift(WINDOW)

    recent_low = g["low"].rolling(WINDOW, min_periods=WINDOW).min()
    previous_low = recent_low.shift(WINDOW)

    high_clearly_lower = recent_high < previous_high * HIGH_DOWN_THRESHOLD
    low_clearly_lower = recent_low < previous_low * LOW_DOWN_THRESHOLD

    out = g[["symbol", "date"]].copy()
    out[FACTOR_COL] = (high_clearly_lower & low_clearly_lower).fillna(False).astype("int8")

    return out


def build_factor_table(daily: pd.DataFrame) -> pd.DataFrame:
    parts = []

    for symbol, g in daily.groupby("symbol", sort=False):
        parts.append(_calc_structure_downtrend_for_symbol(g))

    if not parts:
        return pd.DataFrame(columns=["symbol", "date", FACTOR_COL])

    return pd.concat(parts, ignore_index=True)


def enrich_pool(
    pool_path: Path = DEFAULT_POOL_PATH,
    indicator_path: Path = DEFAULT_INDICATOR_PATH,
    out_path: Path = DEFAULT_OUT_PATH,
) -> None:
    print("[INFO] Loading existing B1 pool...")
    pool = _load_pool(pool_path)

    print("[INFO] Loading full daily indicator data...")
    daily = _load_daily_indicators(indicator_path)

    print("[INFO] Calculating structure downtrend factor...")
    factor = build_factor_table(daily)

    print("[INFO] Merging factor back to pool...")
    if FACTOR_COL in pool.columns:
        pool = pool.drop(columns=[FACTOR_COL])

    out = pool.merge(factor, on=["symbol", "date"], how="left")
    out[FACTOR_COL] = out[FACTOR_COL].fillna(0).astype("int8")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Writing output parquet: {out_path}")
    out.to_parquet(out_path, index=False)

    print("[DONE]")
    print(f"rows: {len(out)}")
    print(f"factor: {FACTOR_COL}")
    print(f"window: {WINDOW}")
    print("value counts:")
    print(out[FACTOR_COL].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    enrich_pool()
