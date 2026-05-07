from __future__ import annotations

from pathlib import Path
import pandas as pd

from core.progress import progress_bar
from selection_strategies import get_selection_strategy
from core.storage import read_table, write_table
from config import POOLS_DIR

POOL_EXPORT_COLUMNS = [
    "symbol", "file", "date", "close", "selection_strategy", "raw_score", "score_pct", "K", "D", "J", "yellow_ma", "z_fast_trend_line", "z_slow_trend_line",
    "hard_brick_turn_strong", "two_day_above_trend_line", "short_trend_above_trend_line",
    "close_below_short_trend_cap", "price_below_50", "j_momentum_or_low",
    "close_above_yellow_ma", "surge_then_shrink_pullback", "small_rise_long_red_brick",
    "risk_filter_pass", "prior_20d_accelerated_huge_volume_bear",
    "prior_20d_shrink_limit_up", "long_lower_shadow_hammer", "limit_up_red_brick",
    "accelerated_huge_volume_bear", "shrink_limit_up", "limit_up", "shrink_volume",
    "pct_change_close", "current_red_height", "red_height_reference", "selected_v7_base",
    "short_trend", "trend_line", "green_to_red", "valid_red_brick", "valid_green_brick",
    "previous_green_height", "green_height_70pct", "brick_reversal_strength", "price_zone_ok",
    "trend_condition_ok", "price_condition_ok", "selected",
]


class PoolStore:
    """Single-file pool storage for all dates of one selection strategy."""

    def __init__(self, pools_dir: str | Path = POOLS_DIR) -> None:
        self.pools_dir = Path(pools_dir)
        self.pools_dir.mkdir(parents=True, exist_ok=True)

    def pool_path(self, selection_strategy: str) -> Path:
        return self.pools_dir / f"{selection_strategy}_pool.parquet"

    def read(self, selection_strategy: str) -> pd.DataFrame:
        df = read_table(self.pool_path(selection_strategy))
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def read_date(self, selection_strategy: str, target_date) -> pd.DataFrame:
        df = self.read(selection_strategy)
        if df.empty:
            return df
        target_ts = pd.Timestamp(target_date).normalize()
        out = df[pd.to_datetime(df["date"]).dt.normalize() == target_ts].copy()
        return out.reset_index(drop=True)

    def write_replace_range(self, selection_strategy: str, pool_df: pd.DataFrame, start_date, end_date) -> Path:
        path = self.pool_path(selection_strategy)
        old = self.read(selection_strategy) if path.exists() else pd.DataFrame()
        start_ts = pd.Timestamp(start_date).normalize()
        end_ts = pd.Timestamp(end_date).normalize()
        if not old.empty and "date" in old.columns:
            old_dates = pd.to_datetime(old["date"]).dt.normalize()
            old = old[(old_dates < start_ts) | (old_dates > end_ts)]
        out = pd.concat([old, pool_df], ignore_index=True) if not pool_df.empty else old
        if not out.empty:
            out["date"] = pd.to_datetime(out["date"])
            sort_cols = [c for c in ["date", "score_pct", "file"] if c in out.columns]
            ascending = [True, False, True][: len(sort_cols)]
            out = out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True) if sort_cols else out.reset_index(drop=True)
        write_table(out, path)
        return path


def build_pool_from_indicators(indicator_df: pd.DataFrame, selection_strategy: str, start_date, end_date, n1: int = 4, n2: int = 6) -> pd.DataFrame:
    if indicator_df.empty:
        return pd.DataFrame(columns=POOL_EXPORT_COLUMNS)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    selector = get_selection_strategy(selection_strategy)
    rows = []
    groups = list(indicator_df.sort_values(["symbol", "date"]).groupby("symbol", sort=False))
    for _, part in progress_bar(groups, desc="Build selection pool", total=len(groups)):
        selected_part = selector(part.copy(), n1=n1, n2=n2)
        selected_part = selected_part[(selected_part["date"] >= start_ts) & (selected_part["date"] <= end_ts)]
        selected_part = selected_part[selected_part["selected"].fillna(0).astype(int) == 1]
        if not selected_part.empty:
            rows.append(selected_part)
    pool_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=POOL_EXPORT_COLUMNS)
    if pool_df.empty:
        return pool_df
    for col in POOL_EXPORT_COLUMNS:
        if col not in pool_df.columns:
            pool_df[col] = None
    pool_df = pool_df[POOL_EXPORT_COLUMNS]
    pool_df = pool_df.sort_values(["date", "score_pct", "file"], ascending=[True, False, True]).reset_index(drop=True)
    return pool_df
