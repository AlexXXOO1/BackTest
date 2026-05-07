from __future__ import annotations

from pathlib import Path
import pandas as pd
from core.progress import progress_bar

from core.data_store import MarketDataStore
from indicators import add_all_indicators
from core.storage import read_table, write_table
from core.cache_meta import build_indicator_meta, write_json_meta
from config import INDICATOR_CACHE_PATH


class IndicatorStore:
    """Build and read the reusable daily indicator cache."""

    def __init__(self, indicator_cache_path: str | Path = INDICATOR_CACHE_PATH) -> None:
        self.indicator_cache_path = Path(indicator_cache_path)

    def exists(self) -> bool:
        return self.indicator_cache_path.exists() and self.indicator_cache_path.stat().st_size > 0

    def read(self) -> pd.DataFrame:
        df = read_table(self.indicator_cache_path)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def build(self, market_store: MarketDataStore, n1: int = 4, n2: int = 6, start_date=None, end_date=None, incremental: bool = False, lookback_days: int = 150) -> pd.DataFrame:
        start_ts = pd.Timestamp(start_date) if start_date is not None else None
        end_ts = pd.Timestamp(end_date) if end_date is not None else None

        existing = self.read() if incremental and self.exists() else pd.DataFrame()
        if incremental and start_ts is None:
            if not existing.empty:
                max_date = pd.to_datetime(existing["date"]).max()
                start_ts = max_date - pd.Timedelta(days=int(lookback_days * 1.6))
            else:
                start_ts = None

        rows = []
        symbols = market_store.list_cached_symbols()
        for symbol in progress_bar(symbols, desc="Build indicators", total=len(symbols)):
            raw = market_store.get_symbol_data(symbol, end_date=end_ts)
            if raw.empty:
                continue
            calc = add_all_indicators(raw, n1=n1, n2=n2)
            if start_ts is not None:
                calc = calc[calc["date"] >= start_ts]
            if end_ts is not None:
                calc = calc[calc["date"] <= end_ts]
            if not calc.empty:
                rows.append(calc)

        new_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if incremental and not existing.empty and not new_df.empty:
            min_new_date = pd.to_datetime(new_df["date"]).min()
            existing = existing[pd.to_datetime(existing["date"]) < min_new_date]
            out = pd.concat([existing, new_df], ignore_index=True)
        elif incremental and not new_df.empty:
            out = new_df
        elif not incremental:
            out = new_df
        else:
            out = existing

        if not out.empty:
            out["date"] = pd.to_datetime(out["date"])
            key_cols = [c for c in ["symbol", "date"] if c in out.columns]
            if key_cols:
                out = out.sort_values(key_cols).drop_duplicates(key_cols, keep="last").reset_index(drop=True)
        write_table(out, self.indicator_cache_path)

        meta = build_indicator_meta(
            df=out,
            n1=n1,
            n2=n2,
            start_date=start_date,
            end_date=end_date,
            incremental=incremental,
            lookback_days=lookback_days,
            market_cache_dir=market_store.market_cache_dir,
            indicator_cache_path=self.indicator_cache_path,
        )
        meta_path = write_json_meta(meta, self.indicator_cache_path)
        print(f"[INFO] Indicator meta saved: {meta_path}")

        return out
