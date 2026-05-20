# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Clean standalone IndicatorStore.

This version does NOT import core.data_store, so it works even if the project
has removed/renamed the old MarketDataStore module.

Input:
    market_cache_dir/*.parquet, one symbol per file if available.
Output:
    indicator_cache/daily_indicators.parquet

Cached indicators:
    base OHLCVA + K-line + MA + volume + MACD + renko_value.
"""

import json
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd

from core.config import INDICATOR_CACHE_PATH, MARKET_CACHE_DIR
from data_engine.indicators import add_all_indicators


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_pickle(path)


def _write_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_pickle(path)
    return path


def _normalize_symbol(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 6:
        return digits[-6:]
    if digits:
        return digits.zfill(6)
    return s


def _standardize_market_df(df: pd.DataFrame, fallback_symbol: str, fallback_file: str) -> pd.DataFrame:
    """Make a cached market dataframe compatible with indicators.basic."""
    out = df.copy()
    lower_map = {str(c).lower(): c for c in out.columns}

    rename_map = {}
    aliases = {
        "date": ["date", "trade_date", "datetime", "日期", "交易日期"],
        "open": ["open", "开盘", "开盘价"],
        "high": ["high", "最高", "最高价"],
        "low": ["low", "最低", "最低价"],
        "close": ["close", "收盘", "收盘价"],
        "volume": ["volume", "vol", "成交量"],
        "amount": ["amount", "成交额", "turnover", "money"],
        "symbol": ["symbol", "code", "stock_code", "ts_code", "股票代码"],
        "file": ["file", "filename", "source_file"],
    }

    for target, cands in aliases.items():
        for cand in cands:
            key = cand.lower()
            if key in lower_map:
                rename_map[lower_map[key]] = target
                break

    if rename_map:
        out = out.rename(columns=rename_map)

    required = ["date", "open", "high", "low", "close", "volume", "amount"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required market columns: {missing}; columns={list(df.columns)}")

    if "symbol" not in out.columns:
        out["symbol"] = fallback_symbol
    else:
        out["symbol"] = out["symbol"].map(_normalize_symbol)

    if "file" not in out.columns:
        out["file"] = fallback_file

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out[out["close"] > 0]
    out = out.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")

    return out[["symbol", "file", "date", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)


class IndicatorStore:
    def __init__(
        self,
        indicator_cache_path: str | Path = INDICATOR_CACHE_PATH,
        market_cache_dir: str | Path = MARKET_CACHE_DIR,
    ) -> None:
        self.indicator_cache_path = Path(indicator_cache_path)
        self.market_cache_dir = Path(market_cache_dir)
        self.indicator_cache_path.parent.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.indicator_cache_path.exists() and self.indicator_cache_path.stat().st_size > 0

    def read(self) -> pd.DataFrame:
        df = _read_table(self.indicator_cache_path)
        if not df.empty and "date" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def _market_files(self) -> list[Path]:
        return sorted(self.market_cache_dir.glob("*.parquet"))

    def build(
        self,
        n1: int = 4,
        n2: int = 6,
        start_date=None,
        end_date=None,
        incremental: bool = False,
        lookback_days: int = 150,
        ma_windows: Iterable[int] = (5, 10, 20, 60),
        volume_windows: Iterable[int] = (5, 10),
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        market_cache_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        if market_cache_dir is not None:
            self.market_cache_dir = Path(market_cache_dir)

        files = self._market_files()
        if not files:
            raise FileNotFoundError(
                f"No market parquet files found in {self.market_cache_dir}. "
                "Please run ops/daily_update/import_tdx_txt.py first, or check MARKET_CACHE_DIR."
            )

        start_ts = pd.Timestamp(start_date) if start_date is not None else None
        end_ts = pd.Timestamp(end_date) if end_date is not None else None

        if incremental:
            print("[WARN] incremental=True still needs existing-cache merge. For low-memory rebuild, prefer full rebuild.", flush=True)

        tmp_dir = self.indicator_cache_path.parent / f"_tmp_{self.indicator_cache_path.stem}_parts"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        part_paths: list[Path] = []
        buffer: list[pd.DataFrame] = []
        buffer_rows = 0
        total_rows = 0
        final_columns: list[str] = []
        total = len(files)
        part_no = 0
        chunk_row_limit = 250_000

        def flush_buffer() -> None:
            nonlocal buffer, buffer_rows, total_rows, final_columns, part_no

            if not buffer:
                return

            part = pd.concat(buffer, ignore_index=True, copy=False)
            if not part.empty:
                part["date"] = pd.to_datetime(part["date"], errors="coerce")
                part = part.sort_values(["symbol", "date"], kind="stable").drop_duplicates(
                    ["symbol", "date"], keep="last"
                ).reset_index(drop=True)

            if not part.empty:
                part_no += 1
                part_path = tmp_dir / f"part-{part_no:05d}.parquet"
                part.to_parquet(part_path, index=False)
                part_paths.append(part_path)
                total_rows += int(len(part))
                final_columns = list(part.columns)
                print(
                    f"[INFO] wrote indicator part {part_no}: rows={len(part):,}, total_rows={total_rows:,}",
                    flush=True,
                )

            buffer = []
            buffer_rows = 0

        for i, path in enumerate(files, start=1):
            if i == 1 or i % 50 == 0 or i == total:
                print(f"[INFO] Build basic indicators: {i}/{total}", flush=True)

            try:
                raw = _read_table(path)
                if raw.empty:
                    continue

                symbol = _normalize_symbol(path.stem)
                base = _standardize_market_df(raw, fallback_symbol=symbol, fallback_file=path.name)

                if end_ts is not None:
                    base = base[base["date"] <= end_ts]
                if base.empty:
                    continue

                calc = add_all_indicators(
                    base,
                    n1=n1,
                    n2=n2,
                    ma_windows=ma_windows,
                    volume_windows=volume_windows,
                    macd_fast=macd_fast,
                    macd_slow=macd_slow,
                    macd_signal=macd_signal,
                )

                if start_ts is not None:
                    calc = calc[pd.to_datetime(calc["date"], errors="coerce") >= start_ts]
                if end_ts is not None:
                    calc = calc[pd.to_datetime(calc["date"], errors="coerce") <= end_ts]

                if not calc.empty:
                    buffer.append(calc)
                    buffer_rows += int(len(calc))

                if buffer_rows >= chunk_row_limit:
                    flush_buffer()

            except Exception as exc:
                print(f"[WARN] skip {path.name}: {exc}", flush=True)

        flush_buffer()

        if not part_paths:
            out = pd.DataFrame()
            _write_table(out, self.indicator_cache_path)
            final_columns = []
            total_rows = 0
        else:
            print(f"[INFO] merging {len(part_paths):,} indicator parts into final parquet...", flush=True)
            try:
                import pyarrow.parquet as pq

                final_tmp = self.indicator_cache_path.with_suffix(self.indicator_cache_path.suffix + ".tmp")
                if final_tmp.exists():
                    final_tmp.unlink()

                writer = None
                try:
                    for part_path in part_paths:
                        table = pq.read_table(part_path)
                        if writer is None:
                            writer = pq.ParquetWriter(final_tmp, table.schema)
                        else:
                            table = table.cast(writer.schema)
                        writer.write_table(table)
                finally:
                    if writer is not None:
                        writer.close()

                final_tmp.replace(self.indicator_cache_path)
            except Exception as exc:
                raise RuntimeError(f"Failed to merge indicator parquet parts: {exc}") from exc
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            out = pd.DataFrame(columns=final_columns)

        meta = {
            "indicator_layer": "clean_basic_indicators_with_renko_v2_no_data_store_dependency_chunked",
            "rows": int(total_rows),
            "columns": list(final_columns),
            "market_cache_dir": str(self.market_cache_dir),
            "indicator_cache_path": str(self.indicator_cache_path),
            "n1": int(n1),
            "n2": int(n2),
            "ma_windows": [int(x) for x in ma_windows],
            "volume_windows": [int(x) for x in volume_windows],
            "macd": {"fast": int(macd_fast), "slow": int(macd_slow), "signal": int(macd_signal)},
            "renko_cached_columns": ["renko_value"],
            "strategy_specific_columns_excluded": True,
            "chunked_write": True,
            "part_count": int(len(part_paths)),
        }
        meta_path = self.indicator_cache_path.with_suffix(self.indicator_cache_path.suffix + ".meta.json")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Basic indicator meta saved: {meta_path}", flush=True)

        return out
