from __future__ import annotations

from pathlib import Path
from typing import Iterable
import pandas as pd

from core.progress import progress_bar
from core.storage import read_table, write_table
from utils import is_main_board_txt, is_st_txt, read_tdx_export_txt
from config import RAW_TDX_TXT_DIR, MARKET_CACHE_DIR

BAR_COLUMNS = ["symbol", "file", "date", "open", "high", "low", "close", "volume", "amount"]


def symbol_from_file(file_path: str | Path) -> str:
    return Path(file_path).stem.upper()


class MarketDataStore:
    """Unified market data access layer.

    Trading and indicator code should read daily bars through this class instead of
    parsing TDX TXT files directly. The cache is stored one file per symbol to keep
    incremental updates simple.
    """

    def __init__(self, txt_dir: str | Path = RAW_TDX_TXT_DIR, market_cache_dir: str | Path = MARKET_CACHE_DIR) -> None:
        self.txt_dir = Path(txt_dir)
        self.market_cache_dir = Path(market_cache_dir)
        self.market_cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, symbol: str) -> Path:
        safe_symbol = str(symbol).replace("/", "_").replace("\\", "_")
        return self.market_cache_dir / f"{safe_symbol}.parquet"

    def list_cached_symbols(self) -> list[str]:
        return sorted(p.stem for p in self.market_cache_dir.glob("*.parquet"))

    def list_txt_files(self) -> list[Path]:
        return sorted(p for p in self.txt_dir.glob("*.txt") if is_main_board_txt(p) and not is_st_txt(p))

    def import_txt_files(self, start_date=None, end_date=None, overwrite: bool = False, show_progress: bool = True) -> dict:
        start_ts = pd.Timestamp(start_date) if start_date is not None else None
        end_ts = pd.Timestamp(end_date) if end_date is not None else None
        report = {"total_txt": 0, "imported": 0, "failed": 0, "skipped": 0, "rows": 0, "failures": []}

        txt_files = self.list_txt_files()
        iterator = progress_bar(txt_files, desc="Import TDX TXT", total=len(txt_files)) if show_progress else txt_files

        for file_path in iterator:
            report["total_txt"] += 1
            symbol = symbol_from_file(file_path)
            cache_path = self.cache_path(symbol)
            try:
                df = read_tdx_export_txt(file_path, end_date=end_ts)
                if start_ts is not None and not df.empty:
                    df = df[df["date"] >= start_ts]
                if df.empty:
                    report["skipped"] += 1
                    continue
                df = df.copy()
                df["symbol"] = symbol
                df["file"] = file_path.name
                df = df[BAR_COLUMNS].sort_values("date").drop_duplicates(["symbol", "date"], keep="last")

                if cache_path.exists() and not overwrite:
                    old = read_table(cache_path)
                    if not old.empty:
                        old["date"] = pd.to_datetime(old["date"])
                        df = pd.concat([old, df], ignore_index=True)
                        df = df.sort_values("date").drop_duplicates(["symbol", "date"], keep="last")

                write_table(df, cache_path)
                report["imported"] += 1
                report["rows"] += int(len(df))
            except Exception as exc:
                report["failed"] += 1
                report["failures"].append({"file": file_path.name, "error": str(exc)})
        return report

    def get_symbol_data(self, symbol_or_file: str, end_date=None) -> pd.DataFrame:
        symbol = Path(str(symbol_or_file)).stem.upper()
        cache_path = self.cache_path(symbol)
        if cache_path.exists():
            df = read_table(cache_path)
        else:
            txt_path = self.txt_dir / (str(symbol_or_file) if str(symbol_or_file).lower().endswith(".txt") else f"{symbol}.txt")
            df = read_tdx_export_txt(txt_path, end_date=end_date)
            if not df.empty:
                df["symbol"] = symbol
                df["file"] = txt_path.name
                df = df[BAR_COLUMNS]
        if df.empty:
            return pd.DataFrame(columns=BAR_COLUMNS)
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        if end_date is not None:
            df = df[df["date"] <= pd.Timestamp(end_date)]
        return df.sort_values("date").reset_index(drop=True)

    def iter_symbol_data(self, symbols: Iterable[str] | None = None, end_date=None, show_progress: bool = False):
        symbols = self.list_cached_symbols() if symbols is None else list(symbols)
        iterator = progress_bar(symbols, desc="Read symbol cache", total=len(symbols)) if show_progress else symbols
        for symbol in iterator:
            df = self.get_symbol_data(symbol, end_date=end_date)
            if not df.empty:
                yield symbol, df

    def get_trade_dates(self, start_date, end_date) -> list[pd.Timestamp]:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        dates: set[pd.Timestamp] = set()
        symbols = self.list_cached_symbols()
        if symbols:
            for symbol in progress_bar(symbols, desc="Collect trade dates", total=len(symbols)):
                df = self.get_symbol_data(symbol)
                if df.empty:
                    continue
                cur_dates = pd.to_datetime(df["date"])
                dates.update(pd.Timestamp(x).normalize() for x in cur_dates[(cur_dates >= start_ts) & (cur_dates <= end_ts)])
        else:
            txt_files = self.list_txt_files()
            for file_path in progress_bar(txt_files, desc="Collect trade dates from TXT", total=len(txt_files)):
                df = read_tdx_export_txt(file_path, end_date=end_ts)
                if df.empty:
                    continue
                cur_dates = pd.to_datetime(df["date"])
                dates.update(pd.Timestamp(x).normalize() for x in cur_dates[(cur_dates >= start_ts) & (cur_dates <= end_ts)])
        return sorted(dates)
