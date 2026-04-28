Unified pool build refactor
===========================

Goal
----
This version keeps the strategy names based on renko_chart_select_strategy_v0 and changes the pool-building pipeline from many daily CSV files to a reusable cache + single pool file design.

New pipeline
------------
1. TXT raw files
   - Keep original TDX TXT files in data/.
   - TXT parsing is no longer repeated by every strategy run.

2. Market data cache
   - Command:
     python scripts/import_tdx_txt.py --txt-dir data --market-cache-dir data/market_cache/daily_bars_by_symbol
   - Output:
     data/market_cache/daily_bars_by_symbol/<SYMBOL>.parquet
   - Purpose:
     each symbol has one cached daily-bar file. Daily updates only need to import changed TXT files.

3. Indicator cache
   - Command:
     python scripts/build_indicators.py --txt-dir data --market-cache-dir data/market_cache/daily_bars_by_symbol --indicator-cache-path data/indicator_cache/daily_indicators.parquet
   - Output:
     data/indicator_cache/daily_indicators.parquet
   - Purpose:
     all reusable indicators are calculated once and reused by future selection strategies.

4. Unified pool file
   - Command:
     python selector.py --start-date 2025-01-01 --end-date 2025-12-31 --strategy renko_chart_select_strategy_v0 --overwrite
   - Output:
     pools/renko_chart_select_strategy_v0_pool.parquet
   - Purpose:
     all selected rows for all dates are stored in one file, instead of creating hundreds of daily CSV files.

5. Backtest
   - Command:
     python main.py --start-date 2025-01-01 --end-date 2025-12-31 --selection-strategy renko_chart_select_strategy_v0 --trade-strategy renko_chart_select_strategy_v0_trade_v4_b_t3_close
   - Backtest reads the unified pool file. If the pool/cache is missing, it will build the missing part automatically.

Interface rules for future maintenance
--------------------------------------
- indicators/ only calculates reusable facts. It should not decide whether a stock is selected.
- selection_strategies/ decides hard filters, scores, thresholds, and selected=1/0.
- trade_strategies/ receives the same interface as before: df, signal_date, capital_alloc, lot_size, fee params, file_name, trade_strategy, pool_row, and pool columns.
- data_store.py is the only layer that should know how to read raw TXT/cache files.
- pool_store.py is the only layer that should know how pool rows are stored.

Compatibility notes
-------------------
- The code tries to write parquet first. If pyarrow/fastparquet is not installed, it falls back to pandas pickle under the same file path. Use the provided read_table/write_table helpers instead of pd.read_parquet directly.
- Existing trade strategy names are kept as renko_chart_select_strategy_v0_trade_*.
- risk.py and scoring.py were not introduced or changed.
