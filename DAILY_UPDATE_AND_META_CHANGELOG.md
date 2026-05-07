# P1 Change Log - daily_update.py and cache meta

## Added

### `scripts/daily_update.py`

One-command daily workflow:

```powershell
python .\scripts\daily_update.py --date 2026-05-07 --strategies renko_chart_select_strategy_v4 b2_confirm_select_strategy_v0
```

This runs:

1. import TDX TXT into market cache
2. build/update indicator cache
3. build/update strategy pool files

Useful variants:

```powershell
# Backfill or rebuild a date range
python .\scripts\daily_update.py --start-date 2026-04-01 --end-date 2026-05-07 --strategies renko_chart_select_strategy_v4 --rebuild-pools

# Rebuild indicators after indicator formula changes
python .\scripts\daily_update.py --start-date 2020-01-01 --end-date 2026-05-07 --strategies renko_chart_select_strategy_v4 --rebuild-indicators --rebuild-pools

# Full rebuild after raw data / adjustment changes
python .\scripts\daily_update.py --start-date 2020-01-01 --end-date 2026-05-07 --strategies renko_chart_select_strategy_v4 --full-rebuild

# Skip TXT import when market cache is already updated
python .\scripts\daily_update.py --date 2026-05-07 --strategies renko_chart_select_strategy_v4 --skip-import

# List registered strategies
python .\scripts\daily_update.py --list-strategies
```

The pool update in `daily_update.py` replaces only the requested date range inside the existing pool file. This is safer than append-only because a stock that was selected before but is no longer selected will be removed for that date range.

## Added

### `core/cache_meta.py`

Central helper for writing and reading cache metadata.

It writes a `.meta.json` next to each cache file:

```text
daily_indicators.parquet
daily_indicators.meta.json

renko_chart_select_strategy_v4_pool.parquet
renko_chart_select_strategy_v4_pool.meta.json
```

## Modified

### `core/indicator_store.py`

After `daily_indicators.parquet` is written, the code now also writes:

```text
daily_indicators.meta.json
```

Recorded fields include:

- `cache_type`
- `created_at`
- `data_root`
- `market_cache_dir`
- `indicator_cache_path`
- `indicator_version`
- `indicator_source_files`
- `n1`
- `n2`
- `requested_start_date`
- `requested_end_date`
- `incremental`
- `lookback_days`
- `rows`
- `columns`
- `min_date`
- `max_date`

### `scripts/selector.py`

When selector CLI saves a pool, it now also writes pool metadata.

Recorded fields include:

- `cache_type`
- `created_at`
- `strategy`
- `strategy_version`
- `required_indicators`
- `indicator_cache_path`
- `indicator_meta_path`
- `indicator_version`
- `market_cache_dir`
- `pools_dir`
- `n1`
- `n2`
- `start_date`
- `end_date`
- `rows`
- `columns`
- `min_date`
- `max_date`

### `core/pool_store.py`

Legacy `scripts/build_pool.py` path also writes pool meta after saving.

## Notes

- If a strategy module defines `STRATEGY_VERSION`, that value is used.
- If not, `strategy_version` falls back to a short SHA256 hash of the strategy file content.
- `indicator_version` is a short SHA256 hash of all `.py` files under `indicators/`.
