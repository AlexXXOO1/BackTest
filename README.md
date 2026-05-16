# BackTest_App

New-architecture-only project package for selection-pool construction, pool schema validation, factor forward-return validation, and Streamlit UI.

## Directory layout

```text
BackTest_App/
  app/ui/                         Streamlit UI only
  analysis/                       factor validation and analysis logic
  core/                           config, path manager, pool schema, validators
  data_engine/                    data import, indicator cache, full-market cache
  data_engine/indicators/         reusable indicator calculation
  strategies/selection/           selection strategy logic
  scripts/                        command-line entry points
  configs/settings.json           code/data path configuration
  tests/                          schema smoke tests
```

Removed old-architecture compatibility code:

```text
selection_strategies/
trade_strategies/
indicators/
legacy/
root config.py
strategies/trade/
```

## Fixed paths

Code root:

```text
C:\Users\zyf37\Desktop\BackTest_App
```

Data root:

```text
C:\Users\zyf37\Desktop\BackTest_Data
```

Large data is not packaged into the app. The app reads and writes to `BackTest_Data` through `core/path_manager.py` and `configs/settings.json`.

## Pool contract

The final saved pool must pass `core/pool_schema.py` before parquet output.

Required core columns:

```text
symbol
date
selection_strategy
open
high
low
close
volume
amount
t1_date
t1_open
t1_close
t2_date
t2_open
t2_close
t3_date
t3_open
t3_close
t4_date
t4_open
t4_close
fwd_return_pct_T1
fwd_return_pct_T2
fwd_return_pct_T3
fwd_return_pct_T4
```

Removed score columns are not allowed in final pool output:

```text
selected
selected_score_base
score
score_rank_key
score_pct
```

Factor columns are dynamic. Any valid numeric strategy-specific column that is not a market absolute value and not a future/forward target can be detected by analysis tools.


## Current finalized dashboard structure

Run main dashboard:

```powershell
streamlit run .\app\ui\pool_dashboard.py
```

Pages:

```text
1. Single Pool Viewer
   - Formal pool viewer. Formal pools should contain selected == 1 only.
   - If a selected column exists, the page automatically filters selected == 1.
   - Default row order preserves the source pool order.
   - Sort by is retained as an optional manual interface for later score/rank work.

2. Analyze Pool Indicator
   - Runs and displays fwd_return_pct_T1, fwd_return_pct_T2, fwd_return_pct_T3, and fwd_return_pct_T4 together when available.
   - Keeps Bucket count and Min samples controls.
   - Uses global quantile buckets. It does not use per-date buckets or min-max equal-width buckets.
   - Removes best bucket and worst bucket display, but keeps best-minus-worst return and bucket up-ratio difference.

3. Single Factor Analysis
   - Select one factor and inspect T+1, T+2, and T+3 bucket behavior together.
   - Shows bucket interval, sample count, mean return, median return, up ratio, win count, and loss count.
   - Keeps bucket mean return, bucket up ratio, and factor-value-by-bucket charts.
   - Removes best bucket, worst bucket, and best-worst return from the page.

4. Multi-Factor Combination Test
   - Select one or more factors.
   - Select one or more buckets for each selected factor.
   - Uses AND logic across factor bucket conditions.
   - Runs only after clicking Run.
   - Uses Strategy Pool Baseline as the benchmark.
   - Outputs summary and matched detail rows.
```

## Main commands

Build B1 stage-low pool:

```powershell
python .\scripts\build_pool.py --strategy b1_stage_low_select_strategy_v0 --no-csv
```

Build Renko v0 pool:

```powershell
python .\scripts\build_pool.py --strategy renko_chart_select_strategy_v0 --no-csv
```

Run main dashboard:

```powershell
streamlit run .\app\ui\pool_dashboard.py
```

Build indicators:

```powershell
python .\scripts\build_indicators.py
```

Import TDX TXT market data:

```powershell
python .\scripts\import_tdx_txt.py --fix-encoding --overwrite
```
