$ErrorActionPreference = "Stop"

$env:PYTHONPATH = (Get-Location).Path

python .\ops\daily_update\import_tdx_txt.py --fix-encoding --overwrite
python .\ops\daily_update\build_indicators.py
python .\ops\daily_update\build_pool.py --strategy b1_stage_low_select_strategy_v0 --incremental --incremental-refresh-days 45 --no-csv
