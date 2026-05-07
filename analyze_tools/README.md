# Analyze Tools

This folder contains analysis-only scripts.

## Compare N pool quality

```powershell
python .\analyze_tools\compare_n_pools_quality.py `
  --pool-paths "C:\Users\zyf37\Desktop\BackTest Data\pools\renko_chart_select_strategy_v0_pool.parquet" "C:\Users\zyf37\Desktop\BackTest Data\pools\renko_chart_select_strategy_v4_pool.parquet" `
  `
  `
  --start-date 2024-01-01 `
  --end-date 2026-04-30
```

Outputs:
- `compare_n_pools_summary_*.csv`
- `compare_n_pools_detail_*.csv`

Main metrics include:
- T+1 open gap from T0 close
- T+2 close return from T+1 open
- T+3 close return from T+1 open
- win rate, 2% hit rate, -2% loss rate
- T+1 to T+3 max high / drawdown from T+1 open
