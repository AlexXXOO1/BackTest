# Path Unification Changelog

本次只完成第一步：统一运行数据路径，不改指标逻辑、不改策略逻辑、不改回测逻辑。

## 统一后的默认路径

所有主流程脚本默认从 `config.py` 的 `BacktestConfig` 读取路径：

```text
DATA_ROOT            = C:\Users\zyf37\Desktop\BackTest Data
RAW_TDX_TXT_DIR      = DATA_ROOT / "data"
MARKET_CACHE_DIR     = DATA_ROOT / "market_cache" / "daily_bars_by_symbol"
INDICATOR_CACHE_PATH = DATA_ROOT / "indicator_cache" / "daily_indicators.parquet"
POOLS_DIR            = DATA_ROOT / "pools"
OUTPUT_DIR           = DATA_ROOT / "output"
```

## 已修改文件

- `config.py`
  - 新增统一路径常量。
  - 支持通过环境变量 `BACKTEST_DATA_ROOT` 切换数据根目录。
  - 修正旧默认路径中 `DATA_ROOT / "data" / "market_cache"` 和 `DATA_ROOT / "data" / "indicator_cache"` 的混用问题。

- `scripts/selector.py`
  - 默认 `market_cache_dir`、`indicator_cache_path`、`pools_dir` 改为从 `BacktestConfig()` 读取。

- `scripts/checkpool.py`
  - 默认 TXT 目录和输出目录改为从 `config.py` 读取。

- `core/data_store.py`
  - `MarketDataStore` 默认 TXT 目录和 market cache 目录改为从 `config.py` 读取。

- `core/indicator_store.py`
  - `IndicatorStore` 默认 indicator cache 路径改为从 `config.py` 读取。

- `core/pool_store.py`
  - `PoolStore` 默认 pools 目录改为从 `config.py` 读取。

- `analyze_tools/compare_n_pools_quality.py`
  - 默认 market cache 和输出目录改为从 `config.py` 读取。

- `analyze_tools/analyze_market_regime.py`
  - `--market-cache-dir` 从必填改为默认读取 `config.MARKET_CACHE_DIR`。

- `selection_strategies/renko_chart_select_strategy_v5.py`
  - 默认大盘指数 TXT 路径改为 `RAW_TDX_TXT_DIR / "SH#999999.txt"`。

- `models/train_xgb_b2_t2_return.py`
  - 默认输入/输出路径改为从 `OUTPUT_DIR` 派生。

- README 文件
  - 更新了路径说明，减少硬编码长路径命令。

## 后续日常命令

以后正常情况下可以少写路径参数：

```powershell
python scripts/import_tdx_txt.py
python scripts/build_indicators.py
python .\scripts\selector.py --start-date 2026-05-07 --end-date 2026-05-07 --strategy renko_chart_select_strategy_v4 --overwrite --debug-summary
```

## 换数据根目录

不要逐个改脚本，设置环境变量即可：

```powershell
$env:BACKTEST_DATA_ROOT = "D:\BackTest Data"
```

如需永久生效，可以在 Windows 系统环境变量中新增 `BACKTEST_DATA_ROOT`。
