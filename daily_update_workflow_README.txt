每日大盘数据更新流程 README

#1. 适用场景

每天更新完通达信导出的 A 股大盘 TXT 数据后，需要固定运行以下流程：

1. 导入 TXT 原始数据，生成 / 更新 market cache
2. 基于 market cache 重新生成 indicator cache
3. 运行 selector，生成指定日期的选股池 pool

本 README 中所有命令均使用绝对路径，避免因为相对路径导致读取错数据。

---

#2. 固定数据目录

当前统一使用的数据根目录：

C:\Users\zyf37\Desktop\BackTest Data

##2.1 原始 TXT 数据目录

C:\Users\zyf37\Desktop\BackTest Data\data

##2.2 行情缓存目录 market cache

C:\Users\zyf37\Desktop\BackTest Data\market_cache\daily_bars_by_symbol

##2.3 指标缓存文件 indicator cache

C:\Users\zyf37\Desktop\BackTest Data\indicator_cache\daily_indicators.parquet

##2.4 选股池输出目录 pools

C:\Users\zyf37\Desktop\BackTest Data\pools

---

#3. 每日固定运行流程

每天更新完 TXT 数据后，建议按顺序运行下面三条命令。

---

#第一步：导入 TXT 数据到 market cache

python scripts/import_tdx_txt.py --txt-dir "C:\Users\zyf37\Desktop\BackTest Data\data" --market-cache-dir "C:\Users\zyf37\Desktop\BackTest Data\market_cache\daily_bars_by_symbol"

##作用

这一步会读取：

C:\Users\zyf37\Desktop\BackTest Data\data

目录下的通达信 TXT 文件，并生成 / 更新每只股票对应的 parquet 行情缓存。

输出目录为：

C:\Users\zyf37\Desktop\BackTest Data\market_cache\daily_bars_by_symbol

---

#第二步：重新生成 indicator cache

python scripts/build_indicators.py --txt-dir "C:\Users\zyf37\Desktop\BackTest Data\data" --market-cache-dir "C:\Users\zyf37\Desktop\BackTest Data\market_cache\daily_bars_by_symbol" --indicator-cache-path "C:\Users\zyf37\Desktop\BackTest Data\indicator_cache\daily_indicators.parquet"

##作用

这一步会基于最新的 market cache 重新计算所有指标，并写入：

C:\Users\zyf37\Desktop\BackTest Data\indicator_cache\daily_indicators.parquet

常见指标包括：

- 砖型图指标
- KDJ / J 值
- 趋势线相关指标
- 涨幅小但红砖长
- 放量上涨后缩量回调
- 其他选股策略需要使用的指标

---

#第三步：运行 selector 生成选股池

##3.1 生成单一日期的选股池

如果只想生成某一天的选股池，例如 `2026-04-29`，运行：

python .\scripts\selector.py --start-date 2026-04-29 --end-date 2026-04-29 --strategy renko_chart_select_strategy_v0 --market-cache-dir "C:\Users\zyf37\Desktop\BackTest Data\market_cache\daily_bars_by_symbol" --indicator-cache-path "C:\Users\zyf37\Desktop\BackTest Data\indicator_cache\daily_indicators.parquet" --pools-dir "C:\Users\zyf37\Desktop\BackTest Data\pools" --overwrite --debug-summary

##3.2 补跑多个日期的选股池

如果想补跑一个区间，例如 `2026-04-28` 到 `2026-04-29`，运行：

python .\scripts\selector.py --start-date 2026-04-28 --end-date 2026-04-30 --strategy renko_chart_select_strategy_v1 --market-cache-dir "C:\Users\zyf37\Desktop\BackTest Data\market_cache\daily_bars_by_symbol" --indicator-cache-path "C:\Users\zyf37\Desktop\BackTest Data\indicator_cache\daily_indicators.parquet" --pools-dir "C:\Users\zyf37\Desktop\BackTest Data\pools" --overwrite --debug-summary

##3.3 更换策略名称

如果要运行其他选股策略，只需要修改：

--strategy renko_chart_select_strategy_v0

例如：

--strategy renko_chart_select_strategy_v1

或：

--strategy renko_chart_select_strategy_v2

---

#4. 推荐每日完整命令模板

假设今天更新的是 `2026-04-29` 的数据，每天完整运行下面三条：

python scripts/import_tdx_txt.py --txt-dir "C:\Users\zyf37\Desktop\BackTest Data\data" --market-cache-dir "C:\Users\zyf37\Desktop\BackTest Data\market_cache\daily_bars_by_symbol"

python scripts/build_indicators.py --txt-dir "C:\Users\zyf37\Desktop\BackTest Data\data" --market-cache-dir "C:\Users\zyf37\Desktop\BackTest Data\market_cache\daily_bars_by_symbol" --indicator-cache-path "C:\Users\zyf37\Desktop\BackTest Data\indicator_cache\daily_indicators.parquet"

python .\scripts\selector.py --start-date 2026-04-29 --end-date 2026-04-29 --strategy renko_chart_select_strategy_v0 --market-cache-dir "C:\Users\zyf37\Desktop\BackTest Data\market_cache\daily_bars_by_symbol" --indicator-cache-path "C:\Users\zyf37\Desktop\BackTest Data\indicator_cache\daily_indicators.parquet" --pools-dir "C:\Users\zyf37\Desktop\BackTest Data\pools" --overwrite --debug-summary

---

#5. 数据流转关系

通达信 TXT 原始数据
        ↓
scripts/import_tdx_txt.py
        ↓
C:\Users\zyf37\Desktop\BackTest Data\market_cache\daily_bars_by_symbol
        ↓
scripts/build_indicators.py
        ↓
C:\Users\zyf37\Desktop\BackTest Data\indicator_cache\daily_indicators.parquet
        ↓
scripts/selector.py
        ↓
C:\Users\zyf37\Desktop\BackTest Data\pools

---

#6. 常见问题检查

##6.1 selector 结果不对

优先检查三类路径是否完全一致：

--market-cache-dir
--indicator-cache-path
--pools-dir

不要混用相对路径和绝对路径。

错误示例：

data\market_cache\daily_bars_by_symbol

正确示例：

C:\Users\zyf37\Desktop\BackTest Data\market_cache\daily_bars_by_symbol

---

##6.2 更新 TXT 后是否必须重新跑 build_indicators.py？

必须重新跑。

因为 selector 读取的是：

C:\Users\zyf37\Desktop\BackTest Data\indicator_cache\daily_indicators.parquet

如果只更新 TXT 和 market cache，但没有重新生成 indicator cache，selector 读到的指标可能还是旧数据。

---

##6.3 今天没有选出股票怎么办？

先检查：

1. 当天 TXT 数据是否已经放入 `C:\Users\zyf37\Desktop\BackTest Data\data`
2. 是否运行了 `import_tdx_txt.py`
3. 是否运行了 `build_indicators.py`
4. selector 的 `--start-date` 和 `--end-date` 是否写对
5. `--market-cache-dir` 是否指向绝对路径
6. `--indicator-cache-path` 是否指向绝对路径
7. 当前策略当天是否真的没有命中

---

#7. 最终结论

每天更新完大盘数据后，固定运行：

1. scripts/import_tdx_txt.py
2. scripts/build_indicators.py
3. scripts/selector.py

核心原则：

全部路径统一使用 C:\Users\zyf37\Desktop\BackTest Data 下的绝对路径
