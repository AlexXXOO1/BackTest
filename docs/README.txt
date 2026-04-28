项目结构说明
====================

当前目录按模块拆分：
1. indicators.py
   - 只放基础指标函数
2. selection_strategies/
   - 每个选股策略一个文件
   - registry.py 统一注册
3. trade_strategies/
   - 每个交易策略一个文件
   - registry.py 统一注册
4. selector.py
   - 负责扫描全市场并生成 pools 下的股票池
5. strategy.py
   - 负责统一调用交易策略模块
6. main.py
   - 主回测入口
   - 自动检查股票池，不存在则调用 selector.py 生成

当前已注册选股策略：
- v1
- v2
- v3

当前已注册交易策略：
- t1_open_t2_up_sell_else_t3_close
- manual_t1_open_green_brick_else_t3_close
- t1_open_hold_red_until_green_t1_sell


新增策略
====================
v4 = v3 + MACD DIFF 白线在零轴上方。


本次更新
====================
1. main.py 会把 --date / --strategy / --n1 / --n2 参数传给 selector.py。
2. selector.py 生成股票池时，不再直接存到 pools 根目录。
3. 每个交易日会先创建一个“策略名_日期”的子文件夹，例如：
   pools/v4_2026-03-05/
4. 当天股票池文件保存在子文件夹中，例如：
   pools/v4_2026-03-05/v4_2026-03-05.csv


补充
====================
股票池恢复为直接写入 pools 根目录，不再为每个日期单独创建子文件夹。
文件命名仍为：策略名_日期.csv，例如 v4_2026-03-05.csv
