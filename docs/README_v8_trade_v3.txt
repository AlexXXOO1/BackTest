v8_trade_v3 说明

本版本新增 trade_strategies/v8_trade_v3.py，并在 registry.py 中注册：v8_trade_v3。
main.py 已默认设置：
SELECTION_STRATEGY = "v8"
TRADE_STRATEGY = "v8_trade_v3"

核心规则：
1. 只买 60 <= score_pct <= 90 的标的。
2. 必须 条件9_涨幅小红砖长 = True。
3. T+1 开盘涨幅必须在 -2% 到 +0.5%。
4. 不再买 score_pct 最高，而是按 T+1 开盘形态健康度排序。
5. T+1 开盘买入，T+2 收盘卖出。

开盘健康度排序：
1. 0% 到 +0.5% 优先。
2. -0.5% 到 0% 次之。
3. -1% 到 -0.5% 再次。
4. -2% 到 -1% 最后。
5. 同一个区间内，选择更接近 +0.2% 的标的。

运行：
python main.py
