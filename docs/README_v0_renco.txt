v0_renco 选股策略说明
====================

本版本新增 selection strategy：v0_renco。

实现公式：
《砖图选股公式（优化红柱长度+连续两天站多空线）》

核心规则：
1. 砖型图核心计算：VAR1A ~ VAR6A 完全按公式实现。
2. 绿转红：REF(AA,1)=0 AND AA=1。
3. 有效红柱：砖型图 > 0。
4. 有效绿柱：前一根为有效绿柱，且前绿柱高 > 0。
5. 长度条件：当天红柱高 > 前绿柱高 * 0.7。
6. 股价区间：连续两天站上多空线，且收盘价 < 短趋势 * 1.02。
7. 趋势条件：短趋势 > 多空线。
8. 价格条件：收盘价 < 50。

运行单日：
python selector.py --date 2025-04-21 --strategy v0_renco

运行区间：
python selector.py --start-date 2025-01-01 --end-date 2025-12-31 --strategy v0_renco

输出股票池文件：
pools/v0_renco_YYYY-MM-DD.csv
