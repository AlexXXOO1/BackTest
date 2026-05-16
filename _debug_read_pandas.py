from pathlib import Path
import time
import pandas as pd

p = Path(r"C:\Users\zyf37\Desktop\BackTest_Data\pools\b1_stage_low_select_strategy_v0_pool.parquet")

cols = [
    "symbol",
    "date",
    "selection_strategy",
    "amplitude_pct",
    "fwd_return_pct_T1",
]

print("start")
t0 = time.time()
df = pd.read_parquet(p, columns=cols, engine="pyarrow")
print("pandas_read_ok:", df.shape, "seconds:", round(time.time() - t0, 2))
print(df.dtypes)
print(df.head())
