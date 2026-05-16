import time
from pathlib import Path

print("step1: before import pandas", flush=True)
import pandas as pd
print("step2: pandas imported", pd.__version__, flush=True)

print("step3: before import pyarrow", flush=True)
import pyarrow.parquet as pq
print("step4: pyarrow imported", flush=True)

p = Path(r"C:\Users\zyf37\Desktop\BackTest_Data\pools\b1_stage_low_select_strategy_v0_pool.parquet")

pf = pq.ParquetFile(p)
print("step5: metadata ok", flush=True)

table = pf.read(columns=["symbol", "date", "selection_strategy", "amplitude_pct", "fwd_return_pct_T1"])
print("step6: arrow table read ok", table.num_rows, table.num_columns, flush=True)

t0 = time.time()
df = table.to_pandas()
print("step7: to_pandas ok", df.shape, "seconds:", round(time.time() - t0, 2), flush=True)
print(df.dtypes, flush=True)
