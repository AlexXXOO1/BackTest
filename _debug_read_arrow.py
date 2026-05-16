from pathlib import Path
import time
import pyarrow.parquet as pq

p = Path(r"C:\Users\zyf37\Desktop\BackTest_Data\pools\b1_stage_low_select_strategy_v0_pool.parquet")

print("start")
t0 = time.time()

pf = pq.ParquetFile(p)
print("metadata_ok:", pf.metadata.num_rows, pf.metadata.num_columns, "seconds:", round(time.time() - t0, 2))

t1 = time.time()
table = pf.read(columns=["amplitude_pct"])
print("arrow_read_ok:", table.num_rows, table.num_columns, "seconds:", round(time.time() - t1, 2))
print(table.schema)
