import faulthandler
import sys

faulthandler.enable()
faulthandler.dump_traceback_later(10, repeat=False)

print("before import pandas", flush=True)
import pandas as pd
print("pandas imported", pd.__version__, flush=True)
