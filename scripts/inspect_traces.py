"""
scripts/inspect_traces.py
Print the last telemetry timestamp and its age (seconds).
"""

import sys
import time
from pathlib import Path

try:
    import pandas as pd
except Exception as e:
    print("pandas not installed (pip install pandas)", file=sys.stderr)
    raise

p = Path("traces")
files = sorted(p.glob("telemetry_*.parquet"))
if not files:
    print("No telemetry parquet files found in traces/")
    sys.exit(2)

df = pd.read_parquet(files[-1])
if df.empty:
    print("Telemetry file is empty")
    sys.exit(2)

ts = float(df["timestamp"].iloc[-1])
age = time.time() - ts
print(f"Latest telemetry file: {files[-1]}")
print(f"Last timestamp: {ts} (unix epoch)")
print(f"Age: {age:.2f} seconds")
if age < 5.0:
    print("Status: ONLINE")
elif age < 30.0:
    print("Status: STALE")
else:
    print("Status: OFFLINE")
