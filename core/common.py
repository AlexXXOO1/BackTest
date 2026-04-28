from __future__ import annotations

import json
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def extract_code(text: str) -> str:
    m = re.search(r"(\d{6})", str(text))
    return m.group(1) if m else ""


def to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) == 1.0
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def make_json_safe(obj: Any) -> Any:
    """Convert pandas/numpy/python objects into JSON-serializable values."""
    if obj is None:
        return None

    if obj is pd.NA or obj is pd.NaT:
        return None

    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value

    if isinstance(obj, (int, str)):
        return obj

    if isinstance(obj, np.ndarray):
        return [make_json_safe(x) for x in obj.tolist()]

    if isinstance(obj, pd.Series):
        return {str(k): make_json_safe(v) for k, v in obj.to_dict().items()}

    if isinstance(obj, pd.DataFrame):
        return [make_json_safe(row) for row in obj.to_dict(orient="records")]

    if isinstance(obj, dict):
        return {str(make_json_safe(k)): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x) for x in obj]

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return str(obj)


def save_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_data = make_json_safe(data)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(safe_data, f, ensure_ascii=False, indent=2)


def get_trade_dates_in_range(txt_dir: Path, start_date: pd.Timestamp, end_date: pd.Timestamp, read_func) -> list[pd.Timestamp]:
    for fp in sorted(txt_dir.glob("*.txt")):
        df = read_func(fp)
        if df.empty:
            continue
        dates = pd.to_datetime(df["date"]).drop_duplicates().sort_values()
        dates = dates[(dates >= start_date) & (dates <= end_date)]
        if len(dates) > 0:
            return list(dates)
    return []
