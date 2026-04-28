from __future__ import annotations

from pathlib import Path
import pandas as pd


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a dataframe with parquet first and pickle fallback.

    The project prefers parquet for speed and stable dtypes. If pyarrow/fastparquet is
    not installed, the same path is written with pandas pickle so the rest of the
    project can still run without extra installation steps.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_pickle(path)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a dataframe written by write_table."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_pickle(path)
