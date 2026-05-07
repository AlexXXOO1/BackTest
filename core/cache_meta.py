from __future__ import annotations

import hashlib
import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from config import DATA_ROOT, MARKET_CACHE_DIR, INDICATOR_CACHE_PATH, POOLS_DIR, ROOT_DIR


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def meta_path_for_cache(cache_path: str | Path) -> Path:
    path = Path(cache_path)
    return path.with_suffix(".meta.json")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def write_json_meta(meta: dict[str, Any], cache_path: str | Path) -> Path:
    meta_path = meta_path_for_cache(cache_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return meta_path


def read_json_meta(cache_path: str | Path) -> dict[str, Any]:
    meta_path = meta_path_for_cache(cache_path)
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def hash_files(files: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(files, key=lambda p: str(p).lower()):
        if not path.exists() or not path.is_file():
            continue
        h.update(str(path.relative_to(ROOT_DIR) if path.is_relative_to(ROOT_DIR) else path).encode("utf-8", errors="ignore"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def indicator_source_files() -> list[Path]:
    indicators_dir = ROOT_DIR / "indicators"
    if not indicators_dir.exists():
        return []
    return [p for p in indicators_dir.rglob("*.py") if "__pycache__" not in p.parts]


def indicator_version() -> str:
    files = indicator_source_files()
    return f"ind_{hash_files(files)}" if files else "ind_unknown"


def strategy_module_from_func(strategy_func: Callable):
    module_name = getattr(strategy_func, "__module__", None)
    if not module_name:
        return None
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def strategy_required_indicators(strategy_func: Callable) -> list[str]:
    module = strategy_module_from_func(strategy_func)
    if module is None:
        return []
    required = (
        getattr(module, "REQUIRED_INDICATORS", None)
        or getattr(module, "REQUIRED_INDICATOR_COLUMNS", None)
        or []
    )
    try:
        return sorted(str(x) for x in required)
    except Exception:
        return []


def strategy_version(strategy_name: str, strategy_func: Callable) -> str:
    module = strategy_module_from_func(strategy_func)
    if module is None:
        return f"{strategy_name}_unknown"

    explicit = getattr(module, "STRATEGY_VERSION", None)
    if explicit:
        return str(explicit)

    module_file = getattr(module, "__file__", None)
    if module_file:
        path = Path(module_file)
        if path.exists():
            return f"{strategy_name}_{hash_files([path])}"

    return f"{strategy_name}_unknown"


def dataframe_date_bounds(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if df.empty or "date" not in df.columns:
        return None, None
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return None, None
    return str(dates.min().date()), str(dates.max().date())


def build_indicator_meta(
    *,
    df: pd.DataFrame,
    n1: int,
    n2: int,
    start_date: Any = None,
    end_date: Any = None,
    incremental: bool = False,
    lookback_days: int | None = None,
    market_cache_dir: str | Path = MARKET_CACHE_DIR,
    indicator_cache_path: str | Path = INDICATOR_CACHE_PATH,
) -> dict[str, Any]:
    min_date, max_date = dataframe_date_bounds(df)
    files = indicator_source_files()
    return {
        "cache_type": "indicator_cache",
        "created_at": now_str(),
        "data_root": str(DATA_ROOT),
        "market_cache_dir": str(market_cache_dir),
        "indicator_cache_path": str(indicator_cache_path),
        "indicator_version": indicator_version(),
        "indicator_source_files": [str(p.relative_to(ROOT_DIR)) for p in files],
        "n1": n1,
        "n2": n2,
        "requested_start_date": str(start_date) if start_date is not None else None,
        "requested_end_date": str(end_date) if end_date is not None else None,
        "incremental": bool(incremental),
        "lookback_days": lookback_days,
        "rows": int(len(df)),
        "columns_count": int(len(df.columns)),
        "columns": [str(c) for c in df.columns],
        "min_date": min_date,
        "max_date": max_date,
    }


def build_pool_meta(
    *,
    pool: pd.DataFrame,
    strategy_name: str,
    strategy_func: Callable,
    start_date: Any,
    end_date: Any,
    n1: int,
    n2: int,
    market_cache_dir: str | Path = MARKET_CACHE_DIR,
    indicator_cache_path: str | Path = INDICATOR_CACHE_PATH,
    pools_dir: str | Path = POOLS_DIR,
) -> dict[str, Any]:
    min_date, max_date = dataframe_date_bounds(pool)
    return {
        "cache_type": "pool",
        "created_at": now_str(),
        "data_root": str(DATA_ROOT),
        "strategy": strategy_name,
        "strategy_version": strategy_version(strategy_name, strategy_func),
        "required_indicators": strategy_required_indicators(strategy_func),
        "indicator_cache_path": str(indicator_cache_path),
        "indicator_meta_path": str(meta_path_for_cache(indicator_cache_path)),
        "indicator_version": read_json_meta(indicator_cache_path).get("indicator_version"),
        "market_cache_dir": str(market_cache_dir),
        "pools_dir": str(pools_dir),
        "n1": n1,
        "n2": n2,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "rows": int(len(pool)),
        "columns_count": int(len(pool.columns)),
        "columns": [str(c) for c in pool.columns],
        "min_date": min_date,
        "max_date": max_date,
    }
