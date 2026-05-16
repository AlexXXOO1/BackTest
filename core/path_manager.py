# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = Path(os.environ.get("BACKTEST_SETTINGS_PATH", PROJECT_ROOT / "configs" / "settings.json"))


def _load_settings() -> dict[str, Any]:
    if SETTINGS_PATH.exists():
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    return {}


_SETTINGS = _load_settings()


def _as_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser()


def _setting_path(key: str, default_relative: str) -> Path:
    data_root = DATA_ROOT
    rel = _SETTINGS.get("paths", {}).get(key, default_relative)
    return data_root / Path(str(rel))


DATA_ROOT = _as_path(
    os.environ.get(
        "BACKTEST_DATA_ROOT",
        _SETTINGS.get("data_root", r"C:\Users\zyf37\Desktop\BackTest_Data"),
    )
)

RAW_TDX_TXT_DIR = _setting_path("raw_tdx_data", "raw_tdx_data")
MARKET_CACHE_DIR = _setting_path("market_cache", "market_cache/daily_bars_by_symbol")
INDICATOR_CACHE_PATH = _setting_path("indicator_cache", "indicator_cache/daily_indicators.parquet")
POOLS_DIR = _setting_path("pools", "pools")
OUTPUT_DIR = _setting_path("output", "output")
RAW_SH_INDEX_DIR = _setting_path("raw_sh_index", "raw_SH_index")


def ensure_data_dirs() -> None:
    for path in [RAW_TDX_TXT_DIR, MARKET_CACHE_DIR, INDICATOR_CACHE_PATH.parent, POOLS_DIR, OUTPUT_DIR, RAW_SH_INDEX_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def describe_paths() -> dict[str, str]:
    return {
        "project_root": str(PROJECT_ROOT),
        "settings_path": str(SETTINGS_PATH),
        "data_root": str(DATA_ROOT),
        "raw_tdx_txt_dir": str(RAW_TDX_TXT_DIR),
        "market_cache_dir": str(MARKET_CACHE_DIR),
        "indicator_cache_path": str(INDICATOR_CACHE_PATH),
        "pools_dir": str(POOLS_DIR),
        "output_dir": str(OUTPUT_DIR),
        "raw_sh_index_dir": str(RAW_SH_INDEX_DIR),
    }
