# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


POOL_SCHEMA_VERSION = "pool_contract_v2_factor_first_t4"

REQUIRED_IDENTITY_COLUMNS = [
    "symbol",
    "date",
    "selection_strategy",
]

REQUIRED_MARKET_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]

REQUIRED_FORWARD_PRICE_COLUMNS = [
    "t1_date",
    "t1_open",
    "t1_close",
    "t2_date",
    "t2_open",
    "t2_close",
    "t3_date",
    "t3_open",
    "t3_close",
    "t4_date",
    "t4_open",
    "t4_close",
]

REQUIRED_FORWARD_RETURN_COLUMNS = [
    "fwd_return_pct_T1",
    "fwd_return_pct_T2",
    "fwd_return_pct_T3",
    "fwd_return_pct_T4",
]

OPTIONAL_FORWARD_COLUMNS = [
    "fwd_up_T1",
    "fwd_up_T2",
    "fwd_up_T3",
    "fwd_up_T4",
    "forward_data_status",
]

REQUIRED_POOL_COLUMNS = [
    *REQUIRED_IDENTITY_COLUMNS,
    *REQUIRED_MARKET_COLUMNS,
    *REQUIRED_FORWARD_PRICE_COLUMNS,
    *REQUIRED_FORWARD_RETURN_COLUMNS,
]

TRANSIENT_SELECTION_COLUMNS = [
    "selected",
]

REMOVED_SCORE_COLUMNS = [
    "selected_score_base",
    "score",
    "score_rank_key",
    "score_pct",
]

FORBIDDEN_FINAL_POOL_COLUMNS = [
    *TRANSIENT_SELECTION_COLUMNS,
    *REMOVED_SCORE_COLUMNS,
]

FORWARD_PREFIXES = (
    "t1_",
    "t2_",
    "t3_",
    "t4_",
    "fwd_",
    "forward_",
)

ALLOWED_FORWARD_COLUMNS = set(REQUIRED_FORWARD_PRICE_COLUMNS) | set(REQUIRED_FORWARD_RETURN_COLUMNS) | set(OPTIONAL_FORWARD_COLUMNS)

NON_FACTOR_COLUMNS = set(REQUIRED_POOL_COLUMNS) | set(OPTIONAL_FORWARD_COLUMNS) | {
    "file",
    "code",
    "name",
    "selection_strategy",
}

ABSOLUTE_VALUE_FACTOR_EXCLUDE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "prev_close",
    "prev_volume",
    "ma5",
    "ma10",
    "ma20",
    "ma50",
    "ma60",
    "volume_ma5",
    "volume_ma10",
}


@dataclass(frozen=True)
class PoolSchemaReport:
    schema_version: str
    row_count: int
    column_count: int
    factor_columns: list[str]
    warnings: list[str]


def _is_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return True
    converted = pd.to_numeric(series, errors="coerce")
    non_na = series.notna()
    if not non_na.any():
        return False
    return converted[non_na].notna().mean() >= 0.95


def detect_factor_columns(df: pd.DataFrame) -> list[str]:
    """Detect strategy-specific factor columns from a final pool dataframe."""
    factors: list[str] = []

    for col in df.columns:
        name = str(col)
        lower = name.lower()

        if name in NON_FACTOR_COLUMNS:
            continue
        if name in FORBIDDEN_FINAL_POOL_COLUMNS:
            continue
        if name in ABSOLUTE_VALUE_FACTOR_EXCLUDE_COLUMNS:
            continue
        if any(lower.startswith(prefix) for prefix in FORWARD_PREFIXES):
            continue
        if name.endswith("_date") or name == "date":
            continue
        if not _is_numeric_like(df[name]):
            continue

        factors.append(name)

    return factors


def validate_pool_schema(df: pd.DataFrame, strategy_name: str | None = None) -> PoolSchemaReport:
    """
    Validate the final saved pool contract.

    Contract v1:
    - required core columns must exist
    - old score columns must not exist in final saved pool
    - future / forward columns are allowed only as target columns
    - factor columns are dynamic and are detected automatically
    """
    errors: list[str] = []
    warnings: list[str] = []

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].astype(str).tolist()
        errors.append(f"Duplicate columns found: {duplicates}")

    missing = [c for c in REQUIRED_POOL_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required pool columns: {missing}")

    forbidden = [c for c in FORBIDDEN_FINAL_POOL_COLUMNS if c in df.columns]
    if forbidden:
        errors.append(f"Final pool must not contain removed score/selection columns: {forbidden}")

    illegal_future = []
    for col in df.columns:
        name = str(col)
        lower = name.lower()
        if any(lower.startswith(prefix) for prefix in FORWARD_PREFIXES) and name not in ALLOWED_FORWARD_COLUMNS:
            illegal_future.append(name)
    if illegal_future:
        errors.append(f"Illegal future-like columns found outside target set: {illegal_future}")

    if strategy_name and "selection_strategy" in df.columns and not df.empty:
        bad_strategy = df["selection_strategy"].astype(str).ne(str(strategy_name))
        if bad_strategy.any():
            bad_values = sorted(df.loc[bad_strategy, "selection_strategy"].astype(str).dropna().unique().tolist())
            errors.append(f"selection_strategy contains unexpected values: {bad_values}")

    for col in ["date", "t1_date", "t2_date", "t3_date", "t4_date"]:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if not df.empty and parsed.notna().sum() == 0:
                errors.append(f"Date column cannot be parsed: {col}")

    numeric_required = [
        *REQUIRED_MARKET_COLUMNS,
        "t1_open",
        "t1_close",
        "t2_open",
        "t2_close",
        "t3_open",
        "t3_close",
        "t4_open",
        "t4_close",
        *REQUIRED_FORWARD_RETURN_COLUMNS,
    ]
    for col in numeric_required:
        if col in df.columns and not _is_numeric_like(df[col]):
            errors.append(f"Required numeric column is not numeric-like: {col}")

    factors = detect_factor_columns(df)
    if not factors:
        warnings.append("No dynamic factor columns detected. Analyze tools will have no factor to validate.")

    if errors:
        joined = "\n".join(f"- {x}" for x in errors)
        raise ValueError(f"Pool schema validation failed. schema={POOL_SCHEMA_VERSION}\n{joined}")

    return PoolSchemaReport(
        schema_version=POOL_SCHEMA_VERSION,
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        factor_columns=factors,
        warnings=warnings,
    )


def drop_removed_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove transient selection and old score columns before final parquet save."""
    return df.drop(columns=[c for c in FORBIDDEN_FINAL_POOL_COLUMNS if c in df.columns], errors="ignore").copy()
