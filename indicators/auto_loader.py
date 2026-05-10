# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


EXCLUDE_MODULES = {
    "__init__",
    "basic",
    "auto_loader",
}

REQUIRED_BASE_COLUMNS = [
    "symbol",
    "file",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]

# indicator 层禁止出现策略判断 / 条件判断 / 距离判断字段
FORBIDDEN_COLUMN_KEYWORDS = [
    "selected",
    "score",
    "rank",
    "signal",
    "xg",
    "buy",
    "sell",
    "position",
    "strategy",
    "pool",
    "market_regime",

    # 判断条件类，不放 indicator cache
    "close_to",
    "distance",
    "gap",
    "above",
    "below",
    "cross",
]

FORBIDDEN_EXACT_COLUMNS = {
    "renko_color",
    "renko_is_red",
    "renko_is_green",
    "renko_is_flat",
}


@dataclass
class IndicatorAuditResult:
    ok: bool
    module_name: str
    func_name: str
    errors: list[str]
    warnings: list[str]
    added_cols: list[str]
    modified_existing_cols: list[str]


def discover_auto_indicator_modules() -> list[str]:
    indicator_dir = Path(__file__).resolve().parent
    modules: list[str] = []

    for item in pkgutil.iter_modules([str(indicator_dir)]):
        name = item.name

        if name in EXCLUDE_MODULES:
            continue

        if name.startswith("_"):
            continue

        modules.append(name)

    return sorted(modules)


def _get_apply_func(module_name: str) -> Callable | None:
    module = importlib.import_module(f"indicators.{module_name}")

    for func_name in ("add_indicators", "apply_indicators"):
        func = getattr(module, func_name, None)
        if callable(func):
            return func

    return None


def _call_apply_func(func: Callable, df: pd.DataFrame, kwargs: dict) -> pd.DataFrame:
    sig = inspect.signature(func)

    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )

    if has_var_kwargs:
        return func(df, **kwargs)

    accepted = {}
    for name in sig.parameters:
        if name in kwargs:
            accepted[name] = kwargs[name]

    return func(df, **accepted)


def _compare_series_exact(a: pd.Series, b: pd.Series) -> bool:
    if len(a) != len(b):
        return False

    try:
        return bool(a.equals(b))
    except Exception:
        return False


def _is_numeric_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def audit_indicator_output(
    module_name: str,
    func_name: str,
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> IndicatorAuditResult:
    """
    indicator 层统一口径审核。

    允许：
    1. 只新增基础事实型数值指标
    2. 不改变行数
    3. 不改变 index / 顺序
    4. 不修改已有字段
    5. 不删除已有字段

    禁止：
    1. selected / score / rank / signal / xg 等策略字段
    2. close_to / distance / above / below / cross 等判断条件字段
    3. renko_color / renko_is_red 等颜色判断字段
    4. bool / object / category 等非数值字段
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(after, pd.DataFrame):
        errors.append(f"return type must be pandas.DataFrame, got {type(after)}")
        return IndicatorAuditResult(
            ok=False,
            module_name=module_name,
            func_name=func_name,
            errors=errors,
            warnings=warnings,
            added_cols=[],
            modified_existing_cols=[],
        )

    before_cols = list(before.columns)
    after_cols = list(after.columns)

    before_col_set = set(before_cols)
    after_col_set = set(after_cols)

    added_cols = [c for c in after_cols if c not in before_col_set]
    deleted_cols = [c for c in before_cols if c not in after_col_set]

    if len(after) != len(before):
        errors.append(f"row count changed: before={len(before)}, after={len(after)}")

    if not before.index.equals(after.index):
        errors.append("index changed; indicator must preserve original row order/index")

    if after.columns.duplicated().any():
        duplicated = after.columns[after.columns.duplicated()].tolist()
        errors.append(f"duplicate columns found: {duplicated}")

    if deleted_cols:
        errors.append(f"deleted existing columns: {deleted_cols}")

    for col in REQUIRED_BASE_COLUMNS:
        if col not in after.columns:
            errors.append(f"required base column missing after indicator: {col}")

    modified_existing_cols = []
    for col in before_cols:
        if col not in after.columns:
            continue

        if not _compare_series_exact(before[col], after[col]):
            modified_existing_cols.append(col)

    if modified_existing_cols:
        errors.append(
            "modified existing columns; indicator files may only add columns: "
            f"{modified_existing_cols}"
        )

    forbidden_added = []
    for col in added_cols:
        col_lower = str(col).lower()

        if col_lower in FORBIDDEN_EXACT_COLUMNS:
            forbidden_added.append(col)
            continue

        if any(key in col_lower for key in FORBIDDEN_COLUMN_KEYWORDS):
            forbidden_added.append(col)

    if forbidden_added:
        errors.append(
            "added strategy/judgement-like columns, not allowed in indicator layer: "
            f"{forbidden_added}"
        )

    non_numeric_added = []
    bool_added = []

    for col in added_cols:
        if col not in after.columns:
            continue

        dtype = after[col].dtype

        if pd.api.types.is_bool_dtype(dtype):
            bool_added.append(f"{col}:{dtype}")
            continue

        if not _is_numeric_dtype(after[col]):
            non_numeric_added.append(f"{col}:{dtype}")

    if bool_added:
        errors.append(
            "added boolean columns; boolean judgement belongs to strategy/analyze layer: "
            f"{bool_added}"
        )

    if non_numeric_added:
        errors.append(
            "added non-numeric indicator columns; indicator cache should keep numeric facts only: "
            f"{non_numeric_added}"
        )

    all_nan_added = []
    inf_added = []

    for col in added_cols:
        s = pd.to_numeric(after[col], errors="coerce")

        if s.notna().sum() == 0:
            all_nan_added.append(col)

        finite_values = s.dropna()
        if len(finite_values) > 0 and not bool(np.isfinite(finite_values).all()):
            inf_added.append(col)

    if all_nan_added:
        errors.append(f"added columns are all NaN: {all_nan_added}")

    if inf_added:
        errors.append(f"added columns contain inf/-inf: {inf_added}")

    high_nan_added = []
    for col in added_cols:
        s = pd.to_numeric(after[col], errors="coerce")
        if len(s) == 0:
            continue

        nan_ratio = float(s.isna().mean())
        if nan_ratio > 0.8:
            high_nan_added.append(f"{col}: nan_ratio={nan_ratio:.2%}")

    if high_nan_added:
        warnings.append(
            "added columns have very high NaN ratio; check min_periods / formula: "
            f"{high_nan_added}"
        )

    return IndicatorAuditResult(
        ok=len(errors) == 0,
        module_name=module_name,
        func_name=func_name,
        errors=errors,
        warnings=warnings,
        added_cols=added_cols,
        modified_existing_cols=modified_existing_cols,
    )


def print_audit_result(result: IndicatorAuditResult) -> None:
    """
    正常情况不打印，避免 build_indicators.py 刷屏。
    只有审核失败时打印 debug 信息。
    """
    if result.ok:
        return

    title = f"indicators.{result.module_name}.{result.func_name}"
    print(f"\n[AUDIT][FAILED] {title}")

    if result.added_cols:
        print(f"  added columns: {result.added_cols}")

    if result.modified_existing_cols:
        print(f"  modified existing columns: {result.modified_existing_cols}")

    for error in result.errors:
        print(f"  [ERROR] {error}")

    for warning in result.warnings:
        print(f"  [WARN] {warning}")


def apply_auto_indicators(
    df: pd.DataFrame,
    audit: bool = True,
    strict_audit: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    自动加载 indicators 文件夹下所有指标模块。

    新指标文件规范：
    1. 放在 indicators/xxx.py
    2. 提供 add_indicators(df, **kwargs) -> pd.DataFrame
       或 apply_indicators(df, **kwargs) -> pd.DataFrame
    3. 只能新增基础事实型数值指标
    4. 不能改旧字段
    5. 不能删行、增行、改变顺序
    6. 不能新增判断条件字段，例如 close_to / distance / above / below / cross
    7. 不能新增策略字段，例如 selected / score / rank / signal / xg

    正常情况下不打印任何内容。
    如果审核失败，会打印失败模块和原因。
    """
    out = df.copy()

    modules = discover_auto_indicator_modules()

    for module_name in modules:
        func = _get_apply_func(module_name)

        if func is None:
            continue

        before = out.copy(deep=True)
        after = _call_apply_func(func, out, kwargs)

        if audit:
            result = audit_indicator_output(
                module_name=module_name,
                func_name=func.__name__,
                before=before,
                after=after,
            )

            print_audit_result(result)

            if not result.ok and strict_audit:
                raise RuntimeError(
                    f"Indicator audit failed for indicators.{module_name}.{func.__name__}. "
                    "See terminal debug info above."
                )

        if not isinstance(after, pd.DataFrame):
            raise TypeError(
                f"indicators.{module_name}.{func.__name__} must return pd.DataFrame, "
                f"got {type(after)}"
            )

        out = after

    return out