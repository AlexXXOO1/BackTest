from __future__ import annotations

from typing import Iterable, TypeVar

T = TypeVar("T")

try:
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover - fallback for environments without tqdm
    _tqdm = None


def progress_bar(iterable: Iterable[T], desc: str = "", total: int | None = None) -> Iterable[T]:
    """Return a tqdm progress bar when tqdm is available, otherwise return the iterable unchanged."""
    if _tqdm is None:
        if desc:
            print(desc)
        return iterable
    return _tqdm(iterable, desc=desc, total=total, unit="item")
