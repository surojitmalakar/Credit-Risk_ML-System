"""Server-side search, filter, sort, and pagination for large tables.

Browser-side tables have to receive every row, which is the slowest part of
exploring a large upload. These helpers keep only the visible page in the
rendered payload and reuse one precomputed search blob per dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_PAGE_SIZE = 25
PAGE_SIZE_OPTIONS = (10, 25, 50, 100, 250)
MAX_FILTER_OPTIONS = 30


def search_blob(frame: pd.DataFrame) -> pd.Series:
    """One lowercase text blob per row, computed once per dataset."""
    if frame.empty:
        return pd.Series(dtype="string")
    text = frame.astype("string").fillna("")
    joined = text.agg(" ".join, axis=1) if len(text.columns) > 1 else text.iloc[:, 0]
    return joined.str.lower()


def apply_search(
    frame: pd.DataFrame,
    blob: pd.Series | None,
    search: str,
) -> pd.DataFrame:
    needle = (search or "").strip().lower()
    if not needle or frame.empty:
        return frame
    if blob is None or len(blob) != len(frame):
        blob = search_blob(frame)
    return frame[blob.str.contains(needle, regex=False, na=False).to_numpy()]


def apply_column_filters(
    frame: pd.DataFrame,
    column_filters: dict[str, list[str]],
) -> pd.DataFrame:
    """Keep rows whose column values are all inside the selected options."""
    if frame.empty or not column_filters:
        return frame
    mask = pd.Series(True, index=frame.index)
    for column, allowed in column_filters.items():
        if column not in frame.columns or not allowed:
            continue
        values = frame[column].astype("string").fillna("Missing")
        mask &= values.isin([str(value) for value in allowed]).to_numpy()
    return frame[mask.to_numpy()]


def numeric_filter_columns(frame: pd.DataFrame, limit: int = 2) -> list[str]:
    """Prefer identifier-like numeric columns for min/max range filters."""
    numeric = [
        column for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) and frame[column].notna().any()
    ]
    preferred = [column for column in numeric if "score" in str(column).lower()]
    return (preferred or numeric)[:limit]


def filter_candidates(frame: pd.DataFrame, max_options: int = MAX_FILTER_OPTIONS) -> dict[str, list[str]]:
    """Low-cardinality categorical columns that make useful filter dropdowns."""
    if frame.empty:
        return {}
    candidates: dict[str, list[str]] = {}
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = frame[column].astype("string").fillna("Missing")
        unique = values.nunique()
        if 1 < unique <= max_options:
            candidates[str(column)] = sorted(values.unique().tolist())
    return dict(sorted(candidates.items(), key=lambda item: (-len(item[1]), item[0])))


def apply_sort(
    frame: pd.DataFrame,
    sort_column: str | None,
    ascending: bool = True,
) -> pd.DataFrame:
    if frame.empty or not sort_column or sort_column not in frame.columns:
        return frame
    return frame.sort_values(sort_column, ascending=ascending, kind="stable", na_position="last")


def page_slice(
    frame: pd.DataFrame,
    page: int,
    page_size: int,
) -> tuple[pd.DataFrame, int, int]:
    """Return the requested page, the total page count, and the resolved page."""
    page_size = max(1, int(page_size))
    pages = max(1, int(np.ceil(len(frame) / page_size)))
    page = min(max(1, int(page or 1)), pages)
    start = (page - 1) * page_size
    return frame.iloc[start:start + page_size], pages, page


def numeric_bounds(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    if column not in frame.columns:
        return 0.0, 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return 0.0, 0.0
    return float(values.min()), float(values.max())
