from __future__ import annotations

import numpy as np
import pandas as pd

from msme_ews.table_tools import (
    apply_column_filters,
    apply_search,
    apply_sort,
    filter_candidates,
    numeric_filter_columns,
    page_slice,
    search_blob,
)


def _frame(rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "client_ref": [f"C-{index:03d}" for index in range(rows)],
        "zone": [["North", "South", "West", "East"][index % 4] for index in range(rows)],
        "amount": rng.uniform(1_000, 90_000, rows).round(2),
        "bureau_score": rng.integers(560, 820, rows),
        "notes": [None if index % 5 == 0 else f"review {index}" for index in range(rows)],
    })


def test_search_blob_is_lowercase_and_covers_every_column():
    frame = _frame(10)
    blob = search_blob(frame)
    assert len(blob) == len(frame)
    assert blob.iloc[0].startswith("c-000 north")
    assert "review 9" in blob.iloc[9]
    assert "nan" not in blob.iloc[0]


def test_search_filters_rows_across_all_columns():
    frame = _frame(60)
    blob = search_blob(frame)
    matched = apply_search(frame, blob, "east")
    assert len(matched) == 15
    assert set(matched["zone"]) == {"East"}
    assert len(apply_search(frame, blob, "")) == len(frame)
    assert apply_search(frame, blob, "no-such-value").empty


def test_search_rebuilds_the_blob_when_the_source_frame_changed():
    frame = _frame(20)
    stale_blob = search_blob(frame)
    other = frame.iloc[:5].copy()
    matched = apply_search(other, stale_blob, "north")
    assert len(matched) <= 5
    assert not matched.empty


def test_column_filters_keep_only_selected_values():
    frame = _frame(40)
    filtered = apply_column_filters(frame, {"zone": ["North", "South"]})
    assert set(filtered["zone"]) == {"North", "South"}
    assert len(filtered) == 20
    assert len(apply_column_filters(frame, {})) == len(frame)
    assert len(apply_column_filters(frame, {"zone": []})) == len(frame)
    assert len(apply_column_filters(frame, {"missing": ["x"]})) == len(frame)


def test_filter_candidates_only_exposes_low_cardinality_columns():
    frame = _frame(40)
    candidates = filter_candidates(frame)
    assert "zone" in candidates
    assert "amount" not in candidates
    assert "client_ref" not in candidates
    assert candidates["zone"] == ["East", "North", "South", "West"]
    assert filter_candidates(pd.DataFrame()) == {}


def test_sort_is_stable_and_places_missing_values_last():
    frame = _frame(20)
    ascending = apply_sort(frame, "amount", ascending=True)
    assert ascending["amount"].is_monotonic_increasing
    assert ascending["amount"].iloc[-1] == frame["amount"].max()
    descending = apply_sort(frame, "amount", ascending=False)
    assert descending["amount"].iloc[0] == frame["amount"].max()
    assert apply_sort(frame, "missing_column").equals(frame)


def test_pagination_returns_only_the_visible_page():
    frame = _frame(60)
    first, pages, page = page_slice(frame, 1, 25)
    assert pages == 3 and page == 1
    assert len(first) == 25

    second, pages, page = page_slice(frame, 2, 25)
    assert pages == 3 and page == 2
    assert len(second) == 25
    assert first.index[0] != second.index[0]

    last, pages, page = page_slice(frame, 3, 25)
    assert len(last) == 10
    assert len(page_slice(frame, 99, 25)[0]) == 10
    assert page_slice(frame, 0, 25)[2] == 1
    assert page_slice(frame, -5, 25)[2] == 1


def test_pagination_handles_empty_and_tiny_frames():
    empty = pd.DataFrame({"a": []})
    page, pages, number = page_slice(empty, 1, 25)
    assert page.empty and pages == 1 and number == 1
    page, pages, _ = page_slice(pd.DataFrame({"a": [1]}), 1, 25)
    assert len(page) == 1 and pages == 1


def test_numeric_filter_columns_prefers_score_like_columns():
    frame = _frame(10)
    assert numeric_filter_columns(frame) == ["bureau_score"]
    assert numeric_filter_columns(pd.DataFrame({"x": ["a", "b"]})) == []
