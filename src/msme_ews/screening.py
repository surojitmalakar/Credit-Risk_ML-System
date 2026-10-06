"""Vectorized portfolio screening and multi-record comparison helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from msme_ews.data import FINANCIAL_COLUMNS
from msme_ews.financial_analysis import (
    analyze_financials,
    observed_field_counts,
    rule_based_assessment,
    rule_based_risk_index,
)
from msme_ews.prediction import risk_category

RULE_METHOD = "Rule-based risk index"
NO_DATA_METHOD = "Insufficient data"
_INDEX_FIELD_TOTAL = len([column for column in FINANCIAL_COLUMNS if column != "Sales_Growth"])
_COMPANY_COLUMNS = ("company_id", "customer_id", "customer_name", "company")
_PERIOD_COLUMNS = ("period", "financial_year", "fiscal_year", "year")


def _identifier_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def score_records(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    flags: pd.DataFrame | None = None,
    bundle: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Screen every record in one vectorized pass.

    All displayed scores are transparent rule-based indices. Supervised metrics
    are evaluated separately against an uploaded target in data intelligence;
    a bundled model is never applied to unrelated uploaded records.
    """
    if frame.empty:
        return pd.DataFrame()

    rule = rule_based_risk_index(frame, features)
    risk_index = rule["Rule risk index"].astype("float64")
    category = risk_index.map(
        lambda value: (
            "Insufficient Data"
            if pd.isna(value)
            else risk_category(float(value) / 100)
        )
    ).astype("string")
    method = pd.Series(RULE_METHOD, index=frame.index, dtype="string").mask(
        risk_index.isna(),
        NO_DATA_METHOD,
    )

    has_flags = flags is not None and len(flags) == len(frame)
    coverage_fields = observed_field_counts(frame)
    scores = pd.DataFrame(index=frame.index)
    scores["_position"] = np.arange(len(frame))
    scores["Record"] = np.arange(1, len(frame) + 1)
    company_column = _identifier_column(frame, _COMPANY_COLUMNS)
    period_column = _identifier_column(frame, _PERIOD_COLUMNS)
    if company_column is not None:
        scores["Company"] = frame[company_column].astype("string")
    if period_column is not None:
        scores["Period"] = frame[period_column].astype("string")
    scores["Method"] = method
    scores["Risk index (0-100)"] = risk_index.round(1)
    scores["Risk category"] = category
    scores["Health score"] = (100 - risk_index).clip(0, 100).round(1)
    scores["Data coverage"] = coverage_fields
    scores["Coverage %"] = (coverage_fields / _INDEX_FIELD_TOTAL * 100).round(0).astype("int64")
    scores["Warning signals"] = (
        flags.sum(axis=1).astype("int64") if has_flags else pd.Series(0, index=frame.index, dtype="int64")
    )
    scores["Warnings"] = (
        flags.apply(lambda row: ", ".join(name for name in flags.columns if bool(row[name])), axis=1)
        if has_flags
        else pd.Series("", index=frame.index, dtype="string")
    )
    scores["ML model used"] = False
    if company_column is None and period_column is None:
        scores["Drivers"] = rule["Rule drivers"]
    return scores.reset_index(drop=True)


def screening_summary(scores: pd.DataFrame) -> dict[str, Any]:
    """Aggregate band mix and coverage for the screening header."""
    if scores.empty:
        return {
            "records": 0,
            "bands": {},
            "model_records": 0,
            "rule_records": 0,
            "flagged_records": 0,
            "mean_risk_index": None,
            "median_risk_index": None,
            "elevated_records": 0,
        }
    risk_index = scores["Risk index (0-100)"]
    bands = scores["Risk category"].value_counts().to_dict()
    return {
        "records": int(len(scores)),
        "bands": {band: int(count) for band, count in bands.items()},
        "model_records": 0,
        "rule_records": int((scores["Method"] == RULE_METHOD).sum()),
        "insufficient_records": int((scores["Method"] == NO_DATA_METHOD).sum()),
        "flagged_records": int((scores["Warning signals"] > 0).sum()),
        "mean_risk_index": float(risk_index.mean()) if risk_index.notna().any() else None,
        "median_risk_index": float(risk_index.median()) if risk_index.notna().any() else None,
        "elevated_records": int((risk_index >= 60).sum()),
    }


def banded_exposure(scores: pd.DataFrame, frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Aggregate an exposure or amount column per risk band when one exists."""
    if column not in frame.columns or scores.empty:
        return pd.DataFrame()
    table = pd.DataFrame({
        "Risk category": scores["Risk category"].to_numpy(),
        column: pd.to_numeric(frame[column], errors="coerce").to_numpy(),
    }).dropna()
    if table.empty:
        return pd.DataFrame()
    grouped = table.groupby("Risk category", dropna=False)[column].agg(["count", "sum", "mean"]).reset_index()
    return grouped.rename(columns={
        "count": "Records",
        "sum": f"Total {column}",
        "mean": f"Average {column}",
    })


def detect_exposure_column(frame: pd.DataFrame) -> str | None:
    """Find an obvious exposure, balance, or amount column for band aggregation."""
    preferred = ("loan_amount", "exposure", "outstanding", "loan_balance", "credit_limit", "balance", "amount")
    lowered = {str(column).lower(): column for column in frame.columns}
    for candidate in preferred:
        if candidate in lowered:
            return lowered[candidate]
    return None


def record_snapshot(
    frame: pd.DataFrame,
    row_index: int,
    features: pd.DataFrame,
    flags: pd.DataFrame | None,
    bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    """Full per-record detail used by the comparison view."""
    row = frame.iloc[row_index]
    analysis = analyze_financials(frame, row_index, features)
    risk_index: float | None = None
    drivers: list[dict[str, Any]] = []
    rule = rule_based_assessment(frame, row_index, analysis)
    risk_index = rule["risk_index"]
    category = rule["risk_category"]
    method = RULE_METHOD if risk_index is not None else NO_DATA_METHOD
    drivers = rule["top_risk_factors"]
    active = (
        flags.columns[flags.iloc[row_index]].tolist()
        if flags is not None and len(flags) == len(frame)
        else []
    )
    company_column = _identifier_column(frame, _COMPANY_COLUMNS)
    company = (
        str(row[company_column])
        if company_column is not None and pd.notna(row[company_column])
        else f"Record {row_index + 1}"
    )
    return {
        "row_index": row_index,
        "company": company,
        "period": str(row.get("period", "Not identified")),
        "distress_probability": None,
        "risk_index": risk_index,
        "risk_category": category,
        "health_score": None if risk_index is None else max(0.0, min(100.0, 100 - risk_index)),
        "method": method,
        "model_available": False,
        "coverage_label": (
            f"{analysis['coverage_count']}/{analysis['coverage_total']} core fields "
            f"({analysis['coverage_percent']:.0f}%)"
        ),
        "coverage_percent": analysis["coverage_percent"],
        "ratios": analysis["ratios"],
        "warnings": active,
        "top_risk_factors": drivers,
        "features": features.iloc[row_index],
        "row": row,
    }
