"""Vectorized portfolio screening and multi-record comparison helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from msme_ews.data import FINANCIAL_COLUMNS
try:
    from msme_ews.financial_analysis import (
        analyze_financials,
        has_sufficient_ml_data,
        model_eligible_rows,
        observed_field_counts,
        rule_based_assessment,
        rule_based_risk_index,
    )
except ImportError:
    from msme_ews.financial_analysis import (
        analyze_financials,
        has_sufficient_ml_data,
        rule_based_assessment,
    )

    # Streamlit can briefly serve mixed file revisions while refreshing a
    # deployment. Keep screening usable with the scalar API from older versions.
    def observed_field_counts(frame: pd.DataFrame) -> pd.Series:
        available = [column for column in FINANCIAL_COLUMNS if column != "Sales_Growth"]
        values = pd.DataFrame(index=frame.index)
        for column in available:
            source = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
            values[column] = pd.to_numeric(source, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).notna()
        return values.sum(axis=1).astype("int64")

    def model_eligible_rows(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(
            [has_sufficient_ml_data(frame, position) for position in range(len(frame))],
            index=frame.index,
            dtype=bool,
        )

    def rule_based_risk_index(
        frame: pd.DataFrame,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []
        coverage = observed_field_counts(frame)
        for position in range(len(frame)):
            analysis = analyze_financials(frame, position, features)
            assessment = rule_based_assessment(frame, position, analysis)
            factors = assessment["top_risk_factors"]
            probability = assessment["distress_probability"]
            rows.append({
                "Rule risk index": probability,
                "Rule risk category": assessment["risk_category"],
                "Rule risk score": sum(factor["contribution"] for factor in factors),
                "Rule risk factors": len(factors),
                "Rule drivers": ", ".join(factor["feature"] for factor in factors),
                "Rule protective factors": len(assessment["protective_factors"]),
                "Observed core fields": coverage.iloc[position],
            })
        return pd.DataFrame(rows, index=frame.index)
from msme_ews.prediction import risk_category

ML_METHOD = "Existing ML model"
RULE_METHOD = "Rule-based risk index"
NO_DATA_METHOD = "Insufficient data"
_INDEX_FIELD_TOTAL = len([column for column in FINANCIAL_COLUMNS if column != "Sales_Growth"])
_COMPANY_COLUMNS = ("company_id", "customer_id", "customer_name", "company")
_PERIOD_COLUMNS = ("period", "financial_year", "fiscal_year", "year")


def _identifier_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def _model_probabilities(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    bundle: dict[str, Any] | None,
) -> pd.Series:
    """Score every row in one predict call; NaN marks rows the model cannot score."""
    unusable = pd.Series(np.nan, index=frame.index, dtype="float64")
    if bundle is None or "model" not in bundle or "features" not in bundle:
        return unusable
    required = list(bundle["features"])
    if any(name not in features.columns for name in required):
        return unusable
    try:
        probabilities = bundle["model"].predict_proba(features[required])
    except (ValueError, TypeError, KeyError):
        return unusable
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        return unusable
    return pd.Series(probabilities[:, 1], index=frame.index, dtype="float64")


def score_records(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    flags: pd.DataFrame | None = None,
    bundle: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Screen every record in one vectorized pass.

    Records with enough observed inputs use the existing ML model; the rest fall
    back to the same transparent rule-based index used for single-record views.
    The returned frame keeps a ``_position`` column for positional lookup back
    into ``frame``.
    """
    if frame.empty:
        return pd.DataFrame()

    eligible = model_eligible_rows(frame)
    model_probability = _model_probabilities(frame, features, bundle)
    rule = rule_based_risk_index(frame, features)

    probability = rule["Rule risk index"].astype("float64")
    model_ready = eligible & model_probability.notna()
    probability = probability.mask(model_ready, model_probability)

    method = pd.Series(RULE_METHOD, index=frame.index, dtype="string")
    method = method.mask(model_ready, ML_METHOD).mask(probability.isna(), NO_DATA_METHOD)
    category = probability.map(
        lambda value: "Insufficient Data" if pd.isna(value) else risk_category(float(value))
    ).astype("string")

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
    scores["Distress probability"] = probability.round(4)
    scores["Risk category"] = category
    scores["Health score"] = (1 - probability).mul(100).clip(0, 100).round(1)
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
    scores["ML model used"] = model_ready
    if company_column is None and period_column is None:
        scores["Drivers"] = rule["Rule drivers"]
    return scores.reset_index(drop=True)


def screening_summary(scores: pd.DataFrame) -> dict[str, Any]:
    """Aggregate band mix and coverage for the screening header."""
    if scores.empty:
        return {"records": 0, "bands": {}, "model_records": 0, "rule_records": 0, "flagged_records": 0}
    probability = scores["Distress probability"]
    bands = scores["Risk category"].value_counts().to_dict()
    scored = probability.dropna()
    return {
        "records": int(len(scores)),
        "bands": {band: int(count) for band, count in bands.items()},
        "model_records": int(scores["ML model used"].sum()),
        "rule_records": int((scores["Method"] == RULE_METHOD).sum()),
        "insufficient_records": int((scores["Method"] == NO_DATA_METHOD).sum()),
        "flagged_records": int((scores["Warning signals"] > 0).sum()),
        "mean_probability": float(scored.mean()) if len(scored) else None,
        "median_probability": float(scored.median()) if len(scored) else None,
        "elevated_records": int((probability >= 0.60).sum()),
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
    model_ready = has_sufficient_ml_data(frame, row_index)
    probability: float | None = None
    if model_ready and bundle is not None:
        required = list(bundle["features"])
        if all(name in features.columns for name in required):
            try:
                probability = float(
                    bundle["model"].predict_proba(features.iloc[[row_index]][required])[0, 1]
                )
            except (ValueError, TypeError, KeyError):
                probability = None
    drivers: list[dict[str, Any]] = []
    if probability is not None:
        category = risk_category(probability)
        method = ML_METHOD
    else:
        rule = rule_based_assessment(frame, row_index, analysis)
        probability = rule["distress_probability"]
        category = rule["risk_category"]
        method = RULE_METHOD if probability is not None else NO_DATA_METHOD
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
        "distress_probability": probability,
        "risk_category": category,
        "health_score": None if probability is None else max(0.0, min(100.0, (1 - probability) * 100)),
        "method": method,
        "model_available": model_ready,
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
