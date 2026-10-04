"""Transparent row-level and aggregate analysis for customer/loan datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_POSITIVE_DEFAULTS = {
    "1", "true", "yes", "y", "default", "defaulted", "delinquent",
    "charged off", "charge off", "npa", "bad", "overdue", "non performing",
}
_NEGATIVE_DEFAULTS = {
    "0", "false", "no", "n", "current", "performing", "paid", "repaid",
    "fully paid", "not default", "no default", "non default", "good", "closed",
}
_HIGH_RISK_WORDS = {"high", "critical", "severe", "very high", "bad", "default", "delinquent"}
_MODERATE_RISK_WORDS = {"moderate", "medium", "watch", "elevated", "at risk"}


def _normalized_status(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())


def _observed_default(value: object) -> bool | None:
    status = _normalized_status(value)
    if not status:
        return None
    if status in _NEGATIVE_DEFAULTS or any(word in status for word in ("not default", "no default", "non default")):
        return False
    if status in _POSITIVE_DEFAULTS or any(word in status for word in ("default", "delinquent", "charged off", "non performing", "past due")):
        return True
    if "paid" in status or "performing" in status or status.startswith("current"):
        return False
    return None


def analyze_credit_portfolio(frame: pd.DataFrame) -> dict[str, Any]:
    """Derive transparent row-level risk indicators and observed portfolio rates."""
    records = frame.copy().reset_index(drop=True)
    id_column = next(
        (name for name in ("customer_id", "loan_id", "customer_name", "company_id") if name in records),
        None,
    )
    status_column = next(
        (name for name in ("default_status", "loan_status") if name in records),
        None,
    )
    risk_column = "risk_status" if "risk_status" in records else None
    score_column = "credit_score" if "credit_score" in records else None
    amount_column = "loan_amount" if "loan_amount" in records else None

    default_labels: list[bool | None] = [
        _observed_default(value) for value in records[status_column]
    ] if status_column else [None] * len(records)
    risks: list[str] = []
    reasons: list[str] = []
    for index, row in records.iterrows():
        reasons_for_row: list[str] = []
        explicit_risk = _normalized_status(row.get(risk_column)) if risk_column else ""
        if explicit_risk:
            if any(term in explicit_risk for term in _HIGH_RISK_WORDS):
                risk = "High Risk"
                reasons_for_row.append(f"Reported risk status: {row[risk_column]}")
            elif any(term in explicit_risk for term in _MODERATE_RISK_WORDS):
                risk = "Moderate Risk"
                reasons_for_row.append(f"Reported risk status: {row[risk_column]}")
            elif any(term in explicit_risk for term in ("low", "good", "healthy")):
                risk = "Low Risk"
                reasons_for_row.append(f"Reported risk status: {row[risk_column]}")
            else:
                risk = "Not Assessed"
        else:
            risk = "Not Assessed"

        defaulted = default_labels[index]
        if defaulted is True:
            risk = "High Risk"
            reasons_for_row.append("Observed default or delinquency status")
        elif defaulted is False and risk == "Not Assessed":
            risk = "Low Risk"
            reasons_for_row.append("Observed non-default/performing status")

        score = pd.to_numeric(row.get(score_column), errors="coerce") if score_column else np.nan
        if pd.notna(score):
            if score < 580:
                risk = "High Risk"
                reasons_for_row.append("Credit score below 580")
            elif score < 670 and risk not in {"High Risk"}:
                risk = "Moderate Risk"
                reasons_for_row.append("Credit score below 670")
            elif score >= 670 and risk == "Not Assessed":
                risk = "Low Risk"
                reasons_for_row.append("Credit score at or above 670")

        risks.append(risk)
        reasons.append("; ".join(dict.fromkeys(reasons_for_row)) or "No recognized risk value was available")

    records["portfolio_risk_category"] = risks
    records["risk_basis"] = reasons
    if id_column and id_column != "customer_id":
        records = records.rename(columns={id_column: "customer_id"})

    observed_defaults = pd.Series(default_labels, dtype="object").dropna()
    score_values = pd.to_numeric(records[score_column], errors="coerce").dropna() if score_column else pd.Series(dtype=float)
    loan_values = pd.to_numeric(records[amount_column], errors="coerce").dropna() if amount_column else pd.Series(dtype=float)
    risk_counts = records["portfolio_risk_category"].value_counts()
    summary = {
        "record_count": int(len(records)),
        "unique_customers": int(records["customer_id"].nunique()) if "customer_id" in records else None,
        "observed_default_count": int(sum(value is True for value in default_labels)),
        "observed_default_coverage": int(len(observed_defaults)),
        "observed_default_rate": (
            float(sum(value is True for value in default_labels) / len(observed_defaults))
            if len(observed_defaults) else None
        ),
        "average_credit_score": float(score_values.mean()) if not score_values.empty else None,
        "average_loan_amount": float(loan_values.mean()) if not loan_values.empty else None,
        "total_loan_amount": float(loan_values.sum()) if not loan_values.empty else None,
        "risk_counts": {str(key): int(value) for key, value in risk_counts.items()},
        "risk_assessed_count": int(records["portfolio_risk_category"].ne("Not Assessed").sum()),
        "available_credit_columns": [
            column for column in ("credit_score", "loan_amount", "default_status", "loan_status", "risk_status")
            if column in frame
        ],
    }
    return {"customers": records, "summary": summary}
