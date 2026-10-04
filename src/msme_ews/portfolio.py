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

# Column candidates used by the concentration and vintage helpers. Matching is
# case-insensitive so exports such as "Loan Amount" or "Origination Date" work.
_AMOUNT_COLUMNS = (
    "loan_amount", "exposure", "outstanding", "loan_balance",
    "credit_limit", "balance", "amount", "sanctioned_amount",
)
_DATE_COLUMNS = (
    "origination_date", "disbursement_date", "disbursed_date", "loan_date",
    "sanction_date", "start_date", "origination_year", "vintage_year",
    "vintage", "cohort",
)
_ID_COLUMNS = ("customer_id", "customer_name", "company_id", "loan_id")
_RISK_CATEGORY_COLUMNS = ("portfolio_risk_category", "risk_category", "risk_status")
_DEFAULT_STATUS_COLUMNS = ("default_status", "loan_status")
_TOP_EXPOSURE_COUNT = 10


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
    return {
        "customers": records,
        "summary": summary,
        "concentration": portfolio_concentration(records),
        "vintage": vintage_analysis(records),
    }


def _identified_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate column present, matched case-insensitively."""
    lowered = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _observed_default_flags(records: pd.DataFrame) -> pd.Series:
    """Observed default status per row (True/False/None), never a model score."""
    status_column = _identified_column(records, _DEFAULT_STATUS_COLUMNS)
    if status_column is None:
        return pd.Series([None] * len(records), dtype="object")
    return pd.Series([_observed_default(value) for value in records[status_column]], dtype="object")


def _cohort_labels(values: pd.Series, freq: str) -> pd.Series:
    """Derive origination cohort labels from dates, year numbers, or vintage text."""
    numeric = pd.to_numeric(values, errors="coerce")
    if len(values) and numeric.notna().all() and numeric.between(1900, 2100).all():
        return numeric.astype("int64").astype("string")
    dates = pd.to_datetime(values, errors="coerce")
    if dates.notna().any():
        return dates.dt.to_period("Q" if freq == "quarter" else "Y").astype("string")
    return values.astype("string").str.strip()


def _top_exposure_table(valued: pd.DataFrame, id_column: str | None, total: float) -> pd.DataFrame:
    """Largest borrowers by aggregated exposure, with each borrower counted once."""
    if id_column is not None:
        grouped = (
            valued.groupby(id_column, dropna=False)
            .agg(Exposure=("_exposure", "sum"), Loans=("_exposure", "size"))
            .sort_values("Exposure", ascending=False)
        )
        grouped.index = grouped.index.astype("string")
        table = grouped.head(_TOP_EXPOSURE_COUNT).reset_index().rename(columns={id_column: "Borrower"})
    else:
        table = (
            valued[["_exposure"]]
            .sort_values("_exposure", ascending=False)
            .head(_TOP_EXPOSURE_COUNT)
            .reset_index(drop=True)
            .rename(columns={"_exposure": "Exposure"})
        )
        table.insert(0, "Borrower", [f"Record {position + 1}" for position in range(len(table))])
        table["Loans"] = 1
    table["Share of exposure (%)"] = (table["Exposure"] / total * 100).round(2)
    return table[["Borrower", "Loans", "Exposure", "Share of exposure (%)"]].reset_index(drop=True)


def _concentration_band_breakdown(valued: pd.DataFrame, total: float) -> pd.DataFrame:
    """Exposure split across the transparent portfolio risk bands."""
    risk_column = _identified_column(valued, _RISK_CATEGORY_COLUMNS)
    if risk_column is None:
        return pd.DataFrame()
    table = valued[[risk_column, "_exposure"]].copy()
    table[risk_column] = table[risk_column].astype("string").fillna("Not Assessed")
    grouped = table.groupby(risk_column)["_exposure"].agg(["count", "sum", "mean"]).reset_index()
    grouped = grouped.rename(columns={
        risk_column: "Risk category",
        "count": "Records",
        "sum": "Exposure",
        "mean": "Average exposure",
    })
    grouped["Average exposure"] = grouped["Average exposure"].round(2)
    grouped["Share of exposure (%)"] = (grouped["Exposure"] / total * 100).round(2)
    return grouped.sort_values("Exposure", ascending=False).reset_index(drop=True)


def _concentration_findings(metrics: dict[str, Any]) -> list[str]:
    """Plain-language concentration notes; descriptive, not a capital measure."""
    findings: list[str] = []
    top_10 = metrics.get("top_10_share")
    largest = metrics.get("largest_exposure_share")
    if top_10 is None or largest is None:
        return findings
    if top_10 >= 0.50:
        findings.append(
            f"The ten largest borrowers hold {top_10:.1%} of total exposure; the book is highly concentrated."
        )
    elif top_10 >= 0.25:
        findings.append(
            f"The ten largest borrowers hold {top_10:.1%} of total exposure, indicating moderate concentration."
        )
    else:
        findings.append(
            f"Exposure is well spread: the ten largest borrowers hold {top_10:.1%} of total exposure."
        )
    findings.append(f"The single largest borrower accounts for {largest:.1%} of exposure.")
    if metrics.get("effective_borrowers") is not None:
        findings.append(
            f"Concentration equates to about {metrics['effective_borrowers']:.1f} equally-sized borrowers "
            f"(inverse HHI from {metrics['herfindahl_index']:.3f})."
        )
    return findings


def portfolio_concentration(
    records: pd.DataFrame,
    amount_column: str | None = None,
) -> dict[str, Any]:
    """Measure how concentrated the observed exposure is across borrowers.

    Aggregation is by borrower wherever an identifier exists, so a customer with
    several loan rows is counted once. The Herfindahl-Hirschman Index reported
    here is a descriptive concentration summary of the uploaded book, not a
    regulatory capital measure, and it uses observed fields only.
    """
    result: dict[str, Any] = {
        "available": False,
        "exposure_column": None,
        "borrower_column": None,
        "total_exposure": None,
        "band_breakdown": pd.DataFrame(),
        "top_exposures": pd.DataFrame(),
        "metrics": {
            "borrowers": 0,
            "herfindahl_index": None,
            "effective_borrowers": None,
            "largest_exposure_share": None,
            "top_5_share": None,
            "top_10_share": None,
        },
        "findings": [],
    }
    if records is None or len(records) == 0:
        result["findings"] = ["No records were available to assess concentration."]
        return result

    column = (
        amount_column
        if amount_column and amount_column in records.columns
        else _identified_column(records, _AMOUNT_COLUMNS)
    )
    if column is None:
        result["findings"] = [
            "No exposure, balance, or loan-amount column was found, so concentration could not be measured."
        ]
        return result

    amounts = pd.to_numeric(records[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valued = records.assign(_exposure=amounts.to_numpy()).dropna(subset=["_exposure"])
    valued = valued[valued["_exposure"] > 0]
    if valued.empty:
        result["findings"] = [
            f"The '{column}' column had no usable positive values, so concentration could not be measured."
        ]
        return result

    total = float(valued["_exposure"].sum())
    id_column = _identified_column(valued, _ID_COLUMNS)
    if id_column is not None:
        by_borrower = (
            valued.groupby(id_column, dropna=False)["_exposure"].sum().sort_values(ascending=False)
        )
    else:
        by_borrower = valued["_exposure"].sort_values(ascending=False)
    shares = by_borrower.to_numpy(dtype="float64") / total
    hhi = float(np.sum(shares ** 2))

    result["available"] = True
    result["exposure_column"] = column
    result["borrower_column"] = id_column
    result["total_exposure"] = total
    result["metrics"] = {
        "borrowers": int(len(shares)),
        "herfindahl_index": round(hhi, 6),
        "effective_borrowers": round(1 / hhi, 3) if hhi > 0 else None,
        "largest_exposure_share": round(float(shares[0]), 6),
        "top_5_share": round(float(shares[:5].sum()), 6),
        "top_10_share": round(float(shares[:10].sum()), 6),
    }
    result["top_exposures"] = _top_exposure_table(valued, id_column, total)
    result["band_breakdown"] = _concentration_band_breakdown(valued, total)
    result["findings"] = _concentration_findings(result["metrics"])
    return result


def _vintage_findings(cohorts: pd.DataFrame) -> list[str]:
    """Compare observed default rates across origination cohorts."""
    if cohorts.empty:
        return []
    if len(cohorts) < 2:
        return ["Only one origination cohort is present; a vintage comparison needs at least two."]
    with_rate = cohorts.dropna(subset=["Observed default rate"])
    if with_rate.empty:
        return [
            "Cohorts were built but no observed default status was available to compare default rates."
        ]
    worst = with_rate.loc[with_rate["Observed default rate"].idxmax()]
    best = with_rate.loc[with_rate["Observed default rate"].idxmin()]
    findings = [
        f"Highest observed default rate: cohort {worst['Cohort']} at {worst['Observed default rate']:.1%}."
    ]
    if worst["Cohort"] != best["Cohort"]:
        findings.append(
            f"Lowest observed default rate: cohort {best['Cohort']} at {best['Observed default rate']:.1%}."
        )
    return findings


def vintage_analysis(
    records: pd.DataFrame,
    date_column: str | None = None,
    *,
    freq: str = "year",
) -> dict[str, Any]:
    """Observed default rate and exposure by origination cohort.

    Cohorts come from an origination/disbursement date column when available,
    otherwise from a year or vintage-style column. Rates are observed statuses
    in the uploaded book at display precision, not model estimates.
    """
    if freq not in {"year", "quarter"}:
        raise ValueError("freq must be 'year' or 'quarter'.")
    result: dict[str, Any] = {
        "available": False,
        "date_column": None,
        "freq": freq,
        "cohorts": pd.DataFrame(),
        "findings": [],
    }
    if records is None or len(records) == 0:
        result["findings"] = ["No records were available to build vintages."]
        return result

    column = (
        date_column
        if date_column and date_column in records.columns
        else _identified_column(records, _DATE_COLUMNS)
    )
    if column is None:
        result["findings"] = [
            "No origination, disbursement, or vintage column was found, so vintages could not be built."
        ]
        return result

    amount_column = _identified_column(records, _AMOUNT_COLUMNS)
    score_column = "credit_score" if "credit_score" in records.columns else None
    working = pd.DataFrame({
        "_cohort": _cohort_labels(records[column], freq).astype("string").to_numpy(),
        "_default": _observed_default_flags(records).to_numpy(),
        "_exposure": (
            pd.to_numeric(records[amount_column], errors="coerce").to_numpy()
            if amount_column is not None else np.nan
        ),
        "_score": (
            pd.to_numeric(records[score_column], errors="coerce").to_numpy()
            if score_column is not None else np.nan
        ),
    })
    working = working[working["_cohort"].notna() & (working["_cohort"].str.strip() != "")]
    if working.empty:
        result["findings"] = [f"The '{column}' column did not contain usable cohort values."]
        return result

    exposure_available = bool(working["_exposure"].notna().any())
    total_exposure = float(working["_exposure"].sum()) if exposure_available else None
    rows: list[dict[str, Any]] = []
    for label, group in working.groupby("_cohort", sort=True):
        observed = group["_default"].dropna()
        exposure_values = group["_exposure"].dropna()
        score_values = group["_score"].dropna()
        exposure = float(exposure_values.sum()) if not exposure_values.empty else None
        rows.append({
            "Cohort": str(label),
            "Records": int(len(group)),
            "Observed default rate": round(float(observed.mean()), 4) if not observed.empty else None,
            "Observed records": int(len(observed)),
            "Exposure": exposure,
            "Exposure share (%)": (
                round(exposure / total_exposure * 100, 2)
                if exposure is not None and total_exposure else None
            ),
            "Average credit score": round(float(score_values.mean()), 1) if not score_values.empty else None,
        })

    result["available"] = True
    result["date_column"] = column
    result["cohorts"] = pd.DataFrame(rows, columns=[
        "Cohort",
        "Records",
        "Observed default rate",
        "Observed records",
        "Exposure",
        "Exposure share (%)",
        "Average credit score",
    ])
    result["findings"] = _vintage_findings(result["cohorts"])
    return result
