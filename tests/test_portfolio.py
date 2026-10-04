from __future__ import annotations

import pandas as pd
import pytest

from msme_ews.portfolio import (
    analyze_credit_portfolio,
    portfolio_concentration,
    vintage_analysis,
)


def _loan_book() -> pd.DataFrame:
    """Five equally-sized loans that carry both risk and default signals."""
    return pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4", "C5"],
        "loan_amount": [100.0, 100.0, 100.0, 100.0, 100.0],
        "credit_score": [700, 700, 700, 700, 700],
        "default_status": ["No", "No", "No", "No", "Yes"],
    })


def test_concentration_computes_borrower_shares_and_hhi():
    concentration = analyze_credit_portfolio(_loan_book())["concentration"]

    assert concentration["available"] is True
    assert concentration["exposure_column"] == "loan_amount"
    assert concentration["total_exposure"] == 500.0
    assert concentration["metrics"]["borrowers"] == 5
    assert concentration["metrics"]["largest_exposure_share"] == 0.2
    assert concentration["metrics"]["herfindahl_index"] == 0.2
    assert concentration["metrics"]["effective_borrowers"] == 5.0
    assert concentration["metrics"]["top_10_share"] == 1.0
    assert set(concentration["band_breakdown"]["Risk category"]) <= {"Low Risk", "High Risk"}
    assert "Borrower" in concentration["top_exposures"].columns


def test_concentration_aggregates_multiple_loans_per_borrower():
    frame = pd.DataFrame({
        "customer_id": ["C1", "C1", "C2"],
        "loan_amount": [50.0, 50.0, 100.0],
        "credit_score": [700, 700, 700],
    })
    concentration = portfolio_concentration(frame)

    assert concentration["metrics"]["borrowers"] == 2
    assert concentration["total_exposure"] == 200.0
    assert concentration["metrics"]["largest_exposure_share"] == 0.5
    assert concentration["metrics"]["herfindahl_index"] == 0.5
    assert concentration["metrics"]["effective_borrowers"] == 2.0


def test_concentration_reports_when_no_amount_column_exists():
    concentration = portfolio_concentration(pd.DataFrame({"customer_id": ["C1", "C2"], "region": ["N", "S"]}))

    assert concentration["available"] is False
    assert concentration["total_exposure"] is None
    assert concentration["band_breakdown"].empty
    assert concentration["findings"]


def test_vintage_analysis_groups_defaults_and_exposure_by_origination_year():
    frame = pd.DataFrame({
        "loan_id": ["L1", "L2", "L3", "L4"],
        "loan_amount": [100.0, 100.0, 100.0, 100.0],
        "origination_date": ["2021-01-15", "2021-06-15", "2022-03-10", "2022-09-20"],
        "default_status": ["Yes", "No", "No", "No"],
        "credit_score": [600, 700, 650, 720],
    })
    vintage = vintage_analysis(frame)

    assert vintage["available"] is True
    assert vintage["date_column"] == "origination_date"
    cohorts = vintage["cohorts"].set_index("Cohort")
    assert list(cohorts.index) == ["2021", "2022"]
    assert cohorts.loc["2021", "Records"] == 2
    assert cohorts.loc["2021", "Observed default rate"] == 0.5
    assert cohorts.loc["2022", "Observed default rate"] == 0.0
    assert cohorts.loc["2021", "Exposure share (%)"] == 50.0


def test_vintage_analysis_supports_quarterly_cohorts():
    frame = pd.DataFrame({
        "loan_id": ["L1", "L2"],
        "loan_amount": [100.0, 100.0],
        "origination_date": ["2021-02-01", "2021-08-01"],
    })
    vintage = vintage_analysis(frame, freq="quarter")

    assert list(vintage["cohorts"]["Cohort"]) == ["2021Q1", "2021Q3"]


def test_vintage_analysis_uses_a_year_column_without_dates():
    frame = pd.DataFrame({
        "loan_id": ["L1", "L2", "L3"],
        "loan_amount": [100.0, 100.0, 100.0],
        "origination_year": [2021, 2021, 2022],
        "default_status": ["No", "No", "Yes"],
    })
    vintage = vintage_analysis(frame)

    assert vintage["available"] is True
    assert vintage["date_column"] == "origination_year"
    cohorts = vintage["cohorts"].set_index("Cohort")
    assert cohorts.loc["2022", "Observed default rate"] == 1.0


def test_vintage_analysis_reports_when_no_cohort_column_exists():
    vintage = vintage_analysis(pd.DataFrame({"loan_id": ["L1"], "loan_amount": [100.0]}))

    assert vintage["available"] is False
    assert vintage["findings"]


def test_vintage_analysis_rejects_unknown_frequency():
    with pytest.raises(ValueError, match="freq"):
        vintage_analysis(_loan_book(), freq="month")


def test_analyze_credit_portfolio_exposes_concentration_and_vintage():
    analysis = analyze_credit_portfolio(_loan_book())

    assert {"customers", "summary", "concentration", "vintage"} <= set(analysis)
    assert analysis["concentration"]["available"] is True
    # The book has no origination column, so vintages are reported as unavailable.
    assert analysis["vintage"]["available"] is False
