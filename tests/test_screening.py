from __future__ import annotations

import math

import numpy as np
import pandas as pd

from msme_ews.demo import make_demo_data
from msme_ews.early_warning import early_warning_indicators
from msme_ews.features import engineer_features
from msme_ews.financial_analysis import (
    analyze_financials,
    has_sufficient_ml_data,
    model_eligible_rows,
    rule_based_assessment,
    rule_based_risk_index,
)
from msme_ews.prediction import DEFAULT_MODEL_PATH, load_model_bundle
from msme_ews.screening import (
    banded_exposure,
    detect_exposure_column,
    record_snapshot,
    score_records,
    screening_summary,
)


def _sparse_frame(rows: int = 240) -> pd.DataFrame:
    """Records with varying field coverage so both the model and rule paths run."""
    rng = np.random.default_rng(11)
    combinations = [
        ["Revenue", "Current_Assets", "Current_Liabilities"],
        ["Revenue", "Debt", "Total_Assets", "Total_Liabilities"],
        ["Revenue", "Net_Profit", "Interest_Expense", "EBITDA"],
        ["Revenue", "Cash_Flow_Operations", "Sales_Growth"],
        ["Revenue", "Net_Profit"],
        ["Revenue"],
    ]
    records = []
    for index in range(rows):
        record: dict[str, object] = {"company_id": f"C{index % 7}", "period": f"20{18 + index % 5}"}
        for column in combinations[index % len(combinations)]:
            if column in {"Current_Assets", "Total_Assets"}:
                record[column] = rng.uniform(1e5, 6e6)
            elif column in {"Current_Liabilities", "Total_Liabilities"}:
                record[column] = rng.uniform(5e4, 4e6)
            elif column == "Debt":
                record[column] = rng.uniform(0, 4e6)
            elif column == "Interest_Expense":
                record[column] = rng.uniform(1e3, 8e4)
            elif column == "EBITDA":
                record[column] = rng.uniform(-2e5, 7e5)
            elif column == "Net_Profit":
                record[column] = rng.uniform(-3e5, 6e5)
            elif column == "Cash_Flow_Operations":
                record[column] = rng.uniform(-4e5, 5e5)
            elif column == "Sales_Growth":
                record[column] = rng.uniform(-0.5, 0.5)
            else:
                record[column] = rng.uniform(5e4, 5e6)
        records.append(record)
    return pd.DataFrame(records)


def round4(value: float) -> float:
    """Scores are displayed to four decimals, so compare at display precision."""
    return round(float(value), 4)


def test_vectorized_rule_index_matches_the_scalar_index_exactly():
    frame = _sparse_frame()
    features = engineer_features(frame)
    vectorized = rule_based_risk_index(frame, features)
    compared = 0
    for row_index in range(len(frame)):
        scalar = rule_based_assessment(frame, row_index, analyze_financials(frame, row_index, features))
        compared += 1
        expected = scalar["risk_index"]
        actual = vectorized["Rule risk index"].iloc[row_index]
        if expected is None:
            assert pd.isna(actual)
        else:
            # Vectorized and scalar paths sum the same weights in a different
            # order, so compare within floating-point tolerance.
            assert math.isclose(float(actual), expected, abs_tol=1e-9)
            assert round4(actual) == round4(expected)
        assert vectorized["Rule risk category"].iloc[row_index] == scalar["risk_category"]
        assert {
            name for name in str(vectorized["Rule drivers"].iloc[row_index]).split(", ") if name
        } == {factor["feature"] for factor in scalar["top_risk_factors"]}
    assert compared == len(frame)


def test_batch_scores_match_single_record_rule_indices():
    frame = make_demo_data()
    features = engineer_features(frame)
    bundle = load_model_bundle(DEFAULT_MODEL_PATH)
    scores = score_records(frame, features, early_warning_indicators(frame), bundle)
    for row_index in range(0, len(frame), 13):
        single = rule_based_assessment(
            frame,
            row_index,
            analyze_financials(frame, row_index, features),
        )
        batch_index = scores["Risk index (0-100)"].iloc[row_index]
        if single["risk_index"] is None:
            assert pd.isna(batch_index)
        else:
            assert float(batch_index) == round4(single["risk_index"])


def test_model_eligibility_is_vectorized_equivalent():
    frame = _sparse_frame(120)
    mask = model_eligible_rows(frame)
    assert mask.sum() > 0
    for row_index in range(len(frame)):
        assert bool(mask.iloc[row_index]) == has_sufficient_ml_data(frame, row_index)


def test_score_records_reports_both_methods_and_keeps_positions():
    frame = _sparse_frame(120)
    features = engineer_features(frame)
    flags = early_warning_indicators(frame)
    scores = score_records(frame, features, flags, load_model_bundle(DEFAULT_MODEL_PATH))

    assert len(scores) == len(frame)
    assert scores["_position"].tolist() == list(range(len(frame)))
    assert set(scores["Method"]) <= {"Rule-based risk index", "Insufficient data"}
    assert not scores["ML model used"].any()
    assert (scores["Method"] == "Rule-based risk index").any()
    assert scores["Health score"].dropna().between(0, 100).all()
    assert scores["Coverage %"].between(0, 100).all()


def test_screening_summary_aggregates_bands_and_coverage():
    frame = make_demo_data()
    features = engineer_features(frame)
    flags = early_warning_indicators(frame)
    scores = score_records(frame, features, flags, load_model_bundle(DEFAULT_MODEL_PATH))
    summary = screening_summary(scores)

    assert summary["records"] == len(frame)
    assert sum(summary["bands"].values()) == len(frame)
    assert summary["model_records"] == int(scores["ML model used"].sum())
    assert summary["flagged_records"] == int((scores["Warning signals"] > 0).sum())
    assert 0 <= summary["mean_risk_index"] <= 100


def test_screening_works_without_a_model_bundle():
    frame = _sparse_frame(90)
    features = engineer_features(frame)
    scores = score_records(frame, features, None, None)
    assert not scores["ML model used"].any()
    assert (scores["Method"] == "Rule-based risk index").any()


def test_banded_exposure_uses_the_detected_amount_column():
    frame = make_demo_data().rename(columns={"Revenue": "loan_amount"})
    features = engineer_features(frame)
    scores = score_records(frame, features, early_warning_indicators(frame))

    assert detect_exposure_column(frame) == "loan_amount"
    exposure = banded_exposure(scores, frame, "loan_amount")
    assert not exposure.empty
    assert set(exposure["Risk category"]) <= set(scores["Risk category"])
    assert banded_exposure(scores, frame, "missing_column").empty
    assert detect_exposure_column(pd.DataFrame({"a": [1], "b": [2]})) is None


def test_record_snapshot_matches_the_selected_record_assessment():
    frame = _sparse_frame(90)
    features = engineer_features(frame)
    flags = early_warning_indicators(frame)
    bundle = load_model_bundle(DEFAULT_MODEL_PATH)
    scores = score_records(frame, features, flags, bundle)

    for row_index in (0, 5, 17, 40):
        snapshot = record_snapshot(frame, row_index, features, flags, bundle)
        assert snapshot["row_index"] == row_index
        assert snapshot["risk_category"] == scores["Risk category"].iloc[row_index]
        batch_index = scores["Risk index (0-100)"].iloc[row_index]
        if snapshot["risk_index"] is None:
            assert pd.isna(batch_index)
        else:
            assert snapshot["risk_index"] == round4(batch_index)
        assert snapshot["method"] == scores["Method"].iloc[row_index]
        assert snapshot["coverage_label"].endswith("%)")
