from __future__ import annotations

import pandas as pd

from msme_ews.copilot import (
    build_credit_context,
    generate_credit_copilot_response,
    resolve_question,
    suggested_questions,
)
from msme_ews.demo import make_demo_data
from msme_ews.early_warning import early_warning_indicators
from msme_ews.features import engineer_features
from msme_ews.financial_analysis import analyze_financials, has_sufficient_ml_data
from msme_ews.prediction import predict_financial_health, risk_category


def _context(row_index: int = 0, bundle=None):
    frame = make_demo_data()
    features = engineer_features(frame)
    flags = early_warning_indicators(frame)
    selected = frame.iloc[[row_index]]
    selected_features = features.iloc[[row_index]]
    analysis = analyze_financials(frame, row_index, features)
    if has_sufficient_ml_data(frame, row_index) and bundle is not None:
        result = predict_financial_health(
            selected,
            bundle=bundle,
            include_explanations=False,
            features=selected_features,
        )
        result["method"] = "Existing ML model"
        result["coverage_label"] = (
            f"{analysis['coverage_count']}/{analysis['coverage_total']} core fields"
        )
        result["coverage_percent"] = analysis["coverage_percent"]
    else:
        from msme_ews.financial_analysis import rule_based_assessment

        result = rule_based_assessment(frame, row_index, analysis)
    return frame, build_credit_context(
        frame, selected, selected_features, flags, row_index, result
    )


def test_resolve_question_defaults_and_expands_follow_ups():
    assert resolve_question("") == "summarize the financial health"
    assert resolve_question("What are the biggest risk factors?") == "What are the biggest risk factors?"
    assert resolve_question("Why is it risky?") == "Why is it risky?"

    expanded = resolve_question("what about liquidity?", history=["Why is this company high risk?"])
    assert expanded.startswith("Why is this company high risk")
    assert "liquidity" in expanded

    standalone = resolve_question("Explain the cash conversion cycle in detail for this lender", history=["Why is it risky?"])
    assert standalone == "Explain the cash conversion cycle in detail for this lender"


def test_suggested_questions_are_context_aware_and_bounded():
    _, context = _context()
    prompts = suggested_questions(context)
    assert 5 <= len(prompts) <= 9
    assert all(isinstance(prompt, str) and prompt for prompt in prompts)
    assert any("reliable" in prompt for prompt in prompts)
    assert any("missing" in prompt for prompt in prompts)
    assert any("20%" in prompt for prompt in prompts)
    assert len(set(prompts)) == len(prompts)


def test_reliability_answer_marks_model_and_rule_paths_differently():
    _, model_context = _context(bundle=_stub_bundle())
    model_answer = generate_credit_copilot_response("How reliable is this?", model_context)
    assert "not a guaranteed outcome" in model_answer.lower()
    assert "calibrated uncertainty" in model_answer

    _, rule_context = _context()
    rule_context["assessment_method"] = "Transparent rule-based risk index"
    rule_answer = generate_credit_copilot_response("how confident are you", rule_context)
    assert "heuristic" in rule_answer.lower()
    assert "probability of default" in rule_answer.lower()


def test_missing_fields_answer_does_not_treat_missing_as_healthy():
    _, context = _context()
    answer = generate_credit_copilot_response("Which financial fields are missing?", context)
    assert "coverage" in answer.lower()
    assert "never treated as healthy" in answer.lower()


def test_sensitivity_answer_points_at_breached_thresholds():
    _, context = _context()
    context["current_ratio"] = 0.8
    context["debt_to_assets"] = 0.7
    answer = generate_credit_copilot_response("what if revenue falls by 20%", context)
    assert "liquidity" in answer and "leverage" in answer
    assert "re-run the analysis" in answer
    assert "Scenario Simulator" not in answer


def test_comparison_answer_uses_observed_history():
    _, context = _context(row_index=2)
    answer = generate_credit_copilot_response("How does this compare with the previous period?", context)
    assert "previous recorded period" in answer
    assert context["prior_period"]["period"] in answer

    _, first_period = _context(row_index=0)
    first_period["prior_period"] = None
    without_history = generate_credit_copilot_response("compare to last year", first_period)
    assert "not an earlier dated company-period" in without_history


def test_follow_up_question_is_answered_in_the_context_of_the_previous_one():
    _, context = _context()
    history = ["What are the biggest risk factors?"]
    answer = generate_credit_copilot_response("what about liquidity?", context, history=history)
    assert answer
    assert len(answer) > 40


def test_rule_based_context_never_claims_a_calibrated_probability():
    _, context = _context()
    context["assessment_method"] = "Transparent rule-based risk index"
    context["default_probability"] = None
    answer = generate_credit_copilot_response("summarize the financial health", context)
    assert "rule-based risk index" in answer.lower()
    assert "not a calibrated probability" in answer.lower()


def test_context_exposes_coverage_and_prior_period():
    frame, context = _context(row_index=3)
    assert context["company"] == str(frame.iloc[3]["company_id"])
    assert "core fields" in context["coverage_label"]
    assert context["history_available"] is True
    assert context["prior_period"] is None or "period" in context["prior_period"]
    assert isinstance(context["warnings"], list)


def _stub_bundle():
    from sklearn.dummy import DummyClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    from msme_ews.features import MODEL_FEATURES

    row = make_demo_data().iloc[[0]]
    model = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                      ("classifier", DummyClassifier(strategy="prior"))])
    model.fit(engineer_features(make_demo_data()), [0, 1] * (len(make_demo_data()) // 2))
    return {"model": model, "features": list(MODEL_FEATURES)}


def test_suggested_questions_drop_shap_wording_for_rule_records():
    _, context = _context()
    context["assessment_method"] = "Transparent rule-based risk index"
    prompts = suggested_questions(context)
    assert not any("shap" in prompt.lower() for prompt in prompts)


def test_summary_answer_handles_a_missing_probability():
    _, context = _context(bundle=_stub_bundle())
    context["default_probability"] = None
    context["health_score"] = None
    answer = generate_credit_copilot_response("xyzzy nonsense", context)
    assert "an unavailable" in answer.lower()
    assert context["risk_category"] in answer


def test_risk_categories_are_reported_in_text():
    assert risk_category(0.1) == "Low Risk"
    frame, context = _context(bundle=_stub_bundle())
    assert context["risk_category"] in {"Low Risk", "Moderate Risk", "High Risk"}
    answer = generate_credit_copilot_response("why is this company risky?", context)
    assert answer


def test_unrecognised_question_falls_back_to_a_useful_summary():
    _, context = _context(bundle=_stub_bundle())
    answer = generate_credit_copilot_response("xyzzy nonsense", context)
    assert context["company"] in answer
    assert "Liquidity, leverage" in answer
