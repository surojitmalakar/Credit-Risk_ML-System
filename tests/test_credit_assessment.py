import pandas as pd
import pytest

from msme_ews.credit_assessment import (
    apply_scenario_adjustments,
    generate_risk_interpretation,
)


def test_risk_interpretation_summarizes_financial_signals_and_shap():
    interpretation = generate_risk_interpretation(
        probability=0.143,
        risk_category="Moderate Risk",
        financials={
            "Sales_Growth": 0.12,
            "EBITDA_Margin": 0.08,
            "Cash_Flow_Operations": -200,
            "Current_Ratio": 1.29,
            "Debt_to_Assets": 0.4,
        },
        warning_signals=["Deteriorating margins", "Negative operating cash flow"],
        shap_factors=[
            {"feature": "EBITDA_Margin", "contribution": 0.2},
            {"feature": "Cash_Flow_Operations", "contribution": 0.1},
        ],
    )

    assert interpretation["risk_category"] == "Moderate Risk"
    assert interpretation["probability"] == "14.3%"
    assert "declining operating margins" in interpretation["explanation"]
    assert "negative operating cash flow" in interpretation["explanation"]
    assert "Revenue growth" in interpretation["explanation"]
    assert "healthy current ratio" in interpretation["explanation"]
    assert "SHAP identifies operating margin, operating cash flow" in interpretation["explanation"]
    assert "Improve operating cash generation" in interpretation["priority"]


def test_risk_interpretation_does_not_invent_revenue_growth():
    interpretation = generate_risk_interpretation(
        probability=0.3,
        risk_category="Moderate Risk",
        financials={"Revenue": 500, "Sales_Growth": -0.02},
        warning_signals=[],
    )

    assert "revenue growth" not in interpretation["explanation"]
    assert "declining revenue" not in interpretation["explanation"]


def test_scenario_adjustments_apply_relative_revenue_debt_and_margin_point_change():
    frame = pd.DataFrame([{"Revenue": 1000, "EBITDA": 100, "Debt": 250, "Sales_Growth": 0.1}])

    scenario = apply_scenario_adjustments(
        frame,
        revenue_change=-0.2,
        operating_margin_change=-0.03,
        debt_change=0.1,
    )

    assert scenario.iloc[0]["Revenue"] == pytest.approx(800)
    assert scenario.iloc[0]["EBITDA"] == pytest.approx(56)
    assert scenario.iloc[0]["Debt"] == pytest.approx(275)
    assert scenario.iloc[0]["Sales_Growth"] == pytest.approx(-0.12)
    assert frame.iloc[0]["Revenue"] == 1000


def test_scenario_adjustments_reject_missing_or_multiple_rows():
    with pytest.raises(ValueError, match="exactly one"):
        apply_scenario_adjustments(pd.DataFrame([{}, {}]), 0, 0, 0)
    with pytest.raises(ValueError, match="requires the Debt field"):
        apply_scenario_adjustments(pd.DataFrame([{"Revenue": 10, "EBITDA": 2}]), 0, 0, 0)
