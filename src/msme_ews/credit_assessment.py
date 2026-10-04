"""Plain-language summaries of model and financial risk signals."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


_WARNING_LABELS = {
    "Rapid revenue decline": "declining revenue",
    "Increasing leverage": "increasing leverage",
    "Falling liquidity": "weakening liquidity",
    "Deteriorating margins": "declining operating margins",
    "Negative operating cash flow": "negative operating cash flow",
    "Increasing receivable days": "slower receivables collection",
    "Falling interest coverage": "weaker interest coverage",
}

_FEATURE_LABELS = {
    "Sales_Growth": "revenue growth",
    "EBITDA_Margin": "operating margin",
    "Net_Profit_Margin": "net profit margin",
    "Cash_Flow_Operations": "operating cash flow",
    "Current_Ratio": "liquidity",
    "Debt_to_Assets": "leverage",
    "Debt_to_Equity": "debt-to-equity",
    "Interest_Coverage": "interest coverage",
    "Receivable_Days": "receivables collection",
    "Revenue": "revenue",
    "Net_Profit": "net profit",
}


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _unique(items: Sequence[str], limit: int = 3) -> list[str]:
    return list(dict.fromkeys(items))[:limit]


def generate_risk_interpretation(
    probability: float,
    risk_category: str,
    financials: Mapping[str, object],
    warning_signals: Sequence[str],
    shap_factors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, str]:
    """Create a concise explanation using observed financials, flags, and SHAP."""
    probability = min(1.0, max(0.0, probability))
    signal_set = set(warning_signals)
    concerns = [
        _WARNING_LABELS[signal]
        for signal in _WARNING_LABELS
        if signal in signal_set
    ]

    growth = _number(financials.get("Sales_Growth"))
    margin = _number(financials.get("EBITDA_Margin"))
    cash_flow = _number(financials.get("Cash_Flow_Operations"))
    liquidity = _number(financials.get("Current_Ratio"))
    leverage = _number(financials.get("Debt_to_Assets"))
    coverage = _number(financials.get("Interest_Coverage"))

    if growth is not None and growth < -0.15 and "Rapid revenue decline" not in signal_set:
        concerns.append("declining revenue")
    if margin is not None and margin < 0 and "Deteriorating margins" not in signal_set:
        concerns.append("negative operating margins")
    if cash_flow is not None and cash_flow < 0 and "Negative operating cash flow" not in signal_set:
        concerns.append("negative operating cash flow")
    if liquidity is not None and liquidity < 1 and "Falling liquidity" not in signal_set:
        concerns.append("liquidity below 1.0x")
    if leverage is not None and leverage >= 0.65 and "Increasing leverage" not in signal_set:
        concerns.append("elevated leverage")
    if coverage is not None and coverage < 1.5 and "Falling interest coverage" not in signal_set:
        concerns.append("weak interest coverage")

    concerns = _unique(concerns)
    strengths = []
    if growth is not None and growth > 0:
        strengths.append("revenue growth")
    if liquidity is not None and liquidity >= 1:
        strengths.append("a healthy current ratio")
    if cash_flow is not None and cash_flow > 0:
        strengths.append("positive operating cash flow")
    if margin is not None and margin > 0 and "Deteriorating margins" not in signal_set:
        strengths.append("positive operating margins")
    strengths = _unique(strengths)

    explanation = f"The model estimates a {probability:.1%} probability of default."
    if concerns:
        explanation += f" The primary concerns are {', '.join(concerns)}."
    else:
        explanation += " No configured financial warning is currently active; the estimate reflects the model's combined patterns and should be reviewed with the underlying data."
    if strengths:
        if len(strengths) == 1:
            strength_summary = strengths[0]
        else:
            strength_summary = f"{', '.join(strengths[:-1])} and {strengths[-1]}"
        explanation += f" {strength_summary[0].upper() + strength_summary[1:]} partially offset these risks."

    positive_shap = sorted(
        (
            factor
            for factor in shap_factors
            if (_number(factor.get("contribution")) or 0.0) > 0
        ),
        key=lambda factor: _number(factor.get("contribution")) or 0.0,
        reverse=True,
    )
    drivers = _unique(
        [
            _FEATURE_LABELS.get(str(factor.get("feature")), str(factor.get("feature", "")).replace("_", " ").lower())
            for factor in positive_shap
            if factor.get("feature")
        ]
    )
    if drivers:
        explanation += f" SHAP identifies {', '.join(drivers)} as the strongest contributors increasing the model's estimate."

    priorities = []
    if "Negative operating cash flow" in signal_set or (cash_flow is not None and cash_flow < 0):
        priorities.append("improve operating cash generation")
    if "Increasing leverage" in signal_set or (leverage is not None and leverage >= 0.65):
        priorities.append("control leverage")
    if "Deteriorating margins" in signal_set or (margin is not None and margin < 0):
        priorities.append("protect operating margins")
    if "Falling liquidity" in signal_set or (liquidity is not None and liquidity < 1):
        priorities.append("strengthen short-term liquidity")
    if not priorities:
        priorities.append("continue monitoring cash flow, liquidity, and leverage")

    return {
        "risk_category": risk_category,
        "probability": f"{probability:.1%}",
        "explanation": explanation,
        "priority": " and ".join(priorities[:2]).capitalize() + ".",
    }


def apply_scenario_adjustments(
    financial_data: pd.DataFrame,
    revenue_change: float,
    operating_margin_change: float,
    debt_change: float,
) -> pd.DataFrame:
    """Apply relative revenue/debt changes and a margin percentage-point shift."""
    if len(financial_data) != 1:
        raise ValueError("Scenario analysis requires exactly one company-period.")

    scenario = financial_data.copy()
    values: dict[str, float] = {}
    for column in ("Revenue", "EBITDA", "Debt"):
        if column not in scenario:
            raise ValueError(f"Scenario analysis requires the {column} field.")
        value = _number(scenario.iloc[0][column])
        if value is None:
            raise ValueError(f"Scenario analysis requires a numeric {column} value.")
        values[column] = value
        scenario[column] = pd.to_numeric(scenario[column], errors="coerce").astype(float)

    if values["Revenue"] == 0:
        raise ValueError("Scenario analysis requires nonzero Revenue to adjust the operating margin.")

    adjusted_revenue = values["Revenue"] * (1 + revenue_change)
    baseline_margin = values["EBITDA"] / values["Revenue"]
    adjusted_margin = baseline_margin + operating_margin_change
    row_index = scenario.index[0]
    scenario.at[row_index, "Revenue"] = adjusted_revenue
    scenario.at[row_index, "EBITDA"] = adjusted_revenue * adjusted_margin
    scenario.at[row_index, "Debt"] = values["Debt"] * (1 + debt_change)
    if "Sales_Growth" in scenario:
        baseline_growth = _number(scenario.at[row_index, "Sales_Growth"])
        if baseline_growth is not None:
            scenario.at[row_index, "Sales_Growth"] = (
                (1 + baseline_growth) * (1 + revenue_change) - 1
            )
    return scenario
