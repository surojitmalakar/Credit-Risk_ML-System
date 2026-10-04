"""Observed financial ratios, coverage, and a transparent sparse-data fallback."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from msme_ews.data import FINANCIAL_COLUMNS
from msme_ews.features import engineer_features
from msme_ews.prediction import risk_category


_MODEL_MINIMUM_FIELDS = 4
_MODEL_CORE_FIELDS = {
    "Revenue", "EBITDA", "Net_Profit", "Total_Assets", "Total_Liabilities",
    "Current_Assets", "Current_Liabilities", "Cash_Flow_Operations", "Debt",
}
_RATIO_FIELDS = {
    "Current Ratio": "Current_Ratio",
    "Quick Ratio": "Quick_Ratio",
    "Debt / Equity": "Debt_to_Equity",
    "Debt / Assets": "Debt_to_Assets",
    "Interest Coverage": "Interest_Coverage",
    "Net Profit Margin": "Net_Profit_Margin",
    "EBITDA Margin": "EBITDA_Margin",
    "Return on Assets": "ROA",
    "Return on Equity": "ROE",
    "Receivable Days": "Receivable_Days",
    "Inventory Days": "Inventory_Days",
}


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def analyze_financials(frame: pd.DataFrame, row_index: int) -> dict[str, Any]:
    """Calculate ratios and data coverage for a selected company-period."""
    row = frame.iloc[row_index]
    feature = engineer_features(frame).iloc[row_index]
    base_fields = [column for column in FINANCIAL_COLUMNS if column != "Sales_Growth"]
    available = [column for column in base_fields if _finite(row.get(column)) is not None]
    ratios = {
        label: _finite(feature.get(column))
        for label, column in _RATIO_FIELDS.items()
    }

    current_assets = _finite(row.get("Current_Assets"))
    current_liabilities = _finite(row.get("Current_Liabilities"))
    if current_assets is not None and current_liabilities is not None:
        ratios["Working Capital"] = current_assets - current_liabilities
    else:
        ratios["Working Capital"] = None

    assets = _finite(row.get("Total_Assets"))
    liabilities = _finite(row.get("Total_Liabilities"))
    debt = _finite(row.get("Debt"))
    equity = _finite(row.get("Equity_Value"))
    if equity is None and assets is not None and liabilities is not None:
        equity = assets - liabilities
    if equity is None and assets is not None and debt is not None:
        equity = assets - debt
    ratios["Equity"] = equity
    if ratios.get("Debt / Equity") is None and debt is not None and equity not in (None, 0):
        ratios["Debt / Equity"] = debt / equity
    net_profit = _finite(row.get("Net_Profit"))
    if ratios.get("Return on Equity") is None and net_profit is not None and equity not in (None, 0):
        ratios["Return on Equity"] = net_profit / equity
    if ratios.get("Return on Assets") is None and net_profit is not None and assets not in (None, 0):
        ratios["Return on Assets"] = net_profit / assets

    revenue = _finite(row.get("Revenue"))
    cash_flow = _finite(row.get("Cash_Flow_Operations"))
    ratios["Operating Cash Flow / Revenue"] = (
        cash_flow / revenue if cash_flow is not None and revenue not in (None, 0) else None
    )
    ratios["Sales Growth"] = _finite(feature.get("Sales_Growth"))
    return {
        "ratios": ratios,
        "coverage_count": len(available),
        "coverage_total": len(base_fields),
        "coverage_percent": 100 * len(available) / len(base_fields),
        "available_fields": available,
        "missing_fields": [column for column in base_fields if column not in available],
        "features": feature,
    }


def has_sufficient_ml_data(frame: pd.DataFrame, row_index: int) -> bool:
    """Require multiple observed financial dimensions before using the ML model."""
    row = frame.iloc[row_index]
    observed = {
        column for column in FINANCIAL_COLUMNS
        if _finite(row.get(column)) is not None
    }
    return (
        len(observed) >= _MODEL_MINIMUM_FIELDS
        and len(observed & _MODEL_CORE_FIELDS) >= 2
    )


def rule_based_assessment(
    frame: pd.DataFrame,
    row_index: int,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a clearly labeled heuristic risk index when ML inputs are sparse."""
    analysis = analysis or analyze_financials(frame, row_index)
    row = frame.iloc[row_index]
    ratio = analysis["ratios"]
    observed: list[tuple[str, float]] = []
    protective: list[str] = []

    current_ratio = ratio.get("Current Ratio")
    if current_ratio is not None:
        if current_ratio < 1.0:
            observed.append(("Current Ratio", 0.22))
        elif current_ratio < 1.2:
            observed.append(("Current Ratio", 0.10))
        elif current_ratio >= 1.5:
            protective.append("Current Ratio")

    debt_equity = ratio.get("Debt / Equity")
    if debt_equity is not None and debt_equity < 0:
        observed.append(("Negative Equity", 0.20))
    elif debt_equity is not None and debt_equity > 2:
        observed.append(("Debt / Equity", 0.16))
    elif debt_equity is not None and debt_equity >= 0:
        protective.append("Debt / Equity")

    debt_assets = ratio.get("Debt / Assets")
    if debt_assets is not None and debt_assets >= 0.65:
        observed.append(("Debt / Assets", 0.18))

    for label, threshold, weight in (
        ("Net Profit Margin", 0.0, 0.15),
        ("Return on Assets", 0.0, 0.10),
        ("Interest Coverage", 1.5, 0.12),
    ):
        value = ratio.get(label)
        if value is not None and value < threshold:
            observed.append((label, weight))
        elif value is not None:
            protective.append(label)

    growth = _finite(analysis["features"].get("Sales_Growth"))
    if growth is not None and growth < -0.15:
        observed.append(("Revenue Growth", 0.12))
    elif growth is not None and growth > 0:
        protective.append("Revenue Growth")

    cash_flow = _finite(row.get("Cash_Flow_Operations"))
    if cash_flow is not None and cash_flow < 0:
        observed.append(("Operating Cash Flow", 0.15))
    elif cash_flow is not None and cash_flow > 0:
        protective.append("Operating Cash Flow")

    observed_fields = analysis["available_fields"]
    if len(observed_fields) < 2:
        probability = None
    else:
        probability = min(0.95, max(0.05, 0.10 + sum(weight for _, weight in observed) - min(0.20, len(protective) * 0.04)))

    category = risk_category(probability) if probability is not None else "Insufficient Data"
    return {
        "distress_probability": probability,
        "risk_category": category,
        "confidence_indicator": analysis["coverage_percent"] / 100,
        "confidence_note": f"Data coverage: {analysis['coverage_count']} of {analysis['coverage_total']} core financial fields; this is not statistical confidence.",
        "coverage_percent": analysis["coverage_percent"],
        "coverage_label": (
            f"{analysis['coverage_count']}/{analysis['coverage_total']} core fields "
            f"({analysis['coverage_percent']:.0f}%)"
        ),
        "top_risk_factors": [
            {"feature": feature, "contribution": weight}
            for feature, weight in sorted(observed, key=lambda item: item[1], reverse=True)
        ],
        "protective_factors": [
            {"feature": feature}
            for feature in dict.fromkeys(protective)
        ],
        "method": "Transparent rule-based risk index; not a calibrated probability of default.",
    }


def recommendations_for(
    row: pd.Series,
    analysis: dict[str, Any],
    warning_signals: list[str],
) -> list[str]:
    """Build recommendations only from observed ratios and warning signals."""
    recommendations: list[str] = []
    signals = set(warning_signals)
    ratios = analysis["ratios"]
    if "Negative operating cash flow" in signals or (
        _finite(row.get("Cash_Flow_Operations")) is not None
        and float(row["Cash_Flow_Operations"]) < 0
    ):
        recommendations.append("Review receivable collection and working-capital conversion to improve operating cash generation.")
    if "Increasing leverage" in signals or (
        ratios.get("Debt / Assets") is not None and ratios["Debt / Assets"] >= 0.65
    ):
        recommendations.append("Monitor debt maturities and evaluate a measured reduction in leverage.")
    if "Falling liquidity" in signals or (
        ratios.get("Current Ratio") is not None and ratios["Current Ratio"] < 1
    ):
        recommendations.append("Strengthen near-term liquidity and align short-term obligations with available current assets.")
    if "Deteriorating margins" in signals or (
        ratios.get("Net Profit Margin") is not None and ratios["Net Profit Margin"] < 0
    ):
        recommendations.append("Review cost drivers and protect operating and net profit margins.")
    if "Rapid revenue decline" in signals:
        recommendations.append("Investigate the reported revenue decline and update forecasts using verified customer demand.")
    if not recommendations:
        if analysis["coverage_count"] < 4:
            recommendations.append("Provide additional periods and balance-sheet fields to improve assessment coverage.")
        else:
            recommendations.append("Continue monitoring cash flow, liquidity, profitability, and leverage as new periods become available.")
    return recommendations
