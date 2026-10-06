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


def analyze_financials(
    frame: pd.DataFrame,
    row_index: int,
    features: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Calculate ratios and data coverage for a selected company-period.

    ``features`` accepts an already engineered frame so repeated single-record
    lookups reuse one feature pass instead of re-deriving every row.
    """
    row = frame.iloc[row_index]
    feature = (
        features.iloc[row_index]
        if features is not None
        else engineer_features(frame).iloc[row_index]
    )
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

    # ROCE: operating return on capital employed (EBITDA over capital employed).
    ebitda = _finite(row.get("EBITDA"))
    capital_employed = (
        assets - current_liabilities
        if assets is not None and current_liabilities is not None
        else None
    )
    ratios["ROCE"] = (
        ebitda / capital_employed
        if ebitda is not None and capital_employed not in (None, 0)
        else None
    )
    # Gross profit / margin are only shown when cost of goods sold is observed.
    cogs = _finite(row.get("Cost_of_Goods_Sold"))
    if revenue is not None and cogs is not None:
        ratios["Gross Profit"] = revenue - cogs
        ratios["Gross Margin"] = (revenue - cogs) / revenue if revenue != 0 else None
    else:
        ratios["Gross Profit"] = None
        ratios["Gross Margin"] = None
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
    risk_index = None
    if len(observed_fields) >= 2:
        risk_index = 100 * min(
            1.0,
            max(
                0.0,
                sum(weight for _, weight in observed)
                - min(0.20, len(protective) * 0.04),
            ),
        )

    category = (
        risk_category(risk_index / 100)
        if risk_index is not None
        else "Insufficient Data"
    )
    return {
        "distress_probability": None,
        "risk_index": risk_index,
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
        "method": "Transparent weighted rule-based risk index (0-100 points), not a probability of default.",
    }


_INDEX_FIELDS = tuple(column for column in FINANCIAL_COLUMNS if column != "Sales_Growth")
_RULE_WEIGHTS = {
    "Negative Equity": 0.20,
    "Debt / Equity": 0.16,
    "Debt / Assets": 0.18,
    "Net Profit Margin": 0.15,
    "Return on Assets": 0.10,
    "Interest Coverage": 0.12,
    "Revenue Growth": 0.12,
    "Operating Cash Flow": 0.15,
}
# Current Ratio is scored in two bands, matching rule_based_assessment.
_CURRENT_RATIO_TIERS = ((1.0, 0.22), (1.2, 0.10))


def observed_field_counts(frame: pd.DataFrame) -> pd.Series:
    """Count usable core financial fields per record without a Python-level loop."""
    observed = pd.DataFrame(index=frame.index)
    for column in _INDEX_FIELDS:
        source = frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index)
        observed[column] = pd.to_numeric(source, errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
    return observed.sum(axis=1).astype("int64")


def model_eligible_rows(frame: pd.DataFrame) -> pd.Series:
    """Vectorized equivalent of :func:`has_sufficient_ml_data` for every record."""
    core = pd.DataFrame(index=frame.index)
    for column in _MODEL_CORE_FIELDS:
        source = frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index)
        core[column] = pd.to_numeric(source, errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
    return (observed_field_counts(frame) >= _MODEL_MINIMUM_FIELDS) & (core.sum(axis=1) >= 2)


def _debt_to_equity(frame: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    ratio = features["Debt_to_Equity"] if "Debt_to_Equity" in features else pd.Series(np.nan, index=frame.index)
    debt = pd.to_numeric(frame["Debt"], errors="coerce") if "Debt" in frame else pd.Series(np.nan, index=frame.index)
    assets = pd.to_numeric(frame["Total_Assets"], errors="coerce") if "Total_Assets" in frame else pd.Series(np.nan, index=frame.index)
    liabilities = (
        pd.to_numeric(frame["Total_Liabilities"], errors="coerce")
        if "Total_Liabilities" in frame else pd.Series(np.nan, index=frame.index)
    )
    equity = assets - liabilities
    equity = equity.where(assets.notna() & liabilities.notna())
    equity = equity.fillna(assets - debt)
    if "Equity_Value" in frame.columns:
        equity = pd.to_numeric(frame["Equity_Value"], errors="coerce").combine_first(equity)
    denominator = equity.where(equity.abs() > 1e-9)
    return ratio.combine_first((debt / denominator).replace([np.inf, -np.inf], np.nan))


def rule_based_risk_index(frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Vectorized equivalent of :func:`rule_based_assessment` for every record.

    Returns one row per record with the heuristic index, band, and the observed
    drivers, so a whole file can be screened in a single vectorized pass.
    """

    def column(name: str) -> pd.Series:
        if name in features:
            return pd.to_numeric(features[name], errors="coerce")
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    current_ratio = column("Current_Ratio")
    debt_equity = _debt_to_equity(frame, features)
    debt_assets = column("Debt_to_Assets")
    net_margin = column("Net_Profit_Margin")
    return_on_assets = column("ROA")
    coverage = column("Interest_Coverage")
    growth = column("Sales_Growth")
    cash_flow = (
        pd.to_numeric(frame["Cash_Flow_Operations"], errors="coerce")
        if "Cash_Flow_Operations" in frame else pd.Series(np.nan, index=frame.index, dtype="float64")
    )

    risk_flags = pd.DataFrame(index=frame.index)
    protective = pd.DataFrame(index=frame.index)
    risk_flags["Negative Equity"] = debt_equity < 0
    risk_flags["Debt / Equity"] = (debt_equity > 2) & ~risk_flags["Negative Equity"]
    risk_flags["Current Ratio"] = current_ratio < 1.2
    risk_flags["Debt / Assets"] = debt_assets >= 0.65
    risk_flags["Net Profit Margin"] = net_margin < 0
    risk_flags["Return on Assets"] = return_on_assets < 0
    risk_flags["Interest Coverage"] = coverage < 1.5
    risk_flags["Revenue Growth"] = growth < -0.15
    risk_flags["Operating Cash Flow"] = cash_flow < 0

    protective["Current Ratio"] = current_ratio >= 1.5
    protective["Debt / Equity"] = (debt_equity >= 0) & ~(risk_flags["Negative Equity"] | risk_flags["Debt / Equity"])
    protective["Net Profit Margin"] = net_margin >= 0
    protective["Return on Assets"] = return_on_assets >= 0
    protective["Interest Coverage"] = coverage >= 1.5
    protective["Revenue Growth"] = growth > 0
    protective["Operating Cash Flow"] = cash_flow > 0

    weight_matrix = risk_flags.astype("float64") * pd.Series(_RULE_WEIGHTS, dtype="float64")
    weight_matrix["Current Ratio"] = risk_flags["Current Ratio"].astype("float64") * pd.Series(
        np.select(
            [current_ratio < _CURRENT_RATIO_TIERS[0][0], current_ratio < _CURRENT_RATIO_TIERS[1][0]],
            [weight for _, weight in _CURRENT_RATIO_TIERS],
            default=0.0,
        ),
        index=frame.index,
        dtype="float64",
    )
    risk_score = weight_matrix.sum(axis=1)
    protective_count = protective.sum(axis=1).astype("int64")
    risk_factor_count = risk_flags.sum(axis=1).astype("int64")
    observed_fields = observed_field_counts(frame)

    risk_index = (
        100 * (risk_score - np.minimum(0.20, protective_count * 0.04)).clip(0.0, 1.0)
    ).where(observed_fields >= 2)
    category = risk_index.map(
        lambda value: (
            "Insufficient Data"
            if pd.isna(value)
            else risk_category(float(value) / 100)
        )
    )

    labels = risk_flags.apply(
        lambda row: ", ".join(name for name in risk_flags.columns if bool(row[name])),
        axis=1,
    )
    return pd.DataFrame({
        "Rule risk index": risk_index,
        "Rule risk category": category.astype("string"),
        "Rule risk score": risk_score.round(4),
        "Rule risk factors": risk_factor_count,
        "Rule drivers": labels.astype("string"),
        "Rule protective factors": protective_count,
        "Observed core fields": observed_fields,
    }, index=frame.index)


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
