"""Deterministic financial feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from msme_ews.data import FINANCIAL_COLUMNS, validate_financial_data

DERIVED_FEATURES = (
    "Current_Ratio", "Quick_Ratio", "Debt_to_Equity", "Debt_to_Assets",
    "Interest_Coverage", "ROA", "ROE", "EBITDA_Margin", "Net_Profit_Margin",
    "Operating_Cash_Flow_Ratio", "Receivable_Days", "Inventory_Days",
)
MODEL_FEATURES = FINANCIAL_COLUMNS + DERIVED_FEATURES
RATIO_LIMIT = 100.0
DAY_LIMIT = 3650.0


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator.abs() > 1e-9)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan).clip(-RATIO_LIMIT, RATIO_LIMIT)


def _days(numerator: pd.Series, revenue: pd.Series) -> pd.Series:
    revenue = revenue.where(revenue.abs() > 1e-9)
    return (numerator / revenue * 365).replace([np.inf, -np.inf], np.nan).clip(0, DAY_LIMIT)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create bounded financial ratios and return a stable numeric feature schema."""
    data = validate_financial_data(frame)
    for column in FINANCIAL_COLUMNS:
        if column not in data:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")

    result = data.loc[:, FINANCIAL_COLUMNS].copy()
    equity = data["Total_Assets"] - data["Total_Liabilities"]
    result["Current_Ratio"] = _ratio(data["Current_Assets"], data["Current_Liabilities"])
    result["Quick_Ratio"] = _ratio(data["Current_Assets"] - data["Inventory"], data["Current_Liabilities"])
    result["Debt_to_Equity"] = _ratio(data["Debt"], equity)
    result["Debt_to_Assets"] = _ratio(data["Debt"], data["Total_Assets"])
    result["Interest_Coverage"] = _ratio(data["EBITDA"], data["Interest_Expense"])
    result["ROA"] = _ratio(data["Net_Profit"], data["Total_Assets"])
    result["ROE"] = _ratio(data["Net_Profit"], equity)
    result["EBITDA_Margin"] = _ratio(data["EBITDA"], data["Revenue"])
    result["Net_Profit_Margin"] = _ratio(data["Net_Profit"], data["Revenue"])
    result["Operating_Cash_Flow_Ratio"] = _ratio(data["Cash_Flow_Operations"], data["Current_Liabilities"])
    result["Receivable_Days"] = _days(data["Accounts_Receivable"], data["Revenue"])
    result["Inventory_Days"] = _days(data["Inventory"], data["Revenue"])

    growth = data["Sales_Growth"]
    if "company_id" in data and "period" in data:
        ordered = data[["company_id", "period", "Revenue"]].copy()
        ordered["_position"] = np.arange(len(ordered))
        ordered["period"] = pd.to_datetime(ordered["period"], errors="coerce")
        ordered = ordered.sort_values(["company_id", "period", "_position"])
        derived = ordered.groupby("company_id", dropna=False)["Revenue"].pct_change(fill_method=None)
        derived.index = ordered.index
        growth = growth.combine_first(derived.reindex(data.index))
    result["Sales_Growth"] = pd.to_numeric(growth, errors="coerce").clip(-10, 10)
    return result.replace([np.inf, -np.inf], np.nan).loc[:, MODEL_FEATURES]