"""Transparent heuristic flags for deterioration in financial indicators."""

from __future__ import annotations

import pandas as pd

from msme_ews.data import validate_financial_data
from msme_ews.features import engineer_features


def _prior_period_values(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    if "company_id" not in frame or "period" not in frame:
        return pd.Series(float("nan"), index=frame.index)
    history = frame[["company_id", "period"]].copy()
    history["_position"] = range(len(frame))
    history["_value"] = values.to_numpy()
    history["period"] = pd.to_datetime(history["period"], errors="coerce")
    history = history.sort_values(["company_id", "period", "_position"])
    history["_prior"] = history.groupby("company_id", dropna=False)["_value"].shift()
    previous = history.set_index("_position")["_prior"].reindex(range(len(frame))).to_numpy()
    return pd.Series(previous, index=frame.index)


def early_warning_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rule-based warning flags; they are not model predictions."""
    data = validate_financial_data(frame)
    features = engineer_features(data)
    previous_leverage = _prior_period_values(data, features["Debt_to_Assets"])
    previous_liquidity = _prior_period_values(data, features["Current_Ratio"])
    previous_margin = _prior_period_values(data, features["EBITDA_Margin"])
    previous_receivable_days = _prior_period_values(data, features["Receivable_Days"])
    previous_coverage = _prior_period_values(data, features["Interest_Coverage"])
    flags = pd.DataFrame(index=features.index)
    flags["Rapid revenue decline"] = features["Sales_Growth"] <= -0.15
    flags["Increasing leverage"] = (features["Debt_to_Assets"] >= 0.65) | (
        features["Debt_to_Assets"] > previous_leverage + 0.05
    )
    flags["Falling liquidity"] = (features["Current_Ratio"] < 1.0) | (
        features["Current_Ratio"] < previous_liquidity * 0.9
    )
    flags["Deteriorating margins"] = (
        (features["EBITDA_Margin"] < 0)
        | (features["Net_Profit_Margin"] < 0)
        | (features["EBITDA_Margin"] < previous_margin)
    )
    flags["Negative operating cash flow"] = features["Cash_Flow_Operations"] < 0
    flags["Increasing receivable days"] = (features["Receivable_Days"] > 90) | (
        features["Receivable_Days"] > previous_receivable_days + 10
    )
    flags["Falling interest coverage"] = (features["Interest_Coverage"] < 1.5) | (
        features["Interest_Coverage"] < previous_coverage
    )
    return flags.fillna(False).astype(bool)


def trend_data(frame: pd.DataFrame, company_id: object) -> pd.DataFrame:
    """Prepare chronological financial health series for one company."""
    if "company_id" not in frame or "period" not in frame:
        return pd.DataFrame()
    selected = frame.loc[frame["company_id"].astype(str) == str(company_id)].copy()
    if selected.empty:
        return selected
    selected["period"] = pd.to_datetime(selected["period"], errors="coerce")
    selected = selected.sort_values("period")
    engineered = engineer_features(selected)
    return pd.DataFrame({
        "period": selected["period"].to_numpy(), "Revenue": selected["Revenue"].to_numpy(),
        "Current_Ratio": engineered["Current_Ratio"].to_numpy(),
        "Debt_to_Assets": engineered["Debt_to_Assets"].to_numpy(),
        "EBITDA_Margin": engineered["EBITDA_Margin"].to_numpy(),
        "Interest_Coverage": engineered["Interest_Coverage"].to_numpy(),
    }, index=selected.index)