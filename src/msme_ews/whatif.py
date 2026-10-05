"""Generic what-if scenario engine for financial risk analysis.

Adjusts observed financial fields by percentage changes, then re-runs the
existing deterministic pipeline (``engineer_features`` +
``analyze_financials`` + ``rule_based_assessment``). Never mutates the
original data. Results are labeled HYPOTHETICAL by callers.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from msme_ews.credit_assessment import apply_scenario_adjustments
from msme_ews.early_warning import early_warning_indicators
from msme_ews.features import engineer_features
from msme_ews.financial_analysis import analyze_financials, rule_based_assessment

# Spec variables -> (canonical column, kind). Margin is percentage-points.
WHATIF_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("revenue", "Revenue", "Revenue (%)"),
    ("profit_margin", "__margin__", "Profit margin (pp)"),
    ("debt", "Debt", "Debt (%)"),
    ("cash", "Cash_Balance", "Cash (%)"),
    ("assets", "Total_Assets", "Assets (%)"),
    ("liabilities", "Total_Liabilities", "Liabilities (%)"),
    ("receivables", "Accounts_Receivable", "Receivables (%)"),
    ("inventory", "Inventory", "Inventory (%)"),
    ("ocf", "Cash_Flow_Operations", "Operating cash flow (%)"),
    ("interest", "Interest_Expense", "Interest expense (%)"),
)


def available_whatif_fields(row: pd.Series) -> list[tuple[str, str, str]]:
    out = []
    for key, col, label in WHATIF_FIELDS:
        if key == "profit_margin":
            if pd.notna(row.get("Revenue")) and pd.notna(row.get("EBITDA")):
                out.append((key, col, label))
        elif col in row.index and pd.notna(pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]):
            out.append((key, col, label))
    return out


def apply_whatif(row_frame: pd.DataFrame, changes: dict[str, float]) -> pd.DataFrame:
    """Apply percent changes (fractions, e.g. -0.2) + margin pp shift."""
    if len(row_frame) != 1:
        raise ValueError("Scenario analysis requires exactly one company-period.")
    scen = row_frame.copy()
    rev_c = float(changes.get("revenue", 0.0))
    marg_c = float(changes.get("profit_margin", 0.0))
    debt_c = float(changes.get("debt", 0.0))
    # Core engine handles revenue/margin/debt when present.
    try:
        scen = apply_scenario_adjustments(scen, revenue_change=rev_c,
                                          operating_margin_change=marg_c,
                                          debt_change=debt_c)
    except ValueError:
        # Fall back to generic scaling when core fields are missing.
        pass
    idx = scen.index[0]
    for key, col, _ in WHATIF_FIELDS:
        if key in {"revenue", "profit_margin", "debt"}:
            continue
        delta = float(changes.get(key, 0.0))
        if delta == 0.0 or col not in scen.columns:
            continue
        try:
            val = float(pd.to_numeric(pd.Series([scen.at[idx, col]]), errors="coerce").iloc[0])
        except (TypeError, ValueError):
            continue
        if pd.isna(val):
            continue
        scen.at[idx, col] = val * (1.0 + delta)
    # Keep Sales_Growth consistent with revenue move.
    if "Sales_Growth" in scen.columns and rev_c != 0.0:
        try:
            g = float(pd.to_numeric(pd.Series([scen.at[idx, "Sales_Growth"]]), errors="coerce").iloc[0])
            if pd.notna(g):
                scen.at[idx, "Sales_Growth"] = (1 + g) * (1 + rev_c) - 1
        except (TypeError, ValueError):
            pass
    return scen


def score_whatif(base_frame: pd.DataFrame, row_index: int,
                 changes: dict[str, float]) -> dict[str, Any]:
    base_row = base_frame.iloc[[row_index]]
    scen = apply_whatif(base_row, changes)
    s_feat = engineer_features(scen)
    s_analysis = analyze_financials(scen, 0, s_feat)
    s_result = rule_based_assessment(scen, 0, s_analysis)
    if s_result.get("risk_index") is not None:
        s_result["health_score"] = 100 - s_result["risk_index"]
    w = early_warning_indicators(scen).iloc[0]
    return {"frame": scen, "features": s_feat.iloc[0],
            "analysis": s_analysis, "result": s_result,
            "signals": w[w].index.tolist()}
