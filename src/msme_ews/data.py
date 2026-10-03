"""Input normalization and validation for financial statement CSV data."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

FINANCIAL_COLUMNS = (
    "Revenue", "EBITDA", "Net_Profit", "Total_Assets", "Total_Liabilities",
    "Current_Assets", "Current_Liabilities", "Cash_Flow_Operations", "Debt",
    "Interest_Expense", "Accounts_Receivable", "Inventory", "Cost_of_Goods_Sold",
    "Sales_Growth",
)

ALIASES = {
    "annual revenue": "Revenue", "sales": "Revenue", "net income": "Net_Profit",
    "total assets": "Total_Assets", "total liabilities": "Total_Liabilities",
    "current assets": "Current_Assets", "current liabilities": "Current_Liabilities",
    "cash flow from operations": "Cash_Flow_Operations", "operating cash flow": "Cash_Flow_Operations",
    "cfo": "Cash_Flow_Operations", "interest expense": "Interest_Expense",
    "accounts receivable": "Accounts_Receivable", "receivables": "Accounts_Receivable",
    "cost of goods sold": "Cost_of_Goods_Sold", "cogs": "Cost_of_Goods_Sold",
    "sales growth": "Sales_Growth", "distress": "distress_label", "default": "distress_label",
    "default label": "distress_label", "distress label": "distress_label",
    "date": "period", "year": "period",
}


def _key(value: object) -> str:
    return re.sub(r"[\s_\-]+", " ", str(value).strip().lower())


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common financial header spellings without discarding columns."""
    renamed: dict[object, str] = {}
    used: set[str] = set()
    for column in frame.columns:
        key = _key(column)
        canonical = ALIASES.get(key)
        if canonical is None:
            canonical = next((name for name in FINANCIAL_COLUMNS if _key(name) == key), str(column).strip())
        if canonical in used:
            raise ValueError(f"Multiple input columns normalize to {canonical!r}.")
        renamed[column] = canonical
        used.add(canonical)
    return frame.rename(columns=renamed).copy()


def validate_financial_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers and reject empty or non-financial input tables."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Financial data must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("The uploaded dataset has no rows.")
    normalized = normalize_columns(frame)
    present = [column for column in FINANCIAL_COLUMNS if column in normalized.columns]
    if not present:
        raise ValueError("No recognized financial columns found. Include a supported field such as Revenue or Total_Assets.")
    numeric = normalized[present].apply(pd.to_numeric, errors="coerce")
    if not numeric.notna().any().any():
        raise ValueError("Recognized financial columns contain no usable numeric values.")
    normalized.loc[:, present] = numeric.replace([np.inf, -np.inf], np.nan)
    return normalized