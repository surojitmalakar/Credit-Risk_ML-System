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
    "annual revenue": "Revenue", "sales": "Revenue", "net sales": "Revenue",
    "total revenue": "Revenue", "operating revenue": "Revenue",
    "turnover": "Revenue", "gross revenue": "Revenue", "gross sales": "Revenue",
    "income from operations": "Revenue", "total income": "Revenue",
    "total sales": "Revenue", "annual sales": "Revenue",
    "net income": "Net_Profit", "net profit": "Net_Profit",
    "profit after tax": "Net_Profit", "pat": "Net_Profit",
    "profit after taxation": "Net_Profit", "profit for the year": "Net_Profit",
    "net profit after tax": "Net_Profit", "profit": "Net_Profit",
    "earnings": "Net_Profit",
    "ebitda": "EBITDA", "operating ebitda": "EBITDA",
    "operating profit": "EBITDA",
    "operating profit before depreciation": "EBITDA",
    "total assets": "Total_Assets", "assets": "Total_Assets",
    "asset": "Total_Assets",
    "total liabilities": "Total_Liabilities", "liabilities": "Total_Liabilities",
    "current assets": "Current_Assets", "total current assets": "Current_Assets",
    "current liabilities": "Current_Liabilities",
    "total current liabilities": "Current_Liabilities",
    "cash flow from operations": "Cash_Flow_Operations", "operating cash flow": "Cash_Flow_Operations",
    "cfo": "Cash_Flow_Operations",
    "cash flow from operating activities": "Cash_Flow_Operations",
    "cash generated from operations": "Cash_Flow_Operations",
    "net cash from operating activities": "Cash_Flow_Operations",
    "cashflow": "Cash_Flow_Operations", "operating cash": "Cash_Flow_Operations",
    "interest expense": "Interest_Expense", "finance costs": "Interest_Expense",
    "finance cost": "Interest_Expense", "interest": "Interest_Expense",
    "accounts receivable": "Accounts_Receivable", "receivables": "Accounts_Receivable",
    "trade receivables": "Accounts_Receivable", "debtors": "Accounts_Receivable",
    "receivable": "Accounts_Receivable",
    "debt": "Debt", "total debt": "Debt", "borrowings": "Debt",
    "total borrowings": "Debt", "loans": "Debt", "total loans": "Debt",
    "loan balance": "Debt", "outstanding": "Debt", "exposure": "Debt",
    "cost of goods sold": "Cost_of_Goods_Sold", "cogs": "Cost_of_Goods_Sold",
    "cost of sales": "Cost_of_Goods_Sold",
    "sales growth": "Sales_Growth", "revenue growth": "Sales_Growth",
    "growth": "Sales_Growth",
    "distress": "distress_label", "default": "distress_label",
    "default label": "distress_label", "distress label": "distress_label",
    "date": "period", "year": "period", "financial year": "period",
    "period": "period", "month": "period",
    "inventory": "Inventory", "inventories": "Inventory", "stock": "Inventory",
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