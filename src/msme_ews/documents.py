"""Offline extraction and normalization for common financial statements."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from msme_ews.data import FINANCIAL_COLUMNS


_METRICS = {
    "revenue": "Revenue",
    "sales": "Revenue",
    "turnover": "Revenue",
    "net sales": "Revenue",
    "gross revenue": "Revenue",
    "total income": "Revenue",
    "income from operations": "Revenue",
    "gross sales": "Revenue",
    "ebitda": "EBITDA",
    "operating profit before depreciation": "EBITDA",
    "operating profit": "EBITDA",
    "net profit": "Net_Profit",
    "profit after tax": "Net_Profit",
    "pat": "Net_Profit",
    "net income": "Net_Profit",
    "profit for the year": "Net_Profit",
    "profit after taxation": "Net_Profit",
    "net profit after tax": "Net_Profit",
    "total assets": "Total_Assets",
    "assets": "Total_Assets",
    "total liabilities": "Total_Liabilities",
    "liabilities": "Total_Liabilities",
    "current assets": "Current_Assets",
    "total current assets": "Current_Assets",
    "current liabilities": "Current_Liabilities",
    "total current liabilities": "Current_Liabilities",
    "operating cash flow": "Cash_Flow_Operations",
    "cash flow from operations": "Cash_Flow_Operations",
    "cash flow from operating activities": "Cash_Flow_Operations",
    "cash generated from operations": "Cash_Flow_Operations",
    "net cash from operating activities": "Cash_Flow_Operations",
    "cash flow operations": "Cash_Flow_Operations",
    "total debt": "Debt",
    "debt": "Debt",
    "borrowings": "Debt",
    "total borrowings": "Debt",
    "long term borrowings": "Long_Term_Debt",
    "long term debt": "Long_Term_Debt",
    "total loans": "Debt",
    "loans": "Debt",
    "interest expense": "Interest_Expense",
    "finance costs": "Interest_Expense",
    "finance cost": "Interest_Expense",
    "accounts receivable": "Accounts_Receivable",
    "trade receivables": "Accounts_Receivable",
    "receivables": "Accounts_Receivable",
    "debtors": "Accounts_Receivable",
    "inventory": "Inventory",
    "inventories": "Inventory",
    "stock": "Inventory",
    "cash and cash equivalents": "Cash_Balance",
    "cash equivalents": "Cash_Balance",
    "cash and bank balances": "Cash_Balance",
    "cash and bank": "Cash_Balance",
    "cash balance": "Cash_Balance",
    "trade payables": "Accounts_Payable",
    "accounts payable": "Accounts_Payable",
    "payables": "Accounts_Payable",
    "short term borrowings": "Short_Term_Debt",
    "short term debt": "Short_Term_Debt",
    "current portion of long term debt": "Short_Term_Debt",
    "shareholders equity": "Equity_Value",
    "shareholder equity": "Equity_Value",
    "owners equity": "Equity_Value",
    "net worth": "Equity_Value",
    "cost of goods sold": "Cost_of_Goods_Sold",
    "cost of sales": "Cost_of_Goods_Sold",
    "cogs": "Cost_of_Goods_Sold",
    "sales growth": "Sales_Growth",
    "revenue growth": "Sales_Growth",
}
_METRIC_KEYS = {
    re.sub(r"[^a-z0-9]+", " ", key).strip(): value
    for key, value in _METRICS.items()
}
_EXTRACTED_COMPONENTS = {
    "Cash_Balance", "Accounts_Payable", "Short_Term_Debt", "Long_Term_Debt",
    "Equity_Value",
}
_META_KEYS = {
    "company", "company id", "company name", "entity", "entity name", "business name",
    "particulars", "particular", "financial item", "line item", "description",
    "metric", "account", "indicator", "item",
}
_VALUE_KEYS = {"value", "amount", "balance", "reported value", "inr", "rs", "rupees"}
_YEAR_RE = re.compile(r"(?<!\d)(?:FY\s*)?(20\d{2})(?:\s*[-/]\s*(\d{2,4}))?(?!\d)", re.I)


@dataclass
class DocumentExtraction:
    """Normalized company-period records and extraction diagnostics."""

    frame: pd.DataFrame
    status: str
    source_type: str
    company_name: str
    tables_found: int = 0
    warnings: list[str] = field(default_factory=list)
    document_type: str = "financial_statement"
    detected_columns: list[str] = field(default_factory=list)


_TABULAR_ALIASES = {
    "company": "company_id",
    "company id": "company_id",
    "company name": "company_id",
    "entity": "company_id",
    "entity name": "company_id",
    "business name": "company_id",
    "customer id": "customer_id",
    "customerid": "customer_id",
    "customer identifier": "customer_id",
    "client id": "customer_id",
    "borrower id": "customer_id",
    "borrowerid": "customer_id",
    "borrower identifier": "customer_id",
    "account id": "customer_id",
    "loan id": "loan_id",
    "loanid": "loan_id",
    "customer name": "customer_name",
    "borrower name": "customer_name",
    "client name": "customer_name",
    "name": "customer_name",
    "credit score": "credit_score",
    "cibil score": "credit_score",
    "bureau score": "credit_score",
    "loan amount": "loan_amount",
    "sanctioned amount": "loan_amount",
    "principal amount": "loan_amount",
    "outstanding amount": "loan_amount",
    "loan balance": "loan_amount",
    "default status": "default_status",
    "defaulted": "default_status",
    "is default": "default_status",
    "is defaulted": "default_status",
    "default flag": "default_status",
    "default label": "default_status",
    "delinquent": "default_status",
    "bad loan": "default_status",
    "loan status": "loan_status",
    "repayment status": "loan_status",
    "payment status": "loan_status",
    "risk status": "risk_status",
    "risk category": "risk_status",
    "risk level": "risk_status",
    "credit risk": "risk_status",
    "application status": "loan_status",
}
_CUSTOMER_ID_COLUMNS = {"customer_id", "loan_id"}
_CREDIT_COLUMNS = {
    "credit_score", "loan_amount", "default_status", "loan_status", "risk_status",
}


def _normalize_tabular_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalize financial and credit headers while retaining unrelated columns."""
    renamed: dict[object, str] = {}
    normalized_values = frame.copy()
    for column in frame.columns:
        key = _clean_key(column)
        financial = _metric(column)
        canonical = financial or _TABULAR_ALIASES.get(key)
        if canonical is None and key in {"period", "year", "financial year", "fiscal year", "date", "reporting period"}:
            canonical = "period"
        if canonical is None and key in {"customer id", "customer number", "borrower number", "member id"}:
            canonical = "customer_id"
        if canonical is None and key in {
            "income", "annual income", "yearly income", "monthly income",
            "salary", "annual salary", "annual revenue",
        }:
            canonical = "Revenue"
        if canonical is None and key == "profit":
            canonical = "Net_Profit"
        if canonical is None and key in {"total debt", "total borrowings", "borrowings"}:
            canonical = "Debt"
        if canonical is None and key in {"default", "defaulted"}:
            canonical = "default_status"
        renamed[column] = canonical or str(column).strip()
        if financial in FINANCIAL_COLUMNS or canonical in {"credit_score", "loan_amount"}:
            normalized_values[column] = normalized_values[column].map(
                lambda value: _number(value, _scale_hint(column))
            )
            if financial == "Sales_Growth" and "%" in str(column):
                normalized_values[column] = normalized_values[column] / 100
    normalized = normalized_values.rename(columns=renamed).copy()
    # Duplicate aliases can occur in exports (for example both Sales and Revenue).
    # Preserve the first non-empty value in the canonical field instead of rejecting
    # the entire borrower table.
    if normalized.columns.duplicated().any():
        merged = pd.DataFrame(index=normalized.index)
        for column in dict.fromkeys(normalized.columns):
            matching = normalized.loc[:, normalized.columns == column]
            merged[column] = matching.bfill(axis=1).iloc[:, 0]
        normalized = merged
    detected = [str(column) for column in normalized.columns]
    return normalized, detected


def _tabular_document_type(frame: pd.DataFrame) -> str:
    columns = set(map(str, frame.columns))
    financial_columns = {
        column for column in columns
        if column in FINANCIAL_COLUMNS
    }
    credit_columns = columns & _CREDIT_COLUMNS
    has_customer_id = bool(columns & _CUSTOMER_ID_COLUMNS) or "customer_name" in columns
    if has_customer_id and (credit_columns or financial_columns):
        return "customer_loan_dataset"
    if credit_columns:
        return "financial_credit_dataset" if financial_columns else "customer_loan_dataset"
    if financial_columns:
        return "financial_credit_dataset"
    return "unsupported"


def _looks_like_statement_layout(frame: pd.DataFrame) -> bool:
    columns = [_clean_key(column) for column in frame.columns]
    has_metric_rows_label = any(
        column in {"particulars", "particular", "line item", "financial item", "description", "account", "metric"}
        for column in columns
    )
    has_year_columns = any(_period(column) is not None for column in frame.columns)
    values_contain_financial_labels = any(
        _metric(value) is not None
        for value in frame.iloc[:, 0].dropna().head(40)
    ) if not frame.empty else False
    return has_year_columns and (has_metric_rows_label or values_contain_financial_labels)


def _read_tabular_file(content: bytes, suffix: str) -> tuple[pd.DataFrame, list[str]]:
    if suffix == ".csv":
        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                frame = pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding=encoding)
                return frame, ["CSV"]
            except (UnicodeDecodeError, pd.errors.ParserError) as error:
                last_error = error
        raise ValueError(f"CSV could not be parsed: {last_error}")
    try:
        sheets = pd.read_excel(io.BytesIO(content), sheet_name=None)
    except ImportError as error:
        raise ValueError(f"Excel support dependency is unavailable: {error}") from error
    tables = [table for table in sheets.values() if not table.empty]
    if not tables:
        return pd.DataFrame(), list(sheets)
    frame = pd.concat(tables, ignore_index=True, sort=False)
    return frame, list(sheets)


def _extract_tabular_document(content: bytes, filename: str, suffix: str) -> DocumentExtraction:
    frame, sources = _read_tabular_file(content, suffix)
    if frame.empty:
        return DocumentExtraction(
            frame=frame,
            status="The workbook or CSV has no data rows.",
            source_type=suffix.lstrip(".").upper(),
            company_name=Path(filename).stem,
            tables_found=len(sources),
            warnings=[f"Read sheet or source: {name}" for name in sources],
            document_type="unsupported",
        )
    normalized, detected = _normalize_tabular_columns(frame)
    document_type = _tabular_document_type(normalized)
    company_column = next(
        (column for column in ("customer_name", "company_id", "customer_id", "loan_id") if column in normalized),
        None,
    )
    company = (
        str(normalized[company_column].dropna().iloc[0])
        if company_column and normalized[company_column].notna().any()
        else Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    )
    if company_column in {"customer_name", "customer_id", "loan_id"}:
        normalized["company_id"] = normalized[company_column].astype("string")
    if "period" not in normalized:
        period_column = next(
            (column for column in ("year", "financial_year", "fiscal_year", "date", "reporting_period") if column in normalized),
            None,
        )
        if period_column:
            normalized["period"] = normalized[period_column].map(_period)
    else:
        normalized["period"] = normalized["period"].map(
            lambda value: _period(value) if pd.notna(value) else None
        )
    if document_type == "financial_credit_dataset" and {"company_id", "period"} <= set(normalized.columns):
        normalized = normalized.groupby(
            ["company_id", "period"],
            as_index=False,
            dropna=False,
            sort=False,
        ).agg(lambda values: values.bfill().iloc[0] if values.notna().any() else np.nan)
    status = {
        "financial_credit_dataset": f"Detected a financial/credit dataset with {len(normalized)} record(s).",
        "customer_loan_dataset": f"Detected a customer/loan dataset with {len(normalized)} borrower record(s).",
        "unsupported": "No recognized financial or credit-risk columns were found.",
    }.get(document_type, f"Detected a financial statement dataset with {len(normalized)} record(s).")
    warnings = [f"Read sheet: {name}" for name in sources if suffix != ".csv"]
    if document_type == "unsupported":
        warnings.append(
            "Detected columns: " + (", ".join(detected) if detected else "none") +
            ". Expected financial values (revenue, profit, assets, liabilities, debt) or credit fields "
            "(customer/loan ID, credit score, loan amount, default/loan/risk status)."
        )
    if document_type == "customer_loan_dataset" and not set(FINANCIAL_COLUMNS) & set(normalized.columns):
        warnings.append(
            "This dataset contains credit/customer fields but no financial statement columns; "
            "the existing financial-statement ML model will not be applied."
        )
    return DocumentExtraction(
        frame=normalized,
        status=status,
        source_type=suffix.lstrip(".").upper(),
        company_name=company,
        tables_found=len(sources),
        warnings=warnings,
        document_type=document_type,
        detected_columns=detected,
    )


def _clean_key(value: object) -> str:
    text = str(value).lower().replace("&", " and ")
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _metric(value: object) -> str | None:
    return _METRIC_KEYS.get(_clean_key(value))


def _scale_hint(value: object) -> float:
    text = str(value).lower()
    if re.search(r"\b(crores?|crs?)\b", text):
        return 10_000_000.0
    if re.search(r"\b(lakhs?|lacs?)\b", text):
        return 100_000.0
    if re.search(r"\bmillions?\b", text):
        return 1_000_000.0
    if re.search(r"\bthousands?\b", text):
        return 1_000.0
    return 1.0


def _document_scale_hint(text: str) -> float:
    for line in text.splitlines():
        normalized = line.lower()
        if re.search(r"\b(amounts?|figures?|values?|financial statements?)\b", normalized):
            scale = _scale_hint(normalized)
            if scale != 1.0:
                return scale
    return 1.0


def _number(value: object, scale: float = 1.0) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float, np.number)):
        return float(value) * scale
    text = str(value).strip()
    if not text or text.lower() in {"-", "–", "—", "na", "n/a", "nil", "none"}:
        return float("nan")
    value_scale = _scale_hint(text)
    local_scale = value_scale if value_scale != 1.0 else scale
    text = re.sub(r"\b(crores?|crs?|lakhs?|lacs?|millions?|thousands?)\b", "", text, flags=re.I)
    is_negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace(",", "").replace(" ", "")
    text = re.sub(r"(?i)(₹|rs\.?|inr|rupees?)", "", text)
    text = re.sub(r"[^0-9.+\-]", "", text)
    if text in {"", "-", ".", "+", "-.", "+."}:
        return float("nan")
    try:
        amount = float(text)
    except ValueError:
        return float("nan")
    return (-amount if is_negative else amount) * local_scale


def _as_growth_fraction(value: object) -> float:
    parsed = _number(value)
    return parsed / 100 if pd.notna(parsed) and "%" in str(value) else parsed


def _period(value: object) -> pd.Timestamp | None:
    if isinstance(value, (int, float, np.number)) and np.isfinite(value):
        year = int(value)
        if float(value) == year and 1900 <= year <= 2100:
            return pd.Timestamp(year=year, month=12, day=31)
        return None
    text = str(value).strip()
    if re.match(r"^20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b", text):
        parsed_date = pd.to_datetime(value, errors="coerce")
        if not pd.isna(parsed_date):
            return pd.Timestamp(parsed_date)
    match = _YEAR_RE.search(text)
    if match is None:
        parsed = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(parsed) else pd.Timestamp(parsed)
    start_year = int(match.group(1))
    end_token = match.group(2)
    if end_token is None:
        return pd.Timestamp(year=start_year, month=12, day=31)
    end_year = int(end_token)
    if end_year < 100:
        end_year += (start_year // 100) * 100
        if end_year < start_year:
            end_year += 100
    if end_year == start_year + 1:
        return pd.Timestamp(year=end_year, month=3, day=31)
    return pd.Timestamp(year=end_year, month=12, day=31)


def _company_from_text(text: str, filename: str) -> str:
    match = re.search(
        r"(?im)^\s*(?:company\s*name|entity\s*name|business\s*name|company|entity)\s*[:\-]\s*(.{2,100})$",
        text,
    )
    if match:
        return match.group(1).strip(" \t|")
    return Path(filename).stem.replace("_", " ").replace("-", " ").strip() or "Uploaded Company"


def _find_header(raw: pd.DataFrame) -> int | None:
    best: tuple[int, int] | None = None
    for index in range(min(15, len(raw))):
        values = [str(value).strip() for value in raw.iloc[index].tolist()]
        metric_count = sum(_metric(value) is not None for value in values)
        has_period = any(
            _period(value) is not None
            or _clean_key(value) in {"period", "year", "financial year", "fiscal year", "date", "reporting period"}
            for value in values
        )
        has_item_label = any(_clean_key(value) in _META_KEYS for value in values)
        has_value_label = any(_clean_key(value) in _VALUE_KEYS for value in values)
        score = metric_count + int(has_period) + int(has_item_label)
        if (
            metric_count >= 2
            or (metric_count >= 1 and (has_period or has_item_label))
            or (has_item_label and (has_period or has_value_label))
        ):
            if best is None or score > best[0]:
                best = (score, index)
    return best[1] if best else None


def _records_from_table(
    raw: pd.DataFrame,
    *,
    company: str,
    default_period: pd.Timestamp | None,
    unit_scale: float,
) -> list[dict[str, object]]:
    if raw.empty:
        return []
    table = raw.copy().dropna(how="all").dropna(axis=1, how="all")
    if table.empty:
        return []
    table = table.map(lambda value: value.strip() if isinstance(value, str) else value)
    records: list[dict[str, object]] = []

    header_row = _find_header(table)
    table_scale = unit_scale
    if header_row is not None:
        header_values = [str(value) for value in table.iloc[header_row].tolist()]
        unit_labels = [
            value for value in header_values
            if _clean_key(value) in _META_KEYS and _scale_hint(value) != 1.0
        ]
        preamble = "\n".join([
            *(str(value) for value in table.iloc[:header_row].to_numpy().ravel()),
            *unit_labels,
        ])
        hinted_scale = _scale_hint("\n".join(unit_labels)) if unit_labels else _document_scale_hint(preamble)
        if hinted_scale == 1.0 and re.search(r"\b(?:in|amounts?|figures?|values?)\b", preamble, re.I):
            hinted_scale = _scale_hint(preamble)
        if hinted_scale != 1.0:
            table_scale = hinted_scale
    if header_row is not None:
        headers = [
            str(value).strip() if value is not None else f"column_{index}"
            for index, value in enumerate(table.iloc[header_row].tolist())
        ]
        body = table.iloc[header_row + 1:].copy()
        body.columns = headers
        canonical: dict[str, str] = {}
        for header in headers:
            mapped = _metric(header)
            if mapped:
                canonical[header] = mapped
            elif _clean_key(header) in {
                "company", "company id", "company name", "entity", "entity name",
                "business name", "borrower name", "customer",
            }:
                canonical[header] = "company_id"
            elif _clean_key(header) in {"period", "year", "financial year", "fiscal year", "fy", "date", "reporting period"}:
                canonical[header] = "period"
            else:
                canonical[header] = header

        metric_columns = [
            header for header, mapped in canonical.items()
            if mapped in FINANCIAL_COLUMNS or mapped in _EXTRACTED_COMPONENTS
        ]
        period_columns = [header for header, mapped in canonical.items() if mapped == "period"]
        company_columns = [header for header, mapped in canonical.items() if mapped == "company_id"]
        if metric_columns:
            for _, source_row in body.iterrows():
                record: dict[str, object] = {"company_id": company, "period": default_period}
                if period_columns:
                    record["period"] = _period(source_row[period_columns[0]]) or default_period
                if company_columns and pd.notna(source_row[company_columns[0]]):
                    record["company_id"] = str(source_row[company_columns[0]]).strip()
                for header in metric_columns:
                    metric_name = canonical[header]
                    scale = _scale_hint(header)
                    record[metric_name] = _number(source_row[header], table_scale if scale == 1 else scale)
                    if metric_name == "Sales_Growth" and pd.notna(record[metric_name]):
                        if "%" in header or "%" in str(source_row[header]):
                            record[metric_name] = float(record[metric_name]) / 100
                if any(pd.notna(record.get(canonical[column])) for column in metric_columns):
                    records.append(record)

    # Financial statements commonly put metric names down rows and years across columns.
    if not records:
        working = table
        column_names: list[str] | None = None
        if header_row is not None:
            column_names = [
                str(value).strip() if value is not None else f"column_{index}"
                for index, value in enumerate(table.iloc[header_row].tolist())
            ]
            working = table.iloc[header_row + 1:].reset_index(drop=True)
        metric_col: int | None = None
        metric_count = 0
        for column_index in range(working.shape[1]):
            count = sum(_metric(value) is not None for value in working.iloc[:, column_index])
            if count > metric_count:
                metric_count, metric_col = count, column_index
        if metric_col is not None and metric_count:
            period_columns: list[tuple[int, pd.Timestamp]] = []
            period_header_column = None
            value_column = None
            if column_names:
                for column_index, name in enumerate(column_names):
                    key = _clean_key(name)
                    if key in {"period", "year", "financial year", "fiscal year", "date", "reporting period"}:
                        period_header_column = column_index
                    elif key in _VALUE_KEYS:
                        value_column = column_index
            if value_column is not None:
                for _, source_row in working.iterrows():
                    metric_name = _metric(source_row.iloc[metric_col])
                    period_value = (
                        _period(source_row.iloc[period_header_column])
                        if period_header_column is not None
                        else default_period
                    )
                    if metric_name is not None:
                        value_scale = (
                            _scale_hint(column_names[value_column])
                            if column_names else table_scale
                        )
                        if value_scale == 1.0:
                            value_scale = table_scale
                        records.append({
                            "company_id": company,
                            "period": period_value or default_period,
                            metric_name: _number(source_row.iloc[value_column], value_scale),
                        })
            if column_names and not records:
                period_columns.extend(
                    (column_index, period)
                    for column_index, name in enumerate(column_names)
                    if column_index != metric_col and (period := _period(name)) is not None
                )
            for column_index in range(working.shape[1]):
                if column_index == metric_col:
                    continue
                period_value = next(
                    (_period(value) for value in working.iloc[:, column_index].head(3) if _period(value) is not None),
                    None,
                )
                if period_value is not None and all(existing[0] != column_index for existing in period_columns):
                    period_columns.append((column_index, period_value))
            if not records and not period_columns and working.shape[1] >= 2 and value_column is None:
                period_columns = [(column_index, default_period) for column_index in range(working.shape[1]) if column_index != metric_col]
            if not records and not period_columns and working.shape[1] == 2:
                period_columns = [(1 - metric_col, default_period)]
            if not records:
                for _, source_row in working.iterrows():
                    metric_name = _metric(source_row.iloc[metric_col])
                    if metric_name is None:
                        continue
                    for column_index, period_value in period_columns:
                        if column_index == period_header_column:
                            continue
                        value_scale = (
                            _scale_hint(column_names[column_index])
                            if column_names else table_scale
                        )
                        if value_scale == 1.0:
                            value_scale = table_scale
                        record = {
                            "company_id": company,
                            "period": period_value or default_period,
                            metric_name: _number(source_row.iloc[column_index], value_scale),
                        }
                        if metric_name == "Sales_Growth" and pd.notna(record[metric_name]):
                            if "%" in str(source_row.iloc[column_index]):
                                record[metric_name] = float(record[metric_name]) / 100
                        records.append(record)
    return records


def _records_from_text(text: str, company: str) -> list[dict[str, object]]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    document_periods = list(dict.fromkeys(period for line in lines for period in [_period(line)] if period is not None))
    default_period = document_periods[0] if len(document_periods) == 1 else None
    unit_scale = _document_scale_hint(text)
    records: list[dict[str, object]] = []
    for line in lines:
        parts = re.split(r"\s{2,}|\s+[|]\s+|:\s*", line)
        metric_name = _metric(parts[0]) if parts else None
        if metric_name is None:
            # PDF extraction often collapses whitespace; accept a known label at line start.
            for label in sorted(_METRICS, key=len, reverse=True):
                if _clean_key(line).startswith(_clean_key(label) + " "):
                    metric_name = _METRICS[label]
                    parts = [label, line[len(label):].strip()]
                    break
        if metric_name is None:
            continue
        value_text = " ".join(parts[1:])
        years = [_period(match.group(0)) for match in _YEAR_RE.finditer(value_text)]
        numbers = re.findall(r"\(?-?₹?\s*(?:Rs\.?\s*)?\d[\d,]*(?:\.\d+)?\s*(?:crores?|crs?|lakhs?|lacs?|millions?|thousands?)?\)?", value_text, re.I)
        numbers = [token for token in numbers if _period(token) is None]
        if not numbers:
            continue
        for index, token in enumerate(numbers):
            period = years[index] if index < len(years) else default_period
            value = (
                _as_growth_fraction(token)
                if metric_name == "Sales_Growth"
                else _number(token, unit_scale)
            )
            records.append({"company_id": company, "period": period, metric_name: value})
    return records


def _merge_records(records: list[dict[str, object]], company: str) -> pd.DataFrame:
    if not records:
        raise ValueError("No financial statement rows or tables could be extracted from this document.")
    frame = pd.DataFrame(records)
    if "company_id" not in frame:
        frame["company_id"] = company
    frame["company_id"] = frame["company_id"].fillna(company).astype(str)
    extracted_columns = list(_EXTRACTED_COMPONENTS)
    for column in (*FINANCIAL_COLUMNS, *extracted_columns):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["period"] = pd.to_datetime(frame["period"], errors="coerce")
    group_columns = ["company_id", "period"]
    aggregations = {column: "last" for column in (*FINANCIAL_COLUMNS, *extracted_columns)}
    frame = frame.sort_values(group_columns, na_position="last")
    frame = frame.groupby(group_columns, as_index=False, dropna=False).agg(aggregations)
    debt_missing = frame["Debt"].isna()
    frame.loc[debt_missing, "Debt"] = frame.loc[
        debt_missing,
        ["Short_Term_Debt", "Long_Term_Debt"],
    ].sum(axis=1, min_count=1)
    if "Current_Assets" not in frame:
        frame["Current_Assets"] = np.nan
    current_assets_missing = frame["Current_Assets"].isna()
    frame.loc[current_assets_missing, "Current_Assets"] = frame.loc[
        current_assets_missing,
        ["Cash_Balance", "Accounts_Receivable", "Inventory"],
    ].sum(axis=1, min_count=1)
    current_liabilities_missing = frame["Current_Liabilities"].isna()
    frame.loc[current_liabilities_missing, "Current_Liabilities"] = frame.loc[
        current_liabilities_missing,
        ["Accounts_Payable", "Short_Term_Debt"],
    ].sum(axis=1, min_count=1)
    # Keep the source metadata columns used by the existing dashboard stable.
    return frame[["company_id", "period", *FINANCIAL_COLUMNS]]


def extract_financial_document(
    file: BinaryIO | bytes,
    filename: str,
) -> DocumentExtraction:
    """Extract supported financial statement documents without network services."""
    content = file if isinstance(file, bytes) else file.read()
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls", ".pdf"}:
        raise ValueError("Supported upload formats are CSV, XLSX, XLS, and PDF.")
    if not content:
        raise ValueError("The uploaded document is empty.")

    if suffix in {".csv", ".xlsx", ".xls"}:
        tabular_result = _extract_tabular_document(content, filename, suffix)
        if (
            tabular_result.document_type != "unsupported"
            or not _looks_like_statement_layout(tabular_result.frame)
        ):
            return tabular_result

    tables: list[pd.DataFrame] = []
    text_parts: list[str] = []
    warnings: list[str] = []
    if suffix == ".csv":
        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                tables = [pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding=encoding, header=None, dtype=object)]
                break
            except (UnicodeDecodeError, pd.errors.ParserError) as error:
                last_error = error
        if not tables:
            raise ValueError(f"CSV could not be parsed: {last_error}")
        text_parts.append(content.decode("utf-8", errors="replace"))
    elif suffix in {".xlsx", ".xls"}:
        try:
            sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None, dtype=object)
        except ImportError as error:
            raise ValueError(f"Excel support dependency is unavailable: {error}") from error
        tables = list(sheets.values())
        warnings.extend(f"Read worksheet: {name}" for name in sheets)
    else:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
                extracted_tables = page.extract_tables() or []
                tables.extend(pd.DataFrame(table) for table in extracted_tables if table)
                if not page_text.strip() and not extracted_tables:
                    warnings.append(f"Page {page_number} contains no extractable text or tables; scanned-image OCR is not included.")

    full_text = "\n".join(text_parts)
    if suffix == ".pdf" and not full_text.strip() and not tables:
        raise ValueError(
            "This PDF has no extractable text or tables. It may be scanned; offline OCR is not included."
        )
    company = _company_from_text(full_text, filename)
    default_periods = list(dict.fromkeys(period for line in full_text.splitlines() for period in [_period(line)] if period is not None))
    default_period = default_periods[0] if len(default_periods) == 1 else None
    unit_scale = _document_scale_hint(full_text)

    records: list[dict[str, object]] = []
    for table in tables:
        records.extend(_records_from_table(
            table,
            company=company,
            default_period=default_period,
            unit_scale=unit_scale,
        ))
    if suffix == ".pdf" and full_text and not records:
        records.extend(_records_from_text(full_text, company))
    frame = _merge_records(records, company)
    company_values = frame["company_id"].dropna().astype(str).unique()
    if len(company_values) == 1:
        company = company_values[0]
    if not frame.loc[:, FINANCIAL_COLUMNS].notna().any().any():
        raise ValueError("The document was read, but no usable financial values were recognized.")
    if len(frame) == 1 and pd.isna(frame.iloc[0]["period"]):
        warnings.append("Financial year was not identified; trend analysis is unavailable.")
    status = f"Extracted {len(frame)} company-period record(s) from {len(tables)} table(s)."
    return DocumentExtraction(
        frame=frame,
        status=status,
        source_type=suffix.lstrip(".").upper(),
        company_name=company,
        tables_found=len(tables),
        warnings=warnings,
    )
