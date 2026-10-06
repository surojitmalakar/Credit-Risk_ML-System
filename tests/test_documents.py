from __future__ import annotations

import io

import pandas as pd
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from msme_ews.documents import extract_financial_document
from msme_ews.portfolio import analyze_credit_portfolio


def test_csv_normalizes_aliases_indian_numbers_and_periods():
    content = (
        "Company Name,Financial Year,Turnover (Rs. Crore),PAT (Rs. Lakh),Borrowings (Crore),Current Assets,Current Liabilities\n"
        "Acme Industries,FY 2023-24,1.25,50,0.4,1000000,500000\n"
    ).encode()

    extracted = extract_financial_document(content, "statement.csv")

    row = extracted.frame.iloc[0]
    assert extracted.company_name == "Acme Industries"
    assert row["period"] == pd.Timestamp("2024-03-31")
    assert row["Revenue"] == 12_500_000
    assert row["Net_Profit"] == 5_000_000
    assert row["Debt"] == 4_000_000
    assert row["Current_Assets"] == 1_000_000


def test_customer_master_csv_uses_dataset_path_and_detects_portfolio_risk():
    content = (
        "CustomerID,Income,Loan Amount,Credit Score,Default Status,City\n"
        "C001,600000,150000,720,No,Pune\n"
        "C002,350000,220000,540,Yes,Mumbai\n"
        "C003,500000,100000,630,Current,Delhi\n"
    ).encode()

    extracted = extract_financial_document(content, "customer_master.csv")
    portfolio = analyze_credit_portfolio(extracted.frame)

    assert extracted.document_type == "customer_loan_dataset"
    assert len(extracted.frame) == 3
    assert extracted.frame.loc[0, "customer_id"] == "C001"
    assert extracted.frame.loc[0, "Revenue"] == 600_000
    assert extracted.frame.loc[1, "loan_amount"] == 220_000
    assert portfolio["summary"]["observed_default_rate"] == 1 / 3
    assert portfolio["customers"]["portfolio_risk_category"].tolist() == [
        "Low Risk",
        "High Risk",
        "Moderate Risk",
    ]


def test_unfamiliar_csv_is_loaded_for_general_profiling():
    content = b"customer_ref,region,signup_date\nA1,North,2024-01-03\n"

    extracted = extract_financial_document(content, "customers.csv")

    assert extracted.document_type == "customer_loan_dataset"
    assert extracted.detected_columns == ["customer_ref", "region", "signup_date"]
    assert len(extracted.frame) == 1
    assert "No recognized" not in extracted.status


def test_financial_customer_dataset_keeps_borrower_rows_for_existing_model_inputs():
    content = (
        "Borrower ID,Annual Income,Profit,Assets,Liabilities,Debt,Credit Score,Default\n"
        "B-01,1000000,100000,900000,400000,250000,710,No\n"
        "B-02,800000,-20000,500000,420000,350000,560,Yes\n"
    ).encode()

    extracted = extract_financial_document(content, "borrowers.csv")

    assert extracted.document_type == "customer_loan_dataset"
    assert len(extracted.frame) == 2
    assert extracted.frame["company_id"].tolist() == ["B-01", "B-02"]
    assert extracted.frame["Revenue"].tolist() == [1_000_000, 800_000]
    assert extracted.frame["Net_Profit"].tolist() == [100_000, -20_000]
    assert extracted.frame["Debt"].tolist() == [250_000, 350_000]
    assert extracted.frame["credit_score"].tolist() == [710, 560]
    assert analyze_credit_portfolio(extracted.frame)["summary"]["observed_default_rate"] == 0.5


def test_csv_transposed_statement_extracts_multiple_financial_years():
    content = (
        "Particulars,FY 2022-23,FY 2023-24\n"
        "Revenue,\"10,00,000\",\"12,00,000\"\n"
        "Net Profit,\"1,00,000\",\"1,50,000\"\n"
    ).encode()

    extracted = extract_financial_document(content, "statement.csv")
    result = extracted.frame.set_index("period")

    assert list(result.index) == [pd.Timestamp("2023-03-31"), pd.Timestamp("2024-03-31")]
    assert result.loc[pd.Timestamp("2023-03-31"), "Revenue"] == 1_000_000
    assert result.loc[pd.Timestamp("2024-03-31"), "Net_Profit"] == 150_000


def test_iso_period_is_not_mistaken_for_a_fiscal_year_range():
    content = b"company_id,period,revenue\nExample Co,2024-03-31,1000000\n"

    extracted = extract_financial_document(content, "statement.csv")

    assert extracted.frame.iloc[0]["company_id"] == "Example Co"
    assert extracted.frame.iloc[0]["period"] == pd.Timestamp("2024-03-31")


def test_vertical_statement_in_crores_derives_current_assets_and_liabilities():
    content = (
        "Particulars (Rs. in Crores),FY 2022-23,FY 2023-24\n"
        "Cash and Cash Equivalents,0.5,0.6\n"
        "Trade Receivables,0.2,0.3\n"
        "Inventory,0.3,0.4\n"
        "Trade Payables,0.5,0.5\n"
        "Short Term Borrowings,0.1,0.1\n"
    ).encode()

    extracted = extract_financial_document(content, "statement.csv")
    latest = extracted.frame.iloc[-1]

    assert latest["period"] == pd.Timestamp("2024-03-31")
    assert latest["Current_Assets"] == 13_000_000
    assert latest["Current_Liabilities"] == 6_000_000
    assert latest["Debt"] == 1_000_000


def test_excel_reads_every_sheet_and_merges_same_period():
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({
            "Company": ["Acme"], "Year": ["2024"], "Sales": ["₹ 10,00,000"],
            "Current Assets": [500_000], "Current Liabilities": [250_000],
        }).to_excel(writer, sheet_name="Income", index=False)
        pd.DataFrame({
            "Company": ["Acme"], "Year": ["2024"], "Borrowings": [100_000],
            "Total Assets": [800_000],
        }).to_excel(writer, sheet_name="Balance", index=False)

    extracted = extract_financial_document(buffer.getvalue(), "accounts.xlsx")

    assert len(extracted.frame) == 1
    row = extracted.frame.iloc[0]
    assert row["Revenue"] == 1_000_000
    assert row["Debt"] == 100_000
    assert row["Total_Assets"] == 800_000
    assert extracted.tables_found == 2


def test_excel_transposed_values_do_not_get_mistaken_for_dates():
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame([
            ["Particulars", 2023, 2024],
            ["Revenue", 1_000_000, 1_200_000],
            ["PAT", 100_000, 150_000],
        ]).to_excel(writer, sheet_name="Statement", index=False, header=False)

    extracted = extract_financial_document(buffer.getvalue(), "statement.xlsx")

    assert list(extracted.frame["period"]) == [
        pd.Timestamp("2023-12-31"),
        pd.Timestamp("2024-12-31"),
    ]
    assert extracted.frame["Revenue"].tolist() == [1_000_000, 1_200_000]


def test_corrupt_ole2_excel_raises_actionable_value_error() -> None:
    """A protected/corrupt .xls must fail as a friendly ValueError.

    xlrd.XLRDError inherits straight from Exception and used to escape the
    app's upload handler as a raw traceback that blocked every tab.
    """
    fake_workbook = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 1024

    with pytest.raises(ValueError, match="password-protected"):
        extract_financial_document(fake_workbook, "statement.xls")


def test_xlsx_bytes_with_xls_filename_still_read_by_content() -> None:
    """Engine selection follows magic bytes, not the filename extension."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({
            "company_id": ["Acme"],
            "period": ["2024"],
            "Revenue": [1_000_000],
            "Total Assets": [700_000],
        }).to_excel(writer, sheet_name="Statement", index=False)

    extracted = extract_financial_document(buffer.getvalue(), "statement.xls")

    assert extracted.frame["Revenue"].iloc[0] == 1_000_000


def test_pdf_text_extracts_report_label_value_company_and_year():
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.drawString(60, 740, "Company Name: Acme Manufacturing")
    pdf.drawString(60, 710, "Financial Year: 2023-24")
    pdf.drawString(60, 680, "Revenue: Rs. 10,00,000")
    pdf.drawString(60, 650, "PAT: Rs. 1,25,000")
    pdf.save()

    extracted = extract_financial_document(buffer.getvalue(), "report.pdf")

    assert extracted.company_name == "Acme Manufacturing"
    assert extracted.frame.iloc[0]["period"] == pd.Timestamp("2024-03-31")
    assert extracted.frame.iloc[0]["Revenue"] == 1_000_000
    assert extracted.frame.iloc[0]["Net_Profit"] == 125_000
