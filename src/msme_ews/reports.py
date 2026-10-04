"""Offline PDF and Excel export for a selected financial assessment."""

from __future__ import annotations

from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _display(value: object) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    if isinstance(value, (int, float)):
        return f"{value:,.4g}"
    return str(value)


def _assessment_table(assessment: dict[str, Any]) -> list[list[str]]:
    probability = assessment.get("distress_probability")
    return [
        ["Assessment method", str(assessment.get("method", "Existing ML model"))],
        ["Financial health score", _display(assessment.get("health_score"))],
        ["Risk index / model estimate", f"{probability:.1%}" if probability is not None else "Not available"],
        ["Risk category", str(assessment.get("risk_category", "Not available"))],
        ["Data coverage", str(assessment.get("coverage_label", "Not available"))],
    ]


def create_credit_assessment_pdf(
    company_name: str,
    statement: pd.DataFrame,
    analysis: dict[str, Any],
    assessment: dict[str, Any],
    warnings: list[str],
    recommendations: list[str],
) -> bytes:
    """Create a compact local PDF report using ReportLab."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title=f"Credit Assessment - {company_name}",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="RiskBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#243449"),
    ))
    story = [
        Paragraph("CREDIT RISK AI", styles["Title"]),
        Paragraph("MSME Financial Assessment", styles["Heading2"]),
        Paragraph(f"Company: {escape(company_name)}", styles["RiskBody"]),
        Spacer(1, 10),
        Paragraph("Assessment summary", styles["Heading3"]),
    ]
    summary = Table(_assessment_table(assessment), colWidths=[2.3 * inch, 4.6 * inch])
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF1F8")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#152238")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([summary, Spacer(1, 10), Paragraph("Financial ratios", styles["Heading3"])])
    ratio_rows = [["Ratio", "Observed value"]] + [
        [str(label), _display(value)]
        for label, value in analysis["ratios"].items()
        if value is not None
    ]
    ratios = Table(ratio_rows, colWidths=[3.4 * inch, 3.5 * inch], repeatRows=1)
    ratios.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0D1B2A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(ratios)
    for title, entries in (
        ("Early warning indicators", warnings or ["No configured warning indicators were triggered."]),
        ("Recommendations", recommendations),
    ):
        story.extend([Spacer(1, 10), Paragraph(title, styles["Heading3"])])
        story.extend(Paragraph(f"- {escape(str(entry))}", styles["RiskBody"]) for entry in entries)

    available = statement.iloc[0].get("period") if not statement.empty else None
    story.extend([
        Spacer(1, 12),
        Paragraph(f"Selected financial period: {escape(_display(available))}", styles["RiskBody"]),
        Spacer(1, 4),
        Paragraph(
            "This report is an analytical research summary, not a lending decision or a guaranteed prediction. "
            "Rule-based results are heuristic indices and are not calibrated probabilities of default.",
            styles["RiskBody"],
        ),
    ])
    document.build(story)
    return output.getvalue()


def create_excel_analysis(
    statement: pd.DataFrame,
    analysis: dict[str, Any],
    assessment: dict[str, Any],
    warnings: list[str],
    recommendations: list[str],
) -> bytes:
    """Create a multi-sheet XLSX analysis workbook using openpyxl."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        statement.to_excel(writer, sheet_name="Financial Data", index=False)
        pd.DataFrame(
            [{"Ratio": label, "Value": value} for label, value in analysis["ratios"].items()]
        ).to_excel(writer, sheet_name="Financial Ratios", index=False)
        pd.DataFrame(
            [{"Metric": key.replace("_", " ").title(), "Value": value} for key, value in assessment.items()
             if key in {"method", "distress_probability", "risk_category", "health_score", "confidence_note", "coverage_label"}]
        ).to_excel(writer, sheet_name="Risk Assessment", index=False)
        pd.DataFrame({"Early Warning Indicator": warnings}).to_excel(
            writer, sheet_name="Early Warnings", index=False,
        )
        pd.DataFrame({"Recommendation": recommendations}).to_excel(
            writer, sheet_name="Recommendations", index=False,
        )
        workbook = writer.book
        header_fill = PatternFill("solid", fgColor="0D1B2A")
        header_font = Font(bold=True, color="FFFFFF")
        for worksheet in workbook.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font
            for column_cells in worksheet.columns:
                width = min(48, max(
                    12,
                    max((len(str(cell.value)) for cell in column_cells if cell.value is not None), default=0) + 2,
                ))
                worksheet.column_dimensions[get_column_letter(column_cells[0].column)].width = width
                for cell in column_cells[1:]:
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
    return output.getvalue()
