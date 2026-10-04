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
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing


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


def create_data_intelligence_pdf(analysis: dict[str, Any], filename: str) -> bytes:
    """Create an offline exploratory report based only on analyzed upload contents."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output, pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch,
        topMargin=0.55 * inch, bottomMargin=0.55 * inch,
        title=f"Data Intelligence Report - {filename}",
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("DATA INTELLIGENCE REPORT", styles["Title"]),
        Paragraph(escape(filename), styles["Heading2"]),
        Paragraph(escape(analysis["executive_summary"]), styles["BodyText"]),
        Spacer(1, 10),
        Paragraph("Dataset profile", styles["Heading3"]),
        Paragraph(
            f"Type: {escape(analysis['dataset_type'])}; rows: {len(analysis['data']):,}; "
            f"variables: {analysis['data'].shape[1] - 2:,}; "
            f"data quality: {analysis['data_quality_percent']:.1f}%; "
            f"duplicates: {analysis['duplicate_count']:,}; "
            f"anomalies: {analysis['anomaly_count']:,}. "
            "The Excel processed-data sheet includes up to the first 50,000 rows.",
            styles["BodyText"],
        ),
    ]
    detected = analysis["profile"].loc[
        analysis["profile"]["Detected meaning"].astype(str).ne(""),
        ["Variable", "Detected meaning"],
    ]
    if not detected.empty:
        story.extend([
            Paragraph("Detected variables", styles["Heading3"]),
            Paragraph(escape("; ".join(
                f"{row['Variable']}: {row['Detected meaning']}"
                for _, row in detected.head(20).iterrows()
            )), styles["BodyText"]),
        ])
    model_details = analysis["model"]
    if model_details.get("status") == "evaluated":
        score_lines = [
            f"{key.replace('_', ' ').title()}: {value:.3f}"
            for key, value in model_details.items()
            if key in {"accuracy", "balanced_accuracy", "mae", "r2"} and isinstance(value, (int, float))
        ]
        story.extend([
            Paragraph("Exploratory model results", styles["Heading3"]),
            Paragraph(
                escape(
                    f"{model_details.get('model')} predicting {model_details.get('target')}; "
                    f"{', '.join(score_lines)}. " + str(model_details.get("limitation", ""))
                ),
                styles["BodyText"],
            ),
        ])
        if model_details.get("top_features"):
            story.append(Paragraph(
                "Top model features: " + escape(", ".join(
                    f"{item['feature']} ({item['importance']:.3f})"
                    for item in model_details["top_features"][:8]
                )),
                styles["BodyText"],
            ))
    numeric_profile = analysis["profile"].loc[
        analysis["profile"]["Type"].eq("Numeric"), ["Variable", "Missing %"]
    ].head(8)
    if not numeric_profile.empty:
        chart = VerticalBarChart()
        chart.x = 30
        chart.y = 25
        chart.height = 130
        chart.width = 480
        chart.data = [numeric_profile["Missing %"].astype(float).tolist()]
        chart.categoryAxis.categoryNames = numeric_profile["Variable"].astype(str).tolist()
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = max(10, float(numeric_profile["Missing %"].max()))
        chart.valueAxis.valueStep = max(1, chart.valueAxis.valueMax / 5)
        drawing = Drawing(500, 180)
        drawing.add(chart)
        story.extend([Spacer(1, 8), Paragraph("Missing values by numeric variable (%)", styles["Heading3"]), drawing])
    for title, values in (
        ("Key patterns", analysis["findings"]),
        ("Early warnings", analysis["early_warnings"] or ["No data-derived early warning was triggered."]),
        ("Risk findings and limitations", [
            "Predicted Risk is available only from a supervised model with an evaluable target. "
            "Anomaly means statistical unusualness; a Risk Indicator is a detected data pattern. "
            "Neither establishes default, fraud, or misconduct.",
            analysis["model"].get(
                "limitation",
                "No explicit default target was found. Anomaly and risk-pattern analysis was performed instead."
                if not analysis["targets"]
                else "A supervised model could not be evaluated; statistical risk-pattern analysis was performed instead.",
            ),
        ]),
        ("Recommendations", analysis["recommendations"]),
    ):
        story.append(Paragraph(title, styles["Heading3"]))
        story.extend(Paragraph(f"- {escape(str(value))}", styles["BodyText"]) for value in values)
    if not analysis["correlations"].empty:
        story.extend([Paragraph("Strongest numeric associations", styles["Heading3"])])
        rows = [["Variable A", "Variable B", "Absolute correlation"]]
        rows.extend([
            [str(row["Variable A"]), str(row["Variable B"]), f"{row['Absolute correlation']:.3f}"]
            for _, row in analysis["correlations"].head(10).iterrows()
        ])
        table = Table(rows, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0D1B2A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(table)
    if not analysis["categorical_summary"].empty:
        story.extend([
            Paragraph("Categorical distributions", styles["Heading3"]),
            Table(
                [["Variable", "Value", "Records", "Share %"]]
                + [
                    [str(row["Variable"]), str(row["Value"]), str(row["Records"]), str(row["Share %"])]
                    for _, row in analysis["categorical_summary"].head(20).iterrows()
                ],
                repeatRows=1,
            ),
        ])
    if not analysis["high_risk_groups"].empty:
        story.extend([Paragraph("Observed high-risk segments", styles["Heading3"])])
        rows = [["Variable", "Group", "Records", "Observed risk pattern %"]]
        rows.extend([
            [
                str(row["Variable"]), str(row["Group"]), str(row["Records"]),
                f"{row['Observed risk pattern %']:.1f}",
            ]
            for _, row in analysis["high_risk_groups"].head(12).iterrows()
        ])
        story.append(Table(rows, repeatRows=1))
    document.build(story)
    return output.getvalue()


def create_data_intelligence_excel(analysis: dict[str, Any]) -> bytes:
    """Create a multi-sheet workbook for profiling, statistics, and risk results."""
    output = BytesIO()
    model = analysis["model"]
    model_rows = []
    if analysis["prediction_available"]:
        model_rows.extend([
            {"Metric": "Analysis type", "Value": "Supervised Risk Prediction"},
            {"Metric": "Target", "Value": model["target"]},
            {"Metric": "Model", "Value": model["model"]},
            {"Metric": "Training records", "Value": model["train_rows"]},
            {"Metric": "Evaluation records", "Value": model["test_rows"]},
            {"Metric": "Evaluation limitation", "Value": model.get("limitation", "")},
        ])
        model_rows.extend(
            {"Metric": key.replace("_", " ").title(), "Value": value}
            for key, value in model.items()
            if key in {"accuracy", "balanced_accuracy", "mae", "r2"}
        )
        model_rows.extend(
            {"Metric": f"Feature importance: {item['feature']}", "Value": item["importance"]}
            for item in model.get("top_features", [])
        )
    else:
        model_rows.extend([
            {
                "Metric": "Analysis type",
                "Value": analysis["analysis_result_title"],
            },
            {
                "Metric": "ML target",
                "Value": analysis["targets"][0] if analysis["targets"] else "Not detected",
            },
            {
                "Metric": "Risk prediction",
                "Value": analysis["risk_prediction_summary"],
            },
            {
                "Metric": "Alternative analysis",
                "Value": analysis["alternative_analysis"],
            },
        ])
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([
            {"Metric": "Dataset type", "Value": analysis["dataset_type"]},
            {"Metric": "Rows", "Value": len(analysis["data"])},
            {"Metric": "Variables", "Value": len(analysis["profile"])},
            {"Metric": "Data quality %", "Value": analysis["data_quality_percent"]},
            {"Metric": "Missing cells %", "Value": analysis["missing_percent"]},
            {"Metric": "Duplicate records", "Value": analysis["duplicate_count"]},
            {"Metric": "Anomalies", "Value": analysis["anomaly_count"]},
            {"Metric": "Processed rows included in workbook", "Value": min(len(analysis["data"]), 50_000)},
        ]).to_excel(writer, sheet_name="Overview", index=False)
        analysis["data"].head(50_000).to_excel(writer, sheet_name="Processed Data", index=False)
        analysis["profile"].to_excel(writer, sheet_name="Data Profile", index=False)
        analysis["statistics"].to_excel(writer, sheet_name="Statistics", index=False)
        analysis["categorical_summary"].to_excel(writer, sheet_name="Categorical Summary", index=False)
        analysis["correlations"].to_excel(writer, sheet_name="Correlations", index=False)
        analysis["trends"].to_excel(writer, sheet_name="Trends", index=False)
        analysis["risk_results"].to_excel(writer, sheet_name="Risk Results", index=False)
        analysis["segments"].to_excel(writer, sheet_name="Segments", index=False)
        analysis["high_risk_groups"].to_excel(writer, sheet_name="Risk Segments", index=False)
        analysis["target_distribution"].to_excel(writer, sheet_name="Target Distribution", index=False)
        pd.DataFrame(model_rows).to_excel(writer, sheet_name="Model Results", index=False)
        pd.DataFrame({"Finding": analysis["findings"]}).to_excel(writer, sheet_name="Key Findings", index=False)
        pd.DataFrame({"Early Warning": analysis["early_warnings"]}).to_excel(writer, sheet_name="Early Warnings", index=False)
        pd.DataFrame({"Recommendation": analysis["recommendations"]}).to_excel(writer, sheet_name="Recommendations", index=False)
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


_BAND_COLOURS = {
    "Low Risk": colors.HexColor("#10B981"),
    "Moderate Risk": colors.HexColor("#F59E0B"),
    "High Risk": colors.HexColor("#F97316"),
    "Critical Risk": colors.HexColor("#EF4444"),
    "Insufficient Data": colors.HexColor("#94A3B8"),
}


def _style_table(table: Table, widths: list[float] | None = None) -> Table:
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5F5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
    ]))
    if widths:
        table._argW = widths
    return table


def create_screening_pdf(
    scores: pd.DataFrame,
    summary: dict[str, Any],
    filename: str,
    exposure: pd.DataFrame | None = None,
    max_rows: int = 120,
) -> bytes:
    """Create a portfolio screening report with band mix and ranked records."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=letter,
        title="Portfolio screening report", author="MSME Credit Risk Research Dashboard",
    )
    styles = getSampleStyleSheet()
    story: list[Any] = [
        Paragraph("Portfolio Screening Report", styles["Title"]),
        Paragraph(escape(f"Source: {filename}"), styles["Normal"]),
        Spacer(1, 0.18 * inch),
        Paragraph(
            "Screening scores are research estimates produced by an existing statistical model "
            "or, for records with too few observed fields, a transparent rule-based index. "
            "They are not lending decisions and carry no calibration guarantee.",
            styles["Normal"],
        ),
        Spacer(1, 0.2 * inch),
    ]
    summary_rows = [["Records screened", f"{summary.get('records', 0):,}"]]
    summary_rows.extend(
        [band, f"{count:,}"] for band, count in (summary.get("bands") or {}).items()
    )
    summary_rows.extend([
        ["ML model scored", f"{summary.get('model_records', 0):,}"],
        ["Rule-based index", f"{summary.get('rule_records', 0):,}"],
        ["Records with warning signals", f"{summary.get('flagged_records', 0):,}"],
        ["Mean risk estimate", _display(summary.get("mean_probability"))],
    ])
    story.append(Paragraph("Screening summary", styles["Heading2"]))
    summary_cells = [
        [Paragraph(str(cell), styles["BodyText"]) for cell in row]
        for row in summary_rows
    ]
    story.append(_style_table(Table(summary_cells, colWidths=[4.2 * inch, 1.6 * inch])))
    story.append(Spacer(1, 0.22 * inch))

    if exposure is not None and not exposure.empty:
        story.append(Paragraph("Exposure by risk band", styles["Heading2"]))
        exposure_rows = [[str(column) for column in exposure.columns]]
        exposure_rows.extend([[_display(cell) for cell in row] for row in exposure.itertuples(index=False)])
        story.append(_style_table(Table(exposure_rows, repeatRows=1)))
        story.append(Spacer(1, 0.22 * inch))

    ordered = scores.sort_values("Distress probability", ascending=False)
    story.append(Paragraph(
        f"Highest estimated risk records (up to {max_rows} of {len(ordered):,})", styles["Heading2"],
    ))
    listed = [
        "Record", "Company", "Period", "Risk category", "Distress probability",
        "Health score", "Method", "Warning signals",
    ]
    rows = [listed]
    for record in ordered.head(max_rows).itertuples(index=False):
        probability = record._asdict().get("Distress probability")
        rows.append([
            _display(record._asdict().get("Record")),
            _display(record._asdict().get("Company", "Not identified")),
            _display(record._asdict().get("Period", "Not identified")),
            str(record._asdict().get("Risk category")),
            "Not available" if pd.isna(probability) else f"{float(probability):.1%}",
            _display(record._asdict().get("Health score")),
            str(record._asdict().get("Method")),
            _display(record._asdict().get("Warning signals")),
        ])
    widths = [0.45 * inch, 1.35 * inch, 0.95 * inch, 0.95 * inch, 0.8 * inch, 0.6 * inch, 1.15 * inch, 0.6 * inch]
    table = Table(rows, repeatRows=1)
    _style_table(table, widths)
    for index, record in enumerate(ordered.head(max_rows).itertuples(index=False), start=1):
        band = str(record._asdict().get("Risk category"))
        table.setStyle(TableStyle([("TEXTCOLOR", (3, index), (3, index), _BAND_COLOURS.get(band, colors.black))]))
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "Records shown are sorted by the model's own estimate. Review underlying statements, "
        "data coverage, and early-warning signals before drawing any conclusion.",
        styles["Italic"],
    ))
    document.build(story)
    return buffer.getvalue()


def create_screening_excel(
    scores: pd.DataFrame,
    summary: dict[str, Any],
    exposure: pd.DataFrame | None = None,
    notes: pd.DataFrame | None = None,
    max_rows: int = 50_000,
) -> bytes:
    """Create a screening workbook with summary, bands, exposure, and records."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([
            {"Metric": "Records screened", "Value": summary.get("records", 0)},
            {"Metric": "ML model scored", "Value": summary.get("model_records", 0)},
            {"Metric": "Rule-based index", "Value": summary.get("rule_records", 0)},
            {"Metric": "Insufficient data", "Value": summary.get("insufficient_records", 0)},
            {"Metric": "Records with warning signals", "Value": summary.get("flagged_records", 0)},
            {"Metric": "Mean risk estimate", "Value": summary.get("mean_probability")},
            {"Metric": "Median risk estimate", "Value": summary.get("median_probability")},
            {"Metric": "Elevated records (>=60%)", "Value": summary.get("elevated_records", 0)},
            {"Metric": "Interpretation", "Value": (
                "Screening scores are research estimates, not lending decisions. Records scored by "
                "the rule-based index are heuristics rather than calibrated probabilities."
            )},
        ]).to_excel(writer, sheet_name="Overview", index=False)
        pd.DataFrame([
            {"Risk category": band, "Records": count}
            for band, count in (summary.get("bands") or {}).items()
        ]).to_excel(writer, sheet_name="Band Summary", index=False)
        if exposure is not None and not exposure.empty:
            exposure.to_excel(writer, sheet_name="Exposure", index=False)
        if notes is not None and not notes.empty:
            notes.to_excel(writer, sheet_name="Analyst Notes", index=False)
        ordered = scores.sort_values("Distress probability", ascending=False)
        ordered.head(max_rows).to_excel(writer, sheet_name="Screening", index=False)
        for sheet in writer.book.worksheets:
            for column_cells in sheet.columns:
                letter = get_column_letter(column_cells[0].column)
                width = max((len(str(cell.value)) for cell in column_cells if cell.value is not None), default=10)
                sheet.column_dimensions[letter].width = min(max(width + 2, 10), 60)
            sheet.freeze_panes = "A2"
            for cell in sheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="0F172A")
                cell.alignment = Alignment(horizontal="left", vertical="center")
    return output.getvalue()
