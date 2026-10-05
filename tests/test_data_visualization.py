from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from msme_ews.data_intelligence import analyze_dataset
from msme_ews.data_visualization import (
    _CHART_COLORS,
    _match_financial_columns,
    _meaningful_numeric,
    build_data_visualizations,
)


def _customer_master() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_ref": [f"C-{index:03}" for index in range(40)],
        "customer_age": [22 + (index * 3) % 45 for index in range(40)],
        "gender": ["Female", "Male", "Female", "Other"] * 10,
        "customer_segment": ["Retail", "SME", "Premium", "Retail"] * 10,
        "state": ["Gujarat", "Maharashtra", "Karnataka", "Delhi"] * 10,
        "acquisition_cost": [450 + index * 25 + (500 if index == 19 else 0) for index in range(40)],
        "joined_date": pd.date_range("2023-01-01", periods=40, freq="MS"),
    })


def test_customer_master_gets_real_distributions_relationships_trends_and_anomalies():
    source = _customer_master()
    analysis = analyze_dataset(source, "customer_master.csv")
    visualizations = build_data_visualizations(analysis)

    assert "Financial Analysis" not in visualizations
    assert any(item["title"] == "Distribution: customer_age" for item in visualizations["Distributions"])
    assert any(item["title"] == "Records by state" for item in visualizations["Distributions"])
    assert any("acquisition_cost by customer_segment" in item["title"] for item in visualizations["Relationships"])
    assert any("over time" in item["title"] for item in visualizations["Time Trends"])
    assert "Risk and Anomalies" in visualizations
    histogram = next(
        item["figure"] for item in visualizations["Distributions"]
        if item["title"] == "Distribution: customer_age"
    )
    assert set(histogram.data[0].x) <= set(source["customer_age"])
    assert any("Correlation Heatmap" in item["title"] for item in visualizations["Relationships"])


def test_sparse_or_constant_data_yields_no_invalid_or_fabricated_charts():
    analysis = analyze_dataset(pd.DataFrame({
        "only_value": [1, 1, 1],
        "category": ["same", "same", "same"],
    }), "small.csv")

    assert build_data_visualizations(analysis) == {}


def test_financial_charts_only_appear_when_the_required_columns_exist():
    financial = analyze_dataset(pd.DataFrame({
        "Revenue": [100, 120, 130],
        "Net_Profit": [10, 12, 14],
        "Debt": [40, 38, 35],
        "Current_Assets": [60, 65, 72],
        "Current_Liabilities": [30, 32, 36],
        "Total_Assets": [100, 110, 120],
        "Total_Liabilities": [50, 52, 55],
        "Equity_Value": [50, 58, 65],
        "Cash_Flow_Operations": [12, 15, 18],
        "period": pd.date_range("2022-12-31", periods=3, freq="YE"),
    }), "financials.csv")
    visualizations = build_data_visualizations(financial)
    titles = [item["title"] for item in visualizations["Financial Analysis"]]

    assert any("revenue trend" == title.lower() for title in titles)
    assert any("profit trend" == title.lower() for title in titles)
    assert any("debt trend" == title.lower() for title in titles)
    assert "Assets vs liabilities" in titles
    assert "Current ratio" in titles
    assert "Debt-to-equity" in titles
    assert any("cash flow trend" == title.lower() for title in titles)


def test_sales_growth_is_never_presented_as_revenue():
    frame = pd.DataFrame({
        "company_id": [f"CO-{index:02d}" for index in range(30)],
        "period": ["2021-12-31", "2022-12-31", "2023-12-31"] * 10,
        "Sales_Growth": [-0.1, 0.05, 0.2] * 10,
        "Revenue": [100 + index for index in range(30)],
        "Net_Profit": [5 + index % 4 for index in range(30)],
    })
    analysis = analyze_dataset(frame, "growth_first.csv")
    matched = _match_financial_columns(
        _meaningful_numeric(analysis),
        {column: " ".join(analysis["roles"].get(column, [])).lower() for column in _meaningful_numeric(analysis)},
    )

    assert matched["revenue"] == "Revenue"
    assert matched["profit"] == "Net_Profit"

    visualizations = build_data_visualizations(analysis)
    trend = next(item for item in visualizations["Financial Analysis"] if item["title"] == "Revenue trend")

    assert trend["figure"].layout.yaxis.title.text == "Revenue"


def test_current_assets_are_never_presented_as_total_assets():
    frame = pd.DataFrame({
        "Current_Assets": [60, 70, 80, 90],
        "Current_Liabilities": [30, 35, 40, 45],
        "Cash_Flow_Operations": [12, 15, 18, 21],
    })
    visualizations = build_data_visualizations(analyze_dataset(frame, "current_only.csv"))
    titles = [item["title"] for item in visualizations["Financial Analysis"]]

    assert "Current ratio" in titles
    assert "Assets vs liabilities" not in titles


def test_identifier_style_columns_do_not_drive_charts():
    frame = pd.DataFrame({
        "Customer_ID": [f"C-{index:03d}" for index in range(40)],
        "Account Number": [f"{4000 + index}" for index in range(40)],
        "Amount": [100 + index for index in range(40)],
        "Segment": ["Retail", "SME", "Premium"] * 13 + ["Retail"],
    })
    visualizations = build_data_visualizations(analyze_dataset(frame, "ids.csv"))
    titled = " ".join(item["title"] for items in visualizations.values() for item in items)

    assert "Customer_ID" not in titled
    assert "Account Number" not in titled
    assert any(item["title"] == "Distribution: Amount" for item in visualizations["Distributions"])


def test_infinite_values_are_treated_as_missing_everywhere():
    frame = pd.DataFrame({
        "Revenue": [100.0, 110.0, np.inf, 130.0] * 10,
        "Total_Assets": [200.0, 210.0, 220.0, 230.0] * 10,
        "period": ["2021", "2022", "2023", "2024"] * 10,
    })

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        analysis = analyze_dataset(frame, "infinities.csv")
        visualizations = build_data_visualizations(analysis)

    assert analysis["data"]["Revenue"].replace([np.inf, -np.inf], np.nan).isna().sum() == 10
    histogram = next(
        item for item in visualizations["Distributions"]
        if item["title"] == "Distribution: Revenue"
    )

    assert not np.isinf(np.asarray(histogram["figure"].data[0].x, dtype=float)).any()


def test_anomaly_scatter_colours_match_the_boolean_flag():
    frame = pd.DataFrame({
        "Amount": [10, 12, 14, 40, 11, 13] * 10,
        "Balance": [5, 6, 7, 30, 5, 6] * 10,
        "Debt": [2, 3, 3, 25, 2, 3] * 10,
    })
    visualizations = build_data_visualizations(analyze_dataset(frame, "anomalies.csv"))
    scatter = next(item for item in visualizations["Risk and Anomalies"] if item["title"] == "Anomaly scatter")
    colours = {trace.name: trace.marker.color for trace in scatter["figure"].data}

    assert set(colours) == {"True", "False"}
    assert colours["True"] == _CHART_COLORS[4]
    assert colours["False"] == _CHART_COLORS[2]


def test_financial_chart_states_what_it_sums():
    frame = pd.DataFrame({
        "Total_Assets": [100, 110, 120, 130],
        "Total_Liabilities": [50, 55, 60, 65],
        "period": ["2021", "2022", "2023", "2024"],
    })
    visualizations = build_data_visualizations(analyze_dataset(frame, "totals.csv"))
    chart = next(item for item in visualizations["Financial Analysis"] if item["title"] == "Assets vs liabilities")

    assert "sum of uploaded values" in chart["figure"].layout.title.text


def test_large_dataset_labels_visual_sample_without_sampling_analysis_input():
    source = pd.DataFrame({
        "monthly_income": range(50_001),
        "region": ["North", "South", "West", "East"] * 12_500 + ["North"],
    })
    analysis = analyze_dataset(source, "large_customers.csv")
    visualizations = build_data_visualizations(analysis)
    histogram = next(
        chart for chart in visualizations["Distributions"]
        if chart["title"].startswith("Distribution: monthly_income")
    )

    assert len(analysis["data"]) == 50_001
    assert "visual sample: 50,000 of 50,001 records" in histogram["figure"].layout.title.text
    assert len(histogram["figure"].data[0].x) == 50_000
