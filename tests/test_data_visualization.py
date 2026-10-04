from __future__ import annotations

import pandas as pd

from msme_ews.data_intelligence import analyze_dataset
from msme_ews.data_visualization import build_data_visualizations


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
