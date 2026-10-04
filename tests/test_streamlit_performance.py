from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_upload_is_analyzed_on_request_and_charts_are_lazy():
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py").run(timeout=30)
    csv = (
        b"client_ref,region,budget\n"
        b"A-1,North,100\n"
        b"A-2,South,120\n"
        b"A-3,West,140\n"
    )
    app.get("file_uploader")[0].upload("customers.csv", csv, "text/csv").run(timeout=60)

    assert not app.exception
    assert len(app.get("status")) == 0
    assert any(button.label == "Analyze Dataset" for button in app.button)

    app.button[0].click().run(timeout=60)

    assert not app.exception
    assert not app.error
    assert len(app.get("status")) == 1
    assert any(button.label == "Re-analyze" for button in app.button)

    app.radio[0].set_value("Data Intelligence").run(timeout=60)

    assert not app.exception
    assert len(app.get("status")) == 0
    assert not app.get("plotly_chart")
    assert any(metric.label == "Records" and metric.value == "3" for metric in app.metric)

    load_charts = next(button for button in app.button if button.label == "Load Visual Analytics")
    load_charts.click().run(timeout=60)

    assert not app.exception
    assert len(app.get("plotly_chart")) > 0

    reanalyze = next(button for button in app.button if button.label == "Re-analyze")
    reanalyze.click().run(timeout=60)

    assert not app.exception
    assert len(app.get("status")) == 1


def test_large_upload_does_not_repeat_analysis_when_navigation_changes():
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py").run(timeout=30)
    rows = 50_001
    csv = "customer_ref,age_years,acquisition_cost,region\n" + "\n".join(
        f"C-{index},{20 + index % 45},{300 + index % 1700},Region-{index % 8}"
        for index in range(rows)
    )
    app.get("file_uploader")[0].upload(
        "large_customers.csv",
        csv.encode(),
        "text/csv",
    ).run(timeout=60)

    assert not app.exception
    assert len(app.get("status")) == 0
    app.button[0].click().run(timeout=120)

    assert not app.exception
    assert len(app.get("status")) == 1
    app.radio[0].set_value("Data Intelligence").run(timeout=120)

    assert not app.exception
    assert len(app.get("status")) == 0
    assert any(metric.label == "Records" and metric.value == "50,001" for metric in app.metric)
    assert not app.get("plotly_chart")
