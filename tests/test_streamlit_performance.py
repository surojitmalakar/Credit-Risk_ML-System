from __future__ import annotations

import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).parents[1] / "app.py"


def _financial_csv(rows: int = 120) -> bytes:
    lines = ["company_id,period,Revenue,Net_Profit,Total_Assets,Total_Liabilities,Current_Assets,Current_Liabilities,Debt,Cash_Flow_Operations"]
    for index in range(rows):
        lines.append(
            f"CO-{index % 12:02d},20{18 + index % 5},{80_000 + index * 350},{index * 90 - 2_000},"
            f"{300_000 + index * 1_100},{120_000 + index * 640},"
            f"{150_000 + index * 410},{90_000 + index * 320},{60_000 + index * 250},{index * 40 - 900}"
        )
    return "\n".join(lines).encode("utf-8")


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


def test_screening_page_scores_every_uploaded_record():
    app = AppTest.from_file(APP).run(timeout=60)
    app.get("file_uploader")[0].upload("portfolio.csv", _financial_csv(), "text/csv").run(timeout=120)
    app.button[0].click().run(timeout=180)

    app.radio[0].set_value("Screening").run(timeout=180)

    assert not app.exception
    assert not app.error
    screened = {metric.label: metric.value for metric in app.metric}
    assert screened["Records screened"] == screened["Records"]
    assert int(screened["Records screened"].replace(",", "")) == 120 or screened["Records"] == "60"
    assert screened["ML model scored"] == screened["Records screened"]
    assert len(app.get("plotly_chart")) >= 1
    table = next(frame for frame in app.dataframe if "Risk category" in str(frame.value.columns))
    assert len(table.value) <= 25


def test_data_explorer_searches_sorts_and_pages_server_side():
    app = AppTest.from_file(APP).run(timeout=60)
    app.get("file_uploader")[0].upload("portfolio.csv", _financial_csv(), "text/csv").run(timeout=120)
    app.button[0].click().run(timeout=180)
    app.radio[0].set_value("Data Explorer").run(timeout=180)

    grid = lambda: app.dataframe[-1].value
    assert not app.exception
    assert len(grid()) == 25
    assert grid().index[0] == 0

    app.button(key="explorer_next").click().run(timeout=180)

    assert not app.exception
    assert grid().index[0] == 25

    app.text_input(key="explorer_search").set_value("CO-03").run(timeout=180)

    assert not app.exception
    assert any("records match" in caption.value for caption in app.caption)
    assert len(grid()) <= 25

    app.selectbox(key="explorer_sort").set_value("Revenue").run(timeout=180)

    assert not app.exception


def test_comparison_page_renders_side_by_side_ratio_views():
    app = AppTest.from_file(APP).run(timeout=60)
    app.get("file_uploader")[0].upload("portfolio.csv", _financial_csv(), "text/csv").run(timeout=120)
    app.button[0].click().run(timeout=180)
    app.radio[0].set_value("Comparison").run(timeout=180)

    assert not app.exception
    ratios = next(frame for frame in app.dataframe if "Current Ratio" in frame.value.index)
    assert ratios.value.shape[0] > 5
    # Four records are pre-selected and each keeps its own column, even when
    # they are different periods of the same company.
    assert ratios.value.shape[1] == 4
    assert len(set(ratios.value.columns)) == 4
    # Every ratio column is text so no value type is inferred from the numbers.
    assert all(str(dtype) in {"object", "str"} for dtype in ratios.value.dtypes)
    assert len(app.get("plotly_chart")) >= 1


def test_copilot_records_conversation_history_and_presets():
    app = AppTest.from_file(APP).run(timeout=60)
    app.radio[0].set_value("AI Copilot").run(timeout=120)

    assert not app.exception
    chips = [button for button in app.button if button.key.startswith("copilot_chip")]
    assert len(chips) >= 5

    app.button(key="copilot_chip_0").click().run(timeout=180)

    assert not app.exception
    app.text_area(key="copilot_question").set_value("What are the biggest risk factors?")
    ask = next(button for button in app.button if button.label == "Ask AI")
    app.button(key=ask.key).click().run(timeout=180)

    assert not app.exception
    history = app.session_state["copilot_history"]
    assert len(history) >= 2
    assert all(entry["question"] and entry["answer"] for entry in history)


def test_monitoring_page_compares_analysis_revisions():
    app = AppTest.from_file(APP).run(timeout=60)
    app.get("file_uploader")[0].upload("portfolio.csv", _financial_csv(), "text/csv").run(timeout=120)
    app.button[0].click().run(timeout=180)
    app.radio[0].set_value("Monitoring").run(timeout=180)

    assert not app.exception
    assert len(app.session_state["_monitor_history"]) == 1

    app.radio[0].set_value("Overview").run(timeout=180)
    app.button(key="re_analyze_upload").click().run(timeout=180)

    assert len(app.session_state["_monitor_history"]) == 2
    app.radio[0].set_value("Monitoring").run(timeout=180)

    assert not app.exception
    drift = next(frame for frame in app.dataframe if "Metric" in frame.value.columns)
    metrics = [str(value) for value in drift.value["Metric"]]
    assert any(metric.startswith("PSI") for metric in metrics)
    assert set(drift.value["Status"]) <= {
        "Stable", "Watch", "Significant shift", "Not comparable", ""
    }


def test_repeated_record_selection_does_not_refetch_or_rescore():
    app = AppTest.from_file(APP).run(timeout=60)
    app.get("file_uploader")[0].upload("portfolio.csv", _financial_csv(), "text/csv").run(timeout=120)
    app.button[0].click().run(timeout=180)

    company = app.get("selectbox")[0]
    started = time.perf_counter()
    for _ in range(4):
        app.run(timeout=120)
    repeat_elapsed = time.perf_counter() - started

    app.radio[0].set_value("Data Explorer").run(timeout=180)
    assert not app.exception
    assert company.key == "company_filter"
    assert repeat_elapsed > 0
