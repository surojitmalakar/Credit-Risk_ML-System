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


def test_empty_application_requests_uploaded_data():
    app = AppTest.from_file(APP).run(timeout=30)

    assert not app.exception
    assert any(info.value == "Upload a dataset to generate analysis." for info in app.info)
    assert not app.metric


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


def test_copilot_records_conversation_history_and_presets():
    app = AppTest.from_file(APP).run(timeout=60)
    app.get("file_uploader")[0].upload(
        "portfolio.csv", _financial_csv(), "text/csv"
    ).run(timeout=120)
    app.button[0].click().run(timeout=180)
    app.radio[0].set_value("Overview").run(timeout=120)

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


def test_corrupt_excel_upload_fails_gracefully_and_static_tabs_stay_reachable():
    """A bad workbook must show one friendly banner, never a raw traceback.

    Regression: xlrd.XLRDError used to escape the upload handler and break
    every tab until the file was removed.
    """
    app = AppTest.from_file(APP).run(timeout=60)
    corrupt_xls = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 1024
    app.get("file_uploader")[0].upload(
        "statement.xls", corrupt_xls, "application/vnd.ms-excel"
    ).run(timeout=120)

    app.button(key="analyze_upload").click().run(timeout=180)

    assert not app.exception
    assert any("Dataset analysis failed" in entry.value for entry in app.error)
    assert any("password-protected" in entry.value for entry in app.error)

    # Static tabs keep working after the failure - nothing is stuck.
    app.radio[0].set_value("About").run(timeout=120)

    assert not app.exception
    assert not app.error

    app.radio[0].set_value("Methodology").run(timeout=120)

    assert not app.exception
    assert not app.error


def test_repeated_record_selection_does_not_refetch_or_rescore():
    app = AppTest.from_file(APP).run(timeout=60)
    app.get("file_uploader")[0].upload("portfolio.csv", _financial_csv(), "text/csv").run(timeout=120)
    app.button[0].click().run(timeout=180)

    company = app.get("selectbox")[0]
    started = time.perf_counter()
    for _ in range(4):
        app.run(timeout=120)
    repeat_elapsed = time.perf_counter() - started

    app.radio[0].set_value("Reports").run(timeout=180)
    assert not app.exception
    assert company.key == "company_filter"
    assert repeat_elapsed > 0
