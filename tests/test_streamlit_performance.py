from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_upload_is_reused_on_navigation_and_only_reanalyzed_on_request():
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py").run(timeout=30)
    csv = (
        b"client_ref,region,budget\n"
        b"A-1,North,100\n"
        b"A-2,South,120\n"
        b"A-3,West,140\n"
    )
    app.get("file_uploader")[0].upload("customers.csv", csv, "text/csv").run(timeout=60)

    assert not app.exception
    assert len(app.get("status")) == 1
    assert any(metric.label == "Records" and metric.value == "3" for metric in app.metric)

    app.radio[0].set_value("Financial Health").run(timeout=60)

    assert not app.exception
    assert not app.error
    assert len(app.get("status")) == 0

    reanalyze = next(button for button in app.button if button.label == "Re-analyze dataset")
    reanalyze.click().run(timeout=60)

    assert not app.exception
    assert len(app.get("status")) == 1
