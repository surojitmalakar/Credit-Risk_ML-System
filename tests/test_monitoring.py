from __future__ import annotations

import numpy as np
import pandas as pd

from msme_ews.demo import make_demo_data
from msme_ews.features import engineer_features
from msme_ews.monitoring import (
    build_snapshot,
    drift_report,
    monitoring_alerts,
    population_stability_index,
    prediction_stability,
    psi_label,
    snapshot_table,
)


def _psi_inputs(shift: float = 0.0, rows: int = 4_000, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(shift, 1.0, rows))


def test_identical_distributions_have_a_stable_psi():
    baseline = _psi_inputs()
    psi = population_stability_index(baseline, baseline.copy())
    assert psi == psi
    assert psi < 0.01
    assert psi_label(psi) == "Stable"


def test_large_shifts_raise_the_psi_and_change_the_label():
    baseline = _psi_inputs()
    stable = population_stability_index(baseline, _psi_inputs(0.2))
    shifted = population_stability_index(baseline, _psi_inputs(2.5))
    assert shifted > stable
    assert psi_label(shifted) == "Significant shift"
    assert psi_label(0.15) == "Watch"
    assert psi_label(0.05) == "Stable"


def test_psi_is_not_computable_for_degenerate_inputs():
    empty = pd.Series(dtype="float64")
    assert np.isnan(population_stability_index(empty, _psi_inputs(10)))
    assert np.isnan(population_stability_index(empty, empty))
    assert np.isnan(population_stability_index(pd.Series([1.0] * 50), pd.Series([2.0] * 50)))
    assert np.isnan(population_stability_index(_psi_inputs(), empty))
    assert psi_label(float("nan")) == "Not comparable"


def test_snapshot_captures_shape_completeness_and_distributions():
    frame = make_demo_data()
    features = engineer_features(frame)
    snapshot = build_snapshot("revision 0", frame, features)

    assert snapshot["records"] == len(frame)
    assert snapshot["columns"] == len(frame.columns)
    assert 0 <= snapshot["missing_percent"] <= 100
    assert snapshot["duplicates"] >= 0
    assert "Current_Ratio" in snapshot["distributions"]
    assert "Revenue" in snapshot["distributions"]
    assert len(snapshot["distributions"]["Current_Ratio"]) == len(frame)


def test_snapshot_records_model_fields_when_analysis_is_supplied():
    frame = make_demo_data()
    analysis = {
        "dataset_type": "MSME/business dataset",
        "anomaly_count": 12,
        "anomaly_rows_scored": 720,
        "missing_percent": 3.5,
        "duplicate_count": 2,
        "model": {"status": "evaluated", "model": "RandomForestClassifier", "accuracy": 0.81},
    }
    snapshot = build_snapshot("revision 1", frame, engine_features_placeholder(), analysis)

    assert snapshot["dataset_type"] == "MSME/business dataset"
    assert snapshot["anomaly_count"] == 12
    assert snapshot["missing_percent"] == 3.5
    assert snapshot["duplicates"] == 2
    assert snapshot["model_status"] == "evaluated"
    assert snapshot["accuracy"] == 0.81


def engine_features_placeholder() -> pd.DataFrame:
    return engineer_features(make_demo_data())


def test_snapshot_table_hides_internal_fields():
    frame = make_demo_data()
    snapshot = build_snapshot("revision 0", frame)
    snapshot["_dataset_key"] = "abc"
    table = snapshot_table([snapshot, snapshot])

    assert "distributions" not in table.columns
    assert "_dataset_key" not in table.columns
    assert len(table) == 2
    assert table["records"].tolist() == [len(frame), len(frame)]


def test_drift_report_marks_a_shifted_dataset_and_keeps_psi_rows():
    baseline_frame = make_demo_data()
    shifted_frame = make_demo_data()
    shifted_frame.loc[shifted_frame.index[: len(shifted_frame) // 2], "Revenue"] *= 4

    baseline = build_snapshot("baseline", baseline_frame, engineer_features(baseline_frame))
    current = build_snapshot("current", shifted_frame, engineer_features(shifted_frame))
    drift = drift_report(baseline, current)

    assert list(drift["Metric"][:4]) == ["Records", "Variables", "Missing cells", "Duplicate records"]
    psi_rows = drift[drift["Metric"].str.startswith("PSI")]
    assert not psi_rows.empty
    assert psi_rows["Status"].isin({"Stable", "Watch", "Significant shift", "Not comparable"}).all()
    revenue_psi = psi_rows.loc[psi_rows["Metric"] == "PSI · Revenue", "Current"].iloc[0]
    assert revenue_psi > 0.10


def test_prediction_stability_tracks_the_elevated_share():
    stability = prediction_stability([
        pd.Series([0.1, 0.2, 0.3, 0.7]),
        pd.Series([0.1, 0.2, 0.8, 0.9]),
    ])
    assert list(stability["Revision"]) == [1, 2]
    assert stability["Scored records"].tolist() == [4, 4]
    assert stability["Elevated (>=60%)"].tolist() == [1, 2]
    assert stability["Elevated share"].iloc[1] > stability["Elevated share"].iloc[0]
    assert 0 <= stability["Mean risk"].iloc[0] <= 1
    assert prediction_stability([pd.Series(dtype="float64")])["Mean risk"].isna().all()


def test_monitoring_alerts_describe_observed_movement():
    frame = make_demo_data()
    baseline = build_snapshot("baseline", frame, engineer_features(frame))
    unchanged = drift_report(baseline, baseline)
    assert monitoring_alerts(unchanged, []) == []

    shifted_frame = make_demo_data()
    shifted_frame.loc[shifted_frame.index[:300], "Revenue"] *= 5
    current = build_snapshot("current", shifted_frame, engineer_features(shifted_frame))
    drift = drift_report(baseline, current)
    alerts = monitoring_alerts(drift, [pd.Series([0.1] * 8 + [0.9] * 2), pd.Series([0.1] * 4 + [0.9] * 6)])
    assert alerts
    assert any("shift" in alert.lower() for alert in alerts)
    assert any("elevated-risk records" in alert for alert in alerts)
