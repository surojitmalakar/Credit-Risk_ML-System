"""Dataset and prediction drift monitoring across analysis revisions.

These diagnostics compare successive uploads or re-analyses of the same file.
Population Stability Index is a screening signal for distribution change, not a
model validity test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PSI_BINS = 10
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
MONITORED_FEATURES = (
    "Revenue",
    "Net_Profit",
    "Total_Assets",
    "Current_Liabilities",
    "Debt",
    "Cash_Flow_Operations",
    "Current_Ratio",
    "Debt_to_Assets",
    "EBITDA_Margin",
    "Interest_Coverage",
)
_LABELS = {
    0.0: "Stable",
    PSI_MODERATE: "Watch",
    PSI_SIGNIFICANT: "Significant shift",
}


def population_stability_index(
    baseline: pd.Series,
    current: pd.Series,
    bins: int = PSI_BINS,
) -> float:
    """PSI between two numeric distributions; NaN when it cannot be computed."""
    base = pd.to_numeric(baseline, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    latest = pd.to_numeric(current, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if base.empty or latest.empty:
        return float("nan")
    if base.nunique() <= 1 or latest.nunique() <= 1:
        return float("nan")
    edges = np.unique(np.quantile(base.to_numpy(), np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return float("nan")
    edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
    base_share = np.histogram(base.to_numpy(), bins=edges)[0].astype("float64")
    latest_share = np.histogram(latest.to_numpy(), bins=edges)[0].astype("float64")
    base_share = np.divide(base_share, base_share.sum(), out=np.zeros_like(base_share), where=base_share.sum() > 0)
    latest_share = np.divide(latest_share, latest_share.sum(), out=np.zeros_like(latest_share), where=latest_share.sum() > 0)
    floor = 1e-6
    return float(np.sum((latest_share - base_share) * np.log((latest_share + floor) / (base_share + floor))))


def psi_label(value: float) -> str:
    if pd.isna(value):
        return "Not comparable"
    for threshold, label in reversed(list(_LABELS.items())):
        if value >= threshold:
            return label
    return "Stable"


def distribution_summary(frame: pd.DataFrame, features: pd.DataFrame | None = None) -> dict[str, pd.Series]:
    """Numeric distributions used for drift comparison."""
    summary: dict[str, pd.Series] = {}
    for column in MONITORED_FEATURES:
        source = None
        if features is not None and column in features.columns:
            source = features[column]
        elif column in frame.columns:
            source = frame[column]
        if source is None:
            continue
        summary[column] = pd.to_numeric(source, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return summary


def build_snapshot(
    label: str,
    frame: pd.DataFrame,
    features: pd.DataFrame | None = None,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture the comparable state of one analysis revision."""
    snapshot: dict[str, Any] = {
        "label": label,
        "records": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_percent": round(float(frame.isna().to_numpy().mean() * 100), 3) if frame.size else 0.0,
        "duplicates": int(frame.duplicated().sum()),
        "distributions": distribution_summary(frame, features),
    }
    if analysis is not None:
        snapshot["dataset_type"] = analysis.get("dataset_type")
        snapshot["anomaly_count"] = analysis.get("anomaly_count")
        snapshot["anomaly_rows_scored"] = analysis.get("anomaly_rows_scored")
        snapshot["missing_percent"] = analysis.get("missing_percent", snapshot["missing_percent"])
        snapshot["duplicates"] = analysis.get("duplicate_count", snapshot["duplicates"])
        model = analysis.get("model", {})
        snapshot["model_status"] = model.get("status")
        snapshot["model"] = model.get("model")
        snapshot["accuracy"] = model.get("accuracy")
        snapshot["balanced_accuracy"] = model.get("balanced_accuracy")
    return snapshot


def snapshot_table(snapshots: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per recorded revision, without internal bookkeeping fields."""
    return pd.DataFrame([
        {key: value for key, value in snapshot.items() if key != "distributions" and not key.startswith("_")}
        for snapshot in snapshots
    ])


def drift_report(
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> pd.DataFrame:
    """Compare two snapshots field by field."""
    rows: list[dict[str, Any]] = []

    def add(name: str, before: Any, after: Any, unit: str = "", severity: str = "") -> None:
        change = None
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            change = after - before
        rows.append({
            "Metric": name,
            "Baseline": before,
            "Current": after,
            "Change": change,
            "Unit": unit,
            "Status": severity,
        })

    add("Records", baseline["records"], current["records"])
    add("Variables", baseline["columns"], current["columns"])
    add("Missing cells", baseline["missing_percent"], current["missing_percent"], "%")
    add("Duplicate records", baseline["duplicates"], current["duplicates"])
    for feature in MONITORED_FEATURES:
        if feature in baseline["distributions"] or feature in current["distributions"]:
            value = population_stability_index(
                baseline["distributions"].get(feature, pd.Series(dtype="float64")),
                current["distributions"].get(feature, pd.Series(dtype="float64")),
            )
            add(f"PSI · {feature}", None, value, "PSI", psi_label(value))
    return pd.DataFrame(rows)


def prediction_stability(scores_history: list[pd.Series]) -> pd.DataFrame:
    """Track the 0-100 rule-index distribution across successive revisions."""
    rows: list[dict[str, Any]] = []
    for position, scores in enumerate(scores_history, start=1):
        values = pd.to_numeric(pd.Series(scores), errors="coerce").dropna()
        rows.append({
            "Revision": position,
            "Scored records": int(len(values)),
            "Mean risk index": None if values.empty else round(float(values.mean()), 1),
            "Median risk index": None if values.empty else round(float(values.median()), 1),
            "Elevated (index >=60)": int((values >= 60).sum()),
            "Elevated share": None if values.empty else round(float((values >= 60).mean()), 4),
        })
    return pd.DataFrame(rows)


def monitoring_alerts(drift: pd.DataFrame, scores_history: list[pd.Series]) -> list[str]:
    """Plain-language alerts; they describe observed movement, not model failure."""
    alerts: list[str] = []
    if not drift.empty:
        significant = drift.loc[drift["Status"] == "Significant shift", "Metric"].tolist()
        watch = drift.loc[drift["Status"] == "Watch", "Metric"].tolist()
        if significant:
            alerts.append(
                "Material distribution shift since the baseline revision in: " + ", ".join(significant)
                + ". Re-validate before relying on scores for this population."
            )
        if watch:
            alerts.append(
                "Moderate shift worth monitoring in: " + ", ".join(watch) + "."
            )
        record_change = drift.loc[drift["Metric"] == "Records", "Change"]
        if not record_change.empty and pd.notna(record_change.iloc[0]) and int(record_change.iloc[0]) != 0:
            alerts.append(
                f"Record count moved by {int(record_change.iloc[0]):+,} since the baseline revision."
            )
    stability = prediction_stability(scores_history)
    if len(stability) >= 2:
        latest = stability.iloc[-1]
        first = stability.iloc[0]
        if latest["Elevated share"] is not None and first["Elevated share"] is not None:
            delta = float(latest["Elevated share"]) - float(first["Elevated share"])
            if abs(delta) >= 0.05:
                direction = "more" if delta > 0 else "fewer"
                alerts.append(
                    f"The share of elevated-risk records is {abs(delta):.1%} {direction} than in the first screened revision."
                )
    return alerts
