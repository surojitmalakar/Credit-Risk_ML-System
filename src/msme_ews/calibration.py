"""Calibration diagnostics and adjustable risk-band cutoffs.

These metrics describe how well a probability matches an observed outcome on the
rows it was evaluated on. They are research diagnostics: they do not make a
model suitable for a lending decision, and an in-sample comparison will always
look better than a held-out one.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from sklearn.metrics import brier_score_loss, roc_auc_score

from msme_ews.prediction import risk_category

DEFAULT_CUTOFFS = (0.20, 0.40, 0.70)
CALIBRATION_BINS = 10


def _aligned(probabilities: Sequence[float] | pd.Series, labels: Sequence[int] | pd.Series) -> tuple[pd.Series, pd.Series]:
    probability = pd.to_numeric(pd.Series(probabilities), errors="coerce").reset_index(drop=True)
    outcome = pd.Series(labels).reset_index(drop=True)
    if len(probability) != len(outcome):
        raise ValueError("Probabilities and labels must have the same length.")
    valid = probability.notna() & outcome.notna()
    probability = probability[valid].astype(float).clip(0.0, 1.0)
    outcome = pd.to_numeric(outcome[valid], errors="coerce")
    keep = outcome.isin([0, 1])
    return probability[keep].reset_index(drop=True), outcome[keep].astype(int).reset_index(drop=True)


def calibration_table(
    probabilities: Sequence[float] | pd.Series,
    labels: Sequence[int] | pd.Series,
    bins: int = CALIBRATION_BINS,
) -> pd.DataFrame:
    """Predicted versus observed default rate per probability band."""
    probability, outcome = _aligned(probabilities, labels)
    if probability.empty:
        return pd.DataFrame(columns=[
            "Band", "Lower", "Upper", "Records", "Predicted default rate",
            "Observed default rate", "Calibration gap",
        ])
    edges = np.linspace(0.0, 1.0, bins + 1)
    frame = pd.DataFrame({"probability": probability, "observed": outcome})
    frame["band"] = pd.cut(frame["probability"], bins=edges, include_lowest=True, labels=False)
    grouped = frame.groupby("band", observed=True).agg(
        Records=("probability", "size"),
        Predicted=("probability", "mean"),
        Observed=("observed", "mean"),
    ).reset_index(drop=True)
    grouped["Lower"] = edges[:-1]
    grouped["Upper"] = edges[1:]
    grouped["Band"] = [
        f"{lower:.0%}-{upper:.0%}" for lower, upper in zip(grouped["Lower"], grouped["Upper"], strict=True)
    ]
    grouped["Predicted default rate"] = (grouped["Predicted"] * 100).round(1)
    grouped["Observed default rate"] = (grouped["Observed"] * 100).round(1)
    grouped["Calibration gap"] = (
        grouped["Predicted default rate"] - grouped["Observed default rate"]
    ).round(1)
    return grouped[[
        "Band", "Lower", "Upper", "Records", "Predicted default rate",
        "Observed default rate", "Calibration gap",
    ]]


def reliability_metrics(
    probabilities: Sequence[float] | pd.Series,
    labels: Sequence[int] | pd.Series,
    table: pd.DataFrame | None = None,
) -> dict[str, float | None]:
    """Brier score, expected calibration error, base rates, and separation."""
    probability, outcome = _aligned(probabilities, labels)
    if probability.empty:
        return {
            "records": 0, "observed_default_rate": None, "predicted_default_rate": None,
            "brier_score": None, "expected_calibration_error": None, "roc_auc": None,
            "lift": None, "calibration_slope": None,
        }
    table = calibration_table(probability, outcome) if table is None else table
    observed_mean = float(outcome.mean())
    predicted_mean = float(probability.mean())
    error = (
        float((table["Calibration gap"].abs() * table["Records"]).sum() / table["Records"].sum() / 100)
        if not table.empty else None
    )
    lift = (
        float(outcome[probability >= 0.5].mean() / observed_mean)
        if observed_mean > 0 and (probability >= 0.5).any() else None
    )
    return {
        "records": int(len(probability)),
        "observed_default_rate": observed_mean,
        "predicted_default_rate": predicted_mean,
        "brier_score": float(brier_score_loss(outcome, probability)),
        "expected_calibration_error": error,
        "roc_auc": float(roc_auc_score(outcome, probability)) if outcome.nunique() == 2 else None,
        "lift": lift,
        "calibration_slope": _calibration_slope(probability, outcome),
    }


def _calibration_slope(probability: pd.Series, outcome: pd.Series) -> float | None:
    """Logistic recalibration slope; 1.0 means no systematic bias."""
    if len(probability) < 10 or outcome.nunique() < 2:
        return None
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000)
        model.fit(
            np.log(np.clip(probability.to_numpy(), 1e-6, 1 - 1e-6)).reshape(-1, 1),
            outcome.to_numpy(),
        )
        return float(model.coef_[0][0])
    except (ValueError, np.linalg.LinAlgError):
        return None


def threshold_table(
    probabilities: Sequence[float] | pd.Series,
    labels: Sequence[int] | pd.Series,
    cutoffs: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
) -> pd.DataFrame:
    """Precision, recall, and volume at each decision cutoff."""
    probability, outcome = _aligned(probabilities, labels)
    if probability.empty:
        return pd.DataFrame(columns=[
            "Cutoff", "Flagged", "Flagged share", "Precision", "Recall",
            "Precision (balanced)", "Observed default rate of flagged",
        ])
    base_rate = float(outcome.mean())
    rows = []
    for cutoff in cutoffs:
        flagged = probability >= cutoff
        selected = outcome[flagged]
        precision = float(selected.mean()) if len(selected) else None
        recall = float(flagged[outcome.to_numpy() == 1].sum() / max((outcome == 1).sum(), 1))
        balanced_precision = (
            precision / base_rate if precision is not None and base_rate > 0 else None
        )
        rows.append({
            "Cutoff": float(cutoff),
            "Flagged": int(flagged.sum()),
            "Flagged share": float(flagged.mean()),
            "Precision": None if precision is None else round(precision, 4),
            "Recall": round(recall, 4),
            "Precision (balanced)": None if balanced_precision is None else round(balanced_precision, 2),
        })
    return pd.DataFrame(rows)


def band_distribution(
    probabilities: Sequence[float] | pd.Series,
    cutoffs: Sequence[float] = DEFAULT_CUTOFFS,
) -> pd.DataFrame:
    """Record counts per band for adjustable cutoffs."""
    moderate, high, critical = cutoffs
    probability = pd.to_numeric(pd.Series(probabilities), errors="coerce")
    bands = pd.Series("Insufficient Data", index=probability.index, dtype="object")
    bands = bands.mask(probability.isna(), "Insufficient Data")
    bands = bands.mask(probability.notna() & (probability < moderate), "Low Risk")
    bands = bands.mask(probability.between(moderate, high, inclusive="left"), "Moderate Risk")
    bands = bands.mask(probability.between(high, critical, inclusive="left"), "High Risk")
    bands = bands.mask(probability >= critical, "Critical Risk")
    counts = bands.value_counts()
    order = ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk", "Insufficient Data"]
    return pd.DataFrame([
        {
            "Risk category": band,
            "Records": int(counts.get(band, 0)),
            "Share %": round(100 * counts.get(band, 0) / max(len(probability), 1), 1),
        }
        for band in order
    ])


def shifted_band_counts(
    probabilities: Sequence[float] | pd.Series,
    cutoffs: Sequence[float],
    reference_cutoffs: Sequence[float] = DEFAULT_CUTOFFS,
) -> pd.DataFrame:
    """Compare band counts under tuned cutoffs against the default cutoffs."""
    tuned = band_distribution(probabilities, cutoffs).rename(
        columns={"Records": "Tuned records", "Share %": "Tuned share %"}
    )
    default = band_distribution(probabilities, reference_cutoffs).rename(
        columns={"Records": "Default records", "Share %": "Default share %"}
    )
    merged = tuned.merge(default, on="Risk category")
    merged["Change"] = merged["Tuned records"] - merged["Default records"]
    return merged


def cutoff_summary(probabilities: Sequence[float] | pd.Series, cutoffs: Sequence[float]) -> str:
    """Human-readable description of the tuned cutoffs."""
    moderate, high, critical = cutoffs
    return (
        f"Moderate at {moderate:.0%}, High at {high:.0%}, Critical at {critical:.0%}"
    )


def default_band(probability: float | None) -> str:
    """Band under the model's shipped cutoffs."""
    return risk_category(probability) if probability is not None else "Insufficient Data"