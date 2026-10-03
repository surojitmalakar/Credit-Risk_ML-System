"""SHAP-based explanations of model behavior."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from msme_ews.features import engineer_features


def _transform(bundle: dict[str, Any], raw_features: pd.DataFrame) -> np.ndarray:
    model = bundle["model"]
    values = model.named_steps["imputer"].transform(raw_features[bundle["features"]])
    if "scaler" in model.named_steps:
        values = model.named_steps["scaler"].transform(values)
    return np.asarray(values)


def _shap_explainer(bundle: dict[str, Any]) -> Any:
    import shap

    classifier, background = bundle["model"].named_steps["classifier"], bundle["background"]
    if hasattr(classifier, "feature_importances_"):
        return shap.TreeExplainer(classifier, data=background, feature_perturbation="interventional")
    return shap.LinearExplainer(classifier, background)


def explain_prediction(bundle: dict[str, Any], row: pd.DataFrame) -> dict[str, Any]:
    """Explain the model output with signed per-feature SHAP values."""
    features = engineer_features(row)
    explanation = _shap_explainer(bundle)(_transform(bundle, features)[:1])
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[0, :, 1] if values.shape[-1] > 1 else values[0, :, 0]
    elif values.ndim == 2:
        values = values[0]
    contributions = pd.DataFrame({"feature": bundle["features"], "value": features.iloc[0].to_numpy(),
                                  "shap_value": values.astype(float)})
    risks = contributions[contributions["shap_value"] > 0].nlargest(5, "shap_value")
    protective = contributions[contributions["shap_value"] < 0].nsmallest(5, "shap_value")
    return {"risk_factors": _records(risks), "protective_factors": _records(protective),
            "contributions": contributions.sort_values("shap_value", ascending=False)}


def global_importance(bundle: dict[str, Any], frame: pd.DataFrame, max_rows: int = 150) -> pd.DataFrame:
    """Compute mean absolute SHAP values on a bounded sample."""
    features = engineer_features(frame).iloc[:max_rows]
    if features.empty:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    values = np.asarray(_shap_explainer(bundle)(_transform(bundle, features)).values)
    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[-1] > 1 else values[:, :, 0]
    return pd.DataFrame({"feature": bundle["features"], "mean_abs_shap": np.mean(np.abs(values), axis=0)}).sort_values("mean_abs_shap", ascending=False)


def _records(frame: pd.DataFrame) -> list[dict[str, float | str | None]]:
    rows = []
    for record in frame.to_dict(orient="records"):
        value = record["value"]
        rows.append({"feature": str(record["feature"]),
                     "value": float(value) if pd.notna(value) else None,
                     "contribution": float(record["shap_value"])})
    return rows