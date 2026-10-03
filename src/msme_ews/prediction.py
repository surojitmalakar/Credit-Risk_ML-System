"""Risk scoring API shared by Streamlit and FastAPI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from msme_ews.data import validate_financial_data
from msme_ews.explain import explain_prediction
from msme_ews.features import engineer_features

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "msme_model.joblib"


def load_model_bundle(path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    return joblib.load(path)


def risk_category(probability: float) -> str:
    if probability < 0.25:
        return "Low Risk"
    if probability < 0.60:
        return "Moderate Risk"
    return "High Risk"


def predict_financial_health(financial_data: dict[str, Any] | pd.DataFrame,
                             bundle: dict[str, Any] | None = None,
                             include_explanations: bool = True) -> dict[str, Any]:
    """Predict distress probability; scores are not lending decisions."""
    data = pd.DataFrame([financial_data]) if isinstance(financial_data, dict) else financial_data.copy()
    data = validate_financial_data(data)
    if len(data) != 1:
        raise ValueError("Prediction accepts exactly one company-period at a time.")
    if bundle is None:
        bundle = load_model_bundle()
    features = engineer_features(data)
    probability = float(bundle["model"].predict_proba(features[bundle["features"]])[0, 1])
    result: dict[str, Any] = {
        "distress_probability": probability, "risk_category": risk_category(probability),
        "confidence_indicator": abs(probability - 0.5) * 2,
        "confidence_note": "Distance from the decision midpoint only; not calibrated uncertainty.",
        "top_risk_factors": [], "protective_factors": [],
        "disclaimer": "Research estimate only; not a guaranteed prediction or financial decision.",
    }
    if include_explanations:
        explanation = explain_prediction(bundle, data)
        result["top_risk_factors"] = explanation["risk_factors"]
        result["protective_factors"] = explanation["protective_factors"]
    return result