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


def load_model_bundle(path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any] | None:
    """Load a model bundle if it exists; return None if not found."""
    path = Path(path)
    if not path.exists():
        return None
    return joblib.load(path)


def risk_category(probability: float) -> str:
    if probability < 0.25:
        return "Low Risk"
    if probability < 0.60:
        return "Moderate Risk"
    return "High Risk"


def predict_financial_health(financial_data: dict[str, Any] | pd.DataFrame,
                             bundle: dict[str, Any] | None = None,
                             include_explanations: bool = True,
                             features: pd.DataFrame | None = None) -> dict[str, Any]:
    """Predict distress probability; scores are not lending decisions.

    ``features`` accepts an already engineered single-row frame so callers can
    reuse one feature pass and keep single-record scores identical to
    whole-file screening scores.
    """
    data = pd.DataFrame([financial_data]) if isinstance(financial_data, dict) else financial_data.copy()
    data = validate_financial_data(data)
    if len(data) != 1:
        raise ValueError("Prediction accepts exactly one company-period at a time.")
    if bundle is None:
        bundle = load_model_bundle()
    if bundle is None or bundle.get("model") is None:
        raise ValueError(
            "No trained model available. Upload a labeled dataset to train a model, "
            "or use the rule-based assessment for transparent risk indices."
        )
    if bundle.get("is_demo") is True:
        raise ValueError(
            "The configured model was trained on synthetic demo data and cannot be used for production predictions."
        )
    if features is None:
        features = engineer_features(data)
    elif len(features) != 1:
        raise ValueError("Precomputed features must contain exactly one company-period.")
    probability = float(bundle["model"].predict_proba(features[bundle["features"]])[0, 1])
    result: dict[str, Any] = {
        "distress_probability": probability, "risk_category": risk_category(probability),
        "confidence_indicator": abs(probability - 0.5) * 2,
        "confidence_note": "Distance from the decision midpoint only; not calibrated uncertainty.",
        "top_risk_factors": [], "protective_factors": [],
        "disclaimer": "Research estimate only; not a guaranteed prediction or financial decision.",
    }
    if include_explanations:
        explanation = explain_prediction(bundle, data, features=features)
        result["top_risk_factors"] = explanation["risk_factors"]
        result["protective_factors"] = explanation["protective_factors"]
    return result