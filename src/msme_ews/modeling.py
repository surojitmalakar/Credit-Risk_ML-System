"""Leakage-conscious grouped model comparison and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from msme_ews.data import validate_financial_data
from msme_ews.features import MODEL_FEATURES, engineer_features


def _target_values(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        target = pd.to_numeric(series, errors="coerce")
    else:
        mapping = {"1": 1, "true": 1, "yes": 1, "distress": 1, "default": 1,
                   "0": 0, "false": 0, "no": 0, "healthy": 0, "no distress": 0, "non-default": 0}
        target = series.astype(str).str.strip().str.lower().map(mapping)
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("Target must contain only binary labels 0 and 1 (or recognized text labels).")
    return target.astype(int)


def _pipeline(classifier: Any, scale: bool = True) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def _metrics(y_true: pd.Series, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float | None]:
    both_classes = y_true.nunique() == 2
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if both_classes else None,
        "pr_auc": float(average_precision_score(y_true, probabilities)) if both_classes else None,
    }


def _fairness_summary(y: pd.Series, pred: np.ndarray, groups: pd.Series) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for group_value in groups.dropna().unique():
        mask = groups.eq(group_value).to_numpy()
        group_y, group_pred = y.loc[mask], pred[mask]
        positive, negative = group_y == 1, group_y == 0
        result[str(group_value)] = {
            "n": int(mask.sum()), "positive_label_rate": float(group_y.mean()),
            "true_positive_rate": float(group_pred[positive].mean()) if positive.any() else None,
            "false_positive_rate": float(group_pred[negative].mean()) if negative.any() else None,
        }
    return result


def train_models(frame: pd.DataFrame, target_column: str = "distress_label",
                 protected_attribute: str | None = None, random_state: int = 42) -> dict[str, Any]:
    """Compare classifiers with group-disjoint test and CV splits when possible."""
    data = validate_financial_data(frame)
    if target_column not in data:
        raise ValueError(f"Target column {target_column!r} is missing.")
    target = _target_values(data[target_column])
    features = engineer_features(data)
    if target.nunique() != 2:
        raise ValueError("Training requires both distress classes (0 and 1).")
    groups = data["company_id"].astype(str) if "company_id" in data else None
    if groups is not None and groups.nunique() >= 5:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        train_indices, test_indices = next(splitter.split(features, target, groups))
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state + 1)
        cv_splits = list(cv.split(features.iloc[train_indices], target.iloc[train_indices], groups.iloc[train_indices]))
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        train_indices, test_indices = next(splitter.split(features, target))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state + 1)
        cv_splits = list(cv.split(features.iloc[train_indices], target.iloc[train_indices]))

    train_x, test_x = features.iloc[train_indices], features.iloc[test_indices]
    train_y, test_y = target.iloc[train_indices], target.iloc[test_indices]
    if train_y.nunique() != 2 or test_y.nunique() != 2:
        raise ValueError("A split contains only one target class; supply more labeled examples.")
    positive_weight = float((train_y == 0).sum() / max((train_y == 1).sum(), 1))
    candidates = {
        "Logistic Regression": _pipeline(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        "Random Forest": _pipeline(RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=random_state), scale=False),
        "XGBoost": _pipeline(XGBClassifier(n_estimators=220, max_depth=3, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0, scale_pos_weight=positive_weight,
            eval_metric="logloss", n_jobs=1, random_state=random_state), scale=False),
    }
    cv_scores: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    scorers = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1",
               "roc_auc": "roc_auc", "pr_auc": "average_precision"}
    for name, pipeline in candidates.items():
        scores = cross_validate(pipeline, train_x, train_y, cv=cv_splits, scoring=scorers,
                                n_jobs=1, error_score="raise")
        cv_scores[name] = {metric: float(np.mean(scores[f"test_{metric}"])) for metric in scorers}
        pipeline.fit(train_x, train_y)
        fitted[name] = pipeline

    selected_name = max(cv_scores, key=lambda name: cv_scores[name]["roc_auc"])
    selected_pipeline = fitted[selected_name]
    test_probabilities = selected_pipeline.predict_proba(test_x)[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(int)
    is_demo = data.attrs.get("dataset_kind") == "synthetic_demo"
    if "dataset_source" in data:
        is_demo = is_demo or data["dataset_source"].eq("synthetic_demo").any()
    report: dict[str, Any] = {
        "selected_model": selected_name, "selection_metric": "mean cross-validated ROC-AUC",
        "cv_metrics": cv_scores, "test_metrics": _metrics(test_y, test_probabilities, test_predictions),
        "test_rows": int(len(test_indices)),
        "test_companies": int(groups.iloc[test_indices].nunique()) if groups is not None else None,
        # Retained so the app can report calibration on held-out rows.
        "test_probabilities": [float(value) for value in test_probabilities],
        "test_labels": [int(value) for value in test_y],
        "evaluation_note": "Synthetic demo data results are illustrative only." if is_demo else "Dataset-specific research estimates, not lending decisions.",
    }
    if protected_attribute and protected_attribute in data:
        report["fairness"] = _fairness_summary(test_y.reset_index(drop=True), test_predictions,
            data[protected_attribute].iloc[test_indices].reset_index(drop=True))

    selected_pipeline.fit(features, target)
    background = selected_pipeline.named_steps["imputer"].transform(features)
    if "scaler" in selected_pipeline.named_steps:
        background = selected_pipeline.named_steps["scaler"].transform(background)
    return {"model": selected_pipeline, "model_name": selected_name, "features": list(MODEL_FEATURES),
            "background": np.asarray(background)[:min(100, len(background))], "report": report,
            "is_demo": is_demo}


def calibration_sample(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    target_column: str = "distress_label",
) -> dict[str, Any]:
    """Probabilities and outcomes used for calibration diagnostics.

    Prefers the held-out test split recorded during training so the comparison is
    out-of-sample. When those values are absent, scores the supplied frame and
    reports that the sample is in-sample, which flatters calibration.
    """
    report = bundle.get("report", {})
    probabilities = report.get("test_probabilities")
    labels = report.get("test_labels")
    if probabilities is not None and labels is not None and len(probabilities) == len(labels):
        return {
            "probabilities": list(probabilities),
            "labels": list(labels),
            "basis": "held-out test split",
            "is_in_sample": False,
        }
    if target_column not in frame.columns:
        return {"probabilities": [], "labels": [], "basis": "unavailable", "is_in_sample": False}
    target = _target_values(frame[target_column])
    features = engineer_features(frame)
    model = bundle.get("model")
    if model is None or not all(name in features.columns for name in bundle.get("features", [])):
        return {"probabilities": [], "labels": [], "basis": "unavailable", "is_in_sample": False}
    predictions = model.predict_proba(features[bundle["features"]])[:, 1]
    return {
        "probabilities": [float(value) for value in predictions],
        "labels": [int(value) for value in target],
        "basis": "scored on the loaded dataset, so it is in-sample",
        "is_in_sample": True,
    }