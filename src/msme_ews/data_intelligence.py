"""Offline, schema-flexible profiling and risk-pattern analysis."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, r2_score


_ROLE_PATTERNS: dict[str, tuple[str, ...]] = {
    "revenue / income": ("revenue", "sales", "turnover", "net sales", "income", "earnings"),
    "profit": ("profit", "pat", "net income", "earnings", "ebitda", "margin"),
    "debt / exposure": ("debt", "borrow", "loan balance", "outstanding", "exposure", "principal"),
    "credit score": ("credit score", "bureau score", "cibil", "fico", "score"),
    "customer identifier": ("customer id", "client id", "account id", "borrower id", "member id"),
    "default / risk target": ("default", "delinquen", "bad loan", "risk flag", "risk status", "fraud", "churn", "target", "label", "outcome"),
    "date / time": ("date", "time", "period", "year", "month", "timestamp"),
    "transaction amount": ("transaction", "amount", "payment", "spend", "purchase", "credit", "debit"),
    "geography": ("city", "state", "country", "region", "location", "pin", "postal"),
    "assets": ("asset", "cash", "receivable", "inventory"),
    "liabilities": ("liabilit", "payable"),
}
_FINANCIAL_TERMS = {"revenue / income", "profit", "debt / exposure", "assets", "liabilities"}
_TARGET_WORDS = ("default", "delinquen", "risk", "fraud", "churn", "target", "label", "outcome", "status")
_IDENTIFIER_WORDS = ("id", "identifier", "account", "reference", "ref", "code", "number", "uuid")


def _key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _numeric_value(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"na", "n/a", "null", "none", "-", "--"}:
        return np.nan
    negative = text.startswith("(") and text.endswith(")")
    percent = "%" in text
    text = re.sub(r"[₹$€£]", "", text)
    text = re.sub(r"(?i)\b(?:rs\.?|inr|usd|eur|gbp)\b", "", text)
    text = text.replace(",", "").replace("(", "").replace(")", "").strip()
    match = re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return np.nan
    number = float(text)
    if negative:
        number = -abs(number)
    return number if not percent else number


def _role_for_column(column: object, values: pd.Series) -> list[str]:
    name = _key(column)
    roles = []
    for role, patterns in _ROLE_PATTERNS.items():
        if role == "date / time":
            matched = bool(re.search(r"\b(date|time|period|year|month|timestamp)\b", name))
        else:
            matched = any(pattern in name for pattern in patterns)
        if matched:
            roles.append(role)
    if not roles:
        for role, patterns in _ROLE_PATTERNS.items():
            if any(SequenceMatcher(None, name, pattern).ratio() >= 0.88 for pattern in patterns):
                roles.append(role)
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if "credit score" not in roles and len(numeric) and numeric.between(300, 900).mean() > 0.8:
        roles.append("credit score (range-inferred)")
    if not roles and re.search(r"\b(?:amt|value|balance|total|amount)\b", name) and len(numeric):
        roles.append("potential financial / exposure")
    return list(dict.fromkeys(roles))


def _parse_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    parsed = frame.copy()
    numeric_columns: list[str] = []
    date_columns: list[str] = []
    for original in parsed.columns:
        column = str(original)
        source = parsed[original]
        if pd.api.types.is_numeric_dtype(source):
            parsed[original] = pd.to_numeric(source, errors="coerce")
            numeric_columns.append(column)
            continue
        nonempty = source.dropna()
        if nonempty.empty:
            continue
        name = _key(column)
        date_hint = any(word in name for word in ("date", "time", "period", "year", "month"))
        date_like_values = nonempty.astype(str).str.contains(r"[-/:]", regex=True).mean() >= 0.6
        if date_hint or date_like_values:
            dates = pd.to_datetime(source, errors="coerce", format="mixed")
            plausible = dates.dropna().between("1900-01-01", "2100-12-31").mean() if dates.notna().any() else 0
            if dates.notna().sum() / len(nonempty) >= 0.6 and plausible >= 0.8:
                parsed[original] = dates
                date_columns.append(column)
                continue
        converted = source.map(_numeric_value)
        if converted.notna().sum() / len(nonempty) >= 0.8 and not any(word in name for word in _IDENTIFIER_WORDS):
            parsed[original] = converted
            numeric_columns.append(column)
    return parsed, numeric_columns, date_columns


def _dataset_type(
    frame: pd.DataFrame,
    roles: dict[str, list[str]],
    filename: str,
    numeric_columns: list[str],
    date_columns: list[str],
) -> str:
    filename_key = _key(filename)
    role_set = {role for matches in roles.values() for role in matches}
    kinds: list[str] = []
    if role_set & _FINANCIAL_TERMS:
        kinds.append("Financial statement" if "date / time" in role_set else "MSME/business dataset")
    if "credit score" in role_set or "default / risk target" in role_set:
        kinds.append("Loan/credit dataset")
    if "customer identifier" in role_set or any(word in filename_key for word in ("customer", "client", "borrower")):
        kinds.append("Customer dataset")
    if "transaction amount" in role_set and date_columns:
        kinds.append("Transaction dataset")
    if any(word in filename_key for word in ("sale", "sales", "order")) or "revenue / income" in role_set:
        kinds.append("Sales dataset")
    if not kinds and len(frame.columns) >= 2:
        if date_columns and numeric_columns:
            kinds.append("Transaction dataset")
        elif role_set:
            kinds.append("MSME/business dataset")
    unique = list(dict.fromkeys(kinds))
    if len(unique) > 1:
        return "Mixed dataset"
    return unique[0] if unique else "Unknown dataset"


def _target_columns(frame: pd.DataFrame, roles: dict[str, list[str]]) -> list[str]:
    targets = []
    for column in frame.columns:
        name = _key(column)
        values = frame[column].dropna()
        if any(word in name for word in _TARGET_WORDS) and 1 < values.nunique() <= 20:
            targets.append(str(column))
        elif any("default / risk target" in role for role in roles.get(str(column), [])):
            if 1 < values.nunique() <= 20:
                targets.append(str(column))
    return list(dict.fromkeys(targets))


def _is_risk_label(value: object) -> bool:
    label = _key(value)
    if not label or any(phrase in label for phrase in ("no default", "not default", "non default", "not late", "not risky")):
        return False
    return (
        label in {"1", "true", "yes", "y"}
        or any(term in label for term in ("default", "delinquen", "late", "high risk", "critical", "severe", "fraud", "bad loan", "charge off", "failed"))
    )


def _model_result(frame: pd.DataFrame, target: str, identifiers: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "not trained", "target": target}
    usable = frame.dropna(subset=[target]).head(20_000).copy()
    y = usable[target]
    classification = not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10
    if len(usable) < 30:
        result["reason"] = "At least 30 labeled rows are required for a held-out evaluation."
        return result
    if y.nunique() < 2 or y.nunique() > 20:
        result["reason"] = "Target needs 2 to 20 observed classes for automatic classification."
        return result
    features = usable.drop(columns=[target, *[name for name in identifiers if name in usable]])
    features = features.loc[:, features.notna().mean().ge(0.2)]
    features = features.loc[:, features.nunique(dropna=True).gt(1)]
    features = features.loc[:, [
        column for column in features.columns
        if pd.api.types.is_numeric_dtype(features[column])
        or features[column].nunique(dropna=True) / max(len(features), 1) <= 0.8
    ]].iloc[:, :50]
    if features.empty:
        result["reason"] = "No varying, sufficiently observed predictor columns remain."
        return result
    numeric = [str(c) for c in features if pd.api.types.is_numeric_dtype(features[c])]
    categorical = [str(c) for c in features if str(c) not in numeric]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
            ]), categorical),
        ],
        remainder="drop",
    )
    if classification:
        counts = y.value_counts()
        stratify = y if counts.min() >= 2 and counts.size <= 10 else None
        model = RandomForestClassifier(n_estimators=120, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=1)
    else:
        stratify = None
        model = RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=42, n_jobs=1)
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features, y, test_size=0.25, random_state=42, stratify=stratify,
        )
        pipeline = Pipeline([("prepare", preprocessor), ("model", model)])
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        result.update(status="evaluated", model=type(model).__name__, train_rows=len(x_train), test_rows=len(x_test))
        if classification:
            result["accuracy"] = float(accuracy_score(y_test, prediction))
            result["balanced_accuracy"] = float(balanced_accuracy_score(y_test, prediction))
        else:
            result["mae"] = float(mean_absolute_error(y_test, prediction))
            result["r2"] = float(r2_score(y_test, prediction))
        transformed_names = pipeline.named_steps["prepare"].get_feature_names_out()
        importances = pipeline.named_steps["model"].feature_importances_
        ranked = sorted(zip(transformed_names, importances, strict=False), key=lambda item: item[1], reverse=True)[:10]
        result["top_features"] = [{"feature": str(name), "importance": float(value)} for name, value in ranked]
        result["limitation"] = "Exploratory hold-out model; not calibrated and not a probability of default."
    except (ValueError, TypeError) as error:
        result["reason"] = f"Automatic model could not be evaluated: {error}"
    return result


def analyze_dataset(frame: pd.DataFrame, filename: str = "uploaded_data") -> dict[str, Any]:
    """Profile arbitrary tabular data and produce cautious, data-derived intelligence."""
    if frame.empty or not len(frame.columns):
        raise ValueError("The uploaded file contains no analyzable records or columns.")
    data, numeric, dates = _parse_frame(frame)
    categorical = [
        str(column) for column in data.columns
        if str(column) not in numeric and str(column) not in dates
        and data[column].nunique(dropna=True) <= max(50, int(len(data) * 0.5))
    ]
    roles = {str(column): _role_for_column(column, data[column]) for column in data.columns}
    identifiers = [
        str(column) for column in data.columns
        if any(word in _key(column).split() for word in _IDENTIFIER_WORDS)
        or (data[column].nunique(dropna=True) / max(len(data), 1) >= 0.95 and data[column].nunique(dropna=True) > 10)
    ]
    targets = _target_columns(data, roles)
    missing = data.isna().sum()
    missing_rate = float(data.isna().to_numpy().mean()) if data.size else 0.0
    profile_rows = []
    for column in data.columns:
        values = data[column]
        profile_rows.append({
            "Variable": str(column),
            "Type": "Numeric" if str(column) in numeric else "Date" if str(column) in dates else "Categorical",
            "Missing": int(missing[column]),
            "Missing %": round(float(missing[column] / max(len(data), 1) * 100), 2),
            "Unique": int(values.nunique(dropna=True)),
            "Sample values": ", ".join(map(str, values.dropna().head(3).tolist())),
            "Detected meaning": ", ".join(roles[str(column)]) or ("Potential ID" if str(column) in identifiers else ""),
        })
    profile = pd.DataFrame(profile_rows)
    stats = data[numeric].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.reset_index(names="Variable") if numeric else pd.DataFrame()
    category_rows: list[dict[str, Any]] = []
    for column in categorical:
        values = data[column].astype("string").fillna("Missing")
        counts = values.value_counts().head(10)
        category_rows.extend({
            "Variable": column,
            "Value": str(value),
            "Records": int(count),
            "Share %": round(float(count / len(data) * 100), 2),
        } for value, count in counts.items())
    categorical_summary = pd.DataFrame(category_rows)

    correlation_rows: list[dict[str, Any]] = []
    if len(numeric) >= 2:
        correlations = data[numeric].corr().abs()
        for left_index, left in enumerate(numeric):
            for right in numeric[left_index + 1:]:
                value = correlations.loc[left, right]
                if pd.notna(value):
                    correlation_rows.append({"Variable A": left, "Variable B": right, "Absolute correlation": float(value)})
    correlations_frame = pd.DataFrame(sorted(correlation_rows, key=lambda row: row["Absolute correlation"], reverse=True)[:15])
    trend_rows: list[dict[str, Any]] = []
    if dates:
        trend_dates = pd.to_datetime(data[dates[0]], errors="coerce")
        for column in numeric[:12]:
            trend_values = pd.DataFrame({"date": trend_dates, "value": data[column]}).dropna()
            by_date = trend_values.groupby("date")["value"].median().sort_index()
            if len(by_date) < 3:
                continue
            first_value = float(by_date.iloc[0])
            latest_value = float(by_date.iloc[-1])
            change = (
                (latest_value - first_value) / abs(first_value) * 100
                if first_value != 0 else None
            )
            trend_rows.append({
                "Variable": column,
                "From": str(by_date.index[0]),
                "To": str(by_date.index[-1]),
                "First value": first_value,
                "Latest value": latest_value,
                "Change %": round(change, 2) if change is not None else None,
                "Direction": "Increasing" if latest_value > first_value else "Declining" if latest_value < first_value else "Stable",
            })
        frequency = trend_dates.dropna().dt.to_period("M").value_counts().sort_index()
        if len(frequency) >= 3:
            trend_rows.append({
                "Variable": "Record frequency per month",
                "From": str(frequency.index[0]),
                "To": str(frequency.index[-1]),
                "First value": int(frequency.iloc[0]),
                "Latest value": int(frequency.iloc[-1]),
                "Change %": round((frequency.iloc[-1] - frequency.iloc[0]) / frequency.iloc[0] * 100, 2)
                if frequency.iloc[0] else None,
                "Direction": "Increasing" if frequency.iloc[-1] > frequency.iloc[0] else "Declining" if frequency.iloc[-1] < frequency.iloc[0] else "Stable",
            })
    trends = pd.DataFrame(
        trend_rows,
        columns=["Variable", "From", "To", "First value", "Latest value", "Change %", "Direction"],
    )

    anomaly_scores = pd.Series(0.0, index=data.index)
    anomalies = pd.Series(False, index=data.index)
    if numeric and len(data) >= 10:
        anomaly_data = data[numeric].replace([np.inf, -np.inf], np.nan)
        anomaly_data = anomaly_data.loc[:, anomaly_data.notna().any()]
        if not anomaly_data.empty:
            filled = SimpleImputer(strategy="median").fit_transform(anomaly_data)
            if np.isfinite(filled).all():
                detector = IsolationForest(contamination="auto", random_state=42, n_estimators=100, n_jobs=1)
                detector.fit(filled[:10_000])
                scores = -detector.score_samples(filled)
                anomaly_scores = pd.Series(scores, index=data.index)
                cutoff = float(np.quantile(scores, 0.95))
                anomalies = anomaly_scores >= cutoff
    segment_frame = pd.DataFrame()
    segment_columns = [
        column for column in numeric
        if data[column].nunique(dropna=True) > 1 and data[column].notna().any()
    ][:20]
    if len(segment_columns) >= 2 and len(data) >= 10:
        segment_data = data[segment_columns].replace([np.inf, -np.inf], np.nan)
        filled_segments = SimpleImputer(strategy="median").fit_transform(segment_data)
        if np.isfinite(filled_segments).all():
            scaled = StandardScaler().fit_transform(filled_segments[:20_000])
            cluster_count = min(4, max(2, int(np.sqrt(len(scaled) / 2))))
            labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(scaled)
            segment_frame = pd.DataFrame({
                "Segment": [f"Segment {label + 1}" for label in labels],
                "Records": 1,
            }).groupby("Segment", as_index=False).sum()
            segment_frame["Share %"] = (segment_frame["Records"] / len(labels) * 100).round(2)
            if target := next(iter(_target_columns(data, roles)), None):
                target_share = data[target].head(len(labels)).astype("string").fillna("Missing").reset_index(drop=True)
                segment_frame["Most common target value"] = [
                    target_share[np.asarray(labels) == label].mode().iloc[0]
                    if (np.asarray(labels) == label).any() else "Not available"
                    for label in range(cluster_count)
                ]
    missing_risk = data.isna().mean(axis=1) if len(data.columns) else pd.Series(0, index=data.index)
    risk_results = pd.DataFrame({
        "Source row": np.arange(len(data)) + 1,
        "Missing %": (missing_risk * 100).round(2),
        "Anomaly score": anomaly_scores.round(5),
        "Anomaly": anomalies.to_numpy(),
        "Detected Risk Pattern": np.where(
            anomalies.to_numpy(),
            "Unusual multivariate pattern; review required",
            np.where(missing_risk.to_numpy() >= 0.5, "High missingness; interpretation is limited", "No anomaly flag"),
        ),
    })
    target = targets[0] if targets else None
    model_result = _model_result(data, target, identifiers) if target else {
        "status": "not trained",
        "reason": "No explicit low-cardinality target variable was detected; no predicted risk is reported.",
    }
    dataset_type = _dataset_type(data, roles, filename, numeric, dates)
    high_risk_groups: list[dict[str, Any]] = []
    target_distribution = pd.DataFrame()
    if target:
        target_values = data[target].astype("string").fillna("Missing")
        target_distribution = target_values.value_counts(dropna=False).rename_axis("Observed target value").reset_index(name="Records")
        target_distribution["Share %"] = (target_distribution["Records"] / len(data) * 100).round(2)
        risk_mask = data[target].map(_is_risk_label)
        if risk_mask.any():
            for category in categorical:
                if category == target or data[category].nunique(dropna=True) > 20:
                    continue
                group_data = pd.DataFrame({
                    "Group": data[category].astype("string").fillna("Missing"),
                    "Observed risk label": risk_mask,
                }).groupby("Group", dropna=False).agg(
                    Records=("Observed risk label", "size"),
                    Observed_risk_count=("Observed risk label", "sum"),
                ).reset_index()
                group_data = group_data.loc[group_data["Records"] >= 3].copy()
                if group_data.empty:
                    continue
                group_data["Observed risk pattern %"] = (
                    group_data["Observed_risk_count"] / group_data["Records"] * 100
                ).round(2)
                group_data["Variable"] = category
                high_risk_groups.extend(
                    group_data.nlargest(3, "Observed risk pattern %").to_dict("records")
                )
        high_risk_groups_frame = pd.DataFrame(high_risk_groups)
    else:
        high_risk_groups_frame = pd.DataFrame()
    findings: list[str] = [
        f"Dataset contains {len(data):,} rows and {len(data.columns):,} columns; {missing_rate:.1%} of cells are missing.",
        f"{len(numeric)} numeric, {len(categorical)} categorical, and {len(dates)} date/time variables were detected.",
        f"{int(data.duplicated().sum()):,} duplicate records were found.",
    ]
    if targets:
        findings.append(f"Potential target '{target}' has {int(data[target].nunique(dropna=True))} observed values.")
    if not high_risk_groups_frame.empty:
        findings.append(
            f"Observed risk-label patterns vary across {high_risk_groups_frame['Variable'].nunique()} categorical segment variable(s)."
        )
    if not correlations_frame.empty:
        top = correlations_frame.iloc[0]
        findings.append(
            f"Strongest numeric association: {top['Variable A']} with {top['Variable B']} "
            f"(absolute correlation {top['Absolute correlation']:.2f}); association is not causation."
        )
    for _, trend in trends.iterrows():
        change_text = f" ({trend['Change %']:.1f}% from first to last observed period)" if pd.notna(trend["Change %"]) else ""
        findings.append(f"Observed trend for {trend['Variable']}: {trend['Direction'].lower()}{change_text}.")
    if anomalies.any():
        findings.append(f"{int(anomalies.sum()):,} records fall in the top 5% of anomaly scores.")
    if not segment_frame.empty:
        findings.append(f"Numeric patterns support exploratory segmentation into {len(segment_frame)} groups.")
    recommendations = [
        "Review columns with missing values before operational use.",
        "Verify detected variable meanings against the source data dictionary.",
    ]
    if anomalies.any():
        recommendations.append("Review flagged anomaly records; an anomaly is not by itself evidence of default or misconduct.")
    if not trends.empty and trends["Direction"].eq("Declining").any():
        recommendations.append("Review declining time-series measures and verify that reporting periods are comparable.")
    if target and model_result["status"] == "evaluated":
        recommendations.append("Treat model metrics as exploratory hold-out results and validate on representative future data.")
    concentration: list[dict[str, Any]] = []
    exposure_columns = [
        str(column) for column, column_roles in roles.items()
        if any(term in " ".join(column_roles).lower() for term in ("exposure", "transaction amount", "revenue / income"))
        and str(column) in numeric
    ]
    for column in exposure_columns[:3]:
        values = data[column].dropna()
        total = values.sum()
        if len(values) and total > 0:
            concentration.append({
                "Variable": column,
                "Top 10% share": float(values.nlargest(max(1, int(np.ceil(len(values) * 0.1)))).sum() / total),
            })
    enriched = data.copy()
    enriched["detected_anomaly"] = anomalies.to_numpy()
    enriched["anomaly_score"] = anomaly_scores.to_numpy()
    return {
        "data": enriched,
        "dataset_type": dataset_type,
        "profile": profile,
        "statistics": stats,
        "categorical_summary": categorical_summary,
        "correlations": correlations_frame,
        "trends": trends,
        "risk_results": risk_results,
        "roles": roles,
        "identifiers": identifiers,
        "targets": targets,
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "date_columns": dates,
        "duplicate_count": int(data.duplicated().sum()),
        "missing_percent": round(missing_rate * 100, 2),
        "data_quality_percent": round((1 - missing_rate) * 100, 2),
        "anomaly_count": int(anomalies.sum()),
        "model": model_result,
        "findings": findings,
        "early_warnings": [
            f"{row['Variable']} is declining across observed periods."
            for _, row in trends.loc[trends["Direction"].eq("Declining")].iterrows()
        ] + (
            [f"{anomalies.sum():,} records have unusual multivariate patterns."]
            if anomalies.any() else []
        ),
        "recommendations": recommendations,
        "concentration": pd.DataFrame(concentration),
        "segments": segment_frame,
        "target_distribution": target_distribution,
        "high_risk_groups": high_risk_groups_frame,
        "executive_summary": (
            f"Automatic profiling classified this upload as {dataset_type.lower()}. "
            f"It contains {len(data):,} rows and {len(data.columns):,} variables, with "
            f"{(1 - missing_rate) * 100:.1f}% non-missing cell coverage. "
            + (
                "A supervised model was evaluated on a hold-out split; its results are exploratory."
                if model_result["status"] == "evaluated"
                else "No predicted risk probability is reported."
            )
        ),
    }
