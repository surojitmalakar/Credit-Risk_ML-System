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
    "revenue / income": ("revenue", "annual sales", "total sales", "sales", "turnover", "net sales", "income"),
    "profit": ("net profit", "profit after tax", "profit after taxation", "profit", "pat", "net income", "earnings", "ebitda", "margin"),
    "debt / exposure": ("total debt", "debt", "borrowings", "borrow", "loan balance", "loans", "outstanding", "exposure", "principal"),
    "credit score": ("credit score", "bureau score", "cibil", "fico", "score"),
    "customer identifier": ("customer id", "client id", "account id", "borrower id", "member id"),
    "default / risk target": ("default", "delinquen", "bad loan", "risk flag", "risk status", "fraud", "churn", "target", "label", "outcome"),
    "date / time": ("date", "time", "period", "year", "month", "timestamp"),
    "transaction amount": ("transaction", "amount", "payment", "spend", "purchase", "credit", "debit"),
    "geography": ("city", "state", "country", "region", "location", "pin", "postal"),
    "assets": ("asset", "receivable", "inventory"),
    "liabilities": ("liabilit", "payable"),
    "cash flow": ("cash flow", "cashflow", "operating cash"),
}
_FINANCIAL_TERMS = {"revenue / income", "profit", "debt / exposure", "assets", "liabilities"}
_TARGET_WORDS = (
    "default", "delinquen", "bad loan", "risk flag", "risk status",
    "fraud", "churn", "target", "label", "outcome", "loan status",
    "repayment status",
)
_IDENTIFIER_WORDS = ("id", "identifier", "account", "reference", "ref", "code", "number", "uuid")
# Enough values to decide whether a column holds dates without parsing all of them.
_DATE_PROBE_ROWS = 2_000
# Model and segmentation stages are estimated from bounded samples.
_MODEL_SAMPLE_ROWS = 20_000
_ANOMALY_TRAIN_ROWS = 10_000
_ANOMALY_SCORE_ROWS = 30_000


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
    scale = 1.0
    scale_match = re.search(
        r"\s*(thousand|million|billion|lakhs?|lacs?|crores?|cr|mn|bn|k)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if scale_match:
        scale_name = scale_match.group(1).lower()
        scale = {
            "thousand": 1_000,
            "k": 1_000,
            "lakh": 100_000,
            "lakhs": 100_000,
            "lac": 100_000,
            "lacs": 100_000,
            "crore": 10_000_000,
            "crores": 10_000_000,
            "cr": 10_000_000,
            "million": 1_000_000,
            "mn": 1_000_000,
            "billion": 1_000_000_000,
            "bn": 1_000_000_000,
        }[scale_name]
        text = text[:scale_match.start()]
    text = re.sub(r"[₹$€£]", "", text)
    text = re.sub(r"(?i)\b(?:rs\.?|inr\b|usd\b|eur\b|gbp\b)", "", text)
    text = text.replace(",", "").replace("%", "").replace("(", "").replace(")", "").strip()
    match = re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return np.nan
    number = float(text)
    if negative:
        number = -abs(number)
    result = number * scale
    return result if np.isfinite(result) else np.nan


def _role_for_column(column: object, values: pd.Series) -> list[str]:
    name = _key(column)
    roles = []
    for role, patterns in _ROLE_PATTERNS.items():
        if role == "date / time":
            matched = bool(re.search(r"\b(date|time|period|year|month|timestamp)\b", name))
        elif role == "assets":
            matched = any(pattern in name for pattern in patterns) or (
                bool(re.search(r"\bcash\b", name)) and "flow" not in name
            )
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


def _looks_numeric(series: pd.Series, nonempty: int) -> pd.Series | None:
    """Fast vectorized numeric parse, falling back to text cleanup when needed."""
    try:
        converted = pd.to_numeric(series, errors="coerce")
    except (TypeError, ValueError):
        return series.map(_numeric_value)
    converted = converted.replace([np.inf, -np.inf], np.nan)
    probe = series.dropna().head(_DATE_PROBE_ROWS).astype(str)
    needs_cleanup = probe.str.contains(
        r"[₹$€£,()%]|\b(?:rs\.?|inr|usd|eur|gbp|thousand|million|billion|lakhs?|lacs?|crores?|cr|mn|bn|k)\b",
        case=False,
        regex=True,
    ).any()
    if needs_cleanup or converted.notna().sum() / nonempty < 0.8:
        return series.map(_numeric_value)
    return converted


def _parse_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    parsed = frame.copy()
    numeric_columns: list[str] = []
    date_columns: list[str] = []
    for original in parsed.columns:
        column = str(original)
        source = parsed[original]
        name = _key(column)
        if pd.api.types.is_numeric_dtype(source):
            numeric_source = pd.to_numeric(source, errors="coerce")
            if (
                re.search(r"\b(year|financial year|fiscal year|period)\b", name)
                and numeric_source.notna().any()
                and numeric_source.dropna().between(1900, 2100).all()
                and numeric_source.dropna().mod(1).eq(0).all()
            ):
                parsed[original] = pd.to_datetime(
                    numeric_source.astype("Int64").astype("string"),
                    format="%Y",
                    errors="coerce",
                )
                date_columns.append(column)
                continue
            parsed[original] = pd.to_numeric(source, errors="coerce")
            numeric_columns.append(column)
            continue
        nonempty = source.dropna()
        if nonempty.empty:
            continue
        date_hint = any(word in name for word in ("date", "time", "period", "year", "month"))
        # Date detection only needs a representative sample; parsing every value of a
        # large object column costs seconds and almost never changes the verdict.
        probe = nonempty.head(_DATE_PROBE_ROWS)
        date_like_values = probe.astype(str).str.contains(r"[-/:]", regex=True).mean() >= 0.6
        if date_hint or date_like_values:
            dates = pd.to_datetime(probe, errors="coerce", format="mixed")
            plausible = dates.dropna().between("1900-01-01", "2100-12-31").mean() if dates.notna().any() else 0
            if dates.notna().sum() / len(probe) >= 0.6 and plausible >= 0.8:
                parsed[original] = (
                    dates if len(probe) == len(nonempty)
                    else pd.to_datetime(source, errors="coerce", format="mixed")
                )
                date_columns.append(column)
                continue
        converted = _looks_numeric(source, len(nonempty))
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
    substantive_financial = role_set & {
        "profit", "debt / exposure", "assets", "liabilities", "cash flow",
    }
    if substantive_financial:
        kinds.append("Financial statement" if date_columns else "MSME/business dataset")
    elif "revenue / income" in role_set:
        kinds.append("Sales dataset")
    if "credit score" in role_set or "default / risk target" in role_set:
        kinds.append("Loan/credit dataset")
    if "customer identifier" in role_set or any(word in filename_key for word in ("customer", "client", "borrower")):
        kinds.append("Customer dataset")
    if "transaction amount" in role_set and date_columns:
        kinds.append("Transaction dataset")
    if any(word in filename_key for word in ("sale", "sales", "order")) and not kinds:
        kinds.append("Sales dataset")
    if not kinds and len(frame.columns) >= 2:
        if date_columns and numeric_columns:
            kinds.append("Time series dataset")
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
        if any(word in name for word in _IDENTIFIER_WORDS):
            continue
        unique_count = values.nunique()
        is_class_target = 1 < unique_count <= 20
        is_numeric_regression_target = (
            pd.api.types.is_numeric_dtype(frame[column])
            and unique_count > 20
        )
        is_target = (
            any(word in name for word in _TARGET_WORDS)
            or any("default / risk target" in role for role in roles.get(str(column), []))
        )
        if is_target and (is_class_target or is_numeric_regression_target):
            targets.append(str(column))
    return list(dict.fromkeys(targets))


def _financial_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate financial indicators only from recognized, observed columns."""
    normalized = {_key(column): str(column) for column in data.columns}

    def find(*names: str) -> str | None:
        return next((normalized[name] for name in names if name in normalized), None)

    columns = {
        "Revenue": find(
            "revenue", "annual revenue", "annual sales", "net sales",
            "total sales", "sales", "turnover",
        ),
        "Profit": find("net profit", "profit after tax", "profit after taxation", "pat", "net income"),
        "EBITDA": find("ebitda", "operating profit"),
        "Debt": find("total debt", "debt", "total borrowings", "borrowings", "loan balance", "loans"),
        "Assets": find("total assets", "assets"),
        "Liabilities": find("total liabilities", "liabilities"),
        "Current assets": find("current assets", "total current assets"),
        "Current liabilities": find("current liabilities", "total current liabilities"),
        "Inventory": find("inventory", "inventories", "stock"),
        "Interest expense": find("interest expense", "finance costs", "finance cost"),
        "Operating cash flow": find("operating cash flow", "cash flow from operations", "cash flow operations"),
        "Receivables": find("accounts receivable", "trade receivables", "receivables", "debtors"),
        "Equity": find("equity value", "shareholders equity", "shareholder equity", "owners equity", "net worth"),
        "Revenue growth": find("revenue growth", "sales growth"),
        "Profit margin": find("profit margin", "net profit margin"),
        "Current ratio": find("current ratio"),
        "Quick ratio": find("quick ratio"),
        "Debt-to-equity": find("debt to equity", "debt equity"),
        "Debt-to-assets": find("debt to assets", "debt assets ratio"),
        "Interest coverage": find("interest coverage"),
        "Operating margin": find("operating margin", "ebitda margin"),
    }

    def values(key: str) -> pd.Series | None:
        column = columns[key]
        if column is None:
            return None
        parsed = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return parsed if parsed.notna().any() else None

    def median(key: str, basis: str) -> tuple[float | None, str]:
        series = values(key)
        return (float(series.median()), basis) if series is not None else (None, basis)

    def ratio(
        numerator: pd.Series | None,
        denominator: pd.Series | None,
        basis: str,
    ) -> tuple[float | None, str]:
        if numerator is None or denominator is None:
            return None, basis
        aligned = pd.concat([numerator, denominator], axis=1).dropna()
        aligned = aligned.loc[aligned.iloc[:, 1] != 0]
        if aligned.empty:
            return None, basis
        return float((aligned.iloc[:, 0] / aligned.iloc[:, 1]).median()), basis

    revenue = values("Revenue")
    profit = values("Profit")
    ebitda = values("EBITDA")
    debt = values("Debt")
    assets = values("Assets")
    liabilities = values("Liabilities")
    current_assets = values("Current assets")
    current_liabilities = values("Current liabilities")
    inventory = values("Inventory")
    interest = values("Interest expense")
    equity = values("Equity")
    growth = values("Revenue growth")
    if equity is None and assets is not None and liabilities is not None:
        equity = assets - liabilities
    if equity is None and assets is not None and debt is not None:
        equity = assets - debt
    quick_assets = (
        current_assets - inventory
        if current_assets is not None and inventory is not None
        else None
    )
    if growth is None and revenue is not None:
        period_column = find("period", "date", "year", "financial year", "fiscal year")
        if period_column is not None:
            periods = pd.to_datetime(data[period_column], errors="coerce", format="mixed")
            company_column = find("company id", "company", "customer id", "borrower id")
            grouping = (
                data[company_column].astype("string").fillna("Missing")
                if company_column is not None
                else pd.Series("all", index=data.index)
            )
            growth_frame = pd.DataFrame({
                "group": grouping,
                "period": periods,
                "revenue": revenue,
            }).dropna(subset=["period", "revenue"])
            changes = []
            for _, group in growth_frame.groupby("group", sort=False):
                ordered = group.sort_values("period")
                prior = ordered["revenue"].shift(1)
                valid = prior.notna() & prior.ne(0)
                changes.extend((ordered.loc[valid, "revenue"] / prior.loc[valid] - 1).tolist())
            if changes:
                growth = pd.Series(changes, dtype="float64")

    metrics: dict[str, tuple[float | None, str]] = {
        "Revenue": median("Revenue", "Median of observed uploaded values; source units retained."),
        "Revenue growth": (
            float(growth.median()) if growth is not None else None,
            "Median of the uploaded growth field or derived period-over-period revenue change."
            if growth is not None else "A growth field or multiple dated revenue periods are required.",
        ),
        "Net profit": median("Profit", "Median of observed uploaded values; source units retained."),
        "Profit margin": (
            median("Profit margin", "Median of uploaded profit-margin field.")
            if values("Profit margin") is not None
            else ratio(profit, revenue, "Median of row-level profit / revenue; zero revenue rows omitted.")
        ),
        "Current ratio": (
            median("Current ratio", "Median of uploaded current-ratio field.")
            if values("Current ratio") is not None
            else ratio(current_assets, current_liabilities, "Median of row-level current assets / current liabilities.")
        ),
        "Quick ratio": (
            median("Quick ratio", "Median of uploaded quick-ratio field.")
            if values("Quick ratio") is not None
            else ratio(quick_assets, current_liabilities, "Median of (current assets - inventory) / current liabilities.")
        ),
        "Debt-to-equity": (
            median("Debt-to-equity", "Median of uploaded debt-to-equity field.")
            if values("Debt-to-equity") is not None
            else ratio(debt, equity, "Median of row-level debt / equity; zero equity rows omitted.")
        ),
        "Debt-to-assets": (
            median("Debt-to-assets", "Median of uploaded debt-to-assets field.")
            if values("Debt-to-assets") is not None
            else ratio(debt, assets, "Median of row-level debt / total assets; zero assets rows omitted.")
        ),
        "Interest coverage": (
            median("Interest coverage", "Median of uploaded interest-coverage field.")
            if values("Interest coverage") is not None
            else ratio(ebitda, interest, "Median of row-level EBITDA / interest expense; zero interest rows omitted.")
        ),
        "Operating margin": (
            median("Operating margin", "Median of uploaded operating-margin field.")
            if values("Operating margin") is not None
            else ratio(ebitda, revenue, "Median of row-level EBITDA / revenue; zero revenue rows omitted.")
        ),
        "Cash flow": median("Operating cash flow", "Median of observed uploaded operating cash-flow values."),
        "Receivables": median("Receivables", "Median of observed uploaded receivables values."),
        "Inventory": median("Inventory", "Median of observed uploaded inventory values."),
    }
    return pd.DataFrame([
        {
            "Metric": name,
            "Value": value if value is not None else "Not available - required data not found.",
            "Basis": basis,
        }
        for name, (value, basis) in metrics.items()
    ])


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
    usable = frame.dropna(subset=[target]).head(_MODEL_SAMPLE_ROWS).copy()
    y = usable[target]
    classification = not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10
    if len(usable) < 30:
        result["reason"] = "At least 30 labeled rows are required for a held-out evaluation."
        return result
    if y.nunique() < 2 or (classification and y.nunique() > 20):
        result["reason"] = (
            "Classification targets need 2 to 20 observed classes."
            if classification
            else "The numeric target has fewer than two observed values."
        )
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
        or (
            not pd.api.types.is_numeric_dtype(data[column])
            and data[column].nunique(dropna=True) / max(len(data), 1) >= 0.95
            and data[column].nunique(dropna=True) > 10
        )
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
    anomaly_rows_scored = 0
    if numeric and len(data) >= 10:
        anomaly_data = data[numeric].replace([np.inf, -np.inf], np.nan)
        anomaly_data = anomaly_data.loc[:, anomaly_data.notna().any()]
        if not anomaly_data.empty:
            filled = SimpleImputer(strategy="median").fit_transform(anomaly_data)
            if np.isfinite(filled).all():
                detector = IsolationForest(contamination="auto", random_state=42, n_estimators=100, n_jobs=1)
                training_rows = min(len(filled), _ANOMALY_TRAIN_ROWS)
                if training_rows < len(filled):
                    train_indices = np.random.default_rng(42).choice(
                        len(filled),
                        size=training_rows,
                        replace=False,
                    )
                    detector.fit(filled[train_indices])
                else:
                    detector.fit(filled)
                # Scoring is linear in rows; bound it and record what was scored.
                scored = filled[:_ANOMALY_SCORE_ROWS]
                scores = -detector.score_samples(scored)
                anomaly_rows_scored = len(scored)
                anomaly_scores.iloc[:len(scored)] = scores
                anomalies.iloc[:len(scored)] = detector.predict(scored) == -1
    target = targets[0] if targets else None
    segment_frame = pd.DataFrame()
    segment_columns = [
        column for column in numeric
        if data[column].nunique(dropna=True) > 1 and data[column].notna().any()
    ][:20]
    if len(segment_columns) >= 2 and len(data) >= 10:
        segment_data = data[segment_columns].replace([np.inf, -np.inf], np.nan)
        filled_segments = SimpleImputer(strategy="median").fit_transform(segment_data)
        if np.isfinite(filled_segments).all():
            scaled = StandardScaler().fit_transform(filled_segments[:_MODEL_SAMPLE_ROWS])
            cluster_count = min(4, max(2, int(np.sqrt(len(scaled) / 2))))
            labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(scaled)
            segment_frame = pd.DataFrame({
                "Segment": [f"Segment {label + 1}" for label in labels],
                "Records": 1,
            }).groupby("Segment", as_index=False).sum()
            segment_frame["Share %"] = (segment_frame["Records"] / len(labels) * 100).round(2)
            if target:
                target_share = data[target].head(len(labels)).astype("string").fillna("Missing").reset_index(drop=True)
                segment_frame["Most common target value"] = [
                    target_share[np.asarray(labels) == label].mode().iloc[0]
                    if (np.asarray(labels) == label).any() else "Not available"
                    for label in range(cluster_count)
                ]
    missing_risk = data.isna().mean(axis=1) if len(data.columns) else pd.Series(0, index=data.index)
    detected_pattern = np.select(
        [
            anomalies.to_numpy(),
            missing_risk.to_numpy() >= 0.5,
            np.arange(len(data)) >= anomaly_rows_scored,
        ],
        [
            "Unusual multivariate pattern; review required",
            "High missingness; interpretation is limited",
            "Not anomaly-scored due to the runtime row limit",
        ],
        default="No anomaly flag",
    )
    risk_results = pd.DataFrame({
        "Source row": np.arange(len(data)) + 1,
        "Missing %": (missing_risk * 100).round(2),
        "Anomaly score": anomaly_scores.round(5),
        "Anomaly": anomalies.to_numpy(),
        "Detected Risk Pattern": detected_pattern,
    })
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
        findings.append(
            f"{int(anomalies.sum()):,} of {anomaly_rows_scored:,} scored records were flagged by Isolation Forest."
            + ("" if anomaly_rows_scored >= len(data) else " Scoring was capped for runtime; unscored rows are not anomaly-free.")
        )
    if not segment_frame.empty:
        findings.append(f"Numeric patterns support exploratory segmentation into {len(segment_frame)} groups.")
    recommendations = []
    if missing_rate > 0:
        missing_columns = [
            str(column) for column in data.columns
            if data[column].isna().mean() >= 0.2
        ]
        if missing_columns:
            recommendations.append(
                "Prioritize validation or completion of high-missingness fields: "
                + ", ".join(missing_columns[:5]) + "."
            )
        else:
            recommendations.append("Review missing values before operational use; avoid assuming missing entries are zero.")
    if roles:
        detected_roles = [
            column for column, column_roles in roles.items()
            if column_roles
        ]
        recommendations.append(
            "Confirm inferred field meanings against the source data dictionary, especially: "
            + ", ".join(detected_roles[:5]) + "."
        )
    else:
        recommendations.append(
            "No domain-specific variables were confidently inferred; confirm field meanings before assigning business risk."
        )
    if anomalies.any():
        recommendations.append(
            f"Review the {int(anomalies.sum()):,} statistically unusual records and verify source values; "
            "an anomaly is not evidence of default, fraud, or misconduct."
        )
    if not trends.empty and trends["Direction"].eq("Declining").any():
        declining = trends.loc[trends["Direction"].eq("Declining"), "Variable"].astype(str).tolist()
        recommendations.append(
            "Investigate observed declines in " + ", ".join(declining[:5])
            + " and verify that periods and measurement units are comparable."
        )
    if not high_risk_groups_frame.empty:
        first_group = high_risk_groups_frame.iloc[0]
        recommendations.append(
            f"Review the observed {first_group['Variable']} segment '{first_group['Group']}' "
            f"({first_group['Observed risk pattern %']:.1f}% of labeled records match the detected risk label); "
            "this is a historical pattern, not a prediction."
        )
    if not correlations_frame.empty and correlations_frame.iloc[0]["Absolute correlation"] >= 0.9:
        top_pair = correlations_frame.iloc[0]
        recommendations.append(
            f"Check {top_pair['Variable A']} and {top_pair['Variable B']} for redundant or overlapping measurements "
            "before interpreting model drivers."
        )
    target_proportions = data[target].value_counts(normalize=True, dropna=True) if target else pd.Series(dtype=float)
    if len(target_proportions) > 1:
        if target_proportions.iloc[0] >= 0.9:
            recommendations.append(
                f"Target '{target}' is imbalanced ({target_proportions.iloc[0]:.1%} in its most common class); "
                "review class-wise metrics and representativeness."
            )
    if target and model_result["status"] == "evaluated":
        recommendations.append(
            "Treat supervised results as exploratory hold-out metrics; validate on representative future data "
            "before using predictions for decisions."
        )
    elif not target:
        recommendations.append(
            "No explicit default target was detected. Use anomaly and statistical risk-pattern findings "
            "for review; no predicted risk probability is available."
        )
    warning_details: list[dict[str, str]] = []
    if missing_rate >= 0.2:
        missing_columns = data.isna().mean().sort_values(ascending=False)
        highest_missing = [
            f"{column}: {data[column].isna().mean():.1%}"
            for column in missing_columns.index[:3]
            if data[column].isna().any()
        ]
        warning_details.append({
            "Severity": "High" if missing_rate >= 0.4 else "Medium",
            "Evidence": f"{missing_rate:.1%} of all cells are missing; " + ", ".join(highest_missing),
            "Reason": "High or concentrated missingness can bias summaries and risk signals.",
            "Recommended action": "Validate source completeness and document missing-value handling before relying on analysis.",
        })
    for _, trend in trends.iterrows():
        change = trend["Change %"]
        if trend["Direction"] == "Declining" and pd.notna(change):
            warning_details.append({
                "Severity": "High" if change <= -20 else "Medium",
                "Evidence": (
                    f"{trend['Variable']} changed from {_numeric_value(trend['First value']):,.4g} "
                    f"to {_numeric_value(trend['Latest value']):,.4g} ({change:.1f}%) "
                    f"between {trend['From']} and {trend['To']}."
                ),
                "Reason": "Observed values declined over the available time periods.",
                "Recommended action": "Verify period alignment and units, then investigate the underlying operating or reporting change.",
            })
    if anomalies.any():
        anomaly_share = float(anomalies.sum() / max(anomaly_rows_scored, 1))
        warning_details.append({
            "Severity": "High" if anomaly_share >= 0.1 else "Medium",
            "Evidence": (
                f"Isolation Forest flagged {int(anomalies.sum()):,} of "
                f"{anomaly_rows_scored:,} scored records ({anomaly_share:.1%})."
            ),
            "Reason": "The flagged rows have unusual multivariate patterns relative to this uploaded dataset.",
            "Recommended action": "Review the flagged rows and verify their source values; unusualness alone is not evidence of default or fraud.",
        })
    for column in numeric:
        name = _key(column)
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if not len(values):
            continue
        if any(term in name for term in ("profit", "earnings", "pat", "cash flow", "cashflow")):
            negative_count = int((values < 0).sum())
            if negative_count:
                warning_details.append({
                    "Severity": "High" if negative_count / len(values) >= 0.25 else "Medium",
                    "Evidence": f"{negative_count:,} of {len(values):,} observed values in '{column}' are negative.",
                    "Reason": "Negative profit or operating cash-flow values can indicate financial pressure.",
                    "Recommended action": "Confirm sign conventions and review the affected periods or records with supporting statements.",
                })
        if "current ratio" in name:
            weak_count = int((values < 1).sum())
            if weak_count:
                warning_details.append({
                    "Severity": "High" if weak_count / len(values) >= 0.25 else "Medium",
                    "Evidence": f"{weak_count:,} of {len(values):,} observed values in '{column}' are below 1.0.",
                    "Reason": "Current assets are below current liabilities where this ratio is below 1.",
                    "Recommended action": "Review near-term liquidity and confirm the ratio's numerator and denominator definitions.",
                })
        if "debt to asset" in name or "debt asset ratio" in name:
            elevated_count = int((values >= 0.65).sum())
            if elevated_count:
                warning_details.append({
                    "Severity": "High" if elevated_count / len(values) >= 0.25 else "Medium",
                    "Evidence": f"{elevated_count:,} of {len(values):,} observed values in '{column}' are at least 0.65.",
                    "Reason": "The reported debt-to-asset ratio indicates elevated leverage.",
                    "Recommended action": "Verify debt and asset scope and review the affected records' repayment capacity.",
                })
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
        "financial_metrics": _financial_metrics(data),
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
        "anomaly_rows_scored": anomaly_rows_scored,
        "anomaly_rows_total": int(len(data)),
        "model": model_result,
        "analysis_mode": "Automatic Dataset Intelligence",
        "model_evaluation_available": model_result["status"] == "evaluated",
        "analysis_result_title": (
            "Supervised Model Evaluation" if model_result["status"] == "evaluated"
            else "Anomaly Analysis" if not target
            else "Risk Pattern Analysis"
        ),
        "alternative_analysis": (
            "Supervised model evaluation with hold-out metrics"
            if model_result["status"] == "evaluated"
            else "Anomaly + Statistical Risk Pattern Detection"
        ),
        "risk_prediction_summary": (
            "Model evaluated; see hold-out metrics below. No calibrated probability of default is reported."
            if model_result["status"] == "evaluated"
            else "Not available without a target."
            if not target
            else "Not available; the detected target could not be evaluated."
        ),
        "findings": findings,
        "early_warnings": [
            f"{row['Variable']} is declining across observed periods."
            for _, row in trends.loc[trends["Direction"].eq("Declining")].iterrows()
        ] + (
            [f"{anomalies.sum():,} records have unusual multivariate patterns."]
            if anomalies.any() else []
        ),
        "warning_details": warning_details,
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
                else (
                    "No explicit default target was found. Anomaly and risk-pattern analysis was performed instead."
                    if not target else
                    "A target was detected, but the available data did not support a reliable held-out evaluation. "
                    "Anomaly and risk-pattern analysis was performed instead."
                )
            )
        ),
    }
