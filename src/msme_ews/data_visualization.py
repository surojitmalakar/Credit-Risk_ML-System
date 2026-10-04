"""Dynamic Plotly charts built only from observed upload data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


_CHART_COLORS = ["#2563EB", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
_FINANCIAL_FIELDS = {
    "revenue": ("revenue", "sales", "turnover", "income"),
    "profit": ("profit", "pat", "net income", "earnings"),
    "debt": ("debt", "borrow", "loan balance", "outstanding"),
    "assets": ("asset",),
    "liabilities": ("liabilit",),
    "cash flow": ("cash flow", "cashflow", "operating cash"),
    "current assets": ("current_assets", "current assets"),
    "current liabilities": ("current_liabilities", "current liabilities"),
}
_ID_TERMS = ("id", "identifier", "reference", "account", "uuid", "code", "number", "ref")


def _meaningful_categories(analysis: dict[str, Any]) -> list[str]:
    data = analysis["data"]
    identifiers = set(analysis["identifiers"])
    candidates = [
        column for column in analysis["categorical_columns"]
        if column not in identifiers
        and 1 < data[column].nunique(dropna=True) <= 20
        and not any(term in str(column).lower().split() for term in _ID_TERMS)
    ]

    def priority(column: str) -> tuple[int, int, int]:
        name = column.lower()
        if "segment" in name or "category" in name or "group" in name:
            semantic_rank = 0
        elif any(word in name for word in ("state", "city", "region", "country", "location", "postal")):
            semantic_rank = 1
        elif any(word in name for word in ("gender", "status", "type", "channel", "source")):
            semantic_rank = 2
        else:
            semantic_rank = 3
        return (semantic_rank, data[column].nunique(dropna=True), -data[column].notna().sum())

    return sorted(candidates, key=priority)


def _meaningful_numeric(analysis: dict[str, Any]) -> list[str]:
    data = analysis["data"]
    identifiers = set(analysis["identifiers"])
    candidates = [
        column for column in analysis["numeric_columns"]
        if column in data
        and data[column].nunique(dropna=True) > 1
        and not any(term in str(column).lower().split() for term in _ID_TERMS)
        and column not in {"anomaly_score"}
    ]

    def priority(column: str) -> tuple[int, float]:
        roles = " ".join(analysis["roles"].get(column, [])).lower()
        name = column.lower()
        if "acquisition" in name or "cost" in name:
            importance_rank = 0
        elif "age" in name:
            importance_rank = 1
        elif any(term in roles or term in name for term in (
            "revenue / income", "profit", "debt / exposure", "credit score",
            "assets", "liabilities", "transaction amount", "income", "balance", "amount",
        )):
            importance_rank = 2
        else:
            importance_rank = 3
        unique_ratio = data[column].nunique(dropna=True) / max(len(data), 1)
        return (importance_rank + int(column in identifiers) * 10, -unique_ratio)

    return sorted(candidates, key=priority)


def _layout(figure: go.Figure, height: int = 310) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=12, r=12, t=48, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    figure.update_xaxes(showgrid=False, automargin=True)
    figure.update_yaxes(gridcolor="rgba(148,163,184,0.14)", automargin=True)
    return figure


def build_data_visualizations(analysis: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return chart groups selected from columns and statistics in the uploaded data."""
    data = analysis["data"]
    charts: dict[str, list[dict[str, Any]]] = {
        "Distributions": [],
        "Relationships": [],
        "Time Trends": [],
        "Risk and Anomalies": [],
        "Financial Analysis": [],
    }
    numeric = _meaningful_numeric(analysis)
    categories = _meaningful_categories(analysis)
    visual_data = data
    sample_note = ""
    if len(data) > 50_000:
        visual_data = data.sample(50_000, random_state=42)
        sample_note = f" (visual sample: 50,000 of {len(data):,} records)"

    for column in numeric[:4]:
        values = pd.to_numeric(visual_data[column], errors="coerce").dropna()
        if values.empty:
            continue
        figure = px.histogram(
            visual_data, x=column, nbins=min(32, max(8, int(np.sqrt(len(visual_data))))),
            title=f"Distribution: {column}{sample_note}", color_discrete_sequence=[_CHART_COLORS[0]],
        )
        figure.update_traces(hovertemplate=f"{column}: %{{x}}<br>Records: %{{y}}<extra></extra>")
        charts["Distributions"].append({"title": f"Distribution: {column}", "figure": _layout(figure, 280)})

    numeric_for_corr = [
        column for column in numeric
        if pd.to_numeric(data[column], errors="coerce").notna().sum() >= 3
    ][:12]
    if len(numeric_for_corr) >= 2:
        correlation = data[numeric_for_corr].corr(min_periods=3)
        correlation = correlation.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if correlation.shape[0] >= 2:
            figure = go.Figure(go.Heatmap(
                z=correlation.to_numpy(),
                x=correlation.columns.astype(str),
                y=correlation.index.astype(str),
                zmin=-1,
                zmax=1,
                colorscale=[[0, "#EF4444"], [0.5, "#111C2B"], [1, "#06B6D4"]],
                colorbar=dict(title="Correlation"),
                hovertemplate="%{y} × %{x}<br>Correlation: %{z:.2f}<extra></extra>",
            ))
            charts["Relationships"].append({
                "title": "Numeric Correlation Heatmap",
                "figure": _layout(figure, max(300, min(500, 35 * len(correlation) + 110))),
            })

    for column in categories[:4]:
        counts = data[column].astype("string").fillna("Missing").value_counts().head(12).sort_values()
        if counts.empty:
            continue
        figure = go.Figure(go.Bar(
            x=counts.values,
            y=counts.index.astype(str),
            orientation="h",
            marker_color=_CHART_COLORS[1],
            hovertemplate=f"{column}: %{{y}}<br>Records: %{{x}}<extra></extra>",
        ))
        figure.update_layout(title=f"Records by {column}", xaxis_title="Records", yaxis_title=column)
        charts["Distributions"].append({"title": f"Records by {column}", "figure": _layout(figure, 280)})

    comparison_metric = next(
        (
            column for column in numeric
            if any(word in column.lower() for word in ("acquisition", "cost"))
        ),
        next(
            (
                column for column in numeric
                if any(word in column.lower() for word in ("age", "income", "revenue", "sales", "loan", "amount", "balance"))
            ),
            numeric[0] if numeric else None,
        ),
    )
    if comparison_metric:
        for category in categories[:3]:
            grouped = data[[category, comparison_metric]].copy()
            grouped[comparison_metric] = pd.to_numeric(grouped[comparison_metric], errors="coerce")
            grouped[category] = grouped[category].astype("string").fillna("Missing")
            grouped = grouped.dropna(subset=[comparison_metric])
            if grouped[category].nunique() > 12 or grouped.empty:
                continue
            comparison = grouped.groupby(category)[comparison_metric].median().sort_values(ascending=False).head(12).reset_index()
            figure = px.bar(
                comparison,
                x=category,
                y=comparison_metric,
                title=f"Median {comparison_metric} by {category}",
                color=category,
                color_discrete_sequence=_CHART_COLORS,
            )
            charts["Relationships"].append({
                "title": f"{comparison_metric} by {category}",
                "figure": _layout(figure, 300),
            })

    if numeric:
        best_pair: tuple[str, str] | None = None
        best_correlation = -1.0
        candidate_numeric = numeric[:10]
        corr = data[candidate_numeric].corr(min_periods=4).abs()
        for index, left in enumerate(candidate_numeric):
            for right in candidate_numeric[index + 1:]:
                value = corr.loc[left, right]
                if pd.notna(value) and value > best_correlation:
                    best_pair = (left, right)
                    best_correlation = float(value)
        if best_pair:
            x_column, y_column = best_pair
            sample = data.loc[:, [x_column, y_column]].replace([np.inf, -np.inf], np.nan).dropna()
            scatter_note = ""
            if len(sample) > 5_000:
                sample = sample.sample(5_000, random_state=42)
                scatter_note = f" (visual sample: 5,000 of {len(data):,} records)"
            if len(sample) >= 3:
                figure = px.scatter(
                    sample,
                    x=x_column,
                    y=y_column,
                    title=f"{y_column} vs {x_column}{scatter_note}",
                    opacity=0.7,
                    color_discrete_sequence=[_CHART_COLORS[2]],
                )
                figure.update_traces(marker=dict(size=7))
                charts["Relationships"].append({
                    "title": f"{y_column} vs {x_column}",
                    "figure": _layout(figure, 320),
                })

    date_columns = analysis["date_columns"]
    if date_columns and numeric:
        date_column = date_columns[0]
        for column in numeric[:4]:
            trend = pd.DataFrame({
                "Date": pd.to_datetime(data[date_column], errors="coerce"),
                column: pd.to_numeric(data[column], errors="coerce"),
            }).dropna()
            if trend.empty or trend["Date"].nunique() < 2:
                continue
            trend = trend.groupby("Date", as_index=False)[column].median().sort_values("Date")
            figure = px.line(trend, x="Date", y=column, markers=True, title=f"{column} over time")
            figure.update_traces(line_color=_CHART_COLORS[2])
            charts["Time Trends"].append({"title": f"{column} over time", "figure": _layout(figure, 290)})

    risk_results = analysis.get("risk_results", pd.DataFrame())
    if not risk_results.empty and "Anomaly" in risk_results and risk_results["Anomaly"].any():
        counts = risk_results["Anomaly"].map({False: "Typical pattern", True: "Anomaly"}).value_counts()
        figure = go.Figure(go.Bar(
            x=counts.index.astype(str),
            y=counts.values,
            marker_color=[_CHART_COLORS[2] if "Typical" in str(label) else _CHART_COLORS[4] for label in counts.index],
            hovertemplate="%{x}<br>Records: %{y}<extra></extra>",
        ))
        figure.update_layout(title="Anomaly Analysis", xaxis_title="Statistical pattern", yaxis_title="Records")
        charts["Risk and Anomalies"].append({"title": "Typical vs anomalous records", "figure": _layout(figure, 280)})

        scatter_columns = numeric[:2]
        if len(scatter_columns) >= 2 and "detected_anomaly" in data:
            scatter = visual_data.loc[:, [*scatter_columns, "detected_anomaly"]].dropna()
            scatter_note = ""
            if len(scatter) > 5_000:
                scatter = scatter.sample(5_000, random_state=42)
                scatter_note = f" (visual sample: 5,000 of {len(data):,} records)"
            figure = px.scatter(
                scatter,
                x=scatter_columns[0],
                y=scatter_columns[1],
                color="detected_anomaly",
                color_discrete_map={"False": _CHART_COLORS[2], "True": _CHART_COLORS[4]},
                title=f"Anomalies across numeric variables{scatter_note}",
                labels={"detected_anomaly": "Anomaly"},
            )
            charts["Risk and Anomalies"].append({"title": "Anomaly scatter", "figure": _layout(figure, 320)})

    segments = analysis.get("segments", pd.DataFrame())
    if not segments.empty and {"Segment", "Records"} <= set(segments.columns):
        figure = px.bar(
            segments,
            x="Segment",
            y="Records",
            color="Segment",
            title="Exploratory Numeric Segments",
            color_discrete_sequence=_CHART_COLORS,
        )
        charts["Risk and Anomalies"].append({"title": "Exploratory segments", "figure": _layout(figure, 280)})

    role_names = {
        column: " ".join(analysis["roles"].get(column, [])).lower()
        for column in numeric
    }
    financial_columns: dict[str, str] = {}
    for label, terms in _FINANCIAL_FIELDS.items():
        match = next(
            (
                column for column in numeric
                if any(term in column.lower().replace("_", " ") or term in role_names[column] for term in terms)
            ),
            None,
        )
        if match:
            financial_columns[label] = match
    for field, prefix in (("assets", "Total_Assets"), ("liabilities", "Total_Liabilities")):
        total_name = prefix.lower()
        total_column = next((column for column in numeric if column.lower() == total_name), None)
        if total_column:
            financial_columns[field] = total_column
    financial_trends = [label for label in ("revenue", "profit", "debt", "cash flow") if label in financial_columns]
    if date_columns:
        date_column = date_columns[0]
        for label in financial_trends:
            column = financial_columns[label]
            trend = pd.DataFrame({
                "Date": pd.to_datetime(data[date_column], errors="coerce"),
                column: pd.to_numeric(data[column], errors="coerce"),
            }).dropna()
            if trend["Date"].nunique() >= 2:
                trend = trend.groupby("Date", as_index=False)[column].median().sort_values("Date")
                figure = px.line(trend, x="Date", y=column, markers=True, title=f"{label.title()} Trend")
                figure.update_traces(line_color=_CHART_COLORS[financial_trends.index(label) % len(_CHART_COLORS)])
                charts["Financial Analysis"].append({"title": f"{label.title()} trend", "figure": _layout(figure, 290)})
    if "assets" in financial_columns and "liabilities" in financial_columns:
        rows = [
            {"Measure": "Assets", "Value": pd.to_numeric(data[financial_columns["assets"]], errors="coerce").sum()},
            {"Measure": "Liabilities", "Value": pd.to_numeric(data[financial_columns["liabilities"]], errors="coerce").sum()},
        ]
        figure = px.bar(
            pd.DataFrame(rows), x="Measure", y="Value", color="Measure",
            title="Assets vs Liabilities", color_discrete_sequence=[_CHART_COLORS[2], _CHART_COLORS[4]],
        )
        charts["Financial Analysis"].append({"title": "Assets vs liabilities", "figure": _layout(figure, 280)})

    if "current assets" in financial_columns and "current liabilities" in financial_columns:
        current_assets = pd.to_numeric(data[financial_columns["current assets"]], errors="coerce")
        current_liabilities = pd.to_numeric(data[financial_columns["current liabilities"]], errors="coerce")
        ratio = current_assets / current_liabilities.replace(0, np.nan)
        if ratio.notna().any():
            figure = px.histogram(
                pd.DataFrame({"Current ratio": ratio.dropna()}),
                x="Current ratio",
                title="Current Ratio",
                nbins=24,
                color_discrete_sequence=[_CHART_COLORS[1]],
            )
            charts["Financial Analysis"].append({"title": "Current ratio", "figure": _layout(figure, 280)})
    if "debt / exposure" in role_names.values():
        debt_column = financial_columns.get("debt")
        equity_column = next((column for column in numeric if "equity" in column.lower()), None)
        if debt_column and equity_column:
            debt_equity = pd.to_numeric(data[debt_column], errors="coerce") / pd.to_numeric(data[equity_column], errors="coerce").replace(0, np.nan)
            if debt_equity.notna().any():
                figure = px.histogram(
                    pd.DataFrame({"Debt-to-equity": debt_equity.dropna()}),
                    x="Debt-to-equity",
                    title="Debt-to-Equity",
                    nbins=24,
                    color_discrete_sequence=[_CHART_COLORS[3]],
                )
                charts["Financial Analysis"].append({"title": "Debt-to-equity", "figure": _layout(figure, 280)})

    return {section: items for section, items in charts.items() if items}
