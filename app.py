"""Eight-page Streamlit research dashboard for MSME financial distress."""

from __future__ import annotations

import sys
import hashlib
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from html import escape

from msme_ews.data import FINANCIAL_COLUMNS, validate_financial_data
from msme_ews.credit_assessment import apply_scenario_adjustments, generate_risk_interpretation
from msme_ews.copilot import (
    build_credit_context,
    generate_credit_copilot_response,
    suggested_questions,
)
from msme_ews.data_intelligence import analyze_dataset
from msme_ews.data_visualization import build_data_visualizations
from msme_ews.demo import make_demo_data
from msme_ews.documents import extract_financial_document
from msme_ews.early_warning import early_warning_indicators, trend_data
from msme_ews.explain import explain_prediction, global_importance
from msme_ews.financial_analysis import (
    analyze_financials,
    has_sufficient_ml_data,
    recommendations_for,
    rule_based_assessment,
)
from msme_ews.features import engineer_features
from msme_ews.modeling import train_models
from msme_ews.monitoring import (
    build_snapshot,
    drift_report,
    monitoring_alerts,
    prediction_stability,
    snapshot_table,
)
from msme_ews.portfolio import analyze_credit_portfolio
from msme_ews.prediction import DEFAULT_MODEL_PATH, predict_financial_health
from msme_ews.reports import (
    create_credit_assessment_pdf,
    create_data_intelligence_excel,
    create_data_intelligence_pdf,
    create_excel_analysis,
)
from msme_ews.screening import (
    banded_exposure,
    detect_exposure_column,
    record_snapshot,
    score_records,
    screening_summary,
)
from msme_ews.table_tools import (
    PAGE_SIZE_OPTIONS,
    apply_column_filters,
    apply_search,
    apply_sort,
    filter_candidates,
    page_slice,
    search_blob,
)

st.set_page_config(
    page_title="CREDIT RISK AI",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="auto",
)


def _validate_upload_cache_key(
    filename: str,
    revision: int,
    content_signature: str,
) -> None:
    if not filename or revision < 0 or len(content_signature) != 64:
        raise ValueError("Invalid upload cache identity.")


_HASH_CHUNK_BYTES = 1 << 20
_DATASET_REGISTRY_LIMIT = 3
_DATASET_REGISTRY: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}


def _register_dataset(
    dataset_key: str,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    flags: pd.DataFrame,
) -> None:
    """Hold a dataset in memory so cached work can key on a short string.

    Passing a DataFrame to ``st.cache_data`` re-hashes every cell on each rerun;
    keying on this identifier keeps repeated interaction work O(1) in the dataset.
    """
    _DATASET_REGISTRY[dataset_key] = (frame, features, flags)
    while len(_DATASET_REGISTRY) > _DATASET_REGISTRY_LIMIT:
        _DATASET_REGISTRY.pop(next(iter(_DATASET_REGISTRY)))


def _dataset_parts(dataset_key: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        return _DATASET_REGISTRY[dataset_key]
    except KeyError:
        raise RuntimeError(
            "The selected dataset is no longer loaded; upload the file again."
        ) from None


def _content_signature(uploaded) -> str:
    """Hash upload bytes in chunks so a large file is never copied twice."""
    digest = hashlib.sha256()
    view = uploaded.getbuffer()
    for start in range(0, len(view), _HASH_CHUNK_BYTES):
        digest.update(view[start:start + _HASH_CHUNK_BYTES])
    return digest.hexdigest()


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_document_extraction(
    _content: bytes,
    filename: str,
    revision: int,
    content_signature: str,
):
    _validate_upload_cache_key(filename, revision, content_signature)
    return extract_financial_document(_content, filename)


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_dataset_profile(
    _frame: pd.DataFrame,
    filename: str,
    revision: int,
    content_signature: str,
):
    _validate_upload_cache_key(filename, revision, content_signature)
    intelligence = analyze_dataset(_frame, filename)
    financial_columns = [
        column for column in FINANCIAL_COLUMNS
        if column in _frame
        and pd.to_numeric(_frame[column], errors="coerce").notna().any()
    ]
    return intelligence, financial_columns


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_upload_visualizations(
    _intelligence: dict,
    filename: str,
    revision: int,
    content_signature: str,
):
    _validate_upload_cache_key(filename, revision, content_signature)
    return build_data_visualizations(_intelligence)


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_intelligence_pdf(
    _intelligence: dict,
    filename: str,
    revision: int,
    content_signature: str,
):
    _validate_upload_cache_key(filename, revision, content_signature)
    return create_data_intelligence_pdf(_intelligence, filename)


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_intelligence_excel(
    _intelligence: dict,
    filename: str,
    revision: int,
    content_signature: str,
):
    _validate_upload_cache_key(filename, revision, content_signature)
    return create_data_intelligence_excel(_intelligence)


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_financial_frame(
    _frame: pd.DataFrame,
    filename: str,
    revision: int,
    content_signature: str,
):
    _validate_upload_cache_key(filename, revision, content_signature)
    return validate_financial_data(_frame)


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_financial_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return engineer_features(frame), early_warning_indicators(frame)


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_screening_scores(dataset_key: str, bundle_signature: str) -> pd.DataFrame:
    frame, features, flags = _dataset_parts(dataset_key)
    return score_records(frame, features, flags, bundle=get_bundle())


@st.cache_data(show_spinner=False, max_entries=128)
def _cached_financial_analysis(dataset_key: str, row_index: int) -> dict:
    frame, features, _ = _dataset_parts(dataset_key)
    return analyze_financials(frame, row_index, features)


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_search_blob(dataset_key: str, source: str = "dataset") -> pd.Series:
    if source == "screening":
        return search_blob(_cached_screening_scores(dataset_key, _bundle_signature(get_bundle())))
    frame, _, _ = _dataset_parts(dataset_key)
    return search_blob(frame)


def _bundle_signature(bundle: dict) -> str:
    """Content fingerprint so cached screening scores invalidate on a new model."""
    report = bundle.get("report", {})
    payload = repr([
        list(bundle.get("features", [])),
        report.get("selected_model"),
        report.get("selection_metric"),
        report.get("test_metrics"),
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _record_monitor_snapshot(
    label: str,
    lineage: str,
    dataset_key: str,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    analysis: dict | None,
) -> None:
    """Keep one comparable snapshot per analysis revision for the monitoring page.

    History is scoped to one dataset lineage, so revisions of an uploaded file are
    never compared against unrelated demo data or a different upload.
    Screened risk scores are attached lazily by the monitoring page so ordinary
    page views do not pay for a full screening pass.
    """
    if st.session_state.get("_monitor_lineage") != lineage:
        st.session_state["_monitor_lineage"] = lineage
        st.session_state["_monitor_history"] = []
    snapshot = build_snapshot(label, frame, features, analysis)
    snapshot["_dataset_key"] = dataset_key
    history = st.session_state.setdefault("_monitor_history", [])
    history[:] = [entry for entry in history if entry["label"] != label]
    history.append(snapshot)
    del history[:-12]


def _snapshot_risk_scores(snapshot: dict, bundle: dict) -> pd.Series:
    key = snapshot.get("_dataset_key")
    if not key:
        return pd.Series(dtype="float64")
    try:
        scores = _cached_screening_scores(key, _bundle_signature(bundle))
    except RuntimeError:
        return pd.Series(dtype="float64")
    return scores["Distress probability"]


def _session_shap(cache_key: str, bundle: dict, selected: pd.DataFrame, features: pd.DataFrame) -> dict:
    cache = st.session_state.setdefault("_local_shap_cache", {})
    if cache_key not in cache:
        cache[cache_key] = explain_prediction(bundle, selected, features=features)
        while len(cache) > 8:
            cache.pop(next(iter(cache)))
    return cache[cache_key]


def _session_global_shap(cache_key: str, bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    cache = st.session_state.setdefault("_global_shap_cache", {})
    if cache_key not in cache:
        cache[cache_key] = global_importance(bundle, frame)
        while len(cache) > 4:
            cache.pop(next(iter(cache)))
    return cache[cache_key]


def _render_data_intelligence(
    document_result,
    intelligence: dict,
    filename: str,
    revision: int,
    content_signature: str,
    financial_columns: list[str],
) -> None:
    frame = document_result.frame
    st.success("File loaded · Analysis complete")
    st.markdown("<div class='panel-header'><h3>AI Data Detective</h3></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Automatic Dataset Intelligence**")
        summary_cols = st.columns(4)
        summary_cols[0].metric("Dataset", intelligence["dataset_type"])
        summary_cols[1].metric("Records", f"{len(frame):,}")
        summary_cols[2].metric("Variables", f"{len(frame.columns):,}")
        summary_cols[3].metric("Data quality", f"{intelligence['data_quality_percent']:.1f}%")
        st.caption(intelligence["executive_summary"])

    overview_cols = st.columns(4)
    overview_cols[0].metric("Missing values", f"{int(frame.isna().sum().sum()):,}")
    overview_cols[1].metric("Duplicate records", f"{intelligence['duplicate_count']:,}")
    overview_cols[2].metric("Numeric variables", len(intelligence["numeric_columns"]))
    overview_cols[3].metric("Categorical variables", len(intelligence["categorical_columns"]))

    with st.expander("Data Profile and Detected Variables"):
        st.dataframe(intelligence["profile"], width="stretch", hide_index=True)
        if intelligence["identifiers"]:
            st.caption("Potential identifiers: " + ", ".join(intelligence["identifiers"]))
        if intelligence["targets"]:
            st.caption("Potential target variables: " + ", ".join(intelligence["targets"]))
        st.caption(
            f"Numeric: {', '.join(intelligence['numeric_columns']) or 'none'} | "
            f"Categorical: {', '.join(intelligence['categorical_columns']) or 'none'} | "
            f"Date/time: {', '.join(intelligence['date_columns']) or 'none'}"
        )

    show_risk_key = f"_show_dataset_risk_{content_signature}"
    if st.button("Show Risk Analysis", key=f"show_dataset_risk_{content_signature}"):
        st.session_state[show_risk_key] = True
    if st.session_state.get(show_risk_key):
        st.markdown("<div class='panel-header'><h3>Risk and Anomaly Analysis</h3></div>", unsafe_allow_html=True)
        st.caption(
            "Predicted Risk is supervised model output. Anomaly is an unusual statistical pattern. "
            "Risk Indicator is a data-derived business pattern. None alone establishes default, fraud, or misconduct."
        )
        st.markdown(f"**{intelligence['analysis_result_title']}**")
        if intelligence["prediction_available"]:
            model = intelligence["model"]
            metrics = st.columns(3)
            metrics[0].metric("Model", model["model"])
            metrics[1].metric("Target", model["target"])
            metrics[2].metric("Evaluation records", f"{model['test_rows']:,}")
            if "accuracy" in model:
                st.caption(
                    f"Accuracy {model['accuracy']:.1%} · "
                    f"Balanced accuracy {model['balanced_accuracy']:.1%}"
                )
            else:
                st.caption(f"MAE {model['mae']:.3g} · R² {model['r2']:.3f}")
            if model.get("top_features"):
                st.dataframe(
                    pd.DataFrame(model["top_features"]).rename(columns={
                        "feature": "Feature",
                        "importance": "Relative importance",
                    }),
                    width="stretch",
                    hide_index=True,
                )
        elif not intelligence["targets"]:
            st.info("No explicit default target was found. Anomaly and risk-pattern analysis was performed instead.")
        else:
            st.info("A target was detected, but a reliable held-out model evaluation was not available.")
        st.dataframe(intelligence["risk_results"].head(100), width="stretch", hide_index=True)
        if not intelligence["findings"]:
            st.info("No material risk patterns were detected in the available data.")
        else:
            st.markdown("**Key findings**")
            for finding in intelligence["findings"]:
                st.write(f"- {finding}")
        if not intelligence["correlations"].empty:
            st.dataframe(intelligence["correlations"], width="stretch", hide_index=True)
        if not intelligence["trends"].empty:
            st.dataframe(intelligence["trends"], width="stretch", hide_index=True)
        if not intelligence["high_risk_groups"].empty:
            st.dataframe(intelligence["high_risk_groups"], width="stretch", hide_index=True)
        if not intelligence["concentration"].empty:
            st.dataframe(intelligence["concentration"], width="stretch", hide_index=True)
        st.markdown("**Recommendations**")
        for recommendation in intelligence["recommendations"]:
            st.write(f"- {recommendation}")
        st.markdown("**Early warnings**")
        for warning in intelligence["early_warnings"]:
            st.write(f"- {warning}")

    st.markdown("<div class='panel-header'><h3>Visual Analytics</h3></div>", unsafe_allow_html=True)
    show_charts_key = f"_show_dataset_charts_{content_signature}"
    if st.button("Load Visual Analytics", key=f"load_dataset_charts_{content_signature}"):
        st.session_state[show_charts_key] = True
    if st.session_state.get(show_charts_key):
        try:
            charts = _cached_upload_visualizations(
                intelligence,
                filename,
                revision,
                content_signature,
            )
            chart_items = [
                (section, item)
                for section, items in charts.items()
                for item in items
            ]
            if not chart_items:
                st.info("No charts could be generated from the available non-constant values.")
            else:
                st.caption(f"Showing up to 6 of {len(chart_items)} relevant charts.")
                for index, (section, item) in enumerate(chart_items[:6]):
                    st.markdown(f"**{section} · {item['title']}**")
                    st.plotly_chart(
                        item["figure"],
                        width="stretch",
                        config={"displayModeBar": False},
                        key=f"data_intelligence_{content_signature}_{index}",
                    )
        except Exception as error:
            st.warning(f"Visual analytics are unavailable for this dataset: {error}")

    st.markdown("<div class='panel-header'><h3>Reports</h3></div>", unsafe_allow_html=True)
    report_key = (content_signature, revision, filename)
    report_cache = st.session_state.setdefault("_intelligence_report_cache", {})
    report = report_cache.setdefault(report_key, {})
    pdf_button, excel_button = st.columns(2)
    with pdf_button:
        if st.button("Generate PDF Report", key=f"generate_intelligence_pdf_{content_signature}"):
            try:
                report["pdf"] = _cached_intelligence_pdf(
                    intelligence,
                    filename,
                    revision,
                    content_signature,
                )
            except Exception as error:
                st.warning(f"PDF report generation failed: {error}")
        if "pdf" in report:
            st.download_button(
                "Download PDF Report",
                data=report["pdf"],
                file_name=f"{Path(filename).stem}_data_intelligence.pdf",
                mime="application/pdf",
                key=f"download_intelligence_pdf_{content_signature}",
            )
    with excel_button:
        if st.button("Prepare Excel Analysis", key=f"generate_intelligence_excel_{content_signature}"):
            try:
                report["excel"] = _cached_intelligence_excel(
                    intelligence,
                    filename,
                    revision,
                    content_signature,
                )
            except Exception as error:
                st.warning(f"Excel report generation failed: {error}")
        if "excel" in report:
            st.download_button(
                "Download Excel Analysis",
                data=report["excel"],
                file_name=f"{Path(filename).stem}_data_intelligence.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_intelligence_excel_{content_signature}",
            )
    if not financial_columns:
        st.info(
            "No canonical financial-statement fields were detected. The general dataset analysis "
            "remains available; no financial-model probability is fabricated."
        )
    for warning in document_result.warnings:
        st.caption(warning)


st.markdown("""
<style>
:root {
    color-scheme: dark;
    --page:#07111F;
    --panel:#0D1B2A;
    --sidebar:#0B1524;
    --primary:#2563EB;
    --primary-soft:#1E40AF;
    --cyan:#06B6D4;
    --healthy:#10B981;
    --warning:#F59E0B;
    --danger:#EF4444;
    --text:#F8FAFC;
    --muted:#94A3B8;
    --input:#111C2B;
    --input-hover:#16263A;
    --input-border:#26384D;
    --placeholder:#64748B;
    --border:#1E293B;
    --shadow:rgba(2, 6, 23, 0.16);
}
html, body, [class*="css"] { font-family:'Inter','Manrope',sans-serif; color:var(--text); }
html, body { width:100%; max-width:100%; overflow-x:hidden; }
.stApp, [data-testid="stAppViewContainer"] { background:var(--page); }
header[data-testid="stHeader"] { background:rgba(7, 17, 31, 0.9); }
h1, h2, h3, h4, h5, h6 { color:var(--text) !important; letter-spacing:-.02em; }
p, li, label, legend, small, [data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"], [data-testid="stMarkdownContainer"],
[data-testid="stMetricLabel"], [data-testid="stMetricValue"],
[data-testid="stMetricDelta"], [data-testid="stMarkdownContainer"] *,
[data-testid="stWidgetLabel"] * { color:var(--text) !important; }
[data-testid="stCaptionContainer"], .stCaption { color:var(--muted) !important; }
[data-testid="stMainBlockContainer"] { max-width:100%; padding-top:.8rem; padding-bottom:2rem; }
[data-testid="stMetric"], [data-testid="stDataFrame"], [data-testid="stTable"],
[data-testid="stPlotlyChart"], [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stExpander"], [data-testid="stForm"], .block-container {
    min-width:0;
    background:var(--panel);
    border:1px solid var(--border);
    border-radius:16px;
    box-shadow:0 8px 22px var(--shadow);
}
[data-testid="stMetric"] { padding:14px 16px 12px; }
[data-testid="stMetricLabel"] { color:var(--muted) !important; }
[data-testid="stMetricValue"] { color:var(--text) !important; }
[data-testid="stDataFrame"], [data-testid="stTable"] { max-width:100%; overflow-x:auto; }
[data-testid="stPlotlyChart"] {
    width:100%;
    max-width:100%;
    min-width:0;
}
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background:var(--sidebar) !important;
    border-right:1px solid #1B2C3E;
}
[data-testid="stSidebar"] {
    min-width:14.5rem !important;
    max-width:14.5rem !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color:#F8FAFC !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#CBD5E1 !important; }
[data-testid="stSidebar"] .eyebrow { color:#94A3B8 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color:#E2E8F0 !important;
    border-radius:10px;
    padding:.55rem .7rem;
    border:1px solid rgba(148, 163, 184, 0.15);
    background:rgba(15, 23, 42, 0.2);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background:linear-gradient(180deg, var(--primary) 0%, #1D4ED8 100%);
    border-color:rgba(96, 165, 250, 0.7);
    color:#FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) * {
    color:#FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {
    display:none;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stFileUploader"] section,
[data-testid="stSidebar"] [data-testid="stFileUploader"] div,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    background:var(--input);
    border-color:var(--input-border);
    color:#F8FAFC;
}
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
    color:#F8FAFC !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    border:1px dashed #475569;
    border-radius:12px;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background:var(--primary) !important;
    border-color:var(--primary) !important;
    color:#FFFFFF !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background:var(--input) !important;
    background-color:var(--input) !important;
    border:1px solid var(--input-border) !important;
    border-radius:10px;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div * {
    background:var(--input) !important;
    background-color:var(--input) !important;
    color:var(--text) !important;
}
[data-testid="stSelectbox"] [role="group"] {
    background:var(--input) !important;
    border:1px solid var(--input-border) !important;
    border-radius:10px !important;
}
[data-testid="stSelectbox"] [role="combobox"] {
    background:var(--input) !important;
    border:0 !important;
    color:var(--text) !important;
    -webkit-text-fill-color:var(--text);
}
[data-testid="stSelectbox"] button[aria-label="Open"] {
    background:var(--input) !important;
    border:0 !important;
    color:var(--text) !important;
    border-radius:0 10px 10px 0 !important;
}
[data-testid="stSelectbox"] button[aria-label="Open"]:hover {
    background:var(--input-hover) !important;
}
[data-testid="stSelectbox"] [role="group"]:focus-within {
    background:var(--input-hover) !important;
    border-color:var(--primary) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
    background:var(--input-hover) !important;
    background-color:var(--input-hover) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stSelectbox"] input {
    color:var(--text) !important;
    -webkit-text-fill-color:var(--text);
}
[data-testid="stSelectbox"] input::placeholder,
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
    color:var(--placeholder) !important;
    -webkit-text-fill-color:var(--placeholder);
    opacity:1;
}
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
[role="listbox"] {
    background:#111827 !important;
    border:1px solid var(--input-border) !important;
    color:#E5E7EB !important;
}
[role="option"] {
    background:#111827 !important;
    color:#E5E7EB !important;
}
[role="option"]:hover,
[role="option"][aria-selected="true"] {
    background:#1D4ED8 !important;
    color:#FFFFFF !important;
}
[data-testid="stFileUploader"] section {
    background:var(--input) !important;
    border:1px dashed var(--input-border) !important;
    border-radius:12px !important;
    color:var(--text) !important;
}
[data-testid="stFileUploader"] section *,
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] {
    color:var(--text) !important;
}
[data-testid="stFileChip"] {
    background:var(--input) !important;
    border:1px solid var(--input-border) !important;
    color:var(--text) !important;
}
[data-testid="stFileChip"] [data-testid="stFileChipName"],
[data-testid="stFileChip"] [data-testid="stFileChipDeleteBtn"] {
    color:var(--text) !important;
}
[data-testid="stFileChip"] > div:first-of-type {
    background:#26384D !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] [data-testid="stCaptionContainer"] {
    color:var(--muted) !important;
}
[data-testid="stFileUploader"] button {
    background:var(--primary) !important;
    border:1px solid var(--primary) !important;
    border-radius:8px !important;
    color:#FFFFFF !important;
}
[data-testid="stFileUploader"] button:hover {
    background:#1D4ED8 !important;
}
.stApp input,
.stApp textarea,
.stApp [data-baseweb="input"] > div,
.stApp [data-baseweb="textarea"] {
    background:var(--input);
    border-color:var(--input-border);
    color:var(--text);
}
.stApp input:focus,
.stApp textarea:focus,
.stApp [data-baseweb="input"]:focus-within,
.stApp [data-baseweb="textarea"]:focus-within,
.stApp [data-testid="stSelectbox"]:focus-within [data-baseweb="select"] > div {
    background:var(--input-hover);
    border-color:var(--primary) !important;
    outline:none;
}
.eyebrow { font:600 11px 'Inter','Manrope',sans-serif; color:#94A3B8; text-transform:uppercase; letter-spacing:.08em; }
.topbar {
    display:flex;
    flex-wrap:wrap;
    justify-content:space-between;
    align-items:center;
    gap:.75rem;
    padding:.25rem 0 .65rem;
}
.topbar-brand h1 {
    margin:0;
    font-size:clamp(2rem, 4vw, 3rem);
    line-height:1.05;
    letter-spacing:-0.045em;
    font-weight:800;
}
.topbar-brand .subtitle {
    margin-top:.35rem;
    color:var(--muted);
    font-size:.96rem;
}
.live-badge {
    display:inline-flex;
    align-items:center;
    gap:.5rem;
    padding:.3rem .7rem;
    border-radius:999px;
    background:rgba(16, 185, 129, 0.12);
    border:1px solid rgba(16, 185, 129, 0.38);
    color:#A7F3D0;
    font:600 11px 'Inter','Manrope',sans-serif;
    letter-spacing:.06em;
    text-transform:uppercase;
}
.live-badge::before {
    content:"";
    width:.55rem;
    height:.55rem;
    border-radius:50%;
    background:var(--healthy);
    box-shadow:0 0 0 4px rgba(16, 185, 129, 0.12);
}
.kpi-card {
    background:var(--panel);
    border:1px solid var(--border);
    border-radius:14px;
    padding:.8rem .9rem;
    min-height:118px;
    position:relative;
    border-top:2px solid var(--kpi-accent, var(--primary));
}
.kpi-label {
    color:var(--muted);
    font-size:.74rem;
    font-weight:600;
    letter-spacing:.02em;
}
.kpi-value {
    margin-top:.4rem;
    font-size:clamp(1.55rem, 2.6vw, 2rem);
    font-weight:800;
    letter-spacing:-.04em;
}
.kpi-trend {
    margin-top:.3rem;
    display:inline-flex;
    align-items:center;
    gap:.4rem;
    padding:.25rem .55rem;
    border-radius:999px;
    font:600 12px 'Inter','Manrope',sans-serif;
    background:rgba(148, 163, 184, 0.08);
    border:1px solid rgba(148, 163, 184, 0.14);
}
.kpi-trend.healthy { color:#A7F3D0; }
.kpi-trend.watch { color:#FCD34D; }
.kpi-trend.high { color:#FDBA74; }
.kpi-trend.critical { color:#FCA5A5; }
.kpi-trend.neutral { color:#BFDBFE; }
.kpi-card.healthy { --kpi-accent:var(--healthy); }
.kpi-card.watch { --kpi-accent:var(--warning); }
.kpi-card.high { --kpi-accent:#F97316; }
.kpi-card.critical { --kpi-accent:var(--danger); }
.kpi-card.neutral { --kpi-accent:var(--cyan); }
.risk-note {
    border:1px solid rgba(239, 68, 68, 0.32);
    border-left:4px solid var(--danger);
    border-radius:18px;
    padding:1rem 1.1rem;
    background:rgba(127, 29, 29, 0.18);
    color:#FECACA;
}
.risk-category-card {
    min-height:100%;
    padding:1rem 1.1rem;
    background:var(--panel);
    border:1px solid var(--border);
    border-radius:18px;
}
.risk-category-label { color:var(--muted) !important; font-size:.78rem; letter-spacing:.04em; font-family:'Inter','Manrope',sans-serif; }
.risk-category-value {
    display:inline-flex;
    margin-top:.65rem;
    padding:.42rem .7rem;
    border-radius:999px;
    font-size:1.2rem;
    font-weight:800;
    letter-spacing:.03em;
}
.risk-category-value.healthy { background:rgba(16, 185, 129, 0.14); color:#A7F3D0 !important; }
.risk-category-value.watch { background:rgba(245, 158, 11, 0.12); color:#FCD34D !important; }
.risk-category-value.high { background:rgba(249, 115, 22, 0.12); color:#FDBA74 !important; }
.risk-category-value.critical { background:rgba(239, 68, 68, 0.12); color:#FCA5A5 !important; }
.risk-category-value.neutral { background:rgba(148, 163, 184, 0.12); color:#E2E8F0 !important; }
.panel-header {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:1rem;
    margin:1rem 0 .55rem;
    padding:0;
}
.panel-header h3 {
    margin:0;
    font-size:1.15rem;
    letter-spacing:-0.03em;
}
.panel {
    background:var(--panel);
    border:1px solid var(--border);
    border-radius:16px;
    padding:.8rem;
    box-shadow:0 8px 22px var(--shadow);
}
.copilot-box {
    background:var(--panel);
    border:1px solid rgba(59, 130, 246, 0.24);
    border-radius:16px;
    padding:.85rem;
}
.copilot-input {
    background:#07111F !important;
    color:var(--text) !important;
    border:1px solid rgba(96, 165, 250, 0.22) !important;
    border-radius:12px !important;
    min-height:76px !important;
}
.copilot-output {
    background:#07111F;
    border:1px solid rgba(148, 163, 184, 0.18);
    border-radius:14px;
    padding:.75rem .85rem;
    color:#E2E8F0;
    line-height:1.6;
}
.chart-card {
    background:var(--panel);
    border:1px solid var(--border);
    border-radius:18px;
    padding:1rem;
    box-shadow:0 18px 40px var(--shadow);
}
.risk-summary-card {
    background:var(--panel);
    border:1px solid var(--border);
    border-left:3px solid var(--risk-accent, var(--cyan));
    border-radius:14px;
    padding:.8rem .9rem;
}
.risk-summary-card.healthy { --risk-accent:var(--healthy); }
.risk-summary-card.watch { --risk-accent:var(--warning); }
.risk-summary-card.high { --risk-accent:#F97316; }
.risk-summary-card.critical { --risk-accent:var(--danger); }
.risk-summary-card .risk-state {
    display:flex;
    align-items:center;
    gap:.45rem;
    font-weight:700;
    margin-bottom:.35rem;
}
.risk-summary-card .risk-state::before {
    content:"";
    width:.58rem;
    height:.58rem;
    flex:0 0 .58rem;
    border-radius:50%;
    background:var(--risk-accent, var(--cyan));
}
.risk-summary-card p { color:var(--muted) !important; margin:.3rem 0; line-height:1.5; }
.driver-legend { color:var(--muted); font-size:.78rem; margin:.1rem 0 .45rem; }
.signal-list { display:grid; gap:.45rem; }
.signal-item {
    display:flex;
    align-items:flex-start;
    gap:.55rem;
    padding:.55rem .65rem;
    border:1px solid var(--border);
    border-radius:11px;
    background:var(--panel);
    line-height:1.35;
}
.signal-indicator {
    flex:0 0 .58rem;
    width:.58rem;
    height:.58rem;
    margin-top:.28rem;
    border-radius:50%;
    background:var(--signal-color, var(--muted));
}
.signal-indicator.healthy { --signal-color:var(--healthy); }
.signal-indicator.watch { --signal-color:var(--warning); }
.signal-indicator.warning { --signal-color:#F97316; }
.signal-indicator.critical { --signal-color:var(--danger); }
.signal-item strong { display:block; font-size:.88rem; color:var(--text); }
.signal-item span { display:block; color:var(--muted); font-size:.78rem; margin-top:.08rem; }
.creator-credit { color:var(--muted); font-size:.75rem; font-weight:500; }
.main .stButton button, [data-testid="stFormSubmitButton"] button { width:100%; }
.assessment-card {
    background:var(--panel);
    border:1px solid var(--border);
    border-radius:16px;
    padding:1rem 1.1rem;
    margin:.75rem 0 1rem;
    box-shadow:0 8px 22px var(--shadow);
}
.assessment-card h3 {
    margin:0 0 .9rem;
    font-size:1.12rem;
    letter-spacing:0.08em;
    text-transform:uppercase;
    color:#E2E8F0;
}
.assessment-grid {
    display:grid;
    grid-template-columns:repeat(3, minmax(0, 1fr));
    gap:1rem;
    margin-bottom:1rem;
}
.assessment-item {
    padding:.7rem .75rem;
    border-radius:12px;
    background:rgba(15, 23, 42, 0.55);
    border:1px solid rgba(148, 163, 184, 0.14);
}
.assessment-label {
    font:600 11px 'Inter','Manrope',sans-serif;
    color:#94A3B8;
    letter-spacing:.08em;
    text-transform:uppercase;
}
.assessment-value {
    margin-top:.35rem;
    font-size:1.15rem;
    font-weight:800;
    color:#F8FAFC;
}
.assessment-meta {
    display:grid;
    grid-template-columns:repeat(2, minmax(0, 1fr));
    gap:1rem;
    margin-top:.5rem;
}
.assessment-block {
    background:rgba(15, 23, 42, 0.5);
    border:1px solid rgba(148, 163, 184, 0.14);
    border-radius:12px;
    padding:.8rem .9rem;
}
.assessment-block h4 {
    margin:0 0 .5rem;
    font-size:.8rem;
    letter-spacing:.08em;
    text-transform:uppercase;
    color:#94A3B8;
    font-family:'Inter','Manrope',sans-serif;
}
.assessment-bullets {
    list-style:none;
    padding:0;
    margin:0;
    color:#E2E8F0;
}
.assessment-bullets li {
    display:flex;
    align-items:center;
    gap:.55rem;
    margin:.35rem 0;
    line-height:1.4;
}
.assessment-bullets li::before {
    content:"";
    width:.48rem;
    height:.48rem;
    flex:0 0 .48rem;
    border-radius:50%;
    background:var(--healthy);
}
.assessment-bullets.risk li::before {
    background:var(--warning);
}
.assessment-recommendation {
    margin-top:1rem;
    padding:.9rem 1rem;
    border-left:4px solid #06B6D4;
    border-radius:12px;
    background:rgba(6, 182, 212, 0.08);
    color:#E0F2FE;
    line-height:1.6;
    font-weight:600;
}
.assessment-priorities {
    margin-top:1rem;
    background:rgba(15, 23, 42, 0.45);
    border:1px solid rgba(148, 163, 184, 0.14);
    border-radius:12px;
    padding:.8rem .9rem;
}
.assessment-priorities ol {
    margin:.5rem 0 0 1.1rem;
    padding:0;
    color:#E2E8F0;
}
.workflow-wrapper {
    margin:1rem 0 1.5rem;
}
.workflow-grid {
    display:grid;
    grid-template-columns: repeat(7, minmax(0, 1fr));
    gap:.85rem;
    align-items:stretch;
    margin-top:.6rem;
}
.workflow-step {
    position:relative;
    background:var(--panel);
    border:1px solid rgba(96, 165, 250, 0.18);
    border-radius:14px;
    padding:.8rem .7rem;
    min-height:108px;
    display:flex;
    flex-direction:column;
    justify-content:center;
    text-align:center;
    color:#E2E8F0;
}
.workflow-step .title {
    font-size:0.82rem;
    font-weight:700;
    line-height:1.4;
}
.workflow-step .tag {
    display:inline-block;
    margin-top:.4rem;
    font:600 10px 'Inter','Manrope',sans-serif;
    letter-spacing:.08em;
    color:#94A3B8;
    text-transform:uppercase;
}
.workflow-step::after {
    content:"↓";
    position:absolute;
    right:-0.55rem;
    top:50%;
    transform:translateY(-50%);
    color:#60A5FA;
    font-weight:800;
    font-size:1.1rem;
}
.workflow-step:last-child::after { content:""; }
.feature-row {
    display:flex;
    flex-wrap:wrap;
    gap:.55rem;
    margin-top:.75rem;
}
.feature-pill {
    display:inline-flex;
    align-items:center;
    padding:.38rem .7rem;
    border-radius:999px;
    background:rgba(37, 99, 235, 0.12);
    border:1px solid rgba(96, 165, 250, 0.22);
    color:#DBEAFE;
    font:600 12px 'Inter','Manrope',sans-serif;
    letter-spacing:.04em;
}
@media (max-width: 767px) {
    .assessment-grid,
    .assessment-meta,
    .workflow-grid {
        grid-template-columns:1fr;
    }
    .workflow-step::after {
        content:"";
    }
}
.stApp button, .stApp [role="button"] {
    min-height:44px;
    color:var(--text) !important;
    border-color:var(--border) !important;
}
.stApp [data-testid="stDownloadButton"] button,
.stApp [data-testid="stFormSubmitButton"] button,
.stApp [data-testid="baseButton-secondary"] button {
    background:linear-gradient(180deg, var(--primary) 0%, #1D4ED8 100%);
    border-color:rgba(96, 165, 250, 0.8);
    color:#FFFFFF !important;
}
.stApp input, .stApp textarea, .stApp [data-baseweb="select"] * { color:var(--text); }
.stApp [data-testid="stImage"] img { max-width:100%; height:auto; object-fit:contain; }
[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stAlert"] p { color:var(--text) !important; }
@media (max-width: 767px) {
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        width:100%;
        max-width:100%;
        min-width:0;
        overflow-x:hidden;
    }
    [data-testid="stMainBlockContainer"] {
        padding:1rem clamp(12px, 4vw, 18px) 2rem;
    }
    [data-testid="stHorizontalBlock"] {
        flex-wrap:wrap;
        gap:.65rem;
        min-width:0;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        flex:1 1 100% !important;
        width:100% !important;
        min-width:0 !important;
    }
    [data-testid="stSidebar"][aria-expanded="true"] {
        width:min(72vw, 16rem) !important;
        min-width:min(72vw, 16rem) !important;
        max-width:72vw !important;
    }
    [data-testid="stSidebar"] { min-width:0 !important; max-width:none !important; }
    [data-testid="stSelectbox"],
    [data-testid="stFileUploader"],
    [data-baseweb="popover"],
    [role="listbox"] {
        width:100%;
        max-width:100vw;
        min-width:0;
    }
    [role="listbox"] { max-height:min(55vh, 24rem); overflow-y:auto; }
    [data-testid="stFileUploader"] section { padding:.75rem !important; }
    [data-testid="stFileUploader"] button { min-height:42px; }
    [data-testid="stMetric"] { width:100%; padding:14px; }
    .kpi-card { min-height:0; padding:.75rem .8rem; }
    .topbar { padding:.15rem 0 .4rem; }
    .topbar-brand .subtitle { font-size:.86rem; line-height:1.4; max-width:32rem; }
    .panel-header { margin:.8rem 0 .45rem; }
    .panel { padding:.65rem; }
    .assessment-card { padding:.85rem; margin:.7rem 0 1rem; }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        max-width:100%;
        overflow-x:auto;
        white-space:nowrap;
    }
    [data-testid="stPlotlyChart"], [data-testid="stImage"],
    [data-testid="stFileUploader"], [data-baseweb="select"] {
        width:100%;
        max-width:100%;
        min-width:0;
    }
    [data-testid="stFileUploader"] section { width:100%; }
    [data-testid="stImage"] img { width:auto; max-width:min(100%, 220px); }
    .stApp button, .stApp [role="button"] { min-height:44px; }
    h1 { font-size:clamp(1.65rem, 7vw, 2.1rem); }
    h2 { font-size:clamp(1.3rem, 5.5vw, 1.65rem); }
    h3 { font-size:clamp(1.1rem, 4.5vw, 1.35rem); }
}
</style>
""", unsafe_allow_html=True)


def style_chart(
    figure: go.Figure,
    *,
    height: int | None = None,
    margin: dict[str, int] | None = None,
) -> None:
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#0D1B2A",
        font={"family": "Inter, Manrope, sans-serif", "color": "#F8FAFC", "size": 12},
        title_font={"color": "#F8FAFC", "size": 16},
        legend={
            "font": {"color": "#F8FAFC", "size": 12},
            "bgcolor": "#0D1B2A",
            "bordercolor": "#1E293B",
            "borderwidth": 1,
        },
        margin=margin or {"t": 36, "r": 12, "b": 38, "l": 12},
        height=height,
        autosize=True,
    )
    figure.update_xaxes(
        color="#E2E8F0",
        title_font={"color": "#E2E8F0"},
        tickfont={"color": "#94A3B8"},
        gridcolor="rgba(148, 163, 184, 0.12)",
        zerolinecolor="rgba(148, 163, 184, 0.24)",
        automargin=True,
    )
    figure.update_yaxes(
        color="#E2E8F0",
        title_font={"color": "#E2E8F0"},
        tickfont={"color": "#94A3B8"},
        gridcolor="rgba(148, 163, 184, 0.12)",
        zerolinecolor="rgba(148, 163, 184, 0.24)",
        automargin=True,
    )


def render_chart(
    figure: go.Figure,
    *,
    height: int | None = None,
    margin: dict[str, int] | None = None,
) -> None:
    style_chart(figure, height=height, margin=margin)
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displayModeBar": False},
    )


def _money(value: float | pd.Series | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    numeric = float(value)
    return f"{numeric:,.0f}"


def _state_color(value: str) -> str:
    value = (value or "").lower()
    if "low" in value:
        return "healthy"
    if "moderate" in value:
        return "watch"
    if "high" in value:
        return "high"
    if "critical" in value:
        return "critical"
    return "neutral"


def render_kpi_card(label: str, value: str, status: str, state: str = "neutral") -> None:
    st.markdown(
        f"""
        <div class="kpi-card {state}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-trend {state}">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_shap_drivers(result: dict) -> None:
    drivers = []
    for direction, key in (("Risk increased", "top_risk_factors"), ("Risk reduced", "protective_factors")):
        for factor in result.get(key, []):
            contribution = factor.get("contribution")
            if contribution is None or pd.isna(contribution):
                continue
            feature = str(factor.get("feature", "Feature")).replace("_", " ")
            drivers.append({
                "Feature": feature,
                "Contribution": float(contribution),
                "Direction": direction,
            })
    if not drivers:
        st.info("SHAP drivers are not available for this record.")
        return

    driver_frame = pd.DataFrame(drivers).sort_values("Contribution")
    figure = px.bar(
        driver_frame,
        x="Contribution",
        y="Feature",
        color="Direction",
        orientation="h",
        color_discrete_map={"Risk increased": "#EF4444", "Risk reduced": "#10B981"},
        category_orders={"Direction": ["Risk increased", "Risk reduced"]},
    )
    figure.update_layout(
        title=None,
        xaxis_title="SHAP contribution to distress output",
        yaxis_title=None,
        showlegend=True,
        legend_title_text=None,
        coloraxis_showscale=False,
    )
    render_chart(figure, height=min(320, 145 + 24 * len(driver_frame)), margin={"t": 12, "r": 12, "b": 38, "l": 12})
    st.caption("Positive SHAP values increase the model output for distress; negative values reduce it. SHAP contributions are not causal.")


_WARNING_PRESENTATION = {
    "Rapid revenue decline": ("critical", "Revenue declined beyond the configured warning threshold."),
    "Negative operating cash flow": ("critical", "Operating cash flow is negative for this period."),
    "Increasing leverage": ("warning", "Leverage is high or rising compared with the prior period."),
    "Falling liquidity": ("warning", "Liquidity is below threshold or declining."),
    "Deteriorating margins": ("watch", "Margin indicators are negative or deteriorating."),
    "Increasing receivable days": ("watch", "Receivables are taking longer to collect."),
    "Falling interest coverage": ("warning", "Interest coverage is low or falling."),
}


def render_warning_signals(active_signals: list[str]) -> None:
    st.markdown("<div class='panel-header'><h3>Early Warning Signals</h3></div>", unsafe_allow_html=True)
    if not active_signals:
        st.markdown(
            "<div class='signal-item'><span class='signal-indicator'></span>"
            "<div><strong>No active warning signals</strong>"
            "<span>No configured condition was triggered by observed values; missing fields are not treated as healthy.</span></div></div>",
            unsafe_allow_html=True,
        )
        return

    labels = {"watch": "Watch", "warning": "Warning", "critical": "Critical"}
    items = []
    for signal in active_signals:
        severity, detail = _WARNING_PRESENTATION.get(signal, ("watch", "Configured early-warning condition is triggered."))
        items.append(
            f"<div class='signal-item'><span class='signal-indicator {severity}'></span>"
            f"<div><strong>{escape(signal)} · {labels[severity]}</strong>"
            f"<span>{escape(detail)}</span></div></div>"
        )
    st.markdown(f"<div class='signal-list'>{''.join(items)}</div>", unsafe_allow_html=True)


def render_credit_assessment_panel(company_name: str, result: dict, selected_features: pd.DataFrame, flags: pd.DataFrame, row_index: int) -> None:
    feat = selected_features.iloc[0]
    current_ratio = feat.get("Current_Ratio", float("nan"))
    margin = feat.get("EBITDA_Margin", 0.0)
    cash_flow = feat.get("Cash_Flow_Operations", 0.0)
    leverage = feat.get("Debt_to_Assets", 0.0)
    sales_growth = feat.get("Sales_Growth", float("nan"))
    active_flags = set(flags.iloc[row_index][flags.iloc[row_index]].index.tolist())
    health_score = result.get("health_score")

    strengths = []
    if pd.notna(sales_growth) and float(sales_growth) > 0:
        strengths.append("Revenue growth")
    if pd.notna(current_ratio) and float(current_ratio) >= 1.0:
        strengths.append("Healthy current ratio")

    risks = []
    if "Deteriorating margins" in active_flags or (pd.notna(margin) and float(margin) < 0):
        risks.append("Declining operating margin")
    if "Negative operating cash flow" in active_flags or (pd.notna(cash_flow) and float(cash_flow) < 0):
        risks.append("Negative operating cash flow")
    if "Increasing leverage" in active_flags or (pd.notna(leverage) and float(leverage) >= 0.65):
        risks.append("Increasing leverage")

    if not risks:
        risks = ["No configured risk threshold was triggered by the available figures"]
    if not strengths:
        strengths = ["No strength could be confirmed from the available figures"]

    priorities = []
    if "Negative operating cash flow" in active_flags or (
        pd.notna(cash_flow) and float(cash_flow) < 0
    ):
        priorities.append("Monitor operating cash flow")
    if "Increasing leverage" in active_flags or (
        pd.notna(leverage) and float(leverage) >= 0.65
    ):
        priorities.append("Reduce leverage")
    if "Deteriorating margins" in active_flags or (
        pd.notna(margin) and float(margin) < 0
    ):
        priorities.append("Protect operating margins")
    if "Falling liquidity" in active_flags or (
        pd.notna(current_ratio) and float(current_ratio) < 1
    ):
        priorities.append("Strengthen short-term liquidity")
    if "Rapid revenue decline" in active_flags:
        priorities.append("Investigate the revenue decline")
    if not priorities:
        priorities.append("Continue monitoring the financial indicators available in this statement")

    risk_label = result["risk_category"]
    if result.get("method") == "Existing ML model":
        recommendation = (
            "The company may be considered for credit subject to tighter monitoring and improved cash-flow generation."
            if risk_label in {"Moderate Risk", "Low Risk"}
            else "The company requires deeper credit review due to elevated distress indicators."
        )
    else:
        recommendation = (
            "This is a sparse-data, rule-based assessment only. It does not determine credit eligibility; "
            "review the observed indicators and obtain additional financial records."
        )
    displayed_health = f"{health_score:.0f}/100" if health_score is not None else "Not available"

    st.markdown(
        f"""
        <div class="assessment-card">
            <h3>MSME CREDIT ASSESSMENT</h3>
            <div class="assessment-grid">
                <div class="assessment-item">
                    <div class="assessment-label">Company</div>
                    <div class="assessment-value">{company_name}</div>
                </div>
                <div class="assessment-item">
                    <div class="assessment-label">Assessment</div>
                    <div class="assessment-value">{risk_label.upper()}</div>
                </div>
                <div class="assessment-item">
                    <div class="assessment-label">Financial Health Score</div>
                    <div class="assessment-value">{displayed_health}</div>
                </div>
            </div>
            <div class="assessment-meta">
                <div class="assessment-block">
                    <h4>Key strengths</h4>
                    <ul class="assessment-bullets">
                        {''.join(f'<li>{item}</li>' for item in strengths)}
                    </ul>
                </div>
                <div class="assessment-block">
                    <h4>Key risks</h4>
                    <ul class="assessment-bullets risk">
                        {''.join(f'<li>{item}</li>' for item in risks)}
                    </ul>
                </div>
            </div>
            <div class="assessment-recommendation">{recommendation}</div>
            <div class="assessment-priorities">
                <h4 style="margin:0; font-size:.8rem; letter-spacing:.04em; color:#94A3B8; font-family:'Inter','Manrope',sans-serif;">Early-warning priorities</h4>
                <ol>
                    {''.join(f'<li>{item}</li>' for item in priorities)}
                </ol>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_risk_interpretation(
    result: dict,
    financials: pd.Series,
    warning_signals: list[str],
) -> None:
    interpretation = generate_risk_interpretation(
        probability=float(result["distress_probability"]),
        risk_category=str(result["risk_category"]),
        financials=financials.to_dict(),
        warning_signals=warning_signals,
        shap_factors=result.get("top_risk_factors", []),
    )
    state = _state_color(interpretation["risk_category"])
    risk_heading = {
        "Low Risk": "LOW CREDIT RISK",
        "Moderate Risk": "MODERATE CREDIT RISK",
        "High Risk": "HIGH CREDIT RISK",
        "Critical Risk": "CRITICAL CREDIT RISK",
    }.get(interpretation["risk_category"], interpretation["risk_category"].upper())
    st.markdown("<div class='panel-header'><h3>AI Risk Summary</h3></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="risk-summary-card {state}">
            <div class="risk-state">{escape(risk_heading)}</div>
            <p>{escape(interpretation['explanation'])}</p>
            <p><strong>Priority:</strong> {escape(interpretation['priority'])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if result.get("top_risk_factors"):
        st.caption("SHAP contributions explain model behavior; they are associations, not causal findings.")


def _band_colour(band: str) -> str:
    return {
        "Low Risk": "#10B981",
        "Moderate Risk": "#F59E0B",
        "High Risk": "#F97316",
        "Critical Risk": "#EF4444",
    }.get(band, "#94A3B8")


def _record_label(frame: pd.DataFrame, position: int) -> str:
    row = frame.iloc[position]
    company = row.get("company_id", f"Record {position + 1}")
    period = row.get("period")
    return f"{company} · {period}" if pd.notna(period) else str(company)


def render_filtered_grid(
    frame: pd.DataFrame,
    key: str,
    *,
    blob_source: str = "dataset",
    page_size_options: tuple[int, ...] = PAGE_SIZE_OPTIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Render a server-side searchable, filterable, sorted table of one page only."""
    st.markdown("<div class='panel-header'><h3>Records</h3></div>", unsafe_allow_html=True)
    control_cols = st.columns([2, 1, 1])
    with control_cols[0]:
        search = st.text_input(
            "Search",
            key=f"{key}_search",
            placeholder="Search any column",
        )
    candidates = filter_candidates(frame)
    with control_cols[1]:
        band_column = next(
            (column for column in candidates if column.lower() in {"risk category", "risk_category"}),
            None,
        )
        band_filter = (
            st.multiselect(
                "Risk band",
                candidates[band_column],
                key=f"{key}_band",
                placeholder="All bands",
            )
            if band_column
            else []
        )
    with control_cols[2]:
        page_size = st.selectbox(
            "Rows per page",
            page_size_options,
            index=1 if len(page_size_options) > 1 else 0,
            key=f"{key}_size",
        )

    extra_cols = [column for column in candidates if column != band_column]
    column_filters: dict[str, list[str]] = {}
    if extra_cols:
        with st.expander("Column filters"):
            filter_cols = st.columns(min(3, len(extra_cols)))
            for position, column in enumerate(extra_cols):
                with filter_cols[position % len(filter_cols)]:
                    column_filters[column] = st.multiselect(
                        column,
                        candidates[column],
                        key=f"{key}_filter_{column}",
                        placeholder="All",
                    )

    filtered = apply_search(frame, _cached_search_blob(dataset_key, blob_source), search)
    if band_column and band_filter:
        filtered = filtered[filtered[band_column].astype("string").isin(band_filter).to_numpy()]
    filtered = apply_column_filters(filtered, column_filters)
    visible, pages, page = page_slice(filtered, st.session_state.get(f"{key}_page", 1), page_size)

    sort_cols = st.columns([2, 1, 1])
    with sort_cols[0]:
        sort_column = st.selectbox(
            "Sort by",
            [""] + [str(column) for column in frame.columns],
            key=f"{key}_sort",
        )
    with sort_cols[1]:
        ascending = st.toggle("Ascending", value=False, key=f"{key}_asc")
    with sort_cols[2]:
        st.caption(f"{len(filtered):,} of {len(frame):,} records match")
    if sort_column:
        visible = apply_sort(visible, sort_column, ascending)

    nav_cols = st.columns([1, 3, 1])
    with nav_cols[0]:
        if st.button("Previous", key=f"{key}_prev", disabled=page <= 1):
            st.session_state[f"{key}_page"] = max(1, page - 1)
            st.rerun()
    with nav_cols[1]:
        st.caption(f"Page {page} of {pages}")
    with nav_cols[2]:
        if st.button("Next", key=f"{key}_next", disabled=page >= pages):
            st.session_state[f"{key}_page"] = min(pages, page + 1)
            st.rerun()
    return filtered, visible


def render_snapshot_detail(snapshot: dict, key: str) -> None:
    probability = snapshot["distress_probability"]
    state = _state_color(snapshot["risk_category"])
    heading = (
        f"{escape(snapshot['company'])} · {escape(snapshot['period'])}"
    )
    st.markdown(
        f"<div class='risk-summary-card {state}'><div class='risk-state'>{heading}</div>"
        f"<p>{escape(snapshot['risk_category'])} · "
        f"{'not calculable' if probability is None else format(probability, '.1%')} · "
        f"{escape(snapshot['method'])}</p>"
        f"<p>Data coverage: {escape(snapshot['coverage_label'])}</p></div>",
        unsafe_allow_html=True,
    )
    if snapshot["warnings"]:
        render_warning_signals(snapshot["warnings"])
    else:
        st.info("No early-warning signal is active for this record.")
    ratios = pd.DataFrame(
        [{"Ratio": label, "Value": value if value is not None else "Not available"}
         for label, value in snapshot["ratios"].items()]
    ).astype({"Value": "string"})
    st.dataframe(ratios, width="stretch", hide_index=True, key=key)
    if snapshot["top_risk_factors"]:
        st.dataframe(
            pd.DataFrame(snapshot["top_risk_factors"]),
            width="stretch",
            hide_index=True,
            key=f"{key}_factors",
        )


def render_screening_page(frame: pd.DataFrame, features: pd.DataFrame, flags: pd.DataFrame, bundle: dict) -> None:
    page_header(
        "Portfolio Screening",
        "Every record is scored in one vectorized pass, so a whole file can be ranked without "
        "opening each company-period.",
    )
    scores = _cached_screening_scores(dataset_key, _bundle_signature(bundle))
    if scores.empty:
        st.info("There are no records to screen in this dataset.")
        return
    summary = screening_summary(scores)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Records screened", f"{summary['records']:,}")
    metric_cols[1].metric("ML model scored", f"{summary['model_records']:,}")
    metric_cols[2].metric("Rule-based index", f"{summary['rule_records']:,}")
    metric_cols[3].metric("Elevated risk (>=60%)", f"{summary['elevated_records']:,}")
    mean_text = (
        "Not available"
        if summary["mean_probability"] is None
        else f"{summary['mean_probability']:.1%}"
    )
    metric_cols[4].metric("Mean risk estimate", mean_text)
    st.caption(
        "Band membership is a research classification of this model's output. "
        "Rule-based records use a transparent heuristic index, not a calibrated probability of default."
    )

    if summary["bands"]:
        band_frame = pd.DataFrame(
            [{"Risk category": band, "Records": count} for band, count in summary["bands"].items()]
        )
        figure = px.bar(
            band_frame,
            x="Records",
            y="Risk category",
            orientation="h",
            color="Risk category",
            color_discrete_map={band: _band_colour(band) for band in summary["bands"]},
            title="Screened records by risk band",
        )
        figure.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
        render_chart(figure, height=250, margin={"t": 42, "r": 12, "b": 38, "l": 120})

    exposure_column = detect_exposure_column(frame)
    if exposure_column:
        exposure = banded_exposure(scores, frame, exposure_column)
        if not exposure.empty:
            st.markdown("<div class='panel-header'><h3>Exposure by risk band</h3></div>", unsafe_allow_html=True)
            st.dataframe(exposure, width="stretch", hide_index=True)
            st.caption(
                f"Aggregated from the observed '{exposure_column}' column. Band totals describe the "
                "current dataset only and are not a portfolio forecast."
            )

    filtered, visible = render_filtered_grid(scores, "screening", blob_source="screening")
    st.dataframe(visible, width="stretch", hide_index=True, key="screening_table")
    if filtered.empty:
        st.info("No screened record matches the current search and filters.")
        return

    position_column = "Distress probability"
    candidate_positions = visible["_position"].astype(int).tolist()
    with st.expander("Focused record detail"):
        label_lookup = {
        int(position): _record_label(frame, int(position))
        for position in visible["_position"]
    }
        choice = st.selectbox(
            "Record",
            candidate_positions,
            format_func=lambda position: label_lookup[position],
            key="screening_focus",
        )
        snapshot = record_snapshot(frame, int(choice), features, flags, bundle)
        render_snapshot_detail(snapshot, "screening_focus_detail")
        st.caption(
            "This detail uses the same feature pass and model scoring as the table above, so the "
            "two views agree."
        )
    st.caption(
        f"Screening ranks {len(filtered):,} of {summary['records']:,} records. "
        "Sorting and paging happen on the server, so only the visible page is sent to the browser."
    )


def _snapshot_label(snapshot: dict, seen: set[str]) -> str:
    """Unique column label for a compared record.

    Two periods of the same company share a name, so a counter is appended only
    when the label is already taken.
    """
    base = f"{snapshot['company']} · {snapshot['period']}"
    label = base
    counter = 2
    while label in seen:
        label = f"{base} ({counter})"
        counter += 1
    seen.add(label)
    return label


def _format_ratio(value: object) -> str:
    """Format one ratio for display; mixed text keeps the column a plain string."""
    if value is None:
        return "Not available"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "Not available" if pd.isna(value) else f"{float(value):,.3f}"
    return str(value)


def render_comparison_page(frame: pd.DataFrame, features: pd.DataFrame, flags: pd.DataFrame, bundle: dict, company_rows: list[int]) -> None:
    page_header(
        "Compare Records",
        "Place up to four company-periods side by side to see where their risk differs.",
    )
    limit = min(len(company_rows), 4)
    options = list(company_rows)[: max(1, min(len(company_rows), 500))]
    chosen = st.multiselect(
        "Records",
        options,
        default=company_rows[:limit],
        format_func=lambda position: _record_label(frame, int(position)),
        key="comparison_records",
        help="Up to four records are compared; the first four are pre-selected.",
    )
    if not chosen:
        st.info("Select at least one record to compare.")
        return
    chosen = [int(position) for position in chosen][:4]
    snapshots = [record_snapshot(frame, position, features, flags, bundle) for position in chosen]

    head_cols = st.columns(len(snapshots))
    for column, snapshot in zip(head_cols, snapshots, strict=False):
        probability = snapshot["distress_probability"]
        column.metric(
            snapshot["company"],
            "Not calculable" if probability is None else f"{probability:.1%}",
            snapshot["risk_category"],
        )

    ratio_labels: list[str] = []
    for snapshot in snapshots:
        for label in snapshot["ratios"]:
            if label not in ratio_labels:
                ratio_labels.append(label)
    # Records of the same company collapse into one column if the company name is
    # used as the key, so label every column with company and period.
    seen_labels: set[str] = set()
    comparison_columns: dict[str, dict[str, str]] = {}
    for snapshot in snapshots:
        comparison_columns[_snapshot_label(snapshot, seen_labels)] = {
            label: _format_ratio(snapshot["ratios"].get(label))
            for label in ratio_labels
        }
    comparison = pd.DataFrame(comparison_columns)
    st.markdown("<div class='panel-header'><h3>Ratio comparison</h3></div>", unsafe_allow_html=True)
    st.dataframe(comparison, width="stretch", key="comparison_ratios")
    st.caption("Zero denominators are treated as missing rather than zero; missing inputs are not treated as healthy.")

    chart_rows = [
        {
            "Record": f"{snapshot['company']} · {snapshot['period']}",
            "Risk category": snapshot["risk_category"],
            "Distress probability": (
                snapshot["distress_probability"]
                if snapshot["distress_probability"] is not None
                else 0.0
            ),
        }
        for snapshot in snapshots
    ]
    figure = px.bar(
        pd.DataFrame(chart_rows),
        x="Distress probability",
        y="Record",
        orientation="h",
        color="Risk category",
        color_discrete_map={band: _band_colour(band) for band in {row["Risk category"] for row in chart_rows}},
        title="Risk estimate by record",
    )
    figure.update_layout(showlegend=False, xaxis_title="Risk estimate", yaxis_title=None, xaxis_range=[0, 1])
    render_chart(figure, height=max(220, 120 + 60 * len(chart_rows)), margin={"t": 42, "r": 12, "b": 38, "l": 150})

    coverage = pd.DataFrame([
        {
            "Record": f"{snapshot['company']} · {snapshot['period']}",
            "Method": snapshot["method"],
            "Data coverage": snapshot["coverage_label"],
            "Warning signals": len(snapshot["warnings"]),
            "Active warnings": ", ".join(snapshot["warnings"]) or "None active",
        }
        for snapshot in snapshots
    ])
    st.markdown("<div class='panel-header'><h3>Assessment basis</h3></div>", unsafe_allow_html=True)
    st.dataframe(coverage, width="stretch", hide_index=True, key="comparison_basis")

    if "company_id" in frame.columns and "period" in frame.columns and len(snapshots) >= 2:
        shared = {snapshot["company"] for snapshot in snapshots}
        if len(shared) == 1:
            st.markdown("<div class='panel-header'><h3>Trend across the selected periods</h3></div>", unsafe_allow_html=True)
            series = trend_data(frame, snapshots[0]["company"])
            if len(series) > 1:
                metric = st.selectbox(
                    "Measure",
                    ["Revenue", "Current_Ratio", "Debt_to_Assets", "EBITDA_Margin", "Interest_Coverage"],
                    key="comparison_metric",
                )
                trend_figure = px.line(
                    series,
                    x="period",
                    y=metric,
                    markers=True,
                    color_discrete_sequence=["#2563EB"],
                    title=f"{metric.replace('_', ' ')} over time",
                )
                render_chart(trend_figure, height=300)
            else:
                st.info("At least two dated periods are needed for a trend line.")
    disclaimer()


def render_data_explorer(frame: pd.DataFrame) -> None:
    page_header(
        "Data Explorer",
        "Search, filter, sort, and page through the loaded records without sending the whole "
        "table to the browser.",
    )
    filtered, visible = render_filtered_grid(frame, "explorer")
    st.dataframe(
        visible,
        width="stretch",
        hide_index=True,
        key="explorer_table",
        column_config={
            str(column): st.column_config.NumberColumn(str(column), format="%.2f")
            for column in visible.columns
            if pd.api.types.is_float_dtype(visible[column])
        },
    )
    st.caption(
        f"Showing {len(visible):,} of {len(filtered):,} matching rows across "
        f"{len(frame.columns):,} variables. Filters, search, and sorting run on the server."
    )
    download_cols = st.columns(2)
    with download_cols[0]:
        st.download_button(
            "Download filtered rows (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_records.csv",
            mime="text/csv",
            key="explorer_download_filtered",
            width="stretch",
        )
    with download_cols[1]:
        st.download_button(
            "Download visible page (CSV)",
            data=visible.to_csv(index=False).encode("utf-8"),
            file_name="visible_page.csv",
            mime="text/csv",
            key="explorer_download_page",
            width="stretch",
        )


def render_monitoring_page(bundle: dict) -> None:
    page_header(
        "Model Monitoring",
        "Compare successive analyses of the same dataset and track how the screened risk "
        "distribution moves.",
    )
    history = st.session_state.get("_monitor_history", [])
    if len(history) < 1:
        st.info("No analysis revision has been recorded in this session yet.")
        return
    snapshots = history
    st.markdown("<div class='panel-header'><h3>Recorded revisions</h3></div>", unsafe_allow_html=True)
    st.dataframe(
        snapshot_table(snapshots),
        width="stretch",
        hide_index=True,
        key="monitor_snapshots",
    )
    st.caption(
        "Each row is one Analyze Dataset or Re-analyze run. Revisions are compared on shape, "
        "completeness, and observed distributions."
    )
    if len(snapshots) < 2:
        st.info("Run Re-analyze, or load a revised file, to produce a second revision and compare.")
    else:
        baseline, current = snapshots[0], snapshots[-1]
        drift = drift_report(baseline, current)
        st.markdown(
            f"<div class='panel-header'><h3>Drift vs baseline ({escape(str(baseline['label']))} → "
            f"{escape(str(current['label']))})</h3></div>",
            unsafe_allow_html=True,
        )
        st.dataframe(drift, width="stretch", hide_index=True, key="monitor_drift")
        st.caption(
            "Population Stability Index (PSI) is a screening signal for distribution movement. "
            "PSI below 0.10 is treated as stable, 0.10 to 0.25 as a watch item, and above 0.25 as "
            "a material shift that warrants revalidation. PSI is not a model validity test."
        )
        score_history = [
            _snapshot_risk_scores(snapshot, bundle) for snapshot in snapshots
        ]
        alerts = monitoring_alerts(drift, score_history)
        st.markdown("<div class='panel-header'><h3>Alerts</h3></div>", unsafe_allow_html=True)
        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.success("No material movement was detected between these revisions.")

        st.markdown("<div class='panel-header'><h3>Screened risk stability</h3></div>", unsafe_allow_html=True)
        stability = prediction_stability(score_history)
        if stability.empty or stability["Scored records"].sum() == 0:
            st.info("Screened risk scores are not stored for these revisions.")
        else:
            st.dataframe(stability, width="stretch", hide_index=True, key="monitor_stability")
            latest = stability.iloc[-1]
            figure = px.line(
                stability,
                x="Revision",
                y="Elevated share",
                markers=True,
                color_discrete_sequence=["#2563EB"],
                title="Share of records in the elevated-risk band by revision",
            )
            figure.update_yaxes(tickformat=".0%")
            render_chart(figure, height=260)

    report = bundle.get("report", {})
    st.markdown("<div class='panel-header'><h3>Model in service</h3></div>", unsafe_allow_html=True)
    model_cols = st.columns(4)
    model_cols[0].metric("Selected model", str(report.get("selected_model", "Not available")))
    model_cols[1].metric("Selection metric", str(report.get("selection_metric", "Not available")))
    accuracy = report.get("test_metrics", {}).get("accuracy")
    model_cols[2].metric("Hold-out accuracy", f"{accuracy:.1%}" if isinstance(accuracy, float) else "Not available")
    model_cols[3].metric("Bundle feature count", len(bundle.get("features", [])))
    st.caption(
        "Monitoring here observes the served model and the dataset it scores. It does not retrain, "
        "recalibrate, or certify the model; those steps need separate validation and governance."
    )
    disclaimer()


NAVIGATION = {
    "Overview": "Executive Overview",
    "Screening": "Portfolio Screening",
    "Comparison": "Compare Records",
    "Data Explorer": "Data Explorer",
    "Data Intelligence": "Data Intelligence",
    "Financial Health": "MSME Financial Health",
    "Risk Prediction": "Risk Prediction",
    "AI Copilot": "AI Copilot",
    "Explainable AI": "Explainable AI",
    "Early Warning": "Early-Warning Indicators",
    "Scenario Simulator": "Scenario Simulator",
    "Credit Assessment": "Credit Assessment",
    "Model Performance": "Model Performance",
    "Monitoring": "Model Monitoring",
    "Methodology": "Methodology",
    "About": "About",
}


@st.cache_resource
def get_bundle() -> dict:
    if DEFAULT_MODEL_PATH.exists():
        return joblib.load(DEFAULT_MODEL_PATH)
    with st.spinner("Preparing the illustrative demo model..."):
        bundle = train_models(make_demo_data(), protected_attribute="owner_gender")
        DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, DEFAULT_MODEL_PATH)
    return bundle


st.sidebar.markdown("<div class='eyebrow'>CREDIT INTELLIGENCE</div>", unsafe_allow_html=True)
st.sidebar.title("CREDIT RISK AI")
selected_navigation = st.sidebar.radio("Workspace", list(NAVIGATION), label_visibility="collapsed")
page = NAVIGATION[selected_navigation]

st.markdown(
    """
    <div class="topbar">
        <div class="topbar-brand">
            <h1>CREDIT RISK AI</h1>
            <div class="subtitle">AI-Powered MSME Financial Intelligence &amp; Early Warning System</div>
        </div>
        <div class="creator-credit">Made by Surojit Malakar • SkillseED India</div>
    </div>
    """,
    unsafe_allow_html=True,
)
selector_cols = st.columns([1.25, 1, 1.2])
with selector_cols[2]:
    uploaded = st.file_uploader(
        "Upload financial statement",
        type=["csv", "xlsx", "xls", "pdf"],
        key="financial_csv_upload",
        help="CSV, Excel workbooks, and text-based PDFs are processed locally. Scanned PDF images require OCR and are not supported.",
    )
document_result = None
data_intelligence = None
data_visualizations = {}
if uploaded is not None:
    try:
        upload_token = (uploaded.file_id, uploaded.name, uploaded.size)
        same_upload = st.session_state.get("_upload_token") == upload_token
        if same_upload:
            upload_signature = st.session_state["_upload_signature"]
            uploaded_content = b""
        else:
            upload_signature = _content_signature(uploaded)
            uploaded_content = uploaded.getvalue()
            st.session_state["_upload_token"] = upload_token
            st.session_state["_upload_signature"] = upload_signature
            st.session_state["_upload_analysis_revision"] = 0
        analysis_revision = int(st.session_state.get("_upload_analysis_revision", 0))
        upload_pipeline_key = (upload_signature, uploaded.name, analysis_revision)
        has_current_analysis = st.session_state.get("_upload_pipeline_key") == upload_pipeline_key
        analyze_clicked = st.button(
            "Analyze Dataset",
            disabled=has_current_analysis,
            key="analyze_upload",
            use_container_width=True,
        )
        reanalyze_clicked = st.button(
            "Re-analyze",
            disabled=not has_current_analysis,
            key="re_analyze_upload",
            use_container_width=True,
        )
        if reanalyze_clicked:
            analysis_revision += 1
            st.session_state["_upload_analysis_revision"] = analysis_revision
        upload_pipeline_key = (upload_signature, uploaded.name, analysis_revision)
        if not has_current_analysis and not analyze_clicked and not reanalyze_clicked:
            st.success("File loaded")
            st.info("Click Analyze Dataset when you are ready to process this file.")
            st.stop()
        if st.session_state.get("_upload_pipeline_key") != upload_pipeline_key:
            if not uploaded_content:
                uploaded_content = uploaded.getvalue()
            with st.status("Analyzing dataset...", expanded=True) as progress:
                document_result = _cached_document_extraction(
                    uploaded_content,
                    uploaded.name,
                    analysis_revision,
                    upload_signature,
                )
                progress.update(label="Detecting variables...")
                data_intelligence, financial_columns = _cached_dataset_profile(
                    document_result.frame,
                    uploaded.name,
                    analysis_revision,
                    upload_signature,
                )
                progress.update(label="Running risk analysis...")
                financial_frame = (
                    _cached_financial_frame(
                        document_result.frame,
                        uploaded.name,
                        analysis_revision,
                        upload_signature,
                    )
                    if financial_columns else None
                )
                financial_base = (
                    _cached_financial_features(financial_frame)
                    if financial_frame is not None else None
                )
                progress.update(label="Analysis complete", state="complete", expanded=False)
            st.session_state["_upload_pipeline_key"] = upload_pipeline_key
            st.session_state["_upload_pipeline"] = {
                "document": document_result,
                "intelligence": data_intelligence,
                "financial_columns": financial_columns,
                "financial_frame": financial_frame,
                "financial_base": financial_base,
            }
        else:
            cached_upload = st.session_state["_upload_pipeline"]
            document_result = cached_upload["document"]
            data_intelligence = cached_upload["intelligence"]
            financial_columns = cached_upload["financial_columns"]
            financial_frame = cached_upload["financial_frame"]
            financial_base = cached_upload["financial_base"]
        if page == "Data Intelligence":
            _render_data_intelligence(
                document_result,
                data_intelligence,
                uploaded.name,
                analysis_revision,
                upload_signature,
                financial_columns,
            )
            st.stop()
        if not financial_columns:
            st.info("This upload has no recognized financial statement fields. Use Data Intelligence for its profile, risk patterns, and reports.")
            st.stop()
        st.success(
            f"File successfully loaded. AI Data Detective completed. "
            f"{document_result.status} Source: {document_result.source_type}."
        )
        st.markdown("<div class='panel-header'><h3>AI Data Detective</h3></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='panel-header'><h3>Automatic Analysis Summary</h3></div>",
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown("**Key Metrics**")
            summary_cols = st.columns(4)
            summary_cols[0].metric("Analysis Mode", data_intelligence["analysis_mode"])
            summary_cols[1].metric("Dataset", data_intelligence["dataset_type"])
            summary_cols[2].metric("Records", f"{len(document_result.frame):,}")
            summary_cols[3].metric("Anomalies", f"{data_intelligence['anomaly_count']:,}")
            result_cols = st.columns(3)
            result_cols[0].metric(
                "ML Target",
                data_intelligence["targets"][0] if data_intelligence["targets"] else "Not detected",
            )
            result_cols[1].metric(
                "Risk Prediction",
                "Model evaluated"
                if data_intelligence["prediction_available"]
                else data_intelligence["risk_prediction_summary"],
            )
            result_cols[2].metric("Alternative Analysis", data_intelligence["alternative_analysis"])
            if data_intelligence["targets"] and not data_intelligence["prediction_available"]:
                st.caption(
                    "A potential target was detected, but the available labeled data did not support "
                    "a held-out model evaluation. Statistical pattern analysis is shown instead."
                )
            elif not data_intelligence["targets"]:
                st.caption(
                    "No explicit default target was found. Anomaly and risk-pattern analysis was performed instead."
                )
            else:
                st.caption(
                    "Predicted Risk refers to supervised model evaluation. It is exploratory and is not a "
                    "calibrated probability of default."
                )
        st.markdown("<div class='panel-header'><h3>Executive Summary</h3></div>", unsafe_allow_html=True)
        st.caption(data_intelligence["executive_summary"])
        st.markdown("<div class='panel-header'><h3>Data Intelligence Dashboard</h3></div>", unsafe_allow_html=True)
        st.markdown("<div class='panel-header'><h3>Dataset Overview</h3></div>", unsafe_allow_html=True)
        overview_cols = st.columns(3)
        overview_cols[0].metric("Rows", f"{len(document_result.frame):,}")
        overview_cols[1].metric("Columns", f"{len(document_result.frame.columns):,}")
        overview_cols[2].metric("Missing values", f"{int(document_result.frame.isna().sum().sum()):,}")
        overview_cols = st.columns(3)
        overview_cols[0].metric("Duplicate records", f"{data_intelligence['duplicate_count']:,}")
        overview_cols[1].metric("Numeric variables", len(data_intelligence["numeric_columns"]))
        overview_cols[2].metric("Categorical variables", len(data_intelligence["categorical_columns"]))
        st.markdown("<div class='panel-header'><h3>Visual Analytics</h3></div>", unsafe_allow_html=True)
        if data_visualizations:
            for section, chart_items in data_visualizations.items():
                st.markdown(f"**{section}**")
                for chart_index in range(0, len(chart_items), 2):
                    chart_cols = st.columns(2)
                    for chart_col, chart_item in zip(
                        chart_cols,
                        chart_items[chart_index:chart_index + 2],
                        strict=False,
                    ):
                        with chart_col:
                            st.plotly_chart(
                                chart_item["figure"],
                                width="stretch",
                                config={"displayModeBar": False},
                                key=f"data_intelligence_{section}_{chart_index}_{chart_item['title']}",
                            )
        else:
            st.info(
                "No charts could be generated from the available non-constant values. "
                "The data profile and quality summary are still available below."
            )
        with st.expander("Data Profile and Detected Variables", expanded=not financial_columns):
            st.dataframe(data_intelligence["profile"], width="stretch", hide_index=True)
            if data_intelligence["identifiers"]:
                st.caption("Potential identifiers: " + ", ".join(data_intelligence["identifiers"]))
            if data_intelligence["targets"]:
                st.caption("Potential target variables: " + ", ".join(data_intelligence["targets"]))
            st.caption(
                f"Numeric: {', '.join(data_intelligence['numeric_columns']) or 'none'} | "
                f"Categorical: {', '.join(data_intelligence['categorical_columns']) or 'none'} | "
                f"Date/time: {', '.join(data_intelligence['date_columns']) or 'none'}"
            )
        if data_intelligence["findings"]:
            with st.expander("Risk / Anomaly Analysis, Early Warnings, and Recommendations", expanded=not financial_columns):
                for finding in data_intelligence["findings"]:
                    st.write(f"- {finding}")
                if not data_intelligence["correlations"].empty:
                    st.dataframe(data_intelligence["correlations"], width="stretch", hide_index=True)
                if not data_intelligence["statistics"].empty:
                    st.dataframe(data_intelligence["statistics"], width="stretch", hide_index=True)
                if not data_intelligence["categorical_summary"].empty:
                    st.markdown("**Categorical distributions**")
                    st.dataframe(data_intelligence["categorical_summary"], width="stretch", hide_index=True)
                if not data_intelligence["target_distribution"].empty:
                    st.markdown("**Observed target / outcome distribution**")
                    st.dataframe(data_intelligence["target_distribution"], width="stretch", hide_index=True)
                if not data_intelligence["high_risk_groups"].empty:
                    st.markdown("**High-risk groups from observed target labels**")
                    st.dataframe(data_intelligence["high_risk_groups"], width="stretch", hide_index=True)
                if not data_intelligence["trends"].empty:
                    st.markdown("**Observed trends**")
                    st.dataframe(data_intelligence["trends"], width="stretch", hide_index=True)
                if not data_intelligence["segments"].empty:
                    st.markdown("**Exploratory numeric segments**")
                    st.dataframe(data_intelligence["segments"], width="stretch", hide_index=True)
                st.markdown(
                    f"**Risk / Anomaly Analysis: {data_intelligence['analysis_result_title']}**"
                )
                st.caption(
                    "Predicted Risk = supervised model output. Anomaly = unusual statistical pattern. "
                    "Risk Indicator = a data-derived business pattern. An anomaly or indicator is not, by itself, "
                    "a default, fraud, or misconduct determination."
                )
                st.dataframe(data_intelligence["risk_results"].head(100), width="stretch", hide_index=True)
                model = data_intelligence["model"]
                if data_intelligence["prediction_available"]:
                    st.markdown("**Supervised Risk Prediction — hold-out evaluation**")
                    st.write(f"Model: {model['model']}")
                    st.write(f"Target: {model['target']}")
                    model_metrics = st.columns(3)
                    if "accuracy" in model:
                        model_metrics[0].metric("Accuracy", f"{model['accuracy']:.1%}")
                        model_metrics[1].metric("Balanced Accuracy", f"{model['balanced_accuracy']:.1%}")
                    else:
                        model_metrics[0].metric("Mean Absolute Error", f"{model['mae']:.3g}")
                        model_metrics[1].metric("R²", f"{model['r2']:.3f}")
                    model_metrics[2].metric("Evaluation records", f"{model['test_rows']:,}")
                    if model.get("top_features"):
                        st.markdown("**Top model features**")
                        st.dataframe(
                            pd.DataFrame(model["top_features"]).rename(columns={
                                "feature": "Feature",
                                "importance": "Relative importance",
                            }),
                            width="stretch",
                            hide_index=True,
                        )
                    st.caption(model.get("limitation", "Exploratory hold-out evaluation; validate before operational use."))
                else:
                    st.markdown(f"**{data_intelligence['analysis_result_title']}**")
                    if not data_intelligence["targets"]:
                        st.info(
                            "No explicit default target was found. Anomaly and risk-pattern analysis was performed instead."
                        )
                    else:
                        st.info(
                            "A candidate target was found, but a reliable hold-out model could not be evaluated. "
                            "Anomaly and statistical risk-pattern analysis is shown instead."
                        )
                if not data_intelligence["concentration"].empty:
                    st.dataframe(data_intelligence["concentration"], width="stretch", hide_index=True)
                st.markdown("**Recommendations**")
                for recommendation in data_intelligence["recommendations"]:
                    st.write(f"- {recommendation}")
                st.markdown("**Early Warnings**")
                for warning in data_intelligence["early_warnings"]:
                    st.write(f"- {warning}")
        frame = financial_frame
        for warning in document_result.warnings:
            st.caption(warning)
        st.caption("Document contents are processed locally; verify extracted figures against the source statement.")
        is_demo = False
    except Exception:
        st.error("Dataset analysis failed. Confirm the file is readable, then use Re-analyze to try again.")
        st.stop()
else:
    uploaded_content = b""
    upload_signature = "synthetic-demo"
    analysis_revision = 0
    upload_pipeline_key = ("synthetic-demo",)
    if st.session_state.get("_demo_pipeline_key") != upload_pipeline_key:
        frame = make_demo_data()
        financial_base = _cached_financial_features(frame)
        st.session_state["_demo_pipeline_key"] = upload_pipeline_key
        st.session_state["_demo_pipeline"] = {
            "frame": frame,
            "financial_base": financial_base,
        }
    else:
        demo_pipeline = st.session_state["_demo_pipeline"]
        frame = demo_pipeline["frame"]
        financial_base = demo_pipeline["financial_base"]
    is_demo = True
    st.caption("Using synthetic illustrative demo data.")

if page == "Data Intelligence":
    st.info("Upload and analyze a dataset to open Data Intelligence.")
    st.stop()

bundle = get_bundle()
features, flags = financial_base
dataset_key = repr(upload_pipeline_key)
_register_dataset(dataset_key, frame, features, flags)
_monitor_label = (
    f"{uploaded.name} · revision {analysis_revision}" if uploaded is not None else "Synthetic demo data"
)
if dataset_key not in st.session_state.setdefault("_monitor_recorded", set()):
    st.session_state["_monitor_recorded"].add(dataset_key)
    _record_monitor_snapshot(
        _monitor_label,
        uploaded.name if uploaded is not None else "synthetic-demo",
        dataset_key,
        frame,
        features,
        data_intelligence,
    )
if "company_id" in frame:
    company_values = frame["company_id"].astype(str)
    company_options = company_values.drop_duplicates().tolist()
else:
    company_values = pd.Series(["All records"] * len(frame), index=frame.index)
    company_options = ["All records"]

with selector_cols[0]:
    selected_company = st.selectbox("Company", company_options, key="company_filter")

if "company_id" in frame:
    company_rows = [
        position
        for position, company in enumerate(company_values)
        if company == selected_company
    ]
else:
    company_rows = list(range(len(frame)))

with selector_cols[1]:
    row_index = st.selectbox(
        "Period",
        options=company_rows,
        format_func=lambda index: str(
            frame.iloc[index].get("period", f"Record {index + 1}")
        ),
        key="period_filter",
    )

def selected_assessment() -> dict:
    if model_available_for_record:
        result = predict_financial_health(
            selected,
            bundle=bundle,
            include_explanations=False,
            features=selected_features,
        )
        result["method"] = "Existing ML model"
        result["health_score"] = max(
            0.0,
            min(100.0, (1 - float(result["distress_probability"])) * 100.0),
        )
    else:
        result = rule_based_assessment(frame, row_index, financial_analysis)
        probability = result["distress_probability"]
        result["health_score"] = (
            max(0.0, min(100.0, (1 - probability) * 100.0))
            if probability is not None
            else None
        )
    result["coverage_percent"] = financial_analysis["coverage_percent"]
    result["coverage_label"] = (
        f"{financial_analysis['coverage_count']}/{financial_analysis['coverage_total']} "
        f"core fields ({financial_analysis['coverage_percent']:.0f}%)"
    )
    return result


selection_cache_key = (upload_pipeline_key, int(row_index))
selection_cache = st.session_state.setdefault("_financial_selection_cache", {})
if selection_cache_key in selection_cache:
    cached_selection = selection_cache[selection_cache_key]
    selected = cached_selection["selected"]
    selected_features = cached_selection["selected_features"]
    financial_analysis = cached_selection["financial_analysis"]
    active_warning_signals = cached_selection["active_warning_signals"]
    recommendations = cached_selection["recommendations"]
    model_available_for_record = cached_selection["model_available"]
    current_assessment = cached_selection["assessment"]
else:
    selected, selected_features = frame.iloc[[row_index]], features.iloc[[row_index]]
    financial_analysis = _cached_financial_analysis(dataset_key, row_index)
    active_warning_signals = flags.iloc[row_index][flags.iloc[row_index]].index.tolist()
    recommendations = recommendations_for(
        frame.iloc[row_index],
        financial_analysis,
        active_warning_signals,
    )
    model_available_for_record = has_sufficient_ml_data(frame, row_index)
    current_assessment = selected_assessment()
    selection_cache[selection_cache_key] = {
        "selected": selected,
        "selected_features": selected_features,
        "financial_analysis": financial_analysis,
        "active_warning_signals": active_warning_signals,
        "recommendations": recommendations,
        "model_available": model_available_for_record,
        "assessment": current_assessment,
    }
    if len(selection_cache) > 24:
        oldest_key = next(iter(selection_cache))
        selection_cache.pop(oldest_key, None)

portfolio_analysis = None
if (
    data_intelligence is not None
    and data_intelligence["dataset_type"] in {"Customer dataset", "Loan/credit dataset"}
):
    portfolio_cache_key = upload_pipeline_key
    portfolio_cache = st.session_state.setdefault("_portfolio_analysis_cache", {})
    if portfolio_cache_key not in portfolio_cache:
        portfolio_cache[portfolio_cache_key] = analyze_credit_portfolio(document_result.frame)
    portfolio_analysis = portfolio_cache[portfolio_cache_key]

if portfolio_analysis is not None and page == "Executive Overview":
    portfolio_summary = portfolio_analysis["summary"]
    st.markdown("<div class='panel-header'><h3>Portfolio Risk Summary</h3></div>", unsafe_allow_html=True)
    portfolio_metrics = st.columns(4)
    portfolio_metrics[0].metric("Borrower / loan records", portfolio_summary["record_count"])
    portfolio_metrics[1].metric(
        "Observed default rate",
        f"{portfolio_summary['observed_default_rate']:.1%}"
        if portfolio_summary["observed_default_rate"] is not None
        else "Not available",
    )
    portfolio_metrics[2].metric(
        "Average credit score",
        f"{portfolio_summary['average_credit_score']:.0f}"
        if portfolio_summary["average_credit_score"] is not None
        else "Not available",
    )
    portfolio_metrics[3].metric(
        "Total loan amount",
        f"{portfolio_summary['total_loan_amount']:,.2f}"
        if portfolio_summary["total_loan_amount"] is not None
        else "Not available",
    )
    st.caption(
        f"Observed default status for {portfolio_summary['observed_default_coverage']} of "
        f"{portfolio_summary['record_count']} records. Portfolio bands use observed statuses "
        "and transparent credit-score thresholds; they are not model-generated default probabilities."
    )
    st.dataframe(portfolio_analysis["customers"], width="stretch", hide_index=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown("<div class='eyebrow'>AI-POWERED MSME FINANCIAL INTELLIGENCE</div>", unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def disclaimer() -> None:
    st.markdown("<div class='risk-note'>Analytical research estimate only. Not a guaranteed prediction, lending recommendation, or financial decision.</div>", unsafe_allow_html=True)


def show_risk() -> dict:
    result = current_assessment
    left, middle, right = st.columns(3)
    risk_value = result.get("distress_probability")
    risk_label = (
        f"{risk_value:.1%}" if risk_value is not None else "Insufficient data"
    )
    left.metric(
        "Estimated distress probability" if model_available_for_record else "Rule-based risk index",
        risk_label,
    )
    risk_class = _state_color(result["risk_category"])
    middle.markdown(
        f"<div class='risk-category-card' role='group' aria-label='Risk category: "
        f"{result['risk_category']}'><div class='risk-category-label'>Risk category</div>"
        f"<div class='risk-category-value {risk_class}'>{result['risk_category']}</div></div>",
        unsafe_allow_html=True,
    )
    right.metric(
        "Model confidence indicator" if model_available_for_record else "Data coverage",
        f"{result['confidence_indicator']:.0%}",
    )
    st.caption(result["confidence_note"])
    disclaimer()
    return result


if page in {"Executive Overview", "AI Copilot", "Scenario Simulator", "Credit Assessment"}:
    is_overview = page == "Executive Overview"
    result = current_assessment
    company_name = str(frame.iloc[row_index].get("company_id", "Selected Company"))
    health_score = result.get("health_score")
    current_ratio = selected_features.iloc[0].get("Current_Ratio", float("nan"))
    risk_class = _state_color(result["risk_category"])
    if is_overview:
        ratio_state = (
            "neutral"
            if pd.isna(current_ratio)
            else "healthy"
            if current_ratio >= 1
            else "watch"
        )
        health_label = (
            "Strong" if health_score >= 70
            else "Watch" if health_score >= 40
            else "Critical"
        ) if health_score is not None else "Insufficient data"
        metric_cols = st.columns(4)
        with metric_cols[0]:
            render_kpi_card(
                "Financial Health",
                f"{health_score:.0f}/100" if health_score is not None else "N/A",
                health_label if health_score is not None else "Insufficient data",
                risk_class,
            )
        with metric_cols[1]:
            probability = result.get("distress_probability")
            render_kpi_card(
                "Default Risk" if model_available_for_record else "Rule-Based Risk Index",
                f"{probability:.1%}" if probability is not None else "N/A",
                result["risk_category"],
                risk_class,
            )
        with metric_cols[2]:
            render_kpi_card(
                "Current Ratio",
                f"{current_ratio:.2f}x" if pd.notna(current_ratio) else "N/A",
                "Healthy" if ratio_state == "healthy" else "Watch" if ratio_state == "watch" else "Unavailable",
                ratio_state,
            )
        with metric_cols[3]:
            render_kpi_card(
                "AI Risk Status" if model_available_for_record else "Risk Status",
                result["risk_category"].upper(),
                "Stable" if risk_class == "healthy" else result["risk_category"],
                risk_class,
            )

        if not model_available_for_record and page in {"Executive Overview", "Credit Assessment"}:
            st.info(
                "The selected record does not contain enough observed financial fields for the ML model. "
                "This assessment uses transparent rules and is not a calibrated probability of default."
            )
        if uploaded is not None:
            st.markdown("<div class='panel-header'><h3>Extracted Financial Data</h3></div>", unsafe_allow_html=True)
            st.caption(
                f"Data coverage: {current_assessment['coverage_label']}. "
                f"Company: {company_name}; period: "
                f"{selected.iloc[0].get('period') if pd.notna(selected.iloc[0].get('period')) else 'Not identified'}."
            )
            st.dataframe(selected, width="stretch", hide_index=True)
            st.markdown("<div class='panel-header'><h3>Financial Ratios</h3></div>", unsafe_allow_html=True)
            ratios_frame = pd.DataFrame(
                [{"Ratio": label, "Value": value if value is not None else "Not available"}
                 for label, value in financial_analysis["ratios"].items()]
            ).astype({"Value": "string"})
            st.dataframe(ratios_frame, width="stretch", hide_index=True)

    if page == "Credit Assessment":
        render_credit_assessment_panel(company_name, result, selected_features, flags, row_index)
        disclaimer()

    if page in {"Executive Overview", "AI Copilot"}:
        st.markdown(
            "<div class='panel-header'><h3>AI Credit Copilot</h3></div>"
            "<div class='creator-credit'>Ask anything about this company's financial health.</div>",
            unsafe_allow_html=True,
        )
        copilot_context = build_credit_context(frame, selected, selected_features, flags, row_index, result)
        transcript = st.session_state.setdefault("copilot_history", [])
        prompt_chips = suggested_questions(copilot_context)
        chip_cols = st.columns(3)
        preset = None
        for position, prompt in enumerate(prompt_chips):
            with chip_cols[position % 3]:
                if st.button(prompt, key=f"copilot_chip_{position}", width="stretch"):
                    preset = prompt
        if preset is not None:
            # Set before the widget is created; Streamlit forbids writing to a
            # widget's session-state key after it has been instantiated.
            st.session_state["copilot_question"] = preset
        with st.form("credit_copilot"):
            question = st.text_area(
                "Question",
                value=f"Why is this company currently {result['risk_category'].lower()}?",
                key="copilot_question",
                help="Answers use the selected company-period's financials, model output, and warning signals.",
                height=76,
                label_visibility="collapsed",
                placeholder="Ask about this company's financial health...",
            )
            submitted = st.form_submit_button("Ask AI")
        response_signature = (
            str(selected_company),
            row_index,
            float(result["distress_probability"] or 0.0),
            uploaded.name if uploaded is not None else "synthetic-demo",
            uploaded.size if uploaded is not None else 0,
            question,
        )
        if preset is not None:
            response_signature = (*response_signature[:-1], preset)
        if submitted or preset is not None or st.session_state.get("credit_copilot_signature") != response_signature:
            history = [entry["question"] for entry in transcript]
            response = generate_credit_copilot_response(question, copilot_context, history=history)
            transcript.append({"question": question, "answer": response, "company": str(selected_company)})
            del transcript[:-12]
            st.session_state["credit_copilot_response"] = response
            st.session_state["credit_copilot_signature"] = response_signature
        if st.session_state.get("credit_copilot_response"):
            st.markdown(
                f"<div class='copilot-output'>{escape(st.session_state['credit_copilot_response'])}</div>",
                unsafe_allow_html=True,
            )
        if len(transcript) > 1:
            with st.expander(f"Conversation history ({len(transcript)})"):
                for entry in reversed(transcript):
                    st.markdown(f"**Q · {escape(str(entry['question']))}**")
                    st.caption(escape(str(entry["answer"])))
            if st.button("Clear conversation", key="copilot_clear"):
                transcript.clear()
                st.session_state.pop("credit_copilot_response", None)
                st.session_state.pop("credit_copilot_signature", None)

    if page in {"Executive Overview", "AI Copilot"}:
        if model_available_for_record:
            render_ai_risk_interpretation(result, selected_features.iloc[0], active_warning_signals)
        else:
            st.markdown("<div class='panel-header'><h3>Risk Assessment Summary</h3></div>", unsafe_allow_html=True)
            risks = [factor["feature"] for factor in result.get("top_risk_factors", [])]
            coverage_pct = financial_analysis["coverage_percent"]
            explanation = (
                f"Observed financial data coverage is {coverage_pct:.0f}%. "
                f"The transparent rules classify this record as {result['risk_category'].lower()}."
            )
            if risks:
                explanation += f" Risk drivers supported by available figures include {', '.join(risks[:3])}."
            st.markdown(
                f"<div class='risk-summary-card {_state_color(result['risk_category'])}'>"
                f"<div class='risk-state'>{escape(result['risk_category'].upper())}</div>"
                f"<p>{escape(explanation)}</p>"
                f"<strong>{escape(result['method'])}</strong></div>",
                unsafe_allow_html=True,
            )

    if is_overview or page == "Credit Assessment":
        st.markdown("<div class='panel-header'><h3>Recommendations</h3></div>", unsafe_allow_html=True)
        for recommendation in recommendations:
            st.write(f"- {recommendation}")
        credit_report_cache = st.session_state.setdefault("_credit_report_cache", {})
        credit_report_key = (selection_cache_key, company_name, tuple(company_rows))
        report = credit_report_cache.setdefault(credit_report_key, {})
        report_cols = st.columns(2)
        with report_cols[0]:
            if st.button("Generate Credit Assessment PDF", key="generate_credit_assessment_pdf"):
                try:
                    report["pdf"] = create_credit_assessment_pdf(
                        company_name,
                        selected,
                        financial_analysis,
                        result,
                        active_warning_signals,
                        recommendations,
                    )
                except Exception as error:
                    st.warning(f"PDF report generation failed: {error}")
            if "pdf" in report:
                st.download_button(
                    "Download Credit Assessment PDF",
                    data=report["pdf"],
                    file_name=f"{company_name.replace(' ', '_')}_credit_assessment.pdf",
                    mime="application/pdf",
                    key="credit_assessment_pdf",
                    width="stretch",
                )
        with report_cols[1]:
            if st.button("Prepare Excel Analysis", key="prepare_financial_analysis_xlsx"):
                try:
                    report["excel"] = create_excel_analysis(
                        frame.iloc[company_rows] if uploaded is not None else selected,
                        financial_analysis,
                        result,
                        active_warning_signals,
                        recommendations,
                    )
                except Exception as error:
                    st.warning(f"Excel report generation failed: {error}")
            if "excel" in report:
                st.download_button(
                    "Download Excel Analysis",
                    data=report["excel"],
                    file_name=f"{company_name.replace(' ', '_')}_financial_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="financial_analysis_xlsx",
                    width="stretch",
                )
            if len(credit_report_cache) > 16:
                credit_report_cache.pop(next(iter(credit_report_cache)), None)

    if is_overview or page == "AI Copilot":
        st.markdown("<div class='panel-header'><h3>Top Risk Drivers</h3></div>", unsafe_allow_html=True)
        if model_available_for_record:
            st.markdown(
                "<div class='driver-legend'>Positive contribution increases distress output · Negative contribution reduces it</div>",
                unsafe_allow_html=True,
            )
            render_shap_drivers(result)
            if not result.get("top_risk_factors") and not result.get("protective_factors"):
                st.info("Open Explainable AI to calculate on-demand SHAP drivers.")
        elif result.get("top_risk_factors"):
            st.caption("Rule-based indicators are derived from observed values and transparent thresholds; they are not SHAP contributions.")
            st.dataframe(
                pd.DataFrame(result["top_risk_factors"]).rename(
                    columns={"feature": "Observed risk driver", "contribution": "Rule weight"}
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No risk driver could be calculated from the available fields.")

    if page == "Scenario Simulator":
        page_header("Scenario Simulator", "Explore how adjusted financial assumptions change the model's estimate.")
        st.caption("Adjust one-period assumptions to see how the existing risk model responds. Margin change is in percentage points; revenue and debt changes are relative percentages.")
        with st.form("credit_scenario"):
            scenario_cols = st.columns(3)
            with scenario_cols[0]:
                revenue_change = st.slider("Revenue change", min_value=-20, max_value=20, value=0, step=1, format="%d%%")
            with scenario_cols[1]:
                margin_change = st.slider("Operating margin change", min_value=-10, max_value=10, value=0, step=1, format="%d pp")
            with scenario_cols[2]:
                debt_change = st.slider("Debt change", min_value=-20, max_value=20, value=0, step=1, format="%d%%")
            run_scenario = st.form_submit_button("Run AI simulation")

        if run_scenario:
            try:
                scenario_data = apply_scenario_adjustments(
                    selected,
                    revenue_change=revenue_change / 100,
                    operating_margin_change=margin_change / 100,
                    debt_change=debt_change / 100,
                )
                scenario_feature_frame = engineer_features(scenario_data)
                scenario_features = scenario_feature_frame.iloc[0]
                scenario_analysis = analyze_financials(scenario_data, 0, scenario_feature_frame)
                if model_available_for_record:
                    scenario_result = predict_financial_health(
                        scenario_data,
                        bundle=bundle,
                        include_explanations=False,
                        features=scenario_feature_frame,
                    )
                    scenario_result["method"] = "Existing ML model"
                    scenario_result["health_score"] = (
                        1 - scenario_result["distress_probability"]
                    ) * 100
                else:
                    scenario_result = rule_based_assessment(
                        scenario_data,
                        0,
                        scenario_analysis,
                    )
                    if scenario_result["distress_probability"] is not None:
                        scenario_result["health_score"] = (
                            1 - scenario_result["distress_probability"]
                        ) * 100
                scenario_warnings = early_warning_indicators(scenario_data).iloc[0]
                scenario_signals = scenario_warnings[scenario_warnings].index.tolist()
                if model_available_for_record:
                    try:
                        scenario_explanation = explain_prediction(
                            bundle,
                            scenario_data,
                            features=scenario_feature_frame,
                        )
                        scenario_result["top_risk_factors"] = scenario_explanation["risk_factors"]
                        scenario_result["protective_factors"] = scenario_explanation["protective_factors"]
                    except Exception as error:
                        st.warning(f"SHAP scenario drivers are unavailable: {error}")

                probability_change_pp = (
                    (scenario_result["distress_probability"] - result["distress_probability"]) * 100
                    if scenario_result.get("distress_probability") is not None
                    and result.get("distress_probability") is not None
                    else None
                )
                scenario_metric, scenario_status = st.columns(2)
                scenario_metric.metric(
                    "Projected risk" if model_available_for_record else "Projected rule-based risk index",
                    f"{scenario_result['distress_probability']:.1%}" if scenario_result.get("distress_probability") is not None else "Insufficient data",
                    delta=f"{probability_change_pp:+.1f} percentage points" if probability_change_pp is not None else None,
                    delta_color="inverse",
                )
                scenario_status.metric(
                    "Current risk" if model_available_for_record else "Current rule-based risk index",
                    f"{result['distress_probability']:.1%}" if result.get("distress_probability") is not None else "Insufficient data",
                )
                if model_available_for_record:
                    render_ai_risk_interpretation(
                        scenario_result,
                        scenario_features,
                        scenario_signals,
                    )
                else:
                    st.info(
                        "Scenario results use the same transparent rules as the current sparse-data assessment; "
                        "they are not calibrated default probabilities."
                    )
                st.caption(
                    f"Applied assumptions: revenue {revenue_change:+d}%, "
                    f"operating margin {margin_change:+d} percentage points, "
                    f"debt {debt_change:+d}%."
                )
            except (TypeError, ValueError) as error:
                st.error(f"Scenario could not be calculated: {error}")
        disclaimer()

    if is_overview:
        if is_demo:
            st.info("Synthetic demo records and labels are illustrative only; they do not represent actual MSMEs.")

        st.markdown("<div class='panel-header'><h3>Financial trends</h3></div>", unsafe_allow_html=True)
        if "company_id" in frame and "period" in frame:
            series = trend_data(frame, selected.iloc[0]["company_id"])
            if not series.empty:
                trend_cols = st.columns(2)
                revenue_chart = px.line(
                    series,
                    x="period",
                    y="Revenue",
                    markers=True,
                    color_discrete_sequence=["#2563EB"],
                )
                revenue_chart.update_layout(title="Revenue", yaxis_title=None)
                with trend_cols[0]:
                    render_chart(revenue_chart, height=220, margin={"t": 24, "r": 10, "b": 38, "l": 10})

                liquidity_chart = px.line(
                    series,
                    x="period",
                    y="Current_Ratio",
                    markers=True,
                    color_discrete_sequence=["#06B6D4"],
                )
                liquidity_chart.update_layout(title="Current ratio", yaxis_title=None)
                with trend_cols[1]:
                    render_chart(liquidity_chart, height=220, margin={"t": 24, "r": 10, "b": 38, "l": 10})
            else:
                st.info("Trend history is unavailable for the selected company.")
        else:
            st.info("Add company and period fields to the CSV to view financial trends.")

        render_warning_signals(active_warning_signals)
        with st.expander("How this analysis works"):
            st.markdown(
                """
                CSV / financial data → financial analysis → ML risk model → SHAP explainability
                → early-warning engine → AI Credit Copilot

                The Copilot provides risk explanation, recommendations, scenario analysis, and a credit assessment.
                """
            )
        disclaimer()

elif page == "Portfolio Screening":
    render_screening_page(frame, features, flags, bundle)
    disclaimer()

elif page == "Compare Records":
    render_comparison_page(frame, features, flags, bundle, company_rows)

elif page == "Data Explorer":
    render_data_explorer(frame)
    disclaimer()

elif page == "MSME Financial Health":
    page_header("MSME Financial Health", "Extracted statement inputs, available-data coverage, and derived ratios.")
    financial_summary = pd.concat(
        [selected.reset_index(drop=True), selected_features.reset_index(drop=True)],
        axis=1,
    ).T.rename(columns={0: "Value"}).astype(str)
    st.dataframe(financial_summary, width="stretch")
    st.caption(f"Core-field coverage: {current_assessment['coverage_label']}")
    st.dataframe(
        pd.DataFrame(
            [{"Ratio": label, "Value": value if value is not None else "Not available"}
             for label, value in financial_analysis["ratios"].items()]
        ).astype({"Value": "string"}),
        width="stretch",
        hide_index=True,
    )
    st.caption("Ratios use bounded calculations; zero denominators are treated as missing. Inventory days use a revenue proxy when COGS is unavailable.")
    disclaimer()

elif page == "Risk Prediction":
    page_header(
        "Risk Prediction",
        "Estimated probability, category, and the factors behind this assessment.",
    )
    result = show_risk()
    health_score = result.get("health_score")
    if health_score is None:
        st.info("There is not enough numeric financial data to calculate a health score.")
    else:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            number={"suffix": "%", "font": {"color": "#F8FAFC", "size": 42}},
            title={"text": "Financial health score", "font": {"color": "#F8FAFC", "size": 18}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": "#94A3B8"},
                "bar": {"color": "#10B981", "thickness": 0.25},
                "bgcolor": "#0D1B2A",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(239, 68, 68, 0.18)"},
                    {"range": [40, 70], "color": "rgba(245, 158, 11, 0.18)"},
                    {"range": [70, 100], "color": "rgba(16, 185, 129, 0.18)"},
                ],
                "threshold": {
                    "line": {"color": "#F8FAFC", "width": 3},
                    "thickness": 0.8,
                    "value": health_score,
                },
            },
        ))
        render_chart(gauge, height=290, margin={"t": 42, "b": 12, "l": 12, "r": 12})
    st.caption(
        "Rule-based assessments are transparent heuristic indices, not calibrated probabilities of default."
        if not model_available_for_record
        else "Health score is calculated as 100% minus the model's estimated distress probability; it is not a separate diagnosis."
    )
    risk, protective = st.columns(2)
    with risk:
        st.subheader("Risk-increasing factors")
        st.dataframe(pd.DataFrame(result["top_risk_factors"]), width="stretch", hide_index=True)
    with protective:
        st.subheader("Risk-reducing factors")
        st.dataframe(pd.DataFrame(result["protective_factors"]), width="stretch", hide_index=True)

elif page == "Explainable AI":
    page_header("Explainable AI", "How the selected model behaves globally and for this individual record.")
    if not model_available_for_record:
        st.info("There are not enough observed financial fields for model-based SHAP. The available rule-based drivers are shown instead.")
        if current_assessment.get("top_risk_factors"):
            st.dataframe(
                pd.DataFrame(current_assessment["top_risk_factors"]).rename(
                    columns={"feature": "Observed risk driver", "contribution": "Rule weight"}
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No rule-based risk drivers were triggered by the available values.")
    else:
        try:
            local = _session_shap(
                f"{upload_pipeline_key!r}:{row_index}",
                bundle,
                selected,
                selected_features,
            )
            st.subheader("Why did the model assign this risk?")
            st.caption("Positive SHAP contributions increase the model output for distress; negative contributions reduce it. Associations are not causal.")
            bars = local["contributions"].copy()
            bars["direction"] = bars["shap_value"].map(lambda value: "Risk increased" if value > 0 else "Risk reduced")
            fig = px.bar(
                bars.sort_values("shap_value"),
                x="shap_value",
                y="feature",
                color="direction",
                orientation="h",
                color_discrete_map={"Risk increased": "#EF4444", "Risk reduced": "#2563EB"},
                title="Individual SHAP contributions",
            )
            fig.update_layout(
                legend_title_text="Contribution",
                xaxis_title="Contribution to distress score",
                yaxis_title=None,
            )
            render_chart(fig, height=400, margin={"t": 58, "r": 12, "b": 52, "l": 120})
            importance = _session_global_shap(
                repr(upload_pipeline_key),
                bundle,
                frame,
            )
            global_fig = px.bar(
                importance.head(15).sort_values("mean_abs_shap"),
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                title="Global feature importance · mean absolute SHAP",
                color_discrete_sequence=["#3B82F6"],
            )
            render_chart(global_fig, height=400, margin={"t": 58, "r": 12, "b": 48, "l": 120})
        except Exception as error:
            st.error(f"SHAP explanation could not be calculated in this environment: {error}")
    disclaimer()

elif page == "Early-Warning Indicators":
    page_header("Early-Warning Indicators", "Transparent heuristics that flag potentially deteriorating conditions.")
    st.caption("Rules are configurable research heuristics, not learned predictions or universal thresholds.")
    selected_flags = flags.iloc[[row_index]].T.rename(columns={row_index: "Triggered"})
    feature_row = selected_features.iloc[0]
    warning_requirements = {
        "Rapid revenue decline": ("Sales_Growth",),
        "Increasing leverage": ("Debt_to_Assets",),
        "Falling liquidity": ("Current_Ratio",),
        "Deteriorating margins": ("EBITDA_Margin", "Net_Profit_Margin"),
        "Negative operating cash flow": ("Cash_Flow_Operations",),
        "Increasing receivable days": ("Receivable_Days",),
        "Falling interest coverage": ("Interest_Coverage",),
    }
    selected_flags["Status"] = [
        "Review" if bool(triggered)
        else "Not triggered" if any(pd.notna(feature_row.get(name)) for name in warning_requirements.get(signal, ()))
        else "Not assessed"
        for signal, triggered in selected_flags["Triggered"].items()
    ]
    st.dataframe(selected_flags.drop(columns="Triggered"), width="stretch")
    st.caption(f"Core-field coverage: {current_assessment['coverage_label']}")
    if "company_id" in frame and "period" in frame:
        series = trend_data(frame, selected.iloc[0]["company_id"])
        if len(series) > 1:
            metric = st.selectbox("Historical measure", ["Revenue", "Current_Ratio", "Debt_to_Assets", "EBITDA_Margin", "Interest_Coverage"])
            fig = px.line(
                series,
                x="period",
                y=metric,
                markers=True,
                title=f"{metric.replace('_', ' ')} over time",
                color_discrete_sequence=["#2563EB"],
            )
            render_chart(fig)
        else:
            st.info("At least two dated company-period records are needed for a trend chart.")
    else:
        st.info("Add company_id and period columns to a longitudinal CSV to view historical trends.")
    disclaimer()

elif page == "Model Performance":
    page_header("Model Performance", "Held-out and cross-validated metrics for the selected model bundle.")
    report = bundle["report"]
    st.caption(f"Selected: {report['selected_model']} · {report['selection_metric']}")
    st.info(report["evaluation_note"])
    st.dataframe(pd.DataFrame([report["test_metrics"]]).T.rename(columns={0: "Held-out test"}), width="stretch")
    st.subheader("Cross-validation comparison")
    st.dataframe(pd.DataFrame(report["cv_metrics"]).T.style.format("{:.3f}"), width="stretch")
    if report.get("fairness"):
        st.subheader("Audit-only group diagnostics")
        st.dataframe(pd.DataFrame(report["fairness"]).T, width="stretch")
        st.caption("Group attributes are excluded from model inputs. Descriptive diagnostics may be unstable for small samples.")
    else:
        st.caption("No held-out protected-group diagnostics are stored in this model bundle.")
    disclaimer()

elif page == "Model Monitoring":
    render_monitoring_page(bundle)

elif page == "Methodology":
    page_header("Methodology & Research Mode", "Definitions, evaluation choices, limitations, and responsible interpretation.")
    st.subheader("Dataset")
    st.write("The included company-period dataset is synthetic, generated with a fixed seed, and labeled by a noisy illustrative rule. It is not external evidence or real default data.")
    st.subheader("Model methodology")
    st.write("Logistic Regression, class-weighted Random Forest, and XGBoost are compared. Median imputation and scaling are fitted inside each model pipeline. Group-disjoint stratified splits are used when company identifiers exist. The selected model maximizes mean training-fold ROC-AUC.")
    st.subheader("Evaluation metrics")
    st.write("Accuracy, precision, recall, F1, ROC-AUC, and average precision (PR-AUC). These depend on the label, sample, split, and 0.5 classification threshold.")
    st.subheader("Fairness and sources of bias")
    st.write("Protected attributes are never model inputs. If supplied for a legitimate audit, held-out group label rates, true-positive rates, and false-positive rates are descriptive only. Selection bias, historical decisions, missingness, sector/geographic representation, and label construction can all produce biased estimates.")
    st.subheader("Limitations and ethical use")
    st.write("The score is not causal, calibrated uncertainty, a guaranteed prediction, lending advice, or a decision engine. Validate definitions, data rights, external and temporal performance, subgroup behavior, and calibration with qualified reviewers before any consequential use. Provide human oversight and recourse.")
    st.subheader("Feature definitions")
    st.dataframe(pd.DataFrame({"Feature": features.columns, "Definition": ["Raw numeric statement field" if name in frame.columns else "Engineered ratio; see project README" for name in features.columns]}), width="stretch", hide_index=True)
    st.subheader("Sparse-data risk index")
    st.write(
        "When fewer than four financial fields are observed, or fewer than two model-core fields are present, "
        "the app uses an uncalibrated rule-based index instead of the ML model. It starts at 10 index points; "
        "observed liquidity, leverage, profitability, coverage, growth, and cash-flow signals adjust the index, "
        "observed strengths can reduce it, and the result is bounded between 5 and 95 index points. "
        "The output is not a default probability."
    )
    st.dataframe(
        pd.DataFrame([
            {"Observed condition": "Current ratio below 1.0x", "Index adjustment": "+22 points"},
            {"Observed condition": "Current ratio below 1.2x", "Index adjustment": "+10 points"},
            {"Observed condition": "Debt / equity above 2.0x", "Index adjustment": "+16 points"},
            {"Observed condition": "Negative equity", "Index adjustment": "+20 points"},
            {"Observed condition": "Debt / assets at least 0.65x", "Index adjustment": "+18 points"},
            {"Observed condition": "Negative net profit margin", "Index adjustment": "+15 points"},
            {"Observed condition": "Negative return on assets", "Index adjustment": "+10 points"},
            {"Observed condition": "Interest coverage below 1.5x", "Index adjustment": "+12 points"},
            {"Observed condition": "Revenue growth below -15%", "Index adjustment": "+12 points"},
            {"Observed condition": "Negative operating cash flow", "Index adjustment": "+15 points"},
            {"Observed condition": "Observed strengths", "Index adjustment": "Up to -20 points"},
        ]),
        width="stretch",
        hide_index=True,
    )
    disclaimer()

elif page == "About":
    page_header("About Surojit Malakar", "Finance, research, operations, and technology in service of practical impact.")
    photo_column, intro_column = st.columns([1, 2], gap="large")
    with photo_column:
        st.markdown(
            "<div class='assessment-card'><div class='eyebrow'>CREDIT RISK AI</div>"
            "<h3>Surojit Malakar</h3><p>SkillseED India</p></div>",
            unsafe_allow_html=True,
        )
    with intro_column:
        st.subheader("A little about me")
        st.write(
            "I am a BBA (Honours with Research) student at Techno India University, "
            "Kolkata, specializing in Finance, Business Analytics, and Operations. "
            "My interests lie at the intersection of financial analysis, business "
            "strategy, operations, research, and social entrepreneurship."
        )
        st.write(
            "My long-term goal is to build a career where I can combine financial "
            "and analytical thinking with research, strategic decision-making, and "
            "practical business execution, while continuing to develop initiatives "
            "that create measurable economic and social impact."
        )

    st.subheader("Entrepreneurship & community impact")
    st.write(
        "I am the Founder of SkillseED India, a rural social enterprise focused on "
        "skill development, agribusiness education, and farmer empowerment. Through "
        "SkillseED India, I have worked with students and farmers across West Bengal "
        "and developed initiatives focused on practical education, skill "
        "development, and agricultural business opportunities."
    )
    st.write(
        "I am also the Founder and Developer of ZynoqIndia, where I have worked on "
        "an eco-tourism and open-access research publishing platform. This experience "
        "has allowed me to combine business strategy with technology, including "
        "product development, UI/UX, React, Vite, Supabase, and digital publishing systems."
    )

    st.subheader("Research & analytical frameworks")
    st.write(
        "My research work focuses on solving practical business and economic problems "
        "through structured analytical frameworks. I developed the Farm-to-Consumer "
        "(F2C 4.0) Model, which focuses on farmer entrepreneurship education, "
        "institutional market access, and cooperative business development."
    )
    st.write(
        "I have also developed Decision Paralysis Economics (DPE), an analytical "
        "framework examining the organizational and economic costs associated with "
        "delayed decision-making in environments affected by information overload "
        "and AI-mediated ambiguity."
    )
    st.write(
        "My academic research experience includes primary data collection, survey "
        "research, statistical analysis, SPSS-based reporting, and the development "
        "of analytical frameworks."
    )

    st.subheader("Experience & interests")
    st.write(
        "Alongside my research and entrepreneurial work, I have gained experience "
        "in HR operations, project management, financial analysis, business "
        "operations, and community development. My professional experience has "
        "allowed me to work on recruitment, process improvement, cross-functional "
        "coordination, research, and program management."
    )
    st.write(
        "I am particularly interested in finance, business analytics, operations "
        "strategy, agribusiness, financial decision-making, research, and "
        "technology-enabled business models."
    )