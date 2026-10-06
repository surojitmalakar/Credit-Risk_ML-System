"""Eight-page Streamlit research dashboard for MSME financial distress."""

from __future__ import annotations

import sys
import hashlib
from pathlib import Path

# Robust path resolution for Streamlit Cloud and local development
# Try multiple possible locations for the src directory
def _resolve_src_dir() -> Path:
    """Resolve the src directory path robustly across environments."""
    candidates = [
        Path(__file__).resolve().parent / "src",  # Standard: app.py in root, src/ sibling
        Path.cwd() / "src",                         # Streamlit Cloud: cwd is repo root
        Path(__file__).resolve().parent.parent / "src",  # Nested deployment
    ]
    for candidate in candidates:
        if candidate.exists() and (candidate / "msme_ews").exists():
            return candidate
    # Fallback: use the first candidate even if not found (will raise ImportError later with clear message)
    return candidates[0]

SRC_DIR = _resolve_src_dir()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from html import escape

from msme_ews.data import FINANCIAL_COLUMNS, validate_financial_data
from msme_ews.calibration import (
    DEFAULT_CUTOFFS,
    band_distribution,
    calibration_table,
    cutoff_summary,
    reliability_metrics,
    shifted_band_counts,
    threshold_table,
)
from msme_ews.credit_assessment import generate_risk_interpretation
from msme_ews.copilot import (
    build_credit_context,
    generate_credit_copilot_answer,
    generate_credit_copilot_response,
    generate_dataset_copilot_answer,
    generate_dataset_copilot_response,
    suggested_questions,
)
from msme_ews.data_intelligence import analyze_dataset
from msme_ews.data_visualization import build_data_visualizations
from msme_ews.documents import extract_financial_document
from msme_ews.early_warning import early_warning_indicators, trend_data
from msme_ews.explain import explain_prediction, global_importance
from msme_ews.financial_analysis import (
    analyze_financials,
    recommendations_for,
    rule_based_assessment,
)
from msme_ews.features import engineer_features
from msme_ews.llm import load_settings

# Streamlit Cloud can serve a partially refreshed revision, so a helper that moves
# between modules must never take the whole dashboard down. Calibration diagnostics
# degrade to an "unavailable" state instead of raising ImportError on startup.
try:
    from msme_ews.modeling import calibration_sample
except ImportError:  # pragma: no cover - depends on the deployed module revision
    try:
        from msme_ews.calibration import calibration_sample
    except ImportError:
        calibration_sample = None
from msme_ews.monitoring import (
    build_snapshot,
    drift_report,
    monitoring_alerts,
    prediction_stability,
    snapshot_table,
)
from msme_ews.notes import (
    NOTE_DECISIONS,
    add_note,
    audit_csv,
    audit_event,
    audit_frame,
    decision_counts,
    log_event,
    make_note,
    notes_csv,
    notes_frame,
    validate_note,
)
from msme_ews.portfolio import analyze_credit_portfolio
from msme_ews.portfolio_intel import portfolio_intelligence
from msme_ews.warning_detail import detailed_warnings, triggered_warnings
from msme_ews.whatif import available_whatif_fields, score_whatif
from msme_ews.reports import (
    create_credit_assessment_pdf,
    create_credit_risk_pdf,
    create_data_intelligence_excel,
    create_data_intelligence_pdf,
    create_early_warning_pdf,
    create_excel_analysis,
    create_executive_pdf,
    create_financial_health_pdf,
    create_screening_excel,
    create_screening_pdf,
)
from msme_ews.scenarios import (
    MAX_GRID_SCENARIOS,
    STRESS_FACTORS,
    STRESS_PRESETS,
    available_factors,
    breakeven_step,
    grid_size,
    run_stress_grid,
    summarize_stress,
    tornado_rows,
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
from msme_ews.theme import (
    BIO_SUMMARY,
    CAREER_INTENT,
    CRITICAL,
    ENTERPRENEURSHIP,
    EXPERIENCE_INTERESTS,
    G1,
    G3,
    G4,
    NEUTRAL,
    RESEARCH_FRAMEWORKS,
    STYLESHEET,
    TEXT,
    TEXT_MUTED,
    TEXT_STRONG,
    WHITE,
    about_block_html,
    about_tiles_html,
    band_colour,
    page_hero_html,
    profile_card_html,
    site_footer_html,
)

st.set_page_config(
    page_title="CREDIT RISK AI",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="auto",
)

# Optional free LLM for the AI Copilot. Reads LLM_API_BASE / LLM_API_KEY /
# LLM_MODEL / LLM_TIMEOUT from Streamlit secrets or the environment.
# Leave unconfigured to keep the transparent rule-based engine.
LLM_SETTINGS = load_settings()


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
_DATASET_LABELS: dict[str, str] = {}


def _register_dataset(
    dataset_key: str,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    flags: pd.DataFrame,
    label: str = "",
) -> None:
    """Hold a dataset in memory so cached work can key on a short string.

    Passing a DataFrame to ``st.cache_data`` re-hashes every cell on each rerun;
    keying on this identifier keeps repeated interaction work O(1) in the dataset.
    """
    _DATASET_REGISTRY[dataset_key] = (frame, features, flags)
    if label:
        _DATASET_LABELS[dataset_key] = label
    while len(_DATASET_REGISTRY) > _DATASET_REGISTRY_LIMIT:
        stale = next(iter(_DATASET_REGISTRY))
        _DATASET_REGISTRY.pop(stale)
        _DATASET_LABELS.pop(stale, None)


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


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_stress_grid(
    dataset_key: str,
    row_index: int,
    factor_keys: tuple[str, ...],
    bundle_signature: str,
) -> pd.DataFrame:
    frame, _, _ = _dataset_parts(dataset_key)
    factors = [factor for factor in STRESS_FACTORS if factor.key in factor_keys]
    return run_stress_grid(frame.iloc[[row_index]], factors, get_bundle())


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_tornado(dataset_key: str, row_index: int, bundle_signature: str) -> pd.DataFrame:
    frame, _, _ = _dataset_parts(dataset_key)
    factors = available_factors(frame.iloc[row_index])
    return tornado_rows(frame.iloc[[row_index]], factors, get_bundle())


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_calibration(dataset_key: str, frame_hash: str, bundle_signature: str) -> dict:
    frame, _, _ = _dataset_parts(dataset_key)
    if calibration_sample is None:  # an older deployed revision lacks the helper
        table = calibration_table([], [])
        return {
            "probabilities": [],
            "labels": [],
            "basis": "calibration diagnostics are unavailable in this deployment",
            "is_in_sample": False,
            "table": table,
            "metrics": reliability_metrics([], [], table),
        }
    sample = calibration_sample(get_bundle(), frame)
    table = calibration_table(sample["probabilities"], sample["labels"])
    return {
        **sample,
        "table": table,
        "metrics": reliability_metrics(sample["probabilities"], sample["labels"], table),
    }


def _cached_screening_pdf(
    dataset_key: str,
    bundle_signature: str,
    exposure_column: str,
) -> bytes:
    frame, _, _ = _dataset_parts(dataset_key)
    scores = _cached_screening_scores(dataset_key, bundle_signature)
    return create_screening_pdf(
        scores,
        screening_summary(scores),
        _dataset_label(dataset_key),
        banded_exposure(scores, frame, exposure_column) if exposure_column else None,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_screening_workbook(
    dataset_key: str,
    bundle_signature: str,
    exposure_column: str,
    notes_signature: str,
) -> bytes:
    frame, _, _ = _dataset_parts(dataset_key)
    scores = _cached_screening_scores(dataset_key, bundle_signature)
    notes = notes_frame(st.session_state.get("underwriter_notes", []))
    return create_screening_excel(
        scores,
        screening_summary(scores),
        banded_exposure(scores, frame, exposure_column) if exposure_column else None,
        notes if not notes.empty else None,
    )


def _dataset_label(dataset_key: str) -> str:
    """Human-readable name for a dataset cache key."""
    labels = _DATASET_LABELS
    return labels.get(dataset_key, "Current dataset")


def page_header(title: str, subtitle: str, page: str | None = None) -> None:
    st.markdown("<div class='eyebrow'>AI-POWERED MSME FINANCIAL INTELLIGENCE</div>", unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)
    if page is not None:
        st.markdown(page_hero_html(page), unsafe_allow_html=True)


def disclaimer() -> None:
    st.markdown("<div class='risk-note'>Analytical research estimate only. Not a guaranteed prediction, lending recommendation, or financial decision.</div>", unsafe_allow_html=True)


def render_about_page() -> None:
    """About page: static content, available without an upload."""
    page_header(
        "About Surojit Malakar",
        "Finance, research, operations, and technology in service of practical impact.",
        page="About",
    )
    photo_column, intro_column = st.columns([1, 2], gap="large")
    with photo_column:
        st.markdown(profile_card_html(), unsafe_allow_html=True)
    with intro_column:
        st.subheader("A little about me")
        st.write(BIO_SUMMARY)
        st.write(CAREER_INTENT)

    st.markdown(about_tiles_html(), unsafe_allow_html=True)
    st.markdown(about_block_html("Entrepreneurship & community impact", ENTERPRENEURSHIP), unsafe_allow_html=True)
    st.markdown(about_block_html("Research & analytical frameworks", RESEARCH_FRAMEWORKS), unsafe_allow_html=True)
    st.markdown(about_block_html("Experience & interests", EXPERIENCE_INTERESTS), unsafe_allow_html=True)


def render_methodology_page(
    frame: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
) -> None:
    """Methodology page: definitions and limitations; feature definitions need an upload."""
    page_header("Methodology & Research Mode", "Definitions, evaluation choices, limitations, and responsible interpretation.", page="Methodology")
    st.subheader("Dataset")
    st.write("All production analysis is based on the active uploaded file. The synthetic dataset generator is retained only for isolated tests and development; it is not loaded by the application.")
    st.subheader("Model methodology")
    st.write(
        "When a usable target is detected, a dataset-specific Random Forest classifier or regressor "
        "is evaluated using a reproducible 75/25 random hold-out split. Numeric predictors use median "
        "imputation; categorical predictors use most-frequent imputation and one-hot encoding. "
        "Identifiers are excluded from predictors. No bundled model is applied to uploaded records."
    )
    st.subheader("Evaluation metrics")
    st.write(
        "Classification reports accuracy and balanced accuracy; regression reports mean absolute "
        "error and R². These exploratory hold-out metrics depend on the target definition, sample, "
        "and split. They are not calibrated probabilities or row-level predictions."
    )
    st.subheader("Fairness and sources of bias")
    st.write("Protected attributes are never model inputs. If supplied for a legitimate audit, held-out group label rates, true-positive rates, and false-positive rates are descriptive only. Selection bias, historical decisions, missingness, sector/geographic representation, and label construction can all produce biased estimates.")
    st.subheader("Limitations and ethical use")
    st.write("The score is not causal, calibrated uncertainty, a guaranteed prediction, lending advice, or a decision engine. Validate definitions, data rights, external and temporal performance, subgroup behavior, and calibration with qualified reviewers before any consequential use. Provide human oversight and recourse.")
    st.subheader("Feature definitions")
    if frame is not None and features is not None:
        st.dataframe(pd.DataFrame({"Feature": features.columns, "Definition": ["Raw numeric statement field" if name in frame.columns else "Engineered ratio; see project README" for name in features.columns]}), width="stretch", hide_index=True)
    else:
        st.info("Upload a dataset to inspect its detected feature definitions.")
    st.subheader("Sparse-data risk index")
    st.write(
        "The app does not apply a bundled model to uploaded records. Its transparent rule-based risk index "
        "adds disclosed points only for observed adverse financial conditions, subtracts a limited offset "
        "for observed strengths, and is bounded between 0 and 100. It is an uncalibrated review signal, "
        "not a probability of default."
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
    return scores["Risk index (0-100)"]


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


def _financial_metrics_for_display(metrics: pd.DataFrame) -> pd.DataFrame:
    """Keep numeric and unavailable KPI values Arrow-compatible in Streamlit."""
    displayed = metrics.copy()
    displayed["Value"] = displayed["Value"].map(
        lambda value: (
            f"{value:,.4g}"
            if isinstance(value, (int, float, np.integer, np.floating))
            else str(value)
        )
    )
    return displayed


def _render_data_intelligence(
    document_result,
    intelligence: dict,
    filename: str,
    revision: int,
    content_signature: str,
    financial_columns: list[str],
) -> None:
    frame = document_result.frame
    financial_metrics = _financial_metrics_for_display(intelligence["financial_metrics"])
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
    with st.expander("Financial analysis from detected fields"):
        st.dataframe(financial_metrics, width="stretch", hide_index=True)

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
            "A supervised model result is a held-out evaluation against the uploaded target, not a prediction for each row. "
            "Anomaly is an unusual statistical pattern. Neither establishes default, fraud, or misconduct."
        )
        st.markdown(f"**{intelligence['analysis_result_title']}**")
        if intelligence["model_evaluation_available"]:
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
        if intelligence.get("warning_details"):
            st.dataframe(
                pd.DataFrame(intelligence["warning_details"]),
                width="stretch",
                hide_index=True,
            )
        elif intelligence["early_warnings"]:
            for warning in intelligence["early_warnings"]:
                st.write(f"- {warning}")
        else:
            st.info("No data-derived early warning was triggered.")

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
                    # Re-apply the brand chart frame: these figures are built by
                    # data_visualization and would otherwise keep their own look.
                    style_chart(item["figure"])
                    st.plotly_chart(
                        item["figure"],
                        width="stretch",
                        config={"displayModeBar": False},
                        key=f"data_intelligence_{content_signature}_{index}",
                    )
        except Exception as error:
            st.warning(f"Visual analytics are unavailable for this dataset: {error}")

    st.markdown("<div class='panel-header'><h3>Dataset Copilot</h3></div>", unsafe_allow_html=True)
    if LLM_SETTINGS.enabled:
        st.caption(
            f"Answers are generated by {LLM_SETTINGS.label} from this upload's "
            "profile, statistics, risk findings, trends, and detected columns. "
            "Only compact dataset metadata is sent to the endpoint, never the "
            "uploaded rows."
        )
    else:
        st.caption(
            "Answers are generated deterministically from this upload's profile, "
            "statistics, risk findings, trends, and detected columns. Configure a "
            "free LLM endpoint (LLM_API_BASE) to enable AI-generated answers."
        )
    copilot_key = f"dataset_copilot_{content_signature}"
    with st.form(f"dataset_copilot_form_{content_signature}"):
        question = st.text_input(
            "Ask about this dataset",
            key=f"{copilot_key}_question",
            placeholder="For example: What are the main warnings?",
        )
        ask_dataset = st.form_submit_button("Ask about dataset")
    if ask_dataset:
        answer, answer_source = generate_dataset_copilot_answer(
            question,
            intelligence,
            llm=LLM_SETTINGS,
        )
        st.session_state[f"{copilot_key}_answer"] = answer
        st.session_state[f"{copilot_key}_source"] = answer_source
    if st.session_state.get(f"{copilot_key}_answer"):
        st.info(st.session_state[f"{copilot_key}_answer"])
        answer_source = st.session_state.get(f"{copilot_key}_source", "rule-based")
        st.caption(
            "Answer source: "
            + (LLM_SETTINGS.label if answer_source == "llm" else "rule-based engine")
        )

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


st.markdown(STYLESHEET, unsafe_allow_html=True)



def style_chart(
    figure: go.Figure,
    *,
    height: int | None = None,
    margin: dict[str, int] | None = None,
) -> None:
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font={"family": "Outfit, sans-serif", "color": TEXT, "size": 12},
        title_font={"color": TEXT, "size": 16},
        legend={
            "font": {"color": TEXT, "size": 12},
            "bgcolor": WHITE,
            "bordercolor": "rgba(16,185,129,.18)",
            "borderwidth": 1,
        },
        margin=margin or {"t": 36, "r": 12, "b": 38, "l": 12},
        # Keep the figure's own height (auto-generated charts pre-size
        # themselves in ``data_visualization._layout``) unless one is given.
        height=height if height is not None else figure.layout.height,
        autosize=True,
    )
    figure.update_xaxes(
        color=TEXT_STRONG,
        title_font={"color": TEXT_STRONG},
        tickfont={"color": TEXT_MUTED},
        gridcolor="rgba(16, 185, 129, 0.14)",
        zerolinecolor="rgba(16, 185, 129, 0.28)",
        automargin=True,
    )
    figure.update_yaxes(
        color=TEXT_STRONG,
        title_font={"color": TEXT_STRONG},
        tickfont={"color": TEXT_MUTED},
        gridcolor="rgba(16, 185, 129, 0.14)",
        zerolinecolor="rgba(16, 185, 129, 0.28)",
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


def render_report_card(
    title: str,
    subtitle: str,
    cache: dict,
    data_key: str,
    build,
    file_name: str,
    mime: str = "application/pdf",
) -> None:
    """Render one on-demand report card with a generate and download action."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(subtitle)
        if st.button(f"Generate {title}", key=f"generate_{data_key}", width="stretch"):
            try:
                cache[data_key] = build()
            except Exception as error:
                st.warning(f"{title} generation failed: {error}")
        if data_key in cache:
            st.download_button(
                f"Download {title}",
                data=cache[data_key],
                file_name=file_name,
                mime=mime,
                key=f"download_{data_key}",
                width="stretch",
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
        color_discrete_map={"Risk increased": CRITICAL, "Risk reduced": G4},
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


_WARNING_ACTIONS = {
    "Rapid revenue decline": "Verify revenue periods and investigate the operating change.",
    "Negative operating cash flow": "Review cash conversion, receivables, and working-capital needs.",
    "Increasing leverage": "Verify debt scope and review repayment capacity and maturities.",
    "Falling liquidity": "Review near-term obligations and available current assets.",
    "Deteriorating margins": "Confirm cost classifications and investigate margin pressure.",
    "Increasing receivable days": "Review overdue receivables and collection practices.",
    "Falling interest coverage": "Verify interest expense and assess debt-service capacity.",
}


def render_warning_signals(
    active_signals: list[str],
    row: pd.Series | None = None,
    features: pd.Series | None = None,
) -> None:
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
        evidence = "Triggered rule; supporting value is unavailable."
        feature_name = {
            "Rapid revenue decline": "Sales_Growth",
            "Negative operating cash flow": "Cash_Flow_Operations",
            "Increasing leverage": "Debt_to_Assets",
            "Falling liquidity": "Current_Ratio",
            "Deteriorating margins": "EBITDA_Margin",
            "Increasing receivable days": "Receivable_Days",
            "Falling interest coverage": "Interest_Coverage",
        }.get(signal)
        observed = features.get(feature_name) if features is not None and feature_name else None
        if observed is None and row is not None and feature_name:
            observed = row.get(feature_name)
        if observed is not None and pd.notna(observed):
            value = float(observed)
            if feature_name in {"Sales_Growth", "EBITDA_Margin"}:
                evidence = f"Observed {feature_name.replace('_', ' ').lower()}: {value:.1%}."
            elif feature_name in {"Current_Ratio", "Debt_to_Assets"}:
                evidence = f"Observed {feature_name.replace('_', ' ').lower()}: {value:.2f}x."
            elif feature_name == "Receivable_Days":
                evidence = f"Observed receivable days: {value:.1f}."
            else:
                evidence = f"Observed {feature_name.replace('_', ' ').lower()}: {value:,.2f}."
        items.append(
            f"<div class='signal-item'><span class='signal-indicator {severity}'></span>"
            f"<div><strong>{escape(signal)} · {labels[severity]}</strong>"
            f"<span>Evidence: {escape(evidence)}</span>"
            f"<span>Reason: {escape(detail)}</span>"
            f"<span>Recommended action: {escape(_WARNING_ACTIONS.get(signal, 'Review the source values and confirm the detected condition.'))}</span>"
            f"</div></div>"
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
                <h4>Early-warning priorities</h4>
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
    """Return the brand colour for a risk band; kept as a thin alias over the theme."""
    return band_colour(band)


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
    risk_index = snapshot["risk_index"]
    state = _state_color(snapshot["risk_category"])
    heading = (
        f"{escape(snapshot['company'])} · {escape(snapshot['period'])}"
    )
    st.markdown(
        f"<div class='risk-summary-card {state}'><div class='risk-state'>{heading}</div>"
        f"<p>{escape(snapshot['risk_category'])} · "
        f"{'not calculable' if risk_index is None else format(risk_index, '.1f') + '/100 index points'} · "
        f"{escape(snapshot['method'])}</p>"
        f"<p>Data coverage: {escape(snapshot['coverage_label'])}</p></div>",
        unsafe_allow_html=True,
    )
    if snapshot["warnings"]:
        render_warning_signals(snapshot["warnings"], snapshot["row"], snapshot["features"])
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
        page="Portfolio Screening",
    )
    scores = _cached_screening_scores(dataset_key, _bundle_signature(bundle))
    if scores.empty:
        st.info("There are no records to screen in this dataset.")
        return
    summary = screening_summary(scores)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Records screened", f"{summary['records']:,}")
    metric_cols[1].metric("Rule-based index", f"{summary['rule_records']:,}")
    metric_cols[2].metric("Elevated risk index (>=60)", f"{summary['elevated_records']:,}")
    mean_text = (
        "Not available"
        if summary["mean_risk_index"] is None
        else f"{summary['mean_risk_index']:.1f}/100"
    )
    metric_cols[3].metric("Mean rule risk index", mean_text)
    st.caption(
        "Every displayed screening score is a transparent 0-100 rule-based index. "
        "It is not a probability of default or an ML prediction."
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

    export_cols = st.columns(3)
    with export_cols[0]:
        st.download_button(
            "Download screening workbook (Excel)",
            data=_cached_screening_workbook(
                dataset_key,
                _bundle_signature(bundle),
                exposure_column or "",
                str(len(st.session_state.get("underwriter_notes", []))),
            ),
            file_name="portfolio_screening.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="screening_workbook",
            width="stretch",
        )
    with export_cols[1]:
        st.download_button(
            "Download screening report (PDF)",
            data=_cached_screening_pdf(dataset_key, _bundle_signature(bundle), exposure_column or ""),
            file_name="portfolio_screening.pdf",
            mime="application/pdf",
            key="screening_pdf",
            width="stretch",
            on_click="ignore",
        )
    with export_cols[2]:
        st.caption(
            "The workbook includes the screening table, band mix, exposure totals, and any analyst "
            "notes saved in this session."
        )

    filtered, visible = render_filtered_grid(scores, "screening", blob_source="screening")
    st.dataframe(visible, width="stretch", hide_index=True, key="screening_table")
    if filtered.empty:
        st.info("No screened record matches the current search and filters.")
        return

    position_column = "Risk index (0-100)"
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
            "This detail uses the same feature pass and rule-based scoring as the table above, so the "
            "two views agree."
        )
        render_notes_panel(
            snapshot["company"],
            snapshot["period"],
            snapshot["distress_probability"],
            snapshot["risk_category"],
            "screening",
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
        page="Compare Records",
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
        risk_index = snapshot["risk_index"]
        column.metric(
            snapshot["company"],
            "Not calculable" if risk_index is None else f"{risk_index:.1f}/100",
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
            "Risk index (0-100)": (
                snapshot["risk_index"]
                if snapshot["risk_index"] is not None
                else 0.0
            ),
        }
        for snapshot in snapshots
    ]
    figure = px.bar(
        pd.DataFrame(chart_rows),
        x="Risk index (0-100)",
        y="Record",
        orientation="h",
        color="Risk category",
        color_discrete_map={band: _band_colour(band) for band in {row["Risk category"] for row in chart_rows}},
        title="Rule-based risk index by record",
    )
    figure.update_layout(showlegend=False, xaxis_title="Index points", yaxis_title=None, xaxis_range=[0, 100])
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
                    color_discrete_sequence=[G3],
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
        page="Data Explorer",
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
        "Dataset Monitoring",
        "Compare successive analyses of the active dataset and track its data quality, feature "
        "distributions, and transparent rule-index movements.",
        page="Model Monitoring",
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

        st.markdown("<div class='panel-header'><h3>Rule risk-index stability</h3></div>", unsafe_allow_html=True)
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
                color_discrete_sequence=[G3],
                title="Share of records with rule index >=60 by revision",
            )
            figure.update_yaxes(tickformat=".0%")
            render_chart(figure, height=260)

    st.markdown("<div class='panel-header'><h3>Dataset-specific model evaluation</h3></div>", unsafe_allow_html=True)
    model = data_intelligence.get("model", {})
    if model.get("status") == "evaluated":
        st.write(
            f"{model['model']} evaluated against uploaded target '{model['target']}' "
            f"using {model['test_rows']:,} hold-out records."
        )
        accuracy = model.get("accuracy")
        if accuracy is not None:
            st.metric("Hold-out accuracy", f"{accuracy:.1%}")
        st.caption(model.get("limitation", "Exploratory evaluation only; not a deployed prediction model."))
    else:
        st.info("No supervised model was evaluated for this upload. Monitoring uses observed data and rule-based indices only.")
    render_audit_panel()
    disclaimer()


def render_stress_page(
    frame: pd.DataFrame,
    company_rows: list[int],
    row_index: int,
    bundle: dict,
) -> None:
    page_header(
        "Stress Testing",
        "Combine revenue, margin, leverage, interest, and liquidity shocks in a single "
        "multi-factor grid, then isolate each lever to see what actually moves the estimate.",
        page="Stress Testing",
    )
    factors = available_factors(frame.iloc[row_index])
    if not factors:
        st.warning(
            "This record has too few observed fields to build a stress grid. Revenue and EBITDA "
            "are required at minimum."
        )
        return

    control_cols = st.columns([2, 3])
    with control_cols[0]:
        chosen = st.multiselect(
            "Levers",
            [factor.key for factor in factors],
            default=[factor.key for factor in factors[:3]],
            format_func=lambda key: next(f.label for f in factors if f.key == key),
            key="stress_factors",
            help="Each selected lever multiplies the size of the grid.",
        )
    selected_factors = [factor for factor in factors if factor.key in chosen]
    with control_cols[1]:
        combos = grid_size(selected_factors)
        st.caption(
            f"{combos:,} scenario combinations. "
            + (
                f"Grids above {MAX_GRID_SCENARIOS:,} combinations are blocked to keep the page responsive."
                if combos > MAX_GRID_SCENARIOS
                else "The whole grid is scored in one pass."
            )
        )
    st.session_state["_stress_blocked"] = combos > MAX_GRID_SCENARIOS

    preset_cols = st.columns(4)
    for position, (name, changes) in enumerate(STRESS_PRESETS.items()):
        with preset_cols[position]:
            if st.button(name, key=f"stress_preset_{position}", width="stretch"):
                st.session_state["stress_factors"] = [key for key in changes if key in {f.key for f in factors}]
                st.rerun()

    if st.session_state.get("_stress_blocked"):
        st.error(
            f"Select fewer levers. {combos:,} combinations exceeds the "
            f"{MAX_GRID_SCENARIOS:,}-scenario limit for an interactive page."
        )
        return
    if not selected_factors:
        st.info("Select at least one lever to run a stress grid.")
        return

    with st.spinner("Scoring the stress grid..."):
        results = _cached_stress_grid(
            dataset_key,
            row_index,
            tuple(factor.key for factor in selected_factors),
            _bundle_signature(bundle),
        )
    base_risk_index = float(results.attrs["base_risk_index"])
    summary = summarize_stress(results, base_risk_index)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Scenarios scored", f"{summary['scenarios']:,}")
    metric_cols[1].metric("Baseline risk index", f"{base_risk_index:.1f}/100")
    metric_cols[2].metric(
        "Worst case",
        f"{summary['worst_risk_index']:.1f}/100",
        summary["worst_band"],
    )
    metric_cols[3].metric("Best case", f"{summary['best_risk_index']:.1f}/100")
    metric_cols[4].metric("Scenarios at index >=60", f"{summary['elevated']:,}")
    st.caption(
        f"Worst case scenario: {summary['worst_scenario']}. "
        f"{summary['worse_than_reported']:,} of {summary['scenarios']:,} combinations have a higher "
        "rule-based index than the baseline."
    )

    st.markdown("<div class='panel-header'><h3>Scenario sensitivity (tornado)</h3></div>", unsafe_allow_html=True)
    tornado = _cached_tornado(dataset_key, row_index, _bundle_signature(bundle))
    if tornado.empty:
        st.info("No single-lever sensitivity is available for this record.")
    else:
        downside_columns = [column for column in tornado.columns if column.startswith("Downside ")]
        upside_columns = [column for column in tornado.columns if column.startswith("Upside ")]
        plotted = tornado.copy()
        plotted["Downside move"] = tornado["Downside impact"]
        plotted["Upside move"] = tornado["Upside impact"]
        figure = px.bar(
            plotted.sort_values("Total impact"),
            x=["Downside move", "Upside move"],
            y="Factor",
            orientation="h",
            color_discrete_sequence=[CRITICAL, G4],
            title="Change in rule-based risk index under each offered scenario",
        )
        figure.update_xaxes(title_text="Change in index points")
        figure.update_layout(barmode="group")
        render_chart(figure, height=max(240, 90 + 46 * len(tornado)), margin={"t": 52, "r": 16, "b": 48, "l": 150})
        st.dataframe(
            tornado[["Factor", *downside_columns, *upside_columns, "Worst band", "Best band"]],
            width="stretch",
            hide_index=True,
            key="stress_tornado",
        )
        flat = tornado[tornado["Total impact"] <= 1e-9]
        if not flat.empty:
            st.info(
                "These levers show no measurable effect for this record: "
                + ", ".join(str(name) for name in flat["Factor"])
                + ". A tree-based model responds to combinations of features, so a single lever "
                "can leave the estimate unchanged even when its ratio moves."
            )

    st.markdown("<div class='panel-header'><h3>Breakeven headroom</h3></div>", unsafe_allow_html=True)
    headroom = pd.DataFrame([
        {
            "Lever": factor.label,
            "Largest downside shock tested": factor.format_step(factor.steps[0]),
            "Stays below index 60 at": breakeven_step(
                frame.iloc[[row_index]], factor, bundle, 0.60, "downside"
            ),
        }
        for factor in selected_factors
    ])
    headroom["Stays below index 60 at"] = headroom["Stays below index 60 at"].map(
        lambda value: "No offered level" if value is None else str(value)
    )
    st.dataframe(headroom, width="stretch", hide_index=True, key="stress_headroom")
    st.caption(
        "Headroom is the largest offered shock whose outcome still scores below index 60. "
        "It is a screening signal, not a default trigger."
    )

    st.markdown("<div class='panel-header'><h3>All combinations</h3></div>", unsafe_allow_html=True)
    ordered = results.sort_values("Risk index (0-100)", ascending=False)
    st.dataframe(
        ordered.head(500),
        width="stretch",
        hide_index=True,
        key="stress_grid",
    )
    if len(ordered) > 500:
        st.caption(f"Showing the 500 highest-index of {len(ordered):,} scored combinations.")
    st.download_button(
        "Download full stress grid (CSV)",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="stress_grid.csv",
        mime="text/csv",
        key="stress_download",
        width="stretch",
    )
    st.warning(
        "Stress results are model responses, not forecasts. The response is only meaningful "
        "inside the range of values seen in training, and a tree-based model can be non-monotonic: "
        "a larger shock is not always the higher estimate."
    )
    disclaimer()


def render_calibration_page(frame: pd.DataFrame, bundle: dict) -> None:
    page_header(
        "Model Calibration",
        "Compare predicted risk with observed outcomes, then move the risk-band cutoffs to see "
        "how the portfolio would be reclassified.",
        page="Model Calibration",
    )
    if bundle.get("model") is None:
        st.info(
            "Calibration is unavailable because this application does not apply a bundled model "
            "to uploaded data. Upload a labeled dataset for an independent hold-out model evaluation."
        )
        return
    sample = _cached_calibration(dataset_key, repr(len(frame)), _bundle_signature(bundle))
    probabilities = sample["probabilities"]
    labels = sample["labels"]
    metrics = sample["metrics"]
    if not probabilities:
        st.info(
            "No labeled outcomes are available for calibration. Upload a dataset with a binary "
            "target column, or retrain the model so the held-out test split is recorded."
        )
        return
    st.caption(f"Sample basis: {sample['basis']} · {metrics['records']:,} records.")

    metric_cols = st.columns(6)
    metric_cols[0].metric("Observed default rate", f"{metrics['observed_default_rate']:.1%}")
    metric_cols[1].metric("Mean predicted risk", f"{metrics['predicted_default_rate']:.1%}")
    metric_cols[2].metric("Brier score", f"{metrics['brier_score']:.4f}")
    metric_cols[3].metric(
        "Expected calibration error",
        f"{metrics['expected_calibration_error']:.1%}"
        if metrics["expected_calibration_error"] is not None else "Not available",
    )
    metric_cols[4].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}" if metrics["roc_auc"] is not None else "Not available")
    metric_cols[5].metric(
        "Recalibration slope",
        f"{metrics['calibration_slope']:.2f}"
        if metrics["calibration_slope"] is not None else "Not available",
    )
    if sample["is_in_sample"]:
        st.warning(
            "These probabilities come from the same data the model was fitted on, so calibration "
            "looks better than it will be out of sample. Retrain the model to record held-out "
            "predictions."
        )

    st.markdown("<div class='panel-header'><h3>Reliability by predicted band</h3></div>", unsafe_allow_html=True)
    table = sample["table"]
    figure = px.line(
        table,
        x="Predicted default rate",
        y="Observed default rate",
        markers=True,
        color_discrete_sequence=[G3],
        title="Predicted versus observed default rate",
    )
    maximum = max(float(table["Predicted default rate"].max()), float(table["Observed default rate"].max()), 0.01)
    figure.add_shape(type="line", x0=0, y0=0, x1=maximum, y1=maximum, line=dict(dash="dash", color=NEUTRAL))
    figure.update_xaxes(title_text="Predicted default rate %", range=[0, maximum * 1.1])
    figure.update_yaxes(title_text="Observed default rate %", range=[0, maximum * 1.1])
    render_chart(figure, height=300)
    st.dataframe(table, width="stretch", hide_index=True, key="calibration_table")
    st.caption(
        "A gap above zero means the model predicts more risk than was observed; a gap below zero "
        "means it predicts less. Small bands are noisy, so read the shape of the curve before the "
        "individual bands."
    )

    st.markdown("<div class='panel-header'><h3>Risk-band cutoffs</h3></div>", unsafe_allow_html=True)
    cutoff_cols = st.columns(3)
    tuned: list[float] = []
    for position, (label, default) in enumerate(
        zip(("Moderate", "High", "Critical"), DEFAULT_CUTOFFS, strict=True)
    ):
        with cutoff_cols[position]:
            value = st.slider(
                f"{label} at or above",
                min_value=0.05,
                max_value=0.95,
                value=round(float(default), 2),
                step=0.05,
                key=f"cutoff_{position}",
            )
            tuned.append(value)
    ordered = tuple(sorted(tuned))
    st.caption(
        f"Tuned cutoffs: {cutoff_summary(probabilities, ordered)}. "
        f"Model defaults: {cutoff_summary(probabilities, DEFAULT_CUTOFFS)}."
    )
    if ordered[0] >= ordered[1] or ordered[1] >= ordered[2]:
        st.error("Cutoffs must increase from Moderate to High to Critical.")
        return
    shifted = shifted_band_counts(probabilities, ordered)
    band_figure = px.bar(
        shifted.melt(id_vars="Risk category", value_vars=["Default records", "Tuned records"]),
        x="Risk category",
        y="value",
        color="variable",
        barmode="group",
        color_discrete_sequence=[NEUTRAL, G3],
        title="Records per band: default versus tuned cutoffs",
    )
    render_chart(band_figure, height=280)
    st.dataframe(shifted, width="stretch", hide_index=True, key="cutoff_shift")
    moved = shifted[shifted["Change"] != 0]
    if not moved.empty:
        st.warning(
            "Cutoff changes reclassify "
            + ", ".join(f"{row['Risk category']} ({row['Change']:+,})" for _, row in moved.iterrows())
            + " records on this sample."
        )
    else:
        st.info("The tuned cutoffs do not change any band on this sample.")

    st.markdown("<div class='panel-header'><h3>Decision cutoffs</h3></div>", unsafe_allow_html=True)
    st.dataframe(
        threshold_table(probabilities, labels, (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)),
        width="stretch",
        hide_index=True,
        key="threshold_table",
    )
    st.caption(
        "Precision at a cutoff is the share of flagged records that actually defaulted. "
        "Precision (balanced) divides by the observed default rate, so 1.0 means no lift."
    )
    disclaimer()


def render_notes_panel(company: str, period: str, probability: float | None, band: str, key: str) -> None:
    """Save a review note for the selected record and list existing notes."""
    st.markdown("<div class='panel-header'><h3>Analyst notes</h3></div>", unsafe_allow_html=True)
    store = st.session_state.setdefault("underwriter_notes", [])
    with st.form(f"note_form_{key}"):
        text = st.text_area("Note", key=f"note_text_{key}", height=70, label_visibility="collapsed",
                            placeholder="Record your review conclusion, conditions, or follow-ups...")
        decision = st.selectbox("Decision", NOTE_DECISIONS, key=f"note_decision_{key}")
        submitted = st.form_submit_button("Save note")
    if submitted:
        error = validate_note(text, decision)
        if error:
            st.error(error)
        else:
            record_key = f"{company}|{period}"
            add_note(store, make_note(
                record_key=record_key,
                company=company,
                period=period,
                text=text,
                decision=decision,
                probability=probability,
                risk_category=band,
            ))
            log_event(st.session_state.setdefault("audit_events", []), audit_event(
                "Note saved",
                f"{decision} recorded for {company} ({period}).",
                _dataset_label(dataset_key),
            ))
            st.success("Note saved to this session.")

    frame = notes_frame(store)
    if not frame.empty:
        st.caption(f"{len(frame)} note(s) saved in this session. Notes are session-scoped and are not persisted.")
        st.dataframe(frame.head(50), width="stretch", hide_index=True, key=f"notes_table_{key}")
        st.download_button(
            "Download notes (CSV)",
            data=notes_csv(store),
            file_name="analyst_notes.csv",
            mime="text/csv",
            key=f"notes_download_{key}",
        )
        counts = decision_counts(store)
        if not counts.empty:
            st.caption("Decisions recorded: " + ", ".join(
                f"{row['Decision']} {int(row['Notes'])}" for _, row in counts.iterrows()
            ))


def render_audit_panel() -> None:
    st.markdown("<div class='panel-header'><h3>Audit trail</h3></div>", unsafe_allow_html=True)
    events = st.session_state.setdefault("audit_events", [])
    log_event(events, audit_event("View opened", "Reviewer opened the audit view.", _dataset_label(dataset_key)))
    frame = audit_frame(events)
    if frame.empty:
        st.info("No recorded activity yet.")
        return
    st.caption(
        f"{len(frame)} recorded action(s) in this session, newest first. "
        "This trail is a working record, not a compliance-grade log."
    )
    st.dataframe(frame, width="stretch", hide_index=True, key="audit_table")
    st.download_button(
        "Download audit trail (CSV)",
        data=audit_csv(events),
        file_name="audit_trail.csv",
        mime="text/csv",
        key="audit_download",
    )


NAVIGATION = {
    "Overview": "Executive Overview",
    "Financial Health": "MSME Financial Health",
    "Credit Risk": "Risk Prediction",
    "Early Warnings": "Early-Warning Indicators",
    "AI Copilot": "AI Copilot",
    "Scenario Simulator": "Scenario Simulator",
    "Credit Assessment": "Credit Assessment",
    "Data Intelligence": "Data Intelligence",
    "Portfolio Intelligence": "Portfolio Intelligence",
    "Model Performance": "Model Performance",
    "Reports": "Reports",
    "Methodology": "Methodology",
    "About": "About",
    "Screening": "Portfolio Screening",
    "Comparison": "Compare Records",
    "Data Explorer": "Data Explorer",
    "Explainable AI": "Explainable AI",
    "Stress Testing": "Stress Testing",
    "Calibration": "Model Calibration",
    "Monitoring": "Model Monitoring",
}


@st.cache_resource
def get_bundle() -> dict:
    """Return an empty model bundle; uploaded datasets are never scored by a demo model."""
    return {
        "model": None,
        "features": [],
        "report": {},
        "is_demo": False,
    }


st.sidebar.markdown("<div class='eyebrow'>CREDIT INTELLIGENCE</div>", unsafe_allow_html=True)
st.sidebar.title("CREDIT RISK AI")
selected_navigation = st.sidebar.radio("Workspace", list(NAVIGATION), label_visibility="collapsed")
page = NAVIGATION[selected_navigation]
# One authoritative page state, mirrored from the navigation radio so every
# page switch is explicit and the active upload is never reset.
st.session_state["current_page"] = page

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
            width="stretch",
        )
        reanalyze_clicked = st.button(
            "Re-analyze",
            disabled=not has_current_analysis,
            key="re_analyze_upload",
            width="stretch",
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
        if page == "Data Intelligence" or not financial_columns:
            _render_data_intelligence(
                document_result,
                data_intelligence,
                uploaded.name,
                analysis_revision,
                upload_signature,
                financial_columns,
            )
            st.stop()
        st.success(
            f"File successfully loaded. AI Data Detective completed. "
            f"{document_result.status} Source: {document_result.source_type}."
        )
        # Compact analysis context. The full automatic analysis (dataset
        # profile, detected variables, patterns, and recommendations) lives on
        # the Data Intelligence page; every other page shows only this summary
        # strip so no page duplicates the dashboard layout.
        context_cols = st.columns(4)
        context_cols[0].metric("Analysis Mode", data_intelligence["analysis_mode"])
        context_cols[1].metric("Dataset", data_intelligence["dataset_type"])
        context_cols[2].metric("Records", f"{len(document_result.frame):,}")
        context_cols[3].metric("Anomalies", f"{data_intelligence['anomaly_count']:,}")
        st.caption(
            f"AI Data Detective: {document_result.status} Source: {document_result.source_type}. "
            f"{data_intelligence['executive_summary']} "
            "Open Data Intelligence for the full dataset profile, detected variables, "
            "correlations, and recommendations."
        )
        frame = financial_frame
        for warning in document_result.warnings:
            st.caption(warning)
        st.caption("Document contents are processed locally; verify extracted figures against the source statement.")
    except (ValueError, TypeError, OSError, ImportError, KeyError) as error:
        st.error(f"Dataset analysis failed: {error}")
        st.stop()
else:
    if page == "About":
        render_about_page()
    elif page == "Methodology":
        render_methodology_page()
    else:
        st.info("Upload a dataset to generate analysis.")
        st.caption(
            f"The '{selected_navigation}' page analyses the active upload. "
            "About and Methodology are available without data."
        )
    st.stop()

bundle = get_bundle()
features, flags = financial_base
dataset_key = repr(upload_pipeline_key)
_register_dataset(
    dataset_key,
    frame,
    features,
    flags,
    uploaded.name,
)
_monitor_label = (
    f"{uploaded.name} · revision {analysis_revision}"
)
if dataset_key not in st.session_state.setdefault("_monitor_recorded", set()):
    st.session_state["_monitor_recorded"].add(dataset_key)
    _record_monitor_snapshot(
        _monitor_label,
        uploaded.name,
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
    result = rule_based_assessment(frame, row_index, financial_analysis)
    risk_index = result["risk_index"]
    result["health_score"] = (
        max(0.0, min(100.0, 100 - risk_index))
        if risk_index is not None
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
    model_available_for_record = False
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

    concentration = portfolio_analysis.get("concentration", {})
    if concentration.get("available"):
        st.markdown("<div class='panel-header'><h3>Exposure Concentration</h3></div>", unsafe_allow_html=True)
        metrics = concentration["metrics"]
        concentration_metrics = st.columns(4)
        concentration_metrics[0].metric(
            "Effective borrowers (1/HHI)",
            f"{metrics['effective_borrowers']:.1f}"
            if metrics["effective_borrowers"] is not None
            else "Not available",
        )
        concentration_metrics[1].metric(
            "Largest borrower share",
            f"{metrics['largest_exposure_share']:.1%}"
            if metrics["largest_exposure_share"] is not None
            else "Not available",
        )
        concentration_metrics[2].metric(
            "Top-10 exposure share",
            f"{metrics['top_10_share']:.1%}"
            if metrics["top_10_share"] is not None
            else "Not available",
        )
        concentration_metrics[3].metric(
            "Measured exposure",
            f"{concentration['total_exposure']:,.2f}"
            if concentration["total_exposure"] is not None
            else "Not available",
        )
        st.caption(
            f"Measured from the observed '{concentration['exposure_column']}' column by borrower. "
            "The Herfindahl-Hirschman Index here is a descriptive concentration summary, "
            "not a regulatory capital measure."
        )
        if not concentration["band_breakdown"].empty:
            st.dataframe(concentration["band_breakdown"], width="stretch", hide_index=True)
        if not concentration["top_exposures"].empty:
            st.dataframe(concentration["top_exposures"], width="stretch", hide_index=True)
        for note in concentration["findings"]:
            st.write(note)

    vintage = portfolio_analysis.get("vintage", {})
    if vintage.get("available"):
        st.markdown("<div class='panel-header'><h3>Origination Vintages</h3></div>", unsafe_allow_html=True)
        st.dataframe(vintage["cohorts"], width="stretch", hide_index=True)
        for note in vintage["findings"]:
            st.write(note)


def show_risk() -> dict:
    result = current_assessment
    left, middle, right = st.columns(3)
    risk_value = (
        result.get("distress_probability")
        if model_available_for_record
        else result.get("risk_index")
    )
    risk_label = (
        f"{risk_value:.1%}" if model_available_for_record and risk_value is not None
        else f"{risk_value:.1f}/100" if risk_value is not None
        else "Insufficient data"
    )
    left.metric(
        "Model prediction" if model_available_for_record else "Rule-based risk index",
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
            risk_index = (
                result.get("distress_probability")
                if model_available_for_record
                else result.get("risk_index")
            )
            render_kpi_card(
                "Model prediction" if model_available_for_record else "Rule-Based Risk Index",
                f"{risk_index:.1%}" if model_available_for_record and risk_index is not None
                else f"{risk_index:.1f}/100" if risk_index is not None
                else "Not available",
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

        # Domain KPI strip: liquidity, profitability, leverage, cash flow.
        net_margin = financial_analysis["ratios"].get("Net Profit Margin")
        debt_assets = financial_analysis["ratios"].get("Debt / Assets")
        ocf_revenue = financial_analysis["ratios"].get("Operating Cash Flow / Revenue")
        domain_cols = st.columns(4)
        with domain_cols[0]:
            render_kpi_card(
                "Liquidity",
                f"{current_ratio:.2f}x" if pd.notna(current_ratio) else "N/A",
                "Healthy" if pd.notna(current_ratio) and current_ratio >= 1 else "Watch" if pd.notna(current_ratio) else "Unavailable",
                "healthy" if pd.notna(current_ratio) and current_ratio >= 1 else "watch" if pd.notna(current_ratio) else "neutral",
            )
        with domain_cols[1]:
            render_kpi_card(
                "Profitability",
                f"{net_margin:.1%}" if net_margin is not None else "N/A",
                "Positive" if net_margin is not None and net_margin > 0 else "Negative" if net_margin is not None else "Unavailable",
                "healthy" if net_margin is not None and net_margin > 0 else "high" if net_margin is not None else "neutral",
            )
        with domain_cols[2]:
            render_kpi_card(
                "Leverage",
                f"{debt_assets:.2f}x" if debt_assets is not None else "N/A",
                "Elevated" if debt_assets is not None and debt_assets >= 0.65 else "Moderate" if debt_assets is not None else "Unavailable",
                "high" if debt_assets is not None and debt_assets >= 0.65 else "healthy" if debt_assets is not None else "neutral",
            )
        with domain_cols[3]:
            render_kpi_card(
                "Cash Flow",
                f"{ocf_revenue:.1%}" if ocf_revenue is not None else "N/A",
                "Positive" if ocf_revenue is not None and ocf_revenue > 0 else "Negative" if ocf_revenue is not None else "Unavailable",
                "healthy" if ocf_revenue is not None and ocf_revenue > 0 else "high" if ocf_revenue is not None else "neutral",
            )

        # Concise company-level executive summary generated from the actual analysis.
        period_value = selected.iloc[0].get("period")
        period_label = str(period_value) if pd.notna(period_value) else "period not identified"
        summary_parts = [
            (
                f"{company_name} ({period_label}) is classified as "
                f"{result['risk_category']} with a rule-based risk index of "
                f"{result['risk_index']:.1f}/100"
                if result.get("risk_index") is not None
                else f"{company_name} ({period_label}) risk could not be calculated from the available fields"
            )
        ]
        if health_score is not None:
            summary_parts.append(f"financial health score {health_score:.0f}/100")
        top_drivers = [
            str(factor.get("feature")).replace("_", " ")
            for factor in (result.get("top_risk_factors") or [])[:3]
        ]
        if top_drivers:
            summary_parts.append("primary risk drivers: " + ", ".join(top_drivers))
        if active_warning_signals:
            summary_parts.append("active early warnings: " + ", ".join(active_warning_signals[:3]))
        if recommendations:
            summary_parts.append(f"priority action: {recommendations[0]}")
        st.markdown("<div class='panel-header'><h3>Company Executive Summary</h3></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='risk-summary-card {risk_class}'>"
            f"<div class='risk-state'>{escape(result['risk_category'])}</div>"
            f"<p>{escape('. '.join(summary_parts))}.</p>"
            f"<strong>{escape(result['method'])}</strong></div>",
            unsafe_allow_html=True,
        )

        if not model_available_for_record and page in {"Executive Overview", "Credit Assessment"}:
            st.info(
                "No dataset-specific ML model is served for this upload. This is a transparent rule-based "
                "index derived from observed values, not a probability of default."
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
        if LLM_SETTINGS.enabled:
            st.caption(
                f"Answers are generated by {LLM_SETTINGS.label}. Only the derived "
                "credit context for the selected record is sent to the endpoint."
            )
        else:
            st.caption(
                "Answers come from the built-in rule-based engine. Configure a free "
                "LLM endpoint (LLM_API_BASE, e.g. a local Ollama server) to enable "
                "AI-generated answers."
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
            uploaded.name,
            uploaded.size,
            question,
        )
        if preset is not None:
            response_signature = (*response_signature[:-1], preset)
        if submitted or preset is not None or st.session_state.get("credit_copilot_signature") != response_signature:
            history = [entry["question"] for entry in transcript]
            with st.spinner("Generating answer..."):
                response, source = generate_credit_copilot_answer(
                    question,
                    copilot_context,
                    history=history,
                    llm=LLM_SETTINGS,
                )
            transcript.append({"question": question, "answer": response, "company": str(selected_company), "source": source})
            del transcript[:-12]
            st.session_state["credit_copilot_response"] = response
            st.session_state["credit_copilot_source"] = source
            st.session_state["credit_copilot_signature"] = response_signature
        if st.session_state.get("credit_copilot_response"):
            answer_source = st.session_state.get("credit_copilot_source", "rule-based")
            st.caption(
                "Answer source: "
                + (LLM_SETTINGS.label if answer_source == "llm" else "rule-based engine")
            )
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
                st.session_state.pop("credit_copilot_source", None)
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
        page_header("What-If Lab", "Test hypothetical financial changes against the actual selected company-period.", page="Scenario Simulator")
        st.caption("HYPOTHETICAL SCENARIO — NOT ACTUAL RESULTS. Adjustments are applied to a copy of the selected record; the uploaded dataset is never modified.")
        available_fields = available_whatif_fields(selected.iloc[0])
        if not available_fields:
            st.info(
                "No adjustable financial fields were detected in the selected record. "
                "Provide Revenue, EBITDA, Debt, Cash, Assets, Liabilities, Receivables, "
                "Inventory, Operating Cash Flow, or Interest Expense to run a scenario."
            )
        else:
            changes: dict[str, float] = {}
            with st.form("whatif_scenario"):
                slider_cols = st.columns(3)
                for position, (field_key, column, label) in enumerate(available_fields):
                    with slider_cols[position % 3]:
                        if field_key == "profit_margin":
                            raw_change = st.slider(label, min_value=-20, max_value=20, value=0, step=1, format="%d pp", key=f"whatif_{field_key}")
                        else:
                            raw_change = st.slider(label, min_value=-50, max_value=50, value=0, step=5, format="%d%%", key=f"whatif_{field_key}")
                        changes[field_key] = raw_change / 100.0
                run_scenario = st.form_submit_button("Run hypothetical scenario")
            if run_scenario:
                try:
                    scored = score_whatif(frame, row_index, changes)
                    scenario_result = scored["result"]
                    scenario_analysis = scored["analysis"]
                    risk_index_change = (
                        scenario_result["risk_index"] - result["risk_index"]
                        if scenario_result.get("risk_index") is not None
                        and result.get("risk_index") is not None
                        else None
                    )
                    health_change = (
                        scenario_result.get("health_score") - result.get("health_score")
                        if scenario_result.get("health_score") is not None
                        and result.get("health_score") is not None
                        else None
                    )
                    scenario_metric, scenario_status, scenario_health = st.columns(3)
                    scenario_metric.metric(
                        "Scenario risk index",
                        f"{scenario_result['risk_index']:.1f}/100" if scenario_result.get("risk_index") is not None else "Insufficient data",
                        delta=f"{risk_index_change:+.1f} points" if risk_index_change is not None else None,
                        delta_color="inverse",
                    )
                    scenario_status.metric(
                        "Baseline risk index",
                        f"{result['risk_index']:.1f}/100" if result.get("risk_index") is not None else "Insufficient data",
                    )
                    scenario_health.metric(
                        "Health score change",
                        f"{health_change:+.1f} points" if health_change is not None else "Not available",
                    )
                    comparison_rows = []
                    for field_key, column, label in available_fields:
                        if field_key == "profit_margin":
                            continue
                        if column in selected.columns and column in scored["frame"].columns:
                            baseline_value = selected.iloc[0][column]
                            scenario_value = scored["frame"].iloc[0][column]
                            try:
                                change = float(scenario_value) - float(baseline_value)
                            except (TypeError, ValueError):
                                change = None
                            comparison_rows.append({
                                "Metric": label.replace(" (%)", ""),
                                "Current": baseline_value,
                                "Scenario": scenario_value,
                                "Change": change,
                            })
                    for ratio_name in ("Current Ratio", "Debt / Assets", "Net Profit Margin", "Interest Coverage"):
                        baseline_value = financial_analysis["ratios"].get(ratio_name)
                        scenario_value = scenario_analysis["ratios"].get(ratio_name)
                        if baseline_value is not None or scenario_value is not None:
                            comparison_rows.append({
                                "Metric": ratio_name,
                                "Current": baseline_value,
                                "Scenario": scenario_value,
                                "Change": (
                                    scenario_value - baseline_value
                                    if baseline_value is not None and scenario_value is not None
                                    else None
                                ),
                            })
                    if comparison_rows:
                        st.markdown("<div class='panel-header'><h3>Current vs Scenario</h3></div>", unsafe_allow_html=True)
                        st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)
                    st.markdown("<div class='panel-header'><h3>Scenario Risk Drivers</h3></div>", unsafe_allow_html=True)
                    if scenario_result.get("top_risk_factors"):
                        st.dataframe(
                            pd.DataFrame(scenario_result["top_risk_factors"]).rename(
                                columns={"feature": "Scenario risk driver", "contribution": "Rule weight"}
                            ),
                            width="stretch",
                            hide_index=True,
                        )
                    else:
                        st.info("No scenario risk driver could be calculated from the adjusted values.")
                    if scored["signals"]:
                        st.markdown("**Scenario early warnings triggered:** " + ", ".join(scored["signals"]))
                    else:
                        st.caption("No early-warning indicator is triggered under these assumptions.")
                    applied = [
                        (f"{label} {changes[field_key] * 100:+.0f} pp" if field_key == "profit_margin"
                         else f"{label} {changes[field_key]:+.0%}")
                        for field_key, column, label in available_fields
                        if changes.get(field_key)
                    ]
                    st.caption("Applied assumptions: " + (", ".join(applied) if applied else "no changes"))
                except (TypeError, ValueError) as error:
                    st.error(f"Scenario could not be calculated: {error}")
        disclaimer()

    if is_overview:
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
                    color_discrete_sequence=[G3],
                )
                revenue_chart.update_layout(title="Revenue", yaxis_title=None)
                with trend_cols[0]:
                    render_chart(revenue_chart, height=220, margin={"t": 24, "r": 10, "b": 38, "l": 10})

                liquidity_chart = px.line(
                    series,
                    x="period",
                    y="Current_Ratio",
                    markers=True,
                    color_discrete_sequence=[G4],
                )
                liquidity_chart.update_layout(title="Current ratio", yaxis_title=None)
                with trend_cols[1]:
                    render_chart(liquidity_chart, height=220, margin={"t": 24, "r": 10, "b": 38, "l": 10})
            else:
                st.info("Trend history is unavailable for the selected company.")
        else:
            st.info("Add company and period fields to the CSV to view financial trends.")

        render_warning_signals(
            active_warning_signals,
            selected.iloc[0],
            selected_features.iloc[0],
        )
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
    page_header("Financial Health", "Deep financial analysis of the selected company-period from observed values.", page="MSME Financial Health")
    health_company = str(frame.iloc[row_index].get("company_id", "Selected Company"))
    st.caption(f"Company: {health_company} | Period: {selected.iloc[0].get('period', 'Not identified')} | Coverage: {current_assessment['coverage_label']}")
    row = frame.iloc[row_index]
    feat0 = selected_features.iloc[0]

    def _num(name):
        try:
            v = float(pd.to_numeric(pd.Series([feat0.get(name) if name in feat0.index else row.get(name)]), errors="coerce").iloc[0])
            return v if pd.notna(v) else None
        except (TypeError, ValueError):
            return None

    def _raw(name):
        try:
            v = float(pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0])
            return v if pd.notna(v) else None
        except (TypeError, ValueError):
            return None

    rev, rev_g = _raw("Revenue"), _num("Sales_Growth")
    ebitda, netp = _raw("EBITDA"), _raw("Net_Profit")
    ca, cl = _raw("Current_Assets"), _raw("Current_Liabilities")
    debt, ta, tl = _raw("Debt"), _raw("Total_Assets"), _raw("Total_Liabilities")
    ocf = _raw("Cash_Flow_Operations")
    rec, inv, pay = _raw("Accounts_Receivable"), _raw("Inventory"), _raw("Accounts_Payable")
    intexp = _raw("Interest_Expense")
    r = financial_analysis["ratios"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Revenue", f"{rev:,.0f}" if rev is not None else "Not available", f"{rev_g:.1%}" if rev_g is not None else None)
    k2.metric("Net profit", f"{netp:,.0f}" if netp is not None else "Not available",
              f"{(r.get('Net Profit Margin') or 0):.1%} margin" if r.get("Net Profit Margin") is not None else None)
    k3.metric("Current ratio", f"{(r.get('Current Ratio') or float('nan')):.2f}x" if r.get("Current Ratio") is not None else "Not available")
    k4.metric("Debt / assets", f"{(r.get('Debt / Assets') or float('nan')):.2f}x" if r.get("Debt / Assets") is not None else "Not available")
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("ROA", f"{r['Return on Assets']:.1%}" if r.get("Return on Assets") is not None else "Not available")
    k6.metric("ROE", f"{r['Return on Equity']:.1%}" if r.get("Return on Equity") is not None else "Not available")
    k7.metric("Interest coverage", f"{r['Interest Coverage']:.2f}x" if r.get("Interest Coverage") is not None else "Not available")
    wc = r.get("Working Capital")
    k8.metric("Working capital", f"{wc:,.0f}" if wc is not None else "Not available")
    k9, k10, k11, k12 = st.columns(4)
    k9.metric("Operating cash flow", f"{ocf:,.0f}" if ocf is not None else "Not available")
    fcf = (ocf - 0) if ocf is not None else None
    k10.metric("Free cash flow (proxy)", f"{fcf:,.0f}" if fcf is not None else "Not available")
    st.caption("Free cash flow is shown as operating cash flow when capex is not in the upload; never fabricated.")
    k11.metric("Receivables", f"{rec:,.0f}" if rec is not None else "Not available")
    k12.metric("Inventory", f"{inv:,.0f}" if inv is not None else "Not available")
    st.markdown("<div class='panel-header'><h3>Profitability</h3></div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        {"Metric": "Observed profit", "Revenue": rev, "EBITDA": ebitda,
         "Net profit": netp, "Gross profit": r.get("Gross Profit")},
        {"Metric": "Margins", "EBITDA margin": r.get("EBITDA Margin"),
         "Net margin": r.get("Net Profit Margin"), "Gross margin": r.get("Gross Margin")},
        {"Metric": "Returns", "ROA": r.get("Return on Assets"),
         "ROE": r.get("Return on Equity"), "ROCE": r.get("ROCE")},
    ]), width="stretch", hide_index=True)
    st.caption("ROCE = EBITDA / capital employed (total assets minus current liabilities). Gross profit/margin require an observed cost-of-goods-sold field; otherwise they are not available.")
    st.markdown("<div class='panel-header'><h3>Liquidity, leverage & cash flow</h3></div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        {"Area": "Liquidity", "Current ratio": r.get("Current Ratio"), "Quick ratio": r.get("Quick Ratio"),
         "Working capital": r.get("Working Capital"), "OCF ratio": r.get("Operating Cash Flow / Revenue")},
        {"Area": "Leverage", "Debt/equity": r.get("Debt / Equity"), "Debt/assets": r.get("Debt / Assets"),
         "Interest coverage": r.get("Interest Coverage"), "Equity": r.get("Equity")},
        {"Area": "Working capital", "Receivables": rec, "Inventory": inv, "Payables (if uploaded)": pay,
         "Receivable days": r.get("Receivable Days"), "Inventory days": r.get("Inventory Days")},
    ]), width="stretch", hide_index=True)
    st.caption("Only observed metrics are shown; missing fields are 'Not available', never zero-filled.")
    if "company_id" in frame and "period" in frame:
        series = trend_data(frame, selected.iloc[0]["company_id"])
        if len(series) > 1:
            st.markdown("<div class='panel-header'><h3>Financial trends</h3></div>", unsafe_allow_html=True)
            metric = st.selectbox("Historical measure", ["Revenue", "Current_Ratio", "Debt_to_Assets", "EBITDA_Margin", "Interest_Coverage"], key="finhealth_trend")
            fig = px.line(series, x="period", y=metric, markers=True, title=f"{metric.replace('_', ' ')} over time", color_discrete_sequence=[G3])
            render_chart(fig, height=280)
            st.markdown("<div class='panel-header'><h3>Period Comparison</h3></div>", unsafe_allow_html=True)
            st.caption("Every observed company-period, sorted chronologically. Ratios are engineered from the uploaded values.")
            st.dataframe(series, width="stretch", hide_index=True)
        else:
            st.info("At least two dated company-period records are needed for a trend chart and period comparison.")
    st.dataframe(pd.DataFrame([{"Ratio": k, "Value": v if v is not None else "Not available"} for k, v in financial_analysis["ratios"].items()]).astype({"Value": "string"}), width="stretch", hide_index=True)
    disclaimer()

elif page == "Risk Prediction":
    page_header(
        "Credit Risk",
        "Dedicated credit-risk assessment of the selected company-period.",
        page="Risk Prediction",
    )
    result = show_risk()
    st.caption(f"Model used: {'No bundled ML model is applied to uploads (transparent rule-based index).' if not model_available_for_record else 'Existing ML model.'} | Coverage: {current_assessment['coverage_label']}")
    if not model_available_for_record:
        st.info("ML cannot produce a calibrated default probability for this upload because no bundled model is applied to unrelated uploaded records. The transparent rule-based index below is the deterministic fallback.")
    health_score = result.get("health_score")
    if health_score is None:
        st.info("There is not enough numeric financial data to calculate a health score.")
    else:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            number={"suffix": "%", "font": {"color": G1, "size": 42}},
            title={"text": "Financial health score", "font": {"color": G1, "size": 18}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": NEUTRAL},
                "bar": {"color": G4, "thickness": 0.25},
                "bgcolor": WHITE,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(239, 68, 68, 0.18)"},
                    {"range": [40, 70], "color": "rgba(245, 158, 11, 0.18)"},
                    {"range": [70, 100], "color": "rgba(16, 185, 129, 0.18)"},
                ],
                "threshold": {
                    "line": {"color": G1, "width": 3},
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

    # Professional credit-risk summary: why this risk level, what reduces it,
    # and what could increase it — all derived from the actual factors above.
    risk_class = _state_color(result["risk_category"])
    risk_driver_names = [
        str(factor.get("feature")).replace("_", " ")
        for factor in (result.get("top_risk_factors") or [])
    ]
    protective_names = [
        str(factor.get("feature")).replace("_", " ")
        for factor in (result.get("protective_factors") or [])
    ]
    if risk_driver_names:
        why_risky = "Risk is elevated primarily by " + ", ".join(risk_driver_names[:3]) + "."
    else:
        why_risky = "No risk-increasing factor was triggered by the available values."
    if protective_names:
        reduces_risk = "Risk is reduced by " + ", ".join(protective_names[:3]) + "."
    else:
        reduces_risk = "No risk-reducing factor was triggered by the available values."
    could_increase = (
        "Conditions that could increase risk include declining revenue, negative "
        "operating margins, weakening liquidity, rising leverage, and negative "
        "operating cash flow."
    )
    st.markdown("<div class='panel-header'><h3>Credit Risk Summary</h3></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='risk-summary-card {risk_class}'>"
        f"<div class='risk-state'>{escape(result['risk_category'])}</div>"
        f"<p><strong>Why this risk level:</strong> {escape(why_risky)}<br>"
        f"<strong>What reduces risk:</strong> {escape(reduces_risk)}<br>"
        f"<strong>What could increase risk:</strong> {escape(could_increase)}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "SHAP/explainability is available on the Explainable AI page when a model "
        "is served for the upload; otherwise the rule-based drivers above are shown."
    )

elif page == "Explainable AI":
    page_header("Explainable AI", "How the selected model behaves globally and for this individual record.", page="Explainable AI")
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
                color_discrete_map={"Risk increased": CRITICAL, "Risk reduced": G3},
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
                color_discrete_sequence=[G3],
            )
            render_chart(global_fig, height=400, margin={"t": 58, "r": 12, "b": 48, "l": 120})
        except Exception as error:
            st.error(f"SHAP explanation could not be calculated in this environment: {error}")
    disclaimer()

elif page == "Early-Warning Indicators":
    page_header("Early Warning Center", "Deterioration alerts ranked by severity, built only from observed values.", page="Early-Warning Indicators")
    st.caption("Rules are configurable research heuristics, not learned predictions or universal thresholds.")
    detailed = detailed_warnings(frame, features, flags, row_index)
    triggered = triggered_warnings(detailed)
    severity_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
    severity_counts = {level: sum(1 for w in detailed if w.get("Severity") == level) for level in severity_order}
    severity_cols = st.columns(len(severity_order))
    for severity_col, level in zip(severity_cols, severity_order):
        severity_col.metric(f"{level} alerts", severity_counts[level])
    st.caption(
        f"Core-field coverage: {current_assessment['coverage_label']}. "
        f"{len(triggered)} of {len(detailed)} indicators are currently triggered."
    )
    if triggered:
        st.markdown("<div class='panel-header'><h3>Active Alerts</h3></div>", unsafe_allow_html=True)
        for warning in triggered:
            severity = warning.get("Severity", "MEDIUM")
            severity_class = {
                "CRITICAL": "critical", "HIGH": "high", "MEDIUM": "watch",
                "LOW": "healthy", "INFO": "healthy",
            }.get(severity, "watch")
            st.markdown(
                f"<div class='risk-summary-card {severity_class}'>"
                f"<div class='risk-state'>{escape(severity)} &middot; {escape(str(warning.get('Indicator')))}</div>"
                f"<p><strong>Actual value:</strong> {escape(str(warning.get('Actual value')))}<br>"
                f"<strong>Reference:</strong> {escape(str(warning.get('Threshold/reference')))}<br>"
                f"<strong>Why it matters:</strong> {escape(str(warning.get('Why it matters')))}<br>"
                f"<strong>Recommended action:</strong> {escape(str(warning.get('Recommended action')))}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No early-warning indicator is currently triggered by the observed values.")
    st.markdown("<div class='panel-header'><h3>Severity Matrix</h3></div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(detailed), width="stretch", hide_index=True)
    if "company_id" in frame and "period" in frame:
        series = trend_data(frame, selected.iloc[0]["company_id"])
        if len(series) > 1:
            st.markdown("<div class='panel-header'><h3>Alert Timeline</h3></div>", unsafe_allow_html=True)
            metric = st.selectbox("Historical measure", ["Revenue", "Current_Ratio", "Debt_to_Assets", "EBITDA_Margin", "Interest_Coverage"])
            fig = px.line(
                series,
                x="period",
                y=metric,
                markers=True,
                title=f"{metric.replace('_', ' ')} over time",
                color_discrete_sequence=[G3],
            )
            render_chart(fig)
        else:
            st.info("At least two dated company-period records are needed for a trend chart.")
    else:
        st.info("Add company_id and period columns to a longitudinal CSV to view historical trends.")
    disclaimer()

elif page == "Model Performance":
    page_header("Model Lab", "Actual quality of the exploratory model trained from a target detected in the active upload.", page="Model Performance")
    model = data_intelligence.get("model", {})
    if model.get("status") != "evaluated":
        st.info(
            "Model performance unavailable because no valid target/model exists. "
            "No bundled or demo model is applied to uploads. Upload data with a "
            "meaningful target label and sufficient examples to evaluate a model."
        )
    else:
        st.info(model.get("limitation", "Exploratory hold-out model evaluation; not calibrated or deployable for decisions."))
        st.markdown("<div class='panel-header'><h3>Model &amp; Training</h3></div>", unsafe_allow_html=True)
        identity_cols = st.columns(4)
        identity_cols[0].metric("Model used", model.get("model", "Not available"))
        identity_cols[1].metric("Target", model.get("target", "Not available"))
        identity_cols[2].metric("Train / test split", model.get("train_test_split", "Not available"))
        identity_cols[3].metric("Evaluation records", f"{model.get('test_rows', 0):,}")
        st.caption(
            f"Training records: {model.get('train_rows', 0):,}. The model is trained "
            "only from the active upload on a random hold-out split; it is not a "
            "calibrated probability of default."
        )
        classification_metrics = {
            key: value for key, value in model.items()
            if key in {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"}
        }
        regression_metrics = {
            key: value for key, value in model.items()
            if key in {"mae", "r2"}
        }
        st.markdown("<div class='panel-header'><h3>Hold-out Metrics</h3></div>", unsafe_allow_html=True)
        if classification_metrics:
            classification_cols = st.columns(len(classification_metrics))
            for metric_col, (metric_name, metric_value) in zip(classification_cols, classification_metrics.items()):
                metric_col.metric(
                    metric_name.replace("_", " ").title(),
                    f"{metric_value:.1%}" if metric_value is not None else "Not available",
                )
        if regression_metrics:
            regression_cols = st.columns(len(regression_metrics))
            for metric_col, (metric_name, metric_value) in zip(regression_cols, regression_metrics.items()):
                metric_col.metric(
                    metric_name.replace("_", " ").upper(),
                    f"{metric_value:.3g}" if metric_value is not None else "Not available",
                )
        confusion = model.get("confusion_matrix")
        if confusion:
            st.markdown("<div class='panel-header'><h3>Confusion Matrix</h3></div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(confusion), width="stretch", hide_index=True)
        roc_curve_data = model.get("roc_curve")
        if roc_curve_data and model.get("roc_auc") is not None:
            st.markdown("<div class='panel-header'><h3>ROC Curve</h3></div>", unsafe_allow_html=True)
            roc_figure = go.Figure()
            roc_figure.add_trace(go.Scatter(
                x=roc_curve_data["fpr"],
                y=roc_curve_data["tpr"],
                mode="lines",
                name=f"ROC (AUC = {model['roc_auc']:.2f})",
                line={"color": G4, "width": 2},
            ))
            roc_figure.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line={"color": NEUTRAL, "width": 1, "dash": "dash"},
            ))
            roc_figure.update_layout(
                xaxis_title="False positive rate",
                yaxis_title="True positive rate",
                showlegend=True,
            )
            render_chart(roc_figure, height=320)
        if model.get("top_features"):
            st.markdown("<div class='panel-header'><h3>Feature Importance</h3></div>", unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(model["top_features"]).rename(
                    columns={"feature": "Feature", "importance": "Relative importance"}
                ),
                width="stretch",
                hide_index=True,
            )
        st.caption(
            "SHAP explainability is available on the Explainable AI page when a "
            "model is served for the upload."
        )
    disclaimer()

elif page == "Stress Testing":
    render_stress_page(frame, company_rows, row_index, bundle)

elif page == "Model Calibration":
    render_calibration_page(frame, bundle)

elif page == "Model Monitoring":
    render_monitoring_page(bundle)

elif page == "Portfolio Intelligence":
    page_header("Portfolio Command", "Entity-level screening summary across the uploaded records.", page="Portfolio Intelligence")
    scores = _cached_screening_scores(dataset_key, _bundle_signature(bundle))
    intel = portfolio_intelligence(frame, scores)
    if not intel.get("available"):
        st.info(intel.get("reason", "No screening scores are available for this dataset."))
    else:
        if not intel["multi_entity"]:
            st.caption(
                "This dataset contains a single entity, so entity-level comparisons are "
                "limited. Portfolio analysis requires multiple entities."
            )
        intel_metrics = st.columns(4)
        intel_metrics[0].metric("Entities", f"{intel['entities']:,}")
        intel_metrics[1].metric("Mean risk index", f"{intel['avg_risk']:.1f}/100" if intel["avg_risk"] is not None else "Not available")
        intel_metrics[2].metric("Mean health score", f"{intel['avg_health']:.1f}/100" if intel["avg_health"] is not None else "Not available")
        intel_metrics[3].metric("High-risk share", f"{intel['high_share']:.1%}" if intel["high_share"] is not None else "Not available")
        st.caption("Screening indices are transparent rule-based scores (0-100), not model default probabilities.")
        tiers = intel.get("risk_tiers") or {}
        if tiers:
            tier_cols = st.columns(len(tiers))
            for tier_col, (tier_name, tier_count) in zip(tier_cols, tiers.items()):
                tier_col.metric(tier_name, f"{tier_count:,}")
        st.markdown("<div class='panel-header'><h3>Portfolio Risk Distribution</h3></div>", unsafe_allow_html=True)
        band_frame = pd.DataFrame(
            [{"Risk band": str(band), "Records": count} for band, count in intel["bands"].items()]
        ).sort_values("Records", ascending=True)
        if not band_frame.empty:
            band_figure = go.Figure(go.Bar(
                x=band_frame["Records"],
                y=band_frame["Risk band"],
                orientation="h",
                marker={"color": [band_colour(str(band)) for band in band_frame["Risk band"]]},
            ))
            band_figure.update_layout(xaxis_title="Records", yaxis_title=None, showlegend=False)
            render_chart(band_figure, height=280)
        else:
            st.info("No risk band could be calculated from the available fields.")
        entity_cols = st.columns(2)
        with entity_cols[0]:
            st.markdown("<div class='panel-header'><h3>Highest Mean Risk Index</h3></div>", unsafe_allow_html=True)
            if intel["riskiest"]:
                st.dataframe(pd.DataFrame(intel["riskiest"], columns=["Entity", "Mean risk index"]), width="stretch", hide_index=True)
            else:
                st.info("No entity risk could be calculated from the available fields.")
        with entity_cols[1]:
            st.markdown("<div class='panel-header'><h3>Lowest Mean Risk Index</h3></div>", unsafe_allow_html=True)
            if intel["strongest"]:
                st.dataframe(pd.DataFrame(intel["strongest"], columns=["Entity", "Mean risk index"]), width="stretch", hide_index=True)
            else:
                st.info("No entity risk could be calculated from the available fields.")
        if intel["largest_entity"]:
            st.markdown("<div class='panel-header'><h3>Entity Concentration</h3></div>", unsafe_allow_html=True)
            st.metric("Largest entity share", f"{intel['largest_entity_share']:.1%}")
            st.caption(f"Largest entity: {intel['largest_entity']}")
        geo = intel.get("geo_concentration") or []
        if geo:
            st.markdown("<div class='panel-header'><h3>Geographic Concentration</h3></div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(geo), width="stretch", hide_index=True)
        anomaly = intel.get("anomaly_concentration")
        if anomaly:
            st.markdown("<div class='panel-header'><h3>Anomaly Concentration</h3></div>", unsafe_allow_html=True)
            st.metric("Anomalous record share", f"{anomaly['share']:.1%}")
            st.caption(f"Detected from the '{anomaly['column']}' column.")
        if intel["concentrations"]:
            st.markdown("<div class='panel-header'><h3>Segment Concentration</h3></div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(intel["concentrations"]), width="stretch", hide_index=True)
        trend = intel.get("trend")
        if trend:
            st.markdown("<div class='panel-header'><h3>Portfolio Trend</h3></div>", unsafe_allow_html=True)
            trend_frame = pd.DataFrame(trend)
            trend_figure = px.line(
                trend_frame,
                x="period",
                y="mean_risk",
                markers=True,
                title="Mean risk index by period",
                color_discrete_sequence=[G3],
            )
            render_chart(trend_figure, height=280)
        disclaimer()

elif page == "Reports":
    page_header("Reports", "Export the active upload's analysis, screening, and company reports.", page="Reports")
    st.caption("Reports are generated locally from the active upload. Each export is built on demand; no data leaves this session.")
    report_export_cols = st.columns(2)
    with report_export_cols[0]:
        st.markdown("<div class='panel-header'><h3>Data Intelligence</h3></div>", unsafe_allow_html=True)
        st.caption("Dataset profile, detected variables, patterns, and recommendations.")
        intelligence_report_cache = st.session_state.setdefault("_reports_intelligence_cache", {})
        intelligence_report = intelligence_report_cache.setdefault(upload_pipeline_key, {})
        if st.button("Generate Data Intelligence PDF", key="reports_generate_intelligence_pdf"):
            try:
                intelligence_report["pdf"] = _cached_intelligence_pdf(
                    data_intelligence,
                    uploaded.name,
                    analysis_revision,
                    upload_signature,
                )
            except Exception as error:
                st.warning(f"PDF report generation failed: {error}")
        if "pdf" in intelligence_report:
            st.download_button(
                "Download Data Intelligence PDF",
                data=intelligence_report["pdf"],
                file_name=f"{Path(uploaded.name).stem}_data_intelligence.pdf",
                mime="application/pdf",
                key="reports_download_intelligence_pdf",
                width="stretch",
            )
        if st.button("Generate Data Intelligence Excel", key="reports_generate_intelligence_excel"):
            try:
                intelligence_report["excel"] = _cached_intelligence_excel(
                    data_intelligence,
                    uploaded.name,
                    analysis_revision,
                    upload_signature,
                )
            except Exception as error:
                st.warning(f"Excel report generation failed: {error}")
        if "excel" in intelligence_report:
            st.download_button(
                "Download Data Intelligence Excel",
                data=intelligence_report["excel"],
                file_name=f"{Path(uploaded.name).stem}_data_intelligence.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="reports_download_intelligence_excel",
                width="stretch",
            )
    with report_export_cols[1]:
        st.markdown("<div class='panel-header'><h3>Portfolio Screening</h3></div>", unsafe_allow_html=True)
        st.caption("Rule-based screening scores, band summary, exposure, and underwriter notes.")
        reports_exposure_column = detect_exposure_column(frame)
        st.download_button(
            "Download screening workbook (Excel)",
            data=_cached_screening_workbook(
                dataset_key,
                _bundle_signature(bundle),
                reports_exposure_column or "",
                str(len(st.session_state.get("underwriter_notes", []))),
            ),
            file_name="portfolio_screening.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="reports_screening_workbook",
            width="stretch",
        )
        st.download_button(
            "Download screening report (PDF)",
            data=_cached_screening_pdf(dataset_key, _bundle_signature(bundle), reports_exposure_column or ""),
            file_name="portfolio_screening.pdf",
            mime="application/pdf",
            key="reports_screening_pdf",
            width="stretch",
        )
    st.markdown("<div class='panel-header'><h3>Company Reports</h3></div>", unsafe_allow_html=True)
    reports_company = str(frame.iloc[row_index].get("company_id", "Selected Company"))
    reports_period = str(selected.iloc[0].get("period", "Not identified"))
    st.caption(
        f"Reports are built on demand from the current analysis of "
        f"{reports_company} ({reports_period}). No data leaves this session."
    )
    company_report_cache = st.session_state.setdefault("_company_reports_cache", {})
    company_report_key = (selection_cache_key, reports_company, reports_period)
    company_report = company_report_cache.setdefault(company_report_key, {})
    report_detailed = detailed_warnings(frame, features, flags, row_index)
    report_drivers = [dict(factor) for factor in (current_assessment.get("top_risk_factors") or [])]
    report_cols = st.columns(2)
    with report_cols[0]:
        render_report_card(
            "Executive Report",
            "Assessment, top risk drivers, early warnings, and recommendations.",
            company_report, "executive_pdf",
            lambda: create_executive_pdf(
                reports_company, reports_period, current_assessment,
                financial_analysis, active_warning_signals, recommendations,
            ),
            f"{reports_company.replace(' ', '_')}_executive_report.pdf",
        )
        render_report_card(
            "Credit Risk Report",
            "Risk assessment, risk drivers, and early warnings.",
            company_report, "credit_risk_pdf",
            lambda: create_credit_risk_pdf(
                reports_company, reports_period, current_assessment,
                active_warning_signals, report_drivers,
            ),
            f"{reports_company.replace(' ', '_')}_credit_risk_report.pdf",
        )
    with report_cols[1]:
        render_report_card(
            "Financial Health Report",
            "Observed financial ratios for the selected company-period.",
            company_report, "financial_health_pdf",
            lambda: create_financial_health_pdf(
                reports_company, reports_period, financial_analysis,
            ),
            f"{reports_company.replace(' ', '_')}_financial_health_report.pdf",
        )
        render_report_card(
            "Early Warning Report",
            "Severity-ranked early-warning indicators with recommended actions.",
            company_report, "early_warning_pdf",
            lambda: create_early_warning_pdf(
                reports_company, reports_period, report_detailed,
            ),
            f"{reports_company.replace(' ', '_')}_early_warning_report.pdf",
        )
    if len(company_report_cache) > 16:
        company_report_cache.pop(next(iter(company_report_cache)), None)
    disclaimer()

elif page == "Methodology":
    render_methodology_page(frame, features)

elif page == "About":
    render_about_page()

st.markdown(site_footer_html(), unsafe_allow_html=True)