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
from msme_ews.credit_assessment import generate_risk_interpretation
from msme_ews.copilot import (
    build_credit_context,
    generate_credit_copilot_answer,
    generate_dataset_copilot_answer,
    suggested_questions,
)
from msme_ews.data_intelligence import analyze_dataset
from msme_ews.data_visualization import build_data_visualizations
from msme_ews.documents import extract_financial_document
from msme_ews.early_warning import early_warning_indicators, trend_data
from msme_ews.financial_analysis import (
    analyze_financials,
    recommendations_for,
    rule_based_assessment,
)
from msme_ews.features import engineer_features
from msme_ews.llm import load_settings
from msme_ews.notes import notes_frame
from msme_ews.portfolio import analyze_credit_portfolio
from msme_ews.warning_detail import detailed_warnings, triggered_warnings
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
from msme_ews.screening import (
    banded_exposure,
    detect_exposure_column,
    score_records,
    screening_summary,
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

# Optional free LLM for the embedded AI Credit Copilot panels. Reads LLM_API_BASE / LLM_API_KEY /
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
    st.caption(
        f"DATA SOURCE · ✓ Uploaded dataset connected ({escape(str(filename))}, "
        f"{len(frame):,} row(s) × {len(frame.columns):,} column(s)) · "
        "file name, row/column counts, sheets, detected and missing fields, quality and "
        "the preview below all describe this upload only."
    )
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


NAVIGATION = {
    "Overview": "Executive Overview",
    "Financial Health": "MSME Financial Health",
    "Credit Risk": "Risk Prediction",
    "Early Warnings": "Early-Warning Indicators",
    "Data Intelligence": "Data Intelligence",
    "Reports": "Reports",
    "Methodology": "Methodology",
    "About": "About",
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
# Read-only mirror of the navigation radio. Never written back into a widget
# key, so tab switches can never fight Streamlit's widget state and freeze.
if st.session_state.get("current_page") != page:
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
            # New file invalidates every downstream result immediately so stale
            # metrics can never linger while navigating between pages.
            for _stale_key in (
                "_upload_pipeline", "_upload_pipeline_key",
                "_financial_selection_cache", "_portfolio_analysis_cache",
                "_intelligence_report_cache", "_credit_report_cache",
                "_company_reports_cache", "_reports_intelligence_cache",
                "copilot_history", "credit_copilot_response",
                "credit_copilot_source", "credit_copilot_signature",
                "company_filter", "period_filter",
            ):
                st.session_state.pop(_stale_key, None)
            # Canonical aliases required by the acceptance spec. They mirror the
            # authoritative pipeline state so external checks can introspect it.
            st.session_state.pop("uploaded_data", None)
            st.session_state.pop("financial_metrics", None)
            st.session_state.pop("risk_result", None)
            st.session_state.pop("warnings", None)
            # Reset company/period selectors for the new file. They are popped
            # BEFORE the widgets are instantiated below, so Streamlit never sees
            # a stale widget value from the previous file (the classic freeze).
            st.session_state.pop("company_filter", None)
            st.session_state.pop("period_filter", None)
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
            if page == "About":
                render_about_page()
            elif page == "Methodology":
                render_methodology_page()
            else:
                st.success("File loaded")
                st.info("Click Analyze Dataset when you are ready to process this file.")
            st.stop()
        if st.session_state.get("_upload_pipeline_key") != upload_pipeline_key:
            if not uploaded_content:
                uploaded_content = uploaded.getvalue()
            with st.status("Analyzing dataset...", expanded=True) as progress:
                try:
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
                except Exception:
                    # Mark the status as failed so the spinner can never hang,
                    # then let the outer handler render one friendly banner.
                    # Streamlit control-flow stops inherit BaseException and
                    # pass through this handler untouched.
                    progress.update(label="Analysis failed", state="error", expanded=True)
                    raise
            st.session_state["_upload_pipeline_key"] = upload_pipeline_key
            st.session_state["_upload_pipeline"] = {
                "document": document_result,
                "intelligence": data_intelligence,
                "financial_columns": financial_columns,
                "financial_frame": financial_frame,
                "financial_base": financial_base,
            }
            # Canonical aliases: single source of truth, mirrored for the spec.
            st.session_state["uploaded_data"] = document_result.frame
            st.session_state["financial_metrics"] = data_intelligence.get("financial_metrics")
        else:
            cached_upload = st.session_state["_upload_pipeline"]
            document_result = cached_upload["document"]
            data_intelligence = cached_upload["intelligence"]
            financial_columns = cached_upload["financial_columns"]
            financial_frame = cached_upload["financial_frame"]
            financial_base = cached_upload["financial_base"]
            st.session_state["uploaded_data"] = document_result.frame
            st.session_state["financial_metrics"] = data_intelligence.get("financial_metrics")
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
    except Exception as error:
        # One friendly banner instead of a raw traceback that takes down every
        # tab. Streamlit's st.stop()/st.rerun control flow inherits
        # BaseException, so it always passes through this handler untouched.
        st.error(f"Dataset analysis failed: {error}")
        if page == "About":
            render_about_page()
        elif page == "Methodology":
            render_methodology_page()
        st.stop()
else:
    if page == "About":
        render_about_page()
    elif page == "Methodology":
        render_methodology_page()
    else:
        st.info("Upload a dataset to generate analysis.")
        st.markdown(
            "<div class='panel-header'><h3>DATA SOURCE · No dataset uploaded</h3></div>",
            unsafe_allow_html=True,
        )
        st.warning("No financial dataset uploaded.")
        st.caption(
            "Upload an Excel, CSV, or supported financial statement to begin analysis. "
            "Every metric, risk score, chart, warning, and report on this page is "
            "computed from the uploaded file — nothing is shown until data arrives. "
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

if "company_id" in frame:
    company_values = frame["company_id"].astype(str)
    company_options = company_values.drop_duplicates().tolist()
else:
    company_values = pd.Series(["All records"] * len(frame), index=frame.index)
    company_options = ["All records"]
# Guard against a stale company value carried over from a previous upload:
# if it is no longer in this file's options, reset BEFORE the widget is built.
if st.session_state.get("company_filter") not in company_options:
    st.session_state.pop("company_filter", None)

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

# Resolve the period row from the selected company. The widget below is the
# single owner of the "period_filter" value — never write to that key after
# the widget exists, or Streamlit raises StreamlitAPIException and the tab
# appears frozen. So compute the safe default here, pass it as index=, and
# read the widget's return value instead of poking session_state.
if company_rows:
    _stored_period = st.session_state.get("period_filter")
    _default_row = _stored_period if _stored_period in company_rows else company_rows[0]
    _default_index = company_rows.index(_default_row)
else:
    _default_index = 0

with selector_cols[1]:
    _period_position = st.selectbox(
        "Period",
        options=company_rows,
        index=_default_index,
        format_func=lambda index: str(
            frame.iloc[index].get("period", f"Record {index + 1}")
        ),
        key="period_filter",
    )
row_index = int(_period_position) if company_rows else 0

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
    st.caption(
        f"DATA SOURCE · ✓ Uploaded dataset connected ({escape(str(uploaded.name))}) · "
        "all portfolio figures below are observed or calculated from that file."
    )
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


if page == "Executive Overview":
    is_overview = page == "Executive Overview"
    result = current_assessment
    st.session_state["risk_result"] = dict(result)
    company_name = str(frame.iloc[row_index].get("company_id", "Selected Company"))
    health_score = result.get("health_score")
    current_ratio = selected_features.iloc[0].get("Current_Ratio", float("nan"))
    risk_class = _state_color(result["risk_category"])
    st.caption(
        f"DATA SOURCE · ✓ Uploaded dataset connected ({escape(str(uploaded.name))}, "
        f"{len(document_result.frame):,} record(s)) · score, category, drivers and trends "
        "below are calculated from that file. Missing inputs show 'Not available'."
    )
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

        if not model_available_for_record:
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

    if is_overview:
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

    if is_overview:
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

    if is_overview:
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

    if is_overview:
        st.markdown("<div class='panel-header'><h3>Top Risk Drivers</h3></div>", unsafe_allow_html=True)
        if model_available_for_record:
            st.markdown(
                "<div class='driver-legend'>Positive contribution increases distress output · Negative contribution reduces it</div>",
                unsafe_allow_html=True,
            )
            render_shap_drivers(result)
            if not result.get("top_risk_factors") and not result.get("protective_factors"):
                st.info("No on-demand SHAP drivers were calculated for this record.")
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

                The Copilot provides risk explanation, recommendations, and answers to
                scenario questions about this credit assessment.
                """
            )
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
    st.caption(
        f"DATA SOURCE · ✓ Uploaded dataset connected ({escape(str(uploaded.name))}, "
        f"{len(document_result.frame):,} record(s)) · every value below is read from or "
        "calculated from that file. 'Not available' means the required field was not "
        "found in the upload — it is never invented."
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Revenue", f"{rev:,.0f}" if rev is not None else "Not available", f"{rev_g:.1%}" if rev_g is not None else None)
    k2.metric("Net profit", f"{netp:,.0f}" if netp is not None else "Not available",
              f"{(r.get('Net Profit Margin') or 0):.1%} margin" if r.get("Net Profit Margin") is not None else None)
    k3.metric("Current ratio", f"{(r.get('Current Ratio') or float('nan')):.2f}x" if r.get("Current Ratio") is not None else "Not available")
    k4.metric("Debt / assets", f"{(r.get('Debt / Assets') or float('nan')):.2f}x" if r.get("Debt / Assets") is not None else "Not available")
    st.caption("Source: Uploaded Dataset → Revenue, Net Profit · Calculated from uploaded data: margins, Current Ratio, Debt-to-Assets.")
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
    st.session_state["warnings"] = list(active_warning_signals)
    st.caption(
        f"DATA SOURCE · ✓ Uploaded dataset connected ({escape(str(uploaded.name))}) · "
        "the index, category and drivers below are calculated from that file."
    )
    st.caption(f"Model used: {'No bundled ML model is applied to uploads (transparent rule-based index).' if not model_available_for_record else 'Existing ML model.'} | Coverage: {current_assessment['coverage_label']}")
    if not model_available_for_record:
        st.info("ML cannot produce a calibrated default probability for this upload because no bundled model is applied to unrelated uploaded records. The transparent rule-based index below is the deterministic fallback.")
    health_score = result.get("health_score")
    if health_score is None:
        st.info("Risk score unavailable — required variables are missing.")
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
        "SHAP explainability is shown on the Overview when a model "
        "is served for the upload; otherwise the rule-based drivers above are shown."
    )

elif page == "Early-Warning Indicators":
    page_header("Early Warning Center", "Deterioration alerts ranked by severity, built only from observed values.", page="Early-Warning Indicators")
    st.caption("Rules are configurable research heuristics, not learned predictions or universal thresholds.")
    st.caption(
        f"DATA SOURCE · ✓ Uploaded dataset connected ({escape(str(uploaded.name))}) · "
        "every warning below shows its actual uploaded value and threshold. "
        "Fields absent from the upload are reported as 'Not available', never assumed."
    )
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

elif page == "Reports":
    page_header("Reports", "Export the active upload's analysis, screening, and company reports.", page="Reports")
    st.caption("Reports are generated locally from the active upload. Each export is built on demand; no data leaves this session.")
    st.caption(
        f"DATA SOURCE · ✓ Uploaded dataset connected ({escape(str(uploaded.name))}, "
        f"{len(document_result.frame):,} record(s)) · every section below re-states the "
        "current upload's figures; nothing is sampled or reused from another file."
    )
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