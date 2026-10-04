"""Eight-page Streamlit research dashboard for MSME financial distress."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from msme_ews.data import validate_financial_data
from msme_ews.credit_assessment import apply_scenario_adjustments, generate_risk_interpretation
from msme_ews.demo import make_demo_data
from msme_ews.early_warning import early_warning_indicators, trend_data
from msme_ews.explain import explain_prediction, global_importance
from msme_ews.features import engineer_features
from msme_ews.modeling import train_models
from msme_ews.prediction import DEFAULT_MODEL_PATH, predict_financial_health

st.set_page_config(
    page_title="CREDIT RISK AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="auto",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
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
    --border:#1E293B;
    --shadow:rgba(15, 23, 42, 0.48);
}
html, body, [class*="css"] { font-family:'Manrope',sans-serif; color:var(--text); }
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
[data-testid="stMainBlockContainer"] { max-width:100%; padding-top:1.25rem; padding-bottom:3rem; }
[data-testid="stMetric"], [data-testid="stDataFrame"], [data-testid="stTable"],
[data-testid="stPlotlyChart"], [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stExpander"], [data-testid="stForm"], .block-container {
    min-width:0;
    background:var(--panel);
    border:1px solid var(--border);
    border-radius:18px;
    box-shadow:0 18px 40px var(--shadow);
}
[data-testid="stMetric"] { padding:18px 18px 14px; }
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
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stFileUploader"] section,
[data-testid="stSidebar"] [data-testid="stFileUploader"] div,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    background:#0F172A;
    border-color:#334155;
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
    background:rgba(37, 99, 235, 0.15) !important;
    border-color:#60A5FA !important;
    color:#EFF6FF !important;
}
.eyebrow { font:600 11px 'DM Mono',monospace; color:#94A3B8; text-transform:uppercase; letter-spacing:.12em; }
.topbar {
    display:flex;
    flex-wrap:wrap;
    justify-content:space-between;
    align-items:center;
    gap:1rem;
    padding:1.05rem 0 1.5rem;
}
.topbar-brand h1 {
    margin:0;
    font-size:clamp(2rem, 4vw, 3rem);
    line-height:1.05;
    letter-spacing:-0.06em;
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
    font:600 11px 'DM Mono',monospace;
    letter-spacing:.06em;
    text-transform:uppercase;
}
.live-badge::before {
    content:"";
    width:.55rem;
    height:.55rem;
    border-radius:50%;
    background:var(--healthy);
    box-shadow:0 0 0 6px rgba(16, 185, 129, 0.18);
}
.kpi-card {
    background:var(--panel);
    border:1px solid var(--border);
    border-radius:18px;
    padding:1rem 1.1rem;
    min-height:160px;
    position:relative;
    overflow:hidden;
}
.kpi-card::after {
    content:"";
    position:absolute;
    inset:auto -10% -55% auto;
    width:130px;
    height:130px;
    background:radial-gradient(circle, rgba(37, 99, 235, 0.25), transparent 68%);
    pointer-events:none;
}
.kpi-label {
    color:var(--muted);
    font-size:.78rem;
    letter-spacing:.08em;
    text-transform:uppercase;
    font-family:'DM Mono',monospace;
}
.kpi-value {
    margin-top:.7rem;
    font-size:clamp(1.8rem, 3vw, 2.5rem);
    font-weight:800;
    letter-spacing:-.05em;
}
.kpi-trend {
    margin-top:.65rem;
    display:inline-flex;
    align-items:center;
    gap:.4rem;
    padding:.25rem .55rem;
    border-radius:999px;
    font:600 12px 'DM Mono',monospace;
    background:rgba(148, 163, 184, 0.08);
    border:1px solid rgba(148, 163, 184, 0.14);
}
.kpi-trend.up { color:#A7F3D0; }
.kpi-trend.down { color:#FECACA; }
.kpi-trend.neutral { color:#BFDBFE; }
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
.risk-category-label { color:var(--muted) !important; font-size:.78rem; letter-spacing:.08em; font-family:'DM Mono',monospace; }
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
.risk-category-value.warning { background:rgba(245, 158, 11, 0.12); color:#FCD34D !important; }
.risk-category-value.danger { background:rgba(239, 68, 68, 0.12); color:#FCA5A5 !important; }
.risk-category-value.neutral { background:rgba(148, 163, 184, 0.12); color:#E2E8F0 !important; }
.panel-header {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:1rem;
    margin:1.5rem 0 .8rem;
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
    border-radius:18px;
    padding:1rem;
    box-shadow:0 18px 40px var(--shadow);
}
.copilot-box {
    background:var(--panel);
    border:1px solid rgba(59, 130, 246, 0.28);
    border-radius:18px;
    padding:1rem;
}
.copilot-input {
    background:#0B1728 !important;
    color:var(--text) !important;
    border:1px solid rgba(96, 165, 250, 0.22) !important;
    border-radius:12px !important;
    min-height:120px !important;
}
.copilot-output {
    background:rgba(15, 23, 42, 0.6);
    border:1px solid rgba(148, 163, 184, 0.18);
    border-radius:14px;
    padding:1rem;
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
.assessment-card {
    background:linear-gradient(180deg, rgba(12, 21, 31, 1), rgba(10, 18, 28, 1));
    border:1px solid rgba(96, 165, 250, 0.2);
    border-radius:18px;
    padding:1.15rem 1.2rem;
    margin:1rem 0 1.5rem;
    box-shadow:0 18px 40px rgba(2, 6, 23, 0.42);
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
    font:600 11px 'DM Mono',monospace;
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
    font-family:'DM Mono',monospace;
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
    content:"✓";
    color:#10B981;
    font-weight:800;
}
.assessment-bullets.risk li::before {
    content:"⚠";
    color:#F59E0B;
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
    font:600 10px 'DM Mono',monospace;
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
    font:600 12px 'DM Mono',monospace;
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
        width:min(86vw, 20rem) !important;
        min-width:min(86vw, 20rem) !important;
        max-width:86vw !important;
    }
    [data-testid="stMetric"] { width:100%; padding:14px; }
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
    .stApp button, .stApp [role="button"] { min-height:46px; }
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
        template="plotly_white",
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#0D1B2A",
        font={"family": "Manrope, sans-serif", "color": "#F8FAFC", "size": 12},
        title_font={"color": "#F8FAFC", "size": 16},
        legend={
            "font": {"color": "#F8FAFC", "size": 12},
            "bgcolor": "#0D1B2A",
            "bordercolor": "#1E293B",
            "borderwidth": 1,
        },
        margin=margin or {"t": 56, "r": 16, "b": 44, "l": 16},
        height=height,
        autosize=True,
    )
    figure.update_xaxes(
        color="#E2E8F0",
        title_font={"color": "#E2E8F0"},
        tickfont={"color": "#94A3B8"},
        gridcolor="#1E293B",
        zerolinecolor="#334155",
        automargin=True,
    )
    figure.update_yaxes(
        color="#E2E8F0",
        title_font={"color": "#E2E8F0"},
        tickfont={"color": "#94A3B8"},
        gridcolor="#1E293B",
        zerolinecolor="#334155",
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
        return "warning"
    if "high" in value:
        return "danger"
    return "neutral"


def render_kpi_card(label: str, value: str, trend: str | None = None, trend_direction: str = "neutral") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {f'<div class="kpi-trend {trend_direction}">{trend}</div>' if trend else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_credit_assessment_panel(company_name: str, result: dict, selected_features: pd.DataFrame, flags: pd.DataFrame, row_index: int) -> None:
    feat = selected_features.iloc[0]
    current_ratio = feat.get("Current_Ratio", float("nan"))
    margin = feat.get("EBITDA_Margin", 0.0)
    cash_flow = feat.get("Cash_Flow_Operations", 0.0)
    leverage = feat.get("Debt_to_Assets", 0.0)
    sales_growth = feat.get("Sales_Growth", float("nan"))
    active_flags = set(flags.iloc[row_index][flags.iloc[row_index]].index.tolist())
    health_score = max(0.0, min(100.0, (1 - float(result["distress_probability"])) * 100.0))

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
        risks = ["Monitoring required on margin pressure"]
    if not strengths:
        strengths = ["Stable operating base"]

    priorities = [
        "Monitor operating cash flow",
        "Reduce leverage",
        "Protect operating margins",
    ]

    risk_label = result["risk_category"]
    recommendation = (
        "The company may be considered for credit subject to tighter monitoring and improved cash-flow generation."
        if risk_label in {"Moderate Risk", "Low Risk"}
        else "The company requires deeper credit review due to elevated distress indicators."
    )

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
                    <div class="assessment-value">{health_score:.0f}/100</div>
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
                <h4 style="margin:0; font-size:.8rem; letter-spacing:.08em; text-transform:uppercase; color:#94A3B8; font-family:'DM Mono',monospace;">Early-warning priorities</h4>
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
    st.subheader("🤖 AI Risk Interpretation")
    st.markdown(f"**{interpretation['risk_category']}**")
    st.write(interpretation["explanation"])
    st.markdown(f"**Priority:** {interpretation['priority']}")
    if result.get("top_risk_factors"):
        st.caption("SHAP contributions explain model behavior; they are associations, not causal findings.")


def build_credit_context(frame: pd.DataFrame, selected: pd.DataFrame, selected_features: pd.DataFrame, flags: pd.DataFrame, row_index: int, result: dict) -> dict:
    row = selected.iloc[0].copy()
    feat = selected_features.iloc[0].copy()
    active_flags = list(flags.iloc[row_index][flags.iloc[row_index]].index)
    company_id = row.get("company_id", "Selected company")
    revenue = float(feat.get("Revenue", 0.0) or 0.0)
    profit = float(feat.get("Net_Profit", 0.0) or 0.0)
    current_ratio = float(feat.get("Current_Ratio", float("nan")) or float("nan")) if pd.notna(feat.get("Current_Ratio")) else float("nan")
    leverage = float(feat.get("Debt_to_Assets", 0.0) or 0.0)
    margin = float(feat.get("EBITDA_Margin", 0.0) or 0.0)
    coverage = float(feat.get("Interest_Coverage", 0.0) or 0.0)
    cash_flow = float(feat.get("Cash_Flow_Operations", 0.0) or 0.0)
    health_score = max(0.0, min(100.0, (1 - float(result["distress_probability"])) * 100.0))
    return {
        "company": company_id,
        "period": row.get("period", "latest identified period"),
        "revenue": revenue,
        "profit": profit,
        "current_ratio": current_ratio,
        "debt_to_assets": leverage,
        "ebitda_margin": margin,
        "interest_coverage": coverage,
        "cash_flow_operations": cash_flow,
        "default_probability": float(result["distress_probability"]),
        "health_score": health_score,
        "risk_category": result["risk_category"],
        "warnings": active_flags,
        "top_risk": result.get("top_risk_factors", []),
        "protective": result.get("protective_factors", []),
    }


def generate_credit_copilot_response(question: str, context: dict) -> str:
    q = (question or "").lower()
    warnings = context["warnings"]
    risk_prob = context["default_probability"]
    current_ratio = context["current_ratio"]
    leverage = context["debt_to_assets"]
    margin = context["ebitda_margin"]
    cash_flow = context["cash_flow_operations"]
    revenue = context["revenue"]
    profit = context["profit"]

    if not q:
        q = "summarize the financial health"

    if any(keyword in q for keyword in ["why", "risky", "risk", "risk factors", "biggest risk"]):
        reasons = []
        if pd.notna(current_ratio) and current_ratio < 1.0:
            reasons.append(f"liquidity is thin at {current_ratio:.2f}x")
        if leverage >= 0.65:
            reasons.append(f"debt loads are elevated at {leverage:.2f} debt-to-assets")
        if margin < 0 or cash_flow < 0:
            reasons.append("operating margin and cash generation are under pressure")
        if warnings:
            reasons.append("early-warning flags such as " + ", ".join(warnings[:3]))
        if not reasons:
            reasons.append("the model is pricing in a moderate probability of distress across revenue, leverage, and margin signals")
        return (
            f"{context['company']} is considered risky because " + "; ".join(reasons) + ". "
            f"The current risk estimate sits at {risk_prob:.1%}, and the model is particularly sensitive to leverage, liquidity, and operating cash generation."
        )

    if any(keyword in q for keyword in ["shap", "feature importance", "factors", "top risk"]):
        top_risk = context["top_risk"][:3] if context.get("top_risk") else ["Revenue pressure", "Leverage strain", "Liquidity deterioration"]
        return (
            f"The largest contributors to the distress score are {', '.join(str(item) for item in top_risk[:3])}. "
            "In practice, elevated leverage, weaker liquidity, and pressure on profitability are the strongest drivers of the current model output."
        )

    if any(keyword in q for keyword in ["credit", "lend", "considered for credit", "should this company"]):
        if risk_prob < 0.25 and current_ratio > 1.0 and cash_flow >= 0:
            return f"Based on current signals, {context['company']} appears broadly creditworthy for a cautious structure: distress probability is {risk_prob:.1%}, cash flow is positive, and liquidity remains above 1.0x."
        return f"This company should be treated cautiously. With a {risk_prob:.1%} distress probability and several active warning signals, underwriting should require tighter covenants, collateral review, or a lower exposure limit."

    if any(keyword in q for keyword in ["weakness", "weakest", "financial weaknesses"]):
        weaknesses = []
        if pd.notna(current_ratio) and current_ratio < 1.0:
            weaknesses.append(f"liquidity is weak at {current_ratio:.2f}x")
        if leverage >= 0.65:
            weaknesses.append(f"leverage is elevated at {leverage:.2f} debt-to-assets")
        if margin < 0 or cash_flow < 0:
            weaknesses.append("operating margin and cash generation are deteriorating")
        if not weaknesses:
            weaknesses.append("operating resilience is uneven despite an otherwise stable revenue base")
        return f"The main weaknesses are {', '.join(weaknesses)}. These are the items most likely to pressure repayment capacity if conditions worsen."

    if any(keyword in q for keyword in ["strength", "strengths", "what are the strengths", "healthy"]):
        strengths = []
        if revenue > 0:
            strengths.append(f"revenue base remains at {revenue:,.0f}")
        if profit > 0:
            strengths.append(f"net profit remains positive at {profit:,.0f}")
        if current_ratio > 1.0:
            strengths.append(f"liquidity remains above 1.0x at {current_ratio:.2f}x")
        if not strengths:
            strengths.append("the model still sees some positive operating characteristics even under moderate risk")
        return f"The main strengths are {', '.join(strengths)}. Those fundamentals help offset the risk profile, but they are not yet strong enough to eliminate the monitoring need."

    if any(keyword in q for keyword in ["management improve", "improve", "should management"]):
        actions = [
            "tighten working-capital discipline and reduce receivable days",
            "lower dependency on debt-funded growth and improve debt service capacity",
            "improve operating cash conversion to sustain repayment capability",
        ]
        return f"Management should prioritize {', '.join(actions)}. These steps would strengthen liquidity, reduce leverage pressure, and improve the credit standing most quickly."

    if any(keyword in q for keyword in ["increase risk", "what could cause", "credit risk to increase", "cause risk"]):
        return (
            "The risk would rise if revenue contracts materially, operating cash flow turns negative, debt builds further, "
            "or liquidity slips below 1.0x. In this scenario, the model would likely move toward a higher default probability."
        )

    if any(keyword in q for keyword in ["summarize", "summary", "financial health"]):
        return (
            f"{context['company']} is currently assessed as {context['risk_category']} with a {context['default_probability']:.1%} distress probability and a health score of {context['health_score']:.0f}/100. "
            f"Revenue is {revenue:,.0f}, net profit is {profit:,.0f}, liquidity is {current_ratio:.2f}x, and leverage is {leverage:.2f} debt-to-assets. "
            f"The main watchpoints are {', '.join(warnings[:3]) if warnings else 'limited but active risk factors'} ."
        )

    if any(keyword in q for keyword in ["early warning", "warning signals", "signals"]):
        return (
            f"The early-warning flags currently active are: {', '.join(warnings) if warnings else 'none at the current period'}. "
            "These are designed to highlight deteriorating liquidity, leverage drift, weak margins, and negative cash generation before distress becomes more visible."
        )

    if any(keyword in q for keyword in ["revenue falls", "20%", "fall by 20", "if revenue falls"]):
        adjusted_revenue = revenue * 0.8
        return (
            f"If revenue were to fall by 20%, the company would likely see a weaker margin profile and reduced cash conversion. "
            f"At an adjusted revenue level of {adjusted_revenue:,.0f}, the risk profile would likely move toward higher distress because leverage and fixed-cost pressure would become more visible."
        )

    return (
        f"{context['company']} currently shows a {context['risk_category']} profile with a {context['default_probability']:.1%} distress probability. "
        f"Liquidity, leverage, and operating cash flow are the key variables shaping the assessment, and the most immediate actions are to improve margin resilience and funding stability."
    )


PAGES = ["Executive Overview", "MSME Financial Health", "Risk Prediction", "Explainable AI",
         "Early-Warning Indicators", "Model Performance", "Methodology", "About"]


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
st.sidebar.caption("AI-Powered MSME Financial Intelligence & Early Warning System")
page = st.sidebar.radio("Workspace", PAGES, label_visibility="collapsed")
uploaded = st.sidebar.file_uploader("Upload financial CSV", type="csv")
bundle = get_bundle()

if uploaded is not None:
    try:
        frame = validate_financial_data(pd.read_csv(uploaded))
        st.sidebar.caption("Using uploaded data. Validate its provenance and target definition.")
        is_demo = False
    except Exception as error:
        st.sidebar.error(str(error))
        frame, is_demo = make_demo_data(), True
else:
    frame, is_demo = make_demo_data(), True
    st.sidebar.caption("Using synthetic illustrative demo data.")

features = engineer_features(frame)
flags = early_warning_indicators(frame)
row_index = st.sidebar.selectbox("Company-period", options=list(range(len(frame))), format_func=lambda index: (
    f"{frame.iloc[index].get('company_id', 'Row')} · {frame.iloc[index].get('period', index)}"))
selected, selected_features = frame.iloc[[row_index]], features.iloc[[row_index]]


def page_header(title: str, subtitle: str) -> None:
    st.markdown("<div class='eyebrow'>AI-POWERED MSME FINANCIAL INTELLIGENCE</div>", unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def disclaimer() -> None:
    st.markdown("<div class='risk-note'>Analytical research estimate only. Not a guaranteed prediction, lending recommendation, or financial decision.</div>", unsafe_allow_html=True)


def show_risk() -> dict:
    result = predict_financial_health(selected, bundle=bundle)
    left, middle, right = st.columns(3)
    left.metric("Estimated distress probability", f"{result['distress_probability']:.1%}")
    risk_class = {
        "Low Risk": "healthy",
        "Moderate Risk": "warning",
        "High Risk": "danger",
    }.get(result["risk_category"], "neutral")
    middle.markdown(
        f"<div class='risk-category-card' role='group' aria-label='Risk category: "
        f"{result['risk_category']}'><div class='risk-category-label'>Risk category</div>"
        f"<div class='risk-category-value {risk_class}'>{result['risk_category']}</div></div>",
        unsafe_allow_html=True,
    )
    right.metric("Confidence indicator", f"{result['confidence_indicator']:.0%}")
    st.caption(result["confidence_note"])
    disclaimer()
    return result


if page == "Executive Overview":
    result = predict_financial_health(selected, bundle=bundle, include_explanations=False)
    try:
        local_explanation = explain_prediction(bundle, selected)
        result["top_risk_factors"] = local_explanation["risk_factors"]
        result["protective_factors"] = local_explanation["protective_factors"]
    except Exception as error:
        st.warning(f"SHAP risk drivers are unavailable for this record: {error}")
    company_name = str(frame.iloc[row_index].get("company_id", "Selected Company"))
    health_score = max(0.0, min(100.0, (1 - result["distress_probability"]) * 100.0))
    current_ratio = selected_features.iloc[0].get("Current_Ratio", float("nan"))
    total_flags = int(flags.iloc[row_index].sum())
    risk_class = _state_color(result["risk_category"])
    trend_label = "Stable" if health_score >= 70 else "Watchlist" if health_score >= 45 else "Elevated"
    trend_direction = "up" if result["risk_category"] in {"Low Risk", "Moderate Risk"} else "down"

    st.markdown(
        """
        <div class="topbar">
            <div class="topbar-brand">
                <div class="eyebrow">CREDIT RISK AI</div>
                <h1>CREDIT RISK AI</h1>
                <div class="subtitle">AI-Powered MSME Financial Intelligence & Early Warning System</div>
            </div>
            <div class="live-badge">Live</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selector_cols = st.columns([2.2, 2, 2, 1.1])
    with selector_cols[0]:
        st.selectbox("Company", options=list(range(len(frame))), index=row_index, format_func=lambda idx: str(frame.iloc[idx].get("company_id", f"Company {idx + 1}")), key="company_selector")
    with selector_cols[1]:
        st.selectbox("Period", options=list(range(len(frame))), index=row_index, format_func=lambda idx: str(frame.iloc[idx].get("period", f"P{idx + 1}")), key="period_selector")
    with selector_cols[2]:
        st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
    with selector_cols[3]:
        st.button("Refresh")

    st.markdown("<div class='panel-header'><h3>Executive dashboard</h3><div class='eyebrow'>Made by Surojit Malakar • SkillseED India</div></div>", unsafe_allow_html=True)
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_kpi_card("Financial Health Score", f"{health_score:.0f}/100", "▲ Strong", "up")
    with metric_cols[1]:
        render_kpi_card("Default Risk Probability", f"{result['distress_probability']:.1%}", "▼ Risk", "down")
    with metric_cols[2]:
        render_kpi_card("Current Ratio", f"{current_ratio:.2f}x" if pd.notna(current_ratio) else "N/A", "◎ Liquid", "up")
    with metric_cols[3]:
        risk_tag = result["risk_category"].upper()
        render_kpi_card("AI Risk Status", risk_tag, trend_label, "neutral")

    render_credit_assessment_panel(company_name, result, selected_features, flags, row_index)
    active_warning_signals = flags.iloc[row_index][flags.iloc[row_index]].index.tolist()
    render_ai_risk_interpretation(result, selected_features.iloc[0], active_warning_signals)

    st.markdown("<div class='panel-header'><h3>Credit intelligence workflow</h3></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="workflow-wrapper">
            <div class="workflow-grid">
                <div class="workflow-step"><div class="title">CSV / Financial Data</div><div class="tag">Input</div></div>
                <div class="workflow-step"><div class="title">Financial Analysis</div><div class="tag">Ratio Logic</div></div>
                <div class="workflow-step"><div class="title">ML Risk Model</div><div class="tag">Prediction</div></div>
                <div class="workflow-step"><div class="title">SHAP Explainability</div><div class="tag">Drivers</div></div>
                <div class="workflow-step"><div class="title">Early Warning Engine</div><div class="tag">Signals</div></div>
                <div class="workflow-step"><div class="title">AI</div><div class="tag">Copilot</div></div>
                <div class="workflow-step"><div class="title">AI Credit Copilot</div><div class="tag">Actions</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="feature-row">
            <div class="feature-pill">Risk Explanation</div>
            <div class="feature-pill">Recommendations</div>
            <div class="feature-pill">Scenario Analysis</div>
            <div class="feature-pill">Credit Report</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_demo:
        st.info("Synthetic demo records and labels are illustrative only; they do not represent actual MSMEs.")

    st.markdown("<div class='panel-header'><h3>AI CREDIT COPILOT</h3></div>", unsafe_allow_html=True)
    copilot_context = build_credit_context(frame, selected, selected_features, flags, row_index, result)
    with st.form("credit_copilot"):
        question = st.text_area(
            "Ask the copilot about this company",
            value="Why is this company considered risky?",
            key="copilot_question",
            help="Use the company context, model output, and early-warning indicators to guide the answer.",
            height=120,
        )
        submitted = st.form_submit_button("Analyze")
    if submitted or True:
        response = generate_credit_copilot_response(question, copilot_context)
        st.markdown(f"<div class='copilot-output'>{response}</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel-header'><h3>AI Scenario Analysis</h3></div>", unsafe_allow_html=True)
    st.caption("Adjust one-period assumptions to see how the existing risk model responds. Margin change is in percentage points; revenue and debt changes are relative percentages.")
    with st.form("credit_scenario"):
        scenario_cols = st.columns(3)
        with scenario_cols[0]:
            revenue_change = st.slider("Revenue change", min_value=-20, max_value=20, value=0, step=1, format="%d%%")
        with scenario_cols[1]:
            margin_change = st.slider("Operating margin change", min_value=-10, max_value=10, value=0, step=1, format="%d pp")
        with scenario_cols[2]:
            debt_change = st.slider("Debt change", min_value=-20, max_value=20, value=0, step=1, format="%d%%")
        run_scenario = st.form_submit_button("RUN AI SIMULATION")

    if run_scenario:
        try:
            scenario_data = apply_scenario_adjustments(
                selected,
                revenue_change=revenue_change / 100,
                operating_margin_change=margin_change / 100,
                debt_change=debt_change / 100,
            )
            scenario_result = predict_financial_health(
                scenario_data,
                bundle=bundle,
                include_explanations=False,
            )
            scenario_features = engineer_features(scenario_data).iloc[0]
            scenario_warnings = early_warning_indicators(scenario_data).iloc[0]
            scenario_signals = scenario_warnings[scenario_warnings].index.tolist()
            try:
                scenario_explanation = explain_prediction(bundle, scenario_data)
                scenario_result["top_risk_factors"] = scenario_explanation["risk_factors"]
            except Exception as error:
                st.warning(f"SHAP scenario drivers are unavailable: {error}")

            probability_change_pp = (
                scenario_result["distress_probability"] - result["distress_probability"]
            ) * 100
            scenario_metric, scenario_status = st.columns(2)
            scenario_metric.metric(
                "Projected Risk",
                f"{scenario_result['distress_probability']:.1%}",
                delta=f"{probability_change_pp:+.1f} percentage points",
                delta_color="inverse",
            )
            scenario_status.metric("Current Risk", f"{result['distress_probability']:.1%}")
            render_ai_risk_interpretation(
                scenario_result,
                scenario_features,
                scenario_signals,
            )
            st.caption(
                f"Applied assumptions: revenue {revenue_change:+d}%, "
                f"operating margin {margin_change:+d} percentage points, "
                f"debt {debt_change:+d}%."
            )
        except (TypeError, ValueError) as error:
            st.error(f"Scenario could not be calculated: {error}")

    chart_cols = st.columns([1.4, 1])
    if "company_id" in frame and "period" in frame:
        series = trend_data(frame, selected.iloc[0]["company_id"])
        with chart_cols[0]:
            st.markdown("<div class='panel-header'><h3>Financial trend</h3></div>", unsafe_allow_html=True)
            if not series.empty:
                chart = px.line(
                    series,
                    x="period",
                    y=["Revenue", "Current_Ratio"],
                    markers=True,
                    title="Revenue and liquidity trend",
                    color_discrete_sequence=["#2563EB", "#10B981"],
                )
                render_chart(chart, height=260)
            else:
                st.info("Trend history is unavailable for the selected company.")
    with chart_cols[1]:
        st.markdown("<div class='panel-header'><h3>Risk drivers</h3></div>", unsafe_allow_html=True)
        top_risk = result.get("top_risk_factors", [])
        if top_risk:
            ordered = top_risk[:5]
            labels = [str(item.get("feature", "Risk factor")) for item in ordered]
            values = [float(item.get("contribution", 0.0)) for item in ordered]
            risk_fig = px.bar(
                x=values,
                y=labels,
                orientation="h",
                color=values,
                color_continuous_scale=["#1E293B", "#EF4444"],
                title="Top model drivers",
            )
            render_chart(risk_fig, height=260)
        else:
            st.write("The model has not surfaced a strong local explanation for the current record.")

    st.markdown("<div class='panel-header'><h3>Early warning signals</h3></div>", unsafe_allow_html=True)
    active = flags.iloc[row_index][flags.iloc[row_index]].index.tolist()
    if active:
        for signal in active:
            st.markdown(f"- {signal}")
    else:
        st.write("No configured warning rules are currently triggered.")

    disclaimer()

elif page == "MSME Financial Health":
    page_header("MSME Financial Health", "Statement inputs and derived ratios for the selected company-period.")
    st.dataframe(pd.concat([selected.reset_index(drop=True), selected_features.reset_index(drop=True)], axis=1).T.rename(columns={0: "Value"}), width="stretch")
    st.caption("Ratios use bounded calculations; zero denominators are treated as missing. Inventory days use a revenue proxy when COGS is unavailable.")
    disclaimer()

elif page == "Risk Prediction":
    page_header("Risk Prediction", "Estimated probability, category, and the factors behind this model score.")
    result = show_risk()
    health_score = (1 - result["distress_probability"]) * 100
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        number={"suffix": "%", "font": {"color": "#F8FAFC", "size": 42}},
        title={"text": "Estimated financial health", "font": {"color": "#F8FAFC", "size": 18}},
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
    st.caption("Health score is calculated as 100% minus the model's estimated distress probability; it is not a separate diagnosis.")
    risk, protective = st.columns(2)
    with risk:
        st.subheader("Risk-increasing factors")
        st.dataframe(pd.DataFrame(result["top_risk_factors"]), width="stretch", hide_index=True)
    with protective:
        st.subheader("Risk-reducing factors")
        st.dataframe(pd.DataFrame(result["protective_factors"]), width="stretch", hide_index=True)

elif page == "Explainable AI":
    page_header("Explainable AI", "How the selected model behaves globally and for this individual record.")
    try:
        local = explain_prediction(bundle, selected)
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
        importance = global_importance(bundle, frame)
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
    selected_flags["Status"] = selected_flags["Triggered"].map({True: "Review", False: "Not triggered"})
    st.dataframe(selected_flags.drop(columns="Triggered"), width="stretch")
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
    disclaimer()

elif page == "About":
    page_header("About Surojit Malakar", "Finance, research, operations, and technology in service of practical impact.")
    photo_column, intro_column = st.columns([1, 2], gap="large")
    with photo_column:
        st.image(
            "https://github.com/surojitmalakar.png?size=440",
            caption="Surojit Malakar",
            width=220,
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