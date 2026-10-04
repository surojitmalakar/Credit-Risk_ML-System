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
from html import escape

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
    page_icon="C",
    layout="wide",
    initial_sidebar_state="auto",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');
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
            "<div class='signal-item'><span class='signal-indicator healthy'></span>"
            "<div><strong>Healthy</strong><span>No configured early-warning conditions are triggered.</span></div></div>",
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


NAVIGATION = {
    "Overview": "Executive Overview",
    "Financial Health": "MSME Financial Health",
    "Risk Prediction": "Risk Prediction",
    "AI Copilot": "AI Copilot",
    "Explainable AI": "Explainable AI",
    "Early Warning": "Early-Warning Indicators",
    "Scenario Simulator": "Scenario Simulator",
    "Credit Assessment": "Credit Assessment",
    "Model Performance": "Model Performance",
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
bundle = get_bundle()

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
    uploaded = st.file_uploader("Upload financial CSV", type="csv", key="financial_csv_upload")
if uploaded is not None:
    try:
        frame = validate_financial_data(pd.read_csv(uploaded))
        st.caption("Using uploaded data. Validate its provenance and target definition.")
        is_demo = False
    except Exception as error:
        st.error(str(error))
        frame, is_demo = make_demo_data(), True
else:
    frame, is_demo = make_demo_data(), True
    st.caption("Using synthetic illustrative demo data.")

features = engineer_features(frame)
flags = early_warning_indicators(frame)
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
    risk_class = _state_color(result["risk_category"])
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


if page in {"Executive Overview", "AI Copilot", "Scenario Simulator", "Credit Assessment"}:
    is_overview = page == "Executive Overview"
    result = predict_financial_health(selected, bundle=bundle, include_explanations=False)
    active_warning_signals = flags.iloc[row_index][flags.iloc[row_index]].index.tolist()
    if page in {"Executive Overview", "AI Copilot"}:
        try:
            local_explanation = explain_prediction(bundle, selected)
            result["top_risk_factors"] = local_explanation["risk_factors"]
            result["protective_factors"] = local_explanation["protective_factors"]
        except Exception as error:
            st.warning(f"SHAP risk drivers are unavailable for this record: {error}")
    company_name = str(frame.iloc[row_index].get("company_id", "Selected Company"))
    health_score = max(0.0, min(100.0, (1 - result["distress_probability"]) * 100.0))
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
        health_label = "Strong" if health_score >= 70 else "Watch" if health_score >= 40 else "Critical"
        metric_cols = st.columns(4)
        with metric_cols[0]:
            render_kpi_card("Financial Health", f"{health_score:.0f}/100", health_label, risk_class)
        with metric_cols[1]:
            render_kpi_card("Default Risk", f"{result['distress_probability']:.1%}", result["risk_category"], risk_class)
        with metric_cols[2]:
            render_kpi_card(
                "Current Ratio",
                f"{current_ratio:.2f}x" if pd.notna(current_ratio) else "N/A",
                "Healthy" if ratio_state == "healthy" else "Watch" if ratio_state == "watch" else "Unavailable",
                ratio_state,
            )
        with metric_cols[3]:
            render_kpi_card("AI Risk Status", result["risk_category"].upper(), "Stable" if risk_class == "healthy" else result["risk_category"], risk_class)

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
            float(result["distress_probability"]),
            uploaded.name if uploaded is not None else "synthetic-demo",
            uploaded.size if uploaded is not None else 0,
            question,
        )
        if submitted or st.session_state.get("credit_copilot_signature") != response_signature:
            response = generate_credit_copilot_response(question, copilot_context)
            st.session_state["credit_copilot_response"] = response
            st.session_state["credit_copilot_signature"] = response_signature
        if st.session_state.get("credit_copilot_response"):
            st.markdown(
                f"<div class='copilot-output'>{escape(st.session_state['credit_copilot_response'])}</div>",
                unsafe_allow_html=True,
            )

    if page in {"Executive Overview", "AI Copilot"}:
        render_ai_risk_interpretation(result, selected_features.iloc[0], active_warning_signals)

    if is_overview or page == "AI Copilot":
        st.markdown("<div class='panel-header'><h3>Top Risk Drivers</h3></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='driver-legend'>Positive contribution increases distress output · Negative contribution reduces it</div>",
            unsafe_allow_html=True,
        )
        render_shap_drivers(result)

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
                    scenario_result["protective_factors"] = scenario_explanation["protective_factors"]
                except Exception as error:
                    st.warning(f"SHAP scenario drivers are unavailable: {error}")

                probability_change_pp = (
                    scenario_result["distress_probability"] - result["distress_probability"]
                ) * 100
                scenario_metric, scenario_status = st.columns(2)
                scenario_metric.metric(
                    "Projected risk",
                    f"{scenario_result['distress_probability']:.1%}",
                    delta=f"{probability_change_pp:+.1f} percentage points",
                    delta_color="inverse",
                )
                scenario_status.metric("Current risk", f"{result['distress_probability']:.1%}")
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

elif page == "MSME Financial Health":
    page_header("MSME Financial Health", "Statement inputs and derived ratios for the selected company-period.")
    financial_summary = pd.concat(
        [selected.reset_index(drop=True), selected_features.reset_index(drop=True)],
        axis=1,
    ).T.rename(columns={0: "Value"}).astype(str)
    st.dataframe(financial_summary, width="stretch")
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