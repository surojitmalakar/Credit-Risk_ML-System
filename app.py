"""Eight-page Streamlit research dashboard for MSME financial distress."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from msme_ews.data import validate_financial_data
from msme_ews.demo import make_demo_data
from msme_ews.early_warning import early_warning_indicators, trend_data
from msme_ews.explain import explain_prediction, global_importance
from msme_ews.features import engineer_features
from msme_ews.modeling import train_models
from msme_ews.prediction import DEFAULT_MODEL_PATH, predict_financial_health

st.set_page_config(page_title="MSME Financial Early Warning", page_icon="◈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#19342d; --muted:#60736d; --mint:#d9efe4; --coral:#d97757; --paper:#f5f7f3; }
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; color:var(--ink); }
.stApp { background: radial-gradient(ellipse at 90% 0%, #e9f2e8 0, transparent 36%), var(--paper); }
h1, h2, h3 { color:var(--ink); letter-spacing:0; }
[data-testid="stMetric"] { background:rgba(255,255,255,.72); border:1px solid #dfe8df; padding:14px 16px; border-radius:6px; }
[data-testid="stSidebar"] { background:#edf3ec; border-right:1px solid #dbe5dc; }
[data-testid="stDataFrame"] { border:1px solid #dfe8df; border-radius:6px; }
.eyebrow { font:500 11px 'DM Mono',monospace; color:#61776e; text-transform:uppercase; }
.risk-note { border-left:3px solid #d97757; padding:10px 14px; background:#fff8f4; color:#573b31; }
</style>
""", unsafe_allow_html=True)

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


st.sidebar.markdown("<div class='eyebrow'>RESEARCH / MSME CREDIT HEALTH</div>", unsafe_allow_html=True)
st.sidebar.title("Early Warning")
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
    st.markdown("<div class='eyebrow'>MSME FINANCIAL DISTRESS / RESEARCH CONSOLE</div>", unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def disclaimer() -> None:
    st.markdown("<div class='risk-note'>Analytical research estimate only. Not a guaranteed prediction, lending recommendation, or financial decision.</div>", unsafe_allow_html=True)


def show_risk() -> dict:
    result = predict_financial_health(selected, bundle=bundle)
    left, middle, right = st.columns(3)
    left.metric("Estimated distress probability", f"{result['distress_probability']:.1%}")
    middle.metric("Risk category", result["risk_category"])
    right.metric("Confidence indicator", f"{result['confidence_indicator']:.0%}")
    st.caption(result["confidence_note"])
    disclaimer()
    return result


if page == "Executive Overview":
    page_header("Financial distress, before it compounds", "A research view of liquidity, leverage, and operating resilience.")
    if is_demo:
        st.info("Synthetic demo records and labels are illustrative only; they do not represent actual MSMEs.")
    result = predict_financial_health(selected, bundle=bundle, include_explanations=False)
    total_flags = int(flags.iloc[row_index].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Businesses / periods", f"{len(frame):,}")
    c2.metric("Selected distress estimate", f"{result['distress_probability']:.1%}")
    c3.metric("Active warning signals", str(total_flags))
    current_ratio = selected_features.iloc[0]["Current_Ratio"]
    c4.metric("Current ratio", f"{current_ratio:.2f}x" if pd.notna(current_ratio) else "Unavailable")
    disclaimer()
    if "company_id" in frame and "period" in frame:
        series = trend_data(frame, selected.iloc[0]["company_id"])
        if not series.empty:
            chart = px.line(series, x="period", y=["Revenue", "Current_Ratio"], markers=True, title="Selected business: revenue and liquidity trend")
            chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title_text="Measure")
            st.plotly_chart(chart, use_container_width=True)
    st.subheader("Current signals")
    active = flags.iloc[row_index][flags.iloc[row_index]].index.tolist()
    st.write(", ".join(active) if active else "No configured warning rules are currently triggered.")

elif page == "MSME Financial Health":
    page_header("MSME Financial Health", "Statement inputs and derived ratios for the selected company-period.")
    st.dataframe(pd.concat([selected.reset_index(drop=True), selected_features.reset_index(drop=True)], axis=1).T.rename(columns={0: "Value"}), use_container_width=True)
    st.caption("Ratios use bounded calculations; zero denominators are treated as missing. Inventory days use a revenue proxy when COGS is unavailable.")
    disclaimer()

elif page == "Risk Prediction":
    page_header("Risk Prediction", "Estimated probability, category, and the factors behind this model score.")
    result = show_risk()
    health_score = (1 - result["distress_probability"]) * 100
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        number={"suffix": "%", "font": {"color": "#19342d", "size": 42}},
        title={"text": "Estimated financial health", "font": {"color": "#19342d", "size": 18}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": "#448568", "thickness": 0.25},
            "bgcolor": "rgba(255,255,255,.55)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#f1d4c9"},
                {"range": [40, 70], "color": "#f4e6bd"},
                {"range": [70, 100], "color": "#d9efe4"},
            ],
            "threshold": {
                "line": {"color": "#19342d", "width": 3},
                "thickness": 0.8,
                "value": health_score,
            },
        },
    ))
    gauge.update_layout(
        height=260,
        margin={"t": 55, "b": 20, "l": 35, "r": 35},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(gauge, use_container_width=True)
    st.caption("Health score is calculated as 100% minus the model's estimated distress probability; it is not a separate diagnosis.")
    risk, protective = st.columns(2)
    with risk:
        st.subheader("Risk-increasing factors")
        st.dataframe(pd.DataFrame(result["top_risk_factors"]), use_container_width=True, hide_index=True)
    with protective:
        st.subheader("Risk-reducing factors")
        st.dataframe(pd.DataFrame(result["protective_factors"]), use_container_width=True, hide_index=True)

elif page == "Explainable AI":
    page_header("Explainable AI", "How the selected model behaves globally and for this individual record.")
    try:
        local = explain_prediction(bundle, selected)
        st.subheader("Why did the model assign this risk?")
        st.caption("Positive SHAP contributions increase the model output for distress; negative contributions reduce it. Associations are not causal.")
        bars = local["contributions"].copy()
        bars["direction"] = bars["shap_value"].map(lambda value: "Risk increased" if value > 0 else "Risk reduced")
        fig = px.bar(bars.sort_values("shap_value"), x="shap_value", y="feature", color="direction", orientation="h", color_discrete_map={"Risk increased": "#c66849", "Risk reduced": "#448568"}, title="Individual SHAP contributions")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        importance = global_importance(bundle, frame)
        global_fig = px.bar(importance.head(15).sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h", title="Global feature importance · mean absolute SHAP")
        global_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(global_fig, use_container_width=True)
    except Exception as error:
        st.error(f"SHAP explanation could not be calculated in this environment: {error}")
    disclaimer()

elif page == "Early-Warning Indicators":
    page_header("Early-Warning Indicators", "Transparent heuristics that flag potentially deteriorating conditions.")
    st.caption("Rules are configurable research heuristics, not learned predictions or universal thresholds.")
    selected_flags = flags.iloc[[row_index]].T.rename(columns={row_index: "Triggered"})
    selected_flags["Status"] = selected_flags["Triggered"].map({True: "Review", False: "Not triggered"})
    st.dataframe(selected_flags.drop(columns="Triggered"), use_container_width=True)
    if "company_id" in frame and "period" in frame:
        series = trend_data(frame, selected.iloc[0]["company_id"])
        if len(series) > 1:
            metric = st.selectbox("Historical measure", ["Revenue", "Current_Ratio", "Debt_to_Assets", "EBITDA_Margin", "Interest_Coverage"])
            fig = px.line(series, x="period", y=metric, markers=True, title=f"{metric.replace('_', ' ')} over time")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
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
    st.dataframe(pd.DataFrame([report["test_metrics"]]).T.rename(columns={0: "Held-out test"}), use_container_width=True)
    st.subheader("Cross-validation comparison")
    st.dataframe(pd.DataFrame(report["cv_metrics"]).T.style.format("{:.3f}"), use_container_width=True)
    if report.get("fairness"):
        st.subheader("Audit-only group diagnostics")
        st.dataframe(pd.DataFrame(report["fairness"]).T, use_container_width=True)
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
    st.dataframe(pd.DataFrame({"Feature": features.columns, "Definition": ["Raw numeric statement field" if name in frame.columns else "Engineered ratio; see project README" for name in features.columns]}), use_container_width=True, hide_index=True)
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