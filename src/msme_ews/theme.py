"""Design system for the Credit Risk AI workspace.

The dashboard borrows its visual language from SkillseED India: an emerald and
cream palette, Cormorant Garamond display headings over Outfit body copy,
frosted glass cards, pill controls and soft green shadows. Every public design
token lives here so the stylesheet, the Plotly charts and the per-tab feature
cards can never drift apart.

Two things are exported:

``STYLESHEET``
    A single ``<style>`` block injected once by ``app.py``.
``PAGE_FEATURES``
    The identity of each navigation tab: its icon, accent colour, headline and
    the concrete things that tab does. ``app.py`` renders this as a feature
    hero so no two tabs look or read the same.
"""

from __future__ import annotations

from html import escape

# ── Brand palette ───────────────────────────────────────────────────────
# Mirrors the SkillseED India green ramp plus its gold accent.
G1 = "#064e3b"          # deepest forest green, headings
G2 = "#065f46"          # deep green
G3 = "#047857"          # primary action green
G4 = "#059669"          # brand green
G5 = "#10b981"
G6 = "#34d399"
G7 = "#6ee7b7"
G8 = "#a7f3d0"
G9 = "#d1fae5"
GOLD = "#d4a843"
GOLD_LIGHT = "#f0c96a"
DARK_1 = "#021a0e"
DARK_2 = "#052e16"
DARK_3 = "#063a1d"
TEXT = "#0f2d1a"
TEXT_STRONG = "#1f5235"
TEXT_MUTED = "#4d8c63"
CREAM = "#f0fdf4"
CREAM_LIGHT = "#f7fef9"
WHITE = "#ffffff"

# ── Risk semantics ─────────────────────────────────────────────────────
HEALTHY = G4
WATCH = GOLD
HIGH = "#ea7317"
CRITICAL = "#dc2626"
NEUTRAL = "#5b8c72"

CHART_SEQUENCE: tuple[str, ...] = (G3, G6, GOLD, G4, CRITICAL, DARK_3)

BAND_COLOURS: dict[str, str] = {
    "Low Risk": G4,
    "Moderate Risk": GOLD,
    "High Risk": HIGH,
    "Critical Risk": CRITICAL,
}


def band_colour(band: str) -> str:
    """Return the brand colour for a risk band label."""
    return BAND_COLOURS.get(band, NEUTRAL)


# ── Stylesheet ─────────────────────────────────────────────────────────
# Injected once by app.py. It mirrors the SkillseED India visual language:
# cream page, emerald brand ramp, frosted glass surfaces and pill controls.
STYLESHEET = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    color-scheme: light;
    /* Brand ramp lifted from the SkillseED India palette */
    --g1:#064E3B; --g2:#065F46; --g3:#047857; --g4:#059669; --g5:#10B981;
    --g6:#34D399; --g7:#6EE7B7; --g8:#A7F3D0; --g9:#D1FAE5;
    --gold:#D4A843; --gold2:#F0C96A;
    --dk:#021A0E; --dk2:#052E16; --dk3:#063A1D;
    --tx:#0F2D1A; --tx2:#1F5235; --txm:#4D8C63;
    --wh:#FFFFFF; --cr:#F0FDF4; --cf:#F7FEF9;
    --bd:rgba(16,185,129,.18); --bds:rgba(16,185,129,.09);
    --glass:rgba(255,255,255,.78); --glass2:rgba(255,255,255,.5);
    --sh1:0 4px 24px rgba(6,78,59,.08);
    --sh2:0 12px 48px rgba(6,78,59,.14);
    --sh3:0 24px 80px rgba(6,78,59,.2);
    --r1:8px; --r2:16px; --r3:24px; --r4:36px; --r5:48px;
    /* Semantic aliases used by the app */
    --page:var(--cr);
    --panel:var(--wh);
    --sidebar:var(--cf);
    --primary:var(--g3);
    --primary-soft:var(--g2);
    --accent:var(--g6);
    --healthy:var(--g4);
    --warning:var(--gold);
    --high:#EA7317;
    --danger:#DC2626;
    --text:var(--tx);
    --muted:var(--txm);
    --border:var(--bd);
    --input:#FFFFFF;
    --input-border:rgba(4,120,87,.22);
}

/* ── Page shell ── */
html, body, [class*="css"] {
    font-family:'Outfit',system-ui,-apple-system,'Segoe UI',sans-serif;
    color:var(--text);
}
html, body { width:100%; max-width:100%; overflow-x:hidden; }
.stApp, [data-testid="stAppViewContainer"] { background:var(--page); }
header[data-testid="stHeader"] { background:rgba(240,253,244,.85); }

/* Soft gradient wash, the dashboard echo of the SkillseED agri backdrop */
[data-testid="stAppViewContainer"]::before {
    content:"";
    position:fixed;
    inset:0;
    z-index:0;
    pointer-events:none;
    background:
        radial-gradient(60rem 40rem at 12% -8%, rgba(52,211,153,.16), transparent 62%),
        radial-gradient(48rem 36rem at 92% 6%, rgba(212,168,67,.13), transparent 64%),
        radial-gradient(52rem 40rem at 68% 104%, rgba(16,185,129,.14), transparent 62%);
}
[data-testid="stMain"] { position:relative; z-index:1; }

h1, h2, h3, h4, h5, h6 {
    color:var(--g1) !important;
    letter-spacing:-.02em;
    font-family:'Cormorant Garamond',Georgia,serif;
    font-weight:700;
}
p, li, label, legend, small, [data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"], [data-testid="stMarkdownContainer"],
[data-testid="stMetricLabel"], [data-testid="stMetricValue"],
[data-testid="stMetricDelta"], [data-testid="stMarkdownContainer"] *,
[data-testid="stWidgetLabel"] * { color:var(--tx2) !important; }
[role="tabpanel"] *, .stMarkdown * { color:var(--tx2) !important; }
[data-testid="stCaptionContainer"], .stCaption { color:var(--txm) !important; }
strong, b, h1 strong, h2 strong, h3 strong { color:var(--g1) !important; }
a { color:var(--g3) !important; }
[data-testid="stMainBlockContainer"] { max-width:100%; padding-top:.6rem; padding-bottom:2.5rem; }

/* ── Glass surfaces ── */
[data-testid="stMetric"], [data-testid="stDataFrame"], [data-testid="stTable"],
[data-testid="stPlotlyChart"], [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stExpander"], [data-testid="stForm"], .block-container {
    min-width:0;
    background:var(--glass);
    backdrop-filter:blur(16px) saturate(150%);
    border:1px solid var(--bd);
    border-radius:var(--r3);
    box-shadow:var(--sh1);
}
[data-testid="stMetric"] { padding:14px 16px 12px; }
[data-testid="stMetricLabel"] { color:var(--txm) !important; }
[data-testid="stMetricValue"] {
    color:var(--g1) !important;
    font-family:'Cormorant Garamond',serif;
    font-size:1.6rem;
}
[data-testid="stMetricDelta"] { font-weight:600; }
[data-testid="stDataFrame"], [data-testid="stTable"] { max-width:100%; overflow-x:auto; }
[data-testid="stPlotlyChart"] { width:100%; max-width:100%; min-width:0; }
[data-testid="stExpander"] details summary { color:var(--g2) !important; font-weight:600; }
[data-testid="stExpander"] details[open] { box-shadow:var(--sh2); }
[data-testid="stAlert"] {
    border-radius:var(--r2);
    border:1px solid var(--bd);
    background:var(--glass);
}
[data-testid="stAlert"] p { color:var(--tx2) !important; }
/* ── Sidebar ── */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background:var(--sidebar) !important;
    border-right:1px solid rgba(16,185,129,.14);
}
[data-testid="stSidebar"] { min-width:15rem !important; max-width:15rem !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6 {
    color:var(--g1) !important;
    font-family:'Cormorant Garamond',serif;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color:var(--tx2) !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:var(--txm) !important; }
[data-testid="stSidebar"] .eyebrow { color:var(--g4) !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color:var(--tx2) !important;
    border-radius:50px;
    padding:.5rem .8rem;
    margin-bottom:.12rem;
    font-size:.9rem;
    font-weight:500;
    border:1px solid transparent;
    background:transparent;
    transition:background .25s,color .25s,border-color .25s,transform .25s;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background:rgba(16,185,129,.08);
    border-color:rgba(16,185,129,.18);
    transform:translateX(2px);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background:linear-gradient(135deg,var(--g3),var(--g4));
    border-color:rgba(4,120,87,.25);
    color:#FFFFFF !important;
    font-weight:600;
    box-shadow:0 4px 18px rgba(5,150,105,.28);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) * {
    color:#FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child { display:none; }
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stFileUploader"] section,
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea {
    background:var(--input);
    border-color:var(--input-border);
    color:var(--tx);
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    border:1.5px dashed rgba(4,120,87,.35);
    border-radius:var(--r2);
    background:rgba(255,255,255,.6);
}

/* ── Inputs ── */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background:var(--input) !important;
    border:1px solid var(--input-border) !important;
    border-radius:50px !important;
    min-height:42px;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div * {
    background:var(--input) !important;
    color:var(--tx2) !important;
}
[data-baseweb="popover"] [role="listbox"] {
    background:var(--wh) !important;
    border:1px solid var(--bd) !important;
    border-radius:var(--r2) !important;
    box-shadow:var(--sh2) !important;
}
[data-baseweb="popover"] [role="listbox"] * { color:var(--tx2) !important; }
[data-baseweb="popover"] [role="option"]:hover { background:var(--g9) !important; }
.stApp input, .stApp textarea { color:var(--tx2) !important; }
.stApp input::placeholder, .stApp textarea::placeholder { color:var(--txm) !important; }
[data-testid="stWidgetLabel"] p { font-weight:600; color:var(--tx2) !important; }
[data-testid="stFileUploaderDropzone"] {
    background:var(--glass2) !important;
    border:1.5px dashed rgba(4,120,87,.35) !important;
    border-radius:var(--r3) !important;
}

/* ── Buttons ── */
.stApp button, .stApp [role="button"] {
    min-height:44px;
    color:var(--g2) !important;
    border-radius:50px;
    border:1.5px solid rgba(4,120,87,.24) !important;
    background:var(--wh);
    font-weight:600;
    transition:transform .25s cubic-bezier(.23,1,.32,1),box-shadow .25s,background .25s;
}
.stApp button:hover, .stApp [role="button"]:hover {
    background:rgba(16,185,129,.09) !important;
    border-color:rgba(4,120,87,.4) !important;
    transform:translateY(-1px);
    box-shadow:0 8px 22px rgba(6,78,59,.14);
}
.stApp [data-testid="stDownloadButton"] button,
.stApp [data-testid="stFormSubmitButton"] button,
.stApp [data-testid="baseButton-primary"] button,
.stApp button[kind="primary"] {
    background:linear-gradient(135deg,var(--g3),var(--g4)) !important;
    border-color:rgba(4,120,87,.3) !important;
    color:#FFFFFF !important;
    box-shadow:0 6px 22px rgba(5,150,105,.28);
}
.stApp [data-testid="stDownloadButton"] button:hover,
.stApp [data-testid="stFormSubmitButton"] button:hover {
    box-shadow:0 12px 32px rgba(5,150,105,.38);
}
.stApp button:disabled, .stApp [role="button"]:disabled {
    opacity:.55;
    background:rgba(255,255,255,.7) !important;
    color:var(--txm) !important;
    border-color:var(--bd) !important;
    box-shadow:none;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background:var(--g4) !important; }
[data-testid="stSlider"] [data-testid="stSliderTickBar"] { background:var(--g8) !important; }

/* ── Brand typography ── */
.eyebrow {
    display:inline-flex;
    align-items:center;
    gap:.4rem;
    font:600 11px 'Outfit',sans-serif;
    letter-spacing:.18em;
    text-transform:uppercase;
    color:var(--g4);
    margin-bottom:.1rem;
}
.topbar {
    display:flex;
    align-items:flex-end;
    justify-content:space-between;
    gap:1rem;
    flex-wrap:wrap;
    padding:.5rem 0 .9rem;
    margin-bottom:.2rem;
    border-bottom:1px solid var(--bd);
}
.brand-logo {
    width:42px; height:42px;
    border-radius:50%;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    font-size:1.15rem;
    background:linear-gradient(135deg,var(--g3),var(--g5));
    box-shadow:0 4px 16px rgba(6,78,59,.22);
    border:2px solid rgba(16,185,129,.24);
}
.topbar-brand h1 {
    margin:0;
    font-family:'Cormorant Garamond',serif;
    font-size:2.1rem;
    line-height:1.1;
    letter-spacing:-.03em;
    color:var(--g1);
}
.topbar-brand .subtitle {
    margin-top:.3rem;
    color:var(--txm);
    font-size:.95rem;
    font-weight:300;
}
.live-badge {
    display:inline-flex;
    align-items:center;
    gap:.5rem;
    padding:.32rem .8rem;
    border-radius:999px;
    background:rgba(16,185,129,.1);
    border:1px solid rgba(16,185,129,.34);
    color:var(--g3);
    font:600 11px 'Outfit',sans-serif;
    letter-spacing:.08em;
    text-transform:uppercase;
}
.live-badge::before {
    content:"";
    width:.55rem; height:.55rem;
    border-radius:50%;
    background:var(--g5);
    animation:skPulse 2.4s ease-in-out infinite;
}
@keyframes skPulse {
    0%,100% { box-shadow:0 0 0 4px rgba(16,185,129,.14); }
    50% { box-shadow:0 0 0 8px rgba(16,185,129,.04); }
}

/* ── Page hero: each tab gets its own identity ── */
.page-hero {
    position:relative;
    overflow:hidden;
    display:flex;
    align-items:center;
    gap:1.25rem;
    flex-wrap:wrap;
    padding:1.25rem 1.4rem;
    margin:.2rem 0 1.1rem;
    border-radius:var(--r4);
    background:var(--glass);
    backdrop-filter:blur(18px) saturate(160%);
    border:1px solid var(--bd);
    box-shadow:var(--sh2);
    animation:skRise .55s cubic-bezier(.23,1,.32,1) both;
}
.page-hero::before {
    content:"";
    position:absolute;
    inset:0;
    background:linear-gradient(120deg,var(--hero-soft,rgba(16,185,129,.14)) 0%,transparent 62%);
    pointer-events:none;
}
@keyframes skRise {
    from { opacity:0; transform:translateY(14px); }
    to { opacity:1; transform:none; }
}
.hero-icon {
    position:relative;
    flex:0 0 auto;
    width:58px; height:58px;
    border-radius:var(--r2);
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:1.6rem;
    background:var(--hero-tint,linear-gradient(135deg,var(--g3),var(--g5)));
    box-shadow:0 8px 26px rgba(6,78,59,.22);
    animation:skFloat 6s ease-in-out infinite;
}
@keyframes skFloat {
    0%,100% { transform:translateY(0); }
    50% { transform:translateY(-5px); }
}
.hero-text { position:relative; min-width:14rem; flex:1 1 20rem; }
.hero-tag {
    display:inline-flex;
    align-items:center;
    gap:.35rem;
    background:rgba(16,185,129,.09);
    color:var(--g3);
    border:1px solid rgba(16,185,129,.18);
    font:600 10px 'Outfit',sans-serif;
    letter-spacing:.18em;
    text-transform:uppercase;
    padding:.24rem .7rem;
    border-radius:50px;
    margin-bottom:.35rem;
}
.hero-text h2 {
    margin:0;
    font-family:'Cormorant Garamond',serif;
    font-size:1.65rem;
    font-weight:700;
    line-height:1.2;
    color:var(--g1);
    letter-spacing:-.02em;
}
.hero-text p {
    margin:.3rem 0 0;
    color:var(--txm);
    font-size:.92rem;
    font-weight:300;
    line-height:1.6;
    max-width:56ch;
}
.hero-chips { display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.6rem; }
.hero-chip {
    display:inline-flex;
    align-items:center;
    gap:.35rem;
    padding:.3rem .7rem;
    border-radius:999px;
    background:var(--wh);
    border:1px solid var(--bd);
    color:var(--tx2);
    font:600 11.5px 'Outfit',sans-serif;
    letter-spacing:.02em;
    box-shadow:var(--sh1);
}
.hero-chip .dot {
    width:.42rem; height:.42rem;
    border-radius:50%;
    background:var(--hero-chip,var(--g4));
}

/* ── Panels ── */
.panel-header {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:1rem;
    margin:1.4rem 0 .6rem;
    padding:0;
}
.panel-header h3 {
    margin:0;
    font-family:'Cormorant Garamond',serif;
    font-size:1.35rem;
    font-weight:700;
    letter-spacing:-.01em;
    color:var(--g1);
}
.panel-header h3::after {
    content:"";
    display:block;
    width:34px; height:2px;
    margin-top:.28rem;
    background:linear-gradient(90deg,var(--g4),var(--g6));
    border-radius:2px;
}
.panel {
    background:var(--glass);
    backdrop-filter:blur(16px) saturate(150%);
    border:1px solid var(--bd);
    border-radius:var(--r3);
    padding:1rem;
    box-shadow:var(--sh1);
}
.copilot-box {
    background:linear-gradient(135deg,rgba(16,185,129,.09),rgba(255,255,255,.85));
    border:1px solid rgba(16,185,129,.24);
    border-radius:var(--r3);
    padding:1rem;
    box-shadow:var(--sh1);
}
.copilot-input {
    background:var(--wh) !important;
    color:var(--tx) !important;
    border:1px solid rgba(4,120,87,.24) !important;
    border-radius:var(--r2) !important;
    min-height:76px !important;
}
.copilot-output {
    background:linear-gradient(135deg,rgba(240,253,244,.9),rgba(255,255,255,.9));
    border:1px solid rgba(16,185,129,.2);
    border-radius:var(--r2);
    padding:.9rem 1rem;
    color:var(--tx2);
    line-height:1.7;
    box-shadow:var(--sh1);
}
.chart-card {
    background:var(--glass);
    backdrop-filter:blur(16px) saturate(150%);
    border:1px solid var(--bd);
    border-radius:var(--r3);
    padding:1rem;
    box-shadow:var(--sh2);
}
.risk-summary-card {
    background:var(--glass);
    border:1px solid var(--bd);
    border-left:4px solid var(--risk-accent,var(--g4));
    border-radius:var(--r2);
    padding:.9rem 1rem;
    box-shadow:var(--sh1);
}
.risk-summary-card.healthy { --risk-accent:var(--healthy); }
.risk-summary-card.watch { --risk-accent:var(--warning); }
.risk-summary-card.high { --risk-accent:var(--high); }
.risk-summary-card.critical { --risk-accent:var(--danger); }
.risk-summary-card .risk-state {
    display:flex;
    align-items:center;
    gap:.45rem;
    font-weight:700;
    margin-bottom:.35rem;
    color:var(--g1);
    letter-spacing:.04em;
}
.risk-summary-card .risk-state::before {
    content:"";
    width:.58rem; height:.58rem;
    flex:0 0 .58rem;
    border-radius:50%;
    background:var(--risk-accent,var(--g4));
}
.risk-summary-card p { color:var(--txm) !important; margin:.3rem 0; line-height:1.6; }
.risk-note {
    border:1px solid rgba(212,168,67,.4);
    border-left:4px solid var(--gold);
    border-radius:var(--r2);
    padding:.9rem 1.05rem;
    background:rgba(253,246,222,.72);
    color:#6B5316;
    font-size:.9rem;
}
.risk-category-card {
    min-height:100%;
    padding:1rem 1.1rem;
    background:var(--glass);
    border:1px solid var(--bd);
    border-radius:var(--r3);
    box-shadow:var(--sh1);
}
.risk-category-label {
    color:var(--txm) !important;
    font-size:.74rem;
    letter-spacing:.12em;
    text-transform:uppercase;
    font-weight:600;
    font-family:'Outfit',sans-serif;
}
.risk-category-value {
    display:inline-flex;
    margin-top:.6rem;
    padding:.42rem .8rem;
    border-radius:999px;
    font-size:1.05rem;
    font-weight:700;
    letter-spacing:.04em;
}
.risk-category-value.healthy { background:rgba(16,185,129,.13); color:var(--g2) !important; }
.risk-category-value.watch { background:rgba(212,168,67,.16); color:#7A5A14 !important; }
.risk-category-value.high { background:rgba(234,115,23,.14); color:#B45309 !important; }
.risk-category-value.critical { background:rgba(220,38,38,.12); color:var(--danger) !important; }
.risk-category-value.neutral { background:rgba(15,45,26,.07); color:var(--tx2) !important; }
.driver-legend { color:var(--txm); font-size:.78rem; margin:.1rem 0 .45rem; }

/* ── Signal list ── */
.signal-list { display:grid; gap:.5rem; }
.signal-item {
    display:flex;
    align-items:flex-start;
    gap:.6rem;
    padding:.7rem .8rem;
    border:1px solid var(--bd);
    border-radius:var(--r2);
    background:var(--glass);
    line-height:1.4;
    box-shadow:var(--sh1);
}
.signal-indicator {
    flex:0 0 .58rem;
    width:.58rem; height:.58rem;
    margin-top:.3rem;
    border-radius:50%;
    background:var(--signal-color,var(--g4));
    box-shadow:0 0 0 3px rgba(16,185,129,.1);
}
.signal-indicator.critical { --signal-color:var(--danger); }
.signal-item strong { display:block; font-size:.88rem; color:var(--g1); }
.signal-item span { display:block; color:var(--txm); font-size:.79rem; margin-top:.12rem; }
.creator-credit { color:var(--txm); font-size:.78rem; font-weight:500; }
.main .stButton button, [data-testid="stFormSubmitButton"] button { width:100%; }

/* ── Assessment card ── */
.assessment-card {
    background:var(--glass);
    backdrop-filter:blur(18px) saturate(160%);
    border:1px solid var(--bd);
    border-radius:var(--r3);
    padding:1.15rem 1.25rem;
    margin:.6rem 0 1rem;
    box-shadow:var(--sh2);
}
.assessment-card h3 {
    margin:0 0 .95rem;
    font-size:1.35rem;
    letter-spacing:.02em;
    color:var(--g1);
}
.assessment-grid {
    display:grid;
    grid-template-columns:repeat(3, minmax(0, 1fr));
    gap:1rem;
    margin-bottom:1rem;
}
.assessment-item {
    padding:.75rem .85rem;
    border-radius:var(--r2);
    background:var(--glass2);
    border:1px solid var(--bd);
}
.assessment-label {
    font:600 10.5px 'Outfit',sans-serif;
    color:var(--txm);
    letter-spacing:.12em;
    text-transform:uppercase;
}
.assessment-value {
    margin-top:.35rem;
    font-family:'Cormorant Garamond',serif;
    font-size:1.25rem;
    font-weight:700;
    color:var(--g1);
}
.assessment-meta {
    display:grid;
    grid-template-columns:repeat(2, minmax(0, 1fr));
    gap:1rem;
    margin-top:.5rem;
}
.assessment-block {
    background:var(--glass2);
    border:1px solid var(--bd);
    border-radius:var(--r2);
    padding:.85rem .95rem;
}
.assessment-block h4 {
    margin:0 0 .5rem;
    font-size:.95rem;
    letter-spacing:.1em;
    text-transform:uppercase;
    color:var(--g3);
    font-family:'Outfit',sans-serif;
}
.assessment-bullets { list-style:none; padding:0; margin:0; color:var(--tx2); }
.assessment-bullets li {
    display:flex;
    align-items:center;
    gap:.55rem;
    margin:.35rem 0;
    line-height:1.4;
}
.assessment-bullets li::before {
    content:"";
    width:.48rem; height:.48rem;
    flex:0 0 .48rem;
    border-radius:50%;
    background:var(--g4);
    box-shadow:0 0 0 3px rgba(16,185,129,.12);
}
.assessment-bullets.risk li::before {
    background:var(--gold);
    box-shadow:0 0 0 3px rgba(212,168,67,.14);
}
.assessment-recommendation {
    margin-top:1rem;
    padding:.95rem 1.05rem;
    border-left:4px solid var(--g4);
    border-radius:var(--r2);
    background:linear-gradient(120deg,rgba(16,185,129,.1),rgba(255,255,255,.7));
    color:var(--g1);
    line-height:1.65;
    font-weight:500;
}
.assessment-priorities {
    margin-top:1rem;
    background:var(--glass2);
    border:1px solid var(--bd);
    border-radius:var(--r2);
    padding:.85rem .95rem;
}
.assessment-priorities ol { margin:.5rem 0 0 1.1rem; padding:0; color:var(--tx2); }
.assessment-priorities h4 {
    margin:0;
    font-size:.95rem;
    letter-spacing:.1em;
    text-transform:uppercase;
    color:var(--g3);
    font-family:'Outfit',sans-serif;
}

/* ── Workflow strip ── */
.workflow-wrapper { margin:1rem 0 1.5rem; }
.workflow-grid {
    display:grid;
    grid-template-columns:repeat(7, minmax(0, 1fr));
    gap:.85rem;
    align-items:stretch;
    margin-top:.6rem;
}
.workflow-step {
    position:relative;
    background:var(--glass);
    border:1px solid var(--bd);
    border-radius:var(--r3);
    padding:.85rem .7rem;
    min-height:110px;
    display:flex;
    flex-direction:column;
    justify-content:center;
    text-align:center;
    color:var(--tx2);
    box-shadow:var(--sh1);
    transition:transform .3s,box-shadow .3s;
}
.workflow-step:hover { transform:translateY(-3px); box-shadow:var(--sh2); }
.workflow-step .title { font-size:.84rem; font-weight:600; line-height:1.4; color:var(--g1); }
.workflow-step .tag {
    display:inline-block;
    margin-top:.4rem;
    font:600 10px 'Outfit',sans-serif;
    letter-spacing:.1em;
    color:var(--g4);
    text-transform:uppercase;
}
.workflow-step::after {
    content:"→";
    position:absolute;
    right:-.62rem;
    top:50%;
    transform:translateY(-50%);
    color:var(--g5);
    font-weight:700;
    font-size:1.05rem;
}
.workflow-step:last-child::after { content:""; }

/* ── Feature pills ── */
.feature-row { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:.8rem; }
.feature-pill {
    display:inline-flex;
    align-items:center;
    padding:.4rem .78rem;
    border-radius:999px;
    background:rgba(16,185,129,.08);
    border:1px solid rgba(16,185,129,.2);
    color:var(--g2);
    font:600 12px 'Outfit',sans-serif;
    letter-spacing:.04em;
}

/* ── KPI cards ── */
.kpi-card {
    background:var(--glass);
    backdrop-filter:blur(16px) saturate(150%);
    border:1px solid var(--bd);
    border-radius:var(--r3);
    padding:.95rem 1.05rem;
    min-height:120px;
    position:relative;
    overflow:hidden;
    box-shadow:var(--sh1);
    transition:transform .35s cubic-bezier(.23,1,.32,1),box-shadow .35s,border-color .35s;
}
.kpi-card:hover {
    transform:translateY(-4px);
    box-shadow:var(--sh3);
    border-color:rgba(16,185,129,.3);
}
.kpi-card::after {
    content:"";
    position:absolute;
    inset:auto 0 0 0;
    height:3px;
    background:linear-gradient(90deg,var(--kpi-accent,var(--g4)),transparent);
}
.kpi-label {
    color:var(--txm);
    font-size:.78rem;
    font-weight:600;
    letter-spacing:.06em;
    text-transform:uppercase;
}
.kpi-value {
    margin-top:.4rem;
    font-family:'Cormorant Garamond',serif;
    font-size:clamp(1.7rem,2.8vw,2.15rem);
    font-weight:700;
    letter-spacing:-.03em;
    color:var(--g1);
    line-height:1.1;
}
.kpi-trend {
    margin-top:.45rem;
    display:inline-flex;
    align-items:center;
    gap:.4rem;
    padding:.26rem .62rem;
    border-radius:999px;
    font:600 12px 'Outfit',sans-serif;
    background:rgba(16,185,129,.07);
    border:1px solid rgba(16,185,129,.18);
}
.kpi-trend.healthy { color:var(--g3); }
.kpi-trend.watch { color:#9A6F1E; }
.kpi-trend.high { color:#B45309; }
.kpi-trend.critical { color:var(--danger); }
.kpi-trend.neutral { color:var(--tx2); }
.kpi-card.healthy { --kpi-accent:var(--healthy); }
.kpi-card.watch { --kpi-accent:var(--warning); }
.kpi-card.high { --kpi-accent:var(--high); }
.kpi-card.critical { --kpi-accent:var(--danger); }
.kpi-card.neutral { --kpi-accent:var(--accent); }

/* ── About ── */
.profile-card {
    background:var(--glass);
    backdrop-filter:blur(18px) saturate(160%);
    border:1px solid var(--bd);
    border-radius:var(--r4);
    padding:1.5rem;
    text-align:center;
    box-shadow:var(--sh2);
}
.profile-photo {
    width:150px; height:150px;
    border-radius:50%;
    object-fit:cover;
    border:3px solid var(--wh);
    box-shadow:0 10px 34px rgba(6,78,59,.2);
    margin:0 auto .9rem;
    display:block;
}
.profile-name {
    font-family:'Cormorant Garamond',serif;
    font-size:1.5rem;
    font-weight:700;
    color:var(--g1);
    margin:0;
    line-height:1.2;
}
.profile-role { font-size:.85rem; color:var(--g3); font-weight:600; margin-top:.15rem; }
.profile-org { font-size:.82rem; color:var(--txm); margin-top:.35rem; }
.profile-links { display:flex; flex-wrap:wrap; gap:.45rem; justify-content:center; margin-top:1rem; }
.profile-link {
    display:inline-flex;
    align-items:center;
    gap:.4rem;
    padding:.42rem .85rem;
    border-radius:999px;
    background:linear-gradient(135deg,var(--g3),var(--g4));
    color:#FFFFFF !important;
    font:600 12px 'Outfit',sans-serif;
    text-decoration:none;
    box-shadow:0 4px 16px rgba(5,150,105,.26);
    transition:transform .25s,box-shadow .25s;
}
.profile-link:hover { transform:translateY(-2px); box-shadow:0 8px 24px rgba(5,150,105,.34); }
.profile-link.ghost {
    background:var(--wh);
    color:var(--g2) !important;
    border:1.5px solid rgba(4,120,87,.26);
    box-shadow:var(--sh1);
}
.about-block {
    background:var(--glass);
    backdrop-filter:blur(16px) saturate(150%);
    border:1px solid var(--bd);
    border-left:4px solid var(--about-accent,var(--g4));
    border-radius:var(--r3);
    padding:1.1rem 1.25rem;
    margin-bottom:1rem;
    box-shadow:var(--sh1);
}
.about-block h3 {
    margin:0 0 .5rem;
    font-family:'Cormorant Garamond',serif;
    font-size:1.3rem;
    color:var(--g1);
}
.about-grid {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));
    gap:1rem;
    margin:.2rem 0 1rem;
}
.about-tile {
    background:var(--glass);
    backdrop-filter:blur(16px) saturate(150%);
    border:1px solid var(--bd);
    border-radius:var(--r3);
    padding:1.05rem 1.15rem;
    box-shadow:var(--sh1);
    transition:transform .35s cubic-bezier(.23,1,.32,1),box-shadow .35s;
}
.about-tile:hover { transform:translateY(-4px); box-shadow:var(--sh2); }
.about-tile .tile-icon { font-size:1.5rem; }
.about-tile h4 { margin:.45rem 0 .3rem; font-size:1.1rem; color:var(--g1); }
.about-tile p { margin:0; font-size:.88rem; color:var(--txm); line-height:1.6; font-weight:300; }

/* ── Site footer ── */
.site-footer {
    margin:2rem 0 1.2rem;
    padding:1.6rem clamp(1rem, 4vw, 2.4rem);
    background:linear-gradient(135deg,var(--dk2),var(--dk));
    border-radius:var(--r3);
    color:rgba(255,255,255,.82);
    box-shadow:var(--sh2);
}
.site-footer .footer-brand {
    margin:0;
    font-family:'Cormorant Garamond',serif;
    font-size:1.35rem;
    font-weight:700;
    color:#fff;
}
.site-footer .footer-tag {
    margin:.15rem 0 .9rem;
    font-size:.74rem;
    font-weight:600;
    letter-spacing:.12em;
    text-transform:uppercase;
    color:var(--g8);
}
.site-footer .footer-row {
    display:flex;
    flex-wrap:wrap;
    gap:.4rem 1.5rem;
    font-size:.85rem;
    font-weight:300;
}
.site-footer .footer-note {
    display:flex;
    flex-wrap:wrap;
    justify-content:space-between;
    gap:.4rem 1.5rem;
    margin-top:1rem;
    padding-top:.9rem;
    border-top:1px solid rgba(255,255,255,.14);
    font-size:.78rem;
    color:rgba(255,255,255,.6);
}
.site-footer a { color:var(--g8); text-decoration:none; }
.site-footer a:hover { color:#fff; }

/* ── Responsive ── */
@media (max-width: 767px) {
    [data-testid="stMainBlockContainer"] { padding:1rem clamp(12px, 4vw, 18px) 2rem; }
    [data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:.65rem; min-width:0; }
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
    [data-testid="stSelectbox"], [data-testid="stFileUploader"],
    [data-baseweb="popover"], [role="listbox"] {
        width:100%; max-width:100vw; min-width:0;
    }
    [role="listbox"] { max-height:min(55vh, 24rem); overflow-y:auto; }
    [data-testid="stMetric"] { width:100%; padding:14px; }
    .kpi-card { min-height:0; padding:.8rem .9rem; }
    .topbar { padding:.15rem 0 .5rem; }
    .topbar-brand .subtitle { font-size:.86rem; line-height:1.4; }
    .panel-header { margin:.9rem 0 .5rem; }
    .panel { padding:.7rem; }
    .assessment-card { padding:.85rem; margin:.7rem 0 1rem; }
    .assessment-grid, .assessment-meta, .workflow-grid { grid-template-columns:1fr; }
    .workflow-step::after { content:""; }
    .about-grid { grid-template-columns:1fr; }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        max-width:100%; overflow-x:auto; white-space:nowrap;
    }
    [data-testid="stPlotlyChart"], [data-testid="stImage"],
    [data-testid="stFileUploader"], [data-baseweb="select"] {
        width:100%; max-width:100%; min-width:0;
    }
    .page-hero { padding:1rem; gap:.9rem; }
    .hero-icon { width:48px; height:48px; font-size:1.3rem; }
    .hero-text h2 { font-size:1.4rem; }
    .profile-photo { width:120px; height:120px; }
    h1 { font-size:clamp(1.7rem, 7vw, 2.1rem); }
    h2 { font-size:clamp(1.35rem, 5.5vw, 1.65rem); }
    h3 { font-size:clamp(1.15rem, 4.5vw, 1.35rem); }
}
@media (prefers-reduced-motion: reduce) {
    .page-hero, .hero-icon, .live-badge::before, .kpi-card { animation:none !important; }
    .kpi-card:hover, .about-tile:hover, .workflow-step:hover { transform:none; }
}
</style>
"""

# ── Per-tab identity ────────────────────────────────────────────────────
# Every navigation tab owns a distinct feature set, icon and accent colour so
# the workspace never reads as one long undifferentiated page. Each entry:
#   icon   emoji shown in the hero medallion
#   accent gradient + soft wash used by the hero
#   tag    short category label
#   title  the tab's own headline
#   blurb  one sentence on what this tab is for
#   chips  the concrete features the tab offers
PAGE_FEATURES: dict[str, dict[str, object]] = {
    "Executive Overview": {
        "icon": "🌿", "accent": G3, "tag": "Command centre",
        "title": "Every credit signal, one screen",
        "blurb": "The landing view of the workspace: the selected company's health, risk drivers and trend in a single pass.",
        "chips": ["Health score card", "Top risk drivers", "Revenue & liquidity trend", "Recommendations"],
    },
    "Data Intelligence": {
        "icon": "🤖", "accent": G1, "tag": "Data detective",
        "title": "Let the dataset describe itself",
        "blurb": "Schema detection, profiling and anomaly spotting, with lazy-loaded charts that never block the first paint.",
        "chips": ["Schema detection", "Anomaly detection", "Lazy visual analytics", "PDF + Excel brief"],
    },
    "MSME Financial Health": {
        "icon": "📊", "accent": G6, "tag": "Statement health",
        "title": "See exactly what the statement contains",
        "blurb": "Shows the fields that were actually extracted and how complete they are, so coverage is never overstated.",
        "chips": ["Extracted fields", "Core-field coverage", "Bounded ratios", "Zero-denominator safety"],
    },
    "Risk Prediction": {
        "icon": "🎯", "accent": GOLD, "tag": "Risk estimate",
        "title": "One number, with its evidence attached",
        "blurb": "A health score gauge paired with the drivers that moved it, so the estimate is never a bare figure.",
        "chips": ["Health gauge", "Rule-based index", "Risk vs protective", "Coverage disclosure"],
    },
    "Early-Warning Indicators": {
        "icon": "⚠️", "accent": HIGH, "tag": "Warning engine",
        "title": "Catch deterioration before the default",
        "blurb": "Transparent heuristics flag deteriorating conditions and name the value behind each trigger.",
        "chips": ["7 heuristic rules", "Evidence per signal", "Trigger vs not-assessed", "Historical measure"],
    },
    "Reports": {
        "icon": "📤", "accent": G3, "tag": "Exports",
        "title": "Take the analysis out of the session",
        "blurb": "Every export is built locally from the active upload: dataset briefs, screening workbooks, and company reports.",
        "chips": ["PDF + Excel briefs", "Screening workbook", "Company reports", "Generated on demand"],
    },
    "Methodology": {
        "icon": "📚", "accent": G1, "tag": "Research notes",
        "title": "Read the rules before trusting the score",
        "blurb": "Definitions, evaluation choices, fairness limits and the complete sparse-data risk index, disclosed in full.",
        "chips": ["Feature glossary", "Fairness & bias", "Limitations", "Full rule table"],
    },
    "About": {
        "icon": "👤", "accent": GOLD, "tag": "The builder",
        "title": "Built by Surojit Malakar",
        "blurb": "The student, researcher and social entrepreneur behind Credit Risk AI and SkillseED India.",
        "chips": ["GitHub profile", "Academic background", "Social enterprise", "Research frameworks"],
    },
}


def page_feature(page: str) -> dict[str, object]:
    """Return the identity of a tab, falling back to a neutral default."""
    return PAGE_FEATURES.get(
        page,
        {
            "icon": "📄", "accent": G3, "tag": "Workspace",
            "title": str(page),
            "blurb": "This part of the Credit Risk AI workspace.",
            "chips": [],
        },
    )


# ── About: the person behind the project ───────────────────────────────
GITHUB_USERNAME = "surojitmalakar"
GITHUB_URL = f"https://github.com/{GITHUB_USERNAME}"
GITHUB_AVATAR_URL = f"https://avatars.githubusercontent.com/u/198892596?v=4"
PORTFOLIO_URL = "https://www.skillseedindia.com"

BIO_SUMMARY = (
    "I am Surojit Malakar, a BBA (Honours with Research) student at Techno India "
    "University, Kolkata, specializing in Finance, Business Analytics, and Operations. "
    "My interests lie at the intersection of financial analysis, business strategy, "
    "operations, research, and social entrepreneurship."
)

CAREER_INTENT = (
    "My long-term goal is to build a career where I can combine financial and analytical "
    "thinking with research, strategic decision-making, and practical business execution, "
    "while continuing to develop initiatives that create measurable economic and social impact."
)

ENTERPRENEURSHIP = (
    "I am the Founder of SkillseED India, a rural social enterprise focused on skill "
    "development, agribusiness education, and farmer empowerment. Through SkillseED India, "
    "I have worked with students and farmers across West Bengal and developed initiatives "
    "focused on practical education, skill development, and agricultural business opportunities. "
    "I am also the Founder and Developer of ZynoqIndia, an eco-tourism and open-access research "
    "publishing platform that combines business strategy with technology, including product "
    "development, UI/UX, React, Vite, Supabase, and digital publishing systems."
)

RESEARCH_FRAMEWORKS = (
    "My research work focuses on solving practical business and economic problems through "
    "structured analytical frameworks. I developed the Farm-to-Consumer (F2C 4.0) Model, which "
    "focuses on farmer entrepreneurship education, institutional market access, and cooperative "
    "business development. I have also developed Decision Paralysis Economics (DPE), an analytical "
    "framework examining the organizational and economic costs associated with delayed "
    "decision-making in environments affected by information overload and AI-mediated ambiguity. "
    "My academic research experience includes primary data collection, survey research, "
    "statistical analysis, SPSS-based reporting, and the development of analytical frameworks."
)

EXPERIENCE_INTERESTS = (
    "Alongside my research and entrepreneurial work, I have gained experience in HR operations, "
    "project management, financial analysis, business operations, and community development. My "
    "professional experience has allowed me to work on recruitment, process improvement, "
    "cross-functional coordination, research, and program management. I am particularly "
    "interested in finance, business analytics, operations strategy, agribusiness, financial "
    "decision-making, research, and technology-enabled business models."
)

# Short cards under the profile photo; each highlights a different facet.
ABOUT_TILES: tuple[tuple[str, str, str], ...] = (
    (
        "🎓",
        "Finance & Analytics",
        "BBA (Honours with Research) at Techno India University, Kolkata, with a focus on "
        "Finance, Business Analytics and Operations.",
    ),
    (
        "🌾",
        "Social entrepreneurship",
        "Founder of SkillseED India, a registered rural social enterprise working across "
        "West Bengal on skills, education and farmer empowerment.",
    ),
    (
        "💻",
        "Technology builder",
        "Founder and developer of ZynoqIndia, an eco-tourism and open-access research "
        "publishing platform built with React, Vite and Supabase.",
    ),
    (
        "📐",
        "Research frameworks",
        "Author of the Farm-to-Consumer (F2C 4.0) Model and Decision Paralysis Economics (DPE), "
        "grounded in survey research and statistical analysis.",
    ),
)


def profile_card_html() -> str:
    """Return the About-tab profile card with the GitHub photograph."""
    return f"""
<div class="profile-card">
    <img class="profile-photo" src="{escape(GITHUB_AVATAR_URL)}"
         alt="GitHub profile photograph of Surojit Malakar" width="150" height="150">
    <p class="profile-name">Surojit Malakar</p>
    <div class="profile-role">BBA (Honours with Research) · Techno India University, Kolkata</div>
    <div class="profile-org">Finance · Business Analytics · Operations</div>
    <div class="profile-links">
        <a class="profile-link" href="{escape(GITHUB_URL)}" target="_blank" rel="noopener noreferrer">
            GitHub Profile
        </a>
        <a class="profile-link ghost" href="{escape(PORTFOLIO_URL)}" target="_blank" rel="noopener noreferrer">
            SkillseED India
        </a>
    </div>
</div>
"""


def about_tiles_html() -> str:
    """Return the grid of focus-area tiles for the About tab."""
    tiles = "".join(
        f'<div class="about-tile"><div class="tile-icon">{escape(icon)}</div>'
        f'<h4>{escape(title)}</h4><p>{escape(body)}</p></div>'
        for icon, title, body in ABOUT_TILES
    )
    return f'<div class="about-grid">{tiles}</div>'


def about_block_html(heading: str, body: str, accent: str = G3) -> str:
    """Return one narrative block on the About tab."""
    return (
        f'<div class="about-block" style="--about-accent:{escape(accent)}">'
        f'<h3>{escape(heading)}</h3>'
        f'<p>{escape(body)}</p>'
        f'</div>'
    )
def page_hero_html(page: str) -> str:
    """Build the hero banner HTML for one navigation tab.

    Kept free of Streamlit imports so it can be unit-tested directly.
    """
    feature = page_feature(page)
    accent = str(feature["accent"])
    chips = "".join(
        f'<span class="hero-chip" style="--hero-chip:{accent}">'
        f'<span class="dot"></span>{escape(str(chip))}</span>'
        for chip in feature["chips"]
    )
    return (
        f'<div class="page-hero" style="--hero-tint:linear-gradient(135deg,{accent},{accent}dd);'
        f'--hero-soft:{accent}1f">'
        f'<div class="hero-icon">{escape(str(feature["icon"]))}</div>'
        f'<div class="hero-text">'
        f'<div class="hero-tag">{escape(str(feature["tag"]))}</div>'
        f'<h2>{escape(str(feature["title"]))}</h2>'
        f'<p>{escape(str(feature["blurb"]))}</p>'
        f'<div class="hero-chips">{chips}</div>'
        f'</div></div>'
    )


def site_footer_html() -> str:
    """Return the SkillseED-style footer rendered under every dashboard tab.

    Mirrors the live skillseedindia.com footer: the deep-green brand block, the
    tagline, contact channels and the standing research-estimate disclaimer.
    Kept free of Streamlit imports so it can be unit-tested directly.
    """
    return f"""
    <footer class="site-footer">
        <div class="footer-brand">SkillseED India</div>
        <div class="footer-tag">Beyond Knowledge Into Action</div>
        <div class="footer-row">
            <span>✉ <a href="mailto:skillseedindia@gmail.com">skillseedindia@gmail.com</a></span>
            <span>🌐 <a href="{escape(PORTFOLIO_URL)}" target="_blank" rel="noopener noreferrer">skillseedindia.com</a></span>
            <span>📍 Rural Development Center, India</span>
        </div>
        <div class="footer-note">
            <span>Credit Risk AI · analytical research estimates only, not lending decisions.</span>
            <span>© 2026 SkillseED India · Built by Surojit Malakar</span>
        </div>
    </footer>
    """