"""Ortak "Liquid Glass" tasarım sistemi (CSS).

streamlit_app.py, app_finetune_rag.py (Sohbet sayfası) ve pages_ui/portfolio_page.py
(myPortfolio sayfası) tarafından ortak kullanılır — böylece iki sayfa da aynı
görsel dile sahip olur.
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --ekofin-bg: #f5f5f7;
    --ekofin-surface: #ffffff;
    --ekofin-text: #1d1d1f;
    --ekofin-text-secondary: #6e6e73;
    --ekofin-accent: #0071e3;
    --ekofin-accent-2: #5e5ce6;
    --ekofin-accent-soft: rgba(0, 113, 227, 0.10);
    --ekofin-border: rgba(0, 0, 0, 0.07);
    --ekofin-radius: 22px;
    --ekofin-ease: cubic-bezier(0.16, 1, 0.3, 1);

    --ekofin-glass-bg: rgba(255, 255, 255, 0.62);
    --ekofin-glass-border: rgba(255, 255, 255, 0.55);
    --ekofin-glass-blur: blur(26px) saturate(180%);

    --ekofin-shadow-sm: 0 1px 2px rgba(0,0,0,.05), 0 1px 1px rgba(0,0,0,.03);
    --ekofin-shadow-md: 0 14px 34px rgba(0,0,0,.09), 0 3px 10px rgba(0,0,0,.05);
    --ekofin-shadow-lg: 0 30px 70px rgba(0,0,0,.16), 0 10px 24px rgba(0,0,0,.08);
}

* { -webkit-font-smoothing: antialiased; }

html {
    color-scheme: light !important;
    min-height: 100%;
    background: #f5f5f7 !important;
}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    color: var(--ekofin-text);
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
    background: transparent !important;
}

/* ---- Canlı, akışkan "Liquid Glass" arka plan (Apple Design dilinden ilham) ---- */
body {
    min-height: 100vh;
    background-color: #f5f5f7 !important;
    background-image: linear-gradient(180deg, #fbfbfd 0%, #f5f5f7 55%, #eef0f3 100%) !important;
}
body::before {
    content: "";
    position: fixed;
    inset: -12%;
    z-index: 0;
    pointer-events: none;
    background:
        radial-gradient(640px circle at 12% 15%, rgba(10,132,255,0.30), transparent 62%),
        radial-gradient(620px circle at 88% 22%, rgba(255,55,95,0.20), transparent 60%),
        radial-gradient(700px circle at 45% 92%, rgba(94,92,230,0.22), transparent 62%),
        radial-gradient(520px circle at 92% 88%, rgba(48,213,200,0.20), transparent 60%),
        radial-gradient(480px circle at 5% 85%, rgba(255,159,10,0.14), transparent 58%);
    filter: blur(10px);
    animation: ekofin-aurora-drift 26s var(--ekofin-ease) infinite alternate;
}
@keyframes ekofin-aurora-drift {
    0%   { transform: translate3d(0,0,0) scale(1) rotate(0deg); }
    50%  { transform: translate3d(-2.5%, 2%, 0) scale(1.06) rotate(2deg); }
    100% { transform: translate3d(2.5%, -2%, 0) scale(1) rotate(-2deg); }
}

#MainMenu, footer, [data-testid="stStatusWidget"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

.block-container {
    max-width: 940px;
    padding-top: 1.6rem;
    padding-bottom: 6rem;
    margin: 0 auto;
    position: relative;
    z-index: 1;
    perspective: 1400px;
}

h1, h2, h3, h4 { color: var(--ekofin-text) !important; letter-spacing: -0.02em; font-weight: 700; }
p, li, label, .stMarkdown { color: var(--ekofin-text); }

/* ---- Hero başlık ---- */
.ekofin-hero { text-align: center; padding: 1.6rem 1rem 2rem; animation: ekofin-rise .7s var(--ekofin-ease) both; }
.ekofin-hero .eyebrow {
    display: inline-flex; align-items: center; gap: .45rem;
    font-size: .76rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase;
    color: var(--ekofin-accent); background: var(--ekofin-accent-soft);
    border: 1px solid rgba(0,113,227,0.18);
    padding: .4rem .95rem; border-radius: 999px; margin-bottom: 1.1rem;
    box-shadow: 0 4px 14px rgba(0,113,227,0.12);
}
.ekofin-hero .eyebrow .dot {
    width: 6px; height: 6px; border-radius: 50%; background: var(--ekofin-accent);
    box-shadow: 0 0 0 0 rgba(0,113,227,0.6);
    animation: ekofin-pulse 2s ease-in-out infinite;
}
@keyframes ekofin-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(0,113,227,0.55); }
    70%  { box-shadow: 0 0 0 8px rgba(0,113,227,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,113,227,0); }
}
.ekofin-hero h1 {
    font-size: clamp(2.3rem, 5vw, 3.6rem); font-weight: 800; margin: 0 0 .6rem; line-height: 1.06;
    letter-spacing: -0.035em;
    background: linear-gradient(100deg, #1d1d1f 10%, var(--ekofin-accent) 45%, var(--ekofin-accent-2) 60%, #1d1d1f 90%);
    background-size: 300% auto;
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    animation: ekofin-title-shine 9s linear infinite;
}
@keyframes ekofin-title-shine { to { background-position: -300% center; } }
.ekofin-hero p { font-size: 1.1rem; color: var(--ekofin-text-secondary); max-width: 600px; margin: 0 auto; }
@keyframes ekofin-rise {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---- Sidebar (koyu likit-cam panel) ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(200deg, rgba(30,30,32,0.94) 0%, rgba(0,0,0,0.96) 100%);
    backdrop-filter: blur(30px) saturate(180%);
    -webkit-backdrop-filter: blur(30px) saturate(180%);
    border-right: 1px solid rgba(255,255,255,0.08);
    box-shadow: 30px 0 60px rgba(0,0,0,0.25);
    perspective: 1200px;
}
[data-testid="stSidebar"] * { color: #f5f5f7 !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: rgba(245,245,247,0.55) !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12); margin: 1.1rem 0; }

.ekofin-brand {
    display: flex; align-items: center; gap: .65rem; padding: .2rem 0 1.3rem; font-size: 1.2rem; font-weight: 700;
}
.ekofin-brand .logo-badge {
    width: 38px; height: 38px; border-radius: 12px; flex-shrink: 0;
    background: linear-gradient(135deg, #0a84ff, #5e5ce6 55%, #34c9ff);
    background-size: 200% 200%;
    display: flex; align-items: center; justify-content: center; font-size: 1.2rem;
    box-shadow: 0 8px 20px rgba(10,132,255,0.4), inset 0 1px 0 rgba(255,255,255,0.35);
    animation: ekofin-gradient-shift 7s ease infinite;
}

/* Segmented control görünümlü persona seçici (Liquid Glass pill) */
[data-testid="stSidebar"] div[role="radiogroup"] {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(10px);
    border-radius: 16px; padding: 5px; gap: 2px;
    border: 1px solid rgba(255,255,255,0.09);
}
[data-testid="stSidebar"] div[role="radiogroup"] label {
    border-radius: 12px; padding: .55rem .7rem !important;
    transition: background .25s var(--ekofin-ease), transform .25s var(--ekofin-ease);
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: rgba(255,255,255,0.97);
    box-shadow: 0 8px 22px rgba(10,132,255,0.35), inset 0 1px 0 rgba(255,255,255,0.9);
    transform: scale(1.01);
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #000 !important; font-weight: 650;
}
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none;
}

/* ---- Butonlar (likit cam / dokunmatik + parıltı geçişi) ---- */
.stButton > button, .stDownloadButton > button {
    position: relative; overflow: hidden;
    border-radius: 980px !important;
    border: 1px solid rgba(255,255,255,0.5) !important;
    background: var(--ekofin-glass-bg) !important;
    backdrop-filter: var(--ekofin-glass-blur);
    -webkit-backdrop-filter: var(--ekofin-glass-blur);
    color: var(--ekofin-text) !important;
    font-weight: 600 !important;
    padding: .6rem 1.35rem !important;
    box-shadow: var(--ekofin-shadow-sm);
    transition: transform .4s var(--ekofin-ease), box-shadow .4s var(--ekofin-ease);
}
.stButton > button::after, .stDownloadButton > button::after {
    content: ""; position: absolute; top: 0; left: -60%; width: 35%; height: 100%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,.65), transparent);
    transform: skewX(-20deg); transition: left .7s ease;
}
.stButton > button:hover::after, .stDownloadButton > button:hover::after { left: 140%; }
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-3px) scale(1.015) rotateX(3deg);
    box-shadow: var(--ekofin-shadow-md);
}
.stButton > button:active, .stDownloadButton > button:active {
    transform: translateY(-1px) scale(0.985);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, #0a84ff, #5e5ce6 55%, #0071e3) !important;
    background-size: 220% 220% !important;
    color: #fff !important; border: none !important;
    box-shadow: 0 10px 26px rgba(10,132,255,.4), var(--ekofin-shadow-sm);
    animation: ekofin-gradient-shift 7s ease infinite;
}
@keyframes ekofin-gradient-shift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    color: #f5f5f7 !important;
    backdrop-filter: blur(14px);
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(120deg, #0a84ff, #5e5ce6 55%, #34c9ff) !important;
    background-size: 220% 220% !important;
    border: none !important; color: #fff !important;
}

/* ---- Sohbet balonları (cam kartlar + 3B hover eğimi) ---- */
[data-testid="stChatMessage"] {
    background: var(--ekofin-glass-bg);
    backdrop-filter: var(--ekofin-glass-blur);
    -webkit-backdrop-filter: var(--ekofin-glass-blur);
    border: 1px solid var(--ekofin-glass-border);
    border-radius: var(--ekofin-radius);
    padding: 1.05rem 1.3rem;
    margin-bottom: 1rem;
    box-shadow: var(--ekofin-shadow-md);
    transform-style: preserve-3d;
    transition: transform .5s var(--ekofin-ease), box-shadow .5s var(--ekofin-ease);
    animation: ekofin-msg-in .5s var(--ekofin-ease) both;
}
[data-testid="stChatMessage"]:hover {
    transform: translateY(-3px) rotateX(1.4deg) rotateY(-1deg);
    box-shadow: var(--ekofin-shadow-lg);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(165deg, rgba(224,240,255,0.75), rgba(240,248,255,0.55));
    border-color: rgba(0,113,227,0.22);
}
@keyframes ekofin-msg-in {
    from { opacity: 0; transform: translateY(14px) scale(.985); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

/* ---- Sohbet giriş kutusu ----
   Not: bu kapsayıcıya backdrop-filter/transform/filter EKLEME — Streamlit'in
   otomatik yükseklik hesaplayan gizli textarea "mirror"ı bu kapsayıcıyı yeni bir
   containing block sayıyor ve genişliği yanlış ölçüp kutuyu dev boyuta çıkarıyor. */
[data-testid="stChatInput"] {
    border-radius: 980px !important;
    box-shadow: var(--ekofin-shadow-md) !important;
    border: 1px solid rgba(255,255,255,.7) !important;
    background: rgba(255, 255, 255, 0.86) !important;
    transition: box-shadow .35s var(--ekofin-ease);
    overflow: hidden;
}
[data-testid="stChatInput"]:focus-within {
    box-shadow: 0 0 0 5px rgba(10,132,255,.16), var(--ekofin-shadow-lg) !important;
}
/* BaseWeb'in iç textarea sarmalayıcıları kendi köşeli, opak beyaz kutusunu ve
   (odakta) mavi kenarlığını çiziyor — bunlar dış pil tasarımıyla çakışıp
   "üst üste binen iki kutu" görüntüsüne yol açıyordu. Tamamen şeffaf yapıp
   sadece dış pilin görünmesini sağlıyoruz. */
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 980px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--ekofin-text) !important;
    caret-color: var(--ekofin-accent);
}

/* ---- Grafik kartı ---- */
[data-testid="stPlotlyChart"] {
    background: var(--ekofin-glass-bg);
    backdrop-filter: var(--ekofin-glass-blur);
    -webkit-backdrop-filter: var(--ekofin-glass-blur);
    border-radius: var(--ekofin-radius);
    border: 1px solid var(--ekofin-glass-border);
    padding: 1.1rem;
    box-shadow: var(--ekofin-shadow-md);
    transition: transform .5s var(--ekofin-ease), box-shadow .5s var(--ekofin-ease);
}
[data-testid="stPlotlyChart"]:hover {
    transform: translateY(-3px);
    box-shadow: var(--ekofin-shadow-lg);
}

/* ---- Uyarılar / expander / metric kartları ---- */
[data-testid="stAlert"] {
    border-radius: 16px; border: 1px solid var(--ekofin-glass-border);
    background: var(--ekofin-glass-bg); backdrop-filter: var(--ekofin-glass-blur);
}
[data-testid="stAlert"] p, [data-testid="stAlert"] span, [data-testid="stAlert"] div {
    color: var(--ekofin-text) !important;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
    color: var(--ekofin-text) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
    color: #f5f5f7 !important;
}
[data-testid="stExpander"] {
    border-radius: 18px; border: 1px solid var(--ekofin-glass-border);
    background: var(--ekofin-glass-bg); backdrop-filter: var(--ekofin-glass-blur);
    overflow: hidden; box-shadow: var(--ekofin-shadow-sm);
    transition: box-shadow .35s var(--ekofin-ease);
}
[data-testid="stExpander"]:hover { box-shadow: var(--ekofin-shadow-md); }
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.06); border-color: rgba(255,255,255,0.14);
}
[data-testid="stMetric"] {
    background: var(--ekofin-glass-bg); backdrop-filter: var(--ekofin-glass-blur);
    border-radius: 18px; padding: 1.1rem;
    border: 1px solid var(--ekofin-glass-border); box-shadow: var(--ekofin-shadow-md);
    transition: transform .4s var(--ekofin-ease), box-shadow .4s var(--ekofin-ease);
}
[data-testid="stMetric"]:hover { transform: translateY(-3px); box-shadow: var(--ekofin-shadow-lg); }
[data-testid="stMetricValue"] {
    color: var(--ekofin-text) !important;
    font-size: clamp(1.15rem, 2.1vw, 1.7rem) !important;
    white-space: normal !important;
    overflow-wrap: anywhere;
    line-height: 1.15 !important;
}
[data-testid="stMetricLabel"] { color: var(--ekofin-text-secondary) !important; }
[data-testid="stMetricLabel"] p { color: var(--ekofin-text-secondary) !important; }

/* ---- Yasal uyarı şeridi ---- */
.ekofin-disclaimer {
    font-size: .78rem; color: var(--ekofin-text-secondary); line-height: 1.55;
    background: rgba(255,255,255,0.5); backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,.5);
    border-radius: 16px;
    padding: .8rem 1.15rem; margin-top: .5rem;
}

/* ---- Sekmeler (Giriş/Kayıt formu için) ---- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: rgba(0,0,0,0.03); border-radius: 14px; padding: 4px; gap: 2px;
}
[data-testid="stTabs"] [data-baseweb="tab"] { border-radius: 10px; }
[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--ekofin-glass-bg); box-shadow: var(--ekofin-shadow-sm);
}

/* ---- Veri tabloları (st.dataframe) — cam temayla uyumlu, koyu tema sızıntısı yok ---- */
[data-testid="stDataFrame"] {
    border-radius: 16px; overflow: hidden;
    border: 1px solid var(--ekofin-glass-border);
    box-shadow: var(--ekofin-shadow-sm);
}
[data-testid="stDataFrame"] * { color: var(--ekofin-text) !important; }
[data-testid="stDataFrame"] [class*="glide"], [data-testid="stDataFrame"] canvas {
    background: var(--ekofin-surface) !important;
}
[data-testid="stDataFrame"] [role="columnheader"] {
    background: rgba(0,0,0,0.03) !important; font-weight: 600 !important;
}

/* ---- İnce, gradyanlı scrollbar ---- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, rgba(10,132,255,.45), rgba(94,92,230,.45));
    border-radius: 999px;
}
::-webkit-scrollbar-track { background: transparent; }
</style>
"""
