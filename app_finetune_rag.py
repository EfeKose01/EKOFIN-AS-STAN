# app_finetune_rag.py — EkoFin Asistan (Nihai Sürüm - Hallüsinasyon Azaltılmış)

import os
import json
import pickle
from typing import List, Dict, Any

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import time
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from datetime import datetime
import re
import yfinance as yf
import PyPDF2
import io

# Selenium kaldırıldı — Streamlit Cloud'da Chrome binary yok.
# get_current_loan_rates yerine kullanıcı web_search ile BDDK/hangikredi yönlendirilir.

# Teknik analiz, risk ve portföy motorları
from technical_engine import compute_technical_snapshot
from risk_engine import risk_snapshot
from portfolio_engine import parse_portfolio_text, portfolio_summary
from fundamental_engine import fundamental_snapshot

# Güncel mevzuat/haber toplayıcı
from news_scraper import run_scraper, SOURCE_MAP

# --- PERSONA'LAR ---

PERSONA_PROMPTS = {
    "Genel Asistan": """Sen EkoFin Asistan adında, Türkiye ekonomisi ve finans piyasaları konusunda geniş bilgiye sahip, yardımcı ve objektif bir yapay zeka asistanısın.
Görevin, araç sonuçlarını ve verileri kullanarak kullanıcının sorusuna net ve anlaşılır bir yanıt vermek. Asla yatırım tavsiyesi verme.

Cevap formatın:
1. **Ana Yanıt:** Kullanıcının sorusuna doğrudan, veriye dayalı ve öz bir yanıt ver.
2. **Şunları da merak edebilirsiniz:** başlığı altında EN AZ 3 devam sorusu öner (liste formatında, '- Soru' şeklinde).
""",

    "Analist Modu": """Sen EkoFin Asistan'ın Analist modusun. Hem teknik hem temel analizi bir arada kullanarak yorum yaparsın.

**Teknik boyut:** MA/EMA, ATR, Supertrend gibi göstergeleri yorumla; trend yönü, momentum ve volatilite hakkında net bir tablo çiz.
**Temel boyut:** Sektör dinamikleri, makro bağlam, bilanço kalitesi ve değerleme çerçevesini göz önünde bulundur.
**Kural:** Asla "al" veya "sat" deme. Senaryoları ve riskleri anlat.

Cevap formatın:
1. **Özet Görünüm:** 2–3 cümleyle genel tablo (trend + temel bağlam).
2. **Teknik Bulgular:** Gösterge bazlı maddeler (MA/EMA pozisyonu, Supertrend yönü, ATR seviyesi, volatilite).
3. **Temel Bağlam:** Sektör, makro ortam veya şirkete dair öne çıkan nokta.
4. **Riskler:** Öne çıkan teknik ve temel riskler.
5. **İlgili diğer analizler:** başlığı altında EN AZ 3 devam sorusu öner.
""",

    "Mevzuat Asistanı": """Sen EkoFin Asistan'ın Mevzuat modusun. BDDK, SPK, TCMB ve ilgili Türk finans mevzuatı konularında uzmanlaşmış bir bilgi asistanısın.

Öncelik sıran:
1. Dahili RAG belgeleri (tebliğler, yönetmelikler, genelgeler) — bunlar birincil kaynağın.
2. Web araması — güncel değişiklik veya son kararlar için.
3. Eğitim bilgin — yalnızca genel çerçeve ve kavram açıklamaları için; tarih veya karar uydurma.

**Yanıt kuralları:**
- Her sayısal sınır, oran veya kural için kaynağını belirt (tebliğ adı, madde numarası varsa).
- Bilginin RAG belgelerinden mi, web aramasından mı, yoksa genel eğitimden mi geldiğini parantez içinde kısaca işaretle.
- Güncel olmayabilecek bilgilerde açıkça uyar: "Bu bilgi güncel olmayabilir, ilgili kurumun resmi sitesini kontrol edin."
- Yoruma değil kurala odaklan; hukuki tavsiye verme.

Cevap formatın:
1. **Kural / Hüküm:** İlgili mevzuat maddesini veya gereksinimi özetle.
2. **Kaynak:** Hangi tebliğ / yönetmelik / karar (tarih varsa belirt).
3. **Pratik Uygulama:** Kuralın pratikte ne anlama geldiğini bir paragrafla açıkla.
4. **İlgili diğer mevzuat konuları:** başlığı altında EN AZ 3 devam sorusu öner.
""",
}

# --- TÜM PERSONA'LAR İÇİN GENEL FİNANS GÜVENLİK KURALLARI EKİ ---

FINANCE_SAFETY_SUFFIX = """
GENEL FİNANS KURALLARI:
- Güncel fiyat, seviye, oran gibi sayısal veriler konusunda asla kendi tahminini kullanma.
- Hisse, döviz, kripto ve faiz oranları için verdiğin her sayısal bilgi mutlaka sistemdeki araçlardan
  (get_market_data, analyze_technical, analyze_risk, analyze_portfolio, calculate_loan_payment vb.)
  veya kullanıcının verdiği tablodan/veri setinden gelmelidir.
- Eğer bu kaynaklarda ilgili veri yoksa, 'Bu konuda elimde sayısal veri yok, fiyat söyleyemem' de. Tahmin yapma.
- 2023 sonrası için eğitim verin güncel olmayabilir. 2023 sonrasına dair tarih, düzenleme veya kurumsal karar bilgisi
  verirken emin değilsen, 'Bu bilgi güncel olmayabilir' diye özellikle uyar ve asla tarih/karar uydurma.
- Matematiksel hesaplamalarda (faiz, taksit, volatilite, Sharpe benzeri oranlar vb.) kendi kafandan hesaplamaya çalışmak yerine
  mümkün olduğunda ilgili araçların çıkardığı sonuçları yorumla. Araç çıktısı yoksa, 'Bu hesabı doğrudan yapamıyorum' demeyi tercih et.
"""

for _k in list(PERSONA_PROMPTS.keys()):
    PERSONA_PROMPTS[_k] = PERSONA_PROMPTS[_k] + "\n" + FINANCE_SAFETY_SUFFIX

APP_NAME = "EkoFin Asistan"
st.set_page_config(page_title=APP_NAME, page_icon="🤖", layout="wide")

# --- GÖRSEL TASARIM SİSTEMİ (Apple tarzı arayüz katmanı) ---

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
}
[data-testid="stChatInput"]:focus-within {
    box-shadow: 0 0 0 5px rgba(10,132,255,.16), var(--ekofin-shadow-lg) !important;
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
[data-testid="stMetricValue"] { color: var(--ekofin-text) !important; }
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

/* ---- İnce, gradyanlı scrollbar ---- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, rgba(10,132,255,.45), rgba(94,92,230,.45));
    border-radius: 999px;
}
::-webkit-scrollbar-track { background: transparent; }
</style>
"""


@st.cache_resource
def load_bist_tickers() -> set:
    """bist_tickers.txt'den geçerli BIST sembollerini yükler (örn: GARAN.IS → GARAN)."""
    path = "bist_tickers.txt"
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        tickers = set()
        for line in f:
            t = line.strip().upper().replace(".IS", "")
            if t:
                tickers.add(t)
    return tickers


BIST_TICKERS: set = set()  # run_streamlit_app içinde doldurulur


def load_dotenv(path: str = ".env"):
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    except Exception:
        pass


load_dotenv(".env")

# Streamlit Secrets'tan anahtarları ortam değişkenlerine aktar (Streamlit Cloud uyumluluğu)
def _load_streamlit_secrets():
    try:
        for key in ["ANTHROPIC_API_KEY", "SERPER_API_KEY"]:
            val = st.secrets.get(key)
            if val and not os.environ.get(key):
                os.environ[key] = val
    except Exception:
        pass  # st.secrets yoksa (lokal geliştirme) sessizce devam et

_load_streamlit_secrets()

# --- RAG İndeksi ---

FAISS_INDEX_PATH = "rag_index.faiss"
CONTENT_MAP_PATH = "rag_content.pkl"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"


@st.cache_resource
def load_semantic_search_engine():
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CONTENT_MAP_PATH, "rb") as f:
            content_map = pickle.load(f)
        return model, index, content_map
    except Exception as e:
        print(f"RAG yüklenemedi (devre dışı): {e}")
        return None, None, None


RAG_ENABLED = False
SEARCH_MODEL, RAG_INDEX, RAG_CONTENT_MAP = None, None, None

if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CONTENT_MAP_PATH):
    try:
        SEARCH_MODEL, RAG_INDEX, RAG_CONTENT_MAP = load_semantic_search_engine()
        if SEARCH_MODEL is not None:
            RAG_ENABLED = True
    except Exception as e:
        print(f"RAG başlatılamadı: {e}")


# --- Yardımcı Fonksiyonlar ---

def _update_stock_session_state(symbols_str: str, result: Dict[str, Any]) -> None:
    """get_market_data sonucunu st.session_state'e yazar (cache ile uyumsuz olduğu için ayrı)."""
    gecerli = result.get("gecerli_semboller") or (
        [result["sembol"]] if "sembol" in result else []
    )
    if not gecerli:
        return

    chart_records = result.get("_chart_records")
    if chart_records:
        history_df = pd.DataFrame.from_dict(chart_records, orient="index")
        history_df.index = pd.to_datetime(history_df.index)
        st.session_state.stock_history = history_df

    st.session_state.stock_company_name = ", ".join(gecerli)
    st.session_state.last_symbols = gecerli


# --- Araç Fonksiyonları ---

def loan_payment(principal: float, annual_rate: float, years: float, payments_per_year: int = 12) -> float:
    r = annual_rate / payments_per_year
    n = int(round(years * payments_per_year))
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def search_documents(query: str, k: int = 4) -> List[Dict[str, Any]]:
    if not RAG_ENABLED or SEARCH_MODEL is None or RAG_INDEX is None:
        return []
    try:
        query_vector = SEARCH_MODEL.encode([query])
        distances, indices = RAG_INDEX.search(np.array(query_vector).astype("float32"), k)
        return [RAG_CONTENT_MAP[i] for i in indices[0]]
    except Exception:
        return []


def _fetch_single_symbol_close_series(yf_symbol: str):
    """download boşsa, tek tek history() fallback."""
    try:
        ticker = yf.Ticker(yf_symbol)
        for period in ["1y", "2y", "max"]:
            hist = ticker.history(period=period, interval="1d")
            if not hist.empty and "Close" in hist.columns:
                try:
                    hist.index = hist.index.tz_localize(None)
                except Exception:
                    pass
                return hist["Close"]
        return None
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def get_intraday_stats(yf_symbol: str) -> Dict[str, float]:
    """
    Tek bir sembol için son 1 günlük (intraday) temel istatistikleri döner.
    Destek/direnç ve gün içi volatiliteyi GERÇEK grafiğe dayandırmak için kullanıyoruz.
    """
    try:
        intraday = yf.download(
            tickers=yf_symbol,
            period="1d",
            interval="5m",
            auto_adjust=False,
            progress=False,
        )
        if intraday.empty:
            return {}

        try:
            intraday.index = intraday.index.tz_localize(None)
        except Exception:
            pass

        # MultiIndex sütunları düzleştir (yfinance tek sembol için de MultiIndex gönderebilir)
        if isinstance(intraday.columns, pd.MultiIndex):
            intraday.columns = intraday.columns.get_level_values(0)

        day_low   = float(intraday["Low"].min().iloc[0]   if hasattr(intraday["Low"].min(),   "iloc") else intraday["Low"].min())
        day_high  = float(intraday["High"].max().iloc[0]  if hasattr(intraday["High"].max(),  "iloc") else intraday["High"].max())
        day_close = float(intraday["Close"].iloc[-1].iloc[0] if hasattr(intraday["Close"].iloc[-1], "iloc") else intraday["Close"].iloc[-1])
        day_open  = float(intraday["Open"].iloc[0].iloc[0]  if hasattr(intraday["Open"].iloc[0],  "iloc") else intraday["Open"].iloc[0])

        intraday_range = day_high - day_low
        vol_pct = (intraday_range / day_open * 100) if day_open != 0 else 0.0

        return {
            "gunluk_en_dusuk": round(day_low, 2),
            "gunluk_en_yuksek": round(day_high, 2),
            "gunluk_acilis": round(day_open, 2),
            "gunluk_kapanis": round(day_close, 2),
            "gunluk_volatilite_yuzde": round(vol_pct, 2),
        }
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def get_market_data(symbols: str) -> Dict[str, Any]:
    """
    Bir veya daha fazla sembol için fiyat geçmişini çeker.
    - BIST sembolleri için .IS ekler.
    - download + history fallback
    - çoklu sembolde tek grafikte normalize edilmiş çizgi
    - istatistikler: ilk/son fiyat + yüzde değişim
    - TEK SEMBOLDE ekstradan 1 günlük intraday istatistikleri de döner
    """
    raw_symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not raw_symbols:
        return {"hata": "En az bir geçerli sembol girilmelidir."}

    yf_symbols = []
    symbol_map = {}
    for rs in raw_symbols:
        if len(rs) in [4, 5] and rs.isalpha() and "." not in rs:
            yf_symbol = rs + ".IS"
        else:
            yf_symbol = rs
        symbol_map[rs] = yf_symbol
        yf_symbols.append(yf_symbol)

    print(f"--- yfinance.download çağrısı: {yf_symbols} ---")

    close_df = None
    try:
        data = yf.download(
            tickers=yf_symbols,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )

        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                if "Close" in data.columns.get_level_values(0):
                    close_df = data["Close"].copy()
                else:
                    st.warning("Yfinance verisinde 'Close' sütunu bulunamadı (MultiIndex).")
            else:
                if "Close" in data.columns:
                    close_df = data[["Close"]].copy()
                    close_df.columns = [yf_symbols[0]]
        else:
            st.warning("yfinance.download boş veri döndürdü, fallback ile tek tek sembolleri deneyeceğim.")
    except Exception as e:
        st.warning(f"yfinance.download hatası: {e}. Tek tek sembol fallback'e geçiliyor.")

    if close_df is not None:
        try:
            close_df.index = close_df.index.tz_localize(None)
        except Exception:
            pass

    valid_cols: Dict[str, pd.Series] = {}
    empty_symbols: List[str] = []

    for rs in raw_symbols:
        ys = symbol_map[rs]
        series = None

        if close_df is not None and ys in close_df.columns:
            candidate = close_df[ys]
            if not candidate.dropna().empty:
                series = candidate

        if series is None or series.dropna().empty:
            series = _fetch_single_symbol_close_series(ys)

        if series is None or series.dropna().empty:
            empty_symbols.append(rs)
            continue

        valid_cols[rs] = series

    if not valid_cols:
        return {
            "hata": "Girilen semboller için anlamlı fiyat verisi bulunamadı.",
            "detay": f"Boş semboller: {', '.join(empty_symbols)}" if empty_symbols else "",
        }

    comparison_df = pd.DataFrame(valid_cols)
    comparison_df = comparison_df.ffill()

    # Not: st.session_state yazımları cache ile uyumsuz; çağıran taraf günceller.

    stats: Dict[str, Dict[str, str]] = {}
    for col in comparison_df.columns:
        first = float(comparison_df[col].iloc[0])
        last = float(comparison_df[col].iloc[-1])
        change = last - first
        pct = (last / first - 1) * 100 if first != 0 else 0.0
        stats[col] = {
            "ilk_fiyat": f"{first:.2f}",
            "son_fiyat": f"{last:.2f}",
            "mutlak_degisim": f"{change:.2f}",
            "yuzde_degisim": f"{pct:.2f}",
        }

    last_date = comparison_df.index[-1].date()
    today = datetime.now().date()
    uyari_parts = []
    if (today - last_date).days > 3:
        uyari_parts.append(f"Son fiyat verisi {last_date} tarihli; çok güncel olmayabilir.")
    if empty_symbols:
        uyari_parts.append("Veri alınamayan semboller: " + ", ".join(empty_symbols))

    # --- TEK SEMBOL DURUMU: detaylı JSON + intraday ---
    if len(comparison_df.columns) == 1:
        col = comparison_df.columns[0]
        last_close = float(comparison_df[col].iloc[-1])
        previous_close = float(comparison_df[col].iloc[-2]) if len(comparison_df) > 1 else last_close
        change = last_close - previous_close
        percent_change = (change / previous_close * 100) if previous_close != 0 else 0.0

        yf_symbol = symbol_map[col]
        intraday_stats = get_intraday_stats(yf_symbol)

        result: Dict[str, Any] = {
            "sembol": col,
            "guncel_fiyat": f"{last_close:.2f}",
            "veri_tarihi": comparison_df.index[-1].strftime("%Y-%m-%d"),
            "gunluk_degisim": f"{change:.2f}",
            "gunluk_degisim_yuzde": f"{percent_change:.2f}%",
            "yillik_istatistik": stats[col],
        }

        # Gün içi istatistikleri de ekle (varsa)
        result.update(intraday_stats)

        if uyari_parts:
            result["uyari"] = " ".join(uyari_parts)
        # Grafik için seri verisi (ISO tarih → fiyat dict)
        result["_chart_records"] = {
            str(k): v for k, v in comparison_df.to_dict(orient="index").items()
        }
        return result

    # --- ÇOKLU SEMBOL DURUMU: karşılaştırma özeti ---
    summary_text = (
        f"{len(comparison_df.columns)} adet hisse için karşılaştırmalı veriler çekilmiştir. "
        f"Hisseler: {', '.join(comparison_df.columns)}. Normalleştirilmiş grafikte birlikte göster."
    )

    response: Dict[str, Any] = {
        "ozet": summary_text,
        "veri_var": True,
        "gecerli_semboller": list(comparison_df.columns),
        "gecersiz_semboller": empty_symbols,
        "istatistikler": stats,
        # Grafik için seri verisi
        "_chart_records": {
            str(k): v for k, v in comparison_df.to_dict(orient="index").items()
        },
    }
    if uyari_parts:
        response["uyari"] = " ".join(uyari_parts)
    return response


def web_search(query: str) -> Dict[str, Any]:
    """
    SERPER ile web araması yapar.
    Çıktı: { "kaynak": "serper", "query": "...", "results": [ {title,snippet,link}, ... ] }
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return {"hata": "Serper API anahtarı .env dosyasında bulunamadı."}

    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "gl": "tr", "hl": "tr"})
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        results = response.json()
        if "organic" in results:
            items = []
            for result in results["organic"][:4]:
                items.append(
                    {
                        "title": result.get("title", "N/A"),
                        "snippet": result.get("snippet", "N/A"),
                        "link": result.get("link", "N/A"),
                    }
                )
            return {"kaynak": "serper", "query": query, "results": items}
        else:
            return {"hata": "Arama sonucu bulunamadı."}
    except Exception as e:
        return {"hata": f"Web araması sırasında genel bir hata oluştu: {e}"}


def get_current_loan_rates(amount: int, term: int) -> Dict[str, Any]:
    """
    Gerçek zamanlı kredi oranı karşılaştırması şu an kullanılamıyor.
    Kullanıcıyı web_search ile BDDK/hangikredi.com'a yönlendir.
    """
    return {
        "hata": "Güncel kredi oranı karşılaştırması şu an bu platformda desteklenmiyor.",
        "oneri": (
            f"Güncel ihtiyaç kredisi oranlarını karşılaştırmak için "
            f"hangikredi.com veya BDDK'nın resmi faiz tablosunu ziyaret edebilirsiniz. "
            f"İsterseniz web araması yapabilirim: "
            f"'{amount} TL {term} ay ihtiyaç kredisi faiz oranları 2025' gibi."
        ),
    }


# --- LLM Katmanı ---

def _http_post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: int = 120) -> Dict[str, Any]:
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        error_body = e.response.text
        raise ConnectionError(f"API sunucusuna bağlanırken hata (HTTP {e.response.status_code}): {error_body}")
    except Exception as e:
        raise RuntimeError(f"HTTP isteği sırasında bilinmeyen hata: {e}")


def call_claude(messages: List[Dict[str, str]]) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY ortam değişkeni bulunamadı.")
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    system_prompt = ""
    user_assistant_messages = []
    for msg in messages:
        clean_msg = {"role": msg["role"], "content": msg["content"]}
        if msg["role"] == "system":
            system_prompt = msg["content"]
        else:
            user_assistant_messages.append(clean_msg)
    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 2048,
        "system": system_prompt,
        "messages": user_assistant_messages,
        "temperature": 0.2,
    }
    try:
        data = _http_post_json("https://api.anthropic.com/v1/messages", payload, headers)
        return data["content"][0]["text"].strip()
    except Exception as e:
        st.error(f"Claude API çağrısı sırasında bir hata oluştu: {e}")
        return "Üzgünüm, şu anda cevap veremiyorum. Lütfen daha sonra tekrar deneyin."


# --- PORTFÖY WRAPPER ---

def analyze_portfolio_wrapper(weights_text: str) -> Dict[str, Any]:
    import re as _re
    # Girilen ham toplamı hesapla (normalleştirmeden önce)
    raw_values = [float(v) for v in _re.findall(r"[:\=]\s*([\d\.]+)", weights_text) if v]
    raw_total = sum(raw_values) if raw_values else 0.0

    weights = parse_portfolio_text(weights_text)
    if not weights:
        return {
            "hata": "Portföy tanımı çözümlenemedi. Örnek format:\n"
                    "GARAN:0.3\nAKBNK:0.4\nTHYAO:0.3"
        }
    result = portfolio_summary(weights)

    # Normalleştirme uyarısı: giriş toplamı 1.0'dan farklıysa bildir
    if raw_total > 0 and abs(raw_total - 1.0) > 0.01:
        result["normalizasyon_uyarisi"] = (
            f"Girilen ağırlıklar toplamı {raw_total:.2f} idi; "
            f"analiz için otomatik olarak 1.00'a normalize edildi."
        )
    return result


# --- TOOL ROUTER ---

TOOLS = {
    "calculate_loan_payment":    {"function": loan_payment,              "required_params": ["principal", "annual_rate", "years"]},
    "search_financial_documents":{"function": search_documents,          "required_params": ["query"]},
    "get_market_data":           {"function": get_market_data,           "required_params": ["symbols"]},
    "web_search":                {"function": web_search,                "required_params": ["query"]},
    "get_current_loan_rates":    {"function": get_current_loan_rates,    "required_params": ["amount", "term"]},
    "analyze_technical":         {"function": compute_technical_snapshot,"required_params": ["symbol"]},
    "analyze_risk":              {"function": risk_snapshot,             "required_params": ["symbol"]},
    "analyze_fundamental":       {"function": fundamental_snapshot,      "required_params": ["symbol"]},
    "analyze_portfolio":         {"function": analyze_portfolio_wrapper, "required_params": ["weights_text"]},
}

TOOL_SYSTEM_PROMPT = """Sen bir araç yönlendiricisin. Kullanıcının mesajını analiz et ve en uygun aracı `TOOL_CALL` formatında çağır. Başka hiçbir metin yazma.

# Araçlar
- `get_market_data(symbols)`: Bir veya daha fazla hisse/forex/kripto için fiyat geçmişi ve normalize grafik. Semboller virgülle ayrılır.
- `analyze_technical(symbol)`: Tek sembol için kapsamlı teknik analiz — RSI, MACD, Bollinger, MA/EMA, ATR, Supertrend, OBV, destek/direnç.
- `analyze_risk(symbol)`: Tek sembol için risk metrikleri — yıllık volatilite, max drawdown, Sharpe benzeri oran.
- `analyze_fundamental(symbol)`: Tek sembol için temel analiz — F/K, P/DD, ROE, kâr marjı, temettü verimi, borç/özsermaye, piyasa değeri.
- `analyze_portfolio(weights_text)`: Çok hisseli portföy analizi. `weights_text` serbest metin (GARAN:0.3, AKBNK:0.4).
- `web_search(query)`: "güncel", "en son", "son haber", "bugün", SPK/BDDK/TCMB/FED/TÜİK haberleri için. `query` = kullanıcının son mesajı.
- `search_financial_documents(query)`: Dahili mevzuat/kavram sorguları için.
- `calculate_loan_payment(principal, annual_rate, years)`: Kredi taksit hesabı.
- `get_current_loan_rates(amount, term)`: Güncel ihtiyaç kredisi oranları.

**KURALLAR:**
- Fiyat sorusu → `get_market_data`. Model kendi bilgisinden fiyat verme.
- Teknik göstergeler (RSI, MA, MACD, Bollinger, Supertrend, destek/direnç) → `analyze_technical`.
- Risk metrikleri (volatilite, drawdown, Sharpe) → `analyze_risk`.
- Temel analiz (F/K, ROE, temettü, bilanço) → `analyze_fundamental`.
- Güncel haber/karar/düzenleme → `web_search`.
- Portföy sorusu → `analyze_portfolio`.
- Eksik zorunlu parametre varsa TOOL_CALL üretme, parametreyi kullanıcıdan iste.
- `web_search` için query eksikse kullanıcının son mesajını kullan.
"""


def run_tool_calling_logic(chat_history: List[Dict[str, Any]], persona: str) -> str:
    system_prompt = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["Genel Asistan"])

    if "stock_history" in st.session_state:
        del st.session_state.stock_history

    last_user_msg = chat_history[-1]["content"]
    lower = last_user_msg.lower()

    analysis_triggers = [
        # Daha agresif tetikleyici liste: fiyat/hisse sorularını mutlaka yakala
        "grafik",
        "karşılaştır",
        "karşılaştırma",
        "performans",
        "yıllık",
        "yıllık getiri",
        "fiyat",
        "fiyatı",
        "güncel fiyat",
        "şu anki fiyat",
        "şu an ne kadar",
        "şu anda ne kadar",
        "kaç tl",
        "kaç para",
        "değeri ne",
        "hisse",
        "lot",
        "endeks",
        "borsa",
        "trend",
        "teknik",
        "analiz",
        "usd",
        "eur",
        "dolar",
        " tl",
        "₺",
    ]

    news_triggers = [
        "güncel",
        "en son",
        "son haber",
        "son gelişme",
        "bugün",
        "bu sene",
        "bu yıl",
        "hangi yıl",
        "hangi tarihte",
        "duyuru",
        "açıklama",
        "karar",
        "yeni düzenleme",
        "yeni tebliğ",
        "yeni yönetmelik",
        "şu anda",
        "şu an",
        "yakın zamanda",
        "spk",
        "sermaye piyasası kurulu",
        "bddk",
        "bankacılık düzenleme",
        "tcmb",
        "merkez bankası",
        "tüik",
        "fed",
        "ecb",
    ]

    should_force_market = any(t in lower for t in analysis_triggers)
    should_force_news = any(t in lower for t in news_triggers)

    # Ek güvenlik: Sembol + fiyat sorusu paterni yakala
    symbol_like_tokens = re.findall(r"\b[A-ZÇĞİÖŞÜ]{3,5}\b", last_user_msg.upper())
    price_keywords = ["kaç tl", "kaç para", "ne kadar", "güncel fiyat", "şu an", "şu anda"]
    if symbol_like_tokens and any(pk in lower for pk in price_keywords):
        should_force_market = True

    # Mevzuat Asistanı'nda piyasa/fiyat araması devre dışı — kapsam dışı soruları reddet
    if persona == "Mevzuat Asistanı":
        should_force_market = False

    messages_for_tool_choice = [{"role": "system", "content": TOOL_SYSTEM_PROMPT}] + chat_history
    tool_call_str = call_claude(messages_for_tool_choice)

    # ---- TOOL_CALL YOKSA: fallback mantıkları ----
    if not tool_call_str.strip().startswith("TOOL_CALL:"):
        # Haber fallback
        if should_force_news:
            result = web_search(last_user_msg)

            if isinstance(result, dict) and "hata" in result:
                tool_output = json.dumps(result, indent=2, ensure_ascii=False)
                final_prompt_text = f"""Bir web araması yapmak istedik ancak araç hata döndürdü:
--- ARAÇ SONUCU (web_search HATASI) ---
{tool_output}
---
SENİN GÖREVİN:
1. Bu hatayı kullanıcıya dürüstçe açıkla (örneğin API anahtarı yok, sonuç bulunamadı vb.).
2. Güncel veri uydurma, tarih veya karar ismi uydurma.
3. Mümkünse kullanıcının resmi kaynakları (SPK, BDDK, TCMB, FED vb. siteleri) kendisinin kontrol etmesi gerektiğini söyle.
4. Yine de, genel çerçeveyi anlatmak için eğitimdeki eski bilgilere dayanabilirsin; ama bunların güncel olmadığını özellikle belirt.
5. Cevabının sonunda 'Şunları da merak edebilirsiniz:' başlığıyla EN AZ 3 devam sorusu öner.
"""
            else:
                tool_output = json.dumps(result, indent=2, ensure_ascii=False)
                final_prompt_text = f"""Kullanıcının '{last_user_msg}' sorusuna cevap vermek için doğrudan web_search aracı çağrıldı ve aşağıdaki JSON sonuç alındı:
--- ARAÇ SONUCU (web_search JSON, PYTHON FALLBACK) ---
{tool_output}
---
JSON yapısı:
- "query": Kullanıcının arama ifadesi
- "results": Her elemanda "title", "snippet" ve "link" alanları olan bir liste

SENİN GÖREVİN:
1. "results" içindeki "title" ve "snippet" alanlarını kullanarak, kullanıcının sorusuna net, özet ve güncel bir cevap yaz.
2. Cevabının EN ALTINDA mutlaka ayrı bir blok olarak şu formatta kaynakları listele:
   Kaynaklar:
   - <link1>
   - <link2>
   - ...
3. Linkleri sadece "results" içindeki "link" alanlarından al. Yeni kaynak uydurma.
4. Cevabının sonunda, her zamanki gibi, kullanıcının merak edebileceği EN AZ 3 devam sorusunu "Şunları da merak edebilirsiniz:" başlığıyla madde madde yaz.
"""

            history_without_last_prompt = chat_history[:-1]
            messages_for_final_answer = [{"role": "system", "content": system_prompt}] + history_without_last_prompt
            messages_for_final_answer.append(
                {"role": "assistant", "content": "TOOL_CALL: web_search(query=...) [python_fallback]"}
            )
            messages_for_final_answer.append({"role": "user", "content": final_prompt_text})
            return call_claude(messages_for_final_answer)

        # Hisse grafiği / fiyat analizi fallback
        if should_force_market:
            # Mesajdan BIST listesine göre filtreli semboller ayıkla
            raw_tokens = re.findall(r"\b[A-Z0-9\.]{3,8}\b", last_user_msg.upper())

            if BIST_TICKERS:
                # BIST listesi yüklüyse, yalnızca listede olan tokenları al
                symbols_found = [t for t in raw_tokens if t in BIST_TICKERS]
            else:
                # Fallback: kara liste filtresi
                BLACKLIST_TOKENS = {
                    "TRADE", "YAP", "AL", "SAT", "GRAFIK", "GRAFİK",
                    "GUNLUK", "GÜNLÜK", "BACKTEST", "TEST", "ICIN", "İÇİN",
                    "STRATEJI", "STRATEJİ", "SON", "AY", "YIL", "BIST",
                }
                symbols_found = [t for t in raw_tokens if t not in BLACKLIST_TOKENS]

            symbols_unique = list(dict.fromkeys(symbols_found))

            # Bu mesajda sembol yoksa, önceki sembolleri kullan
            if not symbols_unique and "last_symbols" in st.session_state:
                symbols_unique = st.session_state.last_symbols

            # Hâlâ sembol yoksa, kullanıcıdan hisse adı iste
            if not symbols_unique:
                history_without_last_prompt = chat_history[:-1]
                messages_for_final_answer = [{"role": "system", "content": system_prompt}] + history_without_last_prompt
                messages_for_final_answer.append(
                    {
                        "role": "assistant",
                        "content": (
                            "Kısa vadeli fiyat / grafik analizi yapabilmem için hangi hisse senedi veya endeks "
                            "için bilgi istediğinizi belirtir misiniz? Örn: GARAN, THYAO, BIST100. "
                            "Fiyat, seviye veya destek/direnç söyleyebilmem için mutlaka sembol bilmem gerekiyor."
                        ),
                    }
                )
                return call_claude(messages_for_final_answer)

            # Buraya geldiysek elimizde en az bir sembol var
            symbols_str = ",".join(symbols_unique)
            with st.spinner(f"Piyasa verisi çekiliyor: {symbols_str}..."):
                result = get_market_data(symbols_str)
            if isinstance(result, dict) and "hata" not in result:
                # cache'li fonksiyon session_state yazamaz; burada güncelliyoruz
                _update_stock_session_state(symbols_str, result)

            if isinstance(result, dict):
                if "hata" in result:
                    detay = result.get("detay", "")
                    if detay:
                        tool_output = f"Hata: {result['hata']}\nDetay: {detay}"
                    else:
                        tool_output = f"Hata: {result['hata']}"
                else:
                    tool_output = json.dumps(result, indent=2, ensure_ascii=False)
            elif isinstance(result, list):
                tool_output = json.dumps(result, indent=2, ensure_ascii=False)
            elif isinstance(result, str):
                tool_output = result
            else:
                tool_output = str(result)

            final_prompt_text = f"""Kullanıcının '{last_user_msg}' sorusuna cevap vermek için doğrudan hisse verileri getirildi:
--- ARAÇ SONUCU (get_market_data, zorunlu fallback) ---
{tool_output}
---
SENİN GÖREVİN:
1. Bu sonucu analiz et ve kullanıcıya net, tutarlı bir cevap oluştur.
2. Eğer JSON içinde 'guncel_fiyat', 'gunluk_en_dusuk' ve 'gunluk_en_yuksek' alanları varsa, kısa vadeli destek ve direnç seviyelerini bu sayılara DAYANDIR. Bu aralığın biraz içini veya çevresini kullanabilirsin ama kesinlikle bambaşka fiyat seviyeleri uydurma.
3. Özellikle '1 günlük grafikte', 'gün içi' veya 'şu an kaç TL' gibi ifadeler geçiyorsa, yorumlarında önceliği bu günlük aralık ve volatilite bilgisine ver.
4. Cevabını, sana atanan kimliğin (persona) gerektirdiği formata uygun şekilde, sonunda EN AZ 3 adet devam sorusu önererek tamamla.
5. Eğer bazı semboller için veri yoksa, bunu belirt ama veri olan semboller üzerinden mutlaka analiz yap.
6. Bu uygulamada sadece gerçek fiyat verisi ve bunlardan türetilen yüzdesel değişimler ve basit karşılaştırmalar kullanılabilir.
7. RSI, hacim, 50/200 günlük ortalama vb. teknik göstergeler için SAYISAL değeri veya yüzdesel değişimi UYDURMA; bu göstergeler için sadece fiyat ve yüzdesel değişim üzerinden yorum yapabileceğini açıkla.
8. Kendi eğitimindeki (örn. 2023 tarihli) eski fiyatları KULLANMA; sadece bu araçtan gelen güncel veriyi esas al.
"""

            history_without_last_prompt = chat_history[:-1]
            messages_for_final_answer = [{"role": "system", "content": system_prompt}] + history_without_last_prompt
            messages_for_final_answer.append(
                {"role": "assistant", "content": "TOOL_CALL: get_market_data(...) [python_fallback]"}
            )
            messages_for_final_answer.append({"role": "user", "content": final_prompt_text})
            return call_claude(messages_for_final_answer)

        # Ne haber, ne hisse → normal direkt cevap
        messages_for_direct_answer = [{"role": "system", "content": system_prompt}] + chat_history
        return call_claude(messages_for_direct_answer)

    # ---- TOOL_CALL VARSA buraya gelir ----
    tool_command = tool_call_str.replace("TOOL_CALL:", "").strip()
    tool_name = tool_command.split("(", 1)[0]
    tool_output = f"Bilinmeyen araç: {tool_name}"

    # Mevzuat Asistanı: piyasa/teknik/risk araçları bu modda çalışmaz
    MARKET_TOOLS = {"get_market_data", "analyze_technical", "analyze_risk", "analyze_portfolio"}
    if persona == "Mevzuat Asistanı" and tool_name in MARKET_TOOLS:
        messages_for_redirect = [{"role": "system", "content": system_prompt}] + chat_history[:-1]
        messages_for_redirect.append({
            "role": "user",
            "content": (
                f"Kullanıcı '{last_user_msg}' diye sordu. "
                "Bu soru piyasa fiyatı veya teknik analiz içeriyor. "
                "Mevzuat Asistanı olarak bu konuda analiz yapamayacağını, "
                "sol panelden Analist Modu'na geçmelerini öneren kısa ve nazik bir yanıt ver. "
                "Hisse fiyatı, destek/direnç veya teknik gösterge üretme."
            ),
        })
        return call_claude(messages_for_redirect)

    if tool_name in TOOLS:
        try:
            params_str = tool_command[len(tool_name) + 1: -1]
            params = {k: v.strip().strip("'\"") for k, v in re.findall(r"(\w+)=([^,)]+)", params_str)}

            # web_search için parametre otomatik dolsun
            if tool_name == "web_search":
                if "query" not in params or params["query"].lower() in ["none", "null", ""]:
                    params["query"] = last_user_msg

            required_params = TOOLS[tool_name].get("required_params", [])
            missing_params = [
                p for p in required_params if p not in params or params[p].lower() in ["none", "null", ""]
            ]

            if missing_params:
                return (
                    f"İsteğinizi yerine getirebilmek için şu ek bilgilere ihtiyacım var: "
                    f"**{', '.join(missing_params)}**."
                )

            typed_params: Dict[str, Any] = {}
            for k, v in params.items():
                if k in ["amount", "term"]:
                    typed_params[k] = int(v)
                elif k in ["principal", "annual_rate", "years"]:
                    typed_params[k] = float(v)
                else:
                    typed_params[k] = v

            SPINNER_MESSAGES = {
                "get_market_data":           "Piyasa verisi çekiliyor...",
                "analyze_technical":         "Teknik analiz hesaplanıyor...",
                "analyze_risk":              "Risk metrikleri hesaplanıyor...",
                "analyze_fundamental":       "Temel veriler çekiliyor...",
                "analyze_portfolio":         "Portföy analizi yapılıyor...",
                "web_search":                "Web araması yapılıyor...",
                "search_financial_documents":"Belgeler taranıyor...",
                "calculate_loan_payment":    "Kredi hesaplanıyor...",
                "get_current_loan_rates":    "Kredi oranları kontrol ediliyor...",
            }
            spinner_msg = SPINNER_MESSAGES.get(tool_name, "İşleniyor...")
            with st.spinner(spinner_msg):
                result = TOOLS[tool_name]["function"](**typed_params)

            # Analist Modu: analyze_technical çağrıldığında risk + temel veriyi de otomatik zincirle
            if persona == "Analist Modu" and tool_name == "analyze_technical":
                symbol_arg = typed_params.get("symbol", "")
                if symbol_arg:
                    extra_parts = []
                    with st.spinner("Risk metrikleri hesaplanıyor..."):
                        risk_res = risk_snapshot(symbol_arg)
                    if "hata" not in risk_res:
                        extra_parts.append(
                            "--- RİSK METRİKLERİ ---\n"
                            + json.dumps(risk_res, indent=2, ensure_ascii=False)
                        )
                    with st.spinner("Temel veriler çekiliyor..."):
                        fund_res = fundamental_snapshot(symbol_arg)
                    if "hata" not in fund_res:
                        extra_parts.append(
                            "--- TEMEL ANALİZ VERİLERİ ---\n"
                            + json.dumps(fund_res, indent=2, ensure_ascii=False)
                        )
                    if extra_parts:
                        result["_ek_analizler"] = extra_parts

            # get_market_data cache'li; session_state burada güncelle
            if tool_name == "get_market_data" and isinstance(result, dict) and "hata" not in result:
                symbols_arg = typed_params.get("symbols", "")
                _update_stock_session_state(symbols_arg, result)

                # Analist Modu: multi-sembol sorgularında her sembol için teknik+risk+temel zincirle
                if persona == "Analist Modu" and symbols_arg:
                    semboller = [s.strip().upper() for s in symbols_arg.replace(",", " ").split() if s.strip()]
                    if len(semboller) >= 1:
                        extra_parts = []
                        for sym in semboller[:3]:  # max 3 sembol
                            with st.spinner(f"{sym} teknik analiz..."):
                                t_res = compute_technical_snapshot(sym)
                            if "hata" not in t_res:
                                extra_parts.append(f"--- {sym} TEKNİK ANALİZ ---\n" + json.dumps(t_res, indent=2, ensure_ascii=False))
                            with st.spinner(f"{sym} risk analiz..."):
                                r_res = risk_snapshot(sym)
                            if "hata" not in r_res:
                                extra_parts.append(f"--- {sym} RİSK ---\n" + json.dumps(r_res, indent=2, ensure_ascii=False))
                            with st.spinner(f"{sym} temel analiz..."):
                                f_res = fundamental_snapshot(sym)
                            if "hata" not in f_res:
                                extra_parts.append(f"--- {sym} TEMEL ---\n" + json.dumps(f_res, indent=2, ensure_ascii=False))
                        if extra_parts:
                            if isinstance(result, dict):
                                result["_ek_analizler"] = extra_parts

            if tool_name == "search_financial_documents":
                tool_output = "\n\n".join(f"Dahili Belge İçeriği:\n{d['text']}" for d in result)
            elif isinstance(result, list):
                tool_output = json.dumps(result, indent=2, ensure_ascii=False)
            elif isinstance(result, dict):
                if "hata" in result:
                    detay = result.get("detay", "")
                    if detay:
                        tool_output = f"Hata: {result['hata']}\nDetay: {detay}"
                    else:
                        tool_output = f"Hata: {result['hata']}"
                else:
                    tool_output = json.dumps(result, indent=2, ensure_ascii=False)
            elif isinstance(result, str):
                tool_output = result
            else:
                tool_output = f"{result:,.2f}"

            if not tool_output:
                tool_output = "İlgili sonuç bulunamadı."
        except Exception as e:
            # Stack trace kullanıcıya gösterilmez; kategorik Türkçe mesaj üretilir
            err_str = str(e).lower()
            if "timeout" in err_str or "timed out" in err_str:
                tool_output = "Hata: Veri kaynağına bağlanırken zaman aşımı oluştu. Lütfen tekrar deneyin."
            elif "api" in err_str or "key" in err_str or "401" in err_str or "403" in err_str:
                tool_output = "Hata: API anahtarı geçersiz veya erişim reddedildi. Lütfen yapılandırmayı kontrol edin."
            elif "connection" in err_str or "network" in err_str:
                tool_output = "Hata: Ağ bağlantısı kurulamadı. İnternet bağlantınızı kontrol edin."
            elif "sembol" in err_str or "symbol" in err_str or "ticker" in err_str:
                tool_output = "Hata: Sembol bulunamadı veya geçersiz. Sembolü kontrol edip tekrar deneyin."
            else:
                tool_output = "Hata: İşlem sırasında beklenmeyen bir sorun oluştu. Lütfen tekrar deneyin."

    # web_search için özel final prompt (Kaynaklar + Öneriler)
    if tool_name == "web_search":
        final_prompt_text = f"""Kullanıcının '{last_user_msg}' sorusuna cevap vermek için bir web arama aracı çalıştırıldı ve aşağıdaki JSON sonuç döndü:
--- ARAÇ SONUCU (web_search JSON) ---
{tool_output}
---
Bu JSON, şu alanları içeriyor:
- "query": Kullanıcının arama ifadesi
- "results": Her elemanda "title", "snippet" ve "link" alanları olan bir liste

SENİN GÖREVİN:
1. "results" içindeki "title" ve "snippet" alanlarını kullanarak kullanıcının sorusuna net, özet ve güncel bir cevap yaz.
2. Her bilgi maddesinin sonuna parantez içinde **(Web araması)** etiketini ekle.
3. Cevabının EN ALTINDA mutlaka ayrı bir blok olarak şu formatta kaynakları listele:
   Kaynaklar:
   - <link1>
   - <link2>
   - ...
4. Linkleri sadece "results" içindeki "link" alanlarından al. Yeni kaynak uydurma.
5. Genel anlatım kısmında uzun URL yazma; linkleri sadece "Kaynaklar:" bölümünde ver.
6. Cevabının sonunda, her zamanki gibi, kullanıcının merak edebileceği EN AZ 3 devam sorusunu "Şunları da merak edebilirsiniz:" başlığıyla madde madde yaz.
"""
    elif tool_name == "search_financial_documents":
        final_prompt_text = f"""Kullanıcının '{last_user_msg}' sorusuna cevap vermek için dahili mevzuat belgeleri tarandı:
--- ARAÇ SONUCU (Dahili Belgeler) ---
{tool_output}
---
SENİN GÖREVİN:
1. Yukarıdaki belge içeriklerini kullanarak kullanıcının sorusuna net ve doğru bir yanıt ver.
2. Her bilgi maddesinin sonuna parantez içinde **(Dahili Belge)** etiketini ekle.
3. Belgede kural/madde numarası veya tebliğ adı geçiyorsa mutlaka belirt.
4. Belgede açıkça yanıt yoksa, "Dahili belgelerimde bu konuya dair doğrudan bir hüküm bulunamadı" de ve genel mevzuat çerçevesini kısaca anlat; tarih veya madde uydurma.
5. Cevabının sonunda EN AZ 3 devam sorusu öner.
"""
    else:
        # Analist Modu zincirleme verilerini tool_output'a ekle
        ek_analizler = ""
        if isinstance(result, dict) and "_ek_analizler" in result:
            ek_analizler = "\n\n" + "\n\n".join(result["_ek_analizler"])
            # JSON çıktısından _ek_analizler alanını temizle
            clean_result = {k: v for k, v in result.items() if k != "_ek_analizler"}
            tool_output = json.dumps(clean_result, indent=2, ensure_ascii=False)

        final_prompt_text = f"""Kullanıcının '{last_user_msg}' sorusuna cevap vermek için araçlar çalıştırıldı:
--- TEKNİK ANALİZ SONUCU ---
{tool_output}
{ek_analizler}
---
SENİN GÖREVİN:
1. Araç çıktısındaki TÜM sayısal verileri kullanarak kapsamlı analiz yaz. UYDURMA YOK — sadece gelen veri.
2. TEKNİK göstergeler ZORUNLU olarak cevaba yansımalı: RSI14 (sayı + yorum), MACD (yorum), Bollinger pozisyonu, MA20/50/200 kıyası, Supertrend yönü, ATR, destek/direnç seviyeleri.
3. Birden fazla sembol varsa: Her hisse için ayrı teknik tablo yap, sonra karşılaştırmalı özet ekle.
4. Temel veriler (_ek_analizler) varsa: F/K, P/DD, ROE, net kâr marjı, piyasa değeri — hepsini "Temel Bağlam" bölümüne yaz.
5. Risk metrikleri varsa: Yıllık volatilite %, max drawdown %, Sharpe oranı — "Risk" bölümüne yaz.
6. Destek/direnç sayılarını (destek_guclu, S1, pivot, R1, direnc_guclu) doğrudan yaz, yuvarlama.
7. Cevabını persona formatına uygun yaz, sonunda EN AZ 3 devam sorusu öner.
"""

    history_without_last_prompt = chat_history[:-1]
    messages_for_final_answer = [{"role": "system", "content": system_prompt}] + history_without_last_prompt
    messages_for_final_answer.append({"role": "assistant", "content": tool_call_str})
    messages_for_final_answer.append({"role": "user", "content": final_prompt_text})
    return call_claude(messages_for_final_answer)


# --- Grafik ---

def display_market_chart(history_df, company_name, key=None):
    fig = go.Figure()
    for column in history_df.columns:
        normalized_series = (history_df[column] / history_df[column].iloc[0]) * 100
        fig.add_trace(go.Scatter(x=history_df.index, y=normalized_series, mode="lines", name=column))

    fig.update_layout(
        title=dict(
            text=f"{company_name} · Normalize Edilmiş Performans (Başlangıç=100)",
            font=dict(size=16, color="#1d1d1f"),
        ),
        xaxis_title="Tarih",
        yaxis_title="Yüzdesel Performans Değişimi",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=["#0071e3", "#34c759", "#ff9500", "#ff3b30", "#5e5ce6", "#af52de"],
        font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif", color="#1d1d1f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False)
    fig.update_traces(line=dict(width=2.5))
    st.plotly_chart(fig, use_container_width=True, key=key)
    st.caption(
        "Her hisse başlangıç günü = 100 olarak normalize edilmiştir. "
        "Grafik fiyat düzeylerini değil, göreli performansı karşılaştırır."
    )


# --- Dosya Analizi Yardımcı Fonksiyonları ---

def summarize_dataframe_with_llm(df: pd.DataFrame) -> str:
    """Yüklenen DataFrame için daha detaylı Türkçe özet üretir (persona içerikten otomatik)."""
    try:
        sample_rows = df.head(20)
        sample_markdown = sample_rows.to_markdown(index=False)
    except Exception:
        sample_markdown = str(df.head(20))

    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    shape = df.shape
    columns = list(df.columns)

    prompt = f"""
Aşağıda bir veri setinin özeti var.

Veri seti boyutu: {shape[0]} satır x {shape[1]} sütun
Tüm sütun adları: {columns}
Sayısal sütunlar: {numeric_cols}

İlk 20 satır:
{sample_markdown}

GÖREVİN:
1. Bu veri setinin genel yapısını ve ne tür bilgiler içerdiğini Türkçe olarak detaylı bir şekilde özetle.
2. Eğer veri seti finansal/ekonomik/veri analizi bağlamındaysa, bunu belirt ve buna göre yorum yap (örneğin hisse, kredi, müşteri verisi, makro veri vb.).
3. Sayısal sütunlar açısından genel bir değerlendirme yap (hangi sütunlar önemli görünüyor, kabaca gözlemlenen trendler veya farklılıklar neler).
4. Önemli gördüğün noktaları ve veriyle ilgili dikkat edilmesi gereken hususları maddeler halinde yaz.
5. Son olarak, bu veri setiyle ilgili kullanıcıya önerebileceğin 3 adet analiz veya araştırma sorusu yaz.

Yanıtın biraz uzun olabilir; özetleyici ama yüzeysel olmayacak kadar detaylı olsun.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "Sen EkoFin Asistan'ın Dosya Analizi modusun. "
                "Yüklenen veri setinin içeriğine göre Genel Asistan, Teknik Analist, Temel Analist veya Bankacı "
                "bakış açılarından en uygun olanını zihninde seçerek açıklama yaparsın. "
                "Okuyan kişinin finans öğrencisi veya finans meraklısı olduğunu varsay ve eğitici bir dil kullan."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    return call_claude(messages)


def read_pdf_text(file_bytes: bytes) -> str:
    """PDF dosyadan metin çıkarır, ilk ~15 sayfa/20000 karakterle sınırlar."""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        texts = []
        for i, page in enumerate(reader.pages):
            if i >= 15:  # çok uzun PDF'lerde kısaltma
                break
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            texts.append(page_text)
        full_text = "\n\n".join(texts)
        if len(full_text) > 20000:
            full_text = full_text[:20000]
        return full_text
    except Exception as e:
        return f"[PDF okuma hatası: {e}]"


def summarize_pdf_with_llm(text: str) -> str:
    """PDF metni için detaylı Türkçe özet üretir (içeriğe göre bakış açısı seçerek)."""
    prompt = f"""
Aşağıda bir PDF dokümanından çıkarılmış metin bulunuyor (kısaltılmış olabilir):

{text}

GÖREVİN:
1. Bu dokümanın konusunu, amacını ve ana mesajlarını Türkçe olarak detaylı bir şekilde özetle.
2. Eğer doküman finansal / ekonomik / hukuki bir içerik barındırıyorsa, bunu açıkça belirt ve açıklamanı buna göre şekillendir (örneğin: düzenleme metni, rapor, akademik çalışma, şirket analizi vb.).
3. Önemli başlıkları, kritik noktaları ve okuyucunun dikkat etmesi gereken uyarıları maddeler halinde belirt.
4. Eğer uygunsa, metinde geçen temel kavramları (örneğin: risk, getiri, sermaye piyasası, enflasyon, faiz, regülasyon vb.) kısaca tanımla.
5. Son olarak, bu dokümanla ilgili kullanıcıya önerebileceğin 3 adet araştırma veya değerlendirme sorusu yaz.

Yanıtın özet ama görece detaylı olsun; birkaç paragraf + madde listeleri yazmaktan çekinme.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "Sen EkoFin Asistan'ın Dosya Analizi modusun. "
                "Önce dokümanın içeriğini anlamaya çalışır, sonra ona en uygun bakış açısını seçersin: "
                "eğer düzenleme/tebliğ ise Bankacı Asistanı gibi, hisse/rapor ise Temel veya Teknik Analist gibi, "
                "daha genel bir eğitim metni ise Genel Asistan gibi konuşursun. "
                "Ancak hangi persona olduğunu kullanıcıya söylemek zorunda değilsin; sadece üslubunu buna göre ayarla."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    return call_claude(messages)


# --- Streamlit Uygulaması ---

def run_streamlit_app() -> None:
    global BIST_TICKERS
    BIST_TICKERS = load_bist_tickers()

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="ekofin-hero">
            <div class="eyebrow">🤖 Yapay Zekâ Destekli Finans Asistanı</div>
            <h1>{APP_NAME}</h1>
            <p>Türkiye finans piyasaları, BDDK/SPK mevzuatı ve şirket analizleri için sade, hızlı ve zarif bir deneyim.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None
    if "active_persona" not in st.session_state:
        st.session_state.active_persona = "Genel Asistan"
    if "_prev_persona" not in st.session_state:
        st.session_state._prev_persona = st.session_state.active_persona

    # --- Karşılama Mesajı ---
    WELCOME_MESSAGE = (
        f"Merhaba! Ben **{APP_NAME}**.\n\n"
        "Türkiye finans piyasaları, BDDK/SPK mevzuatı ve ekonomik gelişmeler konularında yanınızdayım.\n\n"
        "**Üç modda çalışıyorum:**\n"
        "- 🧭 **Genel Asistan** — Piyasa, ekonomi ve finans sorularına genel yanıtlar\n"
        "- 📊 **Analist Modu** — Teknik ve temel analiz bir arada\n"
        "- ⚖️ **Mevzuat Asistanı** — BDDK, SPK, TCMB tebliğ ve yönetmelikleri\n\n"
        "Sol panelden modu seçip sormaya başlayabilirsiniz."
    )

    if not st.session_state.chats:
        first_chat_id = f"chat_{time.time()}"
        st.session_state.chats[first_chat_id] = [{"role": "assistant", "content": WELCOME_MESSAGE}]
        st.session_state.active_chat_id = first_chat_id

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(
            f"""
            <div class="ekofin-brand">
                <span class="logo-badge">💡</span>
                <span>{APP_NAME}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        persona_labels = {
            "Genel Asistan": "🧭 Genel Asistan",
            "Analist Modu": "📊 Analist Modu",
            "Mevzuat Asistanı": "⚖️ Mevzuat Asistanı",
        }
        persona_descriptions = {
            "Genel Asistan": "Piyasa, ekonomi ve finans sorularına genel yanıtlar.",
            "Analist Modu": "Teknik + temel analiz bir arada: MA, Supertrend, sektör ve makro bağlam.",
            "Mevzuat Asistanı": "BDDK, SPK, TCMB mevzuatı — tebliğ ve yönetmelik odaklı yanıtlar.",
        }

        selected_persona = st.radio(
            "Asistan Modu",
            options=list(persona_labels.keys()),
            format_func=lambda k: persona_labels[k],
            index=list(persona_labels.keys()).index(st.session_state.active_persona),
        )
        st.caption(persona_descriptions[selected_persona])

        if selected_persona != st.session_state._prev_persona:
            st.toast(f"Mod değiştirildi: {persona_labels[selected_persona]}", icon="🔄")
            st.session_state._prev_persona = selected_persona
        st.session_state.active_persona = selected_persona

        st.markdown("---")
        if st.button("➕ Yeni Sohbet"):
            new_chat_id = f"chat_{time.time()}"
            st.session_state.chats[new_chat_id] = [{"role": "assistant", "content": WELCOME_MESSAGE}]
            st.session_state.active_chat_id = new_chat_id
            st.rerun()

        # --- GÜNCEL MEVZUAT TOPLAYICI ---
        st.markdown("---")
        st.markdown("#### 🌐 Güncel Veri Taraması")

        all_sources = list(SOURCE_MAP.keys())
        regulatory = ["Resmi Gazete", "BDDK", "SPK", "TCMB", "Hazine", "KAP"]
        news_sources = ["Bloomberg HT", "Ekonomim", "Para Analiz", "Dünya Gazetesi"]

        with st.expander("Kaynak seçimi", expanded=False):
            sel_reg = st.multiselect(
                "Düzenleyici Kurumlar",
                options=regulatory,
                default=regulatory,
                key="scraper_regulatory",
            )
            sel_news = st.multiselect(
                "Finans Medyası",
                options=news_sources,
                default=["Bloomberg HT", "Ekonomim"],
                key="scraper_news",
            )

        selected_sources = sel_reg + sel_news

        if st.button("🔄 Şimdi Tara", use_container_width=True, type="primary"):
            if not selected_sources:
                st.warning("En az bir kaynak seçin.")
            else:
                progress_placeholder = st.empty()
                progress_bar = st.progress(0)
                log_lines = []
                total = len(selected_sources)

                def _progress_cb(label: str, count: int):
                    log_lines.append(f"✓ {label}: {count} belge")
                    done = len(log_lines)
                    progress_bar.progress(done / total)
                    progress_placeholder.markdown("\n".join(log_lines))

                with st.spinner("Kaynaklar taranıyor..."):
                    summary = run_scraper(
                        sources=selected_sources,
                        progress_callback=_progress_cb,
                    )

                progress_bar.empty()
                progress_placeholder.empty()

                toplam = summary["toplam"]
                if toplam == 0:
                    st.info("Yeni belge bulunamadı (zaten güncel veya siteler erişilemez).")
                else:
                    st.success(f"✅ {toplam} yeni belge kaydedildi.")
                    with st.expander("Kaynak bazlı özet"):
                        for src, cnt in summary["kaynaklar"].items():
                            if cnt > 0:
                                st.markdown(f"- **{src}**: {cnt}")
                    st.caption(
                        f"Belgeler `resmi_gazete_guncel/` klasörüne kaydedildi. "
                        f"RAG indeksini güncellemek için `python create_index.py` çalıştırın."
                    )

        # Son tarama tarihi
        saved_files = sorted(os.listdir("resmi_gazete_guncel")) if os.path.exists("resmi_gazete_guncel") else []
        if saved_files:
            last_date = saved_files[-1][:10]
            st.caption(f"Son kayıt: {last_date} · {len(saved_files)} belge")

        st.markdown("### Sohbet Geçmişi")
        for chat_id in reversed(list(st.session_state.chats.keys())):
            history = st.session_state.chats[chat_id]
            title = history[1]["content"][:40] if len(history) > 1 and history[1]["role"] == "user" else "Yeni Sohbet"
            button_type = "primary" if chat_id == st.session_state.active_chat_id else "secondary"
            if st.button(title, key=chat_id, use_container_width=True, type=button_type):
                st.session_state.active_chat_id = chat_id
                st.rerun()

    # --- DOSYA ANALİZİ ve PORTFÖY MODU kaldırıldı (MVP) ---
    if False and st.session_state.get("work_mode") == "Dosya Analizi":
        st.header("📂 Dosya Analizi Modu (LLM + Sade Dashboard)")

        uploaded_file = st.file_uploader(
            "CSV / Excel / PDF dosyası yükleyin; içeriğini okuyup özetleyeyim. "
            "Finansal tabloysa ek olarak sade bir dashboard da çıkarırım.",
            type=["csv", "xlsx", "xls", "pdf"],
        )

        # Dosya yüklenince state'e hem içerik hem de orijinal byte'lar kaydediliyor
        if uploaded_file is not None:
            file_changed = (
                "last_uploaded_name" not in st.session_state
                or st.session_state.last_uploaded_name != uploaded_file.name
            )
            if file_changed:
                # Her yeni dosyada state'i sıfırla
                st.session_state.last_uploaded_name = uploaded_file.name
                st.session_state.uploaded_df = None
                st.session_state.uploaded_pdf_text = None
                st.session_state.uploaded_summary = None
                st.session_state.uploaded_file_bytes = None
                st.session_state.uploaded_file_ext = None

                ext = uploaded_file.name.lower().split(".")[-1]
                file_bytes = uploaded_file.getvalue()
                st.session_state.uploaded_file_bytes = file_bytes
                st.session_state.uploaded_file_ext = ext

                if ext in ["csv", "xlsx", "xls"]:
                    # Tablo dosyası
                    try:
                        if ext == "csv":
                            df = pd.read_csv(io.BytesIO(file_bytes))
                        else:
                            df = pd.read_excel(io.BytesIO(file_bytes))
                        st.session_state.uploaded_df = df
                        st.session_state.uploaded_summary = summarize_dataframe_with_llm(df)
                    except Exception as e:
                        st.error(f"Dosya okunurken bir hata oluştu: {e}")
                elif ext == "pdf":
                    # PDF dosyası
                    try:
                        _tmp_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                        total_pages = len(_tmp_reader.pages)
                    except Exception:
                        total_pages = 0
                    if total_pages > 15:
                        st.info(
                            f"PDF {total_pages} sayfa içeriyor. "
                            f"Analiz için yalnızca ilk 15 sayfa kullanıldı."
                        )
                    text = read_pdf_text(file_bytes)
                    st.session_state.uploaded_pdf_text = text
                    st.session_state.uploaded_summary = summarize_pdf_with_llm(text)
                else:
                    st.error("Desteklenmeyen dosya formatı.")

        # 1) Başta: Yapay zeka ile sözel özet
        if "uploaded_summary" in st.session_state and st.session_state.uploaded_summary:
            st.markdown("### 🧠 Dosya Özeti (Yapay Zeka)")
            st.markdown(st.session_state.uploaded_summary)

        # 2) Tablo dosyaları için: sade dashboard + mantıklı ortalamalar
        if "uploaded_df" in st.session_state and st.session_state.uploaded_df is not None:
            df = st.session_state.uploaded_df.copy()
            st.markdown("### 📊 Sayısal Özet (Konsantre KPI'lar)")

            numeric_cols = df.select_dtypes(include=["number"]).columns
            df_dash = df.copy()

            # tarih alanı varsa normalize et
            has_date = False
            if "date" in df_dash.columns:
                df_dash["date"] = pd.to_datetime(df_dash["date"], errors="coerce")
                if df_dash["date"].notna().any():
                    has_date = True

            # --- Ana KPI'lar (total_amount varsa daha anlamlı göster) ---
            c1, c2, c3 = st.columns(3)

            if "total_amount" in df_dash.columns:
                total_amount = float(df_dash["total_amount"].sum())

                if "transaction_id" in df_dash.columns:
                    tx_totals = df_dash.groupby("transaction_id")["total_amount"].sum()
                    avg_ticket = float(tx_totals.mean()) if not tx_totals.empty else 0.0
                    num_tx = int(tx_totals.shape[0])
                else:
                    avg_ticket = float(df_dash["total_amount"].mean())
                    num_tx = int(df_dash["total_amount"].count())

                c1.metric("Toplam Tutar", f"{total_amount:,.2f}")
                c2.metric("Ort. İşlem Tutarı", f"{avg_ticket:,.2f}")
                c3.metric("İşlem Sayısı", f"{num_tx:,}")
            else:
                # Daha genel fallback: veri kümesi çok finansal değilse
                total_rows = len(df_dash)
                c1.metric("Kayıt Sayısı", f"{total_rows:,}")
                if len(numeric_cols) > 0:
                    first_num = numeric_cols[0]
                    c2.metric(f"{first_num} Ortalaması", f"{df_dash[first_num].mean():,.2f}")
                c3.metric("Sayısal Sütun Sayısı", f"{len(numeric_cols)}")

            # --- Kısa sayısal yorumlar (metinle) ---
            yorumlar = []

            if has_date and "total_amount" in df_dash.columns:
                daily = (
                    df_dash.dropna(subset=["date"])
                    .set_index("date")
                    .resample("D")["total_amount"]
                    .sum()
                )
                if not daily.empty:
                    daily_avg = float(daily.mean())
                    yorumlar.append(
                        f"- Günlük ortalama toplam tutar yaklaşık **{daily_avg:,.2f}** birim."
                    )

            if "customer_id" in df_dash.columns and "total_amount" in df_dash.columns:
                cust_totals = df_dash.groupby("customer_id")["total_amount"].sum()
                yorumlar.append(
                    f"- Toplam **{cust_totals.index.nunique()}** farklı müşteri kaydı var."
                )
                top_cust = cust_totals.sort_values(ascending=False).head(1)
                if not top_cust.empty:
                    cid = top_cust.index[0]
                    val = float(top_cust.iloc[0])
                    yorumlar.append(
                        f"- En çok harcama yapan müşteri **{cid}**, toplam **{val:,.2f}** birim harcamış."
                    )

            if "product_category" in df_dash.columns and "total_amount" in df_dash.columns:
                cat_totals = (
                    df_dash.groupby("product_category")["total_amount"]
                    .sum()
                    .sort_values(ascending=False)
                )
                top3 = cat_totals.head(3)
                if not top3.empty:
                    items = "; ".join([f"{idx}: {float(val):,.0f}" for idx, val in top3.items()])
                    yorumlar.append(f"- Ciroda öne çıkan ilk 3 kategori: {items}.")

            if "product_name" in df_dash.columns and "total_amount" in df_dash.columns:
                prod_totals = (
                    df_dash.groupby("product_name")["total_amount"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(3)
                )
                if not prod_totals.empty:
                    items = ", ".join(prod_totals.index.tolist())
                    yorumlar.append(f"- En çok ciro yapan ürünler: **{items}**.")

            if yorumlar:
                st.markdown("**Kısa Sayısal Yorumlar:**")
                st.markdown("\n".join(yorumlar))

            # 3) Görselleştirmeler: sade, pür, gereksiz tablo yok
            st.markdown("### 📈 Görselleştirmeler")

            if has_date and "total_amount" in df_dash.columns:
                daily_df = (
                    df_dash.dropna(subset=["date"])
                    .set_index("date")
                    .resample("D")["total_amount"]
                    .sum()
                    .reset_index()
                )
                if not daily_df.empty:
                    fig_line = px.line(
                        daily_df,
                        x="date",
                        y="total_amount",
                        labels={"date": "Tarih", "total_amount": "Toplam Tutar"},
                        title="Günlük Toplam Tutar",
                    )
                    st.plotly_chart(fig_line, use_container_width=True)

            if "product_category" in df_dash.columns and "total_amount" in df_dash.columns:
                cat_sales = (
                    df_dash.groupby("product_category")["total_amount"]
                    .sum()
                    .sort_values(ascending=False)
                    .reset_index()
                )
                if not cat_sales.empty:
                    fig_cat = px.bar(
                        cat_sales,
                        x="product_category",
                        y="total_amount",
                        labels={"product_category": "Kategori", "total_amount": "Toplam Tutar"},
                        title="Kategori Bazında Toplam Tutar",
                    )
                    fig_cat.update_layout(bargap=0.35, bargroupgap=0.25)
                    st.plotly_chart(fig_cat, use_container_width=True)

            if "product_name" in df_dash.columns and "total_amount" in df_dash.columns:
                top_products = (
                    df_dash.groupby("product_name")["total_amount"]
                    .sum()
                    .nlargest(10)
                    .sort_values(ascending=True)
                    .reset_index()
                )
                if not top_products.empty:
                    fig_prod = px.bar(
                        top_products,
                        y="product_name",
                        x="total_amount",
                        orientation="h",
                        labels={"product_name": "Ürün", "total_amount": "Toplam Tutar"},
                        title="En Çok Ciro Yapan Ürünler",
                    )
                    fig_prod.update_layout(bargap=0.35, bargroupgap=0.25)
                    st.plotly_chart(fig_prod, use_container_width=True)

            # 4) Detay isteyen için: veri önizleme ve histogramlar expander içinde
            with st.expander("Detaylı veri önizlemesi (isteğe bağlı)"):
                st.dataframe(df.head(20))

            if len(numeric_cols) > 0:
                with st.expander("Sayısal dağılımlar (isteğe bağlı)"):
                    max_plots = min(3, len(numeric_cols))
                    for col in numeric_cols[:max_plots]:
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=df[col].dropna(), nbinsx=20))
                        fig.update_layout(
                            title=f"{col} dağılımı",
                            xaxis_title=col,
                            yaxis_title="Frekans",
                            bargap=0.2,
                            template="plotly_dark",
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"upload_hist_{col}")

        # PDF için kısa metin kesiti
        if "uploaded_pdf_text" in st.session_state and st.session_state.uploaded_pdf_text:
            st.markdown("### PDF İçeriğinden Kısa Bir Kesit")
            preview = st.session_state.uploaded_pdf_text[:1000]
            if len(st.session_state.uploaded_pdf_text) > 1000:
                preview += "\n\n... (metin kısaltıldı)"
            st.code(preview, language="markdown")

        # --- İNDİRME SEÇENEKLERİ ---
        if "uploaded_file_bytes" in st.session_state and st.session_state.uploaded_file_bytes:
            st.markdown("### 🔽 İndirme Seçenekleri")

            # Orijinal dosyayı indir
            st.download_button(
                label="📁 Orijinal dosyayı indir",
                data=st.session_state.uploaded_file_bytes,
                file_name=st.session_state.last_uploaded_name if "last_uploaded_name" in st.session_state else "yuklenen_dosya",
                mime="application/octet-stream",
            )

            # İşlenmiş tabloyu CSV olarak indir (sadece DataFrame varsa)
            if "uploaded_df" in st.session_state and st.session_state.uploaded_df is not None:
                df_to_save = st.session_state.uploaded_df
                csv_bytes = df_to_save.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📊 İşlenmiş veriyi CSV olarak indir",
                    data=csv_bytes,
                    file_name="ekofin_analiz_verisi.csv",
                    mime="text/csv",
                )

            # Özet metni indir
            if "uploaded_summary" in st.session_state and st.session_state.uploaded_summary:
                summary_text = st.session_state.uploaded_summary
                st.download_button(
                    label="📝 Özet metni indir (.txt)",
                    data=summary_text,
                    file_name="ekofin_dosya_ozeti.txt",
                    mime="text/plain",
                )

        # --- DOSYA SİL / SIFIRLA BUTONU ---
        if (
            ("uploaded_df" in st.session_state and st.session_state.uploaded_df is not None)
            or ("uploaded_pdf_text" in st.session_state and st.session_state.uploaded_pdf_text)
            or ("uploaded_file_bytes" in st.session_state and st.session_state.uploaded_file_bytes)
        ):
            st.markdown("---")
            if st.button("🗑️ Yüklenen dosyayı sil ve baştan başla"):
                for key in [
                    "uploaded_df",
                    "uploaded_pdf_text",
                    "uploaded_summary",
                    "last_uploaded_name",
                    "uploaded_file_bytes",
                    "uploaded_file_ext",
                ]:
                    st.session_state.pop(key, None)
                st.rerun()

        return  # Dosya Analizi modunda sohbet kısmına inmiyoruz.

    if False and st.session_state.get("work_mode") == "Portföy Takibi":
        st.header("📊 Portföy Takibi ve Risk Analizi")

        st.markdown(
            """
            Portföyünü şu formatta girebilirsin:

            ```text
            GARAN:0.3
            AKBNK:0.4
            THYAO:0.3
            ```
            veya

            ```text
            GARAN=0.3, AKBNK=0.4, THYAO=0.3
            ```

            Ağırlıkların toplamı otomatik normalize edilir (0.3+0.4+0.3 → 1.0).
            """
        )

        text_input = st.text_area("Portföy tanımını yaz:", height=150)
        period = st.selectbox("Analiz dönemi:", ["6mo", "1y", "2y"], index=1)

        if st.button("Portföyü Analiz Et") and text_input.strip():
            weights = parse_portfolio_text(text_input)
            if not weights:
                st.error("Portföy tanımı çözümlenemedi. Formatı kontrol et.")
                return

            with st.spinner("Portföy analizi yapılıyor..."):
                summary = portfolio_summary(weights, period=period)

            if "hata" in summary:
                st.error(summary["hata"])
                return

            st.subheader("Özet Metrikler")
            c1, c2, c3 = st.columns(3)
            c1.metric("Toplam Dönem Getirisi", f"{summary['toplam_donem_getirisi_yuzde']:.2f}%")
            c2.metric("Yıllık Getiri Tahmini", f"{summary['yillik_getiri_tahmini_yuzde']:.2f}%")
            c3.metric("Yıllık Volatilite", f"{summary['yillik_volatilite_yuzde']:.2f}%")

            c4, c5 = st.columns(2)
            c4.metric("Max Drawdown", f"{summary['max_drawdown_yuzde']:.2f}%")
            if summary["sharpe_benzeri_oran"] is not None:
                c5.metric("Sharpe benzeri oran", f"{summary['sharpe_benzeri_oran']:.2f}")

            st.markdown("**Ağırlıklar:**")
            st.json(summary["agirliklar"])

            st.markdown("**Sembol Bazlı Korelasyon Katkısı (portföyle):**")
            st.json(summary["sembol_korelasyon_katkisi"])

            st.info(
                f"Analiz dönemi: {summary['tarih_araligi']['ilk_tarih']} - {summary['tarih_araligi']['son_tarih']}"
            )

        return  # Portföy modunda sohbet kısmına inmiyoruz.

    # --- SOHBET MODU ---
    active_chat_history = st.session_state.chats[st.session_state.active_chat_id]

    prompt = None
    last_suggestions = None

    for idx, msg in enumerate(active_chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if "stock_chart_data" in msg:
                    history_data = pd.DataFrame.from_dict(msg["stock_chart_data"]["history"], orient="index")
                    history_data.index = pd.to_datetime(history_data.index)
                    chart_key = f"history_chart_{st.session_state.active_chat_id}_{idx}"
                    display_market_chart(history_data, msg["stock_chart_data"]["name"], key=chart_key)
                if "suggestions" in msg:
                    last_suggestions = msg["suggestions"]

    # Giriş ekranı butonları
    if len(active_chat_history) == 1:
        st.markdown("---")
        st.markdown("**Hızlı Başlangıç Önerileri:**")

        initial_questions = [
            "GARAN ve AKBNK son 6 ay performansını karşılaştır",
            "BDDK'nın son kredi düzenlemeleri neler?",
            "THYAO teknik analizi",
            "SPK'nın halka arz mevzuatında temel şartlar neler?",
        ]

        col1, col2 = st.columns(2)
        with col1:
            if st.button(initial_questions[0], key="init_btn_0", use_container_width=True):
                prompt = initial_questions[0]
            if st.button(initial_questions[2], key="init_btn_2", use_container_width=True):
                prompt = initial_questions[2]
        with col2:
            if st.button(initial_questions[1], key="init_btn_1", use_container_width=True):
                prompt = initial_questions[1]
            if st.button(initial_questions[3], key="init_btn_3", use_container_width=True):
                prompt = initial_questions[3]

    # Son asistandan gelen öneriler
    if last_suggestions:
        st.markdown("---")
        cols = st.columns(len(last_suggestions))
        for i, suggestion in enumerate(last_suggestions):
            with cols[i]:
                btn_key = f"suggestion_btn_{st.session_state.active_chat_id}_{i}"
                if st.button(suggestion, key=btn_key):
                    prompt = suggestion

    user_input = st.chat_input("Sorunuzu yazın…")
    st.markdown(
        """
        <div class="ekofin-disclaimer">
        ⚠️ EkoFin Asistan tarafından sunulan içerikler yalnızca bilgilendirme amaçlıdır ve yatırım tavsiyesi niteliği taşımaz.
        Burada yer alan hiçbir bilgi, Sermaye Piyasası Kurulu (SPK) veya herhangi bir yetkili kurum tarafından yetkilendirilmiş
        yatırım danışmanlığı hizmeti olarak değerlendirilemez. Yatırım kararlarınızı vermeden önce lisanslı bir yatırım danışmanına
        başvurmanız tavsiye edilir.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if user_input:
        prompt = user_input

    if prompt:
        active_chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"{st.session_state.active_persona} düşünüyor ve araştırıyor..."):
                full_answer = run_tool_calling_logic(active_chat_history, st.session_state.active_persona)

                suggestion_headings = [
                    "Şunları da merak edebilirsiniz:",
                    "İlgili diğer analizler:",
                    "Detaylı araştırma konuları:",
                    "İlgili diğer prosedürler:",
                    "Öneriler:",
                    "Ek Sorular:",
                ]
                main_answer = full_answer
                suggestions: List[str] = []
                for heading in suggestion_headings:
                    if heading in full_answer:
                        parts = full_answer.split(heading, 1)
                        main_answer = parts[0].strip()
                        suggestion_lines = parts[1].strip().split("\n")
                        suggestions = [
                            line.strip().lstrip("-•0123456789). ")
                            for line in suggestion_lines
                            if line.strip()
                        ][:3]
                        break

                st.markdown(main_answer)
                assistant_response_to_save: Dict[str, Any] = {"role": "assistant", "content": main_answer}

                if "stock_history" in st.session_state and st.session_state.stock_history is not None:
                    history_data = st.session_state.stock_history
                    company_name = st.session_state.get("stock_company_name", "Piyasa Aracı")
                    chart_key = f"live_chart_{st.session_state.active_chat_id}_{len(active_chat_history)}_{time.time()}"
                    assistant_response_to_save["stock_chart_data"] = {
                        "history": history_data.to_dict(orient="index"),
                        "name": company_name,
                    }
                    display_market_chart(history_data, company_name, key=chart_key)

                if suggestions:
                    assistant_response_to_save["suggestions"] = suggestions

                active_chat_history.append(assistant_response_to_save)

        st.rerun()


if __name__ == "__main__":
    run_streamlit_app()
