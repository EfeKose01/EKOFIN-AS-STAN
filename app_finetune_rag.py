# app_finetune_rag.py — EkoFin Asistan
# (Nihai Sürüm: Chatbot + RAG + Web Arama + Dosya Analizi + Güncellenmiş Arayüz)

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

# Web Otomasyonu (kredi oranları aracı için)
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- PERSONA'LAR ---

PERSONA_PROMPTS = {
    "Genel Asistan": """Sen, EkoFin Asistan adında, Türkiye ekonomisi ve finans piyasaları konusunda geniş bilgiye sahip, yardımcı ve objektif bir yapay zeka asistanısın.
Görevin, sana sunulan araç sonuçlarını ve verileri kullanarak kullanıcının sorusuna net ve anlaşılır bir cevap vermek. Asla yatırım tavsiyesi verme.

CEVAP FORMATIN ŞU ŞEKİLDE OLMALI:
1.  **Ana Cevap:** Kullanıcının sorusuna doğrudan, veriye dayalı ve net bir yanıt ver.
2.  **Öneriler:** Cevabınla ilgili olarak kullanıcının merak edebileceği EN AZ 3 adet devam sorusu öner. Bu soruları her zaman "Şunları da merak edebilirsiniz:" başlığı altında, liste formatında ('- Soru 1') sun.
""",
    "Teknik Analist": """Sen, bir Borsa Teknik Analistisin. Görevin, sadece hisse senetleri ve endekslerin grafiklerini ve teknik göstergelerini yorumlamak. Veri odaklı, kısa, net ve objektif ol. Asla "al" veya "sat" deme.
Cevabının sonunda, "İlgili diğer analizler:" başlığı altında EN AZ 3 adet devam sorusu öner. Örn: '- Bu hissenin hacim analizini yapabilir misin?'
""",
    "Temel Analist / Araştırmacı": """Sen, bir Finansal Araştırmacı ve Temel Analistsin. Bir hissenin veya endeksin arkasındaki temel dinamikleri analiz et.
Cevaplarını daima kaynaklarla destekle ve sonunda "Detaylı araştırma konuları:" başlığı altında EN AZ 3 adet devam sorusu öner.
""",
    "Bankacı Asistanı": """Sen, bir Kurumsal Bilgi Asistanısın. Banka ürünleri ve prosedürleri hakkındaki sorulara öncelikle DAHİLİ BELGELERDEN yararlanarak doğru cevaplar ver.
Cevabının sonunda, "İlgili diğer prosedürler:" başlığı altında EN AZ 3 adet devam sorusu öner.
"""
}

APP_NAME = "EkoFin Asistan"
st.set_page_config(page_title=APP_NAME, page_icon="🤖", layout="wide")


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

# --- RAG İndeksi ---

FAISS_INDEX_PATH = "rag_index.faiss"
CONTENT_MAP_PATH = "rag_content.pkl"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"


@st.cache_resource
def load_semantic_search_engine():
    model = SentenceTransformer(EMBEDDING_MODEL)
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CONTENT_MAP_PATH, "rb") as f:
        content_map = pickle.load(f)
    return model, index, content_map


if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CONTENT_MAP_PATH):
    SEARCH_MODEL, RAG_INDEX, RAG_CONTENT_MAP = load_semantic_search_engine()
else:
    st.error("RAG indeks dosyaları bulunamadı. Lütfen önce `python create_index.py` komutunu çalıştırın.")
    st.stop()


# --- Araç Fonksiyonları ---

def loan_payment(principal: float, annual_rate: float, years: float, payments_per_year: int = 12) -> float:
    r = annual_rate / payments_per_year
    n = int(round(years * payments_per_year))
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def search_documents(query: str, k: int = 4) -> List[Dict[str, Any]]:
    query_vector = SEARCH_MODEL.encode([query])
    distances, indices = RAG_INDEX.search(np.array(query_vector).astype("float32"), k)
    return [RAG_CONTENT_MAP[i] for i in indices[0]]


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


def get_market_data(symbols: str) -> Dict[str, Any]:
    """
    Bir veya daha fazla sembol için fiyat geçmişini çeker.
    - BIST sembolleri için .IS ekler.
    - download + history fallback
    - çoklu sembolde tek grafikte normalize edilmiş çizgi
    - istatistikler: ilk/son fiyat + yüzde değişim
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

    st.session_state.stock_history = comparison_df
    st.session_state.stock_company_name = ", ".join(valid_cols.keys())
    st.session_state.last_symbols = list(comparison_df.columns)

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

    if len(comparison_df.columns) == 1:
        col = comparison_df.columns[0]
        last_close = float(comparison_df[col].iloc[-1])
        previous_close = float(comparison_df[col].iloc[-2]) if len(comparison_df) > 1 else last_close
        change = last_close - previous_close
        percent_change = (change / previous_close * 100) if previous_close != 0 else 0.0

        result: Dict[str, Any] = {
            "sembol": col,
            "guncel_fiyat": f"{last_close:.2f}",
            "veri_tarihi": comparison_df.index[-1].strftime("%Y-%m-%d"),
            "gunluk_degisim": f"{change:.2f}",
            "gunluk_degisim_yuzde": f"{percent_change:.2f}%",
            "yillik_istatistik": stats[col],
        }
        if uyari_parts:
            result["uyari"] = " ".join(uyari_parts)
        return result

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


# app_finetune_rag.py içinde bu fonksiyonu bul ve değiştir:

def get_current_loan_rates(amount: int, term: int) -> Dict[str, Any]:
    print(f"--- CHROME TARAYICISI (HEADLESS) BAŞLATILDI: Tutar={amount}, Vade={term} ay ---")
    try:
        # --- LINUX/STREAMLIT CLOUD UYUMLU AYARLAR ---
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--headless")  # Arayüz olmadan çalış
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")

        # Streamlit Cloud'da driver otomatik bulunur, path vermeye gerek yok
        driver = webdriver.Chrome(options=chrome_options)

        url = f"https://www.hangikredi.com/kredi/ihtiyac-kredisi?amount={amount}&term={term}"
        driver.get(url)
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.XPATH, "//table[contains(@class, 'offer-table')]//tbody/tr")))
        offers = driver.find_elements(By.XPATH, "//table[contains(@class, 'offer-table')]//tbody/tr")
        results = []
        for offer in offers[:5]:
            try:
                bank_name = offer.find_element(
                    By.XPATH, ".//div[contains(@class, 'bank-logo')]//img"
                ).get_attribute("alt")
                interest_rate = offer.find_element(By.XPATH, ".//td[2]/div").text
                monthly_payment = offer.find_element(By.XPATH, ".//td[3]/div").text
                results.append(
                    {"banka": bank_name, "aylik_faiz_orani": interest_rate, "aylik_taksit": monthly_payment}
                )
            except Exception:
                continue
        driver.quit()
        if not results:
            return {"hata": "HangiKredi sitesinden kredi teklifleri alınamadı, site yapısı değişmiş olabilir."}
        return {"kredi_teklifleri": results}
    except Exception as e:
        return {"hata": f"Web otomasyonu sırasında hata oluştu (Linux/Chrome): {e}"}


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
        "model": "claude-3-haiku-20240307",
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


# --- TOOL ROUTER ---

TOOLS = {
    "calculate_loan_payment": {"function": loan_payment, "required_params": ["principal", "annual_rate", "years"]},
    "search_financial_documents": {"function": search_documents, "required_params": ["query"]},
    "get_market_data": {"function": get_market_data, "required_params": ["symbols"]},
    "web_search": {"function": web_search, "required_params": ["query"]},
    "get_current_loan_rates": {"function": get_current_loan_rates, "required_params": ["amount", "term"]},
}

TOOL_SYSTEM_PROMPT = """Sen bir araç yönlendiricisin. Kullanıcının mesajını analiz et ve aşağıdaki araçlardan en uygununu, doğru parametrelerle `TOOL_CALL` formatında çağır. Başka hiçbir metin yazma.

# Araçlar
- `get_current_loan_rates(amount, term)`: Kullanıcı, bankaların GÜNCEL ihtiyaç kredisi faiz oranlarını karşılaştırmalı olarak istediğinde kullanılır.
- `get_market_data(symbols)`: Bir veya daha fazla hisse senedini (GARAN, THYAO), forex'i (EUR/USD) veya kriptoyu (BTC/USD) karşılaştırmalı olarak analiz etmek veya grafiğini çizmek için kullanılır. Semboller virgülle ayrılmalıdır.
- `web_search(query)`: "güncel", "en son", "son haber", "bugün", "bu sene", "hangi yıl" gibi kelimeler geçen sorular ile SPK, BDDK, TCMB, MERKEZ BANKASI, FED, ECB, TÜİK gibi kurumların SON kararları / haberleri sorulduğunda MUTLAKA kullanılmalıdır. `query` parametresi, doğrudan kullanıcının son mesajı olmalıdır.
- `calculate_loan_payment(principal, annual_rate, years)`: Belirli bir faiz oranı verilerek kredi taksiti hesaplamak için.
- `search_financial_documents(query)`: "Enflasyon nedir?", "müşteri sırrı nedir?" gibi teorik kavramlar veya dahili mevzuat bilgisinde kullanılır.

**ÖNEMLİ KURALLAR:**
- Eksik zorunlu parametre varsa, `TOOL_CALL` üretme. Bunun yerine, hangi parametrenin eksik olduğunu kullanıcıdan iste. SADECE `web_search` için istisna: Eğer `query` eksikse, query olarak doğrudan kullanıcının son mesajını kullan.
- Kullanıcının sorusu GÜNCEL BİR OLAY/HABER içeriyorsa (özellikle SPK, BDDK, TCMB, FED, "en son", "güncel", "son karar", "hangi yıl" vb.), KESİNLİKLE `web_search` çağır. Asla modelin kendi bilgisiyle uydurma yapma.
- `get_market_data` çıktısında bazı semboller için veri yoksa, yine de veri olan sembollerle analiz yapılabilir. Asla "karşılaştırma mümkün değildir" deme; hangi semboller için veri olmadığını belirt, ama mevcut verilerle karşılaştırma yap.
"""


def run_tool_calling_logic(chat_history: List[Dict[str, Any]], persona: str) -> str:
    system_prompt = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["Genel Asistan"])

    if "stock_history" in st.session_state:
        del st.session_state.stock_history

    last_user_msg = chat_history[-1]["content"]
    lower = last_user_msg.lower()

    analysis_triggers = [
        "grafik",
        "karşılaştır",
        "karşılaştırma",
        "performans",
        "yıllık",
        "fiyat",
        "teknik",
        "analiz",
        "hisse",
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

        # Hisse grafiği fallback
        if should_force_market:
            symbols_found = re.findall(r"\b[A-ZÇĞİÖŞÜ]{3,5}\b", last_user_msg.upper())
            symbols_unique = list(dict.fromkeys(symbols_found))
            if not symbols_unique and "last_symbols" in st.session_state:
                symbols_unique = st.session_state.last_symbols

            if symbols_unique:
                symbols_str = ",".join(symbols_unique)
                result = get_market_data(symbols_str)

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
2. Cevabını, sana atanan kimliğin (persona) gerektirdiği formata uygun şekilde, sonunda EN AZ 3 adet devam sorusu önererek tamamla.
3. Eğer bazı semboller için veri yoksa, bunu belirt ama veri olan semboller üzerinden mutlaka analiz yap.
4. Bu uygulamada sadece kapanış fiyatları ve bunlardan türetilen yüzdesel değişimler ve basit karşılaştırmalar kullanılabilir.
5. RSI, hacim, 50/200 günlük ortalama vb. teknik göstergeler için SAYISAL değeri veya yüzdesel değişimi UYDURMA; bu göstergeler için sadece fiyat ve yüzdesel değişim üzerinden yorum yapabileceğini açıkla.
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

            result = TOOLS[tool_name]["function"](**typed_params)

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
            tool_output = f"Araç çalıştırılırken bir hata oluştu: {e}"

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
2. Cevabının EN ALTINDA mutlaka ayrı bir blok olarak şu formatta kaynakları listele:
   Kaynaklar:
   - <link1>
   - <link2>
   - ...
3. Linkleri sadece "results" içindeki "link" alanlarından al. Yeni kaynak uydurma.
4. Genel anlatım kısmında uzun URL yazma; linkleri sadece "Kaynaklar:" bölümünde ver.
5. Cevabının sonunda, her zamanki gibi, kullanıcının merak edebileceği EN AZ 3 devam sorusunu "Şunları da merak edebilirsiniz:" başlığıyla madde madde yaz.
"""
    else:
        final_prompt_text = f"""Kullanıcının '{last_user_msg}' sorusuna cevap vermek için bir araç çalıştırıldı ve şu sonuç bulundu:
--- ARAÇ SONUCU ---
{tool_output}
---
SENİN GÖREVİN:
1. Bu sonucu analiz et ve kullanıcıya net bir cevap oluştur.
2. Cevabını, sana atanan kimliğin (persona) gerektirdiği formata uygun şekilde, sonunda EN AZ 3 adet devam sorusu önererek tamamla.
3. Araç çıktısında bazı semboller için veri yoksa, yine de veri olan semboller üzerinden analiz yap ve eksik sembolleri ayrıca belirt.
4. Bu uygulamada sadece kapanış fiyatları ve bunlardan türetilen yüzdesel değişimler ve basit karşılaştırmalar kullanılabilir.
5. RSI, hacim, 50/200 günlük ortalama vb. teknik göstergeler için SAYISAL değeri veya yüzdesel değişimi UYDURMA; bu göstergeler sorulursa yalnızca fiyat hareketi ve yüzdesel değişim üzerinden yorum yapabileceğini açıkla.
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
        title=f"{company_name} - Normalize Edilmiş Performans Grafiği (Başlangıç=100)",
        xaxis_title="Tarih",
        yaxis_title="Yüzdesel Performans Değişimi",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


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
    st.title(f"📈 {APP_NAME}")

    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None
    if "active_persona" not in st.session_state:
        st.session_state.active_persona = "Genel Asistan"
    if "work_mode" not in st.session_state:
        st.session_state.work_mode = "Sohbet"

    # --- Karşılama Mesajı (GÜNCELLENDİ) ---
    WELCOME_MESSAGE = (
        f"Merhaba! Ben {APP_NAME}. Bankacılık mevzuatı (BDDK), finansal piyasalar ve "
        "ekonomik gelişmeler konusunda size destek olmak için buradayım.\n\n"
        "İster karmaşık yasal düzenlemeleri sorun, ister piyasa analizi isteyin; "
        "güvenilir verilerle yanınızdayım."
    )

    if not st.session_state.chats:
        first_chat_id = f"chat_{time.time()}"
        st.session_state.chats[first_chat_id] = [{"role": "assistant", "content": WELCOME_MESSAGE}]
        st.session_state.active_chat_id = first_chat_id

    # --- Sidebar ---
    with st.sidebar:
        st.title(f"💡 {APP_NAME}")
        st.session_state.work_mode = st.selectbox(
            "Çalışma Modu:",
            options=["Sohbet", "Dosya Analizi"],
            index=["Sohbet", "Dosya Analizi"].index(st.session_state.work_mode),
        )

        if st.session_state.work_mode == "Sohbet":
            st.session_state.active_persona = st.selectbox(
                "Asistan Modu Seçin:",
                options=list(PERSONA_PROMPTS.keys()),
                index=list(PERSONA_PROMPTS.keys()).index(st.session_state.active_persona),
            )
        else:
            st.markdown(
                "> Dosya Analizi modunda persona seçimi otomatik yapılır. "
                "Özetler dokümanın içeriğine göre uygun bakış açısından yazılır."
            )

        st.markdown("---")
        if st.button("➕ Yeni Sohbet"):
            new_chat_id = f"chat_{time.time()}"
            st.session_state.chats[new_chat_id] = [{"role": "assistant", "content": WELCOME_MESSAGE}]
            st.session_state.active_chat_id = new_chat_id
            st.rerun()

        st.markdown("### Sohbet Geçmişi")
        for chat_id in reversed(list(st.session_state.chats.keys())):
            history = st.session_state.chats[chat_id]
            title = history[1]["content"][:40] if len(history) > 1 and history[1]["role"] == "user" else "Yeni Sohbet"
            button_type = "primary" if chat_id == st.session_state.active_chat_id else "secondary"
            if st.button(title, key=chat_id, use_container_width=True, type=button_type):
                st.session_state.active_chat_id = chat_id
                st.rerun()

    # --- DOSYA ANALİZİ MODU ---
    if st.session_state.work_mode == "Dosya Analizi":
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

    # --- SOHBET MODU ---

    st.header(f"Sohbet Modu: {st.session_state.active_persona}")
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

    # --- GÜNCELLENMİŞ GİRİŞ BUTONLARI ---
    # Giriş ekranı butonları
    if len(active_chat_history) == 1:
        st.markdown("---")
        st.markdown("**Hızlı Başlangıç Önerileri:**")

        initial_questions = [
            "Ticari kredilerde döviz varlığı sınırı nedir?",
            "GARAN ve THYAO hisselerini karşılaştır",
            "BDDK'nın son yayınladığı düzenlemeler neler?",
            "Enflasyon muhasebesi kimleri kapsıyor?"
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