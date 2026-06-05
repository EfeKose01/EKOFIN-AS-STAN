# news_scraper.py — EkoFin Güncel Mevzuat ve Finans Haberleri Toplayıcı
#
# Kapsanan kaynaklar:
#   Düzenleyici : Resmi Gazete, BDDK, SPK, TCMB, Hazine ve Maliye Bakanlığı
#   Piyasa      : KAP (önemli açıklamalar), Borsa İstanbul duyuruları
#   Haber       : Bloomberg HT, Ekonomim, Para Analiz, Dünya Gazetesi

import os
import re
import json
import hashlib
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Any

SAVE_DIR = os.path.join(os.path.dirname(__file__), "resmi_gazete_guncel")
os.makedirs(SAVE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8",
}

FINANCE_KEYWORDS = [
    "bddk", "spk", "tcmb", "merkez bankası", "hazine",
    "bankacılık", "kredi", "faiz", "mevduat", "sermaye piyasası",
    "borsa", "hisse", "tahvil", "bono", "döviz", "enflasyon",
    "tebliğ", "yönetmelik", "karar", "düzenleme", "genelge",
    "finansal", "ekonomi", "kur", "swap", "repo", "halka arz",
    "fon", "sigorta", "emeklilik", "leasing", "faktoring",
]


def _is_finance_relevant(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in FINANCE_KEYWORDS)


def _safe_filename(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:max_len]


def _short_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:6]


def _save_document(title: str, source: str, url: str, date_str: str, content: str) -> str:
    """Belgeyi dosyaya yazar; dosya adını döner."""
    safe = _safe_filename(title)
    fname = f"{date_str}_{source}_{safe}_{_short_hash(url)}.txt"
    fpath = os.path.join(SAVE_DIR, fname)
    body = (
        f"BAŞLIK: {title}\n"
        f"KAYNAK: {source}\n"
        f"TARİH: {date_str}\n"
        f"LİNK: {url}\n"
        f"{'=' * 60}\n\n"
        f"{content.strip()}"
    )
    with open(fpath, "w", encoding="utf-8", errors="replace") as f:
        f.write(body)
    return fname


def _get(url: str, timeout: int = 15) -> requests.Response | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        # Encoding tespiti: önce Content-Type header'ına bak,
        # yoksa apparent_encoding kullan (chardet/charset-normalizer)
        if r.encoding is None or r.encoding.lower() in ("iso-8859-1", "latin-1"):
            detected = r.apparent_encoding
            if detected:
                r.encoding = detected
        return r
    except Exception:
        return None


def _text_from_html(html_bytes: bytes) -> str:
    # Encoding'i kendin tespit et: önce UTF-8 dene, sonra apparent_encoding
    for enc in ("utf-8", "utf-8-sig", "iso-8859-9", "windows-1254", "latin-1"):
        try:
            text = html_bytes.decode(enc, errors="strict")
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        text = html_bytes.decode("utf-8", errors="replace")

    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


# ─────────────────────────────────────────────
#  KAYNAK FONKSİYONLARI
# ─────────────────────────────────────────────

def scrape_resmi_gazete(today: datetime) -> List[Dict[str, Any]]:
    """Bugünün Resmi Gazetesi'nden finans/bankacılık mevzuatını çeker."""
    results = []
    date_str = today.strftime("%Y-%m-%d")
    ymd = today.strftime("%Y%m%d")
    y, m = today.strftime("%Y"), today.strftime("%m")

    url = f"https://www.resmigazete.gov.tr/eskiler/{y}/{m}/{ymd}.htm"
    resp = _get(url)
    if not resp:
        return results

    soup = BeautifulSoup(resp.content, "html.parser")
    for a in soup.find_all("a", href=True):
        title = a.get_text(strip=True)
        if not title or not _is_finance_relevant(title):
            continue
        href = a["href"]
        if not href.startswith("http"):
            href = f"https://www.resmigazete.gov.tr/eskiler/{y}/{m}/{href}"

        content_resp = _get(href)
        if not content_resp:
            continue
        content = _text_from_html(content_resp.content)
        fname = _save_document(title, "ResmiGazete", href, date_str, content)
        results.append({"kaynak": "Resmi Gazete", "baslik": title, "link": href, "dosya": fname})

    return results


def scrape_bddk(today: datetime) -> List[Dict[str, Any]]:
    """BDDK mevzuat ve duyuru sayfalarından son eklentileri çeker."""
    results = []
    date_str = today.strftime("%Y-%m-%d")

    sources = [
        ("https://www.bddk.org.tr/Mevzuat/Liste/51", "BDDK-Mevzuat"),
        ("https://www.bddk.org.tr/Mevzuat/Liste/56", "BDDK-Mevzuat"),
        ("https://www.bddk.org.tr/Duyuru/Liste/1",   "BDDK-Duyuru"),
    ]

    for url, label in sources:
        resp = _get(url)
        if not resp:
            continue
        soup = BeautifulSoup(resp.content, "html.parser")

        for a in soup.find_all("a", href=True):
            title = a.get_text(strip=True)
            if len(title) < 10:
                continue
            href = a["href"]
            if not href.startswith("http"):
                href = "https://www.bddk.org.tr" + href

            # Zaten kaydedilmiş mi kontrol et (hash ile)
            existing = [f for f in os.listdir(SAVE_DIR) if _short_hash(href) in f]
            if existing:
                continue

            content_resp = _get(href)
            if not content_resp:
                content = title
            else:
                content = _text_from_html(content_resp.content)

            fname = _save_document(title, label, href, date_str, content)
            results.append({"kaynak": label, "baslik": title, "link": href, "dosya": fname})
            if len(results) >= 10:
                break

    return results


def scrape_spk(today: datetime) -> List[Dict[str, Any]]:
    """SPK duyuru ve bülten sayfalarından çeker."""
    results = []
    date_str = today.strftime("%Y-%m-%d")

    pages = [
        ("https://www.spk.gov.tr/Duyuru/Listele", "SPK-Duyuru"),
        ("https://www.spk.gov.tr/Bulten",          "SPK-Bulten"),
    ]

    for url, label in pages:
        resp = _get(url)
        if not resp:
            continue
        soup = BeautifulSoup(resp.content, "html.parser")

        for a in soup.find_all("a", href=True):
            title = a.get_text(strip=True)
            if len(title) < 10:
                continue
            href = a["href"]
            if not href.startswith("http"):
                href = "https://www.spk.gov.tr" + href

            existing = [f for f in os.listdir(SAVE_DIR) if _short_hash(href) in f]
            if existing:
                continue

            content_resp = _get(href)
            content = _text_from_html(content_resp.content) if content_resp else title
            fname = _save_document(title, label, href, date_str, content)
            results.append({"kaynak": label, "baslik": title, "link": href, "dosya": fname})
            if len(results) >= 8:
                break

    return results


def scrape_tcmb(today: datetime) -> List[Dict[str, Any]]:
    """TCMB basın duyuruları sayfasından çeker."""
    results = []
    date_str = today.strftime("%Y-%m-%d")
    url = "https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Duyurular/Basin+Duyurulari"
    resp = _get(url)
    if not resp:
        return results

    soup = BeautifulSoup(resp.content, "html.parser")
    for a in soup.find_all("a", href=True):
        title = a.get_text(strip=True)
        if len(title) < 10 or not _is_finance_relevant(title):
            continue
        href = a["href"]
        if not href.startswith("http"):
            href = "https://www.tcmb.gov.tr" + href

        existing = [f for f in os.listdir(SAVE_DIR) if _short_hash(href) in f]
        if existing:
            continue

        content_resp = _get(href)
        content = _text_from_html(content_resp.content) if content_resp else title
        fname = _save_document(title, "TCMB", href, date_str, content)
        results.append({"kaynak": "TCMB", "baslik": title, "link": href, "dosya": fname})
        if len(results) >= 6:
            break

    return results


def scrape_hazine(today: datetime) -> List[Dict[str, Any]]:
    """Hazine ve Maliye Bakanlığı duyurularını çeker."""
    results = []
    date_str = today.strftime("%Y-%m-%d")
    url = "https://www.hmb.gov.tr/duyurular"
    resp = _get(url)
    if not resp:
        return results

    soup = BeautifulSoup(resp.content, "html.parser")
    for a in soup.find_all("a", href=True):
        title = a.get_text(strip=True)
        if len(title) < 10 or not _is_finance_relevant(title):
            continue
        href = a["href"]
        if not href.startswith("http"):
            href = "https://www.hmb.gov.tr" + href

        existing = [f for f in os.listdir(SAVE_DIR) if _short_hash(href) in f]
        if existing:
            continue

        content_resp = _get(href)
        content = _text_from_html(content_resp.content) if content_resp else title
        fname = _save_document(title, "Hazine", href, date_str, content)
        results.append({"kaynak": "Hazine", "baslik": title, "link": href, "dosya": fname})
        if len(results) >= 6:
            break

    return results


def scrape_kap(today: datetime) -> List[Dict[str, Any]]:
    """KAP'tan önemli açıklamaları (BDDK/SPK/finansal) çeker."""
    results = []
    date_str = today.strftime("%Y-%m-%d")
    # KAP bildirim arama endpoint'i (public JSON)
    url = "https://www.kap.org.tr/tr/api/disclosures"
    resp = _get(url)
    if not resp:
        return results

    try:
        data = resp.json()
        items = data if isinstance(data, list) else data.get("data", [])
        for item in items[:20]:
            title = item.get("title") or item.get("name") or ""
            if not _is_finance_relevant(title):
                continue
            link = item.get("url") or item.get("link") or "https://www.kap.org.tr"
            content = item.get("description") or title
            fname = _save_document(title, "KAP", link, date_str, content)
            results.append({"kaynak": "KAP", "baslik": title, "link": link, "dosya": fname})
            if len(results) >= 5:
                break
    except Exception:
        pass

    return results


def scrape_bloomberght(today: datetime) -> List[Dict[str, Any]]:
    """Bloomberg HT finans haberlerini çeker."""
    results = []
    date_str = today.strftime("%Y-%m-%d")

    pages = [
        ("https://www.bloomberght.com/ekonomi",    "BloombergHT-Ekonomi"),
        ("https://www.bloomberght.com/borsa",       "BloombergHT-Borsa"),
        ("https://www.bloomberght.com/dunya",       "BloombergHT-Dunya"),
    ]

    for url, label in pages:
        resp = _get(url)
        if not resp:
            continue
        soup = BeautifulSoup(resp.content, "html.parser")

        for a in soup.find_all("a", href=True):
            title = a.get_text(strip=True)
            if len(title) < 20 or not _is_finance_relevant(title):
                continue
            href = a["href"]
            if not href.startswith("http"):
                href = "https://www.bloomberght.com" + href

            existing = [f for f in os.listdir(SAVE_DIR) if _short_hash(href) in f]
            if existing:
                continue

            content_resp = _get(href)
            content = _text_from_html(content_resp.content) if content_resp else title
            # İlk 3000 karakterle sınırla (haber gövdesi)
            content = content[:3000]
            fname = _save_document(title, label, href, date_str, content)
            results.append({"kaynak": label, "baslik": title, "link": href, "dosya": fname})
            if len(results) >= 8:
                break

    return results


def scrape_ekonomim(today: datetime) -> List[Dict[str, Any]]:
    """Ekonomim.com güncel finans haberlerini çeker."""
    results = []
    date_str = today.strftime("%Y-%m-%d")

    pages = [
        ("https://www.ekonomim.com/ekonomi",   "Ekonomim-Ekonomi"),
        ("https://www.ekonomim.com/finans",    "Ekonomim-Finans"),
    ]

    for url, label in pages:
        resp = _get(url)
        if not resp:
            continue
        soup = BeautifulSoup(resp.content, "html.parser")

        for a in soup.find_all("a", href=True):
            title = a.get_text(strip=True)
            if len(title) < 20:
                continue
            href = a["href"]
            if not href.startswith("http"):
                href = "https://www.ekonomim.com" + href

            existing = [f for f in os.listdir(SAVE_DIR) if _short_hash(href) in f]
            if existing:
                continue

            content_resp = _get(href)
            content = (_text_from_html(content_resp.content) if content_resp else title)[:3000]
            fname = _save_document(title, label, href, date_str, content)
            results.append({"kaynak": label, "baslik": title, "link": href, "dosya": fname})
            if len(results) >= 6:
                break

    return results


def scrape_paraanaliz(today: datetime) -> List[Dict[str, Any]]:
    """Para Analiz güncel haberleri çeker."""
    results = []
    date_str = today.strftime("%Y-%m-%d")
    url = "https://www.paraanaliz.com/haberler/"
    resp = _get(url)
    if not resp:
        return results

    soup = BeautifulSoup(resp.content, "html.parser")
    for a in soup.find_all("a", href=True):
        title = a.get_text(strip=True)
        if len(title) < 20 or not _is_finance_relevant(title):
            continue
        href = a["href"]
        if not href.startswith("http"):
            href = "https://www.paraanaliz.com" + href

        existing = [f for f in os.listdir(SAVE_DIR) if _short_hash(href) in f]
        if existing:
            continue

        content_resp = _get(href)
        content = (_text_from_html(content_resp.content) if content_resp else title)[:3000]
        fname = _save_document(title, "ParaAnaliz", href, date_str, content)
        results.append({"kaynak": "ParaAnaliz", "baslik": title, "link": href, "dosya": fname})
        if len(results) >= 6:
            break

    return results


def scrape_dunya(today: datetime) -> List[Dict[str, Any]]:
    """Dünya Gazetesi ekonomi haberlerini çeker."""
    results = []
    date_str = today.strftime("%Y-%m-%d")
    url = "https://www.dunya.com/ekonomi"
    resp = _get(url)
    if not resp:
        return results

    soup = BeautifulSoup(resp.content, "html.parser")
    for a in soup.find_all("a", href=True):
        title = a.get_text(strip=True)
        if len(title) < 20 or not _is_finance_relevant(title):
            continue
        href = a["href"]
        if not href.startswith("http"):
            href = "https://www.dunya.com" + href

        existing = [f for f in os.listdir(SAVE_DIR) if _short_hash(href) in f]
        if existing:
            continue

        content_resp = _get(href)
        content = (_text_from_html(content_resp.content) if content_resp else title)[:3000]
        fname = _save_document(title, "DunyaGazetesi", href, date_str, content)
        results.append({"kaynak": "DunyaGazetesi", "baslik": title, "link": href, "dosya": fname})
        if len(results) >= 6:
            break

    return results


# ─────────────────────────────────────────────
#  ANA TOPLAMA FONKSİYONU
# ─────────────────────────────────────────────

SOURCE_MAP = {
    "Resmi Gazete":    scrape_resmi_gazete,
    "BDDK":            scrape_bddk,
    "SPK":             scrape_spk,
    "TCMB":            scrape_tcmb,
    "Hazine":          scrape_hazine,
    "KAP":             scrape_kap,
    "Bloomberg HT":    scrape_bloomberght,
    "Ekonomim":        scrape_ekonomim,
    "Para Analiz":     scrape_paraanaliz,
    "Dünya Gazetesi":  scrape_dunya,
}


def run_scraper(
    sources: List[str] | None = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Belirtilen kaynaklardan veri çeker.

    Args:
        sources: Çekilecek kaynak isimleri (None → hepsi).
        progress_callback: Her kaynak tamamlandığında çağrılır (label, count).

    Returns:
        { "toplam": N, "kaynaklar": { kaynak: count }, "belgeler": [...] }
    """
    today = datetime.now()
    active = sources or list(SOURCE_MAP.keys())
    all_docs = []
    source_counts: Dict[str, int] = {}

    for label in active:
        fn = SOURCE_MAP.get(label)
        if not fn:
            continue
        try:
            docs = fn(today)
        except Exception:
            docs = []
        source_counts[label] = len(docs)
        all_docs.extend(docs)
        if progress_callback:
            progress_callback(label, len(docs))

    return {
        "toplam": len(all_docs),
        "kaynaklar": source_counts,
        "belgeler": all_docs,
    }


if __name__ == "__main__":
    def cb(label, count):
        print(f"  {label}: {count} belge")

    print("EkoFin Mevzuat Tarayıcı başlatılıyor...\n")
    result = run_scraper(progress_callback=cb)
    print(f"\nToplam {result['toplam']} yeni belge kaydedildi → {SAVE_DIR}")
