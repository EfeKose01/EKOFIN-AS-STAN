"""Kişiselleştirilmiş haber akışı: mevcut news_scraper.py çıktısını kullanıcının
portföyündeki sembol/sektörlere göre filtreler; ayrıca RAG sohbet asistanına
verilecek kısa "kullanıcı portföyü" bağlam metnini üretir.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta

from backend.services.portfolio_service import SECTOR_MAP

_NEWS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "resmi_gazete_guncel")

# Sektör bazlı anahtar kelimeler — haberin ilgili sektörle alakalı olup olmadığını
# ucuz bir şekilde (embedding gerektirmeden) kestirmek için kullanılır.
SECTOR_KEYWORDS: dict[str, list[str]] = {
    "Bankacılık": ["banka", "bddk", "kredi", "mevduat", "faiz oranı"],
    "Ulaştırma": ["havayolu", "havacılık", "uçak", "ulaştırma"],
    "Savunma Sanayi": ["savunma sanayi", "askeri", "roket", "radar"],
    "Perakende": ["perakende", "market zinciri", "mağaza"],
    "Enerji": ["petrol", "rafineri", "akaryakıt", "enerji piyasası"],
    "Demir-Çelik": ["çelik", "demir çelik", "maden"],
    "Kimya": ["kimya sanayi", "polyester", "elyaf"],
    "Holding": ["holding"],
    "Telekomünikasyon": ["telekom", "mobil operatör", "5g", "fiber"],
}


def _parse_news_file(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except OSError:
        return None

    header, _, body = raw.partition("=" * 60)
    fields = {"title": "", "source": "", "date": "", "url": ""}
    for line in header.splitlines():
        if line.startswith("BAŞLIK:"):
            fields["title"] = line.split(":", 1)[1].strip()
        elif line.startswith("KAYNAK:"):
            fields["source"] = line.split(":", 1)[1].strip()
        elif line.startswith("TARİH:"):
            fields["date"] = line.split(":", 1)[1].strip()
        elif line.startswith("LİNK:"):
            fields["url"] = line.split(":", 1)[1].strip()

    return {**fields, "content": body.strip(), "file": os.path.basename(path)}


def load_recent_news(max_age_days: int = 3) -> list[dict]:
    """resmi_gazete_guncel/ klasöründeki son N günün haberlerini yükler.

    Not: Bu klasör news_scraper.run_scraper() tarafından doldurulur (sidebar'daki
    "Şimdi Tara" butonu veya günlük zamanlanmış bir görevle). Klasör yoksa/boşsa
    sessizce boş liste döner — üretim ortamında bir cron/scheduled task ile
    run_scraper()'ın her sabah çalıştırılması önerilir.
    """
    if not os.path.isdir(_NEWS_DIR):
        return []

    cutoff = datetime.now() - timedelta(days=max_age_days)
    items = []
    for fname in os.listdir(_NEWS_DIR):
        if not fname.endswith(".txt"):
            continue
        date_prefix = fname[:10]  # "YYYY-MM-DD_..."
        try:
            file_date = datetime.strptime(date_prefix, "%Y-%m-%d")
        except ValueError:
            file_date = None
        if file_date and file_date < cutoff:
            continue
        parsed = _parse_news_file(os.path.join(_NEWS_DIR, fname))
        if parsed:
            items.append(parsed)

    items.sort(key=lambda x: x.get("date", ""), reverse=True)
    return items


def filter_news_for_portfolio(
    news_items: list[dict], symbols: list[str], sector_map: dict[str, str] | None = None
) -> list[dict]:
    """Haberleri kullanıcının sembollerine veya sembollerin sektörlerine göre filtreler."""
    if not symbols:
        return []
    sector_map = sector_map or SECTOR_MAP
    symbols_upper = [s.upper() for s in symbols]
    user_sectors = {sector_map.get(s, "Diğer") for s in symbols_upper}
    sector_kw = [kw for sec in user_sectors for kw in SECTOR_KEYWORDS.get(sec, [])]

    symbol_patterns = [re.compile(rf"\b{re.escape(s)}\b", re.IGNORECASE) for s in symbols_upper]

    matched = []
    for item in news_items:
        haystack = f"{item['title']} {item['content'][:500]}"
        hit_symbol = next((s for s, pat in zip(symbols_upper, symbol_patterns) if pat.search(haystack)), None)
        hit_sector_kw = next((kw for kw in sector_kw if kw.lower() in haystack.lower()), None)
        if hit_symbol or hit_sector_kw:
            matched.append({**item, "matched_symbol": hit_symbol, "matched_keyword": hit_sector_kw})

    return matched


def build_portfolio_context_block(symbols_with_sectors: list[tuple[str, str]]) -> str:
    """RAG sohbet asistanının system prompt'una eklenecek kısa bağlam metni.

    Amaç: asistanın kullanıcının hangi hisselere sahip olduğunu bilerek yanıt
    vermesi (örn. "GARAN'daki pozisyonunuzu düşününce..." gibi), ama YİNE DE
    kesinlikle yatırım tavsiyesi vermemesi — bu kural persona prompt'larında
    zaten mevcut (FINANCE_SAFETY_SUFFIX), burada sadece veri ekleniyor.
    """
    if not symbols_with_sectors:
        return ""
    lines = [f"- {sym} ({sector})" for sym, sector in symbols_with_sectors]
    return (
        "\n\nKULLANICI BAĞLAMI (giriş yapmış kullanıcının portföyü — sadece bilgi amaçlı, "
        "asla bu bilgiyi yatırım tavsiyesi vermek için kullanma, sadece örnek/bağlam olarak "
        "kullanıcı sorduğunda ilişkilendir):\n" + "\n".join(lines)
    )
