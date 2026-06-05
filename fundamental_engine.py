# fundamental_engine.py — EkoFin Temel Analiz Motoru
# yfinance .info endpoint'inden anlık temel veriler.
# Bilanço ağırlıklı değil; değerleme çarpanları + şirket profili odaklı.

from typing import Dict, Any, Optional
import yfinance as yf


_FIELD_MAP = {
    # Değerleme
    "trailingPE":        ("f_k_orani",              "F/K (son 12 ay)"),
    "forwardPE":         ("f_k_beklenen",           "F/K (beklenen)"),
    "priceToBook":       ("p_dd_orani",              "P/DD"),
    "priceToSalesTrailing12Months": ("f_s_orani",   "F/S"),
    "enterpriseToEbitda":("ev_ebitda",              "EV/EBITDA"),

    # Büyüme & Karlılık
    "revenueGrowth":     ("gelir_buyumesi_yuzde",   "Gelir büyümesi (YoY %)"),
    "earningsGrowth":    ("kar_buyumesi_yuzde",      "Kâr büyümesi (YoY %)"),
    "profitMargins":     ("net_kar_marji_yuzde",    "Net kâr marjı %"),
    "operatingMargins":  ("faaliyet_marji_yuzde",   "Faaliyet marjı %"),
    "returnOnEquity":    ("oz_sermaye_getirisi",     "ROE %"),
    "returnOnAssets":    ("aktif_getirisi",          "ROA %"),

    # Temettü
    "dividendYield":     ("temettü_verimi_yuzde",   "Temettü verimi %"),
    "payoutRatio":       ("dagitiim_orani",          "Dağıtım oranı"),

    # Piyasa
    "marketCap":         ("piyasa_degeri_tl",       "Piyasa değeri"),
    "fiftyTwoWeekHigh":  ("52_hafta_yuksek",        "52 hafta yüksek"),
    "fiftyTwoWeekLow":   ("52_hafta_dusuk",         "52 hafta düşük"),
    "beta":              ("beta",                    "Beta (piyasa korelasyonu)"),

    # Borç
    "debtToEquity":      ("borc_oz_sermaye",        "Borç/Özsermaye"),
    "currentRatio":      ("cari_oran",               "Cari oran"),
    "quickRatio":        ("asit_test_orani",         "Asit-test oranı"),

    # Genel
    "sector":            ("sektor",                  "Sektör"),
    "industry":          ("sanayi",                  "Endüstri"),
    "fullTimeEmployees": ("tam_zamanli_calisan",     "Çalışan sayısı"),
    "country":           ("ulke",                    "Ülke"),
    "currency":          ("para_birimi",             "Para birimi"),
}


def _pct(val) -> Optional[float]:
    """Oran → yüzde dönüşümü (0.05 → 5.0)."""
    return round(float(val) * 100, 2) if val is not None else None


def fundamental_snapshot(symbol: str) -> Dict[str, Any]:
    """
    Sembol için temel analiz özeti. yfinance .info kullanır.
    Yüzdelik alanlar (marjlar, büyüme, verim) otomatik *100 yapılır.
    """
    raw = symbol.strip().upper()
    if len(raw) in [4, 5] and "." not in raw and raw.isalpha():
        yf_symbol = raw + ".IS"
        display_symbol = raw
    else:
        yf_symbol = raw
        display_symbol = raw

    try:
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info or {}
    except Exception as e:
        return {"sembol": display_symbol, "hata": f"Temel veri alınamadı: {e}"}

    if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
        return {
            "sembol": display_symbol,
            "hata": "Bu sembol için temel veri bulunamadı (yfinance .info boş döndü)."
        }

    PCT_FIELDS = {
        "revenueGrowth", "earningsGrowth", "profitMargins",
        "operatingMargins", "returnOnEquity", "returnOnAssets",
        "dividendYield", "payoutRatio",
    }

    result: Dict[str, Any] = {"sembol": display_symbol}

    for yf_key, (tr_key, label) in _FIELD_MAP.items():
        val = info.get(yf_key)
        if val is None:
            continue
        if yf_key in PCT_FIELDS:
            result[tr_key] = _pct(val)
        elif isinstance(val, float):
            result[tr_key] = round(val, 4)
        else:
            result[tr_key] = val

    # Piyasa değerini milyar olarak formatla
    if "piyasa_degeri_tl" in result:
        mc = result["piyasa_degeri_tl"]
        if mc and mc > 1e9:
            result["piyasa_degeri_milyar"] = round(mc / 1e9, 2)
        elif mc and mc > 1e6:
            result["piyasa_degeri_milyon"] = round(mc / 1e6, 2)

    # Şirket özeti (varsa, 300 karla sınırla)
    desc = info.get("longBusinessSummary") or info.get("shortName", "")
    if desc:
        result["sirket_ozeti"] = desc[:300]

    if len(result) <= 2:
        return {
            "sembol": display_symbol,
            "hata": "Temel veri alanları boş — bu sembol için yfinance .info dolu değil.",
        }

    return result
