"""Portföy CRUD + NumPy/Pandas ile değerleme ve sektörel dağılım.

Bu modül Streamlit'ten habersizdir (framework-agnostic) — tüm fonksiyonlar
düz Python veri yapıları (DataFrame, dict, dataclass) ile çalışır ve saf
`Session` nesnesi alır, böylece backend bağımsız test edilebilir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from backend.models import PortfolioItem
from backend.schemas import PortfolioItemCreate, PortfolioItemUpdate

# Hisse sembolü -> sektör eşlemesi. Gerçek projede ayrı bir referans tablosu/CSV
# olabilir; burada dashboard'un çalışması için küçük, genişletilebilir bir örnek.
SECTOR_MAP: dict[str, str] = {
    "GARAN": "Bankacılık", "AKBNK": "Bankacılık", "ISCTR": "Bankacılık", "YKBNK": "Bankacılık",
    "THYAO": "Ulaştırma", "PGSUS": "Ulaştırma",
    "ASELS": "Savunma Sanayi",
    "BIMAS": "Perakende", "MGROS": "Perakende", "SOKM": "Perakende",
    "TUPRS": "Enerji", "PETKM": "Enerji",
    "EREGL": "Demir-Çelik", "KRDMD": "Demir-Çelik",
    "SASA": "Kimya",
    "KCHOL": "Holding", "SAHOL": "Holding",
    "TCELL": "Telekomünikasyon", "TTKOM": "Telekomünikasyon",
}
DEFAULT_SECTOR = "Diğer"


# ---- CRUD ----

def list_portfolio(db: Session, user_id: str) -> list[PortfolioItem]:
    return (
        db.query(PortfolioItem)
        .filter(PortfolioItem.user_id == user_id)
        .order_by(PortfolioItem.symbol)
        .all()
    )


def upsert_portfolio_item(db: Session, user_id: str, data: PortfolioItemCreate) -> PortfolioItem:
    """Sembol zaten varsa miktar/maliyeti ağırlıklı ortalamayla günceller, yoksa yeni satır ekler."""
    existing = (
        db.query(PortfolioItem)
        .filter(PortfolioItem.user_id == user_id, PortfolioItem.symbol == data.symbol)
        .first()
    )
    if existing:
        total_qty = float(existing.quantity) + data.quantity
        blended_cost = (
            float(existing.quantity) * float(existing.cost_basis) + data.quantity * data.cost_basis
        ) / total_qty
        existing.quantity = total_qty
        existing.cost_basis = blended_cost
        existing.source = data.source
        db.flush()
        return existing

    item = PortfolioItem(
        user_id=user_id,
        symbol=data.symbol,
        quantity=data.quantity,
        cost_basis=data.cost_basis,
        source=data.source,
    )
    db.add(item)
    db.flush()
    return item


def update_portfolio_item(db: Session, user_id: str, item_id: str, data: PortfolioItemUpdate) -> PortfolioItem | None:
    item = (
        db.query(PortfolioItem)
        .filter(PortfolioItem.id == item_id, PortfolioItem.user_id == user_id)
        .first()
    )
    if item is None:
        return None
    if data.quantity is not None:
        item.quantity = data.quantity
    if data.cost_basis is not None:
        item.cost_basis = data.cost_basis
    db.flush()
    return item


def delete_portfolio_item(db: Session, user_id: str, item_id: str) -> bool:
    item = (
        db.query(PortfolioItem)
        .filter(PortfolioItem.id == item_id, PortfolioItem.user_id == user_id)
        .first()
    )
    if item is None:
        return False
    db.delete(item)
    db.flush()
    return True


# ---- Değerleme (NumPy / Pandas) ----

def portfolio_to_dataframe(items: list[PortfolioItem]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=["item_id", "symbol", "quantity", "cost_basis"])
    return pd.DataFrame(
        [
            {
                "item_id": it.id,
                "symbol": it.symbol,
                "quantity": float(it.quantity),
                "cost_basis": float(it.cost_basis),
            }
            for it in items
        ]
    )


def value_portfolio(df: pd.DataFrame, current_prices: dict[str, float]) -> pd.DataFrame:
    """Her satır için piyasa değeri, maliyet ve Kar/Zarar (PnL) hesaplar.

    `current_prices`: {"GARAN": 118.5, ...} — çağıran taraf (ör. yfinance/get_market_data)
    fiyatları sağlar; bu fonksiyon fiyat kaynağından bağımsızdır (test edilebilir).
    """
    if df.empty:
        cols = ["symbol", "quantity", "cost_basis", "current_price", "cost_value", "market_value", "pnl", "pnl_pct"]
        return pd.DataFrame(columns=cols)

    out = df.copy()
    out["current_price"] = out["symbol"].map(current_prices).astype(float)
    # Fiyatı bulunamayan semboller için NaN yerine maliyeti kullan (PnL=0 gösterir,
    # sessizce yanlış rakam üretmek yerine "veri yok" izlenimi verir).
    missing_price_mask = out["current_price"].isna()
    out.loc[missing_price_mask, "current_price"] = out.loc[missing_price_mask, "cost_basis"]

    out["cost_value"] = out["quantity"] * out["cost_basis"]
    out["market_value"] = out["quantity"] * out["current_price"]
    out["pnl"] = out["market_value"] - out["cost_value"]
    out["pnl_pct"] = np.where(out["cost_value"] != 0, out["pnl"] / out["cost_value"] * 100, 0.0)
    out["price_missing"] = missing_price_mask
    return out


def portfolio_summary(valued_df: pd.DataFrame) -> dict:
    if valued_df.empty:
        return {
            "total_cost": 0.0, "total_market_value": 0.0, "total_pnl": 0.0,
            "total_pnl_pct": 0.0, "holdings_count": 0,
        }
    total_cost = float(valued_df["cost_value"].sum())
    total_market_value = float(valued_df["market_value"].sum())
    total_pnl = total_market_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0.0
    return {
        "total_cost": round(total_cost, 2),
        "total_market_value": round(total_market_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "holdings_count": int(len(valued_df)),
    }


def sector_allocation(valued_df: pd.DataFrame, sector_map: dict[str, str] | None = None) -> dict[str, float]:
    """Sektöre göre piyasa değeri ağırlığını (%) döner — pasta grafik için hazır JSON'a uygun dict.

    Örn: {"Bankacılık": 41.2, "Teknoloji": 58.8}
    """
    if valued_df.empty:
        return {}
    sector_map = sector_map or SECTOR_MAP
    df = valued_df.copy()
    df["sector"] = df["symbol"].map(sector_map).fillna(DEFAULT_SECTOR)

    grouped = df.groupby("sector")["market_value"].sum()
    total = grouped.sum()
    if total == 0:
        return {}
    pct = (grouped / total * 100).round(2)
    return pct.sort_values(ascending=False).to_dict()


# ---- Canlı fiyat kaynağı ----

def fetch_current_prices(symbols: list[str]) -> dict[str, float]:
    """BIST/kripto/döviz sembolleri için yfinance üzerinden son kapanış fiyatını çeker.

    Not: app_finetune_rag.py'deki get_market_data ile kasıtlı olarak AYRI tutuldu —
    o modül RAG/embedding gibi ağır bağımlılıkları import-time'da yüklüyor; sadece
    fiyat çekmek için o zinciri tetiklememek adına burada hafif, bağımsız bir
    yfinance çağrısı kullanıyoruz.
    """
    import yfinance as yf

    if not symbols:
        return {}

    yf_symbols = []
    reverse_map: dict[str, str] = {}
    for s in symbols:
        s = s.strip().upper()
        yf_symbol = f"{s}.IS" if len(s) in (4, 5) and s.isalpha() else s
        yf_symbols.append(yf_symbol)
        reverse_map[yf_symbol] = s

    prices: dict[str, float] = {}
    try:
        data = yf.download(tickers=yf_symbols, period="5d", interval="1d", auto_adjust=False, progress=False)
        if not data.empty:
            close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
            if not isinstance(close, pd.DataFrame):
                close = close.to_frame()
            if len(yf_symbols) == 1 and close.shape[1] == 1:
                close.columns = yf_symbols
            for ys in yf_symbols:
                if ys in close.columns:
                    series = close[ys].dropna()
                    if not series.empty:
                        prices[reverse_map[ys]] = float(series.iloc[-1])
    except Exception:
        pass  # aşağıdaki tekli fallback devreye girer

    # Toplu indirme başarısız olan/eksik kalan semboller için tek tek dene.
    missing = [s for s in symbols if s.upper() not in prices]
    for s in missing:
        ys = f"{s.upper()}.IS" if len(s) in (4, 5) and s.isalpha() else s.upper()
        try:
            hist = yf.Ticker(ys).history(period="5d")
            if not hist.empty:
                prices[s.upper()] = float(hist["Close"].iloc[-1])
        except Exception:
            continue

    return prices
