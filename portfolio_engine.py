# portfolio_engine.py
# EkoFin Portföy Analizi Motoru
# Kullanıcının sembol + ağırlık bilgisine göre portföy zaman serisi ve risk analizi üretir.

from typing import Dict, Any, Tuple
import yfinance as yf
import pandas as pd
import numpy as np


def parse_portfolio_text(text: str) -> Dict[str, float]:
    """
    Kullanıcıdan gelen portföy tanımı:
    Örn:
        GARAN:0.3
        AKBNK:0.4
        THYAO:0.3

    veya tek satır:
        GARAN=0.3, AKBNK=0.4, THYAO=0.3

    Çıktı: { "GARAN": 0.3, "AKBNK": 0.4, "THYAO": 0.3 }
    """
    lines = text.replace(",", "\n").splitlines()
    weights = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            sym, w = line.split(":", 1)
        elif "=" in line:
            sym, w = line.split("=", 1)
        else:
            continue
        try:
            w_val = float(w.strip())
        except ValueError:
            continue
        sym_clean = sym.strip().upper()
        weights[sym_clean] = w_val

    total = sum(weights.values())
    if total <= 0:
        return {}

    # normalize to 1
    for k in list(weights.keys()):
        weights[k] = weights[k] / total

    return weights


def _map_to_yf_symbol(raw: str) -> str:
    if len(raw) in [4, 5] and "." not in raw and raw.isalpha():
        return raw + ".IS"
    return raw


def _download_price_matrix(symbols: Dict[str, float], period: str = "1y") -> pd.DataFrame:
    yf_symbols = [_map_to_yf_symbol(s) for s in symbols.keys()]
    data = yf.download(
        tickers=yf_symbols,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close_df = data["Close"].copy()
        else:
            return pd.DataFrame()
    else:
        close_df = data[["Close"]].copy()
        close_df.columns = [yf_symbols[0]]

    try:
        close_df.index = close_df.index.tz_localize(None)
    except Exception:
        pass

    # Kolon isimlerini tekrar orijinal sembollere çevir
    rename_map = {}
    for raw, yf_sym in zip(symbols.keys(), yf_symbols):
        if yf_sym in close_df.columns:
            rename_map[yf_sym] = raw
    close_df = close_df.rename(columns=rename_map)

    return close_df.dropna(how="all")


def portfolio_timeseries(weights: Dict[str, float], period: str = "1y") -> Tuple[pd.DataFrame, pd.Series]:
    """
    weights: {"GARAN": 0.3, "AKBNK": 0.4, "THYAO": 0.3}
    Dönen:
        prices_df: her sembolün fiyat serisi
        portfolio_index: normalize portföy endeksi (ilk gün = 100)
    """
    prices = _download_price_matrix(weights, period=period)
    if prices.empty:
        return prices, pd.Series(dtype=float)

    prices = prices[sorted(weights.keys())]  # kolon sırası sabit

    # log/normal getiriler
    returns = prices.pct_change().fillna(0.0)
    w_vec = np.array([weights[s] for s in prices.columns])
    port_ret = returns.values @ w_vec
    port_series = (1 + pd.Series(port_ret, index=prices.index)).cumprod() * 100.0

    return prices, port_series


def portfolio_summary(weights: Dict[str, float], period: str = "1y") -> Dict[str, Any]:
    prices, port_index = portfolio_timeseries(weights, period=period)
    if prices.empty or port_index.empty:
        return {"hata": "Portföy analizi için yeterli veri bulunamadı."}

    # risk metrikleri
    daily_ret = port_index.pct_change().dropna()
    if daily_ret.empty:
        return {"hata": "Portföy getirisi hesaplanamadı."}

    daily_vol = float(daily_ret.std())
    annual_vol = daily_vol * (252 ** 0.5)
    avg_daily = float(daily_ret.mean())
    annual_ret = avg_daily * 252

    cum_ret = float(port_index.iloc[-1] / port_index.iloc[0] - 1.0)

    # max drawdown
    cum_max = port_index.cummax()
    drawdown = (port_index / cum_max) - 1.0
    max_dd = float(drawdown.min())

    if annual_vol > 0:
        sharpe_like = annual_ret / annual_vol
    else:
        sharpe_like = None

    contribs = {}
    for col in prices.columns:
        col_ret = prices[col].pct_change().dropna()
        if col_ret.empty:
            continue
        contribs[col] = float(col_ret.corr(daily_ret))

    return {
        "agirliklar": weights,
        "toplam_donem_getirisi_yuzde": round(cum_ret * 100, 2),
        "yillik_getiri_tahmini_yuzde": round(annual_ret * 100, 2),
        "yillik_volatilite_yuzde": round(annual_vol * 100, 2),
        "max_drawdown_yuzde": round(max_dd * 100, 2),
        "sharpe_benzeri_oran": round(sharpe_like, 2) if sharpe_like is not None else None,
        "sembol_korelasyon_katkisi": contribs,
        "tarih_araligi": {
            "ilk_tarih": port_index.index[0].strftime("%Y-%m-%d"),
            "son_tarih": port_index.index[-1].strftime("%Y-%m-%d"),
        },
    }
