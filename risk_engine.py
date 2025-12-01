# risk_engine.py
# EkoFin Risk Analizi Motoru
# Volatilite, max drawdown, basit Sharpe benzeri metrikler hesaplar.

from typing import Dict, Any
import yfinance as yf
import pandas as pd
import numpy as np


def _fetch_close_series(yf_symbol: str, period: str = "1y") -> pd.Series:
    data = yf.download(
        tickers=yf_symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if data is None or data.empty:
        return pd.Series(dtype=float)
    try:
        data.index = data.index.tz_localize(None)
    except Exception:
        pass
    return data["Close"].dropna()


def _max_drawdown(series: pd.Series) -> float:
    cum_max = series.cummax()
    drawdown = (series / cum_max) - 1.0
    return float(drawdown.min())


def risk_snapshot(symbol: str) -> Dict[str, Any]:
    """
    Tek sembol için risk metrikleri:
    - yıllık volatilite
    - max drawdown
    - son 1 yıllık basit Sharpe benzeri oran
    """
    raw = symbol.strip().upper()
    if len(raw) in [4, 5] and "." not in raw and raw.isalpha():
        yf_symbol = raw + ".IS"
        display_symbol = raw
    else:
        yf_symbol = raw
        display_symbol = raw

    close = _fetch_close_series(yf_symbol, period="1y")
    if close.empty or close.shape[0] < 60:
        return {
            "sembol": display_symbol,
            "hata": "Risk analizi için yeterli tarihsel veri yok."
        }

    returns = close.pct_change().dropna()

    daily_vol = float(returns.std())
    annual_vol = daily_vol * (252 ** 0.5)

    dd = _max_drawdown(close)

    avg_daily_ret = float(returns.mean())
    annual_ret = avg_daily_ret * 252

    if annual_vol > 0:
        sharpe_like = annual_ret / annual_vol
    else:
        sharpe_like = None

    return {
        "sembol": display_symbol,
        "yillik_volatilite": round(annual_vol * 100, 2),
        "max_drawdown_yuzde": round(dd * 100, 2),
        "tahmini_yillik_getiri_yuzde": round(annual_ret * 100, 2),
        "sharpe_benzeri_oran": round(sharpe_like, 2) if sharpe_like is not None else None,
    }
