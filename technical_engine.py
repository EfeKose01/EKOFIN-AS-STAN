# technical_engine.py
# EkoFin Teknik Analiz Motoru
# Yfinance verisinden MA, EMA, ATR, Supertrend hesaplar.

from typing import Dict, Any
import yfinance as yf
import pandas as pd
import numpy as np


def _fetch_price_data(yf_symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(
        tickers=yf_symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()
    try:
        data.index = data.index.tz_localize(None)
    except Exception:
        pass
    return data.dropna()


def _ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def _atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def _supertrend(data: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """
    Basitleştirilmiş Supertrend hesaplaması.
    Tam tradingview kopyası değil ama yön + ortalama trendi verir.
    """
    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    atr = _atr(data, period)
    hl2 = (high + low) / 2.0

    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()

    for i in range(1, len(close)):
        if close.iloc[i - 1] <= upper_band.iloc[i - 1]:
            upper_band.iloc[i] = min(upper_basic.iloc[i], upper_band.iloc[i - 1])
        else:
            upper_band.iloc[i] = upper_basic.iloc[i]

        if close.iloc[i - 1] >= lower_band.iloc[i - 1]:
            lower_band.iloc[i] = max(lower_basic.iloc[i], lower_band.iloc[i - 1])
        else:
            lower_band.iloc[i] = lower_basic.iloc[i]

    supertrend = pd.Series(index=close.index, dtype=float)
    direction = True  # True = uptrend, False = downtrend

    for i in range(len(close)):
        if i == 0:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction = close.iloc[i] >= supertrend.iloc[i]
        else:
            if close.iloc[i] > upper_band.iloc[i - 1]:
                direction = True
            elif close.iloc[i] < lower_band.iloc[i - 1]:
                direction = False

            if direction:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]

    return supertrend


def compute_technical_snapshot(symbol: str) -> Dict[str, Any]:
    """
    Kullanıcıya tek sembol için özet teknik analiz JSON'u döner.
    - MA20, MA50, MA200
    - EMA20, EMA50
    - ATR14
    - Supertrend son değeri ve yönü
    """
    # BIST kısaları için .IS ekleyelim
    raw = symbol.strip().upper()
    if len(raw) in [4, 5] and "." not in raw and raw.isalpha():
        yf_symbol = raw + ".IS"
        display_symbol = raw
    else:
        yf_symbol = raw
        display_symbol = raw

    data = _fetch_price_data(yf_symbol, period="6mo", interval="1d")
    if data.empty:
        return {
            "sembol": display_symbol,
            "hata": "Bu sembol için teknik analiz yapacak kadar veri bulunamadı."
        }

    close = data["Close"]

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)

    atr14 = _atr(data, 14)
    st = _supertrend(data, period=10, multiplier=3.0)

    last = close.iloc[-1]
    last_ma20 = ma20.iloc[-1]
    last_ma50 = ma50.iloc[-1]
    last_ma200 = ma200.iloc[-1]
    last_ema20 = ema20.iloc[-1]
    last_ema50 = ema50.iloc[-1]
    last_atr = atr14.iloc[-1]
    last_st = st.iloc[-1]

    trend_label = "belirsiz"
    trend_score = 0
    if last > last_ma20 > last_ma50 > last_ma200:
        trend_label = "yükselen trend"
        trend_score = 1
    elif last < last_ma20 < last_ma50 < last_ma200:
        trend_label = "düşen trend"
        trend_score = -1

    st_direction = "yukarı trend"
    if last < last_st:
        st_direction = "aşağı trend"

    return {
        "sembol": display_symbol,
        "son_fiyat": round(float(last), 2),
        "ma20": round(float(last_ma20), 2) if pd.notna(last_ma20) else None,
        "ma50": round(float(last_ma50), 2) if pd.notna(last_ma50) else None,
        "ma200": round(float(last_ma200), 2) if pd.notna(last_ma200) else None,
        "ema20": round(float(last_ema20), 2) if pd.notna(last_ema20) else None,
        "ema50": round(float(last_ema50), 2) if pd.notna(last_ema50) else None,
        "atr14": round(float(last_atr), 4) if pd.notna(last_atr) else None,
        "supertrend": round(float(last_st), 2) if pd.notna(last_st) else None,
        "supertrend_yonu": st_direction,
        "trend_label": trend_label,
        "trend_score": trend_score,
    }
