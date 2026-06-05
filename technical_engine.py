# technical_engine.py — EkoFin Teknik Analiz Motoru
# Göstergeler: MA, EMA, RSI, MACD, Bollinger Bands, ATR, Supertrend,
#               Hacim (OBV), Pivot Destek/Direnç

from typing import Dict, Any, Optional
import yfinance as yf
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
#  VERİ ÇEKME
# ─────────────────────────────────────────────

def _fetch_price_data(yf_symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(
        tickers=yf_symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()
    # MultiIndex sütunları düzleştir (tek sembol bile MultiIndex gelebilir)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    try:
        data.index = data.index.tz_localize(None)
    except Exception:
        pass
    return data.dropna()


# ─────────────────────────────────────────────
#  GÖSTERGE HESAPLAMA
# ─────────────────────────────────────────────

def _ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def _atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = data["High"], data["Low"], data["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Returns (upper, middle, lower, bandwidth_pct)."""
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    bandwidth = ((upper - lower) / middle) * 100
    return upper, middle, lower, bandwidth


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume — hacim tabanlı trend doğrulayıcı."""
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def _supertrend(data: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    high, low, close = data["High"], data["Low"], data["Close"]
    atr = _atr(data, period)
    hl2 = (high + low) / 2.0

    upper_arr = (hl2 + multiplier * atr).to_numpy(dtype=float)
    lower_arr = (hl2 - multiplier * atr).to_numpy(dtype=float)
    close_arr = close.to_numpy(dtype=float)
    n = len(close_arr)

    for i in range(1, n):
        if close_arr[i - 1] <= upper_arr[i - 1]:
            upper_arr[i] = min(upper_arr[i], upper_arr[i - 1])
        if close_arr[i - 1] >= lower_arr[i - 1]:
            lower_arr[i] = max(lower_arr[i], lower_arr[i - 1])

    direction = np.ones(n, dtype=bool)
    st_arr = np.empty(n, dtype=float)
    direction[0] = close_arr[0] >= lower_arr[0]
    st_arr[0] = lower_arr[0] if direction[0] else upper_arr[0]

    for i in range(1, n):
        if close_arr[i] > upper_arr[i - 1]:
            direction[i] = True
        elif close_arr[i] < lower_arr[i - 1]:
            direction[i] = False
        else:
            direction[i] = direction[i - 1]
        st_arr[i] = lower_arr[i] if direction[i] else upper_arr[i]

    return pd.Series(st_arr, index=close.index, dtype=float)


def _pivot_levels(data: pd.DataFrame, lookback: int = 20):
    """
    Son `lookback` günün high/low'undan basit destek ve direnç seviyeleri.
    Yöntem: rolling window içindeki swing high ve swing low'lar.
    """
    recent = data.tail(lookback)
    support  = float(recent["Low"].min())
    resistance = float(recent["High"].max())
    pivot = (float(recent["High"].iloc[-1]) + float(recent["Low"].iloc[-1]) + float(recent["Close"].iloc[-1])) / 3
    s1 = 2 * pivot - float(recent["High"].iloc[-1])
    r1 = 2 * pivot - float(recent["Low"].iloc[-1])
    return {
        "destek_guclu": round(support, 2),
        "destek_s1":    round(s1, 2),
        "pivot":        round(pivot, 2),
        "direnc_r1":    round(r1, 2),
        "direnc_guclu": round(resistance, 2),
    }


def _rsi_label(rsi_val: float) -> str:
    if rsi_val >= 70:
        return "aşırı alım bölgesi"
    if rsi_val <= 30:
        return "aşırı satım bölgesi"
    if rsi_val >= 55:
        return "güçlü alan"
    if rsi_val <= 45:
        return "zayıf alan"
    return "nötr"


def _bb_position(close_val: float, upper: float, lower: float, middle: float) -> str:
    if close_val >= upper:
        return "üst banda yakın (aşırı alım sinyali olabilir)"
    if close_val <= lower:
        return "alt banda yakın (aşırı satım sinyali olabilir)"
    if close_val > middle:
        return "orta bandın üzerinde"
    return "orta bandın altında"


# ─────────────────────────────────────────────
#  ANA FONKSİYON
# ─────────────────────────────────────────────

def compute_technical_snapshot(symbol: str, period: str = "1y") -> Dict[str, Any]:
    """
    Tek sembol için kapsamlı teknik analiz JSON'u.

    Göstergeler:
      Trend    : MA20/50/200, EMA20/50, Supertrend
      Momentum : RSI14, MACD(12,26,9)
      Volatilite: ATR14, Bollinger Bands(20,2)
      Hacim    : OBV yönü
      Fiyat yapısı: Pivot destek/direnç (son 20 gün)
    """
    raw = symbol.strip().upper()
    if len(raw) in [4, 5] and "." not in raw and raw.isalpha():
        yf_symbol = raw + ".IS"
        display_symbol = raw
    else:
        yf_symbol = raw
        display_symbol = raw

    data = _fetch_price_data(yf_symbol, period=period, interval="1d")
    if data.empty or len(data) < 30:
        return {
            "sembol": display_symbol,
            "hata": "Teknik analiz için yeterli veri bulunamadı (min. 30 gün gerekli)."
        }

    close  = data["Close"]
    volume = data.get("Volume", pd.Series(dtype=float))

    # ── Trend ──
    ma20  = close.rolling(20).mean()
    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    st    = _supertrend(data, period=10, multiplier=3.0)

    # ── Momentum ──
    rsi14 = _rsi(close, 14)
    macd_line, signal_line, histogram = _macd(close)

    # ── Volatilite ──
    atr14        = _atr(data, 14)
    bb_up, bb_mid, bb_low, bb_bw = _bollinger_bands(close, 20, 2.0)

    # ── Hacim ──
    has_volume = not volume.empty and volume.sum() > 0
    obv_trend  = None
    if has_volume:
        obv       = _obv(close, volume)
        obv_slope = float(obv.iloc[-1]) - float(obv.iloc[-10]) if len(obv) >= 10 else 0
        obv_trend = "yükselen" if obv_slope > 0 else "düşen"

    # ── Destek/Direnç ──
    pivots = _pivot_levels(data, lookback=20)

    # ── Son değerler ──
    def _f(s) -> Optional[float]:
        v = s.iloc[-1]
        return round(float(v), 4) if pd.notna(v) else None

    last        = _f(close)
    last_ma20   = _f(ma20)
    last_ma50   = _f(ma50)
    last_ma200  = _f(ma200)
    last_ema20  = _f(ema20)
    last_ema50  = _f(ema50)
    last_atr    = _f(atr14)
    last_st     = _f(st)
    last_rsi    = _f(rsi14)
    last_macd   = _f(macd_line)
    last_signal = _f(signal_line)
    last_hist   = _f(histogram)
    last_bb_up  = _f(bb_up)
    last_bb_mid = _f(bb_mid)
    last_bb_low = _f(bb_low)
    last_bb_bw  = _f(bb_bw)

    # ── Trend etiketi ──
    trend_label = "belirsiz"
    if last and last_ma20 and last_ma50:
        if last > last_ma20 > last_ma50:
            trend_label = "kısa vadeli yükselen"
            if last_ma200 and last_ma50 > last_ma200:
                trend_label = "güçlü yükselen trend"
        elif last < last_ma20 < last_ma50:
            trend_label = "kısa vadeli düşen"
            if last_ma200 and last_ma50 < last_ma200:
                trend_label = "güçlü düşen trend"

    st_direction = "yukarı trend" if (last_st and last and last > last_st) else "aşağı trend"

    # ── MACD yorumu ──
    macd_yorum = "nötr"
    if last_macd is not None and last_signal is not None:
        if last_macd > last_signal and last_hist and last_hist > 0:
            macd_yorum = "yükselen momentum (MACD sinyal üzerinde)"
        elif last_macd < last_signal and last_hist and last_hist < 0:
            macd_yorum = "düşen momentum (MACD sinyal altında)"

    return {
        "sembol": display_symbol,
        "analiz_donemi": period,
        "son_fiyat": round(last, 2) if last else None,

        # Trend
        "trend_ozet": trend_label,
        "supertrend_yonu": st_direction,
        "supertrend_degeri": last_st,
        "ma20":  last_ma20,
        "ma50":  last_ma50,
        "ma200": last_ma200,
        "ema20": round(last_ema20, 2) if last_ema20 else None,
        "ema50": round(last_ema50, 2) if last_ema50 else None,

        # Momentum
        "rsi14": round(last_rsi, 1) if last_rsi else None,
        "rsi_yorum": _rsi_label(last_rsi) if last_rsi else None,
        "macd_yorum": macd_yorum,
        "macd_deger": round(last_macd, 4) if last_macd else None,
        "macd_sinyal": round(last_signal, 4) if last_signal else None,
        "macd_histogram": round(last_hist, 4) if last_hist else None,

        # Volatilite
        "atr14": last_atr,
        "bollinger_ust": round(last_bb_up, 2) if last_bb_up else None,
        "bollinger_orta": round(last_bb_mid, 2) if last_bb_mid else None,
        "bollinger_alt": round(last_bb_low, 2) if last_bb_low else None,
        "bollinger_bant_genisligi_yuzde": round(last_bb_bw, 2) if last_bb_bw else None,
        "bollinger_pozisyon": _bb_position(last, last_bb_up, last_bb_low, last_bb_mid) if all([last, last_bb_up, last_bb_low, last_bb_mid]) else None,

        # Hacim
        "obv_trend": obv_trend,

        # Destek / Direnç
        "destek_direnc": pivots,
    }
