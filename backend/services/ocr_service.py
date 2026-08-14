"""OCR pipeline TASLAĞI — aracı kurum portföy ekran görüntüsünden satır satır veri çıkarımı.

ÖNEMLİ (taslak/draft niteliği): Aracı kurum uygulamalarının ekran tasarımları
birbirinden çok farklı (sütun sırası, para birimi gösterimi, ondalık ayracı vb.)
olduğu için bu modül %100 doğruluk hedeflemez. Amaç, kullanıcının elle girmesi
gereken veriyi olabildiğince azaltmak; düşük güven skorlu veya hiç bulunamayan
satırlar için UI katmanı (pages_ui/portfolio_page.py) manuel düzeltme/onay
adımına düşürür. Bu yüzden her satır bir `confidence` skoruyla birlikte döner.

Bağımlılıklar: pytesseract (+ sistemde `tesseract-ocr` binary'si), opencv-python.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

import cv2
import numpy as np

from backend.schemas import OcrParsedRow

# --- Sayı / sembol ayrıştırma yardımcıları ---

_SYMBOL_RE = re.compile(r"\b[A-ZÇĞİÖŞÜ]{3,6}\b")
# "1.234,56" (TR) ya da "1,234.56" (US) ya da düz "1234.56" gibi biçimleri yakalar.
_NUMBER_RE = re.compile(r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|\d+[.,]\d+|\d+")


@lru_cache(maxsize=1)
def _load_bist_tickers() -> frozenset[str]:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "bist_tickers.txt")
    if not os.path.exists(path):
        return frozenset()
    with open(path, "r", encoding="utf-8") as f:
        return frozenset(line.strip().upper().replace(".IS", "") for line in f if line.strip())


def _parse_tr_number(raw: str) -> float | None:
    """'1.234,56' / '1,234.56' / '43,33' / '150' gibi biçimleri float'a çevirir."""
    raw = raw.strip()
    if not raw:
        return None
    # Hem nokta hem virgül varsa: sonuncusu ondalık ayraçtır, diğeri binlik.
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif "," in raw:
        # Tek virgül + 3 haneli sonrası varsa muhtemelen binlik ayracı, değilse ondalık.
        parts = raw.split(",")
        if len(parts) == 2 and len(parts[1]) == 3:
            raw = raw.replace(",", "")
        else:
            raw = raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


# --- Görüntü ön işleme ---

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Ekran görüntüsünü OCR için daha okunabilir hale getirir.

    Adımlar: gri tonlama -> gürültü azaltma -> adaptif eşikleme -> hafif büyütme.
    Broker ekranları genelde küçük fontlu ve renkli olduğundan bu adımlar
    Tesseract'ın karakter tanıma isabetini belirgin şekilde artırır.
    """
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Görüntü okunamadı; desteklenmeyen veya bozuk dosya.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    # Küçük fontları büyütmek OCR isabetini artırır.
    scaled = cv2.resize(thresh, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    return scaled


def extract_text(image_bytes: bytes) -> str:
    import pytesseract

    processed = preprocess_image(image_bytes)
    try:
        return pytesseract.image_to_string(processed, lang="tur+eng")
    except pytesseract.TesseractError:
        # "tur" dil paketi kurulu değilse İngilizce'ye düş.
        return pytesseract.image_to_string(processed, lang="eng")


# --- Satır ayrıştırma ---

def parse_portfolio_rows(text: str) -> list[OcrParsedRow]:
    """OCR çıktısındaki her satırı 'sembol + adet + maliyet' adayına çevirir."""
    tickers = _load_bist_tickers()
    rows: list[OcrParsedRow] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if len(line) < 3:
            continue

        symbol_matches = _SYMBOL_RE.findall(line.upper())
        if not symbol_matches:
            continue

        number_tokens = [_parse_tr_number(n) for n in _NUMBER_RE.findall(line)]
        number_tokens = [n for n in number_tokens if n is not None]
        if not number_tokens:
            # Satırda hiç sayı yoksa (başlık, açıklama vb.) portföy satırı olamaz.
            continue

        # Satırdaki en olası sembol: BIST listesinde olan varsa onu tercih et.
        symbol = next((s for s in symbol_matches if s in tickers), symbol_matches[0])

        quantity = number_tokens[0] if len(number_tokens) >= 1 else None
        cost_basis = number_tokens[1] if len(number_tokens) >= 2 else None

        confidence = 0.3
        if symbol in tickers:
            confidence += 0.4
        if quantity is not None and cost_basis is not None:
            confidence += 0.3
        confidence = min(confidence, 1.0)

        rows.append(
            OcrParsedRow(
                symbol=symbol,
                quantity=quantity,
                cost_basis=cost_basis,
                confidence=round(confidence, 2),
                raw_line=line,
            )
        )

    return rows


def run_ocr_pipeline(image_bytes: bytes) -> list[OcrParsedRow]:
    """Uçtan uca: görüntü -> ham metin -> ayrıştırılmış satır adayları.

    Dönen liste güven skoruna göre büyükten küçüğe sıralanır; UI katmanı
    düşük skorlu (örn. <0.7) satırları "gözden geçirin" olarak işaretlemeli.
    """
    text = extract_text(image_bytes)
    rows = parse_portfolio_rows(text)
    return sorted(rows, key=lambda r: r.confidence, reverse=True)
