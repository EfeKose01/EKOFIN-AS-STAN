FROM python:3.11-slim

WORKDIR /app

# OCR (pytesseract) ve OpenCV için gerekli sistem paketleri.
# NOT: packages.txt sadece Streamlit Community Cloud tarafından okunur;
# Railway/Docker dağıtımında sistem paketleri buraya elle eklenmeli.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-tur \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# NOT: Ana giriş noktası artık streamlit_app.py — "Sohbet" (herkese açık) ve
# "myPortfolio" (girişli) sayfaları arasındaki yönlendirmeyi bu dosya kurar.
# app_finetune_rag.py'ı doğrudan çalıştırmak myPortfolio'yu ATLAR.
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]
