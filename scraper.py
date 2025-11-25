# scraper.py - Resmi Gazete ve BDDK Veri Çekici
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import io
import PyPDF2  # PDF okumak için gerekli: pip install pypdf

# Verilerin kaydedileceği klasör (Senin yapına uygun)
SAVE_DIR = "datalar_extracted/resmi_gazete_guncel"
os.makedirs(SAVE_DIR, exist_ok=True)

# Takip edilecek anahtar kelimeler
KEYWORDS = ["BDDK", "Bankacılık", "Kredi", "Mevduat", "Faiz", "Kart", "Finansal"]


def extract_text_from_pdf(pdf_content):
    """PDF içeriğini metne çevirir."""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"[PDF Okuma Hatası: {e}]"


def fetch_daily_resmi_gazete():
    """Bugünün Resmi Gazetesini tarar ve ilgili verileri indirir."""
    today = datetime.now()
    date_str = today.strftime("%Y%m%d")
    year = today.strftime("%Y")
    month = today.strftime("%m")

    url = f"https://www.resmigazete.gov.tr/eskiler/{year}/{month}/{date_str}.htm"
    print(f"🌍 Resmi Gazete taranıyor: {url}")

    # --- DÜZELTME BURADA ---
    # 1. Tarayıcı gibi görünmek için Header ekliyoruz
    # 2. Timeout süresini 30 saniyeye çıkarıyoruz
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code != 200:
            print(f"⚠️ Bugünün Resmi Gazetesi henüz yayınlanmamış veya erişilemiyor (Kod: {response.status_code})")
            return False

        soup = BeautifulSoup(response.content, "html.parser")
        links = soup.find_all("a")

        found_count = 0

        for link in links:
            title = link.text.strip()
            href = link.get("href")

            if not title or not href:
                continue

            # Anahtar kelime kontrolü
            if any(k.lower() in title.lower() for k in KEYWORDS):
                print(f"✅ İlgili Mevzuat Bulundu: {title}")

                # Linki tamamlama
                if not href.startswith("http"):
                    # Resmi Gazete link yapısına göre düzeltme
                    full_link = f"https://www.resmigazete.gov.tr/eskiler/{year}/{month}/{href}"
                else:
                    full_link = href

                # İçeriği İndirme
                try:
                    content_resp = requests.get(full_link, timeout=15)
                    content_text = ""

                    if full_link.endswith(".pdf"):
                        print("   📄 PDF indiriliyor ve işleniyor...")
                        content_text = extract_text_from_pdf(content_resp.content)
                    else:
                        print("   🌐 HTML içeriği alınıyor...")
                        sub_soup = BeautifulSoup(content_resp.content, "html.parser")
                        # Sadece metin içeriğini al (Javascript vs temizle)
                        content_text = sub_soup.get_text(separator="\n")

                    # Dosyayı Kaydetme
                    # Dosya ismini güvenli hale getir
                    safe_title = "".join([c if c.isalnum() else "_" for c in title])[:50]
                    file_name = f"{date_str}_{safe_title}.txt"
                    file_path = os.path.join(SAVE_DIR, file_name)

                    # Dosya başına Metadata ekleyerek kaydediyoruz (RAG için çok önemli)
                    final_content = f"BAŞLIK: {title}\nKAYNAK: Resmi Gazete\nTARİH: {today.strftime('%d.%m.%Y')}\nLİNK: {full_link}\n\n{content_text}"

                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(final_content)

                    print(f"   💾 Kaydedildi: {file_path}")
                    found_count += 1

                except Exception as e:
                    print(f"   ❌ İçerik indirilemedi: {e}")

        if found_count == 0:
            print("ℹ️ Bugün finans/bankacılık ile ilgili bir karar bulunamadı.")
            return False
        else:
            print(f"🚀 Toplam {found_count} yeni mevzuat sisteme eklendi.")
            return True

    except Exception as e:
        print(f"Genel Hata: {e}")
        return False


if __name__ == "__main__":
    fetch_daily_resmi_gazete()