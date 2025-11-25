# create_index.py - GÜNCELLENMİŞ SÜRÜM
import os
import json
import pickle
from glob import glob
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict

# --- YENİ EKLENEN: Scraper Modülü ---
import scraper  # scraper.py dosyasını import ediyoruz

# --- Yapılandırma ---
SOURCE_FILES_GLOB = [
    "jsonl_out/*.jsonl",  # DÜZELTME: Klasörün içindeki .jsonl dosyaları
    "datalar_extracted/**/*.md",
    "datalar_extracted/**/*.txt"
]
FAISS_INDEX_PATH = "rag_index.faiss"
CONTENT_MAP_PATH = "rag_content.pkl"
EMBEDDING_MODEL = 'paraphrase-multilingual-mpnet-base-v2'
MAX_CHUNK_CHARS = 1000
OVERLAP = 150


def read_and_chunk_files() -> List[Dict[str, str]]:
    """Tüm kaynak dosyaları okur ve metin parçalarına (chunk) ayırır."""
    chunks = []
    paths = []
    # Glob desenlerini genişlet
    for pattern in SOURCE_FILES_GLOB:
        paths.extend(glob(pattern, recursive=True))

    print(f"📂 Taranacak dosya sayısı: {len(paths)}")

    for path in paths:
        # print(f"-> İşleniyor: {path}") # Çok dosya varsa konsolu kirletmesin diye kapattım
        try:
            full_text = ""
            if path.endswith('.jsonl'):
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        data = json.loads(line)
                        full_text += data.get('content', '') + "\n"
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    full_text = f.read()

            if not full_text.strip(): continue

            start = 0
            while start < len(full_text):
                end = start + MAX_CHUNK_CHARS
                chunk_text = full_text[start:end]
                if chunk_text.strip():
                    # Metadata için dosya yolunu da ekliyoruz
                    chunks.append({
                        "path": path,
                        "text": chunk_text,
                        "tokens": chunk_text.lower().split()
                    })
                start += MAX_CHUNK_CHARS - OVERLAP
        except Exception as e:
            print(f"HATA: {path} dosyası işlenemedi - {e}")
            continue
    return chunks


def build_and_save_index():
    # 1. ADIM: Önce yeni verileri kontrol et ve indir
    print("🌍 Resmi Gazete güncellemeleri kontrol ediliyor...")
    scraper.fetch_daily_resmi_gazete()
    print("------------------------------------------------")

    # 2. ADIM: İndeksleme
    print("Anlamsal indeksleme başlıyor...")
    content_chunks = read_and_chunk_files()
    if not content_chunks:
        print("UYARI: İndekslenecek içerik bulunamadı.")
        return

    print(f"📦 Toplam {len(content_chunks)} adet metin parçası (chunk) işleniyor.")

    print(f"🧠 '{EMBEDDING_MODEL}' modeli yükleniyor...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("🔢 Metinler vektörlere dönüştürülüyor (Bu işlem biraz sürebilir)...")
    texts_to_encode = [chunk["text"] for chunk in content_chunks]
    embeddings = model.encode(texts_to_encode, show_progress_bar=True)

    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(np.array(embeddings).astype('float32'))

    print(f"💾 '{FAISS_INDEX_PATH}' dosyasına kaydediliyor...")
    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"💾 '{CONTENT_MAP_PATH}' dosyasına kaydediliyor...")
    with open(CONTENT_MAP_PATH, 'wb') as f:
        pickle.dump(content_chunks, f)

    print("\n✅ SİSTEM GÜNCELLENDİ VE İNDEKS BAŞARIYLA OLUŞTURULDU!")


if __name__ == "__main__":
    build_and_save_index()