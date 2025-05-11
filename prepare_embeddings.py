import os
import glob
import pickle
import faiss
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# === CONFIG ===
CATEGORIES = ["strategy_act", "ethic"]
PDF_DIR = "data"
INDEX_DIR = "chroma_index"
CHUNK_SIZE = 500  # kelime sayısı
EMBED_MODEL = "all-MiniLM-L6-v2"

def pdf_to_text(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_height = page.rect.height
            margin = 50  # Üst ve alt kenar boşluğu
            clip_rect = fitz.Rect(0, margin, page.rect.width, page_height - margin)
            
            # Doğrudan metin çıkarma denemesi
            page_text = page.get_text(clip=clip_rect)
            
            # Eğer metin boşsa OCR kullan
            if not page_text.strip():
                tp = page.get_textpage_ocr(clip=clip_rect)
                page_text = page.get_text(textpage=tp)
            
            text += page_text + " "
        
        return text
    except Exception as e:
        print(f"[HATA] {file_path}: {e}")
        return ""

def clean_text(text):
    import re
    return re.sub(r"\s+", " ", text).strip()

def split_into_chunks(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def process_category(category):
    print(f"\n📂 İşleniyor: {category}")
    folder_path = os.path.join(PDF_DIR, category)
    files = glob.glob(os.path.join(folder_path, "*.pdf"))

    if not files:
        print("⚠️ PDF bulunamadı.")
        return

    model = SentenceTransformer(EMBED_MODEL)
    all_chunks = []
    all_metadata = []

    for file_path in files:
        print(f"📄 {os.path.basename(file_path)} okunuyor...")
        raw_text = pdf_to_text(file_path)
        if not raw_text:
            continue
        cleaned = clean_text(raw_text)
        chunks = split_into_chunks(cleaned, chunk_size=CHUNK_SIZE)
        print(f"  ➤ {len(chunks)} parça üretildi.")

        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "pdf": os.path.basename(file_path),
                "chunk": chunk,
                "order": idx
            })

    # Embedding
    print("🧠 Embedding hesaplanıyor...")
    embeddings = model.encode(all_chunks)
    if len(embeddings.shape) == 1:
        embeddings = np.expand_dims(embeddings, axis=0)

    # FAISS index oluştur
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    # Kayıt et
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, f"{category}_index.faiss"))

    with open(os.path.join(INDEX_DIR, f"{category}_meta.pkl"), "wb") as f:
        pickle.dump(all_metadata, f)

    print(f"✅ {category} kategorisi için index oluşturuldu ({len(all_chunks)} chunk)")

if __name__ == "__main__":
    for cat in CATEGORIES:
        process_category(cat)