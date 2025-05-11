import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from conversation_utils import (
    get_recent_qa_pairs,
    compress_conversation_context,
    expand_user_question,
    create_final_prompt
)

from chat_prompt import create_prompt
from model_utils import query_gemini

# === Sabit Ayarlar ===
INDEX_DIR = "chroma_index"
EMBED_MODEL = "all-MiniLM-L6-v2"
CATEGORIES = ["strategy_act", "ethic"]

# === Streamlit Başlangıç ===
st.set_page_config(page_title="RAGemini", layout="wide")
st.title("📚 RAGemini Scholar ")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Kullanıcıdan kategori seçimi ===
selected_category = st.sidebar.selectbox("Kategori Seçin", CATEGORIES)

# === Soru girişi ===
st.subheader("Soru Sor")
user_query = st.text_input("Sormak istediğiniz soruyu yazın:")

if st.button("Cevapla") and user_query:
    # === Embed modeli yükle
    emb_model = SentenceTransformer(EMBED_MODEL)

    # === FAISS ve metadata dosyalarını yükle
    index_path = f"{INDEX_DIR}/{selected_category}_index.faiss"
    meta_path = f"{INDEX_DIR}/{selected_category}_meta.pkl"

    try:
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
    except Exception as e:
        st.error(f"Hata: Index veya metadata yüklenemedi. ({e})")
        st.stop()

    # === Soruya uygun chunk'ları getir
    query_vector = emb_model.encode([user_query])
    distances, indices = index.search(np.array(query_vector), k=3)

    retrieved_chunks = []
    for i in indices[0]:
        retrieved_chunks.append(metadata[i])

    # === Eğer geçmiş yoksa create_prompt, varsa compress-expand
    if not st.session_state.chat_history:
        final_prompt = create_prompt(user_query, retrieved_chunks, [])
    else:
        qa_pairs = get_recent_qa_pairs(st.session_state.chat_history)
        compressed_context = compress_conversation_context(qa_pairs, st.secrets["AIzaSyCBGYO7v1Sf7dsXey4mNtNs-u06ljxL1UM"])
        expanded_question = expand_user_question(user_query, compressed_context, st.secrets["AIzaSyCBGYO7v1Sf7dsXey4mNtNs-u06ljxL1UM"])
        final_prompt = create_final_prompt(compressed_context, expanded_question, retrieved_chunks)

    # === Prompt'u göster
    with st.expander("🧾 Üretilen Prompt"):
        st.code(final_prompt)

    # === Gemini'den yanıt al
    api_key = st.secrets["API_KEY"]
    gemini_response = query_gemini(final_prompt, api_key)

    answer = ""
    for candidate in gemini_response.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            answer += part.get("text", "")

    # === Sonucu ekrana yaz ve geçmişe ekle
    st.subheader("🧠 Cevap")
    st.write(answer)
    st.session_state.chat_history.append({"question": user_query, "answer": answer})

# === Sohbet Geçmişi Gösterimi ===
if st.session_state.chat_history:
    st.subheader("📜 Sohbet Geçmişi")
    for msg in st.session_state.chat_history:
        st.markdown(f"**Soru:** {msg['question']}")
        st.markdown(f"**Cevap:** {msg['answer']}")
        st.markdown("---")
