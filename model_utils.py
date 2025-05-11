import numpy as np
import faiss
import time
import os
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from pdf_utils import pdf_to_text, clean_text, split_into_chunks

@st.cache_resource
def load_model(model_name: str):
    # Transformer’ı tamamen offline moda al
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Local cache klasöründeki model yolunu belirle
    local_path = f"model_cache/{model_name.split('/')[-1]}"
    return SentenceTransformer(
        local_path,
        cache_folder="model_cache",
        local_files_only=True
    )

def build_combined_index(uploaded_files, model_name="all-MiniLM-L6-v2", chunk_size=500):
    model_emb = load_model(model_name)
    all_chunks = []
    all_metadata = []
    
    for pdf_file in uploaded_files:
        st.info(f"**Reading file:** {pdf_file.name}")
        text = pdf_to_text(pdf_file)
        if not text:
            continue
        st.write(f"- Extracted **{len(text)} characters** from {pdf_file.name}.")
        st.write(f"- Cleaning text for {pdf_file.name}...")
        cleaned = clean_text(text)
        st.write(f"- Text length after cleaning: **{len(cleaned)} characters**")
        st.write(f"- Splitting into chunks of {chunk_size} words...")
        chunks = split_into_chunks(cleaned, chunk_size=chunk_size)
        st.write(f"- Created **{len(chunks)} chunks**.")
        
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "pdf": pdf_file.name,
                "chunk": chunk
            })
    
    st.write(f"- Total chunks from all PDFs: **{len(all_chunks)}**")
    st.write(f"- Generating embeddings for all chunks...")
    embeddings = model_emb.encode(all_chunks)
    if len(embeddings.shape) == 1:
        embeddings = np.expand_dims(embeddings, axis=0)
    st.write(f"- Embedding shape: **{embeddings.shape}**")
    
    st.write(f"- Building combined FAISS index...")
    dimension = embeddings.shape[1]
    combined_index = faiss.IndexFlatL2(dimension)
    combined_index.add(np.array(embeddings))
    st.success(f"- Combined FAISS index built for all PDFs.")
    
    return model_emb, combined_index, all_metadata

def retrieve_relevant_chunks_combined(query, model_emb, combined_index, all_metadata, top_k=3):
    query_embedding = model_emb.encode([query])
    distances, idxs = combined_index.search(np.array(query_embedding), top_k)
    results = []
    for distance, idx in zip(distances[0], idxs[0]):
        metadata = all_metadata[idx]
        results.append({
            "pdf": metadata["pdf"],
            "chunk": metadata["chunk"],
            "distance": distance
        })
    results = sorted(results, key=lambda x: x["distance"])
    return results


def query_gemini(prompt, api_key, show_time=False):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    start_time = time.time()
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        elapsed = time.time() - start_time
        if show_time:
            st.info(f"API call took **{elapsed:.2f} seconds**.")
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request error: {e}")
        return {}
