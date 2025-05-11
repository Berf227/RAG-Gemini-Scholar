import re
import zipfile
from io import BytesIO
import PyPDF2
import streamlit as st

def pdf_to_text(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
    except PyPDF2.errors.PdfReadError as e:
        st.error(f"Error reading {pdf_file.name}: {e}")
        return ""
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text
    return text

def clean_text(text):
    return re.sub(r'\s+', ' ', text)

def split_into_chunks(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_pdf_files(uploaded_files):
    pdf_files = []
    for file in uploaded_files:
        # Eğer doğrudan PDF ise
        if file.name.lower().endswith(".pdf"):
            pdf_files.append(file)
        
        # Eğer ZIP ise
        elif file.name.lower().endswith(".zip"):
            with zipfile.ZipFile(file) as z:
                for filename in z.namelist():
                    # MacOS metadata dosyalarını atla
                    # (ör. __MACOSX klasörü veya "._" ile başlayan dosyalar)
                    if filename.startswith("__MACOSX/"):
                        continue
                    if filename.startswith("._"):
                        continue

                    if filename.lower().endswith(".pdf"):
                        pdf_bytes = z.read(filename)
                        # Boş dosyaları atla
                        if len(pdf_bytes) == 0:
                            continue
                        
                        # PDF'i bellek üstünde temsil eden BytesIO nesnesi
                        pdf_file = BytesIO(pdf_bytes)
                        pdf_file.name = filename
                        pdf_files.append(pdf_file)
    return pdf_files

