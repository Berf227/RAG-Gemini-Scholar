# RAG‑Gemini

> **Retrieval‑Augmented Generation** (RAG) with Google’s Gemini LLM and custom PDF/document retrieval.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Project Structure](#Project-Structure)  
---

## Project Overview

This project provides a **Retrieval‑Augmented Generation** pipeline that takes a chat history and related documents, uses the **Gemini** model to generate comprehensive, self‑contained answers on the topic.

---

## Features

- Selects the last N QA pairs from chat history and compresses if needed  
- Expands the user’s question with context for improved responses  
- Retrieves text chunks from PDFs and other documents  
- Builds queries for the Gemini API and processes model replies  

---

## Installation

1. **Python 3.9+** is required.  
2. Clone the repository:  
   ```bash
   git clone https://github.com/aycayk/RAG-Gemini.git
   cd RAG-Gemini
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Set your Gemini API key as an environment variable:
     ```bash
   export GEMINI_API_KEY="your_api_key_here"

---

## Project Structure

```plain
RAG‑Gemini/
├── chat_prompt.py         # Defines query and instruction prompts
├── conversation_utils.py  # Selects, compresses, expands chat history and builds final prompt
├── download_model.py      # Script to download local model or tokenizer
├── main.py                # Main flow: document loading, retrieval, model queries
├── model_utils.py         # Communicates with Gemini API (query_gemini function)
├── pdf_utils.py           # Splits PDFs into chunks
└── requirements.txt       # List of required packages
