from model_utils import query_gemini

def get_recent_qa_pairs(chat_history, max_tokens=1500, max_pairs=5):
    selected_pairs = []
    total_tokens = 0
    
    for pair in reversed(chat_history):
        q_tokens = len(pair["question"].split())
        a_tokens = len(pair["answer"].split())
        pair_tokens = q_tokens + a_tokens

        if total_tokens + pair_tokens > max_tokens:
            break

        selected_pairs.insert(0, pair)  # Baştan ekle
        total_tokens += pair_tokens

        if len(selected_pairs) >= max_pairs:
            break

    return selected_pairs


def compress_conversation_context(qa_pairs, api_key):
    qa_text = ""
    for pair in qa_pairs:
        qa_text += f"Q: {pair['question']}\nA: {pair['answer']}\n"

    compression_prompt = (
        "Below is a conversation history. Extract only the essential facts and context required to answer a new follow-up question.\n"
        "Return this as a compact paragraph (do not repeat full Q&A):\n\n"
        f"{qa_text}\n\n"
        "Compressed Context:"
    )

    response = query_gemini(compression_prompt, api_key)
    compressed_context = ""
    for candidate in response.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            compressed_context += part.get("text", "")
    return compressed_context


def expand_user_question(original_question, compressed_context, api_key):
    expansion_prompt = (
        f"You are expanding the following user question using background knowledge:\n\n"
        f"Background Context:\n{compressed_context}\n\n"
        f"Original Question:\n{original_question}\n\n"
        f"Expanded Question (self-contained, with context):"
    )
    
    response = query_gemini(expansion_prompt, api_key)
    expanded_question = ""
    for candidate in response.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            expanded_question += part.get("text", "")
    return expanded_question


def create_final_prompt(compressed_context, expanded_question, retrieved_docs):
    doc_context = "\n\n".join([
        f"Document ({doc['pdf']}):\n{doc['chunk']}" for doc in retrieved_docs
    ])
    
    prompt = (
        f"Context from conversation:\n{compressed_context}\n\n"
        f"User's expanded question:\n{expanded_question}\n\n"
        f"Relevant document excerpts:\n{doc_context}\n\n"
        f"Based on the above, provide a comprehensive answer:"
    )
    
    return prompt
