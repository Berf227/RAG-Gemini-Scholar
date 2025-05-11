def create_prompt(query, retrieved_results, chat_history):
    # Yeni instruction metni
    instruction = (
        "You are an expert article analyst with deep domain knowledge and a keen ability to compare, contrast, and summarize large sets of documents. "
        "Your task is not only to answer the user's question based on the provided article excerpts, but also to provide a general evaluation of the overall content, "
        "identify common themes, highlight significant differences, and point out any notable trends across the articles. "
        "Ensure your response is comprehensive, well-structured, and includes comparisons when applicable. "
        "If the user's question is ambiguous, ask for clarification. "
        "If the answer cannot be derived from the provided articles, respond with: "
        "\"I'm sorry, the answer to your question is not found in the uploaded articles. Please upload more comprehensive content.\""
    )
    
    # Belirli makale parçalarını içeren context
    context_parts = []
    for result in retrieved_results:
        context_parts.append(f"Article ({result['pdf']}):\n{result['chunk']}")
    context = "\n\n".join(context_parts)
    
    # Chat geçmişini ekle
    conversation = ""
    for msg in chat_history:
        conversation += f"{msg['role']}: {msg['content']}\n"
    conversation += f"User: {query}\n"
    
    # Yeni prompt oluşturma
    prompt = (
        f"Instruction:\n{instruction}\n\n"
        f"Article Excerpts:\n{context}\n\n"
        f"Conversation History:\n{conversation}\n\n"
        f"Based on the above, please provide a comprehensive answer that includes a general evaluation of the articles and draws comparisons between them where relevant.\n"
        f"Answer:"
    )
    
    return prompt
