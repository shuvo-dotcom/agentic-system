from utils.llm_provider import get_llm_response
from typing import List

def summarize_logs(log_messages: List[str], api_key: str) -> str:
    context = "\n".join(log_messages)
    prompt = f"Summarize the following logs in a concise, actionable way for an engineer.\n\nLogs:\n{context}\n\nSummary:" 
    response = get_llm_response([{"role": "user", "content": prompt}])
    return response.strip()

def answer_log_query(question: str, log_messages: List[str], api_key: str) -> str:
    context = "\n".join(log_messages)
    prompt = f"Given the following logs, answer the user's question as precisely as possible.\n\nLogs:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = get_llm_response([{"role": "user", "content": prompt}])
    return response.strip() 