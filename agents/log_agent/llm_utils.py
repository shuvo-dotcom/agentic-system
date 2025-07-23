import openai
from typing import List

def summarize_logs(log_messages: List[str], api_key: str) -> str:
    openai.api_key = api_key
    context = "\n".join(log_messages)
    prompt = f"Summarize the following logs in a concise, actionable way for an engineer.\n\nLogs:\n{context}\n\nSummary:" 
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def answer_log_query(question: str, log_messages: List[str], api_key: str) -> str:
    openai.api_key = api_key
    context = "\n".join(log_messages)
    prompt = f"Given the following logs, answer the user's question as precisely as possible.\n\nLogs:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip() 