# searching/generator.py
from langchain_ollama import ChatOllama

def generate_answer(query, docs):
    llm = ChatOllama(model="gpt-oss:latest")

    context = "\n\n".join([d.page_content[:500] for d in docs])

    prompt = f"""
你是一個法律助理，請根據以下資料回答問題：

[資料]
{context}

[問題]
{query}

請提供：
1. 精簡答案
2. 引用依據
"""

    return llm.invoke(prompt).content