# searching/reranker.py
def simple_rerank(query, docs, top_k=10):
    # 先用最簡單方式：截前 top_k
    return docs[:top_k]