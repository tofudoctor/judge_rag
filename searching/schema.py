# searching/schema.py
from typing import List, TypedDict
from langchain_core.documents import Document

class RAGState(TypedDict):
    query: str
    keywords: str
    case_type: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    is_relevant: str
    hallucination_grade: str  # 'yes' 或 'no'
    retry_count: int         # 避免無限迴圈的計數器
    answer: str