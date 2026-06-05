# searching/schema.py
from typing import Any, List, TypedDict
from langchain_core.documents import Document

class RAGState(TypedDict):
    query: str
    keywords: str
    case_type: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    is_relevant: str
    doc_grade_reason: str
    hallucination_grade: str  # 'yes' 或 'no'
    hallucination_reason: str
    retry_count: int         # 避免無限迴圈的計數器
    answer: str
    generation_history: list[dict[str, str]]
    timing: dict[str, Any]
