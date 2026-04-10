# searching/schema.py
from typing import List, TypedDict
from langchain_core.documents import Document

class RAGState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    answer: str