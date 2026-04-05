# indexing/chunker.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs, chunk_size=1000, chunk_overlap_ratio=0.2):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size*chunk_overlap_ratio,
        separators = ["。", "；", "！", "？", " ", "　"],
    )

    return splitter.split_documents(docs)