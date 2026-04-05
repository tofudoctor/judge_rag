# indexing/embedder.py
from langchain.embeddings import OllamaEmbeddings

class Embedder:
    def __init__(self, model_name=None):
        self.model = OllamaEmbeddings(model=model_name)

    def embed(self, chunks):
        return self.model.embed_documents(
            [c.page_content for c in chunks]
        )