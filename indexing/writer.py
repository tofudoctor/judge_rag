# indexing/writer.py
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import VectorParams, Distance, SparseVectorParams
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

class QdrantWriter:
    def __init__(self, host="localhost", port=6333, embedding_model=None):
        self.client = QdrantClient(url=f"http://{host}:{port}")
        self.embedding = embedding_model
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
    def ensure_collection(self, name, dim, distance="cosine"):
        if self.client.collection_exists(name):
            return

        dist_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclid": Distance.EUCLID,
            "manhattan": Distance.MANHATTAN
        }
        dist = dist_map.get(distance, Distance.COSINE)

        self.client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": VectorParams(size=dim, distance=dist)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )
        print(f"[Qdrant] Collection '{name}' created.")

    def get_vector_store(self, collection_name):
        """回傳封裝好的 LangChain VectorStore"""
        return QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embedding,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )