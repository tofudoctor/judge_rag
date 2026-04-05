# indexing/writer.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore

def get_client(host="localhost", port=6333):
    return QdrantClient(url=f"http://{host}:{port}")

def ensure_collection(client, name, dim, distance):
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

def get_store(client, collection, embedding):
    return QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embedding,
    )