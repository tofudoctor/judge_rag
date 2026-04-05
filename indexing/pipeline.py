# indexing/pipeline.py
from tqdm import tqdm
from langchain_ollama import OllamaEmbeddings
from .loader import data_loader_by_years
from .chunker import chunk_documents
from .writer import get_client, ensure_collection, get_store
from ..utils.batch import batch_iter

class BuildPipeline:
    def __init__(self):
        self.embedder = OllamaEmbeddings(model="bge-m3")
        self.client = get_client(host="localhost", port=6333)
        self.vector_dim = len(self.embedder.embed_query("test"))

    def run(self, base_dir, case_type=None, n_years=None):
        """
        pipeline 入口
        """

        # loader
        docs = data_loader_by_years(
            base_dir=base_dir,
            case_type=case_type,
            n_years=n_years
        )
        print(f"[Loader] Loaded {len(docs)} documents from {base_dir}")

        # chunker + embedder + Qdrant
        for chunk_size in [1000, 500, 300]:
            print(f"---------chunk size = {chunk_size}-----------")

            # chunk
            chunked_docs = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap_ratio=0.2)
            print(f"[Chunker] {len(chunked_docs)} chunks created")

            # Qdrant + embed
            collection_name = f"{case_type}_chunk{chunk_size}"
            ensure_collection(
                client=self.client,
                name=collection_name,
                dim=self.vector_dim,
                distance="cosine"
            )

            store = get_store(
                client=self.client,
                collection=collection_name,
                embedding=self.embedder
            )

            # 批次寫入 + tqdm
            batch_size = 32
            total_batches = (len(chunked_docs) + batch_size - 1) // batch_size
            for batch in tqdm(batch_iter(chunked_docs, batch_size),
                            total=total_batches,
                            desc=f"[Writer] Adding to {collection_name}",
                            unit="batch",
                            dynamic_ncols=True):
                store.add_documents(batch)

            print(f"[Writer] Written to Qdrant collection: {collection_name}")

        