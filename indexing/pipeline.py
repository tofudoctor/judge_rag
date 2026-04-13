# indexing/pipeline.py
import uuid
from tqdm import tqdm
from langchain_ollama import OllamaEmbeddings
from .loader import data_loader_by_years
from .chunker import chunk_documents
from .writer import QdrantWriter
from ..utils.batch import batch_iter

class BuildPipeline:
    def __init__(self):
        self.embedder = OllamaEmbeddings(model="bge-m3")
        self.writer = QdrantWriter(host="localhost", embedding_model=self.embedder)
        self.vector_dim = len(self.embedder.embed_query("test"))

    def run(self, base_dir, case_type=None, n_years=None, distance="cosine"):
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
            collection_name = f"{distance}_chunk"
            self.writer.ensure_collection(
                name=collection_name,
                dim=self.vector_dim,
                distance=distance
            )

            store = self.writer.get_vector_store(
                collection_name=collection_name,
            )

            # 批次寫入 + tqdm
            print(f"[Writer] Adding to {collection_name}")
            batch_size = 256
            total_batches = (len(chunked_docs) + batch_size - 1) // batch_size
            for batch in tqdm(batch_iter(chunked_docs, batch_size),
                            total=total_batches,
                            desc=f"[Writer] Adding to {collection_name}",
                            unit="batch",
                            dynamic_ncols=True,
                            bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                # 過濾空 chunk
                batch = [d for d in batch if d.page_content.strip()]
                if not batch:
                    continue

                # 生成 id
                ids = []
                for chunk in batch:
                    chunk.metadata["chunk_size"] = chunk_size
                    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.page_content.encode("utf-8")))
                    chunk.metadata["id"] = chunk_id
                    ids.append(chunk_id)

                # 檢查哪些 ID 已經存在
                try:
                    existing_points = self.writer.client.retrieve(
                        collection_name=collection_name,
                        ids=ids,
                        with_payload=False, # 只要查存在，不需要內容
                        with_vectors=False
                    )
                    existing_ids = {p.id for p in existing_points}
                except Exception as e:
                    print(f"[ERROR] 查詢重複 ID 失敗: {e}")
                    existing_ids = set()

                # 過濾出「真的不在資料庫內」的 chunk 和 id
                new_batch = []
                new_ids = []
                for i, cid in enumerate(ids):
                    if cid not in existing_ids:
                        new_batch.append(batch[i])
                        new_ids.append(cid)

                # 如果這批全都在資料庫裡了，就直接跳過，不驚動 Ollama
                if not new_batch:
                    continue

                # 嘗試加入新 batch (只針對 new_batch)
                try:
                    store.add_documents(new_batch, ids=new_ids)
                except Exception as e_batch:
                    # 如果整個 batch 失敗，逐個加入
                    for chunk, cid in zip(new_batch, new_ids):
                        try:
                            store.add_documents([chunk], ids=[cid])
                        except Exception as e_chunk:
                            print(f"[WARN] Skipping bad chunk: {e_chunk}")
                            print(f"Chunk preview: {chunk.page_content[:100]}")
                            continue

            print(f"[Writer] Written to Qdrant collection: {collection_name}")

        