# searching/retriever.py
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import FastEmbedSparse

class Retriever:
    def __init__(
        self, 
        distance="cosine", 
        dense_embedding_model="bge-m3", 
        sparse_embedding_model="Qdrant/bm25", 
        url="http://localhost:6333"):

        self.embedder = OllamaEmbeddings(model=dense_embedding_model)
        self.sparse_embeddings = FastEmbedSparse(model_name=sparse_embedding_model)
        self.client = QdrantClient(url=url, timeout=60)
        self.collection_name = f"{distance}_chunk"
        

    def retrieve(self, query, keywords, target_count=50, case_type=None):
        # 生成向量 (參考原始碼變數命名邏輯)
        q_dense = self.embedder.embed_query(query)
        q_sparse = self.sparse_embeddings.embed_query(query)
        kw_dense = self.embedder.embed_query(keywords)
        kw_sparse = self.sparse_embeddings.embed_query(keywords)

        must_conditions = []
        if case_type:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.TYPE",
                    match=models.MatchValue(value=case_type)
                )
            )

        # 執行分組查詢 (使用 query_points_groups API)
        search_result = self.client.query_points_groups(
            collection_name=self.collection_name,
            # 使用 RrfQuery 並指定權重
            # 順序必須與 prefetch 清單完全一致
            query=models.RrfQuery(
                rrf=models.Rrf(
                    # 權重分配邏輯：
                    # 5.0: Query Dense (核心語意)
                    # 2.0: Query Sparse (問題關鍵字)
                    # 0.2: Keywords Dense (關鍵字語意擴展)
                    # 0.5: Keywords Sparse (精確法條)
                    weights=[5.0, 2.0, 0.2, 0.5] 
                )
            ),
            prefetch=[
                # 索引 0: Query - Dense
                models.Prefetch(query=q_dense, using="dense", limit=target_count * 2),
                # 索引 1: Query - Sparse
                models.Prefetch(
                    query=models.SparseVector(indices=q_sparse.indices, values=q_sparse.values),
                    using="sparse", 
                    limit=target_count * 2
                ),
                # 索引 2: Keywords - Dense
                models.Prefetch(query=kw_dense, using="dense", limit=target_count * 2),
                # 索引 3: Keywords - Sparse
                models.Prefetch(
                    query=models.SparseVector(indices=kw_sparse.indices, values=kw_sparse.values),
                    using="sparse", 
                    limit=target_count * 2
                ),
            ],
            group_by="metadata.JID",
            limit=target_count,
            group_size=1,
            query_filter=models.Filter(must=must_conditions) if must_conditions else None,
            with_payload=True
        )

        # 處理分組結果 (search_result 是一個包含 groups 的物件)
        final_docs = []
        for group in search_result.groups:
            if not group.hits:
                continue
            
            # 取得該組最高分的點
            top_hit = group.hits[0]
            
            # 封裝回 LangChain Document
            doc = Document(
                page_content=top_hit.payload.get("page_content", ""),
                metadata=top_hit.payload.get("metadata", {})
            )
            # 注入 RRF 分數
            doc.metadata["relevance_score"] = top_hit.score
            final_docs.append(doc)

        return final_docs