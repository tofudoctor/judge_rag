# searching/retriever.py
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings

class MultiRetriever:
    def __init__(self, case_type):
        self.embedder = OllamaEmbeddings(model="bge-m3")
        self.client = QdrantClient(url="http://localhost:6333")

        self.collections = [
            f"{case_type}_chunk1000",
            f"{case_type}_chunk500",
            f"{case_type}_chunk300",
        ]

        self.stores = [
            QdrantVectorStore(
                client=self.client,
                collection_name=col,
                embedding=self.embedder,
            )
            for col in self.collections
        ]

    def retrieve(self, query, target_count=30):
        all_results = []
        
        # 為了去重後仍有足夠數量，每個庫抓取 target_count 的兩倍
        search_k = target_count * 2
        
        for store in self.stores:
            # 使用 score 版本
            results = store.similarity_search_with_score(query, k=search_k)
            all_results.extend(results)

        # 根據 Score（相似度分數）進行降冪排序 (從高到低)
        all_results.sort(key=lambda x: x[1], reverse=True)

        # 進行 JID 去重
        seen_jids = set()
        unique_docs = []

        for doc, score in all_results:
            jid = doc.metadata.get("JID")
            
            if jid not in seen_jids:
                seen_jids.add(jid)
                # 將分數存入 metadata，方便後續 node（如 rerank）參考
                doc.metadata["relevance_score"] = score
                unique_docs.append(doc)
            
            # 4. 收集到目標篇數就停止
            if len(unique_docs) >= target_count:
                break

        return unique_docs