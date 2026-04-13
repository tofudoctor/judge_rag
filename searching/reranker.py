# searching/reranker.py
import torch
from flashrank import Ranker, RerankRequest
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from typing import List

class FlashReranker:
    def __init__(self, model_name: str = "ms-marco-MultiBERT-L-12", cache_dir: str = "/opt"):
        """
        初始化時加載模型，避免重複加載
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        # 在啟動時就先載入模型到記憶體
        self.ranker = Ranker(model_name=self.model_name, cache_dir=self.cache_dir)

    def rerank(self, query: str, docs: List[Document], top_k: int = 10) -> List[Document]:
        """
        對 LangChain Document 物件進行重排
        """
        if not docs:
            return []

        # 1. 格式轉換：FlashRank 要求的格式是 List[Dict]
        # 我們將 Document 轉為含有 "id", "text", "metadata" 的 dict
        passages = []
        for i, doc in enumerate(docs):
            passages.append({
                "id": doc.metadata.get("JID", str(i)), # 優先用 JID 作為 ID
                "text": doc.page_content,
                "metadata": doc.metadata
            })

        # 2. 建立 Rerank 請求
        rerank_request = RerankRequest(query=query, passages=passages)
        
        # 3. 執行重排
        results = self.ranker.rerank(rerank_request)

        # 4. 轉回 LangChain Document 物件
        final_docs = []
        for r in results[:top_k]:
            # 從結果中重建 Document
            final_docs.append(
                Document(
                    page_content=r["text"],
                    metadata={
                        **r["metadata"],           # 保留原始所有 metadata (JID, TYPE 等)
                        "rerank_score": r["score"] # 加入 Rerank 分數供後續參考
                    }
                )
            )

        return final_docs
    
class BGEReranker:
    def __init__(self, model_name='BAAI/bge-reranker-v2-m3', cache_dir="./cache"):
        # 這裡會直接從 HuggingFace 下載完整的 BGE 模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(
            model_name, 
            max_length=1024, 
            device=device, # 如果有 GPU 改成 "cuda"
            automodel_args={"cache_dir": cache_dir}
        )

    def rerank(self, query: str, docs: List[Document], top_k: int = 10) -> List[Document]:
        if not docs: return []
        
        # 準備輸入格式：[[query, text1], [query, text2], ...]
        sentence_pairs = [[query, doc.page_content] for doc in docs]
        
        # 獲得分數
        scores = self.model.predict(sentence_pairs)
        
        # 結合分數並排序
        for i, doc in enumerate(docs):
            doc.metadata["rerank_score"] = float(scores[i])
            
        sorted_docs = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)
        return sorted_docs[:top_k]
    
    def simple_rerank(self, docs: List[Document], top_k: int = 20) -> List[Document]:
        """
        保留原本的簡易截斷方法
        """
        sorted_docs = sorted(docs, key=lambda x: x.metadata["relevance_score"], reverse=True)
        return sorted_docs[:top_k]