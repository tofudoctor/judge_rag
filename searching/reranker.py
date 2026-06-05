# searching/reranker.py
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document
from typing import List

RECENCY_BOOST_MAX = 0.5
AUTHORITY_BOOST = 2.0

class FlashReranker:
    def __init__(self, model_name: str = "ms-marco-MultiBERT-L-12", cache_dir: str = "./cache"):
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
    
class MixedbreadReranker:
    def __init__(self, model_name='mixedbread-ai/mxbai-rerank-large-v2', cache_dir="./cache"):
        # 這裡會直接從 HuggingFace 下載完整的 Mixedbread reranker 模型
        import torch
        from sentence_transformers import CrossEncoder

        requested_device = os.getenv("RERANKER_DEVICE", "cuda").lower()
        if requested_device == "cuda" and not torch.cuda.is_available():
            print("[Reranker] CUDA unavailable; fallback to CPU.")
            requested_device = "cpu"

        model_kwargs = {"cache_dir": cache_dir}
        if requested_device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16

        print(f"[Reranker] Loading {model_name} on {requested_device}.")

        self.model = CrossEncoder(
            model_name,
            max_length=1024,
            device=requested_device,
            model_kwargs=model_kwargs,
        )

    def rerank(self, query: str, docs: List[Document], top_k: int = 10) -> List[Document]:
        if not docs: return []
        
        # 準備輸入格式：[[query, text1], [query, text2], ...]
        sentence_pairs = [[query, doc.page_content] for doc in docs]
        
        # 獲得分數
        scores = self.model.predict(sentence_pairs)

        # 同一次搜尋結果中，依 JDATE 越新給越高的新近性加權。
        dated_docs = [
            (str(doc.metadata.get("JDATE", "")), doc)
            for doc in docs
            if doc.metadata.get("JDATE")
        ]
        dated_docs.sort(key=lambda item: item[0], reverse=True)

        recency_boost_by_id = {}
        date_count = len(dated_docs)
        for rank, (_, doc) in enumerate(dated_docs):
            if date_count <= 1:
                boost = RECENCY_BOOST_MAX
            else:
                boost = RECENCY_BOOST_MAX * (1 - rank / (date_count - 1))
            recency_boost_by_id[id(doc)] = boost
        
        # 結合分數並排序
        for i, doc in enumerate(docs):
            raw_score = float(scores[i])
            jid = doc.metadata.get("JID", "")
            authority_boost = AUTHORITY_BOOST if "大" in jid and raw_score > 10 else 0.0
            recency_boost = recency_boost_by_id.get(id(doc), 0.0)

            doc.metadata["rerank_raw_score"] = raw_score
            doc.metadata["authority_boost"] = round(authority_boost, 4)
            doc.metadata["recency_boost"] = round(recency_boost, 4)
            doc.metadata["rerank_score"] = raw_score + authority_boost + recency_boost
            
        sorted_docs = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)
        return sorted_docs[:top_k]
    
    def simple_rerank(self, docs: List[Document], top_k: int = 20) -> List[Document]:
        """
        保留原本的簡易截斷方法
        """
        sorted_docs = sorted(docs, key=lambda x: x.metadata["relevance_score"], reverse=True)
        return sorted_docs[:top_k]


class MixedbreadBaseReranker(MixedbreadReranker):
    def __init__(self, cache_dir="./cache"):
        super().__init__(
            model_name='mixedbread-ai/mxbai-rerank-base-v2',
            cache_dir=cache_dir,
        )

