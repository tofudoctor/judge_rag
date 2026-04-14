# searching/pipeline.py
from .graph import quick_search_graph, full_search_graph
import time

class BaseSearchPipeline:
    """基礎 Pipeline 類別，封裝共同邏輯"""
    def __init__(self):
        self.app = None

    def _format_output(self, result):
        """統一格式化輸出結果"""
        ref_details = []
        reranked_docs = result.get("reranked_docs", [])
        for doc in reranked_docs:
            jid = doc.metadata.get("JID")
            score = doc.metadata.get("rerank_score")

            if jid:
                    # 處理分數顯示，若無分數則顯示 N/A
                    score_val = round(score, 4) if isinstance(score, (int, float)) else "N/A"
                    ref_details.append({
                        "JID": jid,
                        "score": score_val
                    })

        return {
            "answer": result.get("answer", "抱歉，系統無法產出回答。"),
            "ref_details": ref_details,  # 這是你要的新格式
            "ref_jids": [d["JID"] for d in ref_details] # 保留舊的方便比對
        }

class QuickSearchPipeline(BaseSearchPipeline):
    """快速模式：直接檢索、重排、生成"""
    def __init__(self, case_type, model="gpt-oss:latest"):
        super().__init__()
        self.app = quick_search_graph(case_type, model=model)

    def run(self, query: str):
        overall_start = time.time()

        result = self.app.invoke({
            "query": query
        })

        output = self._format_output(result)
        
        total_time = time.time() - overall_start
        print(f"\n✅ [任務完成] 總檢索生成耗時: {total_time:.2f} 秒")
        
        output["total_time"] = total_time

        return output

class FullSearchPipeline(BaseSearchPipeline):
    """完整模式：改寫、檢索、重排、文件評分、幻覺檢查、失敗重試"""
    def __init__(self, case_type, model="gpt-oss:120b"):
        super().__init__()
        self.app = full_search_graph(case_type, model=model)

    def run(self, query: str):
        # Full 模式需要初始化 retry_count
        overall_start = time.time()

        result = self.app.invoke({
            "query": query,
            "retry_count": 0
        })
        
        output = self._format_output(result)
        # 額外提供評分狀態供除錯或 UI 顯示
        output["is_relevant"] = result.get("is_relevant", "no")
        output["hallucination_grade"] = result.get("hallucination_grade", "unknown")
        
        total_time = time.time() - overall_start
        print(f"\n✅ [任務完成] 總檢索生成耗時: {total_time:.2f} 秒")
        
        output["total_time"] = total_time

        return output