# searching/graph.py
from langgraph.graph import StateGraph, END
from .schema import RAGState
from .retriever import Retriever
from .reranker import BGEReranker
from .generator import LegalGenerator
from .query_rewriter import QueryRewriter
from .doc_grader import DocGrader
from .hallucination_grader import HallucinationGrader
import json
import time

def quick_search_graph(case_type, model="gpt-oss:latest"):

    retriever = Retriever(distance="cosine")
    reranker = BGEReranker()
    generator = LegalGenerator(model=model)

    # ------------------------
    # 1. Retrieve
    # ------------------------
    def retrieve_node(state: RAGState):
        start_time = time.time()
        current_case_type = state.get("case_type") or case_type
        print(f"--- [階段 1] 正在檢索法律判決 (Case Type: {current_case_type}) ---")
        docs = retriever.retrieve(
            query=state["query"],
            keywords=state["query"],
            target_count=100,
            case_type=current_case_type
        )
        print(f"    成功抓取 {len(docs)} 筆原始資料，耗時: {time.time() - start_time:.2f} 秒")
        return {"retrieved_docs": docs}

    # ------------------------
    # 2. Rerank
    # ------------------------
    def rerank_node(state: RAGState):
        start_time = time.time()
        print("--- [階段 2] 執行 BGE Rerank 二次重排 ---")
        docs = reranker.rerank(state["query"], state["retrieved_docs"], top_k=20)
        print(f"    耗時: {time.time() - start_time:.2f} 秒")
        return {"reranked_docs": docs}

    # ------------------------
    # 3. Generate
    # ------------------------
    def generate_node(state: RAGState):
        start_time = time.time()
        print("--- [階段 3] 法律 AI 正在生成回答 ---")
        answer = generator.generate(state["query"], state["reranked_docs"])
        print(f"    生成完畢，耗時: {time.time() - start_time:.2f} 秒")
        return {"answer": answer}

    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

def full_search_graph(case_type, model="gpt-oss:latest"):

    retriever = Retriever(distance="cosine")
    reranker = BGEReranker()
    generator = LegalGenerator(model=model)
    rewriter = QueryRewriter(model=model)
    doc_grader = DocGrader(model=model)
    hallucination_grader = HallucinationGrader(model=model)

    MAX_RETRY = 1

    # ------------------------
    # 1. Query Rewrite
    # ------------------------
    def rewrite_node(state: RAGState):
        start_time = time.time()
        print(f"使用{model}作為主要模型")
        print("--- [階段 1] 正在重寫問題並提取關鍵字 ---")
        keywords  = rewriter.rewrite(state["query"])
        print(f"    耗時: {time.time() - start_time:.2f} 秒")
        return {"keywords": keywords}

    # ------------------------
    # 2. Retrieve
    # ------------------------
    def retrieve_node(state: RAGState):
        start_time = time.time()
        current_case_type = state.get("case_type") or case_type
        print(f"--- [階段 2] 正在檢索法律判決 (Case Type: {current_case_type}) ---")
        docs = retriever.retrieve(
            query=state["query"],
            keywords=state["query"],
            target_count=100,
            case_type=current_case_type
        )
        print(f"    成功抓取 {len(docs)} 筆原始資料，耗時: {time.time() - start_time:.2f} 秒")
        return {"retrieved_docs": docs}

    # ------------------------
    # 3. Rerank
    # ------------------------
    def rerank_node(state: RAGState):
        start_time = time.time()
        print("--- [階段 3] 執行 BGE Rerank 二次重排 ---")
        docs = reranker.rerank(state["query"], state["retrieved_docs"], top_k=20)
        print(f"    耗時: {time.time() - start_time:.2f} 秒")
        return {"reranked_docs": docs}

    # ------------------------
    # 4. Doc Grader
    # ------------------------
    def grade_node(state: RAGState):
        start_time = time.time()
        print("--- [階段 4] 評估檢索文件相關性 ---")
        result = doc_grader.grade(state["query"], state["reranked_docs"])
        try:
            # 確保提取 json 欄位
            score_val = json.loads(result)["binary_score"]
        except:
            score_val = result
        
        print(f"    相關性檢查結果: {score_val}，耗時: {time.time() - start_time:.2f} 秒")
        return {"is_relevant": score_val}

    # ------------------------
    # 5. Generate
    # ------------------------
    def generate_node(state: RAGState):
        start_time = time.time()
        print("--- [階段 5] 法律 AI 正在生成回答 ---")
        answer = generator.generate(state["query"], state["reranked_docs"])
        print(f"    生成完畢，耗時: {time.time() - start_time:.2f} 秒")
        return {"answer": answer}

    # ------------------------
    # 6. Hallucination Check
    # ------------------------
    def hallucination_node(state: RAGState):
        start_time = time.time()
        print("--- [階段 6] 執行幻覺檢查 (Hallucination Check) ---")
        result = hallucination_grader.grade(
            state["answer"],
            state["reranked_docs"]
        )
        try:
            score_val = json.loads(result)["binary_score"]
        except:
            score_val = result
            
        print(f"    幻覺檢查結果: {score_val}，耗時: {time.time() - start_time:.2f} 秒")
        return {"hallucination_grade": score_val}

    # ------------------------
    # 7. Retry
    # ------------------------
    def retry_node(state: RAGState):
        retry_count = state.get("retry_count", 0) + 1
        print(f"⚠️ 檢查不通過，正在執行第 {retry_count} 次重新生成...")
        return {"retry_count": retry_count}

    # ------------------------
    # 8. 判斷 doc 是否可回答
    # ------------------------
    def decide_after_grade(state: RAGState):
        if state["is_relevant"] == "yes":
            return "generate"
        else:
            return "fail"

    # ------------------------
    # 9. 判斷 hallucination
    # ------------------------
    def decide_after_hallucination(state: RAGState):
        if state["hallucination_grade"] == "yes":
            return "end"
        else:
            if state.get("retry_count", 0) >= MAX_RETRY:
                return "end"
            return "retry"

    # ------------------------
    # 10. 無法回答 fallback
    # ------------------------
    def fail_node(state: RAGState):
        return {
            "answer": "根據目前檢索到的判決資料，無法找到足夠依據回答該問題。"
        }

    # ------------------------
    # Graph
    # ------------------------
    graph = StateGraph(RAGState)

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("grade", grade_node)
    graph.add_node("generate", generate_node)
    graph.add_node("hallucination", hallucination_node)
    graph.add_node("retry", retry_node)
    graph.add_node("fail", fail_node)

    graph.set_entry_point("rewrite")

    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "grade")

    graph.add_conditional_edges(
        "grade",
        decide_after_grade,
        {
            "generate": "generate",
            "fail": "fail"
        }
    )

    graph.add_edge("generate", "hallucination")

    graph.add_conditional_edges(
        "hallucination",
        decide_after_hallucination,
        {
            "end": END,
            "retry": "retry"
        }
    )

    graph.add_edge("retry", "generate")  # ⚠️ 重跑 generate，不重抓資料
    graph.add_edge("fail", END)

    return graph.compile()