# searching/graph.py
from langgraph.graph import StateGraph, END
from .schema import RAGState
from .retriever import Retriever
from .reranker import MixedbreadBaseReranker, MixedbreadReranker
from .generator import LegalGenerator
from .query_rewriter import QueryRewriter
from .doc_grader import DocGrader
from .hallucination_grader import HallucinationGrader
import json
import time

def quick_search_graph(case_type, model="gpt-oss:latest"):

    retriever = Retriever(distance="cosine")
    reranker = MixedbreadBaseReranker()
    generator = LegalGenerator(model=model)

    def add_time(state, key, duration):
        timing = dict(state.get("timing", {}))
        timing[key] = round(duration, 2)
        return timing

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
        duration = time.time() - start_time
        print(f"    成功抓取 {len(docs)} 筆原始資料，耗時: {duration:.2f} 秒")
        return {
            "retrieved_docs": docs,
            "timing": add_time(state, "retrieve", duration)
        }

    # ------------------------
    # 2. Rerank
    # ------------------------
    def rerank_node(state: RAGState):
        start_time = time.time()
        print("--- [階段 2] 執行 Mixedbread Base Rerank 二次重排 ---")
        docs = reranker.rerank(state["query"], state["retrieved_docs"], top_k=20)
        duration = time.time() - start_time
        print(f"    耗時: {duration:.2f} 秒")
        return {
            "reranked_docs": docs,
            "timing": add_time(state, "rerank", duration)
        }

    # ------------------------
    # 3. Generate
    # ------------------------
    def generate_node(state: RAGState):
        start_time = time.time()
        print("--- [階段 3] 法律 AI 正在生成回答 ---")
        hallucination_feedback = ""
        if state.get("retry_count", 0) > 0:
            hallucination_feedback = state.get("hallucination_reason", "")
            if hallucination_feedback:
                print(f"    套用前次幻覺檢查意見: {hallucination_feedback}")

        answer, generation_history = generator.generate_conversation(
            query=state["query"],
            docs=state["reranked_docs"],
            generation_history=state.get("generation_history", []),
            hallucination_feedback=hallucination_feedback,
        )
        duration = time.time() - start_time
        print(f"    生成完畢，耗時: {duration:.2f} 秒")
        return {
            "answer": answer,
            "generation_history": generation_history,
            "timing": add_time(state, "generate", duration)
        }

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
    reranker = MixedbreadReranker()
    generator = LegalGenerator(model=model)
    rewriter = QueryRewriter(model=model)
    doc_grader = DocGrader(model=model)
    hallucination_grader = HallucinationGrader(model=model)

    MAX_RETRY = 2

    def add_time(state, key, duration):
        timing = dict(state.get("timing", {}))
        timing[key] = round(duration, 2)
        return timing

    def add_retry_time(state, duration):
        timing = dict(state.get("timing", {}))
        timing["retry"] = round(timing.get("retry", 0) + duration, 2)
        return timing

    # ------------------------
    # 1. Query Rewrite
    # ------------------------
    def rewrite_node(state: RAGState):
        start_time = time.time()
        print(f"使用{model}作為主要模型")
        print("--- [階段 1] 正在重寫問題並提取關鍵字 ---")
        keywords = rewriter.rewrite(state["query"])
        duration = time.time() - start_time
        print(f"    耗時: {duration:.2f} 秒")

        return {
            "keywords": keywords,
            "timing": add_time(state, "rewrite", duration)
        }

    # ------------------------
    # 2. Retrieve
    # ------------------------
    def retrieve_node(state: RAGState):
        start_time = time.time()
        current_case_type = state.get("case_type") or case_type
        print(f"--- [階段 2] 正在檢索法律判決 (Case Type: {current_case_type}) ---")

        docs = retriever.retrieve(
            query=state["query"],
            keywords=state["keywords"],
            target_count=200,
            case_type=current_case_type
        )

        duration = time.time() - start_time
        print(f"    成功抓取 {len(docs)} 筆原始資料，耗時: {duration:.2f} 秒")

        return {
            "retrieved_docs": docs,
            "timing": add_time(state, "retrieve", duration)
        }

    # ------------------------
    # 3. Rerank
    # ------------------------
    def rerank_node(state: RAGState):
        start_time = time.time()
        print("--- [階段 3] 執行 Mixedbread Large Rerank 二次重排 ---")

        docs = reranker.rerank(state["query"], state["retrieved_docs"], top_k=20)

        duration = time.time() - start_time
        print(f"    耗時: {duration:.2f} 秒")

        return {
            "reranked_docs": docs,
            "timing": add_time(state, "rerank", duration)
        }

    # ------------------------
    # 4. Doc Grader
    # ------------------------
    def grade_node(state: RAGState):
        start_time = time.time()
        print("--- [階段 4] 評估檢索文件相關性 ---")

        result = doc_grader.grade(state["query"], state["reranked_docs"])

        if isinstance(result, dict):
            score_val = result.get("binary_score", "no")
            reason = result.get("reason", "")
        else:
            try:
                parsed = json.loads(result)
                score_val = parsed.get("binary_score", "no")
                reason = parsed.get("reason", "")
            except Exception:
                score_val = result
                reason = ""

        duration = time.time() - start_time
        print(f"    相關性檢查結果: {score_val}，耗時: {duration:.2f} 秒")

        return {
            "is_relevant": score_val,
            "doc_grade_reason": reason,
            "timing": add_time(state, "doc_grade", duration)
        }

    # ------------------------
    # 5. Generate
    # ------------------------
    def generate_node(state: RAGState):
        start_time = time.time()
        print("--- [階段 5] 法律 AI 正在生成回答 ---")

        hallucination_feedback = ""
        if state.get("retry_count", 0) > 0:
            hallucination_feedback = state.get("hallucination_reason", "")
            if hallucination_feedback:
                print(f"    套用前次幻覺檢查意見: {hallucination_feedback}")

        answer, generation_history = generator.generate_conversation(
            query=state["query"],
            docs=state["reranked_docs"],
            generation_history=state.get("generation_history", []),
            hallucination_feedback=hallucination_feedback,
        )

        duration = time.time() - start_time
        print(f"    生成完畢，耗時: {duration:.2f} 秒")

        attempt = state.get("retry_count", 0) + 1

        return {
            "answer": answer,
            "generation_history": generation_history,
            "timing": add_time(state, f"generate{attempt}", duration)
        }

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

        if isinstance(result, dict):
            score_val = result.get("binary_score", "yes")
            reason = result.get("reason", "")
        else:
            try:
                parsed = json.loads(result)
                score_val = parsed.get("binary_score", "yes")
                reason = parsed.get("reason", "")
            except Exception:
                score_val = result
                reason = ""

        duration = time.time() - start_time
        print(f"    幻覺檢查結果: {score_val}，耗時: {duration:.2f} 秒")

        attempt = state.get("retry_count", 0) + 1

        return {
            "hallucination_grade": score_val,
            "hallucination_reason": reason,
            "timing": add_time(state, f"hallucination{attempt}", duration)
        }

    # ------------------------
    # 7. Retry
    # ------------------------
    def retry_node(state: RAGState):
        retry_count = state.get("retry_count", 0) + 1
        print(f"⚠️ 檢查不通過，正在執行第 {retry_count} 次重新生成...")

        return {
            "retry_count": retry_count,
            "timing": dict(state.get("timing", {}))
        }

    # ------------------------
    # 8. 判斷 doc 是否可回答
    # ------------------------
    def decide_after_grade(state: RAGState):
        if state["is_relevant"] == "yes":
            return "generate"
        else:
            return "fail"

    # ------------------------
    # 9. 判斷生成後是否還需要 hallucination check
    # ------------------------
    def decide_after_generate(state: RAGState):
        if state.get("retry_count", 0) >= MAX_RETRY:
            return "end"
        return "hallucination"

    # ------------------------
    # 10. 判斷 hallucination
    # ------------------------
    def decide_after_hallucination(state: RAGState):
        if state["hallucination_grade"] == "no":
            return "end"
        return "retry"

    # ------------------------
    # 11. 無法回答 fallback
    # ------------------------
    def fail_node(state: RAGState):
        reason = state.get("doc_grade_reason") or "根據目前檢索到的判決資料，無法找到足夠依據回答該問題。"
        return {
            "answer": reason,
            "timing": state.get("timing", {})
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

    graph.add_conditional_edges(
        "generate",
        decide_after_generate,
        {
            "hallucination": "hallucination",
            "end": END
        }
    )

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