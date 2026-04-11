# searching/graph.py
from langgraph.graph import StateGraph, END
from .schema import RAGState
from .retriever import MultiRetriever
from .reranker import FlashReranker
from .generator import LegalGenerator
from .query_rewriter import MultiQueryRewriter
from .doc_grader import DocGrader
from .hallucination_grader import HallucinationGrader
import json

def quick_search_graph(case_type):

    retriever = MultiRetriever()
    reranker = FlashReranker()

    def retrieve_node(state: RAGState):
        docs = retriever.retrieve(state["query"], case_type=case_type)
        return {"retrieved_docs": docs}

    def rerank_node(state: RAGState):
        docs = reranker.rerank(state["query"], state["retrieved_docs"])
        return {"reranked_docs": docs}

    def generate_node(state: RAGState):
        answer = generate_answer(state["query"], state["reranked_docs"])
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

    retriever = MultiRetriever()
    reranker = FlashReranker()
    generator = LegalGenerator(model=model)
    rewriter = MultiQueryRewriter(model=model)
    doc_grader = DocGrader(model=model)
    hallucination_grader = HallucinationGrader(model=model)

    MAX_RETRY = 2

    # ------------------------
    # 1. Query Rewrite
    # ------------------------
    def rewrite_node(state: RAGState):
        queries = rewriter.rewrite(state["query"])
        return {"queries": queries}

    # ------------------------
    # 2. Retrieval（多 query）
    # ------------------------
    def retrieve_node(state: RAGState):
        all_docs = []

        for q in state["queries"]:
            docs = retriever.retrieve(q, case_type=case_type)
            all_docs.extend(docs)

        # 去重（用 JID）
        seen = set()
        unique_docs = []

        for d in all_docs:
            jid = d.metadata.get("JID")
            if jid not in seen:
                seen.add(jid)
                unique_docs.append(d)

        return {"retrieved_docs": unique_docs}

    # ------------------------
    # 3. Rerank
    # ------------------------
    def rerank_node(state: RAGState):
        docs = reranker.rerank(state["query"], state["retrieved_docs"])
        return {"reranked_docs": docs}

    # ------------------------
    # 4. Doc Grader（能不能回答）
    # ------------------------
    def grade_node(state: RAGState):
        result = doc_grader.grade(
            state["query"],
            state["reranked_docs"]
        )

        try:
            score = json.loads(result)["binary_score"]
        except:
            score = "no"

        return {"is_relevant": score}

    # ------------------------
    # 5. Generate
    # ------------------------
    def generate_node(state: RAGState):
        answer = generator.generate(
            state["query"],
            state["reranked_docs"]
        )
        return {"answer": answer}

    # ------------------------
    # 6. Hallucination Check
    # ------------------------
    def hallucination_node(state: RAGState):
        result = hallucination_grader.grade(
            state["answer"],
            state["reranked_docs"]
        )

        try:
            score = json.loads(result)["binary_score"]
        except:
            score = "no"

        return {"hallucination_grade": score}

    # ------------------------
    # 7. Retry 控制
    # ------------------------
    def retry_node(state: RAGState):
        retry_count = state.get("retry_count", 0) + 1
        return {"retry_count": retry_count}

    # ------------------------
    # 8. 條件判斷：Doc 是否可回答
    # ------------------------
    def decide_after_grade(state: RAGState):
        if state["is_relevant"] == "yes":
            return "generate"
        else:
            if state.get("retry_count", 0) >= MAX_RETRY:
                return "fail"
            return "retry"

    # ------------------------
    # 9. 條件判斷：Hallucination
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
    # Graph 建立
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

    # doc grader 分支
    graph.add_conditional_edges(
        "grade",
        decide_after_grade,
        {
            "generate": "generate",
            "retry": "retry",
            "fail": "fail",
        }
    )

    graph.add_edge("retry", "rewrite")

    # generate → hallucination
    graph.add_edge("generate", "hallucination")

    # hallucination 分支
    graph.add_conditional_edges(
        "hallucination",
        decide_after_hallucination,
        {
            "end": END,
            "retry": "retry",
        }
    )

    graph.add_edge("fail", END)

    return graph.compile()