# searching/graph.py
from langgraph.graph import StateGraph, END
from .schema import RAGState
from .retriever import MultiRetriever
from .reranker import simple_rerank
from .generator import generate_answer

def quick_search_graph(case_type):

    retriever = MultiRetriever(case_type)

    def retrieve_node(state: RAGState):
        docs = retriever.retrieve(state["query"])
        return {"retrieved_docs": docs}

    def rerank_node(state: RAGState):
        docs = simple_rerank(state["query"], state["retrieved_docs"])
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