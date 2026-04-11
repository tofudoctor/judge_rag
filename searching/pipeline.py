# searching/pipeline.py
from .graph import quick_search_graph

class QuickSearchPipeline:
    def __init__(self, case_type):
        self.app = quick_search_graph(case_type)

    def run(self, query):
        result = self.app.invoke({
            "query": query
        })

        reranked_docs = result.get("reranked_docs", [])
        jids = [doc.metadata.get("JID") for doc in reranked_docs]


        return {
            "answer": result["answer"],
            "ref_jids": jids
        }