# searching/doc_grader.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class GradeDocuments(BaseModel):
    """判斷檢索到的文件是否與問題相關"""
    binary_score: str = Field(
        description="文件是否與問題相關, 'yes' 或 'no'"
    )

class DocGrader:
    def __init__(self, model="gpt-oss:latest"):
        # temperature 設為 0 以確保判斷穩定
        self.llm = ChatOllama(model=model, format="json", temperature=0)
        
        system = """你是一個法律文件相關性過濾員。
        你的唯一任務是判斷【檢索到的法律判決】與【使用者的法律問題】是否在討論「同一個法律主題」。
        
        判斷標準：
        1. 只要文件中包含與問題相關的法律概念、法條、構成要件、或類似事實，即評為 'yes'。
        2. 不要求文件可以完整回答問題，只要「部分相關」或「可能提供推理依據」即為 'yes'。
        3. 僅在文件完全無關時（例如不同法律領域），才評為 'no'。
        
        請僅輸出 JSON 格式，包含 'binary_score' 欄位。"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("user", "法律問題: {query} \n\n 檢索到的法律判決: \n\n {documents}")
        ])

    def grade(self, query: str, docs: list) -> str:
        # 取得最高分的文件分數 (假設 metadata 中存有 're_score')
        top_score = docs[0].metadata.get("rerank_score", 0)
        
        # 門檻 1：高分信心區 (直接通過)
        if top_score >= 0.75:
            print(f"    [Rerank 高分跳過 LLM] Score: {top_score:.4f}")
            return "yes"
        
        # 門檻 2：極低分區 (直接捨棄)
        if top_score < 0.2:
            print(f"    [Rerank 分數過低直接排除] Score: {top_score:.4f}")
            return "no"

        # 門檻 3：模糊地帶，交給 LLM 判斷是否「法律議題相關」
        print(f"    [Rerank 分數中等 ({top_score:.4f}), 啟動 LLM 議題審核]")

        # 將所有文件的內容合併起來供審核
        doc_text = "\n\n".join([d.page_content for d in docs[:5]])
        chain = self.prompt | self.llm
        res = chain.invoke({"query": query, "documents": doc_text})
        return self.sanitize_score(res.content)

    def sanitize_score(self, score):
        """將多種可能的肯定回覆統一轉為 'yes'，其餘皆為 'no'"""
        positive_values = ["yes", "y", "YES", "Yes",
                            "1", 1, 
                            True, "True", "true", "T", "t"]
        
        # 先轉字串並去空白，確保比對精準
        if str(score).strip() in [str(v) for v in positive_values]:
            return "yes"
        return "no"