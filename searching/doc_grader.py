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
        
        system = """你是一個法律文件審核員。
        你的任務是評估【檢索到的法律判決】是否與【使用者的法律問題】相關。
        
        判斷標準：
        1. 只要文件中包含可以回答問題的關鍵字、法條、或類似案例事實，即評為 'yes'。
        2. 如果文件內容完全無關（例如問詐欺卻給了離婚判決），請評為 'no'。
        
        請僅輸出 JSON 格式，包含 'binary_score' 欄位。"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("user", "法律問題: {query} \n\n 檢索到的文件內容: \n\n {documents}")
        ])

    def grade(self, query: str, docs: list) -> str:
        # 將所有文件的內容合併起來供審核
        doc_text = "\n\n".join([d.page_content for d in docs])
        chain = self.prompt | self.llm
        res = chain.invoke({"query": query, "documents": doc_text})
        return res.content