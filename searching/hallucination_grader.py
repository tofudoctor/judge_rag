# searching/hallucination_grader.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class GradeHallucination(BaseModel):
    """判斷回答是否基於文件內容（無幻覺）"""
    binary_score: str = Field(
        description="回答是否基於文件, 'yes' 代表無幻覺, 'no' 代表有幻覺"
    )

class HallucinationGrader:
    def __init__(self, model="gpt-oss:latest"):
        # 設為 json 模式確保輸出穩定
        self.llm = ChatOllama(model=model, format="json", temperature=0)
        
        system = """你是一個嚴格的法律事實查核員。
        你會收到【參考判決】以及【AI 產出的答案】。
        
        你的唯一任務：判斷【AI 產出的答案】中提到的所有事實、法條、結論，是否都能從【參考判決】中找到對應的根據。
        
        評分標準：
        - 'yes': 答案中所有內容均有判決書支撐，無幻覺。
        - 'no': 答案中包含了判決書沒提到的資訊（例如：編造了判決字號、引用了錯誤的法條、或是引用了中國法律）。
        
        請僅以 JSON 格式輸出，包含 'binary_score'。"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("user", "參考判決：\n\n {documents} \n\n AI 產出的答案：\n\n {answer}")
        ])

    def grade(self, answer: str, docs: list) -> str:
        doc_text = "\n\n".join([d.page_content for d in docs])
        chain = self.prompt | self.llm
        res = chain.invoke({"documents": doc_text, "answer": answer})
        return res.content