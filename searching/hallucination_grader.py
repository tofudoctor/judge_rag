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
        
        system = """你是一個「寬鬆但可靠」的法律事實查核員。
        你會收到【參考判決】以及【AI 產出的答案】。
        你的任務是確保【AI 產出的答案】沒有「嚴重的法律事實錯誤」，確保核心結論一致就可以。
        
        查核標準：
        1. 【核心結論】：答案對法律問題的結論（如：有效或無效）是否與判決書一致？
        2. 【禁止編造】：答案是否提到了判決書中「完全不存在」的判決字號或當事人？
        3. 【容忍度】：
        - 只要結論正確且引用了正確的判決字號，即使 AI 的語氣與原文略有不同，也應評為 'yes'。
        - 除非答案公然違背判決書內容，或引用了判決書中完全沒出現的法條，才評為 'no'。
        - 忽略以下情況（仍應評為 yes）：
            - 語句改寫、摘要
            - 多篇判決整理後的統整說法
            - 未涵蓋所有細節
            - 合理推論
        
        請僅以 JSON 格式輸出，包含 'binary_score'。"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("user", "參考判決：\n\n {documents} \n\n AI 產出的答案：\n\n {answer}")
        ])

    def grade(self, answer: str, docs: list) -> str:
        doc_text = "\n\n".join([d.page_content for d in docs])
        chain = self.prompt | self.llm
        res = chain.invoke({"documents": doc_text, "answer": answer})
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