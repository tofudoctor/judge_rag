# searching/hallucination_grader.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class GradeHallucination(BaseModel):
    """判斷回答是否基於文件內容（無幻覺）"""
    binary_score: str = Field(
        description="回答是否完全基於提供的判決文獻, 'yes' 代表無幻覺, 'no' 代表有幻覺"
    )
    reason: str = Field(description="判定為 yes 或 no 的簡短理由")

class HallucinationGrader:
    def __init__(self, model="gpt-oss:latest"):
        # 設為 json 模式確保輸出穩定
        self.llm = ChatOllama(model=model, format="json", temperature=0)
        self.structured_llm = self.llm.with_structured_output(GradeHallucination)
        
        system = """你是一個「寬鬆但可靠」的法律事實查核員。
        你會收到多份【參考判決】以及一份【AI 產出的法律摘要】。
        
        任務：
        核對摘要中的「法律見解」與「判決字號」是否真有出現在提供的參考判決中。
        
        判斷標準：

        判定為 yes (無幻覺) 的標準：
        1. 摘要的核心法律結論可以從提供的判決中找到支持。
        2. 摘要中引用的所有判決字號 [ID]，都必須出現在參考判決的標題或內容中。
        3. 容許適度的語句改寫，只要法律邏輯不變。

        判定為 no (有幻覺) 的標準：
        1. 摘要引用了「完全不在」提供文獻中的判決字號。
        2. 摘要虛構了文獻中沒有提到的法律規則。
        3. 摘要結論與判決意旨明顯相反。

        【重要】
        - 不要求逐字一致
        - 不因語句改寫或摘要而判為錯誤
        - 只要「合理支持」即可

        請在 reason 欄位詳細說明你的核對過程（例如：核對了 JID... 與 JID...，確認其見解與文獻相符）。

        請輸出 JSON：
        {{ "binary_score": "yes" 或 "no" }}
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("user", "【參考判決】:\n\n {documents} \n\n 【AI 產出的摘要】：\n\n {answer}")
        ])

    def grade(self, answer: str, docs: list) -> str:
        doc_contents = []
        for d in docs[:10]:
            jid = d.metadata.get("JID", "未知字號")
            doc_contents.append(f"### [判決字號：{jid}] ###\n{d.page_content}")
        doc_text = "\n\n".join(doc_contents)

        chain = self.prompt | self.structured_llm
        res = chain.invoke({"documents": doc_text, "answer": answer})
        # --- 直接在這裡 Print 理由 ---
        print("\n" + "="*50)
        print(f"【🔍 幻覺檢查詳細報告】")
        print(f"判定結果: {res.binary_score}")
        print(f"審核理由: {res.reason}")
        print("="*50 + "\n")
        # ---------------------------
        return self.sanitize_score(res.binary_score)
    
    def sanitize_score(self, score):
        """將多種可能的肯定回覆統一轉為 'yes'，其餘皆為 'no'"""
        positive_values = ["yes", "y", "YES", "Yes",
                            "1", 1, 
                            True, "True", "true", "T", "t"]
        
        # 先轉字串並去空白，確保比對精準
        if str(score).strip() in [str(v) for v in positive_values]:
            return "yes"
        return "no"