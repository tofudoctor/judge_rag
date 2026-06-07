# searching/doc_grader.py
import json
import re

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class GradeDocuments(BaseModel):
    """判斷檢索到的文件是否與問題相關"""
    binary_score: str = Field(
        description="文件是否與問題相關, 'yes' 或 'no'"
    )
    reason: str = Field(description="判定為 yes 或 no 的簡短理由")

class DocGrader:
    def __init__(self, model="gpt-oss:latest"):
        # temperature 設為 0 以確保判斷穩定
        self.llm = ChatOllama(
            model=model, 
            reasoning=False, 
            temperature=0,
            seed=0,
            # num_ctx=128000,
            num_predict=16384,)
        
        system = """你是一個法律 RAG 文件可回答性審核員。

        你的任務是判斷【目前提供的檢索判決】是否足以支撐 AI 回答【使用者問題】。

        請注意：你的判斷目標不是文件與問題是否有關鍵字相似，而是這批文件是否包含足以回答問題的法律依據、法院見解、構成要件、法律效果或高度相似事實。

        判斷標準請採取「可回答性」而非「完全命中」標準：
        1. 評為 'yes'：
        - 至少有一篇判決直接或間接討論使用者問題的核心法律爭點；或
        - 判決雖未完整回答問題，但包含可支撐保守、有限度回答的法院見解、證據評價、法律標準或事實認定方法；或
        - 多篇判決合併後，可以合理歸納出回答方向。
        2. 評為 'no'：
        - 判決完全沒有觸及使用者問題所需的核心法律爭點；
        - 判決只出現零散關鍵字，且無任何法院判斷可供引用；
        - 判決只有程序事項、當事人主張或背景事實，缺乏法院見解；
        - 即使採取保守、有限度回答，也找不到可引用的判決依據。
        3. 若文件有疑義但看得出與問題核心爭點有實質關聯，傾向評為 'yes'，並在 reason 說明只能支持有限度回答。
        4. 不需要所有判決都相關；只要目前提供的判決中至少有一篇可作為回答依據，即可判 'yes'。
        5. 若使用者問題是抽象法律標準型問題，不要求判決文字完全出現同一問句；只要判決實質討論同一法律標準、相鄰標準或同義爭點，即可判 'yes'。
        6. 對於「認定兩造已盡舉證責任的標準」這類問題，若判決討論舉證責任、證明責任、舉證程度、證明標準、舉證已足、舉證不足、自由心證、經驗法則、論理法則、法院如何認定事實或證據取捨，即屬核心爭點命中。
        7. 不要因為 rerank_score 高就判 'yes'；rerank_score 只代表排序參考，不代表文件足以回答問題。

        reason 欄位請使用繁體中文，簡短說明：
        - 若判 yes，指出哪些完整 JID 足以支撐回答，以及支撐的是哪個核心爭點。
        - 若判 no，說明目前文件缺少哪個回答問題所必需的法律依據或法院見解。
        - reason 必須使用完整 JID，格式為 [完整JID]，例如「[TPSV,108,台上大,1636,20210917,1] 直接討論...」。
        - 不得使用「文件1」、「文件2」、「第一篇」、「第幾篇」等文件編號代稱。

        請只輸出以下兩行，不要輸出其他文字：
        SCORE: yes 或 no
        REASON: 中文簡短理由
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("user", "使用者的問題: {query} \n\n 檢索到的法律判決: \n\n {documents}")
        ])

    def grade(self, query: str, docs: list) -> str:
        if not docs:
            print("    [DocGrader] 無檢索文件，直接判定 no")
            return {"binary_score": "no", "reason": "沒有檢索到可供判斷的判決文件，因此無法根據目前資料回答該問題。"}

        top_score = docs[0].metadata.get("rerank_score", 0)
        print(f"    [DocGrader] Top rerank score: {top_score:.4f}，啟動 LLM 法律爭點審核")

        doc_text = self.build_doc_text(docs, limit=8, chars_per_doc=1500)
        chain = self.prompt | self.llm
        raw_res = chain.invoke({"query": query, "documents": doc_text})
        parsed = self.parse_response(raw_res)

        if parsed.reason.startswith("模型回傳空內容"):
            print("    [DocGrader] 模型回傳空內容，改用較短文件重試一次")
            doc_text = self.build_doc_text(docs, limit=5, chars_per_doc=1000)
            raw_res = chain.invoke({"query": query, "documents": doc_text})
            parsed = self.parse_response(raw_res)

        print("\n" + "-"*30)
        print("【文件相關性審核】")
        print(f"判定結果: {parsed.binary_score}")
        print(f"審核理由: {parsed.reason}")
        print("-"*30 + "\n")

        return {
            "binary_score": self.sanitize_score(parsed.binary_score),
            "reason": parsed.reason,
        }

    def build_doc_text(self, docs, limit=5, chars_per_doc=1800):
        doc_texts = []
        for doc in docs[:limit]:
            metadata = doc.metadata
            score = metadata.get("rerank_score", "N/A")
            jid = metadata.get("JID", "未知JID")
            jdate = metadata.get("JDATE", "未知日期")
            content = doc.page_content[:chars_per_doc]
            doc_texts.append(
                f"### JID: {jid} ###\n"
                f"JDATE: {jdate} | rerank_score: {score}\n"
                f"{content}"
            )
        return "\n\n".join(doc_texts)

    def parse_response(self, response):
        content = getattr(response, "content", response)
        if not isinstance(content, str):
            content = str(content)
        content = content.strip()

        if not content:
            return GradeDocuments(binary_score="no", reason="模型回傳空內容，保守判定為 no")

        score_match = re.search(r"(?im)^\s*(?:SCORE|判定結果|結果)\s*[:：]\s*(yes|no)\b", content)
        if not score_match:
            first_line = content.splitlines()[0].strip() if content.splitlines() else ""
            if re.fullmatch(r"(?i)yes|no", first_line):
                score = first_line.lower()
            else:
                score = None
        else:
            score = score_match.group(1).lower()

        reason_match = re.search(r"(?is)^\s*(?:REASON|理由|審核理由)\s*[:：]\s*(.+)\s*\Z", content, flags=re.MULTILINE)
        if reason_match:
            reason = reason_match.group(1).strip()
        else:
            lines = content.splitlines()
            reason = "\n".join(lines[1:]).strip() if len(lines) > 1 else "模型未提供理由"

        if score:
            return GradeDocuments(binary_score=score, reason=reason or "模型未提供理由")

        # 相容舊格式：如果模型仍回 JSON，也能解析。
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    data = None
            else:
                data = None

        if isinstance(data, dict):
            return GradeDocuments(
                binary_score=str(data.get("binary_score", "no")),
                reason=str(data.get("reason", "模型未提供理由")),
            )

        return GradeDocuments(
            binary_score="no",
            reason=f"模型未回傳可解析的 yes/no，保守判定為 no。原始輸出: {content[:120]}",
        )

    def sanitize_score(self, score):
        """將多種可能的肯定回覆統一轉為 'yes'，其餘皆為 'no'"""
        positive_values = ["yes", "y", "YES", "Yes",
                            "1", 1,
                            True, "True", "true", "T", "t"]
        
        # 先轉字串並去空白，確保比對精準
        if str(score).strip() in [str(v) for v in positive_values]:
            return "yes"
        return "no"
