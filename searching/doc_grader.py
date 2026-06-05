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
            # num_ctx=32768,
            num_predict=16384,)
        
        system = """你是一個法律文件相關性審核員。
        你的任務是判斷【檢索到的法律判決】是否足以作為【使用者問題】的法律回答依據。

        請用較嚴格的標準判斷：
        1. 評為 'yes'：至少有一篇判決直接討論使用者問題中的核心法律爭點、法條、構成要件、法律效果或高度相似事實。
        2. 評為 'no'：文件只是同一大類法律領域、只出現零散關鍵字、或無法支撐問題的法律結論。
        3. 若使用者問題本身不是法律問題，且判決無法合理回答該問題，必須評為 'no'。
        4. 不要因為 rerank 分數高就評為 'yes'；分數只代表語意排序，最終仍要看法律爭點是否相同。

        reason 欄位請使用繁體中文，簡短說明判斷理由，並指出是否有命中的核心爭點。
        reason 必須直接使用完整 JID 作為依據，例如「TPSV,108,台上大,1636,20210917,1 直接討論...」。
        不得使用「文件1」、「文件2」、「第一篇」、「第幾篇」等文件編號代稱。
        
        請只輸出以下兩行，不要輸出其他文字：
        SCORE: yes 或 no
        REASON: 中文簡短理由，必須包含相關完整 JID
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("user", "使用者的問題: {query} \n\n 檢索到的法律判決: \n\n {documents}")
        ])

    def grade(self, query: str, docs: list) -> str:
        if not docs:
            print("    [DocGrader] 無檢索文件，直接判定 no")
            return {"binary_score": "no", "reason": "無檢索文件"}

        top_score = docs[0].metadata.get("rerank_score", 0)
        print(f"    [DocGrader] Top rerank score: {top_score:.4f}，啟動 LLM 法律爭點審核")

        doc_text = self.build_doc_text(docs, limit=5, chars_per_doc=1800)
        chain = self.prompt | self.llm
        raw_res = chain.invoke({"query": query, "documents": doc_text})
        parsed = self.parse_response(raw_res)

        if parsed.reason.startswith("模型回傳空內容"):
            print("    [DocGrader] 模型回傳空內容，改用較短文件重試一次")
            doc_text = self.build_doc_text(docs, limit=3, chars_per_doc=900)
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
