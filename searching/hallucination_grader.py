# searching/hallucination_grader.py
import json
import re

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class GradeHallucination(BaseModel):
    """判斷回答是否有幻覺或無法被提供文件支持"""
    binary_score: str = Field(
        description="回答是否有幻覺, 'yes' 代表有幻覺需 retry, 'no' 代表無幻覺可通過"
    )
    reason: str = Field(description="判定為 yes 或 no 的簡短理由")

class HallucinationGrader:
    def __init__(self, model="gpt-oss:latest"):
        # 設為 json 模式確保輸出穩定
        self.llm = ChatOllama(
            model=model,
            reasoning=False, 
            temperature=0, 
            # num_ctx=32768,
            num_predict=16384,)
        
        system = """你是一個「寬鬆但可靠」的法律事實查核員。
        你會收到多份【參考判決】以及一份【AI 產出的法律摘要】。
        
        任務：
        核對摘要中的「法律見解」與「判決字號」是否真有出現在提供的參考判決中。
        
        判斷標準：

        判定為 no (無幻覺、通過) 的標準：
        1. 摘要的核心法律結論可以從提供的判決中找到支持。
        2. 摘要中引用的所有判決字號 [ID]，都必須出現在參考判決的標題或內容中。
        3. 容許適度的語句改寫，只要法律邏輯不變。

        判定為 yes (有幻覺、應 retry) 的標準：
        1. 摘要引用了「完全不在」提供文獻中的判決字號。
        2. 摘要虛構了文獻中沒有提到的法律規則。
        3. 摘要結論與判決意旨明顯相反。

        【重要】
        - 不要求逐字一致
        - 不因語句改寫或摘要而判為錯誤
        - 只要「合理支持」即可
        - 先用【提供的判決字號清單】核對摘要引用的判決字號；只要引用字號出現在清單中，就不得說該判決未提供。
        - 若摘要完全沒有引用任何判決 JID，判定為 yes，因為無法確認法律見解來源。
        - 若摘要使用較短字號（例如省略日期或流水號），只要可由清單中的完整 JID 唯一辨識，也視為已提供。

        請在 reason 欄位使用繁體中文說明你的核對過程。
        reason 必須直接使用完整 JID 作為依據，例如「TPSV,108,台上大,1636,20210917,1 支持摘要中的...」。
        不得使用「文件1」、「文件2」、「第一篇」、「第幾篇」等文件編號代稱。

        請只輸出以下兩行，不要輸出其他文字：
        SCORE: yes 或 no
        REASON: 中文簡短理由，必須包含相關完整 JID
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            (
                "user",
                "【提供的判決字號清單】：\n{id_reference}\n\n"
                "【參考判決】：\n\n{documents}\n\n"
                "【AI 產出的摘要】：\n\n{answer}"
            )
        ])

    def grade(self, answer: str, docs: list) -> str:
        answer = answer or ""
        if not answer.strip():
            return self.report_result(
                GradeHallucination(
                    binary_score="yes",
                    reason="AI 回答為空，無法確認是否基於提供判決，判定為有幻覺並要求 retry",
                )
            )

        provided_jids = [d.metadata.get("JID", "") for d in docs[:10] if d.metadata.get("JID")]
        if provided_jids and not any(jid in answer for jid in provided_jids):
            return self.report_result(
                GradeHallucination(
                    binary_score="yes",
                    reason="AI 回答未引用任何提供清單中的完整 JID，無法核對來源，判定為有幻覺並要求 retry",
                )
            )

        id_reference = self.build_id_reference(docs, limit=10)
        doc_text = self.build_doc_text(docs, limit=10, chars_per_doc=900)
        answer_text = answer[:3000]

        chain = self.prompt | self.llm
        raw_res = chain.invoke({"id_reference": id_reference, "documents": doc_text, "answer": answer_text})
        parsed = self.parse_response(raw_res)

        if parsed.reason.startswith("模型回傳空內容"):
            print("    [HallucinationGrader] 模型回傳空內容，保留 JID 清單並改用較短文件重試一次")
            doc_text = self.build_doc_text(docs, limit=10, chars_per_doc=450)
            answer_text = answer[:1500]
            raw_res = chain.invoke({"id_reference": id_reference, "documents": doc_text, "answer": answer_text})
            parsed = self.parse_response(raw_res)

        return self.report_result(parsed)

    def report_result(self, parsed: GradeHallucination):
        print("\n" + "="*50)
        print("【幻覺檢查詳細報告】")
        print(f"判定結果: {parsed.binary_score}")
        print(f"審核理由: {parsed.reason}")
        print("="*50 + "\n")

        return {
            "binary_score": self.sanitize_score(parsed.binary_score),
            "reason": parsed.reason,
        }

    def build_id_reference(self, docs, limit=10):
        return "\n".join(
            f"- {d.metadata.get('JID', '未知字號')}"
            for d in docs[:limit]
        )

    def build_doc_text(self, docs, limit=5, chars_per_doc=1800):
        doc_contents = []
        for d in docs[:limit]:
            jid = d.metadata.get("JID", "未知字號")
            content = d.page_content[:chars_per_doc]
            doc_contents.append(f"### [判決字號：{jid}] ###\n{content}")
        return "\n\n".join(doc_contents)

    def parse_response(self, response):
        content = getattr(response, "content", response)
        if not isinstance(content, str):
            content = str(content)
        content = content.strip()

        if not content:
            return GradeHallucination(binary_score="yes", reason="模型回傳空內容，保守判定為有幻覺")

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
            return GradeHallucination(binary_score=score, reason=reason or "模型未提供理由")

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
            return GradeHallucination(
                binary_score=str(data.get("binary_score", "no")),
                reason=str(data.get("reason", "模型未提供理由")),
            )

        return GradeHallucination(
            binary_score="yes",
            reason=f"模型未回傳可解析的 yes/no，保守判定為有幻覺。原始輸出: {content[:120]}",
        )

    def sanitize_score(self, score):
        """將多種可能的肯定回覆統一轉為 'no'，其餘皆為 'yes'"""
        positive_values = ["no", "n", "NO", "No",
                            "0", 0,
                            False, "False", "false", "F", "f"]
        
        # 先轉字串並去空白，確保比對精準
        if str(score).strip() in [str(v) for v in positive_values]:
            return "no"
        return "yes"
