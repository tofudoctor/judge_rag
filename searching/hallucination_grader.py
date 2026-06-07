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
            # num_ctx=128000,
            num_predict=16384,)
        
        system = """你是一個「寬鬆但可靠」的法律事實查核員。
        你會收到【提供的判決字號清單】、多份【參考判決】以及一份【AI 產出的法律摘要】。

        任務：
        核對摘要中的「法律見解」與「判決字號」是否能被提供的參考判決支持。

        判斷標準：

        判定為 no（無幻覺、通過）的標準：
        1. 摘要的核心法律結論可以從提供的判決中合理支持。
        2. 摘要中引用的每一個 JID，都必須「完整且逐字」出現在【提供的判決字號清單】中。
        3. 容許適度摘要、改寫、合併判決見解，只要法律邏輯沒有明顯改變。

        判定為 yes（有幻覺、應 retry）的標準：
        1. 摘要引用了未出現在【提供的判決字號清單】中的 JID。
        2. 摘要使用JID縮寫，均視為引用不精確，必須判定為 yes。
        3. 摘要完全沒有引用任何提供的完整 JID。
        4. 摘要加入參考判決未支持的法律規則、構成要件、法律效果或結論。
        5. 摘要結論與提供判決意旨明顯相反。
        6. 若無法判斷摘要是否被文獻支持，保守判定為 yes。

        重要規則：
        - 不要求法律見解逐字一致。
        - 不因語句改寫、濃縮或學術化表述而判為 yes。
        - 但引用格式必須嚴格，JID 不得縮寫、不得省略、不得自行改寫。
        - 核對引用時，只能使用【提供的判決字號清單】中的完整 JID。
        - 只要摘要中有任何一個引用 JID 不完整、不在清單中、或使用短字號，即使法律結論合理，也必須判定為 yes。
        - 若摘要完全沒有引用任何判決 JID，判定為 yes，因為無法確認法律見解來源。
        - AI 摘要不需要引用【提供的判決字號清單】中的所有 JID；只要摘要中實際引用的 JID 都完整、逐字、且出現在清單中即可。

        reason 欄位請使用繁體中文說明你的核對過程。
        reason 必須直接使用完整 JID 作為依據，例如「[TPSV,108,台上大,1636,20210917,1] 支持摘要中的...」。
        若判定為 yes，請明確指出是哪一個引用不完整、不在清單中，或哪一個法律見解缺乏提供判決支持。
        不得使用「文件1」、「文件2」、「第一篇」、「第幾篇」等文件編號代稱。

        請只輸出以下兩行，不要輸出其他文字：
        SCORE: yes 或 no
        REASON: 中文簡短理由，必須包含相關完整 JID；若沒有可引用 JID，請明確說明。
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
