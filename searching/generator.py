# searching/generator.py
from langchain_ollama import ChatOllama

class LegalGenerator:
    def __init__(self, model="gpt-oss:latest"):
        self.llm = ChatOllama(
            model=model,
            reasoning=False,
            temperature=0.2,
            # num_ctx=65536,
            num_predict=32768,
        )
        
        # 修正：將變數放入 prompt 範本中
        self.system_message = """
        你是一位極度嚴謹、具備台灣法學專業的法官助理。

        你的任務是：
        閱讀【判決書 ID 對照表】與【參考判決文獻】，針對【使用者問題】整理出一段「可以直接回答問題的法律見解摘要」。
        請先在心中整理各判決的共通法律見解，再輸出最重要的結論與理由。

        【判決書 ID 對照表】（引用時必須使用以下 ID，不得自行更改）：
        {id_reference}

        【輸出格式要求】（極重要）：
        1. 請使用 Markdown 格式輸出。
        2. 請固定使用以下兩個標題：
        - ### 結論
        - ### 理由
        3. 「結論」請先直接回答使用者問題，不要先鋪陳背景。
        4. 「理由」請再詳細整理各判決中所提及的法律事實、見解或適用原則。
        5. 不要使用條列，請使用自然段落。
        6. 每個法律見解或法院判定邏輯，必須在句末以中括號標註來源 ID。

        【內容結構要求】（極重要）：
        1. 結論先行：先用一小段話直接回答使用者問題。
        2. 理由在後：接著說明該結論是如何從參考判決文獻中歸納出來的。
        3. 理由部分應先整理判決共通見解，再說明如何推導到本題結論。
        4. 自然語句：請將上述邏輯串聯成語法流暢、如學術論文般的段落。

        【執行規則】（請嚴格遵守）：
        1. 證據導向：內容必須 100% 基於提供的文獻。若文獻中未提及相關法律依據，請據實回報，絕不可自行補償外部法律知識。
        2. 精確引用：每一個法律見解或法院判定邏輯，必須在語句末尾以中括號標註其來源 ID。格式範例：...應構成侵權行為 [TPSV,109,台上,3172,20210219,1]。若該見解由多個判決共識得出，請併列標註：...認定具備因果關係 [TPSV,109,台上,99,20201224,1][TPSV,113,台上,1688,20241016,1]。
        3. 論文式寫作：請模仿學術論文或判決書理由欄，以流暢的自然語句將多篇判決的見解串聯成結構完整的段落。
        4. 排除雜訊：不要在輸出中包含引號、原始段落內容或無關的案號文字。
        """


    def generate_conversation(
        self,
        query: str,
        docs: list,
        generation_history: list | None = None,
        hallucination_feedback: str = "",
    ) -> tuple[str, list]:
        if not docs:
            answer = "找不到相關的判決書資料，無法提供回答。"
            return answer, (generation_history or []) + [{"role": "assistant", "content": answer}]

        generation_history = list(generation_history or [])
        id_reference, context = self.build_reference_context(docs)
        messages = [("system", self.system_message.format(id_reference=id_reference))]

        if generation_history:
            messages.extend(
                (item["role"], item["content"])
                for item in generation_history
                if item.get("role") in {"user", "assistant"} and item.get("content")
            )
            user_content = self.build_retry_user_message(hallucination_feedback)
            log_content = user_content
        else:
            user_content = self.build_initial_user_message(context, query)
            log_content = f"初次生成：根據參考判決回答問題：{query}"

        messages.append(("user", user_content))

        try:
            answer = self.invoke_for_content(messages)
            if not answer.strip():
                print("    [Generator] 模型回傳空答案，追加明確輸出要求後重試一次")
                retry_messages = messages + [(
                    "user",
                    "你剛剛沒有輸出任何答案。請立刻輸出完整答案，必須包含 ### 結論 與 ### 理由，且每個法律見解都要引用完整 JID。",
                )]
                answer = self.invoke_for_content(retry_messages)
        except Exception as e:
            answer = f"生成答案時發生錯誤: {str(e)}"

        generation_history.append({
            "role": "user",
            "content": user_content,
            "log_content": log_content,
        })
        generation_history.append({
            "role": "assistant",
            "content": answer,
            "log_content": answer,
        })
        return answer, generation_history

    def invoke_for_content(self, messages: list) -> str:
        response = self.llm.invoke(messages)
        answer = getattr(response, "content", response)
        if not isinstance(answer, str):
            answer = str(answer)
        return answer.strip()

    def build_reference_context(self, docs: list) -> tuple[str, str]:
        jid_list = []
        context_list = []

        for d in docs[:10]:
            jid = d.metadata.get("JID", "未知字號")
            jid_list.append(f"- {jid}")
            context_list.append(f"### [判決字號：{jid}] ###\n{d.page_content}")

        return "\n".join(jid_list), "\n\n---\n\n".join(context_list)

    def build_initial_user_message(self, context: str, query: str) -> str:
        return (
            "### [參考判決文獻] ###\n"
            f"{context}\n\n"
            "### [使用者的問題] ###\n"
            f"{query}"
        )

    def build_retry_user_message(self, hallucination_feedback: str) -> str:
        feedback = (hallucination_feedback or "").strip()
        if not feedback:
            feedback = "HallucinationGrader 未提供具體理由，但上一版未通過檢查。"
        return self.build_retry_instruction(feedback)

    def build_retry_instruction(self, hallucination_feedback: str) -> str:
        feedback = (hallucination_feedback or "").strip()
        if not feedback:
            return "無。這是第一次生成，請直接依據參考判決回答。"

        return (
            "你上一次回答被 HallucinationGrader 判定有幻覺或引用不精確。\n"
            f"前次檢查理由：{feedback}\n"
            "請直接修正上一版回答，並只輸出修正後的完整答案。上一版回答只供修正參考；"
            "若上一版與參考判決衝突，必須以參考判決為準。"
            "不得引用未出現在【判決書 ID 對照表】中的 JID；"
            "不得新增參考判決未支持的法律規則；若某個見解缺乏判決支持，請刪除或改寫成保守表述；"
            "所有引用都必須使用完整 JID。"
        )
