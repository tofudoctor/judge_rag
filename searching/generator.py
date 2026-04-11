# searching/generator.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class LegalGenerator:
    def __init__(self, model="gpt-oss:latest"):
        self.llm = ChatOllama(
            model=model, 
            temperature=0.2
        )
        
        # 修正：將變數放入 prompt 範本中
        self.system_message = """
        你是一位極度嚴謹、具備台灣法學專業的法官助理。依據【使用者的問題】，從提供的【參考判決文獻】中歸納出準確的法律見解摘要。

        【判決書 ID 對照表】（引用時必須使用以下 ID，不得自行更改）：
        {id_reference}

        【執行規則】(請嚴格遵守)：
        1. 內容必須 100% 基於提供的判決文獻，絕不可使用外部知識或捏造資訊。
        2. 每一個法律見解必須明確說明是哪一篇判決提到的，並於句末加上該判決的 [ID]。
        3. 同一段落若有多篇判決各自提出見解，每個見解單獨一行並各自標註來源 [ID]。
        4. 絕不採納當事人主張：略過「上訴人主張」、「被上訴人則以」等當事人陳述段落。
        5. 鎖定法官心證：尋找以「按」、「惟按」、「查」、「本院認為」等字眼開頭的段落。

        【輸出規則】：
        - 直接開始寫法律見解，不要加任何額外標題或開場白。
        - 不要輸出「原始關鍵段落」或其他額外區塊。
        - 不要加入多餘的問候語或說明。
        """

        # 將 system_message 定義為一個含有變數的 Template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            ("user", "### [參考判決文獻] ###\n{context}\n\n### [使用者的問題] ###\n{query}")
        ])
        
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def generate(self, query: str, docs: list) -> str:
        if not docs:
            return "找不到相關的判決書資料，無法提供回答。"

        # 1. 產生 id_reference 字串
        # 格式：- ID 1: [JID_A] \n - ID 2: [JID_B]
        id_list = []
        context_list = []
        
        for i, d in enumerate(docs):
            jid = d.metadata.get("JID", "未知字號")
            id_list.append(f"- ID {i+1}: [{jid}]")
            
            # 2. 產生 context_text
            content = d.page_content[:800]
            context_list.append(f"【參考判決 {i+1} / 字號：{jid}】\n{content}")
            
        id_reference = "\n".join(id_list)
        context = "\n\n---\n\n".join(context_list)

        # 3. 執行生成，傳入所有在 Prompt 中定義的變數
        try:
            response = self.chain.invoke({
                "id_reference": id_reference, # 這裡成功傳入
                "context": context,
                "query": query
            })
            return response
        except Exception as e:
            return f"生成答案時發生錯誤: {str(e)}"