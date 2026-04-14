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
        你是一位極度嚴謹、具備台灣法學專業的法官助理。
        
        你的任務是：
        閱讀【判決書 ID 對照表】與【參考判決文獻】，針對【使用者問題】整理出一段「可以直接回答問題的法律見解摘要」。
        請先在心中整理各判決的共通法律見解，再輸出最重要的結論。

        【判決書 ID 對照表】（引用時必須使用以下 ID，不得自行更改）：
        {id_reference}

        【結構要求】（極重要）：
        1. 論據先行：先歸納各判決中所提及的法律事實、見解或適用原則，並於句末以中括號引用完整 ID。
        2. 總結在後：在彙整完各判決見解後，以「因此」、「綜上所述」或「準此」等銜接詞，推導出針對問題的最終結論。
        3. 自然語句：請將上述邏輯串聯成一個語法流暢、如學術論文般的段落。

        【執行規則】（請嚴格遵守）：
        1. 證據導向：內容必須 100% 基於提供的文獻。若文獻中未提及相關法律依據，請據實回報，絕不可自行補償外部法律知識。
        2. 精確引用：每一個法律見解或法院判定邏輯，必須在語句末尾以中括號標註其來源 ID。格式範例：...應構成侵權行為 [TPSV,109,台上,3172,20210219,1]。若該見解由多個判決共識得出，請併列標註：...認定具備因果關係 [TPSV,109,台上,99,20201224,1][TPSV,113,台上,1688,20241016,1]。
        3. 論文式寫作：請勿使用列點或標題。請模仿學術論文或判決書理由欄，以「流暢的自然語句」將多篇判決的見解串聯成一個結構完整的段落。
        4. 排除雜訊：不要在輸出中包含引號、原始段落內容或無關的案號文字。
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

        jid_list = []
        context_list = []
        
        for i, d in enumerate(docs[:10]):
            jid = d.metadata.get("JID", "未知字號")
            # 這裡把 JID 存進清單
            jid_list.append(f"- {jid}")
            
            # 內容前面依然標註字號，加強關聯
            context_list.append(f"### [判決字號：{jid}] ###\n{d.page_content}")
            
        # 將 JID 清單轉為字串
        id_reference = "\n".join(jid_list)
        context = "\n\n---\n\n".join(context_list)

        try:
            # 確保這裡的 keys 與 ChatPromptTemplate 裡的一模一樣
            return self.chain.invoke({
                "id_reference": id_reference, 
                "context": context,
                "query": query
            })
        except Exception as e:
            return f"生成答案時發生錯誤: {str(e)}"