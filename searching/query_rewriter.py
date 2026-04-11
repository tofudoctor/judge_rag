# searching/qurey_rewriter.py
from langchain_ollama import ChatOllama

class MultiQueryRewriter:

    def __init__(self, model):

        self.llm = ChatOllama(model=model)

    def rewrite(self, query: str) -> str:

        prompt = f"""
        你是一個資深法官助理。請將以下法律問題改寫成 3 個專業的檢索查詢，以確保能從資料庫搜到最相關的判決。

        ### 改寫規則：
        1. 嚴禁輸出任何解釋、前綴（如：維度一、檢索Query）或說明文字。
        2. 嚴禁引入原問題以外的法條編號，但若原問題未提法條，請僅使用法律概念名詞進行改寫。
        3. 每個 Query 應由 3-5 個專業法律詞彙組成，以空格分隔。
        4. 語言統一使用繁體中文。
        5. 僅回傳改寫後的查詢，每行一個，不需序號。

        ### 改寫策略：
        1. 維度一（條文與術語）：聚焦在「{query}」 涉及的法條與標準法律用語。若原問題未提及具體法條，嚴禁自行加入編號。
        2. 維度二（實務爭點）：聚焦在該問題於實務上最常辯論的法律爭點（如時效、構成要件）。
        3. 維度三（特定情境）：聚焦在案件的事實特徵（如特定行為）。

        ### 範例一：
        原問題：裁定准予交付審判之法官，是否需要迴避本案審判？
        1. 准予交付審判 法官迴避 職務參與 自行迴避
        2. 預斷偏見 執行職務有偏頗之虞 程序正義 迴避事由
        3. 法官裁定交付審判 參與本案審判 心證受損 審判公正

        ### 範例二：
        原問題：法人之名譽或信用受侵害，可否依民法第195條第1項規定請求賠償？
        1. 民法第195條第1項 法人名譽權 信用權 損害賠償
        2. 非財產上損害 精神痛苦 慰撫金請求權 權利主體
        3. 公司名譽受損 妨害名譽 信用受侵害 損害認定

        ### 請開始執行：
        原問題：{query}

        """

        response = self.llm.invoke(prompt).content.strip()

       

        # 將字串按行切割，並去除空白行與序號（以防 LLM 不聽話加了序號）

        queries = []

        for line in response.split('\n'):

            clean_line = line.strip()

            # 移除可能出現的 "1. ", "2. ", "- " 等前綴

            if clean_line:

                # 簡單的清理：移除開頭的數字點號或符號

                import re

                clean_line = re.sub(r'^(\d+\.|[\-\*])\s*', '', clean_line)

                queries.append(clean_line)

       

        # 確保只回傳前三個，並回傳 list
        return queries[:3]