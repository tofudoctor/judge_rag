# searching/qurey_rewriter.py
from langchain_ollama import ChatOllama
import re

class QueryRewriter:

    def __init__(self, model="gpt-oss:latest"):

        self.llm = ChatOllama(model=model)

    def rewrite(self, query: str) -> str:

        prompt = f"""
        你是一個資深法官助理。
        請從以下法律問題中提取「核心法律術語」，組成一行檢索關鍵字，以確保能從資料庫搜到最相關的判決。

        ### 改寫規則：
        1. 僅輸出法律詞彙，以空格分隔，嚴禁任何解釋或標點符號。
        2. 嚴禁引入原問題以外的法條編號，但若原問題未提法條，請僅使用法律概念名詞進行改寫。
        3. 總共提取 4-6 個專業法律詞彙，以空格分隔。
        4. 語言統一使用繁體中文。

        ### 改寫策略：
        1. 維度一（條文與術語）：聚焦在「{query}」 涉及的法條與標準法律用語。若原問題未提及具體法條，嚴禁自行加入編號。
        2. 維度二（實務爭點）：聚焦在該問題於實務上最常辯論的法律爭點（如時效、構成要件）。
        3. 維度三（特定情境）：聚焦在案件的事實特徵（如特定行為）。

        ### 範例：
        原問題：裁定准予交付審判之法官，是否需要迴避本案審判？
        准予交付審判 法官迴避 職務參與 自行迴避 審判公正 程序正義

        原問題：法人之名譽或信用受侵害，可否依民法第195條第1項規定請求賠償？
        民法第195條第1項 法人名譽權 信用權 損害賠償 非財產上損害

        ### 請開始執行：
        原問題：{query}

        """

        response = self.llm.invoke(prompt).content.strip()

        # 強力清理：只取第一行，並移除所有數字序號與特殊符號
        first_line = response.split('\n')[0].strip()
        clean_keywords = re.sub(r'^(\d+\.|[\-\*])\s*', '', first_line)
        
        # 再次確保沒有多餘標點（只保留空格分隔詞彙）
        clean_keywords = clean_keywords.replace(',', ' ').replace('，', ' ')
        
        return clean_keywords