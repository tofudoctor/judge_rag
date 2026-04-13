# searching/test_retriever_weights.py
from judge_rag.searching.retriever import Retriever

def test_weighted_retrieval():
    print("正在初始化雙 Hybrid Retriever...")
    # 確保你的 Qdrant 已啟動並有對應的 collection
    retriever = Retriever()

    # 測試情境：案情描述較模糊，但關鍵字極度精確
    query = "河川浮覆地原所有權人依民法第767條第1項規定行使物上請求權時，有無消滅時效規定之適用？"
    keywords = "民法第767條 消滅時效 物上請求權 河川浮覆地 所有權人"

    print("\n" + "="*60)
    print(f"測試問題: {query}")
    print(f"關鍵字串: {keywords}")
    print("="*60)

    # 執行檢索
    docs = retriever.retrieve(
        query=query,
        keywords=keywords,
        target_count=5, # 測試前 5 筆即可
    )

    if not docs:
        print("❌ 檢索結果為空，請檢查 Qdrant 資料庫或 Collection 名稱。")
        return

    print(f"\n成功檢索到 {len(docs)} 筆判決：\n")
    
    for i, doc in enumerate(docs):
        score = doc.metadata.get("relevance_score", 0.0)
        jid = doc.metadata.get("JID", "未知字號")
        content_snippet = doc.page_content.replace("\n", "")[:80]
        
        print(f"排名 {i+1} [分數: {score:.4f}]")
        print(f"字號: {jid}")
        print(f"內容: {content_snippet}...")
        print("-" * 30)

    # 驗證邏輯
    top_jid = docs[0].metadata.get("JID", "")
    if "1153" in top_jid:
        print("\n✅ 測試成功：Keywords Sparse 權重發揮作用，精確字號判決排在第一。")
    else:
        print("\n判定：Query Dense 佔據主導，或是關鍵字未精確命中。")

if __name__ == "__main__":
    test_weighted_retrieval()