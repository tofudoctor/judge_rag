# 測試腳本：直接觀察 Query 生成品質
from .query_rewriter import MultiQueryRewriter

# 1. 實例化
rewriter = MultiQueryRewriter(model="qwen3:0.6b") # 請確保 ollama 已啟動

# 2. 測試不同類型的法律問題
test_samples = [
    "河川浮覆地原所有權人依民法第767條第1項規定行使物上請求權時，有無消滅時效規定之適用？",                      # 民事：損害賠償
    "裁定准予交付審判之法官，是否需要迴避本案審判？",           # 刑事：妨害名譽
    "非原住民為購買取得原住民保留地，與原住民成立借名登記契約，再由該出名人向土地所有權人購買取得土地。其等間之借名登記契約是否有效？"        # 民事：債權債務
]

print("="*50)
print("法官助理 AI：Multi-Query 改寫品質測試")
print("="*50)

for i, raw_q in enumerate(test_samples, 1):
    print(f"\n【案例 {i}】原始輸入：{raw_q}")
    print("-" * 20)
    
    # 執行改寫
    rewritten_queries = rewriter.rewrite(raw_q)
    
    # 印出結果
    for j, q in enumerate(rewritten_queries, 1):
        print(f"  檢索 Query {j}: {q}")