# Judge RAG Local RAG 技術說明

## 一、系統定位

`judge_rag` 是一套以台灣最高法院判決書為資料來源的本地端法律 RAG 系統。它的目標不是單純做全文搜尋，而是把使用者的法律問題轉換成可檢索、可排序、可引用、可驗證的回答流程。

目前系統主要服務兩種使用情境：

1. **QuickSearch**：快速取得相關判決並直接生成回答，適合互動式查詢與速度測試。
2. **FullSearch**：加入問題重寫、文件可回答性審核、幻覺檢查與對話式 retry，適合需要較高可靠性的法律問答。

所有回答都以檢索出的判決為依據，並要求模型使用完整 JID 引用來源，例如 `[TPSV,108,台上大,1636,20210917,1]`。

---

## 二、資料建置流程

資料建置流程由 `judge_rag/indexing` 負責，核心入口是 `BuildPipeline`。

### 1. 判決資料讀取

`loader.py` 會讀取民事、刑事、家事等資料夾中的 JSON 判決書，並轉成 LangChain `Document`。

每份文件會保留重要 metadata，例如：

- `JID`：判決唯一識別碼，後續引用與去重都以它為核心。
- `JDATE`：判決日期，用於 rerank 後的新近性加權。
- `JCASE`、`COURT`、`TYPE` 等欄位：用於資料篩選與檢索分類。

### 2. Chunk 切分

`chunker.py` 會把長判決切成較小片段，預設建立三種 chunk size：

- `1000`
- `500`
- `300`

切分時使用約 `20%` overlap，降低法律見解被切斷的機率。不過目前三種 chunk 都寫入同一個 Qdrant collection：`{distance}_chunk`，並在 metadata 中保留 `chunk_size`。

### 3. 向量化與 Qdrant 寫入

`indexing/pipeline.py` 使用 Ollama 的 `snowflake-arctic-embed2` 作為 dense embedding 模型。

`writer.py` 同時建立：

- dense vector：語意檢索使用。
- sparse vector：`Qdrant/bm25`，關鍵字檢索使用。

每個 chunk 會用內容產生穩定 UUID，寫入前會先檢查 Qdrant 中是否已存在，避免重複寫入。

---

## 三、檢索策略

目前檢索邏輯在 `searching/retriever.py`。

系統採用 **Hybrid Retrieval**，同時送出四種查詢訊號：

1. 原始問題的 dense embedding。
2. 原始問題的 sparse BM25 embedding。
3. 重寫關鍵字的 dense embedding。
4. 重寫關鍵字的 sparse BM25 embedding。

Qdrant 端使用 `RrfQuery` 融合這四路結果，目前權重為：

```text
Query Dense      5.0
Query Sparse     2.0
Keywords Dense   0.2
Keywords Sparse  0.5
```

這個設計讓系統以原始問題語意為主，同時保留法條、案由、專有名詞等精確詞命中的能力。

檢索時會以 `metadata.JID` 分組，每個判決只取分數最高的 chunk，避免同一判決大量重複佔據結果。案件類型則透過 `metadata.TYPE` 篩選，例如 `civil` 或 `criminal`。

---

## 四、Rerank 策略

Rerank 實作在 `searching/reranker.py`。

目前主要使用 Mixedbread reranker：

- QuickSearch：`mixedbread-ai/mxbai-rerank-base-v2`
- FullSearch：`mixedbread-ai/mxbai-rerank-large-v2`

模型透過 `sentence_transformers.CrossEncoder` 載入，預設使用 CUDA；若 CUDA 不可用或載入失敗，程式會依目前環境降級處理。

### 加權規則

rerank 分數不是只看模型原始分數，還會加入兩種法律資料特化加權：

1. **權威性加權**

   若 JID 中包含「大」且原始 rerank 分數高於門檻，代表可能是大法庭或重要統一見解，會額外加分。

2. **新近性加權**

   同一次搜尋結果中，會依 `JDATE` 由新到舊給予最多 `0.5` 的新近性加權。這是為了降低舊判決被後來見解修正或取代時的風險。

最後每份文件會保留：

- `rerank_raw_score`
- `authority_boost`
- `recency_boost`
- `rerank_score`

benchmark 輸出中顯示的 score 是加權後的 `rerank_score`。

---

## 五、QuickSearch 流程

QuickSearch 定義在 `quick_search_graph()`。

流程如下：

```text
Retrieve -> Rerank -> Generate
```

### 1. Retrieve

使用原始問題同時作為 `query` 與 `keywords`，從 Qdrant 抓取最多 `50` 筆判決。

### 2. Rerank

使用 Mixedbread Base reranker 將檢索結果重排，最後取 Top 20。

### 3. Generate

使用指定 Ollama 模型生成回答。生成器會把 Top 10 判決整理成：

- 判決書 ID 對照表。
- 參考判決文獻。
- 使用者問題。

模型必須根據提供判決回答，並使用完整 JID 引用來源。

QuickSearch 不做 doc grader、不做 hallucination grader，也不做 retry，因此速度較快，但品質保護較少。

---

## 六、FullSearch 流程

FullSearch 定義在 `full_search_graph()`。

流程如下：

```text
Rewrite -> Retrieve -> Rerank -> DocGrader -> Generate -> HallucinationGrader -> Retry / End
```

### 1. Query Rewrite

`query_rewriter.py` 會把使用者問題改寫成法律檢索關鍵字，目標是提升 Qdrant 檢索召回率。FullSearch 的檢索會同時使用原始問題與重寫關鍵字。

### 2. Retrieve

FullSearch 從 Qdrant 抓取最多 `100` 筆判決，再交給 reranker。

### 3. Rerank

使用 Mixedbread Large reranker 重排，最後取 Top 20。

### 4. DocGrader

`doc_grader.py` 判斷目前 rerank 後的判決是否足以回答問題。它不只看關鍵字相似，而是判斷是否包含可支撐回答的法院見解、法律標準、構成要件、法律效果或相似事實。

DocGrader 輸出格式是：

```text
SCORE: yes 或 no
REASON: 中文理由
```

理由必須使用完整 JID，不得使用「文件1」、「第一篇」等代稱。

若判定為 `no`，FullSearch 會直接輸出 DocGrader 的理由作為 fallback，避免模型硬答。

### 5. Generate

`generator.py` 使用對話式生成。第一次生成時，模型會收到完整參考判決與問題；若後續 retry，則會把前一輪回答與修正指示作為 conversation history 傳回模型。

生成規則重點：

- 固定輸出 `### 結論` 與 `### 理由`。
- 只能根據參考判決回答。
- 每個法律見解都要有完整 JID。
- 不得引用未出現在判決書 ID 對照表中的 JID。
- 若判決不足，應保守說明不足，不能補外部知識。

### 6. HallucinationGrader

`hallucination_grader.py` 檢查生成回答是否可由提供判決支持。

此處的定義是：

```text
no  = 無幻覺，通過
yes = 有幻覺，需要 retry
```

檢查重點包含：

- 回答是否引用完整 JID。
- 引用的 JID 是否都出現在提供清單中。
- 法律結論是否能被提供判決支持。
- 是否加入判決未支持的法律規則或構成要件。

若模型輸出空內容或無法解析，會保守判定為有幻覺。

### 7. 對話式 Retry

FullSearch 最多會進行 `MAX_RETRY = 2` 次重新生成。最後一次生成後不再做無意義的幻覺檢查，直接輸出結果。

Retry 時，HallucinationGrader 的 reason 會傳給 generator，提示模型修正前一版回答，例如刪除無法支持的見解、補上缺漏 JID、避免短字號或錯誤引用。

---

## 七、Benchmark 設計

`main.py` 內建 7 題 benchmark：

- 3 題民事問題。
- 3 題刑事問題。
- 1 題非法律問題：「怎麼打籃球？」

可用參數：

```bash
python3 -m judge_rag.main --quick
python3 -m judge_rag.main --full
```

輸出檔案分為：

- `quicksearch_results.txt`
- `fullsearch_results.txt`

QuickSearch 表格只統計：

```text
Retrieve | Rerank | Generate | Total
```

FullSearch 表格統計：

```text
Rewrite | Retrieve | Rerank | doc grade | Generate1 | 幻覺grade1 | Generate2 | 幻覺grade2 | Generate3 | Total
```

若某欄位未執行，表格中以 `X` 表示；平均值不計入 `X`。

結果 txt 也會記錄：

- 題號。
- 模式與模型。
- case type。
- Top 20 引用判決 JID。
- rerank score。
- DocGrader reason。
- HallucinationGrader reason。
- 回答摘要。

---

## 八、目前系統特色

1. **Hybrid Retrieval**

   Dense 語意檢索與 sparse BM25 檢索並用，並以 RRF 權重融合。

2. **JID 去重**

   檢索階段以 `metadata.JID` 分組，避免同一判決的多個 chunk 重複洗版。

3. **Quick / Full 雙模式**

   QuickSearch 重速度；FullSearch 重可靠性。

4. **Mixedbread Rerank**

   Quick 使用 base，Full 使用 large，並加入大法庭權威性與判決日期新近性加權。

5. **法律可回答性審核**

   DocGrader 不只判斷語意相似，而是判斷目前文件是否足以支撐法律回答。

6. **引用幻覺防護**

   HallucinationGrader 專門檢查 JID、法律結論與參考判決的一致性。

7. **對話式 Retry**

   Retry 不是重新從零生成，而是把前次回答與幻覺檢查理由放回對話，要求模型針對問題修正。

8. **可比較的 Benchmark**

   目前 benchmark 同時保存時間表、引用判決、grader reason 與回答摘要，便於比較不同模型與不同搜尋模式。

---

## 九、系統限制與注意事項

1. Reranker 分數是模型內部分數，不是 0 到 1 的機率，也不是絕對相關性分數。
2. 大法庭與新近性加權是排序輔助，不能取代法律判斷。
3. FullSearch 的品質較高，但成本與時間顯著高於 QuickSearch。
4. 若 Ollama 模型過大，首次載入可能成為主要耗時來源。
5. 若檢索結果本身缺少核心判決，generator 仍應保守回答，而不是自行補充外部知識。
