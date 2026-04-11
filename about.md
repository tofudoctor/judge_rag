# 法律判決檢索與問答系統 (Legal-RAG) 技術文件

## 一、 系統概述
本系統為一套基於 **RAG (Retrieval-Augmented Generation)** 架構開發的法律專業工具。系統旨在從海量的法院判決書中，透過語意搜尋精準定位相關案例，並結合大型語言模型 (LLM) 生成具備法律依據與引用出處的分析建議。

系統架構主要分為兩大核心模組：
1.  **資料建置流程 (Indexing Pipeline)**：負責法律文件的預處理、向量化與持久化儲存。
2.  **查詢與回答流程 (Searching & Generation Pipeline)**：負責問題優化、多維度檢索、重排序及具備防護機制的回饋生成。

---

## 二、 資料建置流程 (Indexing)

本階段目標是將原始判決資料轉化為可供高效檢索的向量資料庫。

### 1. 資料讀取與預處理 (`loader.py`)
* **功能描述**：解析 JSON 格式之判決文件，支援特定年份 (`n_years`) 篩選。
* **資料清洗**：自動移除換行符號、多餘空格等噪音資訊。
* **元數據架構**：每筆 Document 均包含完整元數據（如 JID 判決字號、JCASE 案由、COURT 審理法院等），確保後續篩選的靈活性。

### 2. 文件切分策略 (`chunker.py`)
系統採用 `RecursiveCharacterTextSplitter` 進行層次化切分，並針對中文語境進行優化：
* **分割符號**：優先以「。；！？ 」作為切分點。
* **多尺度設計**：同步產生三種不同粒度的文本塊：
    * **Large (1000 tokens)**：保留較完整的上下文背景。
    * **Medium (500 tokens)**：平衡語意與檢索精確度。
    * **Small (300 tokens)**：鎖定極其精確的法律見解。
* **重疊機制**：固定 20% 的 Overlap 以確保語意連貫性。

### 3. 向量化與儲存 (`writer.py`)
* **Embedding 模型**：使用 `bge-m3` 模型進行文本向量化。
* **向量資料庫**：採用 **Qdrant**，並針對不同 Chunk Size 建立對應的 Collection。
* **寫入機制**：支援 Batch 批次寫入（Size=32）與 SHA1 唯一碼去重，確保資料一致性。

---

## 三、 查詢與檢索系統 (Searching)

### 1. 查詢重寫機制 (`query_rewriter.py`)
為克服使用者原始問題描述模糊的問題，系統會將其轉化為 3 組具備法律專業特徵的查詢關鍵字。
> **範例**：若提問「法人名譽權」，重寫器會自動補強「民法195條」、「非財產損害」、「慰撫金」等關鍵詞。

### 2. 多維度檢索與去重 (`retriever.py`)
系統執行 **Multi-Collection Retrieval**：
* 同時檢索 1000/500/300 三種粒度的 Collection。
* 透過 **Metadata Filter** 鎖定案件類型（如民事或刑事）。
* **去重機制**：以 JID（判決字號）為基準進行聚合，僅保留相似度分數最高的文件片段。

### 3. 重排序與篩選 (`reranker.py` & `doc_grader.py`)
* **Reranking**：利用 `FlashRank` (模型：ms-marco-MultiBERT) 進行精準二次排序。
* **可回答性判斷**：透過 `Doc Grader` 模組初步過濾無關資料，若現有文件無法回答則啟動重試機制。

---

## 四、 生成與防護機制 (Generation & Guardrails)

### 1. 具引用依據的生成 (`generator.py`)
系統僅根據檢索到的判決片段進行回答，並嚴格遵守以下準則：
* **聚焦法院見解**：自動過濾當事人主張，僅摘錄法院之判斷理由。
* **來源標註 (Citation)**：在每一段結論後方明確標註對應之 [JID]，確保答案可追溯。

### 2. 幻覺檢測與驗證 (`hallucination_grader.py`)
為確保法律建議的嚴肅性，系統內建 **Hallucination Grader**：
* **一致性檢查**：比對生成的答案是否完全基於檢索到的文本。
* **異常處理**：若偵測到模型捏造內容，將自動觸發重新生成或報錯機制。

---

## 五、 流程編排 (Graph Pipeline)

本系統使用 **LangGraph** 建構動態流程圖，支援複雜的邏輯跳轉：

| 模式 | 流程說明 |
| :--- | :--- |
| **快速搜尋 (Quick)** | `Retrieve` → `Rerank` → `Generate` |
| **完整搜尋 (Full)** | `Rewrite` → `Retrieve` → `Rerank` → `Grader` → `Generate` → `Hallucination Check (Retry if needed)` |

* **重試機制**：針對低品質檢索或生成結果，系統提供最多 2 次自動重試（Retry），確保最終輸出的可靠性。

---

## 六、 系統特色總結

* 🚀 **高精準度**：結合 Multi-Query 與 Multi-Chunk 檢索策略，大幅提升召回率。
* ⚖️ **法律專業化**：透過 Query Rewrite 強化法律詞彙應用，並專注於「法院見解」的提取。
* 🛡️ **雙重驗證**：Doc Grader 與 Hallucination Grader 構成防護網，有效抑止 AI 幻覺。
* 🔍 **透明性**：完整的 Citation 機制，讓每一句法律分析都有據可查。

---