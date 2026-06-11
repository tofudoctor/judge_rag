# Judge RAG 系統技術說明

## 1. 系統定位

`judge_rag` 是一套以台灣最高法院裁判書為資料來源的本地法律 RAG（Retrieval-Augmented Generation）系統。系統將裁判書建立為 dense 與 sparse 混合索引，收到法律問題後依序完成檢索、重排、可回答性審核、答案生成與引用查核。

核心目標：

- 從大量裁判書找出與問題最相關的法院見解。
- 同時利用語意相似度與精確法律詞彙，提高召回率。
- 優先呈現與問題直接相關的大法庭或重要統一見解裁判。
- 只根據檢索文件生成答案，不任意補入外部法律知識。
- 以完整 JID 標示每個法律見解的來源。
- 在完整模式中檢查文件是否足以回答，以及回答是否受到文件支持。

系統提供兩種搜尋模式：

| 模式 | 流程 | 特性 |
| --- | --- | --- |
| QuickSearch | Retrieve → Rerank → Generate | 延遲較低，適合快速查詢與模型比較 |
| FullSearch | Rewrite → Retrieve → Rerank → DocGrader → Generate → HallucinationGrader → Retry / End | 保護機制較完整，適合重視引用與可靠性的查詢 |

## 2. 專案結構

```text
judge_rag/
├── about.md                         系統架構與技術說明
├── README.md                        安裝與操作方式
├── requirements.txt                Python 套件版本
├── preprocess.py                   建立民事、家事、刑事索引的入口
├── main.py                         Quick / Full benchmark 執行入口
├── indexing/
│   ├── loader.py                   讀取與清理裁判書 JSON
│   ├── chunker.py                  裁判書分段
│   ├── writer.py                   建立 Qdrant collection 與向量儲存
│   └── pipeline.py                 串接載入、切分、向量化與寫入
├── searching/
│   ├── schema.py                   LangGraph 共用狀態定義
│   ├── retriever.py                Hybrid Retrieval 與 RRF 融合
│   ├── reranker.py                 CrossEncoder 重排及法律特化加權
│   ├── query_rewriter.py            法律檢索關鍵字改寫
│   ├── doc_grader.py               文件可回答性審核
│   ├── generator.py                法律摘要生成與 JID 引用規則
│   ├── hallucination_grader.py     答案支持度與引用查核
│   ├── graph.py                    Quick / Full LangGraph 工作流
│   ├── pipeline.py                 對外搜尋介面與結果格式化
│   └── test.py                     Retriever 權重人工驗證腳本
└── utils/
    └── batch.py                    批次迭代工具
```

## 3. 整體資料流

### 3.1 索引建置

```text
裁判書 JSON
  → Loader 清理文字與保留 metadata
  → 建立 1000 / 500 / 300 字元 chunk
  → Ollama dense embedding
  → FastEmbed BM25 sparse embedding
  → 寫入 Qdrant cosine_chunk
```

### 3.2 QuickSearch

```text
使用者問題
  → 原問題 dense / sparse 混合檢索
  → 依 JID 去重
  → Mixedbread Base CrossEncoder rerank
  → 大法庭與新近性加權
  → Top 20
  → Top 10 提供給 Generator
  → 產生附完整 JID 的法律摘要
```

### 3.3 FullSearch

```text
使用者問題
  → LLM 提取法律檢索關鍵字
  → 原問題與關鍵字各做 dense / sparse 檢索
  → RRF 融合並依 JID 去重
  → Mixedbread Large CrossEncoder rerank
  → 大法庭與新近性加權
  → DocGrader 判斷文件是否足以回答
  → Generator 產生附完整 JID 的摘要
  → HallucinationGrader 檢查內容與引用
  → 不通過時帶著檢查理由重新生成，最多 retry 2 次
```

## 4. 資料格式與 Metadata

### 4.1 輸入資料

`loader.py` 預期資料目錄以年份分層，每個檔案為 JSON：

```text
最高法院民事/
├── 2024/
│   ├── TPSV,...json
│   └── ...
└── 2025/
```

系統使用的 JSON 欄位：

| 欄位 | 用途 |
| --- | --- |
| `JID` | 裁判唯一識別、檢索分組及答案引用 |
| `JFULL` | 裁判全文，作為切分與檢索內容 |
| `JTITLE` | 裁判標題 |
| `JYEAR` | 裁判年度 |
| `JCASE` | 裁判字別 |
| `JDATE` | 裁判日期及新近性加權 |
| `JPDF` | 原始 PDF 連結 |

Loader 另外加入 `COURT=最高法院` 與 `TYPE=civil/criminal`。家事資料目前以 `civil` 寫入，因此搜尋時會與民事共同使用 `civil` filter。

### 4.2 文字清理

`clean_text()` 會移除 `\r`、`\n`、`\t`，將連續空白壓成單一空白，再去除頭尾空白。這能降低控制字元對 embedding 與 BM25 的干擾，但會失去原裁判書的段落排版。

## 5. Indexing 模組

### 5.1 `indexing/loader.py`

責任：

- 依年份資料夾由新到舊讀取 JSON。
- 可用 `n_years` 限制只載入最近幾年。
- 清理 `JFULL` 並轉成 LangChain `Document`。
- 保存檢索、篩選與引用所需 metadata。

效果是把原始司法資料統一成 LangChain 與 Qdrant 可使用的格式，並透過 `TYPE` 支援民刑事篩選。

### 5.2 `indexing/chunker.py`

使用 `RecursiveCharacterTextSplitter`，依 `。；！？`、半形與全形空格切分。目前建立：

| Chunk size | Overlap |
| --- | --- |
| 1000 | 200 |
| 500 | 100 |
| 300 | 60 |

大 chunk 保留較完整論證，小 chunk 增加精確命中個別爭點的機會，20% overlap 降低法律句子或推論跨 chunk 被切斷的風險。三種 chunk 全部寫入同一 collection，以 `chunk_size` metadata 區分。

### 5.3 `indexing/writer.py`

負責建立 Qdrant collection 與 LangChain vector store。Collection 包含：

- `dense`：由 Ollama embedding 產生。
- `sparse`：由 `Qdrant/bm25` 產生。

預設 collection 為 `cosine_chunk`。程式也支援 dot、euclid、manhattan，但目前正式流程使用 cosine。

### 5.4 `indexing/pipeline.py`

`BuildPipeline` 串接完整建置流程：

1. 以 `snowflake-arctic-embed2` 建立 Ollama embeddings。
2. 用測試字串取得 dense vector 維度。
3. 載入指定資料目錄。
4. 依序建立 1000、500、300 chunk。
5. 每批處理 256 個 chunk。
6. 使用 chunk 內容建立 UUID5。
7. 寫入前向 Qdrant 查詢 ID，跳過已存在內容。
8. 批次寫入失敗時，降級為逐筆寫入並跳過壞資料。

這可避免重複索引、提高大量資料寫入效率，也避免單一異常 chunk 中止整批資料。注意 UUID 只根據 `page_content` 產生，兩份裁判若有完全相同 chunk 文字，可能共用 ID。

### 5.5 `preprocess.py`

索引建置入口：

| 目錄 | TYPE |
| --- | --- |
| `最高法院民事` | `civil` |
| `最高法院家事` | `civil` |
| `最高法院刑事` | `criminal` |

## 6. Retrieval 模組

### 6.1 `searching/query_rewriter.py`

FullSearch 使用指定 Ollama chat model 將問題改寫為 4 至 6 個法律檢索詞。設定為 `temperature=0`、`seed=0`、`reasoning=False`、`num_predict=128`。

規則要求只輸出一行繁體中文法律詞彙，不得自行加入原問題未提及的法條編號。程式取第一行，移除開頭序號與逗號。效果是補充法律術語、實務爭點與事實特徵，讓 sparse 檢索更容易命中法條與構成要件。

### 6.2 `searching/retriever.py`

Retriever 建立四路查詢：

1. 原問題 dense。
2. 原問題 sparse。
3. 重寫關鍵字 dense。
4. 重寫關鍵字 sparse。

Qdrant 使用 weighted RRF 融合：

| 訊號 | 權重 |
| --- | ---: |
| Query Dense | 5.0 |
| Query Sparse | 2.0 |
| Keywords Dense | 0.5 |
| Keywords Sparse | 1.0 |

Query Dense 負責整體語意；Query Sparse 保留原問題法條與精確詞；Keywords Dense 補充法律術語語意；Keywords Sparse 強化改寫後的精確命中。每路 prefetch 數量是 `target_count * 2`。

檢索使用 `group_by=metadata.JID`、`group_size=1`，每個 JID 只保留最高分 chunk，避免同一裁判佔滿候選結果。`case_type` 有值時會篩選 `metadata.TYPE`。Qdrant RRF 分數存入 `relevance_score`。

| 模式 | target_count | 每路 prefetch |
| --- | ---: | ---: |
| QuickSearch | 100 | 200 |
| FullSearch | 200 | 400 |

## 7. Rerank 模組

### 7.1 使用模型

| 模式 | 模型 |
| --- | --- |
| QuickSearch | `mixedbread-ai/mxbai-rerank-base-v2` |
| FullSearch | `mixedbread-ai/mxbai-rerank-large-v2` |

模型由 `sentence_transformers.CrossEncoder` 載入，輸入是 `[使用者問題, 裁判 chunk]`，`max_length=1024`。

裝置策略：

- 預設 `RERANKER_DEVICE=cuda`。
- CUDA 不可用時改用 CPU。
- CUDA 載入發生 out of memory 時清除 cache 並改用 CPU。
- CUDA 使用 `torch.bfloat16`。
- `PYTORCH_CUDA_ALLOC_CONF` 預設為 `expandable_segments:True`。

### 7.2 法律特化加權

CrossEncoder 分數後加入：

```text
若 JID 含「大」且 raw_score > 5：authority_boost = 5.0
```

這讓已具一定相關性的含「大」裁判更容易進入前順位，但不會只因 JID 含「大」就提升低相關文件。

同批候選也會依 `JDATE` 由新到舊，線性給予 0.0 至 1.0 的 `recency_boost`。

```text
rerank_score = rerank_raw_score + authority_boost + recency_boost
```

metadata 會保存 `rerank_raw_score`、`authority_boost`、`recency_boost`、`rerank_score`。兩種模式最後都保留 Top 20。

### 7.3 `FlashReranker`

檔案另保留 FlashRank 實作，預設 `ms-marco-MultiBERT-L-12`，但目前 QuickSearch 與 FullSearch 沒有使用它。

## 8. Generator 模組

`searching/generator.py` 使用 Ollama `ChatOllama`，模型由 pipeline 傳入，設定為 `temperature=0.2`、`reasoning=False`、`num_predict=32768`。

Generator 最多使用 rerank 後前 10 篇，建立判決書 ID 對照表、有順位標示的參考裁判內容與使用者問題。

回答固定使用：

```markdown
### 結論

### 理由
```

每個法律見解必須附完整 JID，例如 `[TPSV,108,台上大,1636,20210917,1]`。中括號與 JID 之間不得有空格，禁止 `[ TPSV,... ]`、`[TPSV,... ]`、`[ TPSV,...]`。

### 8.1 大法庭優先規則

系統以 `"大" in JID` 判斷含「大」裁判，並在 ID 對照表及參考內容標示為重要統一見解裁判。

生成規則要求：

- 與問題直接相關時，含「大」裁判必須優先閱讀、歸納與引用。
- 含「大」裁判與一般裁判支持同一見解時，以含「大」裁判為核心。
- 含「大」裁判提供抽象原則、一般裁判提供具體適用時，可同時引用。
- 含「大」裁判與問題無直接關係時，不得為了權威性強行引用。
- 見解衝突時優先呈現相關含「大」裁判，並保守說明其他見解。

### 8.2 保守生成與 Retry

Prompt 要求內容完全以提供文件為依據，不得引用對照表以外的 JID，不得自行拼接、縮寫或改寫 JID，文件不足時必須明確說明。模型回傳空內容時，Generator 會追加格式與引用要求後重試一次。

FullSearch retry 不重新檢索，而是保留 conversation history，將 HallucinationGrader 理由傳回 Generator，要求刪除無法支持的見解、修正 JID 並優先使用相關含「大」裁判。

## 9. DocGrader 模組

`searching/doc_grader.py` 判斷文件是否包含足以回答問題的法院見解、法律標準、構成要件、法律效果、證據評價方法或高度相似事實，而不只是看關鍵字。

使用與 FullSearch 相同的 Ollama chat model，設定為 `temperature=0`、`seed=0`、`reasoning=False`、`num_predict=16384`。第一次最多提供前 8 篇、每篇 1500 字元；模型回傳空內容時改成前 5 篇、每篇 1000 字元重試。

輸出：

```text
SCORE: yes 或 no
REASON: 繁體中文理由
```

`yes` 代表至少一篇或多篇合併後足以支撐回答；`no` 代表只有零散關鍵字、程序背景或缺少核心法律見解。沒有文件、空輸出或無法解析時保守回傳 `no`。FullSearch 收到 `no` 時不進入生成，直接輸出 reason 或 fallback。

## 10. HallucinationGrader 模組

`searching/hallucination_grader.py` 檢查：

- 回答是否至少引用一個提供的完整 JID。
- 回答中的 JID 是否完整且出現在清單。
- 法律結論是否可由參考裁判支持。
- 是否加入裁判沒有提供的規則、要件或效果。
- 結論是否與裁判意旨明顯相反。

分數語意：

| 分數 | 意義 |
| --- | --- |
| `no` | 沒有幻覺，可以通過 |
| `yes` | 有幻覺，需要 retry |

輸入範圍：JID Top 10、參考內容 Top 10 每篇最多 900 字元、回答最多 3000 字元。空輸出時縮短為每篇 450 字元、回答 1500 字元再試一次。

送入 LLM 前，程式先以字串比對確認回答至少包含一個 Top 10 JID；完全沒有時直接判有幻覺。注意目前「每一個引用是否都在清單」主要仍由 LLM 審核，尚未以正規表示式對所有引用做完全確定性的集合比對。

## 11. LangGraph 工作流

### 11.1 `searching/schema.py`

`RAGState` 保存 `query`、`keywords`、`case_type`、檢索與 rerank 文件、grader 結果、retry 次數、回答、對話歷史與各階段 timing。

### 11.2 QuickSearch graph

```text
retrieve → rerank → generate → END
```

不做 query rewrite、文件可回答性審核、幻覺檢查或 retry。原問題同時作為 query 與 keywords，適合需要較快反應的情境。

### 11.3 FullSearch graph

```text
rewrite → retrieve → rerank → grade
                               ├─ no  → fail → END
                               └─ yes → generate
                                          ├─ 通過或達上限 → END
                                          └─ hallucination
                                                ├─ no  → END
                                                └─ yes → retry → generate
```

`MAX_RETRY=2`，代表初次生成後最多再生成兩次。Retry 不重新檢索或 rerank。最後一次生成後直接結束，不再執行幻覺檢查。

## 12. Pipeline 對外介面

`searching/pipeline.py` 提供 `QuickSearchPipeline` 與 `FullSearchPipeline`：

```python
from judge_rag.searching.pipeline import FullSearchPipeline

pipeline = FullSearchPipeline(model="gpt-oss:latest")
result = pipeline.run(
    query="法人名譽受損能否請求賠償？",
    case_type="civil",
)
```

共同輸出：

| 欄位 | 說明 |
| --- | --- |
| `answer` | 最終回答 |
| `ref_details` | Top 20 JID 與 rerank score |
| `ref_jids` | JID 清單 |
| `total_time` | 整體耗時 |
| `timing` | 各節點耗時 |

FullSearch 另外提供 `is_relevant`、`doc_grade_reason`、`hallucination_grade`、`hallucination_reason`、`generation_history`。`QuickSearchPipeline` 的類別預設模型是 `gpt-oss:latest`；`FullSearchPipeline` 的類別預設模型是 `gpt-oss:120b`，但 `main.py` benchmark 會明確傳入 `gpt-oss:latest`。實際整合時建議總是明確指定 `model`，避免意外載入不同大小的模型。

## 13. Benchmark 與測試

`main.py` 內建 8 題民事、刑事、非法律及簡化問法 benchmark。執行：

```bash
python3 -m judge_rag.main --quick
python3 -m judge_rag.main --full
```

目前啟用的生成模型是 `gpt-oss:latest`，其他比較模型保留於程式註解。

| 模式 | 輸出檔案 |
| --- | --- |
| QuickSearch | `quicksearch_results.txt` |
| FullSearch | `fullsearch_results.txt` |

結果採 append，包含問題、case type、Top 20 JID 與加權分數、各階段耗時、grader reason、對話式生成紀錄與最終回答。

`searching/test.py` 是 Retriever 權重的人工 smoke test，使用河川浮覆地問題抓取前 5 筆；它不是完整自動化測試套件。

### 13.1 `main.py`

負責命令列參數、benchmark 題目、Quick／Full 模式選擇、計時表格、Top 20 引用輸出，以及將 stdout／stderr 同步追加到結果檔。`--quick` 與 `--full` 為互斥且必填參數。

### 13.2 `utils/batch.py`

`batch_iter()` 使用 `itertools.islice` 將任意 iterable 切成固定大小批次。目前 indexing pipeline 以它建立每批 256 個 chunk，避免一次把所有資料送入 embedding 與 Qdrant。

### 13.3 `requirements.txt`

鎖定 LangChain、LangGraph 相依套件、Ollama、Qdrant client、FastEmbed、FlashRank、Sentence Transformers 與 tqdm 的版本，確保 indexing、retrieval、rerank 及工作流 API 相容。

### 13.4 `__init__.py`

根目錄及 `indexing`、`searching`、`utils` 中的 `__init__.py` 用來宣告 Python package；目前沒有額外初始化邏輯。

## 14. 模型與元件總表

| 用途 | 模型／元件 | 執行位置 |
| --- | --- | --- |
| Dense embedding | `snowflake-arctic-embed2` | Ollama |
| Sparse embedding | `Qdrant/bm25` | FastEmbed |
| Quick reranker | `mixedbread-ai/mxbai-rerank-base-v2` | Sentence Transformers |
| Full reranker | `mixedbread-ai/mxbai-rerank-large-v2` | Sentence Transformers |
| 備用 reranker | `ms-marco-MultiBERT-L-12` | FlashRank，目前未接工作流 |
| Query rewrite | 使用 pipeline 傳入模型；benchmark 為 `gpt-oss:latest` | Ollama |
| DocGrader | 使用 pipeline 傳入模型；benchmark 為 `gpt-oss:latest` | Ollama |
| Generator | 使用 pipeline 傳入模型；benchmark 為 `gpt-oss:latest` | Ollama |
| HallucinationGrader | 使用 pipeline 傳入模型；benchmark 為 `gpt-oss:latest` | Ollama |
| 向量資料庫 | Qdrant | `localhost:6333` |
| 工作流 | LangGraph | Python process |

Mixedbread 與 FastEmbed 模型首次使用時通常會下載至本機 cache，因此第一次執行較慢且需要可存取模型來源。

## 15. 系統可達成的效果

- **Hybrid Retrieval**：兼顧自然語言語意召回與法條、案號、法律詞彙精確命中。
- **多尺度 Chunk**：兼顧完整法律論證與精確段落。
- **CrossEncoder Rerank**：對問題與候選裁判成對判斷，通常比只依向量距離更能辨識直接回答爭點的文件。
- **大法庭優先**：相關且 JID 含「大」的裁判在排序與生成階段都得到優先考量。
- **可回答性控制**：FullSearch 在生成前先判斷文件是否足以回答，降低檢索不到仍硬答的情況。
- **引用與幻覺控制**：答案必須使用完整 JID，FullSearch 再檢查引用與內容，不通過時帶理由修正。
- **可觀測性**：保存各階段耗時、排序結果、grader 理由與生成歷程，方便調參與比較模型。

## 16. 已知限制

1. 本系統是裁判檢索與摘要工具，不等同法律意見，也不保證裁判仍是最新或唯一見解。
2. `rerank_score` 是模型分數加 boost，不是機率，也不能跨不同模型直接比較。
3. 新近性只依日期相對排序，尚未判斷裁判是否被變更、廢棄或因修法失去適用性。
4. JID 含「大」即視為重要裁判；未進一步區分提案裁定、正式大法庭裁定或原因案件終局裁判。
5. 檢索依完整 JID 分組，同一原因案件不同日期或程序裁判仍可能分別出現。
6. 三種 chunk 寫入同一 collection；內容完全相同的 chunk 可能因 UUID 相同而只保留一份。
7. Generator、DocGrader 與 HallucinationGrader 只讀取 Top 10 或更少內容，Top 20 清單不代表全部提供給模型。
8. 裁判內容會依字元數截斷，核心見解若位於 chunk 後段可能未進入 grader context。
9. HallucinationGrader 尚未以 deterministic parser 驗證回答中的每一個 JID。
10. QuickSearch 沒有 grader 或 retry，速度較快但保護較少。
11. FullSearch 的 grader 與 generator 使用同一 chat model，模型偏誤可能同時影響生成與審核。
12. Reranker 首次下載、CPU fallback 或大型 chat model 都可能造成明顯延遲。

## 17. 可調整的重要參數

| 參數 | 位置 | 現值 |
| --- | --- | --- |
| Dense embedding model | `indexing/pipeline.py`、`searching/retriever.py` | `snowflake-arctic-embed2` |
| Sparse model | `writer.py`、`retriever.py` | `Qdrant/bm25` |
| Chunk sizes | `indexing/pipeline.py` | 1000、500、300 |
| Chunk overlap | `indexing/pipeline.py` | 20% |
| Batch size | `indexing/pipeline.py` | 256 |
| RRF weights | `searching/retriever.py` | 5.0、2.0、0.5、1.0 |
| Quick candidates | `searching/graph.py` | 100 |
| Full candidates | `searching/graph.py` | 200 |
| Rerank Top K | `searching/graph.py` | 20 |
| Generator context | `searching/generator.py` | Top 10 |
| Authority boost | `searching/reranker.py` | 5.0 |
| Authority raw threshold | `searching/reranker.py` | > 5 |
| Recency boost max | `searching/reranker.py` | 1.0 |
| Full retry count | `searching/graph.py` | 2 |
| Qdrant URL | `writer.py`、`retriever.py` | `localhost:6333` |

調整檢索、rerank 或 prompt 後，應使用固定 benchmark 比較召回、引用正確性、回答品質與延遲，不宜只觀察單一題目。
