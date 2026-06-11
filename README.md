# Judge RAG 操作手冊

本文件說明如何安裝、建立索引，以及執行 QuickSearch 和 FullSearch。系統架構、各模組職責、模型與演算法請參閱 [about.md](about.md)。

## 1. 環境需求

- Python 3.10 以上
- Ollama
- Docker，建議用來執行 Qdrant
- 足夠磁碟空間儲存裁判索引與 reranker 模型
- NVIDIA GPU 可加速 rerank，但不是必要條件

預設 Qdrant 位址為 `http://localhost:6333`。

## 2. 工作目錄

以下命令假設目錄結構為：

```text
tofudoctor/
├── judge_rag/
├── 最高法院民事/
├── 最高法院家事/
└── 最高法院刑事/
```

因 `preprocess.py` 使用相對資料路徑，執行 Python module 時請位於 `tofudoctor` 根目錄，而不是 `judge_rag` 目錄內：

```bash
cd /home/ntu002/Desktop/tofudoctor
```

## 3. 建立 Python 虛擬環境

Linux / macOS：

```bash
python3 -m venv judge_rag/venv
source judge_rag/venv/bin/activate
python -m pip install --upgrade pip
pip install -r judge_rag/requirements.txt
```

Windows PowerShell：

```powershell
py -m venv judge_rag\venv
judge_rag\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r judge_rag\requirements.txt
```

## 4. 安裝並啟動 Ollama

完成 Ollama 安裝後，確認服務可用：

```bash
ollama list
```

下載 dense embedding 模型：

```bash
ollama pull snowflake-arctic-embed2
```

下載目前 benchmark 使用的生成模型：

```bash
ollama pull gpt-oss:latest
```

若改用其他 chat model，也必須先下載：

```bash
ollama pull <model-name>
```

確認模型：

```bash
ollama list
```

## 5. 啟動 Qdrant

建議掛載本地 storage，避免 container 刪除後索引消失：

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$PWD/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

已建立 container 時：

```bash
docker start qdrant
```

確認服務：

```bash
curl http://localhost:6333/collections
```

預設 collection 名稱是 `cosine_chunk`。

## 6. 準備裁判資料

每個資料目錄下應有年份資料夾，年份資料夾內放 JSON：

```text
最高法院民事/2025/*.json
最高法院家事/2025/*.json
最高法院刑事/2025/*.json
```

JSON 至少應提供：

```json
{
  "JID": "TPSV,108,台上大,1636,20210917,1",
  "JFULL": "裁判全文",
  "JTITLE": "裁判標題",
  "JYEAR": "108",
  "JCASE": "台上大",
  "JDATE": "20210917",
  "JPDF": "..."
}
```

## 7. 建立 Qdrant 索引

執行前確認：

- Ollama 正在執行。
- `snowflake-arctic-embed2` 已下載。
- Qdrant 正在 `localhost:6333` 執行。
- 工作目錄是 `tofudoctor`。

使用虛擬環境 Python：

```bash
judge_rag/venv/bin/python -m judge_rag.preprocess
```

若已啟動虛擬環境：

```bash
python -m judge_rag.preprocess
```

程式會依序索引：

1. `最高法院民事`，標記為 `civil`
2. `最高法院家事`，標記為 `civil`
3. `最高法院刑事`，標記為 `criminal`

每份裁判建立 1000、500、300 三種 chunk，寫入 `cosine_chunk`。已存在的 chunk 會依內容 UUID 跳過，因此可重跑以加入新資料；若只修改 metadata 而文字完全不變，舊資料不會自動更新。

## 8. 執行 QuickSearch Benchmark

```bash
judge_rag/venv/bin/python -m judge_rag.main --quick
```

或在已啟動的虛擬環境中：

```bash
python -m judge_rag.main --quick
```

QuickSearch 流程：

```text
Retrieve → Rerank → Generate
```

結果會顯示在終端機，並追加寫入 `judge_rag/quicksearch_results.txt`。

## 9. 執行 FullSearch Benchmark

```bash
judge_rag/venv/bin/python -m judge_rag.main --full
```

或：

```bash
python -m judge_rag.main --full
```

FullSearch 流程：

```text
Rewrite → Retrieve → Rerank → DocGrader
→ Generate → HallucinationGrader → Retry / End
```

結果會顯示在終端機，並追加寫入 `judge_rag/fullsearch_results.txt`。FullSearch 會比 QuickSearch 慢，因為同一題可能多次呼叫 chat model。

## 10. 在 Python 中查詢單一問題

從 `tofudoctor` 根目錄啟動：

```bash
judge_rag/venv/bin/python
```

### QuickSearch

```python
from judge_rag.searching.pipeline import QuickSearchPipeline

pipeline = QuickSearchPipeline(model="gpt-oss:latest")
result = pipeline.run(
    query="河川浮覆地原所有權人的物上請求權有無消滅時效？",
    case_type="civil",
)

print(result["answer"])
print(result["ref_details"])
print(result["timing"])
```

### FullSearch

```python
from judge_rag.searching.pipeline import FullSearchPipeline

pipeline = FullSearchPipeline(model="gpt-oss:latest")
result = pipeline.run(
    query="裁定准予交付審判之法官是否應自行迴避？",
    case_type="criminal",
)

print(result["answer"])
print(result["ref_details"])
print(result["doc_grade_reason"])
print(result["hallucination_reason"])
```

### `case_type` 選擇

| 值 | 範圍 |
| --- | --- |
| `"civil"` | 民事與家事索引 |
| `"criminal"` | 刑事索引 |
| `None` | 不限制類型 |

## 11. 更換生成模型

先下載模型：

```bash
ollama pull <model-name>
```

程式呼叫時傳入：

```python
pipeline = FullSearchPipeline(model="<model-name>")
```

Benchmark 使用模型在 `judge_rag/main.py` 的 `benchmarks` 清單設定。FullSearch 會將同一 model 用於 QueryRewriter、DocGrader、Generator 與 HallucinationGrader。請注意 `FullSearchPipeline()` 類別本身的預設值是 `gpt-oss:120b`，因此自行呼叫時建議像範例一樣明確傳入 `model="gpt-oss:latest"` 或其他已下載模型。

## 12. 指定 Reranker 裝置

預設優先使用 CUDA。強制 CPU：

```bash
RERANKER_DEVICE=cpu judge_rag/venv/bin/python -m judge_rag.main --quick
```

指定 CUDA：

```bash
RERANKER_DEVICE=cuda judge_rag/venv/bin/python -m judge_rag.main --full
```

CUDA 不可用時會自動改用 CPU；載入模型發生 CUDA out of memory 時也會嘗試降級到 CPU。

## 13. 執行 Retriever Smoke Test

Qdrant 已建立索引後：

```bash
judge_rag/venv/bin/python -m judge_rag.searching.test
```

此腳本使用固定河川浮覆地問題抓取前 5 筆，協助觀察 hybrid retrieval 權重是否符合預期。

## 14. 查看結果

```bash
less judge_rag/quicksearch_results.txt
less judge_rag/fullsearch_results.txt
```

每次 benchmark 都會追加內容。若要從空白結果檔重新開始，請先備份，再清空相應檔案。

## 15. 常用檢查

Ollama：

```bash
ollama list
ollama ps
```

Qdrant：

```bash
docker ps
curl http://localhost:6333/collections
```

Python 套件：

```bash
judge_rag/venv/bin/python -m pip check
```

語法檢查：

```bash
judge_rag/venv/bin/python -m compileall judge_rag
```

## 16. 常見問題

### Qdrant connection refused

```bash
docker start qdrant
docker ps
curl http://localhost:6333/collections
```

### Ollama model not found

```bash
ollama pull snowflake-arctic-embed2
ollama pull gpt-oss:latest
```

### 第一次 rerank 很慢

Mixedbread reranker 由 Sentence Transformers 載入，首次執行可能需要下載模型；之後會使用 `./cache` 快取。

### CUDA out of memory

```bash
RERANKER_DEVICE=cpu judge_rag/venv/bin/python -m judge_rag.main --quick
```

也可先改用 QuickSearch，因其 reranker 為 base 版本。

### 搜尋不到資料

依序確認：

1. `preprocess` 已完整執行。
2. Qdrant 有 `cosine_chunk` collection。
3. 查詢 `case_type` 與索引中的 `TYPE` 相符。
4. Dense embedding 模型與建索引時一致。
5. Qdrant volume 使用正確 storage 目錄。

### Top 20 與模型實際讀取數量不同

目前 Pipeline 回傳與 benchmark 顯示 Top 20；Generator 與 HallucinationGrader 使用 Top 10；DocGrader 第一次使用 Top 8。

### 為何 JID 含「大」的裁判排得較前

當 JID 含「大」且 CrossEncoder raw score 大於 5，reranker 增加 5 分；Generator 也將其標示為重要裁判。但若內容與問題沒有直接關係，Prompt 要求不得強行引用。

## 17. 最短操作流程

```bash
cd /home/ntu002/Desktop/tofudoctor

python3 -m venv judge_rag/venv
source judge_rag/venv/bin/activate
pip install -r judge_rag/requirements.txt

ollama pull snowflake-arctic-embed2
ollama pull gpt-oss:latest

docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$PWD/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant

python -m judge_rag.preprocess
python -m judge_rag.main --quick
python -m judge_rag.main --full
```
