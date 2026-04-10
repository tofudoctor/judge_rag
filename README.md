# Judge RAG 專案環境建置指南

本專案用於建立法律判決的 RAG（Retrieval-Augmented Generation）系統，包含資料載入、切分、embedding，以及寫入向量資料庫。

---

# 🧰 環境需求

* Python 3.10+
* pip / virtualenv（或 conda）
* Docker（用來跑 Qdrant，建議）
* Ollama（用來跑 embedding 模型）

---

# 🚀 Step 1：建立 Python 環境

### 使用 venv（建議）

```bash
python3 -m venv venv
```

啟動環境：

* Windows

```bash
venv\Scripts\activate
```

* macOS / Linux

```bash
source venv/bin/activate
```

---

# 📦 Step 2：安裝套件

```bash
pip install -r requirements.txt
```

---

# 🧠 Step 3：安裝 Ollama（Embedding 模型）

下載並安裝：
```bash
# Windows
irm https://ollama.com/install.ps1 | iex
# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

安裝完成後，拉取模型：

```bash
ollama pull bge-m3
```

測試是否成功：

```bash
ollama run bge-m3
```

---

# 🗄️ Step 4：啟動 Qdrant（向量資料庫）

### 使用 Docker（推薦）

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
docker start qdrant
```

啟動後可在瀏覽器確認：
http://localhost:6333

---

# 執行
在 judge_rag 外面：
```bash
python3 -m judge_rag.main
```

---

# 📌 備註

* 預設使用 embedding model：bge-m3（透過 Ollama）
* 預設 Qdrant：localhost:6333
* 支援自訂年份數(n_years)

---

完成以上步驟，即可成功建立 RAG 環境 🎉
