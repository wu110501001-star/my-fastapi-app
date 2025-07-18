import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from llama_index.core.indices import VectorStoreIndex
from google.cloud import storage
import chromadb
import openai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()
import os
import base64

# 從環境變數加載金鑰（base64 解碼）
service_account_key = base64.b64decode(os.getenv('GOOGLE_APPLICATION_CREDENTIALS')).decode('utf-8')

# 將金鑰寫入臨時檔案
with open('/tmp/service-account-key.json', 'w') as f:
    f.write(service_account_key)

# 設定環境變數
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/service-account-key.json"

# 設定認證環境變數（假設你的金鑰在 Google Drive 上）
key_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

# 初始化 FastAPI 應用
app = FastAPI()

# 初始化 OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
# 初始化 Sentence Transformer 模型（支援中文）
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 初始化 ChromaDB 客戶端（GCS 上儲存的向量）
VECTOR_DB_PATH = "gs://mkt_test_ai"  # GCS 或 Google Drive 路徑
client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = client.get_or_create_collection(name="knowledge")

# 初始化 Google Cloud Storage 客戶端
storage_client = storage.Client()

# 使用 Lifespan 事件處理啟動和關閉邏輯
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時檢查資料是否成功加載
    try:
        documents = collection.get()['documents']
        if documents:
            print(f"成功加載 {len(documents)} 篇文件")
        else:
            print("資料庫中尚未儲存任何文件")
    except Exception as e:
        print(f"無法加載資料庫文件: {str(e)}")

    # 在應用啟動後執行其他初始化任務
    yield

# 設定 FastAPI 使用 lifespan
app = FastAPI(lifespan=lifespan)

# 查詢 LlamaIndex 索引
@app.get("/query")
async def query_vector_db(query_text: str, top_n: int = 3):
    try:
        # 用 ChromaDB 進行查詢
        response = collection.query(query_text, n_results=top_n)
        results = []
        for i, doc in enumerate(response['documents']):
            results.append({
                "document": doc,
                "file_name": response['metadatas'][i]['filename'],
                "timestamp": response['metadatas'][i].get('timestamp', '未知'),
                "spotify_link": response['metadatas'][i].get('spotify_link', '無'),
                "apple_link": response['metadatas'][i].get('apple_link', '無')
            })
        return {"query": query_text, "results": results}
    except Exception as e:
        return {"error": f"查詢時發生錯誤: {str(e)}"}
