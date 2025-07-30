# main.py
import os
import json
import base64
import re
import duckdb
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from google.cloud import storage

from llama_index.core import (
    Settings as LlamaSettings,
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

index = None
embed_model = None  # ✅ 新增：儲存 embed_model 方便查詢
DUCKDB_DIR = "./duckdb_storage"

# --- 初始化 ---
load_dotenv()

# 1. 處理 Google Cloud 金鑰
service_account_key = base64.b64decode(os.getenv('GOOGLE_APPLICATION_CREDENTIALS')).decode('utf-8')
with open('/tmp/service-account-key.json', 'w') as f:
    f.write(service_account_key)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/service-account-key.json"

# 2. 設定 OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("未設定 OPENAI_API_KEY")
LlamaSettings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
embed_model = LlamaSettings.embed_model  # ✅ 新增：額外存成變數 embed_model

# 3. CORS & FastAPI（✅ 關閉 docs 與 redoc）
app = FastAPI(docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://vite-react-qa-ui.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 載入資料與建立索引 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global index

    try:
        print("📥 下載 GCS 向量檔案並建立索引中...")
        storage_client = storage.Client()
        bucket = storage_client.bucket("mkt_test_ai")
        blobs = bucket.list_blobs(prefix="converted_chunks/")

        documents = []
        for blob in blobs:
            if blob.name.endswith(".json"):
                try:
                    record = json.loads(blob.download_as_text())
                    text = record.get("text", "")
                    metadata = record.get("metadata", {})
                    doc_id = record.get("id")
                    if text and doc_id:
                        documents.append(Document(text=text, metadata=metadata, doc_id=doc_id))
                except Exception as e:
                    print(f"❌ 載入 {blob.name} 錯誤：{e}")

        if not os.path.exists(DUCKDB_DIR):
            os.makedirs(DUCKDB_DIR)

        vector_store = DuckDBVectorStore(
            database_path=os.path.join(DUCKDB_DIR, "index.duckdb"),
            table_name="vectors"
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        print(f"✅ 向量索引完成，共 {len(documents)} 筆")
    except Exception as e:
        print(f"❌ 初始化失敗：{str(e)}")

    yield

app.router.lifespan_context = lifespan

# --- 查詢介面 ---
@app.get("/query")
async def query_vector_db(query_text: str, top_n: int = 3):
    try:
        if index is None:
            return {"error": "索引尚未初始化"}
        query_engine = index.as_query_engine(similarity_top_k=top_n)
        response = query_engine.query(query_text)
        return {
            "query": query_text,
            "answer": str(response)
        }
    except Exception as e:
        return {"error": f"查詢錯誤：{str(e)}"}

@app.post("/ask")
async def ask(request: Request):
    try:
        data = await request.json()
        query_text = data.get("question", "").strip()
        if not query_text:
            return JSONResponse(status_code=400, content={"error": "請提供問題"})

        if index is None:
            return JSONResponse(status_code=500, content={"error": "索引尚未初始化"})

        print("📥 查詢問題：", query_text)
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(query_text)
        print("📤 回答：", str(response))

        sources = []
        for node in response.source_nodes:
            meta = node.metadata or {}
            filename = meta.get("filename", "未知")
            timestamp = mxeta.get("timestamp", "無")
            spotify = meta.get("spotify_link", "")
            apple = meta.get("apple_link", "")
            website_link = meta.get("website_link", "")  # ✅ 改成 website_link
            publish_date = meta.get("publish_date", "")
            content = node.text.strip()

            source_text = f"\n📁 {filename}"
            if timestamp and timestamp != "無":
                source_text += f"（時間：{timestamp})"
            source_text += f"\n段落：{content}"
            if spotify:
                source_text += f"\nSpotify：{spotify}"
            if apple:
                source_text += f"\nApple：{apple}"
            if website_link:  # ✅ 改變變數名稱
                source_text += f"\n網站連結：{website_link}"
            if publish_date:
                source_text += f"\n發布日期：{publish_date}"
            source_text += "\n"

            sources.append(source_text)

        full_answer = str(response)
        if sources:
            full_answer += "\n\n📎 引用來源：" + "\n".join(sources)
        return { "answer": full_answer }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"查詢發生錯誤：{str(e)}"})

@app.get("/")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
