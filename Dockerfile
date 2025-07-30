# 使用官方 Python 基礎映像
FROM python:3.13

# 設定工作目錄
WORKDIR /app

# 複製依賴檔案
COPY requirements.txt .

# 安裝依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案程式碼
COPY . .

# 啟動應用
CMD ["python", "main.py"]
