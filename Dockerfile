FROM python:3.11-slim

WORKDIR /app

COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY llm.py server.py ./
COPY web ./web

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]