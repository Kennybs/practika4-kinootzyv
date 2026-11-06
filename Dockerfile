# Образ сервиса тональности отзывов «Киноотзыв»
# Собирается: docker build -t kinootzyv-service:1.0.0 .
FROM python:3.14-slim

WORKDIR /app

# Сначала зависимости: слой кэшируется и не пересобирается при правке кода
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Затем код и артефакты модели
COPY app/ app/
COPY artifacts/ artifacts/

EXPOSE 8077

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8077/health')"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8077"]