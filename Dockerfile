FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl unzip \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY recsys recsys
COPY api api
COPY worker worker
COPY scripts scripts

EXPOSE 8000
# Shared metric files so /metrics aggregates all uvicorn workers.
CMD ["sh", "-c", "rm -rf /tmp/prom && mkdir -p /tmp/prom && PROMETHEUS_MULTIPROC_DIR=/tmp/prom exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4"]
