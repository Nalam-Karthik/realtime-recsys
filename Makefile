PY ?= .venv/bin/python

data:
	./scripts/download_data.sh

train:
	$(PY) -m scripts.train

serve:
	rm -rf /tmp/recsys-prom && mkdir -p /tmp/recsys-prom
	PROMETHEUS_MULTIPROC_DIR=/tmp/recsys-prom $(PY) -m uvicorn api.main:app --port 8000 --workers 4

worker:
	$(PY) -m worker.stream_consumer

test:
	$(PY) -m pytest

bench:
	$(PY) -m scripts.benchmark

.PHONY: data train serve worker test bench
