.PHONY: install dev pipeline cnn api dashboard test lint docker clean

install:        ## Install runtime dependencies (editable)
	pip install -r requirements.txt && pip install -e .

dev:            ## Install dev tooling
	pip install -r requirements-dev.txt && pip install -e .

pipeline:       ## Full research pipeline
	python -m visionguard.cli pipeline

cnn:            ## Optional ResNet18 transfer learning (needs requirements-cnn.txt)
	python -m visionguard.cli train-cnn

api:            ## FastAPI on :8002 (docs at /docs)
	uvicorn api.main:app --reload --port 8002

dashboard:      ## Streamlit dashboard on :8503
	streamlit run dashboard/app.py --server.port 8503

test:
	pytest --cov=visionguard --cov-report=term-missing

lint:
	ruff check src api dashboard tests && ruff format --check src api dashboard tests

docker:
	docker build -f docker/Dockerfile -t visionguard-ai:latest . && docker run -p 8002:8002 -p 8503:8503 visionguard-ai:latest

clean:
	rm -rf data/processed checkpoints/*.joblib checkpoints/*.pt reports/figures/*.png .pytest_cache .ruff_cache
