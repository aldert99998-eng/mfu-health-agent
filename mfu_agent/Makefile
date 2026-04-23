.PHONY: install start stop up down llama run test lint format

install:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"
	$(MAKE) up

# One-shot: Qdrant + llama-server + Streamlit. Idempotent — safe to re-run.
start:
	docker compose up -d
	@bash scripts/start_llama.sh
	.venv/bin/streamlit run app.py --server.port 8504 --server.headless true

# Stop everything we own: Streamlit + llama-server + Qdrant.
stop:
	-pkill -f "streamlit run app.py" || true
	-pkill -f "llama.cpp.*llama-server" || true
	docker compose down

up:
	docker compose up -d

down:
	docker compose down

llama:
	@bash scripts/start_llama.sh

run:
	.venv/bin/streamlit run app.py --server.port 8504 --server.headless true

test:
	.venv/bin/pytest -v

lint:
	.venv/bin/ruff check .
	.venv/bin/mypy .

format:
	.venv/bin/ruff format .
