.PHONY: install up down run test lint format

install:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"
	$(MAKE) up

up:
	docker compose up -d

down:
	docker compose down

run:
	.venv/bin/streamlit run app.py --server.port 8504 --server.headless true

test:
	.venv/bin/pytest -v

lint:
	.venv/bin/ruff check .
	.venv/bin/mypy .

format:
	.venv/bin/ruff format .
