PY ?= python

.PHONY: all clean features train recommend pipeline test lint help

help:
	@echo "Targets:"
	@echo "  make all        - clean -> features -> train -> recommend (full pipeline)"
	@echo "  make clean      - normalize raw data + derive target (QA artifact)"
	@echo "  make features   - regenerate data/processed/table_tennis_serves_features.csv"
	@echo "  make train      - train models, write models/*.pkl"
	@echo "  make recommend  - rank serves for the example context"
	@echo "  make test       - run pytest"
	@echo "  make lint       - run ruff"

all: clean features train recommend

clean:
	$(PY) -m src.run_pipeline clean

features:
	$(PY) -m src.run_pipeline features

train:
	$(PY) -m src.run_pipeline train

recommend:
	$(PY) -m src.run_pipeline recommend

# Run the whole chain in a single process.
pipeline:
	$(PY) -m src.run_pipeline all

test:
	pytest -q

lint:
	ruff check src tests
