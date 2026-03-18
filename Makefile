.PHONY: test lint format typecheck docker dashboard train build-engines clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Quality ──────────────────────────────────────────────────────────────────

test: ## Run the test suite
	pytest tests/ -v --timeout=60

lint: format-check typecheck ## Run all linters

format: ## Auto-format code with black and isort
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

format-check: ## Check formatting without modifying files
	black --check --diff src/ tests/ scripts/
	isort --check-only --diff src/ tests/ scripts/

typecheck: ## Run mypy type checking
	mypy --ignore-missing-imports src/

# ── Training ─────────────────────────────────────────────────────────────────

train: ## Run RL training pipeline
	python scripts/collect_traces.py --source synthetic --frames 10000
	python scripts/train_rl.py

build-engines: ## Export ONNX models and build TensorRT engines
	python scripts/train_detection.py
	python scripts/train_segmentation.py
	python scripts/export_onnx.py
	python scripts/build_trt.py

# ── Runtime ──────────────────────────────────────────────────────────────────

dashboard: ## Launch the Streamlit monitoring dashboard
	streamlit run app/dashboard.py

docker: ## Build the Docker image
	docker build -t lyapunov-edge-inference .

# ── Housekeeping ─────────────────────────────────────────────────────────────

clean: ## Remove generated artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
