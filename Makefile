.PHONY: setup setup-ci setup-dev download-data train train-rnn train-encoder train-decoder train-modernbert \
       train-distributed train-quick train-no-checkpoint train-resume serve test test-verbose test-lint lint format check typecheck ci clean docker-build docker-run \
 	   gpu-reset tune tune-rnn tune-encoder tune-decoder tune-standalone \
 	   start-metrics stop-metrics setup-dashboards start-exporter stop-exporter stop-ray \
 	   upload-rnn upload-encoder upload-decoder upload-modernbert upload-router \
 	   download-rnn download-encoder download-decoder download-modernbert \
 	   push-hub pull-hub diagnose diagnose-env diagnose-pipeline \
       router-split router-augment router-train router-evaluate router-pipeline upload-router

# Default device: use auto-detect (cuda > mps > cpu)
DEVICE ?= auto
# Default model type
MODEL ?= encoder
# Default number of lines to load
STOP ?= 300000
# Default run type (new or update)
RUN_TYPE ?= new
# Checkpoint directory — checkpointing is enabled by default for all training.
# To disable checkpointing, use: make train-no-checkpoint
CHECKPOINT_DIR ?= checkpoints/$(MODEL)
# Default metrics exporter port
EXPORTER_PORT ?= 8081
# Sleep prevention — uses systemd-inhibit on Linux to prevent system sleep
# during training. Automatically detected; no action needed.
# To disable: make train INHIBIT_SLEEP=
INHIBIT_SLEEP := $(shell command -v systemd-inhibit >/dev/null 2>&1 && echo "systemd-inhibit --what=sleep --who='training' --why='Model training in progress' --mode=block" || echo "")

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

## Install dependencies with CUDA-enabled PyTorch (for local GPU development)
setup:
	uv sync --extra ray --extra router --extra onnx

## Install dependencies with CPU-only PyTorch (for CI or machines without GPU)
## Installs the CPU-only torch wheel from PyTorch's index, then syncs the rest.
setup-ci:
	uv pip install --index-url https://download.pytorch.org/whl/cpu torch
	uv sync --extra ray --extra router --extra onnx

setup-mps:
	uv sync --extra router --extra onnx

## Install dependencies with dev tools (pytest, ruff, pyright, etc.)
setup-dev:
	uv sync --extra dev --extra ray --extra router --extra onnx

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────

## Download GloVe embeddings and print Yelp dataset instructions
download-data:
	bash scripts/download_data.sh

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

## Train a model (checkpointing enabled by default, saves to checkpoints/<MODEL>/)
train:
	$(INHIBIT_SLEEP) uv run sentimentizer --model $(MODEL) --device $(DEVICE) --run-type $(RUN_TYPE) run --stop $(STOP) \
		--checkpoint-dir $(CHECKPOINT_DIR) --checkpoint-every 1 --save

## Train RNN model
train-rnn:
	$(INHIBIT_SLEEP) uv run sentimentizer --model rnn --device $(DEVICE) --run-type $(RUN_TYPE) run --stop $(STOP) \
		--checkpoint-dir checkpoints/rnn --checkpoint-every 1 --save

## Train Transformer Encoder model (recommended)
train-encoder:
	$(INHIBIT_SLEEP) uv run sentimentizer --model encoder --device $(DEVICE) --run-type $(RUN_TYPE) run --stop $(STOP) \
		--checkpoint-dir checkpoints/encoder --checkpoint-every 1 --save

## Train Transformer Decoder model
train-decoder:
	$(INHIBIT_SLEEP) uv run sentimentizer --model decoder --device $(DEVICE) --run-type $(RUN_TYPE) run --stop $(STOP) \
		--checkpoint-dir checkpoints/decoder --checkpoint-every 1 --save

## Train ModernBERT model
train-modernbert:
	$(INHIBIT_SLEEP) uv run sentimentizer --model modernbert --device $(DEVICE) --run-type $(RUN_TYPE) run --stop $(STOP) \
		--checkpoint-dir checkpoints/modernbert --checkpoint-every 1 --save

## Quick training run with fewer rows for iteration
train-quick:
	$(INHIBIT_SLEEP) uv run sentimentizer --model $(MODEL) --device $(DEVICE) --run-type $(RUN_TYPE) run --stop 5000 \
		--checkpoint-dir $(CHECKPOINT_DIR) --checkpoint-every 1 --save

## Train without checkpointing (override CHECKPOINT_DIR to disable)
train-no-checkpoint:
	$(INHIBIT_SLEEP) uv run sentimentizer --model $(MODEL) --device $(DEVICE) --run-type $(RUN_TYPE) run --stop $(STOP) --save

## Resume training from the latest checkpoint
train-resume:
	$(INHIBIT_SLEEP) uv run sentimentizer --model $(MODEL) --device $(DEVICE) --run-type $(RUN_TYPE) train \
		--checkpoint-dir $(CHECKPOINT_DIR) --resume --save

## Distributed training with Ray Train (1 worker; use train-dist-workers for more)
train-distributed:
	$(INHIBIT_SLEEP) uv run sentimentizer --model $(MODEL) --device $(DEVICE) --run-type $(RUN_TYPE) run --stop $(STOP) \
		--distributed --checkpoint-dir $(CHECKPOINT_DIR) --checkpoint-every 1 --save

## Distributed training with custom worker count (usage: make train-dist-workers WORKERS=4)
train-dist-workers:
	$(INHIBIT_SLEEP) uv run sentimentizer --model $(MODEL) --device $(DEVICE) --run-type $(RUN_TYPE) run --stop $(STOP) \
		--distributed --num-workers $(WORKERS) --checkpoint-dir $(CHECKPOINT_DIR) --checkpoint-every 1 --save

# ──────────────────────────────────────────────
# Individual pipeline stages
# ──────────────────────────────────────────────

## Extract raw reviews into parquet
extract:
	uv run sentimentizer --model $(MODEL) --run-type $(RUN_TYPE) extract --stop $(STOP)

## Tokenize: build/update dictionary and write processed parquet
tokenize:
	uv run sentimentizer --model $(MODEL) --run-type $(RUN_TYPE) tokenize

## Train only (no extract/tokenize, checkpointing enabled)
train-only:
	$(INHIBIT_SLEEP) uv run sentimentizer --model $(MODEL) --device $(DEVICE) --run-type $(RUN_TYPE) train \
		--checkpoint-dir $(CHECKPOINT_DIR) --checkpoint-every 1 --save

# ──────────────────────────────────────────────
# Tuning
# ──────────────────────────────────────────────

## Run tuning workflow with agent-guided loop and model validation
tune:
	uv run sentimentizer --model $(MODEL) tune --save

## Run tuning workflow for RNN
tune-rnn:
	uv run sentimentizer --model rnn tune --save

## Run tuning workflow for Encoder
tune-encoder:
	uv run sentimentizer --model encoder tune --save

## Run tuning workflow for Decoder
tune-decoder:
	uv run sentimentizer --model decoder tune --save

## Run tuning workflow in standalone mode (no LLM agent, single Ray Tune sweep)
tune-standalone:
	uv run sentimentizer --model $(MODEL) tune --mode standalone --save

## Quick tuning test with tiny dataset and few trials
## Usage: make tune-test MODEL=rnn STOP=100 SAMPLES=2
tune-test:
	uv run sentimentizer --model $(MODEL) tune --mode standalone --samples $(SAMPLES) --no-validate --save

## Run tuning workflow with custom samples and iterations (usage: make tune-custom SAMPLES=50 ITERATIONS=10)
tune-custom:
	uv run sentimentizer --model $(MODEL) tune --samples $(SAMPLES) --max-iterations $(ITERATIONS) --save

## Run tuning workflow without model validation
tune-no-validate:
	uv run sentimentizer --model $(MODEL) tune --no-validate --save

# ──────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────

ROUTER_RAW_DATA ?= data/router/reviews.jsonl
ROUTER_DATA_DIR ?= data/router/split

## Split original labeled sources before augmentation
router-split:
	uv run sentimentizer router split --data $(ROUTER_RAW_DATA) --output-dir $(ROUTER_DATA_DIR)

## Augment only the training sources (requires Ollama)
router-augment:
	uv run sentimentizer router augment --seeds $(ROUTER_DATA_DIR)/train.jsonl --output $(ROUTER_DATA_DIR)/train.augmented.jsonl

## Train the sentence encoder router
router-train:
	uv run sentimentizer router train --data $(ROUTER_DATA_DIR)/train.augmented.jsonl --validation-data $(ROUTER_DATA_DIR)/validation.jsonl

## Evaluate on independent held-out sources
router-evaluate:
	uv run sentimentizer router evaluate --model-path models/router --data $(ROUTER_DATA_DIR)/test.jsonl

## Run the full router pipeline (augment -> train -> evaluate)
router-pipeline:
	$(MAKE) router-split
	$(MAKE) router-augment
	$(MAKE) router-train
	$(MAKE) router-evaluate

# ──────────────────────────────────────────────
# Hugging Face Hub (push/pull per model)
# ──────────────────────────────────────────────

## Upload RNN weights + dictionary + model card to Hugging Face Hub
upload-rnn:
	uv run sentimentizer --model rnn hf push

## Upload Encoder weights + dictionary + model card to Hugging Face Hub
upload-encoder:
	uv run sentimentizer --model encoder hf push

## Upload Decoder weights + dictionary + model card to Hugging Face Hub
upload-decoder:
	uv run sentimentizer --model decoder hf push

## Upload ModernBERT weights + configuration to Hugging Face Hub
upload-modernbert:
	uv run sentimentizer --model modernbert hf push

## Upload Router model to Hugging Face Hub
upload-router:
	uv run sentimentizer router push

## Upload all models to Hugging Face Hub
push-hub: upload-rnn upload-encoder upload-decoder upload-modernbert upload-router

## Download RNN weights + dictionary from Hugging Face Hub
download-rnn:
	uv run sentimentizer --model rnn hf pull

## Download Encoder weights + dictionary from Hugging Face Hub
download-encoder:
	uv run sentimentizer --model encoder hf pull

## Download Decoder weights + dictionary from Hugging Face Hub
download-decoder:
	uv run sentimentizer --model decoder hf pull

## Download ModernBERT weights + configuration from Hugging Face Hub
download-modernbert:
	uv run sentimentizer --model modernbert hf pull

## Download all models from Hugging Face Hub
pull-hub: download-rnn download-encoder download-decoder download-modernbert

# ──────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────

## Fast environment check (no torch/ray imports)
diagnose-env:
	uv run sentimentizer diagnose env

## Full pipeline diagnostics (imports ML stack)
diagnose-pipeline:
	uv run sentimentizer --model $(MODEL) diagnose pipeline

## Run diagnostics (defaults to pipeline)
diagnose:
	uv run sentimentizer --model $(MODEL) diagnose pipeline

# ──────────────────────────────────────────────
# Serving
# ──────────────────────────────────────────────

## Start Ray Serve with sentiment + router (no GPU required)
serve:
	uv run --active python -m sentimentizer.serve

# ──────────────────────────────────────────────
# Testing & Linting
# ──────────────────────────────────────────────

## Run all tests
test:
	uv run pytest tests/

## Run all tests with verbose per-test output
test-verbose:
	uv run pytest tests/ -v

## Run lint checks then tests (lint must pass before tests run)
test-lint: lint
	uv run pytest tests/

## Run tests with coverage report
test-cov:
	uv run pytest tests/ --cov=sentimentizer --cov-report=term-missing

## Run only Ray Train tests
test-ray:
	uv run pytest tests/ -k "Ray"

## Lint with ruff
lint:
	uv run ruff check .

## Format with ruff
format:
	uv run ruff format .
	uv run ruff check . --fix --select I

## Run static type checks with pyright
typecheck:
	uv run pyright

## Auto-format, auto-fix, then lint (run after every change)
check:
	uv run ruff format .
	uv run ruff check . --fix
	uv run ruff check .

## Run full CI verification locally (format, lint, typecheck, test, and build)
ci: check typecheck test
	@echo "==> Cleaning old build artifacts..."
	rm -rf dist/
	@echo "==> Verifying package builds successfully..."
	uv build
	@echo "==> CI verification passed successfully!"

# ──────────────────────────────────────────────
# Docker
# ──────────────────────────────────────────────

## Build the Docker image
docker-build:
	docker build -t sentimentizer .

## Run the Docker container
docker-run:
	docker run -p 8000:8000 -p 8265:8265 sentimentizer

# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

## Setup Ray Grafana dashboards using Ray's internal factory
setup-dashboards:
	@mkdir -p metrics/grafana/dashboards
	uv run python scripts/generate_ray_dashboards.py
	@echo "Generated Ray dashboards in metrics/grafana/dashboards/"

## Start the Sentimentizer Prometheus metrics exporter (system, GPU, Ray health)
start-exporter:
	@bash -c '( nohup uv run python sentimentizer/exporter.py --addr 0.0.0.0 >/dev/null 2>&1 & disown )'

## Stop the Sentimentizer metrics exporter
stop-exporter:
	@pgrep -f "[s]entimentizer/exporter.py" | xargs -r kill 2>/dev/null || true

## Start Prometheus, Grafana, and metrics exporter for dashboard metrics
start-metrics: setup-dashboards
	@cd metrics && docker compose up -d
	@echo "Restarting Grafana to load newly generated dashboards..."
	@cd metrics && docker compose restart grafana
	@echo "Stopping old metrics exporter if running..."
	@pgrep -f "[s]entimentizer/exporter.py" | xargs -r kill 2>/dev/null || true
	@echo "Starting metrics exporter (port 8081)..."
	@bash -c '( nohup uv run python sentimentizer/exporter.py --addr 0.0.0.0 >/dev/null 2>&1 & disown )'
	@sleep 2
	@echo "All metrics services running. Grafana: http://localhost:3000 (admin/admin)"

## Stop Prometheus, Grafana, and metrics exporter
stop-metrics:
	@echo "Stopping exporter..."
	@pgrep -f "[s]entimentizer/exporter.py" | xargs -r kill 2>/dev/null || true
	@echo "Stopping Docker containers (this may take a few seconds)..."
	@cd metrics && docker compose down -t 10 || true
	@echo "Metrics stopped."

# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────

## Force stop all local Ray instances
stop-ray:
	uv run ray stop --force

## Remove generated data files, checkpoints, Python caches, and ALL metrics state
clean: stop-metrics stop-ray
	@echo "==> Cleaning generated data files..."
	rm -rf sentimentizer/data/review_data.parquet
	rm -rf sentimentizer/data/review_data_raw.parquet
	rm -rf sentimentizer/data/weights.pth
	rm -rf checkpoints/
	rm -rf tuning_results/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "==> Cleaning Ray temporary files..."
	rm -rf /tmp/ray/*
	@echo "==> Cleaning Ray Tune results..."
	rm -rf ~/ray_results/*
	@echo "==> Cleaning persisted training metrics..."
	rm -f /tmp/sentimentizer_metrics/*_metrics.json
	@echo "==> Cleaning Prometheus TSDB data..."
	@docker volume rm metrics_prometheus-data 2>/dev/null || true
	@docker volume prune -f 2>/dev/null || true
	@echo "Clean complete. Run 'make start-metrics' to start fresh."

## Clean only Ray-related files and logs
clean-ray: stop-ray
	rm -rf /tmp/ray/*
	rm -rf ~/ray_results/*

# Fix NVIDIA driver/library mismatch without rebooting
gpu-reset:
	@echo "==> Stopping services that might use the GPU..."
	-sudo systemctl stop ollama
	-sudo systemctl stop docker
	@echo "==> Unloading NVIDIA kernel modules..."
	-sudo rmmod nvidia_drm
	-sudo rmmod nvidia_modeset
	-sudo rmmod nvidia_uvm
	-sudo rmmod nvidia
	@echo "==> Reloading NVIDIA kernel modules..."
	sudo modprobe nvidia
	sudo modprobe nvidia_uvm
	@echo "==> Restarting services..."
	-sudo systemctl start docker
	-sudo systemctl start ollama
	@echo "==> Verifying NVML initialization..."
	nvidia-smi
