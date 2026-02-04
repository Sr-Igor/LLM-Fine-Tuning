# Makefile for src2 (Planuze LLM Refactored)

PYTHON := python3
VENV := .venv
PYTHON_VENV := $(VENV)/bin/python
PIP := $(PYTHON_VENV) -m pip

# Environment Variables
export PYTHONPATH=.
export MALLOC_NANOSLEEP=0

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m

# ============================================================================
# MAIN COMMANDS
# ============================================================================

.PHONY: help venv mlx\:install cuda\:install mlx\:prepare mlx\:train cuda\:train mlx\:full cuda\:full mlx\:publish cuda\:publish clean

help:
	@echo "$(GREEN)LLM Project $(NC)"
	@echo "Available commands:"
	@echo ""
	@echo "Utility:"
	@echo "  make clean           - Remove heavy artifacts (data/processed, models, adapters)"
	@echo ""
	@echo "Setup:"
	@echo "  make venv            - Create virtual environment"
	@echo ""
	@echo "MLX Commands:"
	@echo "  make mlx:install     - Install dependencies"
	@echo "  make mlx:prepare     - Prepare data"
	@echo "  make mlx:train       - Run training"
	@echo "  make mlx:full        - Full pipeline"
	@echo "  make mlx:publish     - Publish model to HF"
	@echo ""
	@echo "CUDA Commands:"
	@echo "  make cuda:install    - Install dependencies"
	@echo "  make cuda:train      - Run training"
	@echo "  make cuda:full       - Full pipeline"
	@echo "  make cuda:publish    - Publish model to HF"


venv:
	@echo "$(YELLOW)Ensuring virtual environment (.venv) exists...$(NC)"
	@bash scripts/venv.sh
	@echo "$(YELLOW)To activate in your shell, run: source scripts/venv.sh$(NC)"

mlx\:install: venv
	@echo "$(YELLOW)Installing dependencies (Default/Apple)...$(NC)"
	$(PIP) install -r requirements/global.txt
	$(PIP) install -r requirements/apple.txt

cuda\:install: venv
	@echo "$(YELLOW)Installing dependencies (CUDA)...$(NC)"
	$(PIP) install -r requirements/global.txt
	$(PIP) install -r requirements/cuda.txt

mlx\:prepare:
	@echo "$(YELLOW)Preparing data...$(NC)"
	$(PYTHON_VENV) -m src.adapters.cli.main prepare

mlx\:train:
	@echo "$(YELLOW)Starting MLX training...$(NC)"
	TRAINING_BACKEND=mlx $(PYTHON_VENV) -m src.adapters.cli.main train

cuda\:train:
	@echo "$(YELLOW)Starting CUDA training (Unsloth)...$(NC)"
	TRAINING_BACKEND=unsloth $(PYTHON_VENV) -m src.adapters.cli.main train

mlx\:full:
	@echo "$(YELLOW)Running full pipeline (MLX)...$(NC)"
	TRAINING_BACKEND=mlx $(PYTHON_VENV) -m src.adapters.cli.main full

cuda\:full:
	@echo "$(YELLOW)Running full pipeline (CUDA)...$(NC)"
	TRAINING_BACKEND=unsloth $(PYTHON_VENV) -m src.adapters.cli.main full

mlx\:publish:
	@echo "$(YELLOW)Publishing MLX model...$(NC)"
	TRAINING_BACKEND=mlx $(PYTHON_VENV) -m src.adapters.cli.main publish

cuda\:publish:
	@echo "$(YELLOW)Publishing CUDA model...$(NC)"
	TRAINING_BACKEND=unsloth $(PYTHON_VENV) -m src.adapters.cli.main publish

lint:
	@echo "$(YELLOW)Checking code...$(NC)"
	flake8 src/

clean:
	@echo "$(YELLOW)Cleaning artifacts...$(NC)"
	rm -rf data/processed/*
	rm -rf adapters_mlx
	rm -rf adapters_cuda
	rm -rf models/*
	rm -rf outputs/*
	@echo "$(GREEN)Clean complete.$(NC)"
