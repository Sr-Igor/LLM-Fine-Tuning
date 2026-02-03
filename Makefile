# Makefile para src2 (Planuze LLM Refactored)

PYTHON := python3
PIP := $(PYTHON) -m pip
VENV := .venv
PYTHON_VENV := $(VENV)/bin/python

# Variáveis de Ambiente
export PYTHONPATH=.
export MALLOC_NANOSLEEP=0

# Cores
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m

# ============================================================================
# COMANDOS PRINCIPAIS
# ============================================================================

.PHONY: help mlx\:install cuda\:install mlx\:prepare mlx\:train cuda\:train mlx\:full cuda\:full

help:
	@echo "$(GREEN)LLM Project $(NC)"
	@echo "Comandos disponíveis:"
	@echo ""
	@echo "MLX Commands:"
	@echo "  make mlx:install     - Instala dependências"
	@echo "  make mlx:prepare     - Prepara dados"
	@echo "  make mlx:train       - Executa treinamento"
	@echo "  make mlx:full        - Pipeline completo"
	@echo ""
	@echo "CUDA Commands:"
	@echo "  make cuda:install    - Instala dependências"
	@echo "  make cuda:train      - Executa treinamento"
	@echo "  make cuda:full       - Pipeline completo"


mlx\:install:
	@echo "$(YELLOW)Instalando dependências (Padrão/Apple)...$(NC)"
	$(PIP) install -r requirements/global.txt
	$(PIP) install -r requirements/apple.txt

cuda\:install:
	@echo "$(YELLOW)Instalando dependências (CUDA)...$(NC)"
	$(PIP) install -r requirements/global.txt
	$(PIP) install -r requirements/cuda.txt

mlx\:prepare:
	@echo "$(YELLOW)Preparando dados...$(NC)"
	$(PYTHON_VENV) -m src.adapters.cli.main prepare

mlx\:train:
	@echo "$(YELLOW)Iniciando treinamento MLX...$(NC)"
	TRAINING_BACKEND=mlx $(PYTHON_VENV) -m src.adapters.cli.main train

cuda\:train:
	@echo "$(YELLOW)Iniciando treinamento CUDA (Unsloth)...$(NC)"
	TRAINING_BACKEND=unsloth $(PYTHON_VENV) -m src.adapters.cli.main train

mlx\:full:
	@echo "$(YELLOW)Executando pipeline completo (MLX)...$(NC)"
	TRAINING_BACKEND=mlx $(PYTHON_VENV) -m src.adapters.cli.main full

cuda\:full:
	@echo "$(YELLOW)Executando pipeline completo (CUDA)...$(NC)"
	TRAINING_BACKEND=unsloth $(PYTHON_VENV) -m src.adapters.cli.main full

lint:
	@echo "$(YELLOW)Verificando código...$(NC)"
	flake8 src/
