# Makefile - Comandos do Projeto Planus

# Variáveis
PYTHON = venv/bin/python
PIP = venv/bin/pip

.PHONY: help install data train run clean

# O comando padrão quando você roda apenas 'make'
help:
	@echo "🤖 COMANDOS DO PLANUS:"
	@echo "  make install  - Instala as dependências (Mac + Nvidia)"
	@echo "  make data     - Gera e processa os dados (Sintético + Manual)"
	@echo "  make train    - Inicia o treinamento (Requer GPU)"
	@echo "  make run      - Roda o modelo no Ollama"
	@echo "  make clean    - Limpa caches e arquivos temporários"

install:
	$(PIP) install -r requirements.txt

# Roda o pipeline de dados completo (Gerar -> Merge)
data:
	$(PYTHON) src/synthetic_data_gen.py
	$(PYTHON) src/dataset_merger.py

# Roda o treino (No Mac isso vai falhar se não tiver configurado o Google Colab/Remote, mas fica o script)
train:
	$(PYTHON) main.py

# Atalho para registrar e rodar no Ollama
run:
	ollama create planus-pro -f Modelfile
	ollama run planus-pro

clean:
	rm -rf __pycache__
	rm -rf data/processed/*
	find . -type f -name "*.pyc" -delete