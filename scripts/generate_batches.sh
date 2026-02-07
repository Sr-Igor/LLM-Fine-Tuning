#!/bin/bash

# ==========================================
# 🚀 Script de Geração de Múltiplos Batches
# ==========================================
# Este script gera múltiplos batches de dados sintéticos
# com diferentes configurações para maximizar a variação

set -e  # Parar em caso de erro

# Cores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 Gerador de Múltiplos Batches${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Diretório base
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# Definir executável Python
PYTHON_EXEC="${PYTHON_EXECUTABLE:-python}"

# Criar diretório para batches se não existir
mkdir -p data/raw/batches

# Gerar ID de variação única para esta execução (garante novos dados)
VARIATION_ID=$(date +%s)
echo -e "${YELLOW}🆔 Run Variation ID: ${VARIATION_ID}${NC}\n"

# Função para limpar checkpoints
clean_checkpoints() {
    echo -e "${YELLOW}🧹 Limpando checkpoints...${NC}"
    # rm -rf data/raw/synthetic_parts/*
}

# Função para gerar um batch
generate_batch() {
    local batch_num=$1
    local chunk_size=$2
    local overlap=$3
    local model=$4
    local languages=$5
    local output_file=$6
    
    echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}📦 Gerando Batch ${batch_num}${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  Chunk Size: ${chunk_size}"
    echo -e "  Overlap: ${overlap}"
    echo -e "  Model: ${model}"
    echo -e "  Languages: ${languages}"
    echo -e "  Output: ${output_file}\n"
    
    # Exportar variáveis de ambiente
    export SYNTHETIC_CHUNK_SIZE=$chunk_size
    export SYNTHETIC_OVERLAP=$overlap
    export SYNTHETIC_GENERATOR_MODEL=$model
    export SYNTHETIC_LANGUAGES=$languages
    export SYNTHETIC_GENERATOR_MODEL=$model
    export SYNTHETIC_LANGUAGES=$languages
    export SYNTHETIC_OUTPUT_FILE=$output_file
    # Incluir Variation ID para garantir que o cache seja novo e a amostragem varie
    export SYNTHETIC_BATCH_ID="batch_${batch_num}_${VARIATION_ID}"
    
    # Executar geração
    $PYTHON_EXEC src/application/generate_synthetic.py
    
    # Contar exemplos gerados
    local count=$(wc -l < "$output_file" | tr -d ' ')
    echo -e "${GREEN}✅ Batch ${batch_num} completo: ${count} exemplos${NC}"
    
    # Limpar checkpoints para próximo batch
    clean_checkpoints
}

# ==========================================
# CONFIGURAÇÃO DOS BATCHES
# ==========================================

echo -e "${BLUE}Configuração dos batches:${NC}\n"

# Batch 1: Chunks pequenos, foco em português, alta precisão
echo -e "  ${YELLOW}Batch 1${NC}: Chunks pequenos (2500), PT only, alta precisão"

# Batch 2: Chunks médios, multilíngue, balanceado
echo -e "  ${YELLOW}Batch 2${NC}: Chunks médios (3500), PT+EN+ES, balanceado"

# Batch 3: Chunks grandes, multilíngue, contexto amplo
echo -e "  ${YELLOW}Batch 3${NC}: Chunks grandes (4500), PT+EN, contexto amplo"

echo -e "\n${YELLOW}⚠️  Certifique-se de que o Ollama está rodando!${NC}"
echo -e "${YELLOW}    Comando: ollama serve${NC}\n"

read -p "Pressione ENTER para iniciar ou Ctrl+C para cancelar..."

# ==========================================
# GERAÇÃO DOS BATCHES
# ==========================================

# Batch 1: Chunks pequenos, português, precisão
generate_batch \
    1 \
    2500 \
    300 \
    "qwen2.5:14b" \
    "pt" \
    "data/raw/batches/batch_1_small_pt.jsonl"

# Batch 2: Chunks médios, multilíngue
generate_batch \
    2 \
    3500 \
    500 \
    "qwen2.5:14b" \
    "pt,en,es" \
    "data/raw/batches/batch_2_medium_multi.jsonl"

# Batch 3: Chunks grandes, contexto amplo
generate_batch \
    3 \
    4500 \
    700 \
    "qwen2.5:14b" \
    "pt,en" \
    "data/raw/batches/batch_3_large_context.jsonl"

# ==========================================
# MESCLAGEM DOS BATCHES
# ==========================================

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}🔀 Mesclando batches...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# Mesclar todos os batches
cat data/raw/batches/batch_*.jsonl > data/raw/train_data_synthetic.jsonl

# Contar total
total_examples=$(wc -l < data/raw/train_data_synthetic.jsonl | tr -d ' ')

echo -e "${GREEN}✅ Mesclagem completa!${NC}\n"

# ==========================================
# ESTATÍSTICAS FINAIS
# ==========================================

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}📊 Estatísticas Finais${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

batch1_count=$(wc -l < data/raw/batches/batch_1_small_pt.jsonl | tr -d ' ')
batch2_count=$(wc -l < data/raw/batches/batch_2_medium_multi.jsonl | tr -d ' ')
batch3_count=$(wc -l < data/raw/batches/batch_3_large_context.jsonl | tr -d ' ')

echo -e "  Batch 1 (Small PT):      ${GREEN}${batch1_count}${NC} exemplos"
echo -e "  Batch 2 (Medium Multi):  ${GREEN}${batch2_count}${NC} exemplos"
echo -e "  Batch 3 (Large Context): ${GREEN}${batch3_count}${NC} exemplos"
echo -e "  ${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${GREEN}TOTAL:                   ${total_examples} exemplos${NC}\n"

echo -e "${GREEN}✅ Arquivo final: data/raw/train_data_synthetic.jsonl${NC}"
echo -e "${YELLOW}💡 Próximo passo: Executar o treinamento!${NC}\n"

# Opcional: Mostrar preview dos dados
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}👀 Preview dos primeiros exemplos${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

if command -v jq &> /dev/null; then
    head -n 2 data/raw/train_data_synthetic.jsonl | jq -r '.instruction' | head -c 500
    echo -e "\n..."
else
    echo -e "${YELLOW}⚠️  Instale 'jq' para ver preview formatado: brew install jq${NC}"
fi

echo -e "\n${GREEN}🎉 Geração completa!${NC}\n"
