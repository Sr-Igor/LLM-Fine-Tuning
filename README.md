# 🧠 LLM Fine-Tuner & Synthetic Data Pipeline

Este projeto fornece uma pipeline completa e agnóstica para criar **Assistentes de IA Especializados** a partir de documentos brutos.

A arquitetura foi desenhada para ser utilizada em **qualquer domínio de conhecimento** (Jurídico, Médico, Técnico, Educacional, etc.).

A pipeline automatiza três fases críticas:

1.  **Ingestão de Conhecimento:** Extração de texto de manuais, PDFs e TXTs.
2.  **Geração de Dados Sintéticos:** Uso de LLMs locais (via Ollama) para criar datasets de treino de alta qualidade (Perguntas & Respostas).
3.  **Fine-Tuning Eficiente:** Treinamento de modelos estado-da-arte (Llama 3, Qwen 2, Mistral) usando técnicas de QLoRA/Unsloth.
4.  **Exportação:** Conversão automática para GGUF para execução local leve.

---

## � Funcionalidades

- **Gerador de Dados Sintéticos:** Transforma docs estáticos em pares de instrução/resposta usando modelos como Llama 3 ou Qwen via Ollama.
- **Treinamento Otimizado (Unsloth):** Suporte nativo ao Unsloth para treinos 2x mais rápidos e com 60% menos uso de VRAM.
- **Configuração Centralizada:** Todo o controle via `.env` sem necessidade de alterar código.
- **Suporte a Modelos Modernos:** Compatível com Llama 3.1, Qwen 2.5, Mistral Nemo e Gemma 2.
- **Exportação GGUF:** Geração automática de modelos quantizados prontos para uso no Ollama/LM Studio.

---

## 🛠️ 1. Pré-requisitos

### Hardware

- **Para Geração de Dados:** Qualquer CPU decente (Apple Silicon M1/M2/M3 é excelente) com 16GB+ RAM.
- **Para Treinamento:** GPU NVIDIA com suporte a CUDA (mínimo 8GB VRAM para modelos 8B, ideal 24GB para modelos 32B+). Suporta WSL2 no Windows e Linux nativo.

### Software

- **Python 3.10+**
- **Ollama** (para geração de dados sintéticos). [Instalar Ollama](https://ollama.com/)
- **Gestor de Pacotes:** `uv` (recomendado) ou `pip`.

### Contas

- **Hugging Face:** Token (Write) para baixar modelos base e (opcionalmente) subir seu modelo treinado.
- **WandB (Opcional):** Para monitorar métricas de treino.

---

## ⚙️ 2. Instalação e Configuração

### 2.1. Configuração do Projeto

```bash
# 1. Clone o repositório
git clone <URL_DO_REPOSITORIO> my-llm-project
cd my-llm-project

# 2. Crie e ative o ambiente virtual
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate no Windows

# 3. Instale as dependências
make install
```

> **Nota para usuários Mac:** Se estiver usando apenas para gerar dados, edite o `requirements.txt` e comente as dependências exclusivas da NVIDIA (unsloth, triton, xformers) antes de instalar, para evitar erros.

### 2.2. Variáveis de Ambiente

O coração da customização está no arquivo `.env`.

1.  Copie o exemplo:
    ```bash
    cp .env.example .env
    ```
2.  Edite o `.env` com suas configurações:
    - **HF_TOKEN:** Seu token Hugging Face.
    - **MODEL_NAME:** Modelo base (ex: `unsloth/Qwen2.5-7B-Instruct`).
    - **SYNTHETIC_SYSTEM_INSTRUCTION:** O prompt que define a "persona" do seu assistente. **É aqui que você define se ele é um advogado, médico, suporte técnico, etc.**

---

## 📚 3. Pipeline de Dados (Fase 1)

Nesta etapa, você transforma seus documentos brutos em um dataset de treino. Isso pode ser feito num MacBook ou PC sem GPU potente.

### Passo A: Documentos Fonte

Coloque seus arquivos PDF, TXT ou MD na pasta:
📂 **`data/source_documents/`**

### Passo B: Gerar Dataset

Execute o comando:

```bash
make data
```

**O que acontece nos bastidores:**

1.  O script lê cada arquivo em `data/source_documents/`.
2.  Quebra o texto em "chunks" (pedaços) configuráveis.
3.  Envia cada chunk para o Ollama (usando o modelo definido em `GENERATOR_MODEL`) com um prompt especial para criar perguntas e respostas baseadas naquele texto.
4.  Salva tudo em `data/raw/train_data_synthetic.jsonl`.
5.  Opcionalmente, mescla com dados manuais (`data/raw/manual_rules.jsonl`) se você tiver exemplos "gold standard" feitos à mão.
6.  Gera o dataset final: 📂 **`data/processed/train_dataset_final.jsonl`**.

---

## 🏋️ 4. Treinamento / Fine-Tuning (Fase 2)

Nesta etapa é necessária uma GPU NVIDIA. Se você gerou dados no Mac, mova a pasta do projeto (ou apenas `data/processed/`) para sua máquina de treino (Linux/WSL).

### Passo Único: Treinar

```bash
make train
```

**O que acontece nos bastidores (`main.py`):**

1.  Carrega o modelo base (configurado no `.env`) em 4-bit (QLoRA).
2.  Configura os adaptadores LoRA (apenas uma fração dos pesos é treinada).
3.  Inicia o treino usando os hiperparâmetros do `.env` (Learning Rate, Batch Size, etc.).
4.  Ao final, **funde** os adaptadores LoRA no modelo base.
5.  Converte o modelo resultante para o formato **GGUF** (quantizado q4_k_m, por padrão).
6.  Salva o resultado em `models/<SEU_NOME_DE_MODELO>/`.

---

## 💬 5. Executar e Testar

Após o treino, você terá um arquivo `.gguf`. Você pode usá-lo imediatamente no Ollama.

1.  Garanta que o `Modelfile` na raiz do projeto aponte para o caminho correto do seu novo modelo GGUF.
2.  Execute:

```bash
make run
```

Isso criará o modelo no Ollama e iniciará um chat no terminal.

---

## 📂 Estrutura do Projeto

```text
.
├── config/                 # Módulos de configuração (Pydantic models)
├── data/
│   ├── source_documents/   # [ENTRADA] Seus PDFs/Textos originais
│   ├── raw/                # Dados intermediários gerados
│   └── processed/          # [SAÍDA] Dataset JSONL final para treino
├── models/                 # Onde os modelos .gguf e adaptadores serão salvos
├── src/
│   ├── planuze/            # Código fonte principal (pode ser renomeado para seu projeto)
│   │   ├── utils/          # Loggers e utilitários
│   │   ├── synthetic_data_gen.py
│   │   └── data_handler.py
├── .env                    # Configurações globais (Segredos, Hiperparâmetros)
├── Makefile                # Atalhos para comandos comuns
├── main.py                 # Script principal de treinamento
└── requirements.txt        # Dependências Python
```

---

## 🔧 Personalização Avançada

### Alterando a "Persona"

Para mudar o comportamento do modelo (ex: de Suporte Técnico para Assistente Jurídico), altere a variável `SYNTHETIC_SYSTEM_INSTRUCTION` no arquivo `.env`. Isso mudará como os dados sintéticos são gerados e, consequentemente, como o modelo aprende a responder.

### Ajuste de Hiperparâmetros

Se tiver pouca VRAM (ex: 8GB), ajuste no `.env`:

- `TRAINING_BATCH_SIZE=1`
- `TRAINING_GRAD_ACCUMULATION=4`
- `MAX_SEQ_LENGTH=2048` (ou menor)

---

## 🤝 Contribuição

Sinta-se livre para abrir Issues e Pull Requests. Este projeto é um template base para democratizar o fine-tuning de LLMs.

## 📄 Licença

MIT
