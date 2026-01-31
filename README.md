# 🧠 Planus Finetuner - Guia de Utilização

Este projeto automatiza a criação de um Assistente de IA Especializado (Planus) para o ERP Planuze. Ele utiliza documentos PDF/TXT para gerar conhecimento e treina modelos (Llama 3.1 ou Qwen 2.5) para responder perguntas técnicas seguindo regras de negócio estritas.

---

## 🛠️ 1. Pré-requisitos

Antes de começar, certifique-se de que você possui:

- **Para Gerar Dados:** Qualquer computador (Mac, Windows, Linux) com **Python 3.10+** e **Ollama** instalado.
- **Para Treinar (Fine-Tuning):** Um servidor ou PC com **GPU NVIDIA** (mínimo 8GB VRAM, ideal 24GB RTX 3090/4090) rodando Linux ou WSL2.
- **Contas:**
  - **Hugging Face:** Token com permissão de leitura/escrita (para baixar/subir modelos).
  - **WandB (Opcional):** Para acompanhar gráficos de treino em tempo real.

---

## 🚀 2. Configuração Inicial

### 2.1. Clone e Ambiente Virtual

```bash
git clone <URL_DO_REPOSITORIO>
cd planuze-llm

# Crie o ambiente virtual (Python 3.10 recomendado)
python3.10 -m venv venv
source venv/bin/activate
```

### 2.2. Instalação de Dependências

O projeto possui um **Makefile** para facilitar os comandos.

- **Se estiver no Mac (apenas geração de dados):**
  Abra o `requirements.txt` e comente as linhas abaixo de "DEPENDÊNCIAS EXCLUSIVAS NVIDIA". Execute:

  ```bash
  make install
  ```

- **Se estiver no Linux/GPU (para treino):**
  Execute direto:
  ```bash
  make install
  ```

### 2.3. Variáveis de Ambiente

Configure as variáveis copiando o exemplo:

```bash
cp .env.example .env
```

Edite o arquivo `.env`:

- `HF_TOKEN`: Seu token do Hugging Face.
- `OLLAMA_HOST`: URL do Ollama (padrão `http://localhost:11434`).
- Configurações de diretórios (se quiser alterar os padrões).

---

## 📚 3. Fase de Dados (Rodar no Mac/Local)

Transforme PDFs brutos em um dataset JSONL limpo para o treino.

### Passo A: Ingestão de Documentos

Coloque seus manuais, políticas e documentos técnicos (PDF ou TXT) na pasta:
📂 **`data/source_documents/`**

### Passo B: Geração e Processamento

Para gerar os dados sintéticos via Ollama, fundir com dados manuais (se houver) e validar o dataset, apenas execute:

```bash
make data
```

> **O que esse comando faz?**
>
> 1. Executa `src/synthetic_data_gen.py`: Lê PDFs e usa o Ollama para criar pares Pergunta/Resposta.
> 2. Executa `src/dataset_merger.py`: Junta os dados sintéticos com `data/raw/manual_rules.jsonl` (opcional), valida o JSON e embaralha.

**Saída Final:** 📂 `data/processed/train_dataset_final.jsonl`

---

## 🏋️ 4. Fase de Treinamento (Rodar no Servidor GPU)

Mova o projeto (ou a pasta `data/processed`) para a máquina com GPU.

### Passo A: Configuração do Treino

Abra o arquivo `main.py` e ajuste a configuração em `project_config`:

- **Model Name:** `unsloth/Qwen2.5-32B-Instruct` ou `unsloth/Meta-Llama-3.1-8B-Instruct`.
- **Max Steps:** `60` para testes rápidos, `300+` para produção.
- **Final Model Name:** Caminho de saída (ex: `models/planus_qwen_v1`).

### Passo B: Executar o Fine-Tuning

```bash
make train
```

> **O processo:**
>
> 1. Baixa o modelo base e aplica adaptadores LoRA.
> 2. Inicia o treinamento supervisionado (SFT).
> 3. Converte e salva o modelo final em formato GGUF na pasta `models/`.

---

## 💬 5. Fase de Uso (Deploy no Ollama)

Com o modelo GGUF salvo, você pode testá-lo imediatamente no Ollama.

Se o modelo foi salvo e você tem um `Modelfile` configurado na raiz (apontando para o GGUF gerado), execute:

```bash
make run
```

Isso irá criar o modelo `planus-pro` no seu Ollama local e abrir o chat interativo.

---

## 🔄 Resumo do Ciclo de Vida (Cheat Sheet)

| Ação                             | Comando        |
| :------------------------------- | :------------- |
| **Instalar Dependências**        | `make install` |
| **Gerar Dataset (PDF -> JSONL)** | `make data`    |
| **Treinar Modelo (GPU)**         | `make train`   |
| **Rodar Chat (Ollama)**          | `make run`     |
| **Limpar Temporários**           | `make clean`   |

---

## 📂 Estrutura de Pastas

```text
planuze-llm/
├── config/             # Classes de configuração
├── data/
│   ├── source_documents/  # [ENTRADA] Seus PDFs aqui
│   ├── raw/               # Dados intermediários (sintéticos/manuais)
│   └── processed/         # [SAÍDA] Dataset final pronto para treino
├── models/             # Onde o GGUF final será salvo
├── src/                # Scripts de lógica (geração, treino, merge)
├── .env                # Tokens e configurações
├── Makefile            # Atalhos de comando
└── main.py             # Script de treino
```
