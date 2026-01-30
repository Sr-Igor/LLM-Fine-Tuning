# 🧠 Planuze LLM Engine

Este repositório contém um pipeline completo para **Fine-Tuning de Modelos de Linguagem (LLMs)** utilizando a biblioteca **Unsloth**. O projeto foi estruturado para facilitar o carregamento de modelos quantizados (4-bit), aplicação de adaptadores LoRA, treinamento supervisionado (SFT) e exportação para o formato GGUF.

## 🚀 Funcionalidades

- **Carregamento Otimizado**: Suporte a modelos 4-bit via Unsloth (ex: Llama-3, Qwen-2.5).
- **Fine-Tuning Eficiente**: Uso de LoRA/QLoRA para adaptação de modelos grandes com menos memória.
- **Pipeline de Dados**: Processamento automático de datasets no formato JSONL com templates de chat (formato Alpaca).
- **Exportação GGUF**: Conversão automática do modelo treinado para GGUF, pronto para uso em ferramentas como Ollama, llama.cpp ou LM Studio.
- **Configuração Modular**: Separação clara entre configurações de modelo, treino e projeto.

## 📂 Estrutura do Projeto

```text
planuze-llm/
├── config/              # Definições de configuração (Dataclasses)
├── data/                # Diretório para datasets (raw/train_data.jsonl)
├── outputs/             # Checkpoints de treinamento (gerado automaticamente)
├── src/                 # Código fonte principal
│   ├── data_handler.py  # Carregamento e formatação de dados
│   ├── model_loader.py  # Gerenciamento do modelo e adapters
│   ├── prompt_templates.py # Templates de prompt (Alpaca)
│   └── trainer_engine.py   # Configuração do SFTTrainer
├── main.py              # Ponto de entrada (Entrypoint)
├── requirements.txt     # Dependências do projeto
└── trial.json           # Arquivo de exemplo (se aplicável ao formato)
```

## 🛠️ Pré-requisitos

- **Python** 3.10 ou superior.
- **GPU NVIDIA** (Recomendado para treino): Drivers CUDA instalados.
  - _Nota_: O código é compatível com desenvolvimento em Mac/CPU (apenas para estruturação), mas o treinamento efetivo requer GPU compatível com CUDA se usar as features do Unsloth.

## 📦 Instalação

1. **Clone o repositório:**

   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd planuze-llm
   ```

2. **Crie um ambiente virtual:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   venv\Scripts\activate     # Windows
   ```

3. **Instale as dependências:**

   ⚠️ **Atenção:** Verifique o arquivo `requirements.txt`.
   - Se estiver em **Linux com GPU**, descomente as linhas referentes ao `unsloth`, `xformers` e `trl`.
   - Se estiver em **MacOS** (sem GPU NVIDIA), mantenha as linhas do Unsloth comentadas.

   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuração

A configuração principal reside no arquivo `main.py` e `config/settings.py`.

No `main.py`, você ajusta o objeto `ProjectConfig`:

```python
project_config = ProjectConfig(
    model=ModelConfig(
        model_name="unsloth/Qwen2.5-32B-Instruct", # Modelo base
        max_seq_length=2048,
        load_in_4bit=True
    ),
    training=TrainingConfig(
        max_steps=60,         # Passos de treino
        batch_size=2,         # Tamanho do batch
        output_dir="outputs_checkpoints"
    ),
    dataset_path="data/raw/train_data.jsonl", # Caminho do dataset
    final_model_name="models/planus_qwen_v1"  # Caminho de saída do GGUF
)
```

## 📊 Formato dos Dados

O script espera um arquivo **JSONL** (JSON Lines) localizado em `data/raw/train_data.jsonl` (ou conforme configurado).

Cada linha deve conter um objeto JSON com os campos:

- `instruction`: A instrução do usuário.
- `input`: Contexto adicional.
- `output`: A resposta esperada.

**Exemplo:**

```json
{"instruction": "Resuma o texto.", "input": "O texto longo aqui...", "output": "Resumo aqui."}
{"instruction": "Classifique o sentimento.", "input": "Eu adorei este produto!", "output": "Positivo"}
```

O template utilizado (definido em `src/prompt_templates.py`) segue o padrão **Alpaca**.

## ▶️ Como Usar

Com tudo configurado e dependências instaladas, execute o pipeline:

```bash
python main.py
```

O script irá:

1. Carregar o modelo base.
2. Aplicar os adaptadores LoRA.
3. Carregar e formatar o dataset.
4. Executar o treinamento.
5. Salvar o modelo final em formato GGUF na pasta especificada.

## ❓ Solução de Problemas

- **FileNotFoundError**: Certifique-se de criar a pasta `data/raw` e adicionar o arquivo `train_data.jsonl`.
- **Erro de Memória/CUDA**: Reduza o `batch_size` no `training` config ou use um modelo menor.

## 📄 Licença

Este projeto é de uso privado.
