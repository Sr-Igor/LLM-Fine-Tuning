planuze-llm/
│
├── documents/
│ ├── ARCHITECTURE.MD
│
├── config/
│ ├── **init**.py
│ └── settings.py # Definição das classes de configuração (Hyperparâmetros)
│
├── data/
│ ├── raw/ # Onde você coloca o train_data.jsonl
│ └── processed/ # (Opcional) Cache de datasets processados
│
├── src/
│ ├── **init**.py
│ ├── model_loader.py # Lógica para carregar Unsloth/Qwen/Llama
│ ├── data_handler.py # Lógica para carregar e formatar JSONL
│ ├── trainer_engine.py # Configuração do SFTTrainer
│ └── prompt_templates.py # Funções de formatação de strings (Templates)
│
├── outputs/ # Onde os checkpoints e GGUF serão salvos
│
├── main.py # Ponto de entrada (Orquestrador)
├── requirements.txt # Dependências
└── .env # Variáveis de ambiente (Tokens HF, WANDB)
