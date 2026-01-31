# @title 🚀 Pipeline Master: Protocolo Planus (Unsloth + GGUF + Hugging Face)
# @markdown Este script automatiza todo o processo documentado: Setup, Treino, Compilação Segura e Upload.

from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from huggingface_hub import login, HfApi
from dotenv import load_dotenv
import os
import sys
import torch
from google.colab import drive
import psutil

# ==========================================
# ⚙️ ZONA DE CONFIGURAÇÃO (Edite aqui)
# ==========================================

# Nome da pasta raiz do projeto no seu Google Drive
# Exemplo: Se o zip foi extraído em "MyDrive/llm/meu-projeto", coloque "meu-projeto"
PROJECT_ROOT_FOLDER = "planuze-llm-collab"

# Caminho relativo do dataset dentro do projeto
DATASET_RELATIVE_PATH = "data/processed/train_dataset_final.jsonl"

# Nome que você quer dar ao modelo no Hugging Face
MODEL_REPO_NAME = "planus-qwen-v1"

# Configurações de Treino (Otimizadas para Tesla T4)
MAX_SEQ_LENGTH = 1024  # 1024 é o limite seguro para T4. 2048 pode dar OOM.
LOAD_IN_4BIT = True

# ==========================================
# 🛠️ 1. PREPARAÇÃO DO AMBIENTE
# ==========================================
print("🏗️ [1/6] Preparando Ambiente e Montando Drive...")

# 1.1 Montar Drive
drive.mount('/content/drive', force_remount=True)

# 1.2 Definir Caminhos Dinâmicos
# Procura a pasta do projeto recursivamente para evitar erros de caminho
search_cmd = f"find /content/drive/MyDrive -type d -name '{PROJECT_ROOT_FOLDER}' -print -quit"
project_path_list = os.popen(search_cmd).read().strip()

if not project_path_list:
    raise FileNotFoundError(
        f"❌ A pasta '{PROJECT_ROOT_FOLDER}' não foi encontrada no seu Drive via busca.")

PROJECT_PATH = project_path_list
print(f"✅ Diretório do Projeto localizado: {PROJECT_PATH}")

# 1.3 Instalar Dependências (Silencioso)
print("📦 Instalando Unsloth e dependências (pode levar 2-3 min)...")
!pip install - -no-deps - q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install - -no-deps - q "xformers<0.0.29" "trl<0.9.0" peft accelerate bitsandbytes python-dotenv huggingface_hub

# ==========================================
# 🔐 2. AUTENTICAÇÃO E CONFIGURAÇÃO
# ==========================================
print("\n🔐 [2/6] Configurando Autenticação...")


# 2.1 Carregar .env
env_path = os.path.join(PROJECT_PATH, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
    print("✅ Arquivo .env carregado.")
else:
    print(
        f"⚠️ .env não encontrado em {env_path}. Tentando variáveis de ambiente do sistema.")

# 2.2 Login no Hugging Face
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError(
        "❌ ERRO CRÍTICO: HF_TOKEN não encontrado. Verifique seu .env.")

try:
    login(token=hf_token)
    api = HfApi()
    user_info = api.whoami()
    username = user_info['name']

    # Validação de Permissão de Escrita
    if 'write' not in user_info['auth']['accessToken']['role'] and user_info['auth']['accessToken']['role'] != 'write':
        # Nota: A API as vezes retorna estruturas diferentes, mas tentamos validar.
        print("⚠️ AVISO: Verifique se seu token tem permissão 'WRITE'. Tokens 'READ' falharão no upload.")

    FULL_REPO_ID = f"{username}/{MODEL_REPO_NAME}"
    print(f"✅ Autenticado como: {username}")
    print(f"🎯 Target Repo: {FULL_REPO_ID}")

except Exception as e:
    raise RuntimeError(f"❌ Falha na autenticação com Hugging Face: {e}")

# ==========================================
# 🧠 3. TREINAMENTO (UNSLOTH)
# ==========================================
print("\n🧠 [3/6] Iniciando Rotina de Treino...")


# 3.1 Carregar Modelo Base
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)

# 3.2 Configurar LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj",
                    "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 3.3 Carregar Dataset
dataset_full_path = os.path.join(PROJECT_PATH, DATASET_RELATIVE_PATH)
if not os.path.exists(dataset_full_path):
    raise FileNotFoundError(
        f"❌ Dataset não encontrado em: {dataset_full_path}")

dataset = load_dataset("json", data_files=dataset_full_path, split="train")
print(f"📚 Dataset carregado: {len(dataset)} registros.")

# Formatação do Prompt (Alpaca/ChatML Style)


def formatting_prompts_func(examples):
    # Adapte esta função se seu JSONL tiver chaves diferentes de instruction/input/output
    if "messages" in examples:  # Suporte a formato chat direto
        return {"text": [tokenizer.apply_chat_template(m, tokenize=False) for m in examples['messages']]}

    # Fallback genérico
    texts = []
    for instruction, input, output in zip(examples.get("instruction", []), examples.get("input", []), examples.get("output", [])):
        text = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        texts.append(text)
    return {"text": texts, }


# 3.4 Configurar Treinador
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,  # Ajuste conforme necessário
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

# 3.5 Executar Treino
print("🔥 Treinando...")
trainer_stats = trainer.train()
print("✅ Treino concluído!")

# 3.6 Salvar Adaptadores Localmente (Backup no Drive)
backup_dir = os.path.join(PROJECT_PATH, "outputs_checkpoints", "final_adapter")
model.save_pretrained(backup_dir)
tokenizer.save_pretrained(backup_dir)
print(f"💾 Backup dos adaptadores salvo em: {backup_dir}")

# ==========================================
# 🔨 4. COMPILAÇÃO DO LLAMA.CPP (A PROVA DE FALHAS)
# ==========================================
print("\n🔨 [4/6] Compilando llama.cpp (Modo Seguro -j 1)...")

# Script Shell embutido para garantir ambiente limpo
shell_script = """
cd /content
rm -rf llama.cpp
git clone --depth 1 https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build
cd build
cmake .. -DGGML_NATIVE=OFF
cmake --build . --config Release -j 1
"""
os.system(shell_script)

# Link Simbólico Crítico
if os.path.exists("/content/llama.cpp/build/bin/llama-quantize"):
    os.system(
        "ln -sf /content/llama.cpp/build/bin/llama-quantize /content/llama.cpp/llama-quantize")
    print("✅ Compilação OK e Link Simbólico criado.")
else:
    raise RuntimeError(
        "❌ Erro na compilação do llama.cpp. Binário não encontrado.")

# ==========================================
# ☁️ 5. EXPORTAÇÃO E UPLOAD
# ==========================================
print("\n☁️ [5/6] Iniciando Conversão GGUF e Upload...")

# Define o método de quantização
quant_method = "q4_k_m"

print(f"🚀 Enviando para Hugging Face: {FULL_REPO_ID}")
print("☕ Isso pode levar alguns minutos (Conversão + Upload de ~5GB)...")

try:
    model.push_to_hub_gguf(
        FULL_REPO_ID,
        tokenizer,
        quantization_method=quant_method,
        token=hf_token
    )
    print("\n🎉 ===================================================")
    print(f"✅ SUCESSO ABSOLUTO! O MODELO ESTÁ ONLINE.")
    print(f"🔗 Link: https://huggingface.co/{FULL_REPO_ID}")
    print("======================================================")

except Exception as e:
    print(f"\n❌ Erro no Upload Automático: {e}")
    print("💡 Tentativa de recuperação: Verifique se o repo já existe ou tente upload manual do arquivo .gguf gerado.")

# ==========================================
# 🏁 6. INSTRUÇÕES FINAIS
# ==========================================
print("\n🏁 [6/6] Próximos Passos (No seu Mac):")
print(f"1. Instale o Ollama: brew install ollama")
print(f"2. Rode direto: ollama run hf.co/{FULL_REPO_ID}")
