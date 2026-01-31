# @title 🚀 Pipeline Master: Protocolo Planus (100% Driven by .env)
# @markdown Este script lê TODAS as configurações do arquivo .env definido.

from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from huggingface_hub import login, HfApi
import os
import sys
import json
import torch
from google.colab import drive
from dotenv import load_dotenv

# ==========================================
# ⚙️ CONFIGURAÇÃO DE BOOTSTRAP
# ==========================================
# Precisamos apenas saber onde buscar o .env inicial
PROJECT_ROOT_FOLDER = "planuze-llm-collab"

# ==========================================
# 🛠️ 1. PREPARAÇÃO E LEITURA DO AMBIENTE
# ==========================================
print("🏗️ [1/6] Montando Drive e Carregando Variáveis...")

drive.mount('/content/drive', force_remount=True)

# 1.1 Localizar Pasta do Projeto
search_cmd = f"find /content/drive/MyDrive -type d -name '{PROJECT_ROOT_FOLDER}' -print -quit"
project_path_list = os.popen(search_cmd).read().strip()

if not project_path_list:
    raise FileNotFoundError(
        f"❌ A pasta '{PROJECT_ROOT_FOLDER}' não foi encontrada.")

PROJECT_PATH = project_path_list
print(f"✅ Diretório do Projeto: {PROJECT_PATH}")

# 1.2 Carregar .env
env_path = os.path.join(PROJECT_PATH, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
    print("✅ .env carregado com sucesso.")
else:
    raise FileNotFoundError(
        "❌ Arquivo .env não encontrado! O script depende dele.")

# --- HELPER: Conversor de Tipos do .env ---


def get_var(key, default=None, type_cast=str):
    val = os.getenv(key)
    if val is None:
        return default
    try:
        if type_cast == bool:
            return val.lower() in ('true', '1', 't')
        if type_cast == int:
            return int(val)
        if type_cast == float:
            return float(val)
        if type_cast == list:
            # Converte string '["a","b"]' em lista Python
            return json.loads(val)
        return str(val)
    except Exception as e:
        print(f"⚠️ Erro ao converter variável {key}: {e}")
        return default


# 1.3 Instalar Dependências
print("📦 Instalando Unsloth (Isso usa as configs do .env na próxima etapa)...")
!pip install - -no-deps - q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install - -no-deps - q "xformers<0.0.29" "trl<0.9.0" peft accelerate bitsandbytes python-dotenv huggingface_hub

# ==========================================
# 🔐 2. AUTENTICAÇÃO
# ==========================================

hf_token = get_var("HF_TOKEN")
if not hf_token:
    raise ValueError("❌ HF_TOKEN ausente no .env")

login(token=hf_token)
username = HfApi().whoami()['name']
# Constrói o Repo ID usando o nome definido no .env
repo_name = get_var("FINAL_MODEL_NAME", "planus_model")
FULL_REPO_ID = f"{username}/{repo_name}"
print(f"🎯 Target Repo: {FULL_REPO_ID}")

# ==========================================
# 🧠 3. TREINAMENTO (Config via .env)
# ==========================================
print("\n🧠 [3/6] Iniciando Treino com Parâmetros do .env...")


# 3.1 Carregar Modelo
max_seq_length = get_var("MAX_SEQ_LENGTH", 1024, int)
load_in_4bit = get_var("LOAD_IN_4BIT", True, bool)
model_name = get_var("MODEL_NAME", "unsloth/Qwen2.5-7B-Instruct")

print(
    f"⚙️ Model: {model_name} | Context: {max_seq_length} | 4bit: {load_in_4bit}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=load_in_4bit,
)

# 3.2 Configurar LoRA
# Aqui usamos json.loads para pegar a lista correta do .env
target_modules = get_var("LORA_TARGET_MODULES", ["q_proj", "k_proj"], list)

model = FastLanguageModel.get_peft_model(
    model,
    r=get_var("LORA_R", 16, int),
    target_modules=target_modules,
    lora_alpha=get_var("LORA_ALPHA", 16, int),
    lora_dropout=get_var("LORA_DROPOUT", 0, float),
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=get_var("TRAINING_SEED", 3407, int),
)

# 3.3 Carregar Dataset
dataset_path = os.path.join(PROJECT_PATH, get_var("DATASET_PATH"))
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Dataset não encontrado: {dataset_path}")

dataset = load_dataset("json", data_files=dataset_path, split="train")

# Formatação ChatML


def formatting_prompts_func(examples):
    if "messages" in examples:
        return {"text": [tokenizer.apply_chat_template(m, tokenize=False) for m in examples['messages']]}
    texts = []
    for instruction, input, output in zip(examples.get("instruction", []), examples.get("input", []), examples.get("output", [])):
        text = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        texts.append(text)
    return {"text": texts, }


# 3.4 Treinador
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=get_var("TRAINING_DATASET_NUM_PROC", 2, int),
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=get_var("TRAINING_BATCH_SIZE", 2, int),
        gradient_accumulation_steps=get_var(
            "TRAINING_GRAD_ACCUMULATION", 4, int),
        warmup_steps=get_var("TRAINING_WARMUP_STEPS", 10, int),
        max_steps=get_var("TRAINING_MAX_STEPS", 60, int),
        learning_rate=get_var("TRAINING_LEARNING_RATE", 2e-4, float),
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim=get_var("TRAINING_OPTIM", "adamw_8bit"),
        weight_decay=get_var("TRAINING_WEIGHT_DECAY", 0.01, float),
        lr_scheduler_type=get_var("TRAINING_LR_SCHEDULER", "linear"),
        seed=get_var("TRAINING_SEED", 3407, int),
        output_dir="outputs",
    ),
)

print("🔥 Iniciando Treino...")
trainer.train()
print("✅ Treino Concluído.")

# Backup no Drive
backup_dir = os.path.join(PROJECT_PATH, get_var(
    "TRAINING_OUTPUT_DIR"), "final_adapter")
model.save_pretrained(backup_dir)
tokenizer.save_pretrained(backup_dir)

# ==========================================
# 🔨 4. COMPILAÇÃO (Mantida igual pela segurança)
# ==========================================
print("\n🔨 [4/6] Compilando llama.cpp...")
os.system("cd /content && rm -rf llama.cpp && git clone --depth 1 https://github.com/ggerganov/llama.cpp && cd llama.cpp && mkdir build && cd build && cmake .. -DGGML_NATIVE=OFF && cmake --build . --config Release -j 1")
if os.path.exists("/content/llama.cpp/build/bin/llama-quantize"):
    os.system(
        "ln -sf /content/llama.cpp/build/bin/llama-quantize /content/llama.cpp/llama-quantize")
else:
    raise RuntimeError("Erro na compilação do llama.cpp")

# ==========================================
# ☁️ 5. EXPORTAÇÃO
# ==========================================
print("\n☁️ [5/6] Upload para Hugging Face...")
quant_method = get_var("GGUF_QUANTIZATION", "q4_k_m")

try:
    model.push_to_hub_gguf(
        FULL_REPO_ID,
        tokenizer,
        quantization_method=quant_method,
        token=hf_token
    )
    print(f"✅ SUCESSO! Modelo em: https://huggingface.co/{FULL_REPO_ID}")
except Exception as e:
    print(f"❌ Erro no upload: {e}")
