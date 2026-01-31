# @title 🚀 Pipeline Master v6: Protocolo Planus (Memory Safe)

from huggingface_hub import login, HfApi
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import os
import sys
import json
import subprocess
import time
import importlib.util
import gc
import torch
from google.colab import drive
from dotenv import load_dotenv

# ==========================================
# 🧹 0. LIMPEZA DE MEMÓRIA (NOVO)
# ==========================================
print("🧹 [0/6] Higienizando VRAM...")
try:
    del model
    del tokenizer
    del trainer
except:
    pass
gc.collect()
torch.cuda.empty_cache()
print(
    f"✅ VRAM Limpa. Memória livre: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

# ==========================================
# 📺 FUNÇÃO AUXILIAR DE LOGGING
# ==========================================


def run_cmd(command, desc=None):
    if desc:
        print(f"\n⏳ {desc}...")
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True
    )
    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"❌ Falha: {command}")
    print(f"✅ Concluído: {desc}")


# ==========================================
# ⚙️ BOOTSTRAP
# ==========================================
PROJECT_ROOT_FOLDER = "planuze-llm-colab"

# ==========================================
# 🛠️ 1. SETUP DE AMBIENTE
# ==========================================
print("🏗️ [1/6] Setup de Ambiente...")

if not os.path.exists('/content/drive'):
    drive.mount('/content/drive', force_remount=True)

# 1.1 Localizar Projeto
search_cmd = f"find /content/drive/MyDrive -type d -name '{PROJECT_ROOT_FOLDER}' -print -quit"
project_path_list = os.popen(search_cmd).read().strip()

if not project_path_list:
    raise FileNotFoundError(f"❌ Pasta '{PROJECT_ROOT_FOLDER}' não encontrada.")

PROJECT_PATH = project_path_list
print(f"✅ Diretório: {PROJECT_PATH}")

# 1.2 Carregar .env
env_path = os.path.join(PROJECT_PATH, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
    print("✅ .env carregado.")
else:
    raise FileNotFoundError("❌ .env não encontrado!")


def get_var(key, default=None, type_cast=str):
    val = os.getenv(key)
    if val is None:
        return default
    try:
        if type_cast == bool:
            return str(val).lower() in ('true', '1', 't')
        if type_cast == int:
            return int(val)
        if type_cast == float:
            return float(val)
        if type_cast == list:
            return json.loads(val)
        return str(val)
    except:
        return default


# 1.3 Instalação Inteligente
if importlib.util.find_spec("unsloth") is None:
    print("\n📦 Instalando Stack Unsloth...")
    run_cmd(
        'pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" "unsloth_zoo"',
        "Instalando Core & Zoo"
    )
    run_cmd(
        'pip install --no-deps "xformers<0.0.29" "trl<0.9.0" peft accelerate bitsandbytes python-dotenv huggingface_hub tyro',
        "Instalando Dependências"
    )
else:
    print("\n✅ Stack Unsloth já instalada. Pulando.")

# ==========================================
# 🧠 2. IMPORTS E MODELO
# ==========================================
print("\n🧠 [2/6] Carregando Modelo...")


# Autenticação
hf_token = get_var("HF_TOKEN")
if not hf_token:
    raise ValueError("❌ HF_TOKEN ausente.")
login(token=hf_token)
username = HfApi().whoami()['name']
repo_name = get_var("FINAL_MODEL_NAME", "planus_model")
FULL_REPO_ID = f"{username}/{repo_name}"
print(f"🎯 Target: {FULL_REPO_ID}")

# Modelo
model_name = get_var("MODEL_NAME", "unsloth/Qwen2.5-7B-Instruct")
max_seq_length = get_var("MAX_SEQ_LENGTH", 1024, int)
load_in_4bit = get_var("LOAD_IN_4BIT", True, bool)

print(f"⚙️ Load: {model_name}")

# Segunda camada de limpeza antes do load pesado
torch.cuda.empty_cache()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=get_var("LORA_R", 16, int),
    target_modules=get_var("LORA_TARGET_MODULES", ["q_proj", "k_proj"], list),
    lora_alpha=get_var("LORA_ALPHA", 16, int),
    lora_dropout=get_var("LORA_DROPOUT", 0, float),
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=get_var("TRAINING_SEED", 3407, int),
)

# ==========================================
# 🔄 3. PREPARAÇÃO DO DATASET
# ==========================================
print("\n🔄 [3/6] Processando Dataset...")

dataset_path = os.path.join(PROJECT_PATH, get_var("DATASET_PATH"))
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Dataset não encontrado: {dataset_path}")

raw_dataset = load_dataset("json", data_files=dataset_path, split="train")


def formatting_prompts_func(examples):
    # ChatML Format
    if "messages" in examples:
        texts = [tokenizer.apply_chat_template(
            m, tokenize=False) for m in examples['messages']]
        return {"text": texts}

    # Instruct Format
    texts = []
    instructions = examples.get(
        "instruction", [""] * len(examples.get("input", [])))
    inputs = examples.get("input", [""] * len(examples.get("output", [])))
    outputs = examples.get("output", [])

    for instruction, input, output in zip(instructions, inputs, outputs):
        text = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        texts.append(text)
    return {"text": texts}


print("⏳ Mapeando dataset...")
dataset = raw_dataset.map(formatting_prompts_func, batched=True)
print(f"✅ Dataset pronto! Colunas: {dataset.column_names}")

if "text" not in dataset.column_names:
    raise ValueError("❌ Erro Crítico: Coluna 'text' não criada.")

# ==========================================
# 🔥 4. EXECUÇÃO DO TREINO
# ==========================================
print("\n🔥 [4/6] Iniciando Trainer...")

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
        report_to="none",
    ),
)

print("🚀 TREINANDO...")
trainer.train()

# Backup
backup_dir = os.path.join(PROJECT_PATH, get_var(
    "TRAINING_OUTPUT_DIR"), "final_adapter")
print(f"\n💾 Salvando Backup: {backup_dir}")
model.save_pretrained(backup_dir)
tokenizer.save_pretrained(backup_dir)

# ==========================================
# 🔨 5. COMPILAÇÃO LLAMA.CPP
# ==========================================
print("\n🔨 [5/6] Verificando Compilação...")

if os.path.exists("/content/llama.cpp/build/bin/llama-quantize"):
    print("✅ Binários compilados. Pulando.")
    os.system(
        "ln -sf /content/llama.cpp/build/bin/llama-quantize /content/llama.cpp/llama-quantize")
else:
    print("⚠️ Compilando llama.cpp...")
    compile_cmd = """
    cd /content
    rm -rf llama.cpp
    git clone --depth 1 https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    mkdir build
    cd build
    cmake .. -DGGML_NATIVE=OFF
    cmake --build . --config Release -j 1
    """
    run_cmd(compile_cmd, "Compilando C++")

    if os.path.exists("/content/llama.cpp/build/bin/llama-quantize"):
        os.system(
            "ln -sf /content/llama.cpp/build/bin/llama-quantize /content/llama.cpp/llama-quantize")
    else:
        raise RuntimeError("❌ Erro compilação llama.cpp")

# ==========================================
# ☁️ 6. EXPORTAÇÃO
# ==========================================
print("\n☁️ [6/6] Upload Hugging Face...")
quant_method = get_var("GGUF_QUANTIZATION", "q4_k_m")

try:
    print(f"⏳ Enviando {quant_method} para {FULL_REPO_ID}...")
    model.push_to_hub_gguf(
        FULL_REPO_ID,
        tokenizer,
        quantization_method=quant_method,
        token=hf_token
    )
    print(f"\n🎉 SUCESSO FINAL! Modelo: https://huggingface.co/{FULL_REPO_ID}")
except Exception as e:
    print(f"\n❌ Erro Upload: {e}")
