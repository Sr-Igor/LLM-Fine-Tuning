# @title 🚀 Pipeline Master v10: Protocolo Planus (VRAM Guard)


from huggingface_hub import login, HfApi
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from dotenv import load_dotenv
from google.colab import drive
import torch
import shutil
import gc
import importlib.util
import time
import subprocess
import json
import os
import sys

# [CRÍTICO] Otimização de memória para T4
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ==========================================
# 🛡️ 0. VRAM GUARD (Trava de Segurança)
# ==========================================
print("🛡️ [0/6] Verificando integridade da GPU...")

# Tenta limpar primeiro
try:
    del model, tokenizer, trainer
except:
    pass
gc.collect()
torch.cuda.empty_cache()

# Verifica memória real disponível
free_vram = torch.cuda.mem_get_info()[0] / 1024**3
total_vram = torch.cuda.mem_get_info()[1] / 1024**3

print(f"📊 Status VRAM: {free_vram:.2f} GB Livres / {total_vram:.2f} GB Totais")

if free_vram < 8.0:
    print("\n❌ ERRO CRÍTICO: GPU SUJA DETECTADA!")
    print(
        f"Você tem apenas {free_vram:.2f}GB livres. O modelo precisa de ~6GB + Contexto.")
    print("👉 SOLUÇÃO: Vá em 'Ambiente de execução' > 'Reiniciar sessão' e tente novamente.")
    sys.exit("⛔ Execução interrompida para proteger o ambiente.")

print("✅ VRAM Suficiente. Prosseguindo...")

# ==========================================
# 📺 LOGGING
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
# 🛠️ 1. SETUP AMBIENTE
# ==========================================
print("🏗️ [1/6] Setup...")
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive', force_remount=True)

search_cmd = f"find /content/drive/MyDrive -type d -name '{PROJECT_ROOT_FOLDER}' -print -quit"
project_path_list = os.popen(search_cmd).read().strip()
if not project_path_list:
    raise FileNotFoundError(f"❌ Pasta '{PROJECT_ROOT_FOLDER}' não encontrada.")
PROJECT_PATH = project_path_list
print(f"✅ Diretório: {PROJECT_PATH}")

env_path = os.path.join(PROJECT_PATH, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
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


if importlib.util.find_spec("unsloth") is None:
    print("\n📦 Instalando Stack...")
    run_cmd(
        'pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" "unsloth_zoo"', "Instalando Core")
    run_cmd('pip install --no-deps "xformers<0.0.29" "trl<0.9.0" peft accelerate bitsandbytes python-dotenv huggingface_hub tyro', "Instalando Deps")
else:
    print("\n✅ Stack já instalada.")

# ==========================================
# 🧠 2. LOAD MODEL
# ==========================================
print("\n🧠 [2/6] Carregando Modelo...")

hf_token = get_var("HF_TOKEN")
if not hf_token:
    raise ValueError("❌ HF_TOKEN ausente.")
login(token=hf_token)
username = HfApi().whoami()['name']
repo_name = get_var("FINAL_MODEL_NAME", "planus_model")
FULL_REPO_ID = f"{username}/{repo_name}"

model_name = get_var("MODEL_NAME", "unsloth/Qwen2.5-7B-Instruct")
max_seq_length = get_var("MAX_SEQ_LENGTH", 1024, int)
load_in_4bit = get_var("LOAD_IN_4BIT", True, bool)

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
# 🔄 3. DATASET
# ==========================================
print("\n🔄 [3/6] Processando Dataset...")

drive_dataset_path = os.path.join(PROJECT_PATH, get_var("DATASET_PATH"))
if not os.path.exists(drive_dataset_path):
    raise FileNotFoundError(f"❌ Dataset não encontrado: {drive_dataset_path}")

local_dataset_path = "/content/temp_dataset.jsonl"
print(f"⚡ Copiando do Drive para Disco Local...")
shutil.copy(drive_dataset_path, local_dataset_path)

file_size = os.path.getsize(local_dataset_path) / (1024 * 1024)
print(f"✅ Dataset carregado: {file_size:.2f} MB.")

raw_dataset = load_dataset(
    "json", data_files=local_dataset_path, split="train")


def formatting_prompts_func(examples):
    if "messages" in examples:
        texts = [tokenizer.apply_chat_template(
            m, tokenize=False) for m in examples['messages']]
        return {"text": texts}
    texts = []
    instructions = examples.get(
        "instruction", [""] * len(examples.get("input", [])))
    inputs = examples.get("input", [""] * len(examples.get("output", [])))
    outputs = examples.get("output", [])
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        texts.append(text)
    return {"text": texts}


dataset = raw_dataset.map(formatting_prompts_func, batched=True)

# ==========================================
# 🔥 4. TREINO (EXTREME MEMORY SAFE)
# ==========================================
print("\n🔥 [4/6] Iniciando Trainer (T4 Optimized)...")

# Força Batch Size 1 para garantir estabilidade
original_batch = get_var("TRAINING_BATCH_SIZE", 2, int)
safe_batch = 1
# Compensa aumentando a acumulação
safe_grad_accum = get_var(
    "TRAINING_GRAD_ACCUMULATION", 4, int) * original_batch

print(f"⚙️ Ajuste T4: Batch={safe_batch} | GradAccum={safe_grad_accum}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=get_var("TRAINING_DATASET_NUM_PROC", 2, int),
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=safe_batch,
        gradient_accumulation_steps=safe_grad_accum,
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

backup_dir = os.path.join(PROJECT_PATH, get_var(
    "TRAINING_OUTPUT_DIR"), "final_adapter")
print(f"\n💾 Salvando Backup: {backup_dir}")
model.save_pretrained(backup_dir)
tokenizer.save_pretrained(backup_dir)

# ==========================================
# 🔨 5. COMPILAÇÃO
# ==========================================
print("\n🔨 [5/6] Verificando Compilação...")
if os.path.exists("/content/llama.cpp/build/bin/llama-quantize"):
    print("✅ Binários compilados.")
    os.system(
        "ln -sf /content/llama.cpp/build/bin/llama-quantize /content/llama.cpp/llama-quantize")
else:
    print("⚠️ Compilando llama.cpp...")
    run_cmd("""
    cd /content
    rm -rf llama.cpp
    git clone --depth 1 https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    mkdir build
    cd build
    cmake .. -DGGML_NATIVE=OFF
    cmake --build . --config Release -j 1
    """, "Compilando C++")

    if os.path.exists("/content/llama.cpp/build/bin/llama-quantize"):
        os.system(
            "ln -sf /content/llama.cpp/build/bin/llama-quantize /content/llama.cpp/llama-quantize")
    else:
        raise RuntimeError("❌ Erro compilação.")

# ==========================================
# ☁️ 6. UPLOAD
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
