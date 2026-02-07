# LLM Fine-Tuning

Advanced Fine-Tuning pipeline compatible with both **Apple Silicon (MLX)** and **NVIDIA GPUs (CUDA/Unsloth)**.
Designed with Clean Architecture and SOLID principles.

- Author: Igor Rezende
- Professional contact: [sr.igor.dev@gmail.com](mailto:sr.igor.dev@gmail.com)

---

## 🚀 Quick Start Guide

Follow this step-by-step guide to go from zero to a trained model uploaded to Hugging Face.

### 📋 Prerequisites

1.  **Python 3.9+** installed.
2.  **Hardware**:
    - Mac with Apple Silicon (M1/M2/M3) e.g., MacBook Pro, Mac Studio.
    - **OR** Linux/Windows PC with NVIDIA GPU (CUDA).
3.  **Hugging Face Account**: You need a Write Access Token. [Get it here](https://huggingface.co/settings/tokens).
4.  **WandB Account (Optional)**: For tracking training metrics. [Sign up](https://wandb.ai/).

---

### 🛠️ 1. Installation

Clone the repository and install dependencies using `make`.

#### For Apple Silicon (Mac M1/M2/M3):

```bash
make mlx:install
```

#### For CUDA (NVIDIA):

```bash
make cuda:install
```

---

### ⚙️ 2. Configuration

Set up your environment variables.

1.  **Global config**:

    ```bash
    cp envs/.env.global.example envs/.env.global
    ```

    Edit `envs/.env.global`:
    - Set `HF_TOKEN` (Required for uploading models and downloading gated models like Llama 3).
    - Set `WANDB_API_KEY` (Optional, for logging).

2.  **Backend config**:
    - **Mac**: `cp envs/.env.mlx.example envs/.env.mlx`
    - **NVIDIA**: `cp envs/.env.cuda.example envs/.env.cuda`

    Edit the created file (e.g., `.env.mlx`) to adjust:
    - `APPLE_MODEL_NAME`: Base model (e.g., `mlx-community/Qwen2.5-14B-Instruct-4bit`).
    - `APPLE_LORA_RANK`: LoRA rank (default 16).
    - `APPLE_NUM_ITERS`: Total training steps (e.g., 600).

---

### 📚 3. Data Preparation

#### Option A: Manual Data

1.  Create your training data in **JSONL** format.
    - Format: `{"text": "Human: Hello\nAI: Hi there!"}` or ChatML format.
2.  Place your `.jsonl` files in the `data/raw/` folder.
    - Example: `data/raw/my_dataset_v1.jsonl`
3.  Run the preparation command:
    ```bash
    make mlx:prepare
    # or
    make cuda:prepare
    ```
    _This command merges all files in `data/raw`, shuffles them, splits into Train/Validation, and saves to `data/processed/`._

#### Option B: Synthetic Data (From PDFs)

Generate high-quality Q&A pairs and Action examples directly from your PDF documents.

**Prerequisites**:
- [Ollama](https://ollama.com) running (`ollama serve`).
- Model pulled (`ollama pull qwen2.5:14b`).

1.  **Place PDFs**: Add your source PDFs to `data/source_documents/`.
2.  **Configure**: Check `envs/.env.global` for generator settings (model, chunk size).
3.  **Run Generator**:
    ```bash
    make synthetic
    ```
    This command uses Ollama to create a rich dataset interacting with your documents, producing `data/raw/train_data_synthetic.jsonl`.
    
    *See [SYNTHETIC_DATA_README.md](SYNTHETIC_DATA_README.md) for full details.*

---

### 🧠 4. Training

Start the fine-tuning process. This will download the base model and train the adapters.

#### Apple Silicon (MLX):

```bash
make mlx:train
```

_Note: A generic start may require `wandb login` if enabled._

#### CUDA (Unsloth):

```bash
make cuda:train
```

---

### ☁️ 5. Publish / Upload

After training, you can upload your unified model (or adapters) to Hugging Face.

```bash
make mlx:publish
# or
make cuda:publish
```

_Make sure `HF_REPO_ID` is set in your `.env.global`._

---

### ⚡ One-Command Pipeline

You can run Preparation -> Training -> Publish in a single command:

```bash
make mlx:full
```

---

## 📂 Project Structure

```
src/
├── domain/           # Business Logic & Interfaces (Clean Architecture)
├── application/      # Use Cases (Prepare, Train, Publish)
├── infrastructure/   # Implementations (MLX, Unsloth, FileSystem)
├── adapters/         # CLI Entry Point & DI Container
└── config/           # Configuration Management
```

## 🐛 Troubleshooting

- **WandB Error**: Ensure you have run `wandb login <key>` locally or set `WANDB_API_KEY` in `.env.global`.
- **Out of Memory (OOM)**: Reduce `BATCH_SIZE` in your `.env` file.
- **"Killed" process on Mac**: Usually means RAM exhaustion. Reduce batch size or close other apps.

---

License: MIT
