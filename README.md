# Planuze LLM (v2)

Fine-Tuning LLM Project optimized for Apple Silicon (MLX) and CUDA (Unsloth).

```
src/
├── domain/           # Business Rules (Configuration, Interfaces)
├── application/      # Use Cases (Prepare Data, Train Model)
├── infrastructure/   # Implementations (MLX Trainer, Unsloth Trainer, etc.)
├── adapters/         # Entry Points (CLI, DI Container)
└── config/           # Unified Configurations
```

## Requirements

- Python 3.9+
- Dependencies listed in `requirements/`
- **Apple Silicon**: Requires `macosx` 13.0+ (Metal Performance Shaders)
- **CUDA**: Requires Linux/Windows with NVIDIA GPU and installed drivers.

## How to Use

Use the `Makefile` to manage the project lifecycle.

### 1. Installation

Default (Apple Silicon):

```bash
make mlx:install
```

For CUDA support (Unsloth):

```bash
make cuda:install
```

### 2. Configuration

Copy `.env.global.example` to `.env.global` and adjust global variables (HF_TOKEN, etc.).
For specific backends, create `.env.mlx` (from `envs/.env.mlx.example`) or `.env.cuda` (from `envs/.env.cuda.example`).

### 3. Data Preparation

Validates and prepares datasets into the required format.

```bash
make mlx:prepare
```

### 4. Training

**Apple Silicon (MLX):**

```bash
make mlx:train
```

**CUDA (Unsloth):**

```bash
make cuda:train
```

### 5. Full Pipeline

Runs preparation followed by training.

```bash
make mlx:full
# or
make cuda:full
```

## Architecture

The system uses Dependency Injection via a Container (`src/adapters/cli/container.py`).

- **Trainer**: Abstracted via `ITrainer`. Current implementations: `MLXTrainerAdapter`, `UnslothTrainerAdapter`.
- **UI**: All visual output is managed by `TerminalPresenter` (using Rich).
- **Data**: File access via `JSONLDataRepository`.
