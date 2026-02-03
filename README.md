# Planuze LLM (v2)

Projeto de Fine-Tuning de LLMs otimizado para Apple Silicon (MLX) e CUDA (Unsloth).

```
src/
├── domain/           # Regras de Negócio (Configurações, Interfaces)
├── application/      # Casos de Uso (Preparar Dados, Treinar Modelo)
├── infrastructure/   # Implementações (MLX Trainer, Unsloth Trainer, etc.)
├── adapters/         # Entry Points (CLI, Container DI)
└── config/           # Configurações unificadas
```

## Requisitos

- Python 3.9+
- Dependências listadas em `requirements/`
- **Apple Silicon**: Requer `macosx` 13.0+ (Metal Performance Shaders)
- **CUDA**: Requer Linux/Windows com NVIDIA GPU e drivers instalados.

## Como Usar

Utilize o `Makefile` para gerenciar o ciclo de vida do projeto.

### 1. Instalação

Padrão (Apple Silicon):

```bash
make mlx:install
```

Para suporte CUDA (Unsloth):

```bash
make cuda:install
```

### 2. Configuração

Copie o `.env.example` para `.env` e ajuste as variáveis.
Para Apple Silicon, configure `.env.apple`, se necessário.

### 3. Preparação de Dados

Valida e prepara os datasets para o formato exigido.

```bash
make mlx:prepare
```

### 4. Treinamento

**Apple Silicon (MLX):**

```bash
make mlx:train
```

**CUDA (Unsloth):**

```bash
make cuda:train
```

### 5. Pipeline Completo

Executa preparação seguida de treinamento.

```bash
make mlx:full
# ou
make cuda:full
```

## Arquitetura

O sistema utiliza Injeção de Dependência através de um Container (`src2/adapters/cli/container.py`).

- **Trainer**: Abstraído via `ITrainer`. Implementação atual: `MLXTrainerAdapter`.
- **UI**: Toda saída visual é gerenciada pelo `TerminalPresenter` (usando Rich).
- **Dados**: Acesso a arquivos via `JSONLDataRepository`.
