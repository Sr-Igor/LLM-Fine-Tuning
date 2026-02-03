"""Module to load and manage project settings from environment variables."""

import os

from dotenv import load_dotenv

from ..domain.entities.configuration import (
    BackendConfig,
    DataConfig,
    LoRAConfig,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
)

# Carrega .env
# Carrega .env.global (reaproveitar variáveis comuns)
load_dotenv(".env.global")

# Carrega ambiente específico baseado no backend
# O backend pode ser setado via environment variable no Makefile
# Padrão: mlx (se não especificado)
_backend = os.getenv("TRAINING_BACKEND", "mlx").lower()

if _backend == "unsloth":
    load_dotenv("envs/.env.cuda")
    print(f"Loaded envs/.env.cuda (Backend: {_backend})")
elif _backend == "mlx":
    load_dotenv("envs/.env.mlx")
    print(f"Loaded envs/.env.mlx (Backend: {_backend})")
else:
    # Fallback or other backends
    pass


class Settings:
    """Carregador de configurações do ambiente."""

    @staticmethod
    def load() -> ProjectConfig:
        """Load and return the complete project configuration."""
        return ProjectConfig(
            model=Settings._load_model(),
            lora=Settings._load_lora(),
            training=Settings._load_training(),
            data=Settings._load_data(),
            backend=Settings._load_backend(),
        )

    @staticmethod
    def _load_backend() -> BackendConfig:
        """Load backend configuration settings."""
        return BackendConfig(type=os.getenv("TRAINING_BACKEND", "mlx").lower())

    @staticmethod
    def _load_model() -> ModelConfig:
        """Load model configuration settings based on the backend."""
        backend = os.getenv("TRAINING_BACKEND", "mlx").lower()
        if backend == "unsloth":
            # Unsloth env usually uses MODEL_NAME, prefixes are optional
            model_name = os.getenv(
                "MODEL_NAME",
                os.getenv("CUDA_MODEL_NAME", "unsloth/Qwen2.5-7B-Instruct"),
            )
            max_seq = int(os.getenv("MAX_SEQ_LENGTH", os.getenv("CUDA_MAX_SEQ_LENGTH", "2048")))
            load_4bit = (
                os.getenv("LOAD_IN_4BIT", os.getenv("CUDA_LOAD_IN_4BIT", "true")).lower() == "true"
            )
            return ModelConfig(
                name=model_name,
                max_seq_length=max_seq,
                load_in_4bit=load_4bit,
                dtype=None,
                hf_repo_id=os.getenv("HF_REPO_ID"),
            )

        # Default MLX
        model_name = os.getenv(
            "APPLE_MODEL_NAME",
            os.getenv("MODEL_NAME", "mlx-community/Qwen2.5-7B-Instruct-4bit"),
        )
        return ModelConfig(
            name=model_name,
            max_seq_length=int(os.getenv("APPLE_MAX_SEQ_LENGTH", "2048")),
            load_in_4bit=True,  # Forçado para MLX 4bit quant models geralmente
            hf_repo_id=os.getenv("HF_REPO_ID"),
        )

    @staticmethod
    def _load_lora() -> LoRAConfig:
        """Load LoRA configuration settings."""
        backend = os.getenv("TRAINING_BACKEND", "mlx").lower()
        if backend == "unsloth":
            return LoRAConfig(
                r=int(os.getenv("LORA_R", os.getenv("CUDA_LORA_RANK", "16"))),
                alpha=int(os.getenv("LORA_ALPHA", os.getenv("CUDA_LORA_ALPHA", "16"))),
                dropout=float(os.getenv("LORA_DROPOUT", os.getenv("CUDA_LORA_DROPOUT", "0.0"))),
                quantization_method=os.getenv("GGUF_QUANTIZATION", "q4_k_m"),
            )

        return LoRAConfig(
            r=int(os.getenv("APPLE_LORA_RANK", "16")),
            alpha=int(os.getenv("APPLE_LORA_ALPHA", "16")),
            dropout=float(os.getenv("APPLE_LORA_DROPOUT", "0.0")),
            quantization_method="q8_0",
        )

    @staticmethod
    def _load_training() -> TrainingConfig:
        """Load training configuration settings."""
        backend = os.getenv("TRAINING_BACKEND", "mlx").lower()
        if backend == "unsloth":
            return TrainingConfig(
                batch_size=int(os.getenv("TRAINING_BATCH_SIZE", os.getenv("CUDA_BATCH_SIZE", "2"))),
                max_steps=int(os.getenv("TRAINING_MAX_STEPS", os.getenv("CUDA_NUM_ITERS", "60"))),
                learning_rate=float(
                    os.getenv(
                        "TRAINING_LEARNING_RATE",
                        os.getenv("CUDA_LEARNING_RATE", "2e-4"),
                    )
                ),
                steps_per_eval=100,  # Not in cuda env explicitly/standardized?
                steps_per_report=10,
                val_batches=25,
                save_every_steps=100,
                mlx_use_metal=False,
                mlx_grad_checkpoint=True,
                wandb_project=os.getenv("WANDB_PROJECT"),
                wandb_watch=os.getenv("WANDB_WATCH", "false"),
            )

        return TrainingConfig(
            batch_size=int(os.getenv("APPLE_BATCH_SIZE", "2")),
            max_steps=int(os.getenv("APPLE_NUM_ITERS", "60")),
            learning_rate=float(os.getenv("APPLE_LEARNING_RATE", "1e-5")),
            steps_per_eval=int(os.getenv("APPLE_STEPS_PER_EVAL", "100")),
            steps_per_report=int(os.getenv("APPLE_STEPS_PER_REPORT", "10")),
            val_batches=int(os.getenv("APPLE_VAL_BATCHES", "25")),
            save_every_steps=int(os.getenv("APPLE_SAVE_EVERY", "100")),
            mlx_use_metal=os.getenv("APPLE_USE_METAL", "true").lower() == "true",
            mlx_grad_checkpoint=os.getenv("APPLE_GRAD_CHECKPOINT", "true").lower() == "true",
            wandb_project=os.getenv("WANDB_PROJECT"),
            wandb_watch=os.getenv("WANDB_WATCH", "false"),
        )

    @staticmethod
    def _load_data() -> DataConfig:
        """Load data configuration settings."""
        backend = os.getenv("TRAINING_BACKEND", "mlx").lower()
        if backend == "unsloth":
            return DataConfig(
                data_dir=os.getenv("CUDA_DATA_DIR", "data"),  # Generic
                output_dir=os.getenv("TRAINING_OUTPUT_DIR", "outputs"),
                adapter_path=os.getenv(
                    "CHECKPOINT_DIR", os.getenv("CUDA_ADAPTER_PATH", "adapters_cuda")
                ),
                fused_model_path=os.getenv(
                    "FINAL_MODEL_NAME",
                    os.getenv("CUDA_FUSED_MODEL", "models/fused_cuda"),
                ),
                gguf_output_path=os.getenv("FINAL_MODEL_NAME", "models/final") + ".gguf",
            )

        return DataConfig(
            data_dir=os.getenv("APPLE_DATA_DIR", "data"),
            output_dir="outputs",  # Genérico
            adapter_path=os.getenv("APPLE_ADAPTER_PATH", "adapters"),
            fused_model_path=os.getenv("APPLE_FUSED_MODEL", "models/fused"),
            gguf_output_path=os.getenv("APPLE_GGUF_OUTPUT", "models/final.gguf"),
        )
