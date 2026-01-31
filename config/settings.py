"""
Configuration settings for the LLM project.

This module defines configuration classes for the model, training, and
project settings, loading values from environment variables with default
fallbacks.
"""

from dataclasses import dataclass
from typing import Optional, List
import os
import json


@dataclass
class ModelConfig:
    """Configuration for the Large Language Model settings."""
    model_name: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: Optional[str] = None  # None = Auto detection

    @classmethod
    def from_env(cls):
        """Creates a ModelConfig instance from environment variables."""
        return cls(
            model_name=os.getenv("MODEL_NAME", "unsloth/Qwen2.5-32B-Instruct"),
            max_seq_length=int(os.getenv("MAX_SEQ_LENGTH", "2048")),
            load_in_4bit=os.getenv("LOAD_IN_4BIT", "true").lower() == "true"
        )


@dataclass
class LoraConfig:
    """Configuration for LoRA (Low-Rank Adaptation)."""
    r: int = 16
    alpha: int = 16
    dropout: float = 0
    target_modules: Optional[List[str]] = None
    random_state: int = 3407
    quantization_method: str = "q4_k_m"

    @classmethod
    def from_env(cls):
        """Creates a LoraConfig instance from environment variables."""
        target_modules_str = os.getenv(
            "LORA_TARGET_MODULES",
            '["q_proj", "k_proj", "v_proj", "o_proj", '
            '"gate_proj", "up_proj", "down_proj"]'
        )
        try:
            target_modules = json.loads(target_modules_str)
        except json.JSONDecodeError:
            # Fallback safe default if parsing fails
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        return cls(
            r=int(os.getenv("LORA_R", "16")),
            alpha=int(os.getenv("LORA_ALPHA", "16")),
            dropout=float(os.getenv("LORA_DROPOUT", "0")),
            target_modules=target_modules,
            random_state=int(os.getenv("LORA_RANDOM_STATE", "3407")),
            quantization_method=os.getenv("GGUF_QUANTIZATION", "q4_k_m")
        )


@dataclass
class TrainingConfig:
    """Configuration for the model training hyperparameters."""
    batch_size: int = 2
    grad_accumulation: int = 4
    warmup_steps: int = 5
    max_steps: int = 60
    learning_rate: float = 2e-4
    output_dir: str = "outputs_checkpoints"
    seed: int = 3407
    dataset_num_proc: int = 2
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"

    @classmethod
    def from_env(cls):
        """Creates a TrainingConfig instance from environment variables."""
        return cls(
            max_steps=int(os.getenv("TRAINING_MAX_STEPS", "60")),
            batch_size=int(os.getenv("TRAINING_BATCH_SIZE", "2")),
            grad_accumulation=int(
                os.getenv("TRAINING_GRAD_ACCUMULATION", "4")),
            warmup_steps=int(os.getenv("TRAINING_WARMUP_STEPS", "5")),
            learning_rate=float(os.getenv("TRAINING_LEARNING_RATE", "2e-4")),
            output_dir=os.getenv("TRAINING_OUTPUT_DIR", "outputs_checkpoints"),
            seed=int(os.getenv("TRAINING_SEED", "3407")),
            dataset_num_proc=int(os.getenv("TRAINING_DATASET_NUM_PROC", "2")),
            optim=os.getenv("TRAINING_OPTIM", "adamw_8bit"),
            weight_decay=float(os.getenv("TRAINING_WEIGHT_DECAY", "0.01")),
            lr_scheduler_type=os.getenv("TRAINING_LR_SCHEDULER", "linear")
        )


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    source_dir: str
    output_file: str
    generator_model: str
    system_instruction: str
    chunk_size: int = 2000
    overlap: int = 200
    chat_subject: str = "TEMA"
    chat_context: str = "CONTEXTO"
    chat_question: str = "PERGUNTA"
    chat_history: str = "HISTÓRICO"
    chat_language: str = "IDIOMA"

    @classmethod
    def from_env(cls):
        """Creates a SyntheticConfig instance from environment variables."""
        return cls(
            source_dir=os.getenv("SYNTHETIC_SOURCE_DIR",
                                 "data/source_documents"),
            output_file=os.getenv("SYNTHETIC_OUTPUT_FILE",
                                  "data/raw/train_data_synthetic.jsonl"),
            generator_model=os.getenv("SYNTHETIC_GENERATOR_MODEL", "llama3.1"),
            system_instruction=os.getenv("SYNTHETIC_SYSTEM_INSTRUCTION", ""),
            chunk_size=int(os.getenv("SYNTHETIC_CHUNK_SIZE", "2000")),
            overlap=int(os.getenv("SYNTHETIC_OVERLAP", "200")),
            chat_subject=os.getenv("AI_CHAT_SUBJECT", "SUBJECT"),
            chat_context=os.getenv("AI_CHAT_CONTEXT", "CONTEXT"),
            chat_question=os.getenv("AI_CHAT_QUESTION", "QUESTION"),
            chat_history=os.getenv("AI_CHAT_HISTORY", "HISTORY"),
            chat_language=os.getenv("AI_CHAT_LANGUAGE", "LANGUAGE")
        )


@dataclass
class ProjectConfig:
    """Configuration for model, training, and dataset settings."""
    model: ModelConfig
    training: TrainingConfig
    synthetic: SyntheticConfig
    lora: LoraConfig
    dataset_path: str
    final_model_name: str  # Nome do arquivo GGUF

    @classmethod
    def from_env(cls):
        """Creates a ProjectConfig instance from environment variables."""
        return cls(
            model=ModelConfig.from_env(),
            training=TrainingConfig.from_env(),
            synthetic=SyntheticConfig.from_env(),
            lora=LoraConfig.from_env(),
            dataset_path=os.getenv(
                "DATASET_PATH", "data/processed/train_dataset_final.jsonl"),
            final_model_name=os.getenv(
                "FINAL_MODEL_NAME", "models/llm_qwen_v1")
        )
