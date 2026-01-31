"""
Configuration settings for the Planuze LLM project.

This module defines configuration classes for the model, training, and
project settings, loading values from environment variables with default
fallbacks.
"""

from dataclasses import dataclass
from typing import Optional
import os


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
class TrainingConfig:
    """Configuration for the model training hyperparameters."""
    batch_size: int = 2
    grad_accumulation: int = 4
    warmup_steps: int = 5
    max_steps: int = 60
    learning_rate: float = 2e-4
    output_dir: str = "outputs_checkpoints"
    seed: int = 3407

    @classmethod
    def from_env(cls):
        """Creates a TrainingConfig instance from environment variables."""
        return cls(
            max_steps=int(os.getenv("TRAINING_MAX_STEPS", "60")),
            batch_size=int(os.getenv("TRAINING_BATCH_SIZE", "2")),
            output_dir=os.getenv("TRAINING_OUTPUT_DIR", "outputs_checkpoints")
        )


@dataclass
class ProjectConfig:
    """Configuration for model, training, and dataset settings."""
    model: ModelConfig
    training: TrainingConfig
    dataset_path: str
    final_model_name: str  # Nome do arquivo GGUF

    @classmethod
    def from_env(cls):
        """Creates a ProjectConfig instance from environment variables."""
        return cls(
            model=ModelConfig.from_env(),
            training=TrainingConfig.from_env(),
            dataset_path=os.getenv(
                "DATASET_PATH", "data/processed/train_dataset_final.jsonl"),
            final_model_name=os.getenv(
                "FINAL_MODEL_NAME", "models/planus_qwen_v1")
        )
