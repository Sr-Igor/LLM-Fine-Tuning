from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    model_name: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: Optional[str] = None  # None = Auto detection


@dataclass
class TrainingConfig:
    batch_size: int = 2
    grad_accumulation: int = 4
    warmup_steps: int = 5
    max_steps: int = 60
    learning_rate: float = 2e-4
    output_dir: str = "outputs"
    seed: int = 3407


@dataclass
class ProjectConfig:
    model: ModelConfig
    training: TrainingConfig
    dataset_path: str
    final_model_name: str  # Nome do arquivo GGUF
