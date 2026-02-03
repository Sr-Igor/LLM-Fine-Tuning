from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BackendConfig:
    """Configuração do Backend de Treinamento."""

    type: str = "mlx"  # mlx or unsloth


@dataclass
class ModelConfig:
    """Configuração do Modelo Base."""

    name: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: Optional[str] = None


@dataclass
class LoRAConfig:
    """Configuração de Adaptação Low-Rank (LoRA)."""

    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    random_state: int = 3407
    quantization_method: str = "q4_k_m"  # Para exportação GGUF


@dataclass
class TrainingConfig:
    """Configuração de Treinamento."""

    batch_size: int = 2
    grad_accumulation: int = 4
    warmup_steps: int = 5
    max_steps: int = 60
    learning_rate: float = 2e-4
    seed: int = 3407
    optim: str = "adamw_8bit"

    # MLX specifics (poderia ser subclass, mas manterei simples aqui)
    mlx_use_metal: bool = True
    mlx_grad_checkpoint: bool = True

    # Intervals
    steps_per_eval: int = 100
    steps_per_report: int = 10
    save_every_steps: int = 100
    val_batches: int = 25


@dataclass
class DataConfig:
    """Configuração de Dados e Caminhos."""

    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"

    train_file: str = "train.jsonl"
    val_file: str = "valid.jsonl"

    # Caminhos de saída
    output_dir: str = "outputs"
    adapter_path: str = "adapters"
    fused_model_path: str = "models/fused"
    gguf_output_path: str = "models/final.gguf"


@dataclass
class ProjectConfig:
    """Configuração Agregada do Projeto."""

    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig
    data: DataConfig
    backend: BackendConfig
