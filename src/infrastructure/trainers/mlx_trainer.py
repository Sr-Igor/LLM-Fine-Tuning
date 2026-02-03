"""
MLX Trainer Implementation.

This module implements the ITrainer interface using the MLX framework,
specifically designed for training on Apple Silicon hardware.
"""

from typing import Any, Dict, List

from ...domain.entities.configuration import ProjectConfig
from ...domain.interfaces.trainer import ITrainer
from ..ui.terminal_presenter import TerminalPresenter


class MLXTrainerAdapter(ITrainer):
    """Adapter for MLX LLM (Apple Silicon)."""

    def __init__(self, presenter: TerminalPresenter):
        """Initialize the MLXTrainerAdapter with a presenter."""
        self.presenter = presenter

    def _build_train_cmd(self, config: ProjectConfig) -> List[str]:
        """Builds CLI command for mlx_lm.lora."""
        c = config
        cmd = [
            "python",
            "-m",
            "mlx_lm.lora",
            "--model",
            c.model.name,
            "--train",
            "--data",
            c.data.processed_dir,  # Assumes data already prepared here
            "--batch-size",
            str(c.training.batch_size),
            # MLX uses num-layers for rank or similar depending on version, adjust conf
            "--num-layers",
            str(c.lora.r),
            "--iters",
            str(c.training.max_steps),
            "--learning-rate",
            str(c.training.learning_rate),
            "--adapter-path",
            c.data.adapter_path,
            "--save-every",
            str(c.training.save_every_steps),
            "--steps-per-eval",
            str(c.training.steps_per_eval),
            "--val-batches",
            str(c.training.val_batches),
            "--seed",
            str(c.training.seed),
        ]

        if c.training.mlx_grad_checkpoint:
            cmd.append("--grad-checkpoint")

        if c.training.wandb_project:
            cmd.append("--log-to-wandb")
            cmd.append("--wandb-project")
            cmd.append(c.training.wandb_project)

        return cmd

    def train(self, config: ProjectConfig) -> Dict[str, Any]:
        """Executes training."""
        self.presenter.log("Starting MLX training...", "info")

        # Disk space monitoring before
        # (Implement check if necessary)

        cmd = self._build_train_cmd(config)
        self.presenter.run_command_with_spinner(cmd, "Training MLX model...")

        self.presenter.log("Training finished!", "success")
        return {"status": "success", "steps": config.training.max_steps}

    def fuse_adapters(self, config: ProjectConfig) -> str:
        """Fuses adapters."""
        self.presenter.log("Fusing adapters...", "info")

        cmd = [
            "python",
            "-m",
            "mlx_lm.fuse",
            "--model",
            config.model.name,
            "--adapter-path",
            config.data.adapter_path,
            "--save-path",
            config.data.fused_model_path,
        ]

        self.presenter.run_command_with_spinner(cmd, "Fusing model...")
        self.presenter.log(f"Fused model saved at: {config.data.fused_model_path}", "success")
        return config.data.fused_model_path

    def export_to_gguf(self, config: ProjectConfig) -> str:
        """Exports to GGUF (requires llama.cpp)."""
        # Note: GGUF export in native MLX or via llama.cpp
        # For simplicity, we'll assume the user has llama.cpp set up
        # or use mlx script if available.
        # Currently MLX doesn't export GGUF directly, uses conversion scripts.
        # I'll keep a functional placeholder warning about llama.cpp

        self.presenter.log("GGUF export requires external llama.cpp for now.", "warning")
        return ""
