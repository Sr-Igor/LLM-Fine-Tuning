"""
MLX Trainer Implementation.

This module implements the ITrainer interface using the MLX framework,
specifically designed for training on Apple Silicon hardware.
"""

import sys
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
        """Builds CLI command for mlx_lm.lora using a config file."""
        from pathlib import Path

        import yaml

        c = config

        # Ensure adapter directory exists so we can save config there
        adapter_dir = Path(c.data.adapter_path)
        adapter_dir.mkdir(parents=True, exist_ok=True)

        yaml_path = adapter_dir / "train_config.yaml"

        # Create a temporary YAML config for MLX
        # MLX expects a flat yaml or specific structure.
        # Referencing standard mlx_lm config format:
        # Adjusted to flat structure commonly accepted by mlx_lm.lora config parser
        mlx_config = {
            "model": c.model.name,
            "train": True,
            "data": c.data.processed_dir,
            "seed": c.training.seed,
            "lora": {
                "r": c.lora.r,
                "alpha": c.lora.alpha,
                "dropout": c.lora.dropout,
            },
            "batch_size": c.training.batch_size,
            "iters": c.training.max_steps,
            "learning_rate": c.training.learning_rate,
            "steps_per_eval": c.training.steps_per_eval,
            "val_batches": c.training.val_batches,
            "save_every": c.training.save_every_steps,
            "adapter_path": c.data.adapter_path,
            "max_seq_length": c.model.max_seq_length,
            "grad_checkpoint": c.training.mlx_grad_checkpoint,
        }

        # Remove unsupported keys if any
        # items_per_eval removed as it might conflict or be auto-calced

        # We can also pass num_layers (layers to fine tune) if supported in yaml
        # usually key is 'lora_layers' in newer versions, or handled in lora_parameters
        # checking mlx code, 'num_layers' or 'lora_layers' in argparse maps to config.
        # Let's write essential ones to yaml and pass overrides via CLI where safe.

        # Save config for reproducibility and usage
        with open(yaml_path, "w") as f:
            yaml.dump(mlx_config, f)

        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.lora",
            "--config",
            str(yaml_path),
        ]

        if c.training.wandb_project:
            cmd.append("--report-to")
            cmd.append("wandb")
            cmd.append("--project-name")
            cmd.append(c.training.wandb_project)

        return cmd

    def train(self, config: ProjectConfig) -> Dict[str, Any]:
        """Executes training."""
        self.presenter.log("Starting MLX training...", "info")

        # Auto-login to WandB if key is available
        import os

        wandb_key = os.getenv("WANDB_API_KEY")
        if wandb_key:
            try:
                import wandb

                self.presenter.log("Logging in to WandB...", "info")
                wandb.login(key=wandb_key)
            except ImportError:
                self.presenter.log(
                    "WandB configured but package not found. Run pip install wandb", "warning"
                )
            except Exception as e:
                self.presenter.log(f"WandB login failed: {e}", "warning")

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
            sys.executable,
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
