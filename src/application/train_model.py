"""
Model training use case module.

This module contains the implementation of the TrainModelUseCase,
responsible for orchestrating the training process and adapter fusion.
"""

from typing import Any, Dict

from ..domain.entities.configuration import ProjectConfig
from ..domain.interfaces.trainer import ITrainer


class TrainModelUseCase:
    """Use Case: Train Model."""

    def __init__(self, trainer: ITrainer):
        """Initialize the TrainModelUseCase with a trainer instance."""
        self.trainer = trainer

    def execute(self, config: ProjectConfig) -> Dict[str, Any]:
        """Orchestrates the training process."""
        # Could have extra logic here (check if data exists,
        # clear cache, etc.)

        # 1. Train
        result = self.trainer.train(config)

        # 2. Automatic fusion (optional, but common)
        fused_path = self.trainer.fuse_adapters(config)
        result["fused_path"] = fused_path

        return result
