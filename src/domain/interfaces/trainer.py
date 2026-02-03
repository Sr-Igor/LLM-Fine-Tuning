"""
Interface for Model Trainers.

This module defines the ITrainer interface, which abstracts the underlying
training framework (such as MLX or Unsloth) from the application logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..entities.configuration import ProjectConfig


class ITrainer(ABC):
    """Abstract interface for training adapters (MLX, Unsloth)."""

    @abstractmethod
    def train(self, config: ProjectConfig) -> Dict[str, Any]:
        """
        Executes model training.

        Args:
            config: Complete project configuration.

        Returns:
            Dictionary with metrics and training results.
        """
        pass

    @abstractmethod
    def fuse_adapters(self, config: ProjectConfig) -> str:
        """
        Fuses LoRA adapters with the base model.

        Returns:
            Path to the fused model.
        """
        pass

    @abstractmethod
    def export_to_gguf(self, config: ProjectConfig) -> str:
        """
        Exports the model to GGUF format.

        Returns:
            Path to the GGUF file.
        """
        pass
