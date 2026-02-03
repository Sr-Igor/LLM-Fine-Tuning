"""
Interface for Model Publishers.

This module defines the IModelPublisher interface, responsible for pushing
trained model artifacts to external repositories like Hugging Face Hub.
"""

from abc import ABC, abstractmethod

from ..entities.configuration import ProjectConfig


class IModelPublisher(ABC):
    """Interface for model publishing (Hugging Face, etc.)."""

    @abstractmethod
    def publish(self, config: ProjectConfig) -> None:
        """
        Publishes model artifacts (adapters, fused, gguf).

        Args:
            config: Project configuration.
        """
        pass
