"""
Interface for LLM Clients.

This module defines the contract for Large Language Model clients,
typically used for synthetic data generation steps.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class ILLMClient(ABC):
    """Interface for LLM clients (synthetic data generation)."""

    @abstractmethod
    def generate(self, model: str, prompt: str, system: str, options: Optional[Dict] = None) -> str:
        """Generates text using an LLM."""
        pass
