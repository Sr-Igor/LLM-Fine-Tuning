from abc import ABC, abstractmethod
from typing import Dict, Optional


class ILLMClient(ABC):
    """Interface para clientes de LLM (geração de dados sintéticos)."""

    @abstractmethod
    def generate(
        self, model: str, prompt: str, system: str, options: Optional[Dict] = None
    ) -> str:
        """Gera texto usando um LLM."""
        pass
