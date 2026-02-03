from abc import ABC, abstractmethod

from ..entities.configuration import ProjectConfig


class IModelPublisher(ABC):
    """Interface para publicação de modelos (Hugging Face, etc)."""

    @abstractmethod
    def publish(self, config: ProjectConfig) -> None:
        """
        Publica os artefatos do modelo (adaptadores, fused, gguf).

        Args:
            config: Configuração do projeto.
        """
        pass
