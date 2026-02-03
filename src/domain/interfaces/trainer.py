from abc import ABC, abstractmethod
from typing import Any, Dict

from ..entities.configuration import ProjectConfig


class ITrainer(ABC):
    """Interface abstrata para adaptadores de treinamento (MLX, Unsloth)."""

    @abstractmethod
    def train(self, config: ProjectConfig) -> Dict[str, Any]:
        """
        Executa o treinamento do modelo.

        Args:
            config: Configuração completa do projeto.

        Returns:
            Dicionário com métricas e resultados do treino.
        """
        pass

    @abstractmethod
    def fuse_adapters(self, config: ProjectConfig) -> str:
        """
        Funde os adaptadores LoRA com o modelo base.

        Returns:
            Caminho do modelo fundido.
        """
        pass

    @abstractmethod
    def export_to_gguf(self, config: ProjectConfig) -> str:
        """
        Exporta o modelo para formato GGUF.

        Returns:
            Caminho do arquivo GGUF.
        """
        pass
