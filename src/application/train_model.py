"""
Módulo do caso de uso para treinamento de modelos.

Este módulo contém a implementação do caso de uso TrainModelUseCase,
responsável por orquestrar o processo de treinamento e fusão de adaptadores.
"""

from typing import Any, Dict

from ..domain.entities.configuration import ProjectConfig
from ..domain.interfaces.trainer import ITrainer


class TrainModelUseCase:
    """Caso de uso: Treinar Modelo."""

    def __init__(self, trainer: ITrainer):
        """Initialize the TrainModelUseCase with a trainer instance."""
        self.trainer = trainer

    def execute(self, config: ProjectConfig) -> Dict[str, Any]:
        """Orquestra o treinamento."""
        # Poderia ter lógica extra aqui (verificar se dados existem,
        # limpar cache, etc)

        # 1. Treinar
        result = self.trainer.train(config)

        # 2. Fusão automática (opcional, mas comum)
        fused_path = self.trainer.fuse_adapters(config)
        result["fused_path"] = fused_path

        return result
