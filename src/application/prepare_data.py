"""
Module for the data preparation use case.

This module contains the logic for preparing and validating data
before it is used for training the LLM.
"""

from pathlib import Path
from typing import Any, Dict, List

from ..domain.entities.configuration import DataConfig
from ..domain.interfaces.data_repository import IDataRepository


class PrepareDataUseCase:
    """Caso de uso: Preparar e Validar Dados para Treinamento."""

    def __init__(self, repository: IDataRepository):
        """Initialize the PrepareDataUseCase with a data repository."""
        self.repository = repository

    def execute(
        self, config: DataConfig, val_ratio: float = 0.1, seed: int = 42
    ) -> Dict[str, Any]:
        """
        Executa pipeline de preparação.

        1. Carrega dados brutos (merged ou raw).
        2. Valida formato.
        3. Converte para formato MLX (Chat) se necessário.
        4. Divide em treino/validação.
        5. Salva na pasta processed.
        """
        # Define caminhos
        input_path = Path(config.processed_dir) / "train_dataset_final.jsonl"
        # (Idealmente esse input viria de config ou argumento,
        # assumindo padrão por enquanto)

        if not input_path.exists():
            # Tenta raw source se processed não existir
            input_path = Path(config.raw_dir) / "train.jsonl"  # Exemplo

        if not input_path.exists():
            raise FileNotFoundError(
                f"Nenhum dado encontrado em {config.processed_dir} ou {config.raw_dir}"
            )

        # 1. Carregar
        data = self.repository.load_jsonl(str(input_path))

        # 2. Validar
        if not self.repository.validate_alpaca_format(data):
            # Tentar converter ou falhar
            # Por simplicidade, vamos assumir que deve estar em Alpaca ou falha
            raise ValueError("Dados não estão no formato Alpaca esperado!")

        # 3. Converter para MLX (Chat Format)
        # MLX aceita: {"messages": [{"role": "user", "content": ...}, ...]}
        mlx_data = self._convert_to_mlx(data)

        # 4. Dividir
        train, val = self.repository.split_data(mlx_data, val_ratio, seed)

        # 5. Salvar
        train_path = Path(config.processed_dir) / config.train_file
        val_path = Path(config.processed_dir) / config.val_file

        self.repository.save_jsonl(train, str(train_path))
        self.repository.save_jsonl(val, str(val_path))

        return {
            "train_count": len(train),
            "val_count": len(val),
            "train_path": str(train_path),
            "val_path": str(val_path),
        }

    def _convert_to_mlx(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert data from Alpaca format to MLX chat format."""
        mlx_data = []
        for item in data:
            user_content = item["instruction"]
            if item.get("input"):
                user_content += f"\n\n{item['input']}"

            mlx_item = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": item["output"]},
                ]
            }
            mlx_data.append(mlx_item)
        return mlx_data
