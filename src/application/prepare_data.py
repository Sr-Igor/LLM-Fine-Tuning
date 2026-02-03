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
    """Use Case: Prepare and Validate Data for Training."""

    def __init__(self, repository: IDataRepository):
        """Initialize the PrepareDataUseCase with a data repository."""
        self.repository = repository

    def execute(self, config: DataConfig, val_ratio: float = 0.1, seed: int = 42) -> Dict[str, Any]:
        """
        Executes preparation pipeline.

        1. Loads raw data (merged or raw).
        2. Validates format.
        3. Converts to MLX format (Chat) if necessary.
        4. Splits into training/validation.
        5. Saves to processed folder.
        """
        # Define paths
        input_path = Path(config.processed_dir) / "train_dataset_final.jsonl"
        # (Ideally this input would come from config or argument,
        # assuming default for now)

        if not input_path.exists():
            # Tries raw source if processed does not exist
            input_path = Path(config.raw_dir) / "train.jsonl"  # Example

        if not input_path.exists():
            raise FileNotFoundError(f"No data found in {config.processed_dir} or {config.raw_dir}")

        # 1. Load
        data = self.repository.load_jsonl(str(input_path))

        # 2. Validate
        if not self.repository.validate_alpaca_format(data):
            # Try to convert or fail
            # For simplicity, assuming it must be in Alpaca or fail
            raise ValueError("Data is not in expected Alpaca format!")

        # 3. Convert to MLX (Chat Format)
        # MLX accepts: {"messages": [{"role": "user", "content": ...}, ...]}
        mlx_data = self._convert_to_mlx(data)

        # 4. Split
        train, val = self.repository.split_data(mlx_data, val_ratio, seed)

        # 5. Save
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
