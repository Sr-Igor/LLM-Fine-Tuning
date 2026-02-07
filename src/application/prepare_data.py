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
        # 1. Load Data
        data = []
        loaded_files = []

        # Priority 1: Aggregate all .jsonl files in raw_dir
        raw_files = list(Path(config.raw_dir).glob("*.jsonl"))
        for f in raw_files:
            file_data = self.repository.load_jsonl(str(f))
            if file_data:
                data.extend(file_data)
                loaded_files.append(f.name)

        # Priority 2: Fallback to existing processed file if no raw data found
        if not data:
            fallback_path = Path(config.processed_dir) / \
                "train_dataset_final.jsonl"
            if fallback_path.exists():
                data = self.repository.load_jsonl(str(fallback_path))
                loaded_files.append(fallback_path.name)

        if not data:
            raise FileNotFoundError(
                f"No populated JSONL files found in \n"
                f"{config.raw_dir} (*.jsonl) or {config.processed_dir}"
            )

        # 1.5 Deduplicate
        original_count = len(data)
        unique_data = []
        seen = set()

        for item in data:
            # Create a identifying tuple from content
            # We use the main fields that define the example
            key = (
                item.get('instruction', '').strip(),
                item.get('input', '').strip(),
                item.get('output', '').strip()
            )

            if key not in seen:
                seen.add(key)
                unique_data.append(item)

        data = unique_data
        if original_count > len(data):
            print(
                f"  → Removed {original_count - len(data)} duplicate examples.")

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
