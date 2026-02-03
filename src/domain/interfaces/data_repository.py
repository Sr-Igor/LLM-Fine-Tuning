"""
Interface for Data Repositories.

This module defines the IDataRepository interface, which outlines the contract
for loading, saving, and manipulating datasets (e.g., JSONL files).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class IDataRepository(ABC):
    """Interface for data persistence and manipulation."""

    @abstractmethod
    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads a list of objects from a JSONL file."""
        pass

    @abstractmethod
    def save_jsonl(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """Saves a list of objects to a JSONL file."""
        pass

    @abstractmethod
    def validate_alpaca_format(self, data: List[Dict[str, Any]]) -> bool:
        """Validates if data follows the Alpaca format {instruction, input, output}."""
        pass

    @abstractmethod
    def split_data(
        self, data: List[Dict[str, Any]], val_ratio: float, seed: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Splits data into training and validation sets."""
        pass
