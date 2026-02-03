from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class IDataRepository(ABC):
    """Interface para persistência e manipulação de dados."""

    @abstractmethod
    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Carrega lista de objetos de um arquivo JSONL."""
        pass

    @abstractmethod
    def save_jsonl(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """Salva lista de objetos em um arquivo JSONL."""
        pass

    @abstractmethod
    def validate_alpaca_format(self, data: List[Dict[str, Any]]) -> bool:
        """Valida se os dados estão no formato Alpaca {instruction, input, output}."""
        pass

    @abstractmethod
    def split_data(
        self, data: List[Dict[str, Any]], val_ratio: float, seed: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Divide dados em treino e validação."""
        pass
