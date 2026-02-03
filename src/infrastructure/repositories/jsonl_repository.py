import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sklearn.model_selection import train_test_split

from ...domain.interfaces.data_repository import IDataRepository


class JSONLDataRepository(IDataRepository):
    """Implementação de repositório para arquivos JSONL."""

    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from a JSONL file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return data

    def save_jsonl(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """Save data to a JSONL file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def validate_alpaca_format(self, data: List[Dict[str, Any]]) -> bool:
        """Validate if data follows the Alpaca format (instruction/output)."""
        required = {"instruction", "output"}
        for item in data:
            if not required.issubset(item.keys()):
                return False
        return True

    def split_data(
        self, data: List[Dict[str, Any]], val_ratio: float, seed: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split data into training and validation sets."""
        if val_ratio <= 0:
            return data, []

        train, val = train_test_split(data, test_size=val_ratio, random_state=seed)
        return train, val
