"""
Módulo para manipulação e processamento de dados.
"""
from functools import partial
from datasets import load_dataset
from src.planuze.prompt_templates import apply_chat_template


def load_and_process_data(file_path: str, tokenizer):
    """
    Carrega o dataset JSONL e aplica o template de chat.

    Args:
        file_path (str): Caminho para o arquivo de dados.
        tokenizer: Tokenizer a ser utilizado.

    Returns:
        Dataset: Dataset processado.
    """
    print(f"📂 Lendo dataset: {file_path}")

    # Carrega JSONL
    dataset = load_dataset("json", data_files=file_path, split="train")

    # Prepara a função de formatação injetando o tokenizer.
    # Usamos partial para passar o tokenizer sem quebrar a assinatura que o
    # .map espera.
    format_func = partial(apply_chat_template, tokenizer=tokenizer)

    # Aplica mapeamento
    dataset = dataset.map(format_func, batched=True)

    return dataset
