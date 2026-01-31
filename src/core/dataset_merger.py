"""
Módulo responsável pela fusão e validação de datasets brutos.
"""
import os
import json
import random
import glob

from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# ==========================================
# CONFIGURAÇÕES
# ==========================================
# Onde estão os arquivos parciais (rules.jsonl, synthetic.jsonl)
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw")
# Onde será salvo o arquivo final limpo
PROCESSED_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")

# Pega o nome do arquivo final da variável DATASET_PATH
# (se definida) ou usa o padrão
dataset_path_env = os.getenv(
    "DATASET_PATH", "data/processed/train_dataset_final.jsonl")
OUTPUT_FILENAME = os.path.basename(dataset_path_env)


def _validate_entry(entry, index, is_line=True):
    """Retorna True se a entrada for válida, senão imprime aviso."""
    required_keys = ("instruction", "input", "output")
    if all(k in entry for k in required_keys):
        return True

    prefix = "Linha" if is_line else "Item"
    print(f"      ⚠️ {prefix} {index} ignorado: Campos faltando.")
    return False


def _process_json_array(file_path):
    """
    Processa arquivo formatado como Array de JSON (independente da extensão).
    """
    filename = os.path.basename(file_path)
    valid_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            for i, entry in enumerate(data):
                if _validate_entry(entry, i, is_line=False):
                    valid_data.append(entry)
        else:
            print(f"      ⚠️ Arquivo {filename} não é uma lista JSON válida.")

        return valid_data, 0
    except json.JSONDecodeError as e:
        print(f"      ❌ Erro de JSON no arquivo {filename}: {e}")
        return [], 1


def _process_jsonl(file_path):
    """Processa arquivo JSONL (uma validação por linha)."""
    filename = os.path.basename(file_path)
    valid_data = []
    file_errors = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    if _validate_entry(entry, line_num, is_line=True):
                        valid_data.append(entry)
                except json.JSONDecodeError:
                    print(f"      ❌ Erro de JSON no arquivo {filename} "
                          f"na linha {line_num}")
                    file_errors += 1
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"❌ Erro ao ler arquivo {filename}: {e}")

    return valid_data, file_errors


def _process_file(file_path):
    """
    Identifica formato (JSON Array ou JSONL) e processa.
    """
    filename = os.path.basename(file_path)
    print(f"   📄 Processando: {filename}...")

    # Detecta se começa com [ (Array)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Pula espaços em branco iniciais
            while True:
                char = f.read(1)
                if not char or not char.isspace():
                    break
            is_array = char == '['
    except Exception:  # pylint: disable=broad-exception-caught
        is_array = False

    if is_array:
        return _process_json_array(file_path)
    else:
        return _process_jsonl(file_path)


def _save_dataset(data, output_path):
    """
    Salva a lista de dados no arquivo de saída.
    """
    print(f"💾 Salvando {len(data)} exemplos em '{output_path}'...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        return True
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"❌ Erro ao salvar arquivo final: {e}")
        return False


def merge_datasets():
    """
    Funde datasets brutos, valida entradas, embaralha e salva o resultado.
    """
    print(f"🔄 Iniciando fusão de datasets em '{RAW_DATA_DIR}'...")

    # Garante que a pasta de saída existe
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Encontra todos os arquivos .jsonl na pasta raw
    jsonl_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.jsonl"))

    if not jsonl_files:
        print("❌ Nenhum arquivo .jsonl encontrado em data/raw/")
        return

    all_data = []
    total_errors = 0

    # Lê cada arquivo usando a função auxiliar
    for file_path in jsonl_files:
        data, errors = _process_file(file_path)
        all_data.extend(data)
        total_errors += errors

    # Embaralha os dados (Shuffle)
    # Isso é CRUCIAL para o treinamento ser estável
    print("🔀 Embaralhando os dados...")
    random.shuffle(all_data)

    # Salva o arquivo final
    output_path = os.path.join(PROCESSED_DIR, OUTPUT_FILENAME)
    _save_dataset(all_data, output_path)

    print("\n--- RELATÓRIO FINAL ---")
    print(f"✅ Arquivos processados: {len(jsonl_files)}")
    print(f"✅ Linhas válidas: {len(all_data)}")
    print(f"❌ Linhas corrompidas: {total_errors}")
    print(f"🚀 Dataset pronto para treino: {output_path}")


if __name__ == "__main__":
    merge_datasets()
