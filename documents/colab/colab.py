# @title 🚑 Pipeline: Resgate e Upload
"""
Script de resgate para carregar um adaptador LoRA salvo localmente no Google Drive,
corrigir o ID do repositório Hugging Face e realizar o upload no formato GGUF.

Este script é otimizado para rodar no Google Colab e utiliza a biblioteca Unsloth.
"""

import os
import subprocess
from dotenv import load_dotenv
from huggingface_hub import HfApi
from unsloth import FastLanguageModel
from google.colab import drive

# Constantes Globais
PROJECT_ROOT_NAME = "planuze-llm-collab"
DEFAULT_MODEL_NAME = "planus_qwen_v2"
QUANTIZATION_METHOD = "q4_k_m"
MAX_SEQ_LENGTH = 1024


def find_project_path(root_name: str) -> str:
    """
    Localiza o caminho do projeto no Google Drive usando comando find.

    Args:
        root_name (str): Nome da pasta raiz do projeto.

    Returns:
        str: Caminho absoluto do projeto.

    Raises:
        FileNotFoundError: Se a pasta não for encontrada.
    """
    if not os.path.exists('/content/drive'):
        print("ue Montando Google Drive...")
        drive.mount('/content/drive', force_remount=True)

    print(f"🔍 Procurando pasta '{root_name}'...")
    command = [
        "find", "/content/drive/MyDrive",
        "-type", "d",
        "-name", root_name,
        "-print", "-quit"
    ]

    try:
        # Uso seguro de subprocess em vez de os.popen
        result = subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        result = ""

    if not result:
        raise FileNotFoundError(
            f"❌ Pasta do projeto '{root_name}' não encontrada.")

    return result


def get_adapter_path(base_path: str) -> str:
    """
    Verifica os diretórios possíveis onde o adaptador pode ter sido salvo.

    Args:
        base_path (str): Caminho raiz do projeto.

    Returns:
        str: Caminho válido do adaptador.

    Raises:
        FileNotFoundError: Se o adaptador não for encontrado em nenhum local.
    """
    possible_paths = [
        os.path.join(base_path, "outputs", "final_adapter"),
        os.path.join(base_path, "outputs_checkpoints", "final_adapter")
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"❌ Não encontrei o backup do adapter em nenhum destes\
        locais: {possible_paths}"
    )


def correct_repo_id(api: HfApi, env_model_name: str) -> str:
    """
    Gera um ID de repositório válido para o Hugging Face (Usuario/Modelo).

    Args:
        api (HfApi): Instância da API do Hugging Face.
        env_model_name (str): Nome do modelo vindo do arquivo .env.

    Returns:
        str: ID do repositório formatado corretamente.
    """
    user = api.whoami()['name']
    # Pega apenas o último segmento se houver barras (ex: Planuze/models/v2 -> v2)
    model_slug = env_model_name.split("/")[-1]
    return f"{user}/{model_slug}"


def main():
    """
    Função principal de execução do pipeline de resgate.
    """
    print("🏗️ Iniciando Resgate...")

    try:
        # 1. Configuração de Caminhos
        project_path = find_project_path(PROJECT_ROOT_NAME)
        print(f"✅ Diretório do Projeto: {project_path}")

        # 2. Carregar Variáveis de Ambiente
        env_path = os.path.join(project_path, ".env")
        load_dotenv(env_path)

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("❌ Token HF não encontrado no arquivo .env")

        # 3. Validar Nomes de Repositório
        api = HfApi(token=hf_token)
        original_name = os.getenv("FINAL_MODEL_NAME", DEFAULT_MODEL_NAME)
        correct_repo = correct_repo_id(api, original_name)

        print(f"❌ Nome Anterior (Do Env): {original_name}")
        print(f"✅ Nome Corrigido (Final): {correct_repo}")

        # 4. Localizar e Carregar Modelo
        adapter_path = get_adapter_path(project_path)
        print(f"📂 Carregando Adapter de: {adapter_path}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )

        # 5. Upload
        print("\n☁️ Iniciando Conversão e Upload...")
        print("☕ Isso vai levar alguns minutos (GGUF Conversion)...")

        model.push_to_hub_gguf(
            repo_id=correct_repo,
            tokenizer=tokenizer,
            quantization_method=QUANTIZATION_METHOD,
            token=hf_token
        )

        print(
            f"\n🎉 SUCESSO! Modelo salvo em: https://huggingface.co/{correct_repo}")

    except Exception as error:  # pylint: disable=broad-exception-caught
        # Captura genérica intencional para exibir erro final ao usuário no Colab
        print(f"\n❌ Erro Fatal no Pipeline: {error}")


if __name__ == "__main__":
    main()
