# @title 📦 Pipeline v0: Setup & Dependências (Clean Install)
"""
Script de inicialização para preparar o ambiente Google Colab
com as dependências necessárias para o Unsloth e Hugging Face.
"""

import subprocess


def install_package(command: str, description: str):
    """
    Executa um comando de instalação via pip e monitora o sucesso.

    Args:
        command (str): O comando pip completo.
        description (str): Descrição amigável para o log.

    Raises:
        RuntimeError: Se o comando falhar.
    """
    print(f"⏳ {description}...")

    # Redireciona stdout e stderr para capturar logs em tempo real
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Imprime a saída enquanto instala
    for line in process.stdout:
        print(line, end='')

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"❌ Falha na instalação de: {description}")

    print(f"✅ Sucesso: {description}\n")


def check_gpu():
    """Verifica se a GPU está ativa."""
    try:
        gpu_info = subprocess.check_output("nvidia-smi", shell=True, text=True)
        if "Tesla T4" in gpu_info or "A100" in gpu_info or "L4" in gpu_info:
            print("✅ GPU Detectada e Pronta.")
        else:
            print("⚠️ AVISO: GPU não detectada ou modelo desconhecido.")
    except subprocess.CalledProcessError:
        print(
            "❌ ERRO CRÍTICO: Nenhuma GPU encontrada. Ative em 'Runtime >\n"
            " Change runtime type'."
        )


def main():
    """Função principal de setup."""
    print("🏗️ Iniciando Setup do Ambiente Planus...\n")

    try:
        # 1. Verificar GPU
        check_gpu()

        # 2. Instalar Unsloth (Core)
        # Instalação específica para Colab com patches de otimização
        install_package(
            'pip install --upgrade --force-reinstall --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"',
            "Instalando Core do Unsloth"
        )

        # 3. Instalar Dependências de Treino e Inferência
        # --no-deps evita conflitos de versão com o PyTorch pré-instalado no Colab
        deps_command = (
            'pip install --no-deps "xformers<0.0.29" "trl<0.9.0" peft accelerate bitsandbytes '
            'python-dotenv huggingface_hub tyro unsloth_zoo'
        )
        install_package(deps_command, "Instalando TRL, Peft e Utilitários")

        print("🎉 Setup Concluído! O ambiente está pronto para rodar o script de Resgate.")

    except Exception as error:  # pylint: disable=broad-exception-caught
        print(f"\n⛔ Erro Fatal no Setup: {error}")
        print("👉 Sugestão: Tente reiniciar a sessão (Runtime > Restart session) e tente novamente.")


if __name__ == "__main__":
    main()
