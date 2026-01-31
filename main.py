"""
Script principal para execução do pipeline de treinamento do modelo.
"""

from dotenv import load_dotenv

from config.settings import ProjectConfig
from src.core.model_loader import ModelManager
from src.core.data_handler import load_and_process_data
from src.core.trainer_engine import run_training

# Carrega variáveis de ambiente
load_dotenv()

# ==========================================
# CONFIGURAÇÃO DO PROJETO (Setup)
# ==========================================
project_config = ProjectConfig.from_env()


def main():
    """
    Função principal que orquestra o carregamento, processamento e treinamento.
    """
    # 1. Inicializar Gerenciador de Modelo
    manager = ModelManager(project_config.model, project_config.lora)
    model, tokenizer = manager.load_base_model()
    model = manager.apply_lora_adapters()

    # 2. Carregar Dados
    dataset = load_and_process_data(project_config.dataset_path, tokenizer)

    # 3. Executar Treino
    run_training(
        model,
        tokenizer,
        dataset,
        project_config.training,
        project_config.model.max_seq_length
    )

    # 4. Exportar

    manager.save_to_gguf(
        project_config.final_model_name,
        quantization=project_config.lora.quantization_method
    )


if __name__ == "__main__":
    main()
