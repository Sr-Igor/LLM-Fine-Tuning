"""
Script principal para execução do pipeline de treinamento do modelo.
"""
from config.settings import ProjectConfig, ModelConfig, TrainingConfig
from src.model_loader import ModelManager
from src.data_handler import load_and_process_data
from src.trainer_engine import run_training

# ==========================================
# CONFIGURAÇÃO DO PROJETO (Setup)
# ==========================================
project_config = ProjectConfig(
    # Selecione o modelo aqui:
    model=ModelConfig(
        # Ou "unsloth/Meta-Llama-3.1-8B-Instruct"
        model_name="unsloth/Qwen2.5-32B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True
    ),
    training=TrainingConfig(
        max_steps=60,  # Aumente para 300 em produção
        batch_size=2,
        output_dir="outputs_checkpoints"
    ),
    dataset_path="data/processed/train_dataset_final.jsonl",
    final_model_name="models/planus_qwen_v1"  # Pasta onde salva o GGUF final
)


def main():
    """
    Função principal que orquestra o carregamento, processamento e treinamento.
    """
    # 1. Inicializar Gerenciador de Modelo
    manager = ModelManager(project_config.model)
    model, tokenizer = manager.load_base_model()
    model = manager.apply_lora_adapters()

    # 2. Carregar Dados
    dataset = load_and_process_data(project_config.dataset_path, tokenizer)

    # 3. Executar Treino
    run_training(model, tokenizer, dataset, project_config.training)

    # 4. Exportar
    manager.save_to_gguf(project_config.final_model_name)


if __name__ == "__main__":
    main()
