"""
Engine de treinamento utilizando SFTTrainer.
"""
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from config.settings import TrainingConfig
from src.core.utils.logger import logger


def run_training(
    model,
    tokenizer,
    dataset,
    config: TrainingConfig,
    max_seq_length: int
):
    """
    Executa o treinamento do modelo.

    Args:
        model: Modelo a ser treinado.
        tokenizer: Tokenizer associado.
        dataset: Dataset de treino.
        config (TrainingConfig): Configuração de treinamento.
        max_seq_length (int): Comprimento máximo da sequência.

    Returns:
        TrainerStats: Estatísticas do treinamento.
    """
    logger.info("🚀 Configurando SFTTrainer...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=config.dataset_num_proc,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.grad_accumulation,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim=config.optim,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            seed=config.seed,
            output_dir=config.output_dir,
        ),
    )

    trainer_stats = trainer.train()
    logger.info("✅ Treinamento concluído!")
    return trainer_stats
