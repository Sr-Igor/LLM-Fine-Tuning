"""
Engine de treinamento utilizando SFTTrainer.
"""
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from config.settings import TrainingConfig


def run_training(model, tokenizer, dataset, config: TrainingConfig):
    """
    Executa o treinamento do modelo.

    Args:
        model: Modelo a ser treinado.
        tokenizer: Tokenizer associado.
        dataset: Dataset de treino.
        config (TrainingConfig): Configuração de treinamento.

    Returns:
        TrainerStats: Estatísticas do treinamento.
    """
    print("🚀 Configurando SFTTrainer...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,  # Poderia vir da config também
        dataset_num_proc=2,
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
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=config.seed,
            output_dir=config.output_dir,
        ),
    )

    trainer_stats = trainer.train()
    return trainer_stats
