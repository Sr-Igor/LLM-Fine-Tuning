from typing import Any, Dict

from ...domain.entities.configuration import ProjectConfig
from ...domain.interfaces.trainer import ITrainer

# Lazy imports to avoid errors on non-CUDA environments
try:
    import torch  # type: ignore # pylint: disable=import-error
    from transformers import TrainingArguments
    from trl import SFTTrainer  # type: ignore # pylint: disable=import-error
    from unsloth import FastLanguageModel  # type: ignore # pylint: disable=import-error
except ImportError:
    FastLanguageModel = None
    SFTTrainer = None
    TrainingArguments = None
    torch = None


class UnslothTrainerAdapter(ITrainer):
    """Adaptador para treinamento usando Unsloth (CUDA)."""

    def __init__(self, presenter):
        """Initialize the UnslothTrainerAdapter with a presenter."""
        self.presenter = presenter
        if FastLanguageModel is None:
            self.presenter.log(
                "Unsloth not installed. Please install requirements/cuda.txt", "warning"
            )

    def train(self, config: ProjectConfig) -> Dict[str, Any]:
        """Execute the training process using Unsloth."""
        if FastLanguageModel is None:
            raise ImportError("Unsloth is not installed. Cannot train on CUDA backend.")

        self.presenter.log(f"Loading Unsloth model: {config.model.name}", "info")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model.name,
            max_seq_length=config.model.max_seq_length,
            dtype=None,
            load_in_4bit=config.model.load_in_4bit,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora.r,
            target_modules=config.lora.target_modules,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=config.lora.random_state,
            use_rslora=False,
            loftq_config=None,
        )

        training_args = TrainingArguments(
            per_device_train_batch_size=config.training.batch_size,
            gradient_accumulation_steps=config.training.grad_accumulation,
            warmup_steps=config.training.warmup_steps,
            max_steps=config.training.max_steps,
            learning_rate=config.training.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=config.training.steps_per_report,
            optim=config.training.optim,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=config.training.seed,
            output_dir=config.data.output_dir,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=self._load_dataset(config.data.train_file),  # Hypothetical helper
            dataset_text_field="text",
            max_seq_length=config.model.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )

        self.presenter.log("Starting Unsloth training...", "info")
        trainer_stats = trainer.train()

        # Save adapters
        model.save_pretrained(config.data.adapter_path)
        tokenizer.save_pretrained(config.data.adapter_path)

        return {
            "status": "success",
            "global_step": trainer_stats.global_step,
            "training_loss": trainer_stats.training_loss,
        }

    def fuse_adapters(self, config: ProjectConfig) -> str:
        """Fuse the LoRA adapters into the base model."""
        if FastLanguageModel is None:
            raise ImportError("Unsloth is not installed.")

        self.presenter.log(f"Loading model for fusion: {config.data.adapter_path}", "info")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.data.adapter_path,  # Load adapters
            max_seq_length=config.model.max_seq_length,
            dtype=None,
            load_in_4bit=config.model.load_in_4bit,
        )

        # Merge to 16bit
        model.save_pretrained_merged(
            config.data.fused_model_path,
            tokenizer,
            save_method="merged_16bit",
        )
        return config.data.fused_model_path

    def export_to_gguf(self, config: ProjectConfig) -> str:
        """Export the trained model to GGUF format."""
        if FastLanguageModel is None:
            raise ImportError("Unsloth is not installed.")

        self.presenter.log(f"Exporting to GGUF: {config.data.gguf_output_path}", "info")

        # Need to load again or reuse model? Unsloth methods often handle this on the model object.
        # Assuming we can just load the adapter path again to be safe/clean
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.data.adapter_path,
            max_seq_length=config.model.max_seq_length,
            dtype=None,
            load_in_4bit=config.model.load_in_4bit,
        )

        model.save_pretrained_gguf(
            config.data.fused_model_path,
            tokenizer,
            quantization_method=config.lora.quantization_method,
        )
        # Note: save_pretrained_gguf usually saves to the directory. Filename handling might differ.
        return config.data.gguf_output_path

    def _load_dataset(self, path: str):
        """Load a dataset from a file path compatible with HuggingFace."""
        # Implementation to load dataset compatible with HuggingFace/Unsloth
        # This is a bit tricky since we defined JSONLDataRepository.
        # Ideally we should use the repository, but SFTTrainer expects a HF Dataset or similar.
        # For now, simplistic implementation using `datasets` library if available
        from datasets import load_dataset  # type: ignore # pylint: disable=import-error

        return load_dataset("json", data_files=path, split="train")
