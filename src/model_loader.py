"""
Módulo para carregamento e gerenciamento do modelo e adapters.
"""
from typing import Tuple
from unsloth import FastLanguageModel
from config.settings import ModelConfig


class ModelManager:
    """
    Gerenciador de ciclo de vida do modelo (carregamento, adaptação
    e salvamento).
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_base_model(self) -> Tuple[object, object]:
        """
        Carrega o modelo base e o tokenizer.

        Returns:
            Tuple[object, object]: Modelo e Tokenizer carregados.
        """
        print(f"🔄 Carregando modelo base: {self.config.model_name}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=self.config.dtype,
            load_in_4bit=self.config.load_in_4bit,
        )
        return self.model, self.tokenizer

    def apply_lora_adapters(self):
        """
        Aplica a camada de treinamento PEFT/LoRA.

        Returns:
            object: Modelo com adaptadores aplicados.
        """
        print("🔧 Aplicando adaptadores LoRA...")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        return self.model

    def save_to_gguf(self, output_path: str, quantization: str = "q4_k_m"):
        """
        Salva o modelo no formato GGUF.

        Args:
            output_path (str): Caminho para salvar o modelo.
            quantization (str, optional): Método de quantização.
            Default: "q4_k_m".
        """
        print(f"💾 Salvando GGUF em: {output_path}...")
        self.model.save_pretrained_gguf(
            output_path, self.tokenizer, quantization_method=quantization)
