from typing import Any, Dict, List

from ...domain.entities.configuration import ProjectConfig
from ...domain.interfaces.trainer import ITrainer
from ..ui.terminal_presenter import TerminalPresenter


class MLXTrainerAdapter(ITrainer):
    """Adaptador para MLX LLM (Apple Silicon)."""

    def __init__(self, presenter: TerminalPresenter):
        """Initialize the MLXTrainerAdapter with a presenter."""
        self.presenter = presenter

    def _build_train_cmd(self, config: ProjectConfig) -> List[str]:
        """Constrói comando CLI para mlx_lm.lora."""
        c = config
        cmd = [
            "python",
            "-m",
            "mlx_lm.lora",
            "--model",
            c.model.name,
            "--train",
            "--data",
            c.data.processed_dir,  # Assume dados já preparados aqui
            "--batch-size",
            str(c.training.batch_size),
            # MLX usa num-layers para rank ou similar dependendo da versão, ajustar conf
            "--num-layers",
            str(c.lora.r),
            "--iters",
            str(c.training.max_steps),
            "--learning-rate",
            str(c.training.learning_rate),
            "--adapter-path",
            c.data.adapter_path,
            "--save-every",
            str(c.training.save_every_steps),
            "--steps-per-eval",
            str(c.training.steps_per_eval),
            "--val-batches",
            str(c.training.val_batches),
            "--seed",
            str(c.training.seed),
        ]

        if c.training.mlx_grad_checkpoint:
            cmd.append("--grad-checkpoint")

        return cmd

    def train(self, config: ProjectConfig) -> Dict[str, Any]:
        """Executa treinamento."""
        self.presenter.log("Iniciando treinamento MLX...", "info")

        # Monitoramento de espaço antes
        # (Implementar verificação se necessário)

        cmd = self._build_train_cmd(config)
        self.presenter.run_command_with_spinner(cmd, "Treinando modelo MLX...")

        self.presenter.log("Treinamento finalizado!", "success")
        return {"status": "success", "steps": config.training.max_steps}

    def fuse_adapters(self, config: ProjectConfig) -> str:
        """Funde adaptadores."""
        self.presenter.log("Fundindo adaptadores...", "info")

        cmd = [
            "python",
            "-m",
            "mlx_lm.fuse",
            "--model",
            config.model.name,
            "--adapter-path",
            config.data.adapter_path,
            "--save-path",
            config.data.fused_model_path,
        ]

        self.presenter.run_command_with_spinner(cmd, "Fundindo modelo...")
        self.presenter.log(f"Modelo fundido salvo em: {config.data.fused_model_path}", "success")
        return config.data.fused_model_path

    def export_to_gguf(self, config: ProjectConfig) -> str:
        """Exporta para GGUF (requer llama.cpp)."""
        # Nota: Exportação GGUF no MLX nativo ou via llama.cpp
        # Por simplificação, assumiremos que o usuário tem llama.cpp
        # setup ou usaremos script mlx se houver
        # Atualmente MLX não exporta GGUF diretamente, usa scripts conversão.
        # Manterei um placeholder funcional que avisa sobre llama.cpp

        self.presenter.log("Exportação GGUF requer llama.cpp externo por enquanto.", "warning")
        return ""
