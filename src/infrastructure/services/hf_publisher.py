import os

from huggingface_hub import HfApi

from ...domain.entities.configuration import ProjectConfig
from ...domain.interfaces.publisher import IModelPublisher
from ...infrastructure.ui.terminal_presenter import TerminalPresenter


class HuggingFacePublisher(IModelPublisher):
    """Implementação do publicador usando Hugging Face Hub."""

    def __init__(self, presenter: TerminalPresenter):
        self.presenter = presenter
        self.api = HfApi()

    def publish(self, config: ProjectConfig) -> None:
        """Publica artefatos no Hugging Face Hub."""
        repo_id = config.model.hf_repo_id

        if not repo_id:
            self.presenter.log("HF_REPO_ID não definido. Pulei etapa de upload.", "warning")
            return

        self.presenter.log(f"Iniciando publicação para: {repo_id}", "info")

        # Verifica/Cria repo
        try:
            self.api.create_repo(repo_id=repo_id, exist_ok=True)
        except Exception as e:
            self.presenter.log(f"Erro ao acessar repo: {e}", "error")
            return

        # 1. Upload Adapters (Sempre existe se treinou)
        if os.path.exists(config.data.adapter_path):
            self.presenter.log(f"Enviando adaptadores: {config.data.adapter_path}", "info")
            try:
                self.api.upload_folder(
                    repo_id=repo_id,
                    folder_path=config.data.adapter_path,
                    path_in_repo="adapters",
                    commit_message="Upload LoRA adapters",
                )
            except Exception as e:
                self.presenter.log(f"Erro no upload de adapters: {e}", "error")

        # 2. Upload Fused Model (Opcional, se foi gerado)
        if os.path.exists(config.data.fused_model_path):
            self.presenter.log(f"Enviando modelo fundido: {config.data.fused_model_path}", "info")
            try:
                self.api.upload_folder(
                    repo_id=repo_id,
                    folder_path=config.data.fused_model_path,
                    path_in_repo="fused_model",
                    commit_message="Upload fused model",
                )
            except Exception as e:
                self.presenter.log(f"Erro no upload do modelo fundido: {e}", "error")

        # 3. Upload GGUF (Opcional, se foi gerado)
        if os.path.exists(config.data.gguf_output_path):
            self.presenter.log(f"Enviando GGUF: {config.data.gguf_output_path}", "info")
            try:
                self.api.upload_file(
                    repo_id=repo_id,
                    path_or_fileobj=config.data.gguf_output_path,
                    path_in_repo=os.path.basename(config.data.gguf_output_path),
                    commit_message="Upload GGUF model",
                )
            except Exception as e:
                self.presenter.log(f"Erro no upload do GGUF: {e}", "error")

        self.presenter.log("Processo de publicação finalizado!", "success")
