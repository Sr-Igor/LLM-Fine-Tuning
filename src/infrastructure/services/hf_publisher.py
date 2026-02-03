import os

from huggingface_hub import HfApi

from ...domain.entities.configuration import ProjectConfig
from ...domain.interfaces.publisher import IModelPublisher
from ...infrastructure.ui.terminal_presenter import TerminalPresenter


class HuggingFacePublisher(IModelPublisher):
    """Publisher implementation using Hugging Face Hub."""

    def __init__(self, presenter: TerminalPresenter):
        self.presenter = presenter
        self.api = HfApi()

    def publish(self, config: ProjectConfig) -> None:
        """Publish artifacts to Hugging Face Hub."""
        repo_id = config.model.hf_repo_id

        if not repo_id:
            self.presenter.log("HF_REPO_ID not defined. Skipping upload step.", "warning")
            return

        self.presenter.log(f"Starting publication for: {repo_id}", "info")

        # Check/Create repo
        try:
            self.api.create_repo(repo_id=repo_id, exist_ok=True)
        except Exception as e:
            self.presenter.log(f"Error accessing repo: {e}", "error")
            return

        # 1. Upload Adapters (Always exists if trained)
        if os.path.exists(config.data.adapter_path):
            self.presenter.log(f"Uploading adapters: {config.data.adapter_path}", "info")
            try:
                self.api.upload_folder(
                    repo_id=repo_id,
                    folder_path=config.data.adapter_path,
                    path_in_repo="adapters",
                    commit_message="Upload LoRA adapters",
                )
            except Exception as e:
                self.presenter.log(f"Error uploading adapters: {e}", "error")

        # 2. Upload Fused Model (Optional, if generated)
        if os.path.exists(config.data.fused_model_path):
            self.presenter.log(f"Uploading fused model: {config.data.fused_model_path}", "info")
            try:
                self.api.upload_folder(
                    repo_id=repo_id,
                    folder_path=config.data.fused_model_path,
                    path_in_repo="fused_model",
                    commit_message="Upload fused model",
                )
            except Exception as e:
                self.presenter.log(f"Error uploading fused model: {e}", "error")

        # 3. Upload GGUF (Optional, if generated)
        if os.path.exists(config.data.gguf_output_path):
            self.presenter.log(f"Uploading GGUF: {config.data.gguf_output_path}", "info")
            try:
                self.api.upload_file(
                    repo_id=repo_id,
                    path_or_fileobj=config.data.gguf_output_path,
                    path_in_repo=os.path.basename(config.data.gguf_output_path),
                    commit_message="Upload GGUF model",
                )
            except Exception as e:
                self.presenter.log(f"Error uploading GGUF: {e}", "error")

        self.presenter.log("Publication process finished!", "success")
