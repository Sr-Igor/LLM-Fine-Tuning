"""
Container module for Dependency Injection.

This module provides a Container class that handles the instantiation
and lifetime management of application dependencies such as settings,
presenters, repositories, and trainers.
"""

from ...application.prepare_data import PrepareDataUseCase
from ...application.publish_model import PublishModelUseCase
from ...application.train_model import TrainModelUseCase
from ...config.settings import Settings
from ...infrastructure.repositories.jsonl_repository import JSONLDataRepository
from ...infrastructure.services.hf_publisher import HuggingFacePublisher
from ...infrastructure.trainers.mlx_trainer import MLXTrainerAdapter
from ...infrastructure.trainers.unsloth_trainer import UnslothTrainerAdapter
from ...infrastructure.ui.terminal_presenter import TerminalPresenter


class Container:
    """Dependency Injection Container Simples."""

    def __init__(self):
        """Initialize the Container with default values for dependencies."""
        self._config = None
        self._presenter = None
        self._repository = None
        self._trainer = None
        self._publisher = None

    @property
    def config(self):
        """Provide access to the application configuration settings."""
        if not self._config:
            self._config = Settings.load()
        return self._config

    @property
    def presenter(self):
        """Provide access to the terminal presenter instance."""
        if not self._presenter:
            self._presenter = TerminalPresenter()
        return self._presenter

    @property
    def repository(self):
        """Provide access to the data repository instance."""
        if not self._repository:
            self._repository = JSONLDataRepository()
        return self._repository

    @property
    def trainer(self):
        """Provide access to the appropriate trainer adapter."""
        if not self._trainer:
            backend_type = self.config.backend.type
            if backend_type == "unsloth":
                self._trainer = UnslothTrainerAdapter(self.presenter)
            else:
                self._trainer = MLXTrainerAdapter(self.presenter)
        return self._trainer

    @property
    def publisher(self):
        """Provide access to the model publisher instance."""
        if not self._publisher:
            self._publisher = HuggingFacePublisher(self.presenter)
        return self._publisher

    # Use Cases Factories

    def get_prepare_data_use_case(self) -> PrepareDataUseCase:
        """Create and return an instance of the PrepareDataUseCase."""
        return PrepareDataUseCase(self.repository)

    def get_train_model_use_case(self) -> TrainModelUseCase:
        """Create and return an instance of the TrainModelUseCase."""
        return TrainModelUseCase(self.trainer)

    def get_publish_model_use_case(self) -> PublishModelUseCase:
        """Create and return an instance of the PublishModelUseCase."""
        return PublishModelUseCase(self.publisher)
