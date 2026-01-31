"""Tests for configuration settings."""
import os
from unittest.mock import patch
from config.settings import ModelConfig, TrainingConfig, ProjectConfig


def test_model_config_from_env():
    """Test loading ModelConfig from environment variables."""
    with patch.dict(os.environ, {"MODEL_NAME": "test-model"}):
        config = ModelConfig.from_env()
        assert config.model_name == "test-model"


def test_training_config_from_env():
    """Test loading TrainingConfig from environment variables."""
    with patch.dict(os.environ, {"TRAINING_BATCH_SIZE": "8"}):
        config = TrainingConfig.from_env()
        assert config.batch_size == 8


def test_project_config_defaults():
    """Test ProjectConfig default values."""
    config = ProjectConfig.from_env()
    assert config.model.max_seq_length == 2048
