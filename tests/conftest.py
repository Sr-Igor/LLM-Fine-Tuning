"""
Configuration file for Pytest.

This module sets up the test environment, including path adjustments
and common fixtures used across multiple test files.
"""
import os
import sys
import pytest

# Add the project root and src directory to sys.path to allow imports
# from 'config' (at root) and 'core' (under src)
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))


@pytest.fixture
def mock_env_setup(monkeypatch):
    """
    Fixture to setup common environment variables for testing.
    This helps ensure tests run with a consistent environment configuration
    and don't accidetally rely on the real .env file or system environment.
    """
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("PROJECT_NAME", "llm-test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
