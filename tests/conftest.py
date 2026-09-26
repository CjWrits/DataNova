"""Pytest shared test fixtures and configuration."""

import os
import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the absolute path of the project root directory."""
    return Path(__file__).resolve().parent.parent

@pytest.fixture(scope="session")
def configs_dir(project_root: Path) -> Path:
    """Return the absolute path of the configs directory."""
    return project_root / "configs"

@pytest.fixture(scope="session")
def artifacts_dir(project_root: Path) -> Path:
    """Return the absolute path of the artifacts directory."""
    return project_root / "artifacts"
