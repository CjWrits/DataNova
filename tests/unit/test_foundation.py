"""Foundation tests validating directory structure and central configuration."""

from pathlib import Path
import yaml


def test_project_root_structure(project_root: Path):
    """Verify that all core monorepo directories exist."""
    required_dirs = [
        "configs",
        "data",
        "data/raw",
        "data/processed",
        "data/external",
        "research",
        "research/notebooks",
        "research/experiments",
        "research/visualizations",
        "ml",
        "ml/preprocessing",
        "ml/features",
        "ml/training",
        "ml/evaluation",
        "ml/explainability",
        "ml/drift",
        "ml/registry",
        "artifacts",
        "artifacts/models",
        "artifacts/metrics",
        "artifacts/reports",
        "backend",
        "backend/app",
        "backend/app/api",
        "backend/app/core",
        "backend/app/models",
        "backend/app/schemas",
        "backend/app/services",
        "frontend",
        "scripts",
        "tests",
        "docs",
    ]
    for d in required_dirs:
        dir_path = project_root / d
        assert dir_path.is_dir(), f"Required directory missing: {d}"


def test_system_config_validity(configs_dir: Path):
    """Verify system_config.yaml loads cleanly and parses valid configuration keys."""
    config_file = configs_dir / "system_config.yaml"
    assert config_file.is_file(), "system_config.yaml does not exist."
    
    with open(config_file, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
        
    assert isinstance(config_data, dict)
    assert "project" in config_data
    assert "paths" in config_data
    assert "data" in config_data
    assert "preprocessing" in config_data
    assert "imbalance_strategy" in config_data
    assert "model" in config_data
    assert "explainability" in config_data
    assert "drift" in config_data
    assert config_data["project"]["random_seed"] == 42


def test_essential_documentation_files(project_root: Path):
    """Verify that all required Phase 0 documentation files exist."""
    docs_dir = project_root / "docs"
    assert (docs_dir / "architecture.md").is_file()
    assert (docs_dir / "roadmap.md").is_file()
    assert (docs_dir / "methodology.md").is_file()
    assert (docs_dir / "setup.md").is_file()
    assert (project_root / "README.md").is_file()
    assert (project_root / ".env.example").is_file()
    assert (project_root / ".gitignore").is_file()
