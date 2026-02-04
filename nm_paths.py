from __future__ import annotations

from pathlib import Path


def addon_root_dir() -> Path:
    return Path(__file__).resolve().parent


def deps_dir() -> Path:
    return addon_root_dir() / "_deps"


def pip_cache_dir() -> Path:
    return addon_root_dir() / "_pip_cache"


def backend_cli_path() -> Path:
    return addon_root_dir() / "backend" / "cli.py"


def addon_cache_dir() -> Path:
    return addon_root_dir() / "_cache"


def project_data_dir(project_root: Path) -> Path:
    return project_root / "Data"


def project_images_dir(project_root: Path) -> Path:
    return project_data_dir(project_root) / "Images"


def project_sfm_dir(project_root: Path) -> Path:
    return project_data_dir(project_root) / "SfM"


def project_cache_dir(project_root: Path) -> Path:
    return project_data_dir(project_root) / "Cache"


def project_logs_dir(project_root: Path) -> Path:
    return project_data_dir(project_root) / "Logs"


def project_manifest_path(project_root: Path) -> Path:
    return project_data_dir(project_root) / "project.json"
