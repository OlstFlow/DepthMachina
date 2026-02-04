from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import bpy

from . import nm_paths


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _marker_path() -> Path:
    return nm_paths.deps_dir() / ".dm_da3_deps_installed.json"


def requirements_hash(requirements_path: Path) -> str:
    if not requirements_path.exists():
        return ""
    return hashlib.sha256(requirements_path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class DepsStatus:
    installed: bool
    marker_path: Path
    marker: dict | None


def get_deps_status() -> DepsStatus:
    marker_path = _marker_path()
    requirements_path = nm_paths.addon_root_dir() / "requirements_da3.txt"
    current_sha = requirements_hash(requirements_path)
    if not marker_path.exists():
        return DepsStatus(installed=False, marker_path=marker_path, marker={"requirements_sha256": current_sha})
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        marker = {}
    marker_sha = str(marker.get("requirements_sha256", ""))
    installed = bool(marker_sha) and (marker_sha == current_sha)
    marker["requirements_sha256_current"] = current_sha
    return DepsStatus(installed=installed, marker_path=marker_path, marker=marker)


def mark_installed(*, requirements_sha256: str) -> Path:
    marker_path = _marker_path()
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "installed_at": _now_iso(),
        "requirements_sha256": requirements_sha256,
        "blender_version": list(bpy.app.version),
    }
    marker_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return marker_path
