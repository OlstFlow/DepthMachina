from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import IO
from typing import Sequence

from . import nm_paths
from .nm_blender import blender_python_executable


@dataclass(frozen=True)
class BackendResult:
    returncode: int
    stdout: str
    stderr: str


def _backend_cmd(args: Sequence[str]) -> list[str]:
    python_exe = blender_python_executable()
    cli_path = nm_paths.backend_cli_path()

    cmd = [python_exe, str(cli_path)]
    cmd.extend(args)
    return cmd


def _backend_env() -> dict[str, str]:
    env = dict(os.environ)
    addon_dir = str(nm_paths.addon_root_dir())
    deps_dir = str(nm_paths.deps_dir())
    existing = env.get("PYTHONPATH", "")
    parts = [addon_dir, deps_dir]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def run_backend(args: Sequence[str]) -> BackendResult:
    cmd = _backend_cmd(args)
    env = _backend_env()

    completed = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return BackendResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


@dataclass
class BackendProcess:
    popen: subprocess.Popen[str]
    stdout_path: Path
    stderr_path: Path
    stdout_file: IO[str]
    stderr_file: IO[str]
    cmd: list[str]

    def close_files(self) -> None:
        try:
            self.stdout_file.close()
        except Exception:
            pass
        try:
            self.stderr_file.close()
        except Exception:
            pass


def spawn_backend(
    args: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
) -> BackendProcess:
    cmd = _backend_cmd(args)
    env = _backend_env()

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_file = stdout_path.open("w", encoding="utf-8", errors="replace")
    stderr_file = stderr_path.open("w", encoding="utf-8", errors="replace")
    popen = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file, text=True, env=env)
    return BackendProcess(
        popen=popen,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        cmd=cmd,
    )
