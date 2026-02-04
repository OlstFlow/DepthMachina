from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from . import nm_paths


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def addon_dev_dir() -> Path:
    return nm_paths.addon_root_dir() / "dev"


def addon_errolog_path() -> Path:
    return addon_dev_dir() / "Errolog.txt"


def addon_terminal_path() -> Path:
    return addon_dev_dir() / "Terminal.txt"


def _append_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line)


def append_errolog(message: str, *, project_root: Path | None = None) -> Path:
    line = f"{_now_iso()} {message}\n"
    target = addon_errolog_path()
    _append_lines(target, [line])
    return target


def append_terminal(text: str, *, project_root: Path | None = None) -> Path:
    if not text.endswith("\n"):
        text += "\n"
    header = f"{_now_iso()} ---\n"
    target = addon_terminal_path()
    _append_lines(target, [header, text])
    return target
