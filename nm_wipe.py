from __future__ import annotations

import json
import os
import shutil
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import nm_logging
from . import nm_paths


_PENDING_FILENAME = ".dm_wipe_pending.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def pending_path() -> Path:
    return nm_paths.addon_root_dir() / _PENDING_FILENAME


def is_pending() -> bool:
    try:
        return pending_path().exists()
    except Exception:
        return False


@dataclass(frozen=True)
class WipeResult:
    removed: list[Path]
    failed: list[Path]


def _rmtree(path: Path) -> None:
    def _onerror(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            raise

    shutil.rmtree(path, onerror=_onerror)


def wipe_paths(paths: list[Path]) -> WipeResult:
    removed: list[Path] = []
    failed: list[Path] = []

    for p in paths:
        try:
            if not p.exists():
                continue
            if p.is_dir():
                _rmtree(p)
            else:
                p.unlink()
            removed.append(p)
        except Exception:
            failed.append(p)

    return WipeResult(removed=removed, failed=failed)


def write_pending(*, paths: list[Path], note: str = "") -> None:
    try:
        payload = {
            "created_at": _now_iso(),
            "paths": [str(p) for p in paths],
            "note": str(note or ""),
        }
        pending_path().write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def clear_pending() -> None:
    try:
        p = pending_path()
        if p.exists():
            p.unlink()
    except Exception:
        pass


def try_run_pending_wipe() -> bool:
    """
    Run a previously scheduled wipe early during add-on startup, before `_deps` is added to sys.path.
    Returns True if pending wipe is fully cleared.
    """
    p = pending_path()
    if not p.exists():
        return True
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        raw_paths = payload.get("paths", [])
        paths = [Path(x) for x in raw_paths if str(x).strip()]
    except Exception:
        paths = []

    if not paths:
        clear_pending()
        return True

    result = wipe_paths(paths)
    if not result.failed:
        clear_pending()
        nm_logging.append_terminal("Pending wipe completed on startup.", project_root=None)
        return True

    write_pending(paths=result.failed, note="Some paths could not be removed on startup (locked files?).")
    nm_logging.append_errolog("Pending wipe still has locked paths; retry after restarting Blender.", project_root=None)
    nm_logging.append_terminal(
        "Pending wipe still pending:\n" + "\n".join([f"- {str(x)}" for x in result.failed]),
        project_root=None,
    )
    return False

