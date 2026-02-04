from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import bpy


@lru_cache(maxsize=1)
def blender_python_executable() -> str:
    """
    Return a path to Blender's embedded Python executable.

    Blender builds differ: some expose bpy.app.binary_path_python, others don't.
    """
    direct = getattr(bpy.app, "binary_path_python", None)
    if direct:
        return str(direct)

    exe = Path(sys.executable)
    if exe.name.lower().startswith("python"):
        return str(exe)

    blender_path = getattr(bpy.app, "binary_path", "") or ""
    blender_exe = Path(blender_path) if blender_path else exe
    root = blender_exe.resolve().parent

    major, minor = int(bpy.app.version[0]), int(bpy.app.version[1])
    ver_dir = f"{major}.{minor}"
    candidates = [
        root / ver_dir / "python" / "bin" / "python.exe",
        root / ver_dir / "python" / "bin" / "python3.exe",
        root / ver_dir / "python" / "bin" / "python3.11.exe",
        root / "python" / "bin" / "python.exe",
        root / "python" / "bin" / "python3.exe",
        root / "python" / "bin" / "python3.11.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise RuntimeError(
        "Could not locate Blender's embedded Python executable. "
        "bpy.app.binary_path_python is missing and no python.exe was found near Blender."
    )

