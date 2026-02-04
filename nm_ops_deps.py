from __future__ import annotations

import subprocess
import traceback

import bpy

from . import nm_deps
from . import nm_logging
from . import nm_paths
from .nm_blender import blender_python_executable


def _run_python(args: list[str]) -> subprocess.CompletedProcess[str]:
    python_exe = blender_python_executable()
    return subprocess.run([python_exe, *args], capture_output=True, text=True)


class NM_OT_install_dependencies(bpy.types.Operator):
    bl_idname = "nm.install_dependencies"
    bl_label = "Install Dependencies"
    bl_description = "Install Python dependencies next to the add-on into _deps"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            addon_dir = nm_paths.addon_root_dir()
            deps_dir = nm_paths.deps_dir()
            cache_dir = nm_paths.pip_cache_dir()
            requirements = addon_dir / "requirements.txt"

            deps_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)

            if not requirements.exists():
                requirements.write_text("# Add pinned dependencies here\n", encoding="utf-8")

            steps = [
                ["-m", "ensurepip", "--upgrade"],
                ["-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "pip"],
                [
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--upgrade",
                    "-r",
                    str(requirements),
                    "--target",
                    str(deps_dir),
                    "--cache-dir",
                    str(cache_dir),
                ],
            ]

            combined_out: list[str] = []
            for step in steps:
                proc = _run_python(step)
                combined_out.append(f"$ python {' '.join(step)}\n")
                if proc.stdout:
                    combined_out.append(proc.stdout)
                if proc.stderr:
                    combined_out.append(proc.stderr)
                combined_out.append("\n")
                if proc.returncode != 0:
                    terminal_path = nm_logging.append_terminal("".join(combined_out), project_root=None)
                    errolog_path = nm_logging.append_errolog(
                        f"Dependency install failed (exit={proc.returncode}). Terminal log: {terminal_path}",
                        project_root=None,
                    )
                    self.report({"ERROR"}, f"Dependency install failed. See {errolog_path.name}.")
                    print("DepthMachina dependency install failed. See:", errolog_path)
                    return {"CANCELLED"}

            marker = nm_deps.mark_installed(requirements_sha256=nm_deps.requirements_hash(requirements))
            nm_logging.append_terminal("".join(combined_out), project_root=None)
            self.report({"INFO"}, f"Dependencies installed. Marker: {marker.name}")
            return {"FINISHED"}
        except Exception:
            errolog_path = nm_logging.append_errolog("Unhandled exception during dependency install.", project_root=None)
            nm_logging.append_terminal(traceback.format_exc(), project_root=None)
            self.report({"ERROR"}, f"Install failed. See {errolog_path.name}.")
            raise


def register():
    bpy.utils.register_class(NM_OT_install_dependencies)


def unregister():
    bpy.utils.unregister_class(NM_OT_install_dependencies)
