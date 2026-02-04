from __future__ import annotations

import shutil
from pathlib import Path

import bpy

from . import nm_addon
from . import nm_deps_da3
from . import nm_deps_moge2
from . import nm_logging
from . import nm_paths
from . import nm_wipe


def _addon_prefs() -> bpy.types.AddonPreferences | None:
    return nm_addon.get_addon_prefs()


def _try_remove_tree(path: Path) -> bool:
    try:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        return True
    except Exception:
        return False


def _remove_dist_infos(deps_dir: Path, prefix: str) -> int:
    removed = 0
    for p in deps_dir.glob(f"{prefix}-*.dist-info"):
        if _try_remove_tree(p):
            removed += 1
    for p in deps_dir.glob(f"{prefix}-*.egg-info"):
        if _try_remove_tree(p):
            removed += 1
    return removed


class NM_OT_set_da3_mode(bpy.types.Operator):
    bl_idname = "nm.set_da3_mode"
    bl_label = "Set DA3 Mode"
    bl_options = {"INTERNAL"}

    mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ("mesh", "Mesh", ""),
            ("points", "Points", ""),
        ],
        default="mesh",
    )

    def execute(self, context):
        current = str(getattr(context.scene, "nm_da3_export_mode", "none") or "none")
        if current == self.mode:
            context.scene.nm_da3_export_mode = "none"
        else:
            context.scene.nm_da3_export_mode = str(self.mode)
        return {"FINISHED"}


class NM_OT_set_moge2_mode(bpy.types.Operator):
    bl_idname = "nm.set_moge2_mode"
    bl_label = "Set MoGe-2 Mode"
    bl_options = {"INTERNAL"}

    mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ("mesh", "Mesh", ""),
            ("points", "Points", ""),
        ],
        default="mesh",
    )

    def execute(self, context):
        current = str(getattr(context.scene, "nm_moge2_export_mode", "none") or "none")
        if current == self.mode:
            context.scene.nm_moge2_export_mode = "none"
        else:
            context.scene.nm_moge2_export_mode = str(self.mode)
        return {"FINISHED"}


class NM_OT_toggle_moge2_enabled(bpy.types.Operator):
    bl_idname = "nm.toggle_moge2_enabled"
    bl_label = "Toggle MoGe-2 Module"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        prefs = _addon_prefs()
        if prefs is None:
            return {"CANCELLED"}
        prefs.enable_moge2 = not bool(getattr(prefs, "enable_moge2", True))
        # Also hide the section in all scenes to avoid confusion.
        for scene in bpy.data.scenes:
            if hasattr(scene, "nm_show_optional_moge2"):
                scene.nm_show_optional_moge2 = False
            if hasattr(scene, "nm_moge2_export_mode"):
                scene.nm_moge2_export_mode = "none"
        return {"FINISHED"}


class NM_OT_toggle_da3_enabled(bpy.types.Operator):
    bl_idname = "nm.toggle_da3_enabled"
    bl_label = "Toggle Depth Anything 3 Module"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        prefs = _addon_prefs()
        if prefs is None:
            return {"CANCELLED"}
        prefs.enable_da3 = not bool(getattr(prefs, "enable_da3", True))
        for scene in bpy.data.scenes:
            if hasattr(scene, "nm_show_optional_da3"):
                scene.nm_show_optional_da3 = False
            if hasattr(scene, "nm_da3_export_mode"):
                scene.nm_da3_export_mode = "none"
        return {"FINISHED"}


class NM_OT_remove_moge2_dependencies(bpy.types.Operator):
    bl_idname = "nm.remove_moge2_dependencies"
    bl_label = "Remove MoGe-2"
    bl_description = "Disable MoGe-2 and remove its install marker (best-effort cleanup of addon deps)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        prefs = _addon_prefs()
        if prefs is not None:
            prefs.enable_moge2 = False

        deps_dir = nm_paths.deps_dir()
        marker = nm_deps_moge2.get_deps_status().marker_path
        ok_marker = _try_remove_tree(marker)

        # MoGe-2 shares most deps with other modules; we only remove the marker by default.
        msg = "MoGe-2 disabled. Marker removed." if ok_marker else "MoGe-2 disabled. Marker remove failed (restart Blender and retry)."
        nm_logging.append_terminal(msg, project_root=None)
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class NM_OT_remove_da3_dependencies(bpy.types.Operator):
    bl_idname = "nm.remove_da3_dependencies"
    bl_label = "Remove Depth Anything 3"
    bl_description = "Disable DA3 and remove its install marker and core DA3 package files (best-effort)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        prefs = _addon_prefs()
        if prefs is not None:
            prefs.enable_da3 = False

        deps_dir = nm_paths.deps_dir()
        marker = nm_deps_da3.get_deps_status().marker_path
        ok_marker = _try_remove_tree(marker)

        removed = 0
        removed += 1 if _try_remove_tree(deps_dir / "depth_anything_3") else 0
        removed += _remove_dist_infos(deps_dir, "depth_anything_3")
        removed += 1 if _try_remove_tree(deps_dir / "addict") else 0
        removed += _remove_dist_infos(deps_dir, "addict")

        msg = f"DA3 disabled. Removed files={removed}. Marker={'OK' if ok_marker else 'FAIL'}."
        if not ok_marker:
            msg += " (Restart Blender and retry if files are locked.)"
        nm_logging.append_terminal(msg, project_root=None)
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class NM_OT_wipe_all_deps(bpy.types.Operator):
    bl_idname = "nm.wipe_all_deps"
    bl_label = "Wipe All Deps"
    bl_description = "Remove all add-on dependencies and caches (_deps/_pip_cache/_cache). Resets the add-on to a clean state."
    bl_options = {"REGISTER"}

    def invoke(self, context, event):
        try:
            return context.window_manager.invoke_confirm(self, event)
        except Exception:
            return self.execute(context)

    def execute(self, context):
        prefs = _addon_prefs()
        if prefs is not None:
            prefs.enable_moge2 = False
            prefs.enable_da3 = False

        # Reset per-scene UI state.
        for scene in bpy.data.scenes:
            if hasattr(scene, "nm_show_optional_moge2"):
                scene.nm_show_optional_moge2 = False
            if hasattr(scene, "nm_show_optional_da3"):
                scene.nm_show_optional_da3 = False
            if hasattr(scene, "nm_moge2_export_mode"):
                scene.nm_moge2_export_mode = "none"
            if hasattr(scene, "nm_da3_export_mode"):
                scene.nm_da3_export_mode = "none"

        paths = [
            nm_paths.deps_dir(),
            nm_paths.pip_cache_dir(),
            nm_paths.addon_cache_dir(),
            nm_paths.addon_root_dir() / "__pycache__",
        ]

        # Try immediate wipe. On Windows, loaded DLLs (torch/opencv) may prevent deletion while Blender is running.
        result = nm_wipe.wipe_paths(paths)

        msg = "Wipe complete."
        if result.failed:
            nm_wipe.write_pending(paths=paths, note="Scheduled by UI wipe. Restart Blender to finish deleting locked files.")
            msg = (
                "Wipe partially completed. Some files are locked (common on Windows with torch/opencv). "
                "Restart Blender, then the add-on will finish wiping automatically on startup."
            )

        nm_logging.append_terminal(
            msg
            + "\n"
            + "\n".join([f"- OK: {str(p)}" for p in result.removed] + [f"- FAIL: {str(p)}" for p in result.failed]),
            project_root=None,
        )

        if result.failed:
            nm_logging.append_errolog("Wipe All Deps: locked paths remain; restart Blender to finish.", project_root=None)
            self.report({"WARNING"}, msg)
        else:
            nm_wipe.clear_pending()
            self.report({"INFO"}, msg)
        return {"FINISHED"}


def register():
    bpy.utils.register_class(NM_OT_set_da3_mode)
    bpy.utils.register_class(NM_OT_set_moge2_mode)
    bpy.utils.register_class(NM_OT_toggle_moge2_enabled)
    bpy.utils.register_class(NM_OT_toggle_da3_enabled)
    bpy.utils.register_class(NM_OT_remove_moge2_dependencies)
    bpy.utils.register_class(NM_OT_remove_da3_dependencies)
    bpy.utils.register_class(NM_OT_wipe_all_deps)


def unregister():
    bpy.utils.unregister_class(NM_OT_wipe_all_deps)
    bpy.utils.unregister_class(NM_OT_remove_da3_dependencies)
    bpy.utils.unregister_class(NM_OT_remove_moge2_dependencies)
    bpy.utils.unregister_class(NM_OT_toggle_da3_enabled)
    bpy.utils.unregister_class(NM_OT_toggle_moge2_enabled)
    bpy.utils.unregister_class(NM_OT_set_moge2_mode)
    bpy.utils.unregister_class(NM_OT_set_da3_mode)
