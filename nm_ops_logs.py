from __future__ import annotations

import bpy

from . import nm_logging


def _open_in_text_editor(filepath: Path, *, name: str) -> None:
    if not filepath.exists():
        raise FileNotFoundError(str(filepath))
    content = filepath.read_text(encoding="utf-8", errors="replace")
    text = bpy.data.texts.get(name)
    if text is None:
        text = bpy.data.texts.new(name)
    text.clear()
    text.write(content)
    for area in bpy.context.window.screen.areas:
        if area.type == "TEXT_EDITOR":
            area.spaces.active.text = text
            break


class NM_OT_open_logs(bpy.types.Operator):
    bl_idname = "nm.open_logs"
    bl_label = "Open Logs Folder"
    bl_description = "Open the add-on dev logs folder in the OS file browser"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        logs_dir = nm_logging.addon_dev_dir()
        logs_dir.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.path_open(filepath=str(logs_dir))
        return {"FINISHED"}


class NM_OT_view_errolog(bpy.types.Operator):
    bl_idname = "nm.view_errolog"
    bl_label = "View Errolog"
    bl_description = "Load dev/Errolog.txt into Blender Text Editor"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        path = nm_logging.addon_errolog_path()
        _open_in_text_editor(path, name="DepthMachina_Errolog.txt")
        return {"FINISHED"}


class NM_OT_view_terminal(bpy.types.Operator):
    bl_idname = "nm.view_terminal"
    bl_label = "View Terminal"
    bl_description = "Load dev/Terminal.txt into Blender Text Editor"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        path = nm_logging.addon_terminal_path()
        _open_in_text_editor(path, name="DepthMachina_Terminal.txt")
        return {"FINISHED"}


def register():
    bpy.utils.register_class(NM_OT_open_logs)
    bpy.utils.register_class(NM_OT_view_errolog)
    bpy.utils.register_class(NM_OT_view_terminal)


def unregister():
    bpy.utils.unregister_class(NM_OT_view_terminal)
    bpy.utils.unregister_class(NM_OT_view_errolog)
    bpy.utils.unregister_class(NM_OT_open_logs)
