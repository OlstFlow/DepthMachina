from __future__ import annotations

import bpy

from . import nm_deps
from . import nm_deps_da3
from . import nm_deps_moge2


class NM_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__ or "DepthMachina"

    last_images_folder: bpy.props.StringProperty(
        name="Last Images Folder",
        description="Last used images folder directory",
        default="",
        subtype="DIR_PATH",
    )
    hover_preview_enabled: bpy.props.BoolProperty(
        name="Hover Preview Enabled",
        description="Show an image preview overlay near mouse when hovering DepthMachina image list area",
        default=True,
    )
    hover_preview_size: bpy.props.IntProperty(
        name="Hover Preview Size",
        description="Preview overlay size (px)",
        default=280,
        min=128,
        max=768,
    )
    hover_preview_delay_ms: bpy.props.IntProperty(
        name="Hover Preview Delay (ms)",
        description="Delay before the hover preview appears",
        default=300,
        min=0,
        max=2000,
    )
    hover_preview_popup_ms: bpy.props.IntProperty(
        name="Preview Popup Duration (ms)",
        description="How long the preview stays visible after clicking an image",
        default=1200,
        min=200,
        max=10000,
    )
    hover_preview_bg_alpha: bpy.props.FloatProperty(
        name="Hover Preview Background Alpha",
        description="Background opacity for hover preview",
        default=0.85,
        min=0.0,
        max=1.0,
    )

    enable_moge2: bpy.props.BoolProperty(
        name="Enable MoGe-2 Module",
        description="Show MoGe-2 tools in the DepthMachina UI",
        default=True,
    )
    enable_da3: bpy.props.BoolProperty(
        name="Enable Depth Anything 3 Module",
        description="Show Depth Anything 3 tools in the DepthMachina UI",
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "last_images_folder")

        layout.separator()
        layout.label(text="UI")
        col = layout.column(align=True)
        col.prop(self, "hover_preview_enabled")
        col.prop(self, "hover_preview_size")
        col.prop(self, "hover_preview_delay_ms")
        col.prop(self, "hover_preview_popup_ms")
        col.prop(self, "hover_preview_bg_alpha")

        layout.separator()
        layout.label(text="Dependencies (status & maintenance)")

        label_split = 0.45

        deps = nm_deps.get_deps_status()
        row = layout.row(align=True)
        split = row.split(factor=label_split, align=True)
        col_l = split.column(align=True)
        col_r = split.column(align=True)
        col_l.label(text=f"Core: {'INSTALLED' if deps.installed else 'MISSING/OUTDATED'}")
        col_r.operator(
            "nm.install_dependencies",
            text=("Reinstall Core" if deps.installed else "Install Core"),
            icon="IMPORT",
        )

        moge2 = nm_deps_moge2.get_deps_status()
        row = layout.row(align=True)
        split = row.split(factor=label_split, align=True)
        col_l = split.column(align=True)
        col_r = split.column(align=True)
        col_l.label(text=f"MoGe-2: {'INSTALLED' if moge2.installed else 'MISSING/OUTDATED'}")
        btns = col_r.row(align=True)
        split3 = btns.split(factor=1.0 / 3.0, align=True)
        b1 = split3.column(align=True)
        b23 = split3.column(align=True)
        split2 = b23.split(factor=0.5, align=True)
        b2 = split2.column(align=True)
        b3 = split2.column(align=True)
        b1.operator(
            "nm.toggle_moge2_enabled",
            text=("Disable" if self.enable_moge2 else "Enable"),
            depress=bool(self.enable_moge2),
        )
        b2.operator(
            "nm.install_moge2_dependencies",
            text=("Reinstall" if moge2.installed else "Install"),
        )
        b3.operator("nm.remove_moge2_dependencies", text="Remove")

        da3 = nm_deps_da3.get_deps_status()
        row = layout.row(align=True)
        split = row.split(factor=label_split, align=True)
        col_l = split.column(align=True)
        col_r = split.column(align=True)
        col_l.label(text=f"Depth Anything 3: {'INSTALLED' if da3.installed else 'MISSING/OUTDATED'}")
        btns = col_r.row(align=True)
        split3 = btns.split(factor=1.0 / 3.0, align=True)
        b1 = split3.column(align=True)
        b23 = split3.column(align=True)
        split2 = b23.split(factor=0.5, align=True)
        b2 = split2.column(align=True)
        b3 = split2.column(align=True)
        b1.operator(
            "nm.toggle_da3_enabled",
            text=("Disable" if self.enable_da3 else "Enable"),
            depress=bool(self.enable_da3),
        )
        b2.operator(
            "nm.install_da3_dependencies",
            text=("Reinstall" if da3.installed else "Install"),
        )
        b3.operator("nm.remove_da3_dependencies", text="Remove")

        layout.separator()
        row = layout.row()
        row.alert = True
        row.operator("nm.wipe_all_deps", icon="TRASH")


def register():
    bpy.utils.register_class(NM_AddonPreferences)


def unregister():
    bpy.utils.unregister_class(NM_AddonPreferences)
