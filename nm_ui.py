from __future__ import annotations

import bpy

from . import nm_addon
from . import nm_deps
from . import nm_deps_da3
from . import nm_deps_moge2


class NM_UL_images(bpy.types.UIList):
    bl_idname = "NM_UL_images"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        op = row.operator("nm.select_image", text=(item.filename or ""), icon="IMAGE_DATA", emboss=False)
        op.index = int(index)


class NM_PT_main(bpy.types.Panel):
    bl_idname = "NM_PT_main"
    bl_label = "DepthMachina"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DepthMachina"

    def draw(self, context):
        layout = self.layout
        deps = nm_deps.get_deps_status()
        moge2_deps = nm_deps_moge2.get_deps_status()
        da3_deps = nm_deps_da3.get_deps_status()
        prefs = nm_addon.get_addon_prefs(context)

        if not deps.installed:
            layout.operator("nm.install_dependencies", icon="IMPORT")
            return

        box = layout.box()
        box.label(text="Images")
        box.prop(context.scene, "nm_images_folder", text="Folder")
        row = box.row(align=True)
        row.operator("nm.set_images_folder", icon="FILE_FOLDER", text="Browse")
        row.operator("nm.refresh_images", icon="FILE_REFRESH", text="Refresh")

        list_row = box.row()
        list_row.template_list(
            "NM_UL_images",
            "",
            context.scene,
            "nm_images",
            context.scene,
            "nm_images_index",
            rows=6,
        )
        col = list_row.column(align=True)
        col.operator("nm.add_images", text="", icon="ADD")
        col.operator("nm.remove_image", text="", icon="REMOVE")

        # Hover preview is drawn as an overlay (see nm_hover_preview.py)

        layout.separator()
        if prefs is None or bool(getattr(prefs, "enable_moge2", True)):
            box = layout.box()
            row = box.row(align=True)
            show_moge2 = bool(getattr(context.scene, "nm_show_optional_moge2", True))
            row.prop(
                context.scene,
                "nm_show_optional_moge2",
                text="MoGe-2",
                icon="TRIA_DOWN" if show_moge2 else "TRIA_RIGHT",
                emboss=False,
            )
            if show_moge2:
                col = box.column(align=True)
                if not moge2_deps.installed:
                    col.operator("nm.install_moge2_dependencies", icon="IMPORT")
                else:
                    mode = str(getattr(context.scene, "nm_moge2_export_mode", "none") or "none")
                    row = col.row(align=True)
                    op = row.operator(
                        "nm.set_moge2_mode",
                        text="Mesh",
                        depress=(mode == "mesh"),
                        icon="MESH_ICOSPHERE",
                    )
                    op.mode = "mesh"
                    op = row.operator(
                        "nm.set_moge2_mode",
                        text="Points",
                        depress=(mode == "points"),
                        icon="PARTICLES",
                    )
                    op.mode = "points"

                    col.separator(factor=1.8)

                    gen = col.column(align=True)
                    gen.enabled = (mode in {"mesh", "points"}) and (not bool(getattr(context.scene, "nm_job_running", False)))
                    gen_row = gen.row(align=True)
                    gen_row.scale_y = 1.7
                    gen_row.scale_x = 1.3
                    gen_row.alignment = "CENTER"
                    op = gen_row.operator("nm.moge2_seed", text="Generate ▶")
                    op.export_mode = mode if mode in {"mesh", "points"} else "mesh"

                    col.separator(factor=1.2)

                    row = col.row(align=True)
                    show_settings = bool(getattr(context.scene, "nm_moge2_show_settings", False))
                    row.prop(
                        context.scene,
                        "nm_moge2_show_settings",
                        text="Settings",
                        icon="TRIA_DOWN" if show_settings else "TRIA_RIGHT",
                        emboss=False,
                    )
                    if show_settings:
                        settings = col.box()
                        if hasattr(context.scene, "nm_moge2_model_version"):
                            settings.prop(context.scene, "nm_moge2_model_version", text="Model")
                        if hasattr(context.scene, "nm_moge2_device"):
                            settings.prop(context.scene, "nm_moge2_device", text="Device")
                        if mode == "mesh":
                            if hasattr(context.scene, "nm_moge2_mesh_stride"):
                                settings.prop(context.scene, "nm_moge2_mesh_stride", text="Mesh Stride")
                        elif mode == "points":
                            if hasattr(context.scene, "nm_moge2_points_stride"):
                                settings.prop(context.scene, "nm_moge2_points_stride", text="Points Stride")
                            if hasattr(context.scene, "nm_moge2_points_max_points"):
                                settings.prop(context.scene, "nm_moge2_points_max_points", text="Max Points")

        layout.separator()
        if prefs is None or bool(getattr(prefs, "enable_da3", True)):
            box = layout.box()
            row = box.row(align=True)
            show_da3 = bool(getattr(context.scene, "nm_show_optional_da3", False))
            row.prop(
                context.scene,
                "nm_show_optional_da3",
                text="Depth Anything 3",
                icon="TRIA_DOWN" if show_da3 else "TRIA_RIGHT",
                emboss=False,
            )
            if show_da3:
                col = box.column(align=True)
                if not da3_deps.installed:
                    col.operator("nm.install_da3_dependencies", icon="IMPORT")
                else:
                    mode = str(getattr(context.scene, "nm_da3_export_mode", "none") or "none")
                    row = col.row(align=True)
                    op = row.operator(
                        "nm.set_da3_mode",
                        text="Mesh",
                        depress=(mode == "mesh"),
                        icon="MESH_ICOSPHERE",
                    )
                    op.mode = "mesh"
                    op = row.operator(
                        "nm.set_da3_mode",
                        text="Points",
                        depress=(mode == "points"),
                        icon="PARTICLES",
                    )
                    op.mode = "points"

                    col.separator(factor=1.8)

                    gen = col.column(align=True)
                    gen.enabled = (mode in {"mesh", "points"}) and (not bool(getattr(context.scene, "nm_job_running", False)))
                    gen_row = gen.row(align=True)
                    gen_row.scale_y = 1.7
                    gen_row.scale_x = 1.3
                    gen_row.alignment = "CENTER"
                    op = gen_row.operator("nm.da3_seed", text="Generate ▶")
                    op.export_mode = mode if mode in {"mesh", "points"} else "mesh"

                    col.separator(factor=1.2)

                    row = col.row(align=True)
                    show_settings = bool(getattr(context.scene, "nm_da3_show_settings", False))
                    row.prop(
                        context.scene,
                        "nm_da3_show_settings",
                        text="Settings",
                        icon="TRIA_DOWN" if show_settings else "TRIA_RIGHT",
                        emboss=False,
                    )
                    if show_settings:
                        settings = col.box()
                        if hasattr(context.scene, "nm_da3_model_id"):
                            settings.prop(context.scene, "nm_da3_model_id", text="Model")
                        if hasattr(context.scene, "nm_da3_device"):
                            settings.prop(context.scene, "nm_da3_device", text="Device")
                        if hasattr(context.scene, "nm_da3_max_edge"):
                            settings.prop(context.scene, "nm_da3_max_edge", text="Max Edge")
                        if mode == "mesh":
                            if hasattr(context.scene, "nm_da3_mesh_stride"):
                                settings.prop(context.scene, "nm_da3_mesh_stride", text="Mesh Stride")
                        elif mode == "points":
                            if hasattr(context.scene, "nm_da3_points_stride"):
                                settings.prop(context.scene, "nm_da3_points_stride", text="Points Stride")
                            if hasattr(context.scene, "nm_da3_points_max_points"):
                                settings.prop(context.scene, "nm_da3_points_max_points", text="Max Points")

                        advanced = col.box()
                        advanced.label(text="Camera/Scale")
                        if hasattr(context.scene, "nm_da3_fov_x_deg"):
                            advanced.prop(context.scene, "nm_da3_fov_x_deg", text="FOV X")
                        if hasattr(context.scene, "nm_da3_target_median_depth"):
                            advanced.prop(context.scene, "nm_da3_target_median_depth", text="Median Depth")

                        # Setup-from-artifacts intentionally removed in the simplified, no-project workflow.


def register():
    bpy.utils.register_class(NM_UL_images)
    bpy.utils.register_class(NM_PT_main)


def unregister():
    bpy.utils.unregister_class(NM_PT_main)
    bpy.utils.unregister_class(NM_UL_images)
