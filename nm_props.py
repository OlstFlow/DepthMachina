from __future__ import annotations

import bpy


class NM_ImageItem(bpy.types.PropertyGroup):
    image_id: bpy.props.StringProperty(name="ID", default="")
    filename: bpy.props.StringProperty(name="Filename", default="")
    filepath: bpy.props.StringProperty(name="Filepath", default="", subtype="FILE_PATH")
    source: bpy.props.EnumProperty(
        name="Source",
        items=[
            ("folder", "Folder", "From the folder scan"),
            ("manual", "Manual", "Added manually"),
        ],
        default="manual",
    )
    status: bpy.props.StringProperty(name="Status", default="")


class NM_ExcludedImageItem(bpy.types.PropertyGroup):
    filepath: bpy.props.StringProperty(name="Filepath", default="", subtype="FILE_PATH")


def _auto_refresh_on_images_folder_update(self, context) -> None:
    try:
        from . import nm_ops_images

        nm_ops_images.refresh_images_in_scene(context.scene)
    except Exception as e:
        try:
            from . import nm_logging

            nm_logging.append_errolog(f"Auto refresh (Images Folder update) failed: {e}", project_root=None)
        except Exception:
            pass


def register():
    bpy.utils.register_class(NM_ImageItem)
    bpy.utils.register_class(NM_ExcludedImageItem)

    bpy.types.Scene.nm_images_folder = bpy.props.StringProperty(
        name="Images Folder",
        description="Folder to list images from (images are not copied)",
        default="",
        subtype="DIR_PATH",
        update=_auto_refresh_on_images_folder_update,
    )

    bpy.types.Scene.nm_images = bpy.props.CollectionProperty(type=NM_ImageItem)
    bpy.types.Scene.nm_images_index = bpy.props.IntProperty(name="Image Index", default=0)
    bpy.types.Scene.nm_images_excluded = bpy.props.CollectionProperty(type=NM_ExcludedImageItem)

    bpy.types.Scene.nm_job_running = bpy.props.BoolProperty(
        name="Job Running",
        description="True while a background job is running",
        default=False,
        options={"HIDDEN"},
    )
    bpy.types.Scene.nm_job_label = bpy.props.StringProperty(
        name="Job Label",
        description="Current background job label",
        default="",
        options={"HIDDEN"},
    )

    bpy.types.Scene.nm_show_optional_moge2 = bpy.props.BoolProperty(
        name="Show MoGe-2",
        description="Show/hide MoGe-2 UI section",
        default=False,
    )
    bpy.types.Scene.nm_show_optional_da3 = bpy.props.BoolProperty(
        name="Show Depth Anything 3",
        description="Show/hide Depth Anything 3 UI section",
        default=False,
    )

    bpy.types.Scene.nm_da3_export_mode = bpy.props.EnumProperty(
        name="DA3 Mode",
        description="Select what to generate",
        items=[
            ("none", "None", "Select Mesh or Points"),
            ("mesh", "Mesh", "Generate mesh"),
            ("points", "Points", "Generate point cloud"),
        ],
        default="none",
    )
    bpy.types.Scene.nm_da3_show_settings = bpy.props.BoolProperty(
        name="DA3 Settings",
        description="Show/hide DA3 settings",
        default=False,
    )

    bpy.types.Scene.nm_da3_model_id = bpy.props.EnumProperty(
        name="DA3 Model",
        description="Depth Anything 3 model id (weights download on first use)",
        items=[
            ("depth-anything/DA3-SMALL", "DA3-SMALL (Apache-2.0)", "Commercial-friendly (Apache-2.0)"),
            ("depth-anything/DA3-BASE", "DA3-BASE (Apache-2.0)", "Commercial-friendly (Apache-2.0)"),
            ("depth-anything/DA3MONO-LARGE", "DA3MONO-LARGE (Apache-2.0)", "Large monocular depth (Apache-2.0)"),
            ("depth-anything/DA3METRIC-LARGE", "DA3METRIC-LARGE (Apache-2.0)", "Large metric depth (Apache-2.0)"),
        ],
        default="depth-anything/DA3-SMALL",
    )
    bpy.types.Scene.nm_da3_device = bpy.props.EnumProperty(
        name="DA3 Device",
        description="Device for Depth Anything 3",
        items=[
            ("cuda", "CUDA", "Use CUDA if available"),
            ("cpu", "CPU", "Force CPU"),
        ],
        default="cuda",
    )

    bpy.types.Scene.nm_moge2_show_settings = bpy.props.BoolProperty(
        name="MoGe-2 Settings",
        description="Show/hide MoGe-2 advanced settings",
        default=False,
    )
    bpy.types.Scene.nm_moge2_export_mode = bpy.props.EnumProperty(
        name="MoGe-2 Mode",
        description="Select what to generate",
        items=[
            ("none", "None", "Select Mesh or Points"),
            ("mesh", "Mesh", "Generate mesh"),
            ("points", "Points", "Generate point cloud"),
        ],
        default="none",
    )
    bpy.types.Scene.nm_moge2_model_version = bpy.props.EnumProperty(
        name="MoGe-2 Model",
        items=[
            ("v2", "v2 (normals)", "Ruicheng/moge-2-vitl-normal"),
            ("v1", "v1", "Ruicheng/moge-vitl"),
        ],
        default="v2",
    )
    bpy.types.Scene.nm_moge2_device = bpy.props.EnumProperty(
        name="MoGe-2 Device",
        description="Device for MoGe-2",
        items=[
            ("cuda", "CUDA", "Use CUDA if available"),
            ("cpu", "CPU", "Force CPU"),
        ],
        default="cuda",
    )
    bpy.types.Scene.nm_moge2_mesh_stride = bpy.props.IntProperty(
        name="MoGe-2 Mesh Stride",
        description="Sampling stride for mesh generation (1 = densest)",
        default=2,
        min=1,
        max=32,
    )
    bpy.types.Scene.nm_moge2_points_stride = bpy.props.IntProperty(
        name="MoGe-2 Points Stride",
        description="Sampling stride for point cloud generation (1 = densest)",
        default=4,
        min=1,
        max=32,
    )
    bpy.types.Scene.nm_moge2_points_max_points = bpy.props.IntProperty(
        name="MoGe-2 Max Points",
        description="Maximum points to keep in point cloud (randomly subsampled)",
        default=250000,
        min=1000,
        max=5000000,
    )

    bpy.types.Scene.nm_da3_max_edge = bpy.props.IntProperty(
        name="DA3 Max Edge",
        description="Max image edge used for DA3 inference (higher = more detail, more VRAM/time)",
        default=1024,
        min=256,
        max=4096,
    )
    bpy.types.Scene.nm_da3_fov_x_deg = bpy.props.FloatProperty(
        name="DA3 FOV X",
        description="Assumed horizontal FOV for DA3 seed camera (deg)",
        default=60.0,
        min=1.0,
        max=179.0,
    )
    bpy.types.Scene.nm_da3_target_median_depth = bpy.props.FloatProperty(
        name="DA3 Target Median Depth",
        description="Rescale depth so median valid depth matches this value",
        default=2.0,
        min=0.01,
        max=1000.0,
    )
    bpy.types.Scene.nm_da3_mesh_stride = bpy.props.IntProperty(
        name="DA3 Mesh Stride",
        description="Sampling stride for mesh generation (1 = densest)",
        default=2,
        min=1,
        max=32,
    )
    bpy.types.Scene.nm_da3_points_stride = bpy.props.IntProperty(
        name="DA3 Points Stride",
        description="Sampling stride for point cloud generation (1 = densest)",
        default=4,
        min=1,
        max=32,
    )
    bpy.types.Scene.nm_da3_points_max_points = bpy.props.IntProperty(
        name="DA3 Max Points",
        description="Maximum points to keep in point cloud (randomly subsampled)",
        default=250000,
        min=1000,
        max=5000000,
    )


def unregister():
    if hasattr(bpy.types.Scene, "nm_images_excluded"):
        del bpy.types.Scene.nm_images_excluded
    if hasattr(bpy.types.Scene, "nm_moge2_points_max_points"):
        del bpy.types.Scene.nm_moge2_points_max_points
    if hasattr(bpy.types.Scene, "nm_moge2_points_stride"):
        del bpy.types.Scene.nm_moge2_points_stride
    if hasattr(bpy.types.Scene, "nm_moge2_mesh_stride"):
        del bpy.types.Scene.nm_moge2_mesh_stride
    if hasattr(bpy.types.Scene, "nm_moge2_device"):
        del bpy.types.Scene.nm_moge2_device
    if hasattr(bpy.types.Scene, "nm_moge2_model_version"):
        del bpy.types.Scene.nm_moge2_model_version
    if hasattr(bpy.types.Scene, "nm_moge2_export_mode"):
        del bpy.types.Scene.nm_moge2_export_mode
    if hasattr(bpy.types.Scene, "nm_moge2_show_settings"):
        del bpy.types.Scene.nm_moge2_show_settings
    if hasattr(bpy.types.Scene, "nm_da3_points_max_points"):
        del bpy.types.Scene.nm_da3_points_max_points
    if hasattr(bpy.types.Scene, "nm_da3_points_stride"):
        del bpy.types.Scene.nm_da3_points_stride
    if hasattr(bpy.types.Scene, "nm_da3_mesh_stride"):
        del bpy.types.Scene.nm_da3_mesh_stride
    if hasattr(bpy.types.Scene, "nm_da3_target_median_depth"):
        del bpy.types.Scene.nm_da3_target_median_depth
    if hasattr(bpy.types.Scene, "nm_da3_fov_x_deg"):
        del bpy.types.Scene.nm_da3_fov_x_deg
    if hasattr(bpy.types.Scene, "nm_da3_max_edge"):
        del bpy.types.Scene.nm_da3_max_edge
    if hasattr(bpy.types.Scene, "nm_da3_device"):
        del bpy.types.Scene.nm_da3_device
    if hasattr(bpy.types.Scene, "nm_da3_model_id"):
        del bpy.types.Scene.nm_da3_model_id
    if hasattr(bpy.types.Scene, "nm_da3_show_settings"):
        del bpy.types.Scene.nm_da3_show_settings
    if hasattr(bpy.types.Scene, "nm_da3_export_mode"):
        del bpy.types.Scene.nm_da3_export_mode
    if hasattr(bpy.types.Scene, "nm_show_optional_da3"):
        del bpy.types.Scene.nm_show_optional_da3
    if hasattr(bpy.types.Scene, "nm_show_optional_moge2"):
        del bpy.types.Scene.nm_show_optional_moge2
    if hasattr(bpy.types.Scene, "nm_images_index"):
        del bpy.types.Scene.nm_images_index
    if hasattr(bpy.types.Scene, "nm_images"):
        del bpy.types.Scene.nm_images
    if hasattr(bpy.types.Scene, "nm_job_label"):
        del bpy.types.Scene.nm_job_label
    if hasattr(bpy.types.Scene, "nm_job_running"):
        del bpy.types.Scene.nm_job_running
    if hasattr(bpy.types.Scene, "nm_images_folder"):
        del bpy.types.Scene.nm_images_folder

    bpy.utils.unregister_class(NM_ImageItem)
    bpy.utils.unregister_class(NM_ExcludedImageItem)
