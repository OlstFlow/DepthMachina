from __future__ import annotations

from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view


def _load_image(path: Path) -> bpy.types.Image | None:
    if not path.exists():
        return None
    try:
        img = bpy.data.images.load(str(path), check_existing=True)
    except Exception:
        return None
    try:
        if hasattr(img, "colorspace_settings") and hasattr(img.colorspace_settings, "name"):
            img.colorspace_settings.name = "sRGB"
    except Exception:
        pass
    return img


def ensure_projected_uv_from_camera(
    *,
    obj: bpy.types.Object,
    cam_obj: bpy.types.Object,
    scene: bpy.types.Scene,
    uv_name: str = "DM_UV",
) -> str | None:
    if obj.type != "MESH" or obj.data is None:
        return None
    mesh = obj.data
    try:
        if not getattr(mesh, "polygons", None) or len(mesh.polygons) == 0:
            return None
    except Exception:
        return None

    try:
        uv_layers = mesh.uv_layers
    except Exception:
        return None

    try:
        uv = uv_layers.get(uv_name)
    except Exception:
        uv = None
    if uv is None:
        try:
            uv = uv_layers.new(name=uv_name)
        except Exception:
            return None
    try:
        uv_layers.active = uv
        uv_layers.active_index = list(uv_layers).index(uv)
    except Exception:
        pass

    try:
        mw = obj.matrix_world
    except Exception:
        return uv_name

    try:
        loops = mesh.loops
        verts = mesh.vertices
        uv_data = uv.data
    except Exception:
        return uv_name

    for poly in mesh.polygons:
        for li in poly.loop_indices:
            try:
                vi = loops[li].vertex_index
                co_world = mw @ verts[vi].co
                ndc = world_to_camera_view(scene, cam_obj, co_world)
                uv_data[li].uv = (float(ndc.x), float(ndc.y))
            except Exception:
                continue

    try:
        mesh.update()
    except Exception:
        pass
    return uv_name


def ensure_image_material(
    *,
    name: str,
    image: bpy.types.Image,
    uv_name: str = "DM_UV",
) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)

    mat.use_nodes = True
    nt = mat.node_tree
    assert nt is not None

    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (140, 0)
    try:
        bsdf.inputs["Roughness"].default_value = 1.0
    except Exception:
        pass

    uv = nodes.new("ShaderNodeUVMap")
    uv.location = (-360, 0)
    try:
        uv.uv_map = uv_name
    except Exception:
        pass

    tex = nodes.new("ShaderNodeTexImage")
    tex.location = (-120, 0)
    tex.image = image

    links.new(uv.outputs.get("UV"), tex.inputs.get("Vector"))
    links.new(tex.outputs.get("Color"), bsdf.inputs.get("Base Color"))
    links.new(bsdf.outputs.get("BSDF"), out.inputs.get("Surface"))

    try:
        mat.blend_method = "OPAQUE"
    except Exception:
        pass
    return mat


def assign_material(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    if obj.type != "MESH" or obj.data is None:
        return
    try:
        mats = obj.data.materials
        if mats is None:
            return
        if len(mats) == 0:
            mats.append(mat)
        else:
            mats[0] = mat
    except Exception:
        pass


def apply_projected_image_material(
    *,
    obj: bpy.types.Object,
    cam_obj: bpy.types.Object,
    image_path: Path,
    material_name: str,
    uv_name: str = "DM_UV",
) -> bool:
    scene = bpy.context.scene
    uv = ensure_projected_uv_from_camera(obj=obj, cam_obj=cam_obj, scene=scene, uv_name=uv_name)
    if not uv:
        return False

    img = _load_image(image_path)
    if img is None:
        return False

    mat = ensure_image_material(name=material_name, image=img, uv_name=uv)
    assign_material(obj, mat)
    return True

