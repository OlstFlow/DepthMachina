from __future__ import annotations

import json
import math
import os
import subprocess
import traceback
import uuid
from pathlib import Path
from time import monotonic

import bpy
from mathutils import Matrix
from mathutils import Vector

from . import nm_backend
from . import nm_deps
from . import nm_deps_moge2
from . import nm_logging
from . import nm_materials
from . import nm_paths
from . import nm_statusbar
from .nm_blender import blender_python_executable

_ADDON_PREFIX = "DM"
_LEGACY_PREFIXES = ("DepthMachina",)
_MOGE2_COLLECTION_NAME = f"{_ADDON_PREFIX}_MoGe-2_gen"
_MOGE2_ENGINE_TAG = "moge2"
_WORLD_UP_FIX_EULER = (math.pi / 2.0, 0.0, 0.0)


def _legacy_names(new_name: str) -> list[str]:
    names: list[str] = []
    for prefix in _LEGACY_PREFIXES:
        if new_name.startswith(f"{_ADDON_PREFIX}_"):
            names.append(new_name.replace(f"{_ADDON_PREFIX}_", f"{prefix}_", 1))
    return names


def _get_or_migrate_collection(new_name: str) -> bpy.types.Collection | None:
    coll = bpy.data.collections.get(new_name)
    if coll is not None:
        return coll
    for legacy_name in _legacy_names(new_name):
        coll = bpy.data.collections.get(legacy_name)
        if coll is not None:
            try:
                coll.name = new_name
            except Exception:
                pass
            return coll
    return None


def _get_or_migrate_object(new_name: str) -> bpy.types.Object | None:
    obj = bpy.data.objects.get(new_name)
    if obj is not None:
        return obj
    for legacy_name in _legacy_names(new_name):
        obj = bpy.data.objects.get(legacy_name)
        if obj is not None:
            try:
                obj.name = new_name
            except Exception:
                pass
            return obj
    return None


def _get_or_migrate_camera_data(new_name: str) -> bpy.types.Camera | None:
    cam = bpy.data.cameras.get(new_name)
    if cam is not None:
        return cam
    for legacy_name in _legacy_names(new_name):
        cam = bpy.data.cameras.get(legacy_name)
        if cam is not None:
            try:
                cam.name = new_name
            except Exception:
                pass
            return cam
    return None


def _selected_image_id(context) -> str | None:
    idx = int(getattr(context.scene, "nm_images_index", -1))
    images = getattr(context.scene, "nm_images", None)
    if images is None:
        return None
    if idx < 0 or idx >= len(images):
        return None
    return str(images[idx].image_id or "").strip() or None


def _selected_image_path(context) -> Path | None:
    idx = int(getattr(context.scene, "nm_images_index", -1))
    images = getattr(context.scene, "nm_images", None)
    if images is None:
        return None
    if idx < 0 or idx >= len(images):
        return None
    raw = str(getattr(images[idx], "filepath", "") or "").strip()
    if not raw:
        return None
    try:
        return Path(bpy.path.abspath(raw)).resolve()
    except Exception:
        try:
            return Path(raw).resolve()
        except Exception:
            return Path(raw)


def _run_python(args: list[str]) -> subprocess.CompletedProcess[str]:
    python_exe = blender_python_executable()
    return subprocess.run([python_exe, *args], capture_output=True, text=True)


def _import_ply(filepath: str) -> None:
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=filepath)
        return
    if hasattr(bpy.ops.import_mesh, "ply"):
        bpy.ops.import_mesh.ply(filepath=filepath)
        return
    raise RuntimeError("No PLY import operator found (expected bpy.ops.wm.ply_import).")


def _ensure_generation_collection(name: str) -> bpy.types.Collection:
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    scene_coll = bpy.context.scene.collection
    try:
        if name not in scene_coll.children.keys():
            scene_coll.children.link(coll)
    except Exception:
        try:
            scene_coll.children.link(coll)
        except Exception:
            pass
    return coll


def _move_object_to_collection(obj: bpy.types.Object, coll: bpy.types.Collection) -> None:
    try:
        for c in list(obj.users_collection):
            if c != coll:
                c.objects.unlink(obj)
    except Exception:
        pass
    try:
        if obj.name not in coll.objects.keys():
            coll.objects.link(obj)
    except Exception:
        try:
            coll.objects.link(obj)
        except Exception:
            pass


def _safe_name(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return "Image"
    out = []
    for ch in s:
        if ch.isalnum() or ch in (" ", "_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    s2 = "".join(out).strip()
    return s2 or "Image"


def _create_root_empty(*, coll: bpy.types.Collection, base_name: str) -> bpy.types.Object:
    obj = bpy.data.objects.new(base_name, None)
    obj.empty_display_type = "PLAIN_AXES"
    obj.empty_display_size = 0.25
    coll.objects.link(obj)
    obj.location = (0.0, 0.0, 0.0)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    return obj


def _bounds_center_world(objects: list[bpy.types.Object]) -> Vector:
    inf = 1.0e30
    min_v = Vector((inf, inf, inf))
    max_v = Vector((-inf, -inf, -inf))
    any_bb = False
    for obj in objects:
        bb = getattr(obj, "bound_box", None)
        if not bb:
            continue
        try:
            m = obj.matrix_world
            for corner in bb:
                v = m @ Vector(corner)
                min_v.x = min(min_v.x, v.x)
                min_v.y = min(min_v.y, v.y)
                min_v.z = min(min_v.z, v.z)
                max_v.x = max(max_v.x, v.x)
                max_v.y = max(max_v.y, v.y)
                max_v.z = max(max_v.z, v.z)
                any_bb = True
        except Exception:
            continue
    if not any_bb:
        return Vector((0.0, 0.0, 0.0))
    return (min_v + max_v) * 0.5


def _abs_path_str(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _find_group_root(*, collection: bpy.types.Collection, image_path_abs: str) -> bpy.types.Object | None:
    for obj in collection.objects:
        if obj.type != "EMPTY":
            continue
        try:
            if str(obj.get("dm_engine", "")) != _MOGE2_ENGINE_TAG:
                continue
            if str(obj.get("dm_image_path", "")) == image_path_abs:
                return obj
        except Exception:
            continue
    return None


def _ensure_group_root(*, collection: bpy.types.Collection, image_path_abs: str, base_name: str) -> tuple[bpy.types.Object, bool]:
    existing = _find_group_root(collection=collection, image_path_abs=image_path_abs)
    if existing is not None:
        return existing, False
    root = _create_root_empty(coll=collection, base_name=base_name)
    try:
        root["dm_engine"] = _MOGE2_ENGINE_TAG
        root["dm_image_path"] = image_path_abs
        root["dm_kind"] = "group"
    except Exception:
        pass
    return root, True


def _find_group_camera(*, collection: bpy.types.Collection, image_path_abs: str) -> bpy.types.Object | None:
    for obj in collection.objects:
        if obj.type != "CAMERA":
            continue
        try:
            if str(obj.get("dm_engine", "")) != _MOGE2_ENGINE_TAG:
                continue
            if str(obj.get("dm_image_path", "")) == image_path_abs and str(obj.get("dm_kind", "")) == "camera":
                return obj
        except Exception:
            continue
    return None


def _ensure_group_camera(*, collection: bpy.types.Collection, image_path_abs: str, name: str) -> tuple[bpy.types.Object, bool]:
    cam_obj = _find_group_camera(collection=collection, image_path_abs=image_path_abs)
    created = False
    if cam_obj is None:
        cam_data = bpy.data.cameras.new(f"{name}_Data")
        cam_obj = bpy.data.objects.new(name, cam_data)
        collection.objects.link(cam_obj)
        created = True
        try:
            cam_obj["dm_engine"] = _MOGE2_ENGINE_TAG
            cam_obj["dm_image_path"] = image_path_abs
            cam_obj["dm_kind"] = "camera"
        except Exception:
            pass
    if cam_obj.data is None or cam_obj.type != "CAMERA":
        try:
            cam_obj.data = bpy.data.cameras.new(f"{name}_Data")
        except Exception:
            pass
    return cam_obj, created


def _get_group_align_delta(root: bpy.types.Object) -> Vector:
    try:
        raw = root.get("dm_align_delta", None)
        # Blender stores list custom props as an IDProperty array (not a plain list/tuple).
        if raw is not None and len(raw) == 3:
            return Vector((float(raw[0]), float(raw[1]), float(raw[2])))
    except Exception:
        pass
    return Vector((0.0, 0.0, 0.0))


def _set_group_align_delta(root: bpy.types.Object, delta: Vector) -> None:
    try:
        root["dm_align_delta"] = [float(delta.x), float(delta.y), float(delta.z)]
    except Exception:
        pass


def _get_group_align_rot_euler(root: bpy.types.Object) -> tuple[float, float, float] | None:
    try:
        raw = root.get("dm_align_rot_euler", None)
        if raw is not None and len(raw) == 3:
            return (float(raw[0]), float(raw[1]), float(raw[2]))
    except Exception:
        pass
    return None


def _set_group_align_rot_euler(root: bpy.types.Object, euler_xyz: tuple[float, float, float]) -> None:
    try:
        root["dm_align_rot_euler"] = [float(euler_xyz[0]), float(euler_xyz[1]), float(euler_xyz[2])]
    except Exception:
        pass


def _apply_world_rot(obj: bpy.types.Object, euler_xyz: tuple[float, float, float]) -> None:
    try:
        rx, ry, rz = euler_xyz
        rot = Matrix.Rotation(float(rx), 4, "X") @ Matrix.Rotation(float(ry), 4, "Y") @ Matrix.Rotation(float(rz), 4, "Z")
        obj.matrix_world = rot @ obj.matrix_world
    except Exception:
        pass


def _translate_world(obj: bpy.types.Object, delta: Vector) -> None:
    try:
        m = obj.matrix_world.copy()
        m.translation = m.translation + delta
        obj.matrix_world = m
    except Exception:
        try:
            obj.location = (Vector(obj.location) + delta)
        except Exception:
            pass


def _parent_keep_world(child: bpy.types.Object, parent: bpy.types.Object) -> None:
    try:
        mw = child.matrix_world.copy()
        child.parent = parent
        child.matrix_parent_inverse = parent.matrix_world.inverted()
        child.matrix_world = mw
    except Exception:
        try:
            child.parent = parent
        except Exception:
            pass


def _camera_center_ray_world(camera_obj: bpy.types.Object) -> tuple[Vector, Vector]:
    origin = camera_obj.matrix_world.translation.copy()
    direction = (camera_obj.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    return origin, direction


def _raycast_object_world(*, obj: bpy.types.Object, origin_world: Vector, direction_world: Vector, max_dist: float = 1.0e6) -> Vector | None:
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        inv = eval_obj.matrix_world.inverted()
        origin_local = inv @ origin_world
        direction_local = (inv.to_3x3() @ direction_world).normalized()
        res = eval_obj.ray_cast(origin_local, direction_local, distance=max_dist)
        if isinstance(res, tuple) and len(res) >= 2:
            # Blender typically returns: (hit, location, normal, face_index)
            if len(res) >= 4:
                hit = bool(res[0])
                loc = res[1]
                if hit:
                    return eval_obj.matrix_world @ Vector(loc)
            else:
                loc = res[0]
                face_index = int(res[2]) if len(res) >= 3 else -1
                if face_index != -1:
                    return eval_obj.matrix_world @ Vector(loc)
    except Exception:
        return None
    return None


def _closest_vertex_to_camera_center_ray(*, obj: bpy.types.Object, origin_world: Vector, direction_world: Vector) -> Vector | None:
    if obj.type != "MESH" or obj.data is None:
        return None
    verts = getattr(obj.data, "vertices", None)
    if not verts:
        return None
    try:
        m = obj.matrix_world
    except Exception:
        return None

    best_point = None
    best_angle = 1.0e30
    best_t = 1.0e30

    try:
        n = len(verts)
    except Exception:
        n = 0
    step = 1
    if n > 500000:
        try:
            step = max(1, int(math.ceil(float(n) / 500000.0)))
        except Exception:
            step = 1

    # Minimize angle to the ray (perp/t), then choose nearest along the ray (smallest t).
    # This is more stable for sparse point clouds than pure distance-to-ray.
    for i in range(0, n, step):
        try:
            v = verts[i]
            p = m @ v.co
        except Exception:
            continue
        w = p - origin_world
        t = w.dot(direction_world)
        if t <= 0.0:
            continue
        perp = w - (direction_world * t)
        perp2 = perp.length_squared
        angle = perp2 / (t * t)
        if (angle < best_angle) or (abs(angle - best_angle) < 1e-18 and t < best_t):
            best_angle = angle
            best_t = t
            best_point = p.copy()

    return best_point


def _apply_camera_settings_from_payload(
    *,
    cam_obj: bpy.types.Object,
    payload: dict,
    fallback_image_path: Path | None = None,
    reset_transform: bool = False,
) -> None:
    cam_data = getattr(cam_obj, "data", None)
    if cam_data is None:
        return
    if reset_transform:
        cam_obj.location = (0.0, 0.0, 0.0)
        cam_obj.rotation_euler = (0.0, 0.0, 0.0)

    image_w = None
    image_h = None
    try:
        image_w = int(payload.get("image_width", 0) or 0) or None
        image_h = int(payload.get("image_height", 0) or 0) or None
    except Exception:
        image_w = None
        image_h = None

    fov_x_deg = payload.get("fov_x_deg", None)
    fov_y_deg = payload.get("fov_y_deg", None)
    if fov_x_deg is None or fov_y_deg is None:
        intr = payload.get("intrinsics", None)
        try:
            if intr and image_w and image_h:
                fx = float(intr[0][0])
                fy = float(intr[1][1])
                cx = float(intr[0][2])
                if cx <= 1.5:
                    fov_x_deg = math.degrees(2.0 * math.atan(0.5 / max(fx, 1e-9)))
                    fov_y_deg = math.degrees(2.0 * math.atan(0.5 / max(fy, 1e-9)))
                else:
                    fov_x_deg = math.degrees(2.0 * math.atan(float(image_w) / (2.0 * max(fx, 1e-9))))
                    fov_y_deg = math.degrees(2.0 * math.atan(float(image_h) / (2.0 * max(fy, 1e-9))))
        except Exception:
            fov_x_deg = None
            fov_y_deg = None

    try:
        fov_x_deg = float(fov_x_deg) if fov_x_deg is not None else 60.0
    except Exception:
        fov_x_deg = 60.0
    try:
        fov_y_deg = float(fov_y_deg) if fov_y_deg is not None else 45.0
    except Exception:
        fov_y_deg = 45.0

    cam_data.type = "PERSP"
    cam_data.sensor_width = 36.0
    if image_w and image_h:
        cam_data.sensor_height = cam_data.sensor_width * (float(image_h) / float(image_w))
    else:
        cam_data.sensor_height = 24.0

    if image_w and image_h and image_h > image_w:
        fov_y_rad = math.radians(max(1.0, min(179.0, fov_y_deg)))
        cam_data.sensor_fit = "VERTICAL"
        cam_data.lens = cam_data.sensor_height / (2.0 * math.tan(fov_y_rad / 2.0))
    else:
        fov_x_rad = math.radians(max(1.0, min(179.0, fov_x_deg)))
        cam_data.sensor_fit = "HORIZONTAL"
        cam_data.lens = cam_data.sensor_width / (2.0 * math.tan(fov_x_rad / 2.0))
    cam_data.clip_start = 0.001
    cam_data.clip_end = 100000.0

    try:
        if image_w and image_h:
            scene = bpy.context.scene
            scene.render.resolution_x = image_w
            scene.render.resolution_y = image_h
            scene.render.resolution_percentage = 100
            scene.render.pixel_aspect_x = 1.0
            scene.render.pixel_aspect_y = 1.0
    except Exception:
        pass

    try:
        image_path_raw = str(payload.get("image_used_path", "") or payload.get("image_path", "") or "").strip()
        image_path = None
        if image_path_raw:
            try:
                image_path = Path(image_path_raw).resolve()
            except Exception:
                image_path = Path(image_path_raw)
        if image_path is None and fallback_image_path is not None:
            image_path = fallback_image_path

        if image_path is not None and image_path.exists():
            img = bpy.data.images.load(str(image_path), check_existing=True)
            if (image_w is None) or (image_h is None):
                try:
                    image_w, image_h = int(img.size[0]), int(img.size[1])
                    scene = bpy.context.scene
                    scene.render.resolution_x = image_w
                    scene.render.resolution_y = image_h
                except Exception:
                    pass
            if hasattr(cam_data, "show_background_images"):
                cam_data.show_background_images = True
            if hasattr(cam_data, "background_images"):
                bg = cam_data.background_images.new()
                bg.image = img
                if hasattr(bg, "frame_method"):
                    bg.frame_method = "FIT"
                if hasattr(bg, "alpha"):
                    bg.alpha = 1.0
    except Exception:
            pass

    bpy.context.scene.camera = cam_obj
    return


def _activate_vertex_colors_for_object(obj: bpy.types.Object) -> None:
    if obj.type != "MESH" or obj.data is None:
        return
    mesh = obj.data
    color_attrs = getattr(mesh, "color_attributes", None)
    if not color_attrs:
        return
    try:
        if len(color_attrs) > 0:
            mesh.color_attributes.active = color_attrs[0]
            if hasattr(mesh.color_attributes, "active_color"):
                mesh.color_attributes.active_color = color_attrs[0]
    except Exception:
        pass


def _force_camera_view_and_background(camera_obj: bpy.types.Object) -> None:
    scene = bpy.context.scene
    scene.camera = camera_obj
    for area in bpy.context.screen.areas:
        if area.type != "VIEW_3D":
            continue
        space = area.spaces.active
        rv3d = getattr(space, "region_3d", None)
        if rv3d is not None:
            try:
                rv3d.view_perspective = "CAMERA"
            except Exception:
                pass
        try:
            if hasattr(space, "show_background_images"):
                space.show_background_images = True
        except Exception:
            pass
        try:
            if hasattr(space, "camera"):
                space.camera = camera_obj
        except Exception:
            pass


class NM_OT_install_moge2_dependencies(bpy.types.Operator):
    bl_idname = "nm.install_moge2_dependencies"
    bl_label = "Install MoGe-2 Dependencies"
    bl_description = "Install optional MoGe-2 Python dependencies next to the add-on into _deps"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            if not nm_deps.get_deps_status().installed:
                self.report({"ERROR"}, "Install core dependencies first.")
                return {"CANCELLED"}

            addon_dir = nm_paths.addon_root_dir()
            deps_dir = nm_paths.deps_dir()
            cache_dir = nm_paths.pip_cache_dir()
            requirements = addon_dir / "requirements_moge2.txt"

            deps_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)

            if not requirements.exists():
                self.report({"ERROR"}, "requirements_moge2.txt is missing.")
                return {"CANCELLED"}

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
                        f"MoGe-2 dependency install failed (exit={proc.returncode}). Terminal log: {terminal_path}",
                        project_root=None,
                    )
                    self.report({"ERROR"}, f"MoGe-2 install failed. See {errolog_path.name}.")
                    return {"CANCELLED"}

            marker = nm_deps_moge2.mark_installed(requirements_sha256=nm_deps_moge2.requirements_hash(requirements))
            nm_logging.append_terminal("".join(combined_out), project_root=None)
            self.report({"INFO"}, f"MoGe-2 dependencies installed. Marker: {marker.name}")
            return {"FINISHED"}
        except Exception:
            errolog_path = nm_logging.append_errolog("Unhandled exception during MoGe-2 dependency install.", project_root=None)
            nm_logging.append_terminal(traceback.format_exc(), project_root=None)
            self.report({"ERROR"}, f"MoGe-2 install failed. See {errolog_path.name}.")
            raise


class NM_OT_moge2_seed(bpy.types.Operator):
    bl_idname = "nm.moge2_seed"
    bl_label = "MoGe-2 Seed (Mesh/Point Cloud)"
    bl_description = "Run MoGe-2 on the selected image and import a seed artifact (mesh if available)"
    bl_options = {"REGISTER"}

    export_mode: bpy.props.EnumProperty(
        name="Export",
        items=[
            ("mesh", "Mesh", "Generate and import mesh"),
            ("points", "Point Cloud", "Generate and import point cloud"),
            ("both", "Both", "Generate both artifacts (imports one by default)"),
        ],
        default="mesh",
    )
    stride: bpy.props.IntProperty(name="Stride", default=4, min=1, max=32)
    max_points: bpy.props.IntProperty(name="Max Points", default=250000, min=1000, max=5000000)
    device: bpy.props.EnumProperty(
        name="Device",
        items=[
            ("cuda", "CUDA", "Use CUDA if available"),
            ("cpu", "CPU", "Force CPU"),
        ],
        default="cuda",
    )
    model_version: bpy.props.EnumProperty(
        name="Model",
        items=[
            ("v2", "v2 (normals)", "Ruicheng/moge-2-vitl-normal"),
            ("v1", "v1", "Ruicheng/moge-vitl"),
        ],
        default="v2",
    )

    @classmethod
    def poll(cls, context):
        if bool(getattr(context.scene, "nm_job_running", False)):
            return False
        if not nm_deps.get_deps_status().installed:
            return False
        if not nm_deps_moge2.get_deps_status().installed:
            return False
        image_id = _selected_image_id(context)
        image_path = _selected_image_path(context)
        return bool(image_id and image_path and image_path.exists())

    _timer = None
    _bp: nm_backend.BackendProcess | None = None
    _started_at: float = 0.0
    _image_id: str = ""
    _image_path: Path | None = None
    _out_dir: Path | None = None
    _export_mode: str = "mesh"

    def _finish(self, context):
        wm = context.window_manager
        try:
            if self._timer:
                wm.event_timer_remove(self._timer)
        except Exception:
            pass
        self._timer = None
        nm_statusbar.clear_progress(context=context)
        try:
            context.scene.nm_job_running = False
            context.scene.nm_job_label = ""
        except Exception:
            pass

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    def _parse_last_json(self, text: str) -> dict:
        for line in reversed((text or "").splitlines()):
            s = (line or "").strip()
            if not s or not s.startswith("{"):
                continue
            try:
                return json.loads(s)
            except Exception:
                continue
        raise ValueError("No JSON payload found in stdout.")

    def invoke(self, context, event):
        if bool(getattr(context.scene, "nm_job_running", False)):
            self.report({"WARNING"}, "Another job is running.")
            return {"CANCELLED"}

        image_id = _selected_image_id(context)
        image_path = _selected_image_path(context)
        if not image_id or image_path is None:
            self.report({"ERROR"}, "No image selected.")
            return {"CANCELLED"}
        if not image_path.exists():
            self.report({"ERROR"}, f"Image not found: {image_path}")
            return {"CANCELLED"}

        model_version = str(getattr(context.scene, "nm_moge2_model_version", self.model_version) or self.model_version)
        device = str(getattr(context.scene, "nm_moge2_device", self.device) or self.device)
        mesh_stride = int(getattr(context.scene, "nm_moge2_mesh_stride", self.stride) or self.stride)
        points_stride = int(getattr(context.scene, "nm_moge2_points_stride", self.stride) or self.stride)
        max_points = int(getattr(context.scene, "nm_moge2_points_max_points", self.max_points) or self.max_points)
        export_mode = str(self.export_mode or "mesh")

        out_dir = nm_paths.addon_cache_dir() / "artifacts" / "moge2" / str(image_id) / uuid.uuid4().hex
        out_dir.mkdir(parents=True, exist_ok=True)

        run_id = uuid.uuid4().hex
        runs_dir = nm_paths.addon_cache_dir() / "runs"
        stdout_path = runs_dir / f"moge2_{run_id}.stdout.txt"
        stderr_path = runs_dir / f"moge2_{run_id}.stderr.txt"

        try:
            self._bp = nm_backend.spawn_backend(
                [
                    "moge2_seed",
                    "--image_id",
                    image_id,
                    "--image",
                    str(image_path),
                    "--out_dir",
                    str(out_dir),
                    "--model_version",
                    model_version,
                    "--device",
                    device,
                    "--export",
                    export_mode,
                    "--mesh_stride",
                    str(int(mesh_stride)),
                    "--points_stride",
                    str(int(points_stride)),
                    "--max_points",
                    str(int(max_points)),
                    "--json",
                ],
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
        except Exception as e:
            nm_logging.append_errolog(f"Failed to start MoGe-2 backend process: {e}", project_root=None)
            self.report({"ERROR"}, "Failed to start MoGe-2 process. See Errolog.")
            return {"CANCELLED"}

        self._started_at = monotonic()
        self._image_id = str(image_id)
        self._image_path = image_path
        self._out_dir = out_dir
        self._export_mode = export_mode

        try:
            context.scene.nm_job_running = True
            context.scene.nm_job_label = f"MoGe-2 running ({export_mode})..."
        except Exception:
            pass
        nm_statusbar.set_progress(context=context, running=True, factor=0.0, text=f"MoGe-2 ({export_mode})")

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        return self.invoke(context, None)

    def modal(self, context, event):
        if event.type == "ESC" and event.value == "PRESS":
            try:
                if self._bp is not None:
                    self._bp.popen.terminate()
            except Exception:
                pass
            try:
                if self._bp is not None:
                    self._bp.close_files()
            except Exception:
                pass
            self._finish(context)
            self.report({"WARNING"}, "Cancelled.")
            return {"CANCELLED"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        if self._bp is None:
            self._finish(context)
            return {"CANCELLED"}

        code = self._bp.popen.poll()
        if code is None:
            try:
                t = monotonic() - float(self._started_at)
                factor = 1.0 - math.exp(-t / 10.0)
                nm_statusbar.set_progress(
                    context=context,
                    running=True,
                    factor=min(0.95, max(0.0, float(factor))),
                    text=f"MoGe-2 ({self._export_mode})",
                )
            except Exception:
                pass
            return {"PASS_THROUGH"}

        # finished
        nm_statusbar.set_progress(context=context, running=True, factor=0.99, text="MoGe-2 (importing)")
        try:
            self._bp.close_files()
        except Exception:
            pass

        stdout_text = self._read_text(self._bp.stdout_path)
        stderr_text = self._read_text(self._bp.stderr_path)
        nm_logging.append_terminal(
            f"$ {' '.join(self._bp.cmd)}\n\n{stdout_text}\n\n{stderr_text}",
            project_root=None,
        )

        if int(code) != 0:
            errolog_path = nm_logging.append_errolog(
                f"MoGe-2 seed failed (exit={code}). See Terminal.txt.",
                project_root=None,
            )
            self._finish(context)
            self.report({"ERROR"}, f"MoGe-2 seed failed. See {errolog_path.name}.")
            return {"CANCELLED"}

        try:
            payload = self._parse_last_json(stdout_text)
        except Exception:
            nm_logging.append_errolog("MoGe-2 seed returned non-JSON output.", project_root=None)
            self._finish(context)
            self.report({"ERROR"}, "MoGe-2 seed returned invalid output. See logs.")
            return {"CANCELLED"}

        mesh_ply_path = payload.get("mesh_ply_path", "") or ""
        ply_path = payload.get("ply_path", "") or ""
        if self._export_mode == "mesh":
            import_path = mesh_ply_path
        elif self._export_mode == "points":
            import_path = ply_path
        else:
            import_path = mesh_ply_path if mesh_ply_path else ply_path
        if not import_path:
            self._finish(context)
            self.report({"ERROR"}, "MoGe-2 seed did not return output path.")
            return {"CANCELLED"}

        try:
            before = set(bpy.data.objects.keys())
            _import_ply(filepath=str(import_path))
            after = set(bpy.data.objects.keys())
            created_names = list(after - before)

            engine_coll = _ensure_generation_collection(_MOGE2_COLLECTION_NAME)
            image_abs = _abs_path_str(self._image_path)
            image_stem = _safe_name((self._image_path.stem if self._image_path else "") or "")
            root, root_created = _ensure_group_root(collection=engine_coll, image_path_abs=image_abs, base_name=image_stem)
            root_name = str(root.name)

            geom_objs: list[bpy.types.Object] = []
            for created_name in created_names:
                obj = bpy.data.objects.get(created_name)
                if obj is None:
                    continue
                _move_object_to_collection(obj, engine_coll)
                geom_objs.append(obj)
                _activate_vertex_colors_for_object(obj)

            # Name imported artifact object (Blender will auto-suffix .001, .002 on collisions).
            if geom_objs:
                desired = f"{root_name}_{'mesh' if self._export_mode == 'mesh' else 'points'}"
                try:
                    geom_objs[0].name = desired
                except Exception:
                    pass
                try:
                    if geom_objs[0].type == "MESH" and geom_objs[0].data is not None:
                        geom_objs[0].data.name = f"{geom_objs[0].name}_{_MOGE2_ENGINE_TAG}"
                except Exception:
                    pass

            cam, cam_created = _ensure_group_camera(collection=engine_coll, image_path_abs=image_abs, name=f"{root_name}_Cam")
            _apply_camera_settings_from_payload(
                cam_obj=cam,
                payload=payload,
                fallback_image_path=self._image_path,
                reset_transform=cam_created,
            )

            # Compute and freeze the group's alignment only once (first time for this image+engine).
            if root_created:
                pivot = None
                if geom_objs:
                    pick = None
                    for o in geom_objs:
                        if o.type == "MESH" and getattr(o.data, "polygons", None) and len(o.data.polygons) > 0:
                            pick = o
                            break
                    if pick is None:
                        pick = geom_objs[0]
                    ray_o, ray_d = _camera_center_ray_world(cam)
                    hit = _raycast_object_world(obj=pick, origin_world=ray_o, direction_world=ray_d)
                    if hit is None:
                        hit = _raycast_object_world(obj=pick, origin_world=ray_o, direction_world=-ray_d)
                    pivot = hit
                    if pivot is None:
                        p1 = _closest_vertex_to_camera_center_ray(obj=pick, origin_world=ray_o, direction_world=ray_d)
                        p2 = _closest_vertex_to_camera_center_ray(obj=pick, origin_world=ray_o, direction_world=-ray_d)
                        if p1 is not None and p2 is not None:
                            pivot = p1 if (p1 - ray_o).length <= (p2 - ray_o).length else p2
                        else:
                            pivot = p1 or p2
                if pivot is None:
                    pivot = _bounds_center_world(geom_objs)

                delta = Vector((0.0, 0.0, 0.0)) - pivot
                _set_group_align_delta(root, delta)
                _set_group_align_rot_euler(root, _WORLD_UP_FIX_EULER)
                try:
                    root["dm_pivot_world"] = [float(pivot.x), float(pivot.y), float(pivot.z)]
                except Exception:
                    pass
                for o in geom_objs:
                    _translate_world(o, delta)
                _translate_world(cam, delta)

                # World-up fix: rotate the whole assembly (mesh/points + camera) so the camera faces forward.
                for o in geom_objs:
                    _apply_world_rot(o, _WORLD_UP_FIX_EULER)
                _apply_world_rot(cam, _WORLD_UP_FIX_EULER)
            else:
                # Reuse the frozen alignment delta to guarantee perfect overlap between mesh/points runs.
                delta = _get_group_align_delta(root)
                rot_euler = _get_group_align_rot_euler(root) or (0.0, 0.0, 0.0)
                for o in geom_objs:
                    _translate_world(o, delta)
                    _apply_world_rot(o, rot_euler)
                if cam_created:
                    _translate_world(cam, delta)
                    _apply_world_rot(cam, rot_euler)

            _parent_keep_world(cam, root)
            for obj in geom_objs:
                _parent_keep_world(obj, root)

            # Apply projected image material for mesh mode (vertex colors remain available as an alternative).
            try:
                if str(self._export_mode) == "mesh":
                    mesh_obj = None
                    for o in geom_objs:
                        if o.type == "MESH" and getattr(o.data, "polygons", None) and len(o.data.polygons) > 0:
                            mesh_obj = o
                            break
                    if mesh_obj is not None:
                        img_path = None
                        # Use the original image for texturing (full resolution / not model-preprocessed).
                        if self._image_path is not None:
                            img_path = self._image_path
                        else:
                            raw = str(payload.get("image_path", "") or payload.get("image_used_path", "") or "").strip()
                            if raw:
                                try:
                                    img_path = Path(raw).resolve()
                                except Exception:
                                    img_path = Path(raw)
                        if img_path is not None:
                            nm_materials.apply_projected_image_material(
                                obj=mesh_obj,
                                cam_obj=cam,
                                image_path=img_path,
                                material_name=f"{root_name}_mat",
                            )
            except Exception:
                pass

            _force_camera_view_and_background(cam)
            try:
                for o in bpy.context.selected_objects:
                    o.select_set(False)
            except Exception:
                pass
            try:
                root.select_set(True)
                bpy.context.view_layer.objects.active = root
            except Exception:
                pass
        except Exception as e:
            nm_logging.append_errolog(f"MoGe-2 import/setup failed: {e}", project_root=None)
            nm_logging.append_terminal(traceback.format_exc(), project_root=None)
            self._finish(context)
            self.report({"ERROR"}, "MoGe-2 import/setup failed. See logs.")
            return {"CANCELLED"}

        self._finish(context)
        self.report({"INFO"}, "MoGe-2 seed imported.")
        return {"FINISHED"}


def register():
    bpy.utils.register_class(NM_OT_install_moge2_dependencies)
    bpy.utils.register_class(NM_OT_moge2_seed)


def unregister():
    bpy.utils.unregister_class(NM_OT_moge2_seed)
    bpy.utils.unregister_class(NM_OT_install_moge2_dependencies)
