from __future__ import annotations

import hashlib
from pathlib import Path

import bpy

from . import nm_backend
from . import nm_deps
from . import nm_logging


DEFAULT_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".exr",
)


def normalize_image_path(path: str | Path) -> Path:
    p = Path(str(path)).expanduser()
    try:
        p = Path(bpy.path.abspath(str(p)))
    except Exception:
        pass
    try:
        return p.resolve()
    except Exception:
        return p


def image_id_from_path(path: Path) -> str:
    # Windows is case-insensitive; normalize to lower for stable ids.
    raw = str(path).replace("/", "\\").lower().encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()


def collect_images_from_folder(folder: Path, *, recursive: bool = False) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    exts = {e.lower() for e in DEFAULT_IMAGE_EXTENSIONS}
    pattern = "**/*" if recursive else "*"
    results: list[Path] = []
    for p in folder.glob(pattern):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        results.append(p)
    results.sort(key=lambda x: x.name.lower())
    return results


def refresh_images_in_scene(scene: bpy.types.Scene) -> int:
    if not hasattr(scene, "nm_images"):
        return 0

    old_items = list(getattr(scene, "nm_images", []) or [])
    excluded: set[str] = set()
    try:
        for ex in list(getattr(scene, "nm_images_excluded", []) or []):
            fp = str(getattr(ex, "filepath", "") or "").strip()
            if fp:
                excluded.add(str(normalize_image_path(fp)))
    except Exception:
        excluded = set()
    old_status_by_path: dict[str, str] = {}
    old_selected_path = None
    try:
        idx = int(getattr(scene, "nm_images_index", -1))
        if 0 <= idx < len(old_items):
            old_selected_path = str(getattr(old_items[idx], "filepath", "") or "").strip()
    except Exception:
        old_selected_path = None

    for it in old_items:
        fp = str(getattr(it, "filepath", "") or "").strip()
        if fp:
            old_status_by_path[fp] = str(getattr(it, "status", "") or "PENDING")

    manual_paths: list[str] = []
    for it in old_items:
        if str(getattr(it, "source", "") or "") != "manual":
            continue
        fp = str(getattr(it, "filepath", "") or "").strip()
        if fp:
            nfp = str(normalize_image_path(fp))
            if nfp not in excluded:
                manual_paths.append(nfp)

    folder_paths: list[str] = []
    folder_raw = str(getattr(scene, "nm_images_folder", "") or "").strip()
    if folder_raw:
        folder = normalize_image_path(folder_raw)
        for p in collect_images_from_folder(folder, recursive=False):
            nfp = str(normalize_image_path(p))
            if nfp not in excluded:
                folder_paths.append(nfp)

    # Dedup while preserving order (folder first).
    seen: set[str] = set()
    ordered: list[tuple[str, str]] = []
    for fp in folder_paths:
        if fp in seen:
            continue
        seen.add(fp)
        ordered.append((fp, "folder"))
    for fp in manual_paths:
        if fp in seen:
            continue
        seen.add(fp)
        ordered.append((fp, "manual"))

    scene.nm_images.clear()
    selected_index = -1
    for i, (fp, src) in enumerate(ordered):
        p = normalize_image_path(fp)
        item = scene.nm_images.add()
        item.filepath = str(p)
        item.filename = p.name
        item.image_id = image_id_from_path(p)
        item.source = src
        s = old_status_by_path.get(fp, "")
        item.status = "" if str(s).upper() == "PENDING" else str(s or "")
        if old_selected_path and fp == old_selected_path:
            selected_index = i

    if len(scene.nm_images) == 0:
        scene.nm_images_index = -1
    else:
        scene.nm_images_index = selected_index if selected_index >= 0 else 0

    return len(scene.nm_images)


class NM_OT_set_images_folder(bpy.types.Operator):
    bl_idname = "nm.set_images_folder"
    bl_label = "Set Images Folder"
    bl_description = "Pick a folder to list images from (images are not copied)"
    bl_options = {"REGISTER"}

    directory: bpy.props.StringProperty(subtype="DIR_PATH")

    def execute(self, context):
        folder_raw = (self.directory or "").strip()
        if not folder_raw:
            self.report({"WARNING"}, "No folder selected.")
            return {"CANCELLED"}

        context.scene.nm_images_folder = folder_raw
        try:
            prefs = context.preferences.addons[__package__].preferences
            prefs.last_images_folder = str(folder_raw)
        except Exception:
            pass
        refresh_images_in_scene(context.scene)
        self.report({"INFO"}, "Folder set.")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class NM_OT_select_image(bpy.types.Operator):
    bl_idname = "nm.select_image"
    bl_label = "Select Image"
    bl_description = "Select an image and show a preview popup"
    bl_options = {"INTERNAL"}

    index: bpy.props.IntProperty(default=-1)

    @classmethod
    def poll(cls, context):
        images = getattr(context.scene, "nm_images", None)
        return images is not None and len(images) > 0

    def _select(self, context, *, mouse_x: int, mouse_y: int) -> None:
        scene = context.scene
        images = getattr(scene, "nm_images", None)
        idx = int(self.index)
        if images is None or idx < 0 or idx >= len(images):
            return
        scene.nm_images_index = idx
        try:
            from . import nm_hover_preview

            nm_hover_preview.schedule_popup(context, mouse_x=int(mouse_x), mouse_y=int(mouse_y))
        except Exception:
            pass

    def invoke(self, context, event):
        self._select(context, mouse_x=int(event.mouse_x), mouse_y=int(event.mouse_y))
        return {"FINISHED"}

    def execute(self, context):
        # Fallback for contexts without an event (use last tracked mouse coords).
        try:
            from . import nm_hover_preview

            mx = int(getattr(nm_hover_preview._SESSION, "mouse_x", 0) or 0)
            my = int(getattr(nm_hover_preview._SESSION, "mouse_y", 0) or 0)
        except Exception:
            mx, my = 0, 0
        self._select(context, mouse_x=mx, mouse_y=my)
        return {"FINISHED"}


class NM_OT_add_images(bpy.types.Operator):
    bl_idname = "nm.add_images"
    bl_label = "Add Photos"
    bl_description = "Add image files to the list (images are not copied)"
    bl_options = {"REGISTER"}

    directory: bpy.props.StringProperty(subtype="DIR_PATH")
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)

    def execute(self, context):
        if not self.files:
            self.report({"WARNING"}, "No files selected.")
            return {"CANCELLED"}

        for f in self.files:
            p = normalize_image_path(Path(self.directory) / f.name)
            if not p.exists() or not p.is_file():
                continue
            # If user re-adds an excluded image, un-exclude it.
            try:
                ex_items = getattr(context.scene, "nm_images_excluded", None)
                if ex_items is not None:
                    for i in range(len(ex_items) - 1, -1, -1):
                        if str(normalize_image_path(ex_items[i].filepath)) == str(p):
                            ex_items.remove(i)
            except Exception:
                pass
            item = context.scene.nm_images.add()
            item.filepath = str(p)
            item.filename = p.name
            item.image_id = image_id_from_path(p)
            item.source = "manual"
            item.status = ""

        # Dedup + keep folder items.
        refresh_images_in_scene(context.scene)
        self.report({"INFO"}, "Photos added.")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class NM_OT_refresh_images(bpy.types.Operator):
    bl_idname = "nm.refresh_images"
    bl_label = "Refresh List"
    bl_description = "Refresh the image list (folder scan + manual entries)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        count = refresh_images_in_scene(context.scene)
        self.report({"INFO"}, f"Loaded {count} image(s).")
        return {"FINISHED"}


class NM_OT_remove_image(bpy.types.Operator):
    bl_idname = "nm.remove_image"
    bl_label = "Remove Image"
    bl_description = "Remove the selected image from the list (excluded from future refreshes)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        images = getattr(context.scene, "nm_images", None)
        idx = int(getattr(context.scene, "nm_images_index", -1))
        return images is not None and 0 <= idx < len(images)

    def execute(self, context):
        scene = context.scene
        images = getattr(scene, "nm_images", None)
        idx = int(getattr(scene, "nm_images_index", -1))
        if images is None or idx < 0 or idx >= len(images):
            self.report({"WARNING"}, "No image selected.")
            return {"CANCELLED"}

        fp_raw = str(getattr(images[idx], "filepath", "") or "").strip()
        if fp_raw:
            fp = str(normalize_image_path(fp_raw))
            try:
                excluded = getattr(scene, "nm_images_excluded", None)
                if excluded is not None:
                    already = False
                    for it in excluded:
                        if str(normalize_image_path(it.filepath)) == fp:
                            already = True
                            break
                    if not already:
                        ex = excluded.add()
                        ex.filepath = fp
            except Exception:
                pass

        try:
            images.remove(idx)
        except Exception:
            pass

        refresh_images_in_scene(scene)
        self.report({"INFO"}, "Removed.")
        return {"FINISHED"}


class NM_OT_backend_hello(bpy.types.Operator):
    bl_idname = "nm.backend_hello"
    bl_label = "Backend: Hello"
    bl_description = "Run the backend CLI hello command (sanity check)"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return nm_deps.get_deps_status().installed

    def execute(self, context):
        result = nm_backend.run_backend(["hello"])
        if result.returncode != 0:
            self.report({"ERROR"}, "Backend hello failed. See logs.")
            nm_logging.append_terminal(result.stdout + "\n" + result.stderr, project_root=None)
            nm_logging.append_errolog("Backend hello failed (non-zero exit).", project_root=None)
            return {"CANCELLED"}

        self.report({"INFO"}, (result.stdout.strip() or "Backend OK"))
        return {"FINISHED"}


class NM_OT_backend_doctor(bpy.types.Operator):
    bl_idname = "nm.backend_doctor"
    bl_label = "Backend: Doctor"
    bl_description = "Run backend environment checks (prints imports/versions)"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return nm_deps.get_deps_status().installed

    def execute(self, context):
        result = nm_backend.run_backend(["doctor"])
        nm_logging.append_terminal(result.stdout + "\n" + result.stderr, project_root=None)
        if result.returncode != 0:
            nm_logging.append_errolog("Backend doctor failed (non-zero exit).", project_root=None)
            self.report({"ERROR"}, "Backend doctor failed. See logs.")
            return {"CANCELLED"}

        self.report({"INFO"}, "Backend doctor OK. See Terminal.txt.")
        return {"FINISHED"}


def register():
    bpy.utils.register_class(NM_OT_set_images_folder)
    bpy.utils.register_class(NM_OT_select_image)
    bpy.utils.register_class(NM_OT_add_images)
    bpy.utils.register_class(NM_OT_remove_image)
    bpy.utils.register_class(NM_OT_refresh_images)
    bpy.utils.register_class(NM_OT_backend_hello)
    bpy.utils.register_class(NM_OT_backend_doctor)


def unregister():
    bpy.utils.unregister_class(NM_OT_backend_doctor)
    bpy.utils.unregister_class(NM_OT_backend_hello)
    bpy.utils.unregister_class(NM_OT_refresh_images)
    bpy.utils.unregister_class(NM_OT_remove_image)
    bpy.utils.unregister_class(NM_OT_add_images)
    bpy.utils.unregister_class(NM_OT_select_image)
    bpy.utils.unregister_class(NM_OT_set_images_folder)
