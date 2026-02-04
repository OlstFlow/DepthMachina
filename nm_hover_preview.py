from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import bpy

from . import nm_logging
from . import nm_paths


def _prefs():
    try:
        from . import nm_addon

        return nm_addon.get_addon_prefs()
    except Exception:
        return None


@dataclass
class _Session:
    is_running: bool = False
    show_preview: bool = False
    mouse_x: int = 0
    mouse_y: int = 0
    last_key: str = ""
    last_selected_index: int = -1
    scheduled_index: int = -1
    popup_until: float = 0.0
    pending_until: float = 0.0
    last_mouse_move: float = 0.0
    mouse_in_sidebar: bool = False
    click_x: int = 0
    click_y: int = 0
    click_time: float = 0.0
    sidebar_x: int = 0
    sidebar_y: int = 0
    sidebar_w: int = 0
    sidebar_h: int = 0
    image: bpy.types.Image | None = None
    handle: object | None = None
    region_kind: str = "UI"


_SESSION = _Session()


def _norm_path(path: str | Path) -> Path:
    try:
        return Path(bpy.path.abspath(str(path))).resolve()
    except Exception:
        try:
            return Path(str(path)).resolve()
        except Exception:
            return Path(str(path))


def _ensure_preview_path(*, image_path: Path, image_id: str, max_edge: int = 512) -> Path | None:
    if not image_path.exists():
        return None
    try:
        src_mtime = image_path.stat().st_mtime
    except Exception:
        src_mtime = None

    previews_dir = nm_paths.addon_cache_dir() / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(int(src_mtime)) if src_mtime is not None else "0"
    dst = previews_dir / f"{image_id[:32]}_{stamp}.jpg"

    try:
        if dst.exists() and src_mtime is not None and dst.stat().st_mtime >= src_mtime:
            return dst
    except Exception:
        pass

    try:
        from PIL import Image, ImageOps

        pil = Image.open(image_path)
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        pil.thumbnail((int(max_edge), int(max_edge)), resample=Image.BICUBIC)
        pil.save(dst, quality=90)
        return dst if dst.exists() else None
    except Exception:
        return None


def _load_preview_image(*, image_path: Path, image_id: str) -> bpy.types.Image | None:
    preview_path = _ensure_preview_path(image_path=image_path, image_id=image_id, max_edge=512)
    path = preview_path or image_path
    if not path.exists():
        return None
    try:
        img = bpy.data.images.load(str(path), check_existing=True)
        # NOTE: This preview is drawn via a GPU overlay, which bypasses Blender's view-transform pipeline.
        # If the image is converted to linear (sRGB -> linear), then drawing it "as-is" makes mid-tones
        # look too dark while whites stay white. Mark it as data so pixels stay in display space.
        try:
            if hasattr(img, "colorspace_settings") and hasattr(img.colorspace_settings, "name"):
                if hasattr(img.colorspace_settings, "is_data"):
                    img.colorspace_settings.is_data = True
                else:
                    img.colorspace_settings.name = "Non-Color"
        except Exception:
            pass
        try:
            if hasattr(img, "use_view_as_render"):
                img.use_view_as_render = False
        except Exception:
            pass
        try:
            if hasattr(img, "alpha_mode"):
                img.alpha_mode = "STRAIGHT"
        except Exception:
            pass
        return img
    except Exception:
        return None


def _draw_callback():
    prefs = _prefs()
    if not prefs or not getattr(prefs, "hover_preview_enabled", True):
        return

    s = _SESSION
    if not s.show_preview or s.image is None:
        return

    try:
        import gpu
        from gpu_extras.batch import batch_for_shader
    except Exception:
        return

    region = bpy.context.region
    if region is None:
        return

    size = int(getattr(prefs, "hover_preview_size", 256) or 256)
    pad = 14
    # Anchor to the left edge of the sidebar, so the preview appears right after the boundary,
    # and align vertically to the click position.
    sidebar_left_local = int(s.sidebar_x - region.x)
    if sidebar_left_local <= 0:
        return

    x = int(sidebar_left_local - size - pad)
    y = int(s.click_y - region.y - size // 2)

    # Clamp to region bounds
    if x + size > region.width:
        x = max(0, int(region.width - size))
    if y + size > region.height:
        y = max(0, int(region.height - size))
    x = max(0, x)
    y = max(0, y)

    # Never cover the sidebar itself.
    max_right = int(sidebar_left_local - pad)
    if x + size > max_right:
        x = max(0, max_right - size)

    iw = int(getattr(s.image, "size", (0, 0))[0] or 0)
    ih = int(getattr(s.image, "size", (0, 0))[1] or 0)
    if iw <= 0 or ih <= 0:
        return

    # Fit image into a square while keeping aspect
    aspect = float(iw) / float(ih)
    if aspect >= 1.0:
        w = size
        h = int(round(size / aspect))
    else:
        h = size
        w = int(round(size * aspect))
    ox = x + (size - w) // 2
    oy = y + (size - h) // 2

    try:
        s.image.preview_ensure()
    except Exception:
        pass

    # GPU texture
    texture = None
    try:
        tex_from_image = getattr(gpu.texture, "from_image", None)
        if tex_from_image is not None:
            texture = tex_from_image(s.image)
    except Exception:
        texture = None

    if texture is None:
        return

    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("NONE")
    gpu.state.depth_mask_set(False)

    # Background
    bg_shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    bg_batch = batch_for_shader(
        bg_shader,
        "TRI_FAN",
        {"pos": ((x, y), (x + size, y), (x + size, y + size), (x, y + size))},
    )
    bg_shader.bind()
    bg_alpha = float(getattr(prefs, "hover_preview_bg_alpha", 0.85) or 0.85)
    bg_shader.uniform_float("color", (0.05, 0.05, 0.05, bg_alpha))
    bg_batch.draw(bg_shader)

    # Image
    shader = gpu.shader.from_builtin("IMAGE")
    batch = batch_for_shader(
        shader,
        "TRI_FAN",
        {
            "pos": ((ox, oy), (ox + w, oy), (ox + w, oy + h), (ox, oy + h)),
            "texCoord": ((0, 0), (1, 0), (1, 1), (0, 1)),
        },
    )
    shader.bind()
    shader.uniform_sampler("image", texture)
    batch.draw(shader)

    gpu.state.blend_set("NONE")
    gpu.state.depth_mask_set(True)
    gpu.state.depth_test_set("LESS_EQUAL")


def _find_view3d_ui_region(context, *, mouse_x: int, mouse_y: int):
    for area in context.screen.areas:
        if area.type != "VIEW_3D":
            continue
        if not (area.x < mouse_x < area.x + area.width and area.y < mouse_y < area.y + area.height):
            continue
        # Prefer sidebar UI region
        for region in area.regions:
            if region.type != "UI":
                continue
            if region.x <= mouse_x <= region.x + region.width and region.y <= mouse_y <= region.y + region.height:
                return area, region
        # Fallback: still inside View3D, but not inside UI region
        return area, None
    return None, None


def _update_sidebar_metrics(context) -> None:
    s = _SESSION
    for area in context.screen.areas:
        if area.type != "VIEW_3D":
            continue
        for region in area.regions:
            if region.type == "UI":
                s.sidebar_x = int(region.x)
                s.sidebar_y = int(region.y)
                s.sidebar_w = int(region.width)
                s.sidebar_h = int(region.height)
                return


def _tag_redraw(context) -> None:
    for area in context.screen.areas:
        if area.type != "VIEW_3D":
            continue
        area.tag_redraw()


def schedule_popup(context, *, mouse_x: int, mouse_y: int) -> None:
    prefs = _prefs()
    if not prefs or not getattr(prefs, "hover_preview_enabled", True):
        return
    s = _SESSION
    now = monotonic()
    delay_ms = int(getattr(prefs, "hover_preview_delay_ms", 300) or 300)
    popup_ms = int(getattr(prefs, "hover_preview_popup_ms", 1200) or 1200)
    try:
        s.scheduled_index = int(getattr(context.scene, "nm_images_index", -1))
    except Exception:
        s.scheduled_index = -1
    s.click_x = int(mouse_x)
    s.click_y = int(mouse_y)
    s.click_time = now
    s.pending_until = now + (float(delay_ms) / 1000.0)
    s.popup_until = now + (float(delay_ms + popup_ms) / 1000.0)
    _tag_redraw(context)


class NM_OT_hover_preview(bpy.types.Operator):
    bl_idname = "nm.hover_preview"
    bl_label = "DepthMachina Hover Preview"
    bl_options = {"INTERNAL"}

    _timer = None

    def modal(self, context, event):
        prefs = _prefs()
        if not prefs or not getattr(prefs, "hover_preview_enabled", True):
            self.cancel(context)
            return {"FINISHED"}

        s = _SESSION

        if event.type in {"LEFTMOUSE", "RIGHTMOUSE", "MIDDLEMOUSE"} and event.value == "PRESS":
            # Any click outside the image row should instantly dismiss the preview.
            s.show_preview = False
            s.pending_until = 0.0
            s.popup_until = 0.0
            s.scheduled_index = -1
            _tag_redraw(context)
            return {"PASS_THROUGH"}

        if event.type == "MOUSEMOVE":
            s.mouse_x = int(event.mouse_x)
            s.mouse_y = int(event.mouse_y)
            s.last_mouse_move = monotonic()
            area, region = _find_view3d_ui_region(context, mouse_x=s.mouse_x, mouse_y=s.mouse_y)
            s.mouse_in_sidebar = bool(area is not None and region is not None)
            if s.mouse_in_sidebar and region is not None:
                s.sidebar_x = int(region.x)
                s.sidebar_y = int(region.y)
                s.sidebar_w = int(region.width)
                s.sidebar_h = int(region.height)
            return {"PASS_THROUGH"}

        if event.type == "TIMER":
            _update_sidebar_metrics(context)

            now = monotonic()
            delay_ms = int(getattr(prefs, "hover_preview_delay_ms", 300) or 300)
            popup_ms = int(getattr(prefs, "hover_preview_popup_ms", 1200) or 1200)

            scene = context.scene
            images = getattr(scene, "nm_images", None)
            idx = int(getattr(scene, "nm_images_index", -1))
            if images is None or idx < 0 or idx >= len(images):
                s.image = None
                s.last_key = ""
                s.show_preview = False
                return {"PASS_THROUGH"}

            if int(getattr(s, "scheduled_index", -1)) != idx:
                s.show_preview = False
                return {"PASS_THROUGH"}

            item = images[idx]
            image_id = str(getattr(item, "image_id", "") or "").strip()
            image_path_raw = str(getattr(item, "filepath", "") or "").strip()
            if not image_path_raw:
                s.image = None
                s.last_key = ""
                s.show_preview = False
                return {"PASS_THROUGH"}
            image_path = _norm_path(image_path_raw)
            if not image_id:
                image_id = image_path.name
            key = str(image_path)

            if now < float(s.pending_until):
                s.show_preview = False
                return {"PASS_THROUGH"}

            if now >= float(s.popup_until):
                s.show_preview = False
                s.scheduled_index = -1
                return {"PASS_THROUGH"}

            if key != s.last_key:
                s.image = _load_preview_image(image_path=image_path, image_id=image_id)
                s.last_key = key

            s.show_preview = s.image is not None
            _tag_redraw(context)
            return {"PASS_THROUGH"}

        return {"PASS_THROUGH"}

    def execute(self, context):
        s = _SESSION
        if s.is_running:
            return {"CANCELLED"}
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.06, window=context.window)
        wm.modal_handler_add(self)
        s.is_running = True
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
        s = _SESSION
        s.is_running = False
        s.show_preview = False
        s.last_key = ""
        s.scheduled_index = -1
        s.image = None


def ensure_running() -> None:
    try:
        prefs = _prefs()
        if not prefs or not getattr(prefs, "hover_preview_enabled", True):
            return
        if _SESSION.is_running:
            return
        bpy.ops.nm.hover_preview("INVOKE_DEFAULT")
    except Exception as e:
        nm_logging.append_errolog(f"Hover preview start failed: {e}", project_root=None)
        nm_logging.append_terminal(traceback.format_exc(), project_root=None)


def register():
    bpy.utils.register_class(NM_OT_hover_preview)

    # Draw handler on VIEW_3D WINDOW region so the overlay can appear outside the sidebar.
    space = getattr(bpy.types, "SpaceView3D", None)
    if space is None:
        return
    try:
        _SESSION.handle = space.draw_handler_add(_draw_callback, (), "WINDOW", "POST_PIXEL")
        _SESSION.region_kind = "WINDOW"
    except Exception:
        _SESSION.handle = None

    # Auto-start (deferred)
    try:
        bpy.app.timers.register(ensure_running, first_interval=0.2)
    except Exception:
        pass


def unregister():
    # Stop modal if running
    try:
        _SESSION.is_running = False
        _SESSION.show_preview = False
    except Exception:
        pass

    space = getattr(bpy.types, "SpaceView3D", None)
    if space is not None and _SESSION.handle is not None:
        try:
            space.draw_handler_remove(_SESSION.handle, _SESSION.region_kind)
        except Exception:
            pass
        _SESSION.handle = None

    bpy.utils.unregister_class(NM_OT_hover_preview)
