from __future__ import annotations

import bpy
from bpy.app.handlers import persistent

from . import nm_logging
from . import nm_ops_images
from . import nm_hover_preview


def _try_refresh_scene(scene: bpy.types.Scene) -> None:
    if not hasattr(scene, "nm_images"):
        return
    nm_ops_images.refresh_images_in_scene(scene)


@persistent
def nm_on_load_post(_dummy) -> None:
    try:
        for scene in bpy.data.scenes:
            _try_refresh_scene(scene)
        nm_hover_preview.ensure_running()
    except Exception as e:
        nm_logging.append_errolog(f"Auto refresh (load_post) failed: {e}", project_root=None)


def register():
    if nm_on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(nm_on_load_post)


def unregister():
    try:
        if nm_on_load_post in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.remove(nm_on_load_post)
    except Exception:
        pass
