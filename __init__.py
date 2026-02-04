bl_info = {
    "name": "DepthMachina",
    "author": "Olstflow, GPT & Gemini Bros",
    "version": (0, 0, 1),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > DepthMachina",
    "description": "Generate quick mesh/point-cloud seeds from images and set up cameras.",
    "category": "3D View",
}

import sys

from . import nm_paths
from . import nm_wipe

# If a wipe was scheduled previously, try to complete it before we add `_deps` to sys.path.
# This avoids Windows DLL locking issues (torch/opencv) that can prevent deleting `_deps` at runtime.
try:
    wipe_ok = nm_wipe.try_run_pending_wipe()
except Exception:
    wipe_ok = True

_deps_dir = nm_paths.deps_dir()
if wipe_ok and _deps_dir.exists():
    _p = str(_deps_dir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from . import nm_prefs
from . import nm_props
from . import nm_statusbar
from . import nm_ops_deps
from . import nm_ops_logs
from . import nm_ops_da3
from . import nm_ops_moge2
from . import nm_ops_images
from . import nm_ops_ui
from . import nm_handlers
from . import nm_hover_preview
from . import nm_ui


_REGISTER_MODULES = (
    nm_prefs,
    nm_props,
    nm_statusbar,
    nm_ops_deps,
    nm_ops_logs,
    nm_ops_da3,
    nm_ops_moge2,
    nm_ops_images,
    nm_ops_ui,
    nm_handlers,
    nm_hover_preview,
    nm_ui,
)


def register():
    for module in _REGISTER_MODULES:
        module.register()


def unregister():
    for module in reversed(_REGISTER_MODULES):
        module.unregister()
