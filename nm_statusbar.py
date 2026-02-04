from __future__ import annotations

import traceback
from time import monotonic

import bpy


_ORIG_STATUSBAR_DRAW = None
_LOGGED_DRAW_ERROR = False
_LOGGED_REDRAW_ERROR = False
_TIMER_REGISTERED = False
_TIMER_STOP = False
_LAST_FORCE_REDRAW_AT = 0.0


def _get_wm(context):
    try:
        return context.window_manager
    except Exception:
        try:
            return bpy.context.window_manager
        except Exception:
            return None


def _tag_redraw_all_windows() -> None:
    try:
        wm = bpy.context.window_manager
        for win in getattr(wm, "windows", []):
            screen = getattr(win, "screen", None)
            if screen is None:
                continue
            for area in getattr(screen, "areas", []):
                try:
                    area.tag_redraw()
                except Exception:
                    pass
    except Exception:
        return


def _force_ui_redraw_tick() -> None:
    # The Status Bar is not an Area in `screen.areas`, so `tag_redraw()` does not always refresh it.
    # Force a window redraw while our progress is running.
    global _LOGGED_REDRAW_ERROR
    try:
        # Known workaround: poke workspace status text to trigger a status bar redraw.
        ws = getattr(bpy.context, "workspace", None)
        fn = getattr(ws, "status_text_set_internal", None) if ws is not None else None
        if callable(fn):
            fn(None)

        wm = bpy.context.window_manager
        for win in getattr(wm, "windows", []):
            screen = getattr(win, "screen", None)
            if screen is None:
                continue
            try:
                with bpy.context.temp_override(window=win, screen=screen):
                    # Try a couple of redraw modes; different builds behave differently.
                    for typ in ("DRAW", "DRAW_SWAP", "DRAW_WIN", "DRAW_WIN_SWAP"):
                        try:
                            bpy.ops.wm.redraw_timer(type=typ, iterations=1)
                            break
                        except Exception:
                            continue
            except Exception as e:
                if not _LOGGED_REDRAW_ERROR:
                    _LOGGED_REDRAW_ERROR = True
                    try:
                        from . import nm_logging

                        nm_logging.append_errolog(f"Status Bar redraw failed: {e}", project_root=None)
                    except Exception:
                        pass
    except Exception:
        return


def _maybe_force_redraw(context) -> None:
    global _LAST_FORCE_REDRAW_AT
    now = monotonic()
    # Throttle to avoid spamming redraw ops.
    if now - float(_LAST_FORCE_REDRAW_AT) < 0.12:
        return
    _LAST_FORCE_REDRAW_AT = now
    _force_ui_redraw_tick()


def set_progress(*, context, running: bool, factor: float = 0.0, text: str = "") -> None:
    wm = _get_wm(context)
    if wm is None:
        return
    try:
        wm.dm_progress_running = bool(running)
        wm.dm_progress_factor = max(0.0, min(1.0, float(factor)))
        wm.dm_progress_text = str(text or "")
    except Exception:
        return
    try:
        _tag_redraw_all_windows()
    except Exception:
        pass

    if bool(running):
        _maybe_force_redraw(context)


def clear_progress(*, context) -> None:
    set_progress(context=context, running=False, factor=0.0, text="")


def _draw_dm_progress(layout, context) -> None:
    wm = _get_wm(context)
    if wm is None:
        return
    if not bool(getattr(wm, "dm_progress_running", False)):
        return
    factor = float(getattr(wm, "dm_progress_factor", 0.0) or 0.0)
    text = str(getattr(wm, "dm_progress_text", "") or "")

    row = layout.row(align=True)
    row.scale_x = 1.6
    row.label(text="", icon="TIME")
    row.progress(factor=max(0.0, min(1.0, factor)), type="BAR", text=text)

 
def _patched_statusbar_draw(self, context) -> None:
    global _LOGGED_DRAW_ERROR
    try:
        # Re-implement Blender 5.0 Status Bar header draw, injecting our progress bar
        # into the center area (between the two separator_spacer calls) without affecting
        # the right-side status info block.
        layout = self.layout

        # input status
        layout.template_input_status()

        layout.separator_spacer()

        # Messages
        layout.template_reports_banner()

        # Blender jobs
        layout.template_running_jobs()

        # DepthMachina progress
        _draw_dm_progress(layout, context)

        layout.separator_spacer()

        # Stats & Info
        layout.template_status_info()
    except Exception:
        if not _LOGGED_DRAW_ERROR:
            _LOGGED_DRAW_ERROR = True
            try:
                from . import nm_logging

                nm_logging.append_errolog("Status Bar draw error (progress UI).", project_root=None)
                nm_logging.append_terminal(traceback.format_exc(), project_root=None)
            except Exception:
                pass
        try:
            if _ORIG_STATUSBAR_DRAW is not None:
                _ORIG_STATUSBAR_DRAW(self, context)
        except Exception:
            pass


def _progress_redraw_tick() -> float | None:
    # Keep the status bar responsive even when Blender doesn't redraw it frequently (e.g. while cursor is in Viewport).
    global _TIMER_STOP
    if _TIMER_STOP:
        return None
    try:
        wm = bpy.context.window_manager
        running = bool(getattr(wm, "dm_progress_running", False))
    except Exception:
        running = False

    if running:
        _force_ui_redraw_tick()
        return 0.15
    return 0.6


def register():
    global _ORIG_STATUSBAR_DRAW
    global _TIMER_REGISTERED
    global _TIMER_STOP
    bpy.types.WindowManager.dm_progress_running = bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    bpy.types.WindowManager.dm_progress_factor = bpy.props.FloatProperty(default=0.0, min=0.0, max=1.0, options={"HIDDEN"})
    bpy.types.WindowManager.dm_progress_text = bpy.props.StringProperty(default="", options={"HIDDEN"})

    if hasattr(bpy.types, "STATUSBAR_HT_header"):
        cls = bpy.types.STATUSBAR_HT_header
        cur = getattr(cls, "draw", None)
        if getattr(cur, "_dm_is_patched", False):
            return
        _ORIG_STATUSBAR_DRAW = cur
        _patched_statusbar_draw._dm_is_patched = True  # type: ignore[attr-defined]
        cls.draw = _patched_statusbar_draw

    if not _TIMER_REGISTERED:
        _TIMER_STOP = False
        try:
            bpy.app.timers.register(_progress_redraw_tick, first_interval=0.1)
            _TIMER_REGISTERED = True
        except Exception:
            _TIMER_REGISTERED = False


def unregister():
    global _ORIG_STATUSBAR_DRAW
    global _LOGGED_DRAW_ERROR
    global _TIMER_REGISTERED
    global _TIMER_STOP
    _TIMER_STOP = True
    _TIMER_REGISTERED = False
    if hasattr(bpy.types, "STATUSBAR_HT_header"):
        cls = bpy.types.STATUSBAR_HT_header
        cur = getattr(cls, "draw", None)
        if getattr(cur, "_dm_is_patched", False) and _ORIG_STATUSBAR_DRAW is not None:
            cls.draw = _ORIG_STATUSBAR_DRAW
            _ORIG_STATUSBAR_DRAW = None
            _LOGGED_DRAW_ERROR = False
            _LOGGED_REDRAW_ERROR = False
    if hasattr(bpy.types.WindowManager, "dm_progress_text"):
        del bpy.types.WindowManager.dm_progress_text
    if hasattr(bpy.types.WindowManager, "dm_progress_factor"):
        del bpy.types.WindowManager.dm_progress_factor
    if hasattr(bpy.types.WindowManager, "dm_progress_running"):
        del bpy.types.WindowManager.dm_progress_running
