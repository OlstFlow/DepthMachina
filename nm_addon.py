from __future__ import annotations

import bpy


def get_addon_prefs(context=None):
    """
    Best-effort addon preferences lookup.

    Blender stores preferences by the add-on module name (folder name). During development the folder
    name may change (e.g. Spatial_snapshot vs DepthMachina), so we try multiple keys.
    """
    try:
        prefs = (context.preferences if context is not None else bpy.context.preferences)
    except Exception:
        return None

    addon_map = getattr(prefs, "addons", None)
    if not addon_map:
        return None

    # Known module ids for this project
    candidates = []
    try:
        candidates.append(__package__)
    except Exception:
        pass
    candidates.extend(["DepthMachina", "Spatial_snapshot"])

    seen = set()
    for key in candidates:
        if not key or key in seen:
            continue
        seen.add(key)
        addon = addon_map.get(key)
        if addon is not None:
            return addon.preferences

    # Fallback: find by AddonPreferences class idname
    for _key, addon in addon_map.items():
        try:
            prefs_obj = addon.preferences
            if getattr(prefs_obj, "bl_idname", None) in {"DepthMachina", "Spatial_snapshot", __package__}:
                return prefs_obj
        except Exception:
            continue

    return None
