from __future__ import annotations

import json
import os
import struct
import sys
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure_hf_cache() -> Path:
    addon_root = Path(__file__).resolve().parents[1]
    cache_root = addon_root / "_cache" / "huggingface"
    hub_cache = cache_root / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TORCH_HOME", str(addon_root / "_cache" / "torch"))
    return cache_root


def _write_ply_vertices_rgb(path: Path, xyz, rgb) -> None:
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError("xyz and rgb must have same length")
    n = int(xyz.shape[0])

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header\n",
        ]
    ).encode("ascii")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        for i in range(n):
            x, y, z = float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])
            r, g, b = int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])
            f.write(struct.pack("<fffBBB", x, y, z, r, g, b))


def _write_ply_mesh_rgb(path: Path, xyz, rgb, faces) -> None:
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError("xyz and rgb must have same length")
    n = int(xyz.shape[0])
    m = int(len(faces))

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            f"element face {m}",
            "property list uchar int vertex_indices",
            "end_header\n",
        ]
    ).encode("ascii")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        for i in range(n):
            x, y, z = float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])
            r, g, b = int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])
            f.write(struct.pack("<fffBBB", x, y, z, r, g, b))
        for (a, b, c) in faces:
            f.write(struct.pack("<Biii", 3, int(a), int(b), int(c)))


def _fov_y_from_fov_x(*, fov_x_deg: float, w: int, h: int) -> float:
    fov_x = math.radians(max(1.0, min(179.0, float(fov_x_deg))))
    aspect = float(h) / max(float(w), 1e-9)
    fov_y = 2.0 * math.atan(math.tan(0.5 * fov_x) * aspect)
    return math.degrees(fov_y)


def run_da3_seed(
    *,
    image_id: str,
    image_path: Path,
    out_dir: Path,
    model_id: str,
    device: str,
    export: str,
    mesh_stride: int,
    points_stride: int,
    max_points: int,
    fov_x_deg: float,
    target_median_depth: float,
    max_edge: int,
) -> dict[str, Any]:
    _configure_hf_cache()

    import numpy as np
    import torch
    from PIL import Image, ImageOps

    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Read + EXIF transpose to avoid sideways images in Blender.
    pil = Image.open(image_path)
    exif_orientation = None
    try:
        exif_orientation = int(pil.getexif().get(274, 1))
    except Exception:
        exif_orientation = None
    pil = ImageOps.exif_transpose(pil).convert("RGB")

    # Downscale to reduce VRAM/compute while keeping aspect.
    w0, h0 = pil.size
    max_edge = int(max_edge)
    if max(w0, h0) > max_edge:
        scale = float(max_edge) / float(max(w0, h0))
        nw = max(1, int(round(w0 * scale)))
        nh = max(1, int(round(h0 * scale)))
        pil = pil.resize((nw, nh), resample=Image.BICUBIC)

    artifacts_dir = Path(out_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    input_path = artifacts_dir / "input_for_model.jpg"
    pil.save(input_path, quality=95)

    want_cuda = device == "cuda"
    use_cuda = bool(want_cuda and torch.cuda.is_available())
    torch_device = torch.device("cuda" if use_cuda else "cpu")

    try:
        from depth_anything_3.api import DepthAnything3
    except Exception as e:
        raise RuntimeError(
            "Depth Anything 3 is not installed (import depth_anything_3.api failed). "
            "Install DA3 dependencies first."
        ) from e

    model = DepthAnything3.from_pretrained(str(model_id))
    model = model.to(device=torch_device)

    with torch.inference_mode():
        prediction = model.inference([str(input_path)])

    # depth: [N,H,W]
    depth = getattr(prediction, "depth", None)
    if depth is None:
        raise RuntimeError("DA3 inference did not return depth.")
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    depth = depth.astype("float32")
    if depth.ndim == 3:
        depth = depth[0]

    processed = getattr(prediction, "processed_images", None)
    if processed is not None and hasattr(processed, "detach"):
        processed = processed.detach().cpu().numpy()

    # Save the exact pixels used for inference (after DA3 preprocessing) as an artifact.
    used_image_path = artifacts_dir / "image_used.jpg"
    rgb = None
    if processed is not None:
        arr = processed[0]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        rgb = arr
        Image.fromarray(arr).save(used_image_path, quality=95)
    else:
        pil.save(used_image_path, quality=95)
        rgb = np.asarray(pil, dtype=np.uint8)

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise RuntimeError(f"Unexpected RGB shape: {rgb.shape}")

    H, W = int(depth.shape[0]), int(depth.shape[1])
    if int(rgb.shape[0]) != H or int(rgb.shape[1]) != W:
        # Best-effort: resize rgb to match depth.
        rgb = np.asarray(Image.fromarray(rgb).resize((W, H), resample=Image.BICUBIC), dtype=np.uint8)

    valid_full = np.isfinite(depth) & (depth > 0)
    if not bool(valid_full.any()):
        raise RuntimeError("DA3 depth map has no valid positive values.")
    median = float(np.median(depth[valid_full]))
    if not math.isfinite(median) or median <= 0:
        median = 1.0
    depth_scaled = (depth / median) * float(target_median_depth)

    fov_y_deg = _fov_y_from_fov_x(fov_x_deg=float(fov_x_deg), w=W, h=H)
    fx = (0.5 * float(W)) / math.tan(0.5 * math.radians(float(fov_x_deg)))
    fy = (0.5 * float(H)) / math.tan(0.5 * math.radians(float(fov_y_deg)))
    cx = (float(W) - 1.0) * 0.5
    cy = (float(H) - 1.0) * 0.5

    def _grid_from_depth(*, stride: int):
        stride = int(stride)
        zs = depth_scaled[::stride, ::stride]
        rgbs = rgb[::stride, ::stride, :]
        valid = valid_full[::stride, ::stride]
        Hs, Ws = int(zs.shape[0]), int(zs.shape[1])

        us = (np.arange(0, Ws, dtype=np.float32) * float(stride)).reshape(1, Ws)
        vs = (np.arange(0, Hs, dtype=np.float32) * float(stride)).reshape(Hs, 1)
        uu = np.broadcast_to(us, (Hs, Ws))
        vv = np.broadcast_to(vs, (Hs, Ws))

        x = (uu - cx) / max(fx, 1e-9) * zs
        y = (vv - cy) / max(fy, 1e-9) * zs
        pts_grid = np.stack([x, y, zs], axis=-1).astype(np.float32)
        return pts_grid, rgbs, valid, zs

    export = (export or "both").lower()
    if export not in {"mesh", "points", "both"}:
        raise ValueError("export must be: mesh|points|both")

    verts_mesh = None
    cols_mesh = None
    faces: list[tuple[int, int, int]] = []
    if export in {"mesh", "both"}:
        pts_grid, rgbs, valid, zs = _grid_from_depth(stride=int(mesh_stride))
        Hs, Ws = int(valid.shape[0]), int(valid.shape[1])
        idx_map = np.full((Hs, Ws), -1, dtype=np.int32)
        verts_mesh = pts_grid[valid].reshape(-1, 3).astype(np.float32)
        cols_mesh = rgbs[valid].reshape(-1, 3).astype(np.uint8)
        idx_map[valid] = np.arange(verts_mesh.shape[0], dtype=np.int32)

        rtol = 0.05
        for yy in range(Hs - 1):
            row0 = idx_map[yy]
            row1 = idx_map[yy + 1]
            for xx in range(Ws - 1):
                a = int(row0[xx])
                b = int(row0[xx + 1])
                c = int(row1[xx])
                d = int(row1[xx + 1])
                if a >= 0 and b >= 0 and c >= 0 and d >= 0:
                    za = float(zs[yy, xx])
                    zb = float(zs[yy, xx + 1])
                    zc = float(zs[yy + 1, xx])
                    zd = float(zs[yy + 1, xx + 1])

                    zmin = min(za, zb, zc, zd)
                    if not (math.isfinite(zmin) and zmin > 0):
                        continue

                    def _edge_ok(z0: float, z1: float) -> bool:
                        return abs(z0 - z1) <= rtol * min(z0, z1)

                    if not (_edge_ok(za, zb) and _edge_ok(za, zc) and _edge_ok(zb, zd) and _edge_ok(zc, zd)):
                        continue
                    faces.append((a, c, d))
                    faces.append((a, d, b))

    pts = None
    cols_pts = None
    if export in {"points", "both"}:
        pts_grid, rgbs, valid, _zs = _grid_from_depth(stride=int(points_stride))
        pts = pts_grid[valid].reshape(-1, 3).astype(np.float32)
        cols_pts = rgbs[valid].reshape(-1, 3).astype(np.uint8)
        if pts.shape[0] > int(max_points):
            idx = np.random.choice(pts.shape[0], size=int(max_points), replace=False)
            pts = pts[idx]
            cols_pts = cols_pts[idx]

    # Match common DCC coordinates (Blender camera looks along -Z):
    # X right stays, Y down -> up, Z forward -> -Z.
    if verts_mesh is not None:
        verts_mesh = verts_mesh * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    if pts is not None:
        pts = pts * np.array([1.0, -1.0, -1.0], dtype=np.float32)

    ply_path = artifacts_dir / "seed_points.ply"
    mesh_ply_path = artifacts_dir / "seed_mesh.ply"
    meta_path = artifacts_dir / "seed.json"

    if pts is not None and cols_pts is not None:
        _write_ply_vertices_rgb(ply_path, pts, cols_pts)
    if verts_mesh is not None and cols_mesh is not None and faces:
        _write_ply_mesh_rgb(mesh_ply_path, verts_mesh, cols_mesh, faces)

    meta = {
        "created_at": _now_iso(),
        "image_id": str(image_id),
        "image_path": str(image_path),
        "image_used_path": str(used_image_path) if used_image_path.exists() else "",
        "exif_orientation": exif_orientation,
        "model_id": str(model_id),
        "device_used": str(torch_device),
        "export": export,
        "mesh_stride": int(mesh_stride),
        "points_stride": int(points_stride),
        "max_points": int(max_points),
        "max_edge": int(max_edge),
        "target_median_depth": float(target_median_depth),
        "fov_x_deg": float(fov_x_deg),
        "fov_y_deg": float(fov_y_deg),
        "image_width": int(W),
        "image_height": int(H),
        "ply_path": str(ply_path) if (pts is not None) else "",
        "points_count": int(pts.shape[0]) if pts is not None else 0,
        "mesh_ply_path": str(mesh_ply_path) if (verts_mesh is not None and faces) else "",
        "mesh_faces_count": int(len(faces)),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
