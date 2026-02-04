from __future__ import annotations

import json
import os
import struct
import sys
import importlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_moge2_code_to_syspath(addon_root: Path) -> Path:
    candidates = [
        addon_root / "vendor" / "moge2",
        addon_root / "_vendor" / "moge2",
    ]

    for base_dir in candidates:
        if (base_dir / "moge").exists() and (base_dir / "utils3d").exists():
            sys.path.insert(0, str(base_dir))
            return base_dir

    # Dev fallback: legacy repo layout (not shipped in release builds).
    legacy_ref_dir = addon_root / "REF" / "MOGE2"
    if (legacy_ref_dir / "moge").exists() and (legacy_ref_dir / "utils3d").exists():
        sys.path.insert(0, str(legacy_ref_dir))
        return legacy_ref_dir

    raise FileNotFoundError(
        "MoGe-2 reference code not found. Expected 'vendor/moge2/{moge,utils3d}' next to the add-on. "
        "If you are running from a dev checkout, REF/MOGE2 is also supported as a fallback."
    )


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


def run_moge2_seed(
    *,
    image_id: str,
    image_path: Path,
    out_dir: Path,
    model_version: str,
    device: str,
    export: str,
    mesh_stride: int,
    points_stride: int,
    max_points: int,
) -> dict[str, Any]:
    addon_root = Path(__file__).resolve().parents[1]
    _add_moge2_code_to_syspath(addon_root)
    _configure_hf_cache()

    import numpy as np
    import torch
    from PIL import Image, ImageOps

    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if model_version not in {"v1", "v2"}:
        raise ValueError("model_version must be v1 or v2")

    if model_version == "v2":
        pretrained = "Ruicheng/moge-2-vitl-normal"
    else:
        pretrained = "Ruicheng/moge-vitl"

    want_cuda = device == "cuda"
    use_cuda = bool(want_cuda and torch.cuda.is_available())
    torch_device = torch.device("cuda" if use_cuda else "cpu")

    try:
        module = importlib.import_module(f"moge.model.{model_version}")
        ModelCls = getattr(module, "MoGeModel")
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"MoGe-2 import failed: missing module '{e.name}'. "
            "Install MoGe-2 dependencies (opencv-python, scipy, torch, torchvision, pillow, huggingface_hub)."
        ) from e
    except Exception as e:
        raise RuntimeError(f"MoGe-2 import failed while loading model '{model_version}': {e}") from e
    # MoGe's from_pretrained already loads checkpoint with map_location='cpu'.
    # Passing map_location here would be forwarded to hf_hub_download and crash.
    model = ModelCls.from_pretrained(pretrained)
    model.eval()
    model.to(torch_device)

    pil_raw = Image.open(image_path)
    exif_orientation = None
    try:
        exif_orientation = int(pil_raw.getexif().get(274, 1))
    except Exception:
        exif_orientation = None

    pil = ImageOps.exif_transpose(pil_raw).convert("RGB")
    rgb = np.array(pil, dtype=np.uint8)
    img_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

    out = model.infer(img_t, resolution_level=9, use_fp16=(torch_device.type == "cuda"))
    points = out["points"].detach().cpu().numpy()
    depth = out.get("depth", None)
    mask = out.get("mask", None)
    intrinsics = out.get("intrinsics", None)
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    if hasattr(intrinsics, "detach"):
        intrinsics = intrinsics.detach().cpu().numpy()

    if points.ndim != 3 or points.shape[-1] != 3:
        raise RuntimeError(f"Unexpected points shape: {points.shape}")

    def _sample_grid(*, stride: int):
        stride = int(stride)
        grid_pts = points[::stride, ::stride, :]
        grid_rgb = rgb[::stride, ::stride, :]
        valid = np.isfinite(grid_pts).all(axis=-1)
        if mask is not None:
            valid &= mask[::stride, ::stride].astype(bool)
        return grid_pts, grid_rgb, valid

    export = (export or "both").lower()
    if export not in {"mesh", "points", "both"}:
        raise ValueError("export must be: mesh|points|both")

    verts_mesh = None
    cols_mesh = None
    faces: list[tuple[int, int, int]] = []
    if export in {"mesh", "both"}:
        grid_pts, grid_rgb, valid = _sample_grid(stride=int(mesh_stride))
        Hs, Ws = int(grid_pts.shape[0]), int(grid_pts.shape[1])
        idx_map = np.full((Hs, Ws), -1, dtype=np.int32)
        verts_mesh = grid_pts[valid].reshape(-1, 3).astype(np.float32)
        cols_mesh = grid_rgb[valid].reshape(-1, 3).astype(np.uint8)
        idx_map[valid] = np.arange(verts_mesh.shape[0], dtype=np.int32)

        rtol = 0.04
        for y in range(Hs - 1):
            row0 = idx_map[y]
            row1 = idx_map[y + 1]
            for x in range(Ws - 1):
                a = int(row0[x])
                b = int(row0[x + 1])
                c = int(row1[x])
                d = int(row1[x + 1])
                if a >= 0 and b >= 0 and c >= 0 and d >= 0:
                    za = float(grid_pts[y, x, 2])
                    zb = float(grid_pts[y, x + 1, 2])
                    zc = float(grid_pts[y + 1, x, 2])
                    zd = float(grid_pts[y + 1, x + 1, 2])

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
        grid_pts, grid_rgb, valid = _sample_grid(stride=int(points_stride))
        pts = grid_pts[valid].reshape(-1, 3).astype(np.float32)
        cols_pts = grid_rgb[valid].reshape(-1, 3).astype(np.uint8)
        if pts.shape[0] > int(max_points):
            idx = np.random.choice(pts.shape[0], size=int(max_points), replace=False)
            pts = pts[idx]
            cols_pts = cols_pts[idx]

    # Match common DCC coordinates (and vendor/moge2 ComfyUI node behavior):
    # X right stays, Y down -> up, Z forward -> -Z.
    if verts_mesh is not None:
        verts_mesh = verts_mesh * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    if pts is not None:
        pts = pts * np.array([1.0, -1.0, -1.0], dtype=np.float32)

    artifacts_dir = Path(out_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save the exact pixels used for inference as an artifact, so Blender background matches 1:1.
    used_image_path = artifacts_dir / "image_used.jpg"
    try:
        pil.save(used_image_path, quality=95)
    except Exception:
        # Best-effort; if saving fails, we still proceed.
        pass

    ply_path = artifacts_dir / "seed_points.ply"
    mesh_ply_path = artifacts_dir / "seed_mesh.ply"
    meta_path = artifacts_dir / "seed.json"

    if pts is not None and cols_pts is not None:
        _write_ply_vertices_rgb(ply_path, pts, cols_pts)
    if verts_mesh is not None and cols_mesh is not None and faces:
        _write_ply_mesh_rgb(mesh_ply_path, verts_mesh, cols_mesh, faces)

    image_h, image_w = int(rgb.shape[0]), int(rgb.shape[1])
    intrinsics_list = None
    fov_x_deg = None
    fov_y_deg = None
    if intrinsics is not None:
        intr = intrinsics.astype(float)
        intrinsics_list = intr.tolist()
        fx = float(intr[0][0])
        fy = float(intr[1][1])
        cx = float(intr[0][2])
        cy = float(intr[1][2])

        if cx <= 1.5 and cy <= 1.5:
            fov_x = 2.0 * math.atan(0.5 / max(fx, 1e-9))
            fov_y = 2.0 * math.atan(0.5 / max(fy, 1e-9))
        else:
            fov_x = 2.0 * math.atan(image_w / (2.0 * max(fx, 1e-9)))
            fov_y = 2.0 * math.atan(image_h / (2.0 * max(fy, 1e-9)))
        fov_x_deg = float(fov_x * 180.0 / math.pi)
        fov_y_deg = float(fov_y * 180.0 / math.pi)
    meta = {
        "created_at": _now_iso(),
        "image_id": image_id,
        "image_path": str(image_path),
        "image_used_path": str(used_image_path) if used_image_path.exists() else "",
        "exif_orientation": exif_orientation,
        "model_version": model_version,
        "pretrained": pretrained,
        "device_used": str(torch_device),
        "export": export,
        "mesh_stride": int(mesh_stride),
        "points_stride": int(points_stride),
        "max_points": int(max_points),
        "image_width": image_w,
        "image_height": image_h,
        "intrinsics": intrinsics_list,
        "fov_x_deg": fov_x_deg,
        "fov_y_deg": fov_y_deg,
        "ply_path": str(ply_path) if (pts is not None) else "",
        "points_count": int(pts.shape[0]) if pts is not None else 0,
        "mesh_ply_path": str(mesh_ply_path) if (verts_mesh is not None and faces) else "",
        "mesh_faces_count": int(len(faces)),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return meta
