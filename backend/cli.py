from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="depthmachina-backend")

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("hello", help="Sanity check command")
    sub.add_parser("doctor", help="Print environment info and import-check optional dependencies")

    p_moge2 = sub.add_parser("moge2_seed", help="Run MoGe-2 on an image and export a seed mesh/point cloud")
    p_moge2.add_argument("--image_id", type=str, required=True, help="Stable id for cache naming (provided by UI)")
    p_moge2.add_argument("--image", type=Path, required=True, help="Input image path")
    p_moge2.add_argument("--out_dir", type=Path, required=True, help="Output directory for artifacts")
    p_moge2.add_argument("--model_version", choices=["v1", "v2"], default="v2")
    p_moge2.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p_moge2.add_argument("--export", choices=["mesh", "points", "both"], default="both")
    p_moge2.add_argument("--stride", type=int, default=4, help="Legacy: used when mesh_stride/points_stride are not set")
    p_moge2.add_argument("--mesh_stride", type=int, default=None)
    p_moge2.add_argument("--points_stride", type=int, default=None)
    p_moge2.add_argument("--max_points", type=int, default=250000)
    p_moge2.add_argument("--json", action="store_true", help="Print JSON result to stdout")

    p_da3 = sub.add_parser("da3_seed", help="Run Depth Anything 3 on an image and export a seed mesh/point cloud")
    p_da3.add_argument("--image_id", type=str, required=True, help="Stable id for cache naming (provided by UI)")
    p_da3.add_argument("--image", type=Path, required=True, help="Input image path")
    p_da3.add_argument("--out_dir", type=Path, required=True, help="Output directory for artifacts")
    p_da3.add_argument("--model_id", type=str, default="depth-anything/DA3-SMALL")
    p_da3.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p_da3.add_argument("--export", choices=["mesh", "points", "both"], default="both")
    p_da3.add_argument("--mesh_stride", type=int, default=2)
    p_da3.add_argument("--points_stride", type=int, default=4)
    p_da3.add_argument("--max_points", type=int, default=250000)
    p_da3.add_argument("--fov_x_deg", type=float, default=60.0)
    p_da3.add_argument("--target_median_depth", type=float, default=2.0)
    p_da3.add_argument("--max_edge", type=int, default=1024)
    p_da3.add_argument("--json", action="store_true", help="Print JSON result to stdout")

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    ns = _parse_args(argv)

    if ns.cmd == "hello":
        print("DepthMachina backend: hello")
        return 0

    if ns.cmd == "doctor":
        def _try_import(name: str) -> str:
            try:
                mod = __import__(name)
                ver = getattr(mod, "__version__", None)
                return f"OK {name} {ver or ''}".strip()
            except Exception as e:
                return f"FAIL {name}: {e}"

        lines = [
            f"python: {sys.version}",
            f"exe: {sys.executable}",
            _try_import("numpy"),
            _try_import("PIL"),
            _try_import("torch"),
            _try_import("torchvision"),
            _try_import("huggingface_hub"),
            _try_import("cv2"),
            _try_import("scipy"),
            _try_import("depth_anything_3"),
        ]
        out = "\n".join(lines)
        print(out)
        return 0

    if ns.cmd == "moge2_seed":
        try:
            from moge2_seed import run_moge2_seed

            meta = run_moge2_seed(
                image_id=str(ns.image_id),
                image_path=Path(ns.image).resolve(),
                out_dir=Path(ns.out_dir).resolve(),
                model_version=str(ns.model_version),
                device=str(ns.device),
                export=str(ns.export),
                mesh_stride=int(ns.mesh_stride) if ns.mesh_stride is not None else int(ns.stride),
                points_stride=int(ns.points_stride) if ns.points_stride is not None else int(ns.stride),
                max_points=int(ns.max_points),
            )
            if ns.json:
                print(json.dumps(meta, ensure_ascii=False))
            else:
                print("OK")
            return 0
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return 1

    if ns.cmd == "da3_seed":
        try:
            from da3_seed import run_da3_seed

            meta = run_da3_seed(
                image_id=str(ns.image_id),
                image_path=Path(ns.image).resolve(),
                out_dir=Path(ns.out_dir).resolve(),
                model_id=str(ns.model_id),
                device=str(ns.device),
                export=str(ns.export),
                mesh_stride=int(ns.mesh_stride),
                points_stride=int(ns.points_stride),
                max_points=int(ns.max_points),
                fov_x_deg=float(ns.fov_x_deg),
                target_median_depth=float(ns.target_median_depth),
                max_edge=int(ns.max_edge),
            )
            if ns.json:
                print(json.dumps(meta, ensure_ascii=False))
            else:
                print("OK")
            return 0
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return 1

    print(f"ERROR: Unknown command: {ns.cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
