# Third-Party Notices

DepthMachina includes or references third-party components listed below. Some are bundled as reference code
and some are installed on-demand via pip. For on-demand dependencies, the authoritative license is the
upstream project repository and/or the package metadata on PyPI/Hugging Face.

## Bundled reference code

- MoGe-2 reference code in `vendor/moge2/`
  - License texts included in `vendor/moge2/LICENSE` (MIT + Apache-2.0).
  - Sources include:
    - Microsoft MoGe (MIT): `https://github.com/microsoft/MoGe`
    - Meta DINOv2 components under `vendor/moge2/moge/model/dinov2/` (Apache-2.0): `https://github.com/facebookresearch/dinov2`

## Installed on-demand (not distributed)

These are installed into `_deps/` via the add-on installer if the user opts in.

- Depth Anything 3 (`git+https://github.com/ByteDance-Seed/Depth-Anything-3.git`)
- PyTorch (`torch`, `torchvision`)
- NumPy, SciPy
- Pillow
- OpenCV-Python
- Hugging Face Hub (`huggingface_hub`)
- addict

## Model weights (downloaded on-demand)

Model weights are downloaded at runtime into the add-on cache directory. Licenses are defined by the
model owners and may vary by model. The add-on UI should expose only commercial-friendly models by default.
