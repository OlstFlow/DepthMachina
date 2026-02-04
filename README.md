# DepthMachina (Blender Add-on)

Turn a single image into a quick **mesh** or **point cloud** using zero-shot depth estimation.
Supports **MoGe-2** and **Depth Anything 3** (optional modules).

<p align="center">
  <img src="media/animation.gif" alt="DepthMachina demo" />
</p>

Video tutorial (install + quick start): https://youtu.be/YYZUKBSga94

## Requirements

- Blender **5.0+** (Windows recommended for MVP)
- NVIDIA GPU with enough VRAM 
  - Tested on **RTX 2060 6GB**
- Internet access for first-time dependency install and model downloads
- Free disk space (Python deps + model weights)

## Installation

1. Download the add-on `.zip` from Release page:  https://github.com/OlstFlow/DepthMachina/releases/

2. In Blender: `Edit → Preferences → Add-ons → Install…` and select the `.zip`.
3. Enable **DepthMachina**.
4. Open: `View3D → Sidebar (N) → DepthMachina`.
5. On first run click **Install Dependencies** and wait until it finishes.
6. Install only the module(s) you plan to use:
   - **MoGe-2 → Install MoGe-2 Dependencies**
   - **Depth Anything 3 → Install Depth Anything 3 Dependencies**

Tip: you can hide unused modules in add-on preferences.

## Quick Start (MoGe-2)

1. `View3D → Sidebar → DepthMachina`
2. In **Images**:
   - Set **Folder** (batch workflow) and click **Refresh**
   - Select an image from the list
3. Expand **MoGe-2**:
   - Choose **Mesh** (or **Points**)
   - Click **Generate ▶**

Notes:
- Generation can take minutes depending on GPU performance.
- The viewport camera is adjusted to the last generated image (aspect ratio changes may affect framing).
- Materials are assigned automatically (check **Shading** mode).
- For colored point clouds, Blender may require a third-party point cloud viewer add-on.
(You can try to use mine - Point Cloud Essentials:  https://github.com/OlstFlow/Point-Cloud-Essentials)

## Support
Feel free to support me on Patreon: https://www.patreon.com/OlstFlow

## Licenses / Third-party

See `THIRD_PARTY_NOTICES.md`.
Model weights are downloaded on-demand; their licenses are defined by the respective model owners and may vary.
