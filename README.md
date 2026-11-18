# CUDA Media Upscaler (Python)

[![CI](https://github.com/jasonpiedrasantasdk/CUDA-Media-Upscaler-Python-/actions/workflows/ci.yml/badge.svg)](https://github.com/jasonpiedrasantasdk/CUDA-Media-Upscaler-Python-/actions/workflows/ci.yml)

A Tkinter desktop application for upscaling images, videos and GIFs with CuPy
on the GPU and a SciPy CPU fallback. This is the Python implementation of the
same pipeline as the native
[`cuda-media-upscaler`](https://github.com/jasonpiedrasantasdk/cuda-media-upscaler)
Qt6 application.

The app runs without a CUDA GPU: if CuPy is missing or GPU compute fails, it
falls back to SciPy and keeps working (slower, and quintic interpolation is
clamped to cubic on the CPU).

## Features

Four tabs:

| Tab | What it does |
|---|---|
| **Image** | Upscale/downscale a single image (PNG, JPEG, BMP, TIFF, WebP) |
| **Video** | Extract frames with `ffmpeg`, upscale each, reassemble with the original audio |
| **GIF** | Decode frames, upscale, rebuild the GIF with an `ffmpeg` palette |
| **Video → GIF** | Resample to a chosen FPS, upscale, generate an optimized GIF |

Shared controls:

- Nearest, bilinear, bicubic and quintic interpolation (`scipy`/`cupyx` orders)
- Light / medium / strong unsharp-style 3×3 sharpening
- Scale slider (1×–8×) or invert mode for downscaling
- Exact target-resolution entry
- GIF settings: FPS, palette size, dither, gifsicle lossy level and `-O` level
- Native file dialogs via `zenity` or `kdialog` when available (better Wayland
  support), Tk dialogs otherwise
- Live log pane, progress bar, frame counter and cancellation

## Requirements

- Python 3.9+
- `numpy`, `Pillow`, `scipy`
- `tkinter` (Debian/Ubuntu: `sudo apt install python3-tk`)
- **Required for video/GIF:** `ffmpeg` and `ffprobe`
- **Optional:**

  | Tool | Purpose |
  |---|---|
  | `cupy-cuda12x` | GPU acceleration (match your CUDA toolkit) |
  | `gifsicle` | Smaller GIF output |
  | `zenity` / `kdialog` | Native file dialogs |

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install numpy pillow scipy
pip install cupy-cuda12x   # optional, NVIDIA GPU only

./launch.sh
# or
python3 cuda_media_upscaler.py
```

`launch.sh` checks Python, the virtualenv, CuPy, `ffmpeg`, `ffprobe`,
`gifsicle` and `zenity`/`kdialog`, then starts the app. It never installs CUDA
drivers.

### Matching CuPy to your toolkit

```bash
nvcc --version                  # check your CUDA version
pip install cupy-cuda12x        # CUDA 12.x / 13.x
```

If CuPy imports but GPU compute fails, the launcher reports the error and the
app continues on the CPU.

## Project layout

```
cuda_media_upscaler.py   Main application (Tkinter GUI + upscale engine)
launch.sh                Environment checks and launcher
requirements.txt         Python dependencies and system-tool notes
docs/NATIVE_PORT_PLAN.md Background: plan for the C++/Qt6 native port
legacy/                  Original standalone image-only upscaler
```

`legacy/cuda_upscaler.py` is the first, image-only version of the tool. It is
kept for reference and is not part of the main application.

## License

MIT — see [LICENSE](LICENSE).
