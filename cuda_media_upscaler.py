#!/usr/bin/env python3
"""
Unified CUDA Media Upscaler
A professional GUI application for upscaling images, videos, and GIFs using CUDA acceleration.
Features: Video-to-GIF pipeline, gifsicle optimization, native file dialogs.

Author: Jason Piedrasanta
Hardened: Phase 1-2 fixes (crash, OOM, thread safety, resource management)
"""

import os
import sys
import time
import tempfile
import shutil
import subprocess
import threading
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
from enum import Enum

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageSequence

# Pillow version compatibility
_LANCZOS = getattr(Image, 'Resampling', Image).LANCZOS

# Set up logging -- verbose to terminal
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Attempt to import CuPy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy.ndimage import zoom as gpu_zoom
    from cupyx.scipy.ndimage import convolve as gpu_convolve
    CUDA_AVAILABLE = True
    logger.info(f"CuPy {cp.__version__} loaded -- CUDA GPU acceleration available")
except ImportError:
    CUDA_AVAILABLE = False
    cp = None
    logger.warning("CuPy not available -- will use CPU fallback")

# Check scipy availability at startup
try:
    from scipy.ndimage import zoom as cpu_zoom, convolve as cpu_convolve
    import scipy
    SCIPY_AVAILABLE = True
    logger.info(f"SciPy {scipy.__version__} loaded -- CPU fallback available")
except ImportError:
    SCIPY_AVAILABLE = False
    cpu_zoom = None
    cpu_convolve = None
    logger.warning("SciPy not available -- CPU fallback disabled")

# Startup validation
if not CUDA_AVAILABLE and not SCIPY_AVAILABLE:
    logger.error(
        "Neither CuPy (GPU) nor scipy (CPU fallback) is available. "
        "Install at least one: pip install cupy-cuda12x OR pip install scipy"
    )


class InterpolationMethod(Enum):
    """Supported interpolation methods."""
    NEAREST = 0
    BILINEAR = 1
    BICUBIC = 3
    QUINTIC = 5


class SharpenLevel(Enum):
    """Sharpening intensity levels."""
    NONE = "none"
    LIGHT = "light"
    MEDIUM = "medium"
    STRONG = "strong"


@dataclass
class UpscaleSettings:
    """Container for upscale parameters."""
    scale: float = 2.0
    interpolation: InterpolationMethod = InterpolationMethod.BICUBIC
    sharpen: SharpenLevel = SharpenLevel.NONE


@dataclass
class GifSettings:
    """GIF-specific settings."""
    fps: int = 15
    colors: int = 256
    dither: str = "sierra2_4a"
    lossy: int = 0  # gifsicle lossy compression (0-200)
    optimize_level: int = 3  # gifsicle -O level (1-3)


# ===========================================================================
# Native File Dialog Support (Wayland/GTK)
# ===========================================================================

class NativeFileDialog:
    """Native file dialogs using zenity/kdialog for better Wayland support."""

    _cached_tool: Optional[str] = None

    @classmethod
    def _detect_dialog_tool(cls) -> str:
        """Detect available dialog tool (cached)."""
        if cls._cached_tool is None:
            for tool in ['zenity', 'kdialog', 'yad']:
                if shutil.which(tool):
                    cls._cached_tool = tool
                    logger.debug(f"Native dialog tool: {tool}")
                    break
            else:
                cls._cached_tool = 'tkinter'
                logger.debug("Using tkinter fallback for file dialogs")
        return cls._cached_tool

    @staticmethod
    def _expand_filetypes_for_tk(filetypes: List[Tuple]) -> List[Tuple]:
        """Expand multi-pattern filetypes for tkinter compatibility."""
        expanded = []
        for name, patterns in filetypes:
            parts = patterns.split()
            if len(parts) > 1:
                for p in parts:
                    expanded.append((name, p))
            else:
                expanded.append((name, patterns))
        return expanded

    @staticmethod
    def askopenfilename(title: str = "Open File", filetypes: List[Tuple] = None,
                        parent=None) -> str:
        """Open file dialog using native tools."""
        tool = NativeFileDialog._detect_dialog_tool()

        if tool == 'zenity':
            cmd = ['zenity', '--file-selection', f'--title={title}']
            if filetypes:
                for name, patterns in filetypes:
                    for pattern in patterns.split():
                        cmd.extend(['--file-filter', f'{name} | {pattern}'])
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    return result.stdout.strip()
            except subprocess.TimeoutExpired:
                logger.warning("Native file dialog timed out, falling back to tkinter")
            except Exception as e:
                logger.warning(f"Native dialog error: {e}")

        elif tool == 'kdialog':
            cmd = ['kdialog', '--getopenfilename', '.', f'{title}']
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    return result.stdout.strip()
            except subprocess.TimeoutExpired:
                logger.warning("Native file dialog timed out, falling back to tkinter")
            except Exception as e:
                logger.warning(f"Native dialog error: {e}")

        # Fallback to tkinter
        from tkinter import filedialog
        tk_filetypes = NativeFileDialog._expand_filetypes_for_tk(filetypes) if filetypes else []
        return filedialog.askopenfilename(title=title, filetypes=tk_filetypes)

    @staticmethod
    def asksaveasfilename(title: str = "Save File", defaultextension: str = "",
                          filetypes: List[Tuple] = None, parent=None) -> str:
        """Save file dialog using native tools."""
        tool = NativeFileDialog._detect_dialog_tool()

        if tool == 'zenity':
            cmd = ['zenity', '--file-selection', '--save', '--confirm-overwrite',
                   f'--title={title}']
            if filetypes:
                for name, patterns in filetypes:
                    for pattern in patterns.split():
                        cmd.extend(['--file-filter', f'{name} | {pattern}'])
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    path = result.stdout.strip()
                    if defaultextension and not path.endswith(defaultextension):
                        path += defaultextension
                    return path
            except subprocess.TimeoutExpired:
                logger.warning("Native file dialog timed out, falling back to tkinter")
            except Exception as e:
                logger.warning(f"Native dialog error: {e}")

        elif tool == 'kdialog':
            cmd = ['kdialog', '--getsavefilename', '.', f'{title}']
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    path = result.stdout.strip()
                    if defaultextension and not path.endswith(defaultextension):
                        path += defaultextension
                    return path
            except subprocess.TimeoutExpired:
                logger.warning("Native file dialog timed out, falling back to tkinter")
            except Exception as e:
                logger.warning(f"Native dialog error: {e}")

        # Fallback to tkinter
        from tkinter import filedialog
        tk_filetypes = NativeFileDialog._expand_filetypes_for_tk(filetypes) if filetypes else []
        return filedialog.asksaveasfilename(
            title=title, defaultextension=defaultextension, filetypes=tk_filetypes
        )


# ===========================================================================
# CUDA Upscaling Engine
# ===========================================================================

class CUDAUpscaler:
    """CUDA-accelerated image upscaling engine using CuPy."""

    SHARPEN_KERNELS = {
        SharpenLevel.LIGHT: np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0]
        ], dtype=np.float32),
        SharpenLevel.MEDIUM: np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32),
        SharpenLevel.STRONG: np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ], dtype=np.float32),
    }

    @staticmethod
    def get_gpu_info() -> str:
        """Get GPU device information."""
        if not CUDA_AVAILABLE:
            return "CUDA not available - CPU fallback mode"
        try:
            device = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(device.id)
            name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
            total_mem = props['totalGlobalMem'] / (1024**3)
            return f"{name} ({total_mem:.1f} GB)"
        except Exception as e:
            return f"GPU detection error: {e}"

    @staticmethod
    def get_free_memory() -> Tuple[float, float]:
        """Get free and total GPU memory in GB."""
        if not CUDA_AVAILABLE:
            return (0.0, 0.0)
        try:
            mem_info = cp.cuda.runtime.memGetInfo()
            free_mem = mem_info[0] / (1024**3)
            total_mem = mem_info[1] / (1024**3)
            return (free_mem, total_mem)
        except Exception:
            return (0.0, 0.0)

    @classmethod
    def upscale(
        cls,
        img_array: np.ndarray,
        scale: float,
        interpolation: InterpolationMethod = InterpolationMethod.BICUBIC,
        sharpen: SharpenLevel = SharpenLevel.NONE
    ) -> np.ndarray:
        """Scale an image (up or down) using GPU acceleration with CPU fallback."""
        # Validate scale — allow downscale (0.01 min) and upscale (8.0 max)
        scale = max(0.01, min(8.0, scale))

        in_h, in_w = img_array.shape[:2]
        out_h, out_w = int(in_h * scale), int(in_w * scale)
        channels = img_array.shape[2] if img_array.ndim == 3 else 1

        logger.debug(
            f"Upscale: {in_w}x{in_h} -> {out_w}x{out_h} "
            f"(scale={scale}x, interp={interpolation.name}, "
            f"sharpen={sharpen.value}, ch={channels})"
        )

        t0 = time.perf_counter()

        if CUDA_AVAILABLE:
            try:
                result = cls._gpu_upscale(img_array, scale, interpolation, sharpen)
                dt = time.perf_counter() - t0
                logger.info(
                    f"GPU upscale complete: {in_w}x{in_h} -> {result.shape[1]}x{result.shape[0]} "
                    f"in {dt:.2f}s"
                )
                return result
            except Exception as e:
                logger.warning(f"GPU error ({type(e).__name__}: {e}), falling back to CPU...")
                # Ensure GPU memory is freed on error
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass

        if not SCIPY_AVAILABLE:
            raise RuntimeError(
                "No upscaling backend available. Install cupy-cuda12x (GPU) or scipy (CPU)."
            )

        result = cls._cpu_upscale(img_array, scale, interpolation, sharpen)
        dt = time.perf_counter() - t0
        logger.info(
            f"CPU upscale complete: {in_w}x{in_h} -> {result.shape[1]}x{result.shape[0]} "
            f"in {dt:.2f}s"
        )
        return result

    @classmethod
    def _gpu_upscale(
        cls,
        img_array: np.ndarray,
        scale: float,
        interpolation: InterpolationMethod,
        sharpen: SharpenLevel
    ) -> np.ndarray:
        """GPU-accelerated upscaling implementation."""
        gpu_img = None
        gpu_result = None
        try:
            free_mem, total_mem = cls.get_free_memory()
            est_vram_mb = (img_array.nbytes * scale * scale * 4) / (1024**2)  # float32
            logger.debug(f"GPU VRAM: {free_mem:.1f}GB free, estimated need: {est_vram_mb:.0f}MB")

            gpu_img = cp.asarray(img_array, dtype=cp.float32)

            if gpu_img.ndim == 3:
                zoom_factors = (scale, scale, 1.0)
            else:
                zoom_factors = (scale, scale)

            gpu_result = gpu_zoom(
                gpu_img,
                zoom_factors,
                order=interpolation.value,
                mode='reflect'
            )

            if sharpen != SharpenLevel.NONE:
                logger.debug(f"Applying {sharpen.value} sharpening kernel")
                kernel = cp.asarray(cls.SHARPEN_KERNELS[sharpen])
                if gpu_result.ndim == 3:
                    for c in range(gpu_result.shape[2]):
                        gpu_result[:, :, c] = gpu_convolve(
                            gpu_result[:, :, c],
                            kernel,
                            mode='reflect'
                        )
                else:
                    gpu_result = gpu_convolve(gpu_result, kernel, mode='reflect')

            gpu_result = cp.clip(gpu_result, 0, 255)
            result = cp.asnumpy(gpu_result).astype(np.uint8)
            return result
        finally:
            # Always clean up GPU memory
            del gpu_img, gpu_result
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    @classmethod
    def _cpu_upscale(
        cls,
        img_array: np.ndarray,
        scale: float,
        interpolation: InterpolationMethod,
        sharpen: SharpenLevel
    ) -> np.ndarray:
        """CPU fallback upscaling using scipy."""
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is not installed. Install with: pip install scipy")

        img_float = img_array.astype(np.float32)

        if img_float.ndim == 3:
            zoom_factors = (scale, scale, 1.0)
        else:
            zoom_factors = (scale, scale)

        order = min(interpolation.value, 3)
        if order != interpolation.value:
            logger.info(f"CPU mode: interpolation order clamped from {interpolation.value} to {order}")

        result = cpu_zoom(img_float, zoom_factors, order=order, mode='reflect')

        if sharpen != SharpenLevel.NONE:
            kernel = cls.SHARPEN_KERNELS[sharpen]
            if result.ndim == 3:
                for c in range(result.shape[2]):
                    result[:, :, c] = cpu_convolve(result[:, :, c], kernel, mode='reflect')
            else:
                result = cpu_convolve(result, kernel, mode='reflect')

        return np.clip(result, 0, 255).astype(np.uint8)


# ===========================================================================
# Video Processing
# ===========================================================================

class VideoProcessor:
    """Video frame extraction and assembly using ffmpeg."""

    SUPPORTED_FORMATS = ['.mp4', '.mkv', '.webm', '.avi', '.mov']

    @staticmethod
    def check_ffmpeg() -> bool:
        """Check if ffmpeg and ffprobe are available."""
        ffmpeg_ok = shutil.which('ffmpeg') is not None
        ffprobe_ok = shutil.which('ffprobe') is not None
        return ffmpeg_ok and ffprobe_ok

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """Get video metadata using ffprobe."""
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except Exception as e:
            raise RuntimeError(f"Failed to get video info: {e}")

    @staticmethod
    def extract_frames(
        video_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[int, float]:
        """Extract frames from video with error checking."""
        info = VideoProcessor.get_video_info(video_path)

        video_stream = None
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break

        if not video_stream:
            raise RuntimeError("No video stream found")

        fps_str = video_stream.get('r_frame_rate', '30/1')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 30.0
        else:
            fps = float(fps_str)

        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
        codec = video_stream.get('codec_name', 'unknown')
        logger.info(f"Video: {width}x{height} @ {fps:.2f}fps, codec={codec}")

        frame_pattern = os.path.join(output_dir, '%06d.png')
        cmd = [
            'ffmpeg', '-i', video_path,
            '-qscale:v', '2',
            '-vsync', '0',
            frame_pattern,
            '-y'
        ]

        logger.debug(f"Extracting frames: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = process.communicate()

        if process.returncode != 0:
            error_text = stderr.decode('utf-8', errors='replace')[:500] if stderr else 'Unknown error'
            raise RuntimeError(f"ffmpeg frame extraction failed (exit {process.returncode}): {error_text}")

        actual_frames = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
        logger.info(f"Extracted {actual_frames} frames to {output_dir}")

        if actual_frames == 0:
            raise RuntimeError("No frames extracted from video. The file may be corrupted or empty.")

        return actual_frames, fps

    @staticmethod
    def assemble_video(
        frames_dir: str,
        output_path: str,
        fps: float,
        original_video: str,
        codec: str = 'libx264',
        crf: int = 18
    ) -> None:
        """Assemble frames back into video with audio from original."""
        frame_pattern = os.path.join(frames_dir, '%06d.png')

        # Verify frames exist
        frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        if frame_count == 0:
            raise RuntimeError("No frames found to assemble into video")

        logger.info(f"Assembling {frame_count} frames into video at {fps:.2f}fps")

        info = VideoProcessor.get_video_info(original_video)
        has_audio = any(
            s.get('codec_type') == 'audio'
            for s in info.get('streams', [])
        )

        cmd = [
            'ffmpeg',
            '-framerate', str(fps),
            '-i', frame_pattern,
        ]

        if has_audio:
            cmd.extend(['-i', original_video])

        ext = os.path.splitext(output_path)[1].lower()

        if ext == '.webm':
            cmd.extend(['-c:v', 'libvpx-vp9', '-crf', str(crf), '-b:v', '0'])
        elif ext == '.mkv':
            cmd.extend(['-c:v', codec, '-crf', str(crf), '-preset', 'medium'])
        else:
            cmd.extend(['-c:v', codec, '-crf', str(crf), '-pix_fmt', 'yuv420p'])

        if has_audio:
            cmd.extend(['-c:a', 'copy', '-map', '0:v', '-map', '1:a?'])

        cmd.extend(['-y', output_path])
        logger.debug(f"Assembly command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Video assembled: {output_path}")


# ===========================================================================
# GIF Processing with Gifsicle
# ===========================================================================

class GIFProcessor:
    """GIF frame extraction and assembly with gifsicle optimization."""

    DITHER_METHODS = ['sierra2_4a', 'bayer', 'floyd_steinberg', 'sierra2', 'none']

    _gifsicle_available: Optional[bool] = None

    @classmethod
    def check_gifsicle(cls) -> bool:
        """Check if gifsicle is available (cached)."""
        if cls._gifsicle_available is None:
            cls._gifsicle_available = shutil.which('gifsicle') is not None
        return cls._gifsicle_available

    @classmethod
    def refresh_gifsicle_check(cls) -> bool:
        """Force re-check of gifsicle availability."""
        cls._gifsicle_available = None
        return cls.check_gifsicle()

    @staticmethod
    def extract_frames(gif_path: str) -> Tuple[List[np.ndarray], List[int], int]:
        """Extract frames from GIF with proper resource management."""
        with Image.open(gif_path) as gif:
            frames = []
            durations = []

            try:
                loop = gif.info.get('loop', 0)
            except Exception:
                loop = 0

            for frame in ImageSequence.Iterator(gif):
                frame_rgba = frame.convert('RGBA')
                frames.append(np.array(frame_rgba))
                durations.append(frame.info.get('duration', 100))

        if not frames:
            raise RuntimeError("No frames found in GIF. The file may be corrupted or empty.")

        logger.info(f"Extracted {len(frames)} GIF frames, size={frames[0].shape[1]}x{frames[0].shape[0]}")
        return frames, durations, loop

    @staticmethod
    def assemble_gif(
        frames: List[np.ndarray],
        durations: List[int],
        loop: int,
        output_path: str,
        gif_settings: Optional[GifSettings] = None
    ) -> None:
        """Assemble frames back into GIF with optional gifsicle optimization."""
        if not frames:
            raise ValueError("No frames to assemble into GIF")

        settings = gif_settings or GifSettings()
        logger.info(
            f"Assembling GIF: {len(frames)} frames, "
            f"{frames[0].shape[1]}x{frames[0].shape[0]}, "
            f"colors={settings.colors}"
        )

        pil_frames = [Image.fromarray(f) for f in frames]

        # Convert to palette mode
        pil_frames_p = []
        for frame in pil_frames:
            frame_p = frame.convert('P', palette=Image.ADAPTIVE, colors=settings.colors)
            pil_frames_p.append(frame_p)

        # Save initial GIF
        use_gifsicle = GIFProcessor.check_gifsicle()
        temp_gif = None

        try:
            if use_gifsicle:
                temp_gif = output_path + '.temp.gif'
                target = temp_gif
            else:
                target = output_path

            pil_frames_p[0].save(
                target,
                save_all=True,
                append_images=pil_frames_p[1:],
                duration=durations,
                loop=loop,
                optimize=True
            )

            # Optimize with gifsicle if available
            if use_gifsicle and temp_gif:
                GIFProcessor.optimize_with_gifsicle(temp_gif, output_path, settings)
        finally:
            # Always clean up temp file
            if temp_gif and os.path.exists(temp_gif):
                try:
                    os.unlink(temp_gif)
                except OSError:
                    pass

        if os.path.exists(output_path):
            size_kb = os.path.getsize(output_path) / 1024
            logger.info(f"GIF saved: {output_path} ({size_kb:.1f} KB)")

    @staticmethod
    def optimize_with_gifsicle(input_path: str, output_path: str,
                                settings: GifSettings) -> None:
        """Optimize GIF using gifsicle (ezgif-style optimization)."""
        cmd = ['gifsicle']

        # Optimization level (1-3)
        opt_level = max(1, min(3, settings.optimize_level))
        cmd.append(f'-O{opt_level}')

        # Lossy compression (0-200, 0=lossless)
        lossy = max(0, min(200, settings.lossy))
        if lossy > 0:
            cmd.append(f'--lossy={lossy}')

        # Color reduction
        if settings.colors < 256:
            cmd.append(f'--colors={settings.colors}')

        cmd.extend(['-o', output_path, input_path])

        logger.debug(f"gifsicle: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info("gifsicle optimization applied")
        except subprocess.CalledProcessError:
            logger.warning("gifsicle failed, using unoptimized GIF")
            shutil.copy(input_path, output_path)

    @staticmethod
    def video_to_gif(
        video_path: str,
        output_path: str,
        fps: int = 15,
        width: int = -1,
        start: float = 0.0,
        end: float = 0.0,
        gif_settings: Optional[GifSettings] = None,
        on_progress: Optional[Callable[[str], None]] = None
    ) -> None:
        """Convert video to GIF using ffmpeg palettegen (two-pass)."""
        settings = gif_settings or GifSettings()
        palette_path = None

        try:
            palette_path = tempfile.mktemp(suffix='.palette.png')

            # Build filter chain
            filters = []
            fps = max(1, fps)
            filters.append(f"fps={fps}")
            if width > 0:
                filters.append(f"scale={width}:-1:flags=lanczos")

            filter_str = ",".join(filters) if filters else "null"

            # Time range args
            time_args = []
            if start > 0:
                time_args += ["-ss", f"{start:.3f}"]
            if end > 0 and end > start:
                time_args += ["-to", f"{end:.3f}"]

            if on_progress:
                on_progress("Pass 1/2: Generating palette...")

            # Pass 1: palette generation
            cmd1 = (
                ["ffmpeg", "-y"] + time_args +
                ["-i", video_path,
                 "-vf", f"{filter_str},palettegen=max_colors={settings.colors}:stats_mode=diff",
                 "-update", "1", palette_path]
            )
            subprocess.run(cmd1, capture_output=True, check=True, timeout=120)

            if on_progress:
                on_progress("Pass 2/2: Encoding GIF...")

            # Pass 2: apply palette
            dither = settings.dither
            cmd2 = (
                ["ffmpeg", "-y"] + time_args +
                ["-i", video_path,
                 "-i", palette_path,
                 "-lavfi", f"{filter_str} [x]; [x][1:v] paletteuse=dither={dither}",
                 output_path]
            )
            subprocess.run(cmd2, capture_output=True, check=True, timeout=300)

            # Optimize with gifsicle
            if GIFProcessor.check_gifsicle() and settings.optimize_level > 0:
                if on_progress:
                    on_progress("Optimizing with gifsicle...")
                temp_output = None
                try:
                    temp_output = output_path + '.opt.gif'
                    GIFProcessor.optimize_with_gifsicle(output_path, temp_output, settings)
                    shutil.move(temp_output, output_path)
                finally:
                    if temp_output and os.path.exists(temp_output):
                        try:
                            os.unlink(temp_output)
                        except OSError:
                            pass

        finally:
            if palette_path and os.path.exists(palette_path):
                try:
                    os.remove(palette_path)
                except OSError:
                    pass


# ===========================================================================
# Theme
# ===========================================================================

class DarkTheme:
    """Standard dark-blue theme — readable, no conflicting colors."""
    # Backgrounds: darkest -> lightest
    BG_DARK = '#0d1117'       # Main background (GitHub dark)
    BG_MEDIUM = '#161b22'     # Panels, cards
    BG_LIGHT = '#21262d'      # Buttons, inputs, interactive elements
    BG_WIDGET = '#30363d'     # Input fields, spinboxes

    # Foregrounds: all tested for WCAG AA contrast against BG_DARK
    FG_PRIMARY = '#e6edf3'    # Main text (high contrast)
    FG_SECONDARY = '#8b949e'  # Muted labels, info text
    FG_VALUE = '#79c0ff'      # Value readouts (blue, readable on dark)

    # Accents
    ACCENT = '#1f6feb'        # Primary action buttons (blue)
    ACCENT_HOVER = '#388bfd'  # Hover state
    SUCCESS = '#3fb950'       # Success messages, available status
    WARNING = '#d29922'       # Warnings
    ERROR = '#f85149'         # Errors, cancel
    BORDER = '#30363d'        # Borders, separators

    # Log colors (all tested against BG_MEDIUM)
    LOG_INFO = '#3fb950'
    LOG_DEBUG = '#8b949e'
    LOG_WARN = '#d29922'
    LOG_ERROR = '#f85149'
    LOG_HEADER = '#79c0ff'


# ===========================================================================
# Main Application
# ===========================================================================

class UpscalerApp:
    """Main application class."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CUDA Media Upscaler")
        self.root.geometry("1200x900")
        self.root.minsize(1000, 800)

        # State variables
        self.current_file: Optional[str] = None
        self.current_media_type: str = "image"
        self.preview_image: Optional[ImageTk.PhotoImage] = None
        self._last_preview_img: Optional[Image.Image] = None
        self.original_size: Optional[Tuple[int, int]] = None

        # Thread safety
        self._processing_lock = threading.Lock()
        self._processing = False
        self._cancel_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Track file selector buttons to disable during processing
        self._file_selector_buttons: List[tk.Button] = []

        # Settings
        self.settings = UpscaleSettings()
        self.gif_settings = GifSettings()

        # Style the native Combobox popdown listbox (bypasses ttk, must use option_add)
        self.root.option_add('*TCombobox*Listbox.background', DarkTheme.BG_MEDIUM)
        self.root.option_add('*TCombobox*Listbox.foreground', DarkTheme.FG_PRIMARY)
        self.root.option_add('*TCombobox*Listbox.selectBackground', DarkTheme.ACCENT)
        self.root.option_add('*TCombobox*Listbox.selectForeground', DarkTheme.FG_PRIMARY)
        self.root.option_add('*TCombobox*Listbox.relief', 'flat')
        self.root.option_add('*TCombobox*Listbox.borderWidth', '1')

        # Apply dark theme
        self._setup_styles()
        self._create_ui()
        self._update_gpu_info()

        # Clean shutdown handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    @property
    def processing(self) -> bool:
        return self._processing

    @processing.setter
    def processing(self, value: bool):
        self._processing = value

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def _on_close(self):
        """Handle window close - clean up threads."""
        if self.processing:
            self._cancel_event.set()
            if self._worker_thread and self._worker_thread.is_alive():
                logger.info("Waiting for worker thread to finish...")
                self._worker_thread.join(timeout=3.0)
        self.root.destroy()

    def _setup_styles(self):
        """Configure ttk styles for dark theme."""
        self.root.configure(bg=DarkTheme.BG_DARK)

        style = ttk.Style()
        style.theme_use('clam')

        style.configure('Dark.TFrame', background=DarkTheme.BG_DARK)
        style.configure('Dark.TLabel', background=DarkTheme.BG_DARK,
                       foreground=DarkTheme.FG_PRIMARY, font=('Segoe UI', 10))
        style.configure('DarkTitle.TLabel', background=DarkTheme.BG_DARK,
                       foreground=DarkTheme.FG_PRIMARY, font=('Segoe UI', 14, 'bold'))
        style.configure('DarkInfo.TLabel', background=DarkTheme.BG_DARK,
                       foreground=DarkTheme.FG_SECONDARY, font=('Segoe UI', 9))
        style.configure('DarkValue.TLabel', background=DarkTheme.BG_DARK,
                       foreground=DarkTheme.FG_VALUE, font=('Segoe UI', 10, 'bold'))
        style.configure('Dark.TButton', background=DarkTheme.ACCENT,
                       foreground=DarkTheme.FG_PRIMARY, font=('Segoe UI', 10, 'bold'),
                       padding=(20, 10))
        style.map('Dark.TButton', background=[('active', DarkTheme.ACCENT_HOVER)])
        style.configure('Dark.TNotebook', background=DarkTheme.BG_DARK, borderwidth=0)
        style.configure('Dark.TNotebook.Tab', background=DarkTheme.BG_MEDIUM,
                       foreground=DarkTheme.FG_PRIMARY, padding=(20, 10),
                       font=('Segoe UI', 10, 'bold'))
        style.map('Dark.TNotebook.Tab',
                 background=[('selected', DarkTheme.ACCENT)],
                 foreground=[('selected', DarkTheme.FG_PRIMARY)])
        style.configure('Dark.Horizontal.TProgressbar', background=DarkTheme.ACCENT,
                       troughcolor=DarkTheme.BG_MEDIUM, borderwidth=0, thickness=20)
        style.configure('Dark.TScale', background=DarkTheme.BG_DARK,
                       troughcolor=DarkTheme.BG_MEDIUM, sliderlength=20,
                       borderwidth=0)
        style.configure('Dark.TCombobox', fieldbackground=DarkTheme.BG_MEDIUM,
                       background=DarkTheme.BG_MEDIUM, foreground=DarkTheme.FG_PRIMARY,
                       arrowcolor=DarkTheme.FG_PRIMARY,
                       selectbackground=DarkTheme.BG_MEDIUM,
                       selectforeground=DarkTheme.FG_PRIMARY,
                       insertcolor=DarkTheme.FG_PRIMARY)
        style.map('Dark.TCombobox',
                  fieldbackground=[('readonly', DarkTheme.BG_MEDIUM),
                                   ('disabled', DarkTheme.BG_DARK),
                                   ('focus', DarkTheme.BG_MEDIUM),
                                   ('active', DarkTheme.BG_MEDIUM)],
                  foreground=[('readonly', DarkTheme.FG_PRIMARY),
                              ('disabled', DarkTheme.FG_SECONDARY),
                              ('focus', DarkTheme.FG_PRIMARY)],
                  background=[('readonly', DarkTheme.BG_MEDIUM),
                              ('active', DarkTheme.BG_LIGHT)],
                  arrowcolor=[('disabled', DarkTheme.FG_SECONDARY),
                              ('active', DarkTheme.FG_PRIMARY)])
        style.configure('Dark.TRadiobutton', background=DarkTheme.BG_DARK,
                       foreground=DarkTheme.FG_PRIMARY, font=('Segoe UI', 10))
        style.map('Dark.TRadiobutton', background=[('active', DarkTheme.BG_DARK)])

    def _create_ui(self):
        """Create the main UI layout."""
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Left panel - Controls
        left_panel = ttk.Frame(main_frame, style='Dark.TFrame', width=400)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)

        # Right panel - Preview + Log
        right_panel = ttk.Frame(main_frame, style='Dark.TFrame')
        right_panel.pack(side='right', fill='both', expand=True)

        self._create_controls(left_panel)
        self._create_preview_and_log(right_panel)

    def _create_controls(self, parent: ttk.Frame):
        """Create the control panel."""
        # Title
        title_label = ttk.Label(parent, text="CUDA Media Upscaler", style='DarkTitle.TLabel')
        title_label.pack(pady=(0, 5))

        # GPU Info
        self.gpu_label = ttk.Label(parent, text="", style='DarkInfo.TLabel')
        self.gpu_label.pack(pady=(0, 10))

        # Tab control
        notebook = ttk.Notebook(parent, style='Dark.TNotebook')
        notebook.pack(fill='x', pady=(0, 10))

        # Create tabs
        image_tab = ttk.Frame(notebook, style='Dark.TFrame')
        video_tab = ttk.Frame(notebook, style='Dark.TFrame')
        gif_tab = ttk.Frame(notebook, style='Dark.TFrame')
        v2g_tab = ttk.Frame(notebook, style='Dark.TFrame')  # Video to GIF

        notebook.add(image_tab, text=' Image ')
        notebook.add(video_tab, text=' Video ')
        notebook.add(gif_tab, text='  GIF  ')
        notebook.add(v2g_tab, text='Video->GIF')

        # Tab content
        self._create_file_selector(image_tab, "image")
        self._create_file_selector(video_tab, "video")
        self._create_file_selector(gif_tab, "gif")
        self._create_v2g_panel(v2g_tab)

        # Common settings
        self._create_settings_panel(parent)

        # GIF settings (for GIF and Video->GIF modes)
        self._create_gif_settings_panel(parent)

        # Progress section
        self._create_progress_section(parent)

        # Action buttons
        self._create_action_buttons(parent)

    def _create_file_selector(self, parent: ttk.Frame, media_type: str):
        """Create file selector for each media type."""
        frame = ttk.Frame(parent, style='Dark.TFrame')
        frame.pack(fill='x', pady=10, padx=10)

        if media_type == "image":
            filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp")]
            btn_text = "Select Image"
        elif media_type == "video":
            filetypes = [("Video files", "*.mp4 *.mkv *.webm *.avi *.mov")]
            btn_text = "Select Video"
        else:  # gif
            filetypes = [("GIF files", "*.gif")]
            btn_text = "Select GIF"

        btn = tk.Button(
            frame, text=btn_text, bg=DarkTheme.BG_LIGHT, fg=DarkTheme.FG_PRIMARY,
            font=('Segoe UI', 10), relief='flat', cursor='hand2',
            command=lambda: self._select_file(filetypes, media_type)
        )
        btn.pack(fill='x', pady=5)
        self._file_selector_buttons.append(btn)

    def _create_v2g_panel(self, parent: ttk.Frame):
        """Create Video->GIF panel with FPS slider."""
        frame = ttk.Frame(parent, style='Dark.TFrame')
        frame.pack(fill='x', pady=10, padx=10)

        btn = tk.Button(
            frame, text="Select Video for GIF", bg=DarkTheme.BG_LIGHT,
            fg=DarkTheme.FG_PRIMARY, font=('Segoe UI', 10), relief='flat',
            cursor='hand2',
            command=lambda: self._select_file(
                [("Video files", "*.mp4 *.mkv *.webm *.avi *.mov")], "video2gif"
            )
        )
        btn.pack(fill='x', pady=5)
        self._file_selector_buttons.append(btn)

        # --- Output FPS with live value label ---
        fps_label_frame = ttk.Frame(frame, style='Dark.TFrame')
        fps_label_frame.pack(fill='x', pady=(10, 0))
        ttk.Label(fps_label_frame, text="Output FPS:", style='Dark.TLabel').pack(side='left')
        self.v2g_fps_value_label = ttk.Label(fps_label_frame, text="15", style='DarkValue.TLabel')
        self.v2g_fps_value_label.pack(side='right')

        self.v2g_fps_var = tk.IntVar(value=15)
        fps_scale = ttk.Scale(
            frame, from_=5, to=30, variable=self.v2g_fps_var,
            orient='horizontal',
            command=lambda v: self.v2g_fps_value_label.config(text=str(int(float(v))))
        )
        fps_scale.pack(fill='x', pady=2)

        # Width is always original — no pre-scale
        self.v2g_width_var = tk.IntVar(value=0)

    def _create_settings_panel(self, parent: ttk.Frame):
        """Create the settings panel with scale, downscale toggle, and resolution input."""
        settings_frame = ttk.Frame(parent, style='Dark.TFrame')
        settings_frame.pack(fill='x', pady=10)

        # --- Downscale checkbox + Scale factor header ---
        mode_header = ttk.Frame(settings_frame, style='Dark.TFrame')
        mode_header.pack(fill='x', pady=(5, 0))

        self.downscale_var = tk.BooleanVar(value=False)
        downscale_cb = tk.Checkbutton(
            mode_header, text="Invert (Downscale)",
            variable=self.downscale_var,
            bg=DarkTheme.BG_DARK, fg=DarkTheme.FG_PRIMARY,
            selectcolor=DarkTheme.BG_WIDGET, activebackground=DarkTheme.BG_DARK,
            activeforeground=DarkTheme.FG_PRIMARY, font=('Segoe UI', 9),
            command=self._on_scale_mode_change
        )
        downscale_cb.pack(side='left')

        self.scale_value_label = ttk.Label(mode_header, text="2.0x Upscale", style='DarkValue.TLabel')
        self.scale_value_label.pack(side='right')

        # Scale slider
        self.scale_var = tk.DoubleVar(value=2.0)
        self.scale_slider = ttk.Scale(settings_frame, from_=1.0, to=8.0, variable=self.scale_var,
                                      orient='horizontal',
                                      command=self._on_scale_change)
        self.scale_slider.pack(fill='x', pady=(2, 5))

        # --- Target resolution input ---
        res_input_frame = ttk.Frame(settings_frame, style='Dark.TFrame')
        res_input_frame.pack(fill='x', pady=(0, 5))
        ttk.Label(res_input_frame, text="Target W:", style='DarkInfo.TLabel').pack(side='left')
        self.target_w_var = tk.StringVar(value="")
        target_w_entry = tk.Entry(
            res_input_frame, textvariable=self.target_w_var, width=6,
            bg=DarkTheme.BG_WIDGET, fg=DarkTheme.FG_PRIMARY,
            insertbackground=DarkTheme.FG_PRIMARY, relief='flat', font=('Segoe UI', 9)
        )
        target_w_entry.pack(side='left', padx=(5, 10))

        ttk.Label(res_input_frame, text="H:", style='DarkInfo.TLabel').pack(side='left')
        self.target_h_var = tk.StringVar(value="")
        target_h_entry = tk.Entry(
            res_input_frame, textvariable=self.target_h_var, width=6,
            bg=DarkTheme.BG_WIDGET, fg=DarkTheme.FG_PRIMARY,
            insertbackground=DarkTheme.FG_PRIMARY, relief='flat', font=('Segoe UI', 9)
        )
        target_h_entry.pack(side='left', padx=(5, 5))

        apply_res_btn = tk.Button(
            res_input_frame, text="Apply", bg=DarkTheme.BG_LIGHT,
            fg=DarkTheme.FG_PRIMARY, font=('Segoe UI', 8), relief='flat',
            command=self._apply_target_resolution
        )
        apply_res_btn.pack(side='left', padx=(5, 0))

        # Resolution info
        self.resolution_label = ttk.Label(settings_frame, text="Output resolution: --",
                                         style='DarkInfo.TLabel')
        self.resolution_label.pack(anchor='w', pady=(2, 5))

        # Interpolation method
        ttk.Label(settings_frame, text="Interpolation", style='Dark.TLabel').pack(anchor='w', pady=(5, 0))
        self.interp_var = tk.StringVar(value="Bicubic")
        interp_combo = ttk.Combobox(settings_frame, textvariable=self.interp_var,
                                   values=["Nearest", "Bilinear", "Bicubic", "Quintic"],
                                   state='readonly', style='Dark.TCombobox')
        interp_combo.pack(fill='x', pady=3)

        # Sharpening
        ttk.Label(settings_frame, text="Sharpening", style='Dark.TLabel').pack(anchor='w', pady=(5, 0))
        self.sharpen_var = tk.StringVar(value="none")
        sharpen_frame = ttk.Frame(settings_frame, style='Dark.TFrame')
        sharpen_frame.pack(fill='x', pady=3)

        for text, value in [("None", "none"), ("Light", "light"), ("Medium", "medium"), ("Strong", "strong")]:
            rb = ttk.Radiobutton(sharpen_frame, text=text, value=value,
                                variable=self.sharpen_var, style='Dark.TRadiobutton')
            rb.pack(side='left', padx=(0, 10))

        # Output format
        ttk.Label(settings_frame, text="Output Format", style='Dark.TLabel').pack(anchor='w', pady=(5, 0))
        self.format_var = tk.StringVar(value="MP4")
        format_combo = ttk.Combobox(settings_frame, textvariable=self.format_var,
                                   values=["MP4", "MKV", "WebM", "AVI", "MOV"],
                                   state='readonly', style='Dark.TCombobox')
        format_combo.pack(fill='x', pady=3)

    def _create_gif_settings_panel(self, parent: ttk.Frame):
        """Create GIF-specific settings panel."""
        gif_frame = ttk.Frame(parent, style='Dark.TFrame')
        gif_frame.pack(fill='x', pady=5)

        ttk.Label(gif_frame, text="GIF Settings", style='Dark.TLabel').pack(anchor='w')

        # Colors
        colors_frame = ttk.Frame(gif_frame, style='Dark.TFrame')
        colors_frame.pack(fill='x', pady=2)
        ttk.Label(colors_frame, text="Colors:", style='DarkInfo.TLabel').pack(side='left')
        self.gif_colors_var = tk.IntVar(value=256)
        colors_combo = ttk.Combobox(colors_frame, textvariable=self.gif_colors_var,
                                   values=[16, 32, 64, 128, 256], width=8,
                                   state='readonly', style='Dark.TCombobox')
        colors_combo.pack(side='right')

        # Lossy compression (gifsicle)
        lossy_frame = ttk.Frame(gif_frame, style='Dark.TFrame')
        lossy_frame.pack(fill='x', pady=2)
        ttk.Label(lossy_frame, text="Lossy (0-200):", style='DarkInfo.TLabel').pack(side='left')
        self.gif_lossy_var = tk.IntVar(value=0)
        lossy_spin = tk.Spinbox(lossy_frame, from_=0, to=200, textvariable=self.gif_lossy_var,
                               width=5, bg=DarkTheme.BG_WIDGET, fg=DarkTheme.FG_PRIMARY,
                               buttonbackground=DarkTheme.BG_LIGHT,
                               insertbackground=DarkTheme.FG_PRIMARY, relief='flat')
        lossy_spin.pack(side='right')

        # Gifsicle status
        GIFProcessor.refresh_gifsicle_check()
        if GIFProcessor.check_gifsicle():
            status_text = "gifsicle: Available"
        else:
            status_text = "gifsicle: Not found (sudo apt install gifsicle)"

        self.gifsicle_label = ttk.Label(gif_frame, text=status_text, style='DarkInfo.TLabel')
        self.gifsicle_label.pack(anchor='w', pady=(5, 0))

    def _create_progress_section(self, parent: ttk.Frame):
        """Create progress section."""
        progress_frame = ttk.Frame(parent, style='Dark.TFrame')
        progress_frame.pack(fill='x', pady=10)

        self.progress_label = ttk.Label(progress_frame, text="Ready", style='DarkInfo.TLabel')
        self.progress_label.pack(anchor='w', pady=(0, 5))

        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate',
                                           style='Dark.Horizontal.TProgressbar')
        self.progress_bar.pack(fill='x', pady=5)

        self.frame_counter_label = ttk.Label(progress_frame, text="", style='DarkInfo.TLabel')
        self.frame_counter_label.pack(anchor='w')

    def _create_action_buttons(self, parent: ttk.Frame):
        """Create action buttons."""
        btn_frame = ttk.Frame(parent, style='Dark.TFrame')
        btn_frame.pack(fill='x', pady=10)

        self.process_btn = tk.Button(
            btn_frame, text="UPSCALE", bg=DarkTheme.ACCENT, fg='#ffffff',
            font=('Segoe UI', 12, 'bold'), relief='flat', cursor='hand2',
            activebackground=DarkTheme.ACCENT_HOVER, activeforeground='#ffffff',
            command=self._start_processing
        )
        self.process_btn.pack(fill='x', pady=5, ipady=10)

        self.cancel_btn = tk.Button(
            btn_frame, text="Cancel", bg=DarkTheme.BG_LIGHT, fg=DarkTheme.FG_PRIMARY,
            font=('Segoe UI', 10), relief='flat', cursor='hand2', state='disabled',
            activebackground=DarkTheme.ERROR, activeforeground='#ffffff',
            command=self._cancel_processing
        )
        self.cancel_btn.pack(fill='x', pady=5)

    def _create_preview_and_log(self, parent: ttk.Frame):
        """Create preview panel and operation log."""
        # Preview area (top)
        preview_label = ttk.Label(parent, text="Preview", style='DarkTitle.TLabel')
        preview_label.pack(anchor='w', pady=(0, 5))

        canvas_frame = tk.Frame(parent, bg=DarkTheme.BG_MEDIUM, relief='flat', bd=2)
        canvas_frame.pack(fill='both', expand=True)

        self.preview_canvas = tk.Canvas(canvas_frame, bg=DarkTheme.BG_MEDIUM,
                                       highlightthickness=0)
        self.preview_canvas.pack(fill='both', expand=True, padx=2, pady=2)

        self.preview_canvas.create_text(
            200, 200, text="No file selected",
            fill=DarkTheme.FG_SECONDARY,
            font=('Segoe UI', 13), tags='placeholder'
        )

        self.preview_canvas.bind('<Configure>', self._on_canvas_resize)

        # Operation Log (bottom)
        log_label = ttk.Label(parent, text="Operation Log", style='DarkTitle.TLabel')
        log_label.pack(anchor='w', pady=(10, 5))

        log_frame = tk.Frame(parent, bg=DarkTheme.BG_MEDIUM, relief='flat', bd=2, height=180)
        log_frame.pack(fill='x', pady=(0, 5))
        log_frame.pack_propagate(False)

        self.log_text = tk.Text(
            log_frame, bg=DarkTheme.BG_MEDIUM, fg=DarkTheme.FG_SECONDARY,
            font=('Consolas', 9), wrap='word', state='disabled',
            borderwidth=0, highlightthickness=0,
            insertbackground=DarkTheme.FG_PRIMARY
        )
        log_scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        log_scrollbar.pack(side='right', fill='y')
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)

        # Tag colors for log levels
        self.log_text.tag_configure('INFO', foreground=DarkTheme.LOG_INFO)
        self.log_text.tag_configure('DEBUG', foreground=DarkTheme.LOG_DEBUG)
        self.log_text.tag_configure('WARNING', foreground=DarkTheme.LOG_WARN)
        self.log_text.tag_configure('ERROR', foreground=DarkTheme.LOG_ERROR)
        self.log_text.tag_configure('HEADER', foreground=DarkTheme.LOG_HEADER, font=('Consolas', 9, 'bold'))

        # Install custom logging handler that writes to the log widget
        self._log_handler = _TkLogHandler(self)
        self._log_handler.setLevel(logging.DEBUG)
        self._log_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
        logging.getLogger().addHandler(self._log_handler)

        # Initial log message
        self._append_log("CUDA Media Upscaler initialized", 'HEADER')
        self._append_log(f"CUDA: {'Available' if CUDA_AVAILABLE else 'Not available'}", 'INFO' if CUDA_AVAILABLE else 'WARNING')
        self._append_log(f"SciPy: {'Available' if SCIPY_AVAILABLE else 'Not available'}", 'INFO' if SCIPY_AVAILABLE else 'WARNING')
        self._append_log(f"gifsicle: {'Available' if GIFProcessor.check_gifsicle() else 'Not found (optional)'}", 'INFO' if GIFProcessor.check_gifsicle() else 'WARNING')

    def _append_log(self, message: str, level: str = 'INFO'):
        """Append a message to the log widget (thread-safe)."""
        def _do_append():
            try:
                self.log_text.config(state='normal')
                self.log_text.insert('end', message + '\n', level)
                self.log_text.see('end')
                self.log_text.config(state='disabled')
            except Exception:
                pass
        self._safe_after(_do_append)

    def _update_gpu_info(self):
        """Update GPU information display."""
        gpu_info = CUDAUpscaler.get_gpu_info()
        if CUDA_AVAILABLE:
            free, total = CUDAUpscaler.get_free_memory()
            self.gpu_label.config(text=f"GPU: {gpu_info}\nMemory: {free:.1f}/{total:.1f} GB free")
        else:
            self.gpu_label.config(text=f"GPU: {gpu_info}")

    def _on_scale_change(self, value):
        """Handle scale slider change."""
        scale = round(float(value), 1)
        self.scale_var.set(scale)
        is_down = self.downscale_var.get()
        mode_text = "Downscale" if is_down else "Upscale"
        self.scale_value_label.config(text=f"{scale}x {mode_text}")
        self._update_resolution_estimate()

    def _on_scale_mode_change(self):
        """Handle downscale checkbox toggle."""
        is_down = self.downscale_var.get()
        scale = self.scale_var.get()
        mode_text = "Downscale" if is_down else "Upscale"
        self.scale_value_label.config(text=f"{scale}x {mode_text}")
        # Update the button text
        self.process_btn.config(text="DOWNSCALE" if is_down else "UPSCALE")
        self._update_resolution_estimate()

    def _apply_target_resolution(self):
        """Calculate scale factor from target resolution input."""
        if not self.original_size:
            messagebox.showinfo("Info", "Select a file first to set target resolution.")
            return

        w_str = self.target_w_var.get().strip()
        h_str = self.target_h_var.get().strip()

        if not w_str and not h_str:
            return

        orig_w, orig_h = self.original_size

        try:
            if w_str and h_str:
                target_w = int(w_str)
                # Use width to determine scale
                scale = target_w / orig_w
            elif w_str:
                target_w = int(w_str)
                scale = target_w / orig_w
            else:
                target_h = int(h_str)
                scale = target_h / orig_h

            scale = round(abs(scale), 1)
            if scale < 0.1:
                scale = 0.1
            if scale > 8.0:
                scale = 8.0

            # Determine if downscale
            if scale < 1.0:
                self.downscale_var.set(True)
                self.scale_var.set(round(1.0 / scale, 1))
            else:
                self.downscale_var.set(False)
                self.scale_var.set(scale)

            self._on_scale_mode_change()
            logger.info(f"Target resolution applied: scale={self.scale_var.get()}x {'down' if self.downscale_var.get() else 'up'}")
        except ValueError:
            messagebox.showerror("Error", "Enter valid integer resolution values.")

    def _on_canvas_resize(self, event):
        """Handle canvas resize."""
        if self._last_preview_img:
            self._display_preview()

    def _validate_input_file(self, filepath: str) -> bool:
        """Validate the selected input file."""
        if not os.path.exists(filepath):
            messagebox.showerror("Error", f"File does not exist: {filepath}")
            return False
        if not os.path.isfile(filepath):
            messagebox.showerror("Error", f"Not a regular file: {filepath}")
            return False
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            messagebox.showerror("Error", "File is empty (0 bytes)")
            return False
        # Warn on very large files (>2GB)
        if file_size > 2 * 1024**3:
            result = messagebox.askyesno(
                "Large File Warning",
                f"This file is {file_size / (1024**3):.1f} GB. Processing may be slow. Continue?"
            )
            return result
        return True

    def _select_file(self, filetypes: list, media_type: str):
        """Open file dialog and load file."""
        if self.processing:
            return

        filepath = NativeFileDialog.askopenfilename(
            title="Select File", filetypes=filetypes
        )
        if filepath:
            if not self._validate_input_file(filepath):
                return
            self.current_file = filepath
            self.current_media_type = media_type
            logger.info(f"Selected {media_type}: {filepath}")
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            self._append_log(f"File: {os.path.basename(filepath)} ({file_size:.1f} MB)", 'INFO')
            self._load_preview()
            self._update_resolution_estimate()

    def _load_preview(self):
        """Load preview for selected file."""
        if not self.current_file:
            return

        tmp_path = None
        try:
            if self.current_media_type == "image":
                img = Image.open(self.current_file)
                img.load()  # Force decode to catch truncated files early
                self.original_size = img.size
                self._append_log(f"Image: {img.size[0]}x{img.size[1]}, mode={img.mode}", 'DEBUG')
            elif self.current_media_type in ["video", "video2gif"]:
                info = VideoProcessor.get_video_info(self.current_file)
                for stream in info.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        self.original_size = (
                            int(stream.get('width', 0)),
                            int(stream.get('height', 0))
                        )
                        codec = stream.get('codec_name', '?')
                        self._append_log(
                            f"Video: {self.original_size[0]}x{self.original_size[1]}, codec={codec}",
                            'DEBUG'
                        )
                        break
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name
                subprocess.run([
                    'ffmpeg', '-i', self.current_file,
                    '-vf', 'select=eq(n\\,0)',
                    '-vframes', '1',
                    tmp_path, '-y'
                ], capture_output=True)
                img = Image.open(tmp_path)
                img.load()  # Force decode before unlinking
            else:  # gif
                img = Image.open(self.current_file)
                img.seek(0)
                self.original_size = img.size
                # Count frames
                n_frames = 0
                try:
                    while True:
                        n_frames += 1
                        img.seek(n_frames)
                except EOFError:
                    pass
                img.seek(0)
                self._append_log(f"GIF: {img.size[0]}x{img.size[1]}, {n_frames} frames", 'DEBUG')

            self._display_preview(img)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Preview load failed: {error_msg}")
            messagebox.showerror("Error", f"Failed to load preview: {error_msg}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _display_preview(self, img: Optional[Image.Image] = None):
        """Display image in preview canvas."""
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        if img is None and self._last_preview_img:
            img = self._last_preview_img
        elif img:
            self._last_preview_img = img.copy()
        else:
            return

        try:
            img_ratio = img.width / img.height
            canvas_ratio = canvas_width / canvas_height

            if img_ratio > canvas_ratio:
                new_width = min(canvas_width - 20, img.width)
                new_height = int(new_width / img_ratio)
            else:
                new_height = min(canvas_height - 20, img.height)
                new_width = int(new_height * img_ratio)

            if new_width < 1 or new_height < 1:
                return

            img_resized = img.resize((new_width, new_height), _LANCZOS)
            self.preview_image = ImageTk.PhotoImage(img_resized)

            self.preview_canvas.delete('all')
            self.preview_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=self.preview_image, anchor='center'
            )
        except Exception as e:
            logger.warning(f"Preview display error: {e}")

    def _get_effective_scale(self) -> float:
        """Get the effective scale factor accounting for downscale mode."""
        scale = self.scale_var.get()
        if self.downscale_var.get():
            return 1.0 / scale  # invert for downscale
        return scale

    def _update_resolution_estimate(self):
        """Update estimated output resolution."""
        if self.original_size:
            eff_scale = self._get_effective_scale()
            new_w = max(1, int(self.original_size[0] * eff_scale))
            new_h = max(1, int(self.original_size[1] * eff_scale))
            mode = "downscale" if self.downscale_var.get() else "upscale"
            self.resolution_label.config(
                text=f"Output: {new_w}x{new_h} ({mode} from {self.original_size[0]}x{self.original_size[1]})"
            )
            # Auto-populate target resolution fields
            self.target_w_var.set(str(new_w))
            self.target_h_var.set(str(new_h))

    def _get_settings(self) -> UpscaleSettings:
        """Get current settings from UI."""
        interp_map = {
            "Nearest": InterpolationMethod.NEAREST,
            "Bilinear": InterpolationMethod.BILINEAR,
            "Bicubic": InterpolationMethod.BICUBIC,
            "Quintic": InterpolationMethod.QUINTIC
        }
        sharpen_map = {
            "none": SharpenLevel.NONE,
            "light": SharpenLevel.LIGHT,
            "medium": SharpenLevel.MEDIUM,
            "strong": SharpenLevel.STRONG
        }

        return UpscaleSettings(
            scale=self._get_effective_scale(),
            interpolation=interp_map.get(self.interp_var.get(), InterpolationMethod.BICUBIC),
            sharpen=sharpen_map.get(self.sharpen_var.get(), SharpenLevel.NONE)
        )

    def _get_gif_settings(self) -> GifSettings:
        """Get GIF settings from UI."""
        return GifSettings(
            fps=max(1, self.v2g_fps_var.get() if hasattr(self, 'v2g_fps_var') else 15),
            colors=self.gif_colors_var.get(),
            lossy=max(0, min(200, self.gif_lossy_var.get())),
            optimize_level=3
        )

    def _set_ui_processing(self, active: bool):
        """Enable/disable UI controls based on processing state."""
        state = 'disabled' if active else 'normal'
        self.process_btn.config(state=state)
        self.cancel_btn.config(state='normal' if active else 'disabled')
        for btn in self._file_selector_buttons:
            btn.config(state=state)

    def _start_processing(self):
        """Start upscaling process."""
        if not self.current_file:
            messagebox.showwarning("Warning", "Please select a file first.")
            return

        with self._processing_lock:
            if self.processing:
                return
            self.processing = True

        # Capture all UI values on main thread before spawning worker
        v2g_fps = self.v2g_fps_var.get() if hasattr(self, 'v2g_fps_var') else 15
        v2g_width = self.v2g_width_var.get() if hasattr(self, 'v2g_width_var') else -1

        # Get output path using native dialog
        if self.current_media_type == "image":
            default_ext = os.path.splitext(self.current_file)[1]
            filetypes = [("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")]
        elif self.current_media_type == "video":
            format_ext = self.format_var.get().lower()
            default_ext = f".{format_ext}"
            filetypes = [(format_ext.upper(), f"*.{format_ext}")]
        elif self.current_media_type == "video2gif":
            default_ext = ".gif"
            filetypes = [("GIF", "*.gif")]
        else:  # gif
            default_ext = ".gif"
            filetypes = [("GIF", "*.gif")]

        output_path = NativeFileDialog.asksaveasfilename(
            title="Save Output",
            defaultextension=default_ext,
            filetypes=filetypes
        )

        if not output_path:
            with self._processing_lock:
                self.processing = False
            return

        # Start processing thread
        self._cancel_event.clear()
        self._set_ui_processing(True)

        settings = self._get_settings()
        gif_settings = self._get_gif_settings()

        # Log the operation header
        mode = "DOWNSCALE" if self.downscale_var.get() else "UPSCALE"
        self._append_log("=" * 50, 'HEADER')
        self._append_log(f"STARTING: {self.current_media_type.upper()} {mode}", 'HEADER')
        self._append_log(f"Input:  {os.path.basename(self.current_file)}", 'INFO')
        self._append_log(f"Output: {os.path.basename(output_path)}", 'INFO')
        self._append_log(f"Scale: {settings.scale:.2f}x  Interp: {settings.interpolation.name}  Sharpen: {settings.sharpen.value}", 'INFO')
        if self.original_size:
            out_w = max(1, int(self.original_size[0] * settings.scale))
            out_h = max(1, int(self.original_size[1] * settings.scale))
            self._append_log(f"Resolution: {self.original_size[0]}x{self.original_size[1]} -> {out_w}x{out_h}", 'INFO')
        if self.current_media_type == "video2gif":
            self._append_log(f"V2G: fps={v2g_fps}, pre-scale width={v2g_width if v2g_width > 0 else 'original'}", 'INFO')
        self._append_log("-" * 50, 'DEBUG')

        self._worker_thread = threading.Thread(
            target=self._process_media,
            args=(self.current_file, output_path, self.current_media_type,
                  settings, gif_settings, v2g_fps, v2g_width)
        )
        self._worker_thread.daemon = True
        self._worker_thread.start()

    def _process_media(
        self,
        input_path: str,
        output_path: str,
        media_type: str,
        settings: UpscaleSettings,
        gif_settings: GifSettings,
        v2g_fps: int = 15,
        v2g_width: int = -1
    ):
        """Process media file in background thread."""
        error_message = None
        t_start = time.perf_counter()
        try:
            if media_type == "image":
                self._process_image(input_path, output_path, settings)
            elif media_type == "video":
                self._process_video(input_path, output_path, settings)
            elif media_type == "video2gif":
                self._process_video_to_gif(
                    input_path, output_path, settings, gif_settings, v2g_fps, v2g_width
                )
            else:  # gif
                self._process_gif(input_path, output_path, settings, gif_settings)

            dt = time.perf_counter() - t_start

            if not self.cancel_requested:
                self._update_progress("Complete!", 100)
                # Log result
                if os.path.exists(output_path):
                    out_size = os.path.getsize(output_path) / (1024 * 1024)
                    logger.info(f"Done in {dt:.1f}s -- output: {out_size:.2f} MB")
                saved_path = output_path
                self._safe_after(lambda p=saved_path: messagebox.showinfo(
                    "Success", f"File saved to:\n{p}"
                ))
            else:
                # Clean up partial output on cancel
                if os.path.exists(output_path):
                    try:
                        os.unlink(output_path)
                        logger.info(f"Cleaned up partial output: {output_path}")
                    except OSError:
                        pass
                self._safe_after(lambda: messagebox.showinfo(
                    "Cancelled", "Processing was cancelled."
                ))

        except Exception as e:
            error_message = str(e)
            logger.error(f"Processing error: {error_message}")
            self._safe_after(lambda msg=error_message: messagebox.showerror(
                "Error", f"Processing failed: {msg}"
            ))
            # Clean up partial output on error
            if os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except OSError:
                    pass
        finally:
            self._safe_after(self._reset_ui)

    def _safe_after(self, callback):
        """Schedule a callback on the main thread, checking window still exists."""
        try:
            if self.root.winfo_exists():
                self.root.after(0, callback)
        except Exception:
            pass

    def _process_image(self, input_path: str, output_path: str, settings: UpscaleSettings):
        """Process single image."""
        self._update_progress("Loading image...", 10)
        img = Image.open(input_path)
        img.load()
        img_array = np.array(img)
        logger.info(f"Image loaded: {img_array.shape[1]}x{img_array.shape[0]}, channels={img_array.shape[2] if img_array.ndim == 3 else 1}")

        if self.cancel_requested:
            return

        self._update_progress("Upscaling...", 30)
        result_array = CUDAUpscaler.upscale(
            img_array, settings.scale, settings.interpolation, settings.sharpen
        )
        logger.info(f"Upscaled: {img_array.shape[1]}x{img_array.shape[0]} -> {result_array.shape[1]}x{result_array.shape[0]}")

        if self.cancel_requested:
            return

        self._update_progress("Saving...", 90)
        result_img = Image.fromarray(result_array)
        result_img.save(output_path, quality=95)
        logger.info(f"Saved: {output_path}")

    def _process_video(self, input_path: str, output_path: str, settings: UpscaleSettings):
        """Process video file frame by frame."""
        if not VideoProcessor.check_ffmpeg():
            raise RuntimeError("ffmpeg/ffprobe not found. Please install ffmpeg.")

        temp_dir = tempfile.mkdtemp(prefix='upscaler_')
        frames_in_dir = os.path.join(temp_dir, 'frames_in')
        frames_out_dir = os.path.join(temp_dir, 'frames_out')
        os.makedirs(frames_in_dir)
        os.makedirs(frames_out_dir)

        try:
            self._update_progress("Extracting frames...", 5)
            frame_count, fps = VideoProcessor.extract_frames(input_path, frames_in_dir)

            frame_files = sorted([f for f in os.listdir(frames_in_dir) if f.endswith('.png')])

            if not frame_files:
                raise RuntimeError("No frames extracted from video")

            logger.info(f"Processing {len(frame_files)} frames at {settings.scale}x scale")

            for i, frame_file in enumerate(frame_files):
                if self.cancel_requested:
                    return

                frame_path = os.path.join(frames_in_dir, frame_file)
                frame_img = Image.open(frame_path)
                frame_array = np.array(frame_img)

                result_array = CUDAUpscaler.upscale(
                    frame_array, settings.scale, settings.interpolation, settings.sharpen
                )

                output_frame_path = os.path.join(frames_out_dir, frame_file)
                Image.fromarray(result_array).save(output_frame_path)

                progress = 10 + int(80 * (i + 1) / len(frame_files))
                self._update_progress(
                    f"Frame {i+1}/{len(frame_files)} "
                    f"({frame_array.shape[1]}x{frame_array.shape[0]} -> {result_array.shape[1]}x{result_array.shape[0]})",
                    progress,
                    f"Frame: {i+1} / {len(frame_files)}"
                )

            if self.cancel_requested:
                return

            self._update_progress("Assembling video...", 95)
            VideoProcessor.assemble_video(frames_out_dir, output_path, fps, input_path)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _process_video_to_gif(
        self,
        input_path: str,
        output_path: str,
        settings: UpscaleSettings,
        gif_settings: GifSettings,
        v2g_fps: int = 15,
        v2g_width: int = -1
    ):
        """Process video to upscaled GIF -- disk-based frame pipeline to avoid OOM."""
        if not VideoProcessor.check_ffmpeg():
            raise RuntimeError("ffmpeg/ffprobe not found. Please install ffmpeg.")

        temp_dir = tempfile.mkdtemp(prefix='v2g_upscale_')
        frames_in_dir = os.path.join(temp_dir, 'frames_in')
        frames_out_dir = os.path.join(temp_dir, 'frames_out')
        os.makedirs(frames_in_dir)
        os.makedirs(frames_out_dir)

        try:
            # Extract frames at target FPS
            self._update_progress("Extracting frames...", 5)

            fps = max(1, v2g_fps)
            width = v2g_width

            # Build ffmpeg filter
            filters = [f"fps={fps}"]
            if width > 0:
                filters.append(f"scale={width}:-1:flags=lanczos")
                logger.info(f"V2G: extracting at {fps}fps, pre-scaling to {width}px width")
            else:
                logger.info(f"V2G: extracting at {fps}fps, keeping original resolution")

            filter_str = ",".join(filters)

            frame_pattern = os.path.join(frames_in_dir, '%06d.png')
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vf', filter_str,
                '-qscale:v', '2',
                frame_pattern, '-y'
            ]
            logger.debug(f"V2G extract: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True)

            frame_files = sorted([f for f in os.listdir(frames_in_dir) if f.endswith('.png')])

            if not frame_files:
                raise RuntimeError("No frames extracted from video")

            # Check first frame dimensions
            first_frame = Image.open(os.path.join(frames_in_dir, frame_files[0]))
            in_w, in_h = first_frame.size
            out_w, out_h = int(in_w * settings.scale), int(in_h * settings.scale)
            logger.info(
                f"V2G: {len(frame_files)} frames extracted at {in_w}x{in_h}, "
                f"will upscale to {out_w}x{out_h} ({settings.scale}x)"
            )

            frame_duration = int(1000 / fps)  # ms per frame

            # Upscale frames to disk (not RAM)
            for i, frame_file in enumerate(frame_files):
                if self.cancel_requested:
                    return

                frame_path = os.path.join(frames_in_dir, frame_file)
                frame_img = Image.open(frame_path)
                frame_array = np.array(frame_img.convert('RGBA'))

                result_array = CUDAUpscaler.upscale(
                    frame_array, settings.scale, settings.interpolation, settings.sharpen
                )

                # Write upscaled frame to disk instead of accumulating in RAM
                output_frame_path = os.path.join(frames_out_dir, frame_file)
                Image.fromarray(result_array).save(output_frame_path)
                del result_array

                progress = 10 + int(70 * (i + 1) / len(frame_files))
                self._update_progress(
                    f"Upscaling frame {i+1}/{len(frame_files)} ({in_w}x{in_h} -> {out_w}x{out_h})",
                    progress,
                    f"Frame: {i+1} / {len(frame_files)}"
                )

            if self.cancel_requested:
                return

            # Assemble GIF by reading frames back from disk one at a time
            self._update_progress("Assembling GIF...", 85)
            logger.info("Reading upscaled frames from disk for GIF assembly...")
            upscaled_frame_files = sorted(
                [f for f in os.listdir(frames_out_dir) if f.endswith('.png')]
            )

            if not upscaled_frame_files:
                raise RuntimeError("No upscaled frames produced")

            # Build PIL frames for GIF assembly (read from disk)
            durations = [frame_duration] * len(upscaled_frame_files)
            frames_for_gif = []
            for frame_file in upscaled_frame_files:
                frame_path = os.path.join(frames_out_dir, frame_file)
                frame_img = Image.open(frame_path)
                frame_img.load()
                frames_for_gif.append(np.array(frame_img.convert('RGBA')))

            GIFProcessor.assemble_gif(frames_for_gif, durations, 0, output_path, gif_settings)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _process_gif(
        self,
        input_path: str,
        output_path: str,
        settings: UpscaleSettings,
        gif_settings: GifSettings
    ):
        """Process GIF frame by frame -- disk-based pipeline to avoid OOM."""
        self._update_progress("Extracting GIF frames...", 5)
        frames, durations, loop = GIFProcessor.extract_frames(input_path)

        if not frames:
            raise RuntimeError("No frames found in GIF")

        in_h, in_w = frames[0].shape[:2]
        out_w, out_h = int(in_w * settings.scale), int(in_h * settings.scale)
        logger.info(
            f"GIF upscale: {len(frames)} frames, {in_w}x{in_h} -> {out_w}x{out_h} ({settings.scale}x)"
        )

        # Use temp dir for disk-based upscaling
        temp_dir = tempfile.mkdtemp(prefix='gif_upscale_')
        try:
            for i, frame in enumerate(frames):
                if self.cancel_requested:
                    return

                result_array = CUDAUpscaler.upscale(
                    frame, settings.scale, settings.interpolation, settings.sharpen
                )

                # Write to disk instead of accumulating in RAM
                frame_path = os.path.join(temp_dir, f'{i:06d}.png')
                Image.fromarray(result_array).save(frame_path)
                del result_array

                progress = 10 + int(80 * (i + 1) / len(frames))
                self._update_progress(
                    f"Frame {i+1}/{len(frames)} ({in_w}x{in_h} -> {out_w}x{out_h})",
                    progress,
                    f"Frame: {i+1} / {len(frames)}"
                )

            # Free original frames from memory
            del frames

            if self.cancel_requested:
                return

            # Read back upscaled frames from disk for assembly
            self._update_progress("Assembling GIF...", 95)
            logger.info("Reading upscaled frames for GIF assembly...")
            upscaled_frames = []
            frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])
            for frame_file in frame_files:
                frame_path = os.path.join(temp_dir, frame_file)
                frame_img = Image.open(frame_path)
                frame_img.load()
                upscaled_frames.append(np.array(frame_img.convert('RGBA')))

            GIFProcessor.assemble_gif(upscaled_frames, durations, loop, output_path, gif_settings)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _update_progress(self, status: str, percent: int, frame_info: str = ""):
        """Update progress display from any thread."""
        def update():
            try:
                self.progress_label.config(text=status)
                self.progress_bar['value'] = percent
                self.frame_counter_label.config(text=frame_info)
            except Exception:
                pass  # Widget may have been destroyed

        self._safe_after(update)

    def _cancel_processing(self):
        """Cancel ongoing processing."""
        self._cancel_event.set()
        logger.info("Cancel requested by user")
        self._update_progress("Cancelling...", 0)

    def _reset_ui(self):
        """Reset UI state after processing."""
        with self._processing_lock:
            self.processing = False
        self._set_ui_processing(False)
        self.frame_counter_label.config(text="")
        if not self.cancel_requested:
            self.progress_bar['value'] = 0


class _TkLogHandler(logging.Handler):
    """Custom logging handler that writes to the app's log widget."""

    def __init__(self, app: UpscalerApp):
        super().__init__()
        self.app = app

    def emit(self, record):
        try:
            msg = self.format(record)
            level = record.levelname
            self.app._append_log(msg, level)
        except Exception:
            pass


def main():
    """Application entry point."""
    logger.info("=" * 60)
    logger.info("CUDA Media Upscaler starting...")
    logger.info(f"Python {sys.version.split()[0]}")
    logger.info(f"NumPy {np.__version__}")
    logger.info(f"Pillow {Image.__version__}")
    if CUDA_AVAILABLE:
        logger.info(f"CuPy {cp.__version__} -- GPU: {CUDAUpscaler.get_gpu_info()}")
    logger.info("=" * 60)

    # Check for display availability
    try:
        root = tk.Tk()
    except tk.TclError as e:
        print(
            f"Error: Cannot initialize display ({e}).\n"
            "This application requires a graphical display (X11 or Wayland).\n"
            "If using SSH, connect with: ssh -X user@host"
        )
        sys.exit(1)

    # Startup warnings
    if not CUDA_AVAILABLE and not SCIPY_AVAILABLE:
        messagebox.showerror(
            "Missing Dependencies",
            "Neither CuPy (GPU) nor scipy (CPU) is available.\n"
            "Install at least one:\n"
            "  pip install cupy-cuda12x  (GPU)\n"
            "  pip install scipy  (CPU fallback)"
        )
        root.destroy()
        sys.exit(1)

    app = UpscalerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
