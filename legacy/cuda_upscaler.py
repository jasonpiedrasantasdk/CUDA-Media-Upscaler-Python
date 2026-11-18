#!/usr/bin/env python3
"""
CUDA Super-Resolution Upscaler
GPU-accelerated image upscaling with CuPy + Tkinter GUI.
Requires: cupy-cuda12x, pillow, numpy, python3-tk
"""

import sys
import os
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

try:
    import cupy as cp
    from cupyx.scipy.ndimage import zoom as gpu_zoom
    from cupyx.scipy.ndimage import convolve as gpu_convolve
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

# ── Constants ──────────────────────────────────────────────────────────────
MAX_SCALE = 8.0
MIN_SCALE = 1.0
INTERPOLATION_METHODS = {
    "Nearest":  0,
    "Bilinear": 1,
    "Bicubic":  3,
    "Quintic":  5,
}
SHARPEN_KERNELS = {
    "none": None,
    "light": np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32),
    "medium": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    "strong": np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32),
}

# ── CUDA Engine ────────────────────────────────────────────────────────────

class CUDAUpscaler:
    """GPU-accelerated image upscaling engine via CuPy."""

    @staticmethod
    def get_gpu_info() -> dict:
        if not HAS_CUDA:
            return {"name": "N/A", "mem_total": 0, "mem_free": 0, "mem_used": 0}
        dev = cp.cuda.Device(0)
        free, total = dev.mem_info
        return {
            "name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "mem_total": total,
            "mem_free": free,
            "mem_used": total - free,
        }

    @staticmethod
    def upscale(img_array: np.ndarray, scale: float, order: int = 3,
                sharpen: str = "none") -> np.ndarray:
        """
        Upscale image on GPU.
        img_array: HxWxC uint8 numpy array
        scale: multiplier (1.0–8.0)
        order: interpolation order (0=nearest, 1=bilinear, 3=bicubic, 5=quintic)
        sharpen: none/light/medium/strong post-upscale sharpening
        Returns: upscaled uint8 numpy array
        """
        if not HAS_CUDA:
            raise RuntimeError("CuPy/CUDA not available")

        # Transfer to GPU
        gpu_img = cp.asarray(img_array, dtype=cp.float32)

        # Zoom each channel — zoom factors: (scale_h, scale_w, 1.0 for channels)
        zoom_factors = (scale, scale, 1.0) if gpu_img.ndim == 3 else (scale, scale)
        gpu_result = gpu_zoom(gpu_img, zoom_factors, order=order, mode='reflect')

        # Optional sharpening
        kernel_np = SHARPEN_KERNELS.get(sharpen)
        if kernel_np is not None and gpu_result.ndim == 3:
            kernel_gpu = cp.asarray(kernel_np, dtype=cp.float32)
            for c in range(gpu_result.shape[2]):
                gpu_result[:, :, c] = gpu_convolve(gpu_result[:, :, c], kernel_gpu, mode='reflect')

        # Clamp and transfer back to CPU
        gpu_result = cp.clip(gpu_result, 0, 255).astype(cp.uint8)
        result = cp.asnumpy(gpu_result)

        # Free GPU memory
        del gpu_img, gpu_result
        cp.get_default_memory_pool().free_all_blocks()

        return result


# ── GUI ────────────────────────────────────────────────────────────────────

class UpscalerApp:
    BG       = "#1a1a2e"
    BG2      = "#16213e"
    FG       = "#e0e0e0"
    ACCENT   = "#0f3460"
    HIGHLIGHT= "#e94560"
    SUBTLE   = "#7f8c8d"
    ENTRY_BG = "#0a0a1a"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("⚡ CUDA Super-Resolution Upscaler")
        self.root.configure(bg=self.BG)
        self.root.geometry("1100x780")
        self.root.minsize(900, 650)

        # State
        self.src_image: Image.Image | None = None
        self.src_array: np.ndarray | None = None
        self.result_array: np.ndarray | None = None
        self.preview_photo = None  # prevent GC
        self.src_path: str = ""
        self._showing_result = False  # track which image is displayed

        self._build_style()
        self._build_ui()
        self._update_gpu_info()

    # ── Style ──────────────────────────────────────────────────────────
    def _build_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background=self.BG)
        style.configure("Dark2.TFrame", background=self.BG2)
        style.configure("Dark.TLabel", background=self.BG, foreground=self.FG,
                         font=("monospace", 10))
        style.configure("Title.TLabel", background=self.BG, foreground=self.HIGHLIGHT,
                         font=("monospace", 14, "bold"))
        style.configure("Info.TLabel", background=self.BG2, foreground=self.SUBTLE,
                         font=("monospace", 9))
        style.configure("Dark.TButton", background=self.ACCENT, foreground=self.FG,
                         font=("monospace", 10, "bold"), padding=8)
        style.map("Dark.TButton",
                  background=[("active", self.HIGHLIGHT)])
        style.configure("Accent.TButton", background=self.HIGHLIGHT, foreground="#fff",
                         font=("monospace", 11, "bold"), padding=10)
        style.map("Accent.TButton",
                  background=[("active", "#ff6b81")])

    # ── Layout ─────────────────────────────────────────────────────────
    def _build_ui(self):
        # Top bar
        top = ttk.Frame(self.root, style="Dark.TFrame")
        top.pack(fill="x", padx=12, pady=(10, 4))
        ttk.Label(top, text="⚡ CUDA Super-Resolution Upscaler",
                  style="Title.TLabel").pack(side="left")
        self.gpu_label = ttk.Label(top, text="GPU: …", style="Info.TLabel")
        self.gpu_label.pack(side="right")

        # Main split: left = controls, right = preview
        body = ttk.Frame(self.root, style="Dark.TFrame")
        body.pack(fill="both", expand=True, padx=12, pady=4)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # ── Left panel (controls) ──────────────────────────────────────
        left = ttk.Frame(body, style="Dark2.TFrame", width=320)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_propagate(False)

        pad = {"padx": 12, "pady": (8, 2)}

        # File open
        ttk.Button(left, text="📂  Open Image…", style="Dark.TButton",
                   command=self._open_file).pack(fill="x", **pad)
        self.file_label = ttk.Label(left, text="No file loaded", style="Info.TLabel",
                                     wraplength=280)
        self.file_label.pack(fill="x", padx=12)

        # Source info
        self.src_info = ttk.Label(left, text="", style="Info.TLabel")
        self.src_info.pack(fill="x", padx=12, pady=(2, 8))

        sep1 = ttk.Separator(left, orient="horizontal")
        sep1.pack(fill="x", padx=12, pady=4)

        # ── Scale slider ───────────────────────────────────────────────
        ttk.Label(left, text="Scale Factor", style="Dark.TLabel").pack(**pad)
        self.scale_var = tk.DoubleVar(value=2.0)
        scale_frame = ttk.Frame(left, style="Dark2.TFrame")
        scale_frame.pack(fill="x", padx=12)
        self.scale_slider = tk.Scale(
            scale_frame, from_=1.0, to=MAX_SCALE, resolution=0.25,
            orient="horizontal", variable=self.scale_var,
            bg=self.BG2, fg=self.FG, troughcolor=self.ACCENT,
            highlightthickness=0, font=("monospace", 10),
            command=lambda _: self._update_output_info()
        )
        self.scale_slider.pack(fill="x")

        # Output resolution display
        self.out_info = ttk.Label(left, text="Output: –", style="Dark.TLabel")
        self.out_info.pack(fill="x", padx=12, pady=(2, 8))

        # ── Custom resolution override ─────────────────────────────────
        ttk.Label(left, text="Custom Output (0 = auto)", style="Dark.TLabel").pack(**pad)
        res_frame = ttk.Frame(left, style="Dark2.TFrame")
        res_frame.pack(fill="x", padx=12)
        ttk.Label(res_frame, text="W:", style="Dark.TLabel").pack(side="left")
        self.custom_w = tk.IntVar(value=0)
        tk.Entry(res_frame, textvariable=self.custom_w, width=7,
                 bg=self.ENTRY_BG, fg=self.FG, insertbackground=self.FG,
                 font=("monospace", 10)).pack(side="left", padx=(2, 8))
        ttk.Label(res_frame, text="H:", style="Dark.TLabel").pack(side="left")
        self.custom_h = tk.IntVar(value=0)
        tk.Entry(res_frame, textvariable=self.custom_h, width=7,
                 bg=self.ENTRY_BG, fg=self.FG, insertbackground=self.FG,
                 font=("monospace", 10)).pack(side="left", padx=2)

        sep2 = ttk.Separator(left, orient="horizontal")
        sep2.pack(fill="x", padx=12, pady=8)

        # ── Interpolation method ───────────────────────────────────────
        ttk.Label(left, text="Interpolation", style="Dark.TLabel").pack(**pad)
        self.interp_var = tk.StringVar(value="Bicubic")
        interp_menu = ttk.Combobox(left, textvariable=self.interp_var,
                                    values=list(INTERPOLATION_METHODS.keys()),
                                    state="readonly", font=("monospace", 10))
        interp_menu.pack(fill="x", padx=12, pady=2)

        # ── Sharpening ─────────────────────────────────────────────────
        ttk.Label(left, text="Post-Sharpen", style="Dark.TLabel").pack(**pad)
        self.sharpen_var = tk.StringVar(value="none")
        sharp_menu = ttk.Combobox(left, textvariable=self.sharpen_var,
                                   values=list(SHARPEN_KERNELS.keys()),
                                   state="readonly", font=("monospace", 10))
        sharp_menu.pack(fill="x", padx=12, pady=2)

        sep3 = ttk.Separator(left, orient="horizontal")
        sep3.pack(fill="x", padx=12, pady=8)

        # ── Export format ──────────────────────────────────────────────
        ttk.Label(left, text="Export Format", style="Dark.TLabel").pack(**pad)
        self.fmt_var = tk.StringVar(value="PNG")
        fmt_menu = ttk.Combobox(left, textvariable=self.fmt_var,
                                 values=["PNG", "JPEG", "TIFF", "WebP"],
                                 state="readonly", font=("monospace", 10))
        fmt_menu.pack(fill="x", padx=12, pady=2)

        # JPEG/WebP quality
        ttk.Label(left, text="Quality (JPEG/WebP)", style="Dark.TLabel").pack(**pad)
        self.quality_var = tk.IntVar(value=95)
        self.quality_slider = tk.Scale(
            left, from_=10, to=100, resolution=5,
            orient="horizontal", variable=self.quality_var,
            bg=self.BG2, fg=self.FG, troughcolor=self.ACCENT,
            highlightthickness=0, font=("monospace", 9)
        )
        self.quality_slider.pack(fill="x", padx=12)

        sep4 = ttk.Separator(left, orient="horizontal")
        sep4.pack(fill="x", padx=12, pady=8)

        # Action buttons
        ttk.Button(left, text="🚀  Upscale (GPU)", style="Accent.TButton",
                   command=self._run_upscale).pack(fill="x", padx=12, pady=(4, 2))
        ttk.Button(left, text="💾  Export…", style="Dark.TButton",
                   command=self._export).pack(fill="x", padx=12, pady=2)

        # Status / progress
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(left, textvariable=self.status_var,
                                       style="Info.TLabel")
        self.status_label.pack(fill="x", padx=12, pady=(8, 2))

        self.progress = ttk.Progressbar(left, mode="indeterminate", length=280)
        self.progress.pack(padx=12, pady=(0, 8))

        # VRAM info
        self.vram_label = ttk.Label(left, text="VRAM: –", style="Info.TLabel")
        self.vram_label.pack(fill="x", padx=12, pady=(0, 8))

        # ── Right panel (preview canvas) ───────────────────────────────
        right = ttk.Frame(body, style="Dark.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(right, bg=self.ENTRY_BG, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda e: self._draw_preview(use_result=self._showing_result))

        # Preview label overlay
        self.preview_label = ttk.Label(right, text="Drag & drop or open an image",
                                        style="Info.TLabel", anchor="center")
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")

        # Bottom bar
        bottom = ttk.Frame(self.root, style="Dark.TFrame")
        bottom.pack(fill="x", padx=12, pady=(0, 8))
        ttk.Label(bottom, text="Engine: CuPy + CUDA NPP  │  RTX 4070 Super",
                  style="Info.TLabel").pack(side="left")
        self.time_label = ttk.Label(bottom, text="", style="Info.TLabel")
        self.time_label.pack(side="right")

    # ── Helpers ─────────────────────────────────────────────────────────
    def _mb(self, b: int) -> str:
        return f"{b / 1048576:.0f} MB"

    def _update_gpu_info(self):
        if not HAS_CUDA:
            self.gpu_label.config(text="GPU: CUDA not available (CPU fallback)")
            return
        info = CUDAUpscaler.get_gpu_info()
        self.gpu_label.config(
            text=f"GPU: {info['name']}  │  VRAM: {self._mb(info['mem_used'])} / {self._mb(info['mem_total'])}"
        )
        self.vram_label.config(
            text=f"VRAM free: {self._mb(info['mem_free'])}  │  used: {self._mb(info['mem_used'])}"
        )

    def _update_output_info(self):
        if self.src_image is None:
            self.out_info.config(text="Output: –")
            return
        w, h = self.src_image.size
        cw, ch = self.custom_w.get(), self.custom_h.get()
        if cw > 0 and ch > 0:
            ow, oh = cw, ch
            scale_eff = f"custom"
        else:
            s = self.scale_var.get()
            ow, oh = int(w * s), int(h * s)
            scale_eff = f"{s:.2f}x"
        mp = (ow * oh) / 1_000_000
        self.out_info.config(text=f"Output: {ow}×{oh}  ({mp:.1f} MP, {scale_eff})")

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.webp"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return
        try:
            img = Image.open(path)
            # Convert palette / grayscale to RGB, keep RGBA
            if img.mode == "P":
                img = img.convert("RGBA")
            elif img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            self.src_image = img
            self.src_array = np.array(img)
            self.src_path = path
            self.result_array = None
            self._showing_result = False
            w, h = img.size
            ch = self.src_array.shape[2] if self.src_array.ndim == 3 else 1
            self.file_label.config(text=os.path.basename(path))
            self.src_info.config(text=f"Source: {w}×{h}  │  {img.mode}  │  {ch}ch")
            self.preview_label.place_forget()
            self._update_output_info()
            self._draw_preview()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _draw_preview(self, use_result: bool = False):
        """Fit-draw either source or result image into the canvas."""
        arr = self.result_array if (use_result and self.result_array is not None) else (
            self.src_array if self.src_array is not None else None
        )
        if arr is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        img = Image.fromarray(arr)
        iw, ih = img.size
        ratio = min(cw / iw, ch / ih, 1.0)
        new_w, new_h = max(1, int(iw * ratio)), max(1, int(ih * ratio))
        thumb = img.resize((new_w, new_h), Image.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(thumb)
        self.canvas.delete("all")
        x, y = cw // 2, ch // 2
        self.canvas.create_image(x, y, image=self.preview_photo, anchor="center")
        # Overlay info
        label = f"{iw}×{ih}"
        if use_result and self.result_array is not None:
            label = f"RESULT  {label}"
        self.canvas.create_text(10, 10, text=label, fill="#e94560",
                                font=("monospace", 11, "bold"), anchor="nw")

    def _run_upscale(self):
        if self.src_array is None:
            messagebox.showwarning("No Image", "Load an image first.")
            return
        if not HAS_CUDA:
            messagebox.showerror("CUDA Error", "CuPy/CUDA not available.")
            return

        # Determine effective scale or custom dims
        cw_val, ch_val = self.custom_w.get(), self.custom_h.get()
        if cw_val > 0 and ch_val > 0:
            # Custom resolution: compute scale from source
            sw, sh = self.src_image.size
            scale_x = cw_val / sw
            scale_y = ch_val / sh
            # Use uniform scale = max to cover, then resize exact
            scale = max(scale_x, scale_y)
            use_custom_resize = True
            target_size = (cw_val, ch_val)
        else:
            scale = self.scale_var.get()
            use_custom_resize = False
            target_size = None

        order = INTERPOLATION_METHODS[self.interp_var.get()]
        sharpen = self.sharpen_var.get()

        self.status_var.set(f"Upscaling {scale:.2f}x on GPU…")
        self.progress.start(15)

        def _worker():
            try:
                t0 = time.perf_counter()
                result = CUDAUpscaler.upscale(self.src_array, scale, order, sharpen)

                # If custom resolution, do a final exact resize on CPU
                if use_custom_resize and target_size:
                    result_img = Image.fromarray(result)
                    result_img = result_img.resize(target_size, Image.LANCZOS)
                    result = np.array(result_img)

                elapsed = time.perf_counter() - t0
                self.result_array = result
                self._showing_result = True
                h, w = result.shape[:2]
                mp = (w * h) / 1_000_000

                self.root.after(0, lambda: [
                    self.progress.stop(),
                    self.status_var.set(f"Done — {w}×{h} ({mp:.1f} MP)"),
                    self.time_label.config(text=f"⏱ {elapsed:.2f}s"),
                    self._update_gpu_info(),
                    self._draw_preview(use_result=True),
                ])
            except Exception as e:
                self.root.after(0, lambda: [
                    self.progress.stop(),
                    self.status_var.set(f"Error: {e}"),
                ])

        threading.Thread(target=_worker, daemon=True).start()

    def _export(self):
        if self.result_array is None:
            messagebox.showwarning("Nothing to Export", "Run upscale first.")
            return
        fmt = self.fmt_var.get()
        ext_map = {"PNG": ".png", "JPEG": ".jpg", "TIFF": ".tiff", "WebP": ".webp"}
        ext = ext_map.get(fmt, ".png")

        # Default filename
        stem = Path(self.src_path).stem if self.src_path else "upscaled"
        h, w = self.result_array.shape[:2]
        default_name = f"{stem}_{w}x{h}{ext}"

        path = filedialog.asksaveasfilename(
            title="Export Upscaled Image",
            defaultextension=ext,
            initialfile=default_name,
            filetypes=[(fmt, f"*{ext}"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            img = Image.fromarray(self.result_array)
            save_kwargs = {}
            if fmt in ("JPEG", "WebP"):
                save_kwargs["quality"] = self.quality_var.get()
            if fmt == "JPEG" and img.mode == "RGBA":
                img = img.convert("RGB")
            img.save(path, **save_kwargs)
            self.status_var.set(f"Exported → {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))


# ── Entry Point ────────────────────────────────────────────────────────────

def main():
    if not HAS_CUDA:
        print("WARNING: CuPy/CUDA not found. Install with: pip3 install --user cupy-cuda12x")
        print("         The GUI will open but upscaling will fail.")

    root = tk.Tk()
    app = UpscalerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
