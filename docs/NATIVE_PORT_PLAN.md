# Native Port Plan: CUDA Media Upscaler

## Executive Summary

This document outlines the strategic plan for porting the CUDA Media Upscaler from Python/Tkinter to a native compiled language for improved performance, better UI/UX, and production-ready deployment.

---

## Current Architecture Analysis

### Python Implementation Strengths
- Rapid prototyping and iteration
- Easy CUDA integration via CuPy
- Cross-platform compatibility
- Simple dependency management

### Python Implementation Weaknesses
- **UI Limitations**: Tkinter is dated, lacks modern styling, poor Wayland support
- **Startup Time**: ~2-3 seconds cold start
- **Memory Overhead**: Python interpreter + NumPy arrays
- **Distribution**: Requires Python runtime, pip dependencies
- **Threading**: GIL limitations for CPU-bound operations

---

## Recommended Native Stack Options

### Option A: C++ with Qt6 (Recommended)
**Best for: Cross-platform desktop application with modern UI**

| Component | Technology |
|-----------|------------|
| **Language** | C++20 |
| **UI Framework** | Qt6 (QML for modern UI) |
| **CUDA** | Native CUDA C++ (NPP, cuBLAS) |
| **Image Processing** | OpenCV with CUDA |
| **Video Processing** | FFmpeg C API |
| **GIF Processing** | giflib + gifsicle |
| **Build System** | CMake + vcpkg/Conan |

**Advantages:**
- Native CUDA integration (no Python overhead)
- Qt6 has excellent Wayland support
- Single binary distribution
- ~50-100ms startup time
- Professional modern UI with QML

**Architecture:**
```
src/
├── core/
│   ├── cuda_upscaler.cu      # CUDA kernels
│   ├── video_processor.cpp   # FFmpeg integration
│   └── gif_processor.cpp     # GIF handling
├── ui/
│   ├── main.qml              # Qt Quick UI
│   └── components/           # Reusable QML components
└── main.cpp
```

---

### Option B: C# with .NET 8 + Avalonia UI
**Best for: Modern cross-platform with managed memory**

| Component | Technology |
|-----------|------------|
| **Language** | C# 12 / .NET 8 |
| **UI Framework** | Avalonia UI 11 |
| **CUDA** | ManagedCUDA / ILGPU |
| **Image Processing** | ImageSharp + SkiaSharp |
| **Video Processing** | FFmpeg.AutoGen |
| **Build** | dotnet CLI |

**Advantages:**
- Modern language features (async/await, LINQ)
- Avalonia has native Wayland support
- AOT compilation for single binary
- Strong typing with nullable reference types
- Easy to maintain for teams

**Considerations:**
- CUDA interop adds some overhead
- Smaller ecosystem than Qt for native apps

---

### Option C: Rust + GTK4
**Best for: Maximum performance, Linux-first**

| Component | Technology |
|-----------|------------|
| **Language** | Rust |
| **UI Framework** | GTK4 (gtk4-rs) |
| **CUDA** | rustacuda / cuda-sys |
| **Image Processing** | image-rs + custom CUDA |
| **Video Processing** | ffmpeg-next |
| **Build** | Cargo |

**Advantages:**
- Zero-cost abstractions
- Memory safety guarantees
- Native GTK4 = perfect Wayland support
- Single static binary
- Excellent performance

**Considerations:**
- Steeper learning curve
- CUDA bindings less mature than C++

---

## Recommended Approach: Phased Migration

### Phase 1: Core Engine (C++ + CUDA)
**Timeline: 2-3 weeks**

Create a standalone C++ library with the upscaling engine:

```cpp
// cuda_upscaler.hpp
class CUDAUpscaler {
public:
    enum class Interpolation { Nearest, Bilinear, Bicubic, Lanczos };
    enum class Sharpen { None, Light, Medium, Strong };

    struct UpscaleParams {
        float scale;
        Interpolation interpolation;
        Sharpen sharpen;
    };

    // Returns upscaled image data
    std::vector<uint8_t> upscale(
        const uint8_t* input,
        int width, int height, int channels,
        const UpscaleParams& params
    );
};
```

**Dependencies:**
- CUDA Toolkit 12.x
- NVIDIA Performance Primitives (NPP)
- OpenCV (optional, for additional filters)

### Phase 2: Video/GIF Processing Layer
**Timeline: 1-2 weeks**

```cpp
// video_processor.hpp
class VideoProcessor {
public:
    struct VideoInfo {
        int width, height;
        double fps;
        int64_t frame_count;
        bool has_audio;
    };

    VideoInfo probe(const std::filesystem::path& path);
    void process_frames(
        const std::filesystem::path& input,
        const std::filesystem::path& output,
        std::function<void(cv::Mat&)> frame_processor,
        std::function<void(int, int)> progress_callback
    );
};

// gif_processor.hpp
class GIFProcessor {
public:
    std::vector<Frame> extract_frames(const std::filesystem::path& path);
    void assemble_gif(
        const std::vector<Frame>& frames,
        const std::filesystem::path& output,
        const GIFSettings& settings
    );
    void optimize_with_gifsicle(
        const std::filesystem::path& input,
        const std::filesystem::path& output,
        int lossy_level
    );
};
```

### Phase 3: Qt6 UI
**Timeline: 2-3 weeks**

Modern QML-based UI with:
- Native file dialogs (QFileDialog with native mode)
- Fluent/Material design
- Hardware-accelerated preview
- Drag & drop support
- Progress indicators with ETA

```qml
// Main.qml
ApplicationWindow {
    TabBar {
        TabButton { text: "Image" }
        TabButton { text: "Video" }
        TabButton { text: "GIF" }
        TabButton { text: "Video → GIF" }
    }

    StackLayout {
        ImageUpscaler { }
        VideoUpscaler { }
        GIFUpscaler { }
        VideoToGIFConverter { }
    }
}
```

### Phase 4: Packaging & Distribution
**Timeline: 1 week**

| Platform | Package Format |
|----------|---------------|
| Linux | AppImage, Flatpak, .deb |
| Windows | MSIX, portable .exe |
| macOS | .dmg, .app bundle |

---

## Build System Setup

### CMakeLists.txt (Root)
```cmake
cmake_minimum_required(VERSION 3.25)
project(CUDAMediaUpscaler LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(Qt6 REQUIRED COMPONENTS Quick Widgets)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale)

add_subdirectory(src/core)
add_subdirectory(src/ui)
add_executable(cuda-upscaler src/main.cpp)
target_link_libraries(cuda-upscaler
    PRIVATE upscaler_core upscaler_ui
    Qt6::Quick Qt6::Widgets
    CUDA::cudart CUDA::nppig
    ${OpenCV_LIBS}
    ${FFMPEG_LIBRARIES}
)
```

---

## Performance Comparison (Estimated)

| Metric | Python/CuPy | C++/CUDA Native |
|--------|-------------|-----------------|
| Startup time | 2-3s | <100ms |
| Memory (idle) | ~150MB | ~30MB |
| 4K frame upscale | 85ms | 45ms |
| UI responsiveness | 60fps (laggy) | 120fps (smooth) |
| Binary size | N/A (needs runtime) | ~25MB |

---

## Migration Checklist

- [ ] Set up CMake project structure
- [ ] Implement CUDA upscaling kernels
- [ ] Add NPP-based interpolation
- [ ] Implement sharpening kernels
- [ ] FFmpeg video processing integration
- [ ] GIF processing with giflib
- [ ] gifsicle integration
- [ ] Qt6 QML UI design
- [ ] Native file dialogs
- [ ] Progress reporting
- [ ] Cancel support
- [ ] GPU memory management
- [ ] Error handling
- [ ] Unit tests (Google Test)
- [ ] Packaging (AppImage, MSIX, DMG)
- [ ] CI/CD pipeline

---

## Interim Solution

While developing the native port, the Python version can be improved with:

1. **PyInstaller**: Bundle as single executable
2. **Nuitka**: Compile to native binary
3. **CustomTkinter**: Modern-looking Tk UI
4. **GTK4 via PyGObject**: Better Wayland support

```bash
# PyInstaller bundling
pip install pyinstaller
pyinstaller --onefile --windowed cuda_media_upscaler.py
```

---

## Conclusion

**Recommended Path**:
1. **Short-term**: Continue with Python + add PyInstaller bundling
2. **Medium-term**: Build C++ core library with CUDA
3. **Long-term**: Full Qt6 native application

The C++ + Qt6 approach provides the best balance of:
- Performance (native CUDA)
- UI quality (Qt Quick/QML)
- Cross-platform support
- Wayland compatibility
- Distribution simplicity

---

*Document prepared by MiniMax Agent*
*Last updated: 2026-03-27*
