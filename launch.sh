#!/bin/bash
# CUDA Media Upscaler Launcher

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "========================================"
echo "     CUDA Media Upscaler Launcher      "
echo "========================================"
echo -e "${NC}"

# Set up CUDA library path if CUDA is installed
# First, try to detect CUDA path using nvcc
if command -v nvcc &> /dev/null; then
    CUDA_BIN_DIR=$(dirname "$(which nvcc)")
    CUDA_ROOT=$(dirname "$CUDA_BIN_DIR")
    if [ -d "$CUDA_ROOT/lib64" ]; then
        export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$LD_LIBRARY_PATH"
    elif [ -d "$CUDA_ROOT/targets/x86_64-linux/lib" ]; then
        export LD_LIBRARY_PATH="$CUDA_ROOT/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
    fi
elif [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
elif [ -d "/usr/local/cuda-13.2/targets/x86_64-linux/lib" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda-13.2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
elif [ -d "/usr/local/cuda-13.1/targets/x86_64-linux/lib" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda-13.1/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
elif [ -d "/usr/local/cuda-12/targets/x86_64-linux/lib" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda-12/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
elif [ -d "/usr/local/cuda-11/targets/x86_64-linux/lib" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda-11/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
fi

# Check Python version
echo -e "${YELLOW}Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo -e "${RED}Error: Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "  Python version: ${GREEN}$PYTHON_VERSION${NC}"

# Check for virtual environment
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

# Check required packages
echo -e "${YELLOW}Checking Python dependencies...${NC}"

check_package() {
    "$PYTHON" -c "import $1" 2>/dev/null
    return $?
}

MISSING_PACKAGES=()

if ! check_package numpy; then
    MISSING_PACKAGES+=("numpy")
fi

if ! check_package PIL; then
    MISSING_PACKAGES+=("Pillow")
fi

if ! check_package scipy; then
    MISSING_PACKAGES+=("scipy")
fi

# Check CuPy (optional but recommended)
if check_package cupy; then
    echo -e "  CuPy (CUDA): ${GREEN}Available${NC}"

    # Check CUDA device + verify GPU kernels actually work
    CUDA_INFO=$("$PYTHON" -c "
import cupy as cp
try:
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    mem = props['totalGlobalMem'] / (1024**3)
    # Verify GPU compute actually works (catches CUDA version mismatch)
    test = cp.zeros((2,2), dtype=cp.float32)
    _ = cp.asnumpy(test + 1.0)
    print(f'{name} ({mem:.1f} GB) -- GPU compute OK')
except Exception as e:
    print(f'Error: {e}')
    print('WARNING: CuPy imported but GPU compute failed!')
    print('Try: pip install cupy-cuda12x (match your CUDA version)')
" 2>&1)
    echo -e "  GPU Device: ${GREEN}$CUDA_INFO${NC}"

    # Show CuPy version and CUDA runtime
    CUPY_VER=$("$PYTHON" -c "import cupy; print(f'CuPy {cupy.__version__}, CUDA runtime {cupy.cuda.runtime.runtimeGetVersion()}')" 2>/dev/null)
    echo -e "  ${CYAN}$CUPY_VER${NC}"
else
    echo -e "  CuPy (CUDA): ${YELLOW}Not available - will use CPU fallback${NC}"
    echo -e "  ${YELLOW}For GPU acceleration, install: pip install cupy-cuda12x${NC}"
fi

# Check system dependencies
echo -e "${YELLOW}Checking system tools...${NC}"

# ffmpeg (required for video processing)
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n1 | cut -d' ' -f3)
    echo -e "  ffmpeg: ${GREEN}$FFMPEG_VERSION${NC}"
else
    echo -e "  ffmpeg: ${RED}Not found${NC}"
    echo -e "  ${YELLOW}Video processing requires ffmpeg. Install with:${NC}"
    echo -e "    sudo apt install ffmpeg"
fi

# ffprobe (required alongside ffmpeg)
if command -v ffprobe &> /dev/null; then
    echo -e "  ffprobe: ${GREEN}Available${NC}"
else
    echo -e "  ffprobe: ${RED}Not found${NC}"
    echo -e "  ${YELLOW}ffprobe is required and usually ships with ffmpeg${NC}"
fi

# gifsicle (optional for GIF optimization)
if command -v gifsicle &> /dev/null; then
    GIFSICLE_VERSION=$(gifsicle --version 2>&1 | head -n1)
    echo -e "  gifsicle: ${GREEN}Available${NC}"
else
    echo -e "  gifsicle: ${YELLOW}Not found (optional, recommended for GIF optimization)${NC}"
    echo -e "    Install with: sudo apt install gifsicle"
fi

# zenity/kdialog (for native file dialogs on Wayland)
if command -v zenity &> /dev/null; then
    echo -e "  zenity: ${GREEN}Available (native file dialogs)${NC}"
elif command -v kdialog &> /dev/null; then
    echo -e "  kdialog: ${GREEN}Available (native file dialogs)${NC}"
else
    echo -e "  Native dialogs: ${YELLOW}Not found (will use Tk dialogs)${NC}"
    echo -e "    For better Wayland support: sudo apt install zenity"
fi

# Install missing packages if any
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}Installing missing packages: ${MISSING_PACKAGES[*]}${NC}"
    "$PYTHON" -m pip install "${MISSING_PACKAGES[@]}"
fi

# Check for display (for GUI)
if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ] && [ "$(uname)" != "Darwin" ]; then
    echo -e "${YELLOW}Warning: No display detected. GUI may not work.${NC}"
    echo -e "  Set DISPLAY or WAYLAND_DISPLAY environment variable."
fi

echo ""
echo -e "${GREEN}Launching CUDA Media Upscaler...${NC}"
echo ""

# Launch the application
"$PYTHON" cuda_media_upscaler.py "$@"
