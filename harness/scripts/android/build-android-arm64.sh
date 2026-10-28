#!/usr/bin/env bash
# Cross-compile llama.cpp for an Android ABI with the OpenCL backend.
#
# Keeps source clones pristine: all cmake output goes to build/<platform>/ and
# the install tree to dist/<platform>/. Nothing is written into llama.cpp/.
#
# Usage:
#   scripts/build-android-arm64.sh [platform-tag]
#
# Examples:
#   scripts/build-android-arm64.sh                 # -> build/android-arm64-opencl
#   scripts/build-android-arm64.sh m35-test        # -> build/android-arm64-opencl-m35-test
#
# Per-platform variants share one configure cache; give each a distinct tag so
# flags (Adreno kernels, quant coverage) can differ without fighting.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-}"
PLATFORM="android-arm64-opencl${TAG:+-$TAG}"
BUILD_DIR="$ROOT/build/$PLATFORM"
DIST_DIR="$ROOT/dist/$PLATFORM"

export ANDROID_HOME="${ANDROID_HOME:-$HOME/.android/Sdk}"
if [ -z "${ANDROID_NDK:-}" ]; then
    ANDROID_NDK="$ANDROID_HOME/ndk/$(ls "$ANDROID_HOME/ndk" | sort -V | tail -1)"
fi
export ANDROID_NDK

if [ ! -f "$ANDROID_NDK/build/cmake/android.toolchain.cmake" ]; then
    echo "error: no NDK at $ANDROID_NDK - run scripts/setup-ndk-opencl.sh" >&2
    exit 1
fi

# The Khronos loader is only needed if we are not linking the vendor driver.
# scripts/setup-ndk-opencl.sh installs headers plus a link-time libOpenCL.
if [ ! -f "$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include/CL/cl.h" ]; then
    echo "error: OpenCL headers missing from NDK - run scripts/setup-ndk-opencl.sh" >&2
    exit 1
fi

JOBS="${JOBS:-$(nproc)}"

# Adreno kernels default ON in ggml. They are runtime-gated on gpu_family, so
# they are inert on Mali but give Adreno 750 (A7X) the xmem GEMM/SDPA path.
# Override with ADRNENO_KERNELS=0 for a smaller binary on non-Adreno targets.
ADRENO="${ADRENO_KERNELS:-ON}"

echo "==> platform   : $PLATFORM"
echo "==> ndk        : $ANDROID_NDK"
echo "==> build dir  : $BUILD_DIR"
echo "==> dist dir   : $DIST_DIR"
echo "==> adreno kern: $ADRENO"
echo

cmake -S "$ROOT/llama.cpp" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DGGML_OPENCL=ON \
    -DGGML_OPENCL_USE_ADRENO_KERNELS="$ADRENO" \
    -DGGML_OPENCL_EMBED_KERNELS=ON \
    -DGGML_CPU_KLEIDIAI=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_SERVER=OFF

cmake --build "$BUILD_DIR" -j "$JOBS"

mkdir -p "$DIST_DIR"
cmake --install "$BUILD_DIR" --prefix "$DIST_DIR" 2>/dev/null || true

echo
echo "==> built. binaries in $DIST_DIR/bin (if installed)"
ls -la "$BUILD_DIR/bin/llama-mtmd-cli" 2>/dev/null || true
file "$BUILD_DIR/bin/llama-mtmd-cli" 2>/dev/null || true
