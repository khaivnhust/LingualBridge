#!/usr/bin/env bash
# Cross-compile llama.cpp for Android arm64 with the Vulkan backend.
#
# Vulkan is vendor-neutral, so unlike GGML_OPENCL (Adreno/Intel only) it runs on
# Mali too. Use it as a GPU correctness harness for the mtmd + Qwen3-ASR path.
#
# Requires a Vulkan SDK for glslc and the Vulkan/SPIR-V headers. Point VULKAN_SDK
# at an extracted LunarG SDK (x86_64/ subdir), or set GLSLC and the include dirs
# individually. SPIRV-Headers needs a cmake config; if the SDK does not ship
# one, run scripts/setup-spirv-headers.sh first.
#
# Usage:
#   scripts/build-android-arm64-vulkan.sh [platform-tag]
#
# Example:
#   scripts/build-android-arm64-vulkan.sh mali-test
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-}"
PLATFORM="android-arm64-vulkan${TAG:+-$TAG}"
BUILD_DIR="$ROOT/build/$PLATFORM"

export ANDROID_HOME="${ANDROID_HOME:-$HOME/.android/Sdk}"
if [ -z "${ANDROID_NDK:-}" ]; then
    ANDROID_NDK="$ANDROID_HOME/ndk/$(ls "$ANDROID_HOME/ndk" | sort -V | tail -1)"
fi
export ANDROID_NDK

VULKAN_SDK="${VULKAN_SDK:-/tmp/vktools/1.4.363.0/x86_64}"
GLSLC="${GLSLC:-$VULKAN_SDK/bin/glslc}"
SPIRV_DIR="${SPIRV_DIR:-$HOME/.local/spirv-headers/share/spirv-headers}"

for f in "$ANDROID_NDK/build/cmake/android.toolchain.cmake" "$GLSLC"; do
    [ -e "$f" ] || { echo "error: missing $f" >&2; exit 1; }
done
if [ ! -d "$SPIRV_DIR" ]; then
    echo "error: SPIRV-Headers cmake config missing at $SPIRV_DIR" >&2
    echo "       run scripts/setup-spirv-headers.sh" >&2
    exit 1
fi

echo "==> platform : $PLATFORM"
echo "==> ndk      : $ANDROID_NDK"
echo "==> glslc    : $GLSLC"
echo "==> spirv    : $SPIRV_DIR"
echo "==> build    : $BUILD_DIR"
echo

cmake -S "$ROOT/llama.cpp" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DGGML_VULKAN=ON \
    -DVulkan_INCLUDE_DIR="$VULKAN_SDK/include" \
    -DVulkan_GLSLC_EXECUTABLE="$GLSLC" \
    -DSPIRV-Headers_DIR="$SPIRV_DIR" \
    -DGGML_CPU_KLEIDIAI=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_SERVER=OFF

cmake --build "$BUILD_DIR" -j "${JOBS:-$(nproc)}"

mkdir -p "$ROOT/dist/$PLATFORM/bin"
cp "$BUILD_DIR/bin/llama-mtmd-cli" "$ROOT/dist/$PLATFORM/bin/"
"$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" \
    --strip-all "$ROOT/dist/$PLATFORM/bin/llama-mtmd-cli"

echo
ls -lh "$ROOT/dist/$PLATFORM/bin/llama-mtmd-cli"
file "$ROOT/dist/$PLATFORM/bin/llama-mtmd-cli" | cut -c1-100
