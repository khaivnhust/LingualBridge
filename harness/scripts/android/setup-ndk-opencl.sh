#!/usr/bin/env bash
# Install OpenCL headers and a link-time libOpenCL into an Android NDK sysroot.
#
# The NDK ships no OpenCL. We install headers from the Khronos clone and link
# against the vendor's own OpenCL implementation pulled from a connected device.
# At runtime the binary resolves libOpenCL.so from /vendor/lib64, so the link
# and runtime implementations are the same binary.
#
# Idempotent. Reversible via --undo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NDK="${ANDROID_NDK:-$HOME/.android/Sdk/ndk/$(ls "$HOME/.android/Sdk/ndk" 2>/dev/null | sort -V | tail -1)}"
SYSROOT="$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr"
CL_SRC="$ROOT/OpenCL-Headers/CL"
VENDOR_LIB="$ROOT/third_party/libOpenCL.so"

if [ ! -d "$SYSROOT" ]; then
    echo "error: NDK sysroot not found: $SYSROOT" >&2
    echo "       set ANDROID_NDK to your NDK root" >&2
    exit 1
fi

case "${1:-}" in
--undo)
    rm -rf "$SYSROOT/include/CL"
    rm -f "$SYSROOT/lib/aarch64-linux-android/libOpenCL.so"
    echo "removed NDK OpenCL headers and link stub"
    exit 0
    ;;
esac

if [ ! -d "$CL_SRC" ]; then
    echo "error: $CL_SRC not found (clone KhronosGroup/OpenCL-Headers)" >&2
    exit 1
fi

# 1. headers
cp -r "$CL_SRC" "$SYSROOT/include/CL"
echo "installed OpenCL headers -> $SYSROOT/include/CL"

# 2. link-time library. Prefer the device's own implementation so link-time and
#    runtime symbols match. Fall back to a stub import lib if no device.
mkdir -p "$ROOT/third_party"
if [ ! -f "$VENDOR_LIB" ]; then
    if command -v adb >/dev/null 2>&1 && adb get-state >/dev/null 2>&1; then
        echo "pulling vendor OpenCL from $(adb get-serialno)"
        adb shell 'ls /vendor/lib64/libOpenCL.so' >/dev/null 2>&1 || {
            echo "error: device has no /vendor/lib64/libOpenCL.so" >&2
            exit 1
        }
        # libOpenCL.so is a symlink to the real GLES/OpenCL driver on Mali.
        adb pull /vendor/lib64/libGLES_mali.so "$ROOT/third_party/libGLES_mali.so" >/dev/null
        ln -sf "$ROOT/third_party/libGLES_mali.so" "$VENDOR_LIB"
        echo "linked vendor lib from third_party/libGLES_mali.so"
    else
        echo "error: no device available to pull vendor OpenCL from." >&2
        echo "       Connect a device, or build the Khronos ICD loader instead:" >&2
        echo "       see $ROOT/OpenCL-ICD-Loader" >&2
        exit 1
    fi
fi

cp "$VENDOR_LIB" "$SYSROOT/lib/aarch64-linux-android/libOpenCL.so"
echo "installed link stub     -> $SYSROOT/lib/aarch64-linux-android/libOpenCL.so"
echo
echo "NDK OpenCL ready: $NDK"
