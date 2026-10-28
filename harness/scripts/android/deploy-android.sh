#!/usr/bin/env bash
# Push the Android OpenCL build plus Qwen3-ASR weights to a device and run it.
#
# NOT run automatically. Invoke by hand once the device is ready.
#
# Usage:
#   scripts/deploy-android.sh <platform-tag> [audio.wav ...]
#
# Examples:
#   scripts/deploy-android.sh                    test.wav
#   scripts/deploy-android.sh m35-test           test.wav short.wav
#
# Weights are read from the HF cache, not from this repo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-}"; shift || true
PLATFORM="android-arm64-opencl${TAG:+-$TAG}"
DIST="$ROOT/dist/$PLATFORM"
REMOTE="/data/local/tmp/llm"

command -v adb >/dev/null || { echo "error: adb not on PATH" >&2; exit 1; }

if [ ! -x "$DIST/bin/llama-mtmd-cli" ]; then
    echo "error: no build at $DIST - run scripts/build-android-arm64.sh $TAG" >&2
    exit 1
fi

# Resolve weights from the HF cache by globbing snapshots, rather than parsing
# `hf cache ls`, whose output format varies across huggingface_hub versions.
HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
SNAP_GLOB="$HF_HUB/models--ggml-org--Qwen3-ASR-0.6B-GGUF/snapshots"
resolve() {
    ls "$SNAP_GLOB"/*/"$1" 2>/dev/null | head -1
}
LM="$(resolve Qwen3-ASR-0.6B-Q8_0.gguf)"
MMPROJ="$(resolve mmproj-Qwen3-ASR-0.6B-Q8_0.gguf)"

for f in "$LM" "$MMPROJ"; do
    [ -f "$f" ] || { echo "error: not in HF cache: ${f:-<unresolved>}" >&2; exit 1; }
done

echo "==> device: $(adb get-serialno)"
echo "==> platform: $PLATFORM"
adb shell "mkdir -p $REMOTE"

echo "==> pushing binary + vendor OpenCL"
adb push "$DIST/bin/llama-mtmd-cli" "$REMOTE/" >/dev/null
adb push "$DIST/lib/libGLES_mali.so"   "$REMOTE/" >/dev/null

echo "==> pushing weights"
adb push "$LM"     "$REMOTE/" >/dev/null
adb push "$MMPROJ" "$REMOTE/" >/dev/null

# libGLES_mali.so is a vendor library and is not guaranteed to resolve from
# /data/local/tmp, so keep the staged copy next to the binary.
RUN="cd $REMOTE && LD_LIBRARY_PATH=. ./llama-mtmd-cli \
  -m Qwen3-ASR-0.6B-Q8_0.gguf --mmproj mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"

echo
echo "==> device checks"
adb shell "cd $REMOTE && LD_LIBRARY_PATH=. ./llama-mtmd-cli --list-devices"

if [ "$#" -eq 0 ]; then
    echo
    echo "==> no audio given. Run manually, e.g."
    echo "    adb shell \"cd $REMOTE && LD_LIBRARY_PATH=. ./llama-mtmd-cli \\"
    echo "      -m Qwen3-ASR-0.6B-Q8_0.gguf --mmproj mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \\"
    echo "      --audio clip.wav -p 'Transcribe this audio exactly.' -ngl 99\""
    exit 0
fi

for a in "$@"; do
    [ -f "$a" ] || { echo "warn: no such file, skipping: $a" >&2; continue; }
    base="$(basename "$a")"
    adb push "$a" "$REMOTE/" >/dev/null
    echo
    echo "==> transcribing $base"
    adb shell "$RUN --audio $base -p 'Transcribe this audio exactly.' -ngl 99"
done
