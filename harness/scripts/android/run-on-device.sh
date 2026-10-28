#!/usr/bin/env bash
# Run one transcription on the device, pinned to a CPU subset and thread-limited.
#
# Thread/CPU pinning is not a llama.cpp flag for cores: use taskset for the core
# set and -t for llama.cpp's thread pool. Default core set is 4-7 (the big
# cluster on big.LITTLE layouts), leaving 0-3 for the OS and audio pipeline.
#
# The Mali Vulkan driver is fragile enough on this device that we keep runs
# short and bounded: a hard timeout plus a fixed core set.
#
# Usage:
#   scripts/run-on-device.sh <binary> <audio.wav> [threads]
#
# Example:
#   scripts/run-on-device.sh llama-mtmd-cli-vk raokouling.wav 4
#
# Env:
#   PIN_CORES=4-7   also bind to those cores via taskset (needs PIN_CORES set)
#   TIMEOUT=600     hard timeout in seconds
set -euo pipefail

BIN="${1:?binary name in /data/local/tmp/llm}"
AUDIO="${2:?audio file name in /data/local/tmp/llm}"
THREADS="${3:-4}"
REMOTE="/data/local/tmp/llm"
MODEL="Qwen3-ASR-0.6B-Q8_0.gguf"
MMPROJ="mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"
TIMEOUT="${TIMEOUT:-600}"

command -v adb >/dev/null || { echo "error: adb not on PATH" >&2; exit 1; }

# Pin to the given cores if taskset exists on the device; else run unpinned.
# Thread limiting is llama.cpp's own -t flag, which caps its compute pool.
# Optional taskset pinning additionally binds those threads to specific cores.
# Toybox taskset takes a hex bitmask, not a core list. 4-7 -> 0xf0.
PIN=""
if [ -n "${PIN_CORES:-}" ]; then
    if adb shell "command -v taskset" >/dev/null 2>&1; then
        MASK="$(printf '%x' $(( (0xF << 4) )))"
        PIN="taskset $MASK"
        echo "==> pinning to cores $PIN_CORES (mask $MASK)"
    else
        echo "==> taskset not available on device; skipping pin"
    fi
fi

echo "==> binary=$BIN audio=$AUDIO threads=$THREADS timeout=${TIMEOUT}s"
echo

adb shell "cd $REMOTE && $PIN timeout $TIMEOUT ./$BIN \
    -m $MODEL --mmproj $MMPROJ --audio $AUDIO \
    -p 'Transcribe this audio exactly.' -ngl 99 -t $THREADS"
