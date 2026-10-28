#!/bin/bash
# setup.sh -- prepare the S2ST package on the device.
#
# Two failures are specific to pushing a tar over adb, and both have wasted a
# QDC instance before:
#   1. exec bits are lost, so "Permission denied" on a file that is plainly
#      present on disk is this bug, not a missing file
#   2. the NDK's libOpenCL.so is 44 MB, SONAME libGLES_mali.so, and needs vendor
#      libraries that may be absent -- use the device's own loader from
#      /vendor/lib64 instead
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)

echo "=== S2ST package setup in $HERE ==="
chmod +x "$HERE"/bin/* "$HERE"/scripts/*.sh 2>/dev/null || true

if [ ! -e "$HERE/lib/libOpenCL.so" ]; then
    for p in /vendor/lib64/libOpenCL.so /vendor/lib64/libOpenCL.so.1; do
        if [ -f "$p" ]; then
            cp "$p" "$HERE/lib/libOpenCL.so"
            echo "  libOpenCL.so <- $p ($(stat -c%s "$p") bytes)"
            break
        fi
    done
fi
[ -f "$HERE/lib/libOpenCL.so" ] || { echo "FATAL: no libOpenCL.so and none on the device" >&2; exit 1; }

echo ""
echo "=== inventory ==="
printf "  %-34s %s\n" "llama-server"    "$([ -x "$HERE/bin/llama-server" ] && echo ok || echo MISSING)"
printf "  %-34s %s\n" "qdc_s2st_bench"  "$([ -x "$HERE/bin/qdc_s2st_bench" ] && echo ok || echo MISSING)"
printf "  %-34s %s\n" "ASR model"       "$([ -f "$HERE/model/qwen3-asr-1.7b-4lang-q4km.gguf" ] && echo ok || echo MISSING)"
printf "  %-34s %s\n" "mmproj"          "$([ -f "$HERE/model/mmproj-q8_0.gguf" ] && echo ok || echo MISSING)"
printf "  %-34s %s\n" "MT model"        "$([ -f "$HERE/model/hy-mt2-1.8b-q6_k.gguf" ] && echo ok || echo MISSING)"
printf "  %-34s %s\n" "libonnxruntime"  "$([ -f "$HERE/lib/libonnxruntime.so" ] && echo ok || echo MISSING)"
printf "  %-34s %s\n" "libespeak-ng"    "$([ -f "$HERE/lib/libespeak-ng.so" ] && echo ok || echo MISSING)"
printf "  %-34s %s\n" "espeak-ng-data"  "$([ -d "$HERE/share/espeak-ng-data" ] && echo ok || echo MISSING)"
printf "  %-34s %s\n" "benchmark audio" "$(ls "$HERE"/data/audio/*.wav 2>/dev/null | wc -l) wav"

echo ""
echo "=== backends in lib/ ==="
ls "$HERE"/lib/ | grep -E 'ggml-(opencl|hexagon|htp)' | sed 's/^/  /'
echo ""
echo "  (4 HTP skels = DSP generations v73/v75/v79/v81; one is picked at load time)"
echo ""
echo "Verify the manifest with:  cd $HERE && sha256sum -c MANIFEST.sha256 --quiet"
echo "Next:  sh scripts/run_s2st.sh"
