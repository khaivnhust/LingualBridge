#!/bin/bash
# run_qdc.sh -- launch the full S2ST pipeline on QDC and measure it.
#
#   ASR : llama-server, Qwen3-ASR-1.7B Q4_K_M, Adreno GPU via OpenCL, port 8083
#   MT  : llama-server, Hy-MT2-1.8B Q6_K,     Hexagon HTP,        port 8084
#   TTS : Piper VITS via ONNX Runtime, in-process in the harness (CPU)
#
# Two llama-server processes, not one: OpenCL and HTP in the same process is not
# something we have ever measured, and the QDC exposes a single CDSP, so MT
# owns the NPU outright. Localhost HTTP costs ~1 ms against a multi-second budget.
#
# Server flags are copied from asr4/package/qdc-test.tar (run_asr.sh) rather than
# re-derived: -np 1 alone was worth 47 % RSS and 21 % RTF on the ASR server.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
QDC=${QDC:-$HERE}
BIN=$QDC/bin
LIB=$QDC/lib

ASR_PORT=${ASR_PORT:-8083}
MT_PORT=${MT_PORT:-8084}
ASR_MODEL=${ASR_MODEL:-$QDC/model/qwen3-asr-1.7b-4lang-q4km.gguf}
MMPROJ=${MMPROJ:-$QDC/model/mmproj-q8_0.gguf}
MT_MODEL=${MT_MODEL:-$QDC/model/hy-mt2-1.8b-q6_k.gguf}
TTS_ONNX=${TTS_ONNX:-$QDC/voices/piper_ko_female_v1.onnx}
TTS_MAP=${TTS_MAP:-$QDC/voices/piper_ko_female_v1.onnx.json}
AUDIO=${AUDIO:-$QDC/data/audio/factory_en.wav}
SRC=${SRC:-en}
TGT=${TGT:-ko}
TERMS=${TERMS:-}
RESULTS=${RESULTS:-$QDC/results/s2st_$(date +%Y%m%d-%H%M%S).jsonl}

for f in "$ASR_MODEL" "$MMPROJ" "$MT_MODEL"; do
    [ -f "$f" ] || { echo "missing model: $f" >&2; exit 1; }
done
[ -f "$AUDIO" ] || { echo "missing audio: $AUDIO" >&2; exit 1; }
export LD_LIBRARY_PATH="$LIB"

echo "=== bringing up ASR (Adreno/OpenCL) on :$ASR_PORT ==="
cd "$QDC"
# --cache-ram 0: the host-RAM prompt cache (default 8 GiB) stores state after
# every request; ASR never reuses a prompt, so it only grew -- 4.35 GB after a
# few hundred clips, enough to drive the phone into zram thrash (kswapd0 on top,
# MemAvailable 0). With it off the server stays flat at ~378 MB.
./bin/llama-server -m "$ASR_MODEL" --mmproj "$MMPROJ" \
    --host 127.0.0.1 --port "$ASR_PORT" -ngl 99 -c 2048 -np 1 --cache-ram 0 \
    > "$QDC/asr_server.log" 2>&1 < /dev/null &
ASR_PID=$!

# -np 1 on the MT server too: the DSP does not benefit from slots, and extra
# slots contend for the single CDSP.
echo "=== bringing up MT (Hexagon HTP) on :$MT_PORT ==="
./bin/llama-server -m "$MT_MODEL" \
    --host 127.0.0.1 --port "$MT_PORT" -ngl 99 -c 4096 -np 1 --cache-ram 256 \
    > "$QDC/mt_server.log" 2>&1 < /dev/null &
MT_PID=$!

cleanup() {
    echo ""
    echo "=== server logs (check for silent CPU fallback) ==="
    grep -iE "no usable GPU|fallback|htp|hexagon|skel|offload" "$QDC/asr_server.log" | head -5 | sed 's/^/  [asr] /'
    grep -iE "no usable GPU|fallback|htp|hexagon|skel|offload" "$QDC/mt_server.log" | head -5 | sed 's/^/  [mt]  /'
    kill "$ASR_PID" "$MT_PID" 2>/dev/null || true
}
trap cleanup EXIT

wait_http() {
    local port=$1 name=$2 tries=0
    while [ $tries -lt 90 ]; do
        if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            echo "  $name ready after ${tries}s"
            return 0
        fi
        sleep 1; tries=$((tries + 1))
    done
    echo "  $name FAILED to come up on :$port - see $QDC/*_server.log" >&2
    return 1
}
wait_http "$ASR_PORT" "ASR"
wait_http "$MT_PORT" "MT"

echo ""
echo "=== measuring ==="
mkdir -p "$(dirname "$RESULTS")" "$QDC/results/tts"
ARGS="--audio $AUDIO --asr-port $ASR_PORT --mt-port $MT_PORT --src $SRC --tgt $TGT --out $RESULTS"
[ -n "$TTS_ONNX" ] && [ -f "$TTS_ONNX" ] && ARGS="$ARGS --tts $TTS_ONNX --tts-map $TTS_MAP --save-tts $QDC/results/tts"
[ -n "$TERMS" ] && ARGS="$ARGS --terms \"$TERMS\""

# shellcheck disable=SC2086
./bin/qdc_s2st_bench $ARGS || true

echo ""
echo "=== A/B: same run with MT streaming disabled ==="
# The whole point of streaming MT is starting speech before the sentence ends.
# This second pass quantifies what it buys, on identical audio.
NOS="$RESULTS.nostream"
# shellcheck disable=SC2086
./bin/qdc_s2st_bench ${ARGS/--out $RESULTS/--out $NOS} --no-stream --warmup 0 || true

echo ""
echo "PULL THESE NOW (QDC instances are ephemeral):"
echo "  adb pull $QDC/results ./s2st-results-$(date +%Y%m%d)"
echo "  adb pull $QDC/asr_server.log $QDC/mt_server.log ."
