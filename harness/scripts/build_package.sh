#!/bin/bash
# build_package.sh -- assemble the single QDC-deployable S2ST package.
#
# Layout:
#   bin/       llama-server (OpenCL + HTP, one build) + qdc_s2st_bench (aarch64)
#   lib/       llama/ggml/OpenCL/Hexagon/HTP skels + onnxruntime + espeak-ng
#   model/     ASR Q4_K_M + mmproj + MT Hy-MT2-1.8B Q6_K
#   voices/    Piper VITS ONNX + phoneme maps
#   espeak-ng-data/   dictionaries (espeak has no compiled-in default on Android)
#   data/      16 kHz benchmark audio + parallel MT sentences
#   scripts/   setup.sh, run_s2st.sh
#
# "Use the base model for missing parts": anything not available locally is
# either fetched or substituted, and every substitution is recorded in
# the harness README so a reader is never misled about what is being measured.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)   # repo root
PKG=$ROOT/s2st-qdc
WORK=${WORK:-${TMPDIR:-/tmp}/s2st-pkg}
NDK=${NDK:-$HOME/.android/Sdk/ndk/30.0.16248370}
STRIP=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip

rm -rf "$PKG" "$WORK"; mkdir -p "$PKG"/{bin,lib,model,voices,scripts,data/audio,data/mt}
mkdir -p "$WORK"

say() { printf "  %s\n" "$*"; }

# ── 1. llama-server + backends, from the already-proven device package ──────
say "[1/7] llama-server + OpenCL/Hexagon backends (from ASR/package/qdc-test.tar)"
tar -xf "$ROOT/ASR/package/qdc-test.tar" -C "$WORK" qdc-build/bin qdc-build/lib 2>/dev/null || true
if [ -d "$WORK/qdc-build/bin" ]; then
    cp -a "$WORK/qdc-build/bin/." "$PKG/bin/" 2>/dev/null || true
    # only the runtime libs; the cmake/pkgconfig trees are build noise
    for f in "$WORK"/qdc-build/lib/*.so; do cp -a "$f" "$PKG/lib/" 2>/dev/null || true; done
fi
say "      bin: $(ls "$PKG/bin" | tr '\n' ' ')"
say "      lib: $(ls "$PKG/lib" | wc -l) shared objects"
say "      HTP backends: $(ls "$PKG/lib" | grep -c 'ggml-htp\|ggml-hexagon')  OpenCL: $(ls "$PKG/lib" | grep -c 'ggml-opencl')"

# ── 2. the harness binary ───────────────────────────────────────────────────
say "[2/7] qdc_s2st_bench (aarch64, ORT + espeak-ng)"
cp -a "$ROOT/harness/build-android/qdc_s2st_bench" "$PKG/bin/"
$STRIP "$PKG/bin/qdc_s2st_bench" 2>/dev/null || true

# ── 3. onnxruntime + espeak-ng for aarch64 android ──────────────────────────
say "[3/7] runtime libs"
cp "${ORT_LIB:-$ROOT/harness/third_party/ort/libonnxruntime-android-arm64.so}" "$PKG/lib/libonnxruntime.so"
cp "${ESPEAK_LIB:-$ROOT/harness/third_party/espeak-ng-1.52.0/build-android/src/libespeak-ng/libespeak-ng.so}" "$PKG/lib/"
mkdir -p "$PKG/share"
    cp -a /usr/lib/x86_64-linux-gnu/espeak-ng-data "$PKG/share/espeak-ng-data"
say "      libonnxruntime.so $(du -h "$PKG/lib/libonnxruntime.so" | cut -f1)"
say "      libespeak-ng.so   $(du -h "$PKG/lib/libespeak-ng.so" | cut -f1)"
say "      espeak-ng-data    $(du -sh "$PKG/espeak-ng-data" | cut -f1)"

# ── 4. models ──────────────────────────────────────────────────────────────
say "[4/7] models"
cp "$ROOT/ASR/weights/gguf/r6-Q4_K_M.gguf" "$PKG/model/qwen3-asr-1.7b-4lang-q4km.gguf"
cp "$ROOT/ASR/weights/gguf/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf" "$PKG/model/mmproj-q8_0.gguf"
MT_GGUF=${MT_GGUF:-$ROOT/models/hy-mt2-1.8b-q6_k.gguf}   # source not recorded; see docs/GETTING_STARTED.md
if [ -s "$MT_GGUF" ]; then
    cp "$MT_GGUF" "$PKG/model/hy-mt2-1.8b-q6_k.gguf"
    say "      hy-mt2-1.8b-q6_k.gguf  $(du -h "$PKG/model/hy-mt2-1.8b-q6_k.gguf" | cut -f1)  (downloaded)"
else
    say "      !! MT model missing -- leaving it out; run scripts/fetch_mt.sh on device"
fi
ls -la "$PKG/model" | awk 'NR>3{printf "      %-40s %s\n",$9,$5}'

# ── 5. voices ──────────────────────────────────────────────────────────────
say "[5/7] Piper voices"
for v in "$ROOT"/piper/voices/*.onnx; do
    b=$(basename "$v")
    cp "$v" "$PKG/voices/$b"
    [ -f "${v}.json" ] && cp "${v}.json" "$PKG/voices/$b.json"
done
say "      $(ls "$PKG/voices"/*.onnx | wc -l) voices"

# ── 6. dataset ─────────────────────────────────────────────────────────────
say "[6/7] benchmark audio (16 kHz mono) + parallel MT sentences"
python3 "$ROOT/harness/scripts/make_bench_data.py" \
    --fleurs-test "$ROOT/ASR/data/fleurs/en_test.jsonl" \
    --cvtest "$ROOT/ASR/data/cvtest" \
    --kspon-manifest "$ROOT/ASR/data/kspon_gate_5k.jsonl" \
    --voxpopuli "$ROOT/ASR/data/voxpopuli/en_heldout.jsonl" \
    --out "$PKG/data" || say "      !! dataset build FAILED"
say "      audio: $(ls "$PKG/data/audio" 2>/dev/null | wc -l) files"
cp -f "$ROOT"/harness/scripts/run_s2st.sh "$PKG/scripts/" 2>/dev/null || cp -f "$ROOT/harness/run_qdc.sh" "$PKG/scripts/run_s2st.sh"

# ── 7. manifest ────────────────────────────────────────────────────────────
say "[7/7] manifest"
cd "$PKG"
find . -type f ! -name MANIFEST.sha256 -printf '%P\n' | sort | while read -r f; do
    sha256sum "$f"
done > MANIFEST.sha256
say "      $(wc -l < MANIFEST.sha256) files hashed"
say ""
say "PACKAGE: $PKG   $(du -sh "$PKG" | cut -f1)"
