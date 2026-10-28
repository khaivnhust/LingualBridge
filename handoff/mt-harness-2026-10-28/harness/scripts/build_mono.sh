#!/bin/bash
# Build src/mono.cc (whole pipeline in one process) for the phone, against the
# same llama.cpp libs it runs (llama.cpp/build-android-hex == device lib/).
set -euo pipefail
ROOT=/home/may/Projects/AI-Translator
B=$ROOT/s2st-bench
L=$ROOT/llama.cpp
NDK=/home/may/.android/Sdk/ndk/30.0.16248370
CXX=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang++
ESPEAK=$B/third_party/espeak-ng-1.52.0
ORT=$B/third_party/ort
mkdir -p $B/build-android
$CXX -std=c++17 -O2 -DS2ST_WITH_ESPEAK \
  $B/src/mono.cc $B/src/tts.cc $B/src/vad.cc $B/src/http.cc \
  -I$B/src -I$L/include -I$L/ggml/include -I$L/tools/mtmd \
  -I$ORT/onnxruntime-linux-x64-1.22.0/include -I$ESPEAK/src/include \
  -L$L/build-android-hex/bin -lllama -lmtmd -lggml -lggml-base \
  $ORT/libonnxruntime-android-arm64.so $ESPEAK/build-android/src/libespeak-ng/libespeak-ng.so \
  -o $B/build-android/s2st_mono
echo built $B/build-android/s2st_mono
