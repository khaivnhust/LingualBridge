#!/bin/bash
# Build src/coreside.cc for the phone against the SAME llama.cpp libs it runs
# (llama.cpp/build-android-hex == /data/local/tmp/s2st-qdc/lib, md5-identical).
set -euo pipefail
ROOT=/home/may/Projects/AI-Translator
L=$ROOT/llama.cpp
NDK=/home/may/.android/Sdk/ndk/30.0.16248370
CXX=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang++
mkdir -p $ROOT/s2st-bench/build-android
$CXX -std=c++17 -O2 $ROOT/s2st-bench/src/coreside.cc \
  -I$L/include -I$L/ggml/include -I$L/tools/mtmd \
  -L$L/build-android-hex/bin -lllama -lmtmd -lggml -lggml-base \
  -o $ROOT/s2st-bench/build-android/s2st_coreside
echo built $ROOT/s2st-bench/build-android/s2st_coreside
