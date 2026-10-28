# On the phone: build, deploy, benchmark

Target: OnePlus 13 (Snapdragon 8 Elite, Adreno 830 GPU, Hexagon NPU), rooted with Magisk. Everything lives in
`/data/local/tmp/s2st-qdc` on the phone.

## 0. Ground rules

- **Always pass the serial**: `adb -s <serial> …`. A second phone is often plugged in, and plain `adb` then fails or hits
  the wrong device.
- **Keep it cool.** Use the cooling setup, pace long runs, and never let the NPU or CPU sit above about 90 °C
  (`harness/scripts/bench_eos.sh` waits for NSP < 85 °C and CPU < 90 °C before each clip). Uncooled constant load drove the NPU
  past 100 °C and the ASR decoder produced NaN garbage. Do not "test" heat on purpose.
- **Run device scripts as root** (`su -c`). The production servers run as root, and a shell-user script cannot stop
  them. The new server then fails to bind its port and **you silently benchmark the old one**.
- **Stop processes by binary name or PID**, not by matching command lines. `pkill -f <script>` inside `adb shell`
  matches and kills its own shell.
- **Detach long runs**: `(nohup sh script.sh > log 2>&1 < /dev/null &)`, then poll. A host-side `timeout` around
  `adb shell` kills the device script with it.
- **After rebuilding, check `md5sum` on host vs phone.** One whole benchmark ran on a stale binary.

## 1. Build llama.cpp for Android (OpenCL + Hexagon in one build)

Needs NDK r30, the Hexagon SDK **6.6.0.0** and OpenCL headers in the NDK sysroot (`harness/scripts/android/setup-ndk-opencl.sh`):

```bash
cmake -S llama.cpp -B llama.cpp/build-android-hex \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON -DGGML_HEXAGON=ON \
  -DHEXAGON_SDK_ROOT=/opt/hexagon/6.6.0.0 -DPREBUILT_LIB_DIR=android_aarch64
cmake --build llama.cpp/build-android-hex -j
```

Hexagon SDK traps (WORKLOG §12): pass `HEXAGON_SDK_ROOT` as a CMake `-D`, not an environment variable. `PREBUILT_LIB_DIR`
is mandatory. The SDK ships `qaic` at `qaic/Ubuntu/qaic` but the build looks in `qaic/bin/`, so inside the SDK's `qaic`
directory run `ln -s ../Ubuntu/qaic bin/qaic` (otherwise all four skel targets fail with error 127). The maintained alternative is
`llama.cpp/scripts/snapdragon/build.py` (containerised).

## 2. Build our binaries

```bash
# benchmark harness (needs ONNX Runtime 1.22 headers + the phone's libonnxruntime.so, espeak-ng built for Android)
cmake -S harness -B harness/build-android -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 -DCMAKE_BUILD_TYPE=Release \
  -DORT_INCLUDE=<ort>/include -DORT_LIB=<libonnxruntime-android-arm64.so> \
  -DESPEAK_INCLUDE=<espeak>/src/include -DESPEAK_LIB=<libespeak-ng.so>
cmake --build harness/build-android -j          # → qdc_s2st_bench

harness/scripts/build_mono.sh                     # single-process pipeline → s2st_mono
```


## 3. Package layout on the phone

`harness/scripts/build_package.sh` assembles it (edit its paths first):

```
/data/local/tmp/s2st-qdc/
  bin/    llama-server, qdc_s2st_bench_v2, s2st_mono, …
  lib/    libllama/ggml/OpenCL/Hexagon (+ libggml-htp-v*.so skels), libonnxruntime, libespeak-ng
  model/  qwen3-asr-1.7b-4lang-r6-q4km.gguf, mmproj-q8_0.gguf, hy-mt2-1.8b-q6_k.gguf
  voices/ Piper .onnx + .onnx.json
  espeak-ng-data/
  data/audio/{en,vi,zh,ko}_00..19.wav + audio_index.json   (80 bench clips with references)
```

Always `export LD_LIBRARY_PATH=$PWD/lib ADSP_LIBRARY_PATH=$PWD/lib` (the NPU needs the second one).

## 4. Start the servers

```bash
# ASR: decoder on NPU, audio encoder on GPU (best measured; see harness/README.md)
./bin/llama-server -m model/qwen3-asr-1.7b-4lang-r6-q4km.gguf --mmproj model/mmproj-q8_0.gguf \
  --host 127.0.0.1 --port 8083 -ngl 99 -c 2048 -np 1 -dev HTP0 -mmdev GPUOpenCL --cache-ram 0
#   all-GPU alternative: -dev GPUOpenCL (and no -mmdev)
#   NEVER: -dev HTP0 without -mmdev (the audio encoder then runs on the NPU and is broken)

# MT on the NPU
./bin/llama-server -m model/hy-mt2-1.8b-q6_k.gguf \
  --host 127.0.0.1 --port 8084 -ngl 99 -c 4096 -np 1 -dev HTP0 --cache-ram 256
```

`--cache-ram 0` on ASR is not optional: the default 8 GiB host cache grew to 4.4 GB and pushed the phone into swap
thrash. The "failed to fit params … HTP0" warning is harmless (the NPU does not report its memory).

## 5. Benchmark

`harness/scripts/bench_eos.sh` is **the** benchmark: 80 clips, per-language policy, ASR decoder + MT on the NPU and the
audio encoder on the GPU, with a thermal wait (NSP < 85 °C, CPU < 90 °C) before each clip. Copy it to `scripts/` on the
phone and run it as root, detached. To compare a variant, copy it and change one thing (server flags, policy flags).

It writes `eos/Gs.jsonl` (one row per utterance) and `eos/split_temps.log`. Pull them, and summarise against the
references. `eos_report.py` reads `A.jsonl` / `B.jsonl` / `C.jsonl` from a folder, so name the run (and an optional
baseline run to compare with) accordingly:

```bash
mkdir run && adb -s <serial> pull /data/local/tmp/s2st-qdc/eos/Gs.jsonl run/C.jsonl
adb -s <serial> pull /data/local/tmp/s2st-qdc/data/audio_index.json run/   # references, made by make_bench_data.py
python3 harness/scripts/eos_report.py run run/audio_index.json
```

Each JSONL row carries `last_to_mt1_ms` (the headline: last speech sample → first MT token),
`eos_to_voice_from_last_speech_ms` (→ first audible word), `asr_ms`, `mt_first_token_ms`, `mt_first_clause_ms`,
and the ASR/MT text.

## 6. Phone housekeeping already in place

- **Swap**: `/data/swapfile` (10 GB) is the only swap; zram is off. Set up by `/data/adb/service.d/swapfile.sh`.
- **Battery**: charging is held between 60 and 80 % by `/data/adb/service.d/charge_limit.sh`, so the phone runs from USB
  with less heat. Log: `/data/local/tmp/charge_limit.log`.
- **Max-performance mode**: `harness/scripts/root/bench_mode.sh` (stops the UI, background apps and radios; **keeps kernel thermal
  zones on**) and `harness/scripts/root/restore.sh` to undo it.
- **Mic and speaker** work from a native binary through AAudio (NDK `libaaudio`, no app needed). Request 48 kHz output
  for the low-latency path.
