cmake_minimum_required(VERSION 3.16)
project(s2st_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── ONNX Runtime ─────────────────────────────────────────────────────────────
# Point at an Android aarch64 build of onnxruntime for device builds. For host
# testing, -DONNXRUNTIME_ROOT=/usr will find the pip package.
set(ONNXRUNTIME_ROOT "" CACHE PATH "onnxruntime install prefix (contains include/ and lib/)")
if(NOT ONNXRUNTIME_ROOT)
  execute_process(COMMAND python3 -c "import onnxruntime,os;print(os.path.dirname(onnxruntime.__file__))"
                  OUTPUT_VARIABLE _ort_py OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET RESULT_VARIABLE _ort_ok)
  if(_ort_ok EQUAL 0)
    execute_process(COMMAND python3 -c "import onnxruntime,os;print(os.path.dirname(os.path.dirname(onnxruntime.__file__)))"
                    OUTPUT_VARIABLE ONNXRUNTIME_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
endif()
message(STATUS "ONNXRUNTIME_ROOT = ${ONNXRUNTIME_ROOT}")

# ── espeak-ng (optional, enables TEXT phonemisation) ─────────────────────────
set(ESPEAK_ROOT "" CACHE PATH "espeak-ng install prefix")
# The header is speak_lib.h, NOT speak.h -- espeak-ng renamed it and there is no
# espeak-ng/speak.h anywhere. Searching for the old name silently disables TEXT
# phonemisation and drops the build to IDS mode with only a warning.
set(ESPEAK_INCLUDE "" CACHE FILEPATH "path to espeak-ng/speak_lib.h")
set(ESPEAK_LIB "" CACHE FILEPATH "path to libespeak-ng.so")
if(NOT ESPEAK_INCLUDE)
  find_path(ESPEAK_INCLUDE_FOUND espeak-ng/speak_lib.h
            HINTS ${ESPEAK_ROOT}/include /usr/include /usr/local/include)
  set(ESPEAK_INCLUDE ${ESPEAK_INCLUDE_FOUND})
endif()
if(NOT ESPEAK_LIB)
  find_library(ESPEAK_LIB_FOUND espeak-ng
               HINTS ${ESPEAK_ROOT}/lib /usr/lib /usr/local/lib)
  set(ESPEAK_LIB ${ESPEAK_LIB_FOUND})
endif()
message(STATUS "ESPEAK_INCLUDE = ${ESPEAK_INCLUDE}")
message(STATUS "ESPEAK_LIB    = ${ESPEAK_LIB}")

add_executable(qdc_s2st_bench
  src/main.cc
  src/http.cc
  src/vad.cc
  src/tts.cc)

target_include_directories(qdc_s2st_bench PRIVATE src)

find_package(Threads REQUIRED)
target_link_libraries(qdc_s2st_bench PRIVATE Threads::Threads)

# onnxruntime: accept explicit paths (required for cross-compiles, where
# find_library often fails against a .so named for another ABI), else discover.
set(ORT_INCLUDE "" CACHE FILEPATH "onnxruntime_cxx_api.h path")
set(ORT_LIB "" CACHE FILEPATH "libonnxruntime.so path")
if(NOT ORT_INCLUDE OR NOT ORT_LIB)
  if(NOT ORT_LIB)
    find_library(ORT_LIB_FOUND onnxruntime
                 HINTS ${ONNXRUNTIME_ROOT}/lib ${ONNXRUNTIME_ROOT}/lib64 ${ONNXRUNTIME_ROOT})
    set(ORT_LIB ${ORT_LIB_FOUND})
  endif()
  if(NOT ORT_INCLUDE)
    find_path(ORT_INCLUDE_FOUND onnxruntime_cxx_api.h
              HINTS ${ONNXRUNTIME_ROOT}/include /usr/include /usr/local/include)
    set(ORT_INCLUDE ${ORT_INCLUDE_FOUND})
  endif()
endif()
if(NOT ORT_LIB OR NOT ORT_INCLUDE)
  message(FATAL_ERROR "onnxruntime not found. Pass -DONNXRUNTIME_ROOT=<prefix> "
                      "or -DORT_INCLUDE=<...>/onnxruntime_cxx_api.h -DORT_LIB=<...>/libonnxruntime.so")
endif()
message(STATUS "ORT_INCLUDE = ${ORT_INCLUDE}")
message(STATUS "ORT_LIB    = ${ORT_LIB}")
target_include_directories(qdc_s2st_bench PRIVATE ${ORT_INCLUDE})
target_link_libraries(qdc_s2st_bench PRIVATE ${ORT_LIB})

if(ESPEAK_INCLUDE AND ESPEAK_LIB)
  message(STATUS "espeak-ng FOUND -> TEXT mode enabled (real streaming phonemisation)")
  target_compile_definitions(qdc_s2st_bench PRIVATE S2ST_WITH_ESPEAK)
  target_include_directories(qdc_s2st_bench PRIVATE ${ESPEAK_INCLUDE})
  target_link_libraries(qdc_s2st_bench PRIVATE ${ESPEAK_LIB})
else()
  message(WARNING "espeak-ng NOT found -> build IDS mode only. Use --tts-ids, or "
                  "install libespeak-ng-dev and reconfigure.")
endif()

# aarch64 device build
if(CMAKE_SYSTEM_NAME STREQUAL "Android" OR ANDROID)
  set_target_properties(qdc_s2st_bench PROPERTIES
    LINK_FLAGS "-Wl,-rpath,'$ORIGIN/../lib'")
endif()

install(TARGETS qdc_s2st_bench RUNTIME DESTINATION bin)

# ── selftest: VAD / chunker / JSON / WAV, no models or device required ───────
# The harness's whole value is its timing, and the timing anchor is the VAD plus
# the clause chunker. Test them before burning a QDC instance.
add_executable(s2st_selftest src/selftest.cc src/http.cc src/vad.cc)
target_include_directories(s2st_selftest PRIVATE src)
target_link_libraries(s2st_selftest PRIVATE Threads::Threads)
