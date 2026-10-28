#!/usr/bin/env bash
# Provide a cmake package config for SPIRV-Headers.
#
# The LunarG Vulkan SDK ships the SPIRV-Headers sources under include/spirv but
# no SPIRV-HeadersConfig.cmake, which ggml's Vulkan backend requires. This
# installs the headers plus a minimal shim config pointing at them.
#
# Usage:
#   scripts/setup-spirv-headers.sh [sdk-x86_64-dir]
set -euo pipefail

SDK="${1:-/tmp/vktools/1.4.363.0/x86_64}"
PREFIX="${SPIRV_PREFIX:-$HOME/.local/spirv-headers}"

if [ ! -d "$SDK/include/spirv" ]; then
    echo "error: no include/spirv under $SDK" >&2
    echo "       pass the extracted Vulkan SDK x86_64 directory" >&2
    exit 1
fi

rm -rf "$PREFIX"
mkdir -p "$PREFIX/include" "$PREFIX/share/spirv-headers"
cp -r "$SDK/include/spirv" "$PREFIX/include/spirv"

cat > "$PREFIX/share/spirv-headers/SPIRV-HeadersConfig.cmake" <<'EOF'
# Shim for SPIRV-Headers, which ships headers but no cmake config.
get_filename_component(_SPIRV_HEADERS_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(SPIRV-Headers_INCLUDE_DIR "${_SPIRV_HEADERS_PREFIX}/include")
add_library(SPIRV-Headers INTERFACE IMPORTED)
set_target_properties(SPIRV-Headers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SPIRV-Headers_INCLUDE_DIR}")
set(SPIRV-Headers_FOUND TRUE)
EOF

echo "installed SPIRV-Headers -> $PREFIX"
find "$PREFIX" -name spirv.hpp | head -1
