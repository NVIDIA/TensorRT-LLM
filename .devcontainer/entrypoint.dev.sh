#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dev workflow dispatcher for TensorRT-LLM containers.
# Mount this script into the container and use it as entrypoint or CMD.
#
# Usage (via docker compose):
#   docker compose ... run --rm dev build-full
#   docker compose ... run --rm dev build-python
#   docker compose ... run --rm dev test tests/unittest/llmapi/test_llm_args.py
#   docker compose ... run --rm dev build-and-test tests/unittest/ -k "test_foo"
#   docker compose ... up -d          # uses default CMD "wait"
#   docker compose ... exec dev bash  # attach to running container

set -euo pipefail

CUDA_ARCH="${CUDA_ARCH:-90-real}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
CODE_DIR="${CODE_DIR:-/workspaces/tensorrt_llm}"
# Optional: override to isolate concurrent builds on the same source tree.
# Default (empty) lets build_wheel.py use cpp/build which is shared across
# containers that mount the same checkout — so ccache and cmake artifacts
# are reused automatically.  Set to a unique path (e.g. /tmp/build_sm90)
# only when running parallel builds for different configs.
BUILD_DIR="${BUILD_DIR:-}"

cd "$CODE_DIR"

# Ensure git trusts the mounted workspace
git config --global --get-all safe.directory 2>/dev/null | grep -Fxq "*" \
    || git config --global --add safe.directory "*"

# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

# Detect whether a C++ build has been done before (cmake artifacts exist).
_is_first_build() {
    [ ! -d "cpp/build/CMakeFiles" ]
}

_print_build_info() {
    echo "=== Build configuration ==="
    echo "  CUDA_ARCH   : $CUDA_ARCH"
    echo "  BUILD_JOBS  : $BUILD_JOBS"
    echo "  CODE_DIR    : $CODE_DIR"
    if _is_first_build; then
        echo "  Build type  : FIRST BUILD (no cmake cache found)"
        echo "  Expect      : ~30-60 min without ccache, much faster with"
    else
        echo "  Build type  : INCREMENTAL (cmake cache exists)"
    fi
    if command -v ccache &>/dev/null; then
        echo "  ccache stats:"
        ccache -s 2>/dev/null | grep -E "Hits|Misses|Cache size" | sed 's/^/    /'
    fi
    echo "==========================="
}

_build_cpp() {
    _print_build_info
    local extra_args=()
    # First build needs clean cmake configure; incremental reuses cache.
    # build_wheel.py handles this automatically via its first_build detection:
    #   first_build = not Path(build_dir, "CMakeFiles").exists()
    # So we just pass the same flags either way — cmake + ccache do the rest.
    python3 scripts/build_wheel.py \
        --use_ccache \
        -a "$CUDA_ARCH" \
        -j "$BUILD_JOBS" \
        --skip_building_wheel \
        --linking_install_binary \
        ${BUILD_DIR:+--build_dir "$BUILD_DIR"} \
        "$@"
    echo "C++ build complete. ccache stats after build:"
    ccache -s 2>/dev/null | grep -E "Hits|Misses|Cache size" | sed 's/^/  /' || true
}

_build_python() {
    pip install -e ".[devel]"
}

# Use precompiled binaries — skip C++ build entirely.  Useful when you only
# need to iterate on Python code and don't want to wait for a C++ build.
_build_precompiled() {
    echo "Installing with precompiled C++ binaries (skipping C++ build)..."
    TRTLLM_USE_PRECOMPILED=1 pip install -e ".[devel]"
}

# ---------------------------------------------------------------------------
# Command dispatcher
# ---------------------------------------------------------------------------

case "${1:-shell}" in
    build-full)
        # Full C++ + Python build.  First invocation does cmake configure +
        # full compile; subsequent invocations are incremental via ccache.
        shift
        _build_cpp "$@"
        _build_python
        ;;
    build-cpp)
        # C++ only (no pip install).  Use when iterating on C++/CUDA code
        # and you have already done pip install -e . once.
        shift
        _build_cpp "$@"
        ;;
    build-python)
        # Python-only reinstall.  Use after editing .py files when C++
        # artifacts are already present (from a prior build or precompiled).
        _build_python
        ;;
    build-precompiled)
        # Download precompiled C++ binaries and install in editable mode.
        # Fastest path for pure-Python development — no C++ build at all.
        _build_precompiled
        ;;
    test)
        shift
        pytest "$@"
        ;;
    build-and-test)
        shift
        _build_cpp
        _build_python
        pytest "$@"
        ;;
    wait)
        # Background mode — container stays alive, exec into it later.
        echo "Container ready. Attach with:"
        echo "  docker compose exec tensorrt_llm-dev bash"
        if _is_first_build; then
            echo ""
            echo "NOTE: No prior C++ build detected.  Run one of:"
            echo "  build-full        — full C++ + Python build (~30-60 min first time)"
            echo "  build-precompiled — download precompiled binaries (minutes)"
        fi
        exec sleep infinity
        ;;
    shell)
        if _is_first_build; then
            echo "WARNING: No C++ build found.  Run 'build-full' or 'build-precompiled' first."
        fi
        exec bash -l
        ;;
    *)
        exec "$@"
        ;;
esac
