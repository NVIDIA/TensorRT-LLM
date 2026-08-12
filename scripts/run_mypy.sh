#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Wrapper for mypy invoked by the pre-commit type-check hook.
#
# When the compiled TensorRT-LLM bindings are present (bindings.*.so),
# performs a full type check including automatic type-stub installation.
# Otherwise, runs a lightweight check that tolerates missing compiled
# modules so that developers can type-check without building the wheel.
#
# Set MYPY_REQUIRE_BINDINGS=1 to fail when the compiled bindings cannot be
# imported (used by the build stage to enforce the full check after compilation).
#
# Importing the bindings needs no GPU, but it does need libcuda.so.1 to resolve;
# on a driverless machine this script falls back to the CUDA stub library (see
# ensure_cuda_driver_stub below).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Make the bindings loadable on a machine with no CUDA driver installed.
#
# bindings.*.so links against the CUDA driver API (the cuMem*/cuMulticast*
# virtual-memory and NVLink-multicast paths), so it carries a DT_NEEDED on
# libcuda.so.1. DT_NEEDED is resolved when the loader maps the library, before
# any of its code runs, so dlopen fails outright where no driver is installed —
# CPU build nodes, for instance. Nothing on the import path calls a driver
# function or needs a device; only symbol resolution does. The CUDA toolkit
# ships a stub libcuda.so for exactly this case, so point the loader at it when
# no real driver is present. Verified on a driverless node: with the stub the
# bindings module imports and initializes cleanly, and torch correctly reports
# zero devices.
ensure_cuda_driver_stub() {
    if python3 -c "import ctypes; ctypes.CDLL('libcuda.so.1')" 2>/dev/null; then
        return  # A real driver is installed; leave the loader path alone.
    fi
    local stub
    stub=$(ls /usr/local/cuda*/targets/*/lib/stubs/libcuda.so \
              /usr/local/cuda*/lib64/stubs/libcuda.so 2>/dev/null | head -n 1)
    if [[ -z "$stub" ]]; then
        return  # Nothing to offer; the import below reports the real error.
    fi
    # Fixed path, so repeated runs reuse one directory instead of leaking one
    # per invocation. The stub must be seen under its SONAME, libcuda.so.1.
    local stub_dir="${TMPDIR:-/tmp}/trtllm-mypy-cuda-stub"
    mkdir -p "$stub_dir"
    ln -sf "$stub" "$stub_dir/libcuda.so.1"
    export LD_LIBRARY_PATH="$stub_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "No CUDA driver on this machine; using the stub at $stub so the" \
         "compiled bindings can be loaded for type checking"
}

ensure_cuda_driver_stub

if python3 -c 'import tensorrt_llm.bindings'; then
    echo "Compiled bindings detected — running full mypy type check"
    exec mypy "$@"
else
    if [[ "${MYPY_REQUIRE_BINDINGS:-0}" -eq 1 ]]; then
        echo "ERROR: MYPY_REQUIRE_BINDINGS=1 but 'import tensorrt_llm.bindings'" \
             "failed (traceback above)." >&2
        # The traceback is authoritative; these are hints for the two failures
        # that look alike from the outside (never built vs. built but unloadable).
        if compgen -G "$PROJECT_DIR/tensorrt_llm/bindings*.so" > /dev/null; then
            echo "       note: bindings*.so IS present, so the extension was built" \
                 "and something stopped it from importing. If the traceback names" \
                 "libcuda.so.1, this machine has no CUDA driver and no stub" \
                 "libcuda.so was found under /usr/local/cuda*/**/stubs/." >&2
        else
            echo "       note: no bindings*.so under $PROJECT_DIR/tensorrt_llm," \
                 "so the extension was probably not built." >&2
        fi
        exit 1
    fi
    echo "No compiled bindings — running lightweight mypy type check"
    # Without installed dependencies and/or compiled bindings every corresponding type
    # resolves to Any. The flags below suppress the strict-mode checks that cascade
    # from this:
    #   --ignore-missing-imports     → silences import errors for the .so modules
    #   --no-warn-return-any         → [no-any-return] on functions returning Any
    #   --no-warn-unused-ignores     → [unused-ignore] on "# type: ignore"
    #                                  comments that become unnecessary
    #   --allow-untyped-decorators   → [misc] on decorators whose type is Any
    #   --allow-subclassing-any      → [misc] on classes inheriting from Any
    exec mypy \
        --ignore-missing-imports \
        --no-warn-return-any \
        --no-warn-unused-ignores \
        --allow-untyped-decorators \
        --allow-subclassing-any \
        --no-install-types \
        --interactive \
        "$@"
fi
