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
#
# TODO(TRTLLM-15310): evaluate a stronger type checker than mypy. mypy is the
# reference implementation of the typing spec, not the most capable checker:
# pyright has better inference and narrowing, fuller generics/ParamSpec/overload
# handling, and is considerably faster. ty (Astral, aligns with the ruff already
# used here) and pyrefly (Meta) target the same niche and are worth re-checking
# for maturity. Migration cost is bounded — [tool.mypy] files in pyproject.toml
# is a short explicit list, so a candidate can be run over the same set as a
# non-blocking stage first to see what it catches that mypy does not.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Temporary directory holding the CUDA driver stub, if one is needed. Removed on
# exit; note that mypy below is therefore invoked WITHOUT exec, so this trap runs.
STUB_DIR=""
cleanup_stub_dir() {
    if [[ -n "$STUB_DIR" ]]; then
        rm -rf "$STUB_DIR"
    fi
}
trap cleanup_stub_dir EXIT

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
    local stub=""
    local candidate
    for candidate in /usr/local/cuda*/targets/*/lib/stubs/libcuda.so \
                     /usr/local/cuda*/lib64/stubs/libcuda.so; do
        # Unmatched globs stay literal; the [[ -f ]] test rejects them.
        if [[ -f "$candidate" ]]; then
            stub="$candidate"
            break
        fi
    done
    if [[ -z "$stub" ]]; then
        return  # Nothing to offer; the import below reports the real error.
    fi
    # A fresh owner-only directory per invocation, not a fixed path: everything
    # on LD_LIBRARY_PATH gets loaded into this process, so a predictable
    # directory under a shared /tmp would let any local user pre-create it and
    # plant their own libcuda.so.1. Removed on exit by the trap above.
    # The stub must be visible under its SONAME, libcuda.so.1.
    STUB_DIR="$(mktemp -d)"
    ln -s "$stub" "$STUB_DIR/libcuda.so.1"
    export LD_LIBRARY_PATH="$STUB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "No CUDA driver on this machine; using the stub at $stub so the" \
         "compiled bindings can be loaded for type checking"
}

ensure_cuda_driver_stub

# ============================================================================
# TEMPORARY DIAGNOSTIC — REMOVE BEFORE MERGING (DLFW 26.08 upgrade)
#
# On the 26.08 base image the import below dies at tensorrt_llm/_utils.py with
#   mpi4py.MPI.Exception: MPI_ERR_OTHER: known error not in list
# from `mpi_comm().Split_type(split_type=OMPI_COMM_TYPE_HOST)`.
#
# The same call succeeds in the same container on a SLURM node under every
# environment tried (with/without SLURM vars, with CI's env replicated), so the
# trigger is specific to this CI pod. This block runs the call here, where it
# actually fails, to capture (a) whether the standard MPI_COMM_TYPE_SHARED
# fails too and (b) the underlying PMIx error that MPI_ERR_OTHER is hiding.
# Never fails the build — the real gate is still the import below.
# ============================================================================
echo "=== [TEMP] MPI Split_type diagnostic ==============================="
python3 - <<'PY' 2>&1 | sed 's/^/[TEMP] /' || true
import os
import traceback

print("libmpi resolution for the built bindings:")
os.system("ldd tensorrt_llm/bindings*.so 2>/dev/null "
          "| grep -iE 'libmpi|libopen-pal|libpmix' || echo '  (none found)'")

def matrix(tag):
    from mpi4py import MPI
    from mpi4py.util import pkl5
    print(f"--- {tag} ---")
    print("  lib :", MPI.Get_library_version().strip().splitlines()[0])
    print("  size:", MPI.COMM_WORLD.Get_size())
    for name, comm in (("raw ", MPI.COMM_WORLD),
                       ("pkl5", pkl5.Intracomm(MPI.COMM_WORLD))):
        for label, st in (("SHARED(0)", MPI.COMM_TYPE_SHARED), ("HOST(9)", 9)):
            try:
                c = comm.Split_type(split_type=st)
                print(f"  {name} {label:10s} -> OK   size={c.Get_size()}")
            except Exception as exc:                     # noqa: BLE001
                print(f"  {name} {label:10s} -> FAIL {type(exc).__name__}: {exc}")

try:
    matrix("A: mpi4py only")
except Exception:                                        # noqa: BLE001
    traceback.print_exc()
PY

echo "=== [TEMP] same, but with torch imported first (as tensorrt_llm does) ==="
python3 - <<'PY' 2>&1 | tail -25 | sed 's/^/[TEMP] /' || true
import torch
print("torch", torch.__version__)
from mpi4py import MPI
from mpi4py.util import pkl5
for name, comm in (("raw ", MPI.COMM_WORLD),
                   ("pkl5", pkl5.Intracomm(MPI.COMM_WORLD))):
    for label, st in (("SHARED(0)", MPI.COMM_TYPE_SHARED), ("HOST(9)", 9)):
        try:
            c = comm.Split_type(split_type=st)
            print(f"  {name} {label:10s} -> OK   size={c.Get_size()}")
        except Exception as exc:                         # noqa: BLE001
            print(f"  {name} {label:10s} -> FAIL {type(exc).__name__}: {exc}")
PY

echo "=== [TEMP] verbose PMIx/OMPI for the failing Split_type ============"
PMIX_MCA_pmix_base_verbose=5 \
OMPI_MCA_odls_base_verbose=5 \
OMPI_MCA_ess_base_verbose=5 \
python3 -c "
from mpi4py import MPI
MPI.COMM_WORLD.Split_type(split_type=9)
" 2>&1 | tail -60 | sed 's/^/[TEMP] /' || true
echo "=== [TEMP] end diagnostic ========================================="

# What the gate below does and does not establish.
#
# `import tensorrt_llm.bindings` is one coarse check standing in for several
# conditions that fail independently. Worth naming, because "bindings importable"
# reads narrower than what is actually being relied on:
#
#   - the extension was built at all;
#   - the compiled artifacts agree with each other — a bindings.*.so built
#     against a different libth_common.so fails here with an undefined symbol.
#     This is the main reason the check is an import rather than a file test;
#   - the runtime dependencies are installed, so mypy reads real inline types
#     (torch and friends) instead of collapsing them to Any.
#
# It establishes nothing about the .pyi stubs — which are what mypy actually
# consumes, since mypy never imports anything. The bindings stubs are generated
# at build time (tensorrt_llm/bindings/*.pyi) and are not checked in, so a
# stub-generation regression would leave this gate green while the checked code
# is silently compared against a missing or stale API. Asserting stub presence
# separately would close that hole; it is deliberately left as a follow-up
# rather than folded into this check, since the two catch disjoint failures.
if python3 -c 'import tensorrt_llm.bindings'; then
    echo "Compiled bindings detected — running full mypy type check"
    mypy "$@"
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
    mypy \
        --ignore-missing-imports \
        --no-warn-return-any \
        --no-warn-unused-ignores \
        --allow-untyped-decorators \
        --allow-subclassing-any \
        --no-install-types \
        --interactive \
        "$@"
fi
