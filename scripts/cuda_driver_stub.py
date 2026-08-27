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
"""Make CUDA-driver-linked modules importable on a driverless (CPU) machine.

``import tensorrt_llm`` (and the compiled ``bindings.*.so`` it pulls in) links
against the CUDA driver API, so it carries a ``DT_NEEDED`` on ``libcuda.so.1``.
That dependency is resolved by the loader when the library is mapped, before any
of its code runs, so the import fails outright on a node with no CUDA driver
installed -- CPU build/test nodes, for instance. Nothing on the import path
calls a driver function or needs a device; only symbol resolution does. The CUDA
toolkit ships a stub ``libcuda.so`` for exactly this case, so point the loader at
it when no real driver is present.

The stub must be on ``LD_LIBRARY_PATH`` *before* the process that does the import
starts (glibc reads it at startup), so the shell entry point below emits the
value for a caller to export ahead of the importing process, e.g. in the CI doc
build:

    export LD_LIBRARY_PATH=$(python3 scripts/cuda_driver_stub.py)
    make html   # sphinx-build now imports tensorrt_llm cleanly

The same logic is available as :func:`ensure_cuda_driver_stub` for in-process
Python callers (e.g. a future mypy type-check wrapper).
"""

import atexit
import ctypes
import glob
import os
import shutil
import sys
import tempfile

# Where the CUDA toolkit places the driver stub. The aarch64/x86_64 targets
# layout comes first; the flat lib64 layout is the fallback.
_STUB_GLOBS = (
    "/usr/local/cuda*/targets/*/lib/stubs/libcuda.so",
    "/usr/local/cuda*/lib64/stubs/libcuda.so",
)


def _real_driver_present() -> bool:
    """True if a real libcuda.so.1 (the installed driver) can be loaded."""
    try:
        ctypes.CDLL("libcuda.so.1")
        return True
    except OSError:
        return False


def _find_stub() -> str | None:
    """Return the path to a toolkit stub libcuda.so, or None if none exists."""
    for pattern in _STUB_GLOBS:
        for candidate in sorted(glob.glob(pattern)):
            if os.path.isfile(candidate):
                return candidate
    return None


def ensure_cuda_driver_stub(register_cleanup: bool = True) -> str | None:
    """Put the CUDA toolkit's stub libcuda.so on LD_LIBRARY_PATH if needed.

    No-op (returns None) when a real driver is installed or no stub is found.
    Otherwise symlinks the stub as ``libcuda.so.1`` into a fresh, owner-only
    directory, prepends that directory to ``LD_LIBRARY_PATH``, and returns the
    directory path.

    register_cleanup removes the directory at interpreter exit; pass False when
    a separate process must keep using LD_LIBRARY_PATH after this one exits (the
    shell entry point does this -- the CI pod is ephemeral, so the temp dir need
    not be cleaned up).
    """
    if _real_driver_present():
        return None  # A real driver is installed; leave the loader path alone.
    stub = _find_stub()
    if stub is None:
        return None  # Nothing to offer; the import will report the real error.
    # A fresh owner-only directory per invocation, not a fixed path: everything
    # on LD_LIBRARY_PATH is loaded into the process, so a predictable directory
    # under a shared /tmp would let any local user pre-create it and plant their
    # own libcuda.so.1. The stub must be visible under its SONAME, libcuda.so.1.
    stub_dir = tempfile.mkdtemp(prefix="cuda-driver-stub-")
    if register_cleanup:
        atexit.register(shutil.rmtree, stub_dir, ignore_errors=True)
    os.symlink(stub, os.path.join(stub_dir, "libcuda.so.1"))
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = stub_dir + (os.pathsep + existing if existing else "")
    print(
        f"No CUDA driver on this machine; using the stub at {stub} so "
        "CUDA-driver-linked modules can be imported",
        file=sys.stderr,
    )
    return stub_dir


if __name__ == "__main__":
    # Shell entry point: `export LD_LIBRARY_PATH=$(python3 cuda_driver_stub.py)`.
    # register_cleanup=False so the stub dir outlives this process -- the
    # caller's exported LD_LIBRARY_PATH points at it. The setup message goes to
    # stderr so stdout carries only the LD_LIBRARY_PATH value to capture.
    ensure_cuda_driver_stub(register_cleanup=False)
    print(os.environ.get("LD_LIBRARY_PATH", ""))
