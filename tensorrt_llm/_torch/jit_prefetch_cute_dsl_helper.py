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
"""CPU-only CuTe DSL compile helper for ``jit_prefetch.py``. Run as a script.

    jit_prefetch_cute_dsl_helper.py <gpu_arch> <out_dir>

Reads one JSON request per stdin line, ``{"tag": int, "spec": str}``, and
writes ``{"tag": int, "ok": bool, "s": float, "err": str, "built": bool}``.
``spec`` names a kernel builder in ``jit_prefetch_cute_dsl`` and its
compile-time arguments (see ``jit_prefetch_cute_dsl.build``).

The process has no visible GPU and launches nothing. Kernel arguments are
CuTe fake tensors with the layout the real call passes, the target arch is
explicit (``--gpu-arch``), and the result is exported with ``export_to_c`` to
``<out_dir>/<key>.o``, which the executor loads with
``cute.runtime.load_module`` instead of compiling.
"""

import glob
import json
import os
import sys
import time


def _preload_runtime():
    """Make CuTe DSL's runtime library (which defines the host shim's CUDA
    entry points) visible, so export can resolve them without a device."""
    import ctypes

    import nvidia_cutlass_dsl

    # A namespace package: no __file__, possibly several __path__ entries.
    for root in list(nvidia_cutlass_dsl.__path__):
        for path in sorted(glob.glob(os.path.join(root, "cu1*", "lib", "libcute_dsl_runtime.so"))):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            os.environ.setdefault("CUTE_DSL_LIBS", path)
            return
    raise RuntimeError("libcute_dsl_runtime.so not found under nvidia_cutlass_dsl")


def main():
    gpu_arch, out_dir = sys.argv[1:3]
    os.makedirs(out_dir, exist_ok=True)
    # Libraries loaded below (UCX via tensorrt_llm, CuTe DSL) print to fd 1.
    # Keep the protocol on a private copy of stdout; send fd 1 to stderr.
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    _preload_runtime()
    from tensorrt_llm._torch import jit_prefetch_cute_dsl as jcd

    out.write(json.dumps({"ready": True}) + "\n")
    out.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        t0 = time.time()
        try:
            built = jcd.build(req["spec"], gpu_arch, out_dir)
            resp = {"ok": True, "err": "", "built": built}
        except Exception as e:  # noqa: BLE001 - report, never die
            resp = {"ok": False, "err": f"{type(e).__name__}: {e}"[:500], "built": False}
        resp.update(tag=req["tag"], s=time.time() - t0)
        out.write(json.dumps(resp) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
