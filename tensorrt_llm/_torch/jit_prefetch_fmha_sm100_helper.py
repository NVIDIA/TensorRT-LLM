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
"""CPU-only builder of MSA ``fmha_sm100`` kernels for ``jit_prefetch.py``. Run as a script.

    jit_prefetch_fmha_sm100_helper.py <parent_pid> <fmha_sm100 package dir> <cache base>

Reads one JSON request per stdin line, ``{"tag": int, "spec": str}``, where
``spec`` names one kernel (``attention/backends/sparse/minimax_m3/
jit_prefetch_fmha.py``): an FMHA variant (``get_fmha_variant``'s arguments),
the planner module, or the NVFP4 sparse prefill's k2q CSR builder. Builds it
where the parent loads it from and writes ``{"tag": int, "ok": bool,
"built": bool, "s": float, "err": str}``; ``built`` is False for a kernel that
was already built.

The process has no visible GPU and launches nothing. ``fmha_sm100``'s nvcc
flags name their targets, so nothing is read from a device. Only
``fmha_sm100/jit.py`` is loaded (by path, without the package's
``__init__``, which imports the CuTe DSL stack); it derives its cache
directory from the package sources and the environment, both the parent's,
and every request fails if that is not the parent's ``<cache base>``. Each
kernel is built holding the per-kernel file lock ``fmha_sm100`` takes before
its own build, so a parent that needs a kernel a helper is building waits for
that build instead of compiling it again. The k2q CSR builder is a
``torch.utils.cpp_extension`` module: importing it builds it, under torch's
build-directory lock.

The process leads a process group of its own. SIGTERM to it stops a build
(ninja stops the compilers it runs and removes their partial outputs) and
this process exits; it sends that signal to itself once ``parent_pid`` is no
longer its parent.
"""

import importlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time

_stop = False


def _on_sigterm(signum, frame) -> None:
    global _stop
    _stop = True


def _stop_with_parent(parent: int) -> None:
    """Once the parent has exited, SIGTERM this process's group each second (a
    ninja started after the first one gets it as well)."""
    while os.getppid() == parent:
        time.sleep(1.0)
    while True:
        os.killpg(0, signal.SIGTERM)
        time.sleep(1.0)


def _load_jit(pkg_dir: str):
    """``fmha_sm100/jit.py`` as a module of its own."""
    spec = importlib.util.spec_from_file_location(
        "_jit_prefetch_fmha_sm100_jit", os.path.join(pkg_dir, "jit.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_ninja_two_jobs(self, cache_dir) -> None:
    """``FMHAVariantManager._run_ninja`` with two jobs, so a variant's two
    objects compile at once. The build file and its commands are unchanged."""
    result = subprocess.run(["ninja", "-j2"], cwd=str(cache_dir), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FMHA variant compilation failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


def _build(jit, pkg_dir: str, req: dict) -> bool:
    """Build one kernel; return False if it was already built."""
    op = req["op"]
    if op == "variant":
        name, params = jit._variant_key_from_runtime(*req["args"])
        cache_dir = jit.CACHE_BASE / name
        fd = jit._acquire_file_lock(cache_dir / ".compile.lock")
        try:
            if (cache_dir / f"{name}.so").exists():
                return False
            jit._variant_manager._compile_only(name, params)
            return True
        finally:
            jit._release_file_lock(fd)
    if op == "plan":
        fd = jit._acquire_file_lock(jit.CACHE_BASE / "plan.lock")
        try:
            if (jit.CACHE_BASE / "plan" / "fmha_sm100_plan.so").exists():
                return False
            jit._do_compile_plan()
            return True
        finally:
            jit._release_file_lock(fd)
    if op == "k2q_csr":
        # Imported the way the k2q CSR builder imports it (fmha_sm100/cute on
        # sys.path); the import builds the extension or finds it built.
        from torch.utils.cpp_extension import _get_build_directory

        name = "sparse_build_k2q_csr_ext"
        built = not os.path.exists(os.path.join(_get_build_directory(name, False), f"{name}.so"))
        cute_dir = os.path.join(pkg_dir, "cute")
        if cute_dir not in sys.path:
            sys.path.insert(0, cute_dir)
        importlib.import_module("src.sm100.build_k2q_csr")
        return built
    raise ValueError(f"unsupported op {op}")


def main() -> int:
    os.setsid()
    signal.signal(signal.SIGTERM, _on_sigterm)
    parent, pkg_dir, cache_base = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    threading.Thread(target=_stop_with_parent, args=(parent,), daemon=True).start()
    # Replies go to a private copy of stdout; fd 1, which the compilers and
    # the libraries may print to, goes to stderr.
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    setup_err = ""
    try:
        jit = _load_jit(pkg_dir)
        if str(jit.CACHE_BASE) != cache_base:
            setup_err = f"cache base {jit.CACHE_BASE} is not the parent's {cache_base}"
        jit.FMHAVariantManager._run_ninja = _run_ninja_two_jobs
    except Exception as e:  # noqa: BLE001 - every request then fails
        setup_err = f"{type(e).__name__}: {e}"
    out.write(json.dumps({"ready": True}) + "\n")
    for line in sys.stdin:
        line = line.strip()
        if not line or _stop:
            if _stop:
                return 1
            continue
        req = json.loads(line)
        t0 = time.perf_counter()
        ok, built, err = not setup_err, False, setup_err
        if ok:
            try:
                built = _build(jit, pkg_dir, json.loads(req["spec"]))
            except Exception as e:  # noqa: BLE001 - report; the parent builds it in place
                ok, err = False, f"{type(e).__name__}: {str(e)[-2000:]}"
        if _stop:
            return 1
        try:
            out.write(
                json.dumps(
                    {
                        "tag": req["tag"],
                        "ok": ok,
                        "built": built,
                        "s": time.perf_counter() - t0,
                        "err": err,
                    }
                )
                + "\n"
            )
        except OSError:  # the parent has gone
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
