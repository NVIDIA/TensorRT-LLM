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
"""CPU-only FlashInfer JIT build helper for ``jit_prefetch.py``. Run as a script.

    jit_prefetch_flashinfer_helper.py <parent_pid> <module> [<module> ...]

Builds each named FlashInfer JIT module (see ``MODULES``, and ``ssu_args``
for the Mamba ``selective_state_update`` modules), in order, into
FlashInfer's own JIT cache and prints one JSON line per module to stdout,
``{"module": str, "ok": bool, "s": float, "err": str}``, then exits. Anything
else printed goes to stderr.

The process has no visible GPU and launches nothing: ``FLASHINFER_CUDA_ARCH_LIST``
(set by the parent to the targets its FlashInfer resolved) gives FlashInfer the
targets it would otherwise read from the device, which also name the cache
directory, so the module lands where the parent's ``build_and_load`` looks for
it. The build holds the module's FlashInfer file lock, so a parent that
reaches the module first waits for this build instead of compiling it a
second time.

SIGTERM to this process's group stops the build: ninja, which FlashInfer runs
in the same group, stops its compilers and removes their partial outputs, and
this process exits instead of starting another build. The parent sends it to
cancel; this process sends it to itself once ``parent_pid`` is no longer its
parent (the parent has exited).
"""

import json
import os
import signal
import sys
import threading
import time

_stop = False

# FlashInfer JIT modules a helper builds, by JitSpec name (the directory of
# FlashInfer's JIT cache it is built in): (module, generator of its JitSpec[,
# the generator's keyword arguments]). ``fmha_gen`` is FlashInfer's TRTLLM-GEN
# attention launcher; its generator also downloads the kernel headers the
# build includes.
MODULES = {
    "sampling": ("flashinfer.jit.sampling", "gen_sampling_module"),
    "fmha_gen": ("flashinfer.jit.attention", "gen_trtllm_gen_fmha_module"),
}


def _on_sigterm(signum, frame) -> None:
    global _stop
    _stop = True


def _stop_with_parent(parent: int) -> None:
    """Once the parent process has exited, SIGTERM this process's group each
    second (a ninja started after the first one gets it as well)."""
    while os.getppid() == parent:
        time.sleep(1.0)
    while True:
        os.killpg(0, signal.SIGTERM)
        time.sleep(1.0)


_SSU_PREFIX = "selective_state_update_"


def ssu_args(name: str):
    """The generator and its arguments for FlashInfer's
    ``selective_state_update`` module ``name``. The name spells out every
    argument (``get_selective_state_update_uri``): ``key_value`` pairs
    after the prefix, then the generator's ``_sm100``/``_sm90`` suffix."""
    from flashinfer.jit.mamba import selective_state_update as ssu

    rest, generator = name[len(_SSU_PREFIX) :], ssu.gen_selective_state_update_module
    for suffix, gen in (
        ("_sm100", ssu.gen_selective_state_update_sm100_module),
        ("_sm90", ssu.gen_selective_state_update_sm90_module),
    ):
        if rest.endswith(suffix):
            rest, generator = rest[: -len(suffix)], gen
            break
    tokens = rest.split("_")
    kv = dict(zip(tokens[0::2], tokens[1::2]))
    dtype = {v: k for k, v in ssu._filename_safe_dtype_map.items()}
    args = (
        dtype[kv["s"]],
        dtype[kv["i"]],
        dtype[kv["w"]],
        dtype[kv["a"]],
        dtype[kv["si"]],
        dtype[kv["sc"]] if "sc" in kv else None,
        int(kv["d"]),
        int(kv["ds"]),
        int(kv["nt"]),
        dtype[kv["cs"]],
        dtype[kv["na"]],
        int(kv.get("pr", 0)),
    )
    return generator, args


def _gen_spec(name: str):
    """FlashInfer's ``JitSpec`` for module ``name``: one of ``MODULES``, or a
    ``selective_state_update`` module named by its arguments."""
    import importlib

    if name.startswith(_SSU_PREFIX):
        generator, args = ssu_args(name)
        spec = generator(*args)
        if spec.name != name:
            raise ValueError(f"{name} names the module {spec.name}")
        return spec
    module, generator, *kwargs = MODULES[name]
    return getattr(importlib.import_module(module), generator)(**(kwargs[0] if kwargs else {}))


def main() -> int:
    signal.signal(signal.SIGTERM, _on_sigterm)
    parent, modules = int(sys.argv[1]), sys.argv[2:]
    threading.Thread(target=_stop_with_parent, args=(parent,), daemon=True).start()
    # The result lines go to a private copy of stdout; fd 1, which FlashInfer
    # and the compilers it runs may print to, goes to stderr.
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    rc = 0
    for name in modules:
        if _stop:
            return 1
        t0 = time.perf_counter()
        err = ""
        try:
            spec = _gen_spec(name)
            if _stop:
                return 1
            spec.build(need_lock=True)
            ok = True
        except Exception as e:  # noqa: BLE001 - report, the parent compiles in place
            ok, err, rc = False, f"{type(e).__name__}: {str(e)[-2000:]}", 1
        try:
            out.write(
                json.dumps({"module": name, "ok": ok, "s": time.perf_counter() - t0, "err": err})
                + "\n"
            )
        except OSError:  # the parent has gone
            return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
