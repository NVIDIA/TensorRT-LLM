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
"""CPU-only Triton compile helper for ``jit_prefetch.py``. Run as a script.

Reads one JSON request per stdin line, ``{"tag": int, "spec": str}``, where
``spec`` is Triton's own specialization JSON (``serialize_specialization_data``).
Compiles it with ``triton.compile`` into the on-disk Triton cache and writes
``{"tag": int, "ok": bool, "s": float, "err": str}`` to stdout.

This process must never initialize MPI or CUDA. It is started with a scrubbed
environment (no OMPI_/PMIX_/PMI_ variables, no visible GPU), and it does not
import the ``tensorrt_llm`` package: the module that defines a kernel is loaded
from its file with every parent package stubbed out, and every other
``tensorrt_llm`` module that module imports is replaced by an inert stand-in.
Only the kernel's own module and its sibling modules in the same directory are
real. Triton's cache key depends on the kernel source, its line number and the
values of the globals the kernel itself references, so the stand-ins do not
change the key; a kernel that depends on a stand-in fails to compile here and
falls back to compiling at its first launch.
"""

import importlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
import types


class _Inert(types.ModuleType):
    """A module whose every attribute is a harmless placeholder class."""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        placeholder = type(name, (), {})
        setattr(self, name, placeholder)
        return placeholder


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve tensorrt_llm.* imports without running the real package code."""

    def __init__(self, pkg_root: str):
        self.pkg_root = pkg_root  # directory containing tensorrt_llm/
        self.real_dirs = set()

    def allow_dir_of(self, module_name: str):
        rel = module_name.split(".")[:-1]
        self.real_dirs.add(os.path.join(self.pkg_root, *rel))

    def find_spec(self, name, path, target=None):
        if name != "tensorrt_llm" and not name.startswith("tensorrt_llm."):
            return None
        parts = name.split(".")
        as_dir = os.path.join(self.pkg_root, *parts)
        as_file = as_dir + ".py"
        if os.path.isfile(as_file) and os.path.dirname(as_file) in self.real_dirs:
            return importlib.machinery.PathFinder.find_spec(name, [os.path.dirname(as_file)])
        spec = importlib.machinery.ModuleSpec(name, self, is_package=os.path.isdir(as_dir))
        if os.path.isdir(as_dir):
            spec.submodule_search_locations = [as_dir]
        return spec

    def create_module(self, spec):
        mod = _Inert(spec.name)
        if spec.submodule_search_locations is not None:
            mod.__path__ = list(spec.submodule_search_locations)
        return mod

    def exec_module(self, module):
        return None


def _resolve(finder: _StubFinder, full_name: str):
    from triton.runtime.autotuner import Autotuner, Heuristics

    module, _, qual = full_name.rpartition(".")
    finder.allow_dir_of(module)
    obj = importlib.import_module(module)
    for part in qual.split("."):
        obj = getattr(obj, part)
    while isinstance(obj, (Autotuner, Heuristics)):
        obj = obj.fn
    return obj


def main():
    pkg_root = sys.argv[1]
    finder = _StubFinder(pkg_root)
    sys.meta_path.insert(0, finder)

    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.runtime.jit import convert_to_tuple_if_list

    def dec(v):
        if isinstance(v, str) and tl.dtype.is_dtype(v):
            return tl.dtype(v)
        if isinstance(v, dict) and "constexpr" in v:
            return tl.constexpr(convert_to_tuple_if_list(v["constexpr"]))
        return convert_to_tuple_if_list(v)

    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        tag = req["tag"]
        t0 = time.time()
        try:
            d = json.loads(req["spec"])
            jit_fn = _resolve(finder, d["name"])
            constexprs = {tuple(k): dec(v) for k, v in zip(d["constant_keys"], d["constant_vals"])}
            attrs = {tuple(k): v for k, v in zip(d["attrs_keys"], d["attrs_vals"])}
            signature = {k: convert_to_tuple_if_list(v) for k, v in d["signature"].items()}
            options = {k: tuple(v) if isinstance(v, list) else v for k, v in d["options"].items()}
            t = d["target"]
            target = GPUTarget(t["backend"], t["arch"], t["warp_size"])
            triton.compile(
                ASTSource(jit_fn, signature, constexprs, attrs), target=target, options=options
            )
            resp = {"tag": tag, "ok": True, "s": time.time() - t0, "err": ""}
        except Exception as e:  # noqa: BLE001 - report, never die
            resp = {
                "tag": tag,
                "ok": False,
                "s": time.time() - t0,
                "err": f"{type(e).__name__}: {e}"[:500],
            }
        out.write(json.dumps(resp) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
