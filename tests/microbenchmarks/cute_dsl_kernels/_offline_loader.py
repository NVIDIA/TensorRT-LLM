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
"""Offline import fallback shared by the standalone CuTe-DSL driver scripts.

The drivers in this directory are runnable as plain scripts, outside any
TensorRT-LLM install layout. Each one therefore carries a small
``try: ... except ImportError: ...`` around its ``tensorrt_llm`` imports; the
``except`` branch calls :func:`install` from this module and then retries the
same imports.

Mechanism (O-1a, shape E -- "path-merging minimal stub"): exactly two package
objects are seeded into ``sys.modules`` -- ``tensorrt_llm`` and
``tensorrt_llm._torch`` -- with the source checkout *prepended* to their
``__path__``. Only those two ``__init__.py`` are expensive to execute
(``tensorrt_llm/__init__.py`` runs the environment bootstrap and pulls torch;
``tensorrt_llm/_torch/__init__.py`` is ``from .llm import LLM``). Every package
below ``_torch`` that the drivers traverse has an ``__init__.py`` that is either
empty or a bare licence header, so ordinary import machinery can take over from
there -- no per-file loading and no hand-maintained module chain.

Seeding rather than replacing matters: when a built ``tensorrt_llm`` *is*
installed, its own search locations are kept behind the checkout, so compiled
artefacts that exist only in an install -- ``tensorrt_llm.bindings`` above all --
stay reachable while the relocated pure-Python kernel trees resolve to the
checkout.

Scope: this fallback covers "an install exists, only the paths moved". It does
not promise to work with no TensorRT-LLM install at all -- the drivers whose
import closure reaches ``tensorrt_llm.bindings`` need a build artefact that is
not in git, and no loader can synthesise it.
"""

import importlib.machinery
import importlib.util
import os
import sys
import types
from pathlib import Path

__all__ = ["install", "repo_root_from"]

# The two packages whose ``__init__.py`` must not be executed, in parent-first
# order, mapped to their directory relative to the repository root.
_STUBS = (
    ("tensorrt_llm", ("tensorrt_llm",)),
    ("tensorrt_llm._torch", ("tensorrt_llm", "_torch")),
)

_VERBOSE_ENV = "TRTLLM_OFFLINE_LOADER_VERBOSE"


def repo_root_from(file_path, up):
    """Return ``Path(file_path).resolve().parents[up]``.

    Provided so drivers can express the repository root once, without importing
    :mod:`pathlib` themselves.
    """
    return Path(file_path).resolve().parents[up]


def _seed(name, directory):
    """Create or extend the stub package ``name`` so ``directory`` is searched first."""
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        # ``find_spec`` locates an installed copy without executing its
        # ``__init__.py``. For a dotted name it imports the *parent* only, which
        # by construction is already the stub seeded on the previous iteration.
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, AttributeError, ValueError):
            spec = None
        if spec is not None and spec.submodule_search_locations:
            module.__path__ = list(spec.submodule_search_locations)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        module.__spec__.submodule_search_locations = module.__path__
        sys.modules[name] = module
        parent, _, leaf = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], leaf, module)
    wanted = str(directory)
    if wanted not in module.__path__:
        module.__path__.insert(0, wanted)  # checkout wins over any install
    return module


def install(repo_root, verify=()):
    """Make the kernel trees under ``repo_root`` importable as ``tensorrt_llm.*``.

    ``repo_root`` is the repository root -- the directory that *contains*
    ``tensorrt_llm/``. Calling this twice is a no-op the second time.

    ``verify`` is an optional iterable of dotted module names to resolve
    immediately; each must come from inside ``repo_root``. This turns the
    otherwise-silent failure mode -- a stale installed copy shadowing the
    checkout -- into a loud one at the point of use.
    """
    root = Path(repo_root).resolve()
    if not root.joinpath("tensorrt_llm").is_dir():
        raise RuntimeError(
            "offline import fallback: %s is not a TensorRT-LLM checkout "
            "(no tensorrt_llm/ directory underneath it)" % (root,)
        )

    for name, parts in _STUBS:
        directory = root.joinpath(*parts)
        if not directory.is_dir():
            raise RuntimeError(
                "offline import fallback: expected package directory %s to "
                "exist for %s" % (directory, name)
            )
        module = _seed(name, directory)
        if module.__path__[0] != str(directory):
            raise RuntimeError(
                "offline import fallback: %s resolves to %s first, not the "
                "checkout at %s" % (name, module.__path__[0], directory)
            )

    for dotted in verify:
        spec = importlib.util.find_spec(dotted)
        if spec is None or not spec.origin:
            raise RuntimeError(
                "offline import fallback: %s did not resolve after seeding "
                "the checkout at %s" % (dotted, root)
            )
        origin = Path(spec.origin).resolve()
        if root not in origin.parents:
            raise RuntimeError(
                "offline import fallback: %s resolved to %s, which is outside "
                "the checkout at %s (stale installed copy?)" % (dotted, origin, root)
            )

    if os.environ.get(_VERBOSE_ENV):
        for name, _ in _STUBS:
            # stderr on purpose: driver stdout is benchmark output.
            print(
                "[offline-loader] %s.__path__ = %r" % (name, list(sys.modules[name].__path__)),
                file=sys.stderr,
            )
    return sys.modules["tensorrt_llm"]
