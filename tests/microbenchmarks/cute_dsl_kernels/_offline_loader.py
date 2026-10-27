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

Mechanism: exactly two package objects are seeded into ``sys.modules`` --
``tensorrt_llm`` and ``tensorrt_llm._torch`` -- with the source checkout
*prepended* to their ``__path__``. Only those two ``__init__.py`` are expensive
to execute (``tensorrt_llm/__init__.py`` runs the environment bootstrap and
pulls torch; ``tensorrt_llm/_torch/__init__.py`` is ``from .llm import LLM``).
The ``__init__.py`` of every other package the drivers traverse is empty,
resolves its exports lazily (``_torch/attention``) or imports only modules of
its own subtree (``kernels/blackwell/top_k``, for example), so ordinary import
machinery can take over from there -- no per-file loading and no
hand-maintained module chain.

Seeding rather than replacing matters: when a built ``tensorrt_llm`` *is*
installed, both seeded packages search the checkout first and the install after
it, so compiled artefacts that exist only in an install --
``tensorrt_llm.bindings`` above all -- stay reachable while the relocated
pure-Python kernel trees resolve to the checkout.

The drivers call :func:`install` only after a first import has failed, and that
attempt may already have imported modules from the install (for example
``tensorrt_llm``, ``tensorrt_llm._torch`` and ``tensorrt_llm._torch.moe``) and
left them in ``sys.modules``. They are kept: the two seeded packages are extended
as described above, and the matching checkout directory is put first on the
``__path__`` of each cached package under ``tensorrt_llm._torch``, so the
retried import finds the kernel trees that the install does not have. Every
other cached ``tensorrt_llm.*`` package (``tensorrt_llm.executor``, for
example) keeps its installed ``__path__``. Modules the first attempt imported
stay the install's copies, and so do modules first imported later beneath one
of those other cached packages; any other module first imported after
:func:`install` resolves to the checkout wherever the checkout has it. Such a
process therefore runs a mix of the two, and checkout code that imports a name
missing from the install's copy of a cached module still fails. With
``TRTLLM_OFFLINE_LOADER_VERBOSE`` set to a non-empty value, :func:`install`
prints to stderr the ``__path__`` of both seeded packages and the name of each
cached package whose ``__path__`` it changed.

Scope: this fallback covers "an install exists, only the paths moved". It does
not promise to work with no TensorRT-LLM install at all -- the drivers whose
import closure reaches ``tensorrt_llm.bindings`` need a build artefact that is
not in git, and no loader can synthesise it.
"""

import importlib
import importlib.machinery
import importlib.util
import os
import sys
import types
from pathlib import Path

__all__ = ["install"]

# The two packages whose ``__init__.py`` must not be executed, in parent-first
# order, mapped to their directory relative to the repository root.
_STUBS = (
    ("tensorrt_llm", ("tensorrt_llm",)),
    ("tensorrt_llm._torch", ("tensorrt_llm", "_torch")),
)

_VERBOSE_ENV = "TRTLLM_OFFLINE_LOADER_VERBOSE"


def _seed(name, directory):
    """Create the stub package ``name`` or extend the cached one; search ``directory`` first."""
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        parent, _, leaf = name.rpartition(".")
        if parent:
            # The parent was seeded on the previous iteration, as a new stub or
            # by extending the package a failed first import left behind; its
            # ``__path__`` already lists the checkout first and any install
            # after it, so the child searches the same locations in that order.
            module.__path__ = [
                os.path.join(entry, leaf)
                for entry in sys.modules[parent].__path__
                if os.path.isdir(os.path.join(entry, leaf))
            ]
        else:
            module.__path__ = []
            # ``find_spec`` locates an installed copy without executing its
            # ``__init__.py``.
            try:
                spec = importlib.util.find_spec(name)
            except (ImportError, AttributeError, ValueError):
                spec = None
            if spec is not None and spec.submodule_search_locations:
                module.__path__ = list(spec.submodule_search_locations)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        module.__spec__.submodule_search_locations = module.__path__
        sys.modules[name] = module
        if parent:
            setattr(sys.modules[parent], leaf, module)
    wanted = str(directory)
    if wanted not in module.__path__:
        module.__path__.insert(0, wanted)  # checkout wins over any install
    return module


def _prefer_checkout_for_cached_packages(root):
    """Put the checkout first on every cached ``tensorrt_llm._torch.*`` package's ``__path__``.

    A package imported from an install keeps its installed ``__path__``, so a
    subpackage that exists only in the checkout stays unresolvable beneath it
    until the checkout directory is added. Only subpackages of
    ``tensorrt_llm._torch`` are extended, because the relocated kernel trees,
    which an older install lacks, all live under it; the packages outside it
    that the drivers reach indirectly exist in the install as well.

    Returns the names of the packages whose ``__path__`` was changed.
    """
    changed = []
    for name in sorted(sys.modules):
        if not name.startswith("tensorrt_llm._torch."):
            continue
        module = sys.modules[name]
        # Read the namespace directly: a module-level ``__getattr__`` may raise
        # for a missing name, ``__path__`` included, and not necessarily with
        # ``AttributeError``.
        path = vars(module).get("__path__") if isinstance(module, types.ModuleType) else None
        # Only regular packages, whose ``__path__`` is a plain list, are
        # extended; the trees the drivers import contain no namespace packages.
        if not isinstance(path, list):
            continue
        directory = root.joinpath(*name.split("."))
        if directory.is_dir() and str(directory) not in path:
            path.insert(0, str(directory))
            changed.append(name)
    importlib.invalidate_caches()
    return changed


def install(repo_root):
    """Make the kernel trees under ``repo_root`` importable as ``tensorrt_llm.*``.

    ``repo_root`` is the repository root -- the directory that *contains*
    ``tensorrt_llm/``. Calling this twice is a no-op the second time.
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

    changed = _prefer_checkout_for_cached_packages(root)

    if os.environ.get(_VERBOSE_ENV):
        # stderr on purpose: driver stdout is benchmark output.
        for name, _ in _STUBS:
            print(
                "[offline-loader] %s.__path__ = %r" % (name, list(sys.modules[name].__path__)),
                file=sys.stderr,
            )
        # These packages came from the install; submodules they already
        # imported stay the install's copies.
        for name in changed:
            print(
                "[offline-loader] checkout prepended to cached %s.__path__" % (name,),
                file=sys.stderr,
            )
    return sys.modules["tensorrt_llm"]
