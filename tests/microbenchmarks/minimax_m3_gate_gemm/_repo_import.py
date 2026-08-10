# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import a repo module without running the `tensorrt_llm` package __init__.

`import tensorrt_llm` pulls in the whole runtime: `transformers`, the compiled
C++ extension, the LLM API. A kernel microbenchmark needs none of it, and a bare
container may be missing any of it, while the CuTe DSL kernels themselves import
only `cutlass` and their siblings.

So stand in fake parent packages carrying nothing but a `__path__` and let the
normal import machinery resolve the leaf module through them. Relative imports
inside the kernel modules keep working, because the parents exist as far as the
import system is concerned.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[3]


def import_bare(dotted: str) -> ModuleType:
    """Import `dotted` from the repo, stubbing out its parent packages."""
    parts = dotted.split(".")
    for depth in range(1, len(parts)):
        name = ".".join(parts[:depth])
        existing = sys.modules.get(name)
        if existing is not None and getattr(existing, "__path__", None):
            continue
        stub = types.ModuleType(name)
        stub.__path__ = [str(REPO_ROOT.joinpath(*parts[:depth]))]
        sys.modules[name] = stub
        if depth > 1:
            setattr(sys.modules[".".join(parts[: depth - 1])], parts[depth - 1], stub)
    return importlib.import_module(dotted)
