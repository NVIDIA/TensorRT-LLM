# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep CuTe DSL task scheduling's WorkTileInfo rebind inside PrimTS.

Importing ``cutlass.experimental.task_scheduling`` rebinds six attributes of CUTLASS'
``WorkTileInfo`` (``__init__``, the two MLIR value hooks, ``tile_idx``, ``is_valid_tile`` and
``update_from``) to a scalar three-axis form, for every kernel in the process. Kernels that
subclass ``WorkTileInfo`` with a different tile shape (FA4's four-axis tile) then fail to trace.

This module is imported first by the package. It snapshots the original attributes, imports
the task scheduling resources (which apply the rebind), snapshots the scalar set, and puts the
originals back. ``scalar_work_tile_info`` installs the scalar set only while a PrimTS kernel
traces and compiles; compiled kernels do not call these Python methods. Nothing about the
generated code changes. Kill switch: TRTLLM_PRIMS_TS_SCOPED_WORKTILE=0 keeps the global rebind.
"""

import contextlib
import functools
import os
import sys
import threading

from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo as _WorkTileInfo

_ATTRS = (
    "__init__",
    "__extract_mlir_values__",
    "__new_from_mlir_values__",
    "tile_idx",
    "is_valid_tile",
    "update_from",
)
_MISSING = object()
_RESOURCES = "cutlass.experimental.task_scheduling.resources"


def _snapshot() -> dict:
    return {name: _WorkTileInfo.__dict__.get(name, _MISSING) for name in _ATTRS}


def _install(attrs: dict) -> None:
    for name, value in attrs.items():
        if value is _MISSING:
            if name in _WorkTileInfo.__dict__:
                delattr(_WorkTileInfo, name)
        else:
            setattr(_WorkTileInfo, name, value)


# Only scope the rebind when this module saw the originals, i.e. task scheduling was not imported
# before PrimTS. Otherwise keep whatever is installed (the previous global behavior).
_ORIGINAL = None if _RESOURCES in sys.modules else _snapshot()
import cutlass.experimental.task_scheduling.resources  # noqa: E402,F401  (applies the scalar rebind)

_SCALAR = _snapshot()
SCOPED = _ORIGINAL is not None and os.environ.get("TRTLLM_PRIMS_TS_SCOPED_WORKTILE", "1") != "0"
_LOCK = threading.RLock()
_DEPTH = 0
if SCOPED:
    _install(_ORIGINAL)


@contextlib.contextmanager
def scalar_work_tile_info():
    """Install the task scheduling WorkTileInfo form while a PrimTS kernel traces (re-entrant)."""
    global _DEPTH
    if not SCOPED:
        yield
        return
    with _LOCK:
        if _DEPTH == 0:
            _install(_SCALAR)
        _DEPTH += 1
        try:
            yield
        finally:
            _DEPTH -= 1
            if _DEPTH == 0:
                _install(_ORIGINAL)


def with_scalar_work_tile_info(fn):
    """Decorator form of ``scalar_work_tile_info`` for the PrimTS compile functions."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with scalar_work_tile_info():
            return fn(*args, **kwargs)

    return wrapper
