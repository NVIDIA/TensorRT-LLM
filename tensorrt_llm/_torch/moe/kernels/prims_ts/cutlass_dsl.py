# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Require the installed CUTLASS DSL wheel for Prims-TS kernels."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import threading
from typing import Iterator

_BOOTSTRAPPED = False
_BOOTSTRAP_ERROR: BaseException | None = None
_BOOTSTRAP_LOCK = threading.RLock()
_WORK_TILE_INFO_ATTRS = (
    "__init__",
    "__extract_mlir_values__",
    "__new_from_mlir_values__",
    "tile_idx",
    "is_valid_tile",
    "update_from",
)
_MISSING = object()
_ORIGINAL_WORK_TILE_INFO: dict[str, object] | None = None
_TASK_SCHEDULING_WORK_TILE_INFO: dict[str, object] | None = None
_TASK_SCHEDULING_SCOPE_DEPTH = 0


def _work_tile_info_class():
    from cutlass.utils import static_persistent_tile_scheduler as scheduler

    return scheduler.WorkTileInfo


def _snapshot_work_tile_info() -> dict[str, object]:
    work_tile_info = _work_tile_info_class()
    return {
        name: work_tile_info.__dict__.get(name, _MISSING)
        for name in _WORK_TILE_INFO_ATTRS
    }


def _apply_work_tile_info(snapshot: dict[str, object]) -> None:
    work_tile_info = _work_tile_info_class()
    for name, value in snapshot.items():
        if value is _MISSING:
            if name in work_tile_info.__dict__:
                delattr(work_tile_info, name)
        else:
            setattr(work_tile_info, name, value)


def _has_required_modules() -> bool:
    global _ORIGINAL_WORK_TILE_INFO, _TASK_SCHEDULING_WORK_TILE_INFO

    # Capture CUTLASS' general-purpose implementation before importing Task
    # Scheduling. CUTLASS DSL 4.7 mutates this class at module import time.
    importlib.import_module("cutlass")
    importlib.import_module("cutlass.cute")
    if _ORIGINAL_WORK_TILE_INFO is None:
        _ORIGINAL_WORK_TILE_INFO = _snapshot_work_tile_info()

    required = (
        "cutlass.experimental.primitives",
        "cutlass.experimental.task_scheduling",
    )
    try:
        for name in required:
            importlib.import_module(name)
        if _TASK_SCHEDULING_WORK_TILE_INFO is None:
            _TASK_SCHEDULING_WORK_TILE_INFO = _snapshot_work_tile_info()
    finally:
        # Merely checking/loading Prims-TS must not change unrelated CUTLASS
        # kernels in this process. The TS representation is installed only by
        # task_scheduling_scope() while a Prims-TS kernel is built or compiled.
        _apply_work_tile_info(_ORIGINAL_WORK_TILE_INFO)
    return True


def ensure_cutlass_dsl_experimental() -> bool:
    """Return whether Prims-TS dependencies are importable from installed wheels."""

    global _BOOTSTRAPPED, _BOOTSTRAP_ERROR

    if _BOOTSTRAPPED:
        return True
    if _BOOTSTRAP_ERROR is not None:
        return False

    with _BOOTSTRAP_LOCK:
        if _BOOTSTRAPPED:
            return True
        if _BOOTSTRAP_ERROR is not None:
            return False

        try:
            importlib.invalidate_caches()
            _has_required_modules()
        except BaseException as exc:
            _BOOTSTRAP_ERROR = exc
            return False

        _BOOTSTRAPPED = True
        return True


def require_cutlass_dsl_experimental() -> None:
    if ensure_cutlass_dsl_experimental():
        return
    raise RuntimeError(
        "Prims-TS requires the CUTLASS DSL wheel. Install the pinned "
        "release-branch wheel before using Prims-TS kernels."
    ) from _BOOTSTRAP_ERROR


@contextmanager
def task_scheduling_scope() -> Iterator[None]:
    """Temporarily install CUTLASS Task Scheduling's ``WorkTileInfo`` methods."""

    global _TASK_SCHEDULING_SCOPE_DEPTH

    require_cutlass_dsl_experimental()
    assert _ORIGINAL_WORK_TILE_INFO is not None
    assert _TASK_SCHEDULING_WORK_TILE_INFO is not None

    # WorkTileInfo is a process-global class, so hold this re-entrant lock for
    # the full scope. Nested compilation in the same thread is supported;
    # concurrent compilation waits rather than observing a half-restored type.
    with _BOOTSTRAP_LOCK:
        if _TASK_SCHEDULING_SCOPE_DEPTH == 0:
            _apply_work_tile_info(_TASK_SCHEDULING_WORK_TILE_INFO)
        _TASK_SCHEDULING_SCOPE_DEPTH += 1
        try:
            yield
        finally:
            _TASK_SCHEDULING_SCOPE_DEPTH -= 1
            if _TASK_SCHEDULING_SCOPE_DEPTH == 0:
                _apply_work_tile_info(_ORIGINAL_WORK_TILE_INFO)


def get_cutlass_dsl_bootstrap_error() -> BaseException | None:
    return _BOOTSTRAP_ERROR
