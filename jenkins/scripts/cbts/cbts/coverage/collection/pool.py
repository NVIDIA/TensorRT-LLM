# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MPI pool accounting, shared by the bootstrap and the pytest plugin.

Separate from ``cbts_plugin`` so ``sitecustomize`` can reach this without importing
pytest. The bootstrap runs during interpreter startup, before pytest exists; loading
the plugin there would seat a module in ``sys.modules`` that pytest's own
``-p cbts_plugin`` would then inherit.
"""

from __future__ import annotations

import contextlib
import os

from cbts.coverage.collection import hooks
from cbts.coverage.collection.channel import (
    ADDRESS_ENV,
    TAINT_ATTRIBUTION,
    TAINT_INCOMPLETE,
    TAINT_POOL_ENV_UNAVAILABLE,
    safe_print,
)

_POOL_PATCHED_MARKER = "_cbts_patched_pool_init"


def _report_pool_env_unavailable(exc: BaseException) -> None:
    """Report and self-taint a failure to forward the channel address to a pool worker.

    Best-effort and never raises itself: whatever failed is on the critical path of the
    product's own pool construction (see ``install_pool_env_patch``), so the report
    itself must not become a second failure. Unlike a worker whose address merely
    arrives late, a worker that never receives it at all has no later retry to find it
    either -- see ``TAINT_POOL_ENV_UNAVAILABLE``.
    """
    safe_print(f"[cbts] pool env patch failed in pid {os.getpid()}: {exc!r}")
    with contextlib.suppress(Exception):
        hooks.active.record_channel_taints(
            [
                (None, "", TAINT_ATTRIBUTION, TAINT_POOL_ENV_UNAVAILABLE),
                (None, "", TAINT_INCOMPLETE, TAINT_POOL_ENV_UNAVAILABLE),
            ]
        )


def install_pool_env_patch() -> bool:
    """Patch ``MPIPoolExecutor.__init__`` to forward the context channel address; idempotent.

    Patching the constructor rather than ``MpiPoolSession._start_mpi_pool`` catches every pool
    (product ``MpiPoolSession`` and disagg's own raw ``MPIPoolExecutor``) while leaving the
    product's pool setup — ``env_overrides``, the ``wait_shutdown`` worker-identity barrier, … —
    intact. Most coverage env reaches workers by ordinary OS-level inheritance
    (``CBTS_COVERAGE_CONFIG``, ``PYTHONPATH``), but the context channel's address is only
    known once the plugin has bound it, which is after the MPI runtime captured the
    environment it spawns workers with. So it is forwarded through ``mpi4py``'s own ``env``
    payload, applied by the worker's sync handshake before it runs any task -- late for the
    startup bootstrap, in time for the subscribe that happens on the framework import.

    Never raises: this stands in for the product's own pool construction, on the
    critical path of every dynamically-spawned pool, including every multi-GPU
    inference deployment that uses one. An unexpected failure here is reported and
    self-tainted (see ``_report_pool_env_unavailable``) rather than propagated.
    """
    try:
        return _install_pool_env_patch()
    except Exception as exc:
        _report_pool_env_unavailable(exc)
        return False


def _install_pool_env_patch() -> bool:
    try:
        from mpi4py.futures import MPIPoolExecutor
    except ImportError:
        return False

    init = MPIPoolExecutor.__init__
    if getattr(init, _POOL_PATCHED_MARKER, False):
        return False

    def _patched_init(self, *args, **kwargs) -> None:
        # The original call always goes through, address merged in or not: coverage
        # must never be able to prevent a pool from being built.
        try:
            address = os.environ.get(ADDRESS_ENV, "").strip()
            if address:
                # Merged, not replaced: the product passes its own env_overrides here.
                worker_env = dict(kwargs.get("env") or {})
                worker_env.setdefault(ADDRESS_ENV, address)
                kwargs["env"] = worker_env
        except Exception as exc:
            _report_pool_env_unavailable(exc)
        return init(self, *args, **kwargs)

    setattr(_patched_init, _POOL_PATCHED_MARKER, True)
    MPIPoolExecutor.__init__ = _patched_init
    return True
