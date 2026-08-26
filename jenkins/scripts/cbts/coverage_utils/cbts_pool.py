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

import os

from cbts_channel import ADDRESS_ENV

_POOL_PATCHED_MARKER = "_cbts_patched_pool_init"


def sitecustomize_call(func_name, *args):
    """Forward to a sitecustomize bootstrap hook (context switch / outcome / worker count), if active."""
    try:
        import sitecustomize

        fn = getattr(sitecustomize, func_name, None)
    except ImportError:
        fn = None
    if fn is not None:
        fn(*args)


def install_expected_workers_patch():
    """Patch ``MPIPoolExecutor.__init__`` to count each test's spawned pool workers; idempotent.

    Patching the constructor rather than ``MpiPoolSession._start_mpi_pool`` catches every pool
    (product ``MpiPoolSession`` and disagg's own raw ``MPIPoolExecutor``) while leaving the
    product's pool setup — ``env_overrides``, the ``wait_shutdown`` worker-identity barrier, … —
    intact. Most coverage env reaches workers by ordinary OS-level inheritance
    (``CBTS_COVERAGE_CONFIG``, ``PYTHONPATH``), but the context channel's address is only
    known once the plugin has bound it, which is after the MPI runtime captured the
    environment it spawns workers with. So it is forwarded through ``mpi4py``'s own ``env``
    payload, applied by the worker's sync handshake before it runs any task -- late for the
    startup bootstrap, in time for the subscribe that happens on the framework import.
    """
    try:
        from mpi4py.futures import MPIPoolExecutor
    except ImportError:
        return False

    init = MPIPoolExecutor.__init__
    if getattr(init, _POOL_PATCHED_MARKER, False):
        return False

    def _patched_init(self, *args, **kwargs):
        try:
            max_workers = kwargs.get("max_workers", args[0] if args else None)
            n = int(max_workers) if max_workers else 1
        except (ValueError, TypeError):
            n = 1
        sitecustomize_call("note_expected_workers", os.environ.get("CBTS_TEST_ID", ""), n)
        address = os.environ.get(ADDRESS_ENV, "").strip()
        if address:
            # Merged, not replaced: the product passes its own env_overrides here.
            worker_env = dict(kwargs.get("env") or {})
            worker_env.setdefault(ADDRESS_ENV, address)
            kwargs["env"] = worker_env
        return init(self, *args, **kwargs)

    setattr(_patched_init, _POOL_PATCHED_MARKER, True)
    MPIPoolExecutor.__init__ = _patched_init
    return True
