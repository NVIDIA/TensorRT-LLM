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
"""Shared helpers for visual_gen multi-GPU (mp.spawn) tests.

Provides :func:`spawn_with_retry`, which allocates a fresh master port and
retries the spawn when the c10d rendezvous ``TCPStore`` loses the bind race and
fails with ``EADDRINUSE``.

Why the retry is needed: the CI-aware port allocator (``get_free_port_in_ci``)
binds a probe socket, *closes* it, and returns the port number. ``mp.spawn``
then launches fresh worker processes and rank 0's ``TCPStore`` re-binds that
port. Anything on the host (including ephemeral outbound sockets or a parallel
test) can grab the port in the gap between the probe close and the re-bind, so a
single allocation is not enough on busy nodes -- we must re-allocate and retry.

Also provides :func:`single_rank_llm_mapping`, the ``Mapping`` to give a
``tp_size == 1`` model -- typically the single-GPU reference of a parity test --
that is constructed *inside* a multi-rank worker.

Why it is needed: the workers run with ``TLLM_DISABLE_MPI=1``, so
``Mapping(world_size=1, rank=0, tp_size=1)`` is a ``DeviceMeshTopology`` whose
``tp_rank`` / ``pp_rank`` / ``cp_rank`` are read lazily from the live process
group. Before a device mesh exists that lookup falls back to ``dist.group.WORLD``,
so on rank ``r`` the "single-GPU" mapping reports ``tp_rank == r``. ``Linear``
caches ``mapping.tp_rank`` at construction and a ROW-parallel ``Linear`` adds its
bias only when ``tp_rank == 0``; the always-ROW ``MLP.down_proj`` of such a
reference therefore drops its bias on every rank but 0, and the "reference"
differs from rank to rank.

Why a ``DeviceMeshTopology`` subclass rather than the alternatives:

* ``MpiTopology(world_size=1, ...)`` cannot be constructed directly:
  ``Mapping.__new__`` chooses the concrete class from ``mpi_disabled()`` alone
  and, when MPI is disabled, hands back an *uninitialised* ``DeviceMeshTopology``
  (``__init__`` is skipped because the object is not an ``MpiTopology``). Making
  it work means unsetting ``TLLM_DISABLE_MPI`` around the call, a process-global
  toggle that every other ``mpi_disabled()`` reader could observe.
* A plain ``Mapping`` subclass is defeated by the same ``__new__`` for the same
  reason, and a bare ``object.__new__`` + manual ``__init__`` is the same wart
  in a less readable form.
* Building a device mesh with a size-1 ``tp`` dimension so that the lookup
  returns 0 needs NCCL, collides with the mesh the test itself builds next, and
  has to be torn down again.

The subclass keeps the product class, its single-process group handling
(``tp_group == [0]``; ``tp_group_pg`` unchanged, never reached at ``tp_size == 1``
by ``Linear`` or ``AllReduce``) and its ``__init__`` validation; it only pins the
three rank coordinates to 0 and overrides ``__new__`` so that ``Mapping.__new__``
cannot swap the class out from under it.
"""

import torch.multiprocessing as mp

from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.mapping import DeviceMeshTopology, Mapping

# The CI-aware allocator lives in tests/integration/defs/common.py. Adding that
# directory to sys.path lets us reuse it (it tracks allocated ports per-process
# so sequential tests don't collide, and honors CONTAINER_PORT_START/NUM).
__extra_import_path__ = ["~/tests/integration"]

_ADDR_IN_USE_MARKERS = ("EADDRINUSE", "address already in use")


def _is_addr_in_use(exc: BaseException) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _ADDR_IN_USE_MARKERS)


def spawn_with_retry(spawn_fn, max_retries: int = 10):
    """Run ``spawn_fn(port)`` with a fresh free port, retrying on EADDRINUSE.

    ``spawn_fn`` receives a master port and is expected to call ``mp.spawn``
    (passing the port through to the workers). If the rendezvous TCPStore fails
    to bind because the port was grabbed after allocation, a new port is chosen
    and the spawn is retried up to ``max_retries`` times. Any other failure
    (e.g. a real assertion inside the test) propagates immediately.
    """
    # Resolved here, not at module level: ``__extra_import_path__`` is honoured
    # by the pytest import hook, which is absent in the spawned workers that
    # import this module for ``single_rank_llm_mapping``.
    from defs.common import get_free_port_in_ci

    last_exc: BaseException | None = None
    for _ in range(max_retries):
        port = get_free_port_in_ci()
        try:
            spawn_fn(port)
            return
        except mp.ProcessRaisedException as exc:
            if _is_addr_in_use(exc):
                last_exc = exc
                continue
            raise
        except OSError as exc:
            if _is_addr_in_use(exc):
                last_exc = exc
                continue
            raise
    assert last_exc is not None
    raise last_exc


class _SingleRankMapping(DeviceMeshTopology):
    """``Mapping(world_size=1)`` whose rank coordinates are 0 whatever the live process group is."""

    def __new__(cls, *args: object, **kwargs: object) -> "_SingleRankMapping":
        # Mapping.__new__ ignores ``cls`` and instantiates DeviceMeshTopology itself,
        # which would bypass this subclass and skip its __init__ altogether.
        return object.__new__(cls)

    @property
    def tp_rank(self) -> int:
        return 0

    @property
    def pp_rank(self) -> int:
        return 0

    @property
    def cp_rank(self) -> int:
        return 0


def single_rank_llm_mapping() -> Mapping:
    """Return the ``Mapping`` for a ``tp_size == 1`` model built inside a multi-rank worker.

    Behaves like ``Mapping(world_size=1, rank=0, tp_size=1)`` does in a genuinely
    single-process job: ``tp_rank == pp_rank == cp_rank == 0`` and
    ``tp_group == [0]`` on every rank, so ROW-parallel ``Linear`` layers keep
    their bias. Use it wherever a test assigns ``vgm.to_llm_mapping()`` for a
    mapping with ``tp_size == 1``.
    """
    if not mpi_disabled():
        # MpiTopology derives its ranks arithmetically from (rank, sizes); nothing to pin.
        return Mapping(world_size=1, rank=0, tp_size=1)
    return _SingleRankMapping(world_size=1, rank=0, tp_size=1)
