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
"""Token-sharded workspace allocation and per-group lifecycle over the shared CE transport."""

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from typing import TypeVar, cast

import torch
import torch.distributed as dist

from tensorrt_llm.logger import logger

from .ce_transport import CeTransport
from .symm_mem_pool import DEFAULT_TIMEOUT_MS, PoolLayout, SymmMemPool

_View = TypeVar("_View", torch.Tensor, tuple[torch.Tensor, torch.Tensor])


def _is_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class CeUnavailableError(RuntimeError):
    """Collective transport recovery refused further use of the CE path."""


class CeWorkspace:
    """Own layout views and graph-safe capacity for one shared transport.

    Slots select payload storage; boundaries are serialized on the compute stream. The
    transport owns copy streams, ready flags, completion fencing and interrupted recovery.
    """

    def __init__(
        self,
        ce: CeTransport,
        world_size: int,
        rank: int,
        device: torch.device,
        *,
        slots: int,
    ) -> None:
        if slots < 1 or world_size < 2 or not 0 <= rank < world_size:
            raise ValueError("Invalid CE workspace slots or TP rank/size.")
        self.ce = ce
        self.pool = ce.pool
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.slots = slots
        self._in_flight = [False] * slots
        self._views: dict[tuple, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = {}
        self._captured = False
        self._capturing = False
        self._needs_recovery = False
        self._boundary: AbstractContextManager[None] | None = None

    def _covers(self, slot_nbytes: int, slots: int) -> bool:
        """Whether the pool's current allocation holds ``slots`` slots of ``slot_nbytes`` bytes
        (``False`` before the first reservation)."""
        if not self.pool.reserved:
            return False
        want = PoolLayout.for_request(slot_nbytes, slots, self.world_size, self.pool.align)
        return self.pool.slots >= slots and self.pool.slot_nbytes >= want.slot_nbytes

    def _reserve(self, slot_nbytes: int, slots: int, min_channels: int) -> bool:
        """``pool.reserve`` under the shared rules; returns whether the pool (re)allocated.

        Raises with a boundary in flight and when the reservation would GROW an allocation
        after a captured boundary completed (growth frees the buffers the graph holds). A first
        reservation of a never-reserved pool frees nothing and is allowed after captures. The
        pool itself refuses any reservation under capture; that stays the only capture-time
        refusal. Cached views are dropped before the pool frees and after it reallocated.
        The collective-specific region checks wrap this call in the subclass.
        ``slot_nbytes`` and ``slots`` must be rank-symmetric. The growth guard raises
        locally before the collective reserve, so rank-local growth would leave peers
        waiting in the pool's all-gather.
        """
        self.drain()
        if self.pool.reserved and not self._covers(slot_nbytes, slots):
            if self._captured:
                need = PoolLayout.for_request(slot_nbytes, slots, self.world_size, self.pool.align)
                raise RuntimeError(
                    f"{type(self).__name__}: the symmetric pool would grow after a CUDA graph "
                    f"captured its buffers (the pool holds {self.pool.slots} x "
                    f"{self.pool.slot_nbytes} bytes, the request needs {slots} x "
                    f"{need.slot_nbytes}); warm up the largest shape first, or run the warm-up "
                    "with capture disabled before capturing."
                )
            # Drop the views into the current allocation first so the pool can free it before
            # allocating the larger one (otherwise both are mapped at once).
            self._drop_views()
        realloc = self.pool.reserve(slot_nbytes, slots, min_channels)  # collective, eager only
        if realloc:
            self._drop_views()
        return realloc

    def _drop_views(self, slot: int | None = None) -> None:
        """Forget the cached views of every slot, or of ``slot`` only (a slot whose layout
        changes while another slot is in flight keeps that slot's views)."""
        if slot is None:
            self._views.clear()
            return
        for key in [key for key in self._views if key[0] == slot]:
            del self._views[key]

    def view(self, key: tuple, make: Callable[[], _View]) -> _View:
        """The memoized view ``make()`` for ``key``, whose first element is the slot it belongs
        to; valid for one allocation (dropped by :meth:`_reserve` on reallocation) and one slot
        layout (the protocol drops a slot's views when it rebinds the layout)."""
        view = self._views.get(key)
        if view is None:
            view = self._views[key] = make()
        return cast(_View, view)

    def _check_slot(self, slot: int) -> None:
        if not 0 <= slot < self.slots:
            raise ValueError(f"slot {slot} out of range (transport has {self.slots} slots).")

    def _require_in_flight(self, slot: int, op: str) -> None:
        self._check_slot(slot)
        if not self._in_flight[slot]:
            raise RuntimeError(
                f"{type(self).__name__}.{op}(slot={slot}) needs an open({slot}) in flight."
            )

    def _check_peer(self, peer: int, who: str) -> None:
        if peer == self.rank or not 0 <= peer < self.world_size:
            raise ValueError(
                f"{type(self).__name__}.{who}: rank {peer} is not a peer of rank {self.rank}/"
                f"{self.world_size}."
            )

    def open(self, slot: int, expected: Iterable[tuple[int, int]]) -> None:
        self._check_slot(slot)
        self.drain()
        boundary = self.ce.boundary(expected)
        boundary.__enter__()
        self._boundary = boundary
        self._capturing = _is_capturing()
        self._in_flight[slot] = True

    def done(self, slot: int) -> None:
        self._require_in_flight(slot, "done")
        boundary, self._boundary = self._boundary, None
        assert boundary is not None
        try:
            boundary.__exit__(None, None, None)
            self._captured |= self._capturing
        except Exception:
            self._needs_recovery = True
            raise
        finally:
            self._in_flight[slot] = False

    def reset(self) -> None:
        """Mark the active boundary interrupted; recover collectively on the next eager use."""
        boundary, self._boundary = self._boundary, None
        if boundary is not None:
            error = RuntimeError("Interrupted token-sharded CE boundary")
            boundary.__exit__(type(error), error, None)
            self._needs_recovery = True
        self._in_flight = [False] * self.slots

    def drain(self) -> None:
        """Finish interrupted work before preparing, reallocating or releasing its storage."""
        if any(self._in_flight):
            raise RuntimeError("CE workspace has a boundary in flight; call done() first.")
        if self._needs_recovery:
            if _is_capturing():
                raise RuntimeError("Recover CE eagerly before capturing another boundary.")
            if not self.ce.ready():
                raise CeUnavailableError("shared CE transport could not recover the boundary")
            self._needs_recovery = False

    def close(self) -> None:
        self.drain()
        self._drop_views()
        self.ce.close()


class CeState:
    """Per-group lifecycle shared by every protocol: probe once (rank-agreed), ``effective`` /
    ``reason``, one warning naming the fallback, closed-state refusal, and a registry keyed by
    group name -- ONE registry per subclass, so two protocols over one group never collide.

    A subclass builds its pool(s) and transport in :meth:`_make_transport`; the op body looks the
    state up with :meth:`get`, checks ``effective`` rank-symmetrically and runs the protocol
    inside ``try: ... except Exception: state.transport.reset(); raise``.

    Args:
        group: The process group.
        group_name: ``group.group_name`` (the registry key and the op's handle).
        timeout_ms: Device-side deadline of every signal wait in the shared pool.
    """

    _registry: dict[str, "CeState"] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._registry = {}  # one dict per subclass: protocols over one group never collide

    def __init__(
        self, group: dist.ProcessGroup, group_name: str, *, timeout_ms: int = DEFAULT_TIMEOUT_MS
    ) -> None:
        if group is None:
            raise ValueError(
                f"{type(self).__name__} needs a torch.distributed process group; got None."
            )
        if timeout_ms < 0:
            raise ValueError(f"{type(self).__name__} needs timeout_ms >= 0 (got {timeout_ms}).")
        self.group = group
        self.group_name = group_name
        self.timeout_ms = timeout_ms
        self.world_size: int = dist.get_world_size(group)
        self.rank: int = dist.get_rank(group)
        self.effective: bool = False
        """bool: Whether the protocol is in use for this group (False before the probe, after a
        failed one, after :meth:`disable` and after :meth:`close`: the caller keeps its NCCL
        path)."""
        self.reason: str = "not probed yet"
        """str: Why the protocol is not effective, or "" while it is."""
        self.pool: SymmMemPool | None = None
        self.ce: CeTransport | None = None
        self.transport: CeWorkspace | None = None
        self._probed = False
        self._closed = False
        self._warned = False  # the one fallback warning of this state
        self._what = type(self).__name__
        self._device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    # --- registry -----------------------------------------------------------------------------

    @classmethod
    def get(cls, group_name: str) -> "CeState":
        """The registered state of ``group_name``; raises ``KeyError`` when there is none."""
        state = cls._registry.get(group_name)
        if state is None:
            raise KeyError(group_name)
        return state

    @classmethod
    def register(cls, state: "CeState") -> "CeState":
        """Register ``state`` for its group name, or return the one already registered for it."""
        return cls._registry.setdefault(state.group_name, state)

    @classmethod
    def release(cls, group_name: str) -> None:
        """Close and unregister the state of ``group_name`` (no-op when there is none)."""
        state = cls._registry.get(group_name)
        if state is not None:
            state.close()

    # --- probe --------------------------------------------------------------------------------

    def probe_once(self, *, min_channels: int, what: str) -> bool:
        """Decide ``effective`` for the state's lifetime; later calls return the stored verdict.

        The shared transport factory collectively checks pool support and peer atomics,
        and prepares optional fused waits. Every rank must call eagerly in matching order.
        """
        self.require_open()
        if self._probed:
            return self.effective
        self._what = what
        if _is_capturing():
            raise RuntimeError(
                f"{what}: the first call of group {self.group_name!r} ran under CUDA-graph "
                "capture; run an eager warm-up call first."
            )
        self.ce = CeTransport.create(self.group, self._device, min_channels)
        ok = self.ce is not None
        reason = "" if ok else "CE transport unavailable"
        if self.ce is not None:
            self.pool = self.ce.pool
            self.pool.timeout_ms = self.timeout_ms
            try:
                self._make_transport()
            except Exception as exc:
                self._probed = True
                self.reason = f"{what} setup failed: {type(exc).__name__}: {exc}"
                self.ce.close()
                self.ce = None
                self.pool = None
                raise
        else:
            self._warned = True  # The shared factory already logged its detailed verdict.
        self.effective = ok
        self.reason = reason
        self._probed = True
        return ok

    def _make_transport(self) -> None:
        """Build the layout workspace over the collectively created shared transport."""
        raise NotImplementedError(f"{type(self).__name__} must implement _make_transport().")

    def _release_pool(self) -> None:
        pool, self.pool = self.pool, None
        if pool is not None:
            pool.release()

    def _warn(self, message: str) -> None:
        if not self._warned:
            logger.warning(message)
            self._warned = True

    # --- state ------------------------------------------------------------------------------

    def require_open(self) -> None:
        """Raise when the state is closed (a sibling helper still holding it gets a clear error
        rather than a silent fallback)."""
        if self._closed:
            raise RuntimeError(
                f"{self._what}: the state of group {self.group_name!r} is closed; build a new one "
                "for further calls."
            )

    def disable(self, reason: str) -> None:
        """Take the group off the protocol for the rest of the state's lifetime (``effective``
        False, ``reason`` set, the one warning if none was given yet); the pools stay allocated
        until :meth:`close`. The op body calls it when collective recovery refuses reuse."""
        self.effective = False
        self.reason = reason
        self._warn(
            f"{self._what} disabled for group {self.group_name!r} ({reason}); keeping the NCCL "
            "path."
        )

    def close(self) -> None:
        """Close the transport (releasing the pool; collective by symmetry when one exists),
        forget it, mark the state closed and unregister it. Idempotent. Call on every rank before
        the process group is destroyed, after the CUDA-graph runners' ``clear()``. A pool held
        without a transport (a subclass's setup raised after reserving it) is released too."""
        if self.transport is not None:
            self.transport.close()
        self.transport = None
        self.ce = None
        self._release_pool()
        self.effective = False
        self.reason = "closed"
        self._closed = True
        registry = type(self)._registry
        if registry.get(self.group_name) is self:
            del registry[self.group_name]
