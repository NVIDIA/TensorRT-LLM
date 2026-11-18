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
"""Move device memory between the KV cache pool and the rank failure restore.

The restore shrinks the KV cache pool for the lost experts it loads into spare slots and for the
workspace rows it remaps. Once the change is committed, every rank returns the empty spare slots
to the pool, and the replaced workspace rows once no excluded process maps them.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator, NamedTuple, Optional, Tuple

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import OutOfPagesError

if TYPE_CHECKING:
    from ..moe.fused_moe.communication.execution_state import MoeExecutionState
    from ..moe.fused_moe.moe_load_balancer import MoeLoadBalancer
    from .llm_request import LlmRequest
    from .py_executor import PyExecutor

_RELEASE_RETRY_S = 1.0


class KvCacheShrinkError(Exception):
    """The pages of requests that cannot be recomputed keep the KV cache pool from shrinking."""


def _free_device_memory() -> int:
    """Return the free device memory after emptying the PyTorch caching allocator.

    The KV cache sizing at startup reads free memory the same way. Call it only between steps.
    """
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info()[0]


def _free_plus_reserved_memory() -> int:
    """Return the free device memory plus the PyTorch caching allocator's reserve.

    Serving only moves memory between the two, so the sum grows only when memory outside the
    allocator is freed. It reads no allocator state that a step in flight changes.
    """
    return torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved()


class StartupMemory(NamedTuple):
    """This rank's KV cache quota at startup, the base of every membership change's resize."""

    kv_cache_quota: int

    def free_to_keep(self, kv_cache_manager, spare_slot_bytes: int) -> int:
        """Return the free device memory less the pool's shortfall from its quota.

        The quota is the startup quota less ``spare_slot_bytes``, the backed spare slots' size.
        """
        return _free_device_memory() - (
            self.kv_cache_quota - spare_slot_bytes - kv_cache_manager.gpu_quota
        )

    def fit_kv_cache(
        self, kv_cache_manager, spare_slot_bytes: int, needed: int, free_to_keep: int = 0
    ) -> int:
        """Resize the suspended KV cache pool and return its new quota.

        The quota is the startup quota less ``spare_slot_bytes`` and ``needed``, derived afresh at
        every change so that repeated failures do not ratchet it down. The pool grows only into
        memory free beyond ``free_to_keep``, and never shrinks for want of it. If the device cannot
        back the growth, the pool keeps its pages.
        """
        quota = self.kv_cache_quota - spare_slot_bytes - needed
        if quota <= kv_cache_manager.gpu_quota:
            return kv_cache_manager.resize_gpu_quota(quota)
        quota = max(
            kv_cache_manager.gpu_quota,
            quota - self.shortfall(kv_cache_manager, spare_slot_bytes, free_to_keep),
        )
        try:
            return kv_cache_manager.resize_gpu_quota(quota)
        except OutOfPagesError:
            return kv_cache_manager.gpu_quota

    def shortfall(self, kv_cache_manager, spare_slot_bytes: int, free_to_keep: int) -> int:
        """Return the quota bytes the pool lacks and cannot take without ``free_to_keep``."""
        return max(0, free_to_keep - self.free_to_keep(kv_cache_manager, spare_slot_bytes))


@dataclass
class _ReleaseRetry:
    """A short memory release, retried once free memory grew by half a workspace row."""

    free_to_keep: int
    free_seen: int
    check_after_s: float


def check_largest_failure_fits(
    executor: PyExecutor,
    balancer,
    startup_memory: StartupMemory,
    moe_execution_state: MoeExecutionState,
) -> None:
    """Check on every original rank that the KV cache pool can make room to restore any failure."""
    needed = (
        balancer.expert_restore.largest_failure_restore_bytes()
        + moe_execution_state.workspace_bytes()
    )
    quota = startup_memory.kv_cache_quota
    with executor.kv_pools_suspended() as kv_cache_manager:
        try:
            kept = startup_memory.fit_kv_cache(kv_cache_manager, 0, needed)
        except OutOfPagesError:
            kept = _smallest_kv_cache_quota(kv_cache_manager, max(0, quota - needed), quota)
        # A pool holding no pages shrinks to 0 bytes even when ``needed`` exceeds its quota.
        fits = quota - kept >= needed
        startup_memory.fit_kv_cache(kv_cache_manager, 0, 0)
    sizes = executor.dist.allgather((fits, needed, kept, quota))
    short = [rank for rank, (rank_fits, *_) in enumerate(sizes) if not rank_fits]
    if short:
        worst = max(short, key=lambda rank: (sizes[rank][1] + sizes[rank][2]) / sizes[rank][3])
        _, worst_needed, worst_kept, worst_quota = sizes[worst]
        # The KV cache pool takes this fraction of the memory left after weights and
        # activations, so its quota grows with the fraction.
        fraction = executor.llm_args.kv_cache_config.free_gpu_memory_fraction
        passing = math.ceil(fraction * (worst_needed + worst_kept) / worst_quota * 100) / 100
        advice = "Use more ranks, so that each rank restores fewer of a lost node's experts."
        if passing <= 1:
            advice = (
                f"Raise kv_cache_config.free_gpu_memory_fraction from {fraction} to at least "
                f"{passing:.2f}, or use more ranks, so that each rank restores fewer of a lost "
                "node's experts."
            )
        raise ValueError(
            "Wide-EP fault tolerance cannot recover from the loss of a node other than rank 0's, "
            f"or of one rank. Restoring its experts needs more device memory on ranks {short} than "
            f"their KV cache pools can release. Rank {worst} needs "
            f"{worst_needed / 2**30:.2f} GiB, and its pool of {worst_quota / 2**30:.2f} GiB "
            f"keeps up to {worst_kept / 2**30:.2f} GiB for the pages in use. {advice}"
        )


def _smallest_kv_cache_quota(kv_cache_manager, low: int, high: int) -> int:
    """Bisect the smallest quota the suspended pool takes in ``(low, high]``, to within 1%."""
    step = max(1, high // 100)
    while high - low > step:
        middle = (low + high) // 2
        try:
            high = kv_cache_manager.resize_gpu_quota(middle)
        except OutOfPagesError:
            low = middle
    return high


class RecoveryMemory:
    """Shrink the KV cache pool for a membership change's restore and release the memory after.

    The pool shrinks while the change is installed. Once serving resumes, every rank releases its
    empty spare slots and replaced workspace rows at the same step boundary. The release resizes
    the pool, which suspends requests, so a forward vote picks that boundary.
    """

    def __init__(
        self,
        executor: PyExecutor,
        moe_state: MoeExecutionState,
        startup_memory: StartupMemory,
    ):
        self._executor = executor
        self._moe = moe_state
        self._startup_memory = startup_memory
        self._rank = executor.dist.rank
        # Only the executor thread reads and writes the state below, so it takes no lock.
        self._kv_cache_free_to_keep: Optional[int] = None
        # Whether a remap replaced this rank's workspace rows while an excluded process may still
        # import them. Their pages return only once that process ended, which no event announces,
        # so the KV cache pool takes them back by measuring.
        self._workspace_rows_owed = False
        self._release_scheduled = False
        self._release_retry: Optional[_ReleaseRetry] = None
        self._release_requested = False
        # Whether a rank asked for a release in the last forward vote and no rank flagged a change.
        self._release_agreed = False

    @property
    def release_agreed(self) -> bool:
        """Return whether the last forward vote agreed on a release."""
        return self._release_agreed

    def check_restore_fits(self, restore_bytes: int, balancer: Optional[MoeLoadBalancer]) -> None:
        """Check that the KV cache pool can release ``restore_bytes``.

        The pool releases at most its startup quota less the backed spare slots, which hold the
        experts of the failures since the last rejoin, even with every request preempted.

        Raises:
            ValueError: The restore needs more device memory than the KV cache pool can release.
        """
        backed = 0 if balancer is None else balancer.expert_restore.backed_spare_slot_bytes
        quota = self._startup_memory.kv_cache_quota
        if restore_bytes > quota - backed:
            raise ValueError(
                f"Rank {self._rank} needs {restore_bytes} bytes of device memory to restore the "
                f"lost experts and workspaces, but its KV cache pool can release at most "
                f"{quota - backed} bytes: its startup quota of {quota} bytes less the {backed} "
                "bytes of spare slots that hold the experts of earlier failures"
            )

    @contextlib.contextmanager
    def kv_cache_shrunk_by(
        self,
        restore_bytes: int,
        balancer: Optional[MoeLoadBalancer],
        recomputable: Callable[[LlmRequest], bool],
    ) -> Iterator[str]:
        """Suspend the KV cache pool, shrink it by ``restore_bytes`` and yield the quota change.

        While the pages in use do not fit, the newest ``recomputable`` requests are preempted for
        recompute, doubling the number preempted with each attempt.

        Raises:
            KvCacheShrinkError: The pages in use do not fit and no request is left to preempt.
        """
        # Each rank decides on its own. Suspending the pools runs a collective only with
        # disaggregated serving, and fault tolerance rejects that.
        if not restore_bytes:
            yield "unchanged"
            return
        backed = 0 if balancer is None else balancer.expert_restore.backed_spare_slot_bytes
        preempted = 0
        while True:
            with self._executor.kv_pools_suspended() as kv_cache_manager:
                quota_before, free = kv_cache_manager.gpu_quota, _free_device_memory()
                free_to_keep = self._startup_memory.free_to_keep(kv_cache_manager, backed)
                try:
                    quota_after = self._startup_memory.fit_kv_cache(
                        kv_cache_manager, backed, restore_bytes
                    )
                except OutOfPagesError as error:
                    victims = [
                        request
                        for request in reversed(self._executor.active_requests)
                        if recomputable(request)
                    ][: max(1, preempted)]
                    if not victims:
                        raise KvCacheShrinkError(
                            f"Rank {self._rank} cannot shrink its KV cache pool of {quota_before} "
                            f"bytes by the {restore_bytes} bytes that the restore needs: the pages "
                            f"in use do not fit after {preempted} requests were preempted, and no "
                            "request is left to preempt"
                        ) from error
                else:
                    self._kv_cache_free_to_keep = free_to_keep
                    yield (
                        f"{quota_before} -> {quota_after} bytes ({restore_bytes} bytes needed, "
                        f"{free} bytes free, {preempted} requests preempted)"
                    )
                    return
            self._executor.recompute_requests(victims)
            preempted += len(victims)

    def owe_workspace_rows(self) -> None:
        """Record that a remap replaced this rank's workspace rows."""
        self._workspace_rows_owed = True

    def schedule_release(self) -> None:
        """Release the committed change's memory at the step boundary that the next vote picks."""
        self._release_scheduled = True

    def wants_release(self, forward_paused: bool) -> bool:
        """Return whether this rank asks the forward vote for a release.

        A rank whose forward a membership change pauses asks for none.
        """
        self._release_requested = not forward_paused and (
            self._release_scheduled
            or (self._release_retry is not None and self._release_retry_due())
        )
        return self._release_requested

    def record_release_vote(self, agreed: bool) -> None:
        """Record whether the forward vote agreed on a release."""
        self._release_agreed = agreed

    def end_release_vote(self) -> bool:
        """Clear the agreed release and return whether this rank asked for it."""
        self._release_agreed = False
        requested, self._release_requested = self._release_requested, False
        return requested

    def release(self, balancer: Optional[MoeLoadBalancer], generation: int) -> None:
        """Release committed ``generation``'s memory, or retry a release that fell short.

        Call it only after ``end_release_vote`` returned True.
        """
        if self._release_scheduled:
            self._release_scheduled = False
            start_s = time.perf_counter()
            released = (
                0 if balancer is None else balancer.expert_restore.release_empty_spare_slots()
            )
            kept, self._kv_cache_free_to_keep = self._kv_cache_free_to_keep, None
            if released or kept is not None:
                quota, short = self._grow_kv_cache_pool(balancer, kept or 0)
                logger.info(
                    f"Rank {self._rank} returned the empty spare slots of membership generation "
                    f"{generation} to the KV cache pool: released {released} "
                    f"bytes, KV cache quota {quota} bytes, {short} bytes short until device memory "
                    f"is freed, in {(time.perf_counter() - start_s) * 1e3:.0f} ms"
                )
            return
        quota, short = self._grow_kv_cache_pool(balancer, self._release_retry.free_to_keep)
        logger.info(
            f"Rank {self._rank} grew the KV cache pool into device memory freed since membership "
            f"generation {generation} was committed, KV cache quota {quota} bytes, "
            f"{short} bytes short"
        )

    def _release_retry_due(self) -> bool:
        """Return whether free memory grew by half a workspace row, checked at most once a second.

        Half a row still detects a returned row when other memory was taken in the meantime.
        """
        retry = self._release_retry
        now_s = time.monotonic()
        if now_s < retry.check_after_s:
            return False
        retry.check_after_s = now_s + _RELEASE_RETRY_S
        return _free_plus_reserved_memory() - retry.free_seen >= self._moe.workspace_bytes() // 2

    def _grow_kv_cache_pool(
        self, balancer: Optional[MoeLoadBalancer], free_to_keep: int
    ) -> Tuple[int, int]:
        """Grow the KV cache pool towards its quota, retrying later while workspace rows are owed.

        Returns:
            The new quota and the bytes the pool lacks of it while ``free_to_keep`` stays free.
        """
        backed = 0 if balancer is None else balancer.expert_restore.backed_spare_slot_bytes
        with self._executor.kv_pools_suspended() as kv_cache_manager:
            quota = self._startup_memory.fit_kv_cache(kv_cache_manager, backed, 0, free_to_keep)
            short = self._startup_memory.shortfall(kv_cache_manager, backed, free_to_keep)
        if not short:
            self._workspace_rows_owed = False
        self._release_retry = None
        if self._workspace_rows_owed:
            self._release_retry = _ReleaseRetry(
                free_to_keep, _free_plus_reserved_memory(), time.monotonic() + _RELEASE_RETRY_S
            )
        return quota, short
