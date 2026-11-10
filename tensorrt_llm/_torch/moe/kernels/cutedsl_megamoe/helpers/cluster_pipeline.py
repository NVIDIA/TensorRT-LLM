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
"""Cluster-scope waits and releases for pipelines whose stage is handed between CTAs.

A ``cutlass.pipeline`` stage that one CTA publishes into another CTA's shared memory
(``st.async`` or ``cp.async.bulk`` with ``mbarrier::complete_tx``) is released at cluster
scope by a thread of the publishing CTA. The pipeline's default ``consumer_wait`` acquires at
CTA scope, and its default ``consumer_release`` arrives on the publisher's empty barrier with
a CTA-scope release that the publisher's default ``producer_acquire`` acquires at CTA scope.
Under the PTX memory model a release and an acquire synchronize only when each scope includes
the other thread, so neither half of such a handoff is ordered without cluster scope.

The helpers here keep the pipeline object and its state and change only the scope: the full
and empty barriers, stage index and phase are the pipeline's own. For a static single-CTA
cluster they fall back to the pipeline's default operations, which are correct and cheaper.

The cluster-scope acquire is a relaxed cluster-scope ``mbarrier.try_wait`` followed by one
``fence.acquire.sync_restrict::shared::cluster.cluster``, the PTX ISA's acquire pattern for an
observed release (memory consistency model, release and acquire patterns). The
``.acquire.cluster`` qualifier on the wait itself would give the same ordering, but the
assembler then adds an L1 invalidation after every try of the wait loop; the restricted fence
after the loop costs nothing in the generated code.
"""

from typing import Union

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir.dialects import nvvm as _nvvm
from cutlass.cutlass_dsl import Int32
from cutlass.experimental import primitives as prims


def _is_static_single_cta(cluster_size: Union[int, Int32, None]) -> bool:
    return isinstance(cluster_size, int) and cluster_size == 1


@cute.jit
def mbarrier_wait_acquire_cluster(mbar_ptr: cute.Pointer, phase: Int32) -> None:
    """Wait for a phase of this CTA's mbarrier with cluster-scope acquire semantics.

    Use it where the phase is completed by a thread of another CTA: a peer's ``st.async``
    complete-tx or a peer's remote arrive. The default ``.acquire.cta`` wait does not
    synchronize with those releases, so shared memory they wrote would be read unordered.
    Spins on ``mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64`` (the default wait's
    suspend hint) and then issues ``fence.acquire.sync_restrict::shared::cluster.cluster``.
    """
    phase_complete = prims.mbarrier_try_wait_parity(
        mbar_ptr, phase, scope=prims.MBarrierScope.CLUSTER, order=prims.MemOrder.RELAXED
    )
    while not phase_complete:
        phase_complete = prims.mbarrier_try_wait_parity(
            mbar_ptr, phase, scope=prims.MBarrierScope.CLUSTER, order=prims.MemOrder.RELAXED
        )
    prims.fence_sync_restrict(prims.MemOrder.ACQUIRE)


@cute.jit
def consumer_wait_cluster(
    pipe: pipeline.PipelineAsync,
    state: pipeline.PipelineState,
    cluster_size: Union[int, Int32, None] = None,
) -> None:
    """``consumer_wait`` for a stage another CTA publishes into this CTA's shared memory.

    Waits on the pipeline's full barrier for ``state`` with ``.acquire.cluster`` semantics,
    so the read of the published stage is ordered after the publisher's cluster-scope
    release. ``cluster_size`` may be passed to keep the default wait for a static
    single-CTA cluster.
    """
    if cutlass.const_expr(_is_static_single_cta(cluster_size)):
        pipe.consumer_wait(state)
    else:
        mbarrier_wait_acquire_cluster(pipe.sync_object_full.get_barrier(state.index), state.phase)


@cute.jit
def consumer_release_cluster(
    pipe: pipeline.PipelineAsync,
    state: pipeline.PipelineState,
    cluster_size: Union[int, Int32, None] = None,
    publisher_cta_rank: int = 0,
) -> None:
    """``consumer_release`` for a stage that the publishing CTA will overwrite.

    Arrives on the publisher's empty barrier for ``state`` with ``.release.cluster``
    semantics, so this CTA's read of the stage is ordered before the publisher's next
    store into it. ``cluster_size`` may be passed to keep the default arrive for a static
    single-CTA cluster.
    """
    if cutlass.const_expr(_is_static_single_cta(cluster_size)):
        pipe.sync_object_empty.arrive(state.index, Int32(publisher_cta_rank))
    else:
        cute.arch.mbarrier_arrive(
            pipe.sync_object_empty.get_barrier(state.index),
            Int32(publisher_cta_rank),
            scope=_nvvm.MemScopeKind.CLUSTER,
        )


@cute.jit
def producer_acquire_cluster(
    pipe: pipeline.PipelineAsync,
    state: pipeline.PipelineState,
    cluster_size: Union[int, Int32, None] = None,
) -> None:
    """``producer_acquire`` for a stage that consumers in other CTAs release.

    Waits on the pipeline's empty barrier for ``state`` with ``.acquire.cluster`` semantics,
    so the next store into the stage is ordered after every consumer's cluster-scope release,
    then runs the pipeline's own ``producer_acquire`` with the wait skipped (for a CLC fetch
    pipeline that is the arrive with transaction count on every CTA's full barrier).
    ``cluster_size`` may be passed to keep the default acquire for a static single-CTA
    cluster.
    """
    if cutlass.const_expr(_is_static_single_cta(cluster_size)):
        pipe.producer_acquire(state)
    else:
        mbarrier_wait_acquire_cluster(pipe.sync_object_empty.get_barrier(state.index), state.phase)
        # ``try_acquire_token=True`` says the empty barrier is already acquired. ``PipelineAsync``
        # and ``PipelineClcFetchAsync`` (CuTe DSL 4.8.0) skip their CTA-scope wait on a truthy
        # token and run only what follows it, which for the CLC fetch pipeline is the arrive with
        # transaction count on every CTA's full barrier. Should a DSL upgrade change that contract,
        # this would turn into a second, CTA-scope wait: keep it in step with ``producer_acquire``.
        pipe.producer_acquire(state, try_acquire_token=True)


@cute.jit
def producer_tail_cluster(
    pipe: pipeline.PipelineAsync,
    state: pipeline.PipelineState,
    cluster_size: Union[int, Int32, None] = None,
) -> None:
    """``producer_tail`` for a stage that consumers in other CTAs release.

    Drains every stage's empty barrier before the producer exits, as the pipeline's own
    ``producer_tail`` does, with ``.acquire.cluster`` waits to match the consumers'
    cluster-scope releases. ``cluster_size`` may be passed to keep the default tail for a
    static single-CTA cluster.
    """
    if cutlass.const_expr(_is_static_single_cta(cluster_size)):
        pipe.producer_tail(state)
    else:
        for _ in cutlass.range_constexpr(pipe.num_stages):
            mbarrier_wait_acquire_cluster(
                pipe.sync_object_empty.get_barrier(state.index), state.phase
            )
            state.advance()


__all__ = [
    "consumer_release_cluster",
    "consumer_wait_cluster",
    "mbarrier_wait_acquire_cluster",
    "producer_acquire_cluster",
    "producer_tail_cluster",
]
