# {$nv-internal-release file}
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, math, nvvm, vector
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05.mma import CollectorOp
from cutlass.cute.testing import assert_ as runtime_assert
from . import manual_mma_128dp  # packaging: bare `import manual_mma_128dp` -> relative for trtllm package (kernel logic unchanged)
from cutlass.cutlass_dsl import (
    T,
    dsl_user_op,
    extract_mlir_values,
    if_generate,
    new_from_mlir_values,
)
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineAsync,
    PipelineOp,
    PipelineState,
    agent_sync,
    pipeline_init_arrive,
    pipeline_init_wait,
)
from cutlass.pipeline.sm90 import MbarrierArray
from cutlass.utils.gemm.sm100 import (
    transform_partitioned_tensor_layout,
)


# ============================================================================
# Inline utility functions
# ============================================================================


@dataclass(frozen=True)
class Fc12TaskShape:
    """Exact per-M task counts for the two GEMM phases."""

    fc1_tiles_per_m: int
    fc2_tiles_per_m: int


@dataclass(frozen=True)
class FusedSmemStageBytes:
    """Byte model matching the fused kernel's aligned shared-storage fields."""

    a_per_ab_stage: int
    b_per_ab_stage: int
    sfa_per_ab_stage: int
    sfb_per_ab_stage: int
    fc1_c_per_stage: int
    fc2_c: int
    metadata: int
    header_fixed: int
    header_per_ab_stage: int
    alignment: int

    def total(self, *, ab_stages: int, fc1_c_stages: int) -> int:
        """Return the exact aligned storage size for one stage candidate."""

        def align_up(value: int) -> int:
            return (
                (value + self.alignment - 1) // self.alignment
            ) * self.alignment

        header = align_up(
            self.header_fixed + ab_stages * self.header_per_ab_stage
        )
        epilogue = align_up(
            max(fc1_c_stages * self.fc1_c_per_stage, self.fc2_c)
        )
        ab_buffers = sum(
            align_up(ab_stages * bytes_per_stage)
            for bytes_per_stage in (
                self.sfa_per_ab_stage,
                self.a_per_ab_stage,
                self.b_per_ab_stage,
                self.sfb_per_ab_stage,
            )
        )
        return align_up(header + epilogue + ab_buffers + self.metadata)


@dataclass(frozen=True)
class FusedSmemStageConfig:
    """Selected common stage counts for both fused GEMM phases."""

    ab: int
    fc1_c: int


def select_fused_smem_stages(
    *,
    capacity: int,
    preferred_ab: int,
    preferred_fc1_c: int,
    minimum_fc1_c: int,
    stage_bytes: FusedSmemStageBytes,
) -> FusedSmemStageConfig:
    """Choose the highest-priority fused stage pair that fits shared memory.

    AB depth is prioritized over FC1 epilogue depth. This preserves the
    standalone FC1 preference when possible while allowing C depth to shrink
    when fused-only storage makes the standalone combination too large.
    """
    for ab_stages in range(preferred_ab, 0, -1):
        for fc1_c_stages in range(
            preferred_fc1_c, minimum_fc1_c - 1, -1
        ):
            if stage_bytes.total(
                ab_stages=ab_stages, fc1_c_stages=fc1_c_stages
            ) <= capacity:
                return FusedSmemStageConfig(
                    ab=ab_stages, fc1_c=fc1_c_stages
                )
    raise ValueError(
        "no fused FC12 shared-memory stage configuration fits: "
        f"capacity={capacity}, preferred_ab={preferred_ab}, "
        f"preferred_fc1_c={preferred_fc1_c}, "
        f"minimum_fc1_c={minimum_fc1_c}"
    )


FC1_PHASE = 0
FC2_PHASE = 1
END_PHASE = -1


@dsl_user_op
def blk_reduce_bf16(dst_gemm, src_smem, size, loc=None, ip=None):
    """Scatter-add one BF16 row from shared memory into global memory."""
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
        "[$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dataclass(frozen=True)
class Fc12Task:
    """One descriptor in a resident CTA's fused FC12 task stream."""

    phase: int
    m_tile: int
    n_tile: int


def derive_fc12_task_shape(
    *, fc1_gemm_n: int, fc2_n: int, mma_n: int
) -> Fc12TaskShape:
    """Derive exact FC1 and FC2 N-tile counts for the fused scheduler.

    One FC1 MMA N tile contains paired up/gate values. SwiGLU halves both
    the logical output width and the per-tile output width, so the number of
    real FC1 tasks remains ``fc1_gemm_n / mma_n``.
    """
    if mma_n <= 0 or mma_n % 2 != 0:
        raise ValueError(f"MMA N must be a positive even value, got {mma_n}")
    if fc1_gemm_n <= 0 or fc1_gemm_n % mma_n != 0:
        raise ValueError(
            f"FC1 GEMM N must be positive and divisible by MMA N: "
            f"{fc1_gemm_n} vs {mma_n}"
        )
    if fc2_n <= 0 or fc2_n % mma_n != 0:
        raise ValueError(
            f"FC2 N must be positive and divisible by MMA N: {fc2_n} vs {mma_n}"
        )
    return Fc12TaskShape(
        fc1_tiles_per_m=fc1_gemm_n // mma_n,
        fc2_tiles_per_m=fc2_n // mma_n,
    )


def derive_fc1_ready_expected(
    *, fc1_gemm_n: int, mma_n: int, mma_cta_group_size: int
) -> int:
    """Return FC1 publications required before one logical M tile is ready.

    Every CTA in a cooperative MMA group stores its own M-row partition and
    publishes once per FC1 N tile. FC2 may read the complete logical M tile
    only after all CTA-side stores have completed.
    """
    if mma_cta_group_size not in (1, 2):
        raise ValueError(
            f"mma_cta_group_size must be 1 or 2, got {mma_cta_group_size}"
        )
    task_shape = derive_fc12_task_shape(
        fc1_gemm_n=fc1_gemm_n,
        fc2_n=mma_n,
        mma_n=mma_n,
    )
    return task_shape.fc1_tiles_per_m * mma_cta_group_size


def derive_fused_ab_mbarrier_array_count(mma_cta_group_size: int) -> int:
    """Return the number of full/empty AB-pipeline mbarrier arrays.

    Both modes own independent FC1 A/B and FC2 A/B pipelines. The 2CTA mode
    additionally owns an A/SFA relay pipeline that publishes per-CTA cp.async
    completion to the cooperative MMA group.
    """
    if mma_cta_group_size not in (1, 2):
        raise ValueError(
            "mma_cta_group_size must be 1 or 2, got "
            f"{mma_cta_group_size}"
        )
    return 4 + (mma_cta_group_size - 1)


def validate_l2_atomic_descriptor_bounds(
    *,
    phase: str,
    gemm_shape: tuple[int, int, int],
    cta_tile_shape_mnk: tuple[int, int, int],
) -> None:
    """Reject phase coordinates that cannot fit the packed cluster response.

    Multi-CTA L2 scheduling publishes the cluster-origin M, N, and L tile
    coordinates as three unsigned 16-bit fields. The receiving CTA adds its
    local cluster coordinate after unpacking, so the complete CTA tile count
    in each published mode must fit in 16 bits.
    """
    if not phase:
        raise ValueError("phase must be non-empty")
    if len(gemm_shape) != 3 or len(cta_tile_shape_mnk) != 3:
        raise ValueError("GEMM and CTA tile shapes must each have three modes")
    if any(dim <= 0 for dim in (*gemm_shape, *cta_tile_shape_mnk)):
        raise ValueError("GEMM and CTA tile shape dimensions must be positive")

    m, n, l = gemm_shape
    tile_m, tile_n, _ = cta_tile_shape_mnk
    coordinate_count_limit = 1 << 16
    tile_counts = (
        ("M", (m + tile_m - 1) // tile_m),
        ("N", (n + tile_n - 1) // tile_n),
        ("L", l),
    )
    for mode, tile_count in tile_counts:
        if tile_count > coordinate_count_limit:
            raise ValueError(
                f"{phase} {mode} tile count {tile_count} exceeds the "
                f"L2 atomic descriptor limit {coordinate_count_limit}"
            )


def derive_fc12_cta_task_stream(
    *,
    num_m_tiles: int,
    task_shape: Fc12TaskShape,
    num_resident_ctas: int,
    cta_idx: int,
) -> Tuple[Fc12Task, ...]:
    """Model the exact 1CTA device schedule for one resident CTA.

    Each phase owns an independent grid-stride task space. N is the inner
    coordinate, matching the unswizzled persistent scheduler used by the
    first kernel configuration. Reinitializing the linear index to ``cta_idx``
    for FC2 lets fast CTAs enter FC2 without waiting for other CTAs.
    """
    if num_m_tiles <= 0:
        raise ValueError(f"num_m_tiles must be positive, got {num_m_tiles}")
    if num_resident_ctas <= 0:
        raise ValueError(
            f"num_resident_ctas must be positive, got {num_resident_ctas}"
        )
    if cta_idx < 0 or cta_idx >= num_resident_ctas:
        raise ValueError(
            f"cta_idx must be in [0, {num_resident_ctas}), got {cta_idx}"
        )

    stream = []
    for phase, tiles_per_m in (
        (FC1_PHASE, task_shape.fc1_tiles_per_m),
        (FC2_PHASE, task_shape.fc2_tiles_per_m),
    ):
        if tiles_per_m <= 0:
            raise ValueError(
                f"phase {phase} tiles_per_m must be positive, got {tiles_per_m}"
            )
        phase_task_count = num_m_tiles * tiles_per_m
        linear_idx = cta_idx
        while linear_idx < phase_task_count:
            stream.append(
                Fc12Task(
                    phase=phase,
                    m_tile=linear_idx // tiles_per_m,
                    n_tile=linear_idx % tiles_per_m,
                )
            )
            linear_idx += num_resident_ctas

    stream.append(Fc12Task(phase=END_PHASE, m_tile=-1, n_tile=-1))
    return tuple(stream)


class L2AtomicPersistentTileScheduler:
    """CGA-coherent persistent scheduler with a static first wave.

    The first work item is the physical cluster ID. Later items are claimed
    from one GPU-scope counter and offset by the persistent cluster count, so
    the static and dynamic ranges never overlap. For a 2CTA MMA group, only
    the leader CTA claims work and publishes the packed descriptor to every
    CTA in the cluster. The 1CTA path returns the warp-broadcast claim
    directly and therefore does not issue a remote-SMEM store to rank zero.
    """

    def __init__(
        self,
        params: utils.PersistentTileSchedulerParams,
        cta_id_in_cluster: cute.Coord,
        block_idx,
        grid_dim,
        response_smem_ptr: cute.Pointer,
        counter_gmem_ptr: cute.Pointer,
    ):
        self.params = params
        self.cta_id_in_cluster = cta_id_in_cluster
        self._block_idx = block_idx
        self._grid_dim = grid_dim
        self._response_smem_ptr = response_smem_ptr
        self._counter_gmem_ptr = counter_gmem_ptr

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values = extract_mlir_values(self.cta_id_in_cluster)
        values.extend(extract_mlir_values(self._block_idx))
        values.extend(extract_mlir_values(self._grid_dim))
        values.extend(extract_mlir_values(self._response_smem_ptr))
        values.extend(extract_mlir_values(self._counter_gmem_ptr))
        return values

    def __new_from_mlir_values__(
        self, values: list[ir.Value]
    ) -> "L2AtomicPersistentTileScheduler":
        if len(values) != 11:
            raise ValueError(f"expected 11 scheduler values, got {len(values)}")
        return L2AtomicPersistentTileScheduler(
            self.params,
            new_from_mlir_values(self.cta_id_in_cluster, values[0:3]),
            new_from_mlir_values(self._block_idx, values[3:6]),
            new_from_mlir_values(self._grid_dim, values[6:9]),
            new_from_mlir_values(self._response_smem_ptr, [values[9]]),
            new_from_mlir_values(self._counter_gmem_ptr, [values[10]]),
        )

    @staticmethod
    @dsl_user_op
    def create(
        params: utils.PersistentTileSchedulerParams,
        block_idx,
        grid_dim,
        response_smem_ptr: cute.Pointer,
        counter_gmem_ptr: cute.Pointer,
        *,
        loc=None,
        ip=None,
    ) -> "L2AtomicPersistentTileScheduler":
        bidx, bidy, _ = block_idx
        cta_id_in_cluster = (
            cutlass.Int32(bidx % params.cluster_shape_mn[0]),
            cutlass.Int32(bidy % params.cluster_shape_mn[1]),
            cutlass.Int32(0),
        )
        return L2AtomicPersistentTileScheduler(
            params,
            cta_id_in_cluster,
            block_idx,
            grid_dim,
            response_smem_ptr,
            counter_gmem_ptr,
        )

    @dsl_user_op
    def initial_work_tile_info(self, *, loc=None, ip=None) -> utils.WorkTileInfo:
        """Return this cluster's static first-wave assignment."""
        _, _, cluster_idx = self._block_idx
        return self._work_tile_from_linear_idx(cluster_idx, loc=loc, ip=ip)

    @dsl_user_op
    def claim_next_work_local(self, *, loc=None, ip=None) -> utils.WorkTileInfo:
        """Claim and decode one work item without a cross-CTA publication."""
        linear_idx = self._claim_next_linear_idx(loc=loc, ip=ip)
        return self._work_tile_from_linear_idx(linear_idx, loc=loc, ip=ip)

    @dsl_user_op
    def publish_next_work(self, mbarrier_addr, *, loc=None, ip=None) -> None:
        """Claim one work item and publish it to every CTA in the cluster."""
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = tidx % 32
        linear_idx = self._claim_next_linear_idx(loc=loc, ip=ip)
        work_tile = self._work_tile_from_linear_idx(linear_idx, loc=loc, ip=ip)
        m_idx, n_idx, l_idx = work_tile.tile_idx
        packed = (
            (cutlass.Int64(m_idx) << 48)
            | (cutlass.Int64(n_idx) << 32)
            | (cutlass.Int64(l_idx) << 16)
            | cutlass.Int64(work_tile.is_valid_tile)
        )

        def publish_to_peer(value):
            value = cutlass.Int64(value)
            cute.arch.store_async_dsmem(
                self._response_smem_ptr,
                (
                    cutlass.Int32(value & cutlass.Int64(0xFFFF_FFFF)),
                    cutlass.Int32(value >> 32),
                ),
                mbarrier_addr,
                lane_idx,
                loc=loc,
                ip=ip,
            )

        if_generate(
            lane_idx < cute.size(self.params.cluster_shape_mn),
            publish_to_peer,
            input_args=[packed],
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def get_published_work(self, *, loc=None, ip=None) -> utils.WorkTileInfo:
        """Decode the work descriptor published into this CTA's SMEM."""
        response = cute.make_tensor(
            cute.recast_ptr(
                self._response_smem_ptr,
                dtype=cutlass.Int64,
                loc=loc,
                ip=ip,
            ),
            cute.make_layout(1, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        response_vec = response.load(loc=loc, ip=ip)
        packed = vector.extract(response_vec.ir_value(loc=loc, ip=ip), [], [0])
        cute.arch.fence_proxy("async.shared", space="cta")
        cta_m, cta_n, _ = self.cta_id_in_cluster
        return utils.WorkTileInfo(
            (
                cute.arch.make_warp_uniform(
                    cutlass.Int32((packed >> 48) & 0xFFFF) + cta_m
                ),
                cute.arch.make_warp_uniform(
                    cutlass.Int32((packed >> 32) & 0xFFFF) + cta_n
                ),
                cute.arch.make_warp_uniform(
                    cutlass.Int32((packed >> 16) & 0xFFFF)
                ),
            ),
            cute.arch.make_warp_uniform(cutlass.Boolean(packed & 0x1)),
        )

    @dsl_user_op
    @cute.jit
    def _claim_next_linear_idx(self, *, loc=None, ip=None) -> cutlass.Int32:
        """Return ``persistent_clusters + atomic_ticket`` to the scheduler warp."""
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = tidx % 32
        linear_idx = cutlass.Int32(0)
        if lane_idx == 0:
            linear_idx = cute.arch.atomic_add(
                self._counter_gmem_ptr,
                cutlass.Int32(1),
                sem="relaxed",
                scope="gpu",
                loc=loc,
                ip=ip,
            )
        linear_idx = cute.arch.shuffle_sync(linear_idx, 0)
        return linear_idx + cutlass.Int32(self._grid_dim[2])

    @dsl_user_op
    def _work_tile_from_linear_idx(
        self, linear_idx: cutlass.Int32, *, loc=None, ip=None
    ) -> utils.WorkTileInfo:
        problem_size = cute.size(
            self.params.problem_layout_ncluster_mnl, loc=loc, ip=ip
        )
        is_valid = linear_idx < problem_size
        if cutlass.const_expr(self.params.swizzle_size == 1):
            cluster_minor_batch, cluster_major = divmod(
                linear_idx, self.params.cluster_shape_major_fdd
            )
            batch_l, cluster_minor = divmod(
                cluster_minor_batch, self.params.cluster_shape_minor_fdd
            )
            cluster_m = cluster_minor
            cluster_n = cluster_major
        else:
            cluster_m, cluster_n, batch_l = (
                self.params.problem_layout_ncluster_mnl.get_flat_coord(
                    linear_idx, loc=loc, ip=ip
                )
            )

        cta_m, cta_n, _ = self.cta_id_in_cluster
        return utils.WorkTileInfo(
            (
                cute.arch.make_warp_uniform(
                    cutlass.Int32(cluster_m)
                    * cutlass.Int32(self.params.cluster_shape_mn[0])
                    + cta_m
                ),
                cute.arch.make_warp_uniform(
                    cutlass.Int32(cluster_n)
                    * cutlass.Int32(self.params.cluster_shape_mn[1])
                    + cta_n
                ),
                cute.arch.make_warp_uniform(cutlass.Int32(batch_l)),
            ),
            cute.arch.make_warp_uniform(is_valid),
        )


# PipelineCpAsyncUmma: CpAsync producer → UMMA consumer handshake
# with cp.async-typed mbarrier. Used for the cpasync-A path.
@dataclass(frozen=True)
class PipelineCpAsyncUmma(PipelineAsync):
    """
    PipelineCpAsyncUmma is used for CpAsync producers and UMMA consumers.

    This pipeline is specifically designed for scenarios where:
    - Producers use CpAsync instructions to load data from global to shared memory
    - Consumers are UMMA warps that perform MMA operations using the loaded data

    Key differences from PipelineAsyncUmma:
    - Uses AsyncLoad producer type for proper cp.async synchronization
    - Suitable for gather/permutation operations during load
    - Used in this kernel for the A matrix with token-based gather addressing
    """

    cta_group: cute.nvgpu.tcgen05.CtaGroup

    @staticmethod
    def _compute_leading_cta_rank(cta_v_size):
        """Computes the leading CTA rank."""
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        return cta_rank_in_cluster // cta_v_size * cta_v_size

    @staticmethod
    def _compute_peer_cta_mask(cta_layout_vmnk: cute.Layout):
        """Computes a mask for signaling arrivals to multicasting threadblocks."""
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        mask_self = cute.nvgpu.cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=0
        )
        block_in_cluster_coord_vmnk_peer = (
            cta_in_cluster_coord_vmnk[0] ^ 1,
            *cta_in_cluster_coord_vmnk[1:],
        )
        mask_peer = cute.nvgpu.cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=0
        )
        return mask_self | mask_peer

    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        barrier_storage: cute.Pointer = None,
        cta_layout_vmnk: Optional[cute.Layout] = None,
        defer_sync: bool = False,
    ):
        """Creates and initializes a new PipelineCpAsyncUmma instance."""
        if not isinstance(barrier_storage, cute.Pointer):
            raise ValueError(
                f"Expected barrier_storage to be a cute.Pointer, but got {type(barrier_storage)}"
            )

        producer_type = PipelineOp.AsyncLoad
        consumer_type = PipelineOp.TCGen05Mma

        producer = (producer_type, producer_group)
        consumer = (consumer_type, consumer_group)

        sync_object_full = MbarrierArray(
            barrier_storage=barrier_storage.align(min_align=8),
            num_stages=num_stages,
            agent=producer,
        )
        sync_object_empty = MbarrierArray(
            barrier_storage=barrier_storage.align(min_align=8) + num_stages,
            num_stages=num_stages,
            agent=consumer,
        )

        cta_v_size = (
            cute.size(cta_layout_vmnk, mode=[0]) if cta_layout_vmnk is not None else 1
        )
        cta_group = (
            cute.nvgpu.tcgen05.CtaGroup.ONE
            if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, mode=[0]) == 1
            else cute.nvgpu.tcgen05.CtaGroup.TWO
        )
        if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, mode=[0]) == 1:
            producer_mask = None
            consumer_mask = None
        else:
            producer_mask = PipelineCpAsyncUmma._compute_leading_cta_rank(cta_v_size)
            consumer_mask = PipelineCpAsyncUmma._compute_peer_cta_mask(cta_layout_vmnk)

        if not defer_sync:
            if cta_layout_vmnk is None or cute.size(cta_layout_vmnk) == 1:
                agent_sync(Agent.ThreadBlock)
            else:
                agent_sync(Agent.ThreadBlockCluster, is_relaxed=True)

        return PipelineCpAsyncUmma(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            consumer_mask,
            cta_group,
        )

    def consumer_release(self, state: PipelineState):
        """UMMA consumer release buffer empty, cta_group needs to be provided."""
        self.sync_object_empty.arrive(state.index, self.consumer_mask, self.cta_group)


@dsl_user_op
def fmin(
    a: Union[float, cutlass.Float32],
    b: Union[float, cutlass.Float32],
    *,
    nan=False,
    loc=None,
    ip=None,
) -> cutlass.Float32:
    return cutlass.Float32(
        nvvm.fmin(
            cutlass.Float32(a).ir_value(loc=loc, ip=ip),
            cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            nan=nan,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fclip_xorsign(
    a: Union[float, cutlass.Float32],
    limit: Union[float, cutlass.Float32],
    *,
    loc=None,
    ip=None,
) -> cutlass.Float32:
    """Clip to ``[-limit, limit]`` with PTX ``min.xorsign.abs.f32``."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(limit).ir_value(loc=loc, ip=ip),
            ],
            "min.xorsign.abs.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def red_add_release_gpu(base_tensor, index, *, loc=None, ip=None):
    """Release-ordered atomic increment of a uint32 global counter."""
    llvm.inline_asm(
        None,
        [base_tensor.iterator.llvm_ptr, index.ir_value()],
        "{"
        ".reg .b64 addr;"
        " cvt.u64.s32 addr, $1;"
        " shl.b64 addr, addr, 2;"
        " add.u64 addr, $0, addr;"
        " red.release.gpu.global.add.u32 [addr], 1;"
        "}",
        "l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_acquire_gpu(base_tensor, index, *, loc=None, ip=None):
    """Acquire-ordered uint32 load from a per-M readiness counter."""
    result = llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [base_tensor.iterator.llvm_ptr, index.ir_value()],
        "{"
        ".reg .b64 addr;"
        " cvt.u64.s32 addr, $2;"
        " shl.b64 addr, addr, 2;"
        " add.u64 addr, $1, addr;"
        " ld.acquire.gpu.global.b32 $0, [addr];"
        "}",
        "=r,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


@dsl_user_op
def nanosleep(ns, *, loc=None, ip=None):
    """Back off a polling warp while it waits for FC1 readiness."""
    llvm.inline_asm(
        None,
        [ns.ir_value()],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def sigmoid_f32(
    a: Union[float, cutlass.Float32], fastmath: bool = False
) -> Union[float, cutlass.Float32]:
    """Compute the sigmoid of the input tensor."""
    return cute.arch.rcp_approx(1.0 + cute.math.exp(-a, fastmath=fastmath))


def silu_f32(
    a: Union[float, cutlass.Float32], fastmath: bool = False
) -> Union[float, cutlass.Float32]:
    """Compute the silu of the input tensor."""
    return a * sigmoid_f32(a, fastmath=fastmath)


class S2TCopyBundle(NamedTuple):
    """Bundle of tiled copy and partitioned tensors for smem-to-tmem copies."""

    tiled_copy: cute.TiledCopy
    sSF_compact: cute.Tensor  # Partitioned source (smem)
    tSF_compact: cute.Tensor  # Partitioned destination (tmem)


@dsl_user_op
def sm100_tcgen05_st_32x32b_x4(
    tmem_addr,
    r0,
    r1,
    r2,
    r3,
    *,
    loc=None,
    ip=None,
):
    """Issue one tcgen05.st.sync.aligned.32x32b.x4.b32.

    Writes 4 32-bit cells per lane to TMEM[lane_offset, col_base..col_base+3].
    Used by SFA transform warps to write LDS+repacked SF data into TMEM
    bypassing cute.copy auto-partition (which over-tiles the multi-mode
    tCtFC1SFA_layout into x128 STTMs that exceed reg budget).
    """
    addr_i32 = cutlass.Uint32(tmem_addr).ir_value(loc=loc, ip=ip)
    r0_i32 = cutlass.Uint32(r0).ir_value(loc=loc, ip=ip)
    r1_i32 = cutlass.Uint32(r1).ir_value(loc=loc, ip=ip)
    r2_i32 = cutlass.Uint32(r2).ir_value(loc=loc, ip=ip)
    r3_i32 = cutlass.Uint32(r3).ir_value(loc=loc, ip=ip)
    asm = "tcgen05.st.sync.aligned.32x32b.x4.b32 [$0], {$1, $2, $3, $4};"
    llvm.inline_asm(
        None,
        [addr_i32, r0_i32, r1_i32, r2_i32, r3_i32],
        asm,
        "r, r, r, r, r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


"""
Rubin (SM107) persistent blockscaled contiguous grouped GEMM with token gather
and fused SwiGLU activation (FC1 of MoE).

Compute:
  fc1_acc = fc1_alpha * (SFA * A[token_ids]) * (SFB * B)
  C       = up * silu(gate)                                # SwiGLU on interleaved acc
  fc2_acc = fc2_alpha * (SFC * C) * (FC2_SFB * FC2_B)
  + optional NVFP4 quantization (generates SFC) when c_dtype == Float4E2M1FN.

Shapes: A is M×K×1; B is N×K×L (L = num experts), interleaved [up, gate] at
granularity=64; C is M×(N/2)×1 (N halved by SwiGLU). SFA/SFB layouts follow
BlockScaledBasicChunk. ``permuted_idx_to_expanded_idx`` is shared by both
phases: FC1 divides by ``topk`` to recover the receive row, while FC2 also
uses the remainder to select the route scale.

Within a tile, valid_m varies per group; padding rows are handled at load by
predicating CpAsync on `abs_row < mn_limit`.

Constraints: A/B share dtype (mxf8 | mxf4 | nvf4); mma_tiler M in {128, 256};
mma_tiler N in {64, 128, 192, 256}; cluster M/N pow-2, total ≤ 16;
contiguous dim ≥ 16B aligned (16/32 elems for f8/f4).

For CUDA graph, A/C/SFA/permuted_idx_to_expanded_idx/tile_idx_to_expert_idx can
be padded to their framework capacities; inactive tiles and rows are filtered
by scheduler metadata rather than mapping values.
"""


class Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel:
    """Rubin (SM107) fused FC1+FC2 persistent MoE kernel.

    The initial executable skeleton retains the proven 128dp FC1 data path.
    FC2 is integrated phase-by-phase behind the same 12-warp launch and
    persistent descriptor stream.

    Builds on Sm107BlockScaledContiguousGroupedGemmKernel (persistent tile
    scheduling, warp specialization, B-reuse, tcgen05.mma block-scale, TMA
    B/SFB with M-multicast, per-phase/per-group alphas). Refer to backbone for
    those.

    Additions on top of backbone:
      - Token gather: A/SFA receive rows are decoded from the shared
        ``permuted_idx_to_expanded_idx`` mapping.
      - A/SFA load: four gather warps issue CpAsync128.CG into a shared
        fc1_a_pipeline. SFA is consumed directly by a Cp128x128b UTCCP in the MMA
        warp using the 128dp_Unique layout.
      - SwiGLU epilogue: C = up * silu(gate), where up/gate come from
        interleaved accumulator at granularity=64 → output N is halved.
      - Optional NVFP4 quant: when c_dtype == Float4E2M1FN, the epilogue
        also generates SFC and quantizes the output.

    First-version warp roles (12 warps total):
      - 0-3   epilogue (LDTM → SwiGLU → optional quant → TMA store)
      - 4-7   gather A/SFA     (CpAsync128.CG)
      - 8     MMA
      - 9     TMA B / SFB
      - 10    scheduler
      - 11    2CTA A/SFA relay during FC1; FC2 metadata loader in both modes

    :param sf_vec_size: Scale factor vector size (16 or 32).
    :param mma_inst_shape: MMA instruction shape (M, N, K).
    :param mma_tiler: MMA tiler shape (M, N, K).
    :param cluster_shape_mn: Cluster dimensions (M, N).
    :param vectorized_f32: Use vectorized f32x2 ops in epilogue.
    :param topk: Experts selected per token.
    :param swiglu_limit: DS-v4 SwiGLU clamp limit; ``+inf`` disables clamp.
    :param scheduler: Persistent work-ID scheduler: ``static`` or ``l2_atomic``.
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        vectorized_f32: bool,
        topk: cutlass.Int64,
        use_pdl: bool = True,
        swiglu_limit: cutlass.Float32 = float("inf"),
        scheduler: str = "static",
    ):
        self.sf_vec_size = sf_vec_size
        self.topk = topk
        self.acc_dtype = cutlass.Float32
        self.mma_inst_shape = mma_inst_shape
        self.mma_tiler = mma_tiler
        self.cluster_shape_mn = cluster_shape_mn
        self.use_pdl = use_pdl
        if scheduler not in ("static", "l2_atomic"):
            raise ValueError(f"unsupported scheduler: {scheduler}")
        self.scheduler = scheduler
        self.use_l2_atomic_scheduler = scheduler == "l2_atomic"
        self.use_cluster_scheduler_response = (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 1
        )
        self.num_scheduler_stage = 1
        self.scheduler_response_bytes = cutlass.Int64.width // 8
        if swiglu_limit < 0:
            swiglu_limit = float("inf")
        self.swiglu_limit = swiglu_limit
        self.has_swiglu_limit = swiglu_limit != float("inf")

        self.use_2cta_instrs = mma_inst_shape[0] == 256
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        # Hand-encoded UTCOMMA instruction-descriptor base (Rubin sm107).
        # Phase 2: sfa_layout=1 (SFA_128dp_Unique) — SFA populated by Cp128x128b
        # UTCCP directly from gathered smem (no LDS+STTM transform).  n_dim is
        # static (=N>>3).
        self.mma_cta_group_int = 2 if self.use_2cta_instrs else 1
        self.sfa_layout_bit = 1
        self.static_idesc_base = manual_mma_128dp.build_static_idesc_base(
            umma_m=mma_inst_shape[0],
            umma_n=mma_inst_shape[1],
            umma_k=mma_inst_shape[2],
            sfa_layout=self.sfa_layout_bit,
        )
        self.arch = "sm_107"
        self.smem_capacity = cutlass.memory.get_smem_capacity_in_bytes(self.arch)
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.gather_a_warp_id = (
            4,
            5,
            6,
            7,
        )
        self.mma_warp_id = 8
        self.tma_b_warp_id = 9
        self.sched_warp_id = 10
        # Warp 11 relays the per-CTA A/SFA signal for the 2CTA cp.async path,
        # then loads FC2 metadata. In 1CTA it skips the relay and only loads
        # FC2 metadata.
        self.sync_transform_warp_id = 11
        # Register reconfig (setmaxnreg) per warpgroup. Launch alloc is 96/thread,
        # so the CTA budget is 96 * threads_per_cta and the per-warpgroup values
        # below must sum (x 128 threads) to within it. At 12 warps the budget is
        # 96*384 = 36864: epilogue 168 + gather 72 + mma-group 48 = 288 -> exactly
        # 128*288 = 36864. The mma group (warps 8-11) previously never called
        # setmaxnreg and sat at the launch default 96; reconfiguring it down to 48
        # is what keeps epilogue at its tuned 168 while gather still gets 72.
        self.num_regs_epilogue_warps = 168
        self.num_regs_gather_a_warps = 72
        self.num_regs_mma_group_warps = 48
        self.fc2_num_meta_stage = 2
        self.buffer_align_bytes = 1024
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                self.mma_warp_id,
                *self.gather_a_warp_id,
                self.tma_b_warp_id,
                *self.epilog_warp_id,
                self.sched_warp_id,
                self.sync_transform_warp_id,
            )
        )
        # Every non-scheduler warp consumes every fused descriptor. This keeps
        # the tile-info mbarrier balanced when warp 11 changes from idle during
        # FC1 to the FC2 metadata producer during phase 1.
        _wo_sched_warps = (
            *self.epilog_warp_id,
            self.mma_warp_id,
            self.tma_b_warp_id,
            self.sync_transform_warp_id,
            *self.gather_a_warp_id,
        )
        self.warps_wo_sched = len(_wo_sched_warps)
        self.threads_wo_sched = self.threads_per_warp * self.warps_wo_sched

        # Set barrier for cta sync, epilogue sync and tmem ptr sync
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len(self.epilog_warp_id),
        )
        # tmem_alloc_barrier participants: epi (allocator) + mma (consumer).
        # 128dp path: the SFA transform warps are idle (SFA UTCCP is done by
        # the MMA warp), so they no longer arrive here.
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32
            * len(
                (
                    self.mma_warp_id,
                    *self.epilog_warp_id,
                )
            ),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )

        self.num_smem_capacity = self.smem_capacity
        # num_tmem_alloc_cols already set in __init__

        self.vectorized_f32 = vectorized_f32

        # For epilogue compatibility
        self.epilogue_warp_id = self.epilog_warp_id

        # B-reuse pattern control
        self.enable_breuse = True if mma_tiler[0] // mma_inst_shape[0] == 2 else False

        # overlap_accum removed: SFA_128dp_Unique + UTCCP freed the TMEM it
        # existed to reclaim, so the plain (non-overlap) acc path is used.

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        - Computing tensor memory allocation columns
        """

        self.mma_inst_shape_sfb = (
            self.mma_inst_shape[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape[1], 128),
            self.mma_inst_shape[2],
        )

        # Configure tiled mma (Rubin SM107)
        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=self._get_mma_permutation_mnk(),
        )

        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )

        # Compute mma/cluster/tile shapes
        self.mma_tiler_sfb = (
            self.mma_inst_shape_sfb[0],
            self.mma_inst_shape_sfb[1],
            self.mma_tiler[2],
        )

        self.fc1_mma_tiler_c = (
            self.mma_tiler[0],
            self.mma_tiler[1] // 2,
            self.mma_tiler[2],
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Number of CpAsync128.CG loads per thread for A matrix (each loads 16 M-rows)
        self.fc1_a_num_loads = self.cta_tile_shape_mnk[0] // 16

        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        self.fc1_cta_tile_shape_mnk_c = (
            self.fc1_mma_tiler_c[0] // cute.size(tiled_mma.thr_id.shape),
            self.fc1_mma_tiler_c[1],
            self.fc1_mma_tiler_c[2],
        )

        # Compute SFA tiler for CpAsync gather (use mma_inst_shape for M/N, scaled K for SF)
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.mma_tiler[2] // mma_inst_shape_k
        self.fc1_mma_tiler_sfa = (
            self.mma_inst_shape[0],
            self.mma_inst_shape[1],
            mma_inst_shape_k * mma_inst_tile_k // 16,
        )
        self.fc1_cta_tile_shape_mnk_sfa = (
            self.fc1_mma_tiler_sfa[0] // cute.size(tiled_mma.thr_id.shape),
            self.fc1_mma_tiler_sfa[1],
            self.fc1_mma_tiler_sfa[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        # A multicast: cluster_N CTAs share A along N dim. Only meaningful when
        # cluster_N > 1. SFA multicast intentionally NOT enabled — was buggy
        # on cta_tile_N >= 256 and not worth the complexity.
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.is_a_mcast = self.num_mcast_ctas_a > 1

        # Fixed epilogue tile (128, 64). SwiGLU halves N, so the default
        # SM107_TILES lookup (keyed on full cta_n) can pick epi_tile_n too
        # small (wrong TMA store strides + insufficient SFC for cvt_fptrunc
        # 32-bit alignment). (128, 64) works for all configs.
        self.fc1_epi_tile = (128, 64)
        self.fc1_epi_tile_n = cute.size(self.fc1_epi_tile[1])
        self.fc1_epi_tile_cnt = (
            self.fc1_cta_tile_shape_mnk_c[0] // cute.size(self.fc1_epi_tile[0]),
            self.fc1_cta_tile_shape_mnk_c[1] // cute.size(self.fc1_epi_tile[1]),
        )
        self.fc2_epi_tile = sm107_utils.compute_epilogue_tile_shape(
            tiled_mma.op,
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.fc2_c_layout,
            self.fc2_c_dtype,
        )
        # FC2 finalize writes one BF16 row-major CTA tile before issuing
        # cp.reduce.async.bulk. Its size is independent of the FC1 C stage
        # count, so compute it before selecting fused shared-memory stages.
        fc2_swizzled_pad = 16 // (self.fc2_c_dtype.width // 8)
        self.fc2_c_smem_layout_staged = cute.make_layout(
            (
                self.cta_tile_shape_mnk[0],
                self.cta_tile_shape_mnk[1],
                1,
            ),
            stride=(
                self.cta_tile_shape_mnk[1] + fc2_swizzled_pad,
                1,
                self.cta_tile_shape_mnk[0]
                * (self.cta_tile_shape_mnk[1] + fc2_swizzled_pad),
            ),
        )
        fc2_c_smem_bytes = cute.size_in_bytes(
            self.fc2_c_dtype, self.fc2_c_smem_layout_staged
        )
        fc2_metadata_smem_bytes = (
            self.cta_tile_shape_mnk[0]
            * self.fc2_num_meta_stage
            * (cutlass.Int32.width + cutlass.Float32.width)
            // 8
        )

        # Setup A/B/C/Scale stage count in shared memory and ACC stage count in tensor memory
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.fc1_num_c_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.cta_tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.fc1_epi_tile,
            self.fc1_c_dtype,
            self.fc1_c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
            self.enable_breuse,
            fc2_c_smem_bytes,
            fc2_metadata_smem_bytes,
            self.fc2_num_meta_stage,
            self.buffer_align_bytes,
            self.mma_cta_group_int,
        )

        # Compute A/B/C/Scale shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )

        # SFA SMEM is plain linear (M_per_cta, tile_K_sf, stage), no pad.
        # Each thread does one CpAsync128.CG (16B = tile_K_sf=16 × FP8) per row.
        # Layout exposes (row, k_sf_byte, stage) with byte strides.
        sfa_tile_k_sf = self.cta_tile_shape_mnk[2] // self.sf_vec_size
        sf_bytes_per_row = sfa_tile_k_sf * self.sf_dtype.width // 8
        sfa_bytes_per_stage = self.cta_tile_shape_mnk[0] * sf_bytes_per_row
        self.fc1_sfa_smem_layout_staged = cute.make_layout(
            (self.cta_tile_shape_mnk[0], sfa_tile_k_sf, self.num_ab_stage),
            stride=(sf_bytes_per_row, 1, sfa_bytes_per_stage),
        )
        self.sfa_smem_alloc_bytes = self.num_ab_stage * sfa_bytes_per_stage
        self.fc2_sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfa_smem_alloc_bytes = max(
            self.sfa_smem_alloc_bytes,
            cute.size_in_bytes(
                self.sf_dtype, self.fc2_sfa_smem_layout_staged
            ),
        )

        self.fc1_c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.fc1_c_dtype,
            self.fc1_c_layout,
            self.fc1_epi_tile,
            self.fc1_num_c_stage,
        )
        # FC1 and FC2 are sequential within one CTA, so their epilogue views
        # alias one allocation sized for the larger phase.
        self.epilogue_smem_alloc_bytes = max(
            cute.size_in_bytes(
                self.fc1_c_dtype, self.fc1_c_smem_layout_staged.outer
            ),
            cute.size_in_bytes(
                self.fc2_c_dtype, self.fc2_c_smem_layout_staged
            ),
        )
        self.fc2_c_copy_size = (
            self.cta_tile_shape_mnk[1] * self.fc2_c_dtype.width // 8
        )

        # Compute TMEM layouts for SFB (Rubin precomputed)
        self.tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0)),
        )

        # SFA 128dp_Unique TMEM layout. Each of the 128 M-rows (tokens) maps
        # 1:1 to one TMEM datapath; the free dim holds nsf = tile_K /
        # sf_vec_size scale-factor bytes (K-order). This is written 1:1 by a
        # Cp128x128b UTCCP from the gathered linear sFC1SFA smem
        # [128 rows, nsf bytes] — no LDS+STTM transform.
        self.fc1_sfa_nsf = self.cta_tile_shape_mnk[2] // self.sf_vec_size
        # DP (datapath/lane) stride for SF TMEM is 1<<18 (== the 32dp path's
        # 262144; the Cp128x128b atom's TV layout expects this, matching the
        # 32-bit physical column addressing where 4 sf-bytes pack per word).
        self.tCtFC1SFA_layout = cute.make_layout(
            (128, self.fc1_sfa_nsf), stride=(1 << 18, 1)
        )
        self.tCtFC2SFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.select(
                self.fc2_sfa_smem_layout_staged,
                mode=list(range(cute.rank(self.fc2_sfa_smem_layout_staged) - 1)),
            ),
        )
        # Per-UMMA-K SF byte count (UMMA_K / sf_vec_size). The MMA reads this
        # many contiguous SF columns per k-block; k-block stride = this value.
        self.fc1_sfa_sf_per_kblock = self.mma_inst_shape[2] // self.sf_vec_size

        # Compute TMEM column counts.
        # SFA 128dp: nsf sf-bytes per DP → ceil(nsf/4) 32-bit columns.
        # Single TMEM buffer (UTCCP issued in the MMA warp right before the
        # MMA, like SFB) → num_sfa_tmem_stage = 1.
        self.fc1_num_sfa_tmem_cols_per_stage = (self.fc1_sfa_nsf + 3) // 4
        self.fc1_num_sfa_tmem_stage = 1
        self.fc1_num_sfa_tmem_cols = (
            self.fc1_num_sfa_tmem_cols_per_stage * self.fc1_num_sfa_tmem_stage
        )
        self.fc2_num_sfa_tmem_cols = (
            cute.cosize(
                cute.recast_layout(
                    32, self.sf_dtype.width, self.tCtFC2SFA_layout
                )
            )
            & 0x0000FFFF
        )
        self.num_sfa_tmem_cols = max(
            self.fc1_num_sfa_tmem_cols, self.fc2_num_sfa_tmem_cols
        )
        self.num_sfb_tmem_cols = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, self.tCtSFB_layout))
            & 0x0000FFFF
        )
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        # acc TMEM cols: tile_N × num_acc_stage × (2 if breuse).
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1]
            * self.num_acc_stage
            * (2 if self.enable_breuse else 1)
        )
        # SFA TMEM offset (cols, 32-bit each): right after acc.
        self.sfa_tmem_offset = self.num_accumulator_tmem_cols
        # Validation: 512 + 32 + 32 = 576 (exact fit for main target on sm_107)
        _total_used = (
            self.num_accumulator_tmem_cols
            + self.num_sfa_tmem_cols
            + self.num_sfb_tmem_cols
        )
        if _total_used > self.num_tmem_alloc_cols:
            raise ValueError(
                f"TMEM overflow: acc({self.num_accumulator_tmem_cols}) + "
                f"sfa({self.num_sfa_tmem_cols}) + "
                f"sfb({self.num_sfb_tmem_cols}) = {_total_used} > "
                f"max {self.num_tmem_alloc_cols}"
            )

    def _get_mma_permutation_mnk(self):
        if cutlass.const_expr(self.use_2cta_instrs and self.enable_breuse):
            m_layout = cute.make_layout(
                shape=(self.mma_inst_shape[0] // 2, 2, 2),
                stride=(1, self.mma_inst_shape[0], self.mma_inst_shape[0] // 2),
            )
            return (m_layout, self.mma_inst_shape[1], self.mma_inst_shape[2])
        else:
            return (1, 1, 1)

    def _is_interleaved_utccp(self) -> bool:
        """Enable interleaving UTCCP for Bkeep-Breuse case for 4xFP4 kernel."""
        return (
            self.a_dtype.width == 4 and self.b_dtype.width == 4 and self.enable_breuse
        )

    def _mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> S2TCopyBundle:
        """Make tiledCopy for smem to tmem load for scale factor tensor."""
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        def appendMNBroadcastMode(smem_layout: cute.Layout):
            mn_dim = cute.get(smem_layout, mode=[0, 0])
            mn_dim = cute.append(mn_dim, cute.make_layout((4), stride=(0)))
            layout = cute.append(
                cute.group_modes(mn_dim, 0), cute.get(smem_layout, mode=[0, 1])
            )
            layout = cute.append(
                cute.group_modes(layout, 0), cute.get(smem_layout, mode=[1])
            )
            layout = cute.append(layout, cute.get(smem_layout, mode=[2]))
            layout = cute.append(layout, cute.get(smem_layout, mode=[3]))
            return layout

        tCsSF_compact_bcast = cute.make_tensor(
            tCsSF_compact.iterator, appendMNBroadcastMode(tCsSF_compact.layout)
        )

        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact_bcast)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return S2TCopyBundle(tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t)

    def _fc1_sfa_s2t_copy_and_partition_128dp(
        self,
        sFC1SFA: cute.Tensor,
        tCtFC1SFA_128: cute.Tensor,
    ) -> S2TCopyBundle:
        """SFA 128dp_Unique UTCCP (Cp128x128b): 1:1 copy of the gathered
        linear sFC1SFA smem [128 rows, nsf SF-bytes, stage] into 128dp TMEM
        (each token → one datapath, no 4x broadcast, no LDS+STTM transform).
        The gathered smem feeds Cp128x128b directly."""
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp128x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtFC1SFA_128)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # sFC1SFA is (128, nsf, stage): partition_S tiles the (128, nsf) copy
        # atom and keeps the stage mode for per-k_tile indexing.
        tCsSFA_s2t_ = thr_copy_s2t.partition_S(sFC1SFA)
        tCsSFA_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSFA_s2t_)
        tCtFC1SFA_s2t = thr_copy_s2t.partition_D(tCtFC1SFA_128)

        return S2TCopyBundle(tiled_copy_s2t, tCsSFA_s2t, tCtFC1SFA_s2t)

    def _mainloop_s2t_copies(
        self,
        stage_idx: int,
        sfb_s2t_bundle: S2TCopyBundle,
    ):
        """Copy SFB from smem to tmem (UTCCP). SFA path now uses LDS+STTM
        from transform warps, no UTCCP needed here."""
        s2t_stage_coord = (None, None, None, None, stage_idx)

        cute.copy(
            sfb_s2t_bundle.tiled_copy,
            sfb_s2t_bundle.sSF_compact[s2t_stage_coord],
            sfb_s2t_bundle.tSF_compact,
        )

    def _mainloop_s2t_interleaved_copies(
        self,
        k_block: int,
        stage_idx: int,
        sfa_s2t_bundle: S2TCopyBundle,
        sfb_s2t_bundle: S2TCopyBundle,
    ):
        """Interleaved UTCCP for Bkeep-Breuse pattern."""
        s_sfa_crd_keep = (None, 0, None, k_block, stage_idx)
        s_sfa_crd_reuse = (None, 1, None, k_block, stage_idx)
        s_sfb_crd = (None, None, None, k_block, stage_idx)

        t_sfa_crd_keep = (None, 0, None, k_block)
        t_sfa_crd_reuse = (None, 1, None, k_block)
        t_sfb_crd = (None, None, None, k_block)

        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s_sfa_crd_keep],
            sfa_s2t_bundle.tSF_compact[t_sfa_crd_keep],
        )
        cute.copy(
            sfb_s2t_bundle.tiled_copy,
            sfb_s2t_bundle.sSF_compact[s_sfb_crd],
            sfb_s2t_bundle.tSF_compact[t_sfb_crd],
        )
        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s_sfa_crd_reuse],
            sfa_s2t_bundle.tSF_compact[t_sfa_crd_reuse],
        )

    @cute.jit
    def __call__(
        self,
        fc1_a: cute.Tensor,
        fc1_b: cute.Tensor,
        fc1_c: cute.Tensor,
        fc1_sfa: cute.Tensor,
        fc1_sfb: cute.Tensor,
        fc1_sfc: Optional[cute.Tensor],
        fc1_norm_const: Optional[cute.Tensor],
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        fc1_alpha: cute.Tensor,
        fc2_alpha: cute.Tensor,
        fc1_ready: cute.Tensor,
        fc1_scheduler_counter: cute.Tensor,
        fc2_scheduler_counter: cute.Tensor,
        fc2_b: cute.Tensor,
        fc2_c: cute.Tensor,
        fc2_sfb: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        fc2_routing_scales: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the contiguous grouped GEMM with gather operation and SwiGLU fusion.

        This method performs FC1 layer computation:
        1. GEMM: acc = fc1_alpha * (SFA * A[token_ids]) * (SFB * B)
        2. SwiGLU: C = up * silu(gate), where up/gate are extracted from interleaved acc (granularity=64)
        3. Optional Quant: When c_dtype is Float4E2M1FN, generates SFC and quantizes output

        Data loading:
        - A and SFA are loaded using CpAsync instructions with token-based gather
        - B and SFB are loaded using TMA instructions with multicast
        - B weights are interleaved: [up_0:64, gate_64:128, up_128:192, gate_192:256, ...]

        Execution steps:
        1. Setup static attributes before smem/grid computation
        2. Setup TMA load/store atoms for B, SFB, and C (no TMA for A/SFA)
        3. Compute grid size with regard to hardware constraints
        4. Define shared storage for kernel
        5. Launch the kernel synchronously with warp specialization:
           - Scheduler warp: Dispatches tile information
           - CpAsync warps: Load A and SFA with gather
           - A Sync Transform warps: Transform the sync signal of A and SFA from global to
             shared memory when use_2cta_instrs is True
           - TMA warp: Load B and SFB with multicast
           - MMA warp: Perform matrix multiply-accumulate
           - Epilogue warps: Apply SwiGLU activation, optional quantization, and store results

        :param fc1_a: FC1 input A (MxKx1), gathered with the shared route mapping
        :type fc1_a: cute.Tensor
        :param fc1_b: FC1 expert weight B (NxKxL), interleaved for SwiGLU
        :type fc1_b: cute.Tensor
        :param fc1_c: Quantized FC1 SwiGLU output and FC2 input (Mx(N/2)x1)
        :type fc1_c: cute.Tensor
        :param fc1_sfa: FC1 A scale factors, gathered with the route mapping
        :type fc1_sfa: cute.Tensor
        :param fc1_sfb: FC1 B scale factors
        :type fc1_sfb: cute.Tensor
        :param fc1_sfc: FC1 C / FC2 A scale factors (None if not quantizing)
        :type fc1_sfc: Optional[cute.Tensor]
        :param fc1_norm_const: FC1 output normalization constant
            (None if not quantizing)
        :type fc1_norm_const: Optional[cute.Tensor]
        :param tile_idx_to_expert_idx: Mapping from tile index to expert ID,
            shape (permuted_m/cta_tile_m,) where cta_tile_m is the CTA tile M size
        :type tile_idx_to_expert_idx: cute.Tensor
        :param tile_idx_to_mn_limit: Mapping from tile index to M-N dimension limit
            for boundary checking, shape (permuted_m/cta_tile_m,)
        :type tile_idx_to_mn_limit: cute.Tensor
        :param num_non_exiting_tiles: Number of valid tiles to process (valid_m/cta_tile_m), shape (1,)
        :type num_non_exiting_tiles: cute.Tensor
        :param fc1_alpha: FC1 dequantization alpha tensor for each group
        :type fc1_alpha: cute.Tensor
        :param fc2_alpha: FC2 dequantization alpha tensor for each group
        :type fc2_alpha: cute.Tensor
        :param fc1_ready: Per-M uint32 counters released by completed FC1 N tiles
        :type fc1_ready: cute.Tensor
        :param fc1_scheduler_counter: FC1 L2-atomic work-ID counter
        :param fc2_scheduler_counter: FC2 L2-atomic work-ID counter
        :param fc2_b: FC2 expert weights, shape (FC2_N, FC2_K, L)
        :param fc2_c: FC2 final output, shape (sequence_M, FC2_N, 1)
        :param fc2_sfb: FC2 expert weight scale factors
        :param permuted_idx_to_expanded_idx: Shared FC1 gather and FC2 scatter
            mapping. Each entry is ``receive_row * topk + topk_slot``.
        :param fc2_routing_scales: Per-token final routing scales
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = fc1_a.element_type
        self.b_dtype: Type[cutlass.Numeric] = fc1_b.element_type
        self.fc1_c_dtype: Type[cutlass.Numeric] = fc1_c.element_type
        self.sf_dtype: Type[cutlass.Numeric] = fc1_sfa.element_type
        self.fc2_c_dtype: Type[cutlass.Numeric] = fc2_c.element_type
        if cutlass.const_expr(self.fc2_c_dtype != cutlass.BFloat16):
            raise TypeError("fused FC2 finalize currently requires BF16 output")
        self.a_major_mode = cutlass.tensor_utils.LayoutEnum.from_tensor(
            fc1_a
        ).mma_major_mode()
        self.b_major_mode = cutlass.tensor_utils.LayoutEnum.from_tensor(
            fc1_b
        ).mma_major_mode()
        self.fc1_c_layout = cutlass.tensor_utils.LayoutEnum.from_tensor(fc1_c)
        self.fc2_c_layout = cutlass.tensor_utils.LayoutEnum.from_tensor(fc2_c)
        fc2_output_n = fc2_c.shape[1]
        runtime_assert(fc2_output_n > 0, "FC2 output N must be positive")
        runtime_assert(
            fc2_output_n % self.mma_tiler[1] == 0,
            "FC2 output N must be divisible by CTA tile N",
        )

        # Note: Rubin supports mixed A/B dtypes (e.g., Float8E4M3FN x Float8E5M2)

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfb tensor by filling B tensor to scale factor atom layout
        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        fc1_sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            fc1_b.shape, self.sf_vec_size
        )
        fc1_sfb = cute.make_tensor(fc1_sfb.iterator, fc1_sfb_layout)

        # Setup sfc tensor by filling C tensor to scale factor atom layout
        self.fc1_generate_sfc = fc1_sfc is not None and fc1_norm_const is not None
        if cutlass.const_expr(self.fc1_generate_sfc):
            fc1_sfc_layout = blockscaled_utils.tile_atom_to_shape_SF(
                fc1_c.shape, self.sf_vec_size
            )
            fc1_sfc = cute.make_tensor(fc1_sfc.iterator, fc1_sfc_layout)
        else:
            raise ValueError("fused FC12 requires quantized FC1 output and SFC")

        # Both phases reuse one tiled MMA and one set of A/B/SFB buffers, so
        # their corresponding operand and scale-factor types must match.
        if cutlass.const_expr(fc1_c.element_type != self.a_dtype):
            raise TypeError("FC1 C / FC2 A must use the shared A dtype")
        if cutlass.const_expr(fc2_b.element_type != self.b_dtype):
            raise TypeError("FC2 B must use the shared B dtype")
        if cutlass.const_expr(fc1_sfb.element_type != self.sf_dtype):
            raise TypeError("FC1 SFB must use the shared scale-factor dtype")
        if cutlass.const_expr(fc1_sfc.element_type != self.sf_dtype):
            raise TypeError("FC1 SFC / FC2 SFA must use the shared scale-factor dtype")
        if cutlass.const_expr(fc2_sfb.element_type != self.sf_dtype):
            raise TypeError("FC2 SFB must use the shared scale-factor dtype")
        fc2_sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            fc2_b.shape, self.sf_vec_size
        )
        fc2_sfb = cute.make_tensor(fc2_sfb.iterator, fc2_sfb_layout)

        atom_layout_mnk = (1, 1, 1)
        permutation_mnk = self._get_mma_permutation_mnk()

        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=atom_layout_mnk,
            permutation_mnk=permutation_mnk,
        )
        tiled_mma.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma.set(tcgen05.Field.NEGATE_B, False)

        # For 2CTA blockscaled kernels, SFB needs to be replicated across peer CTAs.
        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_B, False)

        tiled_mma_bkeep = None
        tiled_mma_breuse = None
        if cutlass.const_expr(self.enable_breuse):
            tiled_mma_bkeep = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD,
                b_collector_op=CollectorOp.FILL,
                atom_layout_mnk=atom_layout_mnk,
                permutation_mnk=permutation_mnk,
            )
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_B, False)

            tiled_mma_breuse = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD,
                b_collector_op=CollectorOp.LASTUSE,
                atom_layout_mnk=atom_layout_mnk,
                permutation_mnk=permutation_mnk,
            )
            tiled_mma_breuse.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_breuse.set(tcgen05.Field.NEGATE_B, False)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # CpAsync128.CG gmem → sA SMEM. Four gather warps × 32
        # threads issue cp.async.cg.16B per (token row, k chunk). a_num_loads
        # (in _setup_attributes) controls CpAsync iterations per thread per k_tile.
        # The same warps gather SFA into the same pipeline stage.

        # Setup TMA load for B
        fc1_b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_fc1_b, tma_tensor_fc1_b = cute.nvgpu.make_tiled_tma_atom_B(
            fc1_b_op,
            fc1_b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFB
        fc1_sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_fc1_sfb, tma_tensor_fc1_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            fc1_sfb_op,
            fc1_sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # This modifies the layout to handle overlapping 256x(# of scale factors for a single column of B (nNSF))
        # logical blocks for SFB when cta_tile_shape_n=192.
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_fc1_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_fc1_sfb.shape[0][1], 4)

            new_shape = (
                (tma_tensor_fc1_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_fc1_sfb.shape[1],
                tma_tensor_fc1_sfb.shape[2],
            )
            # Use right multiplication for ScaledBasis (3 * x instead of x * 3)
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_fc1_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_fc1_sfb.stride[1],
                tma_tensor_fc1_sfb.stride[2],
            )
            tma_tensor_fc1_sfb_new_layout = cute.make_layout(
                new_shape, stride=new_stride
            )
            tma_tensor_fc1_sfb = cute.make_tensor(
                tma_tensor_fc1_sfb.iterator, tma_tensor_fc1_sfb_new_layout
            )

        # FC2 reuses the FC1 A/B/SF stage buffers but owns separate TMA
        # descriptors and mbarriers. A/SFA and B/SFB intentionally remain two
        # independent producer pipelines so only the A side waits on FC1.
        fc2_a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        fc2_a_smem_layout = cute.select(
            self.a_smem_layout_staged,
            mode=list(range(cute.rank(self.a_smem_layout_staged) - 1)),
        )
        tma_atom_fc2_a, tma_tensor_fc2_a = cute.nvgpu.make_tiled_tma_atom_A(
            fc2_a_op,
            fc1_c,
            fc2_a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        fc2_sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        fc2_sfa_smem_layout = cute.slice_(
            self.fc2_sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_fc2_sfa, tma_tensor_fc2_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            fc2_sfa_op,
            fc1_sfc,
            fc2_sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        fc2_b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        fc2_b_smem_layout = cute.select(
            self.b_smem_layout_staged,
            mode=list(range(cute.rank(self.b_smem_layout_staged) - 1)),
        )
        tma_atom_fc2_b, tma_tensor_fc2_b = cute.nvgpu.make_tiled_tma_atom_B(
            fc2_b_op,
            fc2_b,
            fc2_b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        fc2_sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        fc2_sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_fc2_sfb, tma_tensor_fc2_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            fc2_sfb_op,
            fc2_sfb,
            fc2_sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        fc1_b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        fc1_sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.fc1_b_tma_load_bytes = (
            fc1_b_copy_size + fc1_sfb_copy_size
        ) * atom_thr_size
        self.fc2_a_tma_load_bytes = (
            cute.size_in_bytes(self.a_dtype, fc2_a_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, fc2_sfa_smem_layout)
        ) * atom_thr_size
        self.fc2_b_tma_load_bytes = (
            cute.size_in_bytes(self.b_dtype, fc2_b_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, fc2_sfb_smem_layout)
        ) * atom_thr_size

        # Setup TMA store for C
        tma_atom_fc1_c = None
        tma_tensor_fc1_c = None
        fc1_epi_smem_layout = cute.slice_(
            self.fc1_c_smem_layout_staged, (None, None, 0)
        )
        tma_atom_fc1_c, tma_tensor_fc1_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            fc1_c,
            fc1_epi_smem_layout,
            self.fc1_epi_tile,
        )

        # Both phases share padded M and group dimensions but own independent
        # N task spaces. Build scheduler parameters from logical GEMM shapes;
        # the physical persistent grid is sized once from the larger FC2 task
        # space used by the initial integration.
        fc1_gemm_shape = fc1_c.shape
        fc2_gemm_shape = (
            fc1_gemm_shape[0],
            fc2_output_n,
            fc1_gemm_shape[2],
        )
        if cutlass.const_expr(
            self.use_l2_atomic_scheduler
            and self.use_cluster_scheduler_response
        ):
            # The cluster leader publishes M/N/L as unsigned 16-bit fields.
            # Keep device-side checks here because tensor extents are staged
            # values for callers that bypass the benchmark's host validation.
            descriptor_count_limit = 1 << 16
            runtime_assert(
                fc1_gemm_shape[0]
                <= self.fc1_cta_tile_shape_mnk_c[0] * descriptor_count_limit,
                "FC1 M tile count exceeds the L2 atomic descriptor limit",
            )
            runtime_assert(
                fc1_gemm_shape[1]
                <= self.fc1_cta_tile_shape_mnk_c[1] * descriptor_count_limit,
                "FC1 N tile count exceeds the L2 atomic descriptor limit",
            )
            runtime_assert(
                fc1_gemm_shape[2] <= descriptor_count_limit,
                "FC1 L tile count exceeds the L2 atomic descriptor limit",
            )
            runtime_assert(
                fc2_gemm_shape[0]
                <= self.cta_tile_shape_mnk[0] * descriptor_count_limit,
                "FC2 M tile count exceeds the L2 atomic descriptor limit",
            )
            runtime_assert(
                fc2_gemm_shape[1]
                <= self.cta_tile_shape_mnk[1] * descriptor_count_limit,
                "FC2 N tile count exceeds the L2 atomic descriptor limit",
            )
            runtime_assert(
                fc2_gemm_shape[2] <= descriptor_count_limit,
                "FC2 L tile count exceeds the L2 atomic descriptor limit",
            )
        self.fc1_tile_sched_params = self._compute_tile_sched_params(
            fc1_gemm_shape,
            self.fc1_cta_tile_shape_mnk_c,
            self.cluster_shape_mn,
        )
        self.fc2_tile_sched_params = self._compute_tile_sched_params(
            fc2_gemm_shape,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
        )
        fc1_ready_expected = cutlass.Int32(
            cute.ceil_div(fc1_b.shape[0], self.mma_tiler[1])
            * self.mma_cta_group_int
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            self.fc2_tile_sched_params, max_active_clusters
        )

        # Define shared storage for kernel.
        @cute.struct
        class SharedStorageCpasync1cta:
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 6 * self.num_tile_stage],
                1,
            ]
            # cpasync mode: A and B use separate pipelines (CpAsync A is
            # CpAsync type, B is TmaUmma type — they can't share one mbar).
            # Each mbar set holds num_ab_stage * 2 (full + empty per stage).
            fc1_a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            fc1_b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            fc2_a_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2
            ]
            fc2_b_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2
            ]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2
            ]
            fc1_scheduler_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_scheduler_stage * 2
            ]
            fc1_scheduler_response: cute.struct.MemRange[
                cutlass.Int64, self.num_scheduler_stage
            ]
            fc2_scheduler_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_scheduler_stage * 2
            ]
            fc2_scheduler_response: cute.struct.MemRange[
                cutlass.Int64, self.num_scheduler_stage
            ]
            scheduler_throttle_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_scheduler_stage * 2
            ]
            fc2_meta_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.fc2_num_meta_stage * 2
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sEpilogue: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8,
                    self.epilogue_smem_alloc_bytes,
                ],
                self.buffer_align_bytes,
            ]
            # sFC1SFA placed BEFORE sA so SFA gets a low SMEM offset: CpAsync128.CG
            # destination addr stays under the 248KB threshold (above which the
            # .CG cache-mode hint would degrade to a plain CpAsync).
            sSFAStorage: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, self.sfa_smem_alloc_bytes],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            fc2_meta_token_idx: cute.struct.MemRange[
                cutlass.Int32,
                self.cta_tile_shape_mnk[0] * self.fc2_num_meta_stage,
            ]
            fc2_meta_scale: cute.struct.MemRange[
                cutlass.Float32,
                self.cta_tile_shape_mnk[0] * self.fc2_num_meta_stage,
            ]

        # 2CTA variant: adds fc1_a_sync_transform_mbar_ptr for the warp-11 relay
        # pipeline (PipelineAsyncUmma) that bridges the per-CTA `fc1_a_pipeline`
        # (which now carries A *and* SFA) to the cluster-wide MMA consumer —
        # tcgen05.mma.cta_group::2 and the SFA Cp128x128b(TWO) UTCCP both read
        # the peer CTA's SMEM, and one relay now covers both.
        @cute.struct
        class SharedStorageCpasync2cta:
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 6 * self.num_tile_stage],
                1,
            ]
            fc1_a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            fc1_b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            fc1_a_sync_transform_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2
            ]
            fc2_a_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2
            ]
            fc2_b_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2
            ]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2
            ]
            fc1_scheduler_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_scheduler_stage * 2
            ]
            fc1_scheduler_response: cute.struct.MemRange[
                cutlass.Int64, self.num_scheduler_stage
            ]
            fc2_scheduler_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_scheduler_stage * 2
            ]
            fc2_scheduler_response: cute.struct.MemRange[
                cutlass.Int64, self.num_scheduler_stage
            ]
            scheduler_throttle_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_scheduler_stage * 2
            ]
            fc2_meta_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.fc2_num_meta_stage * 2
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sEpilogue: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8,
                    self.epilogue_smem_alloc_bytes,
                ],
                self.buffer_align_bytes,
            ]
            sSFAStorage: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, self.sfa_smem_alloc_bytes],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            fc2_meta_token_idx: cute.struct.MemRange[
                cutlass.Int32,
                self.cta_tile_shape_mnk[0] * self.fc2_num_meta_stage,
            ]
            fc2_meta_scale: cute.struct.MemRange[
                cutlass.Float32,
                self.cta_tile_shape_mnk[0] * self.fc2_num_meta_stage,
            ]

        self.shared_storage = (
            SharedStorageCpasync2cta
            if self.use_2cta_instrs
            else SharedStorageCpasync1cta
        )

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_bkeep,
            tiled_mma_breuse,
            tiled_mma_sfb,
            fc1_a,
            tma_atom_fc1_b,
            tma_tensor_fc1_b,
            fc1_sfa,
            tma_atom_fc1_sfb,
            tma_tensor_fc1_sfb,
            tma_atom_fc2_a,
            tma_tensor_fc2_a,
            tma_atom_fc2_sfa,
            tma_tensor_fc2_sfa,
            tma_atom_fc2_b,
            tma_tensor_fc2_b,
            tma_atom_fc2_sfb,
            tma_tensor_fc2_sfb,
            fc2_c,
            permuted_idx_to_expanded_idx,
            fc2_routing_scales,
            tma_atom_fc1_c,
            tma_tensor_fc1_c,
            fc1_sfc,
            fc1_norm_const,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            num_non_exiting_tiles,
            fc1_alpha,
            fc2_alpha,
            fc1_ready,
            fc1_scheduler_counter,
            fc2_scheduler_counter,
            fc1_ready_expected,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.fc1_sfa_smem_layout_staged,
            self.fc2_sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.tCtFC1SFA_layout,
            self.tCtFC2SFA_layout,
            self.tCtSFB_layout,
            self.fc1_c_smem_layout_staged,
            self.fc2_c_smem_layout_staged,
            self.fc1_epi_tile,
            self.fc2_epi_tile,
            self.fc1_tile_sched_params,
            self.fc2_tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_bkeep: Optional[cute.TiledMma],
        tiled_mma_breuse: Optional[cute.TiledMma],
        tiled_mma_sfb: cute.TiledMma,
        mFC1A_mkl: cute.Tensor,
        tma_atom_fc1_b: cute.CopyAtom,
        mFC1B_nkl: cute.Tensor,
        mFC1SFA_mkl: cute.Tensor,
        tma_atom_fc1_sfb: cute.CopyAtom,
        mFC1SFB_nkl: cute.Tensor,
        tma_atom_fc2_a: cute.CopyAtom,
        mFC2A_mkl: cute.Tensor,
        tma_atom_fc2_sfa: cute.CopyAtom,
        mFC2SFA_mkl: cute.Tensor,
        tma_atom_fc2_b: cute.CopyAtom,
        mFC2B_nkl: cute.Tensor,
        tma_atom_fc2_sfb: cute.CopyAtom,
        mFC2SFB_nkl: cute.Tensor,
        mFC2C_mnl: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        fc2_routing_scales: cute.Tensor,
        tma_atom_fc1_c: cute.CopyAtom,
        mFC1C_mnl: cute.Tensor,
        mFC1SFC_mnl: Optional[cute.Tensor],
        fc1_norm_const: Optional[cute.Tensor],
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        fc1_alpha: cute.Tensor,
        fc2_alpha: cute.Tensor,
        fc1_ready: cute.Tensor,
        fc1_scheduler_counter: cute.Tensor,
        fc2_scheduler_counter: cute.Tensor,
        fc1_ready_expected: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        fc1_sfa_smem_layout_staged: cute.Layout,
        fc2_sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tCtFC1SFA_layout: cute.Layout,
        tCtFC2SFA_layout: cute.Layout,
        tCtSFB_layout: cute.Layout,
        fc1_c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        fc2_c_smem_layout_staged: cute.Layout,
        fc1_epi_tile: cute.Tile,
        fc2_epi_tile: cute.Tile,
        fc1_tile_sched_params: utils.PersistentTileSchedulerParams,
        fc2_tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_b_warp_id:
            cpasync.prefetch_descriptor(tma_atom_fc1_b)
            cpasync.prefetch_descriptor(tma_atom_fc1_sfb)
            cpasync.prefetch_descriptor(tma_atom_fc2_b)
            cpasync.prefetch_descriptor(tma_atom_fc2_sfb)
            cpasync.prefetch_descriptor(tma_atom_fc1_c)
        if warp_idx == self.gather_a_warp_id[0]:
            cpasync.prefetch_descriptor(tma_atom_fc2_a)
            cpasync.prefetch_descriptor(tma_atom_fc2_sfa)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        is_scheduler_leader_cta = cta_rank_in_cluster == 0
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # (fc1_a_pipeline created below alongside fc1_b_pipeline.)

        cta_v_size = cute.size(cluster_layout_vmnk, mode=[0])

        # Four gather warps issue A and SFA cp.async copies, then arrive once
        # on the shared stage. The MMA warp consumes both operands from this
        # single CpAsyncUmma pipeline.
        fc1_a_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.gather_a_warp_id),
        )
        fc1_a_pipeline = PipelineCpAsyncUmma.create(
            barrier_storage=storage.fc1_a_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=fc1_a_pipeline_producer_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        if cutlass.const_expr(self.use_2cta_instrs):
            fc1_a_sync_transform_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * cta_v_size,
            )
            fc1_a_sync_transform_pipeline = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.fc1_a_sync_transform_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=fc1_a_sync_transform_pipeline_producer_group,
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )

        fc1_b_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_b
        fc1_b_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        fc1_b_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.fc1_b_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=fc1_b_pipeline_producer_group,
            consumer_group=fc1_b_pipeline_consumer_group,
            tx_count=self.fc1_b_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(0, 1),
            defer_sync=True,
        )

        fc2_a_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.fc2_a_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mcast_ctas_a
            ),
            tx_count=self.fc2_a_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(1, 0),
            defer_sync=True,
        )
        fc2_b_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.fc2_b_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mcast_ctas_b
            ),
            tx_count=self.fc2_b_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(0, 1),
            defer_sync=True,
        )

        # Pipeline Init: Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = (
            len(self.epilog_warp_id)
            * self.threads_per_warp
            * (2 if use_2cta_instrs else 1)
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Pipeline Init:Initialize tile info pipeline (barrier) and states
        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        # All four gather A/SFA warps consume every tile descriptor.
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_wo_sched,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        if cutlass.const_expr(self.use_l2_atomic_scheduler):
            # A one-stage local handshake prevents the scheduler from claiming
            # work faster than the TMA-B warp starts the current descriptor.
            scheduler_throttle_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=storage.scheduler_throttle_mbar_ptr.data_ptr(),
                num_stages=self.num_scheduler_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.threads_per_warp
                ),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.threads_per_warp
                ),
                defer_sync=True,
            )

            if cutlass.const_expr(self.use_cluster_scheduler_response):
                scheduler_response_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread
                )
                scheduler_response_consumer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.threads_per_warp * cute.size(self.cluster_shape_mn),
                )
                fc1_scheduler_pipeline = pipeline.PipelineClcFetchAsync.create(
                    barrier_storage=storage.fc1_scheduler_mbar_ptr.data_ptr(),
                    num_stages=self.num_scheduler_stage,
                    producer_group=scheduler_response_producer_group,
                    consumer_group=scheduler_response_consumer_group,
                    tx_count=self.scheduler_response_bytes,
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                )
                fc2_scheduler_pipeline = pipeline.PipelineClcFetchAsync.create(
                    barrier_storage=storage.fc2_scheduler_mbar_ptr.data_ptr(),
                    num_stages=self.num_scheduler_stage,
                    producer_group=scheduler_response_producer_group,
                    consumer_group=scheduler_response_consumer_group,
                    tx_count=self.scheduler_response_bytes,
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                )

        # Warp 11 stages the per-row FC2 scatter metadata independently from
        # the A/B load pipelines. Only the epilogue warpgroup consumes it.
        meta_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.fc2_meta_mbar_ptr.data_ptr(),
            num_stages=self.fc2_num_meta_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_warp
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.epilog_warp_id),
            ),
        )

        # Tensor memory dealloc barrier init
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
            arch=self.arch,
        )

        # Cluster arrive after barrier init (Rubin uses pipeline_init_arrive)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/C/Scale
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sEpilogueBytes = storage.sEpilogue.get_tensor(
            cute.make_layout((self.epilogue_smem_alloc_bytes,))
        )
        sFC1C = cute.make_tensor(
            cute.recast_ptr(
                sEpilogueBytes.iterator,
                fc1_c_smem_layout_staged.inner,
                dtype=self.fc1_c_dtype,
            ),
            fc1_c_smem_layout_staged.outer,
        )
        sFC2C = cute.make_tensor(
            cute.recast_ptr(sEpilogueBytes.iterator, dtype=self.fc2_c_dtype),
            fc2_c_smem_layout_staged,
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # SFA SMEM (linear+pad layout for the cp.async gather).
        sFC1SFA = storage.sSFAStorage.get_tensor(fc1_sfa_smem_layout_staged)
        # FC2 overlays the same raw SFA storage with the standard blockscaled
        # layout expected by TMA and the FC2 MMA descriptor.
        sFC2SFA = storage.sSFAStorage.get_tensor(fc2_sfa_smem_layout_staged)
        # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        # (bidx, bidy, bidz, valid, mn_limit)
        # (m_tile, n_tile, expert, valid, mn_limit, phase)
        info_layout = cute.make_layout((6, self.num_tile_stage), stride=(1, 6))
        sInfo = storage.sInfo.get_tensor(info_layout)
        fc2_meta_layout = cute.make_layout(
            (self.cta_tile_shape_mnk[0], self.fc2_num_meta_stage),
            stride=(1, self.cta_tile_shape_mnk[0]),
        )
        sFC2MetaTokenIdx = storage.fc2_meta_token_idx.get_tensor(fc2_meta_layout)
        sFC2MetaScale = storage.fc2_meta_scale.get_tensor(fc2_meta_layout)

        #
        # Compute multicast mask for A/B buffer full
        #
        b_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )
        # FC1 uses cpasync A, while FC2 uses TMA A/SFA and may multicast both
        # along cluster N.
        a_full_mcast_mask = None
        sfa_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, loopM, loopK, loopL)
        gFC1A_mkl = cute.local_tile(
            mFC1A_mkl,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        gFC1B_nkl = cute.local_tile(
            mFC1B_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )

        # (bM, bK, RestM, RestK, RestL)
        gFC1SFA_mkl = cute.local_tile(
            mFC1SFA_mkl,
            cute.slice_(self.fc1_cta_tile_shape_mnk_sfa, (None, 0, None)),
            (None, None, None),
        )

        # (bN, bK, RestN, RestK, RestL)
        gFC1SFB_nkl = cute.local_tile(
            mFC1SFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )

        gToken_ml = cute.local_tile(
            permuted_idx_to_expanded_idx,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, 0)),
            (None,),
        )

        # (bM, bN, loopM, loopN, loopL)
        gFC1C_mnl = cute.local_tile(
            mFC1C_mnl,
            cute.slice_(self.fc1_mma_tiler_c, (None, None, 0)),
            (None, None, None),
        )
        k_tile_cnt = cutlass.Int32(cute.size(gFC1A_mkl, mode=[3]))

        gFC2A_mkl = cute.local_tile(
            mFC2A_mkl,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        gFC2B_nkl = cute.local_tile(
            mFC2B_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )
        gFC2SFA_mkl = cute.local_tile(
            mFC2SFA_mkl,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        gFC2SFB_nkl = cute.local_tile(
            mFC2SFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        gFC2C_mnl = cute.local_tile(
            mFC2C_mnl,
            cute.slice_(self.mma_tiler, (None, None, 0)),
            (None, None, None),
        )
        fc2_k_tile_cnt = cutlass.Int32(cute.size(gFC2A_mkl, mode=[3]))

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgFC1B = thr_mma.partition_B(gFC1B_nkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgFC1SFB = thr_mma_sfb.partition_B(gFC1SFB_nkl)
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        tCgFC1C = thr_mma.partition_C(gFC1C_mnl)
        tCgFC2A = thr_mma.partition_A(gFC2A_mkl)
        tCgFC2B = thr_mma.partition_B(gFC2B_nkl)
        tCgFC2SFA = thr_mma.partition_A(gFC2SFA_mkl)
        tCgFC2SFB = thr_mma_sfb.partition_B(gFC2SFB_nkl)
        tCgFC2C = thr_mma.partition_C(gFC2C_mnl)

        #
        # Partition global/shared tensor for TMA load B
        #
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tFC1BsB, tFC1BgB = cpasync.tma_partition(
            tma_atom_fc1_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgFC1B, 0, 3),
        )

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tFC1BsSFB, tFC1BgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_fc1_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgFC1SFB, 0, 3),
        )
        tFC1BsSFB = cute.filter_zeros(tFC1BsSFB)
        tFC1BgSFB = cute.filter_zeros(tFC1BgSFB)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tFC2AsA, tFC2AgA = cpasync.tma_partition(
            tma_atom_fc2_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgFC2A, 0, 3),
        )
        tFC2AsSFA, tFC2AgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_fc2_sfa,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sFC2SFA, 0, 3),
            cute.group_modes(tCgFC2SFA, 0, 3),
        )
        tFC2AsSFA = cute.filter_zeros(tFC2AsSFA)
        tFC2AgSFA = cute.filter_zeros(tFC2AgSFA)
        tFC2BsB, tFC2BgB = cpasync.tma_partition(
            tma_atom_fc2_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgFC2B, 0, 3),
        )
        tFC2BsSFB, tFC2BgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_fc2_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgFC2SFB, 0, 3),
        )
        tFC2BsSFB = cute.filter_zeros(tFC2BsSFB)
        tFC2BgSFB = cute.filter_zeros(tFC2BgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        cute.arch.griddepcontrol_wait()

        #
        # Specialized Schedule Warp
        #
        if warp_idx == self.sched_warp_id:
            if cutlass.const_expr(self.use_l2_atomic_scheduler):
                scheduler_throttle_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    self.num_scheduler_stage,
                )
                fc1_tile_sched = L2AtomicPersistentTileScheduler.create(
                    fc1_tile_sched_params,
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                    storage.fc1_scheduler_response.data_ptr(),
                    fc1_scheduler_counter.iterator,
                )
                if cutlass.const_expr(self.use_cluster_scheduler_response):
                    fc1_scheduler_producer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.ProducerConsumer,
                        self.num_scheduler_stage,
                    )
                    fc1_scheduler_consumer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer,
                        self.num_scheduler_stage,
                    )
            else:
                fc1_tile_sched = utils.StaticPersistentTileScheduler.create(
                    fc1_tile_sched_params,
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                )
            work_tile = fc1_tile_sched.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )

            num_non_exiting_tiles_value = num_non_exiting_tiles[0]

            is_continue = cutlass.Boolean(1)
            while work_tile.is_valid_tile and is_continue:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_m = cur_tile_coord[0] // cute.size(
                    tiled_mma.thr_id.shape
                )
                if mma_tile_coord_m < num_non_exiting_tiles_value:
                    tile_info_pipeline.producer_acquire(tile_info_producer_state)
                    cur_tile_coord = work_tile.tile_idx
                    expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                    mn_limit = tile_idx_to_mn_limit[mma_tile_coord_m]
                    with cute.arch.elect_one():
                        sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                        sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                        sInfo[(2, tile_info_producer_state.index)] = expert_idx
                        sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                            work_tile.is_valid_tile
                        )
                        sInfo[(4, tile_info_producer_state.index)] = mn_limit
                        sInfo[(5, tile_info_producer_state.index)] = cutlass.Int32(0)
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )

                    self.sched_sync_barrier.arrive_and_wait()
                    tile_info_pipeline.producer_commit(tile_info_producer_state)
                    tile_info_producer_state.advance()

                    if cutlass.const_expr(self.use_l2_atomic_scheduler):
                        if is_scheduler_leader_cta:
                            scheduler_throttle_pipeline.consumer_wait(
                                scheduler_throttle_consumer_state
                            )
                            scheduler_throttle_pipeline.consumer_release(
                                scheduler_throttle_consumer_state
                            )
                            scheduler_throttle_consumer_state.advance()
                else:
                    is_continue = cutlass.Boolean(0)

                if is_continue:
                    if cutlass.const_expr(self.use_l2_atomic_scheduler):
                        if cutlass.const_expr(self.use_cluster_scheduler_response):
                            if is_scheduler_leader_cta:
                                fc1_scheduler_pipeline.producer_acquire(
                                    fc1_scheduler_producer_state
                                )
                                fc1_tile_sched.publish_next_work(
                                    fc1_scheduler_pipeline.producer_get_barrier(
                                        fc1_scheduler_producer_state
                                    )
                                )
                                fc1_scheduler_producer_state.advance()
                            fc1_scheduler_pipeline.consumer_wait(
                                fc1_scheduler_consumer_state
                            )
                            work_tile = fc1_tile_sched.get_published_work()
                            fc1_scheduler_pipeline.consumer_release(
                                fc1_scheduler_consumer_state
                            )
                            fc1_scheduler_consumer_state.advance()
                        else:
                            work_tile = fc1_tile_sched.claim_next_work_local()
                    else:
                        fc1_tile_sched.advance_to_next_work()
                        work_tile = fc1_tile_sched.get_current_work()

            if cutlass.const_expr(
                self.use_l2_atomic_scheduler
                and self.use_cluster_scheduler_response
            ):
                if is_scheduler_leader_cta:
                    fc1_scheduler_pipeline.producer_tail(
                        fc1_scheduler_producer_state
                    )

            # FC2 owns an independent work-ID stream. There is deliberately no
            # grid-wide phase fence: a fast cluster may start claiming FC2 as
            # soon as its FC1 stream ends, while FC2 A enforces per-M readiness.
            if cutlass.const_expr(self.use_l2_atomic_scheduler):
                fc2_tile_sched = L2AtomicPersistentTileScheduler.create(
                    fc2_tile_sched_params,
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                    storage.fc2_scheduler_response.data_ptr(),
                    fc2_scheduler_counter.iterator,
                )
                if cutlass.const_expr(self.use_cluster_scheduler_response):
                    fc2_scheduler_producer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.ProducerConsumer,
                        self.num_scheduler_stage,
                    )
                    fc2_scheduler_consumer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer,
                        self.num_scheduler_stage,
                    )
            else:
                fc2_tile_sched = utils.StaticPersistentTileScheduler.create(
                    fc2_tile_sched_params,
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                )
            fc2_work_tile = fc2_tile_sched.initial_work_tile_info()
            fc2_is_continue = cutlass.Boolean(1)
            while fc2_work_tile.is_valid_tile and fc2_is_continue:
                fc2_tile_coord = fc2_work_tile.tile_idx
                fc2_m_tile = fc2_tile_coord[0] // cute.size(
                    tiled_mma.thr_id.shape
                )
                if fc2_m_tile < num_non_exiting_tiles_value:
                    tile_info_pipeline.producer_acquire(tile_info_producer_state)
                    expert_idx = tile_idx_to_expert_idx[fc2_m_tile]
                    mn_limit = tile_idx_to_mn_limit[fc2_m_tile]
                    with cute.arch.elect_one():
                        sInfo[(0, tile_info_producer_state.index)] = fc2_tile_coord[0]
                        sInfo[(1, tile_info_producer_state.index)] = fc2_tile_coord[1]
                        sInfo[(2, tile_info_producer_state.index)] = expert_idx
                        sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(1)
                        sInfo[(4, tile_info_producer_state.index)] = mn_limit
                        sInfo[(5, tile_info_producer_state.index)] = cutlass.Int32(
                            FC2_PHASE
                        )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.sched_sync_barrier.arrive_and_wait()
                    tile_info_pipeline.producer_commit(tile_info_producer_state)
                    tile_info_producer_state.advance()

                    if cutlass.const_expr(self.use_l2_atomic_scheduler):
                        if is_scheduler_leader_cta:
                            scheduler_throttle_pipeline.consumer_wait(
                                scheduler_throttle_consumer_state
                            )
                            scheduler_throttle_pipeline.consumer_release(
                                scheduler_throttle_consumer_state
                            )
                            scheduler_throttle_consumer_state.advance()
                else:
                    fc2_is_continue = cutlass.Boolean(0)

                if fc2_is_continue:
                    if cutlass.const_expr(self.use_l2_atomic_scheduler):
                        if cutlass.const_expr(self.use_cluster_scheduler_response):
                            if is_scheduler_leader_cta:
                                fc2_scheduler_pipeline.producer_acquire(
                                    fc2_scheduler_producer_state
                                )
                                fc2_tile_sched.publish_next_work(
                                    fc2_scheduler_pipeline.producer_get_barrier(
                                        fc2_scheduler_producer_state
                                    )
                                )
                                fc2_scheduler_producer_state.advance()
                            fc2_scheduler_pipeline.consumer_wait(
                                fc2_scheduler_consumer_state
                            )
                            fc2_work_tile = fc2_tile_sched.get_published_work()
                            fc2_scheduler_pipeline.consumer_release(
                                fc2_scheduler_consumer_state
                            )
                            fc2_scheduler_consumer_state.advance()
                        else:
                            fc2_work_tile = fc2_tile_sched.claim_next_work_local()
                    else:
                        fc2_tile_sched.advance_to_next_work()
                        fc2_work_tile = fc2_tile_sched.get_current_work()

            if cutlass.const_expr(
                self.use_l2_atomic_scheduler
                and self.use_cluster_scheduler_response
            ):
                if is_scheduler_leader_cta:
                    fc2_scheduler_pipeline.producer_tail(
                        fc2_scheduler_producer_state
                    )

            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = -1
                sInfo[(1, tile_info_producer_state.index)] = -1
                sInfo[(2, tile_info_producer_state.index)] = -1
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(4, tile_info_producer_state.index)] = -1
                sInfo[(5, tile_info_producer_state.index)] = cutlass.Int32(END_PHASE)
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        # Gather A/SFA warps (warps 4-7). Four warps issue CpAsync128.CG for
        # every K tile; the (16, 8) thread layout covers 16 M rows by eight
        # K chunks, and padding rows are predicated off.
        if (
            warp_idx <= self.gather_a_warp_id[-1]
            and warp_idx >= self.gather_a_warp_id[0]
        ):
            cute.arch.setmaxregister_decrease(self.num_regs_gather_a_warps)
            a_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(
                    cache_mode=cpasync.LoadCacheMode.GLOBAL
                ),
                mFC1A_mkl.element_type,
                num_bits_per_copy=128,
            )
            a_thread_layout = cute.make_layout((16, 8), stride=(8, 1))
            a_value_layout = cute.make_layout((1, 32), stride=(32, 1))
            a_tiled_copy = cute.make_tiled_copy_tv(
                a_atom_copy,
                a_thread_layout,
                a_value_layout,
            )
            tidx_in_warpgroup = tidx % 128

            sA_tiled = cute.make_tensor(
                sA.iterator,
                layout=cute.make_layout(
                    (
                        self.cta_tile_shape_mnk[0],
                        self.cta_tile_shape_mnk[2],
                        self.num_ab_stage,
                    ),
                    stride=(
                        self.cta_tile_shape_mnk[2],
                        1,
                        self.cta_tile_shape_mnk[0] * self.cta_tile_shape_mnk[2],
                    ),
                ),
            )
            a_thr_copy = a_tiled_copy.get_slice(tidx_in_warpgroup)
            tFC1AsA_tiled = a_thr_copy.partition_D(sA_tiled)

            a_token_offset_tensor = cute.make_rmem_tensor(
                cute.make_layout((self.fc1_a_num_loads,)),
                cutlass.Int32,
            )
            a_predicate_tensor = cute.make_rmem_tensor(
                cute.make_layout((self.fc1_a_num_loads,)),
                cutlass.Boolean,
            )

            # SFA gather rides these same warps. One CpAsync128.CG per
            # thread per row per k_tile (16B = 16 FP8 SFs); sFC1SFA is plain
            # linear (M, tile_K_sf, stage). Same 128 threads as A, so
            # Each thread gathers one row at cta_tile_M=128, or two rows
            # in the inherited cta_tile_M=256 layout.
            sfa_tile_k_sf = self.cta_tile_shape_mnk[2] // self.sf_vec_size
            sfa_gather_threads = len(self.gather_a_warp_id) * self.threads_per_warp
            sfa_rows_per_thread = self.cta_tile_shape_mnk[0] // sfa_gather_threads
            sfa_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(
                    cache_mode=cpasync.LoadCacheMode.GLOBAL
                ),
                mFC1SFA_mkl.element_type,
                num_bits_per_copy=128,
            )
            sFC1SFA_tiled = cute.make_tensor(
                sFC1SFA.iterator,
                layout=cute.make_layout(
                    (
                        self.cta_tile_shape_mnk[0],
                        sfa_tile_k_sf,
                        self.num_ab_stage,
                    ),
                    stride=(
                        sfa_tile_k_sf,
                        1,
                        self.cta_tile_shape_mnk[0] * sfa_tile_k_sf,
                    ),
                ),
            )

            fc1_a_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            tile_info = cute.make_rmem_tensor((6,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
            tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
            tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
            tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
            is_valid_tile = (tile_info[3] == 1) and (
                tile_info[5] == FC1_PHASE
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                gToken_ml_tile = gToken_ml[(None, tile_info[0])]
                for i in range(self.fc1_a_num_loads):
                    token_ml_tile_offset = (tidx_in_warpgroup // 8) + i * 16
                    a_token_offset_tensor[i] = gToken_ml_tile[token_ml_tile_offset]
                    a_predicate_tensor[i] = (
                        cutlass.Boolean(1)
                        if tile_info[0] * self.cta_tile_shape_mnk[0]
                        + token_ml_tile_offset
                        < tile_info[4]
                        else cutlass.Boolean(0)
                    )
                    a_token_offset_tensor[i] = (
                        a_token_offset_tensor[i] // self.topk
                        if tile_info[0] * self.cta_tile_shape_mnk[0]
                        + token_ml_tile_offset
                        < tile_info[4]
                        else 0
                    )

                tFC1AgA = gFC1A_mkl[(None, None, 0, None, 0)]
                A_gmem_thread_offset = cute.assume(
                    (tidx_in_warpgroup % 8) * 32, divby=32
                )

                fc1_a_producer_state.reset_count()
                peek_fc1_a_empty_status = cutlass.Boolean(1)
                if fc1_a_producer_state.count < k_tile_cnt:
                    peek_fc1_a_empty_status = fc1_a_pipeline.producer_try_acquire(
                        fc1_a_producer_state
                    )

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    fc1_a_pipeline.producer_acquire(
                        fc1_a_producer_state, peek_fc1_a_empty_status
                    )

                    tFC1AgA_ktile = tFC1AgA[(None, None, fc1_a_producer_state.count)]
                    tFC1AsA_ktile = tFC1AsA_tiled[
                        (None, None, None, fc1_a_producer_state.index)
                    ]

                    for i in range(self.fc1_a_num_loads):
                        A_gmem_slice_offset = A_gmem_thread_offset + cute.assume(
                            a_token_offset_tensor[i] * tFC1AgA_ktile.layout[0].stride,
                            divby=32,
                        )
                        A_gmem_slice_offset = cute.assume(
                            A_gmem_slice_offset, divby=32
                        )
                        tFC1AgA_slice_ptr = tFC1AgA_ktile.iterator + A_gmem_slice_offset
                        tFC1AgA_slice = cute.make_tensor(
                            tFC1AgA_slice_ptr, layout=cute.make_layout((32,))
                        )
                        tFC1AsA_slice = cute.make_tensor(
                            tFC1AsA_ktile[(None, i, None)].iterator,
                            layout=cute.make_layout((32,)),
                        )
                        a_predicate_slice = cute.make_rmem_tensor(
                            cute.make_layout((1,)), cutlass.Boolean
                        )
                        a_predicate_slice[0] = a_predicate_tensor[i]
                        cute.copy_atom_call(
                            a_atom_copy,
                            tFC1AgA_slice,
                            tFC1AsA_slice,
                            pred=a_predicate_slice,
                        )

                    # SFA for the same stage, issued by the same threads so
                    # the single producer_commit below covers A and SFA.
                    for r in cutlass.range_constexpr(sfa_rows_per_thread):
                        row_in_cta = tidx_in_warpgroup + r * sfa_gather_threads
                        tok = gToken_ml_tile[row_in_cta]
                        sfa_row_id = (
                            cutlass.Int32(-1) if tok == -1 else tok // self.topk
                        )
                        sfa_pred = cute.make_rmem_tensor(
                            cute.make_layout((1,)), cutlass.Boolean
                        )
                        sfa_pred[0] = (
                            cutlass.Boolean(1)
                            if (
                                tile_info[0] * self.cta_tile_shape_mnk[0]
                                + row_in_cta
                                < tile_info[4]
                            )
                            else cutlass.Boolean(0)
                        )
                        tFC1AgSFA_row = gFC1SFA_mkl[(sfa_row_id, None, 0, None, 0)]
                        tFC1AgSFA_ktile = tFC1AgSFA_row[(None, k_tile)]
                        tFC1AgSFA_slice = cute.make_tensor(
                            tFC1AgSFA_ktile.iterator,
                            layout=cute.make_layout((sfa_tile_k_sf,)),
                        )
                        tFC1AsSFA_slice = cute.make_tensor(
                            sFC1SFA_tiled[
                                (row_in_cta, None, fc1_a_producer_state.index)
                            ].iterator,
                            cute.make_layout((sfa_tile_k_sf,)),
                        )
                        cute.copy_atom_call(
                            sfa_atom_copy,
                            tFC1AgSFA_slice,
                            tFC1AsSFA_slice,
                            pred=sfa_pred,
                        )

                    fc1_a_pipeline.producer_commit(fc1_a_producer_state)

                    fc1_a_producer_state.advance()
                    peek_fc1_a_empty_status = cutlass.Boolean(1)
                    if fc1_a_producer_state.count < k_tile_cnt:
                        peek_fc1_a_empty_status = fc1_a_pipeline.producer_try_acquire(
                            fc1_a_producer_state
                        )

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_tile = (tile_info[3] == 1) and (
                    tile_info[5] == FC1_PHASE
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            fc1_a_pipeline.producer_tail(fc1_a_producer_state)

            fc2_a_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            # The descriptor that ended the FC1 loop is either the first
            # FC2 task or END and has already been released. Only warp 4,
            # which becomes the FC2 A/SFA producer, acquire-waits for the
            # corresponding M tile. Warps 5-7 stay independent so the B
            # side never inherits this dependency.
            is_valid_fc2_tile = tile_info[3] == 1
            while is_valid_fc2_tile:
                if warp_idx == self.gather_a_warp_id[0]:
                    fc2_m_tile = tile_info[0] // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    ready = ld_acquire_gpu(fc1_ready, fc2_m_tile)
                    while ready != fc1_ready_expected:
                        nanosleep(cutlass.Int32(32))
                        ready = ld_acquire_gpu(fc1_ready, fc2_m_tile)

                    fc2_a_gmem = tFC2AgA[(None, fc2_m_tile, None, 0)]
                    fc2_sfa_gmem = tFC2AgSFA[(None, fc2_m_tile, None, 0)]
                    fc2_a_producer_state.reset_count()
                    peek_fc2_a_empty = cutlass.Boolean(1)
                    if fc2_a_producer_state.count < fc2_k_tile_cnt:
                        peek_fc2_a_empty = fc2_a_pipeline.producer_try_acquire(
                            fc2_a_producer_state
                        )
                    for _ in cutlass.range(
                        0, fc2_k_tile_cnt, 1, unroll=1
                    ):
                        fc2_a_pipeline.producer_acquire(
                            fc2_a_producer_state, peek_fc2_a_empty
                        )
                        fc2_a_barrier = fc2_a_pipeline.producer_get_barrier(
                            fc2_a_producer_state
                        )
                        cute.copy(
                            tma_atom_fc2_a,
                            fc2_a_gmem[
                                (None, fc2_a_producer_state.count)
                            ],
                            tFC2AsA[(None, fc2_a_producer_state.index)],
                            tma_bar_ptr=fc2_a_barrier,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_fc2_sfa,
                            fc2_sfa_gmem[
                                (None, fc2_a_producer_state.count)
                            ],
                            tFC2AsSFA[(None, fc2_a_producer_state.index)],
                            tma_bar_ptr=fc2_a_barrier,
                            mcast_mask=sfa_full_mcast_mask,
                        )
                        fc2_a_producer_state.advance()
                        peek_fc2_a_empty = cutlass.Boolean(1)
                        if fc2_a_producer_state.count < fc2_k_tile_cnt:
                            peek_fc2_a_empty = (
                                fc2_a_pipeline.producer_try_acquire(
                                    fc2_a_producer_state
                                )
                            )

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_fc2_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            if warp_idx == self.gather_a_warp_id[0]:
                fc2_a_pipeline.producer_tail(fc2_a_producer_state)

        # MMA-group warps (8-11: mma / tma_b / sched / sync-transform relay).
        # These previously never called setmaxnreg and sat at the launch default
        # of 96 regs/thread. At 12 warps the CTA budget is 96*384 = 36864, so
        # holding them at 96 would leave gather only 24 regs (the PTX minimum)
        # once the epilogue takes its tuned 168. Dropping this group to 48 is
        # what buys gather a workable 72. Budget: 128*(168+72+48) = 36864 exactly.
        if warp_idx >= self.mma_warp_id and warp_idx <= self.sync_transform_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma_group_warps)

        # Sync-transform / metadata warp (warp 11). In 2CTA, it consumes the
        # per-CTA cp.async pipeline and re-publishes it cluster-wide so the MMA
        # sees both CTAs' A and SFA. It then loads FC2 scatter metadata; 1CTA
        # skips the relay and enters the same metadata loop directly.
        if warp_idx == self.sync_transform_warp_id:
            meta_lane = tidx % self.threads_per_warp
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            meta_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.fc2_num_meta_stage
            )
            tile_info = cute.make_rmem_tensor((6,), cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
            tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
            tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
            tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
            tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            if cutlass.const_expr(self.use_2cta_instrs):
                fc1_a_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                a_sync_transform_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage
                )
                is_valid_fc1_tile = (tile_info[3] == 1) and (
                    tile_info[5] == FC1_PHASE
                )

                while is_valid_fc1_tile:
                    fc1_a_consumer_state.reset_count()
                    peek_fc1_a_full_status = cutlass.Boolean(1)
                    if fc1_a_consumer_state.count < k_tile_cnt:
                        peek_fc1_a_full_status = fc1_a_pipeline.consumer_try_wait(
                            fc1_a_consumer_state
                        )
                    a_sync_transform_producer_state.reset_count()
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # Wait per-CTA A full → commit cluster-wide A sync-transform
                        # full. We do NOT release fc1_a_pipeline here; MMA owns its
                        # consumer_release so each CTA's producer sees the empty arrive.
                        fc1_a_pipeline.consumer_wait(
                            fc1_a_consumer_state, peek_fc1_a_full_status
                        )
                        fc1_a_sync_transform_pipeline.producer_commit(
                            a_sync_transform_producer_state
                        )
                        a_sync_transform_producer_state.advance()
                        fc1_a_consumer_state.advance()
                        peek_fc1_a_full_status = cutlass.Boolean(1)
                        if fc1_a_consumer_state.count < k_tile_cnt:
                            peek_fc1_a_full_status = fc1_a_pipeline.consumer_try_wait(
                                fc1_a_consumer_state
                            )

                    # Advance to next tile
                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                    tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
                    tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                    tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                    tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                    is_valid_fc1_tile = (tile_info[3] == 1) and (
                        tile_info[5] == FC1_PHASE
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                # Drain the sync-transform pipeline before exit.
                fc1_a_sync_transform_pipeline.producer_tail(
                    a_sync_transform_producer_state
                )
            else:
                # FC1 has no warp-11 work in the 1CTA kernel. Consume its
                # descriptors so both MMA modes enter the common FC2 metadata
                # loop with the first FC2 descriptor still resident.
                is_valid_fc1_tile = (tile_info[3] == 1) and (
                    tile_info[5] == FC1_PHASE
                )
                while is_valid_fc1_tile:
                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                    tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
                    tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                    tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                    tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                    is_valid_fc1_tile = (tile_info[3] == 1) and (
                        tile_info[5] == FC1_PHASE
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

            is_valid_fc2_tile = (tile_info[3] == 1) and (
                tile_info[5] == FC2_PHASE
            )
            while is_valid_fc2_tile:
                tile_m_start = tile_info[0] * self.cta_tile_shape_mnk[0]
                expert_idx = tile_info[2]
                alpha_val = fc2_alpha[expert_idx]

                meta_pipeline.producer_acquire(meta_producer_state)
                meta_stage = meta_producer_state.index
                for j in cutlass.range(
                    self.cta_tile_shape_mnk[0] // self.threads_per_warp,
                    unroll_full=True,
                ):
                    row = meta_lane + j * self.threads_per_warp
                    permuted_row = tile_m_start + row
                    expanded_idx = permuted_idx_to_expanded_idx[permuted_row]
                    safe_idx = cutlass.max(expanded_idx, cutlass.Int32(0))
                    token_idx = safe_idx // self.topk
                    topk_idx = safe_idx % self.topk
                    is_valid_row = cutlass.Int32(
                        permuted_row < tile_info[4]
                    )
                    gather_token = token_idx * is_valid_row
                    token_scale = fc2_routing_scales[(gather_token, topk_idx)]
                    sFC2MetaTokenIdx[(row, meta_stage)] = token_idx
                    sFC2MetaScale[(row, meta_stage)] = alpha_val * token_scale
                cute.arch.fence_proxy("async.shared", space="cta")
                meta_pipeline.producer_commit(meta_producer_state)
                meta_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_fc2_tile = (tile_info[3] == 1) and (
                    tile_info[5] == FC2_PHASE
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            meta_pipeline.producer_tail(meta_producer_state)

        # TMA B/SFB load warp (warp 9). Loads B/SFB GMEM → SMEM with multicast.
        if warp_idx == self.tma_b_warp_id:
            fc1_b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            if cutlass.const_expr(self.use_l2_atomic_scheduler):
                scheduler_throttle_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.num_scheduler_stage,
                )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((6,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
            tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
            tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
            tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
            tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
            tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
            is_valid_tile = (tile_info[3] == 1) and (
                tile_info[5] == FC1_PHASE
            )
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                if cutlass.const_expr(self.use_l2_atomic_scheduler):
                    if is_scheduler_leader_cta:
                        scheduler_throttle_pipeline.producer_acquire(
                            scheduler_throttle_producer_state
                        )
                        scheduler_throttle_pipeline.producer_commit(
                            scheduler_throttle_producer_state
                        )
                        scheduler_throttle_producer_state.advance()
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), loopK)
                tFC1BgB_slice = tFC1BgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # Apply SFB slicing hack when cta_tile_shape_n=64
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2

                # ((atom_v, rest_v), RestK)
                tFC1BgSFB_slice = tFC1BgSFB[
                    (None, slice_n, None, mma_tile_coord_mnl[2])
                ]

                # Peek (try_wait) B buffer empty for k_tile = prefetch_k_tile_cnt
                fc1_b_producer_state.reset_count()
                peek_fc1_b_empty_status = cutlass.Boolean(1)
                if fc1_b_producer_state.count < k_tile_cnt:
                    peek_fc1_b_empty_status = (
                        fc1_b_pipeline.producer_try_acquire(
                            fc1_b_producer_state
                        )
                    )
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    fc1_b_pipeline.producer_acquire(
                        fc1_b_producer_state, peek_fc1_b_empty_status
                    )

                    tFC1BgB_k = tFC1BgB_slice[(None, fc1_b_producer_state.count)]
                    tFC1BgSFB_k = tFC1BgSFB_slice[(None, fc1_b_producer_state.count)]
                    tFC1BsB_pipe = tFC1BsB[(None, fc1_b_producer_state.index)]
                    tFC1BsSFB_pipe = tFC1BsSFB[(None, fc1_b_producer_state.index)]

                    tma_bar = fc1_b_pipeline.producer_get_barrier(
                        fc1_b_producer_state
                    )

                    # TMA load B
                    cute.copy(
                        tma_atom_fc1_b,
                        tFC1BgB_k,
                        tFC1BsB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )

                    # TMA load SFB
                    cute.copy(
                        tma_atom_fc1_sfb,
                        tFC1BgSFB_k,
                        tFC1BsSFB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # Peek (try_wait) B buffer empty for k_tile + 1
                    fc1_b_producer_state.advance()
                    peek_fc1_b_empty_status = cutlass.Boolean(1)
                    if fc1_b_producer_state.count < k_tile_cnt:
                        peek_fc1_b_empty_status = (
                            fc1_b_pipeline.producer_try_acquire(
                                fc1_b_producer_state
                            )
                        )

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
                tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_tile = (tile_info[3] == 1) and (
                    tile_info[5] == FC1_PHASE
                )
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            fc1_b_pipeline.producer_tail(fc1_b_producer_state)
            fc2_b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            is_valid_fc2_tile = tile_info[3] == 1
            while is_valid_fc2_tile:
                if cutlass.const_expr(self.use_l2_atomic_scheduler):
                    if is_scheduler_leader_cta:
                        scheduler_throttle_pipeline.producer_acquire(
                            scheduler_throttle_producer_state
                        )
                        scheduler_throttle_pipeline.producer_commit(
                            scheduler_throttle_producer_state
                        )
                        scheduler_throttle_producer_state.advance()
                fc2_n_tile = tile_info[1]
                fc2_expert = tile_info[2]
                fc2_b_gmem = tFC2BgB[
                    (None, fc2_n_tile, None, fc2_expert)
                ]
                fc2_sfb_n_tile = fc2_n_tile
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    fc2_sfb_n_tile = fc2_n_tile // 2
                fc2_sfb_gmem = tFC2BgSFB[
                    (None, fc2_sfb_n_tile, None, fc2_expert)
                ]
                fc2_b_producer_state.reset_count()
                peek_fc2_b_empty = cutlass.Boolean(1)
                if fc2_b_producer_state.count < fc2_k_tile_cnt:
                    peek_fc2_b_empty = fc2_b_pipeline.producer_try_acquire(
                        fc2_b_producer_state
                    )
                for _ in cutlass.range(0, fc2_k_tile_cnt, 1, unroll=1):
                    fc2_b_pipeline.producer_acquire(
                        fc2_b_producer_state, peek_fc2_b_empty
                    )
                    fc2_b_barrier = fc2_b_pipeline.producer_get_barrier(
                        fc2_b_producer_state
                    )
                    cute.copy(
                        tma_atom_fc2_b,
                        fc2_b_gmem[(None, fc2_b_producer_state.count)],
                        tFC2BsB[(None, fc2_b_producer_state.index)],
                        tma_bar_ptr=fc2_b_barrier,
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_fc2_sfb,
                        fc2_sfb_gmem[(None, fc2_b_producer_state.count)],
                        tFC2BsSFB[(None, fc2_b_producer_state.index)],
                        tma_bar_ptr=fc2_b_barrier,
                        mcast_mask=sfb_full_mcast_mask,
                    )
                    fc2_b_producer_state.advance()
                    peek_fc2_b_empty = cutlass.Boolean(1)
                    if fc2_b_producer_state.count < fc2_k_tile_cnt:
                        peek_fc2_b_empty = fc2_b_pipeline.producer_try_acquire(
                            fc2_b_producer_state
                        )

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
                tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_fc2_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            fc2_b_pipeline.producer_tail(fc2_b_producer_state)
            if cutlass.const_expr(self.use_l2_atomic_scheduler):
                if is_scheduler_leader_cta:
                    scheduler_throttle_pipeline.producer_tail(
                        scheduler_throttle_producer_state
                    )

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # SFA 128dp_Unique TMEM (single buffer). Written 1:1 each k_tile by
            # a Cp128x128b UTCCP from gathered sFC1SFA smem (issued in this warp,
            # right before the MMA — like SFB). tCtFC1SFA_layout == the 128dp
            # (128, nsf):(1<<18, 1) layout passed from the host.
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtFC1SFA_128 = cute.make_tensor(sfa_tmem_ptr, tCtFC1SFA_layout)
            # FC1 and FC2 reuse the same SFA TMEM allocation, but interpret it
            # through different layouts. FC1 uses the 128-datapath Unique
            # layout required by its gathered activation scales, while FC2
            # consumes the standard block-scaled layout used by TMA-loaded
            # intermediate scales.
            tCtFC2SFA = cute.make_tensor(sfa_tmem_ptr, tCtFC2SFA_layout)
            # Per-k-block SFA base-address view for the manual MMA: k-block k
            # reads 128dp starting at sf-column k * sfa_sf_per_kblock. Trivial
            # (V=1, MN=1) modes — only the base address per k-block matters;
            # HW expands the 128 datapaths when idesc bit26=1.
            tCtFC1SFA_mma = cute.make_tensor(
                sfa_tmem_ptr,
                cute.make_layout(
                    (1, 1, self.fc1_sfa_nsf // self.fc1_sfa_sf_per_kblock),
                    stride=(0, 0, self.fc1_sfa_sf_per_kblock),
                ),
            )

            # Make SFB tmem tensor (using precomputed layout)
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            # UTCCP s2t bundles: SFA (Cp128x128b, 128dp) + SFB (Cp4x32x128b).
            sfa_s2t_bundle = self._fc1_sfa_s2t_copy_and_partition_128dp(
                sFC1SFA, tCtFC1SFA_128
            )
            fc2_sfa_s2t_bundle = self._mainloop_s2t_copy_and_partition(
                sFC2SFA, tCtFC2SFA
            )
            sfb_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            fc1_a_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            fc1_b_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            if cutlass.const_expr(self.use_2cta_instrs):
                a_sync_transform_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info from pipeline (scheduler has filtered out tiles >= num_non_exiting_tiles)
            tile_info = cute.make_rmem_tensor((6,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
            tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
            tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
            tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
            tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
            tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
            is_valid_tile = (tile_info[3] == 1) and (
                tile_info[5] == FC1_PHASE
            )
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # Peek (try_wait) A / B / SFA buffer full for k_tile = 0.
                # 1CTA waits directly on fc1_a_pipeline. The inherited 2CTA path
                # waits on the cluster-wide A relay pipeline.
                if cutlass.const_expr(self.use_2cta_instrs):
                    a_sync_transform_consumer_state.reset_count()
                    peek_a_sync_transform_full_status = cutlass.Boolean(1)
                    if (
                        a_sync_transform_consumer_state.count < k_tile_cnt
                        and is_leader_cta
                    ):
                        peek_a_sync_transform_full_status = (
                            fc1_a_sync_transform_pipeline.consumer_try_wait(
                                a_sync_transform_consumer_state
                            )
                        )
                    fc1_a_consumer_state.reset_count()
                else:
                    fc1_a_consumer_state.reset_count()
                    peek_fc1_a_full_status = cutlass.Boolean(1)
                    if fc1_a_consumer_state.count < k_tile_cnt:
                        peek_fc1_a_full_status = fc1_a_pipeline.consumer_try_wait(
                            fc1_a_consumer_state
                        )
                fc1_b_consumer_state.reset_count()
                peek_fc1_b_full_status = cutlass.Boolean(1)
                if fc1_b_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_fc1_b_full_status = fc1_b_pipeline.consumer_try_wait(
                        fc1_b_consumer_state
                    )

                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                # Accumulator stage.
                acc_stage_index = acc_producer_state.index

                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                # Apply TMEM pointer offset hack when cta_tile_shape_n=192 or
                # cta_tile_shape_n=64
                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    # If this is an ODD tile, shift the TMEM start address for
                    # cta_tile_shape_n=192 case by two words
                    # (ignores first 64 columns of SFB)
                    offset = (
                        cutlass.Int32(2)
                        if mma_tile_coord_mnl[1] % 2 == 1
                        else cutlass.Int32(0)
                    )
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    # Move in increments of 64 columns of SFB
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                    #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                #
                # Mma mainloop
                #

                for k_tile in cutlass.range(k_tile_cnt):
                    # Set tensor memory buffer for current tile
                    # (MMA, MMA_M, MMA_N)

                    if is_leader_cta:
                        # Wait for A / B / SFA buffer full.
                        # A and SFA share fc1_a_pipeline; B has an independent
                        # TMA pipeline. The inherited 2CTA A path waits on its
                        # cluster-wide relay.
                        if cutlass.const_expr(self.use_2cta_instrs):
                            fc1_a_sync_transform_pipeline.consumer_wait(
                                a_sync_transform_consumer_state,
                                peek_a_sync_transform_full_status,
                            )
                        else:
                            fc1_a_pipeline.consumer_wait(
                                fc1_a_consumer_state, peek_fc1_a_full_status
                            )
                        fc1_b_pipeline.consumer_wait(
                            fc1_b_consumer_state, peek_fc1_b_full_status
                        )
                        a_stage_idx = fc1_a_consumer_state.index
                        b_stage_idx = fc1_b_consumer_state.index
                        sfa_stage_idx = a_stage_idx

                        # SFA 128dp UTCCP (Cp128x128b, smem → 128dp TMEM) +
                        # SFB UTCCP (Cp4x32x128b). Both land before the MMA.
                        sfa_s2t_stage_coord = (None, None, None, sfa_stage_idx)
                        cute.copy(
                            sfa_s2t_bundle.tiled_copy,
                            sfa_s2t_bundle.sSF_compact[sfa_s2t_stage_coord],
                            sfa_s2t_bundle.tSF_compact,
                        )
                        self._mainloop_s2t_copies(b_stage_idx, sfb_s2t_bundle)

                        num_kblocks = cute.size(tCrA, mode=[2])

                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            if cutlass.const_expr(
                                self.enable_breuse
                                and cute.size(tCtAcc.layout, mode=[1]) == 2
                                and cute.size(tCtAcc.layout, mode=[2]) == 1
                            ):
                                tCtAcc_bkeep = tCtAcc[(None, 0, 0)]
                                tCtAcc_breuse = tCtAcc[(None, 1, 0)]

                                a_kblk_crd_keep = (None, 0, kblock_idx, a_stage_idx)
                                a_kblk_crd_reuse = (None, 1, kblock_idx, a_stage_idx)
                                b_kblk_crd = (None, 0, kblock_idx, b_stage_idx)

                                sfa_kblk_crd_keep = (None, 0, kblock_idx)
                                sfa_kblk_crd_reuse = (None, 1, kblock_idx)
                                sfb_kblk_crd = (None, 0, kblock_idx)

                                # Bkeep
                                tiled_mma_bkeep.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or kblock_idx != 0,
                                )
                                cute.gemm(
                                    tiled_mma_bkeep,
                                    tCtAcc_bkeep,
                                    [
                                        tCrA[a_kblk_crd_keep],
                                        tCtFC1SFA[sfa_kblk_crd_keep],
                                    ],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_bkeep,
                                )
                                # Breuse
                                tiled_mma_breuse.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or kblock_idx != 0,
                                )
                                cute.gemm(
                                    tiled_mma_breuse,
                                    tCtAcc_breuse,
                                    [
                                        tCrA[a_kblk_crd_reuse],
                                        tCtFC1SFA[sfa_kblk_crd_reuse],
                                    ],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_breuse,
                                )
                            else:
                                a_kblock_coord = (None, None, kblock_idx, a_stage_idx)
                                b_kblock_coord = (None, None, kblock_idx, b_stage_idx)
                                sf_kblock_coord = (None, None, kblock_idx)

                                # Hand-encoded UTCOMMA (replaces cute.gemm) so
                                # idesc bit 26 (SFA layout) is under our control.
                                manual_mma_128dp.issue_manual_block_scaled_mma_atom(
                                    acc_frag=tCtAcc,
                                    a_frag=tCrA[a_kblock_coord],
                                    sfa_frag=tCtFC1SFA_mma[sf_kblock_coord],
                                    b_frag=tCrB[b_kblock_coord],
                                    sfb_frag=tCtSFB_mma[sf_kblock_coord],
                                    static_idesc_base=self.static_idesc_base,
                                    accumulate=(k_tile != 0 or kblock_idx != 0),
                                    cta_group=self.mma_cta_group_int,
                                )

                        # Release A/SFA and B buffer-empty barriers.
                        fc1_a_pipeline.consumer_release(fc1_a_consumer_state)
                        if cutlass.const_expr(self.use_2cta_instrs):
                            fc1_a_sync_transform_pipeline.consumer_release(
                                a_sync_transform_consumer_state
                            )
                        fc1_b_pipeline.consumer_release(fc1_b_consumer_state)

                    # Peek (try_wait) A / B / SFA buffer full for k_tile + 1.
                    if cutlass.const_expr(self.use_2cta_instrs):
                        a_sync_transform_consumer_state.advance()
                        peek_a_sync_transform_full_status = cutlass.Boolean(1)
                        if (
                            a_sync_transform_consumer_state.count < k_tile_cnt
                            and is_leader_cta
                        ):
                            peek_a_sync_transform_full_status = (
                                fc1_a_sync_transform_pipeline.consumer_try_wait(
                                    a_sync_transform_consumer_state
                                )
                            )
                        fc1_a_consumer_state.advance()
                    else:
                        fc1_a_consumer_state.advance()
                        peek_fc1_a_full_status = cutlass.Boolean(1)
                        if fc1_a_consumer_state.count < k_tile_cnt:
                            peek_fc1_a_full_status = fc1_a_pipeline.consumer_try_wait(
                                fc1_a_consumer_state
                            )
                    fc1_b_consumer_state.advance()
                    peek_fc1_b_full_status = cutlass.Boolean(1)
                    if fc1_b_consumer_state.count < k_tile_cnt and is_leader_cta:
                        peek_fc1_b_full_status = fc1_b_pipeline.consumer_try_wait(
                            fc1_b_consumer_state
                        )

                #
                # Async arrive accumulator buffer full(each kblock)
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)

                # Peek (try_wait) Acc buffer empty for k_tile = k_tile + 1
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
                tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_tile = (tile_info[3] == 1) and (
                    tile_info[5] == FC1_PHASE
                )
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            acc_pipeline.producer_tail(acc_producer_state)
            fc2_a_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            fc2_b_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            is_valid_fc2_tile = tile_info[3] == 1
            while is_valid_fc2_tile:
                acc_stage_index = acc_producer_state.index
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                fc2_a_consumer_state.reset_count()
                fc2_b_consumer_state.reset_count()
                peek_fc2_a_full = cutlass.Boolean(1)
                peek_fc2_b_full = cutlass.Boolean(1)
                if (
                    fc2_a_consumer_state.count < fc2_k_tile_cnt
                    and is_leader_cta
                ):
                    peek_fc2_a_full = fc2_a_pipeline.consumer_try_wait(
                        fc2_a_consumer_state
                    )
                    peek_fc2_b_full = fc2_b_pipeline.consumer_try_wait(
                        fc2_b_consumer_state
                    )

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                for k_tile in cutlass.range(0, fc2_k_tile_cnt, 1, unroll=1):
                    if is_leader_cta:
                        fc2_a_pipeline.consumer_wait(
                            fc2_a_consumer_state, peek_fc2_a_full
                        )
                        fc2_b_pipeline.consumer_wait(
                            fc2_b_consumer_state, peek_fc2_b_full
                        )

                        a_stage_idx = fc2_a_consumer_state.index
                        b_stage_idx = fc2_b_consumer_state.index
                        fc2_sfa_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            a_stage_idx,
                        )
                        cute.copy(
                            fc2_sfa_s2t_bundle.tiled_copy,
                            fc2_sfa_s2t_bundle.sSF_compact[
                                fc2_sfa_stage_coord
                            ],
                            fc2_sfa_s2t_bundle.tSF_compact,
                        )
                        self._mainloop_s2t_copies(
                            b_stage_idx, sfb_s2t_bundle
                        )

                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(
                            num_kblocks, unroll_full=True
                        ):
                            a_kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                a_stage_idx,
                            )
                            b_kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                b_stage_idx,
                            )
                            sf_kblock_coord = (None, None, kblock_idx)

                            tiled_mma.set(
                                tcgen05.Field.ACCUMULATE,
                                k_tile != 0 or kblock_idx != 0,
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                [
                                    tCrA[a_kblock_coord],
                                    tCtFC2SFA[sf_kblock_coord],
                                ],
                                [
                                    tCrB[b_kblock_coord],
                                    tCtSFB[sf_kblock_coord],
                                ],
                                tCtAcc,
                            )

                        fc2_a_pipeline.consumer_release(fc2_a_consumer_state)
                        fc2_b_pipeline.consumer_release(fc2_b_consumer_state)
                    fc2_a_consumer_state.advance()
                    fc2_b_consumer_state.advance()
                    peek_fc2_a_full = cutlass.Boolean(1)
                    peek_fc2_b_full = cutlass.Boolean(1)
                    if (
                        fc2_a_consumer_state.count < fc2_k_tile_cnt
                        and is_leader_cta
                    ):
                        peek_fc2_a_full = fc2_a_pipeline.consumer_try_wait(
                            fc2_a_consumer_state
                        )
                        peek_fc2_b_full = fc2_b_pipeline.consumer_try_wait(
                            fc2_b_consumer_state
                        )

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_fc2_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

        #
        # Specialized epilogue warps
        #
        if warp_idx <= self.epilog_warp_id[-1]:
            # Register reconfig: epilogue needs many regs for SwiGLU
            cute.arch.setmaxregister_increase(self.num_regs_epilogue_warps)
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            # Epilogue partition: transform both accumulator and C layout.
            # transform_partitioned_tensor_layout merges (MMA_ATOM, MMA_M) → flat M.
            tCtAcc_transformed = transform_partitioned_tensor_layout(tCtAcc_base)
            tCgFC1C_for_epi = transform_partitioned_tensor_layout(tCgFC1C)
            tCgFC2C_for_epi = transform_partitioned_tensor_layout(tCgFC2C)

            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tFC1TR_tAcc_base,
                tFC1TR_rAcc_up,
                tFC1TR_rAcc_gate,
            ) = self.fc1_epilogue_tmem_copy_and_partition(
                epi_tidx,
                tCtAcc_transformed,
                tCgFC1C_for_epi,
                fc1_epi_tile,
                use_2cta_instrs,
            )

            (
                fc2_tiled_copy_t2r,
                tFC2TR_tAcc_base,
                tFC2TR_rAcc,
            ) = self.fc2_epilogue_tmem_copy_and_partition(
                epi_tidx,
                tCtAcc_transformed,
                tCgFC2C_for_epi,
                fc2_epi_tile,
                use_2cta_instrs,
            )
            tFC2TR_rC = cute.make_rmem_tensor(
                tFC2TR_rAcc.shape, self.fc2_c_dtype
            )
            fc2_r2s_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.fc2_c_dtype
            )
            fc2_tiled_copy_r2s = cute.make_tiled_copy_D(
                fc2_r2s_atom, fc2_tiled_copy_t2r
            )
            fc2_thr_copy_r2s = fc2_tiled_copy_r2s.get_slice(epi_tidx)
            tFC2RS_sC = fc2_thr_copy_r2s.partition_D(sFC2C)
            tFC2RS_rC = fc2_tiled_copy_r2s.retile(tFC2TR_rC)

            tFC1TR_rC = None
            tiled_copy_r2s = None
            tFC1RS_rC = None
            tFC1RS_sC = None
            bSG_sC = None
            bSG_gC_partitioned = None
            tFC1TR_rC = cute.make_rmem_tensor(tFC1TR_rAcc_up.shape, self.fc1_c_dtype)
            tiled_copy_r2s, tFC1RS_rC, tFC1RS_sC = (
                self.fc1_epilogue_smem_copy_and_partition(
                    tiled_copy_t2r, tFC1TR_rC, epi_tidx, sFC1C
                )
            )
            (
                tma_atom_fc1_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.fc1_epilogue_gmem_copy_and_partition(
                epi_tidx, tma_atom_fc1_c, tCgFC1C_for_epi, fc1_epi_tile, sFC1C
            )

            if cutlass.const_expr(self.fc1_generate_sfc):
                norm_const = fc1_norm_const[0]
                # (EPI_TILE_M, EPI_TILE_N, RestM, RestN, RestL)
                gFC1SFC_mnl = cute.local_tile(
                    mFC1SFC_mnl, fc1_epi_tile, (None, None, None)
                )

                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                # (T2R, T2R_M, T2R_N, RestM, RestN, RestL)
                tCgFC1SFC_mnl = thr_copy_t2r.partition_D(gFC1SFC_mnl)
                tCgFC1SFC_mnl = cute.filter_zeros(tCgFC1SFC_mnl)
                # (T2R, T2R_M, T2R_N)
                tCrFC1SFC = cute.make_rmem_tensor(
                    tCgFC1SFC_mnl[(None, None, None, 0, 0, 0)].layout, self.sf_dtype
                )
                tCrFC1SFC_pvscale = cute.make_rmem_tensor_like(
                    tCrFC1SFC, cutlass.Float32
                )

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            meta_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.fc2_num_meta_stage
            )

            fc1_c_pipeline = None
            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            fc1_c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.fc1_num_c_stage,
                producer_group=c_producer_group,
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((6,), cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
            tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
            tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
            tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
            tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
            tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
            is_valid_tile = (tile_info[3] == 1) and (
                tile_info[5] == FC1_PHASE
            )
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            num_prev_subtiles = cutlass.Int32(0)
            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Get alpha for current group
                #

                expert_idx = mma_tile_coord_mnl[2]
                alpha_val = fc1_alpha[expert_idx]

                #
                # Slice to per mma tile index
                #
                bSG_gC = None
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        mma_tile_coord_mnl[0],
                        mma_tile_coord_mnl[1],
                        0,
                    )
                ]

                # Get accumulator stage index.
                acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tFC1TR_tAcc = tFC1TR_tAcc_base[
                    (None, None, None, None, None, acc_stage_index)
                ]

                if cutlass.const_expr(self.fc1_generate_sfc):
                    # (T2R, T2R_M, T2R_N, RestM, RestN)
                    tCgFC1SFC_mn = tCgFC1SFC_mnl[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            0,
                        )
                    ]

                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                # SwiGLU epilogue. Acc has full N cols with interleaved
                # [up, gate] at granularity=64; C has N/2 cols. Iterate M and
                # N output subtiles → up * silu(gate).
                # tFC1TR_tAcc: (T2R, T2R_M, T2R_N, EPI_M, EPI_N,
                # STAGE), sliced on STAGE. bSG_gC: ((ATOM_V, REST_V),
                # EPI_M, EPI_N, loopM, loopN, loopL).
                interleave_granularity = 64
                gate_offset = interleave_granularity // self.fc1_epi_tile_n
                epi_m_cnt = cute.size(tFC1TR_tAcc.shape, mode=[3])
                acc_n_subtile_cnt = cute.size(tFC1TR_tAcc.shape, mode=[4])
                out_n_subtile_cnt = (
                    acc_n_subtile_cnt // 2
                )  # N/2 output subtiles per M subtile

                for epi_m_idx in cutlass.range(epi_m_cnt):
                    for out_n_idx in cutlass.range(out_n_subtile_cnt):
                        # Map output N subtile → acc N subtile. Each
                        # interleave block of 2*gate_offset subtiles is
                        # [up*gate_offset, gate*gate_offset].
                        real_out_n_idx = out_n_idx
                        block_idx = real_out_n_idx // gate_offset
                        within_block = real_out_n_idx % gate_offset
                        up_n_subtile = block_idx * 2 * gate_offset + within_block
                        gate_n_subtile = (
                            block_idx * 2 * gate_offset + gate_offset + within_block
                        )
                        #
                        # Load accumulator from tensor memory buffer to register
                        #
                        tFC1TR_tAcc_mn_up = tFC1TR_tAcc[
                            (None, None, None, epi_m_idx, up_n_subtile)
                        ]
                        tFC1TR_tAcc_mn_gate = tFC1TR_tAcc[
                            (None, None, None, epi_m_idx, gate_n_subtile)
                        ]

                        cute.copy(tiled_copy_t2r, tFC1TR_tAcc_mn_up, tFC1TR_rAcc_up)
                        cute.copy(tiled_copy_t2r, tFC1TR_tAcc_mn_gate, tFC1TR_rAcc_gate)

                        acc_vec_up = tFC1TR_rAcc_up.load()
                        acc_vec_gate = tFC1TR_rAcc_gate.load()

                        # SwiGLU: output = up * silu(gate),  silu(x) = x * sigmoid(x).
                        tCompute = cute.make_rmem_tensor(
                            acc_vec_gate.shape, self.acc_dtype
                        )
                        if cutlass.const_expr(self.vectorized_f32):
                            # SwiGLU Packed Version: uses f32x2 packed operations for better performance
                            # Computes: output = (alpha * up) * silu(alpha * gate)
                            # where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
                            LOG2_E = cutlass.Float32(1.4426950408889634)
                            for i in cutlass.range_constexpr(
                                0, cute.size(tFC1TR_rAcc_up), 2
                            ):
                                acc_vec_up_alpha = cute.arch.mul_packed_f32x2(
                                    (acc_vec_up[i], acc_vec_up[i + 1]),
                                    (
                                        cutlass.Float32(alpha_val),
                                        cutlass.Float32(alpha_val),
                                    ),
                                )
                                acc_vec_gate_alpha = cute.arch.mul_packed_f32x2(
                                    (acc_vec_gate[i], acc_vec_gate[i + 1]),
                                    (
                                        cutlass.Float32(alpha_val),
                                        cutlass.Float32(alpha_val),
                                    ),
                                )
                                if cutlass.const_expr(self.has_swiglu_limit):
                                    acc_vec_gate_alpha = (
                                        fmin(
                                            acc_vec_gate_alpha[0], self.swiglu_limit
                                        ),
                                        fmin(
                                            acc_vec_gate_alpha[1], self.swiglu_limit
                                        ),
                                    )
                                    acc_vec_up_alpha = (
                                        fclip_xorsign(
                                            acc_vec_up_alpha[0], self.swiglu_limit
                                        ),
                                        fclip_xorsign(
                                            acc_vec_up_alpha[1], self.swiglu_limit
                                        ),
                                    )
                                tCompute_log2e = cute.arch.mul_packed_f32x2(
                                    (acc_vec_gate_alpha[0], acc_vec_gate_alpha[1]),
                                    (-LOG2_E, -LOG2_E),
                                )
                                (
                                    tCompute[i],
                                    tCompute[i + 1],
                                ) = cute.arch.add_packed_f32x2(
                                    (
                                        cute.math.exp2(
                                            tCompute_log2e[0], fastmath=True
                                        ),
                                        cute.math.exp2(
                                            tCompute_log2e[1], fastmath=True
                                        ),
                                    ),
                                    (1.0, 1.0),
                                )
                                tCompute[i] = cute.arch.rcp_approx(tCompute[i])
                                tCompute[i + 1] = cute.arch.rcp_approx(tCompute[i + 1])
                                (
                                    tCompute[i],
                                    tCompute[i + 1],
                                ) = cute.arch.mul_packed_f32x2(
                                    (tCompute[i], tCompute[i + 1]),
                                    (acc_vec_gate_alpha[0], acc_vec_gate_alpha[1]),
                                )
                                (
                                    tCompute[i],
                                    tCompute[i + 1],
                                ) = cute.arch.mul_packed_f32x2(
                                    (tCompute[i], tCompute[i + 1]),
                                    (acc_vec_up_alpha[0], acc_vec_up_alpha[1]),
                                )
                        else:
                            # SwiGLU Unpacked Version: scalar operations
                            # Computes: output = (alpha * up) * silu(alpha * gate)
                            for i in cutlass.range_constexpr(cute.size(tFC1TR_rAcc_up)):
                                acc_vec_up_alpha = acc_vec_up[i] * cutlass.Float32(
                                    alpha_val
                                )
                                acc_vec_gate_alpha = acc_vec_gate[i] * cutlass.Float32(
                                    alpha_val
                                )
                                if cutlass.const_expr(self.has_swiglu_limit):
                                    acc_vec_gate_alpha = fmin(
                                        acc_vec_gate_alpha, self.swiglu_limit
                                    )
                                    acc_vec_up_alpha = fclip_xorsign(
                                        acc_vec_up_alpha, self.swiglu_limit
                                    )
                                tCompute[i] = acc_vec_up_alpha * silu_f32(
                                    acc_vec_gate_alpha, fastmath=True
                                )

                        if cutlass.const_expr(self.fc1_generate_sfc):
                            # Float4E2M1FN quantization: per-vector absmax →
                            # SFC → store SFC to gmem → quantize output by
                            # reciprocal of SFC. (Subtile is partitioned on N.)
                            sfc_subtile_idx_mn = (
                                tile_info[0] * self.fc1_epi_tile_cnt[0] + epi_m_idx,
                                tile_info[1] * self.fc1_epi_tile_cnt[1]
                                + real_out_n_idx,
                            )
                            tCgFC1SFC = tCgFC1SFC_mn[
                                (
                                    None,
                                    None,
                                    None,
                                    *sfc_subtile_idx_mn,
                                )
                            ]

                            #
                            # Get absolute max across a vector and Compute SFC
                            #
                            tFC1TR_rAcc_frg = cute.logical_divide(
                                tCompute, cute.make_layout(self.sf_vec_size)
                            )
                            acc_frg = tFC1TR_rAcc_frg.load()
                            acc_frg = epilogue_op(acc_frg)

                            # Apply element-wise absolute value using math.absf (supports vectors)
                            abs_acc_frg_ir = math.absf(acc_frg.ir_value())
                            abs_acc_frg = type(acc_frg)(
                                abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype
                            )

                            if cutlass.const_expr(self.vectorized_f32):
                                for vi in cutlass.range_constexpr(
                                    abs_acc_frg.shape[1]
                                ):
                                    tCrFC1SFC_pvscale[vi] = abs_acc_frg[
                                        None, vi
                                    ].reduce(
                                        cute.ReductionOp.MAX,
                                        cutlass.Float32(0.0),
                                        0,
                                    )
                                for vi in cutlass.range_constexpr(
                                    0, abs_acc_frg.shape[1], 2
                                ):
                                    tCrFC1SFC_pvscale[vi], tCrFC1SFC_pvscale[vi + 1] = (
                                        cute.arch.mul_packed_f32x2(
                                            (
                                                tCrFC1SFC_pvscale[vi],
                                                tCrFC1SFC_pvscale[vi + 1],
                                            ),
                                            (
                                                self.get_dtype_rcp_limits(
                                                    self.fc1_c_dtype
                                                ),
                                                self.get_dtype_rcp_limits(
                                                    self.fc1_c_dtype
                                                ),
                                            ),
                                        )
                                    )
                                    tCrFC1SFC_pvscale[vi], tCrFC1SFC_pvscale[vi + 1] = (
                                        cute.arch.mul_packed_f32x2(
                                            (
                                                tCrFC1SFC_pvscale[vi],
                                                tCrFC1SFC_pvscale[vi + 1],
                                            ),
                                            (norm_const, norm_const),
                                        )
                                    )
                            else:
                                for vi in cutlass.range_constexpr(abs_acc_frg.shape[1]):
                                    tCrFC1SFC_pvscale[vi] = (
                                        abs_acc_frg[None, vi].reduce(
                                            cute.ReductionOp.MAX,
                                            cutlass.Float32(0.0),
                                            0,  # Use 0.0 as init for abs values
                                        )
                                        * self.get_dtype_rcp_limits(self.fc1_c_dtype)
                                        * norm_const
                                    )

                            # TODO: f32x2 -> f8x2 conversion
                            tCrFC1SFC.store(tCrFC1SFC_pvscale.load().to(self.sf_dtype))

                            # Store SFC to gmem.
                            # TODO: predicate (cute.elem_less)
                            cute.autovec_copy(tCrFC1SFC, tCgFC1SFC)

                            # Quantize output and convert to c_dtype.
                            # TODO: need to add f8x2 -> f32x2 conversion
                            tCrFC1SFC_qpvscale_up = tCrFC1SFC.load().to(cutlass.Float32)
                            fp32_max = cutlass.Float32(3.40282346638528859812e38)
                            if cutlass.const_expr(self.vectorized_f32):
                                for vi in cutlass.range_constexpr(
                                    0, cute.size(tCrFC1SFC), 2
                                ):
                                    acc_scale = cute.arch.mul_packed_f32x2(
                                        (
                                            cute.arch.rcp_approx(
                                                tCrFC1SFC_qpvscale_up[vi]
                                            ),
                                            cute.arch.rcp_approx(
                                                tCrFC1SFC_qpvscale_up[vi + 1]
                                            ),
                                        ),
                                        (norm_const, norm_const),
                                    )
                                    acc_scale_min0 = fmin(
                                        acc_scale[0], fp32_max, nan=True
                                    )
                                    acc_scale_min1 = fmin(
                                        acc_scale[1], fp32_max, nan=True
                                    )

                                    vec0 = tFC1TR_rAcc_frg[None, vi]
                                    vec1 = tFC1TR_rAcc_frg[None, vi + 1]
                                    for ei in cutlass.range_constexpr(self.sf_vec_size):
                                        vec0[ei], vec1[ei] = cute.arch.mul_packed_f32x2(
                                            (vec0[ei], vec1[ei]),
                                            (acc_scale_min0, acc_scale_min1),
                                        )
                            else:
                                for vi in cutlass.range_constexpr(cute.size(tCrFC1SFC)):
                                    # TODO:Need to add E8M0 rcp approximation
                                    acc_scale = norm_const * cute.arch.rcp_approx(
                                        tCrFC1SFC_qpvscale_up[vi]
                                    )
                                    acc_scale = fmin(acc_scale, fp32_max, nan=True)

                                    vec = tFC1TR_rAcc_frg[None, vi]
                                    for ei in cutlass.range_constexpr(self.sf_vec_size):
                                        vec[ei] = vec[ei] * acc_scale

                            acc_vec = tiled_copy_r2s.retile(tCompute).load()
                            tFC1RS_rC.store(acc_vec.to(self.fc1_c_dtype))
                        else:
                            #
                            # Convert to C type
                            #
                            acc_vec = tiled_copy_r2s.retile(tCompute).load()
                            acc_vec = epilogue_op(acc_vec.to(self.fc1_c_dtype))
                            tFC1RS_rC.store(acc_vec)

                        #
                        # Store C to shared memory
                        #
                        num_prev_subtiles = num_prev_subtiles + 1
                        c_buffer = num_prev_subtiles % self.fc1_num_c_stage

                        cute.copy(
                            tiled_copy_r2s,
                            tFC1RS_rC,
                            tFC1RS_sC[(None, None, None, c_buffer)],
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        self.epilog_sync_barrier.arrive_and_wait()
                        #
                        # TMA store C to global memory
                        #
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_fc1_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, epi_m_idx, real_out_n_idx)],
                            )
                            # Fence and barrier to make sure shared memory store is visible to TMA store
                            fc1_c_pipeline.producer_commit()
                            fc1_c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()

                # A readiness release must order the global intermediate, not
                # merely permit sFC1C reuse. Therefore wait for full TMA S2G
                # completion (no read=True), synchronize the epilogue warps,
                # then let one elected lane increment this M tile's counter.
                cute.arch.cp_async_bulk_wait_group(0)
                self.epilog_sync_barrier.arrive_and_wait()
                if warp_idx == self.epilog_warp_id[0]:
                    with cute.arch.elect_one():
                        red_add_release_gpu(
                            fc1_ready,
                            mma_tile_coord_mnl[0],
                        )

                #
                # Async arrive accumulator buffer empty.
                #
                acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
                tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_tile = (tile_info[3] == 1) and (
                    tile_info[5] == FC1_PHASE
                )
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            is_valid_fc2_tile = tile_info[3] == 1
            while is_valid_fc2_tile:
                tile_m_start = tile_info[0] * self.cta_tile_shape_mnk[0]
                meta_pipeline.consumer_wait(meta_consumer_state)
                meta_stage = meta_consumer_state.index

                acc_stage_index = acc_consumer_state.index
                tFC2TR_tAcc = tFC2TR_tAcc_base[
                    (None, None, None, None, None, acc_stage_index)
                ]
                acc_pipeline.consumer_wait(acc_consumer_state)

                tFC2TR_tAcc = cute.group_modes(
                    tFC2TR_tAcc, 3, cute.rank(tFC2TR_tAcc)
                )
                tFC2RS_sC_grouped = cute.group_modes(tFC2RS_sC, 1, 3)
                m_iter_cnt = cute.size(tFC2TR_tAcc.shape[3], mode=[0])
                n_subtile_cnt = cute.size(tFC2TR_tAcc.shape[3], mode=[1])

                for m_iter_idx in cutlass.range(m_iter_cnt):
                    permuted_row = (
                        tile_m_start
                        + m_iter_idx * self.cta_tile_shape_mnk[0]
                        + epi_tidx
                    )
                    is_valid_row = permuted_row < tile_info[4]
                    meta_row = (
                        m_iter_idx * self.cta_tile_shape_mnk[0] + epi_tidx
                    )
                    combined_scale = sFC2MetaScale[(meta_row, meta_stage)]

                    for n_subtile_idx in cutlass.range(n_subtile_cnt):
                        tFC2TR_tAcc_mn = tFC2TR_tAcc[
                            (
                                None,
                                None,
                                None,
                                (m_iter_idx, n_subtile_idx),
                            )
                        ]
                        cute.copy(
                            fc2_tiled_copy_t2r,
                            tFC2TR_tAcc_mn,
                            tFC2TR_rAcc,
                        )
                        fc2_acc_vec = fc2_tiled_copy_r2s.retile(
                            tFC2TR_rAcc
                        ).load()
                        tFC2RS_rC.store(
                            (combined_scale * fc2_acc_vec).to(
                                self.fc2_c_dtype
                            )
                        )
                        if is_valid_row:
                            cute.copy(
                                fc2_tiled_copy_r2s,
                                tFC2RS_rC[None, 0, 0],
                                tFC2RS_sC_grouped[
                                    (
                                        None,
                                        (m_iter_idx, n_subtile_idx),
                                        0,
                                    )
                                ],
                            )
                        cute.arch.fence_proxy("async.shared", space="cta")

                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                for m_iter_idx in cutlass.range(m_iter_cnt):
                    permuted_row = (
                        tile_m_start
                        + m_iter_idx * self.cta_tile_shape_mnk[0]
                        + epi_tidx
                    )
                    is_valid_row = permuted_row < tile_info[4]
                    if is_valid_row:
                        meta_row = (
                            m_iter_idx * self.cta_tile_shape_mnk[0]
                            + epi_tidx
                        )
                        token_idx = sFC2MetaTokenIdx[(meta_row, meta_stage)]
                        coord_n = tile_info[1] * self.cta_tile_shape_mnk[1]
                        scatter_out = cute.domain_offset(
                            (token_idx, coord_n, 0), mFC2C_mnl
                        )
                        smem_row = (
                            m_iter_idx * self.cta_tile_shape_mnk[0]
                            + epi_tidx
                        )
                        blk_reduce_bf16(
                            scatter_out,
                            sFC2C[(smem_row, None, 0)],
                            cutlass.Int32(self.fc2_c_copy_size),
                        )

                # The next FC2 tile may reuse sFC2C once the async reduce has
                # finished reading it. Global scatter-add completion is
                # drained once at kernel exit.
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                self.epilog_sync_barrier.arrive_and_wait()

                meta_pipeline.consumer_release(meta_consumer_state)
                meta_consumer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_fc2_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            #
            # Wait for C store complete
            #
            cute.arch.cp_async_bulk_wait_group(0)

        cute.arch.griddepcontrol_launch_dependents()

    def fc2_epilogue_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        tCgFC2C: cute.Tensor,
        fc2_epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Partition FC2 TMEM accumulators for the BF16 finalize epilogue."""
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.fc2_c_layout,
            self.fc2_c_dtype,
            self.acc_dtype,
            fc2_epi_tile,
            use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(tAcc, fc2_epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tFC2TR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        tCgFC2C_epi = cute.flat_divide(tCgFC2C, fc2_epi_tile)
        tFC2TR_gC = thr_copy_t2r.partition_D(tCgFC2C_epi)
        tFC2TR_rAcc = cute.make_rmem_tensor(
            tFC2TR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape,
            self.acc_dtype,
        )
        return tiled_copy_t2r, tFC2TR_tAcc, tFC2TR_rAcc

    def fc1_epilogue_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gFC1C_mnl: cute.Tensor,
        fc1_epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory
        (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gFC1C_mnl: The global tensor C
        :type gFC1C_mnl: cute.Tensor
        :param fc1_epi_tile: The epilogue tiler
        :type fc1_epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing tiled_copy_t2r, tFC1TR_tAcc,
            tFC1TR_rAcc_up, and tFC1TR_rAcc_gate, where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tFC1TR_tAcc: The partitioned accumulator tensor
            - tFC1TR_rAcc_up: The partitioned accumulator tensor for acc up
            - tFC1TR_rAcc_gate: The partitioned accumulator tensor for acc gate
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load (Rubin uses transformed layout)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.fc1_c_layout,
            self.fc1_c_dtype,
            self.acc_dtype,
            fc1_epi_tile,
            use_2cta_instrs,
        )

        # tAcc is already transformed: (M, N, STAGE) layout
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc,
            fc1_epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE)
        tFC1TR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # gFC1C_mnl is already transformed: (M, N_half, loopM, loopN, loopL)
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gFC1C_mnl_epi = cute.flat_divide(gFC1C_mnl, fc1_epi_tile)

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        tFC1TR_gC = thr_copy_t2r.partition_D(gFC1C_mnl_epi)

        # (T2R, T2R_M, T2R_N)
        tFC1TR_rAcc_up = cute.make_rmem_tensor(
            tFC1TR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        # (T2R, T2R_M, T2R_N)
        tFC1TR_rAcc_gate = cute.make_rmem_tensor(
            tFC1TR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tFC1TR_tAcc, tFC1TR_rAcc_up, tFC1TR_rAcc_gate

    def fc1_epilogue_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tFC1TR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sFC1C: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register
        array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tFC1TR_rC: The partitioned accumulator tensor
        :type tFC1TR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sFC1C: The shared memory tensor to be copied and partitioned
        :type sFC1C: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tFC1RS_rC, tFC1RS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tFC1RS_rC: The partitioned tensor C (register source)
            - tFC1RS_sC: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.fc1_c_layout, self.fc1_c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tFC1RS_sC = thr_copy_r2s.partition_D(sFC1C)
        # (R2S, R2S_M, R2S_N)
        tFC1RS_rC = tiled_copy_r2s.retile(tFC1TR_rC)
        return tiled_copy_r2s, tFC1RS_rC, tFC1RS_sC

    def fc1_epilogue_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gFC1C_mnl: cute.Tensor,
        fc1_epi_tile: cute.Tile,
        sFC1C: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        - partition register array (source) and global memory (destination) for none TMA store version;
        - partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gFC1C_mnl: The global tensor C
        :type gFC1C_mnl: cute.Tensor
        :param fc1_epi_tile: The epilogue tiler
        :type fc1_epi_tile: cute.Tile
        :param sFC1C: The shared memory tensor to be copied and partitioned
        :type sFC1C: cute.Tensor

        :return: A tuple containing :
            - For TMA store: (tma_atom_fc1_c, bSG_sC, bSG_gC) where:
                - tma_atom_fc1_c: The TMA copy atom
                - bSG_sC: The partitioned shared memory tensor C
                - bSG_gC: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # gFC1C_mnl is already transformed: (M, N_half, loopM, loopN, loopL)
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gFC1C_epi = cute.flat_divide(gFC1C_mnl, fc1_epi_tile)
        tma_atom_fc1_c = atom
        sFC1C_for_tma_partition = cute.group_modes(sFC1C, 0, 2)
        gFC1C_for_tma_partition = cute.group_modes(gFC1C_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, loopM, loopN, loopL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_fc1_c,
            0,
            cute.make_layout(1),
            sFC1C_for_tma_partition,
            gFC1C_for_tma_partition,
        )
        return tma_atom_fc1_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        cta_tile_shape_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        fc1_epi_tile: cute.Tile,
        fc1_c_dtype: Type[cutlass.Numeric],
        fc1_c_layout: cutlass.tensor_utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        num_smem_capacity: int,
        occupancy: int,
        with_breuse: bool,
        fc2_c_smem_bytes: int,
        fc2_metadata_smem_bytes: int,
        fc2_num_meta_stage: int,
        buffer_align_bytes: int,
        mma_cta_group_size: int,
    ) -> Tuple[int, int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Shared data type of the FC1/FC2 A operands.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Shared data type of the FC1/FC2 B operands.
        :type b_dtype: type[cutlass.Numeric]
        :param fc1_epi_tile: The epilogue tile shape.
        :type fc1_epi_tile: cute.Tile
        :param fc1_c_dtype: Data type of FC1 operand C (output).
        :type fc1_c_dtype: type[cutlass.Numeric]
        :param fc1_c_layout: Layout of FC1 operand C.
        :type fc1_c_layout: cutlass.tensor_utils.LayoutEnum
        :param sf_dtype: Data type of scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Vector size of scale factor.
        :type sf_vec_size: int
        :param num_smem_capacity: Total available shared memory capacity in bytes.
        :type num_smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int
        :param with_breuse: Whether the MMA tiler reuses B across M tiles.
        :type with_breuse: bool
        :param fc2_c_smem_bytes: Bytes required by the FC2 BF16 epilogue view.
        :type fc2_c_smem_bytes: int
        :param fc2_metadata_smem_bytes: Bytes for token-index and scale metadata.
        :type fc2_metadata_smem_bytes: int
        :param fc2_num_meta_stage: Number of FC2 metadata pipeline stages.
        :type fc2_num_meta_stage: int
        :param buffer_align_bytes: Alignment of each major shared-memory buffer.
        :type buffer_align_bytes: int
        :param mma_cta_group_size: Number of CTAs participating in one MMA.
        :type mma_cta_group_size: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages, tile-info stages)
        :rtype: tuple[int, int, int, int]
        """
        # Default ACC stages
        num_acc_stage = 1 if (with_breuse and mma_tiler_mnk[1] in {192, 256}) else 2

        # Default C stages
        fc1_num_c_stage = 2

        # Default Tile info stages
        num_tile_stage = 2

        # Calculate smem layout and size for one stage of A, B, and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )

        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )
        fc2_sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,
        )

        fc1_c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            fc1_c_dtype,
            fc1_c_layout,
            fc1_epi_tile,
            1,
        )

        # SFA SMEM is plain linear (M, tile_K_sf), no pad.
        # Per stage = cta_tile_M × tile_K_sf bytes (FP8 = 1 byte/element).
        sfa_tile_k_sf = cta_tile_shape_mnk[2] // sf_vec_size
        sf_bytes_per_row = sfa_tile_k_sf * sf_dtype.width // 8
        fc1_sfa_bytes_per_stage_one = cta_tile_shape_mnk[0] * sf_bytes_per_row
        sfa_bytes_per_stage_one = max(
            fc1_sfa_bytes_per_stage_one,
            cute.size_in_bytes(sf_dtype, fc2_sfa_smem_layout_staged_one),
        )

        a_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one
        )
        b_bytes_per_stage = cute.size_in_bytes(
            b_dtype, b_smem_layout_staged_one
        )
        sfb_bytes_per_stage = cute.size_in_bytes(
            sf_dtype, sfb_smem_layout_staged_one
        )
        ab_bytes_per_stage = (
            a_bytes_per_stage
            + b_bytes_per_stage
            + sfa_bytes_per_stage_one
            + sfb_bytes_per_stage
        )
        fc1_c_bytes_per_stage = cute.size_in_bytes(
            fc1_c_dtype, fc1_c_smem_layout_staged_one
        )
        c_bytes = fc1_c_bytes_per_stage * fc1_num_c_stage

        # Preserve the standalone FC1 heuristic as the preferred stage pair.
        # The fused selector below then reduces C first within each AB depth,
        # followed by AB depth, until the actual aligned fused storage fits.
        aligned_header_bytes = buffer_align_bytes
        preferred_num_ab_stage = (
            num_smem_capacity // occupancy - (aligned_header_bytes + c_bytes)
        ) // ab_bytes_per_stage
        preferred_fc1_num_c_stage = fc1_num_c_stage + (
            num_smem_capacity
            - occupancy * ab_bytes_per_stage * preferred_num_ab_stage
            - occupancy * (aligned_header_bytes + c_bytes)
        ) // (occupancy * fc1_c_bytes_per_stage)

        int32_bytes = cutlass.Int32.width // 8
        int64_bytes = cutlass.Int64.width // 8
        # Every pipeline owns one full/empty mbarrier pair per stage. In
        # addition to FC1 A/B and FC2 A/B, 2CTA owns an A/SFA relay pipeline.
        num_ab_mbarrier_arrays = derive_fused_ab_mbarrier_array_count(
            mma_cta_group_size
        )
        header_per_ab_stage = num_ab_mbarrier_arrays * 2 * int64_bytes
        scheduler_control_bytes = (
            # FC1 and FC2 each own full/empty mbarriers plus one Int64 response.
            2 * (2 * int64_bytes + int64_bytes)
            # One full/empty throttle mbarrier pair is shared by both phases.
            + 2 * int64_bytes
        )
        header_fixed = (
            6 * num_tile_stage * int32_bytes
            + num_acc_stage * 2 * int64_bytes
            + num_tile_stage * 2 * int64_bytes
            + fc2_num_meta_stage * 2 * int64_bytes
            + int64_bytes
            + int32_bytes
            + scheduler_control_bytes
        )
        stage_bytes = FusedSmemStageBytes(
            a_per_ab_stage=a_bytes_per_stage,
            b_per_ab_stage=b_bytes_per_stage,
            sfa_per_ab_stage=sfa_bytes_per_stage_one,
            sfb_per_ab_stage=sfb_bytes_per_stage,
            fc1_c_per_stage=fc1_c_bytes_per_stage,
            fc2_c=fc2_c_smem_bytes,
            metadata=fc2_metadata_smem_bytes,
            header_fixed=header_fixed,
            header_per_ab_stage=header_per_ab_stage,
            alignment=buffer_align_bytes,
        )
        selected = select_fused_smem_stages(
            capacity=num_smem_capacity // occupancy,
            preferred_ab=preferred_num_ab_stage,
            preferred_fc1_c=preferred_fc1_num_c_stage,
            minimum_fc1_c=fc1_num_c_stage,
            stage_bytes=stage_bytes,
        )
        return num_acc_stage, selected.ab, selected.fc1_c, num_tile_stage

    @staticmethod
    def _compute_tile_sched_params(
        gemm_shape: Tuple[int, int, int],
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> utils.PersistentTileSchedulerParams:
        """Build AlongN persistent scheduler parameters for one logical GEMM."""
        m, n, l = gemm_shape
        num_ctas_mnl = (
            cute.ceil_div(m, cta_tile_shape_mnk[0]),
            cute.ceil_div(n, cta_tile_shape_mnk[1]),
            l,
        )
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        return utils.PersistentTileSchedulerParams(
            num_ctas_mnl,
            cluster_shape_mnl,
            raster_along_m=False,
        )

    @staticmethod
    def _get_tma_atom_kind(
        atom_sm_cnt: cutlass.Int32, mcast: cutlass.Boolean
    ) -> Union[
        cpasync.CopyBulkTensorTileG2SMulticastOp, cpasync.CopyBulkTensorTileG2SOp
    ]:
        """
        Select the appropriate TMA copy atom based on the number of SMs and the multicast flag.

        :param atom_sm_cnt: The number of SMs
        :type atom_sm_cnt: cutlass.Int32
        :param mcast: The multicast flag
        :type mcast: cutlass.Boolean

        :return: The appropriate TMA copy atom kind
        :rtype: cpasync.CopyBulkTensorTileG2SMulticastOp or cpasync.CopyBulkTensorTileG2SOp

        :raise ValueError: If the atom_sm_cnt is invalid
        """
        if atom_sm_cnt == 2 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 2 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 1 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
        elif atom_sm_cnt == 1 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

        raise ValueError(f"Invalid atom_sm_cnt: {atom_sm_cnt} and {mcast}")

    @staticmethod
    def get_dtype_rcp_limits(dtype: Type[cutlass.Numeric]) -> float:
        """
        Calculates the reciprocal of the maximum absolute value for a given data type.

        :param dtype: Data type
        :type dtype: Type[cutlass.Numeric]

        :return: An float representing the reciprocal of the maximum absolute value
        :rtype: float
        """
        if dtype == cutlass.Float4E2M1FN:
            return 1 / 6.0
        if dtype == cutlass.Float8E4M3FN:
            return 1 / 448.0
        if dtype == cutlass.Float8E5M2:
            return 1 / 128.0
        return 1.0

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        fc1_c_dtype: Type[cutlass.Numeric],
        fc2_c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """Return whether both fused phases use the supported NVFP4 types."""
        return (
            a_dtype == cutlass.Float4E2M1FN
            and b_dtype == cutlass.Float4E2M1FN
            and sf_dtype == cutlass.Float8E4M3FN
            and sf_vec_size == 16
            and fc1_c_dtype == a_dtype
            and fc2_c_dtype == cutlass.BFloat16
        )

    @staticmethod
    def is_valid_layouts(
        a_major: str,
        b_major: str,
        fc1_c_major: str,
        fc2_c_major: str,
    ) -> bool:
        """Return whether operands use the layouts implemented by fused FC12."""
        return (
            a_major == "k"
            and b_major == "k"
            and fc1_c_major == "n"
            and fc2_c_major == "n"
        )

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_inst_shape: tuple[int, int, int],
        mma_tiler: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
    ) -> bool:
        """Return whether the MMA and cluster geometry has a kernel path."""
        if (
            len(mma_inst_shape) != 3
            or len(mma_tiler) != 3
            or len(cluster_shape_mn) != 2
        ):
            return False

        inst_m, inst_n, inst_k = mma_inst_shape
        tile_m, tile_n, tile_k = mma_tiler
        cluster_m, cluster_n = cluster_shape_mn
        if inst_m not in (128, 256) or inst_n not in (128, 256):
            return False
        if inst_k != 128 or tile_k != 256:
            return False
        # The 128dp FC1 path does not currently implement B-reuse. Keep the
        # unsupported two-instruction-M geometry out of compilation until its
        # MMA and SFA TMEM path is restored and verified.
        if tile_m != inst_m or tile_n != inst_n:
            return False

        mma_cta_group_size = inst_m // 128
        if cluster_m != mma_cta_group_size:
            return False
        if cluster_n not in (1, 2, 4):
            return False
        return cluster_m * cluster_n <= 16

    @staticmethod
    def is_valid_problem_shapes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        fc1_c_dtype: Type[cutlass.Numeric],
        fc2_c_dtype: Type[cutlass.Numeric],
        mma_tiler: tuple[int, int, int],
        fc1_gemm_shape: tuple[int, int, int, int],
        fc2_gemm_shape: tuple[int, int, int, int],
        a_major: str,
        b_major: str,
        fc1_c_major: str,
        fc2_c_major: str,
    ) -> bool:
        """Validate fused dimension relationships, tiling, and 16-byte alignment."""
        if len(fc1_gemm_shape) != 4 or len(fc2_gemm_shape) != 4:
            return False
        if any(dim <= 0 for dim in (*fc1_gemm_shape, *fc2_gemm_shape)):
            return False

        fc1_m, fc1_n, fc1_k, fc1_l = fc1_gemm_shape
        fc2_m, fc2_n, fc2_k, fc2_l = fc2_gemm_shape
        tile_m, tile_n, tile_k = mma_tiler
        if fc1_m != fc2_m or fc1_l != fc2_l:
            return False
        if fc1_n != 2 * fc2_k or fc1_k != fc2_n:
            return False
        if (
            fc1_m % tile_m != 0
            or fc1_n % tile_n != 0
            or fc2_n % tile_n != 0
            or fc1_k % tile_k != 0
            or fc2_k % tile_k != 0
        ):
            return False

        def is_contiguous_16b_aligned(
            dtype: Type[cutlass.Numeric],
            major: str,
            shape: tuple[int, int, int],
        ) -> bool:
            major_mode = 0 if major == "m" else 1
            contiguous_elements = 16 * 8 // dtype.width
            return shape[major_mode] % contiguous_elements == 0

        return (
            is_contiguous_16b_aligned(
                a_dtype, a_major, (fc1_m, fc1_k, fc1_l)
            )
            and is_contiguous_16b_aligned(
                b_dtype, b_major, (fc1_n, fc1_k, fc1_l)
            )
            and is_contiguous_16b_aligned(
                fc1_c_dtype, fc1_c_major, (fc1_m, fc2_k, fc1_l)
            )
            and is_contiguous_16b_aligned(
                b_dtype, b_major, (fc2_n, fc2_k, fc2_l)
            )
            and is_contiguous_16b_aligned(
                fc2_c_dtype, fc2_c_major, (fc2_m, fc2_n, 1)
            )
        )

    @classmethod
    def can_implement(
        cls,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        fc1_c_dtype: Type[cutlass.Numeric],
        fc2_c_dtype: Type[cutlass.Numeric],
        mma_inst_shape: tuple[int, int, int],
        mma_tiler: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        fc1_gemm_shape: tuple[int, int, int, int],
        fc2_gemm_shape: tuple[int, int, int, int],
        a_major: str,
        b_major: str,
        fc1_c_major: str,
        fc2_c_major: str,
    ) -> bool:
        """Return whether a complete fused FC1+FC2 problem is supported."""
        if not cls.is_valid_dtypes_and_scale_factor_vec_size(
            a_dtype,
            b_dtype,
            sf_dtype,
            sf_vec_size,
            fc1_c_dtype,
            fc2_c_dtype,
        ):
            return False
        if not cls.is_valid_layouts(
            a_major, b_major, fc1_c_major, fc2_c_major
        ):
            return False
        if not cls.is_valid_mma_tiler_and_cluster_shape(
            mma_inst_shape, mma_tiler, cluster_shape_mn
        ):
            return False
        return cls.is_valid_problem_shapes(
            a_dtype,
            b_dtype,
            fc1_c_dtype,
            fc2_c_dtype,
            mma_tiler,
            fc1_gemm_shape,
            fc2_gemm_shape,
            a_major,
            b_major,
            fc1_c_major,
            fc2_c_major,
        )
