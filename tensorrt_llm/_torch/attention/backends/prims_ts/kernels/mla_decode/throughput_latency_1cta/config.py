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

"""Configuration traits for the throughput-latency 1CTA MLA TS path.

This module contains Python-side trait state only. Shape policy is kept in the
explicit kernel selector so the executable schedule stays focused on one
concrete profile at a time.

``make_throughput_latency_mla_config`` is the public factory: it validates user shapes,
page size, and explicit profile values before deriving task, pipeline, and
workspace traits. Invalid inputs fail with ``ValueError`` rather than falling
through to DSL division or GMEM reduction layout errors.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Any

from ....split_kv_mode_policy import select_split_kv_modes
from ..helpers.constants import MAX_MLA_SPLITS_KV, SUPPORTED_MLA_PAGE_SIZES
from ..helpers.mask import MaskType, normalize_mask_type
from ..helpers.query import FlatQueryTileLayout


# 1CTA profiles are specialized for these Q/head tiles.  Tile sizes 8/16 use
# the swaps-MMA-AB schedule; tile sizes 32/64 are supported for explicit or
# keeps-MMA-AB profiles.
SUPPORTED_TILE_SIZE_Q = (8, 16, 32, 64)

# SM100A exposes 227 KiB of dynamic SMEM.  Keep a small TS metadata reservation
# when deciding whether the automatic cluster-reduction scratch can fit.
SM100A_SMEM_CAPACITY_BYTES = 232448
ESTIMATED_TS_BARRIER_BYTES = 512
MAX_CLUSTER_SMEM_DATA_BYTES = SM100A_SMEM_CAPACITY_BYTES - ESTIMATED_TS_BARRIER_BYTES
# CUDA's nonportable cluster launch attribute accepts at most 32 CTAs.  This is
# a launch-geometry limit, not a problem-shape policy.
MAX_CLUSTER_SIZE = 32


def align_up(value: int, alignment: int) -> int:
    """Return ``value`` rounded up to ``alignment`` bytes."""
    return ((value + alignment - 1) // alignment) * alignment


def estimated_throughput_latency_smem_data_bytes(cfg: "MlaConfig") -> int:
    """Estimate steady SMEM data allocation for the 1CTA resource set."""
    stensor = cfg.stensor_align
    total = 0
    total += align_up(cfg.smem_q_tile_bytes * cfg.q_stages, stensor)
    total += align_up(cfg.smem_kv_tile_bytes * cfg.kv_stages, stensor)
    total += align_up(
        cfg.page_offsets_stages * cfg.page_offsets_entries_per_stage * 4,
        128,
    )
    total += 2 * align_up(cfg.p_smem_tile_bytes, stensor)
    total += align_up(16, 8)  # P0/P1 ordering barrier.
    total += 2 * align_up(cfg.softmax_scratch_bytes, 16)
    total += align_up(cfg.o_smem_tile_bytes, stensor)
    total += align_up(cfg.corr_scratch_bytes, 16)
    if cfg.cluster_reduction_smem_bytes:
        total += align_up(cfg.cluster_reduction_smem_bytes, 64)
        total += align_up(8, 8)
    return total


@dataclass(frozen=True)
class MlaConfig:
    """Kernel traits for one throughput-latency 1CTA MLA schedule.

    The dataclass groups shape dimensions, tile sizes, CTA decomposition,
    resource capacities, warp/register budgets, and feature switches consumed by
    the captured task graph. Fields are plain Python values so they can be passed
    as ``Constexpr`` into JIT resources and tasks.
    """

    # DS MLA dimensions.  Dense MLA uses 512 latent channels plus 64 RoPE
    # channels, so the QK head dimension defaults to 576.
    batch_size: int = 1
    num_heads_q: int = 16
    seq_len_q: int = 4
    seq_len_kv: int = 4096
    logical_num_heads_q: int = 16
    logical_seq_len_q: int = 4
    # Causal is bottom-right aligned for speculative decode. Dense keeps only
    # the ordinary per-batch KV-tail predicate.
    mask_type: str = MaskType.CAUSAL.value
    head_dim_qk: int = 576
    head_dim_v: int = 512
    latent_dim: int = 512
    rope_dim: int = 64

    # Data types.
    qkv_dtype: str = "bf16"
    o_dtype: str = "bf16"
    qkv_dtype_bytes: int = 2
    o_dtype_bytes: int = 2
    acc_dtype_bytes: int = 4
    use_bf16_output: int = 1
    use_fp8_output: int = 0

    # Mainloop tile shape for flat query-row decode. Each loop step consumes two
    # 128-token KV tiles: K for the current score update and V for the delayed
    # PV update.
    tile_size_q: int = 16
    tile_size_kv: int = 128
    num_insts_q: int = 1
    num_insts_kv: int = 2
    q_stages: int = 2
    kv_stages: int = 4
    # Keep page-ID stages aligned with the straight-line K/V pipeline so each
    # stage has an explicit, reusable metadata window.
    page_offsets_stages: int = 6
    # Each pipeline stage describes one KV tile. The 32-entry capacity covers
    # the smallest supported page size: 128 tokens / 16 tokens per page = 8.
    page_offsets_entries_per_stage: int = 32
    o_stages: int = 2

    # Default CTA decomposition for SM100-class decode.  A 512-thread CTA
    # provides 16 warps for the two softmax groups, correction, MMA, load,
    # scheduler, and page-offset work.
    threads_per_cta: int = 512
    num_ctas_per_seq_q: int = 4
    num_ctas_per_seq_kv: int = 1
    num_ctas_for_all_heads: int = 1
    num_ctas_per_head_dim: int = 1
    head_dim_per_cta_v: int = 512
    head_dim_per_stage_kv: int = 128
    head_dim_per_stage_v: int = 128
    num_tokens_per_page: int = 32
    max_num_pages_per_seq_kv: int = 1

    # Shared-memory and TMEM allocation traits.  STensor allocations are aligned
    # to 1 KiB, and TMEM column counts follow the tcgen05 16-column granularity
    # used for S/P/O tiles.
    stensor_align: int = 1024
    tmem_s_cols: int = 16
    tmem_stats_cols: int = 32
    tmem_o_cols: int = 16

    # Warp layout for load, MMA, softmax, correction, and scheduling work.  Warp
    # indices are CTA-local; four-warps groups are kept contiguous for
    # setmaxnreg and named-barrier participation.
    softmax0_warp_idx: int = 0
    softmax1_warp_idx: int = 4
    correction_warp_idx: int = 8
    mma_warp_idx: int = 12
    load_warp_idx: int = 15
    page_offsets_warp_idx: int = 13
    scheduler_warp_idx: int = 14
    softmax_num_warps: int = 4
    correction_num_warps: int = 4
    mma_num_warps: int = 1
    load_num_warps: int = 1
    page_offsets_num_warps: int = 1
    scheduler_num_warps: int = 1
    clc_padding_warp_idx: int = 14
    clc_padding_num_warps: int = 1

    # Register budgets used by task-local setmaxregister calls.  Softmax owns
    # the score registers, correction runs with a small budget after softmax,
    # and load/MMA/scheduler warps share the lower budget.
    softmax_regs: int = 176
    correction_regs: int = 64
    mma_load_regs: int = 96
    scheduler_regs: int = 96

    # Feature mode flags consumed by config and schedule construction.  They are
    # integer flags because many branches are passed as Constexpr values into
    # JIT code.
    kernel_variant: str = "swaps_mma_ab"
    use_paged_kv: int = 1
    supports_var_seq_lens: int = 1
    use_persistent_scheduler: int = 1
    use_clc_dynamic_persistent_scheduler: int = 0
    use_multi_ctas_kv: int = 0
    use_cluster_reduction: int = 0
    persistent_wave_sm_count: int | None = None
    use_attention_sinks: int = 0
    use_sliding_window_causal: int = 0
    attention_window_size: int = 0

    @property
    def softmax0_num_warps(self) -> int:
        return self.softmax_num_warps

    @property
    def softmax1_num_warps(self) -> int:
        return self.softmax_num_warps

    @property
    def padding_warp_idx(self) -> int:
        return self.page_offsets_warp_idx

    @property
    def padding_num_warps(self) -> int:
        return 2

    @property
    def qk_smem_tile_bytes(self) -> int:
        return self.tile_size_q * self.head_dim_qk * self.qkv_dtype_bytes

    @property
    def kv_smem_tile_bytes(self) -> int:
        return self.tile_size_kv * self.head_dim_per_stage_kv * self.qkv_dtype_bytes

    @property
    def v_smem_tile_bytes(self) -> int:
        return self.tile_size_kv * self.head_dim_per_stage_v * self.qkv_dtype_bytes

    @property
    def pages_per_kv_tile(self) -> int:
        """Return physical pages consumed by one logical KV tile."""

        return ceil(self.tile_size_kv / self.num_tokens_per_page)

    @property
    def total_kv_tiles(self) -> int:
        return ceil(self.seq_len_kv / self.tile_size_kv)

    @property
    def kv_tiles_per_multi_cta_group(self) -> int:
        return self.num_ctas_per_seq_kv * self.num_insts_kv

    @property
    def num_steps_per_cta_kv(self) -> int:
        tokens_per_multi_cta_step = (
            self.num_ctas_per_seq_kv * self.num_insts_kv * self.tile_size_kv
        )
        return ceil(self.seq_len_kv / tokens_per_multi_cta_step)

    @property
    def p_smem_tile_bytes(self) -> int:
        return self.tile_size_kv * self.tile_size_q * self.qkv_dtype_bytes

    @property
    def smem_q_tile_bytes(self) -> int:
        return self.qk_smem_tile_bytes

    @property
    def smem_kv_tile_bytes(self) -> int:
        return self.kv_smem_tile_bytes

    @property
    def smem_p_tile_bytes(self) -> int:
        return self.p_smem_tile_bytes

    @property
    def partial_o_dtype_bytes(self) -> int:
        if self.num_ctas_per_seq_kv > 1:
            return 2
        return self.o_dtype_bytes

    @property
    def o_smem_tile_bytes(self) -> int:
        staging_dim = max(self.head_dim_per_stage_v, 64)
        return self.tile_size_q * staging_dim * self.partial_o_dtype_bytes

    @property
    def o_copy_segments_per_stage(self) -> int:
        bytes_per_stage = (
            self.tile_size_q * self.head_dim_per_stage_v * self.partial_o_dtype_bytes
        )
        return max(1, ceil(bytes_per_stage / 2048))

    @property
    def softmax_scratch_bytes(self) -> int:
        return 4 * self.tile_size_q * self.acc_dtype_bytes

    @property
    def corr_scratch_bytes(self) -> int:
        return 8 * self.tile_size_q * self.acc_dtype_bytes

    @property
    def cluster_reduction_rows_per_slice(self) -> int:
        # One cluster-reduction SMEM slice is one 16-row x 128B tile.  The row count
        # shrinks as the per-CTA V head dimension grows.
        num_bytes_per_slice = 128 * 16
        num_bytes_per_row_o = self.head_dim_per_cta_v * self.partial_o_dtype_bytes
        return max(1, num_bytes_per_slice // num_bytes_per_row_o)

    @property
    def cluster_reduction_slices(self) -> int:
        return ceil(self.tile_size_q / self.cluster_reduction_rows_per_slice)

    def cluster_reduction_smem_bytes_for(self, num_ctas_kv: int) -> int:
        """Return multi-CTA KV cluster reduction SMEM footprint."""
        if num_ctas_kv <= 1:
            return 0
        rows_per_slice = self.cluster_reduction_rows_per_slice
        num_slices = self.cluster_reduction_slices
        num_slices_per_cta = ceil(num_slices / num_ctas_kv)
        num_rows_per_cta = num_slices_per_cta * rows_per_slice
        num_bytes_per_row_o = self.head_dim_per_cta_v * self.partial_o_dtype_bytes
        num_bytes_per_row_stats = 2 * self.acc_dtype_bytes
        return (
            num_ctas_kv
            * num_rows_per_cta
            * (num_bytes_per_row_o + num_bytes_per_row_stats)
        )

    @property
    def cluster_reduction_smem_bytes(self) -> int:
        if self.use_multi_ctas_kv != 1 or self.use_cluster_reduction != 1:
            return 0
        max_bytes = 0
        for num_ctas_kv in range(2, self.num_ctas_per_seq_kv + 1):
            max_bytes = max(
                max_bytes, self.cluster_reduction_smem_bytes_for(num_ctas_kv)
            )
        return max_bytes

    @property
    def tmem_total_cols(self) -> int:
        return (
            2 * self.tmem_s_cols
            + 2 * self.tmem_stats_cols
            + self.tmem_o_buffer_cols * self.o_stages * self.v_head_dim_stages
        )

    @property
    def tmem_alloc_cols(self) -> int:
        # Reserve the full SM100 TMEM budget for fixed column placement.
        return 512

    @property
    def qk_head_dim_stages(self) -> int:
        return ceil(self.head_dim_qk / self.head_dim_per_stage_kv)

    @property
    def v_head_dim_stages(self) -> int:
        return ceil(self.head_dim_per_cta_v / self.head_dim_per_stage_v)

    @property
    def tmem_o_buffer_cols(self) -> int:
        return self.tmem_o_cols

    @property
    def q_smem_tile_elements(self) -> int:
        return self.qk_smem_tile_bytes // self.qkv_dtype_bytes

    @property
    def kv_smem_stage_elements(self) -> int:
        return self.kv_smem_tile_bytes // self.qkv_dtype_bytes

    def qk_head_stage_width(self, stage_idx: int) -> int:
        start = stage_idx * self.head_dim_per_stage_kv
        return max(0, min(self.head_dim_per_stage_kv, self.head_dim_qk - start))

    def v_head_stage_width(self, stage_idx: int) -> int:
        start = stage_idx * self.head_dim_per_stage_v
        return max(0, min(self.head_dim_per_stage_v, self.head_dim_per_cta_v - start))

    def is_fp8_qkv(self) -> bool:
        """Return whether Q/K/V tensors use E4M3 data."""

        return self.qkv_dtype == "e4m3"

    def local_kv_tiles(self, total_kv_tiles: int) -> int:
        """Return per-CTA KV tiles after multi-CTA KV splitting."""
        if self.use_multi_ctas_kv != 1:
            return total_kv_tiles
        tiles_per_group = self.num_ctas_per_seq_kv * self.num_insts_kv
        num_groups = (total_kv_tiles + tiles_per_group - 1) // tiles_per_group
        return max(self.num_insts_kv, num_groups * self.num_insts_kv)

    def loop_domain(self, local_kv_tiles: int) -> int:
        """Return the decode-gen steady-state loop domain after HEAD."""
        remaining_kv_tiles = max(local_kv_tiles - self.num_insts_kv, 0)
        return (remaining_kv_tiles + self.num_insts_kv - 1) // self.num_insts_kv


def compute_workspace_size(
    *,
    cfg: MlaConfig,
    partial_o_dtype,
    lse_dtype,
) -> int:
    """Return the exact 1CTA split-KV GMEM workspace size in bytes.

    The producer layout is ``[B, SQ, H, split, D]`` for partial O and
    ``[B, SQ, H, split]`` for LSE. Cluster reduction keeps these partials in
    shared memory and therefore needs no GMEM workspace.
    """

    split_kv = cfg.num_ctas_per_seq_kv
    if split_kv == 1 or cfg.use_cluster_reduction == 1:
        return 0
    partial_rows = cfg.batch_size * cfg.seq_len_q * cfg.num_heads_q * split_kv
    return partial_rows * (
        cfg.head_dim_v * partial_o_dtype.width // 8 + lse_dtype.width // 8
    )


@dataclass(frozen=True)
class MlaProfile:
    """Concrete tunable profile for one throughput-latency 1CTA MLA kernel variant.

    Profiles select only the tunable scheduler/decomposition knobs. The config
    factory validates them against the concrete shape and expands them into the
    full ``MlaConfig`` consumed by the kernel.
    """

    name: str
    num_ctas_per_seq_kv: int = 1
    num_ctas_per_head_dim: int = 1
    use_persistent_scheduler: int = 1
    use_clc_dynamic_persistent_scheduler: int = 0
    use_multi_ctas_kv: int = 0
    use_cluster_reduction: int = 0
    kernel_variant: str = "swaps_mma_ab"
    tile_size_q: int | None = None


@dataclass(frozen=True)
class FlatQueryLaunchShape:
    """Logical query geometry and its normalized physical flat-row launch."""

    logical_num_heads_q: int
    logical_seq_len_q: int
    num_heads_q: int
    seq_len_q: int
    tile_size_q: int

    @classmethod
    def for_tile(
        cls,
        logical_num_heads_q: int,
        logical_seq_len_q: int,
        tile_size_q: int,
    ) -> "FlatQueryLaunchShape":
        """Build consecutive physical M-row tiles for one logical query."""

        layout = FlatQueryTileLayout.for_tile(
            logical_num_heads_q,
            logical_seq_len_q,
            tile_size_q,
        )
        return cls(
            logical_num_heads_q=logical_num_heads_q,
            logical_seq_len_q=logical_seq_len_q,
            num_heads_q=layout.tile_size_q,
            seq_len_q=layout.num_tiles,
            tile_size_q=tile_size_q,
        )


def validate_tile_size_q(tile_size_q: int) -> int:
    """Validate a user-supplied Q tile size for 1CTA MLA."""

    if tile_size_q not in SUPPORTED_TILE_SIZE_Q:
        raise ValueError(
            "throughput-latency 1CTA MLA tile_size_q must be one of "
            f"{SUPPORTED_TILE_SIZE_Q}, got {tile_size_q}"
        )
    return tile_size_q


def tile_size_q_from_profile_name(profile: str | None) -> int | None:
    """Return the explicit Q tile encoded by profile names such as ``h32_static``."""

    if not profile:
        return None
    for tile_size_q in SUPPORTED_TILE_SIZE_Q:
        if profile.startswith(f"h{tile_size_q}_"):
            return tile_size_q
    return None


def validate_max_active_clusters(max_active_clusters: int) -> int:
    """Return the explicit hardware active-cluster capacity."""

    if max_active_clusters is None:
        raise ValueError("max_active_clusters must be provided")
    if max_active_clusters <= 0:
        raise ValueError("max_active_clusters must be positive")
    return max_active_clusters


def tile_size_q_for_heads(num_heads_q: int) -> int:
    """Choose the automatic 1CTA Q tile from the physical row extent."""

    return auto_tile_size_q_for_mla_gen(
        num_heads_q=num_heads_q,
        seq_len_q=1,
    )


def auto_tile_size_q_for_mla_gen(
    *,
    num_heads_q: int,
    seq_len_q: int,
) -> int:
    """Choose the smallest native M tile covering the flat query rows.

    Shapes wider than the largest 1CTA tile use repeated M64 tiles. The choice
    depends only on query geometry, so one compiled topology is reusable across
    runtime batch extents and K/V lengths.
    """

    total_q_rows = num_heads_q * seq_len_q
    for tile_size_q in SUPPORTED_TILE_SIZE_Q:
        if total_q_rows <= tile_size_q:
            return tile_size_q
    return SUPPORTED_TILE_SIZE_Q[-1]


def resolve_auto_flat_query_launch_shape(
    *,
    num_heads_q: int,
    seq_len_q: int,
) -> FlatQueryLaunchShape:
    """Resolve the public 1CTA profile tile and flat-row launch extent.

    Equivalent H/Q factorizations resolve to the same physical profile when
    their runtime work is otherwise identical.
    """

    tile_size_q = auto_tile_size_q_for_mla_gen(
        num_heads_q=num_heads_q,
        seq_len_q=seq_len_q,
    )
    return FlatQueryLaunchShape.for_tile(
        num_heads_q,
        seq_len_q,
        tile_size_q,
    )


def profile_name(
    base_name: str, num_heads_q: int, tile_size_q: int | None = None
) -> str:
    tile_size_q = tile_size_q or tile_size_q_for_heads(num_heads_q)
    return f"h{tile_size_q}_{base_name}"


def q_tile_work_count(
    batch_size: int,
    num_heads_q: int,
    seq_len_q: int,
    tile_size_q: int | None = None,
) -> int:
    """Return CTA work count before KV or V-head-dim decomposition."""

    tile_size_q = tile_size_q or tile_size_q_for_heads(num_heads_q)
    return batch_size * ceil(num_heads_q * seq_len_q / tile_size_q)


def automatic_split_kv_step_tokens(tile_size_q: int) -> int:
    """Return one steady-state K step for the selected task graph."""

    num_kv_insts = 1 if tile_size_q >= 64 else MlaConfig.num_insts_kv
    return MlaConfig.tile_size_kv * num_kv_insts


def select_auto_split_kv(
    *,
    seq_len_kv: int,
    tile_size_q: int,
    base_work: int,
    target_work: int,
) -> int:
    """Choose the smallest split count preserving the target K-step depth."""

    split_kv_step_tokens = automatic_split_kv_step_tokens(tile_size_q)
    max_split_kv = max(1, ceil(seq_len_kv / split_kv_step_tokens))
    split_kv = min(
        max_split_kv,
        max(1, target_work // base_work),
        MAX_MLA_SPLITS_KV,
    )
    local_k_steps = ceil(max_split_kv / split_kv)
    return ceil(max_split_kv / local_k_steps)


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def int_flag(value: bool) -> int:
    if value:
        return 1
    return 0


def dtype_num_bytes(dtype: str) -> int:
    if dtype == "e4m3":
        return 1
    return 2


def dtype_config_kwargs(qkv_dtype: str, o_dtype: str) -> dict[str, int]:
    """Return dtype byte widths and feature flags for the config dataclass."""

    kwargs = {
        "qkv_dtype_bytes": dtype_num_bytes(qkv_dtype),
        "o_dtype_bytes": dtype_num_bytes(o_dtype),
        "use_bf16_output": 0,
        "use_fp8_output": 0,
    }
    if o_dtype == "bf16":
        kwargs["use_bf16_output"] = 1
    elif o_dtype == "e4m3":
        kwargs["use_fp8_output"] = 1

    return kwargs


def tile_size_q_for_profile(
    profile: MlaProfile,
    num_heads_q: int,
    seq_len_q: int,
    tile_size_q: int | None = None,
) -> int:
    """Resolve the effective Q tile for the selected profile."""

    del seq_len_q
    if tile_size_q is not None:
        tile_size_q = validate_tile_size_q(tile_size_q)
        if profile.kernel_variant == "keeps_mma_ab" and tile_size_q < 64:
            raise ValueError("keeps_mma_ab profiles require tile_size_q >= 64")
        if profile.kernel_variant != "keeps_mma_ab" and tile_size_q >= 64:
            raise ValueError("swaps_mma_ab profiles require tile_size_q < 64")
        return tile_size_q
    if profile.tile_size_q is not None:
        profile_tile_size_q = validate_tile_size_q(profile.tile_size_q)
        if profile.kernel_variant == "keeps_mma_ab" and profile_tile_size_q < 64:
            raise ValueError("keeps_mma_ab profiles require tile_size_q >= 64")
        if profile.kernel_variant != "keeps_mma_ab" and profile_tile_size_q >= 64:
            raise ValueError("swaps_mma_ab profiles require tile_size_q < 64")
        return profile_tile_size_q
    if profile.kernel_variant == "keeps_mma_ab":
        return 64
    return tile_size_q_for_heads(num_heads_q)


def kv_stage_count(profile: MlaProfile, tile_size_q: int, qkv_dtype: str) -> int:
    if profile.use_clc_dynamic_persistent_scheduler == 1 and tile_size_q == 32:
        return 3
    if qkv_dtype == "e4m3":
        return 9
    return MlaConfig.kv_stages


def softmax_register_budget(tile_size_q: int) -> int:
    if tile_size_q == 32:
        return 160
    return MlaConfig.softmax_regs


def correction_register_budget(tile_size_q: int) -> int:
    if tile_size_q == 32:
        return 96
    return MlaConfig.correction_regs


def keeps_mma_ab_config_kwargs(profile: MlaProfile, qkv_dtype: str) -> dict[str, int]:
    """Return keeps-MMA-AB scheduler, pipeline, and register traits."""

    kv_stages = MlaConfig.kv_stages
    q_stages = 1
    if qkv_dtype == "e4m3":
        kv_stages = 8
        q_stages = MlaConfig.q_stages

    use_persistent_scheduler = profile.use_persistent_scheduler
    use_clc_dynamic_persistent_scheduler = profile.use_clc_dynamic_persistent_scheduler
    if profile.use_multi_ctas_kv == 1:
        use_persistent_scheduler = 0
        use_clc_dynamic_persistent_scheduler = 0

    return {
        # Keeps-MMA-AB has one QK/PV pipe.  Raising this value would change the
        # KV work partition without adding the second stream used by the swaps
        # schedule, and would therefore skip every other KV tile.
        "num_insts_kv": 1,
        "kv_stages": kv_stages,
        "q_stages": q_stages,
        "o_stages": 1,
        "threads_per_cta": 384,
        "correction_warp_idx": 4,
        "mma_warp_idx": 8,
        "load_warp_idx": 9,
        "page_offsets_warp_idx": 10,
        "scheduler_warp_idx": 11,
        "softmax_regs": 200,
        "correction_regs": 192,
        "mma_load_regs": 112,
        "tmem_s_cols": 128,
        "tmem_o_cols": 128,
        "use_persistent_scheduler": use_persistent_scheduler,
        "use_clc_dynamic_persistent_scheduler": use_clc_dynamic_persistent_scheduler,
    }


def resolve_split_kv_reduction_policy(
    *,
    profile: MlaProfile,
    tile_size_q: int,
    head_dim: int,
    head_dim_per_cta_v: int,
    reduction_mode: str | None,
) -> int:
    """Return whether the selected split-KV profile should use cluster reduction."""

    if reduction_mode not in (None, "auto", "cluster", "gmem_separate"):
        raise ValueError(f"unsupported reduction_mode: {reduction_mode}")
    reduction_mode = reduction_mode or "auto"

    if reduction_mode == "gmem_separate":
        return 0
    if reduction_mode == "cluster":
        if profile.use_multi_ctas_kv != 1:
            raise ValueError("explicit cluster reduction requires a split-KV profile")
        if profile.kernel_variant == "keeps_mma_ab":
            raise ValueError(
                "explicit cluster reduction is not supported by keeps-MMA-AB profiles"
            )
        if profile.num_ctas_per_seq_kv > MAX_CLUSTER_SIZE:
            raise ValueError(
                "explicit cluster reduction exceeds the CUDA cluster-size limit: "
                f"split_kv={profile.num_ctas_per_seq_kv}, "
                f"max_cluster_size={MAX_CLUSTER_SIZE}"
            )
        return 1

    cluster_capable_profile = (
        profile.use_multi_ctas_kv == 1
        and profile.use_cluster_reduction == 1
        and profile.kernel_variant != "keeps_mma_ab"
        and profile.num_ctas_per_seq_kv <= MAX_CLUSTER_SIZE
    )
    if not cluster_capable_profile:
        return 0

    modes = select_split_kv_modes(
        family="mla_decode",
        topology="1cta",
        tile_size_q=tile_size_q,
        head_dim=head_dim,
        head_dim_per_cta_v=head_dim_per_cta_v,
        split_kv=profile.num_ctas_per_seq_kv,
        available_modes=("cluster", "gmem_separate"),
    )
    return int_flag(modes[0] == "cluster")


def cluster_reduction_cluster_count(cfg: MlaConfig) -> int:
    """Return the number of clusters launched by the cluster reduction grid."""

    cluster_size = cfg.num_ctas_per_seq_kv
    grid_m = cfg.num_ctas_for_all_heads * cfg.num_ctas_per_seq_q * cluster_size
    grid_n = cfg.num_ctas_per_head_dim
    grid_l = cfg.batch_size
    return ceil(grid_m / cluster_size) * grid_n * grid_l


def cluster_reduction_cluster_shape(cfg: MlaConfig) -> tuple[int, int, int]:
    """Return the cluster reduction cluster shape for the main kernel launch."""

    return (cfg.num_ctas_per_seq_kv, 1, 1)


def resolve_auto_cluster_reduction_mode(
    cfg: MlaConfig,
    *,
    reduction_mode: str | None,
    max_active_clusters: int,
) -> str | None:
    """Return the runtime reduction mode after the auto-cluster capacity check."""

    if reduction_mode not in (None, "auto") or cfg.use_cluster_reduction != 1:
        return reduction_mode

    if cluster_reduction_cluster_count(cfg) > max_active_clusters:
        return "gmem_separate"

    return reduction_mode


def resolve_runtime_cluster_reduction_mode(
    cfg: MlaConfig | None,
    *,
    reduction_mode: str | None,
    hardware_info,
    stream=None,
    log=None,
) -> str | None:
    """Query cluster occupancy and return the final runtime reduction mode."""

    if (
        cfg is None
        or reduction_mode not in (None, "auto")
        or cfg.use_cluster_reduction != 1
    ):
        return reduction_mode

    cluster_shape = cluster_reduction_cluster_shape(cfg)
    cluster_size = cluster_shape[0] * cluster_shape[1] * cluster_shape[2]
    active_clusters = hardware_info.get_max_active_clusters(cluster_size, stream)
    cluster_count = cluster_reduction_cluster_count(cfg)
    if log is not None:
        log(
            "cluster_reduction_occupancy",
            f"clusters={cluster_count}, active={active_clusters}",
        )

    resolved_mode = resolve_auto_cluster_reduction_mode(
        cfg,
        reduction_mode=reduction_mode,
        max_active_clusters=active_clusters,
    )
    if log is not None:
        if resolved_mode == "gmem_separate":
            log(
                "cluster_reduction",
                "disabled: grid requires more than one active cluster wave",
            )
        else:
            log("cluster_reduction", "enabled")
    return resolved_mode


def resolve_auto_cluster_reduction_config(
    cfg: MlaConfig,
    *,
    reduction_mode: str | None,
) -> MlaConfig:
    """Disable auto cluster reduction when the static SMEM footprint is too large."""

    if reduction_mode not in (None, "auto") or cfg.use_cluster_reduction != 1:
        return cfg

    if estimated_throughput_latency_smem_data_bytes(cfg) > MAX_CLUSTER_SMEM_DATA_BYTES:
        return replace(cfg, use_cluster_reduction=0)

    return cfg


def validate_explicit_cluster_reduction_config(
    cfg: MlaConfig,
    *,
    reduction_mode: str | None,
) -> None:
    """Reject explicit cluster requests that exceed the SMEM launch budget."""

    if reduction_mode != "cluster":
        return
    smem_data_bytes = estimated_throughput_latency_smem_data_bytes(cfg)
    if smem_data_bytes > MAX_CLUSTER_SMEM_DATA_BYTES:
        raise ValueError(
            "explicit cluster reduction exceeds the SM100A shared-memory budget: "
            f"required_data_bytes={smem_data_bytes}, "
            f"max_data_bytes={MAX_CLUSTER_SMEM_DATA_BYTES}"
        )


def baseline_profile(
    tile_size_q: int | None = None,
    *,
    name: str = "baseline",
) -> MlaProfile:
    """Return the default nonpersistent, non-split 1CTA profile."""

    return MlaProfile(
        name=name,
        use_persistent_scheduler=0,
        use_clc_dynamic_persistent_scheduler=0,
        tile_size_q=tile_size_q,
    )


def persistent_profiles(
    num_heads_q: int, tile_size_q: int | None = None
) -> tuple[MlaProfile, ...]:
    """Return the persistent policy candidates in preferred order."""

    return (
        MlaProfile(
            name=profile_name("clc", num_heads_q, tile_size_q),
            use_persistent_scheduler=1,
            use_clc_dynamic_persistent_scheduler=1,
            use_multi_ctas_kv=0,
            tile_size_q=tile_size_q,
        ),
        MlaProfile(
            name=profile_name("static", num_heads_q, tile_size_q),
            use_persistent_scheduler=1,
            use_clc_dynamic_persistent_scheduler=0,
            use_multi_ctas_kv=0,
            tile_size_q=tile_size_q,
        ),
        MlaProfile(
            name=profile_name("nonpersistent", num_heads_q, tile_size_q),
            use_persistent_scheduler=0,
            use_clc_dynamic_persistent_scheduler=0,
            use_multi_ctas_kv=0,
            tile_size_q=tile_size_q,
        ),
    )


def persistent_override_profile(
    profile: MlaProfile, explicit_persistent: bool | None
) -> MlaProfile:
    """Apply the user persistent override to a selected non-split profile."""

    if explicit_persistent is None:
        return profile
    if profile.use_multi_ctas_kv == 1:
        if explicit_persistent:
            raise ValueError("persistent scheduling is not supported with split-KV")
        return profile
    if explicit_persistent:
        return replace(
            profile,
            use_persistent_scheduler=1,
            use_clc_dynamic_persistent_scheduler=(
                profile.use_clc_dynamic_persistent_scheduler
            ),
        )
    return replace(
        profile,
        use_persistent_scheduler=0,
        use_clc_dynamic_persistent_scheduler=0,
    )


def forced_persistent_profiles(
    *,
    num_heads_q: int,
    tile_size_q: int | None,
    explicit_persistent: bool,
) -> tuple[MlaProfile, ...]:
    """Return profiles for an explicit persistent/nonpersistent user request."""

    if explicit_persistent:
        return persistent_profiles(num_heads_q, tile_size_q)[:2]
    if tile_size_q is None:
        return (baseline_profile(),)
    return (
        MlaProfile(
            name=profile_name("nonpersistent", num_heads_q, tile_size_q),
            use_persistent_scheduler=0,
            use_clc_dynamic_persistent_scheduler=0,
            use_multi_ctas_kv=0,
            tile_size_q=tile_size_q,
        ),
    )


def automatic_unsplit_profiles(
    *,
    batch_size: int,
    num_heads_q: int,
    seq_len_q: int,
    tile_size_q: int,
    max_active_clusters: int,
) -> tuple[MlaProfile, ...]:
    """Return automatic scheduler candidates for an unsplit 1CTA launch.

    A direct grid that fits one resident wave has no launch work for a
    persistent CTA to replace. Once logical work exceeds resident capacity,
    CLC becomes the preferred scheduler independent of dtype, K length, or a
    shape-specific threshold.
    """

    clc, static, nonpersistent = persistent_profiles(num_heads_q, tile_size_q)
    base_work = q_tile_work_count(batch_size, num_heads_q, seq_len_q, tile_size_q)
    if base_work > max_active_clusters:
        return (clc, static, nonpersistent)
    return (nonpersistent, clc, static)


def keeps_mma_ab_profiles(
    *,
    num_heads_q: int,
    batch_size: int,
    seq_len_q: int,
    max_active_clusters: int,
) -> tuple[MlaProfile, ...]:
    """Return keeps-MMA-AB candidates for large head tiles."""

    # Keeps-MMA-AB is only wired for the q64 head-tiled schedules.  H64 and H128
    # both map to the same q64 tile; H128 is split across two head tiles.
    if num_heads_q not in (64, 128):
        return ()

    nonpersistent = MlaProfile(
        name="h64_keeps_mma_ab",
        num_ctas_per_head_dim=1,
        use_persistent_scheduler=0,
        use_clc_dynamic_persistent_scheduler=0,
        use_multi_ctas_kv=0,
        kernel_variant="keeps_mma_ab",
        tile_size_q=64,
    )
    split = MlaProfile(
        name="h64_keeps_mma_ab_splitkv_gmem",
        num_ctas_per_seq_kv=4,
        num_ctas_per_head_dim=1,
        use_persistent_scheduler=0,
        use_clc_dynamic_persistent_scheduler=0,
        use_multi_ctas_kv=1,
        kernel_variant="keeps_mma_ab",
        tile_size_q=64,
    )
    clc = MlaProfile(
        name="h64_keeps_mma_ab_clc",
        num_ctas_per_head_dim=1,
        use_persistent_scheduler=1,
        use_clc_dynamic_persistent_scheduler=1,
        use_multi_ctas_kv=0,
        kernel_variant="keeps_mma_ab",
        tile_size_q=64,
    )
    base_work = q_tile_work_count(batch_size, num_heads_q, seq_len_q, 64)
    if base_work > max_active_clusters:
        # The Keeps-MMA-AB task graph supports one persistent work tile. A
        # persistent launch would therefore omit the grid suffix once work
        # exceeds the resident wave, so multi-wave shapes use the direct grid.
        return nonpersistent, split
    return nonpersistent, clc, split


def keeps_mma_ab_explicit_split_profile(
    *,
    batch_size: int,
    num_heads_q: int,
    seq_len_q: int,
    split_kv: int,
    latent_dim: int,
    max_active_clusters: int,
) -> MlaProfile | None:
    """Return the keeps-MMA-AB profile constrained by an explicit split-KV count."""

    # Explicit split-KV is accepted only for the q64 keeps-MMA-AB schedules; the
    # swaps-MMA-AB path handles smaller tiles through ``explicit_split_profile``.
    if num_heads_q not in (64, 128):
        return None
    if split_kv <= 1:
        return MlaProfile(
            name="h64_keeps_mma_ab",
            num_ctas_per_head_dim=1,
            use_persistent_scheduler=0,
            use_clc_dynamic_persistent_scheduler=0,
            use_multi_ctas_kv=0,
            kernel_variant="keeps_mma_ab",
            tile_size_q=64,
        )

    base_work = q_tile_work_count(batch_size, num_heads_q, seq_len_q, 64)
    head_dim_split = head_dim_split_for_work(
        work_after_kv=base_work * split_kv,
        target_work=validate_max_active_clusters(max_active_clusters),
        latent_dim=latent_dim,
    )

    profile_suffix = "splitkv_gmem"
    if split_kv != 4:
        profile_suffix = f"splitkv{split_kv}_gmem"
    if head_dim_split > 1:
        profile_suffix = f"{profile_suffix}_hdim{latent_dim // head_dim_split}"
    return MlaProfile(
        name=f"h64_keeps_mma_ab_{profile_suffix}",
        num_ctas_per_seq_kv=split_kv,
        num_ctas_per_head_dim=head_dim_split,
        use_persistent_scheduler=0,
        use_clc_dynamic_persistent_scheduler=0,
        use_multi_ctas_kv=1,
        kernel_variant="keeps_mma_ab",
        tile_size_q=64,
    )


def keeps_mma_ab_forced_persistent_profile(
    *, num_heads_q: int, explicit_persistent: bool
) -> MlaProfile | None:
    """Return a keeps-MMA-AB profile constrained by the user persistent request."""

    if num_heads_q not in (64, 128):
        return None
    return MlaProfile(
        name="h64_keeps_mma_ab_clc" if explicit_persistent else "h64_keeps_mma_ab",
        num_ctas_per_head_dim=1,
        use_persistent_scheduler=int_flag(explicit_persistent),
        use_clc_dynamic_persistent_scheduler=int_flag(explicit_persistent),
        use_multi_ctas_kv=0,
        kernel_variant="keeps_mma_ab",
        tile_size_q=64,
    )


def head_dim_split_for_work(
    *, work_after_kv: int, target_work: int, latent_dim: int
) -> int:
    """Choose the V head-dim split used after the selected split-KV count."""

    # Use a V head-dim split only when the current grid leaves at least a 2x SM
    # gap. Smaller gaps are usually not worth the extra scheduling and epilogue
    # work.
    if work_after_kv * 2 > target_work:
        return 1

    if work_after_kv * 4 <= target_work and latent_dim % 4 == 0:
        return 4
    if latent_dim % 2 == 0:
        return 2
    return 1


def explicit_split_profile(
    *,
    batch_size: int,
    num_heads_q: int,
    seq_len_q: int,
    split_kv: int,
    latent_dim: int,
    max_active_clusters: int,
    tile_size_q: int | None = None,
) -> MlaProfile:
    """Return a swaps-MMA-AB profile constrained by a user split-KV count."""

    max_active_clusters = validate_max_active_clusters(max_active_clusters)
    if split_kv <= 1:
        return baseline_profile(tile_size_q)

    base_work = q_tile_work_count(batch_size, num_heads_q, seq_len_q, tile_size_q)
    target_work = max_active_clusters
    head_dim_split = head_dim_split_for_work(
        work_after_kv=base_work * split_kv,
        target_work=target_work,
        latent_dim=latent_dim,
    )

    suffix = f"splitkv{split_kv}"
    if head_dim_split > 1:
        suffix = f"{suffix}_hdim{latent_dim // head_dim_split}"
    return MlaProfile(
        name=profile_name(suffix, num_heads_q, tile_size_q),
        num_ctas_per_seq_kv=split_kv,
        num_ctas_per_head_dim=head_dim_split,
        use_persistent_scheduler=0,
        use_clc_dynamic_persistent_scheduler=0,
        use_multi_ctas_kv=1,
        use_cluster_reduction=1,
        tile_size_q=tile_size_q,
    )


def auto_split_profile(
    *,
    batch_size: int,
    num_heads_q: int,
    seq_len_q: int,
    seq_len_kv: int,
    latent_dim: int,
    max_active_clusters: int,
    tile_size_q: int | None = None,
) -> MlaProfile | None:
    """Choose split-KV followed by V head-dimension decomposition."""

    max_active_clusters = validate_max_active_clusters(max_active_clusters)
    original_tile_size_q = tile_size_q or tile_size_q_for_heads(num_heads_q)
    base_work = q_tile_work_count(batch_size, num_heads_q, seq_len_q, tile_size_q)
    target_work = max_active_clusters
    if base_work <= 0 or base_work >= target_work:
        return None

    split_kv = select_auto_split_kv(
        seq_len_kv=seq_len_kv,
        tile_size_q=original_tile_size_q,
        base_work=base_work,
        target_work=target_work,
    )
    work_after_kv = base_work * split_kv

    head_dim_split = head_dim_split_for_work(
        work_after_kv=work_after_kv,
        target_work=target_work,
        latent_dim=latent_dim,
    )
    if split_kv == 1 and head_dim_split == 1:
        return None

    suffix = "baseline"
    if split_kv > 1:
        suffix = "splitkv"
    elif head_dim_split > 1:
        suffix = f"hdim{latent_dim // head_dim_split}"
    if split_kv > 1 and head_dim_split > 1:
        suffix = f"{suffix}_hdim{latent_dim // head_dim_split}"
    return MlaProfile(
        name=profile_name(suffix, num_heads_q, original_tile_size_q),
        num_ctas_per_seq_kv=split_kv,
        num_ctas_per_head_dim=head_dim_split,
        use_persistent_scheduler=0,
        use_clc_dynamic_persistent_scheduler=0,
        use_multi_ctas_kv=int_flag(split_kv > 1),
        use_cluster_reduction=int_flag(split_kv > 1),
        kernel_variant=(
            "keeps_mma_ab" if original_tile_size_q >= 64 else "swaps_mma_ab"
        ),
        tile_size_q=original_tile_size_q,
    )


def is_throughput_latency_mla_supported_shape(
    *,
    batch_size: int,
    num_heads_q: int,
    seq_len_q: int,
    seq_len_kv: int,
    latent_dim: int = 512,
    rope_dim: int = 64,
    num_tokens_per_page: int = 32,
) -> bool:
    """Return whether the basic dense DS MLA 1CTA path can be considered."""

    del batch_size
    return (
        latent_dim == 512
        and rope_dim == 64
        and 1 <= num_heads_q <= 128
        and is_power_of_two(num_heads_q)
        and num_tokens_per_page in SUPPORTED_MLA_PAGE_SIZES
        and seq_len_q >= 1
        and seq_len_kv >= 128
    )


def enumerate_throughput_latency_mla_profiles(
    *,
    batch_size: int,
    num_heads_q: int,
    seq_len_q: int,
    seq_len_kv: int,
    latent_dim: int = 512,
    rope_dim: int = 64,
    num_tokens_per_page: int = 32,
    max_active_clusters: int,
    qkv_dtype: str = "bf16",
    tile_size_q: int | None = None,
    explicit_split_kv: int | None = None,
    explicit_persistent: bool | None = None,
) -> tuple[MlaProfile, ...]:
    """Return benchmarkable throughput-latency 1CTA profiles for a problem shape."""

    max_active_clusters = validate_max_active_clusters(max_active_clusters)
    if tile_size_q is not None:
        tile_size_q = validate_tile_size_q(tile_size_q)
    if not is_throughput_latency_mla_supported_shape(
        batch_size=batch_size,
        num_heads_q=num_heads_q,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        num_tokens_per_page=num_tokens_per_page,
    ):
        return ()

    selected_tile_size_q = tile_size_q or auto_tile_size_q_for_mla_gen(
        num_heads_q=num_heads_q,
        seq_len_q=seq_len_q,
    )
    if explicit_split_kv is not None and explicit_split_kv > 0:
        if explicit_split_kv > 1 and explicit_persistent is True:
            raise ValueError("persistent scheduling is not supported with split-KV")
        if explicit_split_kv == 1 and explicit_persistent is not None:
            if selected_tile_size_q >= 64:
                profile = keeps_mma_ab_forced_persistent_profile(
                    num_heads_q=num_heads_q,
                    explicit_persistent=explicit_persistent,
                )
                if profile is None:
                    return ()
                return (profile,)
            return forced_persistent_profiles(
                num_heads_q=num_heads_q,
                tile_size_q=selected_tile_size_q,
                explicit_persistent=explicit_persistent,
            )
        if selected_tile_size_q >= 64:
            profile = keeps_mma_ab_explicit_split_profile(
                batch_size=batch_size,
                num_heads_q=num_heads_q,
                seq_len_q=seq_len_q,
                split_kv=explicit_split_kv,
                latent_dim=latent_dim,
                max_active_clusters=max_active_clusters,
            )
            if profile is None:
                return ()
            return (profile,)

        return (
            explicit_split_profile(
                batch_size=batch_size,
                num_heads_q=num_heads_q,
                seq_len_q=seq_len_q,
                split_kv=explicit_split_kv,
                latent_dim=latent_dim,
                max_active_clusters=max_active_clusters,
                tile_size_q=selected_tile_size_q,
            ),
        )

    if explicit_persistent is not None:
        if selected_tile_size_q >= 64:
            profile = keeps_mma_ab_forced_persistent_profile(
                num_heads_q=num_heads_q,
                explicit_persistent=explicit_persistent,
            )
            if profile is None:
                return ()
            return (profile,)
        return forced_persistent_profiles(
            num_heads_q=num_heads_q,
            tile_size_q=selected_tile_size_q,
            explicit_persistent=explicit_persistent,
        )

    profiles: list[MlaProfile] = []
    split_profile = auto_split_profile(
        batch_size=batch_size,
        num_heads_q=num_heads_q,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        latent_dim=latent_dim,
        max_active_clusters=max_active_clusters,
        tile_size_q=selected_tile_size_q,
    )
    if split_profile is not None:
        profiles.append(split_profile)

    if selected_tile_size_q >= 64:
        profiles.extend(
            profile
            for profile in keeps_mma_ab_profiles(
                num_heads_q=num_heads_q,
                batch_size=batch_size,
                seq_len_q=seq_len_q,
                max_active_clusters=max_active_clusters,
            )
            if profile.name not in {candidate.name for candidate in profiles}
        )
    else:
        profiles.extend(
            automatic_unsplit_profiles(
                batch_size=batch_size,
                num_heads_q=num_heads_q,
                seq_len_q=seq_len_q,
                tile_size_q=selected_tile_size_q,
                max_active_clusters=max_active_clusters,
            )
        )

    if selected_tile_size_q < 64:
        baseline = baseline_profile(selected_tile_size_q)
        if not any(profile.name == baseline.name for profile in profiles):
            profiles.append(baseline)

    return tuple(profiles)


def resolve_throughput_latency_mla_profile(
    *,
    profile: MlaProfile | str | None,
    batch_size: int,
    num_heads_q: int,
    seq_len_q: int,
    seq_len_kv: int,
    latent_dim: int = 512,
    rope_dim: int = 64,
    num_tokens_per_page: int = 32,
    max_active_clusters: int,
    qkv_dtype: str = "bf16",
    tile_size_q: int | None = None,
    explicit_split_kv: int | None = None,
    explicit_persistent: bool | None = None,
) -> MlaProfile:
    """Resolve an explicit profile name or default to the first candidate."""

    if isinstance(profile, MlaProfile):
        return profile

    profiles = enumerate_throughput_latency_mla_profiles(
        batch_size=batch_size,
        num_heads_q=num_heads_q,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        num_tokens_per_page=num_tokens_per_page,
        max_active_clusters=max_active_clusters,
        qkv_dtype=qkv_dtype,
        tile_size_q=tile_size_q,
        explicit_split_kv=explicit_split_kv,
        explicit_persistent=explicit_persistent,
    )
    if profile in (None, "", "default"):
        if profiles:
            return profiles[0]
        return baseline_profile()

    for candidate in profiles:
        if candidate.name == profile:
            return candidate
    available = ", ".join(candidate.name for candidate in profiles) or "none"
    raise ValueError(
        f"throughput-latency 1CTA MLA profile {profile!r} is not valid for this shape; "
        f"available profiles: {available}"
    )


def make_throughput_latency_mla_config(
    *,
    batch_size: int,
    num_heads_q: int,
    seq_len_q: int,
    seq_len_kv: int,
    latent_dim: int = 512,
    rope_dim: int = 64,
    num_tokens_per_page: int = 32,
    qkv_dtype: str = "bf16",
    o_dtype: str = "bf16",
    profile: MlaProfile | str | None = None,
    persistent_wave_sm_count: int | None = None,
    max_active_clusters: int,
    reduction_mode: str | None = None,
    logical_num_heads_q: int | None = None,
    logical_seq_len_q: int | None = None,
    tile_size_q: int | None = None,
    explicit_split_kv: int | None = None,
    explicit_persistent: bool | None = None,
    mask_type: MaskType | str = MaskType.CAUSAL,
) -> MlaConfig:
    """Return throughput-latency 1CTA MLA traits for a concrete profile."""

    mask_type = normalize_mask_type(mask_type)

    if logical_num_heads_q is None:
        logical_num_heads_q = num_heads_q
    if logical_seq_len_q is None:
        logical_seq_len_q = seq_len_q

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_heads_q <= 0:
        raise ValueError("num_heads_q must be positive")
    if logical_num_heads_q <= 0:
        raise ValueError("logical_num_heads_q must be positive")
    if num_heads_q > 128 or not is_power_of_two(num_heads_q):
        raise ValueError(
            "num_heads_q must be a power of two no larger than 128 for "
            f"throughput-latency 1CTA MLA: num_heads_q={num_heads_q}"
        )
    if seq_len_q <= 0:
        raise ValueError("seq_len_q must be positive")
    if logical_seq_len_q <= 0:
        raise ValueError("logical_seq_len_q must be positive")
    if seq_len_kv <= 0:
        raise ValueError("seq_len_kv must be positive")
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    if rope_dim < 0:
        raise ValueError("rope_dim must be non-negative")
    if num_tokens_per_page not in SUPPORTED_MLA_PAGE_SIZES:
        raise ValueError(
            "num_tokens_per_page must be one of "
            f"{SUPPORTED_MLA_PAGE_SIZES}, got {num_tokens_per_page}"
        )
    if MlaConfig.tile_size_kv % num_tokens_per_page != 0:
        raise ValueError(
            "num_tokens_per_page must exactly divide the 1CTA KV tile: "
            f"tile_size_kv={MlaConfig.tile_size_kv}, "
            f"num_tokens_per_page={num_tokens_per_page}"
        )
    pages_per_kv_tile = MlaConfig.tile_size_kv // num_tokens_per_page
    if pages_per_kv_tile > MlaConfig.page_offsets_entries_per_stage:
        raise ValueError(
            "page-offset staging capacity is smaller than one KV tile: "
            f"pages_per_kv_tile={pages_per_kv_tile}, "
            "page_offsets_entries_per_stage="
            f"{MlaConfig.page_offsets_entries_per_stage}"
        )
    if qkv_dtype not in ("bf16", "e4m3"):
        raise ValueError(f"unsupported qkv_dtype={qkv_dtype!r}")
    if o_dtype not in ("bf16", "e4m3"):
        raise ValueError(f"unsupported o_dtype={o_dtype!r}")
    if tile_size_q is not None:
        tile_size_q = validate_tile_size_q(tile_size_q)
    elif isinstance(profile, str):
        tile_size_q = tile_size_q_from_profile_name(profile)
    max_active_clusters = validate_max_active_clusters(max_active_clusters)

    head_dim_qk = latent_dim + rope_dim
    selected_profile = resolve_throughput_latency_mla_profile(
        profile=profile,
        batch_size=batch_size,
        num_heads_q=num_heads_q,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        num_tokens_per_page=num_tokens_per_page,
        max_active_clusters=max_active_clusters,
        qkv_dtype=qkv_dtype,
        tile_size_q=tile_size_q,
        explicit_split_kv=explicit_split_kv,
        explicit_persistent=explicit_persistent,
    )
    selected_profile = persistent_override_profile(
        selected_profile,
        explicit_persistent,
    )
    tile_size_q = tile_size_q_for_profile(
        selected_profile, num_heads_q, seq_len_q, tile_size_q
    )
    flat_layout = FlatQueryTileLayout.for_tile(
        logical_num_heads_q,
        logical_seq_len_q,
        tile_size_q,
    )
    if num_heads_q != flat_layout.tile_size_q or seq_len_q != flat_layout.num_tiles:
        raise ValueError(
            "physical 1CTA launch shape must match the flat query-row layout: "
            f"got num_heads_q={num_heads_q}, seq_len_q={seq_len_q}; expected "
            f"num_heads_q={flat_layout.tile_size_q}, "
            f"seq_len_q={flat_layout.num_tiles} for logical "
            f"H={logical_num_heads_q}, SQ={logical_seq_len_q}"
        )
    if selected_profile.kernel_variant == "keeps_mma_ab" and num_heads_q < tile_size_q:
        raise ValueError(
            "keeps_mma_ab profiles require an effective Q tile of at least "
            f"{tile_size_q}, got num_heads_q={num_heads_q}"
        )
    num_ctas_for_all_heads = ceil(num_heads_q / tile_size_q)
    num_ctas_per_head_dim = selected_profile.num_ctas_per_head_dim

    if num_ctas_per_head_dim <= 0:
        raise ValueError("num_ctas_per_head_dim must be positive")
    if selected_profile.num_ctas_per_seq_kv <= 0:
        raise ValueError("num_ctas_per_seq_kv must be positive")
    if selected_profile.num_ctas_per_seq_kv > seq_len_kv:
        raise ValueError(
            "num_ctas_per_seq_kv must not exceed seq_len_kv: "
            f"num_ctas_per_seq_kv={selected_profile.num_ctas_per_seq_kv}, "
            f"seq_len_kv={seq_len_kv}"
        )
    if latent_dim % num_ctas_per_head_dim != 0:
        raise ValueError(
            "latent_dim must be divisible by num_ctas_per_head_dim: "
            f"latent_dim={latent_dim}, "
            f"num_ctas_per_head_dim={num_ctas_per_head_dim}"
        )
    head_dim_per_cta_v = latent_dim // num_ctas_per_head_dim
    use_cluster_reduction = resolve_split_kv_reduction_policy(
        profile=selected_profile,
        tile_size_q=tile_size_q,
        head_dim=latent_dim,
        head_dim_per_cta_v=head_dim_per_cta_v,
        reduction_mode=reduction_mode,
    )
    if use_cluster_reduction == 1 and flat_layout.tail_rows != tile_size_q:
        if reduction_mode == "cluster":
            raise ValueError(
                "cluster reduction requires every launched Q tile to contain a full "
                "tile_size_q rows: "
                f"tail_rows={flat_layout.tail_rows}, tile_size_q={tile_size_q}"
            )
        use_cluster_reduction = 0
    num_ctas_per_seq_q = max(1, seq_len_q)

    config_kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        num_heads_q=num_heads_q,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        logical_num_heads_q=logical_num_heads_q,
        logical_seq_len_q=logical_seq_len_q,
        mask_type=mask_type,
        head_dim_qk=head_dim_qk,
        head_dim_v=latent_dim,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        qkv_dtype=qkv_dtype,
        o_dtype=o_dtype,
        tile_size_q=tile_size_q,
        num_ctas_per_seq_q=num_ctas_per_seq_q,
        num_ctas_for_all_heads=num_ctas_for_all_heads,
        num_ctas_per_seq_kv=selected_profile.num_ctas_per_seq_kv,
        num_ctas_per_head_dim=num_ctas_per_head_dim,
        head_dim_per_cta_v=head_dim_per_cta_v,
        head_dim_per_stage_v=128,
        kv_stages=kv_stage_count(selected_profile, tile_size_q, qkv_dtype),
        softmax0_warp_idx=0,
        softmax_regs=softmax_register_budget(tile_size_q),
        correction_regs=correction_register_budget(tile_size_q),
        num_tokens_per_page=num_tokens_per_page,
        max_num_pages_per_seq_kv=max(1, ceil(seq_len_kv / num_tokens_per_page)),
        tmem_s_cols=tile_size_q,
        tmem_stats_cols=MlaConfig.tmem_stats_cols,
        tmem_o_cols=tile_size_q,
        kernel_variant=selected_profile.kernel_variant,
        use_multi_ctas_kv=selected_profile.use_multi_ctas_kv,
        use_cluster_reduction=use_cluster_reduction,
        use_persistent_scheduler=selected_profile.use_persistent_scheduler,
        use_clc_dynamic_persistent_scheduler=(
            selected_profile.use_clc_dynamic_persistent_scheduler
        ),
        persistent_wave_sm_count=persistent_wave_sm_count,
    )
    config_kwargs.update(dtype_config_kwargs(qkv_dtype, o_dtype))
    if selected_profile.kernel_variant == "keeps_mma_ab":
        config_kwargs.update(keeps_mma_ab_config_kwargs(selected_profile, qkv_dtype))
    cfg = MlaConfig(**config_kwargs)
    validate_explicit_cluster_reduction_config(cfg, reduction_mode=reduction_mode)
    cfg = resolve_auto_cluster_reduction_config(
        cfg,
        reduction_mode=reduction_mode,
    )
    return cfg
