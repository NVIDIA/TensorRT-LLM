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

"""Configuration for the throughput 2CTA MLA decode TS kernel.

The throughput 2CTA policy uses a 2CTA M128 schedule. BF16 is the public
throughput path; dtype traits are explicit so FP8 bring-up can share the same
configuration structure once its 2CTA V/PV layout is complete.
"""

from dataclasses import dataclass
from typing import Tuple

from ..helpers.constants import SUPPORTED_MLA_PAGE_SIZES
from ..helpers.mask import MaskType, normalize_mask_type


# Softmax converts natural-scale scores to exp2 with log2(e).
LOG2_E = 1.4426950408889634074

# Split-KV reduction allocates one LSE slot per possible split.  Match the
# standalone reducer's qualified workspace and launch capacity.
MAX_SPLITS = 128

# The separate reducer follows the public MLA output contract: partial O is
# stored as BF16 while LSE and the final accumulation remain FP32.  One
# 512-thread CTA owns eight D512 rows, with each thread moving one 16-byte
# BF16 vector.  Keeping these values derived makes the row packing explicit
# and prevents the launch geometry from drifting away from the workspace
# representation.
PARTIAL_O_BITS = 16
REDUCTION_THREADS_PER_CTA = 512
REDUCTION_VECTOR_BYTES = 16
REDUCTION_VALUES_PER_THREAD = REDUCTION_VECTOR_BYTES * 8 // PARTIAL_O_BITS
REDUCTION_THREADS_PER_ROW = 512 // REDUCTION_VALUES_PER_THREAD
REDUCTION_ROWS_PER_CTA = REDUCTION_THREADS_PER_CTA // REDUCTION_THREADS_PER_ROW

# PV consumes V from SMEM in 32-token K blocks.  Physical KV pages may be
# smaller or larger, but TMA must assemble this fixed block geometry before
# tcgen05 advances the V descriptor to the next K block.
V_SMEM_K_BLOCK_TOKENS = 32

# Each transpose-TMA issue stages one 64-element slice of the latent dimension.
V_TMA_LATENT_ELEMENTS = 64


def ceil_div(a: int, b: int) -> int:
    """Return ``ceil(a / b)`` for positive integer divisors."""

    if b <= 0:
        raise ValueError(f"divisor must be positive, got {b}")
    return (a + b - 1) // b


def compute_split_kv(
    *,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    mma_qk_tiler_mn: Tuple[int, int] = (128, 128),
    max_active_blocks: int,
) -> int:
    """Choose the throughput 2CTA split-KV count for a concrete launch.

    The heuristic tries to expose enough K-split work to fill the available CTA
    slots without creating extra partial K waves. The result is capped to keep
    the reduction grid bounded.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if seq_len_q <= 0:
        raise ValueError(f"seq_len_q must be positive, got {seq_len_q}")
    if seq_len_kv <= 0:
        raise ValueError(f"seq_len_kv must be positive, got {seq_len_kv}")
    if max_active_blocks <= 0:
        raise ValueError(f"max_active_blocks must be positive, got {max_active_blocks}")
    if mma_qk_tiler_mn[1] <= 0:
        raise ValueError(
            f"mma_qk_tiler_mn[1] must be positive, got {mma_qk_tiler_mn[1]}"
        )

    max_splits = ceil_div(seq_len_kv, mma_qk_tiler_mn[1])
    blocks_per_batch = max(1, max_active_blocks // batch_size // (seq_len_q * 2))
    split_heur = min(max_splits, blocks_per_batch)
    k_waves = ceil_div(max_splits, split_heur)
    split_wave_aware = ceil_div(max_splits, k_waves)
    # The 2CTA reduction workspace supports more split slots, but the automatic
    # scheduler caps launch-time splits at 32 to avoid excessive reduction work.
    return min(split_wave_aware, 32)


def compute_workspace_size(
    *,
    num_heads: int,
    seq_len_q: int,
    latent_dim: int,
    batch_size: int,
    split_kv: int,
    partial_o_dtype,
    lse_dtype,
) -> int:
    """Return mixed-dtype MLA split-KV workspace size in bytes."""

    if split_kv == 1:
        return 0
    partial_rows = batch_size * num_heads * seq_len_q * split_kv
    return partial_rows * (
        latent_dim * partial_o_dtype.width // 8 + lse_dtype.width // 8
    )


@dataclass
class MlaDecodeConfig:
    """MLA decode kernel configuration.

    All values are plain Python ints/tuples so they can be used as
    ``Constexpr`` in DSL code.
    """

    # Architecture.  Dense MLA uses 512 latent channels plus 64 RoPE channels;
    # the throughput path uses one 2-CTA cluster per M128 tile.
    latent_dim: int = 512
    rope_dim: int = 64
    num_mma_ctas: int = 2
    cluster_shape_mnk: Tuple[int, int, int] = (2, 1, 1)

    # MMA tile shapes.  QK runs over 128x128 K tiles, while PV writes the 512 V
    # head in two 256-column passes.
    mma_qk_tiler_mn: Tuple[int, int] = (128, 128)
    mma_pv_tiler_mn: Tuple[int, int] = (128, 256)
    mma_qk_tiler_k: int = 64  # = rope_dim
    mma_qk_tiler: Tuple[int, int, int] = (128, 128, 64)
    mma_qk_rope_tiler: Tuple[int, int, int] = (128, 128, 64)
    mma_pv_tiler: Tuple[int, int, int] = (128, 256, 32)

    # Iteration counts derived from the fixed dense MLA dimensions and MMA
    # tile shapes above.
    iterations_qk_latent: int = 8  # latent_dim / mma_qk_tiler_k = 512/64
    iterations_qk_rope: int = 1  # rope_dim / mma_qk_tiler_k = 64/64
    iterations_qk: int = 9  # latent + rope
    iterations_pv_k: int = 4  # mma_qk_tiler[1] / mma_pv_tiler[2] = 128/32
    iterations_pv_n: int = 2  # latent_dim / mma_pv_tiler[1] = 512/256
    # BF16 keeps the hardware-legal K64/PV-K32 transactions, but publishes two
    # consecutive transactions under one TMA/UMMA pipeline stage.  This
    # matches a 128-element head-dimension stage without requiring a TMA box
    # wider than the 128-byte swizzle permits. FP8 already uses native K128
    # QK stages and separate whole-tile K/V resources.
    kv_subtiles_per_stage: int = 2
    iterations_qk_latent_stages: int = 4
    iterations_qk_stages: int = 5
    iterations_pv_stages: int = 4

    # Pipeline stage counts for the captured schedule resources.  The combined
    # K/V stage count keeps enough delayed-V stages live for K-before-V overlap.
    load_q_stage: int = 1
    load_k_stage: int = 3
    load_v_stage: int = 2
    load_kv_stage: int = 7
    mma_s_stage: int = 2
    p_mma_stage: int = 2
    p_cor_stage: int = 2
    mma_o_stage: int = 1

    # Base BF16 warp assignments for the 12-warp CTA. Softmax and correction
    # each own a contiguous four-warp group; the remaining warps issue MMA and
    # TMA (including register-held page IDs) or provide scheduler/alignment
    # roles. The FP8 factory extends the CTA to 16 warps for its second softmax
    # group and split QK/PV schedule.
    compute_warp_ids: Tuple[int, ...] = (0, 1, 2, 3)
    correction_warp_ids: Tuple[int, ...] = (4, 5, 6, 7)
    mma_warp_id: int = 8
    load_tma_warp_id: int = 9
    pv_mma_warp_id: int = 11
    empty_warp_ids: Tuple[int, ...] = (11,)
    second_compute_warp_ids: Tuple[int, ...] = ()
    num_softmax_groups: int = 1

    num_compute_warps: int = 4
    threads_per_warp: int = 32
    threads_per_cta: int = 384  # 12 warps * 32
    warps_in_n: int = 2

    # Register budgets passed to setmaxnreg for the high-register softmax and
    # correction groups; all other warps use the lower shared budget.
    softmax_reg_num: int = 192
    correction_reg_num: int = 208
    other_reg_num: int = 96

    # Named barrier IDs and thread counts.  IDs are local to this kernel's
    # manual synchronization protocol and are kept away from the TMEM barrier.
    softmax_sync_bar_id: int = 2
    softmax_sync_threads: int = 128  # 4 warps * 32
    epilogue_sync_bar_id: int = 3
    epilogue_sync_threads: int = 128  # 4 warps * 32
    softmax_order_bar_0_id: int = 5
    softmax_order_bar_1_id: int = 6

    # TMEM sync barrier (for alloc/dealloc)
    tmem_sync_bar_id: int = 1
    tmem_sync_bar_threads: int = 0  # computed in make_config

    # TMEM layout offsets.  The full 512-column TMEM budget is reserved so S,
    # O, and correction-factor columns can use fixed offsets.
    num_tmem_cols: int = 512
    tmem_o_offset: int = 0  # computed
    correction_factor_offset: int = 0  # computed

    # SMEM element counts (per-stage or total)
    smem_q_latent_elems: int = 0
    smem_q_rope_elems: int = 0
    smem_kc_elems: int = 0
    smem_vc_elems: int = 0
    smem_k_stage_elems: int = 0
    smem_v_stage_elems: int = 0
    smem_p_elems: int = 0
    softmax_exchange_elems: int = 128

    # Page geometry.  Physical page-table geometry is independent of the V
    # transpose-TMA microtile used to assemble the tcgen05 SMEM operand.
    page_size: int = 32
    kc_page_tile_size: int = 32
    v_tma_token_count: int = 32

    # Data types
    qkv_dtype: str = "bf16"
    o_dtype: str = "bf16"
    qkv_dtype_bytes: int = 2
    o_dtype_bytes: int = 2
    use_bf16_output: int = 1
    use_fp8_output: int = 0

    # TMA byte counts
    tma_copy_q_bytes: int = 0
    tma_copy_kc_bytes: int = 0
    tma_copy_vc_bytes: int = 0
    tma_copy_k_tile_bytes: int = 0
    tma_copy_v_tile_bytes: int = 0
    tma_kc_subtile_bytes: int = 0
    tma_vc_subtile_bytes: int = 0

    # Scheduling.  The runner normally supplies max_active_clusters from
    # HardwareInfo; 56 is the SM100-class fallback used by local construction
    # paths that do not query hardware.
    use_fp8_split_mma_schedule: bool = False
    use_fp8_dual_softmax_schedule: bool = False
    max_active_clusters: int = 56
    is_persistent: bool = True
    is_var_seq: bool = False
    # Use block_split_kvs[batch] as a per-batch cap before runtime K contracts
    # the useful split prefix. Grid and workspace geometry retain the maximum.
    is_var_split_kv: bool = False
    # Causal is bottom-right aligned for speculative decode. Dense still masks
    # the ordinary per-batch KV tail at ``cache_seqs[batch]``.
    mask_type: str = MaskType.CAUSAL.value

    @property
    def tokens_per_k_tile(self) -> int:
        """Return the logical KV-token count covered by one QK tile."""

        return self.mma_qk_tiler[1]

    @property
    def tokens_per_k_cta(self) -> int:
        """Return the KV-token count owned by one CTA in the 2CTA cluster."""

        return self.tokens_per_k_tile // self.num_mma_ctas

    @property
    def pages_per_k_tile(self) -> int:
        """Return the physical page count spanned by one logical K tile."""

        return self.tokens_per_k_tile // self.page_size

    @property
    def pages_per_k_cta(self) -> int:
        """Return page IDs consumed by one CTA, including shared-page K tiles."""

        return max(1, ceil_div(self.pages_per_k_tile, self.num_mma_ctas))

    @property
    def tokens_per_v_tile(self) -> int:
        """Return the logical KV-token count covered by all PV K iterations."""

        return self.mma_pv_tiler[2] * self.iterations_pv_k

    @property
    def pages_per_v_tile(self) -> int:
        """Return the physical page count spanned by one logical V tile."""

        return self.tokens_per_v_tile // self.page_size

    @property
    def pages_per_v_subtile(self) -> int:
        """Return physical page IDs consumed by one staged BF16 PV iteration."""

        return max(1, ceil_div(self.pages_per_v_tile, self.iterations_pv_k))

    @property
    def v_subtiles_per_page(self) -> int:
        """Return staged BF16 PV iterations that share one physical page ID."""

        return max(1, ceil_div(self.iterations_pv_k, self.pages_per_v_tile))

    @property
    def v_tma_copies_per_subtile(self) -> int:
        """Return transpose-TMA copies needed to assemble one PV K subtile."""

        return self.mma_pv_tiler[2] // self.v_tma_token_count

    def is_fp8_qkv(self) -> bool:
        """Return whether Q/K/V tensors use E4M3 data."""

        return self.qkv_dtype == "e4m3"


def make_mla_decode_config(
    mma_qk_tiler_mn: Tuple[int, int] = (128, 128),
    mma_pv_tiler_mn: Tuple[int, int] = (128, 256),
    rope_dim: int = 64,
    page_size: int = 32,
    qkv_dtype: str = "bf16",
    o_dtype: str = "bf16",
    max_active_clusters: int = 56,
    is_persistent: bool = True,
    is_var_seq: bool = False,
    is_var_split_kv: bool = False,
    mask_type: MaskType | str = MaskType.CAUSAL,
) -> MlaDecodeConfig:
    """Create and populate a MlaDecodeConfig from problem parameters."""
    cfg = MlaDecodeConfig()
    cfg.mma_qk_tiler_mn = mma_qk_tiler_mn
    cfg.mma_pv_tiler_mn = mma_pv_tiler_mn
    cfg.rope_dim = rope_dim
    cfg.page_size = page_size
    cfg.qkv_dtype = qkv_dtype
    cfg.o_dtype = o_dtype
    cfg.max_active_clusters = max_active_clusters
    cfg.is_persistent = is_persistent
    cfg.is_var_seq = is_var_seq
    cfg.is_var_split_kv = is_var_split_kv
    cfg.mask_type = normalize_mask_type(mask_type)

    def _require_positive(name: str, value: int) -> None:
        """Validate that a named configuration value is positive."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    def _require_divisible(
        dividend_name: str, dividend: int, divisor_name: str, divisor: int
    ) -> None:
        """Validate that one named configuration value divides another."""
        _require_positive(dividend_name, dividend)
        _require_positive(divisor_name, divisor)
        if dividend % divisor != 0:
            raise ValueError(
                f"{dividend_name}={dividend} must be divisible by "
                f"{divisor_name}={divisor}"
            )

    for idx, value in enumerate(mma_qk_tiler_mn):
        _require_positive(f"mma_qk_tiler_mn[{idx}]", value)
    for idx, value in enumerate(mma_pv_tiler_mn):
        _require_positive(f"mma_pv_tiler_mn[{idx}]", value)
    if rope_dim < 0:
        raise ValueError(f"rope_dim must be non-negative, got {rope_dim}")
    _require_positive("page_size", page_size)
    if page_size not in SUPPORTED_MLA_PAGE_SIZES:
        raise ValueError(
            f"page_size must be one of {SUPPORTED_MLA_PAGE_SIZES}, got {page_size}"
        )
    if page_size > mma_qk_tiler_mn[1] or mma_qk_tiler_mn[1] % page_size != 0:
        raise ValueError(
            "page_size must exactly partition the throughput 2CTA K tile: "
            f"mma_qk_tiler_mn[1]={mma_qk_tiler_mn[1]}, page_size={page_size}"
        )

    if qkv_dtype not in ("bf16", "e4m3"):
        raise ValueError(f"unsupported qkv_dtype={qkv_dtype!r}")
    if o_dtype not in ("bf16", "e4m3"):
        raise ValueError(f"unsupported o_dtype={o_dtype!r}")
    cfg.qkv_dtype_bytes = 1 if qkv_dtype == "e4m3" else 2
    cfg.o_dtype_bytes = 1 if o_dtype == "e4m3" else 2
    cfg.use_bf16_output = int(o_dtype == "bf16")
    cfg.use_fp8_output = int(o_dtype == "e4m3")
    cfg.use_fp8_split_mma_schedule = qkv_dtype == "e4m3"
    cfg.use_fp8_dual_softmax_schedule = qkv_dtype == "e4m3"
    cfg.empty_warp_ids = () if cfg.use_fp8_split_mma_schedule else (cfg.pv_mma_warp_id,)
    if cfg.use_fp8_split_mma_schedule:
        cfg.second_compute_warp_ids = (12, 13, 14, 15)
        cfg.num_softmax_groups = 2
        cfg.threads_per_cta = cfg.threads_per_warp * 16
        cfg.softmax_reg_num = 160
        cfg.correction_reg_num = 160
        cfg.other_reg_num = 32
        cfg.mma_o_stage = 2
        cfg.epilogue_sync_bar_id = 4

    # Derived MMA tilers. FP8 latent QK uses K=128 while the separate RoPE MMA
    # keeps K=64.
    if cfg.rope_dim > 0:
        cfg.mma_qk_tiler_k = cfg.rope_dim * (2 if qkv_dtype == "e4m3" else 1)
    else:
        cfg.mma_qk_tiler_k = 128 if qkv_dtype == "e4m3" else 64
    _require_divisible(
        "latent_dim", cfg.latent_dim, "mma_qk_tiler_k", cfg.mma_qk_tiler_k
    )
    _require_divisible(
        "mma_qk_tiler_mn[1] * mma_qk_tiler_k",
        mma_qk_tiler_mn[1] * cfg.mma_qk_tiler_k,
        "mma_pv_tiler_mn[1]",
        mma_pv_tiler_mn[1],
    )
    cfg.mma_qk_tiler = (mma_qk_tiler_mn[0], mma_qk_tiler_mn[1], cfg.mma_qk_tiler_k)
    cfg.mma_qk_rope_tiler = (mma_qk_tiler_mn[0], mma_qk_tiler_mn[1], cfg.rope_dim)
    pv_k = mma_qk_tiler_mn[1] * cfg.mma_qk_tiler_k // mma_pv_tiler_mn[1]
    _require_divisible("mma_qk_tiler_mn[1]", mma_qk_tiler_mn[1], "pv_k", pv_k)
    _require_divisible(
        "latent_dim", cfg.latent_dim, "mma_pv_tiler_mn[1]", mma_pv_tiler_mn[1]
    )
    cfg.mma_pv_tiler = (mma_pv_tiler_mn[0], mma_pv_tiler_mn[1], pv_k)

    # Iteration counts
    cfg.iterations_qk_latent = cfg.latent_dim // cfg.mma_qk_tiler_k
    cfg.iterations_qk_rope = 1 if cfg.rope_dim > 0 else 0
    cfg.iterations_qk = cfg.iterations_qk_latent + cfg.iterations_qk_rope
    cfg.iterations_pv_k = cfg.mma_qk_tiler[1] // cfg.mma_pv_tiler[2]
    cfg.iterations_pv_n = cfg.latent_dim // cfg.mma_pv_tiler[1]
    cfg.kv_subtiles_per_stage = 1 if qkv_dtype == "e4m3" else 2
    _require_divisible(
        "iterations_qk_latent",
        cfg.iterations_qk_latent,
        "kv_subtiles_per_stage",
        cfg.kv_subtiles_per_stage,
    )
    _require_divisible(
        "iterations_pv_k * iterations_pv_n",
        cfg.iterations_pv_k * cfg.iterations_pv_n,
        "kv_subtiles_per_stage",
        cfg.kv_subtiles_per_stage,
    )
    cfg.iterations_qk_latent_stages = (
        cfg.iterations_qk_latent // cfg.kv_subtiles_per_stage
    )
    cfg.iterations_qk_stages = cfg.iterations_qk_latent_stages + cfg.iterations_qk_rope
    cfg.iterations_pv_stages = (
        cfg.iterations_pv_k * cfg.iterations_pv_n // cfg.kv_subtiles_per_stage
    )
    if cfg.tokens_per_v_tile != cfg.tokens_per_k_tile:
        raise ValueError(
            "PV K iterations must cover the same token tile as QK: "
            f"tokens_per_v_tile={cfg.tokens_per_v_tile}, "
            f"tokens_per_k_tile={cfg.tokens_per_k_tile}"
        )

    # Page-offset tile sizes
    num_mma_ctas = cfg.cluster_shape_mnk[0]
    cfg.num_mma_ctas = num_mma_ctas
    _require_divisible(
        "mma_qk_tiler_mn[0]", cfg.mma_qk_tiler[0], "num_mma_ctas", num_mma_ctas
    )
    _require_divisible(
        "mma_qk_tiler_mn[1]", cfg.mma_qk_tiler[1], "num_mma_ctas", num_mma_ctas
    )
    if (
        cfg.page_size != cfg.tokens_per_k_tile
        and cfg.tokens_per_k_cta % cfg.page_size != 0
    ):
        raise ValueError(
            "page_size must partition each CTA's K tile unless both CTAs share "
            "one full-tile page: "
            f"tokens_per_k_cta={cfg.tokens_per_k_cta}, page_size={cfg.page_size}"
        )
    _require_divisible(
        "mma_pv_tiler_mn[0]", cfg.mma_pv_tiler[0], "num_mma_ctas", num_mma_ctas
    )
    _require_divisible(
        "mma_pv_tiler_mn[1]", cfg.mma_pv_tiler[1], "num_mma_ctas", num_mma_ctas
    )
    cfg.kc_page_tile_size = min(cfg.page_size, cfg.tokens_per_k_cta)
    cfg.v_tma_token_count = min(cfg.page_size, V_SMEM_K_BLOCK_TOKENS)
    _require_divisible(
        "V_SMEM_K_BLOCK_TOKENS",
        V_SMEM_K_BLOCK_TOKENS,
        "v_tma_token_count",
        cfg.v_tma_token_count,
    )
    _require_divisible(
        "mma_pv_tiler[2]",
        cfg.mma_pv_tiler[2],
        "v_tma_token_count",
        cfg.v_tma_token_count,
    )

    # SMEM sizes (elements)
    cfg.smem_q_latent_elems = (
        (cfg.mma_qk_tiler[0] // num_mma_ctas)
        * cfg.mma_qk_tiler[2]
        * cfg.iterations_qk_latent
        * cfg.load_q_stage
    )
    cfg.smem_q_rope_elems = (
        (cfg.mma_qk_rope_tiler[0] // num_mma_ctas)
        * cfg.mma_qk_rope_tiler[2]
        * cfg.load_q_stage
    )
    k_latent_subtile_elems = cfg.mma_qk_tiler[1] // num_mma_ctas * cfg.mma_qk_tiler[2]
    k_rope_subtile_elems = (
        cfg.mma_qk_rope_tiler[1] // num_mma_ctas * cfg.mma_qk_rope_tiler[2]
    )
    if qkv_dtype == "e4m3":
        cfg.smem_k_stage_elems = (
            k_latent_subtile_elems * cfg.iterations_qk_latent
            + k_rope_subtile_elems * cfg.iterations_qk_rope
        )
        cfg.smem_kc_elems = cfg.smem_k_stage_elems * cfg.load_k_stage
    else:
        cfg.smem_k_stage_elems = k_latent_subtile_elems * cfg.kv_subtiles_per_stage
        cfg.smem_kc_elems = cfg.smem_k_stage_elems * cfg.load_kv_stage
    v_subtile_elems = cfg.mma_pv_tiler[1] // num_mma_ctas * cfg.mma_pv_tiler[2]
    cfg.smem_v_stage_elems = v_subtile_elems * cfg.iterations_pv_k * cfg.iterations_pv_n
    cfg.smem_vc_elems = (
        cfg.smem_v_stage_elems * cfg.load_v_stage if qkv_dtype == "e4m3" else 0
    )
    cfg.smem_p_elems = (
        (cfg.mma_pv_tiler[0] // num_mma_ctas)
        * cfg.mma_pv_tiler[2]
        * cfg.iterations_pv_k
        * cfg.p_mma_stage
    )
    cfg.softmax_exchange_elems = (
        cfg.num_compute_warps
        * cfg.threads_per_warp
        * (1 if not cfg.use_fp8_dual_softmax_schedule else 2)
    )

    # TMEM layout
    cfg.tmem_o_offset = cfg.mma_s_stage * cfg.mma_qk_tiler[1] // cfg.warps_in_n
    cfg.correction_factor_offset = cfg.tmem_o_offset + cfg.latent_dim // cfg.warps_in_n

    # TMA byte counts
    q_latent_tile_bytes = (
        cfg.mma_qk_tiler[0] // num_mma_ctas * cfg.mma_qk_tiler[2] * cfg.qkv_dtype_bytes
    )
    q_rope_tile_bytes = (
        cfg.mma_qk_rope_tiler[0]
        // num_mma_ctas
        * cfg.mma_qk_rope_tiler[2]
        * cfg.qkv_dtype_bytes
    )
    cfg.tma_copy_q_bytes = (
        q_latent_tile_bytes * num_mma_ctas * cfg.iterations_qk_latent
        + q_rope_tile_bytes * num_mma_ctas * cfg.iterations_qk_rope
    )
    cfg.tma_copy_kc_bytes = (
        cfg.mma_qk_tiler[1]
        // num_mma_ctas
        * cfg.mma_qk_tiler[2]
        * cfg.qkv_dtype_bytes
        * num_mma_ctas
    )
    cfg.tma_copy_vc_bytes = (
        cfg.mma_pv_tiler[1]
        // num_mma_ctas
        * cfg.mma_pv_tiler[2]
        * cfg.qkv_dtype_bytes
        * num_mma_ctas
    )
    if cfg.tma_copy_kc_bytes != cfg.tma_copy_vc_bytes:
        raise ValueError(
            "K and V TMA subtile byte counts must match: "
            f"tma_copy_kc_bytes={cfg.tma_copy_kc_bytes}, "
            f"tma_copy_vc_bytes={cfg.tma_copy_vc_bytes}"
        )
    cfg.tma_kc_subtile_bytes = cfg.tma_copy_kc_bytes * cfg.kv_subtiles_per_stage
    cfg.tma_vc_subtile_bytes = cfg.tma_copy_vc_bytes * cfg.kv_subtiles_per_stage
    cfg.tma_copy_k_tile_bytes = (
        cfg.smem_k_stage_elems * cfg.qkv_dtype_bytes * num_mma_ctas
    )
    cfg.tma_copy_v_tile_bytes = (
        cfg.tma_copy_vc_bytes * cfg.iterations_pv_k * cfg.iterations_pv_n
    )

    # TMEM sync barrier thread count: QK MMA + softmax + correction, plus
    # the FP8 PV-MMA warp when it participates in TMEM O production.
    cfg.tmem_sync_bar_threads = (
        cfg.threads_per_warp * (2 if cfg.use_fp8_split_mma_schedule else 1)
        + cfg.threads_per_warp * cfg.num_compute_warps
        + cfg.threads_per_warp * cfg.num_compute_warps
    )
    if cfg.use_fp8_dual_softmax_schedule:
        cfg.tmem_sync_bar_threads += cfg.threads_per_warp * cfg.num_compute_warps

    return cfg
