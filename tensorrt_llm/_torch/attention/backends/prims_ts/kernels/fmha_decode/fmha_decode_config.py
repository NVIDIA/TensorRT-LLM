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

"""
Configuration for the FMHA decode TS kernel.

The class FmhaDecodeConfig encapsulates static configuration parameters
that should be set before kernel compilation.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields, replace

import cutlass.utils as utils
from cutlass import BFloat16, Float16, Float32, Float8E4M3FN

from ..._block_sparse.common import (
    _block_sparse_kv_atom_size,
    _select_block_sparse_q_tile_size,
    _validate_sparse_kv_block_size,
)
from ...split_kv_mode_policy import select_split_kv_modes
from .fmha_decode_constants import (
    AUTO_LAUNCH_TILE_SIZE_KV,
    BITS_PER_BYTE,
    BYTES_PER_KIB,
    FALLBACK_SM_COUNT_B200,
    FP32_BYTES,
    FP8_OUTPUT_ELEMENTS_PER_REG_GROUP,
    FP8_P_PACKED_REGS_PER_Q_REPEAT,
    FP8_VALUES_PER_REG,
    FP16_OUTPUT_ELEMENTS_PER_REG_GROUP,
    FP16_P_PACKED_REGS_PER_Q_REPEAT,
    FP16_VALUES_PER_REG,
    KV_TILE_256_REGISTER_REALLOCATION_MIN_TILES,
    KV_TILE_256_SHARED_FIFO_STAGES,
    MAX_CLUSTER_DIM_X,
    MAX_CLUSTER_PARTIAL_SMEM_BYTES,
    MAX_KV_STAGE_SMEM_KIB,
    MAX_WARP_GROUPS,
    MIN_LOOP_ITERS_PER_SPLIT,
    PARALLEL_REDUCTION_BYTES_PER_SLICE,
    PARALLEL_REDUCTION_THREADS_PER_CTA,
    PARTIAL_O_ELEMENT_BYTES,
    PARTIAL_STATS_VALUES_PER_ROW,
    Q_REPETITION_GROUP_HEADS,
    Q_ROW_ALIGNMENT_BYTES,
    REDUCTION_BYTES_PER_SLICE,
    REDUCTION_THREADS_PER_CTA,
    SPLIT_KV_MIN_TOKENS_PER_CTA,
    TMEM_COLUMNS_PER_ROW,
    TMEM_ROW_STRIDE,
    TOTAL_SMEM_BUDGET_KIB,
    WARP_THREADS,
)

ConfigValue = int | float | bool | str | type | None
_GroupedKeepsProfileKey = tuple[type, type, type, int, int, int, int]

# Public APIs use the strings ``dense`` and ``causal``.  Keep the value carried
# through FmhaDecodeConfig as a small integer so mask selection remains a
# compile-time predicate in CuTe DSL kernels.
DENSE = 0
CAUSAL = 1
MASK_TYPES = ("dense", "causal")

# Every correction lane reduces one packed 16-byte partial-O vector.  The
# default four-warp correction group therefore owns a 2-KiB reducer slice.
SPLIT_REDUCTION_VECTOR_BYTES_PER_THREAD = 16

_GROUPED_KEEPS_MAIN_PROFILE: _GroupedKeepsProfileKey = (
    Float16,
    Float16,
    Float16,
    128,
    0,
    2,
    2,
)
# Block-sparse feature compatibility is validated by
# ``validate_block_sparse_profile``. This set only selects the Keeps MMA
# resource recipes qualified for that already-validated launch domain.
_BLOCK_SPARSE_GROUPED_KEEPS_PROFILES = {
    _GROUPED_KEEPS_MAIN_PROFILE,
    (BFloat16, BFloat16, BFloat16, 128, 0, 2, 2),
}
_GROUPED_KEEPS_STATIC_ONLY_PROFILES = {
    (Float8E4M3FN, Float8E4M3FN, Float16, 128, 0, 2, 2),
    (BFloat16, BFloat16, BFloat16, 64, 0, 2, 2),
    (Float16, Float16, Float16, 256, 128, 1, 1),
}

_KV_TILE_256_PHYSICAL_DEFAULTS: Mapping[str, ConfigValue] = {
    "tmem_s_cols": 128,
    "tmem_stats_cols": 32,
    "tmem_p_cols": 64,
    "tmem_o_cols": 128,
    "mma_tile_m_bmm1": 64,
    "mma_tile_n_bmm1": 256,
    "mma_tile_m_bmm2": 64,
    "mma_tile_n_bmm2": 256,
    "q_stages": 1,
    "kv_stages": KV_TILE_256_SHARED_FIFO_STAGES,
    "head_dim_per_stage_kv": 0,
    "num_insts_kv": 2,
    "o_stages": 2,
}

# KV256 currently uses one validated 16-warp role assignment. Keep these
# defaults next to the physical profile; explicit alternatives remain intact
# until the common profile validator determines whether they are supported.
_KV_TILE_256_TASK_TOPOLOGY_DEFAULTS: Mapping[str, int] = {
    "softmax0_warp_idx": 0,
    "softmax0_num_warps": 4,
    "softmax1_warp_idx": 4,
    "softmax1_num_warps": 4,
    "correction_warp_idx": 8,
    "correction_num_warps": 4,
    "mma_warp_idx": 12,
    "mma_num_warps": 1,
    "load_warp_idx": 13,
    "load_num_warps": 1,
    "page_offsets_warp_idx": 14,
    "page_offsets_num_warps": 1,
    "scheduler_warp_idx": 13,
    "scheduler_num_warps": 1,
}

_KV_TILE_256_TUNABLE_FIELDS = frozenset(("kv_stages",))

# Public cost-model collection uses the FP8 proxy for every source dtype.  The
# original fixed-Q1 ratio-32 requests exercise a partial grouped-Q tile; the
# shape-aware path also admits complete fixed multi-Q tiles at any legal head
# ratio. These are profile families rather than shape exceptions: batch size,
# KV length, tile choice, and legal GMEM split fanout remain unrestricted by
# this declaration.
_GROUPED_KEEPS_PAGED_FP8_PROFILES = {
    (Float8E4M3FN, Float8E4M3FN, Float8E4M3FN, 64, 0, 2, 2),
    (Float8E4M3FN, Float8E4M3FN, Float8E4M3FN, 128, 0, 2, 2),
    (Float8E4M3FN, Float8E4M3FN, Float16, 256, 128, 1, 1),
}


def normalize_mask_type(
    mask_type: str | int | None,
    *,
    sliding_window_causal: bool = False,
) -> int:
    """Return the constexpr mask id for a public mask selection.

    ``None`` uses the decode defaults: ordinary decode is dense, while
    requesting a causal sliding window implies a causal right bound.  An
    explicit dense mask together with a causal sliding window is contradictory
    and rejected instead of silently changing the requested mask.
    """

    if mask_type is None:
        return CAUSAL if sliding_window_causal else DENSE
    if isinstance(mask_type, bool):
        raise ValueError("mask_type must be 'dense' or 'causal', not a boolean")
    if isinstance(mask_type, int):
        if mask_type not in (DENSE, CAUSAL):
            raise ValueError("internal mask_type must be DENSE or CAUSAL")
        normalized = mask_type
    elif isinstance(mask_type, str):
        value = mask_type.lower()
        if value not in MASK_TYPES:
            raise ValueError(
                f"mask_type must be one of {MASK_TYPES}, got {mask_type!r}"
            )
        normalized = CAUSAL if value == "causal" else DENSE
    else:
        raise ValueError(
            f"mask_type must be one of {MASK_TYPES}, got {type(mask_type).__name__}"
        )
    if sliding_window_causal and normalized != CAUSAL:
        raise ValueError(
            "sliding_window_causal requires mask_type='causal'; omit mask_type "
            "to select causal implicitly or pass mask_type='causal' explicitly"
        )
    return normalized


def mask_type_name(mask_type: int) -> str:
    """Return the public string for a normalized constexpr mask id."""

    if mask_type == DENSE:
        return "dense"
    if mask_type == CAUSAL:
        return "causal"
    raise ValueError(f"internal mask_type must be DENSE or CAUSAL, got {mask_type!r}")


def _q_tokens_per_cta(
    rows_per_cta: int, heads_q_per_kv: int, groups_tokens_heads_q: bool
) -> int:
    """Return complete Q tokens packed into one CTA."""
    if groups_tokens_heads_q:
        return rows_per_cta // heads_q_per_kv
    return 1


def _q_tma_rows_per_cta(
    rows_per_cta: int, heads_q_per_kv: int, groups_tokens_heads_q: bool
) -> int:
    """Return rows accounted by Q TMA for every CTA."""
    if groups_tokens_heads_q:
        return (
            _q_tokens_per_cta(rows_per_cta, heads_q_per_kv, groups_tokens_heads_q)
            * heads_q_per_kv
        )
    return rows_per_cta


@dataclass(frozen=True)
class QTileGeometry:
    """Host-side ownership and padding contract for one decode Q CTA."""

    rows_per_cta: int
    heads_q_per_kv: int
    groups_tokens_heads_q: bool

    @property
    def tokens_per_cta(self) -> int:
        """Return complete Q tokens packed into one CTA."""
        return _q_tokens_per_cta(
            self.rows_per_cta,
            self.heads_q_per_kv,
            self.groups_tokens_heads_q,
        )

    @property
    def head_ctas_per_token(self) -> int:
        """Return the number of head-band CTAs assigned to one Q token."""
        if self.groups_tokens_heads_q:
            return 1
        return (self.heads_q_per_kv + self.rows_per_cta - 1) // self.rows_per_cta

    @property
    def tma_rows_per_cta(self) -> int:
        """Return rows accounted by Q TMA for every CTA."""
        return _q_tma_rows_per_cta(
            self.rows_per_cta,
            self.heads_q_per_kv,
            self.groups_tokens_heads_q,
        )

    def num_q_ctas(self, seq_len_q: int) -> int:
        """Return the number of Q CTAs for one ``(batch, KV head)`` pair."""
        if seq_len_q < 0:
            raise ValueError("seq_len_q must be non-negative")
        if self.groups_tokens_heads_q:
            return (seq_len_q + self.tokens_per_cta - 1) // self.tokens_per_cta
        return seq_len_q * self.head_ctas_per_token


@dataclass(frozen=True)
class GroupedQMmaCandidate:
    """One grouped-Q MMA geometry candidate."""

    variant: str
    tile_size_q: int
    q_tokens_per_cta: int
    q_tiles: int


@dataclass(frozen=True)
class GroupedQLaunchCandidate:
    """One production-valid grouped-Q MMA and KV-split launch recipe."""

    mma: GroupedQMmaCandidate
    split_kv_mode: str
    splits_kv: int
    base_ctas: int
    launched_ctas: int
    seq_len_per_cta_kv: int
    waves: int
    modeled_time: float


_GROUPED_Q_MMA_TILES = (
    ("swaps_mma_ab", 8),
    ("swaps_mma_ab", 16),
    ("swaps_mma_ab", 32),
    ("keeps_mma_ab", 64),
    ("keeps_mma_ab", 128),
)

# Empirical GQA-generation factors derived from matched reference measurements.
# The selector reuses the measured relative costs but resolves its own legal
# profiles, Q geometry, split fanout, and cluster promotion instead of copying
# a final shape policy.
_GROUPED_Q_MAINLOOP_COST = {
    8: 1.0,
    16: 1.2,
    32: 1.48,
    64: 1.68,
    128: 2.2,
}
_GROUPED_Q_REDUCTION_COST = {
    8: 1.0,
    16: 1.03,
    32: 1.08,
    64: 1.2,
    128: 1.32,
}
_GROUPED_Q_REDUCTION_SEQ_LEN_FACTOR = 128.0


def enumerate_grouped_q_mma_candidates(
    *, heads_q_per_kv: int, seq_len_q: int
) -> tuple[GroupedQMmaCandidate, ...]:
    """Return grouped-Q candidates without applying launch policy.

    A CTA always owns an integral number of complete Q-head groups.  TileQ may
    leave structural padding rows and the final CTA may own fewer tokens than
    its capacity; the common Q geometry and row masks already represent both
    cases.  This helper deliberately does not inspect dtypes, reduction modes,
    scheduler state, or mutate ``FmhaDecodeConfig``.
    """
    if heads_q_per_kv <= 0:
        raise ValueError("heads_q_per_kv must be positive")
    if seq_len_q <= 0:
        raise ValueError("seq_len_q must be positive")

    candidates = []
    for variant, tile_size_q in _GROUPED_Q_MMA_TILES:
        if tile_size_q < heads_q_per_kv:
            continue
        q_tokens_per_cta = tile_size_q // heads_q_per_kv
        candidates.append(
            GroupedQMmaCandidate(
                variant=variant,
                tile_size_q=tile_size_q,
                q_tokens_per_cta=q_tokens_per_cta,
                q_tiles=(seq_len_q + q_tokens_per_cta - 1) // q_tokens_per_cta,
            )
        )
    return tuple(candidates)


def make_grouped_q_launch_candidate(
    candidate: GroupedQMmaCandidate,
    *,
    splits_kv: int,
    seq_len_kv: int,
    tile_size_kv: int,
    num_insts_kv: int,
    batch_size: int,
    num_heads_kv: int,
    service_capacity: int,
) -> GroupedQLaunchCandidate:
    """Build one empirical grouped-Q launch-cost record.

    The model follows the measured GQA-generation cost structure:

    ``(mainloop_factor[TileQ] * seq_len_per_cta_kv +``
    `` reduction_factor[TileQ] * 128 * splits_kv) * waves``.

    This is a small deterministic default heuristic. It deliberately uses the
    same factors for every dtype; the production profile validator still
    determines which actual-dtype recipes are legal.
    """
    if splits_kv <= 0:
        raise ValueError("splits_kv must be positive")
    if seq_len_kv <= 0:
        raise ValueError("seq_len_kv must be positive")
    if tile_size_kv <= 0:
        raise ValueError("tile_size_kv must be positive")
    if num_insts_kv <= 0:
        raise ValueError("num_insts_kv must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_heads_kv <= 0:
        raise ValueError("num_heads_kv must be positive")
    if service_capacity <= 0:
        raise ValueError("service_capacity must be positive")
    kv_tiles = (seq_len_kv + tile_size_kv - 1) // tile_size_kv
    seq_len_per_cta_kv = ((kv_tiles + splits_kv - 1) // splits_kv) * tile_size_kv
    base_ctas = batch_size * num_heads_kv * candidate.q_tiles
    launched_ctas = base_ctas * splits_kv
    waves = (launched_ctas + service_capacity - 1) // service_capacity
    modeled_time = (
        _GROUPED_Q_MAINLOOP_COST[candidate.tile_size_q] * seq_len_per_cta_kv
        + _GROUPED_Q_REDUCTION_COST[candidate.tile_size_q]
        * _GROUPED_Q_REDUCTION_SEQ_LEN_FACTOR
        * splits_kv
    ) * waves
    return GroupedQLaunchCandidate(
        mma=candidate,
        split_kv_mode=("gmem_reduction" if splits_kv > 1 else "disabled"),
        splits_kv=splits_kv,
        base_ctas=base_ctas,
        launched_ctas=launched_ctas,
        seq_len_per_cta_kv=seq_len_per_cta_kv,
        waves=waves,
        modeled_time=modeled_time,
    )


def select_grouped_q_launch_candidate(
    candidates: Sequence[GroupedQLaunchCandidate],
) -> GroupedQLaunchCandidate:
    """Select the lowest-cost legal grouped-Q launch recipe.

    The score compares actual Q-grid waves, so a wider tile can win only when
    it removes enough repeated KV mainloop work to repay its higher per-tile
    cost.  This is important for partial final tiles: TileQ128 and TileQ64 can
    have different Q-grid multiplicities even when both are legal.
    """
    if not candidates:
        raise ValueError("at least one grouped-Q launch candidate is required")

    return min(
        candidates,
        key=lambda recipe: (
            recipe.modeled_time,
            recipe.waves,
            recipe.splits_kv,
            -recipe.mma.tile_size_q,
        ),
    )


def select_grouped_q_direct_wave_candidate(
    candidates: Sequence[GroupedQLaunchCandidate],
) -> GroupedQLaunchCandidate:
    """Select a direct recipe using the same mainloop-aware launch score."""
    direct = tuple(recipe for recipe in candidates if recipe.splits_kv == 1)
    if not direct:
        raise ValueError("at least one direct grouped-Q candidate is required")
    return select_grouped_q_launch_candidate(direct)


def make_q_tile_geometry(
    *,
    rows_per_cta: int,
    heads_q_per_kv: int,
    groups_tokens_heads_q: bool,
) -> QTileGeometry:
    """Build the common grouped or one-token Q-tile geometry contract."""
    if rows_per_cta <= 0:
        raise ValueError("rows_per_cta must be positive")
    if heads_q_per_kv <= 0:
        raise ValueError("heads_q_per_kv must be positive")

    if groups_tokens_heads_q:
        if rows_per_cta < heads_q_per_kv:
            raise ValueError("grouped Q tiles require rows_per_cta >= heads_q_per_kv")
        return QTileGeometry(
            rows_per_cta=rows_per_cta,
            heads_q_per_kv=heads_q_per_kv,
            groups_tokens_heads_q=True,
        )

    return QTileGeometry(
        rows_per_cta=rows_per_cta,
        heads_q_per_kv=heads_q_per_kv,
        groups_tokens_heads_q=False,
    )


@dataclass
class FmhaDecodeConfig:
    """
    Kernel-wide configuration for fmha_decode.
    """

    # ------------------------------------------------------------------
    # Problem shape
    # ------------------------------------------------------------------
    # Per-head embedding dimension D. Supported profiles use 64, 128, or 256.
    headdim: int = 128
    # Number of KV tiles in the launch (= ceil(seq_len_kv / tile_size_kv)).
    # Populated by the launcher once seq_len_kv is known.
    total_kv_tiles: int = 0
    # Q rows placed on the small MMA N dimension. Shape-aware selection derives
    # this tile from the Q-heads-per-KV-head ratio and grouping policy.
    tile_size_q: int = 8
    # Q-layout metadata. Raw shape-less configs retain neutral values;
    # shape-aware selection fills them and defaults to grouped Q.
    max_seq_len_q: int = 1
    # Select the packed Q/O ABI. Q and O are laid out as
    # [sum_q_tokens, num_heads_q, head_dim] and indexed by cu_seqlens_q.
    use_variable_seqlens_q: bool = False
    heads_q_per_kv: int = 0
    groups_tokens_heads_q: bool = False
    # K/V tokens per tile along the K-sequence dimension; also the MMA "M"
    # dimension for BMM1 under SwapsMmaAb.
    tile_size_kv: int = 128
    # Select a block-sparse launch. A positive semantic Q block is legal when
    # the selected physical Q tile stays within one route row. KV blocks remain
    # restricted to 8/16/32 or positive multiples of 64 and are assembled into
    # a profile-selected fixed KV128 or KV256 route.
    use_block_sparse: bool = False
    q_block_size: int = 0
    kv_block_size: int = 0
    # Optional batch-wide physical-token validity metadata shared by every head
    # and sparse row in that batch item. Keeps profiles derive their route-full
    # fast path from these words at run time; it is intentionally not a separate
    # plan/config specialization.
    use_kv_valid_bits: bool = False
    # Let independent warps issue the two fine-route K/V instruction streams.
    # The wrapper resolves this from immutable route capacity; the kernel does
    # not inspect live BSR morphology to select its task topology.
    use_parallel_sparse_kv_loads: bool = False
    # Number of K/V instances that the loop processes per step. Two instances
    # (K0/V0 and K1/V1) let two parallel SoftmaxTask groups consume alternating
    # tiles, improving SM utilization when tile_size_q is tiny.
    num_insts_kv: int = 2  # K/V instances per loop step

    # ------------------------------------------------------------------
    # Data types
    # ------------------------------------------------------------------
    # Q element type. One of Float16 / BFloat16 / Float8E4M3FN. Must equal
    # kv_dtype — mixed Q/KV element types are not supported yet; enforced by
    # the guard in make_decode_config.
    q_dtype: type = Float16
    # K and V element type. One of Float16 / BFloat16 / Float8E4M3FN.
    kv_dtype: type = Float16
    # Output O element type. One of Float16 / BFloat16 / Float8E4M3FN.
    out_dtype: type = Float16
    # Accumulator type (BMM accumulators and softmax stats), always Float32
    # in the currently supported recipes.
    acc_dtype: type = Float32

    # ------------------------------------------------------------------
    # Software pipeline depths
    # ------------------------------------------------------------------
    # Q TMA pipeline depth. Q is reloaded only across persistent work tiles,
    # so a shallow ring is sufficient.
    q_stages: int = 2
    # K/V TMA pipeline depth. Deeper KV staging hides the long
    # GMEM→SMEM latency in the BMM1↔BMM2 chain.
    kv_stages: int = 4

    # ------------------------------------------------------------------
    # TMEM column counts
    # ------------------------------------------------------------------
    # Columns in TMEM for one BMM1 S = Q·Kᵀ accumulator (per softmax instance).
    # Equal to tile_size_q because SwapsMmaAb puts Q heads on the N axis.
    tmem_s_cols: int = 8  # per instance (tileSizeQ)
    # Columns in TMEM for the per-row softmax statistics (running max/sum and
    # scratch) owned by one softmax instance.
    tmem_stats_cols: int = 32  # per instance (softmax local stats)
    # P operand width for the staged-D256 Keeps TMEM overlay; zero for Swaps.
    tmem_p_cols: int = 0
    # Columns in TMEM for one O = Σ P·V accumulator (per O stage). Mirrors
    # tile_size_q for the same reason as tmem_s_cols.
    tmem_o_cols: int = 8  # per instance (tileSizeQ)
    # Number of O accumulator stages held simultaneously in TMEM so MMA can
    # start the next BMM2 while CorrectionTask rescales the previous one.
    o_stages: int = 2  # O0 and O1 in TMEM

    # ------------------------------------------------------------------
    # MMA atom shapes under SwapsMmaAb
    # ------------------------------------------------------------------
    # BMM1 computes S = K · Qᵀ: K provides
    # the M axis (tile_size_kv), Q provides the N axis (tile_size_q).
    mma_tile_m_bmm1: int = 128
    mma_tile_n_bmm1: int = 8
    # BMM2 computes O = V · Pᵀ: V provides M (headdim_v == headdim), P
    # provides N (tile_size_q), and the K-reduction axis is tile_size_kv.
    mma_tile_m_bmm2: int = 128
    mma_tile_n_bmm2: int = 8

    # ------------------------------------------------------------------
    # Warp specialization layout (4 warp groups × 4 warps = 16 warps total)
    # NOTE: please update `_active_warp_roles` after new roles are added.
    # ------------------------------------------------------------------
    # Softmax0Task: WG0 (warps 0–3) handles even K/V instances (K0/V0).
    softmax0_warp_idx: int = 0  # WG0: warps 0-3
    softmax0_num_warps: int = 4
    # Softmax1Task: WG1 (warps 4–7) handles odd K/V instances (K1/V1).
    softmax1_warp_idx: int = 4  # WG1: warps 4-7
    softmax1_num_warps: int = 4
    # CorrectionTask: WG2 (warps 8–11) rescales O across instances and writes
    # the epilogue to GMEM.
    correction_warp_idx: int = 8  # WG2: warps 8-11
    correction_num_warps: int = 4
    # MmaTask: a single warp in WG3 issues tcgen05 MMA instructions for BMM1/BMM2.
    mma_warp_idx: int = 12  # WG3: warp 12
    mma_num_warps: int = 1
    # LoadTask: a single warp in WG3 issues TMA loads for Q/K/V.
    load_warp_idx: int = 13  # WG3: warp 13
    load_num_warps: int = 1
    # PageTableTask (paged-KV only): warp 14 prefetches logical→physical
    # page IDs that LoadTask consumes when issuing the TMA copies.
    page_offsets_warp_idx: int = 14  # WG3: warp 14 for paged-KV page table prefetch
    page_offsets_num_warps: int = 1
    # SMEM pipeline depth for the prefetched page-offset table.
    page_offsets_stages: int = 6
    # SchedulerTask: under persistent scheduling, warp 13 runs the CLC tile
    # scheduler instead of doing TMA loads.
    scheduler_warp_idx: int = 13  # Persistent: warp 13
    scheduler_num_warps: int = 1
    # ClcLoadTask: under persistent scheduling, warp 15 issues the CLC
    # response loads.
    clc_load_warp_idx: int = 15
    # PaddingTask placement is derived after selecting the active task roles.
    # Each active warp group is compacted first, then its unused tail warps are
    # assigned to the corresponding padding task. Persistent layouts retain
    # all four warp groups even when the last group contains only padding.
    wg0_padding_warp_idx: int = 4
    wg0_padding_num_warps: int = 0
    wg1_padding_warp_idx: int = 8
    wg1_padding_num_warps: int = 0
    wg2_padding_warp_idx: int = 12
    wg2_padding_num_warps: int = 0
    wg3_padding_warp_idx: int = 16
    wg3_padding_num_warps: int = 0

    # ------------------------------------------------------------------
    # Task-local register allocation
    # ------------------------------------------------------------------
    # The full TileQ128 Keeps graph has the largest live softmax/correction
    # fragments and needs registers moved from its descriptor-only task group.
    # A long KV256 graph instead moves a smaller share from Softmax to its
    # heavier correction tail. Short KV256 loops avoid the fixed hand-off cost.
    @property
    def uses_task_register_reallocation(self) -> bool:
        return self.use_keeps_mma_ab and (
            self.tile_size_q == 128
            or (
                self.tile_size_q == 64
                and self.tile_size_kv == 256
                and self.total_kv_tiles >= KV_TILE_256_REGISTER_REALLOCATION_MIN_TILES
            )
        )

    @property
    def softmax_task_num_registers(self) -> int | None:
        if not self.uses_task_register_reallocation:
            return None
        return 176 if self.tile_size_kv == 256 else 184

    @property
    def correction_task_num_registers(self) -> int | None:
        if not self.uses_task_register_reallocation:
            return None
        return 104 if self.tile_size_kv == 256 else 88

    @property
    def mma_load_task_num_registers(self) -> int | None:
        return 56 if self.uses_task_register_reallocation else None

    # ------------------------------------------------------------------
    # SMEM allocation alignment
    # ------------------------------------------------------------------
    # Alignment (bytes) for SMEM tensor allocations; 1024 is required for
    # the swizzled TMA descriptors used here.
    stensor_align: int = 1024

    @property
    def smem_q_tile_bytes(self) -> int:
        """SMEM bytes for one Q tile (tile_size_q × headdim elements)."""
        return self.tile_size_q * self.headdim * self.q_dtype_bytes

    @property
    def q_tokens_per_cta(self) -> int:
        """Complete Q tokens packed into one grouped CTA."""
        return _q_tokens_per_cta(
            self.tile_size_q,
            self.heads_q_per_kv,
            self.groups_tokens_heads_q,
        )

    @property
    def num_q_ctas(self) -> int:
        """Return Q-CTA groups per ``(batch, KV head)`` launch tile.

        Shape-aware decode configs always carry a positive head ratio. Keep
        raw shape-less configs conservative so this scheduling predicate can
        never specialize geometry that has not been resolved yet.
        """
        if self.heads_q_per_kv <= 0 or self.max_seq_len_q <= 0:
            return max(self.max_seq_len_q, 1)
        q_geometry = make_q_tile_geometry(
            rows_per_cta=self.tile_size_q,
            heads_q_per_kv=self.heads_q_per_kv,
            groups_tokens_heads_q=self.groups_tokens_heads_q,
        )
        return max(q_geometry.num_q_ctas(self.max_seq_len_q), 1)

    @property
    def has_single_q_cta(self) -> bool:
        """Whether every physical split maps to one logical Q CTA."""
        return (
            self.heads_q_per_kv > 0 and self.max_seq_len_q > 0 and self.num_q_ctas == 1
        )

    @property
    def q_tma_rows_per_cta(self) -> int:
        """Q rows represented by the configured tensor-map box."""
        return _q_tma_rows_per_cta(
            self.tile_size_q,
            self.heads_q_per_kv,
            self.groups_tokens_heads_q,
        )

    @property
    def q_manual_padding_rows(self) -> int:
        """Grouped structural Q rows completed without a TMA instruction."""
        if self.groups_tokens_heads_q:
            return self.tile_size_q - self.q_tma_rows_per_cta
        return 0

    @property
    def smem_kv_tile_bytes(self) -> int:
        """SMEM bytes for one staged K or V tile."""
        return self.tile_size_kv * self.head_dim_kv_stage * self.kv_dtype_bytes

    @property
    def head_dim_kv_stage(self) -> int:
        """Head-dim slice loaded by one K/V stage."""
        return (
            self.head_dim_per_stage_kv
            if self.head_dim_per_stage_kv != 0
            else self.headdim
        )

    @property
    def num_head_dim_stages_kv(self) -> int:
        """Number of K/V head-dim stages needed to cover headdim."""
        return (self.headdim + self.head_dim_kv_stage - 1) // self.head_dim_kv_stage

    @property
    def tmem_o_cols_per_head_dim_stage(self) -> int:
        """TMEM O columns owned by one head-dim stage."""
        if self.use_keeps_mma_ab:
            return self.head_dim_kv_stage
        return self.tile_size_q

    @property
    def tmem_o_stage_cols(self) -> int:
        """TMEM columns reserved for one logical O pipeline stage."""
        if self.use_keeps_mma_ab:
            return self.tmem_o_cols
        head_dim_stage_cols = self.tmem_o_cols_per_head_dim_stage
        stages_per_tmem_row = max(TMEM_COLUMNS_PER_ROW // head_dim_stage_cols, 1)
        return head_dim_stage_cols * min(
            self.num_head_dim_stages_kv, stages_per_tmem_row
        )

    def swaps_head_dim_stage_tmem_offset(self, head_dim_stage_idx: int) -> int:
        """Return the Swaps O offset for one staged head-dimension slice."""
        if self.use_keeps_mma_ab or self.head_dim_per_stage_kv == 0:
            return 0
        stage_cols = self.tmem_o_cols_per_head_dim_stage
        stages_per_tmem_row = max(TMEM_COLUMNS_PER_ROW // stage_cols, 1)
        return (head_dim_stage_idx // stages_per_tmem_row) * TMEM_ROW_STRIDE + (
            head_dim_stage_idx % stages_per_tmem_row
        ) * stage_cols

    def swaps_o_chunk_tmem_offset(self, chunk_idx: int) -> int:
        """Return the Swaps O offset for one 64-column correction chunk."""
        if self.use_keeps_mma_ab or self.head_dim_per_stage_kv == 0:
            return chunk_idx * TMEM_ROW_STRIDE
        chunks_per_head_dim_stage = max(self.head_dim_kv_stage // 64, 1)
        head_dim_stage_idx, chunk_idx_in_stage = divmod(
            chunk_idx, chunks_per_head_dim_stage
        )
        return (
            self.swaps_head_dim_stage_tmem_offset(head_dim_stage_idx)
            + chunk_idx_in_stage * TMEM_ROW_STRIDE
        )

    def pv_head_dim_stage_tmem_offset(self, head_dim_stage_idx: int) -> int:
        """Return the staged PV destination offset in one logical O tile."""
        if self.head_dim_per_stage_kv == 0:
            return 0
        if self.use_keeps_mma_ab:
            return head_dim_stage_idx * self.head_dim_kv_stage
        return self.swaps_head_dim_stage_tmem_offset(head_dim_stage_idx)

    @property
    def smem_p_tile_bytes(self) -> int:
        """P stored in SMEM for the SwapsMmaAb BMM2 B operand."""
        return self.tile_size_kv * self.tile_size_q * self.q_dtype_bytes

    @property
    def tmem_total_cols(self) -> int:
        """Sum of TMEM columns used by the kernel: 2× S, 2× softmax stats,
        and o_stages× O accumulators (the factor 2 is the two softmax
        instances K0/V0 and K1/V1)."""
        if self.use_keeps_mma_ab:
            if self.num_insts_kv == 1:
                return (
                    self.tmem_s_cols
                    + self.tmem_stats_cols
                    + self.tmem_p_cols
                    + self.tmem_o_stage_cols * self.o_stages
                )
            return (
                2 * self.tmem_s_cols
                + (
                    2 * self.tmem_stats_cols
                    if self.keeps_separates_tmem_s_and_stats
                    else 0
                )
                + self.tmem_o_stage_cols * self.o_stages
            )
        return (
            2 * self.tmem_s_cols
            + 2 * self.tmem_stats_cols
            + self.tmem_o_stage_cols * self.o_stages
        )

    @property
    def tmem_alloc_cols(self) -> int:
        """TMEM allocation rounded up to a power of two and at least 32,
        matching the hardware TMEM allocator's granularity."""
        return max(32, 1 << (self.tmem_total_cols - 1).bit_length())

    @property
    def threads_per_cta(self) -> int:
        """CTA thread count implied by the configured task warp layout."""
        return (
            max(
                role.preferred_warp_idx + role.num_warps
                for role in _active_warp_roles(self)
            )
            * 32
        )

    # ------------------------------------------------------------------
    # Inferred dtype attributes (derived from q_dtype / kv_dtype / out_dtype)
    # ------------------------------------------------------------------
    @property
    def q_dtype_bytes(self) -> int:
        """Byte width of one Q element (fp16/bf16=2, e4m3=1).

        Also the byte width used by anything that feeds the MMA on the Q/S/P
        side (softmax stats, P tile, MMA operand descriptors)."""
        return 1 if self.q_dtype == Float8E4M3FN else 2

    @property
    def kv_dtype_bytes(self) -> int:
        """Byte width of one K/V element (fp16/bf16=2, e4m3=1)."""
        return 1 if self.kv_dtype == Float8E4M3FN else 2

    @property
    def o_dtype_bytes(self) -> int:
        """Byte width of one O element (fp16/bf16=2, e4m3=1)."""
        return 1 if self.out_dtype == Float8E4M3FN else 2

    @property
    def acc_dtype_bytes(self) -> int:
        """Byte width of one accumulator element (fp32=4)."""
        return 4 if self.acc_dtype == Float32 else 2

    @property
    def use_bf16_qkv(self) -> bool:
        """Whether Q/K/V use BF16 storage and MMA inputs."""
        return self.kv_dtype == BFloat16

    @property
    def use_bf16_output(self) -> bool:
        """Whether final O is stored as BF16."""
        return self.out_dtype == BFloat16

    @property
    def use_fp8_qkv(self) -> bool:
        """fp8 (E4M3) Q/K/V path: switches MMA kind and P-quantization."""
        return self.kv_dtype == Float8E4M3FN

    @property
    def use_fp8_output(self) -> bool:
        """Whether final O is stored as FP8 E4M3."""
        return self.out_dtype == Float8E4M3FN

    @property
    def use_bf16_separate_partial_o(self) -> bool:
        """Whether normalized separate-GMEM partial O uses BF16 storage.

        FP16 output keeps FP16 partials to preserve its mantissa/error
        envelope. BF16 and FP8 output use BF16 partials for the wider range.
        """
        return self.use_bf16_output or self.use_fp8_output

    @property
    def supports_reduction_dtypes(self) -> bool:
        """Whether split reduction supports the configured input/output dtypes."""
        return (
            self.q_dtype in (Float16, BFloat16)
            and self.out_dtype in (Float16, BFloat16)
        ) or (
            self.q_dtype == Float8E4M3FN and self.out_dtype in (Float16, Float8E4M3FN)
        )

    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------
    # Enable persistent scheduling: work tiles are fetched from a CLC response
    # queue at runtime instead of mapping one tile per CTA in the launch grid.
    # Mutually exclusive with split-KV mode.
    use_persistent_scheduler: bool = False
    # Split-KV: split the K-sequence across several CTAs that produce partial
    # O/stats, with a GMEM reduction epilogue.
    use_split_kv: bool = False
    # Split-KV fanout: number of CTAs that cooperate on one (batch, head_kv)
    # when use_split_kv is enabled. The launcher picks this based on SM count.
    splits_kv: int = 1
    # Upper bound of splits_kv across all (batch, head_kv) groups,
    # used to size the per-launch partial-O / partial-stats / counter
    # scratch GMEM buffers.
    max_splits_kv: int = 1
    # Paged-KV cache layout: K/V live in fixed-size pages and the kernel
    # follows a logical→physical page index table per request.
    use_paged_kv: bool = False
    # Page size (tokens per page) when use_paged_kv is enabled. Must be one of
    # 16 / 32 / 64 / 128 and must divide the 128-token KV tile.
    num_tokens_per_page: int = 32
    # Maximum number of pages per (batch, head_kv) — sizes the page index
    # table stride.
    max_num_pages_per_seq_kv: int = 1
    # Enable sliding-window-causal masking: KV tiles fully outside
    # [seq_len_kv − attention_window_size, seq_len_kv) are skipped entirely
    # rather than masked element-wise.
    use_sliding_window_causal: bool = False
    # Window size W: each Q attends to the last W KV tokens. Ignored unless
    # use_sliding_window_causal is enabled.
    attention_window_size: int = 0
    # Fixed-launch seq_len_kv after sliding-window trimming (constexpr seen
    # by the kernel; populated by _configure_static_sliding_window). Falls
    # back to the launch seq_len_kv when the bias-TMA path is not used.
    static_seq_len_kv: int = 0
    # Number of leading full KV tiles skipped by the sliding window for the
    # fixed-launch path. Added to runtime tile indices in resource code.
    static_num_skipped_kv_tiles: int = 0
    # Token offset where the sliding window starts (mod tile boundaries),
    # used for partial-tile masking on the leading edge of the window.
    static_window_start_idx: int = 0
    # When set, the TMA descriptors are pre-biased to point at the first
    # in-window token, so the kernel doesn't have to add the skipped-tiles
    # offset at runtime. Trades launch-time flexibility for codegen
    # simplicity in the fixed-length case.
    use_static_sliding_kv_tma_bias: bool = False
    # Attention-sinks: add a per-head "sink" exponent to the softmax
    # denominator (extra logit that absorbs probability mass). Requires the
    # `attention_sinks` tensor argument at launch.
    use_attention_sinks: bool = False
    # Optional profile and reduction knobs selected by launcher policy or tests.
    use_keeps_mma_ab: bool = False
    # Nonzero means each K/V stage covers only this many head-dim columns.
    # H256 SwapsMmaAb uses 128-column stages to keep TMA and TMEM layouts valid.
    head_dim_per_stage_kv: int = 0
    # Ordered softmax barrier mode: 0 disables, 1 auto-enables for supported
    # profiles, and 2 forces the barrier path for targeted validation.
    ordered_softmax_barrier_mode: int = 0
    # Named barrier slot used by ordered softmax. Slots below 8 are already
    # consumed by the main pipeline resources in these schedules.
    softmax_order_barrier_id: int = 8
    # Both softmax task groups participate: 8 warps * 32 lanes.
    softmax_order_barrier_threads: int = 256
    use_cluster_smem_reduction: bool = False
    use_separate_reduction_kernel: bool = False
    # Compile-time attention-mask selection. Public APIs normalize the string
    # names in MASK_TYPES to the integer constants used by CuTe DSL branches.
    mask_type: int = DENSE

    # ------------------------------------------------------------------
    # Derived resource footprints
    # ------------------------------------------------------------------
    @property
    def smem_q_tile_elements(self) -> int:
        """Return Q elements in one staged SMEM tile."""
        return self.smem_q_tile_bytes // self.q_dtype_bytes

    @property
    def use_parallel_separate_reduction(self) -> bool:
        """Use the production standalone reducer for separate-GMEM modes."""
        return self.use_separate_reduction_kernel

    @property
    def use_parallel_separate_reduction_pdl(self) -> bool:
        """Order every production standalone reducer through PDL."""
        return self.use_separate_reduction_kernel

    @property
    def parallel_reduction_padded_splits(self) -> int:
        """Return the power-of-two split capacity of the reducer cluster."""
        if self.max_splits_kv <= 1:
            return 1
        return 1 << (self.max_splits_kv - 1).bit_length()

    @property
    def use_compact_parallel_reduction(self) -> bool:
        """Use one wide CTA when S2-S4 cannot amortize a CTA cluster."""
        return self.use_separate_reduction_kernel and 2 <= self.max_splits_kv <= 4

    @property
    def parallel_reduction_cluster_size(self) -> int:
        """Return reducer CTAs for the padded split capacity."""
        if (
            not self.use_separate_reduction_kernel
            or self.use_compact_parallel_reduction
        ):
            return 1
        return {
            8: 1,
            16: 2,
            32: 4,
            64: 8,
            128: 16,
        }.get(self.parallel_reduction_padded_splits, 1)

    @property
    def parallel_reduction_splits_per_cta(self) -> int:
        """Return split slots owned by each reducer CTA."""
        if self.use_compact_parallel_reduction:
            return self.max_splits_kv
        return (
            self.parallel_reduction_padded_splits
            // self.parallel_reduction_cluster_size
        )

    @property
    def parallel_reduction_threads_per_cta(self) -> int:
        """Return the thread count selected by the reducer schedule."""
        if self.use_compact_parallel_reduction:
            return REDUCTION_THREADS_PER_CTA
        return PARALLEL_REDUCTION_THREADS_PER_CTA

    @property
    def parallel_reduction_bytes_per_slice(self) -> int:
        """Return output bytes covered by one independent reducer group."""
        if self.use_compact_parallel_reduction:
            return REDUCTION_BYTES_PER_SLICE
        return PARALLEL_REDUCTION_BYTES_PER_SLICE

    @property
    def smem_kv_tile_elements(self) -> int:
        """Return K or V elements in one staged SMEM tile."""
        return self.smem_kv_tile_bytes // self.kv_dtype_bytes

    @property
    def num_softmax_scale_groups(self) -> int:
        """Return independent max/sum groups tracked by each softmax lane."""
        if self.use_keeps_mma_ab:
            return 1
        return max(self.tile_size_q // 4, 1)

    @property
    def num_s_regs_per_thread(self) -> int:
        """Return all score values owned by each softmax lane."""
        if self.use_keeps_mma_ab:
            if self.tile_size_q == 128:
                return self.tile_size_kv
            return self.tile_size_kv // 2
        return self.num_softmax_scale_groups * 4

    @property
    def softmax_score_fragment_regs(self) -> int:
        """Return the maximum score fragment kept live in registers.

        KV256 owns 128 score values per lane but streams them as four native
        32-register LDTM atoms. Other profiles retain their complete score
        fragment, so this property is intentionally distinct from
        ``num_s_regs_per_thread`` (the total logical ownership).
        """
        if self.tile_size_kv == 256:
            return 32
        return self.num_s_regs_per_thread

    @property
    def num_softmax_score_fragments(self) -> int:
        """Return score fragments used to cover one logical KV tile."""
        return self.num_s_regs_per_thread // self.softmax_score_fragment_regs

    @property
    def num_packed_p_regs(self) -> int:
        """Return packed P registers stored by each softmax producer lane."""
        if self.use_keeps_mma_ab:
            values_per_reg = (
                FP8_VALUES_PER_REG if self.use_fp8_qkv else FP16_VALUES_PER_REG
            )
            return max(self.num_s_regs_per_thread // values_per_reg, 1)
        q_repeats = max(self.tile_size_q // Q_REPETITION_GROUP_HEADS, 1)
        regs_per_repeat = (
            FP8_P_PACKED_REGS_PER_Q_REPEAT
            if self.use_fp8_qkv
            else FP16_P_PACKED_REGS_PER_Q_REPEAT
        )
        return regs_per_repeat * q_repeats

    @property
    def tmem_p_cols_per_inst(self) -> int:
        """Return the TMEM-P columns owned by one K/V instance.

        Two-instance Keeps publishes one packed row per producer lane, with one
        packed 32-bit register slot per TMEM column. Therefore its footprint is
        ``num_s_regs_per_thread / values_per_reg == num_packed_p_regs``, rather
        than a function of the complete logical KV tile width. Q128/KV128 and
        Q64/KV256 both own 128 packed 16-bit P values per lane and need 64
        columns per instance.
        """
        if self.uses_two_inst_tmem_p:
            return self.num_packed_p_regs
        return self.tmem_p_cols

    @property
    def num_fp8_output_regs(self) -> int:
        """Return packed FP8 output registers owned by each correction lane."""
        return max(
            (self.tile_size_q * self.headdim) // FP8_OUTPUT_ELEMENTS_PER_REG_GROUP,
            1,
        )

    @property
    def num_fp16_output_regs(self) -> int:
        """Return packed 16-bit output registers owned by each correction lane."""
        return max(
            (self.tile_size_q * self.headdim) // FP16_OUTPUT_ELEMENTS_PER_REG_GROUP,
            1,
        )

    @property
    def keeps_output_f32_regs(self) -> int:
        """Return FP32 O registers owned by one Keeps correction lane."""
        if self.tile_size_q == 128 or self.tile_size_kv == 256:
            return self.headdim
        return self.headdim // 2

    @property
    def correction_barrier_threads(self) -> int:
        """Return named-barrier participants for the correction task."""
        return self.correction_num_warps * WARP_THREADS

    @property
    def keeps_p_smem_vector_elements(self) -> int:
        """Return P elements in one aligned 16-byte SMEM store."""
        return 16 // self.q_dtype_bytes

    @property
    def static_local_kv_tiles(self) -> int:
        """Return static KV tiles assigned to one split CTA."""
        if not self.use_split_kv:
            return self.total_kv_tiles
        tiles_per_cta_group = self.splits_kv * self.num_insts_kv
        num_groups = (
            self.total_kv_tiles + tiles_per_cta_group - 1
        ) // tiles_per_cta_group
        return max(self.num_insts_kv, num_groups * self.num_insts_kv)

    @property
    def inferred_kv_stages(self) -> int:
        """Return the deepest K/V ring that fits the shared-memory budget."""
        q_dtype_bits = 8 if self.q_dtype == Float8E4M3FN else 16
        q_row_bytes = (
            (q_dtype_bits * self.headdim // BITS_PER_BYTE + Q_ROW_ALIGNMENT_BYTES - 1)
            // Q_ROW_ALIGNMENT_BYTES
        ) * Q_ROW_ALIGNMENT_BYTES
        q_tile_kib = q_row_bytes * self.tile_size_q // BYTES_PER_KIB
        kv_budget_kib = min(
            MAX_KV_STAGE_SMEM_KIB,
            TOTAL_SMEM_BUDGET_KIB - q_tile_kib * self.q_stages,
        )
        kv_stage_head_dim = self.head_dim_per_stage_kv or self.headdim
        kv_tile_bits = q_dtype_bits * self.tile_size_kv * kv_stage_head_dim
        return max(
            1,
            kv_budget_kib * BYTES_PER_KIB * BITS_PER_BYTE // kv_tile_bits,
        )

    def validate_dtypes(self) -> None:
        """Validate decode input, output, and accumulator dtypes."""
        for name, dtype, supported in (
            ("q_dtype", self.q_dtype, SUPPORTED_IO_DTYPES),
            ("kv_dtype", self.kv_dtype, SUPPORTED_IO_DTYPES),
            ("out_dtype", self.out_dtype, SUPPORTED_IO_DTYPES),
            ("acc_dtype", self.acc_dtype, SUPPORTED_ACC_DTYPES),
        ):
            if dtype not in supported:
                raise ValueError(f"Unsupported {name}: {dtype}")
        if self.q_dtype != self.kv_dtype:
            raise ValueError(
                f"q_dtype ({self.q_dtype}) != kv_dtype ({self.kv_dtype}): "
                "mixed Q/KV element types are not supported"
            )

    def validate_boolean_fields(self) -> None:
        """Require every boolean config field to carry a real Python bool."""
        for name, config_field in self.__dataclass_fields__.items():
            value = getattr(self, name)
            if config_field.type in (bool, "bool") and not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool, got {type(value).__name__}")

    def validate_paged_kv_staging_config(self) -> None:
        """Validate the selected dense or sparse paged-KV staging geometry."""

        if not self.use_paged_kv:
            raise ValueError("paged-KV staging requires use_paged_kv=True")
        validate_page_size(self.num_tokens_per_page)

        if self.use_block_sparse:
            if self.tile_size_kv not in (128, 256):
                raise ValueError(
                    "paged block-sparse supports only KV128 or KV256 routes"
                )
            atom_size = _block_sparse_kv_atom_size(self.kv_block_size)
            if atom_size > self.num_tokens_per_page:
                raise ValueError(
                    "paged block-sparse atom size must not exceed page size"
                )
            if self.num_tokens_per_page % atom_size != 0:
                raise ValueError(
                    "paged block-sparse page size must be divisible by atom size"
                )
            if self.tile_size_kv == 256 and self.num_tokens_per_page not in (
                64,
                128,
            ):
                raise ValueError(
                    "paged block-sparse KV256 routes require page size 64 or 128"
                )
            return

        if self.tile_size_kv <= 0:
            raise ValueError("paged-KV tile_size_kv must be positive")
        if self.tile_size_kv % self.num_tokens_per_page != 0:
            raise ValueError(
                "paged-KV num_tokens_per_page must divide tile_size_kv exactly"
            )
        pages_per_tile = self.tile_size_kv // self.num_tokens_per_page
        if pages_per_tile not in (1, 2, 4, 8, 16):
            raise ValueError(
                "paged-KV staging supports 1, 2, 4, 8, or 16 pages per KV tile"
            )
        if self.page_offsets_num_warps != 1:
            raise ValueError(
                "paged-KV page-offset staging requires exactly one producer warp"
            )

    def validate_block_sparse_profile(self, *, heads_q_per_kv: int) -> None:
        """Validate the qualified host profile for block-sparse."""
        if not self.use_block_sparse:
            if self.use_parallel_sparse_kv_loads:
                raise ValueError(
                    "sparse execution policy requires block-sparse attention"
                )
            return

        if not (self.q_dtype == self.kv_dtype == self.out_dtype):
            raise ValueError("block-sparse requires q_dtype == kv_dtype == out_dtype")
        if self.q_dtype not in (Float16, BFloat16):
            raise ValueError(
                "block-sparse supports only matching Float16 or BFloat16 IO"
            )

        kv_block_size = _validate_sparse_kv_block_size(self.kv_block_size)
        selected_q_tile = _select_block_sparse_q_tile_size(
            q_block_size=self.q_block_size,
            heads_q_per_kv=heads_q_per_kv,
            kv_block_size=kv_block_size,
        )
        if self.use_paged_kv:
            self.validate_paged_kv_staging_config()
        if not self.groups_tokens_heads_q:
            raise ValueError("block-sparse requires groups_tokens_heads_q=True")
        if self.headdim != 128:
            raise ValueError("block-sparse requires headdim=128")
        if self.tile_size_kv not in (128, 256):
            raise ValueError("block-sparse requires tile_size_kv in (128, 256)")
        if self.tile_size_kv == 256 and not (
            self.tile_size_q == 64
            and kv_block_size % 64 == 0
            and not self.use_parallel_sparse_kv_loads
        ):
            raise ValueError(
                "block-sparse tile_size_kv=256 requires the Q64 16-bit Keeps "
                "profile with coarse KV blocks and one load task"
            )
        if self.tile_size_q != selected_q_tile:
            raise ValueError(
                "block-sparse tile_size_q must match its grouped-Q geometry"
            )
        uses_fine_q_tile = selected_q_tile < 64
        uses_fine_kv_blocks = kv_block_size < 64
        if uses_fine_q_tile:
            if self.use_keeps_mma_ab:
                raise ValueError("fine block-sparse Q tiles require SwapsMmaAb")
        elif not self.use_keeps_mma_ab:
            raise ValueError("coarse block-sparse Q tiles require KeepsMmaAb")
        if uses_fine_kv_blocks and not uses_fine_q_tile:
            raise ValueError("fine KV blocks require a SwapsMmaAb Q tile")
        if self.use_parallel_sparse_kv_loads and (
            self.use_keeps_mma_ab
            or kv_block_size not in (8, 16)
            or self.num_insts_kv != 2
        ):
            raise ValueError(
                "parallel sparse K/V loads require two-instance SwapsMmaAb "
                "with KV block size 8 or 16"
            )
        if self.use_variable_seqlens_q:
            raise ValueError("block-sparse does not support variable-Q sequences")
        if self.use_sliding_window_causal:
            raise ValueError("block-sparse does not support sliding window attention")
        if self.use_attention_sinks:
            raise ValueError("block-sparse does not support attention sinks")
        if (
            self.use_split_kv
            or self.use_cluster_smem_reduction
            or self.use_separate_reduction_kernel
        ):
            raise ValueError("block-sparse does not support split-KV reduction")

    def compile_signature(self) -> tuple[tuple[str, object], ...]:
        """Key and reconstruct the batch-dynamic callable in the decode cache.

        The public planner can reuse one compiled topology across batch sizes.
        Keeping the complete static config in the cache key prevents a callable
        compiled for one scheduler, reduction, or tile layout from being reused
        by another.
        """

        return tuple(
            (config_field.name, getattr(self, config_field.name))
            for config_field in fields(self)
        )

    @property
    def uses_q_desc_ref(self) -> bool:
        """Whether QK derives Q's descriptor from shared resource state."""
        return self.use_variable_seqlens_q and self.use_persistent_scheduler

    @property
    def has_odd_kv_tail(self) -> bool:
        """Whether the K/V tile count leaves an unpaired tail instance."""
        return (self.total_kv_tiles % self.num_insts_kv) != 0

    @property
    def uses_uniform_causal_mask(self) -> bool:
        """Whether every Q row in a CTA shares one causal right bound."""
        return self.mask_type == CAUSAL and (
            not self.groups_tokens_heads_q
            or self.q_tokens_per_cta == 1
            or self.uses_guarded_fixed_q1_grouped_keeps
        )

    @property
    def uses_per_row_causal_mask(self) -> bool:
        """Whether grouped Q rows require distinct causal right bounds."""
        return (
            self.mask_type == CAUSAL
            and self.groups_tokens_heads_q
            and self.q_tokens_per_cta > 1
            and not self.uses_guarded_fixed_q1_grouped_keeps
        )

    @property
    def _grouped_keeps_profile_key(self) -> _GroupedKeepsProfileKey:
        """Return the dtype, shape, and staging key used by Keeps recipe tables."""
        return (
            self.q_dtype,
            self.kv_dtype,
            self.out_dtype,
            self.headdim,
            self.head_dim_per_stage_kv,
            self.num_insts_kv,
            self.o_stages,
        )

    @property
    def uses_guarded_fixed_q1_grouped_keeps(self) -> bool:
        """Whether inactive grouped rows are excluded solely at output stores.

        The validated paged FP8 Keeps profiles pack two or four structural Q
        tokens into TileQ64/128 while the public decode problem has exactly one
        logical token.  QK/PV rows are independent and every direct, split
        scratch, and final-reduction store already checks row validity, so the
        inactive score rows do not need per-KV-tile suppression.
        """
        profile = self._grouped_keeps_profile_key
        return (
            self.use_keeps_mma_ab
            and self.groups_tokens_heads_q
            and self.max_seq_len_q == 1
            and not self.use_variable_seqlens_q
            and self.q_manual_padding_rows == 0
            and profile in _GROUPED_KEEPS_PAGED_FP8_PROFILES
            and self.supports_grouped_keeps
            # The staged TileQ64 one-instance TMEM-P schedule keeps per-row
            # score masking because its generated code is sensitive to that
            # control-flow shape.
            and not (self.uses_staged_one_inst_tmem_p and self.tile_size_q == 64)
        )

    @property
    def uses_guarded_grouped_keeps_output_rows(self) -> bool:
        """Whether inactive Keeps rows can be discarded only at publication.

        Keeps MMA, softmax, and PV rows are independent. For a fixed grouped-Q
        launch, structural padding and a partial final token group therefore
        cannot affect a valid row; direct output, split scratch, and reduction
        publication already guard row validity. Keep the staged one-instance
        TileQ64/D256 exception on its per-row score-mask path because its
        generated schedule is sensitive to that control-flow shape.
        """
        profile = self._grouped_keeps_profile_key
        return (
            self.use_keeps_mma_ab
            and self.groups_tokens_heads_q
            and not self.use_variable_seqlens_q
            and profile in _GROUPED_KEEPS_PAGED_FP8_PROFILES
            and self.supports_grouped_keeps
            and not (self.uses_staged_one_inst_tmem_p and self.tile_size_q == 64)
        )

    @property
    def has_static_dense_full_kv_tiles(self) -> bool:
        """Whether static dense KV avoids masking and runtime tile remapping."""
        return (
            not self.use_split_kv
            and not self.use_block_sparse
            and self.mask_type == DENSE
            and not self.use_sliding_window_causal
            and self.static_seq_len_kv != 0
            and (self.static_seq_len_kv % self.tile_size_kv) == 0
        )

    @property
    def uses_ordered_softmax_barrier(self) -> bool:
        """Whether this profile selects the ordered P0/P1 softmax barrier."""
        if self.tile_size_kv == 256:
            # KV256 uses independent four-stage P-fragment pipelines. Ordering
            # the two softmax groups would serialize fragment production and
            # defeat the intended P/PV overlap.
            return False
        if self.ordered_softmax_barrier_mode == 2:
            return True
        return self.ordered_softmax_barrier_mode == 1 and (
            self.headdim == 128 and self.tile_size_q in (32, 64, 128)
        )

    @property
    def ordered_softmax_early_release(self) -> bool:
        """Whether the P0/P1 baton is released at TMEM store issue.

        For two-instance TMEM-P the partner softmax group's publication
        touches only its own registers and TMEM region, so it does not need
        this group's store drain, async fence, or pipeline commit. Releasing
        at TMEM store issue overlaps that tail with the partner's wakeup while the
        exp2 phases stay serialized on the shared MUFU pipes.
        """
        return self.uses_ordered_softmax_barrier and self.uses_two_inst_tmem_p

    @property
    def resolved_softmax_order_barrier_threads(self) -> int:
        """Return the participant count for ordered softmax barriers."""
        heads_q_per_kv = self.heads_q_per_kv or self.tile_size_q
        if self.use_keeps_mma_ab:
            return (self.softmax0_num_warps + self.softmax1_num_warps) * WARP_THREADS
        if (
            self.headdim == 128
            and self.tile_size_q == 32
            and self.tile_size_kv == 128
            and heads_q_per_kv == 32
            and not self.groups_tokens_heads_q
        ):
            return 128
        return self.softmax_order_barrier_threads

    @property
    def uses_nontrivial_grouped_q_layout(self) -> bool:
        """Whether grouped Q differs from one complete token per CTA."""
        return self.groups_tokens_heads_q and (
            self.q_tokens_per_cta > 1 or self.q_manual_padding_rows > 0
        )

    @property
    def q_tiles_are_full(self) -> bool:
        """Whether every launched Q CTA owns all of its MMA rows."""
        if self.use_variable_seqlens_q:
            return False
        if self.uses_nontrivial_grouped_q_layout:
            return (
                self.q_manual_padding_rows == 0
                and self.max_seq_len_q % self.q_tokens_per_cta == 0
            )
        if self.max_seq_len_q > 1:
            return self.heads_q_per_kv % self.tile_size_q == 0
        return self.heads_q_per_kv == self.tile_size_q

    @property
    def q_tiles_need_row_mask(self) -> bool:
        """Whether softmax must suppress inactive Q rows."""
        if self.use_variable_seqlens_q:
            return True
        if self.uses_nontrivial_grouped_q_layout or self.max_seq_len_q > 1:
            return not self.q_tiles_are_full
        return False

    @property
    def q_score_rows_need_mask(self) -> bool:
        """Whether inactive Q rows must be suppressed in every score tile."""
        return (
            self.q_tiles_need_row_mask
            and not self.uses_guarded_grouped_keeps_output_rows
        )

    @property
    def uses_q_cta_sliding_union(self) -> bool:
        """Whether the causal/window KV union depends on the logical Q CTA."""
        return (
            self.mask_type == CAUSAL
            and self.max_seq_len_q > 1
            and not self.has_single_q_cta
        )

    @property
    def uses_runtime_q_kv_union(self) -> bool:
        """Whether task/resource KV geometry must retain runtime Q metadata.

        Multiple causal Q CTAs have distinct right bounds. A multi-token
        sliding-window launch also retains the runtime path even when it fits
        in one Q CTA, because the existing static window-prefix metadata is
        specialized only for SQ1.
        """
        return self.uses_q_cta_sliding_union or (
            self.use_sliding_window_causal and self.max_seq_len_q > 1
        )

    @property
    def uses_tmem_p(self) -> bool:
        """Whether Keeps P is materialized in TMEM for BMM2."""
        return self.uses_staged_one_inst_tmem_p or self.uses_two_inst_tmem_p

    @property
    def uses_staged_one_inst_tmem_p(self) -> bool:
        """Whether P uses the double-buffered D256 TMEM overlay."""
        return (
            self.use_keeps_mma_ab
            and self.headdim == 256
            and self.head_dim_per_stage_kv == 128
            and self.num_insts_kv == 1
            and self.o_stages == 1
        )

    @property
    def uses_two_inst_tmem_p(self) -> bool:
        """Whether a two-instance Keeps profile uses the TMEM-P overlay.

        Q128/KV128 and sparse Q64/KV128 publish a complete packed-P row per
        pipeline token. Q64/KV256 uses the same S-to-P aliasing contract but
        streams four independently ready K32 fragments. Dense Q64/KV128 keeps
        the base kernel's faster SMEM-P cadence.
        """
        # Two-instance Keeps keeps stats outside S, so both static and persistent
        # work tiles can overlay P on the consumed S instance. The split K/V
        # schedule preserves same-instance PV -> QK order, while Softmax delays
        # S release through TMEM store completion and the P-pipeline commit.
        return (
            self.use_keeps_mma_ab
            and (
                (self.tile_size_q == 128 and self.tile_size_kv == 128)
                or (self.tile_size_q == 64 and self.tile_size_kv == 256)
                or (
                    self.use_block_sparse
                    and self.tile_size_q == 64
                    and self.tile_size_kv == 128
                )
            )
            and self.head_dim_per_stage_kv == 0
            and self.num_insts_kv == 2
            and self.o_stages == 2
            and self.tmem_total_cols <= 512
        )

    @property
    def streams_tmem_p_fragments(self) -> bool:
        """Whether P is published as independently ready TMEM fragments."""
        return self.uses_two_inst_tmem_p and self.num_softmax_score_fragments > 1

    @property
    def matches_kv256_task_topology(self) -> bool:
        """Whether task roles match KV256's validated 16-warp layout."""
        return all(
            getattr(self, field) == expected
            for field, expected in _KV_TILE_256_TASK_TOPOLOGY_DEFAULTS.items()
        )

    @property
    def uses_rotating_kv256_exchange(self) -> bool:
        """Whether this profile selects KV-ring scratch for correction.

        Persistent direct output can overlap the next work tile's first two
        K loads with correction by placing its exchange in the third, drained
        KV stage. Split-KV and attention sinks retain the fixed exchange because
        their tail storage and lifetime differ from direct output.
        """
        selects_persistent_kv256 = (
            self.streams_tmem_p_fragments
            and self.tile_size_q == 64
            and self.tile_size_kv == 256
            and self.use_persistent_scheduler
        )
        if not selects_persistent_kv256:
            return False

        has_rotating_kv_ring = (
            self.num_head_dim_stages_kv == 1
            and self.kv_stages == KV_TILE_256_SHARED_FIFO_STAGES
            and self.load_num_warps == 1
        )
        has_direct_output_lifetime = not (self.use_split_kv or self.use_attention_sinks)
        return has_rotating_kv_ring and has_direct_output_lifetime

    @property
    def keeps_separates_tmem_s_and_stats(self) -> bool:
        """Whether two-instance Keeps has room for standalone stats tiles."""
        if not (
            self.use_keeps_mma_ab
            and self.use_fp8_qkv
            and self.tile_size_kv == 128
            and self.head_dim_per_stage_kv == 0
            and self.num_insts_kv == 2
            and self.o_stages == 2
        ):
            return False
        return (
            2 * self.tmem_s_cols
            + 2 * self.tmem_stats_cols
            + self.tmem_o_stage_cols * self.o_stages
            <= 512
        )

    @property
    def keeps_stats_via_smem(self) -> bool:
        """Whether Keeps softmax->correction stats travel through SMEM.

        When two-instance Keeps cannot give the stats payload standalone
        TMEM columns (S/stats/O exceed the 512-column budget), the stats
        slot aliases S and a stats-done credit pipeline must gate every QK
        re-issue on correction's TMEM stats read. Routing the small per-row
        payload through an SMEM ring removes that MMA-side serialization;
        the existing softmax-local pipelines already order the handoff.
        """
        return self.use_keeps_mma_ab and not self.keeps_separates_tmem_s_and_stats

    @property
    def keeps_loop_correction_chunk_regs(self) -> int:
        """Return FP32 registers corrected by one Keeps TMEM pair."""
        if self.tile_size_q == 64:
            return 32
        return 32 if self.tile_size_q == 128 and self.headdim >= 256 else 8

    @property
    def keeps_loop_correction_stage_layout(self) -> tuple[tuple[int, int, int], ...]:
        """Return ``(TMEM offset, half split, chunk count)`` for each O slice."""
        stage_cols = self.head_dim_kv_stage
        lane_regs_per_stage = (
            stage_cols
            if self.tile_size_q == 128 or self.tile_size_kv == 256
            else stage_cols // 2
        )
        chunk_regs = self.keeps_loop_correction_chunk_regs
        assert stage_cols % 2 == 0
        assert lane_regs_per_stage % chunk_regs == 0
        return tuple(
            (
                stage_idx * stage_cols,
                stage_cols // 2,
                lane_regs_per_stage // chunk_regs,
            )
            for stage_idx in range(self.num_head_dim_stages_kv)
        )

    @property
    def fp8_copy_can_use_full_tile_fast_path(self) -> bool:
        """Whether every correction wave owns only in-bounds FP8 bytes."""
        correction_copy_bytes = self.correction_num_warps * WARP_THREADS * 16
        tile_bytes = self.tile_size_q * self.headdim
        return self.q_tiles_are_full and tile_bytes % correction_copy_bytes == 0

    @property
    def can_use_cluster_smem_reduction(self) -> bool:
        """Whether the configured split profile is eligible for cluster reduction."""
        ungrouped_q_layout = (
            not self.groups_tokens_heads_q
            and not self.use_variable_seqlens_q
            and self.max_seq_len_q == 1
        )
        grouped_q_layout = (
            self.groups_tokens_heads_q
            and self.heads_q_per_kv > 0
            and self.tile_size_q >= self.heads_q_per_kv
        )
        return (
            self.use_split_kv
            and not self.use_separate_reduction_kernel
            and not self.use_persistent_scheduler
            and not self.use_keeps_mma_ab
            and self.headdim in (64, 128, 256)
            and self.tile_size_q in (8, 16, 32)
            and (ungrouped_q_layout or grouped_q_layout)
            and self.max_splits_kv >= self.splits_kv >= 2
        )

    @property
    def supports_cluster_smem_reduction(self) -> bool:
        """Whether cluster reduction is both selected and eligible."""
        return self.use_cluster_smem_reduction and self.can_use_cluster_smem_reduction

    @property
    def split_reduction_slice_bytes(self) -> int:
        """Return the minimum byte range assigned to one reducer CTA."""
        return self.correction_barrier_threads * SPLIT_REDUCTION_VECTOR_BYTES_PER_THREAD

    @property
    def split_reduction_rows_per_slice(self) -> int:
        """Return complete partial-O rows covered by one reducer slice."""
        row_bytes = self.headdim * PARTIAL_O_ELEMENT_BYTES
        return max(self.split_reduction_slice_bytes // row_bytes, 1)

    @property
    def split_reduction_slices_per_cta(self) -> int:
        """Return contiguous reducer slices assigned to one owner CTA."""
        rows_per_slice = self.split_reduction_rows_per_slice
        num_slices = (self.tile_size_q + rows_per_slice - 1) // rows_per_slice
        return max((num_slices + self.splits_kv - 1) // self.splits_kv, 1)

    @property
    def cluster_reduction_rows_per_cta(self) -> int:
        """Return the slice-aligned row capacity of one split-reduction owner."""
        return self.split_reduction_slices_per_cta * self.split_reduction_rows_per_slice

    @property
    def cluster_reduction_num_owner_ctas(self) -> int:
        """Return split CTAs that own at least one physical reducer slice."""
        rows_per_slice = self.split_reduction_rows_per_slice
        num_slices = (self.tile_size_q + rows_per_slice - 1) // rows_per_slice
        return min(
            (num_slices + self.split_reduction_slices_per_cta - 1)
            // self.split_reduction_slices_per_cta,
            self.splits_kv,
        )

    @property
    def cluster_max_runtime_partial_rows(self) -> int:
        """Maximum slice-aligned ``split x owner-row`` records at runtime."""
        if not self.supports_cluster_smem_reduction:
            return self.max_splits_kv * self.cluster_reduction_rows_per_cta
        rows_per_slice = self.split_reduction_rows_per_slice
        num_slices = (self.tile_size_q + rows_per_slice - 1) // rows_per_slice
        return max(
            splits * ((num_slices + splits - 1) // splits) * rows_per_slice
            for splits in range(1, self.splits_kv + 1)
        )

    @property
    def correction_sum_scratch_entries(self) -> int:
        """Return correction denominator scratch entries, or zero if unused."""
        return 0 if self.use_keeps_mma_ab else 4 * self.tile_size_q

    @property
    def cluster_transaction_bytes(self) -> int:
        """Return the byte count expected by each cluster owner barrier."""
        row_bytes = (
            self.headdim * PARTIAL_O_ELEMENT_BYTES
            + PARTIAL_STATS_VALUES_PER_ROW * FP32_BYTES
        )
        return self.cluster_max_runtime_partial_rows * row_bytes

    @property
    def max_runtime_row_split_segments(self) -> int:
        """Return the maximum reducer slices assigned to one runtime owner."""
        # Runtime contraction can leave one active owner responsible for the
        # complete Q tile.
        rows_per_slice = self.split_reduction_rows_per_slice
        return max((self.tile_size_q + rows_per_slice - 1) // rows_per_slice, 1)

    @property
    def supports_grouped_keeps(self) -> bool:
        """Whether a validated config uses a qualified grouped-Keeps recipe."""
        if self.tile_size_kv == 256:
            # KV256 reuses the common fixed/packed-Q, page-table, masking,
            # persistent scheduler, attention-sink, and GMEM split-publisher
            # semantics. Keeps has no cluster-SMEM publisher at either KV tile
            # size.
            if (
                not self.use_keeps_mma_ab
                or not self.groups_tokens_heads_q
                or self.tile_size_q != 64
                or self.headdim != 128
                or self.q_dtype not in (Float16, BFloat16)
                or not (self.q_dtype == self.kv_dtype == self.out_dtype)
                or self.use_cluster_smem_reduction
                or not self.matches_kv256_task_topology
            ):
                return False
            direct = not (self.use_split_kv or self.use_separate_reduction_kernel)
            if direct:
                return True
            if not (
                self.use_split_kv
                and self.splits_kv > 1
                and self.max_splits_kv >= self.splits_kv
            ):
                return False
            if self.use_separate_reduction_kernel and self.use_variable_seqlens_q:
                return self.mask_type == CAUSAL
            return self.mask_type == DENSE
        if (
            not self.use_keeps_mma_ab
            or not self.groups_tokens_heads_q
            or self.tile_size_kv != 128
            or self.tile_size_q not in (64, 128)
            or self.use_cluster_smem_reduction
        ):
            return False

        profile = self._grouped_keeps_profile_key

        # Keep block-sparse qualification separate from the dense/paged
        # profile matrix below. Its structural, masking, and reduction
        # constraints are validated separately; both scheduler modes use the
        # same qualified recipe keys.
        if self.use_block_sparse:
            return profile in _BLOCK_SPARSE_GROUPED_KEEPS_PROFILES

        direct = not (self.use_split_kv or self.use_separate_reduction_kernel)

        if profile in _GROUPED_KEEPS_PAGED_FP8_PROFILES:
            fixed_q1_ratio32 = self.max_seq_len_q == 1 and self.heads_q_per_kv == 32
            fixed_grouped_q = self.max_seq_len_q > 1
            return (
                (fixed_q1_ratio32 or fixed_grouped_q)
                and not self.use_variable_seqlens_q
                and self.use_paged_kv
                and self.num_tokens_per_page == 32
                and self.mask_type == CAUSAL
                and not any(
                    (
                        self.use_cluster_smem_reduction,
                        self.use_sliding_window_causal,
                        self.use_attention_sinks,
                    )
                )
                and (
                    direct
                    or (
                        self.use_split_kv
                        and self.splits_kv > 1
                        and self.max_splits_kv >= self.splits_kv
                    )
                )
            )

        if profile in _GROUPED_KEEPS_STATIC_ONLY_PROFILES:
            return (
                self.tile_size_q == 64
                and direct
                and self.mask_type == DENSE
                and not any(
                    (
                        self.use_variable_seqlens_q,
                        self.use_persistent_scheduler,
                        self.use_paged_kv,
                        self.use_sliding_window_causal,
                        self.use_attention_sinks,
                    )
                )
            )
        if profile != _GROUPED_KEEPS_MAIN_PROFILE:
            return False

        if self.tile_size_q == 128:
            return direct and not any(
                (
                    self.use_persistent_scheduler,
                    self.use_paged_kv,
                    self.use_sliding_window_causal,
                    self.use_attention_sinks,
                )
            )

        if self.use_paged_kv or self.use_sliding_window_causal:
            return (
                direct
                and self.mask_type == CAUSAL
                and not self.use_persistent_scheduler
                and not self.use_attention_sinks
                and not (self.use_paged_kv and self.use_sliding_window_causal)
                and (
                    not self.use_sliding_window_causal or self.attention_window_size > 0
                )
            )

        if self.use_persistent_scheduler:
            return (
                direct
                and not self.use_attention_sinks
                and (self.use_variable_seqlens_q or self.mask_type == DENSE)
            )
        if direct:
            return True

        if not (
            self.use_split_kv
            and self.splits_kv > 1
            and self.max_splits_kv >= self.splits_kv
        ):
            return False

        if self.use_separate_reduction_kernel and self.use_variable_seqlens_q:
            return not self.use_attention_sinks or self.mask_type == CAUSAL
        return self.mask_type == DENSE and not self.use_attention_sinks


SUPPORTED_IO_DTYPES = {Float16, BFloat16, Float8E4M3FN}
SUPPORTED_ACC_DTYPES = {Float32}


def _decode_config_items(source: object):
    """Return candidate config key/value pairs from a mapping or args-like object."""
    if isinstance(source, Mapping):
        return source.items()
    namespace = getattr(source, "__dict__", None)
    if namespace is not None:
        return namespace.items()
    field_names = FmhaDecodeConfig.__dataclass_fields__
    return (
        (name, getattr(source, name)) for name in field_names if hasattr(source, name)
    )


def _iter_config_sources(source: object):
    """Yield config sources in precedence order for direct config mutation."""
    if source is None:
        return
    if isinstance(source, (tuple, list)):
        for item in source:
            yield from _iter_config_sources(item)
        return
    yield source


def _apply_config_source(cfg: FmhaDecodeConfig, source: object) -> set[str]:
    """Apply explicit, correctly typed config fields and return names touched."""
    field_names = FmhaDecodeConfig.__dataclass_fields__
    explicit_fields: set[str] = set()
    for source_item in _iter_config_sources(source):
        for key, value in _decode_config_items(source_item):
            if key == "use_causal_spec_decoding":
                raise ValueError(
                    "use_causal_spec_decoding was removed; use "
                    "mask_type='dense' or mask_type='causal' instead"
                )
            if key == "single_token_q_per_cta":
                raise ValueError(
                    "single_token_q_per_cta was removed; select the Q layout "
                    "with groups_tokens_heads_q"
                )
            if key == "mask_type":
                if value is not None:
                    explicit_fields.add(key)
                # Normalize mask strings after all sources have been inspected
                # and the sliding-window request is known.
                continue
            if value is None or key == "headdim" or key not in field_names:
                continue
            if key in ("splits_kv", "max_splits_kv") and int(value) <= 0:
                continue
            setattr(cfg, key, value)
            explicit_fields.add(key)
    cfg.validate_boolean_fields()
    return explicit_fields


def _resolve_explicit_split_controls(
    cfg: FmhaDecodeConfig,
    *,
    explicit_fields: set[str],
    splits_kv: int,
    max_splits_kv: int | None,
) -> tuple[int, int | None]:
    """Merge public and config-source split controls without losing intent."""
    if "splits_kv" in explicit_fields:
        if splits_kv > 0 and splits_kv != cfg.splits_kv:
            raise ValueError(
                "conflicting splits_kv selections: public API requested "
                f"{splits_kv}, while config overrides requested {cfg.splits_kv}"
            )
        if splits_kv <= 0:
            splits_kv = cfg.splits_kv
    if "max_splits_kv" in explicit_fields:
        if (
            max_splits_kv is not None
            and max_splits_kv > 0
            and max_splits_kv != cfg.max_splits_kv
        ):
            raise ValueError(
                "conflicting max_splits_kv selections: public API requested "
                f"{max_splits_kv}, while config overrides requested "
                f"{cfg.max_splits_kv}"
            )
        if max_splits_kv is None or max_splits_kv <= 0:
            max_splits_kv = cfg.max_splits_kv
    return splits_kv, max_splits_kv


def _mask_type_from_config_source(source: object) -> str | int | None:
    """Return the last non-None mask selection from config sources."""
    selected: str | int | None = None
    for source_item in _iter_config_sources(source):
        for key, value in _decode_config_items(source_item):
            if key == "mask_type" and value is not None:
                selected = value
    return selected


def groups_tokens_heads_q_from_config_source(source: object) -> bool | None:
    """Return the last explicit Q-grouping selection from config sources."""
    selected = None
    for source_item in _iter_config_sources(source):
        for key, value in _decode_config_items(source_item):
            if key != "groups_tokens_heads_q" or value is None:
                continue
            if not isinstance(value, bool):
                raise TypeError(
                    f"groups_tokens_heads_q must be a bool, got {type(value).__name__}"
                )
            selected = value
    return selected


def _apply_mask_type_config(
    cfg: FmhaDecodeConfig,
    *,
    source: object,
    mask_type: str | int | None,
    sliding_window_causal: bool,
    explicit_fields: set[str],
) -> None:
    """Resolve public and config-source mask selections into a constexpr id."""
    source_mask_type = _mask_type_from_config_source(source)
    if mask_type is not None:
        explicit_fields.add("mask_type")
    if mask_type is not None and source_mask_type is not None:
        public_mask_type = normalize_mask_type(
            mask_type, sliding_window_causal=sliding_window_causal
        )
        config_mask_type = normalize_mask_type(
            source_mask_type, sliding_window_causal=sliding_window_causal
        )
        if public_mask_type != config_mask_type:
            raise ValueError(
                "conflicting mask_type selections: public API requested "
                f"{mask_type_name(public_mask_type)!r}, while config overrides "
                f"requested {mask_type_name(config_mask_type)!r}"
            )
        cfg.mask_type = public_mask_type
        return
    cfg.mask_type = normalize_mask_type(
        mask_type if mask_type is not None else source_mask_type,
        sliding_window_causal=sliding_window_causal,
    )


def _set_if_implicit(
    cfg: FmhaDecodeConfig,
    field_name: str,
    value: ConfigValue,
    explicit_fields: set[str],
) -> None:
    """Set a derived default only when the caller did not provide the field."""
    if field_name not in explicit_fields:
        setattr(cfg, field_name, value)


def _finalize_static_decode_config(
    cfg: FmhaDecodeConfig,
    explicit_fields: set[str],
) -> None:
    """Fill dtype-dependent, profile-dependent, and SMEM-derived defaults."""
    cfg.validate_boolean_fields()
    cfg.validate_dtypes()

    use_keeps_mma_ab = cfg.use_keeps_mma_ab
    if not use_keeps_mma_ab and cfg.headdim > 128:
        _set_if_implicit(cfg, "head_dim_per_stage_kv", 128, explicit_fields)
        _set_if_implicit(cfg, "num_insts_kv", 2, explicit_fields)

    if use_keeps_mma_ab:
        tile_size_q = cfg.tile_size_q if "tile_size_q" in explicit_fields else 64
        tile_size_kv = cfg.tile_size_kv if "tile_size_kv" in explicit_fields else 128
        if cfg.headdim > 128:
            _set_if_implicit(cfg, "num_insts_kv", 1, explicit_fields)
            _set_if_implicit(cfg, "head_dim_per_stage_kv", 128, explicit_fields)
            _set_if_implicit(cfg, "o_stages", 1, explicit_fields)
        _set_if_implicit(cfg, "tile_size_q", tile_size_q, explicit_fields)
        if cfg.tile_size_kv == 256:
            # Materialize the selected profile without overriding caller
            # tuning. The common validator below decides whether the resulting
            # effective configuration is supported by the kernel.
            for field_name, value in _KV_TILE_256_PHYSICAL_DEFAULTS.items():
                _set_if_implicit(cfg, field_name, value, explicit_fields)
            for field_name, value in _KV_TILE_256_TASK_TOPOLOGY_DEFAULTS.items():
                _set_if_implicit(cfg, field_name, value, explicit_fields)
        else:
            _set_if_implicit(cfg, "tmem_s_cols", tile_size_kv, explicit_fields)
            _set_if_implicit(
                cfg,
                "tmem_p_cols",
                tile_size_kv // 2,
                explicit_fields,
            )
            _set_if_implicit(cfg, "tmem_o_cols", cfg.headdim, explicit_fields)
            _set_if_implicit(cfg, "mma_tile_m_bmm1", tile_size_q, explicit_fields)
            _set_if_implicit(cfg, "mma_tile_n_bmm1", tile_size_kv, explicit_fields)
            _set_if_implicit(cfg, "mma_tile_m_bmm2", tile_size_q, explicit_fields)
            _set_if_implicit(
                cfg,
                "mma_tile_n_bmm2",
                cfg.head_dim_per_stage_kv or cfg.headdim,
                explicit_fields,
            )
        if cfg.num_insts_kv == 1:
            # One-inst static profiles use a compact 12-warp layout. Persistent
            # profiles keep MMA/scheduler/page-or-padding/load together in WG2
            # and add a work-queue-aware padding WG3 to preserve the existing
            # 16-warp CLC and CTA-barrier contract.
            _set_if_implicit(cfg, "correction_warp_idx", 4, explicit_fields)
            _set_if_implicit(cfg, "mma_warp_idx", 8, explicit_fields)
            _set_if_implicit(cfg, "page_offsets_warp_idx", 9, explicit_fields)
            _set_if_implicit(cfg, "load_warp_idx", 11, explicit_fields)
            _set_if_implicit(cfg, "scheduler_warp_idx", 9, explicit_fields)
            _set_if_implicit(cfg, "clc_load_warp_idx", 11, explicit_fields)
        if cfg.tile_size_q == 128 and cfg.tile_size_kv != 256:
            _set_if_implicit(cfg, "q_stages", 1, explicit_fields)
        _set_if_implicit(cfg, "ordered_softmax_barrier_mode", 1, explicit_fields)

    if cfg.tile_size_kv != 256 and "kv_stages" not in explicit_fields:
        cfg.kv_stages = cfg.inferred_kv_stages

    # Split-KV mode forbids persistent scheduling. Canonicalize here so
    # downstream consumers can gate on use_persistent_scheduler alone without
    # having to repeat the (... and not use_split_kv) check.
    if cfg.use_split_kv:
        cfg.use_persistent_scheduler = False


def _validate_kv256_static_config(cfg: FmhaDecodeConfig) -> None:
    """Validate the effective KV256 profile after implicit defaults are filled."""
    if cfg.tile_size_kv != 256:
        return

    def _require_python_int(field_name: str) -> int:
        value = getattr(cfg, field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"KV256 {field_name} must be a Python integer, "
                f"got {type(value).__name__}"
            )
        return value

    for field_name, expected in _KV_TILE_256_PHYSICAL_DEFAULTS.items():
        actual = _require_python_int(field_name)
        if field_name in _KV_TILE_256_TUNABLE_FIELDS:
            continue
        if actual != expected:
            raise ValueError(f"KV256 requires {field_name}={expected}, got {actual}")
    for field_name, expected in _KV_TILE_256_TASK_TOPOLOGY_DEFAULTS.items():
        actual = _require_python_int(field_name)
        if actual != expected:
            raise ValueError(f"KV256 requires {field_name}={expected}, got {actual}")

    if cfg.q_stages <= 0 or cfg.kv_stages <= 0:
        raise ValueError(
            "KV256 shared-memory pipeline requires positive q_stages and "
            f"kv_stages, got q_stages={cfg.q_stages}, "
            f"kv_stages={cfg.kv_stages}"
        )
    pipeline_smem_bytes = (
        cfg.q_stages * cfg.smem_q_tile_bytes + cfg.kv_stages * cfg.smem_kv_tile_bytes
    )
    pipeline_smem_budget_bytes = TOTAL_SMEM_BUDGET_KIB * BYTES_PER_KIB
    if pipeline_smem_bytes > pipeline_smem_budget_bytes:
        raise ValueError(
            "KV256 shared-memory pipeline exceeds the static SMEM budget: "
            f"q_stages={cfg.q_stages}, kv_stages={cfg.kv_stages} require "
            f"{pipeline_smem_bytes} bytes, limit is "
            f"{pipeline_smem_budget_bytes} bytes"
        )
    if (
        cfg.use_persistent_scheduler
        and not cfg.use_split_kv
        and not cfg.use_attention_sinks
        and cfg.kv_stages != KV_TILE_256_SHARED_FIFO_STAGES
    ):
        raise ValueError(
            "persistent KV256 requires kv_stages="
            f"{KV_TILE_256_SHARED_FIFO_STAGES} for the rotating shared-KV "
            f"exchange, got {cfg.kv_stages}"
        )
    if not cfg.supports_grouped_keeps:
        raise ValueError(
            "KV256 currently supports only the qualified Q64 FP16/BF16/D128 "
            "grouped Keeps profile"
        )


@dataclass(frozen=True)
class _WarpRole:
    """One active task's configured warp placement."""

    name: str
    index_field: str
    preferred_warp_idx: int
    num_warps: int
    is_padding: bool = False


def _append_padding_warp_roles(cfg: FmhaDecodeConfig, roles: list[_WarpRole]) -> None:
    """Append padding warps to the tail of each warp group."""
    num_warps_per_wg = [0] * MAX_WARP_GROUPS
    total_num_wgs = 0
    for role in roles:
        if role.num_warps <= 0:
            raise ValueError(f"{role.name} must use at least one warp")
        preferred_wg = role.preferred_warp_idx // 4
        preferred_wg_end = (role.preferred_warp_idx + role.num_warps - 1) // 4
        if preferred_wg != preferred_wg_end:
            raise ValueError(
                f"{role.name} spans warp groups {preferred_wg} and {preferred_wg_end}; "
                "each task role must fit within one warp group"
            )
        if preferred_wg < 0 or preferred_wg >= MAX_WARP_GROUPS:
            raise ValueError(
                f"{role.name} is assigned to unsupported warp group {preferred_wg}"
            )
        num_warps_per_wg[preferred_wg] += role.num_warps
        total_num_wgs = max(total_num_wgs, preferred_wg + 1)
    for wg_idx, num_warps in enumerate(num_warps_per_wg):
        if num_warps > 4:
            raise ValueError(
                f"warp group {wg_idx} has {num_warps} active warps; at most 4 are allowed"
            )

    if cfg.use_persistent_scheduler:
        total_num_wgs = MAX_WARP_GROUPS

    for wg_idx, active_warps in enumerate(num_warps_per_wg):
        padding_index_field = f"wg{wg_idx}_padding_warp_idx"
        padding_count_field = f"wg{wg_idx}_padding_num_warps"
        padding_warp_idx = wg_idx * 4 + active_warps
        padding_num_warps = 4 - active_warps if wg_idx < total_num_wgs else 0
        setattr(cfg, padding_index_field, padding_warp_idx)
        setattr(cfg, padding_count_field, padding_num_warps)
        if padding_num_warps > 0:
            roles.append(
                _WarpRole(
                    f"wg{wg_idx}_padding",
                    padding_index_field,
                    padding_warp_idx,
                    padding_num_warps,
                    is_padding=True,
                )
            )


def _active_warp_roles(cfg: FmhaDecodeConfig) -> list[_WarpRole]:
    """Return the warp roles instantiated by the current kernel profile."""
    roles = [
        _WarpRole(
            "softmax0",
            "softmax0_warp_idx",
            cfg.softmax0_warp_idx,
            cfg.softmax0_num_warps,
        ),
        _WarpRole(
            "correction",
            "correction_warp_idx",
            cfg.correction_warp_idx,
            cfg.correction_num_warps,
        ),
        _WarpRole("mma", "mma_warp_idx", cfg.mma_warp_idx, cfg.mma_num_warps),
    ]
    if cfg.num_insts_kv != 1:
        roles.append(
            _WarpRole(
                "softmax1",
                "softmax1_warp_idx",
                cfg.softmax1_warp_idx,
                cfg.softmax1_num_warps,
            )
        )

    if cfg.use_persistent_scheduler:
        roles.extend(
            (
                _WarpRole(
                    "scheduler",
                    "scheduler_warp_idx",
                    cfg.scheduler_warp_idx,
                    cfg.scheduler_num_warps,
                ),
                _WarpRole(
                    "load",
                    "clc_load_warp_idx",
                    cfg.clc_load_warp_idx,
                    cfg.load_num_warps,
                ),
            )
        )
    else:
        roles.append(
            _WarpRole(
                "load",
                "load_warp_idx",
                cfg.load_warp_idx,
                cfg.load_num_warps,
            )
        )

    if cfg.use_paged_kv and not cfg.use_block_sparse:
        roles.append(
            _WarpRole(
                "page_offsets",
                "page_offsets_warp_idx",
                cfg.page_offsets_warp_idx,
                cfg.page_offsets_num_warps,
            )
        )
    _append_padding_warp_roles(cfg, roles)
    return roles


def _finalize_warp_roles(cfg: FmhaDecodeConfig) -> None:
    """Compact active warp indices and add paddings."""
    warp_roles = sorted(
        _active_warp_roles(cfg),
        key=lambda role: (
            role.preferred_warp_idx // 4,
            role.is_padding,
            role.preferred_warp_idx,
        ),
    )

    next_warp_idx = 0
    for role in warp_roles:
        setattr(cfg, role.index_field, next_warp_idx)
        next_warp_idx += role.num_warps


def _make_static_decode_config(
    headdim: int = 128,
    args: object | None = None,
    *,
    mask_type: str | int | None = None,
    sliding_window_causal: bool = False,
) -> FmhaDecodeConfig:
    """Build a static FmhaDecodeConfig by mutating a default config object.

    ``args`` may be a parser namespace, harness dictionary, fully specified
    config object, or a tuple/list of those sources. Only ``FmhaDecodeConfig``
    fields are read from it. ``None`` values mean "keep the default"; mappings
    are the appropriate source when omitted fields should remain implicit.
    """
    cfg = FmhaDecodeConfig(headdim=headdim)
    explicit_fields = _apply_config_source(cfg, args)
    _apply_mask_type_config(
        cfg,
        source=args,
        mask_type=mask_type,
        sliding_window_causal=(sliding_window_causal or cfg.use_sliding_window_causal),
        explicit_fields=explicit_fields,
    )
    _finalize_static_decode_config(cfg, explicit_fields)
    _validate_kv256_static_config(cfg)
    _finalize_warp_roles(cfg)
    return cfg


def _swaps_tile_fields_for_heads(
    num_heads_q: int,
    num_heads_kv: int,
    *,
    groups_tokens_heads_q: bool,
) -> dict[str, int]:
    """Return SwapsMmaAb tile metadata for a GQA head ratio."""
    h_r = num_heads_q // num_heads_kv
    if groups_tokens_heads_q:
        tile_size_q = next((tile_q for tile_q in (8, 16, 32) if h_r <= tile_q), 0)
        if tile_size_q == 0:
            raise ValueError(
                "default groups_tokens_heads_q=True supports Hq/Hkv <= 32; set "
                "groups_tokens_heads_q=False to use ungrouped SwapsMmaAb head bands"
            )
    else:
        tile_size_q = 8 if h_r <= 8 else 16
    return {
        "tile_size_q": tile_size_q,
        "tmem_s_cols": tile_size_q,
        "tmem_o_cols": tile_size_q,
        "mma_tile_n_bmm1": tile_size_q,
        "mma_tile_n_bmm2": tile_size_q,
    }


def cluster_smem_reduction_partial_smem_bytes(
    *,
    max_splits_kv: int,
    tile_size_q: int,
    headdim: int,
    splits_kv: int | None = None,
    correction_num_warps: int = 4,
) -> int:
    """Bytes staged per owner CTA for every runtime cluster split prefix.

    For active count ``s``, each owner stages ``s`` times its slice-aligned row
    band. The maximum across all prefixes covers non-divisor contractions.
    """
    if max_splits_kv <= 0:
        return 0
    configured_splits_kv = splits_kv if splits_kv is not None else max_splits_kv
    if configured_splits_kv <= 0:
        return 0
    slice_bytes = (
        correction_num_warps * WARP_THREADS * SPLIT_REDUCTION_VECTOR_BYTES_PER_THREAD
    )
    row_bytes = headdim * PARTIAL_O_ELEMENT_BYTES
    rows_per_slice = max(slice_bytes // row_bytes, 1)
    num_slices = (tile_size_q + rows_per_slice - 1) // rows_per_slice
    max_runtime_partial_rows = max(
        active_splits
        * max((num_slices + active_splits - 1) // active_splits, 1)
        * rows_per_slice
        for active_splits in range(1, configured_splits_kv + 1)
    )
    return max_runtime_partial_rows * (
        headdim * PARTIAL_O_ELEMENT_BYTES + PARTIAL_STATS_VALUES_PER_ROW * FP32_BYTES
    )


def cluster_smem_reduction_unsupported_reason(
    *,
    max_splits_kv: int,
    splits_kv: int | None = None,
    tile_size_q: int,
    headdim: int,
    correction_num_warps: int = 4,
    cluster_dim_x: int = 1,
    max_partial_smem_bytes: int = MAX_CLUSTER_PARTIAL_SMEM_BYTES,
) -> str:
    """Return why cluster SMEM reduction must be rejected, or "" if supported."""
    if max_splits_kv <= 1:
        return ""
    if max_splits_kv * cluster_dim_x > MAX_CLUSTER_DIM_X:
        return (
            "splits_kv * clusterDimX exceeds the cluster-size limit of "
            f"{MAX_CLUSTER_DIM_X}"
        )
    partial_smem_bytes = cluster_smem_reduction_partial_smem_bytes(
        max_splits_kv=max_splits_kv,
        splits_kv=splits_kv,
        tile_size_q=tile_size_q,
        headdim=headdim,
        correction_num_warps=correction_num_warps,
    )
    if partial_smem_bytes > max_partial_smem_bytes:
        return (
            "cluster leader-DSMEM partial staging would use "
            f"{partial_smem_bytes} bytes, above the conservative "
            f"{max_partial_smem_bytes}-byte limit"
        )
    return ""


def compute_runtime_active_splits_kv(
    *,
    valid_k: int,
    tile_size_kv: int,
    num_insts_kv: int,
    configured_splits_kv: int,
) -> int:
    """Host mirror of the device runtime split-prefix calculation."""
    if valid_k < 0:
        raise ValueError("valid_k must be non-negative")
    if tile_size_kv <= 0:
        raise ValueError("tile_size_kv must be positive")
    if num_insts_kv <= 0:
        raise ValueError("num_insts_kv must be positive")
    if configured_splits_kv <= 0:
        raise ValueError("configured_splits_kv must be positive")
    total_kv_tiles = (valid_k + tile_size_kv - 1) // tile_size_kv
    groups_per_split = (total_kv_tiles + configured_splits_kv * num_insts_kv - 1) // (
        configured_splits_kv * num_insts_kv
    )
    local_kv_tiles = max(groups_per_split * num_insts_kv, num_insts_kv)
    return (total_kv_tiles + local_kv_tiles - 1) // local_kv_tiles


def _max_splits_kv_by_work(
    *,
    seq_len_kv: int,
    tile_size_kv: int,
    num_insts_kv: int,
    max_splits_kv: int | None = None,
) -> int:
    """Return the fanout cap that retains useful KV work per CTA."""
    tile_size_per_cta_kv = tile_size_kv * num_insts_kv * MIN_LOOP_ITERS_PER_SPLIT
    max_by_seq = max(
        1,
        (seq_len_kv + tile_size_per_cta_kv - 1) // tile_size_per_cta_kv,
    )
    if max_splits_kv is not None and max_splits_kv > 0:
        max_by_seq = min(max_by_seq, max_splits_kv)
    return max_by_seq


def enumerate_auto_splits_kv(
    *,
    seq_len_kv: int,
    batch_size: int,
    num_heads_kv: int,
    tile_size_kv: int,
    num_insts_kv: int,
    num_q_tiles: int,
    service_capacity: int,
) -> tuple[int, ...]:
    """Enumerate direct and useful split fanouts for an under-filled Q grid.

    All one-wave fanouts and the first capacity-crossing fanout participate in
    the empirical score. The latter is important when a partially filled wave
    cannot be completed by any uniform integer fanout.
    """
    if num_q_tiles <= 0:
        raise ValueError("num_q_tiles must be positive")
    if service_capacity <= 0:
        raise ValueError("service_capacity must be positive")
    max_by_work = _max_splits_kv_by_work(
        seq_len_kv=seq_len_kv,
        tile_size_kv=tile_size_kv,
        num_insts_kv=num_insts_kv,
    )
    base_grid = max(1, batch_size * num_heads_kv * num_q_tiles)
    if base_grid >= service_capacity or max_by_work <= 1:
        return (1,)
    first_full_wave_fanout = (service_capacity + base_grid - 1) // base_grid
    max_considered = min(
        max_by_work,
        max(first_full_wave_fanout, 2),
    )
    return tuple(range(1, max_considered + 1))


def select_splits_kv(
    *,
    seq_len_kv: int,
    batch_size: int,
    num_heads_kv: int,
    tile_size_kv: int,
    num_insts_kv: int,
    num_q_tiles: int = 1,
    service_capacity: int | None = None,
    requested_splits_kv: int = -1,
    max_splits_kv: int | None = None,
) -> int:
    """
    Select a Q-grid-aware split-KV fanout.

    Every legal TileQ is considered with its actual number of Q CTAs.  The
    automatic fanout fills otherwise idle cluster-size-one service slots while
    retaining at least ``MIN_LOOP_ITERS_PER_SPLIT`` KV iterations per CTA.
    Positive caller fanouts remain pinned subject only to the KV-work cap.
    """
    if num_q_tiles <= 0:
        raise ValueError("num_q_tiles must be positive")
    max_by_seq = _max_splits_kv_by_work(
        seq_len_kv=seq_len_kv,
        tile_size_kv=tile_size_kv,
        num_insts_kv=num_insts_kv,
        max_splits_kv=max_splits_kv,
    )
    if requested_splits_kv > 0:
        return max(1, min(max_by_seq, requested_splits_kv))

    if service_capacity is None:
        hardware_info = utils.HardwareInfo()
        service_capacity = hardware_info.get_device_multiprocessor_count()
        # B200 fallback when the runtime SM query is unavailable.
        service_capacity = (
            FALLBACK_SM_COUNT_B200 if service_capacity <= 0 else service_capacity
        )
    if service_capacity <= 0:
        raise ValueError("service_capacity must be positive")
    base_grid = max(1, batch_size * num_heads_kv * num_q_tiles)
    return max(1, min(max_by_seq, max(service_capacity // base_grid, 1)))


SPLIT_KV_MODES = (
    "disabled",
    "gmem_reduction",
    "gmem_reduction_with_separate_kernel",
    "cluster_smem_reduction",
)


def _select_auto_launch_mode(
    *,
    batch_size: int,
    num_heads_kv: int,
    seq_len_kv: int,
    num_q_tiles: int = 1,
    tile_size_kv: int = AUTO_LAUNCH_TILE_SIZE_KV,
    persistent_min_waves: int = 1,
    persistent_min_tiles_per_cta: int = 1,
) -> str:
    """Pick the launch mode that best matches the kernel's parallelism budget.

    The kernel can run in three launch modes, each suited to a different
    occupancy regime.

    Returns one of:

      ``"gmem_reduction"``
          Split the K/V sequence across several CTAs (Flash-Decoding GMEM
          reduction). Chosen when the static grid sits under one SM wave
          (``waves < 1``) *and* each CTA has at least 2,048 padded K/V tokens
          to dwarf the GMEM reduction overhead. Legal split fanout is still
          bounded independently by the minimum loop iterations. At ``b=1``
          there are only ``num_heads_kv`` CTAs in the static grid, which
          fills a few percent of an SM-rich device; splitting K/V across
          tens of CTAs unlocks the remaining bandwidth.

      ``"persistent"``
          Switch to the CLC dynamic persistent scheduler whenever the direct
          launch contains more than ``persistent_min_waves`` resident CTA
          waves and at least ``persistent_min_tiles_per_cta`` K/V tiles per
          task. Persistence has no launch work to eliminate within one wave;
          callers with heavier per-task scheduling may require more work to
          amortize that overhead.

      ``"static"``
          Everything else. The default static grid is a good fit when the
          launch already saturates the device with substantial per-CTA
          work.
    """
    if seq_len_kv <= 0 or batch_size <= 0 or num_heads_kv <= 0 or num_q_tiles <= 0:
        return "static"
    hardware_info = utils.HardwareInfo()
    sm_count = hardware_info.get_device_multiprocessor_count()
    sm_count = FALLBACK_SM_COUNT_B200 if sm_count <= 0 else sm_count
    ctas = batch_size * num_heads_kv * num_q_tiles
    waves = ctas / sm_count
    tiles_per_cta = (seq_len_kv + tile_size_kv - 1) // tile_size_kv
    kv_tokens_per_cta = tiles_per_cta * tile_size_kv
    if waves < 1 and kv_tokens_per_cta >= SPLIT_KV_MIN_TOKENS_PER_CTA:
        return "gmem_reduction"
    if (
        ctas > persistent_min_waves * sm_count
        and tiles_per_cta >= persistent_min_tiles_per_cta
    ):
        return "persistent"
    return "static"


def get_max_active_clusters_for_cluster_size(cluster_size: int) -> int:
    """Query cluster occupancy after establishing CUDA's primary context.

    ``HardwareInfo`` compiles and retains a tiny occupancy-query module on its
    first call.  If that first call precedes CUDA context creation, retrying can
    reuse an invalid module handle.  Establish the context before constructing
    ``HardwareInfo`` so a fresh Python process is reliable as well.
    """
    cuda_context_ready = False
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.init()
            torch.empty(0, device="cuda")
            cuda_context_ready = True
    except (ImportError, RuntimeError):
        pass
    try:
        return utils.HardwareInfo().get_max_active_clusters(cluster_size)
    except RuntimeError:
        if cuda_context_ready:
            raise
        return max(FALLBACK_SM_COUNT_B200 // max(cluster_size, 1), 1)


def select_one_wave_cluster_split(
    *,
    initial_splits_kv: int,
    minimum_splits_kv: int = 2,
    num_launched_clusters: int,
    tile_size_q: int,
    headdim: int,
    correction_num_warps: int = 4,
) -> int | None:
    """Return the largest legal cluster split count whose clusters fit one wave.

    The ordinary split heuristic maximizes mainloop parallelism without
    considering cluster residency.  If that cluster size does not fit the
    entire logical grid concurrently, search smaller cluster sizes instead of
    falling back immediately to GMEM reduction.  Every candidate goes through
    the same cluster-size and leader-DSMEM gates; no problem-shape whitelist is
    involved. ``minimum_splits_kv`` can pin an explicit fanout while retaining
    the same eligibility checks.
    """
    if initial_splits_kv < 2 or num_launched_clusters <= 0:
        return None
    lower_bound = max(minimum_splits_kv, 2)
    if lower_bound > initial_splits_kv:
        return None
    for candidate in range(initial_splits_kv, lower_bound - 1, -1):
        cluster_reason = cluster_smem_reduction_unsupported_reason(
            max_splits_kv=candidate,
            splits_kv=candidate,
            tile_size_q=tile_size_q,
            headdim=headdim,
            correction_num_warps=correction_num_warps,
        )
        if cluster_reason:
            continue
        max_active_clusters = get_max_active_clusters_for_cluster_size(candidate)
        if max_active_clusters > 0 and num_launched_clusters <= max_active_clusters:
            return candidate
    return None


def validate_sliding_window_args(
    sliding_window_causal: bool, attention_window_size: int
) -> None:
    """Validate the sliding-window causal command-line arguments."""
    if sliding_window_causal and attention_window_size <= 0:
        raise ValueError(
            "attention_window_size must be positive when sliding_window_causal is enabled"
        )


def validate_causal_decode_lengths(
    *,
    seq_len_q: int,
    seq_len_kv: int,
    mask_type: int,
) -> None:
    """Reject Q/KV lengths that cannot represent causal speculative decode."""
    if mask_type == CAUSAL and seq_len_q > seq_len_kv:
        raise ValueError(
            "causal decode requires seq_len_q <= seq_len_kv; "
            f"got seq_len_q={seq_len_q}, seq_len_kv={seq_len_kv}"
        )


def _effective_seq_len_for_sliding(
    seq_len_kv: int,
    sliding_window_causal: bool,
    attention_window_size: int,
    tile_size_kv: int = 128,
) -> int:
    """Return the effective KV length after sliding-window tile skipping."""
    if not sliding_window_causal:
        return seq_len_kv
    skipped_tiles = max(seq_len_kv - attention_window_size, 0) // tile_size_kv
    return seq_len_kv - skipped_tiles * tile_size_kv


def _validate_split_kv_mode(mode: str) -> str:
    """Validate and return a canonical split-KV launch mode."""
    if mode not in SPLIT_KV_MODES:
        allowed = ", ".join(SPLIT_KV_MODES)
        raise ValueError(
            f"Unsupported split_kv_mode: {mode}. Expected one of: {allowed}"
        )
    return mode


def _apply_swaps_tile_config(
    cfg: FmhaDecodeConfig,
    *,
    explicit_fields: set[str],
    num_heads_q: int,
    num_heads_kv: int,
) -> None:
    """Fill SwapsMmaAb tile fields while preserving explicit user fields."""
    if cfg.use_keeps_mma_ab:
        return

    tile_fields = _swaps_tile_fields_for_heads(
        num_heads_q,
        num_heads_kv,
        groups_tokens_heads_q=cfg.groups_tokens_heads_q,
    )
    for key, value in tile_fields.items():
        _set_if_implicit(cfg, key, value, explicit_fields)

    if "tile_size_q" not in explicit_fields:
        return

    for key in (
        "tmem_s_cols",
        "tmem_o_cols",
        "mma_tile_n_bmm1",
        "mma_tile_n_bmm2",
    ):
        _set_if_implicit(cfg, key, cfg.tile_size_q, explicit_fields)


_MMA_SELECTION_FIELDS = {
    "use_keeps_mma_ab",
    "tile_size_q",
    "tmem_s_cols",
    "tmem_p_cols",
    "tmem_o_cols",
    "mma_tile_m_bmm1",
    "mma_tile_n_bmm1",
    "mma_tile_m_bmm2",
    "mma_tile_n_bmm2",
}

_LAUNCH_SELECTION_FIELDS = {
    "use_split_kv",
    "splits_kv",
    "max_splits_kv",
    "use_separate_reduction_kernel",
    "use_cluster_smem_reduction",
    "use_persistent_scheduler",
}


def _try_apply_auto_kv256_profile(
    cfg: FmhaDecodeConfig,
    *,
    q_candidate: GroupedQMmaCandidate | None,
    explicit_fields: set[str],
    auto_tuner: bool,
    split_kv_mode: str,
    num_heads_q: int,
    num_heads_kv: int,
    splits_kv: int,
    max_splits_kv: int | None,
) -> bool:
    """Promote KV128 after the Q cost model selects an exact Q64 Keeps tile.

    Every other automatic Q result retains the default KV128 tile. Explicit
    MMA, KV-tile, or launch policies remain caller-controlled. Device
    compatibility is enforced by the public PrimTS wrapper, alongside the
    other decode profiles, rather than duplicated in config selection.
    """
    selection_is_unpinned = (
        auto_tuner
        and split_kv_mode == "disabled"
        and splits_kv == -1
        and max_splits_kv is None
        and not (_MMA_SELECTION_FIELDS & explicit_fields)
        and not (_LAUNCH_SELECTION_FIELDS & explicit_fields)
        and "tile_size_kv" not in explicit_fields
    )
    profile_is_eligible = (
        selection_is_unpinned
        and q_candidate is not None
        and q_candidate.variant == "keeps_mma_ab"
        and q_candidate.tile_size_q == 64
        and cfg.use_paged_kv
        and cfg.groups_tokens_heads_q
        and cfg.headdim == 128
        and cfg.q_dtype in (Float16, BFloat16)
        and cfg.q_dtype == cfg.kv_dtype == cfg.out_dtype
        and num_heads_q // num_heads_kv <= 64
    )
    if not profile_is_eligible:
        return False

    cfg.tile_size_kv = 256
    return True


def _apply_grouped_q_mma_candidate(
    cfg: FmhaDecodeConfig,
    candidate: GroupedQMmaCandidate,
) -> None:
    """Apply one internally selected grouped-Q MMA tile to ``cfg``."""
    cfg.use_keeps_mma_ab = candidate.variant == "keeps_mma_ab"
    cfg.tile_size_q = candidate.tile_size_q
    if cfg.use_keeps_mma_ab:
        return
    cfg.tmem_s_cols = candidate.tile_size_q
    cfg.tmem_o_cols = candidate.tile_size_q
    cfg.mma_tile_n_bmm1 = candidate.tile_size_q
    cfg.mma_tile_n_bmm2 = candidate.tile_size_q


def _resolve_grouped_q_launch_candidates(
    cfg: FmhaDecodeConfig,
    candidate: GroupedQMmaCandidate,
    *,
    explicit_fields: set[str],
    seq_len_q: int,
    seq_len_kv: int,
    batch_size: int,
    num_heads_q: int,
    num_heads_kv: int,
    service_capacity: int,
) -> tuple[GroupedQLaunchCandidate, ...]:
    """Resolve one TileQ with Q-grid-aware direct and split recipes.

    The under-filled Q grid always participates in split selection. If the
    GMEM split profile is unsupported, the legal direct recipe remains in the
    candidate set.
    """
    probe = deepcopy(cfg)
    # TileQ is always scored on the common KV128 baseline. This includes an
    # explicitly requested final KV width: explicit materialization remains a
    # downstream constraint and must not feed back into the Q winner.
    probe.tile_size_kv = 128
    _apply_grouped_q_mma_candidate(probe, candidate)
    cost_tile_size_kv = 128
    if (
        candidate.variant == "keeps_mma_ab"
        and candidate.tile_size_q == 64
        and probe.headdim == 128
        and probe.q_dtype == BFloat16
        and probe.q_dtype == probe.kv_dtype == probe.out_dtype
    ):
        # TileQ selection uses a common KV128 cost basis and is independent of
        # the later KV-tile decision. BF16 Q64 has no final KV128 profile, but
        # its Q geometry and modeled cost are identical to the qualified FP16
        # logical candidate. The KV selector materializes and validates the
        # actual BF16 profile only after the Q winner is known.
        probe.q_dtype = Float16
        probe.kv_dtype = Float16
        probe.out_dtype = Float16
    try:
        _finalize_static_decode_config(
            probe,
            explicit_fields | {"tile_size_q"},
        )
        cost_num_insts_kv = probe.num_insts_kv
        _validate_profile_support(
            cfg=probe,
            seq_len_q=seq_len_q,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            split_kv_mode="disabled",
        )
    except ValueError:
        return ()

    split_candidates = enumerate_auto_splits_kv(
        seq_len_kv=seq_len_kv,
        batch_size=batch_size,
        num_heads_kv=num_heads_kv,
        tile_size_kv=cost_tile_size_kv,
        num_insts_kv=cost_num_insts_kv,
        num_q_tiles=candidate.q_tiles,
        service_capacity=service_capacity,
    )
    recipes = []
    for candidate_splits_kv in split_candidates:
        if candidate_splits_kv > 1:
            split_probe = deepcopy(probe)
            split_probe.use_split_kv = True
            split_probe.splits_kv = candidate_splits_kv
            split_probe.max_splits_kv = candidate_splits_kv
            try:
                _validate_profile_support(
                    cfg=split_probe,
                    seq_len_q=seq_len_q,
                    num_heads_q=num_heads_q,
                    num_heads_kv=num_heads_kv,
                    split_kv_mode="gmem_reduction",
                )
            except ValueError:
                continue
        recipes.append(
            make_grouped_q_launch_candidate(
                candidate,
                splits_kv=candidate_splits_kv,
                seq_len_kv=seq_len_kv,
                tile_size_kv=cost_tile_size_kv,
                num_insts_kv=cost_num_insts_kv,
                batch_size=batch_size,
                num_heads_kv=num_heads_kv,
                service_capacity=service_capacity,
            )
        )
    return tuple(recipes)


def _apply_auto_grouped_q_mma_config(
    cfg: FmhaDecodeConfig,
    *,
    explicit_fields: set[str],
    auto_tuner: bool,
    split_kv_mode: str,
    seq_len_q: int,
    seq_len_kv: int,
    batch_size: int,
    num_heads_q: int,
    num_heads_kv: int,
    splits_kv: int,
    max_splits_kv: int | None,
) -> GroupedQLaunchCandidate | None:
    """Select a legal fixed multi-Q MMA and KV-split recipe when unpinned.

    SQ1, packed/variable Q, explicit launch modes, ungrouped layouts, and
    caller-provided MMA fields retain the existing path. Explicit fanout
    controls bypass this selector together with explicit launch and MMA fields.
    """
    if (
        not auto_tuner
        or split_kv_mode != "disabled"
        or seq_len_q <= 1
        or cfg.use_variable_seqlens_q
        or not cfg.groups_tokens_heads_q
        or not cfg.use_paged_kv
        or cfg.num_tokens_per_page != 32
        or cfg.mask_type != CAUSAL
        or cfg.use_sliding_window_causal
        or cfg.use_attention_sinks
        or bool(_MMA_SELECTION_FIELDS & explicit_fields)
        or bool(_LAUNCH_SELECTION_FIELDS & explicit_fields)
        or splits_kv != -1
        or max_splits_kv is not None
    ):
        return None

    candidates = enumerate_grouped_q_mma_candidates(
        heads_q_per_kv=num_heads_q // num_heads_kv,
        seq_len_q=seq_len_q,
    )
    if not candidates:
        return None
    service_capacity = get_max_active_clusters_for_cluster_size(1)
    supported = tuple(
        recipe
        for candidate in candidates
        for recipe in _resolve_grouped_q_launch_candidates(
            cfg,
            candidate,
            explicit_fields=explicit_fields,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            batch_size=batch_size,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            service_capacity=service_capacity,
        )
    )
    if not supported:
        return None

    has_underfilled_q_grid = any(
        batch_size * num_heads_kv * recipe.mma.q_tiles < service_capacity
        for recipe in supported
    )
    if has_underfilled_q_grid:
        selected = select_grouped_q_launch_candidate(supported)
    else:
        # Every legal Q grid already fills the machine, so splitting cannot
        # expose otherwise-idle SMs. Compare direct recipes with the same
        # mainloop-aware score instead of rewarding narrow tiles that reread
        # the complete KV sequence for each additional Q tile.
        direct = tuple(recipe for recipe in supported if recipe.splits_kv == 1)
        if not direct:
            return None
        selected = select_grouped_q_direct_wave_candidate(direct)
    _apply_grouped_q_mma_candidate(cfg, selected.mma)
    if selected.splits_kv == 1 and selected.base_ctas > service_capacity:
        # CLC persistence pays for work discovery by reusing one resident CTA
        # wave. Select it only when the direct launch has more than one wave;
        # a single-wave grid has no launch work for persistence to eliminate.
        persistent_probe = deepcopy(cfg)
        persistent_probe.use_persistent_scheduler = True
        try:
            _finalize_static_decode_config(
                persistent_probe,
                explicit_fields | {"tile_size_q"},
            )
            _validate_profile_support(
                cfg=persistent_probe,
                seq_len_q=seq_len_q,
                num_heads_q=num_heads_q,
                num_heads_kv=num_heads_kv,
                split_kv_mode="disabled",
            )
        except ValueError:
            # Some public profiles support the selected direct grouped tile but
            # not its CLC variant. Keep the valid static recipe rather than
            # turning a successful automatic selection into a late rejection.
            pass
        else:
            cfg.use_persistent_scheduler = True
    # Keeps static defaults otherwise canonicalize an implicit tile to 64.
    # Record only these local derived choices so finalization preserves them;
    # the pre-Q snapshot used by the downstream KV selector remains unpinned.
    explicit_fields.add("tile_size_q")
    return selected


def _apply_default_q_grouping(
    cfg: FmhaDecodeConfig,
    *,
    explicit_fields: set[str],
) -> None:
    """Enable token/head grouping unless the caller explicitly opts out."""
    if "groups_tokens_heads_q" not in explicit_fields:
        cfg.groups_tokens_heads_q = True


def _apply_layout_config(
    cfg: FmhaDecodeConfig,
    *,
    qkv_layout: str,
    num_tokens_per_page: int,
    seq_len_kv: int,
) -> str:
    """Apply contiguous/paged-KV layout fields and return the canonical layout."""
    qkv_layout = normalize_qkv_layout(qkv_layout)
    if qkv_layout == "pagedKv":
        validate_page_size(num_tokens_per_page)
        cfg.use_paged_kv = True
        cfg.num_tokens_per_page = num_tokens_per_page
        cfg.max_num_pages_per_seq_kv = (
            seq_len_kv + num_tokens_per_page - 1
        ) // num_tokens_per_page
    return qkv_layout


def _apply_feature_config(
    cfg: FmhaDecodeConfig,
    *,
    explicit_fields: set[str],
    seq_len_q: int,
    num_heads_q: int,
    num_heads_kv: int,
    sliding_window_causal: bool,
    attention_window_size: int,
    use_attention_sinks: bool,
) -> None:
    """Apply feature flags that affect config construction."""
    if sliding_window_causal:
        cfg.use_sliding_window_causal = True
        cfg.attention_window_size = attention_window_size
    if use_attention_sinks:
        cfg.use_attention_sinks = True
    if seq_len_q > 1 or cfg.use_variable_seqlens_q or cfg.groups_tokens_heads_q:
        _set_if_implicit(cfg, "max_seq_len_q", seq_len_q, explicit_fields)
    _set_if_implicit(
        cfg, "heads_q_per_kv", num_heads_q // num_heads_kv, explicit_fields
    )


def _should_auto_select_launch_mode(
    cfg: FmhaDecodeConfig,
    *,
    auto_tuner: bool,
    split_kv_mode: str,
    seq_len_q: int,
) -> bool:
    """Return whether automatic launch-mode selection is allowed for this shape."""
    single_query = seq_len_q == 1
    grouped_query = cfg.groups_tokens_heads_q and seq_len_q > 1
    return (
        auto_tuner
        and split_kv_mode == "disabled"
        and not cfg.use_persistent_scheduler
        and not cfg.use_attention_sinks
        and (
            single_query
            or grouped_query
            or cfg.use_variable_seqlens_q
            or cfg.use_sliding_window_causal
        )
    )


def _num_q_tiles_for_launch(cfg: FmhaDecodeConfig) -> int:
    """Return the physical Q-CTA multiplicity for launch occupancy."""
    q_geometry = make_q_tile_geometry(
        rows_per_cta=cfg.tile_size_q,
        heads_q_per_kv=cfg.heads_q_per_kv,
        groups_tokens_heads_q=cfg.groups_tokens_heads_q,
    )
    return max(q_geometry.num_q_ctas(cfg.max_seq_len_q), 1)


def _apply_auto_launch_mode(
    cfg: FmhaDecodeConfig,
    *,
    auto_tuner: bool,
    split_kv_mode: str,
    batch_size: int,
    num_heads_q: int,
    num_heads_kv: int,
    seq_len_kv: int,
    seq_len_q: int,
) -> str:
    """Apply the static/persistent/split-KV launch heuristic when it is allowed."""
    if not _should_auto_select_launch_mode(
        cfg,
        auto_tuner=auto_tuner,
        split_kv_mode=split_kv_mode,
        seq_len_q=seq_len_q,
    ):
        return split_kv_mode

    mode = _select_auto_launch_mode(
        batch_size=batch_size,
        num_heads_kv=num_heads_kv,
        seq_len_kv=seq_len_kv,
        num_q_tiles=_num_q_tiles_for_launch(cfg),
        tile_size_kv=cfg.tile_size_kv,
    )
    if (cfg.use_variable_seqlens_q or cfg.use_sliding_window_causal) and mode == (
        "gmem_reduction"
    ):
        # Runtime Q offsets and sliding-window bounds are compatible with CLC
        # work discovery, but they deliberately remain nonsplit. Underfilled
        # grids therefore stay direct while grids above one resident wave use
        # the same structural persistence rule as fixed-Q decode.
        return split_kv_mode
    if mode not in ("gmem_reduction", "persistent"):
        return split_kv_mode

    # An explicitly selected MMA profile can bypass the joint selector while
    # still leaving launch mode automatic. Probe the complete derived launch
    # before committing it, so an unsupported Keeps split/persistent recipe
    # falls back to the caller's valid direct profile.
    probe = deepcopy(cfg)
    if mode == "gmem_reduction":
        _apply_split_kv_config(
            probe,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            batch_size=batch_size,
            num_heads_kv=num_heads_kv,
            split_kv_mode=mode,
            splits_kv=-1,
            max_splits_kv=None,
            sliding_window_causal=False,
            attention_window_size=0,
        )
    else:
        probe.use_persistent_scheduler = True
    try:
        _validate_profile_support(
            cfg=probe,
            seq_len_q=seq_len_q,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            split_kv_mode=(mode if mode == "gmem_reduction" else "disabled"),
        )
    except ValueError:
        return split_kv_mode

    if mode == "persistent":
        cfg.use_persistent_scheduler = True
        return split_kv_mode
    if mode == "gmem_reduction":
        return mode
    return split_kv_mode


def _apply_split_kv_config(
    cfg: FmhaDecodeConfig,
    *,
    seq_len_q: int,
    seq_len_kv: int,
    batch_size: int,
    num_heads_kv: int,
    split_kv_mode: str,
    splits_kv: int,
    max_splits_kv: int | None,
    sliding_window_causal: bool,
    attention_window_size: int,
) -> None:
    """Resolve split-KV fanout and config flags for the selected reduction mode."""
    if split_kv_mode == "disabled":
        return

    heuristic_seq_len_kv = _effective_seq_len_for_sliding(
        seq_len_kv,
        sliding_window_causal,
        attention_window_size,
        cfg.tile_size_kv,
    )
    selected_splits_kv = select_splits_kv(
        seq_len_kv=heuristic_seq_len_kv,
        batch_size=batch_size,
        num_heads_kv=num_heads_kv,
        tile_size_kv=cfg.tile_size_kv,
        num_insts_kv=cfg.num_insts_kv,
        num_q_tiles=_num_q_tiles_for_launch(cfg),
        requested_splits_kv=splits_kv,
        max_splits_kv=max_splits_kv,
    )
    if selected_splits_kv <= 1:
        return

    cfg.use_split_kv = True
    cfg.splits_kv = selected_splits_kv
    cfg.max_splits_kv = selected_splits_kv
    cfg.use_persistent_scheduler = False
    if split_kv_mode == "gmem_reduction_with_separate_kernel":
        cfg.use_separate_reduction_kernel = True
    elif split_kv_mode == "cluster_smem_reduction":
        cfg.use_cluster_smem_reduction = True


_AUTO_SPLIT_KV_REDUCTION_MODES = (
    "gmem_reduction",
    "gmem_reduction_with_separate_kernel",
    "cluster_smem_reduction",
)


def _config_with_split_kv_mode(
    cfg: FmhaDecodeConfig, split_kv_mode: str
) -> FmhaDecodeConfig:
    """Return ``cfg`` with exactly one split-KV reduction mode selected."""

    return replace(
        cfg,
        use_cluster_smem_reduction=split_kv_mode == "cluster_smem_reduction",
        use_separate_reduction_kernel=(
            split_kv_mode == "gmem_reduction_with_separate_kernel"
        ),
    )


def _select_auto_split_kv_reduction_mode(
    cfg: FmhaDecodeConfig,
    *,
    seq_len_q: int,
    batch_size: int,
    num_heads_q: int,
    num_heads_kv: int,
    preserve_exact_cluster_fanout: bool,
) -> tuple[FmhaDecodeConfig, str]:
    """Select the first supported reduction mode from the shared policy.

    FlashInfer retains its general one-wave cluster search for legacy auto-derived
    fanouts. Jointly scored recipes and caller-provided fanouts remain exact,
    because changing their split count after selection would invalidate the
    TileQ/split comparison or the caller's request.
    """

    modes = select_split_kv_modes(
        family="fmha_decode",
        topology="1cta",
        tile_size_q=cfg.tile_size_q,
        head_dim=cfg.headdim,
        head_dim_per_cta_v=None,
        split_kv=cfg.splits_kv,
        available_modes=_AUTO_SPLIT_KV_REDUCTION_MODES,
    )
    for split_kv_mode in modes:
        trial = _config_with_split_kv_mode(cfg, split_kv_mode)
        if split_kv_mode == "cluster_smem_reduction":
            if not trial.can_use_cluster_smem_reduction:
                continue
            q_geometry = make_q_tile_geometry(
                rows_per_cta=trial.tile_size_q,
                heads_q_per_kv=num_heads_q // num_heads_kv,
                groups_tokens_heads_q=trial.groups_tokens_heads_q,
            )
            num_launched_clusters = (
                q_geometry.num_q_ctas(trial.max_seq_len_q) * num_heads_kv * batch_size
            )
            cluster_split = select_one_wave_cluster_split(
                initial_splits_kv=trial.splits_kv,
                minimum_splits_kv=(
                    trial.splits_kv if preserve_exact_cluster_fanout else 2
                ),
                num_launched_clusters=num_launched_clusters,
                tile_size_q=trial.tile_size_q,
                headdim=trial.headdim,
                correction_num_warps=trial.correction_num_warps,
            )
            if cluster_split is None:
                continue
            trial.splits_kv = cluster_split
            trial.max_splits_kv = cluster_split
        try:
            _validate_profile_support(
                cfg=trial,
                seq_len_q=seq_len_q,
                num_heads_q=num_heads_q,
                num_heads_kv=num_heads_kv,
                split_kv_mode=split_kv_mode,
            )
        except ValueError:
            continue
        return trial, split_kv_mode

    # Inline GMEM was the prior automatic behavior. Final validation preserves
    # its canonical error if the base profile itself is unsupported.
    return _config_with_split_kv_mode(cfg, "gmem_reduction"), "gmem_reduction"


def _validate_profile_support(
    *,
    cfg: FmhaDecodeConfig,
    seq_len_q: int,
    num_heads_q: int,
    num_heads_kv: int,
    split_kv_mode: str,
) -> None:
    """Reject unsupported profile combinations before kernel compilation."""

    headdim = cfg.headdim
    use_keeps_mma_ab = cfg.use_keeps_mma_ab
    use_groups_tokens_heads_q = cfg.groups_tokens_heads_q
    tile_size_q = cfg.tile_size_q
    cfg.validate_boolean_fields()
    _validate_kv256_static_config(cfg)
    if cfg.mask_type not in (DENSE, CAUSAL):
        raise ValueError("mask_type must be DENSE or CAUSAL")
    if cfg.use_paged_kv and not cfg.use_block_sparse:
        cfg.validate_paged_kv_staging_config()
    if num_heads_q % num_heads_kv != 0:
        raise ValueError("fmha_decode requires num_heads_q divisible by num_heads_kv")
    heads_q_per_kv = num_heads_q // num_heads_kv
    if cfg.heads_q_per_kv != heads_q_per_kv:
        raise ValueError(
            "heads_q_per_kv metadata must equal num_heads_q / num_heads_kv"
        )
    cfg.validate_block_sparse_profile(heads_q_per_kv=heads_q_per_kv)
    if use_groups_tokens_heads_q:
        make_q_tile_geometry(
            rows_per_cta=tile_size_q,
            heads_q_per_kv=heads_q_per_kv,
            groups_tokens_heads_q=True,
        )
    # Preserve the legacy fixed FP8 Q8/D128 profile. The D64/D256 extensions
    # are deliberately limited to grouped page-32 decode; broader dtypes,
    # head ratios, and variable-Q layouts remain unqualified.
    qualified_fp8_q8_separate_reduction_supported = (
        not use_keeps_mma_ab
        and cfg.use_split_kv
        and cfg.use_separate_reduction_kernel
        and not cfg.use_cluster_smem_reduction
        and cfg.q_dtype == Float8E4M3FN
        and cfg.kv_dtype == Float8E4M3FN
        and (
            (headdim == 128 and cfg.out_dtype == Float8E4M3FN)
            or (
                use_groups_tokens_heads_q
                and cfg.use_paged_kv
                and cfg.num_tokens_per_page == 32
                and (
                    (headdim == 64 and cfg.out_dtype == Float8E4M3FN)
                    or (headdim == 256 and cfg.out_dtype == Float16)
                )
            )
        )
        and tile_size_q == 8
        and heads_q_per_kv == 8
        and seq_len_q == 1
        and cfg.max_seq_len_q == 1
        and not cfg.use_variable_seqlens_q
        and cfg.splits_kv == cfg.max_splits_kv
        and 2 <= cfg.max_splits_kv <= 128
    )
    if cfg.use_separate_reduction_kernel:
        cluster_size = cfg.parallel_reduction_cluster_size
        splits_per_cta = cfg.parallel_reduction_splits_per_cta
        padded_splits = cfg.parallel_reduction_padded_splits
        default_cluster_size = {
            8: 1,
            16: 2,
            32: 4,
            64: 8,
            128: 16,
        }.get(padded_splits)
        compact_topology_supported = (
            cfg.use_compact_parallel_reduction
            and cluster_size == 1
            and splits_per_cta == cfg.max_splits_kv
        )
        clustered_topology_supported = (
            not cfg.use_compact_parallel_reduction
            and default_cluster_size == cluster_size
            and splits_per_cta in (2, 4, 8)
            and cluster_size * splits_per_cta == padded_splits
        )
        if not (
            cfg.use_split_kv
            and cfg.splits_kv == cfg.max_splits_kv
            and 2 <= cfg.max_splits_kv <= 128
            and (compact_topology_supported or clustered_topology_supported)
        ):
            raise ValueError(
                "separate GMEM reduction requires equal static "
                "splits_kv/max_splits_kv in [2,128] using the production "
                "compact or padded clustered topology"
            )
        if (
            cluster_size > 1
            and get_max_active_clusters_for_cluster_size(cluster_size) <= 0
        ):
            raise ValueError(
                "parallel separate reduction cluster size "
                f"{cluster_size} is not supported on this device"
            )
    supports_grouped_keeps = cfg.supports_grouped_keeps
    if cfg.tile_size_kv != 128 and not (
        cfg.tile_size_kv == 256 and supports_grouped_keeps
    ):
        raise ValueError(
            "wide KeepsMmaAb is enabled only for the qualified FP16/BF16/D128 "
            "KV256 native warp-specialized profile"
        )
    if use_keeps_mma_ab and use_groups_tokens_heads_q:
        if not supports_grouped_keeps:
            raise ValueError(
                "grouped KeepsMmaAb currently supports only validated narrow "
                "profiles; pass groups_tokens_heads_q=False to use a supported "
                "ungrouped Keeps profile"
            )
    if cfg.use_variable_seqlens_q:
        # Packed Q supports the broad grouped Swaps matrix plus the narrow
        # grouped Keeps direct profile validated above.
        if use_keeps_mma_ab and not (
            use_groups_tokens_heads_q and supports_grouped_keeps
        ):
            raise ValueError(
                "packed variable-Q KeepsMmaAb requires its supported grouped "
                "static direct profile"
            )
        if not use_groups_tokens_heads_q and cfg.use_cluster_smem_reduction:
            raise ValueError(
                "packed ungrouped SwapsMmaAb does not support cluster SMEM reduction; "
                "use grouped Q or a GMEM reduction mode"
            )
    if use_keeps_mma_ab:
        if headdim not in (64, 128, 256):
            raise ValueError("fmha_decode keepsMmaAb supports headdim=64, 128, or 256")
        if tile_size_q not in (64, 128):
            raise ValueError("fmha_decode keepsMmaAb supports tile_size_q=64 or 128")
        effective_head_dim_stage = cfg.head_dim_per_stage_kv
        effective_num_insts_kv = cfg.num_insts_kv
        effective_o_stages = cfg.o_stages
        if effective_num_insts_kv == 1 and (
            headdim != 256 or effective_head_dim_stage != 128 or effective_o_stages != 1
        ):
            raise ValueError(
                "one-instance KeepsMmaAb is enabled only for the staged "
                "headDim=256 profile with head_dim_per_stage_kv=128 and "
                "o_stages=1"
            )
        if headdim == 256 and (
            effective_head_dim_stage != 128
            or effective_num_insts_kv != 1
            or effective_o_stages != 1
        ):
            raise ValueError(
                "fmha_decode keepsMmaAb headDim=256 requires "
                "head_dim_per_stage_kv=128, num_insts_kv=1, and o_stages=1"
            )
        if headdim != 256 and effective_head_dim_stage != 0:
            raise ValueError(
                "split head_dim_per_stage_kv keepsMmaAb profiles are enabled "
                "only for headDim=256"
            )
        if not use_groups_tokens_heads_q and heads_q_per_kv != tile_size_q:
            raise ValueError(
                "fmha_decode keepsMmaAb requires numHeadsQPerKv == tile_size_q"
            )
        if cfg.q_dtype == Float8E4M3FN and cfg.out_dtype not in (
            Float16,
            Float8E4M3FN,
        ):
            raise ValueError(
                "fmha_decode keepsMmaAb fp8 qkv path supports fp16 or fp8 output"
            )
        use_split_kv = split_kv_mode != "disabled" or cfg.use_split_kv
        if use_split_kv:
            if cfg.q_dtype not in (Float16, BFloat16, Float8E4M3FN):
                raise ValueError(
                    "split-KV keepsMmaAb profiles support only fp16, bf16, or fp8 qkv"
                )
        if cfg.use_cluster_smem_reduction:
            raise ValueError(
                "fmha_decode does not support cluster SMEM reduction with keepsMmaAb"
            )
        use_separate_reduction_kernel = cfg.use_separate_reduction_kernel
        separate_reduction_q_layout_supported = (
            not use_groups_tokens_heads_q and heads_q_per_kv == tile_size_q
        ) or use_groups_tokens_heads_q
        separate_reduction_unstaged_supported = (
            headdim == 128
            and tile_size_q in (64, 128)
            and effective_head_dim_stage == 0
            and effective_num_insts_kv == 2
            and effective_o_stages == 2
        )
        separate_reduction_h256_supported = (
            headdim == 256
            and tile_size_q == 128
            and effective_head_dim_stage == 128
            and effective_num_insts_kv == 1
            and effective_o_stages == 1
        )
        fixed_fp8_new_keeps_separate_reduction_supported = (
            (tile_size_q, headdim) in ((64, 64), (64, 256), (128, 64))
            and not use_groups_tokens_heads_q
            and seq_len_q == 1
            and cfg.max_seq_len_q == 1
            and not cfg.use_variable_seqlens_q
            and cfg.use_paged_kv
            and cfg.num_tokens_per_page == 32
            and cfg.q_dtype == Float8E4M3FN
            and cfg.kv_dtype == Float8E4M3FN
            and cfg.out_dtype == (Float16 if headdim == 256 else Float8E4M3FN)
            and cfg.splits_kv == cfg.max_splits_kv
            and 2 <= cfg.max_splits_kv <= 128
        )
        separate_reduction_profile_supported = (
            separate_reduction_unstaged_supported
            or separate_reduction_h256_supported
            or fixed_fp8_new_keeps_separate_reduction_supported
        )
        if use_separate_reduction_kernel and not (
            separate_reduction_profile_supported
            and separate_reduction_q_layout_supported
            and cfg.supports_reduction_dtypes
            and use_split_kv
        ):
            raise ValueError(
                "separate reduction keepsMmaAb profiles require "
                "the established D128/Q64-Q128 or D256/Q128 profiles, or a "
                "fixed FP8/page-32 D64/Q64-Q128 or D256/Q64 profile; valid "
                "reduction dtypes, Q layout, and static split-KV are required"
            )
        return
    use_split_kv = split_kv_mode != "disabled" or cfg.use_split_kv
    # SwapsMmaAb tensor maps provide OOB fill for a final partial head band.
    # Keep the ungrouped one-token-per-CTA control available at the same tile Q
    # as grouped profiles so explicit ungrouped launches retain the MMA shape.
    effective_head_dim_stage = cfg.head_dim_per_stage_kv
    effective_num_insts_kv = cfg.num_insts_kv
    if headdim == 256:
        if effective_head_dim_stage != 128 or effective_num_insts_kv != 2:
            raise ValueError(
                "fmha_decode SwapsMmaAb headDim=256 requires "
                "head_dim_per_stage_kv=128 and num_insts_kv=2"
            )
    elif effective_head_dim_stage != 0:
        raise ValueError(
            "split head_dim_per_stage_kv SwapsMmaAb profiles are enabled only "
            "for headDim=256"
        )
    if cfg.use_cluster_smem_reduction:
        # Single source of truth for structural eligibility plus dtype and
        # SMEM-budget checks. Grouped cluster is an explicit static choice.
        if not (cfg.supports_reduction_dtypes and cfg.can_use_cluster_smem_reduction):
            raise ValueError(
                "cluster SMEM reduction requires static SwapsMmaAb headDim in "
                "{64,128,256}, TileSizeQ in {8,16,32}, "
                "either ungrouped single-token or complete-token grouped Q, "
                "at least two split CTAs, "
                "and fp16/bf16 or fp8 qkv with fp16/fp8 output"
            )
        cluster_reason = cluster_smem_reduction_unsupported_reason(
            max_splits_kv=cfg.max_splits_kv,
            splits_kv=cfg.splits_kv,
            tile_size_q=cfg.tile_size_q,
            headdim=headdim,
            correction_num_warps=cfg.correction_num_warps,
        )
        if cluster_reason:
            raise ValueError(f"cluster SMEM reduction rejected: {cluster_reason}")
    use_separate_reduction_kernel = cfg.use_separate_reduction_kernel
    separate_reduction_supported = (
        use_split_kv
        and cfg.supports_reduction_dtypes
        and (seq_len_q == 1 or cfg.use_variable_seqlens_q or use_groups_tokens_heads_q)
        and (tile_size_q in (16, 32) or qualified_fp8_q8_separate_reduction_supported)
        and headdim >= 64
    )
    if use_separate_reduction_kernel and not separate_reduction_supported:
        raise ValueError(
            "separate reduction SwapsMmaAb profiles require fixed SQ=1, "
            "fixed grouped Q, or packed variable Q; tile_size_q in {16,32} "
            "or a fixed FP8 Q8/HqPerKv8 profile (legacy D128 with FP8 output, "
            "plus grouped paged-KV/page-32 D64 with FP8 output or D256 with "
            "FP16 output); valid reduction dtypes and static split-KV are required"
        )
    if tile_size_q and tile_size_q not in (8, 16, 32):
        raise ValueError("fmha_decode SwapsMmaAb supports tile_size_q in {8,16,32}")
    if headdim not in (64, 128, 256):
        raise ValueError(
            "fmha_decode SwapsMmaAb supports headDim in "
            "{64,128,256}. headDim=256 uses the staged profile with "
            "head_dim_per_stage_kv=128 and num_insts_kv=2."
        )


def normalize_qkv_layout(qkv_layout: str) -> str:
    """Normalize CLI aliases to the canonical QKV layout names."""
    normalized = qkv_layout.strip().lower()
    if normalized in ("contiguous", "contiguouskv", "dense"):
        return "contiguousKv"
    if normalized in ("paged", "pagedkv", "page-index", "page_index"):
        return "pagedKv"
    raise ValueError(f"Unsupported qkv_layout: {qkv_layout}")


def validate_page_size(num_tokens_per_page: int) -> None:
    """Validate a paged-KV page size against supported tile shapes."""
    if num_tokens_per_page not in (16, 32, 64, 128):
        raise ValueError("num_tokens_per_page must be one of 16, 32, 64, or 128")
    if 128 % num_tokens_per_page != 0:
        raise ValueError("num_tokens_per_page must divide the 128-token KV tile")


def make_decode_config(
    headdim: int = 128,
    args: object | None = None,
    *,
    seq_len_q: int = 1,
    seq_len_kv: int | None = None,
    batch_size: int | None = None,
    num_heads_q: int | None = None,
    num_heads_kv: int | None = None,
    qkv_dtype: type = Float16,
    o_dtype: type = Float16,
    qkv_layout: str = "contiguousKv",
    num_tokens_per_page: int = 32,
    split_kv_mode: str = "disabled",
    splits_kv: int = -1,
    max_splits_kv: int | None = None,
    sliding_window_causal: bool = False,
    attention_window_size: int = 0,
    mask_type: str | None = None,
    use_attention_sinks: bool = False,
    auto_tuner: bool = True,
) -> FmhaDecodeConfig:
    """Build the static decode kernel config and apply auto-selection policy.

    When no launch-shape inputs are supplied, this is the simple static config
    constructor: it reads only ``FmhaDecodeConfig`` fields from ``args`` and
    applies static profile defaults. When launch-shape inputs are supplied, it
    additionally runs the decode kernel-selection workflow below.

    Workflow:
    1. Create a default ``FmhaDecodeConfig`` and apply caller-supplied config
       fields directly onto it, remembering which profile fields were explicit.
    2. Fill profile-derived defaults: dtype fields, paged-KV metadata, the
       dense/causal mask, sliding-window flags, attention-sink flags, default
       grouped-Q metadata for fixed or packed launches, and the SMEM-derived
       KV stage count. For an unpinned fixed multi-Q paged-causal page-32
       launch, the Q selector first enumerates Swaps8/16/32 and Keeps64/128
       tiles and every useful direct/split recipe through the first
       capacity-crossing fanout.
       Production-valid recipes minimize the empirical TileQ
       mainloop-plus-reduction proxy using their actual Q-grid CTA waves.
       Only a selected Q64 Keeps recipe may then promote a qualified FP16/BF16
       D128 launch from KV128 to KV256; every other automatic Q tile retains
       KV128. The final KV width re-derives launch policy instead of reusing a
       fanout scored for KV128. Explicit policies remain caller-controlled.
       TileQ128 remains automatic over TileQ64 only for staged D256. SQ1 is
       outside this grouped-Q cost model and therefore remains KV128.
    3. Shapes outside that qualified Q/launch selector retain the general launch
       policy: under-filled fixed-Q long-sequence grids use split-KV GMEM
       reduction, direct grids above one resident wave use persistent
       scheduling, and the rest stay static. Packed-Q and sliding-window grids
       remain nonsplit but use the same structural persistence boundary. An
       unsupported automatic mode falls back to direct.
    4. If split-KV is selected or requested, compute the split fanout from the
       effective KV length, requested split count, max split cap, SM count, and
       ``batch_size * num_heads_kv * q_tiles`` physical grid size.
    5. Choose the reduction mode for an automatic split-KV launch: prefer a
       legal one-wave cluster configuration, otherwise use the standalone GMEM
       reducer with inline reduction as a support fallback. A jointly scored
       recipe keeps its exact fanout; the legacy launch path may search
       downward for a one-wave cluster split.
    6. Validate the final profile combination before returning the config.

    Attention-sink paths skip automatic launch-mode selection. Explicit launch
    modes are still validated.
    """
    shape_values = (seq_len_kv, batch_size, num_heads_q, num_heads_kv)
    if all(value is None for value in shape_values):
        return _make_static_decode_config(
            headdim,
            args,
            mask_type=mask_type,
            sliding_window_causal=sliding_window_causal,
        )
    if any(value is None for value in shape_values):
        raise ValueError(
            "seq_len_kv, batch_size, num_heads_q, and num_heads_kv are all "
            "required for decode kernel selection"
        )

    validate_sliding_window_args(sliding_window_causal, attention_window_size)
    cfg = FmhaDecodeConfig(headdim=headdim)
    explicit_fields = _apply_config_source(cfg, args)
    splits_kv, max_splits_kv = _resolve_explicit_split_controls(
        cfg,
        explicit_fields=explicit_fields,
        splits_kv=splits_kv,
        max_splits_kv=max_splits_kv,
    )
    _apply_mask_type_config(
        cfg,
        source=args,
        mask_type=mask_type,
        sliding_window_causal=(sliding_window_causal or cfg.use_sliding_window_causal),
        explicit_fields=explicit_fields,
    )
    split_kv_mode = _validate_split_kv_mode(split_kv_mode)
    # Whether the launch mode was left to the auto-tuner rather than explicitly
    # chosen by the caller. Only auto-derived modes are eligible for the cluster
    # promotion below.
    launch_mode_was_auto = split_kv_mode == "disabled"
    _fanout_was_explicit = splits_kv > 0
    if num_heads_q <= 0 or num_heads_kv <= 0:
        raise ValueError("fmha_decode head counts must be positive")
    if num_heads_q % num_heads_kv != 0:
        raise ValueError("fmha_decode requires num_heads_q divisible by num_heads_kv")
    _apply_default_q_grouping(
        cfg,
        explicit_fields=explicit_fields,
    )
    cfg.q_dtype = qkv_dtype
    cfg.kv_dtype = qkv_dtype
    cfg.out_dtype = o_dtype

    qkv_layout = _apply_layout_config(
        cfg,
        qkv_layout=qkv_layout,
        num_tokens_per_page=num_tokens_per_page,
        seq_len_kv=seq_len_kv,
    )
    _apply_feature_config(
        cfg,
        explicit_fields=explicit_fields,
        seq_len_q=seq_len_q,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        sliding_window_causal=sliding_window_causal,
        attention_window_size=attention_window_size,
        use_attention_sinks=use_attention_sinks,
    )
    if not cfg.use_variable_seqlens_q and cfg.max_seq_len_q != seq_len_q:
        raise ValueError(
            "fixed-Q max_seq_len_q must equal seq_len_q; use "
            "use_variable_seqlens_q for a runtime-varying Q length"
        )
    # Resolve Q first. The KV selector consumes the multi-Q cost-model result
    # instead of forcing a Q64 profile ahead of the established Q policy.
    auto_selection_explicit_fields = set(explicit_fields)
    selected_grouped_q_recipe = _apply_auto_grouped_q_mma_config(
        cfg,
        explicit_fields=explicit_fields,
        auto_tuner=auto_tuner,
        split_kv_mode=split_kv_mode,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        batch_size=batch_size,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        splits_kv=splits_kv,
        max_splits_kv=max_splits_kv,
    )
    if selected_grouped_q_recipe is None:
        _apply_swaps_tile_config(
            cfg,
            explicit_fields=explicit_fields,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
        )
    kv_tile_was_promoted = _try_apply_auto_kv256_profile(
        cfg,
        q_candidate=(
            selected_grouped_q_recipe.mma
            if selected_grouped_q_recipe is not None
            else None
        ),
        explicit_fields=auto_selection_explicit_fields,
        auto_tuner=auto_tuner,
        split_kv_mode=split_kv_mode,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        splits_kv=splits_kv,
        max_splits_kv=max_splits_kv,
    )
    if kv_tile_was_promoted:
        # The selected Q tile remains valid, but its KV128 launch fanout, score,
        # and possible persistent side effect do not. Re-derive launch policy
        # below from the final KV256 work granularity.
        cfg.use_persistent_scheduler = False
        selected_grouped_q_recipe = None
    _finalize_static_decode_config(cfg, explicit_fields)
    if not cfg.use_variable_seqlens_q:
        validate_causal_decode_lengths(
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            mask_type=cfg.mask_type,
        )

    # Auto launch-mode selection. Only kicks in if the caller has not already
    # opted into a specific mode. Packed-Q and sliding-window shapes may select
    # CLC persistence, but remain nonsplit.
    if selected_grouped_q_recipe is not None:
        if selected_grouped_q_recipe.splits_kv > 1:
            split_kv_mode = selected_grouped_q_recipe.split_kv_mode
            splits_kv = selected_grouped_q_recipe.splits_kv
            max_splits_kv = selected_grouped_q_recipe.splits_kv
    elif not (_LAUNCH_SELECTION_FIELDS & explicit_fields):
        if split_kv_mode == "disabled" and splits_kv > 1:
            split_kv_mode = "gmem_reduction"
        elif splits_kv <= 0:
            split_kv_mode = _apply_auto_launch_mode(
                cfg,
                auto_tuner=auto_tuner,
                split_kv_mode=split_kv_mode,
                batch_size=batch_size,
                num_heads_q=num_heads_q,
                num_heads_kv=num_heads_kv,
                seq_len_kv=seq_len_kv,
                seq_len_q=seq_len_q,
            )

    reduction_mode_preconfigured = (
        cfg.use_cluster_smem_reduction or cfg.use_separate_reduction_kernel
    )
    auto_split_kv_selected = (
        auto_tuner
        and launch_mode_was_auto
        and split_kv_mode == "gmem_reduction"
        and not reduction_mode_preconfigured
        and not cfg.use_variable_seqlens_q
        and not cfg.use_sliding_window_causal
        and not cfg.use_attention_sinks
    )

    _apply_split_kv_config(
        cfg,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        batch_size=batch_size,
        num_heads_kv=num_heads_kv,
        split_kv_mode=split_kv_mode,
        splits_kv=splits_kv,
        max_splits_kv=max_splits_kv,
        sliding_window_causal=sliding_window_causal,
        attention_window_size=attention_window_size,
    )

    if auto_split_kv_selected and cfg.use_split_kv:
        cfg, split_kv_mode = _select_auto_split_kv_reduction_mode(
            cfg,
            seq_len_q=seq_len_q,
            batch_size=batch_size,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            preserve_exact_cluster_fanout=(
                _fanout_was_explicit or selected_grouped_q_recipe is not None
            ),
        )

    _validate_kv256_static_config(cfg)
    _finalize_warp_roles(cfg)

    _validate_profile_support(
        cfg=cfg,
        seq_len_q=seq_len_q,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        split_kv_mode=split_kv_mode,
    )
    return cfg
