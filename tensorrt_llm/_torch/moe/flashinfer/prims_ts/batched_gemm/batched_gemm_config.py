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

"""Configuration for the BatchedGemm TS example.

All kernel-wide constants, feature-flag enums, and per-variant knobs.
Only this file may use `from __future__ import annotations`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum
from cutlass.experimental import primitives as prims

from tensorrt_llm._torch.moe.flashinfer.tllm_enums import WeightLayout


# `SmemSfLdgstsResource._tail_cp_async_wait_group` lowers the runtime
# `cp.async.wait_group` drain for short K depths by hand-rolling one branch
# per static `cp_async_wait_group_depth` value (depth 0/1/2).  Supporting
# deeper prefetch windows would require either an exponentially larger
# switch or a generic runtime helper, so cap the SFB LDGSTS
# producer-commit prefetch window at this depth and reject anything
# larger at resource construction time.
MAX_PRODUCER_COMMIT_PREFETCH_DEPTH: int = 3


class BatchMode(IntEnum):
    BATCH_N = 0
    BATCH_M = 1


class RouteImpl(IntEnum):
    NONE = 0
    TMA = 1
    LDGSTS = 2
    LDG_PLUS_STS = 3


class TileScheduler(IntEnum):
    STATIC = 0
    PERSISTENT = 1


class ActKind(IntEnum):
    NONE = 0
    SWIGLU = 1
    GEGLU = 2
    RELU2 = 3
    SILU = 4
    SITU = 5


class BiasType(IntEnum):
    NONE = 0
    M = 1


class DType(IntEnum):
    FP32 = 1
    FP16 = 2
    BF16 = 3
    E2M1 = 4
    MXE2M1 = 5
    MXE4M3 = 6
    E4M3 = 7


class SfLayout(IntEnum):
    R8c4 = 1
    LINEAR = 2
    R128c4 = 3


class SfSmemToTmemCopy(IntEnum):
    NONE = 0
    # No separate CopySf task; UTCCP is fused into the MMA warp.
    FUSED_UTCCP = 1
    UTCCP = 2
    LDS_STTM = 3


MX_MMA_FORMAT_E4M3 = 0
MX_MMA_FORMAT_E2M1 = 5

# A 32-bit TMEM SF word packs four 1-byte scale factors.  MMA selects one byte
# from the word with the 2-bit SF id, so one SF K-group spans four SF vectors.
TMEM_SF_PACK_SIZE_BYTES = 4

# S2T_32x128b_WARPX4 UTCCP copies 128 bits at a time.  Each SF TMEM column is
# one 32-bit packed-SF word, so one copy covers 128 / (8 * 4) = 4 columns.
TMEM_SF_UTCCP_COLS_PER_COPY = 4


DTYPE_BITS = {
    int(DType.FP32): 32,
    int(DType.FP16): 16,
    int(DType.BF16): 16,
    int(DType.E2M1): 4,
    int(DType.MXE2M1): 4,
    int(DType.MXE4M3): 8,
    int(DType.E4M3): 8,
}


def dtype_bits(dtype_kind: object) -> int:
    """Return the logical bit width for a concrete DType value."""
    kind = int(dtype_kind)
    if kind not in DTYPE_BITS:
        raise ValueError(f"Unsupported dtype kind: {kind}")
    return DTYPE_BITS[kind]


def tmem_d_cols_per_stage(tile_n: int) -> int:
    """Return accumulator TMEM columns consumed by one D stage."""
    return tile_n


def tmem_sf_col_stride(tile_mn: int) -> int:
    """Return TMEM columns for one group of four scale-factor K vectors."""
    return 2 * ((tile_mn + 63) // 64)


def tmem_sf_k_groups(tile_k: int, sf_vec_size: int) -> int:
    """Return the number of four-vector SF groups in one K tile."""
    if tile_k % sf_vec_size != 0:
        raise ValueError(
            f"tile_k must be divisible by sf_vec_size={sf_vec_size}, got {tile_k}"
        )
    sf_group_k = TMEM_SF_PACK_SIZE_BYTES * sf_vec_size
    return (tile_k + sf_group_k - 1) // sf_group_k


def tmem_sf_cols(tile_mn: int, tile_k: int, sf_vec_size: int) -> int:
    """Return TMEM columns used by one scale-factor operand stage."""
    return tmem_sf_k_groups(tile_k, sf_vec_size) * tmem_sf_col_stride(tile_mn)


def uniform_pipeline_stage_overrides(
    load_stages: int,
    *,
    sf_stages: int | None = None,
    tmem_acc_stages: int | None = None,
) -> dict[str, int]:
    """Build explicit stage overrides for generated variants with one stage count."""
    sf_stages = load_stages if sf_stages is None else sf_stages
    result = {
        "num_stages_a": int(load_stages),
        "num_stages_b": int(load_stages),
        "num_stages_smem_sfa": int(sf_stages),
        "num_stages_smem_sfb": int(sf_stages),
        "num_stages_tmem_sfa": int(sf_stages),
        "num_stages_tmem_sfb": int(sf_stages),
    }
    if tmem_acc_stages is not None:
        result["num_stages_tmem_acc"] = int(tmem_acc_stages)
    return result


@dataclass(kw_only=True)
class BatchedGemmConfig:
    """Compile-time configuration the BatchedGemm TS kernel is generated from.

    Holds every static knob: tile/MMA shape, operand and output dtypes,
    scale-factor settings, pipeline depths, warp/register budgets, routing,
    scheduling, and epilogue options. Fields are plain ``int`` (enums are stored
    as their integer value) so the DSL can trace them. The many read-only
    ``@property`` members derive secondary quantities (MMA kind, SMEM/TMEM
    footprints, layout choices) from these fields.

    Allowed values and cross-field constraints are enforced by
    :func:`validate_config`.

    .. note::
        Warp indices/counts and :attr:`threads_per_cta` are **recomputed in
        place** by :func:`compute_warp_layout` during kernel assembly; their
        field defaults are placeholders. Do not set them, except
        :attr:`num_load_b_warps`, :attr:`num_load_sfa_warps`, and
        :attr:`num_load_sfb_warps`, which are honored only when ``> 1``.
    """

    # --- Shape ---
    cluster_m: int = 1
    """Number of CTAs per cluster along the M axis.

    ``1`` disables clustering; ``2`` forms a 2-CTA cluster that multicasts the
    shared operand over cluster-scoped mbarriers. Drives :attr:`has_cluster`,
    :attr:`cta_group`, and per-CTA B sizing (:attr:`split_b_across_ctas`).
    [FIXME]:attr:`use_max_tmem_overlap` requires ``cluster_m == 2``.
    """

    epi_tile_m: int = 128
    """Epilogue tile M (rows processed per epilogue stage), in elements.

    Typically equals :attr:`tile_m` (``128``). Sets the C SMEM staging size
    (:attr:`num_bytes_c_per_stage`) and the epilogue reference block shuffle.
    """

    epi_tile_n: int = 8
    """Epilogue tile N (output columns per epilogue call), in elements.

    When smaller than :attr:`tile_n` the epilogue loops ``tile_n // epi_tile_n``
    times; also drives the TMEM-to-register (T2R) fragment layout.
    """

    mma_k: int = 64
    """MMA instruction K depth, in elements; one of ``{16, 32, 64}``.

    Must match the MMA dtype (:func:`validate_config`): ``16`` for BF16/CastA,
    ``32`` for MX and FP8, ``64`` for NVFP4. :attr:`tile_k` must be a multiple
    of it.
    """

    mma_m: int = 128
    """MMA instruction M dimension, in elements (passed to ``Tcgen05InstrDesc``).

    Generated TS kernels use ``64`` for DeepSeek FP8 split-epilogue shapes,
    ``128`` for 1-CTA shapes, and ``256`` for 2-CTA high-throughput shapes.
    ``validate_config`` range-checks this broad generated-kernel domain; exact
    tcgen05 descriptor legality is still enforced by ``Tcgen05InstrDesc.build``.
    """

    mma_n: int = 8
    """MMA instruction N dimension, in elements (passed to ``Tcgen05InstrDesc``).

    Generated TS kernels use ``{8, 16, 32, 64, 128, 256}``.
    ``validate_config`` range-checks this broad generated-kernel domain; exact
    tcgen05 descriptor legality is still enforced by ``Tcgen05InstrDesc.build``.
    """

    tile_k: int = 512
    """GEMM tile K depth (reduction length per K-tile), in elements.

    Must be a multiple of :attr:`mma_k` for the active dtype
    (:func:`validate_config`).

    .. note::
        Routed-SF TMA gather4 needs ``sf_k >= 32`` for the routed operand, where
        ``sf_k = tile_k // input_sf_block_size_{a,b}``. The producer-commit
        prefetch guard also requires ``3 * tile_k <= K``. CastA currently
        requires ``tile_k = 256`` because the cast stage produces a fixed
        128x256 BF16 tile in TMEM.
    """

    tile_m: int = 128
    """GEMM tile M dimension (rows per CTA), in elements.

    Restricted to ``128`` by :func:`validate_config`. Sets the SFA row count and
    :attr:`num_bytes_a_per_stage`.
    """

    tile_n: int = 8
    """GEMM tile N dimension (token/output columns per CTA), in elements.

    One of ``{8, 16, 32, 64, 128, 256}`` (:func:`validate_config`). Drives the
    SFB layout (:attr:`uses_sfb_8x4_load`), the tile256 overlap path
    (:attr:`use_tile256_tmem_overlap`), and the epilogue warp count.
    """

    # --- Semantic dtypes ---
    dtype_a: int = int(DType.E2M1)
    """Operand A element type, as a :class:`DType` integer value.

    With :attr:`dtype_b` it selects the MMA kind: NVFP4 (``E2M1``x``E2M1``), MX
    (``MXE2M1``/``MXE4M3`` on both sides), FP8 (``E4M3``x``E4M3``), BF16
    (``BF16``x``BF16``), or CastA (``MXE2M1`` A x ``BF16`` B). See
    :attr:`dtype_a_smem_bits` / :attr:`dtype_a_tma_bits` for the staged widths.
    """

    dtype_acc: int = int(DType.FP32)
    """Accumulator element type. **Must be** ``int(DType.FP32)``.

    Enforced in :meth:`__post_init__`; every MMA path accumulates in FP32
    regardless of input or output dtype.
    """

    dtype_b: int = int(DType.E2M1)
    """Operand B element type, as a :class:`DType` integer value.

    Paired with :attr:`dtype_a` to select the MMA kind (see :attr:`dtype_a`).
    """

    dtype_c: int = int(DType.BF16)
    """Output (epilogue) element type, as a :class:`DType` integer value.

    ``BF16``/``FP16`` store plain; ``E4M3`` is plain FP8 output only for FP8 MMA
    (:attr:`uses_fp8_output`); ``E2M1``, ``MXE2M1``, and ``MXE4M3`` enable output
    quantization with block scales (:attr:`uses_fp4_output_quant` /
    :attr:`uses_mx_output_quant`). The accumulator stays FP32 -- this is the
    final cast. Output quantization requires an :attr:`is_swap_ab` gated epilogue
    (:func:`validate_config`).
    """

    weight_layout: int = int(WeightLayout.MajorK)
    """Physical layout of the per-expert weight operand.

    ``MajorK`` keeps the existing storage, logically ``[expert, Mn, K]``.
    ``BlockMajorK`` stores the weight matrix as
    ``[expert, K_bytes / block_bytes, Mn, block_bytes]`` and is exposed to TMA
    as ``(blockK_elems, Mn, K / blockK_elems, expert)``. ``block_bytes`` follows
    TRT-LLM Gen's BlockMajorK rule: start from the number of elements in a 128B
    K block, then use the padded smaller block only when the padded K tile spans
    more than two 128B SMEM slices. The layout always applies to the weight
    operand: A when :attr:`is_swap_ab`, B otherwise.
    """

    sf_bits: int = 8
    """Scale-factor storage width in bits. Default ``8``.

    .. note::
        Reserved. No code path varies on this value -- block scales are always
        8-bit UE8M0, and the per-token SF dtype is set by
        :attr:`per_token_sf_dtype`.
    """

    sf_block_size_a: int = 0
    """Block length (K elements) per A scale factor; ``0`` = auto.

    When ``0``, :attr:`input_sf_block_size_a` resolves to ``32`` for MX/CastA,
    else ``16``. MX inputs must use ``32`` (:func:`validate_config`).
    """

    sf_block_size_b: int = 0
    """Block length (K elements) per B scale factor; ``0`` = auto.

    When ``0``, :attr:`input_sf_block_size_b` resolves to ``32`` for MX, else
    ``16``. MX inputs must use ``32`` (:func:`validate_config`).
    """

    sf_block_size_c: int = 0
    """Block length per output (C) scale factor; ``0`` = auto.

    When ``0``, :attr:`output_sf_block_size_c` resolves to ``32`` for MXFP4/MXFP8
    output, else ``16``. NVFP4 output must use ``16`` and MX output ``32``
    (:func:`validate_config`). Only meaningful when the output is quantized
    (:attr:`has_epilogue_quant`).
    """

    # --- Pipeline stages ---
    num_stages_a: int = 5
    """Depth of the operand-A SMEM load pipeline ring (TMA prefetch).

    Sizes ``num_bytes_a_per_stage * num_stages_a`` of SMEM.
    """

    num_stages_b: int = 5
    """Depth of the operand-B SMEM load pipeline ring.

    Per-stage SMEM depends on :attr:`split_b_across_ctas` (clustered kernels
    stage only a slice; see :attr:`num_bytes_b_smem_per_stage`).
    """

    num_stages_c_smem: int = 2
    """Depth of the epilogue C SMEM staging pipeline ring.

    Used only with :attr:`use_tma_store`; see :attr:`num_bytes_c_smem_scratch`.
    """

    num_stages_smem_sfa: int = 5
    """Depth of the scale-factor-A SMEM pipeline ring (block-scaled kernels only).

    Distinct from the TMEM-side depth :attr:`num_stages_tmem_sfa`.
    """

    num_stages_smem_sfb: int = 5
    """Depth of the scale-factor-B SMEM pipeline ring (block-scaled kernels only)."""

    num_stages_tmem_acc: int = 2
    """Depth of the accumulator (C) TMEM pipeline ring; ``1`` or ``2``.

    Contributes to the 512-column TMEM budget (:attr:`tmem_required_cols`).

    .. note::
        Must be ``1`` when :attr:`use_max_tmem_overlap` is set
        (:func:`validate_config`).
    """

    num_stages_tmem_sfa: int = 5
    """Depth of the scale-factor-A TMEM pipeline ring after the SMEM->TMEM copy.

    Allocated only on the unfused copy path (:attr:`uses_unfused_tmem_sf_copy`).
    """

    num_stages_tmem_sfb: int = 5
    """Depth of the scale-factor-B TMEM pipeline ring (unfused copy path only)."""

    num_stages_workid: int = 3
    """Depth of the persistent CLC work-id fetch pipeline ring.

    Used only when :attr:`is_persistent`; a value ``> 1`` prevents C scratch from
    aliasing the A/B buffers (:attr:`aliases_c_scratch_with_ab`).
    """

    # --- Warp assignments ---
    # All indices and most counts below are OUTPUTS of compute_warp_layout();
    # the defaults are placeholders. See the class note and R5 in the design doc.
    cast_a_warp_idx_first: int = 0
    """First warp of the CastA group. Output of :func:`compute_warp_layout`."""

    copy_sfa_warp_idx: int = 12
    """Warp index of the CopySfA task. Output of :func:`compute_warp_layout`."""

    copy_sfb_warp_idx: int = 4
    """Warp index of the CopySfB task. Output of :func:`compute_warp_layout`."""

    epilogue_warp_idx: int = 0
    """First warp of the epilogue group. Output of :func:`compute_warp_layout`."""

    load_a_warp_idx: int = 10
    """Warp index of the LoadA task. Output of :func:`compute_warp_layout`."""

    load_b_warp_idx: int = 8
    """Warp index of the LoadB task. Output of :func:`compute_warp_layout`."""

    load_sfa_warp_idx: int = 11
    """Warp index of the LoadSfA task. Output of :func:`compute_warp_layout`."""

    load_sfb_warp_idx: int = 9
    """Warp index of the LoadSfB task. Output of :func:`compute_warp_layout`."""

    mma_warp_idx: int = 13
    """Warp index of the MMA task. Output of :func:`compute_warp_layout`."""

    num_cast_a_warps: int = 0
    """CastA warp count (``4`` when :attr:`has_cast_a`, else ``0``).
    Output of :func:`compute_warp_layout`."""

    num_copy_sfa_warps: int = 1
    """CopySfA warp count. Output of :func:`compute_warp_layout`."""

    num_copy_sfb_warps: int = 4
    """CopySfB warp count (``4`` for the compact R8c4 STTM path).
    Output of :func:`compute_warp_layout`."""

    num_epilogue_warps: int = 4
    """Epilogue warp count (``4``, or ``8`` for some wide configs).
    Output of :func:`compute_warp_layout`."""

    num_load_a_warps: int = 1
    """LoadA warp count (``0`` when A is gathered).
    Output of :func:`compute_warp_layout`."""

    num_load_b_warps: int = 1
    """Requested LoadB warp count.

    .. note::
        Honored by :func:`compute_warp_layout` only when ``> 1``; otherwise the
        count is recomputed from the shape.
    """

    num_load_sfa_warps: int = 1
    """Requested LoadSfA warp count.

    .. note:: Honored by :func:`compute_warp_layout` only when ``> 1``.
    """

    num_load_sfb_warps: int = 1
    """Requested LoadSfB warp count.

    .. note:: Honored by :func:`compute_warp_layout` only when ``> 1``.
    """

    num_mma_warps: int = 1
    """MMA warp count (always ``1``). Output of :func:`compute_warp_layout`."""

    num_padding_warps: int = 2
    """Padding-warp count that rounds the CTA up to whole warpgroups.
    Output of :func:`compute_warp_layout`."""

    num_workid_warps: int = 1
    """CLC work-id warp count (``1`` when persistent, else ``0``).
    Output of :func:`compute_warp_layout`."""

    padding_warp_idx: int = 14
    """First warp of the padding group. Output of :func:`compute_warp_layout`."""

    workid_warp_idx: int = 14
    """Warp index of the CLC work-id task. Output of :func:`compute_warp_layout`."""

    # --- Register budgets (per-thread setmaxnreg) ---
    cast_a_regs: int = 160
    """Per-thread register budget for the CastA warp group."""

    copy_sf_regs: int = 48
    """Per-thread register budget for the CopySf tasks.

    The compact R8c4-STTM SFB path overrides this to ``72``
    (:attr:`copy_sfb_task_regs`).
    """

    epilogue_regs: int = 160
    """Per-thread register budget for the epilogue warp group."""

    load_a_regs: int = 0
    """LoadA register budget; ``0`` falls back to :attr:`load_regs`
    (see :attr:`load_a_task_regs`)."""

    load_b_regs: int = 0
    """LoadB register budget; ``0`` falls back to :attr:`load_regs`
    (see :attr:`load_b_task_regs`)."""

    load_regs: int = 48
    """Default register budget for the A/B TMA load tasks when no per-task
    override (:attr:`load_a_regs` / :attr:`load_b_regs`) is set."""

    load_sf_regs: int = 48
    """Default register budget for the scale-factor load tasks when no per-task
    override is set."""

    load_sfa_regs: int = 0
    """LoadSfA register budget; ``0`` falls back to :attr:`load_sf_regs`."""

    load_sfb_regs: int = 0
    """LoadSfB register budget; ``0`` falls back to :attr:`load_sf_regs`."""

    mma_regs: int = 48
    """Per-thread register budget for the MMA task."""

    padding_regs: int = 48
    """Per-thread register budget for the padding warps."""

    workid_regs: int = 48
    """Per-thread register budget for the CLC work-id task."""

    threads_per_cta: int = 512
    """Total threads per CTA.

    .. note::
        Output of :func:`compute_warp_layout` (``= total_warps * 32``); the field
        default is a placeholder.
    """

    # --- Feature flags (int 0/1 or enum-as-int; never bool/str) ---
    act_kind: int = int(ActKind.NONE)
    """Epilogue activation, as an :class:`ActKind` integer value.

    Allowed: ``NONE``, ``SWIGLU``, ``GEGLU``, ``RELU2``, ``SILU``, ``SITU``. Gated kinds
    (:attr:`has_gated_epilogue`) consume ``(gate, up)`` pairs and halve the
    paired output dimension. In ``BATCH_N`` / swap-A/B mode the paired dimension
    is output M (hidden rows); in ``BATCH_M`` / non-swap mode it is output N
    (columns). Both layouts have epilogue handling.
    """

    batch_mode: int = int(BatchMode.BATCH_N)
    """MoE expert-batching axis, as a :class:`BatchMode` integer value.

    ``BATCH_N`` batches routed activation/token tiles along N, selects the
    swapped-A/B path (A=weights, B=activations), and stores C in the M-major
    (column-major) layout. ``BATCH_M`` batches along M, selects the non-swapped
    path, and stores C row-major. Also selects the expert-coordinate lookup.
    """

    do_pdl_wait_for_num_non_exiting_ctas: int = 0
    """Emit a PDL wait on ``num_non_exiting_ctas`` before the first early-exit
    load; the TS schedule then assumes the wait has completed. ``0``/``1``.

    Requires :attr:`use_pdl` and :attr:`use_early_exit` (:func:`validate_config`).
    """

    per_token_sf_dtype: int = int(DType.FP32)
    """Element type of per-token FP8 scale factors, as a :class:`DType` value.

    One of ``FP32``/``FP16``/``BF16`` (:func:`validate_config`). Relevant only
    when :attr:`use_per_token_sf_b` is set.
    """

    route_act: int = int(RouteImpl.NONE)
    """MoE activation gather transport, as a :class:`RouteImpl` value.

    ``NONE`` (no gather), ``TMA`` (gather4), or ``LDGSTS`` (cp.async).
    ``LDG_PLUS_STS`` is rejected (:func:`validate_config`). See :attr:`has_gather`,
    :attr:`has_tma_route`.
    """

    route_sfs_act: int = int(RouteImpl.NONE)
    """Transport for routed activation scale factors, as a :class:`RouteImpl` value.

    ``NONE`` uses the default SF transport: padded/non-routed normally, but
    TMA-routed block-scaled activations implicitly use the LDGSTS routed-SF
    path. A non-``NONE`` value requires :attr:`route_act` != ``NONE``; ``TMA``
    requires ``sf_k >= 32`` for the routed operand. ``LDG_PLUS_STS`` is rejected.
    See :attr:`has_routed_sfs`.
    """

    sf_layout_a: int = int(SfLayout.R128c4)
    """SMEM layout of the A scale factors, as a :class:`SfLayout` value.

    Default ``R128c4``. ``R8c4`` is rejected for input SFA; CastA requires
    ``R128c4`` (:func:`validate_config`). The effective layout is
    :attr:`smem_sfa_layout` (routing may override it).
    """

    sf_layout_b: int = int(SfLayout.R8c4)
    """SMEM layout of the B scale factors, as a :class:`SfLayout` value.

    Default ``R8c4``, valid only for non-gather B with ``tile_n <= 64`` or routed
    low-N SFB (:func:`validate_config`). Effective layout is
    :attr:`smem_sfb_layout`.
    """

    sf_layout_c: int = int(SfLayout.R8c4)
    """SMEM layout of the output scale factors, as a :class:`SfLayout` value.

    Used only when the output is quantized (:attr:`has_epilogue_quant`), where it
    must be ``R8c4`` or ``R128c4`` (:func:`validate_config`).
    """

    tile_scheduler: int = int(TileScheduler.STATIC)
    """CTA scheduling mode, as a :class:`TileScheduler` value.

    ``STATIC`` (fixed hardware grid) or ``PERSISTENT`` (dynamic CLC work queue).
    See :attr:`is_persistent`.
    """

    transpose_mma_output: int = 1
    """Whether MMA output is transposed before the epilogue. ``0``/``1``.

    The generated ABI exposes this separately from :attr:`batch_mode`. The
    current kernel supports only ``1`` for the ``BATCH_N`` swapped-A/B path and
    ``0`` for the ``BATCH_M`` non-swapped path. FP8 per-token SFB requires ``1``.
    """

    use_clc_fast_drain: int = 0
    """Enable batched CLC cancellation for persistent early-exit overlaunch.

    When a CUDA graph launches more token CTAs than the active batch needs,
    this lets the work queue cancel queued tail CTAs in batches.
    """

    use_early_exit: int = 0
    """Let over-launched token CTAs skip work past the active batch. ``0``/``1``.

    Needs a runtime ``num_non_exiting_ctas`` value; pairs with :attr:`use_pdl` and
    :attr:`do_pdl_wait_for_num_non_exiting_ctas`.
    """

    use_full_tmem_dealloc_barrier: int = 0
    """Full-CTA (vs epilogue-warpgroup-only) barrier before TMEM dealloc. ``0``/``1``.

    ``1`` syncs all warps before ``tcgen05.dealloc``; ``0`` syncs only the
    epilogue warpgroup (lower overhead).
    """

    use_global_scales: int = 1
    """Enable per-tensor FP32 global scales in the epilogue. ``0``/``1``.

    Affects only NVFP4 and plain-FP8 paths (ignored for MX, which carries block
    scales). See :attr:`uses_global_scales` for the effective value.
    """

    fuse_sf_copy_to_mma: int = 0
    """Fuse the scale-factor SMEM-to-TMEM copy into the MMA task. ``0``/``1``.

    This matches TRT-LLM Gen's ``fuseUtccpWithUtcmma`` path and removes the
    otherwise separate CopySf tasks.  Unlike :attr:`use_max_tmem_overlap`, it
    does not require a tile-256 accumulator layout.
    """

    fuse_operand_sf_loads: int = 0
    """Fuse each operand load with its scale-factor load. ``0``/``1``.

    The clustered high-throughput LDGSTS path uses the same four warps for the
    routed activation and activation-SF copies, and the same warp for the
    weight and weight-SF TMA loads.  This matches the generated kernel's
    12-warp schedule and lets both LDGSTS streams share one delayed commit
    window.
    """

    use_max_tmem_overlap: int = 0
    """Enable the tile256 fused-SF / maximum-TMEM-overlap path. ``0``/``1``.

    Valid only for ``tile_n == 256`` block-scaled kernels with ``cluster_m == 2``,
    persistent scheduling, and ``num_stages_tmem_acc == 1``
    (:func:`validate_config`). See :attr:`use_tile256_tmem_overlap`.
    """

    use_pdl: int = 0
    """Enable programmatic dependent launch (PDL). ``0``/``1``.

    Overlaps this kernel's prologue with the prior kernel; pairs with
    :attr:`do_pdl_wait_for_num_non_exiting_ctas`.
    """

    # Per-token scale factors. SF-A is consumed by the epilogue for plain FP8;
    # SF-B is consumed by the B/output-channel epilogue path for plain FP8 and
    # NVFP4/E2M1.
    use_per_token_sf_a: int = 0
    """Plain-FP8 per-token scale factors on A. ``0``/``1``.

    In swap-A/B MoE configs this is the A-side channel row scale.
    """

    use_per_token_sf_b: int = 0
    """Per-token scale factors on B. ``0``/``1``.

    Valid for plain FP8 or NVFP4 MMA. FP8 with :attr:`bias_type` = ``M`` is
    parsed but not implemented. Dtype is
    :attr:`per_token_sf_dtype`.
    """

    has_gemm1_alpha: int = 0
    """Load a per-expert float32 SwiGLU-OAI alpha pointer in the epilogue."""

    has_gemm1_beta: int = 0
    """Load a per-expert float32 SwiGLU-OAI beta pointer in the epilogue."""

    has_gemm1_clamp_limit: int = 0
    """Load a per-expert float32 SwiGLU-OAI clamp-limit pointer in the epilogue."""

    # DeepSeek FP8: per-128-K-chunk FP32 dequant scales loaded into SMEM via
    # cp.async and applied as ffma2 in the epilogue, BEFORE the FP32 output cast.
    # Requires E4M3 MMA
    # with tile_k=128 and is mutually exclusive with MX/NVFP4 block scaling
    # and per-token SF.
    use_deepseek_fp8: int = 0
    load_sfab_warp_idx: int = 12
    num_load_sfab_warps: int = 0
    load_sfab_regs: int = 48
    use_tma_oob_opt: int = 1
    """Use TMA out-of-bounds predication for partial tiles. ``0``/``1``.

    Pads descriptor coordinates so the hardware predicates OOB accesses, cutting
    epilogue predicate math. Applies to A/B loads and C stores; see
    :attr:`use_tma_oob_opt_a` / :attr:`use_tma_oob_opt_b`.
    """

    use_tma_store: int = 0
    """Epilogue store kind: ``0`` = direct STG, ``1`` = SMEM-staged TMA store.

    TMA store improves locality for gated / quantized outputs; it needs the C SMEM
    scratch (:attr:`num_bytes_c_smem_scratch`).
    """

    use_two_tma_load_warps: int = 1
    """Reserved.

    .. note:: Currently unused by the TS implementation (no consumer).
    """

    use_unroll_loop_2x_for_mma: int = 0
    """Unroll the MMA K loop by two stages. ``0``/``1``.

    The captured loop keeps a scalar logical step and asks the DSL compiler to
    unroll it twice, so each K tile retains its own pipeline-stage identity.
    Odd K-tile counts automatically use the scalar loop.
    """

    use_work_throttle: int = 1
    """Throttle CLC work-id fetch in persistent kernels. ``0``/``1``.

    Effective only when :attr:`is_persistent` (see
    :attr:`use_work_throttle_barrier`); paces clustered loads to avoid SMEM
    pipeline ring overrun.
    """

    # --- Gather (LDGSTS route) ---
    gather_regs: int = 48
    """Per-thread register budget for the LDGSTS gather task."""

    gather_warp_idx: int = 6
    """Warp index of the gather task. Output of :func:`compute_warp_layout`."""

    num_gather_warps: int = 0
    """Gather warp count (``0`` = TMA-only; ``> 0`` = LDGSTS activation gather).
    Output of :func:`compute_warp_layout`."""

    num_sync_warps: int = 0
    """Cross-CTA proxy-barrier sync warp count (``1`` only when clustered and
    gathered). Output of :func:`compute_warp_layout`."""

    sync_regs: int = 48
    """Per-thread register budget for the sync warp."""

    sync_warp_idx: int = 10
    """Warp index of the sync task. Output of :func:`compute_warp_layout`."""

    # --- Host-derived schedule constants ---
    # Set by ``validate_config(..., problem_mnk=...)`` on the Python launch
    # path.  The JIT kernel receives ``cfg`` as a constexpr, so resources and
    # task factories can consume this as a static delayed-commit prefetch cap
    # instead of guarding head/tail entries against runtime K.
    ldgsts_sfb_producer_commit_prefetch_depth: int = MAX_PRODUCER_COMMIT_PREFETCH_DEPTH

    # --- Activation and epilogue knobs ---
    bias_type: int = int(BiasType.NONE)
    """Epilogue bias, as a :class:`BiasType` value.

    ``NONE`` or ``M`` (per-channel float32 ``[experts, M]`` bias added before
    activation). Plain FP8 per-token SF-B with ``M`` is parsed but not
    implemented; NVFP4 per-token SF-B with ``M`` is supported. See
    :attr:`has_bias_m`.
    """

    # --- Derived sizes (properties) ---
    def __post_init__(self) -> None:
        """Validate invariants fixed at construction (accumulator must be FP32)."""
        if self.dtype_acc != int(DType.FP32):
            raise ValueError(
                f"BatchedGemm accumulators must be FP32, got {self.dtype_acc}"
            )

    @property
    def dtype_a_kind(self) -> int:
        """Integer :class:`DType` value of operand A (mirror of :attr:`dtype_a`)."""
        return int(self.dtype_a)

    @property
    def dtype_b_kind(self) -> int:
        """Integer :class:`DType` value of operand B (mirror of :attr:`dtype_b`)."""
        return int(self.dtype_b)

    @property
    def dtype_c_kind(self) -> int:
        """Integer :class:`DType` value of the output (mirror of :attr:`dtype_c`)."""
        return int(self.dtype_c)

    @property
    def dtype_a_bits(self) -> int:
        """Logical bit width of operand A (4/8/16/32), via :func:`dtype_bits`."""
        return dtype_bits(self.dtype_a_kind)

    @property
    def dtype_b_bits(self) -> int:
        """Logical bit width of operand B (4/8/16/32)."""
        return dtype_bits(self.dtype_b_kind)

    @property
    def dtype_c_bits(self) -> int:
        """Logical bit width of the output (4/8/16/32)."""
        return dtype_bits(self.dtype_c_kind)

    @property
    def dtype_acc_bits(self) -> int:
        """Logical bit width of the accumulator (always ``32``)."""
        return dtype_bits(self.dtype_acc)

    @property
    def acc_dtype_bytes(self) -> int:
        """Accumulator byte width (always ``4``)."""
        return self.dtype_acc_bits // 8

    @property
    def is_mx_mma(self) -> bool:
        """True when both operands are MX types (``MXE2M1``/``MXE4M3``);
        block-scaled MX MMA with ``mma_k = 32``.
        """
        mx_dtypes = (int(DType.MXE2M1), int(DType.MXE4M3))
        return self.dtype_a_kind in mx_dtypes and self.dtype_b_kind in mx_dtypes

    @property
    def is_fp8_mma(self) -> bool:
        """True when both operands are ``E4M3`` -- plain FP8 MMA (``mma_k = 32``)."""
        return self.dtype_a_kind == int(DType.E4M3) and self.dtype_b_kind == int(
            DType.E4M3
        )

    @property
    def has_cast_a(self) -> bool:
        """True for the MXFP4-weight / BF16-activation MMA path.

        A is loaded as packed ``MXE2M1`` + UE8M0 scales, cast to BF16 in a four-warp
        CastA task, staged in TMEM, then fed to a BF16 MMA. Requires
        ``dtype_a=MXE2M1`` and ``dtype_b=BF16``; the C output may be BF16 or
        FP16 after the FP32 accumulator cast.
        """
        return self.dtype_a_kind == int(DType.MXE2M1) and self.dtype_b_kind == int(
            DType.BF16
        )

    @property
    def is_nvfp4_mma(self) -> bool:
        """True when both operands are ``E2M1`` -- NVFP4 MMA (``mma_k = 64``)."""
        return self.dtype_a_kind == int(DType.E2M1) and self.dtype_b_kind == int(
            DType.E2M1
        )

    @property
    def has_input_block_scaling(self) -> bool:
        """True when the inputs carry block scale factors
        (:attr:`is_mx_mma` or :attr:`is_nvfp4_mma`).
        """
        return self.is_mx_mma or self.is_nvfp4_mma

    @property
    def has_scale_factor_a(self) -> bool:
        """True when operand A has scale factors (block scaling or
        :attr:`has_cast_a`).
        """
        return self.has_input_block_scaling or self.has_cast_a

    @property
    def has_scale_factor_b(self) -> bool:
        """True when operand B has scale factors (block scaling)."""
        return self.has_input_block_scaling

    @property
    def uses_f16_mma(self) -> bool:
        """True when the MMA runs in 16-bit (BF16, or the CastA BF16 path)."""
        return self.is_bf16_mma or self.has_cast_a

    @property
    def uses_plain_mma(self) -> bool:
        """True for non-block-scaled MMA (BF16, CastA, or plain FP8)."""
        return self.uses_f16_mma or self.is_fp8_mma

    @property
    def uses_mx_scale_factors(self) -> bool:
        """True when any UE8M0 MX scale factors are present
        (MX input, CastA, or MXFP8 output).
        """
        return self.is_mx_mma or self.has_cast_a or self.uses_mxfp8_output_quant

    @property
    def uses_global_scales(self) -> bool:
        """True when per-tensor FP32 global scales participate in the epilogue.

        Applies to NVFP4 and plain-FP8 paths and requires :attr:`use_global_scales`
        != 0. MX inputs already carry UE8M0 block scales, so the flag is ignored there.
        """
        return (self.is_nvfp4_mma or self.is_fp8_mma) and self.use_global_scales != 0

    @property
    def uses_fp8_output(self) -> bool:
        """True for plain FP8 MMA writing an ``E4M3`` output
        (no output block scaling).
        """
        return self.is_fp8_mma and self.dtype_c_kind == int(DType.E4M3)

    @property
    def has_deepseek_fp8_c_scale(self) -> bool:
        """DeepSeek FP8 E4M3 output uses the C-scale pointer for FP32 dq scales."""
        return self.has_deepseek_fp8 and self.uses_fp8_output

    @property
    def has_per_token_sf_a(self) -> bool:
        """True when plain-FP8 per-token A scales are active."""
        return self.is_fp8_mma and self.use_per_token_sf_a != 0

    @property
    def has_per_token_sf_b(self) -> bool:
        """True when plain-FP8 or NVFP4 per-token B scales are active
        (:attr:`use_per_token_sf_b`).
        """
        return (self.is_fp8_mma or self.is_nvfp4_mma) and self.use_per_token_sf_b != 0

    @property
    def has_per_token_scaling(self) -> bool:
        """True when any per-token scale factors are active."""
        return self.has_per_token_sf_a or self.has_per_token_sf_b

    @property
    def has_swiglu_oai_params(self) -> bool:
        """True when any runtime per-expert SwiGLU-OAI parameter is active."""
        return self.act_kind == int(ActKind.SWIGLU) and self.has_gemm1_activation_params

    @property
    def has_gemm1_activation_params(self) -> bool:
        """True when any runtime per-expert gated-activation parameter is active."""
        return bool(
            self.has_gemm1_alpha or self.has_gemm1_beta or self.has_gemm1_clamp_limit
        )

    @property
    def has_situ_params(self) -> bool:
        """True when SiTU uses runtime per-expert activation parameters."""
        return self.act_kind == int(ActKind.SITU) and self.has_gemm1_activation_params

    @property
    def has_deepseek_fp8(self) -> bool:
        """Per-K-chunk FP32 dequant SF path (use_deepseek_fp8=1).

        See ``smem_deepseek_sf_resources.SmemDeepSeekSfAbResource``: a single
        cp.async producer feeds per-stage activation+weight float scales that
        are applied in the epilogue before the configured FP32 output cast.
        Requires E4M3 MMA and is mutually exclusive with MX/NVFP4 input block scaling
        and per-token SF.
        """
        return self.use_deepseek_fp8 != 0

    @property
    def has_deepseek_fp8_two_epilogue(self) -> bool:
        """DeepSeek FP8 FC2 path splits M across two epilogues."""
        return (
            self.has_deepseek_fp8
            and self.is_swap_ab
            and self.tile_m == 128
            and self.mma_m == 64
        )

    @property
    def deepseek_fp8_epi_subtile_cnt(self) -> int:
        """Number of output-N epilogue subtiles required by DeepSeek FP8."""
        if self.is_swap_ab:
            warpgroup_count = max(1, self.num_epilogue_warps // 4)
            if self.has_deepseek_fp8_two_epilogue:
                warpgroup_count = 1
            epi_cols_per_call = self.epi_tile_n * warpgroup_count
        else:
            epi_cols_per_call = self.epi_tile_n // 4
        return max(1, self.tile_n // max(1, epi_cols_per_call))

    @property
    def num_dsfp8_act_sfs_per_stage(self) -> int:
        """Number of DeepSeek FP8 activation scales staged per pipeline stage."""
        token_tile_dim = self.tile_n if self.is_swap_ab else self.tile_m
        return max(token_tile_dim, 32)

    @property
    def num_bytes_dsfp8_sfab_per_stage(self) -> int:
        """SMEM bytes per DeepSeek SF stage, rounded for cp.async alignment."""
        raw = (
            self.num_dsfp8_act_sfs_per_stage + 1  # num weight sfs per stage
        ) * 4
        return ((raw + 15) // 16) * 16

    @property
    def num_dsfp8_weight_sfs_per_stage(self) -> int:
        """Number of DeepSeek FP8 weight scales staged per pipeline stage."""
        return 1

    @property
    def has_output_block_scaling(self) -> bool:
        """True when the quantized output carries block scales
        (non-BF16 quantized C).
        """
        return self.has_epilogue_quant and self.dtype_c_kind != int(DType.BF16)

    @property
    def uses_fp4_output_quant(self) -> bool:
        """True when C uses packed E2M1 storage (NVFP4 or MXFP4)."""
        return self.has_epilogue_quant and self.dtype_c_kind in (
            int(DType.E2M1),
            int(DType.MXE2M1),
        )

    @property
    def uses_mxfp4_output_quant(self) -> bool:
        """True when the output is quantized to MXFP4 (``MXE2M1``)."""
        return self.has_epilogue_quant and self.dtype_c_kind == int(DType.MXE2M1)

    @property
    def uses_mxfp8_output_quant(self) -> bool:
        """True when the output is quantized to MXFP8 (``MXE4M3``)."""
        return self.has_epilogue_quant and self.dtype_c_kind == int(DType.MXE4M3)

    @property
    def uses_mx_output_quant(self) -> bool:
        """True when quantized C uses UE8M0 MX scale factors."""
        return self.uses_mxfp4_output_quant or self.uses_mxfp8_output_quant

    @property
    def input_sf_block_size_a(self) -> int:
        """Resolved A scale-factor block length in K elements
        (``32`` MX/CastA, ``16`` NVFP4); honors :attr:`sf_block_size_a`.
        """
        if self.sf_block_size_a > 0:
            return self.sf_block_size_a
        return 32 if (self.is_mx_mma or self.has_cast_a) else 16

    @property
    def input_sf_block_size_b(self) -> int:
        """Resolved B scale-factor block length in K elements
        (``32`` MX, ``16`` NVFP4); honors :attr:`sf_block_size_b`.
        """
        if self.sf_block_size_b > 0:
            return self.sf_block_size_b
        return 32 if self.is_mx_mma else 16

    @property
    def output_sf_block_size_c(self) -> int:
        """Resolved output scale-factor block length
        (``32`` MXFP4/MXFP8, ``16`` NVFP4); honors :attr:`sf_block_size_c`.
        """
        if self.sf_block_size_c > 0:
            return self.sf_block_size_c
        return 32 if self.uses_mx_output_quant else 16

    @property
    def sf_vec_size(self) -> int:
        """Scale-factor K-vector length (``32`` for MX/CastA, else ``16``)."""
        return 32 if (self.is_mx_mma or self.has_cast_a) else 16

    @property
    def has_fused_act(self) -> bool:
        """True when a fused activation is selected (:attr:`act_kind` != NONE)."""
        return self.act_kind != int(ActKind.NONE)

    @property
    def has_activation_epilogue(self) -> bool:
        """True when the epilogue runs an activation, routed or fused."""
        return self.has_routed_act or self.has_fused_act

    @property
    def has_activation_side_effects(self) -> bool:
        """True when the epilogue does activation-side work.

        Covers routed or fused activation, output block scaling, or plain-FP8 output
        (:attr:`has_routed_act`, :attr:`has_fused_act`,
        :attr:`has_output_block_scaling`, :attr:`uses_fp8_output`).
        """
        return (
            self.has_routed_act
            or self.has_fused_act
            or self.has_output_block_scaling
            or self.uses_fp8_output
        )

    @property
    def does_scale_ab(self) -> bool:
        """True when the A/B operands are scaled (block scaling or CastA)."""
        return self.has_input_block_scaling or self.has_cast_a

    @property
    def does_scale_c(self) -> bool:
        """True when the output is quantized and needs scale factors
        (``E2M1``, ``MXE2M1``, or ``MXE4M3``).
        """
        return self.dtype_c_kind in (
            int(DType.E2M1),
            int(DType.MXE2M1),
            int(DType.MXE4M3),
        )

    @property
    def num_stages_cast_a(self) -> int:
        """CastA TMEM pipeline ring depth; mirrors :attr:`num_stages_a`."""
        return self.num_stages_a

    def _smem_bits_for_dtype(self, dtype_kind: int, dtype_bits: int) -> int:
        """Return the SMEM staging bit width for ``dtype_kind``
        (MX/FP8 stage at 8-bit).
        """
        if dtype_kind in (int(DType.MXE2M1), int(DType.MXE4M3), int(DType.E4M3)):
            return 8
        return dtype_bits

    def _tma_bits_for_dtype(self, dtype_kind: int, dtype_bits: int) -> int:
        """Return the TMA transfer bit width for ``dtype_kind``
        (``MXE2M1`` packs to 4-bit).
        """
        if dtype_kind == int(DType.MXE2M1):
            return 4
        if dtype_kind in (int(DType.MXE4M3), int(DType.E4M3)):
            return 8
        return dtype_bits

    @property
    def dtype_a_smem_bits(self) -> int:
        """SMEM staging width of A in bits (``4`` for CastA-packed MXFP4,
        ``8`` for MX/FP8, else the logical width).
        """
        if self.has_cast_a:
            return 4
        return self._smem_bits_for_dtype(self.dtype_a_kind, self.dtype_a_bits)

    @property
    def dtype_b_smem_bits(self) -> int:
        """SMEM staging width of B in bits (``8`` for MX/FP8, else logical)."""
        return self._smem_bits_for_dtype(self.dtype_b_kind, self.dtype_b_bits)

    @property
    def dtype_a_tma_bits(self) -> int:
        """TMA transfer width of A in bits (``4`` for ``MXE2M1``,
        ``8`` for MX/FP8, else logical).
        """
        return self._tma_bits_for_dtype(self.dtype_a_kind, self.dtype_a_bits)

    @property
    def dtype_b_tma_bits(self) -> int:
        """TMA transfer width of B in bits."""
        return self._tma_bits_for_dtype(self.dtype_b_kind, self.dtype_b_bits)

    @property
    def weight_layout_kind(self) -> int:
        """Integer :class:`~tensorrt_llm._torch.moe.flashinfer.tllm_enums.WeightLayout` value."""
        return int(self.weight_layout)

    @property
    def uses_block_major_k_weight(self) -> bool:
        """True when the physical weight operand uses BlockMajorK storage."""
        return self.weight_layout_kind == int(WeightLayout.BlockMajorK)

    @property
    def uses_block_major_k_weight_a(self) -> bool:
        """True when operand A is the BlockMajorK weight tensor."""
        return self.uses_block_major_k_weight and self.is_swap_ab

    @property
    def uses_block_major_k_weight_b(self) -> bool:
        """True when operand B is the BlockMajorK weight tensor."""
        return self.uses_block_major_k_weight and not self.is_swap_ab

    @property
    def weight_dtype_tma_bits(self) -> int:
        """TMA transfer bit width of the physical weight operand."""
        return self.dtype_a_tma_bits if self.is_swap_ab else self.dtype_b_tma_bits

    @property
    def weight_dtype_smem_bits(self) -> int:
        """SMEM staging bit width of the physical weight operand."""
        return self.dtype_a_smem_bits if self.is_swap_ab else self.dtype_b_smem_bits

    @property
    def weight_dtype_kind(self) -> int:
        """Logical dtype kind of the physical weight operand."""
        return self.dtype_a_kind if self.is_swap_ab else self.dtype_b_kind

    @property
    def block_major_k_pad_multiplier(self) -> int:
        """TRT-LLM Gen BlockMajorK padding multiplier for the weight operand."""
        tma_bits = self.weight_dtype_tma_bits
        smem_bits = self.weight_dtype_smem_bits
        if smem_bits % tma_bits != 0:
            raise ValueError(
                "BlockMajorK weight SMEM bit width must be a multiple of TMA "
                f"bit width, got smem_bits={smem_bits}, tma_bits={tma_bits}"
            )
        return max(1, smem_bits // tma_bits)

    @property
    def block_major_k_elems_in_128b(self) -> int:
        """Logical K elements in one unpadded 128B BlockMajorK slice."""
        return 128 * 8 // self.weight_dtype_tma_bits

    @property
    def block_major_k_smem_slices_per_tile(self) -> int:
        """Number of padded 128B SMEM slices covered by one K tile."""
        return (
            self.block_major_k_pad_multiplier * self.tile_k
        ) // self.block_major_k_elems_in_128b

    @property
    def block_major_k_elems(self) -> int:
        """Logical K elements covered by one BlockMajorK K block."""
        elems_in_128b = self.block_major_k_elems_in_128b
        if self.block_major_k_smem_slices_per_tile > 2:
            return elems_in_128b // self.block_major_k_pad_multiplier
        return elems_in_128b

    @property
    def block_major_k_bytes(self) -> int:
        """TRT-LLM Gen BlockMajorK K-block bytes for the weight TensorMap."""
        return self.block_major_k_elems * self.weight_dtype_tma_bits // 8

    @property
    def block_major_k_tile_block_elems(self) -> int:
        """Fastest TMA dimension used for one BlockMajorK tile load."""
        return min(self.block_major_k_elems, self.tile_k)

    @property
    def block_major_k_tile_slices(self) -> int:
        """Number of BlockMajorK K-block slices covered by one tile load."""
        return self.tile_k // self.block_major_k_tile_block_elems

    @property
    def mx_a_format(self) -> int:
        """MX MMA format code for A (:data:`MX_MMA_FORMAT_E4M3` /
        :data:`MX_MMA_FORMAT_E2M1`); raises if A is not an MX dtype.
        """
        if self.dtype_a_kind == int(DType.MXE4M3):
            return MX_MMA_FORMAT_E4M3
        if self.dtype_a_kind == int(DType.MXE2M1):
            return MX_MMA_FORMAT_E2M1
        raise ValueError(f"dtype_a is not an MX MMA dtype: {self.dtype_a_kind}")

    @property
    def mx_b_format(self) -> int:
        """MX MMA format code for B; raises if B is not an MX dtype."""
        if self.dtype_b_kind == int(DType.MXE4M3):
            return MX_MMA_FORMAT_E4M3
        if self.dtype_b_kind == int(DType.MXE2M1):
            return MX_MMA_FORMAT_E2M1
        raise ValueError(f"dtype_b is not an MX MMA dtype: {self.dtype_b_kind}")

    @property
    def tmem_sfa_cols(self) -> int:
        """TMEM columns per stage for the A scale factors
        (``0`` without block scaling).
        """
        return self.tmem_sf_cols("a")

    @property
    def tmem_sfb_cols(self) -> int:
        """TMEM columns per stage for the B scale factors
        (``0`` without block scaling).
        """
        return self.tmem_sf_cols("b")

    def _tmem_sf_tile_mn(self, operand: str) -> int:
        """Return the M/N tile extent for a SF ``operand`` (``tile_m`` for "a", ``tile_n`` for "b")."""
        if operand == "a":
            return self.tile_m
        if operand == "b":
            return self.tile_n
        raise ValueError(f"Unsupported scale-factor operand: {operand}")

    def _tmem_sf_vec_size(self, operand: str) -> int:
        """Return the SF block length for ``operand`` (:attr:`input_sf_block_size_a` / ``_b``)."""
        if operand == "a":
            return self.input_sf_block_size_a
        if operand == "b":
            return self.input_sf_block_size_b
        raise ValueError(f"Unsupported scale-factor operand: {operand}")

    def tmem_sf_cols(self, operand: str) -> int:
        """Return TMEM columns for one stage of scale-factor ``operand``
        (``"a"``/``"b"``); ``0`` without block scaling.
        """
        if not self.has_input_block_scaling:
            return 0
        return tmem_sf_cols(
            self._tmem_sf_tile_mn(operand),
            self.tile_k,
            self._tmem_sf_vec_size(operand),
        )

    def tmem_sf_col_stride(self, operand: str) -> int:
        """Return the TMEM column stride for one four-vector SF group
        of ``operand`` (``"a"``/``"b"``).
        """
        return tmem_sf_col_stride(self._tmem_sf_tile_mn(operand))

    @property
    def tmem_cast_a_cols(self) -> int:
        """TMEM columns per CastA stage (``128`` when :attr:`has_cast_a`,
        else ``0``).
        """
        return 128 if self.has_cast_a else 0

    @property
    def tmem_c_cols_per_stage(self) -> int:
        """TMEM columns per accumulator stage (= :attr:`tile_n`)."""
        return tmem_d_cols_per_stage(self.tile_n)

    @property
    def tmem_total_cols(self) -> int:
        """Alias of :attr:`tmem_required_cols` (pre-rounding TMEM footprint)."""
        return self.tmem_required_cols

    @property
    def num_bytes_a_per_stage(self) -> int:
        """SMEM bytes for one A tile stage
        (``tile_m * tile_k * dtype_a_smem_bits / 8``).
        """
        return self.tile_m * self.tile_k * self.dtype_a_smem_bits // 8

    @property
    def num_bytes_b_per_stage(self) -> int:
        """SMEM bytes for one full (cluster-wide) B tile stage."""
        return self.tile_n * self.tile_k * self.dtype_b_smem_bits // 8

    @property
    def num_bytes_a_tma_per_stage(self) -> int:
        """GMEM bytes moved by one A TMA load
        (uses the packed :attr:`dtype_a_tma_bits`).
        """
        return self.tile_m * self.tile_k * self.dtype_a_tma_bits // 8

    @property
    def num_bytes_b_tma_per_stage(self) -> int:
        """GMEM bytes moved by one B TMA load, per CTA (divided by
        ``cluster_m`` when :attr:`split_b_across_ctas`).
        """
        bytes_per_stage = self.tile_n * self.tile_k * self.dtype_b_tma_bits // 8
        if self.split_b_across_ctas:
            return bytes_per_stage // self.cluster_m
        return bytes_per_stage

    @property
    def num_bytes_b_smem_per_stage(self) -> int:
        """Per-CTA SMEM bytes for one B tile stage.

        When :attr:`split_b_across_ctas`, each CTA stages only its ``1 / cluster_m``
        slice (the TMA transaction stays cluster-total); otherwise equals
        :attr:`num_bytes_b_per_stage`.
        """
        if self.split_b_across_ctas:
            return self.num_bytes_b_per_stage // self.cluster_m
        return self.num_bytes_b_per_stage

    @property
    def num_bytes_sfa_per_stage(self) -> int:
        """SMEM bytes for one A scale-factor stage."""
        return self.tile_m * (self.tile_k // self.input_sf_block_size_a)

    @property
    def num_bytes_sfb_per_stage(self) -> int:
        """SMEM bytes for one B scale-factor stage.

        The physical ``R128c4`` layout stores complete 128-row groups.  Its
        TMA descriptor transfers those padded groups, so both the allocation
        and the pipeline transaction byte count must use the same rounded
        extent for non-power-of-two token tiles such as 96 and 192.
        """
        if self.smem_sfb_layout == int(SfLayout.R128c4):
            mn = ((self.tile_n + 127) // 128) * 128
        else:
            mn = max(16, self.tile_n) if self.is_mx_mma else self.tile_n
        return mn * (self.tile_k // self.input_sf_block_size_b)

    @property
    def num_bytes_sfb_tma_per_stage(self) -> int:
        """GMEM bytes moved by one SFB TMA load.

        Smaller than the SMEM allocation for the compact 8x4 SFB layout
        (:attr:`uses_sfb_8x4_load`); otherwise equals :attr:`num_bytes_sfb_per_stage`.
        """
        if self.uses_sfb_8x4_load:
            k_groups = (
                self.tile_k // self.input_sf_block_size_b // TMEM_SF_PACK_SIZE_BYTES
            )
            outer_groups = (self.tile_n + 7) // 8
            return 32 * k_groups * outer_groups
        return self.num_bytes_sfb_per_stage

    @property
    def num_bytes_c_per_stage(self) -> int:
        """SMEM bytes for one epilogue C tile
        (``epi_tile_m * epi_tile_n * dtype_c_bits / 8``).
        """
        return self.epi_tile_m * self.epi_tile_n * self.dtype_c_bits // 8

    @property
    def num_bytes_c_tma_store_per_group(self) -> int:
        """Bytes staged by one TMA-store epilogue warpgroup (halved for a gated epilogue)."""
        stage_bytes = self.num_bytes_c_per_stage
        if self.has_gated_epilogue:
            stage_bytes //= 2
        return stage_bytes

    @property
    def uses_bf16_gated_tma_c_scratch(self) -> bool:
        """True when a 16-bit gated epilogue stages only gated output rows in SMEM.

        Holds for BF16 MMA with BF16/FP16 output, :attr:`use_tma_store`, a gated
        epilogue, and no output quantization; shrinks the C scratch to half-N per
        warpgroup.
        """
        return (
            self.is_bf16_mma
            and bool(self.use_tma_store)
            and self.has_gated_epilogue
            and not self.has_epilogue_quant
            and not self.uses_fp8_output
        )

    @property
    def uses_deepseek_fp8_split_epilogue_tma_c_scratch(self) -> bool:
        """Whether DeepSeek FP8 stages one split-M BF16/FP16 output tile.

        This path reserves one TMA scratch tile per epilogue warpgroup. It
        requires num_stages_c_smem=1 when this property is true.
        """
        return (
            self.has_deepseek_fp8_two_epilogue
            and bool(self.use_tma_store)
            and self.dtype_c_kind in (int(DType.BF16), int(DType.FP16))
        )

    @property
    def num_bytes_c_smem_scratch(self) -> int:
        """Total SMEM bytes reserved for the C epilogue staging buffer.

        Sized from :attr:`num_bytes_c_per_stage` x :attr:`num_stages_c_smem`, or the
        reduced per-warpgroup form when :attr:`uses_bf16_gated_tma_c_scratch` or
        :attr:`uses_deepseek_fp8_split_epilogue_tma_c_scratch`.
        """
        if self.uses_deepseek_fp8_split_epilogue_tma_c_scratch:
            return (
                (self.tile_m // 2)
                * self.epi_tile_n
                * self.dtype_c_bits
                // 8
                * max(1, self.num_epilogue_warps // 4)
            )
        if self.uses_bf16_gated_tma_c_scratch:
            return (
                self.num_bytes_c_tma_store_per_group
                * self.num_stages_c_smem
                * max(1, self.num_epilogue_warps // 4)
            )
        return self.num_bytes_c_per_stage * self.num_stages_c_smem

    @property
    def aliases_c_scratch_with_ab(self) -> bool:
        """True when the C epilogue scratch may reuse the A/B SMEM buffers.

        ``False`` when :attr:`uses_bf16_gated_tma_c_scratch`, when persistent scheduling
        overlaps the next tile's loads (:attr:`is_persistent` and
        ``num_stages_workid > 1``), or for :attr:`has_cast_a`.
        """
        if self.uses_bf16_gated_tma_c_scratch:
            return False
        if self.is_persistent and self.num_stages_workid > 1:
            return False
        return not self.has_cast_a

    @property
    def tmem_alloc_cols(self) -> int:
        """Hardware TMEM allocation: :attr:`tmem_required_cols` rounded up
        to a power of two (minimum ``32``).
        """
        return max(32, 1 << (self.tmem_total_cols - 1).bit_length())

    @property
    def tmem_required_cols(self) -> int:
        """TMEM column footprint before hardware power-of-two rounding.

        Sums accumulator, CastA, and scale-factor staging across stages; must stay
        <= 512 (:func:`validate_config`). See :attr:`tmem_alloc_cols` for the rounded
        allocation.
        """
        total = self.tmem_c_cols_per_stage * self.num_stages_tmem_acc
        if self.has_cast_a:
            total += self.tmem_cast_a_cols * self.num_stages_cast_a
        if not self.has_scale_factors:
            return total

        if self.use_combined_sfab_copy:
            total += (
                self.tmem_sfa_cols * self.num_stages_tmem_sfa
                + self.tmem_sfb_cols * self.num_stages_tmem_sfb
            )
        elif self.uses_unfused_tmem_sf_copy:
            total += self.tmem_sfa_cols * self.num_stages_tmem_sfa
            total += self.tmem_sfb_cols * self.num_stages_tmem_sfb
        elif self.use_tile256_tmem_overlap:
            total = max(total, 512)
        else:
            total += self.tmem_sfa_cols + self.tmem_sfb_cols
        return total

    @property
    def has_scale_factors(self) -> bool:
        """True when the A/B MMA operands use block scaling (= :attr:`has_input_block_scaling`)."""
        return self.has_input_block_scaling

    @property
    def has_copy_sf(self) -> bool:
        """True when scale factors need a SMEM->TMEM copy
        (= :attr:`has_scale_factors`).
        """
        return self.has_scale_factors

    @property
    def uses_unfused_tmem_sf_copy(self) -> bool:
        """True when SF use a separate CopySf task (block-scaled and
        neither explicitly fused into MMA nor covered by the tile-256
        :attr:`use_max_tmem_overlap` path).
        """
        return (
            self.has_scale_factors
            and self.fuse_sf_copy_to_mma == 0
            and self.use_max_tmem_overlap == 0
        )

    @property
    def smem_sfa_layout(self) -> int:
        """Effective SMEM layout of the A scale factors, as a :class:`SfLayout` value.

        Overrides :attr:`sf_layout_a` for routed SFA: ``LINEAR`` for TMA-routed,
        ``R128c4`` for LDGSTS-routed; otherwise the configured layout.
        """
        if self.has_routed_sfs and not self.is_swap_ab:
            if self.uses_tma_routed_sfs:
                return int(SfLayout.LINEAR)
            return int(SfLayout.R128c4)
        return self.sf_layout_a

    @property
    def smem_sfb_layout(self) -> int:
        """Effective SMEM layout of the B scale factors, as a :class:`SfLayout` value.

        Overrides :attr:`sf_layout_b` for routed :attr:`is_swap_ab` SFB
        (``LINEAR`` for TMA-routed; ``R8c4`` for ``tile_n < 128`` else
        ``R128c4`` for LDGSTS-routed) and forces ``R8c4`` when
        :attr:`uses_sfb_8x4_load`.
        """
        if self.has_routed_sfs and self.is_swap_ab:
            if self.uses_tma_routed_sfs:
                return int(SfLayout.LINEAR)
            if self.tile_n < 128:
                return int(SfLayout.R8c4)
            return int(SfLayout.R128c4)
        if self.uses_sfb_8x4_load:
            return int(SfLayout.R8c4)
        return self.sf_layout_b

    def _sf_smem_to_tmem_copy(self, smem_layout: int) -> int:
        """Map an SMEM SF layout to its :class:`SfSmemToTmemCopy` strategy."""
        if not self.has_scale_factors:
            return int(SfSmemToTmemCopy.NONE)
        if not self.uses_unfused_tmem_sf_copy:
            return int(SfSmemToTmemCopy.FUSED_UTCCP)
        if smem_layout == int(SfLayout.R128c4):
            return int(SfSmemToTmemCopy.UTCCP)
        return int(SfSmemToTmemCopy.LDS_STTM)

    @property
    def sfa_smem_to_tmem_copy(self) -> int:
        """:class:`SfSmemToTmemCopy` strategy for A SF, from
        :attr:`smem_sfa_layout`.
        """
        return self._sf_smem_to_tmem_copy(self.smem_sfa_layout)

    @property
    def sfb_smem_to_tmem_copy(self) -> int:
        """:class:`SfSmemToTmemCopy` strategy for B SF, from
        :attr:`smem_sfb_layout`.
        """
        if (
            self.uses_unfused_tmem_sf_copy
            and self.has_routed_sfs
            and self.is_swap_ab
            and self.uses_ldgsts_routed_sfs
            and self.tile_n % 128 != 0
        ):
            # R128c4 staging pads N=192 to two 128-row blocks. UTCCP copies
            # that padded representation verbatim, but the N=192 MMA consumes
            # six contiguous scale columns. Use LDS+STTM to compact 8 -> 6.
            return int(SfSmemToTmemCopy.LDS_STTM)
        return self._sf_smem_to_tmem_copy(self.smem_sfb_layout)

    @property
    def use_combined_sfab_copy(self) -> bool:
        """True for the single-pipeline combined SFA+SFB SMEM->TMEM copy.

        A high-throughput :attr:`is_swap_ab` path (unfused copy, persistent, clustered,
        ``tile_n >= 128``) that issues one CopySfAb warp instead of two TmemSf pipelines.
        """
        if not (
            self.uses_unfused_tmem_sf_copy
            and self.is_swap_ab
            and self.is_persistent
            and self.has_cluster
            and self.sfb_smem_to_tmem_copy != int(SfSmemToTmemCopy.LDS_STTM)
            and self.tile_n >= 128
        ):
            return False

        if not self.has_activation_side_effects and not self.has_routed_sfs:
            return True

        return self.has_routed_sfs and self.uses_ldgsts_routed_sfs

    @property
    def split_b_across_ctas(self) -> bool:
        """True when a clustered kernel stages only a B slice per CTA.

        Holds for clustered plain-MMA, tile256-overlap, non-routed block-scaled,
        or swap-AB LDGSTS gather kernels.  The LDGSTS gather resource assigns
        ``tile_n / cluster_m`` routed rows to each CTA, so its per-stage SMEM
        allocation and K-group descriptor stride must use the same split.
        See :attr:`num_bytes_b_smem_per_stage`.
        """
        return self.has_cluster and (
            self.uses_plain_mma
            or self.use_tile256_tmem_overlap
            or (self.is_swap_ab and self.has_gather)
            or (self.has_scale_factors and not self.has_routed_act)
            or (self.has_scale_factors and self.is_swap_ab and not self.has_routed_act)
        )

    @property
    def use_tile256_tmem_overlap(self) -> bool:
        """True for the ``tile_n == 256`` fused-SF / max-TMEM-overlap path.

        Enabled by :attr:`use_max_tmem_overlap` on block-scaled ``tile_n == 256``
        kernels; see :func:`validate_config` for the full constraint set.
        """
        return (
            self.has_scale_factors
            and self.use_max_tmem_overlap != 0
            and self.tile_n == 256
        )

    @property
    def has_cluster(self) -> bool:
        """Whether the kernel runs as a multi-CTA cluster (``cluster_m > 1``).

        Gates cluster-scoped SMEM allocation, mbarrier multicast, the cross-CTA
        sync warp, and B-tile splitting. See :attr:`cluster_m`, :attr:`cta_group`.
        """
        return self.cluster_m > 1

    @property
    def is_persistent(self) -> bool:
        """True for dynamic persistent CLC scheduling (:attr:`tile_scheduler` == PERSISTENT)."""
        return self.tile_scheduler == int(TileScheduler.PERSISTENT)

    @property
    def use_work_throttle_barrier(self) -> bool:
        """True when the persistent work-throttle barrier is active.

        Equals :attr:`is_persistent` and :attr:`use_work_throttle`; paces CLC work-id
        fetch so loads cannot outrun the mainloop.
        """
        return self.is_persistent and bool(self.use_work_throttle)

    @property
    def num_work_throttle_producer_warps(self) -> int:
        """Number of load-task warps that signal the work-throttle barrier (``0`` when inactive)."""
        if not self.use_work_throttle_barrier:
            return 0
        if self.has_gather and not self.is_swap_ab:
            return self.num_load_b_warps
        return self.num_load_a_warps

    @property
    def load_a_task_regs(self) -> int:
        """Effective LoadA register budget (:attr:`load_a_regs`, else the
        MMA budget for tile256 overlap, else :attr:`load_regs`).
        """
        if self.load_a_regs > 0:
            return self.load_a_regs
        if self.use_tile256_tmem_overlap and self.has_activation_side_effects:
            return self.mma_regs
        return self.load_regs

    @property
    def load_b_task_regs(self) -> int:
        """Effective LoadB register budget (:attr:`load_b_regs` or
        :attr:`load_regs`).
        """
        if self.load_b_regs > 0:
            return self.load_b_regs
        return self.load_regs

    @property
    def load_sfa_task_regs(self) -> int:
        """Effective LoadSfA register budget (:attr:`load_sfa_regs`, else the
        MMA budget for tile256 overlap, else :attr:`load_sf_regs`).
        """
        if self.load_sfa_regs > 0:
            return self.load_sfa_regs
        if self.use_tile256_tmem_overlap and self.has_activation_side_effects:
            return self.mma_regs
        return self.load_sf_regs

    @property
    def load_sfb_task_regs(self) -> int:
        """Effective LoadSfB register budget (:attr:`load_sfb_regs` or
        :attr:`load_sf_regs`).
        """
        if self.load_sfb_regs > 0:
            return self.load_sfb_regs
        return self.load_sf_regs

    @property
    def copy_sfa_task_regs(self) -> int:
        """Effective CopySfA register budget (= :attr:`copy_sf_regs`)."""
        return self.copy_sf_regs

    @property
    def copy_sfb_task_regs(self) -> int:
        """Effective CopySfB register budget (``72`` for the R8c4-STTM path,
        else :attr:`copy_sf_regs`).
        """
        if self.sfb_smem_to_tmem_copy == int(
            SfSmemToTmemCopy.LDS_STTM
        ) and self.smem_sfb_layout == int(SfLayout.R8c4):
            return 72
        return self.copy_sf_regs

    @property
    def is_swap_ab(self) -> bool:
        """True when A=weights and B=activations."""
        return self.batch_mode == int(BatchMode.BATCH_N)

    @property
    def has_gated_epilogue(self) -> bool:
        """True for gated activations (``SWIGLU``/``GEGLU``/``SILU``/``SITU``);
        the paired output dimension is halved.
        """
        return self.act_kind in (
            int(ActKind.SWIGLU),
            int(ActKind.GEGLU),
            int(ActKind.SILU),
            int(ActKind.SITU),
        )

    @property
    def has_epilogue_quant(self) -> bool:
        """True when the output is quantized (= :attr:`does_scale_c`)."""
        return self.does_scale_c

    @property
    def cta_group(self):
        """tcgen05 CTA-group kind for MMA/TMEM ops.

        Returns ``CTA_2`` for a 2-CTA cluster (``cluster_m >= 2``), else
        ``CTA_1``. Passed to mbarrier descriptor construction; it must agree with
        :attr:`cluster_m` or the barrier lowering fails. See :attr:`has_cluster`.
        """

        return prims.CTAGroup.CTA_2 if self.cluster_m >= 2 else prims.CTAGroup.CTA_1

    @property
    def is_batch_m(self) -> bool:
        """True when experts are batched along M (:attr:`batch_mode` == ``BATCH_M``).

        BatchM is the non-swapped-A/B layout and uses ``tile_coord_m`` for expert
        lookup; BatchN/swapped-A/B uses ``tile_coord_n``.
        """
        return self.batch_mode == int(BatchMode.BATCH_M)

    @property
    def has_gather(self) -> bool:
        """True when activations are gathered via LDGSTS (:attr:`route_act` == ``LDGSTS``)."""
        return self.route_act == int(RouteImpl.LDGSTS)

    @property
    def has_tma_route(self) -> bool:
        """True when activations are gathered via TMA gather4 (:attr:`route_act` == ``TMA``)."""
        return self.route_act == int(RouteImpl.TMA)

    @property
    def has_routed_act(self) -> bool:
        """True when activations are routed by any transport (:attr:`route_act` != ``NONE``)."""
        return self.route_act != int(RouteImpl.NONE)

    @property
    def has_implicit_routed_sfs(self) -> bool:
        """True when activation scale factors are implicitly routed.

        Low-latency TMA-route kernels route the activation SF through a compact cp.async
        path even with :attr:`route_sfs_act` == ``NONE``; kept separate so
        ``route_sfs_act = TMA`` can still pick the TMA gather4 SF loader.
        """
        return (
            self.has_scale_factors
            and self.has_tma_route
            and self.route_sfs_act == int(RouteImpl.NONE)
        )

    @property
    def use_tma_oob_opt_a(self) -> bool:
        """True when TMA out-of-bounds predication applies to operand A.

        Requires :attr:`use_tma_oob_opt`, no activation routing, and
        :attr:`is_swap_ab` false.
        """
        return (
            self.use_tma_oob_opt == 1
            and not self.has_routed_act
            and not self.is_swap_ab
        )

    @property
    def use_tma_oob_opt_b(self) -> bool:
        """True when TMA out-of-bounds predication applies to operand B.

        Requires :attr:`use_tma_oob_opt`, no activation routing, :attr:`is_swap_ab`,
        and not BF16 MMA.
        """
        return (
            self.use_tma_oob_opt == 1
            and not self.has_routed_act
            and self.is_swap_ab
            and not self.is_bf16_mma
        )

    @property
    def has_routed_sfs(self) -> bool:
        """True when activation scale factors are gathered (not padded).

        Explicit (:attr:`route_sfs_act` != ``NONE``) or implicit
        (:attr:`has_implicit_routed_sfs`); always ``False`` for :attr:`has_cast_a`.
        """
        if self.has_cast_a:
            return False
        return self.route_sfs_act != int(RouteImpl.NONE) or self.has_implicit_routed_sfs

    @property
    def uses_ldgsts_routed_sfs(self) -> bool:
        """True when routed activation SF are loaded by cp.async / LDGSTS.

        Covers implicit routing and explicit :attr:`route_sfs_act` == ``LDGSTS``.
        """
        return self.has_implicit_routed_sfs or self.route_sfs_act == int(
            RouteImpl.LDGSTS
        )

    @property
    def uses_tma_routed_sfs(self) -> bool:
        """True when routed activation SF use TMA gather4 (:attr:`route_sfs_act` == ``TMA``).

        Requires ``sf_k >= 32`` for the routed operand; see
        :func:`validate_config`.
        """
        return self.route_sfs_act == int(RouteImpl.TMA)

    @property
    def uses_sfb_8x4_load(self) -> bool:
        """True when the non-gather B scale-factor tensor uses the compact ``R8c4`` layout.

        Requires ``sf_layout_b == R8c4``, an 8-row aligned low-N tile or
        ``tile_n == 192``, no routed
        :attr:`is_swap_ab` SFB, and no tile256 overlap. Forces
        :attr:`smem_sfb_layout` to ``R8c4``.
        """
        return (
            self.sf_layout_b == int(SfLayout.R8c4)
            and (self.tile_n <= 64 or self.tile_n == 192)
            and self.tile_n % 8 == 0
            and not (self.has_routed_sfs and self.is_swap_ab)
            and not self.use_tile256_tmem_overlap
        )

    @property
    def is_bf16_mma(self) -> bool:
        """True when both operands are BF16 -- standard ``tcgen05`` MMA, not block-scaled."""
        return self.dtype_a_kind == int(DType.BF16) and self.dtype_b_kind == int(
            DType.BF16
        )

    @property
    def use_bf16_kbox_tma_a(self) -> bool:
        """True when operand A uses the BF16 K-box TMA descriptor shape.

        For BF16 A, except TMA-routed :attr:`is_swap_ab` false,
        :attr:`use_tma_oob_opt_a`, or
        :attr:`has_cast_a`.
        """
        return (
            self.dtype_a_kind == int(DType.BF16)
            and not (self.has_tma_route and not self.is_swap_ab)
            and not self.use_tma_oob_opt_a
            and not self.has_cast_a
        )

    @property
    def has_bias_m(self) -> bool:
        """True when a per-channel bias is added (:attr:`bias_type` == ``M``)."""
        return self.bias_type == int(BiasType.M)


def uses_routed_sfa_tma_desc(cfg: BatchedGemmConfig) -> bool:
    """Return whether routed SFA needs a TMA gather4 TensorMap descriptor."""
    return cfg.has_routed_sfs and cfg.uses_tma_routed_sfs and not cfg.is_swap_ab


def uses_routed_sfb_tma_desc(cfg: BatchedGemmConfig) -> bool:
    """Return whether routed SFB needs a TMA gather4 TensorMap descriptor."""
    return cfg.has_routed_sfs and cfg.uses_tma_routed_sfs and cfg.is_swap_ab


def _validate_binary_config_option(name: str, value: int) -> None:
    if value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1, got {value}")


def _validate_route_config_option(name: str, value: int) -> None:
    if value in (
        int(RouteImpl.NONE),
        int(RouteImpl.TMA),
        int(RouteImpl.LDGSTS),
    ):
        return
    if value == int(RouteImpl.LDG_PLUS_STS):
        raise ValueError(f"{name}=LDG_PLUS_STS is not implemented by this kernel")
    raise ValueError(f"Unsupported {name}={value}")


@dataclass(frozen=True)
class TaskShape:
    name: str
    count_field: str
    index_field: str


@dataclass(frozen=True)
class WarpLayout:
    task_order: tuple[str, ...]


_WARP_TASKS = {
    task.name: task
    for task in (
        TaskShape("epilogue", "num_epilogue_warps", "epilogue_warp_idx"),
        TaskShape("cast_a", "num_cast_a_warps", "cast_a_warp_idx_first"),
        TaskShape("gather", "num_gather_warps", "gather_warp_idx"),
        TaskShape("load_a", "num_load_a_warps", "load_a_warp_idx"),
        TaskShape("load_b", "num_load_b_warps", "load_b_warp_idx"),
        TaskShape("load_sfa", "num_load_sfa_warps", "load_sfa_warp_idx"),
        TaskShape("load_sfb", "num_load_sfb_warps", "load_sfb_warp_idx"),
        TaskShape("load_sfab", "num_load_sfab_warps", "load_sfab_warp_idx"),
        TaskShape("copy_sfa", "num_copy_sfa_warps", "copy_sfa_warp_idx"),
        TaskShape("copy_sfb", "num_copy_sfb_warps", "copy_sfb_warp_idx"),
        TaskShape("sync", "num_sync_warps", "sync_warp_idx"),
        TaskShape("mma", "num_mma_warps", "mma_warp_idx"),
        TaskShape("workid", "num_workid_warps", "workid_warp_idx"),
    )
}


def _pack_warp_layout(cfg: BatchedGemmConfig, layout: WarpLayout) -> None:
    warp_idx = 0
    for task_name in layout.task_order:
        task = _WARP_TASKS[task_name]
        setattr(cfg, task.index_field, warp_idx)
        warp_idx += int(getattr(cfg, task.count_field))

    # when using deepseek fp8, add another warp to load the scales
    if cfg.has_deepseek_fp8 and "load_sfab" not in layout.task_order:
        task = _WARP_TASKS["load_sfab"]
        setattr(cfg, task.index_field, warp_idx)
        warp_idx += int(getattr(cfg, task.count_field))

    total_warps = ((warp_idx + 3) // 4) * 4
    cfg.padding_warp_idx = warp_idx
    cfg.num_padding_warps = total_warps - warp_idx
    cfg.threads_per_cta = total_warps * 32


def compute_warp_layout(cfg: BatchedGemmConfig) -> None:
    """Compute warp indices/counts and write them into *cfg* (in place).

    Called once inside the kernel assembly.  Tests should never set warp
    indices or counts — this function is the single source of truth.
    """
    requested_load_b_warps = cfg.num_load_b_warps
    requested_load_sfa_warps = cfg.num_load_sfa_warps
    requested_load_sfb_warps = cfg.num_load_sfb_warps

    if cfg.has_deepseek_fp8_two_epilogue:
        cfg.num_epilogue_warps = 8
        cfg.epilogue_warp_idx = 0
        cfg.num_mma_warps = 1
        cfg.num_load_a_warps = 1
        cfg.num_load_b_warps = max(1, requested_load_b_warps)
        cfg.num_load_sfa_warps = 0
        cfg.num_load_sfb_warps = 0
        cfg.num_copy_sfa_warps = 0
        cfg.num_copy_sfb_warps = 0
        cfg.num_cast_a_warps = 0
        cfg.num_gather_warps = 0
        cfg.num_sync_warps = 0
        cfg.num_workid_warps = 1 if cfg.is_persistent else 0
        _pack_warp_layout(
            cfg,
            WarpLayout(
                (
                    "epilogue",
                    "mma",
                    "load_a",
                    "load_b",
                    "load_sfab",
                    "workid",
                )
            ),
        )
        return

    if cfg.use_tile256_tmem_overlap:
        cfg.num_mma_warps = 1
        cfg.num_copy_sfa_warps = 0
        cfg.num_copy_sfb_warps = 0
        cfg.num_gather_warps = 0
        cfg.num_sync_warps = 0
        cfg.num_workid_warps = 1 if cfg.is_persistent else 0

        cfg.epilogue_warp_idx = 0
        if cfg.has_activation_side_effects:
            if cfg.uses_mxfp8_output_quant:
                cfg.num_epilogue_warps = 4
                cfg.num_load_a_warps = 1
                cfg.num_load_b_warps = 8
                cfg.num_load_sfa_warps = 1
                cfg.num_load_sfb_warps = 4

                _pack_warp_layout(
                    cfg,
                    WarpLayout(
                        (
                            "epilogue",
                            "load_b",
                            "load_sfb",
                            "load_a",
                            "load_sfa",
                            "mma",
                            "workid",
                        )
                    ),
                )
                return

            cfg.num_epilogue_warps = 8
            cfg.num_load_a_warps = 1
            cfg.num_load_b_warps = 8
            cfg.num_load_sfa_warps = 1
            cfg.num_load_sfb_warps = 4

            _pack_warp_layout(
                cfg,
                WarpLayout(
                    (
                        "epilogue",
                        "load_b",
                        "load_sfb",
                        "load_a",
                        "load_sfa",
                        "mma",
                        "workid",
                    )
                ),
            )
            return

        cfg.num_epilogue_warps = 4
        cfg.num_load_a_warps = 1
        cfg.num_load_b_warps = 1
        cfg.num_load_sfa_warps = 1
        cfg.num_load_sfb_warps = 1

        _pack_warp_layout(
            cfg,
            WarpLayout(
                (
                    "epilogue",
                    "load_b",
                    "load_sfb",
                    "load_a",
                    "load_sfa",
                    "mma",
                    "workid",
                )
            ),
        )
        return

    if cfg.has_cast_a:
        cfg.num_epilogue_warps = 4
        cfg.epilogue_warp_idx = 0
        cfg.num_cast_a_warps = 4
        cfg.cast_a_warp_idx_first = 4
        cfg.num_mma_warps = 1
        cfg.num_load_a_warps = 1
        cfg.num_load_b_warps = 1
        cfg.num_load_sfa_warps = 1
        cfg.num_load_sfb_warps = 0
        cfg.num_copy_sfa_warps = 0
        cfg.num_copy_sfb_warps = 0
        cfg.num_sync_warps = 0
        cfg.num_workid_warps = 1 if cfg.is_persistent else 0

        c = cfg.cast_a_warp_idx_first + cfg.num_cast_a_warps
        if cfg.has_gather:
            if cfg.is_swap_ab:
                cfg.num_load_b_warps = 0
            else:
                cfg.num_load_a_warps = 0
            cfg.num_gather_warps = 4
            cfg.gather_warp_idx = c
            c += cfg.num_gather_warps
        else:
            cfg.num_gather_warps = 0
            cfg.load_b_warp_idx = c
            c += cfg.num_load_b_warps

        cfg.load_a_warp_idx = c
        c += cfg.num_load_a_warps
        if not cfg.has_gather:
            cfg.gather_warp_idx = c
        cfg.load_sfa_warp_idx = c
        c += cfg.num_load_sfa_warps
        cfg.load_sfb_warp_idx = c
        cfg.copy_sfa_warp_idx = c
        cfg.copy_sfb_warp_idx = c
        cfg.sync_warp_idx = c
        cfg.mma_warp_idx = c
        c += cfg.num_mma_warps
        cfg.workid_warp_idx = c
        c += cfg.num_workid_warps

        # DeepSeek FP8: one cp.async SF load warp.
        if cfg.has_deepseek_fp8:
            cfg.load_sfab_warp_idx = c
            c += cfg.num_load_sfab_warps

        total_warps = ((c + 3) // 4) * 4
        cfg.padding_warp_idx = c
        cfg.num_padding_warps = total_warps - c
        cfg.threads_per_cta = total_warps * 32
        return

    cfg.num_epilogue_warps = 8 if (cfg.is_fp8_mma and cfg.tile_n == 256) else 4
    cfg.epilogue_warp_idx = 0
    cfg.num_cast_a_warps = 0
    cfg.num_mma_warps = 1

    # Load-warps: gather replaces one TMA load operand.
    cfg.num_load_a_warps = 1
    cfg.num_load_b_warps = 1
    if cfg.has_gather:
        if cfg.is_swap_ab:
            cfg.num_load_b_warps = 0
        else:
            cfg.num_load_a_warps = 0
    elif cfg.has_tma_route:
        if cfg.is_swap_ab:
            if requested_load_b_warps > 1:
                cfg.num_load_b_warps = requested_load_b_warps
            elif cfg.is_fp8_mma and cfg.has_cluster:
                cfg.num_load_b_warps = 8
            elif cfg.tile_n == 8 or (cfg.has_cluster and cfg.tile_n == 16):
                cfg.num_load_b_warps = 2
            elif cfg.tile_n == 256:
                cfg.num_load_b_warps = 8
            else:
                cfg.num_load_b_warps = 4
        else:
            cfg.num_load_a_warps = 8 if cfg.tile_m == 256 else 4

    if cfg.has_scale_factors:
        cfg.num_load_sfa_warps = 1
        cfg.num_load_sfb_warps = 1
        if requested_load_sfa_warps > 1:
            cfg.num_load_sfa_warps = requested_load_sfa_warps
        elif cfg.sfa_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM):
            cfg.num_load_sfa_warps = min(4, max(1, (cfg.tile_m + 3) // 4))
        if requested_load_sfb_warps > 1:
            cfg.num_load_sfb_warps = requested_load_sfb_warps
        elif (
            cfg.has_routed_sfs
            and (cfg.uses_tma_routed_sfs or cfg.has_implicit_routed_sfs)
            and cfg.is_swap_ab
            and cfg.tile_n < 128
        ):
            # TMA-routed compact SFB kernels, and FC1 TMA-route kernels with
            # implicit routed SF, distribute activation-SF rows across the
            # generated LoadSfB warpgroup: tile8=>2, tile16=>4, and
            # tile32/tile64=>8 warps.
            cfg.num_load_sfb_warps = min(8, max(1, cfg.tile_n // 4))
    else:
        cfg.num_load_sfa_warps = 0
        cfg.num_load_sfb_warps = 0

    if cfg.has_gather:
        routed_rows = cfg.tile_n if cfg.is_swap_ab else cfg.tile_m
        # Packed FP4 has twice as many 16-byte copies per gather warp as FP8,
        # so two warps saturate the routed operand while avoiding a larger CTA.
        max_gather_warps = (
            2
            if (
                cfg.is_nvfp4_mma
                and cfg.is_swap_ab
                and cfg.dtype_b_smem_bits == 4
                and not cfg.fuse_operand_sf_loads
            )
            else 4
        )
        cfg.num_gather_warps = min(max_gather_warps, max(1, (routed_rows + 3) // 4))
    else:
        cfg.num_gather_warps = 0
    cfg.num_sync_warps = (
        1 if cfg.has_cluster and cfg.has_gather and not cfg.fuse_operand_sf_loads else 0
    )
    cfg.num_workid_warps = 1 if cfg.is_persistent else 0
    # CopySf warps copy scale factors to TMEM before the MMA. The compact low-N
    # SFB path needs the 4-warp STTM warpgroup; all bulk S2T paths use one warp
    # to stay inside the launch envelope.
    if cfg.uses_unfused_tmem_sf_copy:
        cfg.num_copy_sfa_warps = (
            4 if cfg.sfa_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM) else 1
        )
        if cfg.use_combined_sfab_copy:
            cfg.num_copy_sfb_warps = 0
        elif cfg.sfb_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM):
            cfg.num_copy_sfb_warps = 4
        else:
            cfg.num_copy_sfb_warps = 1
    else:
        cfg.num_copy_sfa_warps = 0
        cfg.num_copy_sfb_warps = 0

    if cfg.fuse_operand_sf_loads:
        # Gen's high-throughput FC1 layout:
        #   epilogue 0-3, routed B/SFB 4-7, A/SFA 8, MMA 9, WorkId 10,
        #   padding 11.
        _pack_warp_layout(
            cfg,
            WarpLayout(
                (
                    "epilogue",
                    "gather",
                    "load_a",
                    "mma",
                    "workid",
                )
            ),
        )
        cfg.load_b_warp_idx = cfg.gather_warp_idx
        cfg.load_sfb_warp_idx = cfg.gather_warp_idx
        cfg.load_sfa_warp_idx = cfg.load_a_warp_idx
        return

    sfb_uses_r8c4_sttm = cfg.sfb_smem_to_tmem_copy == int(
        SfSmemToTmemCopy.LDS_STTM
    ) and cfg.smem_sfb_layout == int(SfLayout.R8c4)
    if sfb_uses_r8c4_sttm:
        # The compact R8c4 SFB path dedicates the first post-epilogue warpgroup
        # to STTM. Keeping CopySfB warpgroup-aligned avoids issuing tcgen05_st
        # from a split warpgroup for low-N block-scaled variants.
        _pack_warp_layout(
            cfg,
            WarpLayout(
                (
                    "epilogue",
                    "copy_sfb",
                    "gather",
                    "load_b",
                    "load_sfb",
                    "load_a",
                    "load_sfa",
                    "copy_sfa",
                    "sync",
                    "mma",
                    "workid",
                )
            ),
        )
        if cfg.has_activation_side_effects and cfg.tile_n == 64:
            cfg.epilogue_regs = max(cfg.epilogue_regs, 160)
        return

    if cfg.use_combined_sfab_copy:
        # The combined-copy path issues B/SFB first, then A/SFA, then a single
        # CopySfAb warp before MMA. Keeping this order avoids the lowering hang
        # observed with two independent TmemSf pipelines.
        _pack_warp_layout(
            cfg,
            WarpLayout(
                (
                    "epilogue",
                    "load_b",
                    "load_sfb",
                    "load_a",
                    "load_sfa",
                    "copy_sfa",
                    "copy_sfb",
                    "gather",
                    "sync",
                    "mma",
                    "workid",
                )
            ),
        )
        return

    if (
        cfg.has_tma_route
        and cfg.is_swap_ab
        and cfg.has_cluster
        and cfg.tile_n >= 128
        and cfg.num_load_b_warps == 4
        and cfg.num_load_sfb_warps == 4
    ):
        # Keep the two four-warp routed load tasks warpgroup-aligned.  Besides
        # matching the generated high-throughput schedule, this lets LoadB/SFB
        # use their larger register budgets without sharing a warpgroup with
        # the low-register MMA, LoadA, LoadSfA, and WorkId tasks.
        _pack_warp_layout(
            cfg,
            WarpLayout(
                (
                    "epilogue",
                    "load_b",
                    "load_sfb",
                    "load_a",
                    "load_sfa",
                    "mma",
                    "workid",
                )
            ),
        )
        return

    if (
        cfg.has_gather
        and cfg.is_swap_ab
        and cfg.cluster_m == 1
        and cfg.num_gather_warps < 4
    ):
        # Low-N LDGSTS order: MMA, TMA-A, LDGSTS gather, then CLC work-id. The
        # condition keys on the task shape (is_swap_ab, 1-CTA, < 4 gather warps),
        # not on a dtype- or tile8-specific special case.
        _pack_warp_layout(
            cfg,
            WarpLayout(
                (
                    "epilogue",
                    "mma",
                    "load_a",
                    "load_b",
                    "gather",
                    "load_sfa",
                    "load_sfb",
                    "copy_sfa",
                    "copy_sfb",
                    "sync",
                    "workid",
                )
            ),
        )
        return

    # --- pack tasks sequentially after epilogue ---
    if cfg.has_gather:
        default_order = (
            "epilogue",
            "gather",
            "mma",
            "load_a",
            "load_b",
            "load_sfa",
            "load_sfb",
            "copy_sfa",
            "copy_sfb",
            "sync",
            "workid",
        )
    else:
        default_order = (
            "epilogue",
            "mma",
            "load_a",
            "load_b",
            "load_sfa",
            "load_sfb",
            "copy_sfa",
            "copy_sfb",
            "gather",
            "sync",
            "workid",
        )
    _pack_warp_layout(cfg, WarpLayout(default_order))


def _ldgsts_sfb_base_producer_commit_prefetch_depth(cfg: BatchedGemmConfig) -> int:
    """Return maximum delayed SFB commit depth for this config."""
    if (
        cfg.has_cluster
        and cfg.is_swap_ab
        and cfg.tile_n >= 128
        and cfg.sfb_smem_to_tmem_copy != int(SfSmemToTmemCopy.LDS_STTM)
    ):
        return min(
            MAX_PRODUCER_COMMIT_PREFETCH_DEPTH,
            cfg.num_stages_smem_sfb - 1,
        )
    return 0


def _ldgsts_sfb_producer_commit_prefetch_depth(cfg: BatchedGemmConfig) -> int:
    """Return the host-derived delayed SFB commit depth consumed by resources."""
    return min(
        _ldgsts_sfb_base_producer_commit_prefetch_depth(cfg),
        cfg.ldgsts_sfb_producer_commit_prefetch_depth,
    )


def _validate_pipeline_stage_count(cfg: BatchedGemmConfig, field_name: str) -> None:
    """Reject zero-depth pipeline rings before deriving schedule constants."""
    value = getattr(cfg, field_name)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}")


def _validate_pipeline_stage_counts(cfg: BatchedGemmConfig) -> None:
    """Validate stage counts for pipelines/resources this config instantiates."""
    for field_name in ("num_stages_a", "num_stages_b", "num_stages_tmem_acc"):
        _validate_pipeline_stage_count(cfg, field_name)

    if cfg.use_tma_store != 0:
        _validate_pipeline_stage_count(cfg, "num_stages_c_smem")

    if cfg.has_scale_factor_a or cfg.has_deepseek_fp8:
        _validate_pipeline_stage_count(cfg, "num_stages_smem_sfa")

    if cfg.has_scale_factors:
        _validate_pipeline_stage_count(cfg, "num_stages_smem_sfb")

    if cfg.uses_unfused_tmem_sf_copy:
        _validate_pipeline_stage_count(cfg, "num_stages_tmem_sfa")
        _validate_pipeline_stage_count(cfg, "num_stages_tmem_sfb")

    if cfg.is_persistent:
        _validate_pipeline_stage_count(cfg, "num_stages_workid")

    if cfg.has_gather and cfg.has_cluster and cfg.num_stages_a != cfg.num_stages_b:
        raise ValueError(
            "2-CTA gather requires num_stages_a == num_stages_b, got "
            f"{cfg.num_stages_a} and {cfg.num_stages_b}"
        )


def _validate_output_store_dtype(cfg: BatchedGemmConfig) -> None:
    """Validate dtype_c against the output store formats implemented by GmemC."""
    if cfg.dtype_c_kind in (int(DType.BF16), int(DType.FP16)):
        return

    if cfg.uses_fp8_output:
        return

    if cfg.uses_fp4_output_quant or cfg.uses_mxfp8_output_quant:
        return

    if cfg.dtype_c_kind == int(DType.E4M3):
        raise ValueError("dtype_c=e4m3 plain FP8 output requires dtype_a=dtype_b=e4m3")

    raise ValueError(
        "Unsupported dtype_c output store: "
        f"{cfg.dtype_c_kind}. Supported dtype_c values are bf16/fp16 plain "
        "stores, e4m3 for FP8 MMA output, and e2m1/mxe2m1/mxe4m3 for output "
        "block scaling."
    )


def derive_problem_shape_constants(
    cfg: BatchedGemmConfig,
    problem_mnk: tuple[int, int, int] | None = None,
) -> None:
    """Derive host-known constexpr schedule constants for ``cfg`` in place."""
    if problem_mnk is None:
        cfg.ldgsts_sfb_producer_commit_prefetch_depth = (
            _ldgsts_sfb_base_producer_commit_prefetch_depth(cfg)
        )
        return

    _, _, problem_k = problem_mnk
    # ``validate_config`` also runs inside staged host assembly where problem
    # sizes are DSL values.  On the Python launch path, derive the delayed SFB
    # commit depth from the host-known K tile count so the JIT schedule receives
    # a constexpr-safe prefetch window.
    if isinstance(problem_k, int):
        cfg.ldgsts_sfb_producer_commit_prefetch_depth = (
            _ldgsts_sfb_base_producer_commit_prefetch_depth(cfg)
        )
        num_k_tiles = (problem_k + cfg.tile_k - 1) // cfg.tile_k
        cfg.ldgsts_sfb_producer_commit_prefetch_depth = min(
            cfg.ldgsts_sfb_producer_commit_prefetch_depth,
            num_k_tiles,
        )


def validate_config(
    cfg: BatchedGemmConfig,
    problem_mnk: tuple[int, int, int] | None = None,
) -> None:
    """Validate config before kernel launch.  Raises ValueError for invalid
    or unsupported combinations, RuntimeError for untested-but-plausible ones.

    Called at the top of kernel assembly (both validation and GPU paths).
    """
    # ── Hard errors: combinations that are known to fail or are nonsensical ──

    if cfg.batch_mode not in (int(BatchMode.BATCH_N), int(BatchMode.BATCH_M)):
        raise ValueError(f"batch_mode must be BATCH_N or BATCH_M, got {cfg.batch_mode}")

    if cfg.weight_layout_kind not in (
        int(WeightLayout.MajorK),
        int(WeightLayout.BlockMajorK),
    ):
        if cfg.weight_layout_kind == int(WeightLayout.MajorMn):
            raise ValueError("weight_layout=MajorMn is not supported by Prims-TS")
        raise ValueError(f"Unsupported weight_layout={cfg.weight_layout}")

    _validate_binary_config_option(
        "transpose_mma_output",
        cfg.transpose_mma_output,
    )
    if cfg.transpose_mma_output != int(cfg.is_swap_ab):
        raise ValueError(
            "transpose_mma_output must match the batch_mode-selected swap_ab "
            f"path, got transpose_mma_output={cfg.transpose_mma_output}, "
            f"batch_mode={BatchMode(cfg.batch_mode).name}"
        )

    if cfg.tile_scheduler not in (
        int(TileScheduler.STATIC),
        int(TileScheduler.PERSISTENT),
    ):
        raise ValueError(
            f"tile_scheduler must be STATIC or PERSISTENT, got {cfg.tile_scheduler}"
        )

    if cfg.cluster_m not in (1, 2):
        raise ValueError(
            "cluster_m must be 1 or 2; the kernel only implements CTA_1/CTA_2 "
            f"cluster paths, got {cfg.cluster_m}"
        )

    _validate_pipeline_stage_counts(cfg)
    _validate_output_store_dtype(cfg)

    if cfg.has_cast_a and cfg.dtype_c_kind not in (
        int(DType.BF16),
        int(DType.FP16),
    ):
        raise ValueError(
            "CastA MXFP4 input supports dtype_c=bf16/fp16 plain stores only, "
            f"got {cfg.dtype_c_kind}"
        )

    derive_problem_shape_constants(cfg, problem_mnk)

    if cfg.tile_m != 128:
        raise ValueError(f"Only tile_m=128 supported, got {cfg.tile_m}")

    if cfg.mma_k not in (16, 32, 64):
        raise ValueError(
            f"mma_k must be 16 (BF16), 32 (MX/FP8), or 64 (NVFP4), got {cfg.mma_k}"
        )

    if cfg.is_bf16_mma and cfg.mma_k != 16:
        raise ValueError(f"BF16 MMA requires mma_k=16, got {cfg.mma_k}")

    if cfg.has_cast_a and cfg.mma_k != 16:
        raise ValueError(f"CastA BF16 MMA requires mma_k=16, got {cfg.mma_k}")

    if cfg.is_nvfp4_mma and cfg.mma_k != 64:
        raise ValueError(f"NVFP4 MMA requires mma_k=64, got {cfg.mma_k}")

    if cfg.is_mx_mma and cfg.mma_k != 32:
        raise ValueError(f"MX MMA requires mma_k=32, got {cfg.mma_k}")

    if cfg.is_fp8_mma and cfg.mma_k != 32:
        raise ValueError(f"FP8 MMA requires mma_k=32, got {cfg.mma_k}")

    if cfg.tile_k % cfg.mma_k != 0:
        raise ValueError(
            f"tile_k must be a multiple of mma_k, got tile_k={cfg.tile_k}, "
            f"mma_k={cfg.mma_k}"
        )

    if cfg.uses_block_major_k_weight:
        block_k = cfg.block_major_k_elems
        tile_block_k = cfg.block_major_k_tile_block_elems
        if cfg.tile_k % tile_block_k != 0:
            raise ValueError(
                "BlockMajorK requires tile_k to divide into whole TMA K-block "
                f"slices, got tile_k={cfg.tile_k}, tile_block_k={tile_block_k}"
            )
        if cfg.tile_k < block_k and block_k % cfg.tile_k != 0:
            raise ValueError(
                "BlockMajorK tile_k must either cover whole K blocks or "
                f"divide one block evenly, got tile_k={cfg.tile_k}, "
                f"block_k={block_k}, block_bytes={cfg.block_major_k_bytes}"
            )
        if problem_mnk is not None:
            _, _, problem_k = problem_mnk
            if isinstance(problem_k, int) and problem_k % block_k != 0:
                raise ValueError(
                    "BlockMajorK requires problem K to be a multiple of the "
                    f"K block, got problem_k={problem_k}, block_k={block_k}, "
                    f"block_bytes={cfg.block_major_k_bytes}"
                )

    if cfg.has_cast_a and cfg.tile_k != 256:
        raise ValueError(f"CastA MXFP4 input requires tile_k=256, got {cfg.tile_k}")

    if cfg.use_bf16_kbox_tma_a and cfg.tile_k % 64 != 0:
        raise ValueError(
            "BF16 k-box TMA A path requires tile_k to be a multiple of 64, "
            f"got tile_k={cfg.tile_k}"
        )

    if cfg.is_mx_mma and (
        cfg.input_sf_block_size_a != 32 or cfg.input_sf_block_size_b != 32
    ):
        raise ValueError(
            "MX input block scaling requires sf_block_size_a/b=32, got "
            f"{cfg.input_sf_block_size_a}/{cfg.input_sf_block_size_b}"
        )

    if cfg.uses_mx_output_quant and cfg.output_sf_block_size_c != 32:
        raise ValueError(
            f"MX output block scaling requires sf_block_size_c=32, got {cfg.output_sf_block_size_c}"
        )

    if (
        cfg.uses_fp4_output_quant
        and not cfg.uses_mxfp4_output_quant
        and cfg.output_sf_block_size_c != 16
    ):
        raise ValueError(
            f"NVFP4 output block scaling requires sf_block_size_c=16, got {cfg.output_sf_block_size_c}"
        )

    if cfg.has_cast_a:
        if cfg.input_sf_block_size_a != 32:
            raise ValueError(
                "CastA MXFP4 input requires sf_block_size_a=32, got "
                f"{cfg.input_sf_block_size_a}"
            )
        if cfg.sf_layout_a != int(SfLayout.R128c4):
            raise ValueError(
                f"CastA MXFP4 input currently requires sf_layout_a=128x4, got {cfg.sf_layout_a}"
            )
        if cfg.tmem_cast_a_cols != 128:
            raise ValueError(
                f"CastA TMEM staging expects 128 columns per stage, got {cfg.tmem_cast_a_cols}"
            )

    if cfg.mma_m not in (64, 128, 256):
        raise ValueError(f"mma_m must be 64/128/256, got {cfg.mma_m}")

    if cfg.mma_n not in (8, 16, 32, 64, 128, 192, 256):
        raise ValueError(f"mma_n must be 8/16/32/64/128/192/256, got {cfg.mma_n}")

    if cfg.tile_n % cfg.mma_n != 0:
        raise ValueError(
            f"tile_n must be a multiple of mma_n, got tile_n={cfg.tile_n}, "
            f"mma_n={cfg.mma_n}"
        )

    cluster_tile_m = cfg.tile_m * cfg.cluster_m
    if cluster_tile_m % cfg.mma_m != 0:
        raise ValueError(
            "cluster-wide tile_m must be a multiple of mma_m, got "
            f"tile_m={cfg.tile_m}, cluster_m={cfg.cluster_m}, mma_m={cfg.mma_m}"
        )

    if cfg.cluster_m > 1 and cfg.mma_n < 16:
        raise ValueError(
            f"cluster_m={cfg.cluster_m} requires mma_n >= 16, got mma_n={cfg.mma_n}"
        )

    if cfg.is_nvfp4_mma and cfg.mma_m == 64:
        raise ValueError("NVFP4 MMA does not support mma_m=64")

    if cfg.tile_n not in (8, 16, 32, 64, 128, 192, 256):
        raise ValueError(f"tile_n must be 8/16/32/64/128/192/256, got {cfg.tile_n}")

    if cfg.epi_tile_n <= 0:
        raise ValueError(f"epi_tile_n must be positive, got {cfg.epi_tile_n}")

    if cfg.epi_tile_n % 8 != 0:
        raise ValueError(f"epi_tile_n must be a multiple of 8, got {cfg.epi_tile_n}")

    if cfg.tile_n % cfg.epi_tile_n != 0:
        raise ValueError(
            f"epi_tile_n must divide tile_n, got tile_n={cfg.tile_n}, "
            f"epi_tile_n={cfg.epi_tile_n}"
        )

    if cfg.tmem_required_cols > 512:
        raise ValueError(
            f"TMEM footprint requires {cfg.tmem_required_cols} columns, "
            "which exceeds the 512-column tcgen05 allocation limit"
        )

    # batch_mode selects swap_ab and the expert-coordinate axis. The generated
    # transpose flag is retained but validated above against that kernel path.

    if cfg.use_max_tmem_overlap not in (0, 1):
        raise ValueError(
            f"use_max_tmem_overlap must be 0 or 1, got {cfg.use_max_tmem_overlap}"
        )
    if cfg.fuse_sf_copy_to_mma not in (0, 1):
        raise ValueError(
            f"fuse_sf_copy_to_mma must be 0 or 1, got {cfg.fuse_sf_copy_to_mma}"
        )
    if cfg.fuse_sf_copy_to_mma != 0 and not cfg.has_scale_factors:
        raise ValueError("fuse_sf_copy_to_mma requires block-scaled MMA operands")
    _validate_binary_config_option(
        "fuse_operand_sf_loads",
        cfg.fuse_operand_sf_loads,
    )
    if cfg.fuse_operand_sf_loads:
        if not (
            cfg.has_scale_factors
            and cfg.has_gather
            and cfg.has_cluster
            and cfg.is_swap_ab
            and cfg.uses_ldgsts_routed_sfs
            and cfg.tile_n >= 128
        ):
            raise ValueError(
                "fuse_operand_sf_loads requires clustered swap-AB LDGSTS "
                "activation and scale-factor routing with tile_n >= 128"
            )
        if not cfg.fuse_sf_copy_to_mma:
            raise ValueError("fuse_operand_sf_loads requires fuse_sf_copy_to_mma=1")
        if cfg.num_load_sfa_warps != cfg.num_load_a_warps:
            raise ValueError(
                "fuse_operand_sf_loads requires num_load_sfa_warps == "
                "num_load_a_warps, got "
                f"{cfg.num_load_sfa_warps} and {cfg.num_load_a_warps}"
            )
        if cfg.num_load_sfb_warps != cfg.num_gather_warps:
            raise ValueError(
                "fuse_operand_sf_loads requires num_load_sfb_warps == "
                "num_gather_warps, got "
                f"{cfg.num_load_sfb_warps} and {cfg.num_gather_warps}"
            )
        sfb_prefetch_depth = _ldgsts_sfb_producer_commit_prefetch_depth(cfg)
        if sfb_prefetch_depth >= cfg.num_stages_b:
            raise ValueError(
                "fuse_operand_sf_loads requires the delayed SFB commit depth "
                "to be smaller than the fused B pipeline depth, got "
                f"prefetch_depth={sfb_prefetch_depth}, "
                f"num_stages_b={cfg.num_stages_b}"
            )
    if cfg.use_tma_oob_opt not in (0, 1):
        raise ValueError(f"use_tma_oob_opt must be 0 or 1, got {cfg.use_tma_oob_opt}")

    if cfg.use_early_exit not in (0, 1):
        raise ValueError(f"use_early_exit must be 0 or 1, got {cfg.use_early_exit}")

    if cfg.use_clc_fast_drain not in (0, 1):
        raise ValueError(
            f"use_clc_fast_drain must be 0 or 1, got {cfg.use_clc_fast_drain}"
        )
    if cfg.use_clc_fast_drain and (not cfg.is_persistent or not cfg.use_early_exit):
        raise ValueError(
            "use_clc_fast_drain requires persistent scheduling and use_early_exit=1"
        )

    if cfg.use_work_throttle not in (0, 1):
        raise ValueError(
            f"use_work_throttle must be 0 or 1, got {cfg.use_work_throttle}"
        )

    _validate_binary_config_option(
        "use_unroll_loop_2x_for_mma",
        cfg.use_unroll_loop_2x_for_mma,
    )
    if cfg.use_unroll_loop_2x_for_mma and cfg.has_deepseek_fp8:
        raise ValueError(
            "use_unroll_loop_2x_for_mma is not supported by the DeepSeek FP8 "
            "per-K accumulator pipeline"
        )

    _validate_binary_config_option(
        "use_full_tmem_dealloc_barrier",
        cfg.use_full_tmem_dealloc_barrier,
    )

    if cfg.use_pdl not in (0, 1):
        raise ValueError(f"use_pdl must be 0 or 1, got {cfg.use_pdl}")

    if cfg.do_pdl_wait_for_num_non_exiting_ctas not in (0, 1):
        raise ValueError(
            "do_pdl_wait_for_num_non_exiting_ctas must be 0 or 1, got "
            f"{cfg.do_pdl_wait_for_num_non_exiting_ctas}"
        )

    if cfg.do_pdl_wait_for_num_non_exiting_ctas:
        if not cfg.use_pdl:
            raise ValueError("do_pdl_wait_for_num_non_exiting_ctas requires use_pdl=1")
        if not cfg.use_early_exit:
            raise ValueError(
                "do_pdl_wait_for_num_non_exiting_ctas requires use_early_exit=1"
            )
    elif cfg.use_pdl and cfg.use_early_exit:
        raise ValueError(
            "PDL early exit requires do_pdl_wait_for_num_non_exiting_ctas=1 "
            "before reading the routing-produced active CTA count"
        )

    if cfg.use_global_scales not in (0, 1):
        raise ValueError(
            f"use_global_scales must be 0 or 1, got {cfg.use_global_scales}"
        )

    if cfg.use_per_token_sf_a not in (0, 1):
        raise ValueError(
            f"use_per_token_sf_a must be 0 or 1, got {cfg.use_per_token_sf_a}"
        )

    if cfg.use_per_token_sf_b not in (0, 1):
        raise ValueError(
            f"use_per_token_sf_b must be 0 or 1, got {cfg.use_per_token_sf_b}"
        )

    for name, value in (
        ("has_gemm1_alpha", cfg.has_gemm1_alpha),
        ("has_gemm1_beta", cfg.has_gemm1_beta),
        ("has_gemm1_clamp_limit", cfg.has_gemm1_clamp_limit),
    ):
        _validate_binary_config_option(name, value)

    if cfg.has_gemm1_activation_params and cfg.act_kind not in (
        int(ActKind.SWIGLU),
        int(ActKind.SITU),
    ):
        raise ValueError("GEMM1 activation params require act_kind=SWIGLU or SITU")

    if cfg.per_token_sf_dtype not in (
        int(DType.FP32),
        int(DType.BF16),
        int(DType.FP16),
    ):
        raise ValueError(
            "per_token_sf_dtype must be FP32, BF16, or FP16, got "
            f"{cfg.per_token_sf_dtype}"
        )

    if cfg.use_per_token_sf_a != 0 and not cfg.is_fp8_mma:
        raise ValueError("Per-token sf_a is only valid for plain FP8 MMA")

    if cfg.use_per_token_sf_b != 0:
        if not (cfg.is_fp8_mma or cfg.is_nvfp4_mma):
            raise ValueError("Per-token sf_b is only valid for plain FP8 or NVFP4 MMA")
        if cfg.has_bias_m and cfg.is_fp8_mma:
            raise ValueError(
                "FP8 per-token sf_b with bias is parsed but not implemented"
            )

    if cfg.use_deepseek_fp8 not in (0, 1):
        raise ValueError(f"use_deepseek_fp8 must be 0 or 1, got {cfg.use_deepseek_fp8}")

    if cfg.has_deepseek_fp8:
        if not cfg.is_fp8_mma:
            raise ValueError(
                "use_deepseek_fp8=1 requires E4M3 MMA (dtype_a=dtype_b=E4M3)"
            )
        if cfg.tile_k != 128:
            raise ValueError(
                "use_deepseek_fp8=1 requires tile_k=128 "
                f"for DeepSeek scale-factor blocks, got {cfg.tile_k}"
            )
        if cfg.has_gated_epilogue:
            raise ValueError(
                "use_deepseek_fp8=1 is incompatible with fused gated activation"
            )
        if cfg.use_max_tmem_overlap != 0:
            raise ValueError(
                "use_deepseek_fp8=1 is incompatible with use_max_tmem_overlap"
            )
        if cfg.cluster_m > 1:
            raise ValueError("use_deepseek_fp8=1 is incompatible with cluster m > 1")
        if cfg.has_input_block_scaling:
            raise ValueError(
                "use_deepseek_fp8=1 is incompatible with MX/NVFP4 input block scaling"
            )
        if cfg.use_per_token_sf_a != 0 or cfg.use_per_token_sf_b != 0:
            raise ValueError(
                "use_deepseek_fp8=1 is incompatible with use_per_token_sf_a/b"
            )
        if cfg.num_load_sfab_warps != 1:
            raise ValueError(
                "use_deepseek_fp8=1 requires num_load_sfab_warps=1, "
                f"got {cfg.num_load_sfab_warps}"
            )
        if cfg.has_deepseek_fp8_c_scale and not cfg.is_swap_ab:
            raise ValueError(
                "DeepSeek FP8 dtype_c=e4m3 C-scale output requires is_swap_ab"
            )
        if cfg.deepseek_fp8_epi_subtile_cnt != 1:
            raise ValueError(
                "use_deepseek_fp8=1 requires epi_subtile_cnt=1 because the "
                "current DeepSeek FP8 epilogue stores one output-N subtile, "
                f"got {cfg.deepseek_fp8_epi_subtile_cnt} "
                f"(tile_n={cfg.tile_n}, epi_tile_n={cfg.epi_tile_n}, "
                f"num_epilogue_warps={cfg.num_epilogue_warps})"
            )

    if cfg.use_tma_store not in (0, 1):
        raise ValueError(f"use_tma_store must be 0 or 1, got {cfg.use_tma_store}")

    if (
        cfg.uses_deepseek_fp8_split_epilogue_tma_c_scratch
        and cfg.num_stages_c_smem != 1
    ):
        raise ValueError(
            "DeepSeek FP8 split-epilogue TMA C scratch requires "
            f"num_stages_c_smem=1, got {cfg.num_stages_c_smem}"
        )

    supported_sf_layouts = {
        int(SfLayout.R8c4),
        int(SfLayout.LINEAR),
        int(SfLayout.R128c4),
    }
    if cfg.sf_layout_a not in supported_sf_layouts:
        raise ValueError(f"Unsupported sf_layout_a: {cfg.sf_layout_a}")
    if cfg.sf_layout_b not in supported_sf_layouts:
        raise ValueError(f"Unsupported sf_layout_b: {cfg.sf_layout_b}")
    if cfg.sf_layout_c not in supported_sf_layouts:
        raise ValueError(f"Unsupported sf_layout_c: {cfg.sf_layout_c}")

    if cfg.has_scale_factors and cfg.sf_layout_a == int(SfLayout.R8c4):
        raise ValueError(
            "sf_layout_a=8x4 is not supported for input SF loads; "
            "8x4 load support is only for the non-gather B tensor with an "
            "8-row aligned tile_n <= 64 or tile_n == 192"
        )

    if (
        cfg.has_scale_factors
        and cfg.sf_layout_b == int(SfLayout.R8c4)
        and not cfg.uses_sfb_8x4_load
        and not (cfg.has_routed_sfs and cfg.is_swap_ab)
    ):
        raise ValueError(
            "sf_layout_b=8x4 requires non-gather B scale-factor loading with "
            "an 8-row aligned tile_n <= 64, tile_n == 192, or routed SFB"
        )

    if (
        cfg.has_scale_factors
        and cfg.sf_layout_b == int(SfLayout.R128c4)
        and cfg.tile_n % 128 != 0
        and not (cfg.has_routed_sfs and cfg.is_swap_ab)
    ):
        raise ValueError(
            "sf_layout_b=128x4 requires tile_n to be a multiple of 128 for "
            "non-routed B scale-factor loading"
        )

    if cfg.has_scale_factors and not cfg.uses_unfused_tmem_sf_copy:
        fused_sf_layouts = {
            "SFA": cfg.smem_sfa_layout,
            "SFB": cfg.smem_sfb_layout,
        }
        unsupported_fused_layouts = [
            name
            for name, layout in fused_sf_layouts.items()
            if layout != int(SfLayout.R128c4)
        ]
        if unsupported_fused_layouts:
            raise ValueError(
                "fused SF copy requires effective R128c4 SMEM layout for "
                + "/".join(unsupported_fused_layouts)
            )

    if cfg.has_epilogue_quant:
        if not cfg.has_scale_factors:
            raise ValueError("Output block scaling requires FP4/MX input scale factors")
        if not (cfg.uses_fp4_output_quant or cfg.uses_mxfp8_output_quant):
            raise ValueError(
                "Output block scaling requires dtype_c=e2m1, mxe2m1, or mxe4m3"
            )
        if not cfg.is_swap_ab or not cfg.has_gated_epilogue:
            raise ValueError(
                "Output block scaling currently requires is_swap_ab with a gated epilogue"
            )
        if cfg.sf_layout_c not in (int(SfLayout.R8c4), int(SfLayout.R128c4)):
            raise ValueError("Output block scaling requires sf_layout_c 8x4 or 128x4")

    if (
        cfg.uses_fp8_output
        and not cfg.has_activation_epilogue
        and not cfg.has_deepseek_fp8_c_scale
    ):
        raise ValueError("Plain FP8 C output requires an activation-side epilogue")

    if cfg.use_max_tmem_overlap != 0:
        if not cfg.use_tile256_tmem_overlap:
            raise ValueError(
                "use_max_tmem_overlap is only supported for block-scaled tile_n=256 kernels"
            )
        supported_tile_k = (128, 256) if cfg.is_mx_mma else (256,)
        if (
            cfg.tile_k not in supported_tile_k
            or cfg.cluster_m != 2
            or not cfg.is_persistent
        ):
            raise ValueError(
                "use_max_tmem_overlap currently supports the generated persistent "
                "2-CTA tile256 shape only, got "
                f"tile_k={cfg.tile_k}, cluster_m={cfg.cluster_m}, "
                f"persistent={cfg.is_persistent}"
            )
        if cfg.num_stages_tmem_acc != 1:
            raise ValueError(
                "use_max_tmem_overlap requires one accumulator TMEM stage, got "
                f"num_stages_tmem_acc={cfg.num_stages_tmem_acc}"
            )
        if cfg.tmem_total_cols < 512 or cfg.tmem_c_cols_per_stage != 256:
            raise ValueError(
                "use_max_tmem_overlap tile256 expects 512 TMEM columns with "
                f"256 C columns, got total={cfg.tmem_total_cols}, "
                f"c_cols={cfg.tmem_c_cols_per_stage}"
            )

    _validate_route_config_option("route_act", cfg.route_act)
    _validate_route_config_option("route_sfs_act", cfg.route_sfs_act)

    if cfg.bias_type not in (int(BiasType.NONE), int(BiasType.M)):
        raise ValueError(f"Unsupported bias_type={cfg.bias_type}")
    if cfg.has_bias_m and problem_mnk is not None:
        problem_m, _, _ = problem_mnk
        if isinstance(problem_m, int) and problem_m > 0:
            bias_block_size = 32 if cfg.epi_tile_m % 128 == 0 else 16
            if problem_m % bias_block_size != 0:
                raise ValueError(
                    "BiasType.M uses TRT-LLM Gen shuffleMatrixA storage and "
                    "requires problem M to be a multiple of "
                    f"{bias_block_size}, got {problem_m}"
                )

    route_sfs_explicit = cfg.route_sfs_act != int(RouteImpl.NONE)

    # route_sfs_act=NONE is valid even when route_act!=NONE. For TMA-routed
    # block-scaled activations, it selects the implicit LDGSTS routed-SF path.
    if route_sfs_explicit and cfg.route_act == int(RouteImpl.NONE):
        raise ValueError(
            "route_sfs_act requires route_act != NONE; cannot gather SF without "
            "gathering activations"
        )

    if route_sfs_explicit and not cfg.has_scale_factors:
        raise ValueError(
            "route_sfs_act requires FP4/MX input block scale factors; no routed "
            "SF resource is instantiated for this dtype combination"
        )

    # SF TMA gather4 requires at least 32 SF elements per tile for the routed
    # operand. Use LDGSTS when tile_k is too small for the operand SF block size.
    if cfg.has_routed_sfs and cfg.uses_tma_routed_sfs:
        routed_sf_operand = "b" if cfg.is_swap_ab else "a"
        routed_sf_block_size = (
            cfg.input_sf_block_size_b if cfg.is_swap_ab else cfg.input_sf_block_size_a
        )
        sf_k = cfg.tile_k // routed_sf_block_size
        if sf_k < 32:
            raise ValueError(
                "SF TMA gather4 requires sf_k >= 32 for the routed activation "
                f"SF operand, got sf_{routed_sf_operand} with sf_k={sf_k} "
                f"for tile_k={cfg.tile_k} and "
                f"input_sf_block_size_{routed_sf_operand}={routed_sf_block_size}. "
                "Use route_sfs_act=LDGSTS."
            )

    if cfg.has_gather:
        activation_operand = "b" if cfg.is_swap_ab else "a"
        activation_dtype_bits = cfg.dtype_b_bits if cfg.is_swap_ab else cfg.dtype_a_bits
        if activation_dtype_bits not in (4, 8, 16):
            raise ValueError(
                "LDGSTS activation gather requires a supported packed or "
                "byte-addressable activation "
                f"elements, got dtype_{activation_operand} with "
                f"{activation_dtype_bits} bits."
            )
        if (cfg.tile_k * activation_dtype_bits) % 128 != 0:
            raise ValueError(
                "LDGSTS activation gather requires each row tile to contain an "
                f"integer number of 16-byte copies, got tile_k={cfg.tile_k} "
                f"and dtype_{activation_operand}={activation_dtype_bits} bits."
            )

    if cfg.act_kind not in (
        int(ActKind.NONE),
        int(ActKind.SWIGLU),
        int(ActKind.GEGLU),
        int(ActKind.RELU2),
        int(ActKind.SILU),
        int(ActKind.SITU),
    ):
        raise ValueError(f"Unsupported act_kind={cfg.act_kind}")

    # Activation is valid for any kernel (FC1 convention, but not restricted).


def _normalize_config_overrides(overrides: dict) -> dict:
    """Validate that all overrides target BatchedGemmConfig fields."""
    result = dict(overrides)

    derived_fields = {"ldgsts_sfb_producer_commit_prefetch_depth"}
    valid_fields = {
        field.name
        for field in fields(BatchedGemmConfig)
        if field.name not in derived_fields
    }
    unknown = sorted(key for key in result if key not in valid_fields)
    if unknown:
        raise TypeError(f"Unknown BatchedGemmConfig option(s): {', '.join(unknown)}")

    return result


def make_config(**overrides) -> BatchedGemmConfig:
    """Patch helper for tests / sweep runs."""
    overrides = _normalize_config_overrides(overrides)
    return BatchedGemmConfig(**overrides)


def compute_max_num_ctas_in_token_dim_for_moe(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    token_tile_size: int,
    cluster_dim_in_token: int,
) -> int:
    """Compute the worst-case MoE token-CTA launch extent.

    The generated testbed first gives one token to as many experts as possible,
    then packs the remaining routed tokens into one expert. This maximizes CGA
    count for a fixed ``num_tokens * top_k``. TS early-exit indexes token work
    at CTA granularity, so convert the CGA count by the token-axis cluster width.
    """
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if top_k <= 0 or top_k > num_experts:
        raise ValueError(f"top_k must be in [1, num_experts], got {top_k}")
    if token_tile_size <= 0:
        raise ValueError(f"token_tile_size must be positive, got {token_tile_size}")
    if cluster_dim_in_token <= 0:
        raise ValueError(
            f"cluster_dim_in_token must be positive, got {cluster_dim_in_token}"
        )

    cga_tile_size = token_tile_size * cluster_dim_in_token
    num_remaining_tokens = num_tokens * top_k
    num_filled_experts = min(num_experts, num_remaining_tokens)
    max_num_cgas = num_filled_experts
    num_remaining_tokens -= num_filled_experts
    if num_remaining_tokens > 0:
        max_num_cgas += num_remaining_tokens // cga_tile_size
    return max_num_cgas * cluster_dim_in_token


def resolve_early_exit_max_token_ctas(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    cfg_overrides: dict,
    explicit_early_exit_max_token_ctas: int = 0,
) -> int:
    """Return launch-time token CTA extent for persistent/static early exit."""
    if explicit_early_exit_max_token_ctas < 0:
        raise ValueError(
            "early_exit_max_token_ctas must be non-negative, got "
            f"{explicit_early_exit_max_token_ctas}"
        )
    if not cfg_overrides.get("use_early_exit", 0):
        return 0
    if explicit_early_exit_max_token_ctas:
        return explicit_early_exit_max_token_ctas

    shape_overrides = dict(cfg_overrides)
    shape_overrides["use_early_exit"] = 0
    shape_cfg = make_config(**shape_overrides)
    token_tile_size = shape_cfg.tile_n if shape_cfg.is_swap_ab else shape_cfg.tile_m
    cluster_dim_in_token = 1 if shape_cfg.is_swap_ab else shape_cfg.cluster_m
    return compute_max_num_ctas_in_token_dim_for_moe(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        token_tile_size=token_tile_size,
        cluster_dim_in_token=cluster_dim_in_token,
    )


def compute_num_k_tiles(cfg: BatchedGemmConfig) -> int:
    """Compute the number of K-tiles for a given problem shape."""
    problem_k = cfg.tile_k
    padded_k = ((problem_k + cfg.tile_k - 1) // cfg.tile_k) * cfg.tile_k
    return padded_k // cfg.tile_k
