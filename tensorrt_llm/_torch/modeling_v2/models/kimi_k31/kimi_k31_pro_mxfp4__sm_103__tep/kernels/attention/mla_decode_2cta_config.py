# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
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

Based on
``tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_2cta/config.py``; unused code removed
(the never-read V-cache subtile byte count).

The throughput 2CTA policy uses a 2CTA M128 schedule. BF16 and FP8 inputs share
the same explicit configuration structure and output-reduction contract.
"""

from dataclasses import dataclass

from .mla_decode_helpers_math import MaskType, normalize_mask_type

# PV consumes V from SMEM in 32-token K blocks.  Physical KV pages may be
# smaller or larger, but TMA must assemble this fixed block geometry before
# tcgen05 advances the V descriptor to the next K block.
V_SMEM_K_BLOCK_TOKENS = 32

# Each transpose-TMA issue stages one 64-element slice of the latent dimension.
V_TMA_LATENT_ELEMENTS = 64


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
    cluster_shape_mnk: tuple[int, int, int] = (2, 1, 1)

    # MMA tile shapes.  QK runs over 128x128 K tiles, while PV writes the 512 V
    # head in two 256-column passes.
    mma_qk_tiler_mn: tuple[int, int] = (128, 128)
    mma_pv_tiler_mn: tuple[int, int] = (128, 256)
    mma_qk_tiler_k: int = 64  # = rope_dim
    mma_qk_tiler: tuple[int, int, int] = (128, 128, 64)
    mma_qk_rope_tiler: tuple[int, int, int] = (128, 128, 64)
    mma_pv_tiler: tuple[int, int, int] = (128, 256, 32)

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

    # Final quantized output can use a staged TMA store on padding warp 10.
    use_tma_output: bool = False

    # Pipeline stage counts for the captured schedule resources.  The combined
    # K/V stage count keeps enough delayed-V stages live for K-before-V overlap.
    sparse_offset_stages: int = 4
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
    compute_warp_ids: tuple[int, ...] = (0, 1, 2, 3)
    correction_warp_ids: tuple[int, ...] = (4, 5, 6, 7)
    mma_warp_id: int = 8
    load_tma_warp_id: int = 9
    load_k_warp_id: int = 9
    load_v_warp_id: int = 10
    load_num_warps: int = 1
    pv_mma_warp_id: int = 11
    empty_warp_ids: tuple[int, ...] = (11,)
    second_compute_warp_ids: tuple[int, ...] = ()
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
    cache_uniform_sparse_pages: bool = False
    # Causal is bottom-right aligned for speculative decode. Dense still masks
    # the ordinary per-batch KV tail at ``cache_seqs[batch]``.
    mask_type: str = MaskType.CAUSAL.value
    # Threshold skip correction in log2 units (binades); 0 disables it.  A positive value
    # freezes the running row max while a K tile raises it by at most this
    # amount.  It is a compile-time constant: a runtime scalar stays live in
    # every softmax iteration and costs registers under the 128-register cap.
    # With E4M3 P it also lowers the P scale from 448 to ``448 * 2**-T``;
    # BF16 P stays unscaled.  See ``max_softmax_rescale_threshold_log2`` for
    # the bounds.
    softmax_rescale_threshold_log2: float = 0.0
    lse_base_on_e: bool = False

    @property
    def tokens_per_k_tile(self) -> int:
        """Return the logical KV-token count covered by one QK tile."""

        return self.mma_qk_tiler[1]

    @property
    def tokens_per_k_cta(self) -> int:
        """Return the KV-token count owned by one CTA in the 2CTA cluster."""

        return self.tokens_per_k_tile // self.num_mma_ctas

    @property
    def tokens_per_v_tile(self) -> int:
        """Return the logical KV-token count covered by all PV K iterations."""

        return self.mma_pv_tiler[2] * self.iterations_pv_k

    def is_fp8_qkv(self) -> bool:
        """Return whether Q/K/V tensors use E4M3 data."""

        return self.qkv_dtype == "e4m3"


def make_mla_decode_config(
    mma_qk_tiler_mn: tuple[int, int] = (128, 128),
    mma_pv_tiler_mn: tuple[int, int] = (128, 256),
    rope_dim: int = 64,
    page_size: int = 32,
    qkv_dtype: str = "bf16",
    o_dtype: str = "bf16",
    max_active_clusters: int = 56,
    is_persistent: bool = True,
    is_var_seq: bool = False,
    is_var_split_kv: bool = False,
    mask_type: MaskType | str = MaskType.CAUSAL,
    sparse_gather_warps: int = 1,
    sparse_kv_stages: int = 0,
    cache_uniform_sparse_pages: bool = False,
    balance_sparse_registers: bool = False,
    fuse_output_quant: bool = False,
    softmax_rescale_threshold_log2: float = 0.0,
    lse_base_on_e: bool = False,
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
    # Range-checked by MlaDecodeTs and plan().
    cfg.softmax_rescale_threshold_log2 = softmax_rescale_threshold_log2
    cfg.lse_base_on_e = lse_base_on_e

    def _require_positive(name: str, value: int) -> None:
        """Validate that a named configuration value is positive."""

    def _require_divisible(
        dividend_name: str, dividend: int, divisor_name: str, divisor: int
    ) -> None:
        """Validate that one named configuration value divides another."""
        _require_positive(dividend_name, dividend)
        _require_positive(divisor_name, divisor)

    for idx, value in enumerate(mma_qk_tiler_mn):
        _require_positive(f"mma_qk_tiler_mn[{idx}]", value)
    for idx, value in enumerate(mma_pv_tiler_mn):
        _require_positive(f"mma_pv_tiler_mn[{idx}]", value)
    _require_positive("page_size", page_size)
    # Page size one is the sparse gather path: D512 (no RoPE) or D576 (RoPE is the
    # last 64 channels of each gathered row).

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
        if page_size == 1:
            # Independent QK/PV streams retain two whole K/V tiles each. A
            # third V tile costs 32 KiB per CTA without improving overlap in
            # the qualified sparse schedule. Keep four metadata slots so
            # index prefetch can run ahead of both streams. Two K slots let
            # QK advance while the previous V tile is consumed.
            cfg.load_k_stage, cfg.load_v_stage = (2, 2)
            cfg.use_fp8_dual_softmax_schedule = False
            cfg.second_compute_warp_ids = ()
            cfg.num_softmax_groups = 1
            cfg.load_k_warp_id = 12
            cfg.load_v_warp_id = 16
            cfg.load_num_warps = 4
            cfg.threads_per_cta = cfg.threads_per_warp * 20
            cfg.other_reg_num = 48

    if balance_sparse_registers:
        # D512's register DSMEM stores benefit from a larger loader budget;
        # D576's bulk copies retain more registers for softmax/correction.
        # Base splits claim 61,440 registers in the 640-thread CTA.
        cfg.softmax_reg_num = 144
        cfg.correction_reg_num = 144
        cfg.other_reg_num = 64

    # SETMAXNREG redistributes the CTA's initial allocation, rounded down
    # to eight registers/thread; the remainder of the SM file is not credit.

    # Derived MMA tilers. FP8 latent QK uses K=128 while the separate RoPE MMA
    # keeps K=64.
    if cfg.rope_dim > 0:
        cfg.mma_qk_tiler_k = cfg.rope_dim * (2 if qkv_dtype == "e4m3" else 1)
    _require_divisible("latent_dim", cfg.latent_dim, "mma_qk_tiler_k", cfg.mma_qk_tiler_k)
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
    _require_divisible("latent_dim", cfg.latent_dim, "mma_pv_tiler_mn[1]", mma_pv_tiler_mn[1])
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
    cfg.iterations_qk_latent_stages = cfg.iterations_qk_latent // cfg.kv_subtiles_per_stage
    cfg.iterations_qk_stages = cfg.iterations_qk_latent_stages + cfg.iterations_qk_rope
    cfg.iterations_pv_stages = (
        cfg.iterations_pv_k * cfg.iterations_pv_n // cfg.kv_subtiles_per_stage
    )

    # Page-offset tile sizes
    num_mma_ctas = cfg.cluster_shape_mnk[0]
    cfg.num_mma_ctas = num_mma_ctas
    _require_divisible("mma_qk_tiler_mn[0]", cfg.mma_qk_tiler[0], "num_mma_ctas", num_mma_ctas)
    _require_divisible("mma_qk_tiler_mn[1]", cfg.mma_qk_tiler[1], "num_mma_ctas", num_mma_ctas)
    _require_divisible("mma_pv_tiler_mn[0]", cfg.mma_pv_tiler[0], "num_mma_ctas", num_mma_ctas)
    _require_divisible("mma_pv_tiler_mn[1]", cfg.mma_pv_tiler[1], "num_mma_ctas", num_mma_ctas)
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
        (cfg.mma_qk_rope_tiler[0] // num_mma_ctas) * cfg.mma_qk_rope_tiler[2] * cfg.load_q_stage
    )
    k_latent_subtile_elems = cfg.mma_qk_tiler[1] // num_mma_ctas * cfg.mma_qk_tiler[2]
    k_rope_subtile_elems = cfg.mma_qk_rope_tiler[1] // num_mma_ctas * cfg.mma_qk_rope_tiler[2]
    if qkv_dtype == "e4m3":
        cfg.smem_k_stage_elems = (
            k_latent_subtile_elems * cfg.iterations_qk_latent
            + k_rope_subtile_elems * cfg.iterations_qk_rope
        )
        cfg.smem_kc_elems = cfg.smem_k_stage_elems * cfg.load_k_stage
    v_subtile_elems = cfg.mma_pv_tiler[1] // num_mma_ctas * cfg.mma_pv_tiler[2]
    cfg.smem_v_stage_elems = v_subtile_elems * cfg.iterations_pv_k * cfg.iterations_pv_n
    cfg.smem_vc_elems = cfg.smem_v_stage_elems * cfg.load_v_stage if qkv_dtype == "e4m3" else 0
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
        cfg.mma_qk_rope_tiler[0] // num_mma_ctas * cfg.mma_qk_rope_tiler[2] * cfg.qkv_dtype_bytes
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
    cfg.tma_kc_subtile_bytes = cfg.tma_copy_kc_bytes * cfg.kv_subtiles_per_stage
    cfg.tma_copy_k_tile_bytes = cfg.smem_k_stage_elems * cfg.qkv_dtype_bytes * num_mma_ctas
    cfg.tma_copy_v_tile_bytes = cfg.tma_copy_vc_bytes * cfg.iterations_pv_k * cfg.iterations_pv_n
    # TMEM sync barrier thread count: QK MMA + softmax + correction, plus
    # the FP8 PV-MMA warp when it participates in TMEM O production.
    cfg.tmem_sync_bar_threads = (
        cfg.threads_per_warp * (2 if cfg.use_fp8_split_mma_schedule else 1)
        + cfg.threads_per_warp * cfg.num_compute_warps
        + cfg.threads_per_warp * cfg.num_compute_warps
    )

    return cfg
