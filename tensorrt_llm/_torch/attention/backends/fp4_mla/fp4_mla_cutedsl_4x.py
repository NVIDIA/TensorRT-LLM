# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA decode, four-CTA-cluster (4x) variant of the CuTe DSL decode kernel.

Design: one QK and one score quantization per KV tile; a 256-row O accumulator per CTA pair with a
fixed denominator owner; the two pairs of a cluster split the KV tiles by parity and exchange P over
DSMEM; the 384 threads reuse the S2P warps for the O correction between owned tiles; PV completion is
relayed by the copy warp, separately from MMA issue. The kernel derives from the six-yield FULL-task
decode kernel (internal version tag x79 in the comments below) with its split machinery removed.
Removed: PREFIX/RESUME task modes and the SCHED1 cut table, the second epoch, the FP16 carry / meta /
header (and the carry TMA descriptor), the generation gate and the reset kernel, the cross-cluster flag
publish and CAS spin. Every cluster runs ONE FULL task: pair = blockIdx.z, tiles [0, total_tiles). The
mainloop (K loader, V loader, copy warp, issuer, ALU warps) is byte-identical to x79 modulo indentation.

FULL-only envelope (host-gated: full_validate_envelope here / validate_envelope in fp4_mla_4x_host.py;
the kernel has no batch bound of its own besides grid.z = runtime_l // 2):
  local heads = 128, output cols = 512, query_len_per_seq in SUPPORTED_QUERY_LENS = (2, 4, 6, 8),
  tokens = B * q (even, % q == 0), n_pairs = tokens // 2 <= FULL_ONLY_MAX_PAIRS = 46 (one wave),
  per-sequence valid_k <= SMEM_P4_RUNTIME_MAX_KV, page_size 128, BF16 output [tokens, 128, 512].
Entry: fp4_mla_4x_full_ctm = x79's ordered_v30_six_yield_ctm with the same parameter list minus
workspace_contract; accum_ptr / row_max_ptr / row_sum_ptr / requested_cut are accepted but unused
(the host passes one int32 scratch of HEADER_SCRATCH_WORDS words for the three pointers).
Correctness class vs the exact v30 reference: ROUNDING_ONLY (rsplit FP32 re-association of the
denominator); x79's APPROXIMATION-class FP16 carry is gone.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass as ctm
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.arch.nvvm_wrappers import inline_ptx as cute_inline_ptx
from cutlass.experimental import cuda as cuda_tma
from cutlass.experimental import primitives as prims

from tensorrt_llm._torch.attention.backends.fp4_mla.config import (
    resolve_fp4_mla_lazy_anchor_rebase_log2,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.cute_dsl_utils import (
    _compile_cutedsl as _compile_cutedsl,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.cutedsl_common import (
    CUDA_GRID_Z_MAX,
    FP4_MLA_P_GLOBAL_SCALE,
    LOG2_6,
    _as_shared_cta,
    _compute_smem_p4_lazy_rebase_scale,
    _derive_qk_scale_burst_kblocks,
    _fence_async_shared_cta,
    _lookup_physical_page,
    _mapa_shared_cluster,
    _mask_invalid_score,
    _mbarrier_arrive_release_cta_shared_cluster,
    _mma_block_scale_kind,
    _mma_scale_format,
    _mma_scale_vec_size_enum,
    _pack_e2m1x8,
    _pack_e2m1x8_from_f16x2,
    _pack_e4m3x4_natural,
    _qk_compact_sfb_cols_per_scale_chunk,
    _qk_sfb_smem_nchunks_per_cta,
    _rescale_smem_p4_resident_o_in_tmem,
    _scale_pack_e4m3x4_natural_from_f16x2,
    _scale_words_for_k,
    _sfa_tmem_cols_for_scale_chunks,
    _softmax_exp2_pair,
    _softmax_exp2_pair_packed_f16x2,
    _tcgen05_ld_red_32x32b_x16_max_f32,
    nvvm_fma_packed_f32x2,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.cutedsl_common import LOG2_E as LOG2_E

# FULL-only envelope (BRIEF 'Fair envelope'): host-gated, the kernel has no batch bound of its own
# besides grid.z = runtime_l // 2 (see full_validate_envelope below SMEM_P4_RUNTIME_MAX_KV).
SUPPORTED_QUERY_LENS = (2, 4, 6, 8)  # MTP1/3/5/7; one compiled variant per value (Constexpr)
FULL_ONLY_MAX_PAIRS = 46  # n_pairs = tokens // 2 <= 46: one wave, 2 SMs per token like 2x
FULL_QUERY_LENS = SUPPORTED_QUERY_LENS  # survey (E03) spelling
FULL_FAIR_MAX_PAIRS = FULL_ONLY_MAX_PAIRS  # survey (E03) spelling
# fp4_mla_4x_full_ctm keeps the accum/row_max/row_sum pointer parameters of the x79 ABI; nothing
# dereferences them (no carry, meta or header). The host passes one int32 scratch tensor of at
# least HEADER_SCRATCH_WORDS words for all three.
HEADER_SCRATCH_WORDS = 16


@cute.jit
def _store_async_cluster_b32(dst: ctm.Int32, value: ctm.Int32, mbar: ctm.Int32):
    # Public-DSL stand-in for the internal prims.store_async(dst, value, mbar): st.async of one b32 to a peer CTA's
    # shared memory with mbarrier complete_tx accounting, on the mapa'd (shared::cluster) 32-bit addresses.
    cute_inline_ptx(
        "{\n"
        "\t.reg .b64 sa_dst, sa_mbar;\n"
        "\tcvt.u64.u32 sa_dst, {$r0};\n"
        "\tcvt.u64.u32 sa_mbar, {$r2};\n"
        "\tst.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32 [sa_dst], {$r1}, [sa_mbar];\n"
        "\t}\n",
        read_only_args=[dst, value, mbar],
    )


M_TILE = 256
KV_TILE = 256
KV_TILES = 4
KV_TOTAL = KV_TILE * KV_TILES
OUT_DIM = 256
OUT_DIM_TOTAL = 512
SMEM_P4_N_OUT_TILES = OUT_DIM_TOTAL // OUT_DIM
QK_LOGICAL_DIM = 640
QK_PADDED_DIM = QK_LOGICAL_DIM
TRTLLM_K_HEAD_DIM = 576
TRTLLM_V_HEAD_DIM = 512
TRTLLM_Q_RESIDUAL_DIM = 64
TRTLLM_K_RESIDUAL_DIM = TRTLLM_Q_RESIDUAL_DIM
TRTLLM_K_STORAGE_DIM = TRTLLM_K_HEAD_DIM + TRTLLM_K_RESIDUAL_DIM
QK_EFFECTIVE_DIM = QK_LOGICAL_DIM + TRTLLM_K_RESIDUAL_DIM
TRTLLM_PAGE_SIZE = 128
TRTLLM_K_SF_GROUPS = TRTLLM_K_STORAGE_DIM // 16
TRTLLM_QSF_EVEN_PRMT_SELECTOR = 25632
TRTLLM_QSF_ODD_PRMT_SELECTOR = 30001
SF_VEC_SIZE_OPTIONS = (16,)
SF_VEC_SIZE = 16
MMA_KBLOCK_DIM = 64
MMA_SCALE_CHUNK_K = SF_VEC_SIZE * 4
MMA_SCALE_CHUNKS_PER_KBLOCK = _scale_words_for_k(MMA_KBLOCK_DIM, SF_VEC_SIZE)
MMA_SCALE_VEC_SIZE = _mma_scale_vec_size_enum(SF_VEC_SIZE)
MMA_SCALE_FORMAT = _mma_scale_format(SF_VEC_SIZE)
MMA_BLOCK_SCALE_KIND = _mma_block_scale_kind(SF_VEC_SIZE)
QK_MMA_KBLOCKS = QK_LOGICAL_DIM // MMA_KBLOCK_DIM
FIRST_ORDER_TAIL_A_LOCAL_KBLOCKS = (0, 1, 0)
FIRST_ORDER_TAIL_B_LOCAL_KBLOCKS = (0, 0, 1)
QK_SF_GROUPS = QK_PADDED_DIM // SF_VEC_SIZE
SMEM_P4_RUNTIME_MAX_KV = 160 * 1024
SMEM_P4_RUNTIME_MAX_PAGES = SMEM_P4_RUNTIME_MAX_KV // TRTLLM_PAGE_SIZE
SMEM_P4_PAGE_ID_PLAN_INTS = SMEM_P4_RUNTIME_MAX_PAGES
SMEM_P4_PAGE_ID_PLAN_BYTES = SMEM_P4_PAGE_ID_PLAN_INTS * ctm.Int32.bytes
SMEM_P4_RUNTIME_SCALE_PAIR_FLOATS = 2
SMEM_P4_RUNTIME_SCALE_PAIR_BYTES = SMEM_P4_RUNTIME_SCALE_PAIR_FLOATS * ctm.Float32.bytes
# Lazy-anchor rebase threshold (log2): a tile rescales the running O accumulator ("correction") only when the row
# maximum grew by more than this since the current anchor. Every rebase drains the PV pipeline of all warps (and of
# the whole 4-CTA cluster in the 4x kernel). Measured 2026-11-01 (DS-R1 shapes, logits scaled x1..x32): with 1.0,
# realistic logits (x4) cost the 4x kernel 18-19 % and the 2x kernel 2-6 %; 4.0 removes the stalls up to x8 logits and
# 6.0 up to ~x16 with FP32 cosine unchanged (the P scale saturates only above ~6.5 log2 at kv_gs 89.6; 8.0 breaks
# accuracy). Default 4.0; TRTLLM_FP4_MLA_LAZY_ANCHOR_LOG2 overrides it within [0, 8].
SMEM_P4_LAZY_ANCHOR_REBASE_LOG2 = resolve_fp4_mla_lazy_anchor_rebase_log2(4.0)


def full_validate_envelope(*, queries, heads, output_cols, query_len, max_kv_len):
    """Host gate for the FULL-only 4x variant (BRIEF 'Fair envelope'). CPU metadata only; the kernel
    launches tokens // 2 clusters and has no bound of its own besides grid.z."""
    if type(query_len) is not int or query_len not in SUPPORTED_QUERY_LENS:
        raise ValueError(
            "4x FULL-only requires query_len_per_seq in (2, 4, 6, 8); q=1 and odd q fall back to 2x"
        )
    if type(queries) is not int or queries <= 0 or queries % query_len:
        raise ValueError("queries (tokens) must be a positive multiple of query_len_per_seq")
    if (
        type(heads) is not int
        or heads <= 0
        or heads > 128
        or (heads * query_len) % 256 != 0
        or output_cols != 512
    ):
        raise ValueError(
            "FULL-only rows mode requires 1..128 local heads with heads * query_len % 256 == 0, and O512"
        )
    n_pairs = (
        queries * heads // 256
    )  # exact: heads * query_len % 256 == 0 and query_len divides queries
    if n_pairs > FULL_ONLY_MAX_PAIRS or n_pairs > CUDA_GRID_Z_MAX:
        raise ValueError(
            f"fair envelope: n_pairs={n_pairs} > {FULL_ONLY_MAX_PAIRS}; fall back to 2x"
        )
    if type(max_kv_len) is not int or not 1 <= max_kv_len <= SMEM_P4_RUNTIME_MAX_KV:
        raise ValueError(f"max_kv_len must be in [1, {SMEM_P4_RUNTIME_MAX_KV}]")
    return n_pairs


SMEM_P4_PIPELINE_DIRECT_TARGET_PAGE_SIZE = 128
RUBIN_TMEM_CAPACITY_KB = 288
TMEM_BYTES_PER_COL = 512
NUM_TMEM_ALLOC_COLS = RUBIN_TMEM_CAPACITY_KB * 1024 // TMEM_BYTES_PER_COL
SMEM_P4_BMM1_MNK = (64, KV_TILE, QK_EFFECTIVE_DIM)
SMEM_P4_BMM2_MNK = (64, OUT_DIM, KV_TILE)
SMEM_P4_TMEM_WARP_SHAPE_MN = (2, 2)
SMEM_P4_BMM1_M = SMEM_P4_BMM1_MNK[0]
SMEM_P4_BMM1_N = SMEM_P4_BMM1_MNK[1]
SMEM_P4_BMM2_M = SMEM_P4_BMM2_MNK[0]
SMEM_P4_BMM2_N = SMEM_P4_BMM2_MNK[1]
SMEM_P4_BMM2_K = SMEM_P4_BMM2_MNK[2]
SMEM_P4_TMEM_WARP_M = SMEM_P4_TMEM_WARP_SHAPE_MN[0]
SMEM_P4_TMEM_WARP_N = SMEM_P4_TMEM_WARP_SHAPE_MN[1]
SMEM_P4_TMEM_BANKS = 2
SMEM_P4_QK_SCORE_STAGES = 1
SMEM_P4_SCORE_SLOTS_PER_KV_TILE = 2
SMEM_P4_SCORE_SLOT_N = SMEM_P4_BMM1_N // SMEM_P4_SCORE_SLOTS_PER_KV_TILE
SMEM_P4_BMM2_STAGE_K = SMEM_P4_BMM2_K // SMEM_P4_SCORE_SLOTS_PER_KV_TILE
SMEM_P4_CTA_GROUP_M = SMEM_P4_BMM1_M * SMEM_P4_TMEM_WARP_M
SMEM_P4_PIPELINE_DIRECT_TARGET_M_OPTIONS = (SMEM_P4_CTA_GROUP_M, M_TILE)
SMEM_P4_SFB_ATOM_ROWS = 128
SMEM_P4_P_ARCHIVE_SLOTS = SMEM_P4_TMEM_BANKS
SMEM_P4_ROWMETA_SLOTS = SMEM_P4_TMEM_BANKS
SMEM_P4_QK_PIPELINE_SLOTS = SMEM_P4_TMEM_BANKS
SMEM_P4_BMM1_ACC_TMEM_COLS = SMEM_P4_BMM1_N // SMEM_P4_TMEM_WARP_N
SMEM_P4_SCORE_HALF_TMEM_COLS = SMEM_P4_BMM1_ACC_TMEM_COLS // SMEM_P4_SCORE_SLOTS_PER_KV_TILE
SMEM_P4_QK_SCORE_SLOT_N = SMEM_P4_BMM1_N
SMEM_P4_QK_SCORE_RING_SLOTS = SMEM_P4_QK_SCORE_STAGES
SMEM_P4_QK_PIPELINE_SLOT_STRIDE = SMEM_P4_QK_SCORE_SLOT_N // SMEM_P4_TMEM_WARP_N
SMEM_P4_QK_SCORE_SLOT_STRIDE = SMEM_P4_SCORE_HALF_TMEM_COLS
SMEM_P4_QK_MMA_SCALE_CHUNKS_PER_KBLOCK = MMA_SCALE_CHUNKS_PER_KBLOCK
SMEM_P4_QK_MMA_KBLOCKS = QK_MMA_KBLOCKS
SMEM_P4_P4_OWNER_SMEM_SF_MAILBOX_WORDS_PER_WARP = 32 * 2
SMEM_P4_P4_OWNER_SMEM_SF_MAILBOX_WORDS_PER_SLOT = (
    SMEM_P4_TMEM_WARP_M * SMEM_P4_TMEM_WARP_N * SMEM_P4_P4_OWNER_SMEM_SF_MAILBOX_WORDS_PER_WARP
)
SMEM_P4_P4_OWNER_SMEM_SF_MAILBOX_WORDS = (
    SMEM_P4_SCORE_SLOTS_PER_KV_TILE * SMEM_P4_P4_OWNER_SMEM_SF_MAILBOX_WORDS_PER_SLOT
)
SMEM_P4_P4_SMEM_ROW_STRIDE_BYTES = SMEM_P4_BMM2_STAGE_K // 2
SMEM_P4_P4_SMEM_ROWS = SMEM_P4_BMM2_M
SMEM_P4_P4_SMEM_STAGE_BYTES = SMEM_P4_P4_SMEM_ROWS * SMEM_P4_P4_SMEM_ROW_STRIDE_BYTES
SMEM_P4_P4_SMEM_STAGE_WORDS = SMEM_P4_P4_SMEM_STAGE_BYTES // 4
SMEM_P4_P4_CONTROL_MBAR_STAGES = SMEM_P4_TMEM_BANKS * SMEM_P4_SCORE_SLOTS_PER_KV_TILE
SMEM_P4_P4_SMEM_PHYSICAL_STAGES = SMEM_P4_TMEM_BANKS * SMEM_P4_SCORE_SLOTS_PER_KV_TILE
SMEM_P4_P4_SMEM_WORDS = SMEM_P4_P4_SMEM_PHYSICAL_STAGES * SMEM_P4_P4_SMEM_STAGE_WORDS
SMEM_P4_SCORE_SLOT_ROW_STATE_STRIDE = SMEM_P4_BMM1_M * SMEM_P4_TMEM_WARP_N
SMEM_P4_P_TMEM_KBLOCK_COLS = MMA_KBLOCK_DIM // 2 // 4
SMEM_P4_PV_KBLOCKS_PER_SCORE_SLOT = SMEM_P4_BMM2_STAGE_K // MMA_KBLOCK_DIM
SMEM_P4_PV_KBLOCKS_PER_LI = SMEM_P4_SCORE_SLOTS_PER_KV_TILE * SMEM_P4_PV_KBLOCKS_PER_SCORE_SLOT
SMEM_P4_P_TMEM_STAGE_COLS = SMEM_P4_PV_KBLOCKS_PER_SCORE_SLOT * SMEM_P4_P_TMEM_KBLOCK_COLS
SMEM_P4_SCORE_TMEM_COLS = (
    0,
    SMEM_P4_QK_SCORE_RING_SLOTS * SMEM_P4_QK_PIPELINE_SLOT_STRIDE,
)
SMEM_P4_SCORE_PONG_TMEM_COLS = (SMEM_P4_SCORE_TMEM_COLS[1], 256)
SMEM_P4_RESIDENT_Q_TMEM_COL_OFFSET = SMEM_P4_SCORE_TMEM_COLS[1]
SMEM_P4_RESIDENT_Q_TMEM_KBLOCK_COLS = SMEM_P4_P_TMEM_KBLOCK_COLS
SMEM_P4_RESIDENT_Q_TMEM_COLS = QK_MMA_KBLOCKS * SMEM_P4_RESIDENT_Q_TMEM_KBLOCK_COLS
SMEM_P4_RESIDENT_Q_TMEM_COL_END = SMEM_P4_RESIDENT_Q_TMEM_COL_OFFSET + SMEM_P4_RESIDENT_Q_TMEM_COLS
SMEM_P4_P_SFA_TMEM_COL_OFFSET = SMEM_P4_RESIDENT_Q_TMEM_COL_END
SMEM_P4_ROWSUM_MMA_K = MMA_KBLOCK_DIM
SMEM_P4_ROWSUM_ACC_TMEM_COL_OFFSET = 568
SMEM_P4_ROWSUM_ACC_TMEM_COLS = 8
SMEM_P4_ROWSUM_UNIT_SFB_TMEM_COL_OFFSET = 560
SMEM_P4_ROWSUM_UNIT_SFB_TMEM_COLS = 2
SMEM_P4_ROWSUM_UNIT_B_WORDS = 64
SMEM_P4_ROWSUM_UNIT_B_PAD_WORDS = 64
SMEM_P4_ROWSUM_UNIT_SFB_WORD_OFFSET = SMEM_P4_ROWSUM_UNIT_B_WORDS + SMEM_P4_ROWSUM_UNIT_B_PAD_WORDS
SMEM_P4_ROWSUM_UNIT_SFB_WORDS = 256
SMEM_P4_ROWSUM_STORAGE_WORDS = SMEM_P4_ROWSUM_UNIT_SFB_WORD_OFFSET + SMEM_P4_ROWSUM_UNIT_SFB_WORDS
SMEM_P4_O_ACC_TMEM_COLS = (
    SMEM_P4_SCORE_PONG_TMEM_COLS[1],
    SMEM_P4_SCORE_PONG_TMEM_COLS[1] + SMEM_P4_BMM2_N,
)
SMEM_P4_SFA_COLS_PER_SCALE_CHUNK = 1
SMEM_P4_P_SFA_COLS_PER_SCALE_CHUNK = 1
SMEM_P4_QK_SFB_WARPX2_MODE = 1
SMEM_P4_P4_SCORE_READ_GROUP1_COL = 16
SMEM_P4_P4_SCORE_READ_GROUP2_COL = 32
SMEM_P4_P_SCALE_FEED_STAGE_COLS = _sfa_tmem_cols_for_scale_chunks(
    SMEM_P4_BMM2_K // MMA_KBLOCK_DIM * MMA_SCALE_CHUNKS_PER_KBLOCK, True
)
SMEM_P4_P_SCALE_A_COL_OFFSET = SMEM_P4_SCORE_SLOTS_PER_KV_TILE * SMEM_P4_P_TMEM_STAGE_COLS
SMEM_P4_PV_SFB_SLOT_COL_OFFSET = SMEM_P4_P_SCALE_A_COL_OFFSET + SMEM_P4_P_SCALE_FEED_STAGE_COLS
SMEM_P4_PV_SFB_WARPX2_MODE = 1
SMEM_P4_PV_SFB_N_CHUNKS = (SMEM_P4_BMM2_N + 127) // 128
SMEM_P4_PV_COMPACT_SFB_COLS_PER_SCALE_CHUNK = _qk_compact_sfb_cols_per_scale_chunk(
    SMEM_P4_CTA_GROUP_M, SMEM_P4_BMM2_N, SMEM_P4_PV_SFB_WARPX2_MODE
)
SMEM_P4_PV_SFB_COLS_PER_KBLOCK = (
    MMA_SCALE_CHUNKS_PER_KBLOCK * SMEM_P4_PV_COMPACT_SFB_COLS_PER_SCALE_CHUNK
)
SMEM_P4_PV_SFB_SCORE_SLOT_COLS = SMEM_P4_PV_KBLOCKS_PER_SCORE_SLOT * SMEM_P4_PV_SFB_COLS_PER_KBLOCK
SMEM_P4_SCALE_FEED_STAGE_COLS = 28
SMEM_P4_SCALE_FEED_TMEM_COLS = (
    SMEM_P4_O_ACC_TMEM_COLS[1],
    SMEM_P4_O_ACC_TMEM_COLS[1] + SMEM_P4_SCORE_SLOTS_PER_KV_TILE * SMEM_P4_SCALE_FEED_STAGE_COLS,
)
SMEM_P4_QK_ACC_COL_OFFSET = SMEM_P4_SCORE_TMEM_COLS[0]
SMEM_P4_SCALE_A_COL_OFFSET = SMEM_P4_SCALE_FEED_TMEM_COLS[0]
SMEM_P4_QK_COMPACT_SFB_COLS_PER_SCALE_CHUNK = _qk_compact_sfb_cols_per_scale_chunk(
    SMEM_P4_CTA_GROUP_M, SMEM_P4_QK_SCORE_SLOT_N, SMEM_P4_QK_SFB_WARPX2_MODE
)
SMEM_P4_QK_SFB_COLS_PER_KBLOCK = (
    SMEM_P4_QK_MMA_SCALE_CHUNKS_PER_KBLOCK * SMEM_P4_QK_COMPACT_SFB_COLS_PER_SCALE_CHUNK
)
SMEM_P4_QK_RESIDENT_SF_CHUNKS = _scale_words_for_k(QK_PADDED_DIM, SF_VEC_SIZE)
SMEM_P4_QK_RESIDENT_SF_COLS = SMEM_P4_QK_RESIDENT_SF_CHUNKS
SMEM_P4_QK_SCALE_BURST_KBLOCKS = _derive_qk_scale_burst_kblocks(
    qk_mma_kblocks=QK_MMA_KBLOCKS,
    qk_scale_burst_target_kblocks=QK_MMA_KBLOCKS,
    scale_a_col_offset=SMEM_P4_SCALE_A_COL_OFFSET,
    num_tmem_alloc_cols=NUM_TMEM_ALLOC_COLS,
    qk_mma_m_dim=SMEM_P4_CTA_GROUP_M,
    mma_scale_chunks_per_kblock=SMEM_P4_QK_MMA_SCALE_CHUNKS_PER_KBLOCK,
    use_sfa_128dp_unique=True,
    qk_score_slot_n=SMEM_P4_QK_SCORE_SLOT_N,
    qk_compact_sfb_cols_per_scale_chunk=SMEM_P4_QK_COMPACT_SFB_COLS_PER_SCALE_CHUNK,
)
SMEM_P4_QK_SFA_FEED_COLS = SMEM_P4_QK_RESIDENT_SF_COLS
SMEM_P4_QK_SCALE_B_COL_OFFSET = (
    SMEM_P4_P_SFA_TMEM_COL_OFFSET + SMEM_P4_TMEM_BANKS * SMEM_P4_P_SCALE_FEED_STAGE_COLS
)
SMEM_P4_PV_VSF_TAIL_COL_OFFSET = (
    (SMEM_P4_SCALE_A_COL_OFFSET + SMEM_P4_QK_SFA_FEED_COLS + 15) // 16 * 16
)
SMEM_P4_RESIDENT_Q_SF_COL_OFFSET = SMEM_P4_SCALE_FEED_TMEM_COLS[0]
SMEM_P4_OLD_SCALE_TMEM_SLOTS = SMEM_P4_ROWMETA_SLOTS
SMEM_P4_STAGE_ROWSUM_TMEM_COLS = SMEM_P4_TMEM_BANKS * SMEM_P4_SCORE_SLOTS_PER_KV_TILE
SMEM_P4_STAGE_ROW_STATE_SMEM_FLOATS = (
    SMEM_P4_TMEM_BANKS * SMEM_P4_SCORE_SLOTS_PER_KV_TILE * SMEM_P4_SCORE_SLOT_ROW_STATE_STRIDE
)
SMEM_P4_ROW_SCALE_RING_FLOATS = SMEM_P4_TMEM_BANKS * SMEM_P4_BMM1_M
SMEM_P4_FINAL_ANCHOR_ROWSUM_FLOATS = SMEM_P4_BMM1_M
SMEM_P4_ATOMIC_RUNNING_ROWMAX_SMEM_ROWS = SMEM_P4_BMM1_M
SMEM_P4_TMEM_SCORE_PIPELINE_STAGES = SMEM_P4_QK_SCORE_STAGES
SMEM_P4_MBAR_PARITY_PHASES = 2
SMEM_P4_PAIR_CORR_MBAR_OFFSET = 0
SMEM_P4_PAIR_P4_READY_MBAR_OFFSET = (
    SMEM_P4_PAIR_CORR_MBAR_OFFSET + SMEM_P4_N_OUT_TILES * SMEM_P4_QK_PIPELINE_SLOTS
)
SMEM_P4_PAIR_ROWSUM_DONE_MBAR_OFFSET = SMEM_P4_PAIR_P4_READY_MBAR_OFFSET + SMEM_P4_QK_PIPELINE_SLOTS
SMEM_P4_STAGE_ROWSUM_RING_SLOTS = SMEM_P4_STAGE_ROWSUM_TMEM_COLS
SMEM_P4_P_HALF_COUNT = SMEM_P4_PV_KBLOCKS_PER_LI // SMEM_P4_PV_KBLOCKS_PER_SCORE_SLOT
SMEM_P4_PAIR_P_SOURCE_READY_MBAR_OFFSET = (
    SMEM_P4_PAIR_ROWSUM_DONE_MBAR_OFFSET + SMEM_P4_STAGE_ROWSUM_RING_SLOTS
)
SMEM_P4_PAIR_P_HALF_READY_MBAR_OFFSET = (
    SMEM_P4_PAIR_P_SOURCE_READY_MBAR_OFFSET + SMEM_P4_P_HALF_COUNT * SMEM_P4_QK_PIPELINE_SLOTS
)
SMEM_P4_PAIR_HALF1_ROWSUM_READY_MBAR_OFFSET = SMEM_P4_PAIR_P_HALF_READY_MBAR_OFFSET
SMEM_P4_PAIR_ROWMAX_READ_DONE_MBAR_OFFSET = (
    SMEM_P4_PAIR_P_HALF_READY_MBAR_OFFSET + SMEM_P4_P_HALF_COUNT * SMEM_P4_QK_PIPELINE_SLOTS
)
SMEM_P4_PAIR_DIRECT_P_READY_MBAR_OFFSET = SMEM_P4_PAIR_ROWSUM_DONE_MBAR_OFFSET
SMEM_P4_PAIR_OVERLAP_MBAR_COUNT = (
    SMEM_P4_PAIR_ROWMAX_READ_DONE_MBAR_OFFSET + SMEM_P4_QK_PIPELINE_SLOTS
)
SMEM_P4_V_PIPELINE_STAGES = 3
if (
    SMEM_P4_P_SFA_TMEM_COL_OFFSET + SMEM_P4_TMEM_BANKS * SMEM_P4_P_SCALE_FEED_STAGE_COLS
    > SMEM_P4_SCORE_PONG_TMEM_COLS[1]
):
    raise ValueError("resident Q and two-stage P_SFA must fit in the freed score-bank region")
if SMEM_P4_ROWSUM_ACC_TMEM_COL_OFFSET + SMEM_P4_ROWSUM_ACC_TMEM_COLS != NUM_TMEM_ALLOC_COLS:
    raise ValueError("FP4 rowsum accumulator must occupy the final eight TMEM columns")
if SMEM_P4_ROWSUM_UNIT_SFB_TMEM_COL_OFFSET % 16 != 0:
    raise ValueError("unit rowsum SFB TMEM image must be 16-column aligned")
if SMEM_P4_ROWSUM_UNIT_SFB_TMEM_COL_OFFSET + 4 > SMEM_P4_ROWSUM_ACC_TMEM_COL_OFFSET:
    raise ValueError("physical unit rowsum SFB image must fit below the rowsum accumulator")


SMEM_P4_PRODUCER_WARPS_PER_SCORE_SLOT = SMEM_P4_TMEM_WARP_M * SMEM_P4_TMEM_WARP_N
SMEM_P4_COMPUTE_WARPS = SMEM_P4_PRODUCER_WARPS_PER_SCORE_SLOT
SMEM_P4_CORRECTION_WARPS = SMEM_P4_PRODUCER_WARPS_PER_SCORE_SLOT
SMEM_P4_TOTAL_PRODUCER_WARPS = SMEM_P4_COMPUTE_WARPS + SMEM_P4_CORRECTION_WARPS
SMEM_P4_COMPUTE_THREADS = 32 * SMEM_P4_COMPUTE_WARPS
SMEM_P4_CORRECTION_THREADS = 32 * SMEM_P4_CORRECTION_WARPS
SMEM_P4_SCORE_SLOT_PRODUCER_THREADS = SMEM_P4_COMPUTE_THREADS
SMEM_P4_TMEM_ALLOC_EXCLUSIVE = True
MMA_TILER_MNK = (SMEM_P4_CTA_GROUP_M, SMEM_P4_BMM2_N, SMEM_P4_BMM2_K)
QK_DATA_STAGE_K_DIM = MMA_TILER_MNK[2]
QK_TAIL_DATA_STAGE_K_DIM = QK_LOGICAL_DIM % QK_DATA_STAGE_K_DIM
SMEM_P4_QK_FULL_DATA_STAGES = QK_LOGICAL_DIM // QK_DATA_STAGE_K_DIM
SMEM_P4_QK_TAIL_STAGE_KBLOCK_START = SMEM_P4_QK_FULL_DATA_STAGES * (
    QK_DATA_STAGE_K_DIM // MMA_KBLOCK_DIM
)
QK_K_TAIL_DATA_STAGE_K_DIM = (
    TRTLLM_K_STORAGE_DIM - SMEM_P4_QK_TAIL_STAGE_KBLOCK_START * MMA_KBLOCK_DIM
)
MMA_TILER_MNK_PER_CTA = (SMEM_P4_BMM1_M, SMEM_P4_BMM2_N, SMEM_P4_BMM2_K)
SMEM_P4_QK_DATA_STAGE_KBLOCKS = QK_DATA_STAGE_K_DIM // MMA_KBLOCK_DIM
SMEM_P4_QK_DATA_STAGES = (
    QK_MMA_KBLOCKS + SMEM_P4_QK_DATA_STAGE_KBLOCKS - 1
) // SMEM_P4_QK_DATA_STAGE_KBLOCKS
SMEM_P4_QK_SMEM_PIPELINE_SLOTS = 3
SMEM_P4_QK_BARRIER_STAGES = SMEM_P4_QK_SMEM_PIPELINE_SLOTS
SMEM_P4_QK_KTILE_RING_STAGES = SMEM_P4_QK_SMEM_PIPELINE_SLOTS
SMEM_P4_QK_BULK_READY_MBARS = SMEM_P4_QK_SMEM_PIPELINE_SLOTS
CLUSTER_SHAPE_MNK = (2, 1, 1)
MMA_SCALE_CHUNKS_PER_TILE = _scale_words_for_k(MMA_TILER_MNK[2], SF_VEC_SIZE)
SMEM_P4_V_N_PER_CTA = SMEM_P4_BMM2_N // CLUSTER_SHAPE_MNK[0]
NUM_QK_Q_STAGE = SMEM_P4_QK_DATA_STAGES
NUM_QK_Q_PIPELINE_STAGE = SMEM_P4_QK_DATA_STAGES
NUM_QK_AB_STAGE = SMEM_P4_QK_DATA_STAGES
B_STAGE_BYTES = SMEM_P4_V_N_PER_CTA * MMA_TILER_MNK[2] // 2
SFA_TMA_STAGE_BYTES = 512 * MMA_SCALE_CHUNKS_PER_TILE
SMEM_P4_QK_N_PER_CTA = SMEM_P4_QK_SCORE_SLOT_N // CLUSTER_SHAPE_MNK[0]
QK_A_STAGE_BYTES = MMA_TILER_MNK_PER_CTA[0] * QK_DATA_STAGE_K_DIM // 2
QK_B_STAGE_BYTES = SMEM_P4_QK_N_PER_CTA * QK_DATA_STAGE_K_DIM // 2
QK_TAIL_A_STAGE_BYTES = MMA_TILER_MNK_PER_CTA[0] * QK_TAIL_DATA_STAGE_K_DIM // 2
QK_TAIL_B_STAGE_BYTES = SMEM_P4_QK_N_PER_CTA * QK_K_TAIL_DATA_STAGE_K_DIM // 2
SMEM_P4_QK_DATA_STAGE_SF_CHUNKS = _scale_words_for_k(QK_DATA_STAGE_K_DIM, min(SF_VEC_SIZE_OPTIONS))
QK_SFA_STAGE_BYTES = 512 * SMEM_P4_QK_DATA_STAGE_SF_CHUNKS
SMEM_P4_QK_SFB_N_CHUNKS = _qk_sfb_smem_nchunks_per_cta(SMEM_P4_CTA_GROUP_M, SMEM_P4_QK_SCORE_SLOT_N)
QK_SFB_STAGE_BYTES = QK_SFA_STAGE_BYTES * SMEM_P4_QK_SFB_N_CHUNKS
QK_SFA_TMA_STAGE_BYTES = QK_SFA_STAGE_BYTES
QK_SFB_TMA_STAGE_BYTES = QK_SFB_STAGE_BYTES
SMEM_P4_QK_TAIL_DATA_STAGE_SF_CHUNKS = _scale_words_for_k(
    QK_TAIL_DATA_STAGE_K_DIM, min(SF_VEC_SIZE_OPTIONS)
)
SMEM_P4_QK_K_TAIL_DATA_STAGE_SF_CHUNKS = _scale_words_for_k(
    QK_K_TAIL_DATA_STAGE_K_DIM, min(SF_VEC_SIZE_OPTIONS)
)
QK_TAIL_SFA_STAGE_BYTES = 512 * SMEM_P4_QK_TAIL_DATA_STAGE_SF_CHUNKS
QK_TAIL_SFB_STAGE_BYTES = 512 * SMEM_P4_QK_K_TAIL_DATA_STAGE_SF_CHUNKS * SMEM_P4_QK_SFB_N_CHUNKS
QK_FULL_TMA_SWIZZLE = cuda_tma.TensorMapSwizzle.s128b
QK_TAIL_TMA_SWIZZLE = cuda_tma.TensorMapSwizzle.s64b
QK_K_TAIL_TMA_SWIZZLE = cuda_tma.TensorMapSwizzle.s64b
SMEM_P4_QK_TMA_CHUNKS_PER_CTA = (
    SMEM_P4_QK_N_PER_CTA + SMEM_P4_PIPELINE_DIRECT_TARGET_PAGE_SIZE - 1
) // SMEM_P4_PIPELINE_DIRECT_TARGET_PAGE_SIZE
SMEM_P4_QK_TMA_N_PER_CHUNK = (
    SMEM_P4_QK_N_PER_CTA + SMEM_P4_QK_TMA_CHUNKS_PER_CTA - 1
) // SMEM_P4_QK_TMA_CHUNKS_PER_CTA
SMEM_P4_PAGES_PER_KV_TILE = KV_TILE // TRTLLM_PAGE_SIZE
SMEM_P4_SCALE_CHUNKS_PER_PAGE = TRTLLM_PAGE_SIZE // MMA_SCALE_CHUNK_K
SMEM_P4_Q_ONLY_STAGE_BYTES = QK_A_STAGE_BYTES + QK_SFA_TMA_STAGE_BYTES
SMEM_P4_QK_KONLY_STAGE_BYTES = QK_B_STAGE_BYTES + QK_SFB_TMA_STAGE_BYTES
SMEM_P4_Q_ONLY_TAIL_STAGE_BYTES = QK_TAIL_A_STAGE_BYTES + QK_TAIL_SFA_STAGE_BYTES
SMEM_P4_QK_KONLY_TAIL_STAGE_BYTES = QK_TAIL_B_STAGE_BYTES + QK_TAIL_SFB_STAGE_BYTES
SMEM_P4_QK_KONLY_TILE_BYTES = (
    SMEM_P4_QK_FULL_DATA_STAGES * SMEM_P4_QK_KONLY_STAGE_BYTES + SMEM_P4_QK_KONLY_TAIL_STAGE_BYTES
)
SMEM_P4_Q_COMPACT_TILE_BYTES = (
    SMEM_P4_QK_FULL_DATA_STAGES * QK_A_STAGE_BYTES + QK_TAIL_A_STAGE_BYTES
)
SMEM_P4_QSF_COMPACT_TILE_BYTES = (
    SMEM_P4_QK_FULL_DATA_STAGES * QK_SFA_STAGE_BYTES + QK_TAIL_SFA_STAGE_BYTES
)
SMEM_P4_K_COMPACT_SLOT_BYTES = (
    SMEM_P4_QK_FULL_DATA_STAGES * QK_B_STAGE_BYTES + QK_TAIL_B_STAGE_BYTES
)
SMEM_P4_KSF_COMPACT_SLOT_BYTES = (
    SMEM_P4_QK_FULL_DATA_STAGES * QK_SFB_STAGE_BYTES + QK_TAIL_SFB_STAGE_BYTES
)
SMEM_P4_K_COMPACT_RING_BYTES = SMEM_P4_QK_KTILE_RING_STAGES * SMEM_P4_K_COMPACT_SLOT_BYTES
SMEM_P4_KSF_COMPACT_RING_BYTES = SMEM_P4_QK_KTILE_RING_STAGES * SMEM_P4_KSF_COMPACT_SLOT_BYTES
SMEM_P4_V_DATA_STAGE_BYTES = B_STAGE_BYTES * SMEM_P4_N_OUT_TILES
SMEM_P4_V_SFB_TMA_STAGE_BYTES = SFA_TMA_STAGE_BYTES * SMEM_P4_PV_SFB_N_CHUNKS * SMEM_P4_N_OUT_TILES
SMEM_P4_V_TILE_TMA_BYTES = SMEM_P4_V_DATA_STAGE_BYTES + SMEM_P4_V_SFB_TMA_STAGE_BYTES
SMEM_P4_CONTROL_MBAR_COUNT = (
    SMEM_P4_QK_PIPELINE_SLOTS
    + 1
    + NUM_QK_Q_PIPELINE_STAGE * 2
    + SMEM_P4_QK_BARRIER_STAGES * 2
    + SMEM_P4_V_PIPELINE_STAGES * 2
    + SMEM_P4_TMEM_SCORE_PIPELINE_STAGES * 2
    + SMEM_P4_P4_CONTROL_MBAR_STAGES * 2
    + SMEM_P4_PAIR_OVERLAP_MBAR_COUNT
    + SMEM_P4_QK_SCORE_RING_SLOTS
    + 2
    + SMEM_P4_QK_BULK_READY_MBARS
)
SMEM_P4_CONTROL_BYTES_RAW = SMEM_P4_CONTROL_MBAR_COUNT * 8 + 8
SMEM_P4_CONTROL_AND_ALIGNMENT_BYTES = (SMEM_P4_CONTROL_BYTES_RAW + 127) // 128 * 128


def _refresh_sf_vec_config(sf_vec_size: int) -> None:
    if sf_vec_size not in SF_VEC_SIZE_OPTIONS:
        raise ValueError(f"sf_vec_size must be one of {SF_VEC_SIZE_OPTIONS}, got {sf_vec_size}")
    if QK_LOGICAL_DIM % sf_vec_size != 0:
        raise ValueError(
            f"QK_LOGICAL_DIM={QK_LOGICAL_DIM} must be divisible by sf_vec_size={sf_vec_size}"
        )
    if QK_LOGICAL_DIM % MMA_KBLOCK_DIM != 0:
        raise ValueError(
            f"QK_LOGICAL_DIM={QK_LOGICAL_DIM} must be divisible by MMA_KBLOCK_DIM={MMA_KBLOCK_DIM}"
        )
    if QK_PADDED_DIM < QK_LOGICAL_DIM or QK_PADDED_DIM % MMA_KBLOCK_DIM != 0:
        raise ValueError(
            "QK_PADDED_DIM must cover QK_LOGICAL_DIM and be a whole Kblock; "
            f"got padded={QK_PADDED_DIM}, logical={QK_LOGICAL_DIM}, "
            f"kblock={MMA_KBLOCK_DIM}"
        )
    if QK_DATA_STAGE_K_DIM != MMA_TILER_MNK[2]:
        raise ValueError(
            "QK_DATA_STAGE_K_DIM must remain K=256 for the FP4 s128b full-stage "
            f"descriptor contract; got {QK_DATA_STAGE_K_DIM}"
        )
    if QK_TAIL_DATA_STAGE_K_DIM != 2 * MMA_KBLOCK_DIM:
        raise ValueError(
            f"QK_TAIL_DATA_STAGE_K_DIM must remain K=128 for the logical K640 tail; got {QK_TAIL_DATA_STAGE_K_DIM}"
        )
    if QK_K_TAIL_DATA_STAGE_K_DIM != 2 * MMA_KBLOCK_DIM:
        raise ValueError(
            "QK_K_TAIL_DATA_STAGE_K_DIM must remain physical K=128 for the "
            f"TensorRT-LLM K640 tail; got {QK_K_TAIL_DATA_STAGE_K_DIM}"
        )
    qk_data_stage_kblocks = QK_DATA_STAGE_K_DIM // MMA_KBLOCK_DIM
    qk_mma_kblocks = QK_LOGICAL_DIM // MMA_KBLOCK_DIM
    qk_data_stages = (qk_mma_kblocks + qk_data_stage_kblocks - 1) // qk_data_stage_kblocks
    qk_ktile_ring_data_stages = SMEM_P4_QK_KTILE_RING_STAGES * qk_data_stages
    qk_tail_stage_end = (
        SMEM_P4_QK_TAIL_STAGE_KBLOCK_START * MMA_KBLOCK_DIM + QK_TAIL_DATA_STAGE_K_DIM
    )
    if qk_tail_stage_end != QK_LOGICAL_DIM:
        raise ValueError(
            "QK full-stage plus tail split must cover QK_LOGICAL_DIM exactly; "
            f"got tail_stage_end={qk_tail_stage_end}, logical={QK_LOGICAL_DIM}"
        )
    qk_k_tail_stage_end = (
        SMEM_P4_QK_TAIL_STAGE_KBLOCK_START * MMA_KBLOCK_DIM + QK_K_TAIL_DATA_STAGE_K_DIM
    )
    if qk_k_tail_stage_end != TRTLLM_K_STORAGE_DIM:
        raise ValueError(
            "QK K full-stage plus physical tail split must cover K640 exactly; "
            f"got tail_stage_end={qk_k_tail_stage_end}, physical={TRTLLM_K_STORAGE_DIM}"
        )
    if QK_PADDED_DIM // 2 % 64 != 0:
        raise ValueError(
            f"QK padded FP4 row stride must be 64B aligned for the K128 tail; got {QK_PADDED_DIM // 2} bytes"
        )
    if TRTLLM_K_STORAGE_DIM // 2 % 64 != 0:
        raise ValueError(
            "TensorRT-LLM K640 packed row stride must be 64B aligned for the "
            f"s64b K128 tail; got {TRTLLM_K_STORAGE_DIM // 2} bytes"
        )
    if KV_TILE % sf_vec_size != 0:
        raise ValueError(f"KV_TILE={KV_TILE} must be divisible by sf_vec_size={sf_vec_size}")
    if SMEM_P4_SCORE_SLOT_N % sf_vec_size != 0:
        raise ValueError(
            "SMEM_P4 score slot must divide the FP4 scale-vector size; "
            f"got N={SMEM_P4_SCORE_SLOT_N}, sf_vec_size={sf_vec_size}"
        )
    mma_scale_chunk_k = sf_vec_size * 4
    mma_scale_chunks_per_kblock = _scale_words_for_k(MMA_KBLOCK_DIM, sf_vec_size)
    mma_scale_chunks_per_tile = _scale_words_for_k(MMA_TILER_MNK[2], sf_vec_size)
    qk_resident_sf_chunks = _scale_words_for_k(QK_PADDED_DIM, sf_vec_size)
    qk_resident_sf_cols = qk_resident_sf_chunks
    qk_data_stage_sf_chunks = _scale_words_for_k(QK_DATA_STAGE_K_DIM, sf_vec_size)
    qk_tail_data_stage_sf_chunks = _scale_words_for_k(QK_TAIL_DATA_STAGE_K_DIM, sf_vec_size)
    qk_k_tail_data_stage_sf_chunks = _scale_words_for_k(QK_K_TAIL_DATA_STAGE_K_DIM, sf_vec_size)
    sfa_tma_stage_bytes = 512 * mma_scale_chunks_per_tile
    sfb_tma_stage_bytes = sfa_tma_stage_bytes * 2
    qk_sfa_stage_bytes = 512 * qk_data_stage_sf_chunks
    qk_sfa_tma_stage_bytes = qk_sfa_stage_bytes
    qk_sfb_n_chunks = _qk_sfb_smem_nchunks_per_cta(SMEM_P4_CTA_GROUP_M, SMEM_P4_QK_SCORE_SLOT_N)
    qk_sfb_stage_bytes = qk_sfa_stage_bytes * qk_sfb_n_chunks
    qk_sfb_tma_stage_bytes = qk_sfb_stage_bytes
    pv_sfb_n_chunks = _qk_sfb_smem_nchunks_per_cta(SMEM_P4_CTA_GROUP_M, SMEM_P4_BMM2_N)
    pv_compact_sfb_cols_per_scale_chunk = _qk_compact_sfb_cols_per_scale_chunk(
        SMEM_P4_CTA_GROUP_M, SMEM_P4_BMM2_N, SMEM_P4_PV_SFB_WARPX2_MODE
    )
    pv_sfb_cols_per_kblock = mma_scale_chunks_per_kblock * pv_compact_sfb_cols_per_scale_chunk
    pv_sfb_score_slot_cols = SMEM_P4_PV_KBLOCKS_PER_SCORE_SLOT * pv_sfb_cols_per_kblock
    p_scale_feed_stage_cols = _sfa_tmem_cols_for_scale_chunks(
        SMEM_P4_BMM2_K // MMA_KBLOCK_DIM * mma_scale_chunks_per_kblock, True
    )
    pv_sfb_score_col_offset = (
        SMEM_P4_P_SCALE_A_COL_OFFSET + SMEM_P4_SCORE_SLOTS_PER_KV_TILE * p_scale_feed_stage_cols
    )
    qk_compact_sfb_cols_per_scale_chunk = _qk_compact_sfb_cols_per_scale_chunk(
        SMEM_P4_CTA_GROUP_M, SMEM_P4_QK_SCORE_SLOT_N, SMEM_P4_QK_SFB_WARPX2_MODE
    )
    qk_sfb_cols_per_kblock = mma_scale_chunks_per_kblock * qk_compact_sfb_cols_per_scale_chunk
    qk_scale_burst_kblocks = _derive_qk_scale_burst_kblocks(
        qk_mma_kblocks=qk_mma_kblocks,
        qk_scale_burst_target_kblocks=qk_mma_kblocks,
        scale_a_col_offset=SMEM_P4_SCALE_A_COL_OFFSET,
        num_tmem_alloc_cols=NUM_TMEM_ALLOC_COLS,
        qk_mma_m_dim=SMEM_P4_CTA_GROUP_M,
        mma_scale_chunks_per_kblock=mma_scale_chunks_per_kblock,
        use_sfa_128dp_unique=True,
        qk_score_slot_n=SMEM_P4_QK_SCORE_SLOT_N,
        qk_compact_sfb_cols_per_scale_chunk=qk_compact_sfb_cols_per_scale_chunk,
    )
    qk_sfa_feed_cols = qk_resident_sf_cols
    qk_sfb_feed_cols = qk_scale_burst_kblocks * qk_sfb_cols_per_kblock
    qk_scale_b_col_offset = (
        SMEM_P4_P_SFA_TMEM_COL_OFFSET + SMEM_P4_TMEM_BANKS * p_scale_feed_stage_cols
    )
    qk_scale_b_col_end = qk_scale_b_col_offset + qk_sfb_feed_cols
    if qk_scale_b_col_end > SMEM_P4_SCORE_PONG_TMEM_COLS[1]:
        raise ValueError(
            "QK SFB must fit after resident Q and P-SFA in the freed score-bank region; "
            f"got end={qk_scale_b_col_end}, limit={SMEM_P4_SCORE_PONG_TMEM_COLS[1]}"
        )
    qk_scale_feed_col_end = max(
        SMEM_P4_SCALE_A_COL_OFFSET + qk_sfa_feed_cols,
        qk_scale_b_col_end,
    )
    pv_vsf_tail_col_offset = (SMEM_P4_SCALE_A_COL_OFFSET + qk_sfa_feed_cols + 15) // 16 * 16
    p_sfa_tail_col_offset = pv_vsf_tail_col_offset - p_scale_feed_stage_cols
    pv_vsf_tail_col_end = (
        pv_vsf_tail_col_offset
        + SMEM_P4_N_OUT_TILES * SMEM_P4_SCORE_SLOTS_PER_KV_TILE * pv_sfb_score_slot_cols
    )
    if pv_vsf_tail_col_end > SMEM_P4_ROWSUM_UNIT_SFB_TMEM_COL_OFFSET:
        raise ValueError("PV VSF tail must fit below the unit rowsum SFB image")
    running_rowmeta_tmem_cols = 2 + SMEM_P4_OLD_SCALE_TMEM_SLOTS
    stage_rowmax_tmem_cols = 0
    stage_rowsum_tmem_cols = SMEM_P4_TMEM_BANKS * SMEM_P4_SCORE_SLOTS_PER_KV_TILE
    rowmeta_tmem_cols = stage_rowmax_tmem_cols + stage_rowsum_tmem_cols + running_rowmeta_tmem_cols
    rowmeta_tmem_col_offset = SMEM_P4_SCALE_FEED_TMEM_COLS[1]
    q_compact_tile_bytes = SMEM_P4_QK_FULL_DATA_STAGES * QK_A_STAGE_BYTES + QK_TAIL_A_STAGE_BYTES
    qsf_compact_tile_bytes = (
        SMEM_P4_QK_FULL_DATA_STAGES * qk_sfa_stage_bytes + 512 * qk_tail_data_stage_sf_chunks
    )
    k_compact_slot_bytes = SMEM_P4_QK_FULL_DATA_STAGES * QK_B_STAGE_BYTES + QK_TAIL_B_STAGE_BYTES
    ksf_compact_slot_bytes = (
        SMEM_P4_QK_FULL_DATA_STAGES * qk_sfb_stage_bytes
        + 512 * qk_k_tail_data_stage_sf_chunks * qk_sfb_n_chunks
    )
    static_smem_bytes = (
        SMEM_P4_CONTROL_AND_ALIGNMENT_BYTES
        + q_compact_tile_bytes
        + SMEM_P4_QK_KTILE_RING_STAGES * k_compact_slot_bytes
        + qsf_compact_tile_bytes
        + SMEM_P4_QK_KTILE_RING_STAGES * ksf_compact_slot_bytes
        + SMEM_P4_P4_SMEM_WORDS * 4
        + SMEM_P4_V_PIPELINE_STAGES * B_STAGE_BYTES * SMEM_P4_N_OUT_TILES
        + SMEM_P4_P4_OWNER_SMEM_SF_MAILBOX_WORDS * 4
        + SMEM_P4_V_PIPELINE_STAGES * sfa_tma_stage_bytes * pv_sfb_n_chunks * SMEM_P4_N_OUT_TILES
        + SMEM_P4_STAGE_ROW_STATE_SMEM_FLOATS * 4
        + SMEM_P4_ROW_SCALE_RING_FLOATS * 4
        + SMEM_P4_FINAL_ANCHOR_ROWSUM_FLOATS * 4
        + SMEM_P4_ATOMIC_RUNNING_ROWMAX_SMEM_ROWS * 4
        + SMEM_P4_PAGE_ID_PLAN_BYTES
        + SMEM_P4_RUNTIME_SCALE_PAIR_BYTES
    )
    globals().update(
        SF_VEC_SIZE=sf_vec_size,
        MMA_SCALE_CHUNK_K=mma_scale_chunk_k,
        MMA_SCALE_CHUNKS_PER_KBLOCK=mma_scale_chunks_per_kblock,
        MMA_SCALE_VEC_SIZE=_mma_scale_vec_size_enum(sf_vec_size),
        MMA_SCALE_FORMAT=_mma_scale_format(sf_vec_size),
        MMA_BLOCK_SCALE_KIND=_mma_block_scale_kind(sf_vec_size),
        QK_MMA_KBLOCKS=qk_mma_kblocks,
        SMEM_P4_QK_DATA_STAGE_KBLOCKS=qk_data_stage_kblocks,
        SMEM_P4_QK_DATA_STAGES=qk_data_stages,
        SMEM_P4_QK_DATA_STAGE_SF_CHUNKS=qk_data_stage_sf_chunks,
        SMEM_P4_QK_TAIL_DATA_STAGE_SF_CHUNKS=qk_tail_data_stage_sf_chunks,
        SMEM_P4_QK_K_TAIL_DATA_STAGE_SF_CHUNKS=qk_k_tail_data_stage_sf_chunks,
        SMEM_P4_QK_KTILE_RING_DATA_STAGES=qk_ktile_ring_data_stages,
        SMEM_P4_QK_BARRIER_STAGES=SMEM_P4_QK_SMEM_PIPELINE_SLOTS,
        NUM_QK_Q_STAGE=qk_data_stages,
        NUM_QK_Q_PIPELINE_STAGE=qk_data_stages,
        NUM_QK_AB_STAGE=qk_data_stages,
        SMEM_P4_QK_SFB_N_CHUNKS=qk_sfb_n_chunks,
        NUM_QK_K_STAGE=qk_ktile_ring_data_stages,
        QK_SF_GROUPS=QK_PADDED_DIM // sf_vec_size,
        PV_SF_GROUPS=KV_TILE // sf_vec_size,
        PV_TOTAL_SF_GROUPS=KV_TOTAL // sf_vec_size,
        SMEM_P4_QK_MMA_SCALE_CHUNKS_PER_KBLOCK=mma_scale_chunks_per_kblock,
        SMEM_P4_QK_MMA_KBLOCKS=qk_mma_kblocks,
        SMEM_P4_QK_SCALE_BURST_KBLOCKS=qk_scale_burst_kblocks,
        SMEM_P4_QK_SFA_FEED_COLS=qk_sfa_feed_cols,
        SMEM_P4_QK_SFB_FEED_COLS=qk_sfb_feed_cols,
        SMEM_P4_QK_COMPACT_SFB_COLS_PER_SCALE_CHUNK=qk_compact_sfb_cols_per_scale_chunk,
        SMEM_P4_QK_SFB_COLS_PER_KBLOCK=qk_sfb_cols_per_kblock,
        SMEM_P4_QK_SCALE_B_COL_OFFSET=qk_scale_b_col_offset,
        SMEM_P4_QK_SCALE_FEED_COL_END=qk_scale_feed_col_end,
        SMEM_P4_P_SFA_TAIL_COL_OFFSET=p_sfa_tail_col_offset,
        SMEM_P4_PV_VSF_TAIL_COL_OFFSET=pv_vsf_tail_col_offset,
        SMEM_P4_PV_VSF_TAIL_COL_END=pv_vsf_tail_col_end,
        SMEM_P4_QK_RESIDENT_SF_CHUNKS=qk_resident_sf_chunks,
        SMEM_P4_QK_RESIDENT_SF_COLS=qk_resident_sf_cols,
        SMEM_P4_RESIDENT_Q_SF_COL_OFFSET=SMEM_P4_SCALE_FEED_TMEM_COLS[0],
        SMEM_P4_SCORE_SLOT_SF_GROUPS=SMEM_P4_SCORE_SLOT_N // sf_vec_size,
        SMEM_P4_P_SFA_COLS_PER_SCALE_CHUNK=SMEM_P4_P_SFA_COLS_PER_SCALE_CHUNK,
        SMEM_P4_P_SCALE_FEED_STAGE_COLS=p_scale_feed_stage_cols,
        SMEM_P4_PV_SFB_SLOT_COL_OFFSET=SMEM_P4_P_SCALE_A_COL_OFFSET + p_scale_feed_stage_cols,
        SMEM_P4_PV_SFB_N_CHUNKS=pv_sfb_n_chunks,
        SMEM_P4_PV_COMPACT_SFB_COLS_PER_SCALE_CHUNK=pv_compact_sfb_cols_per_scale_chunk,
        SMEM_P4_PV_SFB_COLS_PER_KBLOCK=pv_sfb_cols_per_kblock,
        SMEM_P4_PV_SFB_SCORE_SLOT_COLS=pv_sfb_score_slot_cols,
        SMEM_P4_PV_SFB_SCORE_COL_OFFSET=pv_sfb_score_col_offset,
        SMEM_P4_P4_OWNER_TMEM_MAILBOX_SLOT_COL_OFFSET=SMEM_P4_PV_SFB_SLOT_COL_OFFSET
        + pv_sfb_score_slot_cols,
        SMEM_P4_RUNNING_ROWMETA_TMEM_COLS=running_rowmeta_tmem_cols,
        SMEM_P4_ROWMETA_TMEM_COLS=rowmeta_tmem_cols,
        SMEM_P4_ROWMETA_TMEM_COL_OFFSET=rowmeta_tmem_col_offset,
        SMEM_P4_STAGE_ROWMAX_TMEM_COLS=stage_rowmax_tmem_cols,
        SMEM_P4_STAGE_ROWSUM_TMEM_COLS=stage_rowsum_tmem_cols,
        SMEM_P4_STAGE_ROWMAX_TMEM_COL_OFFSET=rowmeta_tmem_col_offset,
        SMEM_P4_STAGE_ROWSUM_TMEM_COL_OFFSET=rowmeta_tmem_col_offset + stage_rowmax_tmem_cols,
        SMEM_P4_RUNNING_ROWMAX_TMEM_COL_OFFSET=rowmeta_tmem_col_offset
        + stage_rowmax_tmem_cols
        + stage_rowsum_tmem_cols,
        SMEM_P4_RUNNING_ROWSUM_TMEM_COL_OFFSET=rowmeta_tmem_col_offset
        + stage_rowmax_tmem_cols
        + stage_rowsum_tmem_cols
        + 1,
        SMEM_P4_OLD_SCALE_TMEM_COL_OFFSET=rowmeta_tmem_col_offset
        + stage_rowmax_tmem_cols
        + stage_rowsum_tmem_cols
        + 2,
        SMEM_P4_PAIR_FIRST_OLD_SCALE_TMEM_COL_OFFSET=rowmeta_tmem_col_offset
        + stage_rowmax_tmem_cols
        + stage_rowsum_tmem_cols
        + 2
        + SMEM_P4_OLD_SCALE_TMEM_SLOTS
        - 1,
        SMEM_P4_ROWMETA_TMEM_COL_END=rowmeta_tmem_col_offset
        + stage_rowmax_tmem_cols
        + stage_rowsum_tmem_cols
        + running_rowmeta_tmem_cols,
        MMA_SCALE_CHUNKS_PER_TILE=mma_scale_chunks_per_tile,
        SFA_TMA_STAGE_BYTES=sfa_tma_stage_bytes,
        SFB_TMA_STAGE_BYTES=sfb_tma_stage_bytes,
        QK_SFA_TMA_STAGE_BYTES=qk_sfa_tma_stage_bytes,
        QK_SFB_TMA_STAGE_BYTES=qk_sfb_tma_stage_bytes,
        QK_SFA_STAGE_BYTES=qk_sfa_stage_bytes,
        QK_SFB_STAGE_BYTES=qk_sfb_stage_bytes,
        QK_TAIL_SFA_STAGE_BYTES=512 * qk_tail_data_stage_sf_chunks,
        QK_TAIL_SFB_STAGE_BYTES=512 * qk_k_tail_data_stage_sf_chunks * qk_sfb_n_chunks,
        SMEM_P4_QK_STAGE_BYTES=QK_A_STAGE_BYTES
        + QK_B_STAGE_BYTES
        + qk_sfa_tma_stage_bytes
        + qk_sfb_tma_stage_bytes,
        SMEM_P4_Q_ONLY_STAGE_BYTES=QK_A_STAGE_BYTES + qk_sfa_tma_stage_bytes,
        SMEM_P4_QK_KONLY_STAGE_BYTES=QK_B_STAGE_BYTES + qk_sfb_tma_stage_bytes,
        SMEM_P4_QK_TAIL_STAGE_BYTES=QK_TAIL_A_STAGE_BYTES
        + QK_TAIL_B_STAGE_BYTES
        + 512 * qk_tail_data_stage_sf_chunks
        + 512 * qk_k_tail_data_stage_sf_chunks * qk_sfb_n_chunks,
        SMEM_P4_Q_ONLY_TAIL_STAGE_BYTES=QK_TAIL_A_STAGE_BYTES + 512 * qk_tail_data_stage_sf_chunks,
        SMEM_P4_QK_KONLY_TAIL_STAGE_BYTES=QK_TAIL_B_STAGE_BYTES
        + 512 * qk_k_tail_data_stage_sf_chunks * qk_sfb_n_chunks,
        SMEM_P4_QK_TILE_BYTES=SMEM_P4_QK_FULL_DATA_STAGES
        * (QK_A_STAGE_BYTES + QK_B_STAGE_BYTES + qk_sfa_tma_stage_bytes + qk_sfb_tma_stage_bytes)
        + QK_TAIL_A_STAGE_BYTES
        + QK_TAIL_B_STAGE_BYTES
        + 512 * qk_tail_data_stage_sf_chunks
        + 512 * qk_k_tail_data_stage_sf_chunks * qk_sfb_n_chunks,
        SMEM_P4_QK_KONLY_TILE_BYTES=SMEM_P4_QK_FULL_DATA_STAGES
        * (QK_B_STAGE_BYTES + qk_sfb_tma_stage_bytes)
        + QK_TAIL_B_STAGE_BYTES
        + 512 * qk_k_tail_data_stage_sf_chunks * qk_sfb_n_chunks,
        SMEM_P4_Q_COMPACT_TILE_BYTES=SMEM_P4_QK_FULL_DATA_STAGES * QK_A_STAGE_BYTES
        + QK_TAIL_A_STAGE_BYTES,
        SMEM_P4_QSF_COMPACT_TILE_BYTES=SMEM_P4_QK_FULL_DATA_STAGES * qk_sfa_stage_bytes
        + 512 * qk_tail_data_stage_sf_chunks,
        SMEM_P4_K_COMPACT_SLOT_BYTES=SMEM_P4_QK_FULL_DATA_STAGES * QK_B_STAGE_BYTES
        + QK_TAIL_B_STAGE_BYTES,
        SMEM_P4_KSF_COMPACT_SLOT_BYTES=SMEM_P4_QK_FULL_DATA_STAGES * qk_sfb_stage_bytes
        + 512 * qk_k_tail_data_stage_sf_chunks * qk_sfb_n_chunks,
        SMEM_P4_K_COMPACT_RING_BYTES=SMEM_P4_QK_KTILE_RING_STAGES
        * (SMEM_P4_QK_FULL_DATA_STAGES * QK_B_STAGE_BYTES + QK_TAIL_B_STAGE_BYTES),
        SMEM_P4_KSF_COMPACT_RING_BYTES=SMEM_P4_QK_KTILE_RING_STAGES
        * (
            SMEM_P4_QK_FULL_DATA_STAGES * qk_sfb_stage_bytes
            + 512 * qk_k_tail_data_stage_sf_chunks * qk_sfb_n_chunks
        ),
        SMEM_P4_STATIC_SMEM_BYTES=static_smem_bytes,
        SMEM_P4_V_DATA_STAGE_BYTES=B_STAGE_BYTES * SMEM_P4_N_OUT_TILES,
        SMEM_P4_V_SFB_TMA_STAGE_BYTES=sfa_tma_stage_bytes * pv_sfb_n_chunks * SMEM_P4_N_OUT_TILES,
        SMEM_P4_V_TILE_TMA_BYTES=B_STAGE_BYTES * SMEM_P4_N_OUT_TILES
        + sfa_tma_stage_bytes * pv_sfb_n_chunks * SMEM_P4_N_OUT_TILES,
    )


_refresh_sf_vec_config(SF_VEC_SIZE)
if SMEM_P4_Q_COMPACT_TILE_BYTES // ctm.Int32.bytes < SMEM_P4_ROWSUM_STORAGE_WORDS:
    raise ValueError("dead resident-Q SMEM must hold the unit FP4 rowsum B and SFB templates")


SMEM_P4_P4_SLOT0_BAR_ID = 6
SMEM_P4_P4_SLOT1_BAR_ID = 7
SMEM_P4_P_SFA_OWNER_BAR_ID = 10
SMEM_P4_DUAL_HALF_BAR_ID = SMEM_P4_P_SFA_OWNER_BAR_ID
SMEM_P4_DUAL_HALF_BAR_THREADS = 32 * SMEM_P4_TOTAL_PRODUCER_WARPS


@cute.jit
def fp4_mla_4x_full_ctm(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    q_sf_ptr: cute.Pointer,
    k_sf_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    page_table_ptr: cute.Pointer,
    valid_k_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    accum_ptr: cute.Pointer,  # x79 ABI slot, unused: no carry in FULL-only (host: HEADER_SCRATCH_WORDS int32 scratch)
    row_max_ptr: cute.Pointer,  # x79 ABI slot, unused: no meta
    row_sum_ptr: cute.Pointer,  # x79 ABI slot, unused: no header
    page_indptr_ptr: cute.Pointer,
    q_global_scale_ptr: cute.Pointer,
    kv_global_scale_ptr: cute.Pointer,
    plan_ptr: cute.Pointer,  # split mode: int32 packed plan (fp4_mla_packed_reduce layout); FULL mode: unused scratch
    partial_ptr: cute.Pointer,  # split mode: fp32 [task_capacity, rows_per_seq, 512] unnormalized O
    rowmax_ptr: cute.Pointer,  # split mode: fp32 [task_capacity, rows_per_seq] row anchor (raw score units)
    rowsum_ptr: cute.Pointer,  # split mode: fp32 [task_capacity, rows_per_seq] denominator relative to the anchor
    problem_size: tuple,
    runtime_m: ctm.Int32,
    runtime_l: ctm.Int32,
    q_tma_l: ctm.Int32,
    num_cache_pages: ctm.Int32,
    num_v_cache_pages: ctm.Int32,
    v_page_offset: ctm.Int32,
    kv_page_stride_bytes: ctm.Int64,
    ksf_page_stride_bytes: ctm.Int64,
    vsf_page_stride_bytes: ctm.Int64,
    softmax_scale_log2: ctm.Float32,
    pv_output_scale: ctm.Float32,
    requested_cut: ctm.Int32,  # x79 ABI slot, unused: no PREFIX cut in FULL-only
    task_capacity: ctm.Int32,  # split mode: plan slots (requests * splits) = plan table stride
    used_tasks: ctm.Int32,  # split mode: planned tasks actually launched; grid z = used_tasks * pairs_per_seq
    stream: cuda.CUstream,
    page_size: ctm.Constexpr = KV_TILE,
    use_mixed_imlp: ctm.Constexpr = False,
    query_len_per_seq: ctm.Constexpr = 1,
    use_consecutive_page_pair: ctm.Constexpr = False,
    use_ksf_gather4: ctm.Constexpr = False,
    heads: ctm.Constexpr = 128,
    split_mode: ctm.Constexpr = False,
) -> None:
    n, k = problem_size
    # n is staged here; N512 and the fair envelope are enforced by the host (full_validate_envelope).
    if cutlass.const_expr(query_len_per_seq not in SUPPORTED_QUERY_LENS):
        raise ValueError("4x FULL-only requires query_len_per_seq in (2, 4, 6, 8)")
    # rows mode: a 128-row block = 128 consecutive packed [token, head] rows of one request; a pair (2 blocks,
    # 256 rows) must stay inside one request so both CTAs share the K/V tiles -> heads * q % 256 == 0.
    if cutlass.const_expr((heads * query_len_per_seq) % 256 != 0):
        raise ValueError("4x rows mode requires heads * query_len_per_seq to be a multiple of 256")
    blocks_per_seq = heads * query_len_per_seq // 128
    pairs_per_seq = heads * query_len_per_seq // 256
    rows_per_seq = heads * query_len_per_seq
    m = runtime_m
    batch_size = runtime_l
    num_requests = batch_size // blocks_per_seq
    q_f4_tensor = cute.make_tensor(
        q_ptr,
        cute.make_layout(
            (QK_PADDED_DIM, m, q_tma_l),
            stride=(1, QK_PADDED_DIM, cute.assume(m * QK_PADDED_DIM, 32)),
        ),
    )
    k_f4_tensor = cute.make_tensor(
        k_ptr,
        cute.make_layout(
            (TRTLLM_K_STORAGE_DIM, page_size, num_cache_pages),
            stride=(1, TRTLLM_K_STORAGE_DIM, kv_page_stride_bytes * 2),
        ),
    )
    page_table_tensor = cute.make_tensor(
        cute.recast_ptr(page_table_ptr, dtype=cutlass.Int32),
        cute.make_layout(
            (SMEM_P4_RUNTIME_MAX_PAGES * (batch_size // blocks_per_seq),), stride=(1,)
        ),
    )
    page_indptr_tensor = cute.make_tensor(
        cute.recast_ptr(page_indptr_ptr, dtype=cutlass.Int32),
        cute.make_layout((batch_size // blocks_per_seq + 1,), stride=(1,)),
    )
    q_global_scale_tensor = cute.make_tensor(
        cute.recast_ptr(q_global_scale_ptr, dtype=cutlass.Float32),
        cute.make_layout((1,), stride=(1,)),
    )
    kv_global_scale_tensor = cute.make_tensor(
        cute.recast_ptr(kv_global_scale_ptr, dtype=cutlass.Float32),
        cute.make_layout((1,), stride=(1,)),
    )
    valid_k_tensor = cute.make_tensor(
        cute.recast_ptr(valid_k_ptr, dtype=cutlass.Int32),
        cute.make_layout((batch_size // blocks_per_seq,), stride=(1,)),
    )
    # split mode (Milestone 2): plan + FP32 partial workspace in fp4_mla_packed_reduce's layout; FULL mode binds
    # 1-element views of the host scratch (never touched).
    if cutlass.const_expr(split_mode):
        plan_tensor = cute.make_tensor(
            cute.recast_ptr(plan_ptr, dtype=cutlass.Int32),
            cute.make_layout(
                (ctm.Int32(3) + num_requests + ctm.Int32(3) * task_capacity,), stride=(1,)
            ),
        )
        partial_tensor = cute.make_tensor(
            cute.recast_ptr(partial_ptr, dtype=cutlass.Float32),
            cute.make_layout((task_capacity * ctm.Int32(rows_per_seq * 512),), stride=(1,)),
        )
        rowmax_tensor = cute.make_tensor(
            cute.recast_ptr(rowmax_ptr, dtype=cutlass.Float32),
            cute.make_layout((task_capacity * ctm.Int32(rows_per_seq),), stride=(1,)),
        )
        rowsum_tensor = cute.make_tensor(
            cute.recast_ptr(rowsum_ptr, dtype=cutlass.Float32),
            cute.make_layout((task_capacity * ctm.Int32(rows_per_seq),), stride=(1,)),
        )
    else:
        plan_tensor = cute.make_tensor(
            cute.recast_ptr(plan_ptr, dtype=cutlass.Int32), cute.make_layout((1,), stride=(1,))
        )
        partial_tensor = cute.make_tensor(
            cute.recast_ptr(partial_ptr, dtype=cutlass.Float32), cute.make_layout((1,), stride=(1,))
        )
        rowmax_tensor = cute.make_tensor(
            cute.recast_ptr(rowmax_ptr, dtype=cutlass.Float32), cute.make_layout((1,), stride=(1,))
        )
        rowsum_tensor = cute.make_tensor(
            cute.recast_ptr(rowsum_ptr, dtype=cutlass.Float32), cute.make_layout((1,), stride=(1,))
        )
    grid_z = batch_size // 2  # one FULL task per pair
    if cutlass.const_expr(split_mode):
        grid_z = used_tasks * ctm.Int32(
            pairs_per_seq
        )  # (planned task, pair of the request) per cluster
    v_tma_tensor = cute.make_tensor(
        b_ptr,
        cute.make_layout(
            (
                page_size,
                SMEM_P4_V_N_PER_CTA,
                n // SMEM_P4_V_N_PER_CTA,
                num_v_cache_pages,
            ),
            stride=(1, page_size, page_size * SMEM_P4_V_N_PER_CTA, page_size * n),
        ),
    )
    v_pair_tma_tensor = cute.make_tensor(
        b_ptr,
        cute.make_layout(
            (
                page_size,
                SMEM_P4_V_N_PER_CTA,
                SMEM_P4_N_OUT_TILES,
                num_v_cache_pages,
                CLUSTER_SHAPE_MNK[0],
            ),
            stride=(
                1,
                page_size,
                page_size * SMEM_P4_BMM2_N,
                page_size * n,
                page_size * SMEM_P4_V_N_PER_CTA,
            ),
        ),
    )
    c_layout = cute.make_layout(
        (cute.assume(m, 32), cute.assume(n, 16), batch_size),
        stride=(cute.assume(n, 16), 1, cute.assume(m * n, 512)),
    )
    c_tensor = cute.make_tensor(c_ptr, c_layout)
    # FULL-only: no carry (accum), meta (row_max) or header (row_sum) views and no carry TMA
    # descriptor; accum_ptr / row_max_ptr / row_sum_ptr are accepted for ABI compatibility only.
    q_sf_blocks = QK_SF_GROUPS // 4
    k_sf_blocks = TRTLLM_K_SF_GROUPS // 4
    vsf_mma_tensor = cute.make_tensor(
        cute.recast_ptr(sfb_ptr, dtype=cutlass.Uint16),
        cute.make_layout(
            (
                256,
                SMEM_P4_SCALE_CHUNKS_PER_PAGE,
                n // SMEM_P4_SFB_ATOM_ROWS,
                num_v_cache_pages,
            ),
            stride=(
                1,
                256,
                256 * SMEM_P4_SCALE_CHUNKS_PER_PAGE,
                vsf_page_stride_bytes // 2,
            ),
        ),
    )
    qsf_mma_tile_m = SMEM_P4_CTA_GROUP_M
    qsf_mma_m_tiles = cute.assume(m // qsf_mma_tile_m, 1)
    qsf_mma_tensor = cute.make_tensor(
        cute.recast_ptr(q_sf_ptr, dtype=cutlass.Uint16),
        cute.make_layout(
            (256, q_sf_blocks, qsf_mma_m_tiles, q_tma_l),
            stride=(
                1,
                256,
                256 * q_sf_blocks,
                cute.assume(256 * q_sf_blocks * qsf_mma_m_tiles, 1),
            ),
        ),
    )
    ksf_mma_tensor = cute.make_tensor(
        cute.recast_ptr(k_sf_ptr, dtype=cutlass.Uint16),
        cute.make_layout(
            (256, k_sf_blocks, num_cache_pages, 1),
            stride=(
                1,
                256,
                ksf_page_stride_bytes // 2,
                ksf_page_stride_bytes // 2 * num_cache_pages,
            ),
        ),
    )
    ksf_gather_tensor = cute.make_tensor(
        cute.recast_ptr(k_sf_ptr, dtype=cutlass.Uint16),
        cute.make_layout(
            (256, k_sf_blocks * num_cache_pages),
            stride=(1, 256),
        ),
    )
    tma_q_desc = cuda_tma.create_tensor_map_tiled_from_view(
        q_f4_tensor,
        box_dims=(QK_DATA_STAGE_K_DIM, 128, 1),
        stride_order=(0, 1, 2),
        swizzle=QK_FULL_TMA_SWIZZLE,
        tma_format=cuda_tma.TensorMapDataFormat.B4X16,
    )
    tma_k_page_desc = cuda_tma.create_tensor_map_tiled_from_view(
        k_f4_tensor,
        box_dims=(QK_DATA_STAGE_K_DIM, SMEM_P4_QK_TMA_N_PER_CHUNK, 1),
        stride_order=(0, 1, 2),
        swizzle=QK_FULL_TMA_SWIZZLE,
        tma_format=cuda_tma.TensorMapDataFormat.B4X16,
    )
    tma_q_tail_desc = cuda_tma.create_tensor_map_tiled_from_view(
        q_f4_tensor,
        box_dims=(QK_TAIL_DATA_STAGE_K_DIM, 128, 1),
        stride_order=(0, 1, 2),
        swizzle=QK_TAIL_TMA_SWIZZLE,
        tma_format=cuda_tma.TensorMapDataFormat.B4X16,
    )
    tma_k_tail_page_desc = cuda_tma.create_tensor_map_tiled_from_view(
        k_f4_tensor,
        box_dims=(QK_K_TAIL_DATA_STAGE_K_DIM, SMEM_P4_QK_TMA_N_PER_CHUNK, 1),
        stride_order=(0, 1, 2),
        swizzle=QK_K_TAIL_TMA_SWIZZLE,
        tma_format=cuda_tma.TensorMapDataFormat.B4X16,
    )
    tma_qsf_desc = cuda_tma.create_tensor_map_tiled_from_view(
        qsf_mma_tensor,
        box_dims=(256, SMEM_P4_QK_DATA_STAGE_SF_CHUNKS, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda_tma.TensorMapSwizzle.none,
        tma_format=cuda_tma.TensorMapDataFormat.DEFAULT,
    )
    tma_qsf_tail_desc = cuda_tma.create_tensor_map_tiled_from_view(
        qsf_mma_tensor,
        box_dims=(256, SMEM_P4_QK_TAIL_DATA_STAGE_SF_CHUNKS, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda_tma.TensorMapSwizzle.none,
        tma_format=cuda_tma.TensorMapDataFormat.DEFAULT,
    )
    tma_ksf_desc = cuda_tma.create_tensor_map_tiled_from_view(
        ksf_mma_tensor,
        box_dims=(256, 1, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda_tma.TensorMapSwizzle.none,
        tma_format=cuda_tma.TensorMapDataFormat.DEFAULT,
    )
    if cutlass.const_expr(use_consecutive_page_pair):
        # A caller promise that covers every active tile lets one TMA span the
        # adjacent physical-page pair.  Mode order (atom, page, scale, batch)
        # preserves the consumer's scale-major SMEM image.
        tma_ksf_stage_desc = cuda_tma.create_tensor_map_tiled_from_view(
            ksf_mma_tensor,
            box_dims=(
                256,
                SMEM_P4_QK_DATA_STAGE_SF_CHUNKS,
                SMEM_P4_QK_SFB_N_CHUNKS,
                1,
            ),
            stride_order=(0, 2, 1, 3),
            swizzle=cuda_tma.TensorMapSwizzle.none,
            tma_format=cuda_tma.TensorMapDataFormat.DEFAULT,
        )
        tma_ksf_tail_stage_desc = cuda_tma.create_tensor_map_tiled_from_view(
            ksf_mma_tensor,
            box_dims=(
                256,
                SMEM_P4_QK_K_TAIL_DATA_STAGE_SF_CHUNKS,
                SMEM_P4_QK_SFB_N_CHUNKS,
                1,
            ),
            stride_order=(0, 2, 1, 3),
            swizzle=cuda_tma.TensorMapSwizzle.none,
            tma_format=cuda_tma.TensorMapDataFormat.DEFAULT,
        )
    elif cutlass.const_expr(use_ksf_gather4):
        # Each 512B KSF atom is one gather row. The generic fast path is
        # selected only when physical pages have no padding, so flattening
        # (page, scale chunk) into a uniform row index is exact.
        tma_ksf_stage_desc = cuda_tma.create_tensor_map_tiled_from_view(
            ksf_gather_tensor,
            box_dims=(256, 1),
            stride_order=(0, 1),
            swizzle=cuda_tma.TensorMapSwizzle.none,
            tma_format=cuda_tma.TensorMapDataFormat.DEFAULT,
        )
        tma_ksf_tail_stage_desc = cuda_tma.create_tensor_map_tiled_from_view(
            ksf_gather_tensor,
            box_dims=(256, 1),
            stride_order=(0, 1),
            swizzle=cuda_tma.TensorMapSwizzle.none,
            tma_format=cuda_tma.TensorMapDataFormat.DEFAULT,
        )
    else:
        tma_ksf_stage_desc = cuda_tma.create_tensor_map_tiled_from_view(
            ksf_mma_tensor,
            box_dims=(256, SMEM_P4_QK_DATA_STAGE_SF_CHUNKS, 1, 1),
            stride_order=(0, 1, 2, 3),
            swizzle=cuda_tma.TensorMapSwizzle.none,
            tma_format=cuda_tma.TensorMapDataFormat.DEFAULT,
        )
        tma_ksf_tail_stage_desc = cuda_tma.create_tensor_map_tiled_from_view(
            ksf_mma_tensor,
            box_dims=(256, SMEM_P4_QK_K_TAIL_DATA_STAGE_SF_CHUNKS, 1, 1),
            stride_order=(0, 1, 2, 3),
            swizzle=cuda_tma.TensorMapSwizzle.none,
            tma_format=cuda_tma.TensorMapDataFormat.DEFAULT,
        )
    tma_vsf_desc = cuda_tma.create_tensor_map_tiled_from_view(
        vsf_mma_tensor,
        box_dims=(
            256,
            SMEM_P4_SCALE_CHUNKS_PER_PAGE,
            SMEM_P4_PV_SFB_N_CHUNKS * SMEM_P4_N_OUT_TILES,
            SMEM_P4_PAGES_PER_KV_TILE,
        ),
        stride_order=(0, 2, 1, 3),
        swizzle=cuda_tma.TensorMapSwizzle.none,
        tma_format=cuda_tma.TensorMapDataFormat.DEFAULT,
    )
    tma_vsf_stage_desc = cuda_tma.create_tensor_map_tiled_from_view(
        vsf_mma_tensor,
        box_dims=(
            256,
            SMEM_P4_SCALE_CHUNKS_PER_PAGE,
            SMEM_P4_PV_SFB_N_CHUNKS,
            1,
        ),
        stride_order=(0, 2, 1, 3),
        swizzle=cuda_tma.TensorMapSwizzle.none,
        tma_format=cuda_tma.TensorMapDataFormat.DEFAULT,
    )
    tma_v_tile_desc = cuda_tma.create_tensor_map_tiled_from_view(
        v_tma_tensor,
        box_dims=(TRTLLM_PAGE_SIZE, SMEM_P4_V_N_PER_CTA, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda_tma.TensorMapSwizzle.s64b,
        tma_format=cuda_tma.TensorMapDataFormat.B4X16,
    )
    v_pair_page_box: ctm.Constexpr = 1
    if cutlass.const_expr(use_consecutive_page_pair):
        v_pair_page_box = SMEM_P4_PAGES_PER_KV_TILE
    tma_v_pair_desc = cuda_tma.create_tensor_map_tiled_from_view(
        v_pair_tma_tensor,
        box_dims=(
            TRTLLM_PAGE_SIZE,
            SMEM_P4_V_N_PER_CTA,
            1,  # One O256 half per CTA pair.
            v_pair_page_box,
            1,
        ),
        stride_order=(0, 1, 2, 3, 4),
        swizzle=cuda_tma.TensorMapSwizzle.s64b,
        tma_format=cuda_tma.TensorMapDataFormat.B4X16,
    )
    tma_sfb_desc = tma_vsf_stage_desc
    ordered_v30_six_yield_kernel(
        page_table_tensor,
        page_indptr_tensor,
        valid_k_tensor,
        q_global_scale_tensor,
        kv_global_scale_tensor,
        plan_tensor,
        partial_tensor,
        rowmax_tensor,
        rowsum_tensor,  # split mode
        tma_q_desc,
        tma_k_page_desc,
        tma_q_tail_desc,
        tma_k_tail_page_desc,
        tma_v_tile_desc,
        tma_qsf_desc,
        tma_qsf_tail_desc,
        tma_ksf_desc,
        tma_ksf_stage_desc,
        tma_ksf_tail_stage_desc,
        tma_vsf_desc,
        tma_vsf_stage_desc,
        tma_v_pair_desc,
        tma_sfb_desc,
        c_tensor,
        problem_size,
        runtime_m,
        v_page_offset,
        softmax_scale_log2,
        pv_output_scale,
        task_capacity,
        num_requests,  # split mode
        page_size,
        query_len_per_seq,
        heads,
        split_mode,
        use_mixed_imlp,
        use_consecutive_page_pair,
        use_ksf_gather4,
    ).launch(
        grid=(
            cute.ceil_div(c_tensor.shape[0], 128) * 4,
            cute.ceil_div(OUT_DIM, MMA_TILER_MNK[1]),
            grid_z,  # FULL: one task per pair (host: <= 46); split: used_tasks * pairs_per_seq (multi-wave)
        ),
        block=[384, 1, 1],
        cluster=(4, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
    )


@cute.jit
def _pack_p4_16_scores_from_group(
    group_scores,
    group_base: ctm.Int32,
    valid_k: ctm.Int32,
    group_valid: ctm.Boolean,
    mask_valid_k: ctm.Constexpr,
    use_mixed_imlp: ctm.Constexpr,
    softmax_scale_log2: ctm.Float32,
    p_bias: ctm.Float32,
    elem_base: ctm.Constexpr,
) -> tuple:
    words = ctm.Array(ctm.Int32, 2, space=ctm.AddressSpace.rmem)
    for word_part in ctm.range_constexpr(2):
        packed_p0 = ctm.Float32(0.0)
        packed_p1 = ctm.Float32(0.0)
        packed_p2 = ctm.Float32(0.0)
        packed_p3 = ctm.Float32(0.0)
        packed_p4 = ctm.Float32(0.0)
        packed_p5 = ctm.Float32(0.0)
        packed_p6 = ctm.Float32(0.0)
        packed_p7 = ctm.Float32(0.0)
        for pair_idx in ctm.range_constexpr(4):
            elem0: ctm.Constexpr = elem_base + word_part * 8 + pair_idx * 2
            elem1: ctm.Constexpr = elem0 + 1
            p0 = ctm.Float32(0.0)
            p1 = ctm.Float32(0.0)
            should_compute = ctm.Boolean(True)
            if cutlass.const_expr(mask_valid_k):
                should_compute = group_valid
            if should_compute:
                score0 = group_scores[elem0]
                score1 = group_scores[elem1]
                if cutlass.const_expr(mask_valid_k):
                    global_k_idx0 = group_base + ctm.Int32(elem0)
                    global_k_idx1 = group_base + ctm.Int32(elem1)
                    score0 = _mask_invalid_score(score0, global_k_idx0, valid_k)
                    score1 = _mask_invalid_score(score1, global_k_idx1, valid_k)
                p0, p1 = _softmax_exp2_pair(score0, score1, softmax_scale_log2, p_bias)
            if cutlass.const_expr(pair_idx == 0):
                packed_p0 = p0
                packed_p1 = p1
            elif cutlass.const_expr(pair_idx == 1):
                packed_p2 = p0
                packed_p3 = p1
            elif cutlass.const_expr(pair_idx == 2):
                packed_p4 = p0
                packed_p5 = p1
            else:
                packed_p6 = p0
                packed_p7 = p1
        words[word_part] = _pack_e2m1x8(
            packed_p0,
            packed_p1,
            packed_p2,
            packed_p3,
            packed_p4,
            packed_p5,
            packed_p6,
            packed_p7,
        )
    return (words[0], words[1])


@cute.jit
def _pack_p4_16_scores_from_group_packed_f16x2(
    group_scores,
    group_base: ctm.Int32,
    valid_k: ctm.Int32,
    group_valid: ctm.Boolean,
    mask_valid_k: ctm.Constexpr,
    softmax_scale_log2: ctm.Float32,
    p_bias: ctm.Float32,
    elem_base: ctm.Constexpr,
) -> tuple:
    words = ctm.Array(ctm.Int32, 2, space=ctm.AddressSpace.rmem)
    for word_part in ctm.range_constexpr(2):
        packed_pair0 = ctm.Int32(0)
        packed_pair1 = ctm.Int32(0)
        packed_pair2 = ctm.Int32(0)
        packed_pair3 = ctm.Int32(0)
        for pair_idx in ctm.range_constexpr(4):
            elem0: ctm.Constexpr = elem_base + word_part * 8 + pair_idx * 2
            elem1: ctm.Constexpr = elem0 + 1
            p_h2 = ctm.Int32(0)
            should_compute = ctm.Boolean(True)
            if cutlass.const_expr(mask_valid_k):
                should_compute = group_valid
            if should_compute:
                score0 = group_scores[elem0]
                score1 = group_scores[elem1]
                if cutlass.const_expr(mask_valid_k):
                    global_k_idx0 = group_base + ctm.Int32(elem0)
                    global_k_idx1 = group_base + ctm.Int32(elem1)
                    score0 = _mask_invalid_score(score0, global_k_idx0, valid_k)
                    score1 = _mask_invalid_score(score1, global_k_idx1, valid_k)
                p_h2 = _softmax_exp2_pair_packed_f16x2(score0, score1, softmax_scale_log2, p_bias)
            if cutlass.const_expr(pair_idx == 0):
                packed_pair0 = p_h2
            elif cutlass.const_expr(pair_idx == 1):
                packed_pair1 = p_h2
            elif cutlass.const_expr(pair_idx == 2):
                packed_pair2 = p_h2
            else:
                packed_pair3 = p_h2
        words[word_part] = _pack_e2m1x8_from_f16x2(
            packed_pair0, packed_pair1, packed_pair2, packed_pair3
        )
    return (words[0], words[1])


SMEM_P4_N256_SCORE_HALVES = 2
SMEM_P4_N256_SCORE_N_PER_COL_BAND = SMEM_P4_BMM1_N // SMEM_P4_TMEM_WARP_N
SMEM_P4_N256_SCORE_N_PER_PRODUCER = SMEM_P4_N256_SCORE_N_PER_COL_BAND // SMEM_P4_N256_SCORE_HALVES
SMEM_P4_N256_PSF_S2T_ROWS = SMEM_P4_BMM1_M
SMEM_P4_N256_PSF_S2T_COLS = SMEM_P4_TMEM_WARP_N * SMEM_P4_N256_SCORE_HALVES
SMEM_P4_N256_PSF_S2T_BANK_WORDS = SMEM_P4_N256_PSF_S2T_ROWS * SMEM_P4_N256_PSF_S2T_COLS
SMEM_P4_N256_PSF_S2T_BANKS = SMEM_P4_TMEM_BANKS
if (
    SMEM_P4_N256_PSF_S2T_ROWS != 64
    or SMEM_P4_N256_PSF_S2T_COLS != 4
    or SMEM_P4_N256_PSF_S2T_BANKS * SMEM_P4_N256_PSF_S2T_BANK_WORDS
    != SMEM_P4_P4_OWNER_SMEM_SF_MAILBOX_WORDS
):
    raise ValueError(
        "row-major PSF S2T banks must exactly alias the existing 2 KiB sPSF allocation"
    )


@cute.jit
def _select_smem_p4_lazy_anchor_with_validity(
    new_row_max: ctm.Float32,
    new_row_valid: ctm.Boolean,
    running_row_anchor: ctm.Float32,
    softmax_scale_log2: ctm.Float32,
) -> tuple:
    # Derive rebase anew from this call's inputs. Some DSL builds incorrectly
    # carry a conditionally assigned False/True local through the tile loop.
    negative_infinity = ctm.Float32(-ctm.Float32.inf)
    new_row_anchor = running_row_anchor
    anchor_delta_log2 = ctm.Float32(0.0)
    if new_row_valid:
        if running_row_anchor == negative_infinity:
            new_row_anchor = new_row_max
        else:
            anchor_delta_log2 = (new_row_max - running_row_anchor) * softmax_scale_log2
            if anchor_delta_log2 > ctm.Float32(SMEM_P4_LAZY_ANCHOR_REBASE_LOG2):
                new_row_anchor = new_row_max
    lane_rebase = (
        new_row_valid
        & (running_row_anchor != negative_infinity)
        & (anchor_delta_log2 > ctm.Float32(SMEM_P4_LAZY_ANCHOR_REBASE_LOG2))
    )
    return (new_row_anchor, lane_rebase, anchor_delta_log2)


@cute.jit
def _dealloc_tmem_cluster(
    tmem_dealloc_mbar,
    tmem_holding_buf,
    warp_idx: ctm.Int32,
    cta_rank: ctm.Int32,
    num_tmem_alloc_cols: ctm.Constexpr = NUM_TMEM_ALLOC_COLS,
    dealloc_warp_id: ctm.Constexpr = 0,
) -> None:
    if warp_idx == ctm.Int32(dealloc_warp_id):
        acc_tmem_ptr = ctm.inttoptr(tmem_holding_buf.load(), 6, ctm.Float32)
        prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_2)
        peer_cta_rank = cta_rank ^ 1
        peer_mbar = _mapa_shared_cluster(tmem_dealloc_mbar, peer_cta_rank)
        _mbarrier_arrive_release_cta_shared_cluster(peer_mbar)
        while not prims.mbarrier_try_wait_parity(tmem_dealloc_mbar, 0, time_limit=10000000):
            pass
        prims.tcgen05_dealloc(
            acc_tmem_ptr,
            num_tmem_alloc_cols,
            is_exclusive=SMEM_P4_TMEM_ALLOC_EXCLUSIVE,
            group=prims.CTAGroup.CTA_2,
        )


DS_Q_BYTES = 128 * 640 // 2
DS_K_BYTES = 128 * 640 // 2
DS_QSF_BYTES = 10 * 512
DS_KSF_BYTES = 10 * 2 * 512
# V ring: tile t of a pair occupies V / VSF bank t % DS_V_BANKS.  With four banks the loader issues the TMA load of
# tile t when PV(t - 4) has completed, a full round before PV(t) needs it: V is never what a PV waits for and needs no
# per-tile L2 prefetch (with two banks the load could start only one PV earlier, and its latency sat on the round).
# Only the kernel-start warm of tiles 0 and 1 (issued while Q is in flight) remains.
DS_V_BANKS = 4
DS_V_BANK_BYTES = 2 * 128 * 128 // 2
DS_VSF_BANK_BYTES = 2 * 4 * 512
DS_V_BYTES = DS_V_BANKS * DS_V_BANK_BYTES
DS_VSF_BYTES = DS_V_BANKS * DS_VSF_BANK_BYTES
DS_P_BYTES = 128 * 256 // 2
DS_PSF_BYTES = 128 * 16
DS_META_BYTES = 128 * 4 * 2
DS_PAYLOAD_BYTES = DS_P_BYTES + DS_PSF_BYTES + DS_META_BYTES
# The two P payload banks by role (index into sPayloadBanks): this CTA's softmax warps store into the own bank, which
# the copy engine and the own PV read; the other pair's copy lands in the landing bank, which the peer PV reads.
PB_OWN = 1
PB_LAND = 0

# Shared-memory layout of one CTA: the arrays in the order ordered_v30_six_yield_kernel declares them (the control
# words by hand, the data arrays from this table through _ds_smem_arrays).  Every offset is a multiple of 1024 (the
# s128b swizzle of the operand arrays is taken on the address).
#
# GR100 serves shared memory from two sub-banks split at DS_SMEM_SUBBANK_BYTES.  A tensor-core operand fetch takes
# the read port of a whole sub-bank and wins over the copy engine's source read and over warp loads; an arriving
# copy or TMA load wins the write port over warp stores.  The arrays are therefore placed by who touches them:
#   below the border  the scale factors, V and the landing P bank (the peer PV's operands; where the P copy lands)
#   above the border  the own P bank (softmax stores, P copy source, own PV operand), the alternate [PSF | META]
#                     image (softmax stores, [PSF | META] copy source), then K and Q
# with the border exactly between the two P banks.  The control words (tmem handle, release cookie, mbarriers,
# address relay) come first, padded to DS_SMEM_CONTROL_BYTES.  The total is above what a block gets by default: the
# device must offer the oversized shared-memory mode (fp4_mla_cutedsl_4x_host.device_unsupported_reason).
DS_SMEM_SUBBANK_BYTES = 128 * 1024
DS_SMEM_CONTROL_BYTES = 4096
DS_SMEM_LAYOUT = (  # (array, element type, bytes)
    ("control", None, DS_SMEM_CONTROL_BYTES),
    ("sQSF", ctm.Uint8, DS_QSF_BYTES),
    ("sV", ctm.Int32, DS_V_BYTES),
    ("sVSF", ctm.Uint8, DS_VSF_BYTES),
    ("sKSFBanks", ctm.Uint8, 2 * DS_KSF_BYTES),
    ("sPayloadBanks", ctm.Int32, 2 * DS_PAYLOAD_BYTES),
    # cp_dual_alt: second [PSF|META] source image; own tiles alternate between the in-bank region of the own P bank
    # (even own index) and this buffer (odd), so the ALU stores of own tile i+1 can never race warp 9's [PSF|META]
    # copy of own tile i, which may still be pending then (see _r93_pack).
    ("sPsfMetaAlt", ctm.Int32, DS_PSF_BYTES + DS_META_BYTES),
    ("sKBanks", ctm.Int32, 2 * DS_K_BYTES),
    ("sQ", ctm.Int32, DS_Q_BYTES),
)
DS_SMEM_BYTES = sum(size for _, _, size in DS_SMEM_LAYOUT)


def _ds_smem_offset(name):
    offset = 0
    for array, _, size in DS_SMEM_LAYOUT:
        if array == name:
            return offset
        offset += size
    raise KeyError(name)


def _ds_smem_arrays():
    """Declare the data arrays of DS_SMEM_LAYOUT in table order; call once, right behind the control block."""
    return {
        name: ctm.Array(dtype, size // dtype.bytes, space=ctm.AddressSpace.smem, alignment=128)
        for name, dtype, size in DS_SMEM_LAYOUT[1:]
    }


assert all(size % 1024 == 0 for _, _, size in DS_SMEM_LAYOUT)
assert (PB_LAND, PB_OWN) == (0, 1)
assert _ds_smem_offset("sPayloadBanks") + DS_PAYLOAD_BYTES == DS_SMEM_SUBBANK_BYTES

# mbarriers (indices into ``bars``) of the V ring and of the [PSF | META] exchange; the other barriers are literal
# indices at their use.  A V bank has one "landed" and one "free" barrier, phase (tile // DS_V_BANKS) & 1: the free
# barrier is committed by the PV packet that read the bank, so PV(t) can never complete before V(t) was issued and
# the loader cannot miss a phase.
DS_NUM_BARS = 35
DS_BAR_META_LANDED = 2  # the other pair's [PSF | META] copy has landed
DS_BAR_META_FREE = 26  # the [PSF | META] landing bank may be overwritten
DS_BAR_V_LANDED = 17  # + V bank: the bank's TMA loads have landed
DS_BAR_V_FREE = 31  # + V bank: the PV packet that read the bank has completed
# (comments write bars[META_LANDED] / bars[META_FREE] for the two metadata barriers)
assert DS_V_BANKS & (DS_V_BANKS - 1) == 0  # V bank of a tile = tile & (DS_V_BANKS - 1)
assert DS_BAR_V_LANDED + DS_V_BANKS <= 21  # bars[21 + bank] is the next literal index in use
assert DS_BAR_V_FREE + DS_V_BANKS == DS_NUM_BARS  # the V-free barriers are the last of bars
DS_Q_TMEM = 256
DS_QSF_TMEM = 528
DS_KSF_TMEM = 544
# PV-only TMEM layout: O[0,512), rowsum[512,528), VSF[528,560),
# aligned unit SFB[560,568), unique-row PSF[568,572). QK layout is unchanged.
DS_PSF_TMEM = 568
DS_VSF_TMEM = 528
DS_ROWSUM_SF_TMEM = 560
DS_ROWSUM_ACC_TMEM = 512
assert DS_ROWSUM_SF_TMEM % 16 == 0
assert DS_ROWSUM_ACC_TMEM + 16 <= DS_VSF_TMEM
assert DS_VSF_TMEM + 32 <= DS_ROWSUM_SF_TMEM
assert DS_ROWSUM_SF_TMEM + 8 <= DS_PSF_TMEM
assert DS_PSF_TMEM + 8 <= NUM_TMEM_ALLOC_COLS


# ur_relief: kernel-constant DSMEM (shared::cluster) addresses used by warps 8 and 9.
# NVVM hoists each mapa of a loop-invariant local address to the kernel entry, so the
# base kept these seven values live across the whole kernel, i.e. in uniform
# registers through the issuer's QK/PV packet blocks. They are computed once per CTA
# at kernel start (_ur_relay_init) into the 32-byte sUrRelay table and re-read
# with one LDS at each use. Word 7 is padding.
UR_RELAY_P_DST = (
    0  # mapa(sPayloadBanks[PB_LAND], rank ^ 2): P bulk-copy destination (the peer's landing bank)
)
UR_RELAY_P_BAR = 1  # mapa(&bars[8], rank ^ 2): its complete_tx mbarrier
UR_RELAY_META_DST = (
    2  # mapa([PSF|META] of sPayloadBanks[PB_LAND], rank ^ 2): [PSF|META] bulk-copy destination
)
UR_RELAY_META_BAR = 3  # mapa(&bars[DS_BAR_META_LANDED], rank ^ 2): its complete_tx mbarrier
UR_RELAY_PAIR_BAR = 4  # mapa(&bars[1], rank ^ 1): issuer pair rendezvous
UR_RELAY_P_CREDIT_BAR = 5  # mapa(&bars[6], rank ^ 2): issuer peer-P credit release
UR_RELAY_LANDED_BAR = 6  # mapa(&bars[7], rank ^ 2): landing-bank release (warp 8 start, warp 9)
UR_RELAY_WORDS = 8
# Control block of DS_SMEM_LAYOUT: tmem + sReleaseCookie (4 B each), bars (8 B each) and sUrRelay (aligned 16, so up
# to 8 B of padding in front) must end inside the first 1024 B; sControlPad (alignment 1024) then ends the block at
# DS_SMEM_CONTROL_BYTES.
assert 8 + 8 * DS_NUM_BARS + 8 + 4 * UR_RELAY_WORDS <= 1024 < DS_SMEM_CONTROL_BYTES
# The base's cp.async.bulk (prims.cp_async_bulk_shared_cluster_shared_cta) for the two
# copy sizes; the size is a register operand, as in the base PTX.
_UR_BULK_COPY_PTX = {
    size: (
        "{\n"
        "\t.reg .b32 ur_size;\n"
        "\tmov.b32 ur_size, %d;\n"
        "\tcp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
        " [{$r0}], [{$r1}], ur_size, [{$r2}];\n"
        "\t}\n" % size
    )
    for size in (DS_P_BYTES, DS_PSF_BYTES + DS_META_BYTES)
}


@cute.jit
def _ur_mapa_u32(view, target_rank: ctm.Int32) -> ctm.Int32:
    # ur_relief: _mapa_shared_cluster(view, target_rank) as a u32 word: the same
    # mapa.shared::cluster.u32 of the same shared::cta address and target rank.
    return cute_inline_ptx(
        "mapa.shared::cluster.u32 {$w0}, {$r0}, {$r1};",
        write_only_types=[ctm.Int32],
        read_only_args=[_as_shared_cta(view).data_ptr(), target_rank],
    )


@cute.jit
def _ur_relay_init(relay, bars, sPayloadBanks, rank: ctm.Int32):
    # ur_relief: once per CTA, at kernel start (warp 8, elected lane). rank,
    # pair_id = rank >> 1 and the SMEM layout are CTA constants, so every word equals
    # the base's hoisted mapa result. Same mbarrier views as the use sites; the two copy destinations are the
    # peer's landing bank (below), not the own-bank views the use sites copy from.
    peer = rank ^ ctm.Int32(2)
    # The copy lands in the receiver's landing bank.
    sPayload = sPayloadBanks.subview(ctm.Int32(PB_LAND) * ctm.Int32(DS_PAYLOAD_BYTES // 4))
    relay[UR_RELAY_P_DST] = _ur_mapa_u32(sPayload, peer)
    relay[UR_RELAY_P_BAR] = _ur_mapa_u32(bars.subview(8), peer)
    relay[UR_RELAY_META_DST] = _ur_mapa_u32(sPayload.subview(ctm.Int32(DS_P_BYTES // 4)), peer)
    relay[UR_RELAY_META_BAR] = _ur_mapa_u32(bars.subview(DS_BAR_META_LANDED), peer)
    relay[UR_RELAY_PAIR_BAR] = _ur_mapa_u32(bars.subview(1), rank ^ ctm.Int32(1))
    relay[UR_RELAY_P_CREDIT_BAR] = _ur_mapa_u32(bars.subview(6), peer)
    relay[UR_RELAY_LANDED_BAR] = _ur_mapa_u32(bars.subview(7), peer)


@cute.jit
def _ur_arrive_remote(remote: ctm.Int32):
    # ur_relief: _mbarrier_arrive_release_cta_shared_cluster on a relayed address: the
    # same mbarrier.arrive.release.cta.shared::cluster.b64 _, [a], 1, with a the same
    # zero-extended 64-bit address operand (cvt.u64.u32) as the base's inline PTX.
    cute_inline_ptx(
        "{\n"
        "\t.reg .b64 ur_addr;\n"
        "\tcvt.u64.u32 ur_addr, {$r0};\n"
        "\tmbarrier.arrive.release.cta.shared::cluster.b64 _, [ur_addr], 1;\n"
        "\t}\n",
        read_only_args=[remote],
    )


@cute.jit
def _ur_bulk_copy_to_peer(dst: ctm.Int32, src, mbar: ctm.Int32, size: ctm.Constexpr):
    # ur_relief: prims.cp_async_bulk_shared_cluster_shared_cta(dst, src, mbar, size) with
    # the relayed dst/mbar words: the same PTX instruction and operand values.
    cute_inline_ptx(
        _UR_BULK_COPY_PTX[size],
        read_only_args=[dst, _as_shared_cta(src).data_ptr(), mbar],
    )


@cute.jit
def _vh2_issue(
    sV,
    sVSF,
    bars,
    local_tile: ctm.Int32,
    vpage0: ctm.Int32,
    vpage1: ctm.Int32,
    tma_v_pair_ptr,
    tma_vsf_stage_ptr,
    pair_id: ctm.Int32,
    pair_rank: ctm.Int32,
    v_page_offset: ctm.Int32,
    rank: ctm.Int32,
):
    # One tile's V/VSF packet; the page lookups are hoisted into the loader's rolling window.
    bank = local_tile & ctm.Int32(1)
    vbank = local_tile & ctm.Int32(DS_V_BANKS - 1)
    if local_tile >= ctm.Int32(DS_V_BANKS):
        # This bank's sole V/VSF reader is PV(local_tile - DS_V_BANKS): the previous phase of the bank's free barrier.
        _ds_wait(
            bars.subview(ctm.Int32(DS_BAR_V_FREE) + vbank),
            ((local_tile // ctm.Int32(DS_V_BANKS)) & ctm.Int32(1)) ^ ctm.Int32(1),
        )
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    if prims.elect_sync():
        full = bars.subview(ctm.Int32(DS_BAR_V_LANDED) + vbank)
        prims.mbarrier_arrive_expect_tx(full, DS_V_BANK_BYTES + DS_VSF_BANK_BYTES)
        for np in ctm.range_constexpr(2):
            page = vpage0 if cutlass.const_expr(np == 0) else vpage1
            _ds_tma(
                sV.subview(
                    vbank * ctm.Int32(DS_V_BANK_BYTES // 4)
                    + ctm.Int32(np * DS_V_BANK_BYTES // 2 // 4)
                ),
                tma_v_pair_ptr,
                (ctm.Int32(0), ctm.Int32(0), pair_id, page + v_page_offset, pair_rank),
                full,
                rank,
            )
            if bank == pair_rank:
                # vmc: this CTA issues the pair's VSF for this tile, multicast to both CTAs
                # (same box, same smem offset, same mbarrier offset in each CTA).
                _ds_tma_pair(
                    sVSF.subview(
                        vbank * ctm.Int32(DS_VSF_BANK_BYTES)
                        + ctm.Int32(np * DS_VSF_BANK_BYTES // 2)
                    ),
                    tma_vsf_stage_ptr,
                    (ctm.Int32(0), pair_id * ctm.Int32(2), ctm.Int32(0), page + v_page_offset),
                    full,
                    pair_id,
                )


@cute.jit
def _vh2_lookup4(
    mPageTable_pl,
    tile: ctm.Int32,
    sequence: ctm.Int32,
    page_begin: ctm.Int32,
    page_count: ctm.Int32,
):
    # Pages of tiles tile+4 / tile+5 (tile = even tile 2k of the current pair); clamped lookups,
    # harmless past the chunk.
    n0 = _lookup_physical_page(
        mPageTable_pl, (tile + ctm.Int32(4)) * ctm.Int32(2), sequence, page_begin, page_count
    )
    n1 = _lookup_physical_page(
        mPageTable_pl,
        (tile + ctm.Int32(4)) * ctm.Int32(2) + ctm.Int32(1),
        sequence,
        page_begin,
        page_count,
    )
    m0 = _lookup_physical_page(
        mPageTable_pl, (tile + ctm.Int32(5)) * ctm.Int32(2), sequence, page_begin, page_count
    )
    m1 = _lookup_physical_page(
        mPageTable_pl,
        (tile + ctm.Int32(5)) * ctm.Int32(2) + ctm.Int32(1),
        sequence,
        page_begin,
        page_count,
    )
    return n0, n1, m0, m1


@cute.jit
def _ds_wait(mbar, phase: ctm.Int32):
    while not prims.mbarrier_try_wait_parity(mbar, phase, time_limit=10000000):
        pass


@cute.jit
def _ds_seen(mbar, phase: ctm.Int32):
    # Non-blocking: has the phase with this parity completed?  Issued early so that its latency overlaps compute.
    return cute.arch.mbarrier_test_wait(mbar.data_ptr(), phase)


@cute.jit
def _ds_wait_unless(seen, mbar, phase: ctm.Int32):
    # Wait only when the earlier _ds_seen of the same phase had not found it complete.
    if not seen:
        _ds_wait(mbar, phase)


@cute.jit
def _ds_pair_sync(
    mbar,
    rank: ctm.Int32,
    phase: ctm.Int32,
    warp: ctm.Int32,
    relay,
    issue_warp: ctm.Constexpr = 4,
    cta_threads: ctm.Constexpr = 256,
):
    # Each pair advances independently; the PV pair never gates QK's next tile.
    # ur_relief: mbar is bars[1] at both call sites; relay word UR_RELAY_PAIR_BAR holds
    # mapa(mbar, rank ^ 1).
    if warp == ctm.Int32(issue_warp):
        pair_remote = relay[UR_RELAY_PAIR_BAR]
        if prims.elect_sync():
            prims.mbarrier_arrive(mbar)
            _ur_arrive_remote(pair_remote)
        _ds_wait(mbar, phase)
    prims.barrier(barrier_id=0, number_of_threads=cta_threads)


@cute.jit
def _ds_k2_views(sKBanks, sKSFBanks, bank: ctm.Int32):
    # Two disjoint K/KSF stages; Q is never aliased by the producer.
    return (
        sKBanks.subview(bank * ctm.Int32(DS_K_BYTES // 4)),
        sKSFBanks.subview(bank * ctm.Int32(DS_KSF_BYTES)),
    )


@cute.jit
def _ds_tma(dst, desc, coords, mbar, rank: ctm.Int32):
    prims.cp_async_bulk_tensor_shared_cluster_global(
        dst,
        desc,
        coords,
        _as_shared_cta(mbar),
        [],
        multicast_mask=ctm.Int16(1) << ctm.Int16(rank),
        group=prims.CTAGroup.CTA_1,
    )


@cute.jit
def _ds_tma_pair(dst, desc, coords, mbar, pair_id: ctm.Int32):
    # vmc: TMA load multicast to both CTAs of this cta_group::2 pair (ranks pair_id*2, pair_id*2+1);
    # the complete_tx lands on the mbarrier at the same offset in each destination CTA.
    prims.cp_async_bulk_tensor_shared_cluster_global(
        dst,
        desc,
        coords,
        _as_shared_cta(mbar),
        [],
        multicast_mask=ctm.Int16(3) << ctm.Int16(pair_id * ctm.Int32(2)),
        group=prims.CTAGroup.CTA_1,
    )


@cute.jit
def _ds_tmem_ptr(base: ctm.Int32, row: ctm.Int32, col: ctm.Int32):
    return ctm.inttoptr(base + (row << ctm.Int32(16)) + col, 6, ctm.Int32)


@cute.jit
def _ds_tma_ksf_gather4_local(
    smem_dst,
    tma_desc,
    row0: ctm.Int32,
    row1: ctm.Int32,
    row2: ctm.Int32,
    row3: ctm.Int32,
    mbar,
    rank: ctm.Int32,
):
    # Same gather coordinates/packing as the old 2xFP4 path, but preserve
    # native V15's per-CTA full barrier. Do NOT use the old helper: its
    # cta_group::2 + rank0 barrier would misroute follower completion.
    # Each 256-element Uint16 gather row is one 512B KSF atom.
    cute_inline_ptx(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1"
        " [{$r0}], [{$r1}, {{$r2}, {$r3}, {$r4}, {$r5}, {$r6}}], "
        "[{$r7}], {$r8};",
        read_only_args=[
            smem_dst.data_ptr(),
            tma_desc,
            ctm.Int32(0),
            row0,
            row1,
            row2,
            row3,
            _as_shared_cta(mbar).data_ptr(),
            ctm.Uint16(ctm.Int16(1) << ctm.Int16(rank)),
        ],
    )


@cute.jit
def _s34_q_rows_to_tmem(sQ, sQSF, base: ctm.Int32, wq: ctm.Int32, lane: ctm.Int32):
    # epoch_s34: the per-warp body of _ds_q_to_tmem for TMEM row block wq (0..3):
    # ten QSF STTM columns [DS_QSF_TMEM, +10) and the Q tail permutation of rows
    # wq*32 .. wq*32+31. Called with wq = warp by warps 0-3 at kernel start (FULL-only:
    # the drain-time call by warps 8-11 is gone). The eight core QSF columns are the raw
    # image; the two tail QSF columns and the Q tail rows are written in the tail_k128
    # order [Qr | Qm] (below). sQSF may be any SMEM view holding the 5,120 B raw QSF image.
    row = wq * ctm.Int32(32) + lane
    # Raw TRT QSF encodes four groups of 32 rows per 512B atom.
    for ck in ctm.range_constexpr(8):
        word_ptr = ctm.inttoptr(
            ctm.Int64(sQSF.data_ptr().toint())
            + ctm.Int64(ck * 512)
            + ctm.Int64(lane * ctm.Int32(16) + wq * ctm.Int32(4)),
            3,
            ctm.Int32,
        )
        prims.tcgen05_st(
            prims.Tcgen05LdStShape.SHAPE_32X32B,
            _ds_tmem_ptr(base, wq * ctm.Int32(32), ctm.Int32(DS_QSF_TMEM + ck)),
            word_ptr.load(),
        )
    tail_ptr = ctm.Int64(sQSF.data_ptr().toint()) + ctm.Int64(8 * 512)
    offset = ctm.Int64(lane * ctm.Int32(16) + wq * ctm.Int32(4))
    even = ctm.inttoptr(tail_ptr + offset, 3, ctm.Int32).load()
    odd = ctm.inttoptr(tail_ptr + offset + ctm.Int64(512), 3, ctm.Int32).load()
    for ck in ctm.range_constexpr(2):
        # tail_k128: the tail image is [Qr | Qm]: column + 8 = the four residual scales (odd bytes), + 9 = the main
        # ones.
        selector: ctm.Constexpr = (
            TRTLLM_QSF_ODD_PRMT_SELECTOR if ck == 0 else TRTLLM_QSF_EVEN_PRMT_SELECTOR
        )
        word = prims.prmt(even, selector, prims.PermuteMode.DEFAULT, hi=odd)
        prims.tcgen05_st(
            prims.Tcgen05LdStShape.SHAPE_32X32B,
            _ds_tmem_ptr(base, wq * ctm.Int32(32), ctm.Int32(DS_QSF_TMEM + 8 + ck)),
            word,
        )
    # Q tail rows of this warp, permuted in place in SMEM.
    tail = sQ.subview(128 * 512 // 2 // 4)
    swizzle = ctm.Swizzle.from_name("s64b")
    ptr = tail.data_ptr() + row * ctm.Int32(16)
    # tail_k128: raw 8-B units [m0, r0, m1, r1, m2, r2, m3, r3] -> [r0, r1, r2, r3 | m0, m1, m2, m3] = [Qr | Qm]:
    # position p takes raw unit (1, 3, 5, 7, 0, 2, 4, 6)[p] = the cycles (0 1 3 7 6 4)(2 5). The 6-cycle is issued as
    # six loads then six stores (the base shape: at most six units live), then the 2-cycle.
    u1 = (ptr + ctm.Int32(2)).load_swizzled(swizzle, alignment=8, count=2)
    u3 = (ptr + ctm.Int32(6)).load_swizzled(swizzle, alignment=8, count=2)
    u7 = (ptr + ctm.Int32(14)).load_swizzled(swizzle, alignment=8, count=2)
    u6 = (ptr + ctm.Int32(12)).load_swizzled(swizzle, alignment=8, count=2)
    u4 = (ptr + ctm.Int32(8)).load_swizzled(swizzle, alignment=8, count=2)
    u0 = (ptr + ctm.Int32(0)).load_swizzled(swizzle, alignment=8, count=2)
    (ptr + ctm.Int32(0)).store_swizzled(u1, swizzle=swizzle, alignment=8)
    (ptr + ctm.Int32(2)).store_swizzled(u3, swizzle=swizzle, alignment=8)
    (ptr + ctm.Int32(6)).store_swizzled(u7, swizzle=swizzle, alignment=8)
    (ptr + ctm.Int32(14)).store_swizzled(u6, swizzle=swizzle, alignment=8)
    (ptr + ctm.Int32(12)).store_swizzled(u4, swizzle=swizzle, alignment=8)
    (ptr + ctm.Int32(8)).store_swizzled(u0, swizzle=swizzle, alignment=8)
    u5 = (ptr + ctm.Int32(10)).load_swizzled(swizzle, alignment=8, count=2)
    u2 = (ptr + ctm.Int32(4)).load_swizzled(swizzle, alignment=8, count=2)
    (ptr + ctm.Int32(4)).store_swizzled(u5, swizzle=swizzle, alignment=8)
    (ptr + ctm.Int32(10)).store_swizzled(u2, swizzle=swizzle, alignment=8)
    cute.arch.fence_view_async_tmem_store()


@cute.jit
def _ds_q_to_tmem(
    sQ,
    sQSF,
    base: ctm.Int32,
    warp: ctm.Int32,
    lane: ctm.Int32,
    pairbar,
    rank: ctm.Int32,
    relay,
    issue_warp: ctm.Constexpr = 4,
    cta_threads: ctm.Constexpr = 256,
):
    if warp < 4:
        _s34_q_rows_to_tmem(sQ, sQSF, base, warp, lane)
    prims.barrier(barrier_id=0, number_of_threads=cta_threads)
    _fence_async_shared_cta()
    _ds_pair_sync(pairbar, rank, ctm.Int32(0), warp, relay, issue_warp, cta_threads)
    # C92: Q stays in SMEM after the tail permutation (tail_k128 order [Qr | Qm]).
    # QSF STTM completion plus the pair rendezvous above publishes Q/QSF.


# v9: frozen register-score helpers, integrated without math changes.
@ctm.dsl_user_op
def _c_opaque_i32(value, *, loc=None, ip=None):
    # C_issuer_streamlining v9: identity mov that LLVM cannot see through or hoist (side
    # effects), so `base + const` TMEM operand addresses derived from it are recomputed
    # (one UIADD3) inside the issuer loop instead of being kept live in uniform registers
    # for the whole kernel: UR-pressure relief for the packet descriptors (fewer R2UR).
    result = llvm.inline_asm(
        ctm.Int32.mlir_type,
        [ctm.Int32(value).ir_value(loc=loc, ip=ip)],
        "mov.b32 $0, $1;",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return ctm.Int32(result)


@ctm.dsl_user_op
def _ds_release_cookie_scale(sReleaseCookie, scale, *, loc=None, ip=None):
    # One zero word is initialized before the existing setup cluster barrier.
    # The volatile load and bitwise XOR must survive in actual SASS AFTER
    # score_empty; source order/side effects alone do not prove that ordering.
    cookie_addr = sReleaseCookie.data_ptr().toint(ctm.Int32)
    scale_bits = scale.bitcast(ctm.Int32)
    result = llvm.inline_asm(
        ctm.Int32.mlir_type,
        [cookie_addr.ir_value(loc=loc, ip=ip), scale_bits.ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 cookie;\nld.volatile.shared.u32 cookie, [$1];\nxor.b32 $0, $2, cookie;\n}",
        "=r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return ctm.Int32(result).bitcast(ctm.Float32)


@ctm.dsl_user_op
def _ds_sf_bias_fma_rn(newanchor, scale, *, loc=None, ip=None):
    # Force float-register constraints. The generic inline_ptx lowering chose
    # integer-immediate 'n' for this Float32 constant in frozen v11b LLVM IR.
    negative_scale = -scale
    negative_log2_6 = ctm.Float32(-LOG2_6)
    result = llvm.inline_asm(
        ctm.Float32.mlir_type,
        [
            newanchor.ir_value(loc=loc, ip=ip),
            negative_scale.ir_value(loc=loc, ip=ip),
            negative_log2_6.ir_value(loc=loc, ip=ip),
        ],
        "fma.rn.f32 $0, $1, $2, $3;",
        "=f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return ctm.Float32(result)


@ctm.dsl_user_op
def _ds_release_cookie_scale_and_bound(sReleaseCookie, scale, valid_k, *, loc=None, ip=None):
    # Same one zero-cookie load as the original scale helper, but the tail mask
    # bound also depends on it. Core keeps its original helper and call.
    cookie_addr = sReleaseCookie.data_ptr().toint(ctm.Int32)
    scale_bits = scale.bitcast(ctm.Int32)
    result = llvm.inline_asm(
        llvm.StructType.get_literal([ctm.Int32.mlir_type, ctm.Int32.mlir_type]),
        [
            cookie_addr.ir_value(loc=loc, ip=ip),
            scale_bits.ir_value(loc=loc, ip=ip),
            valid_k.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b32 cookie;\n"
        "ld.volatile.shared.u32 cookie, [$2];\n"
        "xor.b32 $0, $3, cookie;\n"
        "xor.b32 $1, $4, cookie;\n}",
        "=&r,=r,r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    released_scale = ctm.Int32(
        llvm.extractvalue(ctm.Int32.mlir_type, result, position=[0], loc=loc, ip=ip)
    ).bitcast(ctm.Float32)
    released_bound = ctm.Int32(
        llvm.extractvalue(ctm.Int32.mlir_type, result, position=[1], loc=loc, ip=ip)
    )
    return released_scale, released_bound


@cute.jit
def _ds_load_reg_score_half_and_release(
    base: ctm.Int32,
    warp: ctm.Int32,
    score_empty,
    sReleaseCookie,
    scale: ctm.Float32,
    mask_valid_k: ctm.Constexpr,
):
    # Caller has waited score_full(tile&1) and applied AFTER_THREAD_SYNC.
    # The two halves are columns of this tile, not alternating S banks.
    rowbase = (warp & ctm.Int32(3)) * ctm.Int32(32)
    half = warp >> ctm.Int32(2)
    pending_groups = []
    for g in ctm.range_constexpr(8):
        col = half * ctm.Int32(128) + ctm.Int32(g * 16)
        score_ptr = _ds_tmem_ptr(base, rowbase, col)
        if cutlass.const_expr(mask_valid_k):
            values = prims.tcgen05_ld(prims.Tcgen05LdStShape.SHAPE_32X32B, score_ptr, num=16)
        else:
            values = _tcgen05_ld_red_32x32b_x16_max_f32(score_ptr)
        pending_groups.append(values)
    # Match the old last-read ordering: all local score loads complete BEFORE
    # releasing S. No max/mask/exp/pack/P-source credit is on this side.
    cute.arch.fence_view_async_tmem_load()
    if prims.elect_sync():
        _mbarrier_arrive_release_cta_shared_cluster(
            _mapa_shared_cluster(score_empty, cute.arch.block_idx_in_cluster() & ctm.Int32(2))
        )
    # All128 score values (plus8 STAT maxima on the non-tail path) must remain
    # register values after this notification. Actual SASS/spills must verify it.
    release_scale = _ds_release_cookie_scale(sReleaseCookie, scale)
    return tuple(pending_groups), release_scale


@cute.jit
def _ds_load_reg_tail_score_half_and_release(
    base: ctm.Int32,
    warp: ctm.Int32,
    score_empty,
    sReleaseCookie,
    scale: ctm.Float32,
    valid_k: ctm.Int32,
    mask_valid_k: ctm.Constexpr,
):
    # Caller has waited score_full(tile&1) and applied AFTER_THREAD_SYNC.
    # The two halves are columns of this tile, not alternating S banks.
    rowbase = (warp & ctm.Int32(3)) * ctm.Int32(32)
    half = warp >> ctm.Int32(2)
    pending_groups = []
    for g in ctm.range_constexpr(8):
        col = half * ctm.Int32(128) + ctm.Int32(g * 16)
        score_ptr = _ds_tmem_ptr(base, rowbase, col)
        if cutlass.const_expr(mask_valid_k):
            values = prims.tcgen05_ld(prims.Tcgen05LdStShape.SHAPE_32X32B, score_ptr, num=16)
        else:
            values = _tcgen05_ld_red_32x32b_x16_max_f32(score_ptr)
        pending_groups.append(values)
    # Match the old last-read ordering: all local score loads complete BEFORE
    # releasing S. No max/mask/exp/pack/P-source credit is on this side.
    cute.arch.fence_view_async_tmem_load()
    if prims.elect_sync():
        _mbarrier_arrive_release_cta_shared_cluster(
            _mapa_shared_cluster(score_empty, cute.arch.block_idx_in_cluster() & ctm.Int32(2))
        )
    # All128 score values (plus8 STAT maxima on the non-tail path) must remain
    # register values after this notification. Actual SASS/spills must verify it.
    release_scale, release_valid_k = _ds_release_cookie_scale_and_bound(
        sReleaseCookie, scale, valid_k
    )
    return tuple(pending_groups), release_scale, release_valid_k


@cute.jit
def _r93_replay_peer(
    sBlockMax,
    bars,
    peer_index: ctm.Int32,
    peer_count: ctm.Int32,
    rank: ctm.Int32,
    warp: ctm.Int32,
    lane: ctm.Int32,
    running_max: ctm.Float32,
    running_anchor: ctm.Float32,
    scale: ctm.Float32,
):
    bank = peer_index & ctm.Int32(1)
    phase = (peer_index // ctm.Int32(2)) & ctm.Int32(1)
    full = bars.subview(ctm.Int32(21) + bank)
    _ds_wait(full, phase)
    row = (warp & ctm.Int32(3)) * ctm.Int32(32) + lane
    blockmax = sBlockMax[bank * ctm.Int32(128) + row]
    # All eight reader warps finish their rows before the last credit can
    # authorize overwrite. Warp0 re-arms before contributing its credit.
    prims.bar_warp_sync(cute.arch.FULL_MASK)
    if peer_index + ctm.Int32(2) < peer_count:
        if warp == ctm.Int32(0):
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(full, 512)
        if prims.elect_sync():
            _mbarrier_arrive_release_cta_shared_cluster(
                _mapa_shared_cluster(bars.subview(ctm.Int32(23) + bank), rank ^ ctm.Int32(2))
            )
    newmax = cute.arch.fmax(running_max, blockmax)
    anchor, rebase, delta = _select_smem_p4_lazy_anchor_with_validity(
        newmax, ctm.Boolean(newmax != ctm.Float32(-ctm.Float32.inf)), running_anchor, scale
    )
    peer_oldscale = _compute_smem_p4_lazy_rebase_scale(rebase, delta)
    return newmax, anchor, peer_oldscale


@cute.jit
def _r93_prep_pv(
    sPayloadBanks,
    bars,
    base: ctm.Int32,
    local_tile: ctm.Int32,
    tile: ctm.Int32,
    pair_id: ctm.Int32,
    rank: ctm.Int32,
    warp: ctm.Int32,
    lane: ctm.Int32,
    own_sf0: ctm.Int32,
    own_sf1: ctm.Int32,
    own_oldscale: ctm.Float32,
    is_local: ctm.Constexpr,
    chunk_tiles: ctm.Int32,
):
    bank = pair_id if cutlass.const_expr(is_local) else (pair_id ^ ctm.Int32(1))
    phase = (local_tile // ctm.Int32(2)) & ctm.Int32(1)
    half = warp >> ctm.Int32(2)
    rowbase = (warp & ctm.Int32(3)) * ctm.Int32(32)
    _row = rowbase + lane  # kept from the x79 lineage; unused in the FULL-only variant
    oldscale = own_oldscale
    if cutlass.const_expr(not is_local):
        _ds_wait(bars.subview(DS_BAR_META_LANDED), phase)
        _fence_async_shared_cta()
        # The peer prep of a task's last tile runs only in the drain round of the pair whose last tile is a peer tile.
        # That pair's final wait (bars[11 + (last & 1)], the commit of PV(last)) is a parity wait, so it is exact only
        # once the previous commit on the same barrier, PV(last - 2), has arrived, and nothing else orders that commit
        # before the softmax warps get there.  Wait the pair's own PV(local_tile - 1) here always, not only on a
        # rebase: its previous generation is the own PV(local_tile - 3), which _r93_pack of the last own tile waited,
        # and PV(local_tile - 2) is issued ahead of it by the same issuer thread, so its commit has arrived as well.
        if local_tile + ctm.Int32(1) == chunk_tiles:
            if local_tile > ctm.Int32(0):
                drain_prev = local_tile - ctm.Int32(1)
                _ds_wait(
                    bars.subview(ctm.Int32(11) + (drain_prev & ctm.Int32(1))),
                    (drain_prev // ctm.Int32(2)) & ctm.Int32(1),
                )
        # The landed [PSF|META] bank is re-armed (bars[META_LANDED] expect_tx) and released to its producer (remote
        # bars[META_FREE]) by the issuer in _r93_issue_pv: the re-arm after its own bars[META_LANDED] wait, the
        # release by a tcgen05.commit queued behind its PSF tcgen05.cp (_r95_complete_pv is empty).  The TMEM PSF
        # bank of a peer tile is written by that issuer tcgen05.cp, not by the ALU warps, so no wait for the bank's
        # last reader PV(local_tile-2) (bars[11+bank], previous phase) is needed here.  The copy is ordered behind
        # PV(local_tile-2) twice: the issuer first waits bars[8], and P(local_tile) lands only after the producer
        # got the bar7 credit that PV(local_tile-2) commits behind its MMAs; and the copy is submitted by the
        # thread that submitted those MMAs (one thread's tcgen05 operations execute in order).  The issuer's
        # V-landed wait is NOT such a guarantee: with DS_V_BANKS V banks it implies only
        # PV(local_tile - DS_V_BANKS) done.
    if tile > ctm.Int32(0):
        if cute.arch.vote_any_sync(oldscale != ctm.Float32(1.0)):
            if local_tile > ctm.Int32(0):
                previous = local_tile - ctm.Int32(1)
                _ds_wait(
                    bars.subview(ctm.Int32(11) + (previous & ctm.Int32(1))),
                    (previous // ctm.Int32(2)) & ctm.Int32(1),
                )
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            tp = ctm.inttoptr(
                base + (rowbase << ctm.Int32(16)) + ctm.Int32(256) + half * ctm.Int32(128),
                6,
                ctm.Float32,
            )
            _rescale_smem_p4_resident_o_in_tmem(tp, oldscale)
            if half == ctm.Int32(0):  # rsplit: both pairs rescale their own rowsum partial
                sp = _ds_tmem_ptr(base, rowbase, ctm.Int32(DS_ROWSUM_ACC_TMEM))
                value = prims.tcgen05_ld(prims.Tcgen05LdStShape.SHAPE_32X32B, sp, num=1)
                cute.arch.fence_view_async_tmem_load()
                prims.tcgen05_st(
                    prims.Tcgen05LdStShape.SHAPE_32X32B,
                    sp,
                    value[0].bitcast(ctm.Float32) * oldscale,
                )
    if cutlass.const_expr(is_local):
        # C_issuer_streamlining v7: own tile: the eight ALU warps store their PSF rows
        # (the same two E4M3x4 words each warp just stored into the SMEM [PSF|META] image
        # for the DSMEM copy) straight into the TMEM SFA bank [DS_PSF_TMEM + bank*4, +4):
        # lane = row (this warp's 32 rows), column = K64 chunk (half*2 + {0,1}). This is the
        # byte-for-byte image the issuer's PSF tcgen05.cp (removed for own tiles) produced,
        # so the PV MMAs read identical scales. WAR: the bank was last read by PV(own
        # tile-1) = PV(local_tile-2), whose completion (bars[11+bank], previous phase)
        # _r93_pack already waited before storing P. Visibility: tcgen05.wait::st +
        # tcgen05.fence::before_thread_sync below precede the bars[9+bank] arrive; the
        # issuer applies tcgen05.fence::after_thread_sync after its wait / handshake.
        # Orders the store behind PV(local_tile-2)'s MMAs, whose commit barrier _r93_pack waited.
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        psf_words = ctm.Vector.from_elements((own_sf0, own_sf1), dtype=ctm.Int32)
        prims.tcgen05_st(
            prims.Tcgen05LdStShape.SHAPE_32X32B,
            _ds_tmem_ptr(
                base, rowbase, ctm.Int32(DS_PSF_TMEM) + bank * ctm.Int32(4) + half * ctm.Int32(2)
            ),
            psf_words,
        )
    cute.arch.fence_view_async_tmem_store()
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    if prims.elect_sync():
        prims.mbarrier_arrive(bars.subview(ctm.Int32(9) + bank))


# PV-only raw helpers: caller must select one thread after all collective waits.
@cute.jit
def _ep_copy_bsf_elected(
    sSF,
    base: ctm.Int32,
    col: ctm.Constexpr,
    chunks: ctm.Constexpr,
    ntiles: ctm.Constexpr = 1,
    first_nt: ctm.Constexpr = 0,
    source_ntiles: ctm.Constexpr = 1,
    first_ck: ctm.Constexpr = 0,
):
    # The input contains one 512B packed atom per 128 N rows and 64 K.
    # M256 uses all four DP bands, so expand each atom to four TMEM columns.
    shape, multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
    # Invariant unswizzled SF descriptor; offsets stay within its start field.
    desc_root = prims.Tcgen05SmemDesc.build(
        sSF, leading_byte_offset=16, stride_byte_offset=128, layout=0
    )
    for nt in ctm.range_constexpr(ntiles):
        for ck in ctm.range_constexpr(chunks):
            for nb in ctm.range_constexpr(2):
                # first_ck selects source K chunks; destination remains locally packed.
                src_byte: ctm.Constexpr = (
                    ((first_ck + ck) * source_ntiles + nt + first_nt) * 2 + nb
                ) * 512
                desc = desc_root.advance_start_address(src_byte)
                target = _ds_tmem_ptr(
                    base, ctm.Int32(0), ctm.Int32(col + nt * chunks * 8 + ck * 8 + nb * 4)
                )
                prims.tcgen05_cp(
                    shape, target, desc, group=prims.CTAGroup.CTA_2, multicast=multicast
                )


@cute.jit
def _ep_mma_elected(
    sA,
    sB,
    base: ctm.Int32,
    acc: ctm.Constexpr,
    acol: ctm.Constexpr,
    asf: ctm.Constexpr,
    bsf: ctm.Constexpr,
    k64: ctm.Constexpr,
    k128: ctm.Constexpr,
    clear: ctm.Boolean,
    a_tmem: ctm.Constexpr,
    a_byte: ctm.Constexpr = 0,
    b_byte: ctm.Constexpr = 0,
    swizzle64: ctm.Constexpr = False,
    n_dim: ctm.Constexpr = 256,
    a_collector_op: ctm.Constexpr = None,
    asf_offset: ctm.Int32 = 0,
    shared_q: ctm.Constexpr = False,
):
    desc = prims.Tcgen05MxOmmaInstrDesc.build(
        a_dtype=ctm.Float4E2M1FN,
        b_dtype=ctm.Float4E2M1FN,
        scale_format=0,
        n_dim=n_dim,
        m_dim=256,
        k_dim=2 if k128 else 0,
        a_sf_layout=1,
        b_major=0,
    )
    bdesc = prims.Tcgen05SmemDesc.build(
        sB.subview(b_byte // 4),
        leading_byte_offset=0 if swizzle64 else 16,
        stride_byte_offset=512 if swizzle64 else 1024,
        layout=4 if swizzle64 else 2,
    )
    if cutlass.const_expr(a_tmem):
        aoperand = _ds_tmem_ptr(base, ctm.Int32(0), ctm.Int32(acol))
    else:
        if cutlass.const_expr(shared_q):
            # Q core is s128b; its pre-permuted tail is s64b. Encoded
            # address advance preserves descriptor fields at each K step.
            aoperand = prims.Tcgen05SmemDesc.build(
                sA,
                leading_byte_offset=0 if swizzle64 else 16,
                stride_byte_offset=512 if swizzle64 else 1024,
                layout=4 if swizzle64 else 2,
            ).advance_start_address(a_byte)
        else:
            aoperand = prims.Tcgen05SmemDesc.build(
                sA.subview(a_byte // 4),
                leading_byte_offset=16,
                stride_byte_offset=1024,
                layout=2,
            )
    prims.tcgen05_mma_block_scale(
        prims.MMABlockScaleKind.MXF4NVF4,
        prims.CTAGroup.CTA_2,
        _ds_tmem_ptr(base, ctm.Int32(0), ctm.Int32(acc)),
        aoperand,
        bdesc,
        desc,
        enable_input_d=~clear,
        scale_a=_ds_tmem_ptr(base, ctm.Int32(0), ctm.Int32(asf) + asf_offset),
        scale_b=_ds_tmem_ptr(base, ctm.Int32(0), ctm.Int32(bsf)),
        scale_vec_size=prims.Tcgen05MMAScaleVecSize.BLOCK16,
        collector_op=a_collector_op,
    )


@cute.jit
def _r93_issue_pv(
    sPayloadBanks,
    sV,
    sVSF,
    sUnit,
    bars,
    base: ctm.Int32,
    local_tile: ctm.Int32,
    tile: ctm.Int32,
    chunk_tiles: ctm.Int32,
    pair_id: ctm.Int32,
    rank: ctm.Int32,
    pair_phase: ctm.Int32,
    is_local: ctm.Constexpr,
    relay,
):
    bank = pair_id if cutlass.const_expr(is_local) else (pair_id ^ ctm.Int32(1))
    phase = (local_tile // ctm.Int32(2)) & ctm.Int32(1)
    sPayload = sPayloadBanks.subview(
        ctm.Int32(PB_OWN if cutlass.const_expr(is_local) else PB_LAND)
        * ctm.Int32(DS_PAYLOAD_BYTES // 4)
    )
    # rsplit: only own-tile packets carry the rowsum MMA that LASTUSEs the collector.
    o_collector = prims.Tcgen05MMACollectorOp.FILL if cutlass.const_expr(is_local) else None
    vbank = local_tile & ctm.Int32(DS_V_BANKS - 1)
    sVStage = sV.subview(vbank * ctm.Int32(DS_V_BANK_BYTES // 4))
    sVSFStage = sVSF.subview(vbank * ctm.Int32(DS_VSF_BANK_BYTES))
    # ur_relief: relayed mapa(&bars[1], rank ^ 1) / mapa(&bars[6], rank ^ 2), read
    # before the waits so the LDS latency hides behind them.
    pair_remote = relay[UR_RELAY_PAIR_BAR]
    # C_issuer_streamlining v1: the V/VSF landing wait (TMA, lands long before the P
    # prep) is taken FIRST so that the ALU-driven prep-ready wait (bars[9+bank]) is the
    # last wait before the pair handshake; parity waits are non-consuming so the order
    # among a packet's waits is free.
    _ds_wait(
        bars.subview(ctm.Int32(DS_BAR_V_LANDED) + vbank),
        (local_tile // ctm.Int32(DS_V_BANKS)) & ctm.Int32(1),
    )
    if cutlass.const_expr(not is_local):
        credit_remote = relay[UR_RELAY_P_CREDIT_BAR]
        # Note: this credit protects only P. The landed metadata bank has its
        # own credit (the issuer releases it by the commit behind its PSF copy, below).
        _ds_wait(bars.subview(8), phase)
        if prims.elect_sync():
            _ur_arrive_remote(credit_remote)
        # land_commit: re-arm the P landing barrier for the next peer tile here (both CTAs), the cp_dual_v3
        # pattern of bars[META_LANDED] below. The producer's next copy waits for our bar7 credit, which this
        # change issues as a commit after this packet's MMAs (the only readers of the landed P), so the re-arm
        # precedes it.
        if local_tile + ctm.Int32(2) < chunk_tiles:
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(bars.subview(8), DS_P_BYTES)
        # cp_dual: the issuer's tcgen05.cp reads the landed [PSF|META] bank
        # (async-proxy bulk copy completing on bars[META_LANDED]). Same parity as the ALU
        # peer prep; parity waits are non-consuming, so a second waiter is legal.
        _ds_wait(bars.subview(DS_BAR_META_LANDED), phase)
        # cp_dual_v3: re-arm the landing barrier for the next peer tile here
        # (both CTAs). The producer's next copy waits for our bars[META_FREE] release,
        # which is issued below after the PSF copy, so the re-arm precedes it.
        if local_tile + ctm.Int32(2) < chunk_tiles:
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(
                    bars.subview(DS_BAR_META_LANDED), DS_PSF_BYTES + DS_META_BYTES
                )
    # C_issuer_streamlining v1: the own-tile bar5 (P + PSF published) wait is implied by
    # bars[9+bank]: every ALU warp arrives on bar5 (_ds_publish_reg_half_payload, after
    # its fence.proxy.async) BEFORE it arrives on bars[9+bank] in _r93_prep_pv(own), in
    # the same ALU round (same parity), and mbarrier arrive/wait are release/acquire.
    _ds_wait(bars.subview(ctm.Int32(9) + bank), phase)
    # V/VSF are async TMA writes. P's generic writers published before
    # prep-ready; peer P is a completed async copy. No issuer generic write.
    if prims.elect_sync():
        prims.mbarrier_arrive(bars.subview(1))
        _ur_arrive_remote(pair_remote)
    _ds_wait(bars.subview(1), pair_phase)
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    if (rank & ctm.Int32(1)) == ctm.Int32(0):
        # One selected thread submits the peer tile's PSF copy, both PV K terms, adjacent
        # rowsums, and commit.
        if prims.elect_sync():
            if cutlass.const_expr(not is_local):
                # cp_dual: peer tile: one tcgen05.cp.cta_group::2.128x128b moves the landed 128-row x 16 B PSF
                # image (row-major; word c = E4M3x4 scales of K64 chunk c, exactly as the producer's _r93_pack
                # stored it and the DSMEM copy landed it) from each CTA's landing bank into the TMEM SFA bank
                # [DS_PSF_TMEM + bank*4, +4): lane = row, column = chunk.  LBO 16 / SBO 128 / no swizzle is the
                # descriptor of every SF copy in this kernel.  Issued first in the packet: the VSF copies queue
                # behind it (one UTCCP at a time) and the kb=0 MMA's SFA operand exact-matches its destination.
                # Its completion is covered by this packet's commit (bar11 + bank), like the VSF copies.
                # C_issuer_streamlining v7: an own tile has no PSF tcgen05.cp: the ALU warps tcgen05.st'd the
                # same image before arriving on bars[9+bank] (_r93_prep_pv), so its VSF copies and kb=0 MMA
                # start right away.
                psf_shape, psf_multicast = prims.S2TCopyMode.S2T_128x128b
                psf_tmem = _ds_tmem_ptr(
                    base, ctm.Int32(0), ctm.Int32(DS_PSF_TMEM) + bank * ctm.Int32(4)
                )
                prims.tcgen05_cp(
                    psf_shape,
                    psf_tmem,
                    prims.Tcgen05SmemDesc.build(
                        sPayload.subview(ctm.Int32(DS_P_BYTES // 4)),
                        leading_byte_offset=16,
                        stride_byte_offset=128,
                        layout=0,
                    ),
                    group=prims.CTAGroup.CTA_2,
                    multicast=psf_multicast,
                )
                # cp_dual_v3: release the landed [PSF|META] bank to its producer
                # (the other pair) as soon as this copy has completed: one
                # arrival on each of their bars[META_FREE] (count 1). The ALU oldscale
                # reads of this bank finished before bar9, hence before now.
                prims.tcgen05_commit(
                    bars.subview(DS_BAR_META_FREE),
                    multicast_mask=ctm.Int16(3)
                    << ctm.Int16((pair_id ^ ctm.Int32(1)) * ctm.Int32(2)),
                    group=prims.CTAGroup.CTA_2,
                )
            for kb in ctm.range_constexpr(2):
                _ep_copy_bsf_elected(sVSFStage, base, DS_KSF_TMEM, 2, first_ck=kb * 2)
                _ep_mma_elected(
                    sPayload,
                    sVStage,
                    base,
                    256,
                    0,
                    DS_PSF_TMEM + kb * 2,
                    DS_KSF_TMEM,
                    kb * 2,
                    True,
                    ctm.Boolean(local_tile == 0) & ctm.Boolean(kb == 0),
                    False,  # split mode: clear O on the task's first tile
                    a_byte=kb * 64,
                    b_byte=kb * DS_V_BANK_BYTES // 2,
                    swizzle64=True,
                    a_collector_op=o_collector,
                    asf_offset=bank * ctm.Int32(4),
                )
                # Independent accumulators: retain K0,K1 order separately
                # for O and rowsum while placing equal-A operations adjacent.
                # Reuse the immediately preceding P K128 matrix for rowsum.
                # rsplit: each pair accumulates the rowsum of its OWN tiles only (pair 0 even,
                # pair 1 odd) in its own TMEM column 512; partials are added at the epilogue.
                # Pair 1's column starts from the kernel-start zero store.
                if cutlass.const_expr(is_local):
                    _ep_mma_elected(
                        sPayload,
                        sUnit,
                        base,
                        DS_ROWSUM_ACC_TMEM,
                        0,
                        DS_PSF_TMEM + kb * 2,
                        DS_ROWSUM_SF_TMEM,
                        kb * 2,
                        True,
                        ctm.Boolean(local_tile == 0) & ctm.Boolean(kb == 0),
                        False,  # split mode: clear O on the task's first tile
                        a_byte=kb * 64,
                        swizzle64=True,
                        n_dim=16,
                        a_collector_op=prims.Tcgen05MMACollectorOp.LASTUSE,
                        asf_offset=bank * ctm.Int32(4),
                    )
            prims.tcgen05_commit(
                bars.subview(ctm.Int32(11) + bank),
                multicast_mask=ctm.Int16(3) << ctm.Int16(pair_id * ctm.Int32(2)),
                group=prims.CTAGroup.CTA_2,
            )
            # The V bank this packet read is free for tile local_tile + DS_V_BANKS (waited in _vh2_issue).  The last
            # DS_V_BANKS tiles of a task have no such waiter; the commit is not guarded, to keep a compare and a
            # branch out of the packet (see the kernel-end note on unwaited releases).
            prims.tcgen05_commit(
                bars.subview(ctm.Int32(DS_BAR_V_FREE) + vbank),
                multicast_mask=ctm.Int16(3) << ctm.Int16(pair_id * ctm.Int32(2)),
                group=prims.CTAGroup.CTA_2,
            )
            if cutlass.const_expr(not is_local):
                if local_tile + ctm.Int32(2) < chunk_tiles:
                    # land_commit: landing-bank credit straight to the producer pair's bar7 (count 1, one arrive
                    # per CTA of that pair) when this packet's MMAs have completed -- the bars[META_FREE] pattern of
                    # cp_dual_v3 above. Replaces warp 9's relayed remote arrive (_r95_complete_pv).
                    prims.tcgen05_commit(
                        bars.subview(7),
                        multicast_mask=ctm.Int16(3)
                        << ctm.Int16((pair_id ^ ctm.Int32(1)) * ctm.Int32(2)),
                        group=prims.CTAGroup.CTA_2,
                    )
    return pair_phase ^ ctm.Int32(1)


@cute.jit
def _r95_complete_pv(
    bars, local_tile: ctm.Int32, chunk_tiles: ctm.Int32, pair_id: ctm.Int32, rank: ctm.Int32, relay
):
    # The existing copy warp observes completion and relays ownership. The
    # issuer can submit independent TCGEN work while the prior PV runs.
    # land_commit: the peer tile's landing re-arm and credit are done by the issuer (bar8 re-arm after its
    # wait; bar7 of the producer pair by a commit after the packet's MMAs), like bars[META_LANDED] /
    # bars[META_FREE] since cp_dual_v3.
    # Only the own tile's source credit (bar6) is still relayed here.
    # x9: no relay (the ALU waits the commit barrier itself)
    pass


@cute.jit
def _r93_pack(
    pending_groups,
    sHalfMax,
    sPayload,
    source_done,
    warp: ctm.Int32,
    lane: ctm.Int32,
    tile: ctm.Int32,
    local_tile: ctm.Int32,
    valid_k: ctm.Int32,
    scale: ctm.Float32,
    p_scale: ctm.Float32,
    running_max: ctm.Float32,
    running_anchor: ctm.Float32,
    mask_valid_k: ctm.Constexpr,
    use_mixed_imlp: ctm.Constexpr,
    sBlockMax,
    bars,
    rank: ctm.Int32,
    own_index: ctm.Int32,
    peer_count: ctm.Int32,
    sPsfMetaAlt,
):
    half = warp >> ctm.Int32(2)
    row = (warp & ctm.Int32(3)) * ctm.Int32(32) + lane
    half_base = half * ctm.Int32(128)
    local_max = ctm.Float32(-ctm.Float32.inf)
    group_maxima = []
    packed_p4_words = []
    # Compute the same maxima from register-staged scores; no TMEM access.
    for g in ctm.range_constexpr(8):
        vals = pending_groups[g]
        group_base = tile * ctm.Int32(256) + half_base + ctm.Int32(g * 16)
        if cutlass.const_expr(mask_valid_k):
            group_max = ctm.Float32(-ctm.Float32.inf)
            for j in ctm.range_constexpr(16):
                value = _mask_invalid_score(
                    vals[j].bitcast(ctm.Float32), group_base + ctm.Int32(j), valid_k
                )
                group_max = cute.arch.fmax(group_max, value)
        else:
            group_max = vals[16].bitcast(ctm.Float32)
        group_maxima.append(group_max)
        local_max = cute.arch.fmax(local_max, group_max)
    # Publish after four groups: retain overlap while releasing half of
    # the score registers before the existing64-thread joins.
    blockmax = ctm.Float32(-ctm.Float32.inf)
    # alu_chain P2: halfmax stage buffer for this own tile (stage 1 = QSF + 4096).
    hm_base = ctm.Int64(sHalfMax.data_ptr().toint())
    if (own_index & ctm.Int32(1)) != ctm.Int32(0):
        hm_base = hm_base + ctm.Int64(4096)
    hm_ptr = ctm.inttoptr(hm_base, 3, ctm.Float32)
    for g in ctm.range_constexpr(8):
        vals = pending_groups[g]
        group_base = tile * ctm.Int32(256) + half_base + ctm.Int32(g * 16)
        group_max = group_maxima[g]
        scores = ctm.Vector.from_elements(
            tuple(vals[j].bitcast(ctm.Float32) for j in range(16)), dtype=ctm.Float32
        )
        group_valid = ctm.Boolean(group_base < valid_k)
        p_bias = ctm.Float32(LOG2_6) - group_max * scale
        if cutlass.const_expr(use_mixed_imlp):
            word0, word1 = _pack_p4_16_scores_from_group_packed_f16x2(
                scores, group_base, valid_k, group_valid, mask_valid_k, scale, p_bias, elem_base=0
            )
        else:
            word0, word1 = _pack_p4_16_scores_from_group(
                scores,
                group_base,
                valid_k,
                group_valid,
                mask_valid_k,
                use_mixed_imlp,
                scale,
                p_bias,
                elem_base=0,
            )
        packed_p4_words.append(word0)
        packed_p4_words.append(word1)
        if cutlass.const_expr(g == 3):
            # alu_chain P2: two-stage halfmax buffer (stage = own_index & 1; stage 1
            # lives in the dead QSF bytes [4096, 5120)) and an arrive/sync split:
            # this warp publishes its half-row maxima and arrives on named barrier
            # 2 + rowgroup + 4*half without blocking; the partner's value is read
            # after group 7. The two-stage buffer makes the former second barrier
            # (WAR on the shared row) unnecessary.
            (hm_ptr + half * ctm.Int32(128) + row).store(local_max)
            prims.barrier_cta_arrive(
                ctm.Int32(2) + (warp & ctm.Int32(3)) + ctm.Int32(4) * half, thread_count=64
            )
    # Wait only for the row partner's arrival (its halfmax store), then combine.
    prims.barrier_cta_sync(
        ctm.Int32(2) + (warp & ctm.Int32(3)) + ctm.Int32(4) * (half ^ ctm.Int32(1)), thread_count=64
    )
    blockmax = cute.arch.fmax((hm_ptr + row).load(), (hm_ptr + ctm.Int32(128) + row).load())
    # Independent blockmax publication precedes the prior-peer state wait.
    # Consecutive physical lanes store consecutive FP32 rows (+4B/lane).
    if half == ctm.Int32(0):
        max_bank = own_index & ctm.Int32(1)
        if own_index >= ctm.Int32(2):
            _ds_wait(
                bars.subview(ctm.Int32(23) + max_bank),
                ((own_index // ctm.Int32(2)) - ctm.Int32(1)) & ctm.Int32(1),
            )
        _store_async_cluster_b32(
            _mapa_shared_cluster(
                sBlockMax.data_ptr() + max_bank * ctm.Int32(128) + row, rank ^ ctm.Int32(2)
            ),
            blockmax.bitcast(ctm.Int32),
            _mapa_shared_cluster(bars.subview(ctm.Int32(21) + max_bank), rank ^ ctm.Int32(2)),
        )
    peer_oldscale = ctm.Float32(1.0)
    if local_tile > ctm.Int32(0):
        running_max, running_anchor, peer_oldscale = _r93_replay_peer(
            sBlockMax,
            bars,
            (local_tile - ctm.Int32(1)) // ctm.Int32(2),
            peer_count,
            rank,
            warp,
            lane,
            running_max,
            running_anchor,
            scale,
        )
    # earlycheck c: the two P-bank credits of the stores below, tested ahead of the PSF computation.
    credit_phase = (own_index - ctm.Int32(1)) & ctm.Int32(1)
    pv_done_seen = _ds_seen(bars.subview(ctm.Int32(11) + (local_tile & ctm.Int32(1))), credit_phase)
    source_done_seen = _ds_seen(source_done, credit_phase)
    newmax = cute.arch.fmax(running_max, blockmax)
    newanchor, rebase, delta = _select_smem_p4_lazy_anchor_with_validity(
        newmax, ctm.Boolean(newmax != ctm.Float32(-ctm.Float32.inf)), running_anchor, scale
    )
    oldscale = _compute_smem_p4_lazy_rebase_scale(rebase, delta)
    # Preserve v10's actual FFMA rounding in BOTH core and tail paths.
    # v11's cookie dependency had separated MUL/SUB across source-credit wait.
    sf_bias = _ds_sf_bias_fma_rn(newanchor, scale)
    # Note: previous remote metadata readers imply source-copy completion
    # and destination SMEM release. Local preparation uses the RF values.
    # cp_dual_v4: no landing-free wait here.  cp_dual_alt: this own tile's [PSF|META] source image alternates by
    # own-index parity between the in-bank region and sPsfMetaAlt.  Its only reader is warp 9's [PSF|META] copy to
    # the other pair (the issuer no longer copies an own tile's PSF: the ALU warps store it into TMEM themselves,
    # _r93_prep_pv).  The copy of own_index-1 may still be pending when these stores run, hence the other buffer.
    # The copy of own_index-2, the last reader of the buffer written now, has landed: the peer's ALU warps waited
    # bars[META_LANDED] for it (_r93_prep_pv, peer tile) in the round before the one whose blockmax
    # _r93_replay_peer consumed above.
    psf_base = ctm.Int64(sPayload.data_ptr().toint()) + ctm.Int64(DS_P_BYTES)
    if (own_index & ctm.Int32(1)) != ctm.Int32(0):
        psf_base = ctm.Int64(sPsfMetaAlt.data_ptr().toint())
    psf_ptr = ctm.inttoptr(psf_base, 3, ctm.Int32)
    own_sf_words = []
    for chunk in ctm.range_constexpr(2):
        if cutlass.const_expr(mask_valid_k):
            sfs = []
            for g in ctm.range_constexpr(4):
                sf = ctm.Float32(0.0)
                if tile * ctm.Int32(256) + half_base + ctm.Int32((chunk * 4 + g) * 16) < valid_k:
                    sf = cute.exp2(group_maxima[chunk * 4 + g] * scale + sf_bias, fastmath=True)
                sfs.append(sf * p_scale)
            sfword = _pack_e4m3x4_natural(sfs[0], sfs[1], sfs[2], sfs[3])
        elif cutlass.const_expr(use_mixed_imlp):
            h01 = _softmax_exp2_pair_packed_f16x2(
                group_maxima[chunk * 4], group_maxima[chunk * 4 + 1], scale, sf_bias
            )
            h23 = _softmax_exp2_pair_packed_f16x2(
                group_maxima[chunk * 4 + 2], group_maxima[chunk * 4 + 3], scale, sf_bias
            )
            sfword = _scale_pack_e4m3x4_natural_from_f16x2(h01, h23, p_scale)
        else:
            args01 = nvvm_fma_packed_f32x2(
                (group_maxima[chunk * 4], group_maxima[chunk * 4 + 1]),
                (scale, scale),
                (sf_bias, sf_bias),
            )
            args23 = nvvm_fma_packed_f32x2(
                (group_maxima[chunk * 4 + 2], group_maxima[chunk * 4 + 3]),
                (scale, scale),
                (sf_bias, sf_bias),
            )
            sfword = _pack_e4m3x4_natural(
                cute.exp2(args01[0], fastmath=True) * p_scale,
                cute.exp2(args01[1], fastmath=True) * p_scale,
                cute.exp2(args23[0], fastmath=True) * p_scale,
                cute.exp2(args23[1], fastmath=True) * p_scale,
            )
        own_sf_words.append(sfword)
        sf_col = half * ctm.Int32(2) + ctm.Int32(chunk)
        (psf_ptr + row * ctm.Int32(4) + sf_col).store(sfword)
    newsum = ctm.Float32(0.0)  # Quantized P*SF denominator stays in PV TMEM.
    if half == ctm.Int32(0):
        meta = ctm.inttoptr(psf_base + ctm.Int64(DS_PSF_BYTES), 3, ctm.Float32)
        (meta + row).store(oldscale)
        (meta + ctm.Int32(128) + row).store(newsum)
    # Publish all eight metadata writers before a single ALU warp dispatches.
    # Generation i+1 requires receiver credit for i, so a publish cannot be lapped.
    # x7: one publish (bar5) covers P + PSF + META.
    # cp_dual_v5: the metadata copy is issued by warp 9 (which also copies P)
    # after that publish and the peer's landing-free credit; no ALU warp waits here.
    # P and local TMEM PSF retain the original two-consumer PV/source credit.
    if own_index > ctm.Int32(0):
        # x9: PV(own tile own_index-1) done straight from its commit barrier; bar6 is the peer credit only.
        _ds_wait_unless(
            pv_done_seen,
            bars.subview(ctm.Int32(11) + (local_tile & ctm.Int32(1))),
            credit_phase,
        )
        _ds_wait_unless(source_done_seen, source_done, credit_phase)
    swizzle = ctm.Swizzle.from_name("s128b")
    # Adjacent N16 groups occupy one contiguous aligned 16-byte segment
    # after the existing s128b swizzle. Preserve byte positions and publish
    # the same complete payload after four wider stores per row half.
    for g in ctm.range_constexpr(4):
        packed = ctm.Vector.from_elements(
            (
                packed_p4_words[g * 4],
                packed_p4_words[g * 4 + 1],
                packed_p4_words[g * 4 + 2],
                packed_p4_words[g * 4 + 3],
            ),
            dtype=ctm.Int32,
        )
        word_col = half * ctm.Int32(16) + ctm.Int32(g * 4)
        (sPayload.data_ptr() + row * ctm.Int32(32) + word_col).store_swizzled(
            packed, swizzle=swizzle, alignment=16
        )
    return newmax, newanchor, own_sf_words[0], own_sf_words[1], oldscale, peer_oldscale


@cute.jit
def _ds_publish_reg_half_payload(payload_ready):
    # ALL8 ALU warps execute this after their disjoint P/SF/meta writes.
    # Copy warp waits one generation of payload_ready(count8), then destination
    # pempty, applies the async shared fence and issues the existing UBLKCP.
    _fence_async_shared_cta()
    # Every lane hands its stores/fence to its elected publisher; all8 arrivals
    # still gate copy. Both pair64 joins protect their own rowmax reuse.
    prims.bar_warp_sync(cute.arch.FULL_MASK)
    if prims.elect_sync():
        prims.mbarrier_arrive(payload_ready)


# QK-only helpers. Caller must select exactly one thread after all waits.
@cute.jit
def _eq_copy_bsf_elected(
    sSF,
    base: ctm.Int32,
    col: ctm.Constexpr,
    chunks: ctm.Constexpr,
    ntiles: ctm.Constexpr = 1,
    first_nt: ctm.Constexpr = 0,
    source_ntiles: ctm.Constexpr = 1,
    first_ck: ctm.Constexpr = 0,
):
    # The input contains one 512B packed atom per 128 N rows and 64 K.
    # M256 uses all four DP bands, so expand each atom to four TMEM columns.
    shape, multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
    # Invariant unswizzled SF descriptor; offsets stay within its start field.
    desc_root = prims.Tcgen05SmemDesc.build(
        sSF, leading_byte_offset=16, stride_byte_offset=128, layout=0
    )
    for nt in ctm.range_constexpr(ntiles):
        for ck in ctm.range_constexpr(chunks):
            for nb in ctm.range_constexpr(2):
                # first_ck selects source K chunks; destination remains locally packed.
                src_byte: ctm.Constexpr = (
                    ((first_ck + ck) * source_ntiles + nt + first_nt) * 2 + nb
                ) * 512
                desc = desc_root.advance_start_address(src_byte)
                target = _ds_tmem_ptr(
                    base, ctm.Int32(0), ctm.Int32(col + nt * chunks * 8 + ck * 8 + nb * 4)
                )
                prims.tcgen05_cp(
                    shape, target, desc, group=prims.CTAGroup.CTA_2, multicast=multicast
                )


@cute.jit
def _eq_mma_elected(
    sA,
    sB,
    base: ctm.Int32,
    acc: ctm.Constexpr,
    acol: ctm.Constexpr,
    asf: ctm.Constexpr,
    bsf: ctm.Constexpr,
    k64: ctm.Constexpr,
    k128: ctm.Constexpr,
    clear: ctm.Boolean,
    a_tmem: ctm.Constexpr,
    a_byte: ctm.Constexpr = 0,
    b_byte: ctm.Constexpr = 0,
    swizzle64: ctm.Constexpr = False,
    n_dim: ctm.Constexpr = 256,
    a_collector_op: ctm.Constexpr = None,
    asf_offset: ctm.Int32 = 0,
    shared_q: ctm.Constexpr = False,
):
    desc = prims.Tcgen05MxOmmaInstrDesc.build(
        a_dtype=ctm.Float4E2M1FN,
        b_dtype=ctm.Float4E2M1FN,
        scale_format=0,
        n_dim=n_dim,
        m_dim=256,
        k_dim=2 if k128 else 0,
        a_sf_layout=1,
        b_major=0,
    )
    bdesc = prims.Tcgen05SmemDesc.build(
        sB.subview(b_byte // 4),
        leading_byte_offset=0 if swizzle64 else 16,
        stride_byte_offset=512 if swizzle64 else 1024,
        layout=4 if swizzle64 else 2,
    )
    if cutlass.const_expr(a_tmem):
        aoperand = _ds_tmem_ptr(base, ctm.Int32(0), ctm.Int32(acol))
    else:
        if cutlass.const_expr(shared_q):
            # Q core is s128b; its pre-permuted tail is s64b. Encoded
            # address advance preserves descriptor fields at each K step.
            aoperand = prims.Tcgen05SmemDesc.build(
                sA,
                leading_byte_offset=0 if swizzle64 else 16,
                stride_byte_offset=512 if swizzle64 else 1024,
                layout=4 if swizzle64 else 2,
            ).advance_start_address(a_byte)
        else:
            aoperand = prims.Tcgen05SmemDesc.build(
                sA.subview(a_byte // 4),
                leading_byte_offset=16,
                stride_byte_offset=1024,
                layout=2,
            )
    prims.tcgen05_mma_block_scale(
        prims.MMABlockScaleKind.MXF4NVF4,
        prims.CTAGroup.CTA_2,
        _ds_tmem_ptr(base, ctm.Int32(0), ctm.Int32(acc)),
        aoperand,
        bdesc,
        desc,
        enable_input_d=~clear,
        scale_a=_ds_tmem_ptr(base, ctm.Int32(0), ctm.Int32(asf) + asf_offset),
        scale_b=_ds_tmem_ptr(base, ctm.Int32(0), ctm.Int32(bsf)),
        scale_vec_size=prims.Tcgen05MMAScaleVecSize.BLOCK16,
        collector_op=a_collector_op,
    )


@cute.jit
def _s34_task_identity(
    bz: ctm.Int32,
    pair_rank: ctm.Int32,
    mValidK_l,
    mPageIndptr_s,
    mPlan_i,
    task_capacity: ctm.Int32,
    num_requests: ctm.Int32,
    query_len_per_seq: ctm.Constexpr,
    heads: ctm.Constexpr,
    split_mode: ctm.Constexpr,
    warp: ctm.Int32,
    lane: ctm.Int32,
):
    # Rows mode (variants/rows): a 128-row block = 128 packed [token, head] rows of one request; a pair = 2 blocks.
    # FULL mode: pair = bz over the whole KV. Split mode (Milestone 2): bz = (plan slot, pair of the request); the plan
    # (fp4_mla_packed_reduce layout [tiles_per_task, used, begins[R+1], request[cap], split[cap], count[cap]]) gives the
    # request and KV split; the task covers tiles [split * tiles_per_task, +tiles_per_task) of this pair, clipped to
    # the pair's own tile count (or, with plan[0] == 0, the explicit [tile_begin, +tile_count) of the ranges plan);
    # a task beyond it (or an unused slot) runs one fully masked tile so its denominator is
    # 0 and the reducer ignores it. Partials go to row slot begins[request] + split (never written for unused slots).
    rows_per_seq = heads * query_len_per_seq
    blocks_per_seq = rows_per_seq // 128
    pairs_per_seq = rows_per_seq // 256
    split = ctm.Int32(0)
    tiles_per_task = ctm.Int32(0)
    slot = ctm.Int32(0)
    if cutlass.const_expr(split_mode):
        task = bz // ctm.Int32(pairs_per_seq)
        pair_in_seq = bz - task * ctm.Int32(pairs_per_seq)
        tables = ctm.Int32(3) + num_requests
        sequence = mPlan_i[tables + task]
        split = mPlan_i[tables + task_capacity + task]
        request_tasks = mPlan_i[tables + task_capacity + task_capacity + task]
        tiles_per_task = mPlan_i[0]
        slot = mPlan_i[ctm.Int32(2) + sequence] + split
        block = sequence * ctm.Int32(blocks_per_seq) + pair_in_seq * ctm.Int32(2) + pair_rank
        row_in_seq = (pair_in_seq * ctm.Int32(2) + pair_rank) * ctm.Int32(128)
        pair_row_in_seq = pair_in_seq * ctm.Int32(256)
    else:
        pair = bz
        block = pair * ctm.Int32(2) + pair_rank
        row0 = block * ctm.Int32(128)
        sequence = row0 // ctm.Int32(rows_per_seq)
        row_in_seq = row0 - sequence * ctm.Int32(rows_per_seq)
        pair_row_in_seq = (pair * ctm.Int32(256)) % ctm.Int32(rows_per_seq)
    query = block  # 128-row block index == the Q / output "query" slot of the TMA views
    kv = mValidK_l[sequence]
    last_offset = ctm.Int32(query_len_per_seq - 1)
    lane_row = row_in_seq + (warp & ctm.Int32(3)) * ctm.Int32(32) + lane
    t_lane = lane_row // ctm.Int32(heads)
    valid_k = ctm.max(kv - (last_offset - t_lane), ctm.Int32(0))
    t_min = row_in_seq // ctm.Int32(heads)
    valid_k_min = ctm.max(kv - (last_offset - t_min), ctm.Int32(0))
    t_pair_last = (pair_row_in_seq + ctm.Int32(255)) // ctm.Int32(heads)
    pair_valid_k = ctm.max(kv - (last_offset - t_pair_last), ctm.Int32(0))
    total_tiles = ctm.max((pair_valid_k + ctm.Int32(255)) // ctm.Int32(256), ctm.Int32(1))
    begin_tile = ctm.Int32(0)
    chunk_tiles = total_tiles
    slot_row_base = ctm.Int32(-1)
    direct = ctm.Boolean(
        True
    )  # write the normalized output (FULL mode, or a split-mode request with one task)
    if cutlass.const_expr(split_mode):
        unused = split >= ctm.Int32(1 << 15)
        direct = ctm.Boolean(request_tasks == ctm.Int32(1))
        begin_tile = split * tiles_per_task
        chunk_tiles = ctm.min(tiles_per_task, total_tiles - begin_tile)
        if tiles_per_task == ctm.Int32(0):
            # Ranges plan (mixed / heterogeneous splits, harness/mla4x_plan.py): plan[0] == 0 flags two extra per-slot
            # tables appended after the fp4_mla_packed_reduce layout, tile_begin[cap] and tile_count[cap]; the task
            # covers tiles [tile_begin, +tile_count) of its pair, clipped as before. The reducer never reads plan[0].
            begin_tile = mPlan_i[tables + ctm.Int32(3) * task_capacity + task]
            chunk_tiles = ctm.min(
                mPlan_i[tables + ctm.Int32(4) * task_capacity + task], total_tiles - begin_tile
            )
        if unused or chunk_tiles <= ctm.Int32(0):
            begin_tile = total_tiles - ctm.Int32(1)
            chunk_tiles = ctm.Int32(1)
            cap = begin_tile * ctm.Int32(256)
            valid_k = ctm.min(valid_k, cap)
            valid_k_min = ctm.min(valid_k_min, cap)
        if not unused:
            slot_row_base = slot * ctm.Int32(rows_per_seq) + row_in_seq
    page_begin = mPageIndptr_s[sequence]
    page_count = mPageIndptr_s[sequence + ctm.Int32(1)] - page_begin
    return (
        query,
        sequence,
        valid_k,
        valid_k_min,
        begin_tile,
        chunk_tiles,
        page_begin,
        page_count,
        slot_row_base,
        direct,
    )


@cute.kernel
def ordered_v30_six_yield_kernel(
    mPageTable_pl: cute.Tensor,
    mPageIndptr_s: cute.Tensor,
    mValidK_l: cute.Tensor,
    mQGlobalScale: cute.Tensor,
    mKvGlobalScale: cute.Tensor,
    mPlan_i: cute.Tensor,
    mPartial: cute.Tensor,
    mRowMax: cute.Tensor,
    mRowSum: cute.Tensor,  # split mode
    tma_q_desc: ctm.GridConstant[cuda_tma.TensorMap],
    tma_k_page_desc: ctm.GridConstant[cuda_tma.TensorMap],
    tma_q_tail_desc: ctm.GridConstant[cuda_tma.TensorMap],
    tma_k_tail_page_desc: ctm.GridConstant[cuda_tma.TensorMap],
    tma_v_tile_desc: ctm.GridConstant[cuda_tma.TensorMap],
    tma_qsf_desc: ctm.GridConstant[cuda_tma.TensorMap],
    tma_qsf_tail_desc: ctm.GridConstant[cuda_tma.TensorMap],
    tma_ksf_desc: ctm.GridConstant[cuda_tma.TensorMap],
    tma_ksf_stage_desc: ctm.GridConstant[cuda_tma.TensorMap],
    tma_ksf_tail_stage_desc: ctm.GridConstant[cuda_tma.TensorMap],
    tma_vsf_desc: ctm.GridConstant[cuda_tma.TensorMap],
    tma_vsf_stage_desc: ctm.GridConstant[cuda_tma.TensorMap],
    tma_v_pair_desc: ctm.GridConstant[cuda_tma.TensorMap],
    tma_sfb_desc: ctm.GridConstant[cuda_tma.TensorMap],
    mC_mnl: cute.Tensor,
    problem_size: tuple,
    runtime_m: ctm.Int32,
    v_page_offset: ctm.Int32,
    softmax_scale_log2: ctm.Float32,
    pv_output_scale: ctm.Float32,
    task_capacity: ctm.Int32,
    num_requests: ctm.Int32,  # split mode
    page_size: ctm.Constexpr,
    query_len_per_seq: ctm.Constexpr,
    heads: ctm.Constexpr,
    split_mode: ctm.Constexpr,
    use_mixed_imlp: ctm.Constexpr,
    use_consecutive_page_pair: ctm.Constexpr,
    use_ksf_gather4: ctm.Constexpr,
):
    tma_q_ptr = tma_q_desc.get_ptr()
    tma_k_page_ptr = tma_k_page_desc.get_ptr()
    tma_q_tail_ptr = tma_q_tail_desc.get_ptr()
    tma_k_tail_page_ptr = tma_k_tail_page_desc.get_ptr()
    tma_qsf_ptr = tma_qsf_desc.get_ptr()
    tma_qsf_tail_ptr = tma_qsf_tail_desc.get_ptr()
    tma_ksf_ptr = tma_ksf_desc.get_ptr()
    tma_ksf_gather_ptr = tma_ksf_stage_desc.get_ptr()
    tma_vsf_stage_ptr = tma_vsf_stage_desc.get_ptr()
    tma_v_pair_ptr = tma_v_pair_desc.get_ptr()
    tid, _, _ = cute.arch.thread_idx()
    bx, _, bz = cute.arch.block_idx()
    rank = cute.arch.block_idx_in_cluster()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tid & ctm.Int32(31)
    pair_rank = rank & ctm.Int32(1)
    # Shared memory, in the order of DS_SMEM_LAYOUT (the placement rules are documented there).
    tmem = ctm.Array(ctm.Int32, 1, space=ctm.AddressSpace.smem)
    sReleaseCookie = ctm.Array(ctm.Int32, 1, space=ctm.AddressSpace.smem)
    bars = ctm.Array(ctm.Int64, DS_NUM_BARS, space=ctm.AddressSpace.smem, alignment=8)
    # ur_relief: kernel-constant DSMEM address relay (UR_RELAY_*), appended after bars. Written once
    # at kernel start by warp 8's elected lane, read-only afterwards; no TMA, bulk copy, st.async or
    # alias view covers it.
    sUrRelay = ctm.Array(ctm.Int32, UR_RELAY_WORDS, space=ctm.AddressSpace.smem, alignment=16)
    # The control words above end inside the first 1024 B (asserted beside UR_RELAY_WORDS); the pad completes
    # DS_SMEM_CONTROL_BYTES.
    sControlPad = ctm.Array(  # noqa: F841
        ctm.Uint8, DS_SMEM_CONTROL_BYTES - 1024, space=ctm.AddressSpace.smem, alignment=1024
    )
    smem = _ds_smem_arrays()
    sQSF, sV, sVSF, sKSFBanks = smem["sQSF"], smem["sV"], smem["sVSF"], smem["sKSFBanks"]
    sPayloadBanks, sPsfMetaAlt = smem["sPayloadBanks"], smem["sPsfMetaAlt"]
    sKBanks, sQ = smem["sKBanks"], smem["sQ"]
    # FULL-only: one task per cluster; identity bound once at kernel start (its valid_k / page-indptr
    # loads issue first, ahead of the barrier init).
    (
        query,
        sequence,
        valid_k,
        valid_k_min,
        begin_tile,
        chunk_tiles,
        page_begin,
        page_count,
        slot_row_base,
        direct,
    ) = _s34_task_identity(
        bz,
        pair_rank,
        mValidK_l,
        mPageIndptr_s,
        mPlan_i,
        task_capacity,
        num_requests,
        query_len_per_seq,
        heads,
        split_mode,
        warp,
        lane,
    )
    if cutlass.const_expr(split_mode):
        # Static grid (CUDA graphs replay grid z = task_capacity x pairs): a slot past the plan's used tasks exits
        # here, before any mbarrier, TMEM or cluster state exists. All four CTAs of the cluster read the same slot,
        # so the whole cluster leaves together (measured: 174 idle clusters cost ~0.5 us; a masked tile each, 17 us).
        if slot_row_base < ctm.Int32(0):
            prims.exit()
    # ur_relief: publish the relay once per CTA. Every read follows the first cluster barrier and
    # bar.sync 0 (CTA-wide ordering of this store).
    if warp == ctm.Int32(8):
        if prims.elect_sync():
            _ur_relay_init(sUrRelay, bars, sPayloadBanks, rank)
    softscale = softmax_scale_log2 / (mQGlobalScale[0] * mKvGlobalScale[0])
    pscale = ctm.Float32(FP4_MLA_P_GLOBAL_SCALE) / mKvGlobalScale[0]
    if warp == ctm.Int32(8):
        if prims.elect_sync():
            sReleaseCookie[0] = ctm.Int32(0)
            for b in ctm.range_constexpr(DS_NUM_BARS):
                prims.mbarrier_init(
                    bars.subview(b),
                    2
                    if b == 1
                    else 16
                    if b == 3
                    else 32
                    if b == 4
                    else 8
                    if b == 5 or b == 9 or b == 10 or b == 23 or b == 24
                    else 1,
                )
        prims.tcgen05_alloc(
            tmem, NUM_TMEM_ALLOC_COLS, is_exclusive=True, group=prims.CTAGroup.CTA_2
        )
    prims.fence_mbarrier_init()
    # fence_b: .release arrive: warp 8's tcgen05.alloc wrote the TMEM token with a weak store; this
    # arrive/wait pair is the only release/acquire edge before every thread's tmem.load() below.
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    base = tmem.load()
    if warp == ctm.Int32(8):
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(bars.subview(0), DS_Q_BYTES + DS_QSF_BYTES)
            # Reuse one coordinate packet for the two identical core
            # stages; avoid six simultaneously live TMA packets.
            for stage in cutlass.range(0, ctm.Int32(2), 1, unroll=1):
                _ds_tma(
                    sQ.subview(stage * ctm.Int32(128 * 256 // 2 // 4)),
                    tma_q_ptr,
                    (stage * ctm.Int32(256), (bx // ctm.Int32(4)) * ctm.Int32(128), query),
                    bars.subview(0),
                    rank,
                )
                _ds_tma(
                    sQSF.subview(stage * ctm.Int32(4 * 512)),
                    tma_qsf_ptr,
                    (ctm.Int32(0), stage * ctm.Int32(4), bx // ctm.Int32(4), query),
                    bars.subview(0),
                    rank,
                )
            _ds_tma(
                sQ.subview(2 * 128 * 256 // 2 // 4),
                tma_q_tail_ptr,
                (ctm.Int32(512), (bx // ctm.Int32(4)) * ctm.Int32(128), query),
                bars.subview(0),
                rank,
            )
            _ds_tma(
                sQSF.subview(2 * 4 * 512),
                tma_qsf_tail_ptr,
                (ctm.Int32(0), ctm.Int32(8), bx // ctm.Int32(4), query),
                bars.subview(0),
                rank,
            )
        _ds_wait(bars.subview(0), ctm.Int32(0))
    # Epoch-start L2 warming while Q is in flight: the first V/VSF tiles and
    # this CTA's first owned K page otherwise pay full DRAM latency at loop
    # entry. Same descriptors/coordinates as the loaders; L2 only.
    if warp == ctm.Int32(11):
        if prims.elect_sync():
            for t in ctm.range_constexpr(2):
                if ctm.Int32(t) < chunk_tiles:
                    for np in ctm.range_constexpr(2):
                        spf_page = _lookup_physical_page(
                            mPageTable_pl,
                            (begin_tile + ctm.Int32(t)) * ctm.Int32(2) + ctm.Int32(np),
                            sequence,
                            page_begin,
                            page_count,
                        )
                        prims.cp_async_bulk_tensor_prefetch(
                            tma_v_pair_ptr,
                            (
                                ctm.Int32(0),
                                ctm.Int32(0),
                                rank >> ctm.Int32(1),
                                spf_page + v_page_offset,
                                pair_rank,
                            ),
                            [],
                        )
                        prims.cp_async_bulk_tensor_prefetch(
                            tma_vsf_stage_ptr,
                            (
                                ctm.Int32(0),
                                (rank >> ctm.Int32(1)) * ctm.Int32(2),
                                ctm.Int32(0),
                                spf_page + v_page_offset,
                            ),
                            [],
                        )
    if warp == ctm.Int32(10):
        # k0_early: the real K/KSF load of own tile 0 into bank 0, issued now (parallel to the Q
        # load) instead of after Q -> TMEM and the second cluster barrier. Same TMAs, same bank,
        # same bar13 phase 0 as the loader loop would have issued; the loop skips own_index 0.
        if prims.elect_sync():
            if (rank >> ctm.Int32(1)) < chunk_tiles:
                e_tile = begin_tile + (rank >> ctm.Int32(1))
                e_sK0, e_sKSF0 = _ds_k2_views(sKBanks, sKSFBanks, ctm.Int32(0))
                e_own_page = _lookup_physical_page(
                    mPageTable_pl,
                    e_tile * ctm.Int32(2) + pair_rank,
                    sequence,
                    page_begin,
                    page_count,
                )
                e_page0 = _lookup_physical_page(
                    mPageTable_pl, e_tile * ctm.Int32(2), sequence, page_begin, page_count
                )
                e_page1 = _lookup_physical_page(
                    mPageTable_pl,
                    e_tile * ctm.Int32(2) + ctm.Int32(1),
                    sequence,
                    page_begin,
                    page_count,
                )
                prims.mbarrier_arrive_expect_tx(bars.subview(13), DS_K_BYTES + DS_KSF_BYTES)
                for stage in ctm.range_constexpr(3):
                    e_kd = tma_k_page_ptr
                    if cutlass.const_expr(stage == 2):
                        e_kd = tma_k_tail_page_ptr
                    _ds_tma(
                        e_sK0.subview(stage * 128 * 256 // 2 // 4),
                        e_kd,
                        (ctm.Int32(stage * 256), ctm.Int32(0), e_own_page),
                        bars.subview(13),
                        rank,
                    )
                if cutlass.const_expr(use_ksf_gather4):
                    e_row0 = e_page0 * ctm.Int32(TRTLLM_K_SF_GROUPS // 4)
                    e_row1 = e_page1 * ctm.Int32(TRTLLM_K_SF_GROUPS // 4)
                    for scale_pair in ctm.range_constexpr(5):
                        e_ck = ctm.Int32(scale_pair * 2)
                        _ds_tma_ksf_gather4_local(
                            e_sKSF0.subview(scale_pair * 4 * 512),
                            tma_ksf_gather_ptr,
                            e_row0 + e_ck,
                            e_row1 + e_ck,
                            e_row0 + e_ck + ctm.Int32(1),
                            e_row1 + e_ck + ctm.Int32(1),
                            bars.subview(13),
                            rank,
                        )
                else:
                    for ck in ctm.range_constexpr(10):
                        for np in ctm.range_constexpr(2):
                            e_page = e_page0 if cutlass.const_expr(np == 0) else e_page1
                            _ds_tma(
                                e_sKSF0.subview((ck * 2 + np) * 512),
                                tma_ksf_ptr,
                                (ctm.Int32(0), ctm.Int32(ck), e_page, ctm.Int32(0)),
                                bars.subview(13),
                                rank,
                            )
    prims.barrier_cta_sync(0, thread_count=384)
    _ds_q_to_tmem(sQ, sQSF, base, warp, lane, bars.subview(1), rank, sUrRelay, 8, 384)
    # Prevent these extra scheduler counts from remaining live through
    # the startup Q TMA burst. The same initialized zero cookie already
    # used by C28 supplies an exact, non-hoistable runtime dependency.
    unused_scale, work_tiles = _ds_release_cookie_scale_and_bound(
        sReleaseCookie, softscale, chunk_tiles
    )
    pair_id = rank >> ctm.Int32(1)
    own_count = (work_tiles + ctm.Int32(1) - pair_id) // ctm.Int32(2)
    peer_count = (work_tiles + pair_id) // ctm.Int32(2)
    rounds = (work_tiles + ctm.Int32(2) - pair_id) // ctm.Int32(2)
    # QSF SMEM is dead after the store completion and pair rendezvous.
    sHalfMax = ctm.Array(
        ctm.inttoptr(ctm.Int64(sQSF.data_ptr().toint()), 3, ctm.Float32), shape=256
    )
    sBlockMax = ctm.Array(
        ctm.inttoptr(ctm.Int64(sQSF.data_ptr().toint()) + ctm.Int64(1024), 3, ctm.Float32),
        shape=256,
    )
    sUnit = ctm.Array(
        ctm.inttoptr(ctm.Int64(sQSF.data_ptr().toint()) + ctm.Int64(2048), 3, ctm.Int32), shape=384
    )
    sUnitSF = ctm.Array(
        ctm.inttoptr(ctm.Int64(sQSF.data_ptr().toint()) + ctm.Int64(2560), 3, ctm.Uint8), shape=1024
    )
    sDenom = ctm.Array(
        ctm.inttoptr(ctm.Int64(sQSF.data_ptr().toint()) + ctm.Int64(3584), 3, ctm.Float32),
        shape=128,
    )
    unit_value = ctm.Int32(0x38383838)
    if tid < ctm.Int32(128):
        unit_value = ctm.Int32(0x22222222)
    sUnit[tid] = unit_value
    # rsplit: pair 1 accumulates its odd-tile rowsum partial from zero in TMEM column 512
    # (pair 0's column is cleared by its tile-0 MMA). Warps 0-3
    # own rows 0..127; the store is ordered before the first rowsum MMA by these warps'
    # wait::st + tcgen05.fence::before_thread_sync ahead of their prep-ready arrive.
    if pair_id == ctm.Int32(1):
        if warp < ctm.Int32(4):
            prims.tcgen05_st(
                prims.Tcgen05LdStShape.SHAPE_32X32B,
                _ds_tmem_ptr(base, warp * ctm.Int32(32), ctm.Int32(DS_ROWSUM_ACC_TMEM)),
                ctm.Int32(0),
            )
            cute.arch.fence_view_async_tmem_store()
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0, thread_count=384)
    _fence_async_shared_cta()
    if warp == ctm.Int32(8):
        if prims.elect_sync():
            if peer_count > ctm.Int32(0):
                prims.mbarrier_arrive_expect_tx(bars.subview(8), DS_P_BYTES)
                prims.mbarrier_arrive_expect_tx(
                    bars.subview(DS_BAR_META_LANDED), DS_PSF_BYTES + DS_META_BYTES
                )
                _ur_arrive_remote(sUrRelay[UR_RELAY_LANDED_BAR])
            for bank in ctm.range_constexpr(2):
                if ctm.Int32(bank) < peer_count:
                    prims.mbarrier_arrive_expect_tx(bars.subview(21 + bank), 512)
            prims.mbarrier_arrive_expect_tx(
                bars.subview(25), 512
            )  # rsplit: both pairs receive the peer pair's rowsum partial (epilogue)
    # group2 cp reads each CTA's own source SMEM. Publish both
    # unitSF sources before rank0 submits the asynchronous copy.
    # This is the existing startup rendezvous, moved before the copy;
    # all local stores and async-proxy fences precede the arrival.
    # fence_b: this arrive keeps .release -- the peer's
    # tcgen05.cp.cta_group::2 reads this CTA's sUnitSF after its wait,
    # and the first DSMEM writes of the epoch (blockmax st.async, P and
    # [PSF|META] bulk copies, sDenom st.async) all follow the peers' wait
    # here, so it also orders every previous-epoch read of those buffers
    # before the new writes (cross-epoch WAR).
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    if warp == ctm.Int32(8):
        if (rank & ctm.Int32(1)) == ctm.Int32(0) and chunk_tiles > ctm.Int32(
            0
        ):  # rsplit: both pair leaders
            shape, multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
            for ck in ctm.range_constexpr(2):
                desc = prims.Tcgen05SmemDesc.build(
                    sUnitSF.subview(ck * 512),
                    leading_byte_offset=16,
                    stride_byte_offset=128,
                    layout=0,
                )
                if prims.elect_sync():
                    prims.tcgen05_cp(
                        shape,
                        _ds_tmem_ptr(base, ctm.Int32(0), ctm.Int32(DS_ROWSUM_SF_TMEM + ck * 4)),
                        desc,
                        group=prims.CTAGroup.CTA_2,
                        multicast=multicast,
                    )
    # One pair owns each QK/S2P tile. Both pairs own disjoint O256
    # slices and consume every P in original tile/K order.
    if warp == ctm.Int32(10):
        # k0_early issued own tile 0's K/KSF into bank 0 on bar13 phase 0 at kernel start; the loader
        # loop skips own_index 0 (k0_preloaded == 2: no plain arrive; the `== 1` branch below is dead).
        k0_preloaded = ctm.Int32(2)
        for own_index in cutlass.range(0, own_count, 1, unroll=1):
            tile = begin_tile + own_index * ctm.Int32(2) + pair_id
            bank = own_index & ctm.Int32(1)
            phase = (own_index // ctm.Int32(2)) & ctm.Int32(1)
            sK, sKSF = _ds_k2_views(sKBanks, sKSFBanks, bank)
            full = bars.subview(ctm.Int32(13) + bank)
            if own_index == ctm.Int32(0) and k0_preloaded != ctm.Int32(0):
                if k0_preloaded == ctm.Int32(1):
                    if prims.elect_sync():
                        prims.mbarrier_arrive(full)
            else:
                # Resolve this tile's pages before the bank-reuse wait so the
                # TMA packet can issue as soon as the bank is free.
                own_page = _lookup_physical_page(
                    mPageTable_pl, tile * ctm.Int32(2) + pair_rank, sequence, page_begin, page_count
                )
                page0 = _lookup_physical_page(
                    mPageTable_pl, tile * ctm.Int32(2), sequence, page_begin, page_count
                )
                page1 = _lookup_physical_page(
                    mPageTable_pl,
                    tile * ctm.Int32(2) + ctm.Int32(1),
                    sequence,
                    page_begin,
                    page_count,
                )
                # kstage: each K stage slot is released by the previous QK on this bank as soon as
                # the MMAs reading it complete (stage 0 -> bars[27+bank], stage 1 -> bars[29+bank],
                # stage 2 + KSF -> bars[15+bank]); wait and issue per stage, expect_tx armed once.
                for stage in ctm.range_constexpr(3):
                    if own_index >= ctm.Int32(2):
                        if cutlass.const_expr(stage == 0):
                            _ds_wait(bars.subview(ctm.Int32(27) + bank), phase ^ ctm.Int32(1))
                        if cutlass.const_expr(stage == 1):
                            _ds_wait(bars.subview(ctm.Int32(29) + bank), phase ^ ctm.Int32(1))
                        if cutlass.const_expr(stage == 2):
                            _ds_wait(bars.subview(ctm.Int32(15) + bank), phase ^ ctm.Int32(1))
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    if prims.elect_sync():
                        if cutlass.const_expr(stage == 0):
                            prims.mbarrier_arrive_expect_tx(full, DS_K_BYTES + DS_KSF_BYTES)
                        kd = tma_k_page_ptr
                        if cutlass.const_expr(stage == 2):
                            kd = tma_k_tail_page_ptr
                        _ds_tma(
                            sK.subview(stage * 128 * 256 // 2 // 4),
                            kd,
                            (ctm.Int32(stage * 256), ctm.Int32(0), own_page),
                            full,
                            rank,
                        )
                if prims.elect_sync():
                    if cutlass.const_expr(use_ksf_gather4):
                        row0 = page0 * ctm.Int32(TRTLLM_K_SF_GROUPS // 4)
                        row1 = page1 * ctm.Int32(TRTLLM_K_SF_GROUPS // 4)
                        for scale_pair in ctm.range_constexpr(5):
                            ck = ctm.Int32(scale_pair * 2)
                            _ds_tma_ksf_gather4_local(
                                sKSF.subview(scale_pair * 4 * 512),
                                tma_ksf_gather_ptr,
                                row0 + ck,
                                row1 + ck,
                                row0 + ck + ctm.Int32(1),
                                row1 + ck + ctm.Int32(1),
                                full,
                                rank,
                            )
                    else:
                        for ck in ctm.range_constexpr(10):
                            for np in ctm.range_constexpr(2):
                                page = _lookup_physical_page(
                                    mPageTable_pl,
                                    tile * ctm.Int32(2) + ctm.Int32(np),
                                    sequence,
                                    page_begin,
                                    page_count,
                                )
                                _ds_tma(
                                    sKSF.subview((ck * 2 + np) * 512),
                                    tma_ksf_ptr,
                                    (ctm.Int32(0), ctm.Int32(ck), page, ctm.Int32(0)),
                                    full,
                                    rank,
                                )
    if warp == ctm.Int32(11):
        # Tiles are walked in pairs over a rolling window of resolved pages (vhoist2), so no page-table load
        # sits between a V bank's release and its TMA issue.  There is no per-tile L2 prefetch: the four-bank ring
        # issues the load itself a round ahead (only the kernel-start warm of tiles 0 and 1 above remains).
        # Window: a/b = this tile pair, c/d = the next one (resolved before the loop for tiles 2/3, which issue
        # without a wait right after tiles 0/1), e/f = the pair after that, looked up in this iteration.
        vh_a0 = _lookup_physical_page(
            mPageTable_pl, begin_tile * ctm.Int32(2), sequence, page_begin, page_count
        )
        vh_a1 = _lookup_physical_page(
            mPageTable_pl,
            begin_tile * ctm.Int32(2) + ctm.Int32(1),
            sequence,
            page_begin,
            page_count,
        )
        vh_b0 = _lookup_physical_page(
            mPageTable_pl,
            begin_tile * ctm.Int32(2) + ctm.Int32(2),
            sequence,
            page_begin,
            page_count,
        )
        vh_b1 = _lookup_physical_page(
            mPageTable_pl,
            begin_tile * ctm.Int32(2) + ctm.Int32(3),
            sequence,
            page_begin,
            page_count,
        )
        vh_c0 = _lookup_physical_page(
            mPageTable_pl,
            begin_tile * ctm.Int32(2) + ctm.Int32(4),
            sequence,
            page_begin,
            page_count,
        )
        vh_c1 = _lookup_physical_page(
            mPageTable_pl,
            begin_tile * ctm.Int32(2) + ctm.Int32(5),
            sequence,
            page_begin,
            page_count,
        )
        vh_d0 = _lookup_physical_page(
            mPageTable_pl,
            begin_tile * ctm.Int32(2) + ctm.Int32(6),
            sequence,
            page_begin,
            page_count,
        )
        vh_d1 = _lookup_physical_page(
            mPageTable_pl,
            begin_tile * ctm.Int32(2) + ctm.Int32(7),
            sequence,
            page_begin,
            page_count,
        )
        for vh_k in cutlass.range(0, (chunk_tiles + ctm.Int32(1)) // ctm.Int32(2), 1, unroll=1):
            vh_t0 = vh_k * ctm.Int32(2)
            vh_tile = begin_tile + vh_t0
            vh_e0 = ctm.Int32(0)
            vh_e1 = ctm.Int32(0)
            vh_f0 = ctm.Int32(0)
            vh_f1 = ctm.Int32(0)
            _vh2_issue(
                sV,
                sVSF,
                bars,
                vh_t0,
                vh_a0,
                vh_a1,
                tma_v_pair_ptr,
                tma_vsf_stage_ptr,
                pair_id,
                pair_rank,
                v_page_offset,
                rank,
            )
            if pair_id == ctm.Int32(0):
                vh_e0, vh_e1, vh_f0, vh_f1 = _vh2_lookup4(
                    mPageTable_pl, vh_tile, sequence, page_begin, page_count
                )
            if vh_t0 + ctm.Int32(1) < chunk_tiles:
                _vh2_issue(
                    sV,
                    sVSF,
                    bars,
                    vh_t0 + ctm.Int32(1),
                    vh_b0,
                    vh_b1,
                    tma_v_pair_ptr,
                    tma_vsf_stage_ptr,
                    pair_id,
                    pair_rank,
                    v_page_offset,
                    rank,
                )
            if pair_id == ctm.Int32(1):
                vh_e0, vh_e1, vh_f0, vh_f1 = _vh2_lookup4(
                    mPageTable_pl, vh_tile, sequence, page_begin, page_count
                )
            vh_a0 = vh_c0
            vh_a1 = vh_c1
            vh_b0 = vh_d0
            vh_b1 = vh_d1
            vh_c0 = vh_e0
            vh_c1 = vh_e1
            vh_d0 = vh_f0
            vh_d1 = vh_f1
    if warp == ctm.Int32(9):
        for own_index in cutlass.range(0, rounds, 1, unroll=1):
            own_local = own_index * ctm.Int32(2) + pair_id
            if own_local < chunk_tiles:
                sPayload = sPayloadBanks.subview(
                    ctm.Int32(PB_OWN) * ctm.Int32(DS_PAYLOAD_BYTES // 4)
                )
                # ur_relief: relayed [PSF|META] copy destination / mbarrier (mapa rank ^ 2 of the [PSF|META]
                # region of the landing bank sPayloadBanks[PB_LAND] and of &bars[DS_BAR_META_LANDED]), read
                # before the waits.
                meta_dst = sUrRelay[UR_RELAY_META_DST]
                meta_bar = sUrRelay[UR_RELAY_META_BAR]
                # cp_dual_v5: [PSF|META] copy (was ALU warp 0): all eight writers
                # published on bar5; the peer's landing bank is free once its
                # issuer's PSF copy of the previous generation completed (bars[META_FREE],
                # commit-driven). Source alternates by own-index parity.
                _ds_wait(bars.subview(5), own_index & ctm.Int32(1))
                if own_index > ctm.Int32(0):
                    _ds_wait(
                        bars.subview(DS_BAR_META_FREE), (own_index - ctm.Int32(1)) & ctm.Int32(1)
                    )
                _fence_async_shared_cta()
                if prims.elect_sync():
                    sMeta = sPayload.subview(ctm.Int32(DS_P_BYTES // 4))
                    if (own_index & ctm.Int32(1)) == ctm.Int32(0):
                        _ur_bulk_copy_to_peer(
                            meta_dst, sMeta, meta_bar, DS_PSF_BYTES + DS_META_BYTES
                        )
                    else:
                        _ur_bulk_copy_to_peer(
                            meta_dst, sPsfMetaAlt, meta_bar, DS_PSF_BYTES + DS_META_BYTES
                        )
                # ur_relief: relayed P copy destination / mbarrier (mapa rank ^ 2 of the landing bank
                # sPayloadBanks[PB_LAND] and of &bars[8]), read before the waits.
                p_dst = sUrRelay[UR_RELAY_P_DST]
                p_bar = sUrRelay[UR_RELAY_P_BAR]
                _ds_wait(bars.subview(5), own_index & ctm.Int32(1))
                _ds_wait(bars.subview(7), own_index & ctm.Int32(1))
                _fence_async_shared_cta()
                if prims.elect_sync():
                    _ur_bulk_copy_to_peer(p_dst, sPayload, p_bar, DS_P_BYTES)
            first_pv = ctm.max(own_local - ctm.Int32(1), ctm.Int32(0))
            last_pv = ctm.min(own_local + ctm.Int32(1), chunk_tiles)
            for local_tile in cutlass.range(first_pv, last_pv, 1, unroll=1):
                _r95_complete_pv(bars, local_tile, chunk_tiles, pair_id, rank, sUrRelay)
        if own_count > ctm.Int32(0):
            _ds_wait(
                bars.subview(ctm.Int32(11) + pair_id), (own_count - ctm.Int32(1)) & ctm.Int32(1)
            )  # x9
            _ds_wait(bars.subview(6), (own_count - ctm.Int32(1)) & ctm.Int32(1))
            # Drain the final landed-bank release (the peer issuer's commit behind its
            # PSF copy, cp_dual) before epoch invalidation.
            _ds_wait(bars.subview(DS_BAR_META_FREE), (own_count - ctm.Int32(1)) & ctm.Int32(1))
    if warp == ctm.Int32(8):
        pair_phase = ctm.Int32(1)
        for own_index in cutlass.range(0, rounds + ctm.Int32(1), 1, unroll=1):
            own_local = own_index * ctm.Int32(2) + pair_id
            base_i = _c_opaque_i32(base)  # v9: per-iteration TMEM base (see _c_opaque_i32)
            if own_local < chunk_tiles:
                bank = own_index & ctm.Int32(1)
                # ur_relief: relayed mapa(&bars[1], rank ^ 1), read before the K
                # wait; dead before the QK packet below.
                pair_remote = sUrRelay[UR_RELAY_PAIR_BAR]
                _ds_wait(
                    bars.subview(ctm.Int32(13) + bank), (own_index // ctm.Int32(2)) & ctm.Int32(1)
                )
                # K/KSF are complete TMA writes in the async proxy.
                # Keep pair readiness and tcgen thread-sync ordering.
                if prims.elect_sync():
                    prims.mbarrier_arrive(bars.subview(1))
                    _ur_arrive_remote(pair_remote)
                _ds_wait(bars.subview(1), pair_phase)
                pair_phase ^= ctm.Int32(1)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                if pair_rank == ctm.Int32(0):
                    if own_index > ctm.Int32(0):
                        _ds_wait(bars.subview(3), (own_index - ctm.Int32(1)) & ctm.Int32(1))
                    sK, sKSF = _ds_k2_views(sKBanks, sKSFBanks, bank)
                    # One selected thread owns the complete QK/SF/commit packet.
                    if prims.elect_sync():
                        for kb in ctm.range_constexpr(4):
                            _eq_copy_bsf_elected(sKSF, base_i, DS_KSF_TMEM, 2, first_ck=kb * 2)
                            _eq_mma_elected(
                                sQ,
                                sK,
                                base_i,
                                0,
                                0,
                                DS_QSF_TMEM + kb * 2,
                                DS_KSF_TMEM,
                                kb * 2,
                                True,
                                ctm.Boolean(kb == 0),
                                False,
                                a_byte=(kb // 2) * 128 * 256 // 2 + (kb % 2) * 64,
                                b_byte=(kb // 2) * 128 * 256 // 2 + (kb % 2) * 64,
                                shared_q=True,
                            )
                            # kstage: K stage (kb // 2) is fully read after the kb = 1 / kb = 3 MMAs;
                            # release it to the K loader now (multicast: both CTAs' loaders).
                            if cutlass.const_expr(kb == 1):
                                prims.tcgen05_commit(
                                    bars.subview(ctm.Int32(27) + bank),
                                    multicast_mask=ctm.Int16(3)
                                    << ctm.Int16(pair_id * ctm.Int32(2)),
                                    group=prims.CTAGroup.CTA_2,
                                )
                            if cutlass.const_expr(kb == 3):
                                prims.tcgen05_commit(
                                    bars.subview(ctm.Int32(29) + bank),
                                    multicast_mask=ctm.Int16(3)
                                    << ctm.Int16(pair_id * ctm.Int32(2)),
                                    group=prims.CTAGroup.CTA_2,
                                )
                        # tail_k128: the three K64 RoPE first-order tail MMAs (Qm.Kt, Qr.Kt, Qm.Kr) become one K64
                        # (Qm.Kt) and one K128 ([Qr | Qm] . [Kt | Kr] = Qr.Kt + Qm.Kr). The kernel-start permutation
                        # writes the Q tail rows as [Qr | Qm] and the two tail QSF columns as (+8 residual, +9 main);
                        # K / KSF are untouched. Same terms in the same order; Qr.Kt + Qm.Kr is rounded once instead
                        # of twice (rounding order). One 4-copy SFB group serves both steps: chunk 8 (Kt) ->
                        # [DS_KSF_TMEM, +8), chunk 9 (Kr) -> [+8, +16).
                        _eq_copy_bsf_elected(sKSF, base_i, DS_KSF_TMEM, 2, first_ck=8)
                        # tail0 = K64 Qm.Kt: A = second half of the [Qr | Qm] rows, SFA = main scales, B = Kt,
                        # SFB = chunk 8.
                        _eq_mma_elected(
                            sQ,
                            sK,
                            base_i,
                            0,
                            0,
                            DS_QSF_TMEM + 9,
                            DS_KSF_TMEM,
                            9,
                            False,
                            ctm.Boolean(False),
                            False,
                            a_byte=128 * 512 // 2 + 32,
                            b_byte=128 * 512 // 2,
                            swizzle64=True,
                            shared_q=True,
                        )
                        # tail12 = ONE K128 over the whole tail rows: [Qr | Qm] . [Kt | Kr].
                        _eq_mma_elected(
                            sQ,
                            sK,
                            base_i,
                            0,
                            0,
                            DS_QSF_TMEM + 8,
                            DS_KSF_TMEM,
                            8,
                            True,
                            ctm.Boolean(False),
                            False,
                            a_byte=128 * 512 // 2,
                            b_byte=128 * 512 // 2,
                            swizzle64=True,
                            shared_q=True,
                        )
                        prims.tcgen05_commit(
                            bars.subview(ctm.Int32(15) + bank),
                            multicast_mask=ctm.Int16(3) << ctm.Int16(pair_id * ctm.Int32(2)),
                            group=prims.CTAGroup.CTA_2,
                        )
                # Per-K-bank completion serves both score capture and
                # K reuse. Score-buffer release remains independent;
                # the issuer never waits for QK completion here.
            # Score capture releases S independently of P/PV. Submit
            # the next owned QK before the previous round's PV pair.
            # Its K/SF and score slots have explicit credits; the same
            # issuer preserves the shared B-SF CP/MMA ordering.
            if own_index > ctm.Int32(0):
                previous_owned = own_local - ctm.Int32(2)
                # The same at-most-two visits, with fixed peer/local roles.
                # Tile bounds and phase remain runtime values.
                if previous_owned > ctm.Int32(0) and previous_owned - ctm.Int32(1) < chunk_tiles:
                    peer_local = previous_owned - ctm.Int32(1)
                    pair_phase = _r93_issue_pv(
                        sPayloadBanks,
                        sV,
                        sVSF,
                        sUnit,
                        bars,
                        base_i,
                        peer_local,
                        begin_tile + peer_local,
                        chunk_tiles,
                        pair_id,
                        rank,
                        pair_phase,
                        False,
                        sUrRelay,
                    )
                if previous_owned < chunk_tiles:
                    pair_phase = _r93_issue_pv(
                        sPayloadBanks,
                        sV,
                        sVSF,
                        sUnit,
                        bars,
                        base_i,
                        previous_owned,
                        begin_tile + previous_owned,
                        chunk_tiles,
                        pair_id,
                        rank,
                        pair_phase,
                        True,
                        sUrRelay,
                    )
        if pair_rank == ctm.Int32(0) and own_count > ctm.Int32(0):
            _ds_wait(bars.subview(3), (own_count - ctm.Int32(1)) & ctm.Int32(1))
    if warp < ctm.Int32(8):
        running_max = ctm.Float32(-ctm.Float32.inf)
        running_anchor = ctm.Float32(-ctm.Float32.inf)
        for own_index in cutlass.range(0, rounds, 1, unroll=1):
            own_local = own_index * ctm.Int32(2) + pair_id
            # Final peer-only drain rounds never consume these defaults.
            own_sf0, own_sf1 = ctm.Int32(0), ctm.Int32(0)
            own_oldscale = ctm.Float32(1.0)
            peer_oldscale = ctm.Float32(1.0)
            if own_local < chunk_tiles:
                tile = begin_tile + own_local
                _ds_wait(
                    bars.subview(ctm.Int32(15) + (own_index & ctm.Int32(1))),
                    (own_index // ctm.Int32(2)) & ctm.Int32(1),
                )
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                sPayload = sPayloadBanks.subview(
                    ctm.Int32(PB_OWN) * ctm.Int32(DS_PAYLOAD_BYTES // 4)
                )
                if (tile + ctm.Int32(1)) * ctm.Int32(
                    256
                ) > valid_k_min:  # rows mode: any row of the block needs masking
                    pending, released_scale, released_bound = (
                        _ds_load_reg_tail_score_half_and_release(
                            base, warp, bars.subview(3), sReleaseCookie, softscale, valid_k, True
                        )
                    )
                    running_max, running_anchor, own_sf0, own_sf1, own_oldscale, peer_oldscale = (
                        _r93_pack(
                            pending,
                            sHalfMax,
                            sPayload,
                            bars.subview(6),
                            warp,
                            lane,
                            tile,
                            own_local,
                            released_bound,
                            released_scale,
                            pscale,
                            running_max,
                            running_anchor,
                            True,
                            use_mixed_imlp,
                            sBlockMax,
                            bars,
                            rank,
                            own_index,
                            peer_count,
                            sPsfMetaAlt,
                        )
                    )
                else:
                    pending, released_scale = _ds_load_reg_score_half_and_release(
                        base, warp, bars.subview(3), sReleaseCookie, softscale, False
                    )
                    running_max, running_anchor, own_sf0, own_sf1, own_oldscale, peer_oldscale = (
                        _r93_pack(
                            pending,
                            sHalfMax,
                            sPayload,
                            bars.subview(6),
                            warp,
                            lane,
                            tile,
                            own_local,
                            valid_k,
                            released_scale,
                            pscale,
                            running_max,
                            running_anchor,
                            False,
                            use_mixed_imlp,
                            sBlockMax,
                            bars,
                            rank,
                            own_index,
                            peer_count,
                            sPsfMetaAlt,
                        )
                    )
                _ds_publish_reg_half_payload(bars.subview(5))
            else:
                if own_local > ctm.Int32(0) and own_local - ctm.Int32(1) < chunk_tiles:
                    running_max, running_anchor, peer_oldscale = _r93_replay_peer(
                        sBlockMax,
                        bars,
                        (own_local - ctm.Int32(1)) // ctm.Int32(2),
                        peer_count,
                        rank,
                        warp,
                        lane,
                        running_max,
                        running_anchor,
                        softscale,
                    )
            # Compute the next owned tile BEFORE preparing the previous
            # peer PV. This permits both pairs' S2P to overlap.
            # At most two fixed pipeline roles, in the same order as
            # the original runtime range. Extents remain runtime.
            if own_local > ctm.Int32(0) and own_local - ctm.Int32(1) < chunk_tiles:
                peer_local = own_local - ctm.Int32(1)
                _r93_prep_pv(
                    sPayloadBanks,
                    bars,
                    base,
                    peer_local,
                    begin_tile + peer_local,
                    pair_id,
                    rank,
                    warp,
                    lane,
                    own_sf0,
                    own_sf1,
                    peer_oldscale,
                    False,
                    chunk_tiles,
                )
            if own_local < chunk_tiles:
                _r93_prep_pv(
                    sPayloadBanks,
                    bars,
                    base,
                    own_local,
                    begin_tile + own_local,
                    pair_id,
                    rank,
                    warp,
                    lane,
                    own_sf0,
                    own_sf1,
                    own_oldscale,
                    True,
                    chunk_tiles,
                )
        if chunk_tiles > ctm.Int32(0):
            last = chunk_tiles - ctm.Int32(1)
            _ds_wait(
                bars.subview(ctm.Int32(11) + (last & ctm.Int32(1))),
                (last // ctm.Int32(2)) & ctm.Int32(1),
            )
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        half = warp >> ctm.Int32(2)
        rowbase = (warp & ctm.Int32(3)) * ctm.Int32(32)
        row = rowbase + lane
        # rsplit: both pairs hold a rowsum partial (own tiles); half 0 of each pair sends
        # it to the same-query CTA of the other pair (st.async, bar25), both pairs wait
        # for the incoming partial and add. FP32 addition is commutative, so both pairs
        # compute bit-identical denominators.
        rs_ld = prims.tcgen05_ld(
            prims.Tcgen05LdStShape.SHAPE_32X32B,
            _ds_tmem_ptr(base, rowbase, ctm.Int32(DS_ROWSUM_ACC_TMEM)),
            num=1,
        )
        cute.arch.fence_view_async_tmem_load()
        rs_own = rs_ld[0].bitcast(ctm.Float32)
        if half == ctm.Int32(0):
            _store_async_cluster_b32(
                _mapa_shared_cluster(sDenom.data_ptr() + row, rank ^ ctm.Int32(2)),
                rs_own.bitcast(ctm.Int32),
                _mapa_shared_cluster(bars.subview(25), rank ^ ctm.Int32(2)),
            )
        _ds_wait(bars.subview(25), ctm.Int32(0))
        denom = rs_own + sDenom[row]
        # Split mode exported the FP32 partial + row stats for EVERY used plan slot, including requests the plan kept
        # whole (count == 1, `direct`): their normalized output is written below and the reducer never reads their
        # slot, so the export was 128 KB + 1 KB of dead HBM writes per CTA (118 MB per B128/q4 launch). Skip it.
        export_slot_row = slot_row_base
        if direct:
            export_slot_row = ctm.Int32(-1)
        if cutlass.const_expr(split_mode):
            # Milestone 2: export the unnormalized FP32 O slice, the row anchor (raw score units; the reducer applies
            # softmax_scale_log2 / (q_scale * kv_scale)) and the denominator; fp4_mla_packed_reduce.reduce_packed_splits
            # merges the request's tasks with output_scale = 1 / kv_global_scale (== pscale / FP4_MLA_P_GLOBAL_SCALE).
            if export_slot_row >= ctm.Int32(0):
                srow = slot_row_base + row
                part = ctm.make_array_view(mPartial)
                poff = srow * ctm.Int32(512) + pair_id * ctm.Int32(256) + half * ctm.Int32(128)
                for col in cutlass.range(0, ctm.Int32(128), 32, unroll=1):
                    tp = ctm.inttoptr(
                        base
                        + (rowbase << ctm.Int32(16))
                        + ctm.Int32(256)
                        + half * ctm.Int32(128)
                        + col,
                        6,
                        ctm.Float32,
                    )
                    values = prims.tcgen05_ld(prims.Tcgen05LdStShape.SHAPE_32X32B, tp, num=32)
                    cute.arch.fence_view_async_tmem_load()
                    part.subview(poff + col).data_ptr().nvvm_store_ext(
                        values[0:32], evict=prims.EvictPriority.NOALLOCATE
                    )
                if pair_id == ctm.Int32(0) and half == ctm.Int32(0):
                    ctm.make_array_view(mRowMax).subview(srow).data_ptr().store(running_anchor)
                    # denominator pre-multiplied by kv_global_scale: the reducer then normalizes with the constant
                    # output_scale = 1 (no host read of the device scale; CUDA-graph safe); same value as the
                    # direct path's 1 / (denom * kv_global_scale).
                    ctm.make_array_view(mRowSum).subview(srow).data_ptr().store(
                        denom * mKvGlobalScale[0]
                    )
        if direct:
            # FULL mode, or a split-mode request that the plan did not split: the reducer skips it (and is not
            # launched at all when no request is split), so the kernel writes the normalized output itself.
            norm = ctm.Float32(0.0)
            if denom > ctm.Float32(0.0):
                norm = (
                    cute.arch.rcp_approx(denom) * pscale * ctm.Float32(1.0 / FP4_MLA_P_GLOBAL_SCALE)
                )
            output_row = row + (bx // ctm.Int32(4)) * ctm.Int32(128)
            output_offset = (
                (query * runtime_m + output_row) * ctm.Int32(problem_size[0])
                + pair_id * ctm.Int32(256)
                + half * ctm.Int32(128)
            )
            output = ctm.make_array_view(mC_mnl)
            for col in cutlass.range(0, ctm.Int32(128), 32, unroll=1):
                tp = ctm.inttoptr(
                    base
                    + (rowbase << ctm.Int32(16))
                    + ctm.Int32(256)
                    + half * ctm.Int32(128)
                    + col,
                    6,
                    ctm.Float32,
                )
                values = prims.tcgen05_ld(prims.Tcgen05LdStShape.SHAPE_32X32B, tp, num=32)
                cute.arch.fence_view_async_tmem_load()
                out = (values[0:32] * norm).to(mC_mnl.element_type)
                output.subview(output_offset + col).data_ptr().nvvm_store_ext(
                    out, evict=prims.EvictPriority.NOALLOCATE
                )
    # fence_a: no all-thread fence.acq_rel.gpu at the boundary: a FULL task publishes nothing to
    # another CTA or kernel before the kernel boundary; intra-CTA handoffs are ordered by the bar.sync.
    prims.barrier_cta_sync(0, thread_count=384)
    # fence_b: barrier.cluster.arrive.relaxed. Nothing written before this arrive is read by a peer after its wait:
    # every remote arrive or complete_tx on this CTA's mbarriers was consumed by one of its own waits before the
    # bar.sync above, or is a tcgen05.commit release that has no waiter at the end of a task: the K-stage releases
    # (bars[27..30]) of the last two own tiles and the V-free releases (bars[DS_BAR_V_FREE + vbank]) of the last
    # DS_V_BANKS tiles.  Those are issued by the pair leader's issuer ahead of (K stage) or right behind (V free) a
    # commit of the same packet that this CTA does wait (bars[15 + bank] / bars[11 + bank]), so they land with it,
    # the whole epilogue before the bar.sync.  Every DSMEM payload (sDenom st.async included) was consumed in-loop,
    # and TMEM hazards are bracketed by tcgen05.fence::before/after_thread_sync.
    # The barrier keeps every CTA resident until its peers' DSMEM traffic into it has completed;
    # barrier.cluster.wait keeps .acquire.
    cute.arch.cluster_arrive_relaxed()
    cute.arch.cluster_wait()
    _dealloc_tmem_cluster(bars.subview(4), tmem, warp, rank, dealloc_warp_id=8)
