# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, E741, F841

import math
from collections.abc import Callable
from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass as ctm
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass.base_dsl.array import EvictPriority
from cutlass.cute.arch.nvvm_wrappers import inline_ptx as cute_inline_ptx
from cutlass.cute.runtime import make_ptr
from cutlass.experimental import cuda as cuda_tma
from cutlass.experimental import primitives as prims

from .call_cache import LruCache, class_defines_callables
from .cute_dsl_utils import _compile_cutedsl, _current_cu_stream
from .cutedsl_common import (
    CUDA_GRID_Z_MAX,
    FP4_MLA_P_GLOBAL_SCALE,
    INT32_MAX,
    LOG2_6,
    LOG2_E,
    _ceil_div,
    _compact_sfb_mma_cols_per_scale_chunk,
    _compute_smem_p4_lazy_rebase_scale,
    _cutlass_output_dtype,
    _derive_qk_scale_burst_kblocks,
    _fence_async_shared_cta,
    _float_to_ordered_u32_for_atomic_max,
    _lookup_physical_page,
    _mapa_shared_cluster,
    _mask_invalid_score,
    _mma_block_scale_kind,
    _mma_scale_format,
    _mma_scale_vec_size_enum,
    _pack_e2m1x8_from_f16x2,
    _publish_p4_n256_atomic_rowmax,
    _qk_compact_sfb_cols_per_scale_chunk,
    _qk_sfb_smem_nchunks_per_cta,
    _rescale_smem_p4_resident_o_in_tmem,
    _scale_words_for_k,
    _sfa_tmem_cols_for_scale_chunks,
    _softmax_exp2_pair_packed_f16x2,
    _unpack_f16x2_to_f32_pair,
    _validate_output_dtype,
    _validate_tensor_pointer_alignment,
    nvvm_add_packed_f32x2,
)
from .cutedsl_common import _as_shared_cta as _as_tma_completion_mbar
from .cutedsl_common import (
    _mbarrier_arrive_release_cta_shared_cluster as _mbarrier_arrive_shared_cluster,
)
from .cutedsl_common import _pack_p4_16_scores_with_sum as _pack_p4_16_scores_from_group
from .cutedsl_common import _tcgen05_ld_red_32x32b_x16_max_f32 as _tcgen05_ld_red_x16_max_f32

# Defined once in the 2x kernel module: these read nothing the two kernels define differently.
from .fp4_mla_cutedsl_mufu16 import (
    _finalize_p4_n256_score_half_psf_source,
    _issue_mxf4nvf4_mma_tile_from_base,
    _issue_p4_n256_psf_bank_to_tmem_from_desc,
    _issue_smem_p4_qk_score_slot_from_base,
    _kv_cache_3d_layout,
    _load_global_page_native_tile_pair,
    _load_p4_n256_atomic_running_rowmax,
    _load_p4_n256_atomic_running_rowmax_with_validity,
    _load_staged_page_native_tile_pair,
    _make_smem_p4_qk_ksf_stage_desc,
    _permute_raw_trtllm_q_tail_in_smem,
    _prepare_p4_n256_score_half_tmem_addresses,
    _PreparedFusedCall,
    _qk_compact_ring_stage_byte_offset,
    _select_runtime_kv_profile,
    _stage_smem_p4_runtime_scale_pair,
    _store_prepacked_p4_n256_score_half_to_direct_a,
    _store_raw_trtllm_qsf_128dp_unique_owner_to_tmem_from_smem,
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
QK_PREFIX_DIM = TRTLLM_V_HEAD_DIM
TRTLLM_QSF_EVEN_PRMT_SELECTOR = 25632
TRTLLM_QSF_ODD_PRMT_SELECTOR = 30001
SF_VEC_SIZE_OPTIONS = (16,)
SF_VEC_SIZE = 16
MMA_KBLOCK_DIM = 64
MMA_KBLOCK_IDESC_K_DIM = 0
MMA_SCALE_CHUNK_K = SF_VEC_SIZE * 4
MMA_SCALE_CHUNKS_PER_KBLOCK = _scale_words_for_k(MMA_KBLOCK_DIM, SF_VEC_SIZE)
MMA_SCALE_VEC_SIZE = _mma_scale_vec_size_enum(SF_VEC_SIZE)
MMA_SCALE_FORMAT = _mma_scale_format(SF_VEC_SIZE)
MMA_BLOCK_SCALE_KIND = _mma_block_scale_kind(SF_VEC_SIZE)
QK_MMA_KBLOCKS = QK_LOGICAL_DIM // MMA_KBLOCK_DIM
QK_EFFECTIVE_MMA_KBLOCKS = QK_EFFECTIVE_DIM // MMA_KBLOCK_DIM
QK_TAIL_EFFECTIVE_MMA_KBLOCKS = QK_EFFECTIVE_MMA_KBLOCKS - QK_PREFIX_DIM // MMA_KBLOCK_DIM
FIRST_ORDER_TAIL_A_LOCAL_KBLOCKS = (0, 1, 0)
FIRST_ORDER_TAIL_B_LOCAL_KBLOCKS = (0, 0, 1)
QK_SF_GROUPS = QK_PADDED_DIM // SF_VEC_SIZE
SMEM_P4_PAGE_PLAN_PROFILE_KV = 160 * 1024
SMEM_P4_RUNTIME_PROFILE_PAGE_ALIGNMENT = 4
SMEM_P4_RUNTIME_PROFILE_KV_ALIGNMENT = SMEM_P4_RUNTIME_PROFILE_PAGE_ALIGNMENT * TRTLLM_PAGE_SIZE
# Keep the device-side ceil division ``valid_k + KV_TILE - 1`` in Int32 range.
SMEM_P4_RUNTIME_MAX_KV = (
    (INT32_MAX - (KV_TILE - 1)) // SMEM_P4_RUNTIME_PROFILE_KV_ALIGNMENT
) * SMEM_P4_RUNTIME_PROFILE_KV_ALIGNMENT
SMEM_P4_PAGE_ID_PLAN_INTS = SMEM_P4_PAGE_PLAN_PROFILE_KV // TRTLLM_PAGE_SIZE
SMEM_P4_PAGE_ID_PLAN_BYTES = SMEM_P4_PAGE_ID_PLAN_INTS * ctm.Int32.bytes
SMEM_P4_RUNTIME_SCALE_PAIR_FLOATS = 2
SMEM_P4_RUNTIME_SCALE_PAIR_BYTES = SMEM_P4_RUNTIME_SCALE_PAIR_FLOATS * ctm.Float32.bytes
# Lazy-anchor rebase threshold (log2), fixed for this kernel: TRTLLM_FP4_MLA_LAZY_ANCHOR_LOG2 tunes the 2x and 4x
# kernels only.
SMEM_P4_LAZY_ANCHOR_REBASE_LOG2 = 1.0
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
SMEM_P4_QK_SCORE_SLOTS = SMEM_P4_TMEM_BANKS
SMEM_P4_QK_SCORE_RING_SLOTS = SMEM_P4_TMEM_BANKS
SMEM_P4_QK_PIPELINE_SLOT_STRIDE = SMEM_P4_QK_SCORE_SLOT_N // SMEM_P4_TMEM_WARP_N
SMEM_P4_QK_SCORE_SLOT_STRIDE = SMEM_P4_SCORE_HALF_TMEM_COLS
SMEM_P4_QK_MMA_KBLOCK_DIM = MMA_KBLOCK_DIM
SMEM_P4_QK_MMA_KBLOCK_IDESC_K_DIM = MMA_KBLOCK_IDESC_K_DIM
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
SMEM_P4_P4_PIPELINE_STAGES = SMEM_P4_TMEM_BANKS
SMEM_P4_P4_CONTROL_MBAR_STAGES = SMEM_P4_TMEM_BANKS * SMEM_P4_SCORE_SLOTS_PER_KV_TILE
SMEM_P4_P4_SMEM_PHYSICAL_STAGES = SMEM_P4_TMEM_BANKS * SMEM_P4_SCORE_SLOTS_PER_KV_TILE
SMEM_P4_P4_SMEM_WORDS = SMEM_P4_P4_SMEM_PHYSICAL_STAGES * SMEM_P4_P4_SMEM_STAGE_WORDS
SMEM_P4_P4_SMEM_MMA_LEADING_BYTE_OFFSET = 0
SMEM_P4_P4_SMEM_MMA_STRIDE_BYTE_OFFSET = 512
SMEM_P4_P4_SMEM_MMA_LAYOUT = 4
SMEM_P4_SCORE_SLOT_ROW_STATE_STRIDE = SMEM_P4_BMM1_M * SMEM_P4_TMEM_WARP_N
SMEM_P4_P_TMEM_KBLOCK_COLS = MMA_KBLOCK_DIM // 2 // 4
SMEM_P4_PV_KBLOCKS_PER_SCORE_SLOT = SMEM_P4_BMM2_STAGE_K // MMA_KBLOCK_DIM
SMEM_P4_PV_KBLOCKS_PER_LI = SMEM_P4_SCORE_SLOTS_PER_KV_TILE * SMEM_P4_PV_KBLOCKS_PER_SCORE_SLOT
SMEM_P4_P_TMEM_STAGE_COLS = SMEM_P4_PV_KBLOCKS_PER_SCORE_SLOT * SMEM_P4_P_TMEM_KBLOCK_COLS
SMEM_P4_P_TMEM_SLOT_STRIDE = SMEM_P4_P_TMEM_STAGE_COLS
SMEM_P4_SCORE_TMEM_COLS = (
    0,
    SMEM_P4_QK_SCORE_RING_SLOTS * SMEM_P4_QK_PIPELINE_SLOT_STRIDE,
)
SMEM_P4_SCORE_PONG_TMEM_COLS = (SMEM_P4_SCORE_TMEM_COLS[1], 256)
SMEM_P4_O_ACC_TMEM_COLS = (
    SMEM_P4_SCORE_PONG_TMEM_COLS[1],
    SMEM_P4_SCORE_PONG_TMEM_COLS[1] + SMEM_P4_BMM2_N,
)
SMEM_P4_O_ACC_TILE_COLS = SMEM_P4_BMM2_N // SMEM_P4_TMEM_WARP_N
SMEM_P4_O_ACC_COL_OFFSETS = tuple(
    (SMEM_P4_O_ACC_TMEM_COLS[0] + t * SMEM_P4_O_ACC_TILE_COLS for t in range(SMEM_P4_N_OUT_TILES))
)
SMEM_P4_SFA_LAYOUT = 1
SMEM_P4_SFA_COLS_PER_SCALE_CHUNK = 1
SMEM_P4_P_SFA_LAYOUT = 1
SMEM_P4_P_SFA_COLS_PER_SCALE_CHUNK = 1
SMEM_P4_SFA_UNIQUE_CHUNKS_PER_UTCCP = 4
SMEM_P4_QK_B_MMA_LEADING_BYTES = 16
SMEM_P4_QK_B_MMA_STRIDE_BYTES = 1024
SMEM_P4_QK_B_MMA_LAYOUT = 2
SMEM_P4_QK_B_MMA_SMEM_WORD_DELTA = 0
SMEM_P4_QK_B_MMA_K_SEGMENT_OFFSET = 0
SMEM_P4_QK_SFB_MMA_COL_DELTA = 0
SMEM_P4_QK_SFB_EXTRA_ROW_STRIDE = 0
SMEM_P4_QK_SFB_EXTRA_COL_STRIDE = -1
SMEM_P4_QK_SFB_S2T_STRIDE_BYTES = 128
SMEM_P4_QK_SFB_CP_GROUP_ONE = False
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
SMEM_P4_P_ARCHIVE_TMEM_COLS = SMEM_P4_SCORE_TMEM_COLS
SMEM_P4_P_ARCHIVE_SLOT_STRIDE = SMEM_P4_QK_PIPELINE_SLOT_STRIDE
SMEM_P4_PV_VSF_N_TILE_COLS = SMEM_P4_SCORE_SLOTS_PER_KV_TILE * SMEM_P4_PV_SFB_SCORE_SLOT_COLS
SMEM_P4_SCALE_FEED_STAGE_COLS = 28
SMEM_P4_SCALE_FEED_TMEM_COLS = (
    SMEM_P4_O_ACC_TMEM_COLS[1],
    SMEM_P4_O_ACC_TMEM_COLS[1] + SMEM_P4_SCORE_SLOTS_PER_KV_TILE * SMEM_P4_SCALE_FEED_STAGE_COLS,
)
SMEM_P4_QK_ACC_COL_OFFSET = SMEM_P4_SCORE_TMEM_COLS[0]
SMEM_P4_O_ACC_COL_OFFSET = SMEM_P4_O_ACC_TMEM_COLS[0]
SMEM_P4_SCALE_A_COL_OFFSET = SMEM_P4_SCALE_FEED_TMEM_COLS[0]
SMEM_P4_QK_COMPACT_SFB_S2T_BYTE_OFFSET = 0
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
SMEM_P4_SCALE_B_COL_OFFSET = (SMEM_P4_SCALE_A_COL_OFFSET + SMEM_P4_QK_SFA_FEED_COLS + 15) // 16 * 16
SMEM_P4_P_SFA_BANK_REL_COL_OFFSET = 32
SMEM_P4_PV_VSF_TAIL_COL_OFFSET = SMEM_P4_SCALE_B_COL_OFFSET
SMEM_P4_RESIDENT_Q_SF_COL_OFFSET = SMEM_P4_SCALE_FEED_TMEM_COLS[0]
SMEM_P4_QK_SFA_COL_OFFSET = SMEM_P4_RESIDENT_Q_SF_COL_OFFSET
SMEM_P4_OLD_SCALE_TMEM_SLOTS = SMEM_P4_ROWMETA_SLOTS
SMEM_P4_STAGE_ROWSUM_TMEM_COLS = SMEM_P4_TMEM_BANKS * SMEM_P4_SCORE_SLOTS_PER_KV_TILE
SMEM_P4_STAGE_ROW_STATE_SMEM_FLOATS = (
    SMEM_P4_TMEM_BANKS * SMEM_P4_SCORE_SLOTS_PER_KV_TILE * SMEM_P4_SCORE_SLOT_ROW_STATE_STRIDE
)
SMEM_P4_ROW_SCALE_RING_FLOATS = SMEM_P4_TMEM_BANKS * SMEM_P4_BMM1_M
SMEM_P4_FINAL_ANCHOR_ROWSUM_FLOATS = SMEM_P4_BMM1_M
SMEM_P4_ATOMIC_RUNNING_ROWMAX_SMEM_ROWS = SMEM_P4_BMM1_M
SMEM_P4_TMEM_SCORE_PIPELINE_STAGES = SMEM_P4_QK_PIPELINE_SLOTS
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
SMEM_P4_PAIR_FINAL_PV_ISSUED_MBAR_OFFSET = (
    SMEM_P4_PAIR_HALF1_ROWSUM_READY_MBAR_OFFSET + SMEM_P4_QK_PIPELINE_SLOTS
)
SMEM_P4_PAIR_ROWMAX_READ_DONE_MBAR_OFFSET = (
    SMEM_P4_PAIR_P_HALF_READY_MBAR_OFFSET + SMEM_P4_P_HALF_COUNT * SMEM_P4_QK_PIPELINE_SLOTS
)
SMEM_P4_PAIR_DIRECT_P_READY_MBAR_OFFSET = SMEM_P4_PAIR_ROWSUM_DONE_MBAR_OFFSET
SMEM_P4_QK15_STAGE1_MBAR_OFFSET = (
    SMEM_P4_PAIR_DIRECT_P_READY_MBAR_OFFSET + SMEM_P4_QK_PIPELINE_SLOTS
)
SMEM_P4_PAIR_OVERLAP_MBAR_COUNT = (
    SMEM_P4_PAIR_ROWMAX_READ_DONE_MBAR_OFFSET + SMEM_P4_QK_PIPELINE_SLOTS
)
SMEM_P4_V_PIPELINE_STAGES = 2
SMEM_P4_PV_PAGE_B_MMA_LEADING_BYTES = 0
SMEM_P4_PV_PAGE_B_MMA_STRIDE_BYTES = 512
SMEM_P4_PV_PAGE_B_MMA_LAYOUT = 4
SMEM_P4_PV_B_MMA_K_SEGMENT_OFFSET = 0
SMEM_P4_PV_SFB_MMA_COL_DELTA = 0
SMEM_P4_PV_SFB_EXTRA_ROW_STRIDE = 0
SMEM_P4_PV_SFB_EXTRA_COL_STRIDE = -1
SMEM_P4_PV_SFB_S2T_STRIDE_BYTES = 128
SMEM_P4_PV_SFB_CP_GROUP_ONE = False
SMEM_P4_QK_COMPLETION_MBARS = SMEM_P4_TMEM_SCORE_PIPELINE_STAGES


SMEM_P4_PRODUCER_WARPS_PER_SCORE_SLOT = SMEM_P4_TMEM_WARP_M * SMEM_P4_TMEM_WARP_N
SMEM_P4_COMPUTE_WARPS = SMEM_P4_PRODUCER_WARPS_PER_SCORE_SLOT
SMEM_P4_CORRECTION_WARPS = SMEM_P4_PRODUCER_WARPS_PER_SCORE_SLOT
SMEM_P4_COMPUTE_WARP_ID_BEGIN = 0
SMEM_P4_CORRECTION_WARP_ID_BEGIN = SMEM_P4_COMPUTE_WARP_ID_BEGIN + SMEM_P4_COMPUTE_WARPS
SMEM_P4_TOTAL_PRODUCER_WARPS = SMEM_P4_COMPUTE_WARPS + SMEM_P4_CORRECTION_WARPS
SMEM_P4_STORE_WARPS = SMEM_P4_CORRECTION_WARPS
SMEM_P4_COMPUTE_THREADS = 32 * SMEM_P4_COMPUTE_WARPS
SMEM_P4_CORRECTION_THREADS = 32 * SMEM_P4_CORRECTION_WARPS
SMEM_P4_SCORE_SLOT_PRODUCER_THREADS = SMEM_P4_COMPUTE_THREADS
THREADS_PER_CTA = 32 * 12
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
MMA_KBLOCKS_PER_TILE = MMA_TILER_MNK[2] // MMA_KBLOCK_DIM
SMEM_P4_QK_DATA_STAGE_KBLOCKS = QK_DATA_STAGE_K_DIM // MMA_KBLOCK_DIM
SMEM_P4_QK_DATA_STAGES = (
    QK_MMA_KBLOCKS + SMEM_P4_QK_DATA_STAGE_KBLOCKS - 1
) // SMEM_P4_QK_DATA_STAGE_KBLOCKS
SMEM_P4_QK_SMEM_PIPELINE_SLOTS = 3
SMEM_P4_QK_BARRIER_STAGES = SMEM_P4_QK_SMEM_PIPELINE_SLOTS
SMEM_P4_QK_KTILE_RING_STAGES = SMEM_P4_QK_SMEM_PIPELINE_SLOTS
SMEM_P4_QK_BULK_READY_MBARS = SMEM_P4_QK_SMEM_PIPELINE_SLOTS
SMEM_P4_QK_DUAL_ARM_MBARS = SMEM_P4_QK_SMEM_PIPELINE_SLOTS
CLUSTER_SHAPE_MNK = (2, 1, 1)
MMA_SCALE_CHUNKS_PER_TILE = _scale_words_for_k(MMA_TILER_MNK[2], SF_VEC_SIZE)
SMEM_P4_V_N_PER_CTA = SMEM_P4_BMM2_N // CLUSTER_SHAPE_MNK[0]
NUM_QK_Q_STAGE = SMEM_P4_QK_DATA_STAGES
NUM_QK_Q_PIPELINE_STAGE = SMEM_P4_QK_DATA_STAGES
NUM_QK_AB_STAGE = SMEM_P4_QK_DATA_STAGES
MMA_WARP_ID = 8
TMA_QK_WARP_ID = 9
TMA_V_WARP_ID = 10
ROWMETA_WARP_ID = 11
SMEM_P4_ALU_REGISTER_BUDGET = 224
SMEM_P4_AUX_REGISTER_BUDGET = 56
A_STAGE_BYTES = MMA_TILER_MNK_PER_CTA[0] * MMA_TILER_MNK[2] // 2
B_STAGE_BYTES = SMEM_P4_V_N_PER_CTA * MMA_TILER_MNK[2] // 2
SFA_STAGE_BYTES = 512 * _scale_words_for_k(MMA_TILER_MNK[2], min(SF_VEC_SIZE_OPTIONS))
SFB_STAGE_BYTES = SFA_STAGE_BYTES * 2
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
QK_FULL_MMA_LEADING_BYTE_OFFSET = 16
QK_FULL_MMA_STRIDE_BYTE_OFFSET = 1024
QK_FULL_MMA_LAYOUT = 2
QK_TAIL_MMA_LEADING_BYTE_OFFSET = 0
QK_TAIL_MMA_STRIDE_BYTE_OFFSET = 512
QK_TAIL_MMA_LAYOUT = 4
QK_K_TAIL_MMA_LEADING_BYTE_OFFSET = 0
QK_K_TAIL_MMA_STRIDE_BYTE_OFFSET = 512
QK_K_TAIL_MMA_LAYOUT = 4
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
SMEM_P4_V_PAGE_N_TILE_BYTES = TRTLLM_PAGE_SIZE * SMEM_P4_V_N_PER_CTA // 2
SMEM_P4_VSF_PAGE_BYTES = (
    SMEM_P4_SCALE_CHUNKS_PER_PAGE * SMEM_P4_PV_SFB_N_CHUNKS * SMEM_P4_N_OUT_TILES * 512
)
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
SMEM_P4_RAW_V_STAGE_BYTES = SMEM_P4_V_DATA_STAGE_BYTES
SMEM_P4_V_SFB_TMA_STAGE_BYTES = SFA_TMA_STAGE_BYTES * SMEM_P4_PV_SFB_N_CHUNKS * SMEM_P4_N_OUT_TILES
SMEM_P4_V_TILE_TMA_BYTES = SMEM_P4_V_DATA_STAGE_BYTES + SMEM_P4_V_SFB_TMA_STAGE_BYTES
SMEM_P4_CONTROL_MBAR_COUNT = (
    SMEM_P4_QK_COMPLETION_MBARS
    + SMEM_P4_QK_PIPELINE_SLOTS
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
    + SMEM_P4_QK_DUAL_ARM_MBARS
    + 1
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
            f"QK_PADDED_DIM must cover QK_LOGICAL_DIM and be a whole Kblock; got padded={QK_PADDED_DIM}, logical={QK_LOGICAL_DIM}, kblock={MMA_KBLOCK_DIM}"
        )
    if QK_DATA_STAGE_K_DIM != MMA_TILER_MNK[2]:
        raise ValueError(
            f"QK_DATA_STAGE_K_DIM must remain K=256 for the FP4 s128b full-stage descriptor contract; got {QK_DATA_STAGE_K_DIM}"
        )
    if QK_TAIL_DATA_STAGE_K_DIM != 2 * MMA_KBLOCK_DIM:
        raise ValueError(
            f"QK_TAIL_DATA_STAGE_K_DIM must remain K=128 for the logical K640 tail; got {QK_TAIL_DATA_STAGE_K_DIM}"
        )
    if QK_K_TAIL_DATA_STAGE_K_DIM != 2 * MMA_KBLOCK_DIM:
        raise ValueError(
            f"QK_K_TAIL_DATA_STAGE_K_DIM must remain physical K=128 for the TensorRT-LLM K640 tail; got {QK_K_TAIL_DATA_STAGE_K_DIM}"
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
            f"QK full-stage plus tail split must cover QK_LOGICAL_DIM exactly; got tail_stage_end={qk_tail_stage_end}, logical={QK_LOGICAL_DIM}"
        )
    qk_k_tail_stage_end = (
        SMEM_P4_QK_TAIL_STAGE_KBLOCK_START * MMA_KBLOCK_DIM + QK_K_TAIL_DATA_STAGE_K_DIM
    )
    if qk_k_tail_stage_end != TRTLLM_K_STORAGE_DIM:
        raise ValueError(
            f"QK K full-stage plus physical tail split must cover K640 exactly; got tail_stage_end={qk_k_tail_stage_end}, physical={TRTLLM_K_STORAGE_DIM}"
        )
    if QK_PADDED_DIM // 2 % 64 != 0:
        raise ValueError(
            f"QK padded FP4 row stride must be 64B aligned for the K128 tail; got {QK_PADDED_DIM // 2} bytes"
        )
    if TRTLLM_K_STORAGE_DIM // 2 % 64 != 0:
        raise ValueError(
            f"TensorRT-LLM K640 packed row stride must be 64B aligned for the s64b K128 tail; got {TRTLLM_K_STORAGE_DIM // 2} bytes"
        )
    if KV_TILE % sf_vec_size != 0:
        raise ValueError(f"KV_TILE={KV_TILE} must be divisible by sf_vec_size={sf_vec_size}")
    if SMEM_P4_SCORE_SLOT_N % sf_vec_size != 0:
        raise ValueError(
            f"SMEM_P4 score slot must divide the FP4 scale-vector size; got N={SMEM_P4_SCORE_SLOT_N}, sf_vec_size={sf_vec_size}"
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
    qk_scale_b_col_offset = (SMEM_P4_SCALE_A_COL_OFFSET + qk_sfa_feed_cols + 15) // 16 * 16
    qk_scale_feed_col_end = max(
        SMEM_P4_SCALE_A_COL_OFFSET + qk_sfa_feed_cols,
        qk_scale_b_col_offset + qk_sfb_feed_cols,
    )
    p_sfa_tail_col_offset = qk_scale_b_col_offset - p_scale_feed_stage_cols
    pv_vsf_tail_col_offset = qk_scale_b_col_offset
    pv_vsf_tail_col_end = (
        pv_vsf_tail_col_offset
        + SMEM_P4_N_OUT_TILES * SMEM_P4_SCORE_SLOTS_PER_KV_TILE * pv_sfb_score_slot_cols
    )
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
        + SMEM_P4_RAW_V_STAGE_BYTES
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
        SMEM_P4_SCALE_B_COL_OFFSET=qk_scale_b_col_offset,
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


TMEM_BAR_ID = 1
TMEM_BAR_THREADS = 32 * (1 + SMEM_P4_TOTAL_PRODUCER_WARPS)
O_STORE_BAR_ID = 2
SMEM_P4_RAW_V_READY_BAR_ID = 3
SMEM_P4_RAW_V_READY_BAR_THREADS = 32 * (SMEM_P4_TOTAL_PRODUCER_WARPS + 1)
SMEM_P4_P4_SLOT0_BAR_ID = 6
SMEM_P4_P4_SLOT1_BAR_ID = 7
SMEM_P4_QK1_SPLIT_ARM_BAR_ID = 8
SMEM_P4_QK1_SPLIT_ARM_BAR_THREADS = 32 * 2
SMEM_P4_QSF_OWNER_BAR_ID = 9
SMEM_P4_P_SFA_OWNER_BAR_ID = 10
SMEM_P4_DUAL_HALF_BAR_ID = SMEM_P4_P_SFA_OWNER_BAR_ID
SMEM_P4_DUAL_HALF_BAR_THREADS = 32 * SMEM_P4_TOTAL_PRODUCER_WARPS
O_STORE_BAR_THREADS = 32 * SMEM_P4_STORE_WARPS
SMEM_P4_QSF_OWNER_BAR_THREADS = 32 * (SMEM_P4_COMPUTE_WARPS + 1)


@cute.jit
def fused_fp4_mla_decode_ctm(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    q_sf_ptr: cute.Pointer,
    k_sf_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    page_table_ptr: cute.Pointer,
    valid_k_ptr: cute.Pointer,
    helix_kv_bounds_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    accum_ptr: cute.Pointer,
    row_max_ptr: cute.Pointer,
    row_sum_ptr: cute.Pointer,
    page_indptr_ptr: cute.Pointer,
    q_global_scale_ptr: cute.Pointer,
    kv_global_scale_ptr: cute.Pointer,
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
    stream: cuda.CUstream,
    page_size: ctm.Constexpr = KV_TILE,
    use_mixed_imlp: ctm.Constexpr = False,
    query_len_per_seq: ctm.Constexpr = 1,
    use_helix_kv_bounds: ctm.Constexpr = False,
    use_smem_page_plan: ctm.Constexpr = True,
    use_consecutive_page_pair: ctm.Constexpr = False,
    use_ksf_gather4: ctm.Constexpr = False,
    write_softmax_stats: ctm.Constexpr = False,
    store_split: ctm.Constexpr = False,
) -> None:
    n, k = problem_size
    m = runtime_m
    l = runtime_l
    logical_k = cute.assume(k, 32)
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
    k_v_raw_tensor = cute.make_tensor(
        cute.recast_ptr(k_ptr, dtype=cutlass.Uint8),
        cute.make_layout(
            (TRTLLM_V_HEAD_DIM // 2, page_size, num_cache_pages),
            stride=(1, TRTLLM_K_STORAGE_DIM // 2, kv_page_stride_bytes),
        ),
    )
    page_table_tensor = cute.make_tensor(
        cute.recast_ptr(page_table_ptr, dtype=cutlass.Int32),
        cute.make_layout(((k // page_size) * (l // query_len_per_seq),), stride=(1,)),
    )
    page_indptr_tensor = cute.make_tensor(
        cute.recast_ptr(page_indptr_ptr, dtype=cutlass.Int32),
        cute.make_layout((l // query_len_per_seq + 1,), stride=(1,)),
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
        cute.make_layout((l // query_len_per_seq,), stride=(1,)),
    )
    helix_kv_bounds_tensor = cute.make_tensor(
        cute.recast_ptr(helix_kv_bounds_ptr, dtype=cutlass.Int32),
        cute.make_layout((l,), stride=(1,)),
    )
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
        (cute.assume(m, 32), cute.assume(n, 16), l),
        stride=(cute.assume(n, 16), 1, cute.assume(m * n, 512)),
    )
    c_tensor = cute.make_tensor(c_ptr, c_layout)
    accum_tensor = cute.make_tensor(accum_ptr, c_layout)
    row_state_layout = cute.make_layout((cute.assume(m, 32), l), stride=(1, cute.assume(m, 32)))
    row_max_tensor = cute.make_tensor(row_max_ptr, row_state_layout)
    row_sum_tensor = cute.make_tensor(row_sum_ptr, row_state_layout)
    q_sf_blocks = QK_SF_GROUPS // 4
    k_sf_blocks = TRTLLM_K_SF_GROUPS // 4
    pv_sf_blocks = k // SF_VEC_SIZE // 4
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
        box_dims=(QK_DATA_STAGE_K_DIM, MMA_TILER_MNK_PER_CTA[0], 1),
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
    tma_k_v_raw_desc = cuda_tma.create_tensor_map_tiled_from_view(
        k_v_raw_tensor,
        box_dims=(SMEM_P4_V_N_PER_CTA // 2, TRTLLM_PAGE_SIZE, 1),
        stride_order=(0, 1, 2),
        swizzle=cuda_tma.TensorMapSwizzle.none,
        tma_format=cuda_tma.TensorMapDataFormat.BYTE,
    )
    tma_q_tail_desc = cuda_tma.create_tensor_map_tiled_from_view(
        q_f4_tensor,
        box_dims=(QK_TAIL_DATA_STAGE_K_DIM, MMA_TILER_MNK_PER_CTA[0], 1),
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
            SMEM_P4_PV_SFB_N_CHUNKS * SMEM_P4_N_OUT_TILES,
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
            SMEM_P4_N_OUT_TILES,
            v_pair_page_box,
            1,
        ),
        stride_order=(0, 1, 2, 3, 4),
        swizzle=cuda_tma.TensorMapSwizzle.s64b,
        tma_format=cuda_tma.TensorMapDataFormat.B4X16,
    )
    tma_sfb_desc = tma_vsf_stage_desc
    kernel(
        page_table_tensor,
        page_indptr_tensor,
        valid_k_tensor,
        helix_kv_bounds_tensor,
        q_global_scale_tensor,
        kv_global_scale_tensor,
        tma_q_desc,
        tma_k_page_desc,
        tma_k_v_raw_desc,
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
        accum_tensor,
        row_max_tensor,
        row_sum_tensor,
        problem_size,
        runtime_m,
        v_page_offset,
        softmax_scale_log2,
        pv_output_scale,
        page_size,
        query_len_per_seq,
        use_helix_kv_bounds,
        use_smem_page_plan,
        use_mixed_imlp,
        use_consecutive_page_pair,
        use_ksf_gather4,
        write_softmax_stats,
        store_split,
    ).launch(
        grid=(
            cute.ceil_div(c_tensor.shape[0], SMEM_P4_CTA_GROUP_M) * CLUSTER_SHAPE_MNK[0],
            cute.ceil_div(OUT_DIM, MMA_TILER_MNK[1]),
            c_tensor.shape[2],
        ),
        block=[THREADS_PER_CTA, 1, 1],
        cluster=CLUSTER_SHAPE_MNK,
        stream=stream,
        min_blocks_per_mp=1,
    )


@cute.jit
def _load_qk_qonly_kblock_stage(
    tma_q_ptr,
    tma_q_tail_ptr,
    tma_qsf_ptr,
    tma_qsf_tail_ptr,
    q_tma_mbar,
    sA,
    sSFA,
    tidx: ctm.Int32,
    bidx: ctm.Int32,
    bidz: ctm.Int32,
    cta_rank: ctm.Int32,
    qk_kblock_idx: ctm.Constexpr,
    stage: ctm.Constexpr,
    q_tma_phase: ctm.Constexpr,
    manage_mbarrier: ctm.Constexpr = True,
) -> None:
    del tidx
    sA_stage = sA.subview(QK_A_STAGE_BYTES // 4 * stage)
    sSFA_stage = sSFA.subview(QK_SFA_STAGE_BYTES * stage)
    if cutlass.const_expr(qk_kblock_idx >= SMEM_P4_QK_TAIL_STAGE_KBLOCK_START):
        tma_q_stage_ptr = tma_q_tail_ptr
        tma_qsf_stage_ptr = tma_qsf_tail_ptr
        q_tma_bytes: ctm.Constexpr = SMEM_P4_Q_ONLY_TAIL_STAGE_BYTES * CLUSTER_SHAPE_MNK[0]
    else:
        tma_q_stage_ptr = tma_q_ptr
        tma_qsf_stage_ptr = tma_qsf_ptr
        q_tma_bytes: ctm.Constexpr = SMEM_P4_Q_ONLY_STAGE_BYTES * CLUSTER_SHAPE_MNK[0]
    mcast_mask = cutlass.Int16(1) << cutlass.Int16(cta_rank)
    if cutlass.const_expr(manage_mbarrier):
        if cta_rank == ctm.Int32(0):
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(q_tma_mbar, q_tma_bytes)
    if prims.elect_sync():
        m_tile_idx = bidx // ctm.Int32(CLUSTER_SHAPE_MNK[0])
        prims.cp_async_bulk_tensor_shared_cluster_global(
            sA_stage,
            tma_q_stage_ptr,
            (
                ctm.Int32(qk_kblock_idx * MMA_KBLOCK_DIM),
                m_tile_idx * ctm.Int32(SMEM_P4_CTA_GROUP_M) + cta_rank * ctm.Int32(SMEM_P4_BMM1_M),
                bidz,
            ),
            _as_tma_completion_mbar(q_tma_mbar),
            [],
            multicast_mask=mcast_mask,
            group=prims.CTAGroup.CTA_2,
        )
        qsf_m_tile = m_tile_idx
        prims.cp_async_bulk_tensor_shared_cluster_global(
            sSFA_stage,
            tma_qsf_stage_ptr,
            (
                ctm.Int32(0),
                ctm.Int32(qk_kblock_idx * SMEM_P4_QK_MMA_SCALE_CHUNKS_PER_KBLOCK),
                qsf_m_tile,
                bidz,
            ),
            _as_tma_completion_mbar(q_tma_mbar),
            [],
            multicast_mask=mcast_mask,
            group=prims.CTAGroup.CTA_2,
        )
    if cutlass.const_expr(manage_mbarrier):
        if cta_rank == ctm.Int32(0):
            while not prims.mbarrier_try_wait_parity(q_tma_mbar, q_tma_phase, time_limit=10000000):
                pass
        prims.fence_proxy(kind=prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)


@cute.jit
def _tma_gather4_cluster(
    smem_dst,
    tma_desc,
    col_coord: ctm.Int32,
    row0: ctm.Int32,
    row1: ctm.Int32,
    row2: ctm.Int32,
    row3: ctm.Int32,
    barrier,
    multicast_mask,
) -> None:
    """Gather four KSF atoms and multicast the compact image to both CTAs."""
    smem_ptr = smem_dst.data_ptr()
    tma_ptr = tma_desc.data_ptr() if hasattr(tma_desc, "data_ptr") else tma_desc
    leader_barrier = prims.cvta_to(
        _mapa_shared_cluster(barrier, ctm.Int32(0)),
        prims.CvtaSpace.SHARED,
    )
    barrier_ptr = leader_barrier.data_ptr()
    multicast_mask_u16 = ctm.Uint16(multicast_mask)
    cute_inline_ptx(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
        " [{$r0}], [{$r1}, {{$r2}, {$r3}, {$r4}, {$r5}, {$r6}}], "
        "[{$r7}], {$r8};",
        read_only_args=[
            smem_ptr,
            tma_ptr,
            col_coord,
            row0,
            row1,
            row2,
            row3,
            barrier_ptr,
            multicast_mask_u16,
        ],
    )


@cute.jit
def _load_qk_konly_kblock_stage(
    mPageTable_pl: cute.Tensor,
    tma_k_page_ptr,
    tma_k_tail_page_ptr,
    tma_ksf_ptr,
    tma_ksf_stage_ptr,
    tma_ksf_tail_stage_ptr,
    qk_tma_mbar,
    sB_slot,
    sSFB_slot,
    tidx: ctm.Int32,
    bidz: ctm.Int32,
    cta_rank: ctm.Int32,
    physical_sfb_blocks_per_l: ctm.Int32,
    page_begin: ctm.Int32,
    page_count: ctm.Int32,
    tile_physical_page0: ctm.Int32,
    tile_physical_page1: ctm.Int32,
    kv_tile_idx: ctm.Int32,
    qk_kblock_idx: ctm.Constexpr,
    score_slot_idx: ctm.Constexpr,
    stage: ctm.Constexpr,
    page_size: ctm.Constexpr,
    q_tma_phase: ctm.Constexpr,
    issue_ksf: ctm.Constexpr = True,
    manage_mbarrier: ctm.Constexpr = True,
    use_consecutive_page_pair: ctm.Constexpr = False,
    use_ksf_gather4: ctm.Constexpr = False,
) -> None:
    del (
        tidx,
        physical_sfb_blocks_per_l,
        page_begin,
        page_count,
        score_slot_idx,
        q_tma_phase,
        manage_mbarrier,
    )
    sB_stage = sB_slot.subview(ctm.Int32(stage * QK_B_STAGE_BYTES // 4))
    sSFB_stage = sSFB_slot.subview(ctm.Int32(stage * QK_SFB_STAGE_BYTES))
    if cutlass.const_expr(qk_kblock_idx >= SMEM_P4_QK_TAIL_STAGE_KBLOCK_START):
        tma_k_stage_ptr = tma_k_tail_page_ptr
        tma_ksf_merged_stage_ptr = tma_ksf_tail_stage_ptr
        qk_k_stage_dim: ctm.Constexpr = QK_K_TAIL_DATA_STAGE_K_DIM
        qk_k_stage_sf_chunks: ctm.Constexpr = SMEM_P4_QK_K_TAIL_DATA_STAGE_SF_CHUNKS
    else:
        tma_k_stage_ptr = tma_k_page_ptr
        tma_ksf_merged_stage_ptr = tma_ksf_stage_ptr
        qk_k_stage_dim: ctm.Constexpr = QK_DATA_STAGE_K_DIM
        qk_k_stage_sf_chunks: ctm.Constexpr = SMEM_P4_QK_DATA_STAGE_SF_CHUNKS
    mcast_mask = cutlass.Int16(1) << cutlass.Int16(cta_rank)
    tile_page_idx = kv_tile_idx * ctm.Int32(SMEM_P4_PAGES_PER_KV_TILE)
    physical_page = tile_physical_page0
    if cta_rank == ctm.Int32(1):
        physical_page = tile_physical_page1
    physical_n_base = ctm.Int32(0)
    for page_chunk in ctm.range_constexpr(SMEM_P4_QK_TMA_CHUNKS_PER_CTA):
        sB_page_stage = sB_stage.subview(
            ctm.Int32(page_chunk * SMEM_P4_QK_TMA_N_PER_CHUNK * qk_k_stage_dim // 8)
        )
        physical_n = physical_n_base + ctm.Int32(page_chunk * SMEM_P4_QK_TMA_N_PER_CHUNK)
        k_tma_coord = (
            ctm.Int32(qk_kblock_idx * MMA_KBLOCK_DIM),
            ctm.Int32(page_chunk * SMEM_P4_QK_TMA_N_PER_CHUNK),
            physical_page,
        )
        prims.cp_async_bulk_tensor_shared_cluster_global(
            sB_page_stage,
            tma_k_stage_ptr,
            k_tma_coord,
            _as_tma_completion_mbar(qk_tma_mbar),
            [],
            multicast_mask=mcast_mask,
            group=prims.CTAGroup.CTA_2,
        )
    if cutlass.const_expr(use_consecutive_page_pair and issue_ksf):
        # ``stride_order=(0, 2, 1, 3)`` makes the TMA coordinate order
        # (atom, physical page, scale chunk, batch).  The promise guarantees
        # that page1 is exactly page0 + 1 for every active tile.
        prims.cp_async_bulk_tensor_shared_cluster_global(
            sSFB_stage,
            tma_ksf_merged_stage_ptr,
            (
                ctm.Int32(0),
                tile_physical_page0,
                ctm.Int32(qk_kblock_idx * SMEM_P4_QK_MMA_SCALE_CHUNKS_PER_KBLOCK),
                ctm.Int32(0),
            ),
            _as_tma_completion_mbar(qk_tma_mbar),
            [],
            multicast_mask=cutlass.Int16(3),
            group=prims.CTAGroup.CTA_2,
        )
    elif cutlass.const_expr(use_ksf_gather4 and issue_ksf):
        # A gather4 transaction shares one column coordinate, so flatten each
        # (physical page, scale chunk) atom into a row. Pairing two adjacent
        # scale chunks produces the exact compact image expected by WARPX2:
        # [scale0/page0, scale0/page1, scale1/page0, scale1/page1].
        scale_chunk_base = ctm.Int32(qk_kblock_idx * SMEM_P4_QK_MMA_SCALE_CHUNKS_PER_KBLOCK)
        for scale_pair_idx in ctm.range_constexpr(qk_k_stage_sf_chunks // 2):
            scale_chunk0 = scale_chunk_base + ctm.Int32(scale_pair_idx * 2)
            scale_chunk1 = scale_chunk0 + ctm.Int32(1)
            row_page0_base = tile_physical_page0 * ctm.Int32(TRTLLM_K_SF_GROUPS // 4)
            row_page1_base = tile_physical_page1 * ctm.Int32(TRTLLM_K_SF_GROUPS // 4)
            _tma_gather4_cluster(
                sSFB_stage.subview(ctm.Int32(scale_pair_idx * 4 * 512)),
                tma_ksf_merged_stage_ptr,
                ctm.Int32(0),
                row_page0_base + scale_chunk0,
                row_page1_base + scale_chunk0,
                row_page0_base + scale_chunk1,
                row_page1_base + scale_chunk1,
                qk_tma_mbar,
                cutlass.Int16(3),
            )
    elif cutlass.const_expr(issue_ksf):
        # Match the scale-major KSF image consumed by the WARPX2 S2T path:
        # [scale chunk][logical page / N128 band][512-byte atom].  The prior
        # page-major merged transaction left only one compact-ring residue
        # numerically valid for nonzero K payloads.
        for scale_chunk_idx in ctm.range_constexpr(qk_k_stage_sf_chunks):
            for logical_page_idx in ctm.range_constexpr(SMEM_P4_PAGES_PER_KV_TILE):
                physical_sf_page = tile_physical_page0
                if cutlass.const_expr(logical_page_idx == 1):
                    physical_sf_page = tile_physical_page1
                sSFB_atom = sSFB_stage.subview(
                    ctm.Int32(
                        (scale_chunk_idx * SMEM_P4_PAGES_PER_KV_TILE + logical_page_idx) * 512
                    )
                )
                prims.cp_async_bulk_tensor_shared_cluster_global(
                    sSFB_atom,
                    tma_ksf_ptr,
                    (
                        ctm.Int32(0),
                        ctm.Int32(
                            qk_kblock_idx * SMEM_P4_QK_MMA_SCALE_CHUNKS_PER_KBLOCK + scale_chunk_idx
                        ),
                        physical_sf_page,
                        ctm.Int32(0),
                    ),
                    _as_tma_completion_mbar(qk_tma_mbar),
                    [],
                    multicast_mask=cutlass.Int16(3),
                    group=prims.CTAGroup.CTA_2,
                )


@cute.jit
def _issue_runtime_t336_qk_stage(
    mPageTable_pl: cute.Tensor,
    tma_k_page_ptr,
    tma_k_tail_page_ptr,
    tma_ksf_ptr,
    tma_ksf_stage_ptr,
    tma_ksf_tail_stage_ptr,
    qk_stage_mbar,
    sB_slot,
    sSFB_slot,
    tidx: ctm.Int32,
    bidz: ctm.Int32,
    physical_sfb_blocks_per_l: ctm.Int32,
    page_begin: ctm.Int32,
    page_count: ctm.Int32,
    physical_page0: ctm.Int32,
    physical_page1: ctm.Int32,
    kv_tile_idx: ctm.Int32,
    page_size: ctm.Constexpr,
    qk_stage_idx: ctm.Constexpr,
    issue_rank0: ctm.Constexpr = True,
    issue_rank1: ctm.Constexpr = True,
    rank1_first: ctm.Constexpr = False,
    issue_ksf: ctm.Constexpr = True,
    ksf_owner_rank1: ctm.Constexpr = False,
    use_consecutive_page_pair: ctm.Constexpr = False,
    use_ksf_gather4: ctm.Constexpr = False,
) -> None:
    qk_stage_kblock_idx: ctm.Constexpr = qk_stage_idx * SMEM_P4_QK_DATA_STAGE_KBLOCKS
    if cutlass.const_expr(rank1_first and issue_rank1):
        _load_qk_konly_kblock_stage(
            mPageTable_pl,
            tma_k_page_ptr,
            tma_k_tail_page_ptr,
            tma_ksf_ptr,
            tma_ksf_stage_ptr,
            tma_ksf_tail_stage_ptr,
            qk_stage_mbar,
            sB_slot,
            sSFB_slot,
            tidx,
            bidz,
            ctm.Int32(1),
            physical_sfb_blocks_per_l,
            page_begin,
            page_count,
            physical_page0,
            physical_page1,
            kv_tile_idx,
            qk_stage_kblock_idx,
            0,
            qk_stage_idx,
            page_size,
            0,
            issue_ksf=issue_ksf and ksf_owner_rank1,
            manage_mbarrier=False,
            use_consecutive_page_pair=use_consecutive_page_pair,
            use_ksf_gather4=use_ksf_gather4,
        )
    if cutlass.const_expr(issue_rank0):
        _load_qk_konly_kblock_stage(
            mPageTable_pl,
            tma_k_page_ptr,
            tma_k_tail_page_ptr,
            tma_ksf_ptr,
            tma_ksf_stage_ptr,
            tma_ksf_tail_stage_ptr,
            qk_stage_mbar,
            sB_slot,
            sSFB_slot,
            tidx,
            bidz,
            ctm.Int32(0),
            physical_sfb_blocks_per_l,
            page_begin,
            page_count,
            physical_page0,
            physical_page1,
            kv_tile_idx,
            qk_stage_kblock_idx,
            0,
            qk_stage_idx,
            page_size,
            0,
            issue_ksf=issue_ksf and not ksf_owner_rank1,
            manage_mbarrier=False,
            use_consecutive_page_pair=use_consecutive_page_pair,
            use_ksf_gather4=use_ksf_gather4,
        )
    if cutlass.const_expr(not rank1_first and issue_rank1):
        _load_qk_konly_kblock_stage(
            mPageTable_pl,
            tma_k_page_ptr,
            tma_k_tail_page_ptr,
            tma_ksf_ptr,
            tma_ksf_stage_ptr,
            tma_ksf_tail_stage_ptr,
            qk_stage_mbar,
            sB_slot,
            sSFB_slot,
            tidx,
            bidz,
            ctm.Int32(1),
            physical_sfb_blocks_per_l,
            page_begin,
            page_count,
            physical_page0,
            physical_page1,
            kv_tile_idx,
            qk_stage_kblock_idx,
            0,
            qk_stage_idx,
            page_size,
            0,
            issue_ksf=issue_ksf and ksf_owner_rank1,
            manage_mbarrier=False,
            use_consecutive_page_pair=use_consecutive_page_pair,
            use_ksf_gather4=use_ksf_gather4,
        )


@cute.jit
def _load_runtime_t336_qk_tile(
    mPageTable_pl: cute.Tensor,
    tma_k_page_ptr,
    tma_k_tail_page_ptr,
    tma_ksf_ptr,
    tma_ksf_stage_ptr,
    tma_ksf_tail_stage_ptr,
    qk_tma_mbar,
    qk_bulk_ready_mbar_ptr,
    pair_overlap_sync_mbars,
    sB,
    sSFB,
    tidx: ctm.Int32,
    bidz: ctm.Int32,
    physical_sfb_blocks_per_l: ctm.Int32,
    page_begin: ctm.Int32,
    page_count: ctm.Int32,
    physical_page0: ctm.Int32,
    physical_page1: ctm.Int32,
    kv_tile_idx: ctm.Int32,
    qk_slot_idx: ctm.Int32,
    page_size: ctm.Constexpr,
    initial_tile: ctm.Constexpr = False,
    split_prefix_owner: ctm.Constexpr = False,
    qk15_final: ctm.Constexpr = False,
    use_consecutive_page_pair: ctm.Constexpr = False,
    use_ksf_gather4: ctm.Constexpr = False,
) -> None:
    qk_bulk_tma_mbar = qk_tma_mbar
    if cutlass.const_expr(not initial_tile):
        qk_bulk_tma_mbar = ctm.Array(qk_bulk_ready_mbar_ptr + qk_slot_idx, shape=1)
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(
                qk_bulk_tma_mbar,
                (1 if qk15_final else SMEM_P4_QK_FULL_DATA_STAGES)
                * SMEM_P4_QK_KONLY_STAGE_BYTES
                * CLUSTER_SHAPE_MNK[0],
            )
        if cutlass.const_expr(split_prefix_owner):
            prims.barrier(
                barrier_id=SMEM_P4_QK1_SPLIT_ARM_BAR_ID,
                number_of_threads=SMEM_P4_QK1_SPLIT_ARM_BAR_THREADS,
            )
    if prims.elect_sync():
        sB_slot = sB.subview(qk_slot_idx * ctm.Int32(SMEM_P4_K_COMPACT_SLOT_BYTES // 4))
        sSFB_slot = sSFB.subview(qk_slot_idx * ctm.Int32(SMEM_P4_KSF_COMPACT_SLOT_BYTES))
        if cutlass.const_expr(initial_tile):
            _issue_runtime_t336_qk_stage(
                mPageTable_pl,
                tma_k_page_ptr,
                tma_k_tail_page_ptr,
                tma_ksf_ptr,
                tma_ksf_stage_ptr,
                tma_ksf_tail_stage_ptr,
                qk_tma_mbar,
                sB_slot,
                sSFB_slot,
                tidx,
                bidz,
                physical_sfb_blocks_per_l,
                page_begin,
                page_count,
                physical_page0,
                physical_page1,
                kv_tile_idx,
                page_size,
                SMEM_P4_QK_FULL_DATA_STAGES,
                issue_rank0=False,
                issue_rank1=True,
                use_consecutive_page_pair=use_consecutive_page_pair,
                use_ksf_gather4=use_ksf_gather4,
            )
        _issue_runtime_t336_qk_stage(
            mPageTable_pl,
            tma_k_page_ptr,
            tma_k_tail_page_ptr,
            tma_ksf_ptr,
            tma_ksf_stage_ptr,
            tma_ksf_tail_stage_ptr,
            qk_tma_mbar if initial_tile else qk_bulk_tma_mbar,
            sB_slot,
            sSFB_slot,
            tidx,
            bidz,
            physical_sfb_blocks_per_l,
            page_begin,
            page_count,
            physical_page0,
            physical_page1,
            kv_tile_idx,
            page_size,
            0,
            issue_rank1=not split_prefix_owner,
            use_consecutive_page_pair=use_consecutive_page_pair,
            use_ksf_gather4=use_ksf_gather4,
        )
        qk_stage1_mbar = qk_tma_mbar if initial_tile else qk_bulk_tma_mbar
        if cutlass.const_expr(qk15_final):
            qk_stage1_mbar = pair_overlap_sync_mbars.subview(SMEM_P4_QK15_STAGE1_MBAR_OFFSET)
            prims.mbarrier_arrive_expect_tx(
                qk_stage1_mbar, SMEM_P4_QK_KONLY_STAGE_BYTES * CLUSTER_SHAPE_MNK[0]
            )
        _issue_runtime_t336_qk_stage(
            mPageTable_pl,
            tma_k_page_ptr,
            tma_k_tail_page_ptr,
            tma_ksf_ptr,
            tma_ksf_stage_ptr,
            tma_ksf_tail_stage_ptr,
            qk_stage1_mbar,
            sB_slot,
            sSFB_slot,
            tidx,
            bidz,
            physical_sfb_blocks_per_l,
            page_begin,
            page_count,
            physical_page0,
            physical_page1,
            kv_tile_idx,
            page_size,
            1,
            issue_rank1=not split_prefix_owner,
            rank1_first=qk15_final,
            use_consecutive_page_pair=use_consecutive_page_pair,
            use_ksf_gather4=use_ksf_gather4,
        )
        _issue_runtime_t336_qk_stage(
            mPageTable_pl,
            tma_k_page_ptr,
            tma_k_tail_page_ptr,
            tma_ksf_ptr,
            tma_ksf_stage_ptr,
            tma_ksf_tail_stage_ptr,
            qk_tma_mbar,
            sB_slot,
            sSFB_slot,
            tidx,
            bidz,
            physical_sfb_blocks_per_l,
            page_begin,
            page_count,
            physical_page0,
            physical_page1,
            kv_tile_idx,
            page_size,
            SMEM_P4_QK_FULL_DATA_STAGES,
            issue_rank0=True,
            issue_rank1=not initial_tile,
            use_consecutive_page_pair=use_consecutive_page_pair,
            use_ksf_gather4=use_ksf_gather4,
        )


@cute.jit
def _load_runtime_t336_qk_tile_dual_steady(
    mPageTable_pl: cute.Tensor,
    tma_k_page_ptr,
    tma_k_tail_page_ptr,
    tma_ksf_ptr,
    tma_ksf_stage_ptr,
    tma_ksf_tail_stage_ptr,
    qk_tma_mbar,
    qk_bulk_ready_mbar_ptr,
    qk_dual_arm_mbars,
    sB,
    sSFB,
    tidx: ctm.Int32,
    bidz: ctm.Int32,
    cta_rank: ctm.Int32,
    physical_sfb_blocks_per_l: ctm.Int32,
    page_begin: ctm.Int32,
    page_count: ctm.Int32,
    physical_page0: ctm.Int32,
    physical_page1: ctm.Int32,
    kv_tile_idx: ctm.Int32,
    qk_slot_idx: ctm.Int32,
    page_size: ctm.Constexpr,
    use_consecutive_page_pair: ctm.Constexpr = False,
    use_ksf_gather4: ctm.Constexpr = False,
) -> None:
    qk_bulk_local_mbar = ctm.Array(qk_bulk_ready_mbar_ptr + qk_slot_idx, shape=1)
    qk_bulk_leader_mbar = _mapa_shared_cluster(qk_bulk_local_mbar, ctm.Int32(0))
    qk_tail_leader_mbar = _mapa_shared_cluster(qk_tma_mbar, ctm.Int32(0))
    arm_mbar = qk_dual_arm_mbars.subview(qk_slot_idx)
    arm_phase = (
        (kv_tile_idx - ctm.Int32(3))
        // ctm.Int32(SMEM_P4_QK_SMEM_PIPELINE_SLOTS)
        % ctm.Int32(SMEM_P4_MBAR_PARITY_PHASES)
    )
    is_leader_cta = cta_rank == ctm.Int32(0)
    if is_leader_cta:
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(
                qk_bulk_leader_mbar,
                SMEM_P4_QK_FULL_DATA_STAGES * SMEM_P4_QK_KONLY_STAGE_BYTES * CLUSTER_SHAPE_MNK[0],
            )
            peer_arm_mbar = _mapa_shared_cluster(arm_mbar, ctm.Int32(1))
            _mbarrier_arrive_shared_cluster(peer_arm_mbar)
    else:
        while not prims.mbarrier_try_wait_parity(arm_mbar, arm_phase, time_limit=10000000):
            pass
    if prims.elect_sync():
        sB_slot = sB.subview(qk_slot_idx * ctm.Int32(SMEM_P4_K_COMPACT_SLOT_BYTES // 4))
        sSFB_slot = sSFB.subview(qk_slot_idx * ctm.Int32(SMEM_P4_KSF_COMPACT_SLOT_BYTES))
        if is_leader_cta:
            _issue_runtime_t336_qk_stage(
                mPageTable_pl,
                tma_k_page_ptr,
                tma_k_tail_page_ptr,
                tma_ksf_ptr,
                tma_ksf_stage_ptr,
                tma_ksf_tail_stage_ptr,
                qk_bulk_leader_mbar,
                sB_slot,
                sSFB_slot,
                tidx,
                bidz,
                physical_sfb_blocks_per_l,
                page_begin,
                page_count,
                physical_page0,
                physical_page1,
                kv_tile_idx,
                page_size,
                0,
                issue_rank0=True,
                issue_rank1=False,
                issue_ksf=True,
                use_consecutive_page_pair=use_consecutive_page_pair,
                use_ksf_gather4=use_ksf_gather4,
            )
            _issue_runtime_t336_qk_stage(
                mPageTable_pl,
                tma_k_page_ptr,
                tma_k_tail_page_ptr,
                tma_ksf_ptr,
                tma_ksf_stage_ptr,
                tma_ksf_tail_stage_ptr,
                qk_bulk_leader_mbar,
                sB_slot,
                sSFB_slot,
                tidx,
                bidz,
                physical_sfb_blocks_per_l,
                page_begin,
                page_count,
                physical_page0,
                physical_page1,
                kv_tile_idx,
                page_size,
                1,
                issue_rank0=True,
                issue_rank1=False,
                issue_ksf=False,
                use_consecutive_page_pair=use_consecutive_page_pair,
                use_ksf_gather4=use_ksf_gather4,
            )
            _issue_runtime_t336_qk_stage(
                mPageTable_pl,
                tma_k_page_ptr,
                tma_k_tail_page_ptr,
                tma_ksf_ptr,
                tma_ksf_stage_ptr,
                tma_ksf_tail_stage_ptr,
                qk_tail_leader_mbar,
                sB_slot,
                sSFB_slot,
                tidx,
                bidz,
                physical_sfb_blocks_per_l,
                page_begin,
                page_count,
                physical_page0,
                physical_page1,
                kv_tile_idx,
                page_size,
                SMEM_P4_QK_FULL_DATA_STAGES,
                issue_rank0=True,
                issue_rank1=False,
                issue_ksf=True,
                use_consecutive_page_pair=use_consecutive_page_pair,
                use_ksf_gather4=use_ksf_gather4,
            )
        else:
            _issue_runtime_t336_qk_stage(
                mPageTable_pl,
                tma_k_page_ptr,
                tma_k_tail_page_ptr,
                tma_ksf_ptr,
                tma_ksf_stage_ptr,
                tma_ksf_tail_stage_ptr,
                qk_bulk_leader_mbar,
                sB_slot,
                sSFB_slot,
                tidx,
                bidz,
                physical_sfb_blocks_per_l,
                page_begin,
                page_count,
                physical_page0,
                physical_page1,
                kv_tile_idx,
                page_size,
                0,
                issue_rank0=False,
                issue_rank1=True,
                issue_ksf=False,
                use_consecutive_page_pair=use_consecutive_page_pair,
                use_ksf_gather4=use_ksf_gather4,
            )
            _issue_runtime_t336_qk_stage(
                mPageTable_pl,
                tma_k_page_ptr,
                tma_k_tail_page_ptr,
                tma_ksf_ptr,
                tma_ksf_stage_ptr,
                tma_ksf_tail_stage_ptr,
                qk_bulk_leader_mbar,
                sB_slot,
                sSFB_slot,
                tidx,
                bidz,
                physical_sfb_blocks_per_l,
                page_begin,
                page_count,
                physical_page0,
                physical_page1,
                kv_tile_idx,
                page_size,
                1,
                issue_rank0=False,
                issue_rank1=True,
                issue_ksf=True,
                ksf_owner_rank1=True,
                use_consecutive_page_pair=use_consecutive_page_pair,
                use_ksf_gather4=use_ksf_gather4,
            )
            _issue_runtime_t336_qk_stage(
                mPageTable_pl,
                tma_k_page_ptr,
                tma_k_tail_page_ptr,
                tma_ksf_ptr,
                tma_ksf_stage_ptr,
                tma_ksf_tail_stage_ptr,
                qk_tail_leader_mbar,
                sB_slot,
                sSFB_slot,
                tidx,
                bidz,
                physical_sfb_blocks_per_l,
                page_begin,
                page_count,
                physical_page0,
                physical_page1,
                kv_tile_idx,
                page_size,
                SMEM_P4_QK_FULL_DATA_STAGES,
                issue_rank0=False,
                issue_rank1=True,
                issue_ksf=False,
                use_consecutive_page_pair=use_consecutive_page_pair,
                use_ksf_gather4=use_ksf_gather4,
            )


@cute.jit
def _issue_runtime_t336_qk_prefix_rank1(
    mPageTable_pl: cute.Tensor,
    tma_k_page_ptr,
    tma_k_tail_page_ptr,
    tma_ksf_ptr,
    tma_ksf_stage_ptr,
    tma_ksf_tail_stage_ptr,
    qk_bulk_ready_mbar_ptr,
    sB,
    sSFB,
    tidx: ctm.Int32,
    bidz: ctm.Int32,
    physical_sfb_blocks_per_l: ctm.Int32,
    page_begin: ctm.Int32,
    page_count: ctm.Int32,
    physical_page0: ctm.Int32,
    physical_page1: ctm.Int32,
    kv_tile_idx: ctm.Int32,
    qk_slot_idx: ctm.Int32,
    page_size: ctm.Constexpr,
) -> None:
    prims.barrier(
        barrier_id=SMEM_P4_QK1_SPLIT_ARM_BAR_ID,
        number_of_threads=SMEM_P4_QK1_SPLIT_ARM_BAR_THREADS,
    )
    qk_bulk_tma_mbar = ctm.Array(qk_bulk_ready_mbar_ptr + qk_slot_idx, shape=1)
    if prims.elect_sync():
        sB_slot = sB.subview(qk_slot_idx * ctm.Int32(SMEM_P4_K_COMPACT_SLOT_BYTES // 4))
        sSFB_slot = sSFB.subview(qk_slot_idx * ctm.Int32(SMEM_P4_KSF_COMPACT_SLOT_BYTES))
        for qk_stage_idx in ctm.range_constexpr(SMEM_P4_QK_FULL_DATA_STAGES):
            _issue_runtime_t336_qk_stage(
                mPageTable_pl,
                tma_k_page_ptr,
                tma_k_tail_page_ptr,
                tma_ksf_ptr,
                tma_ksf_stage_ptr,
                tma_ksf_tail_stage_ptr,
                qk_bulk_tma_mbar,
                sB_slot,
                sSFB_slot,
                tidx,
                bidz,
                physical_sfb_blocks_per_l,
                page_begin,
                page_count,
                physical_page0,
                physical_page1,
                kv_tile_idx,
                page_size,
                qk_stage_idx,
                issue_rank0=False,
                issue_rank1=True,
            )


@cute.jit
def _issue_raw_k_pages_to_v_staging(
    tma_k_v_raw_ptr,
    raw_v_tma_mbar,
    sRawV,
    sPageIdPlan,
    cta_rank: ctm.Int32,
    kv_tile_idx: ctm.Int32,
    direct_physical_page0: ctm.Int32,
    direct_physical_page1: ctm.Int32,
    use_smem_page_plan: ctm.Constexpr,
) -> None:
    """Stage the V-bearing K prefix in a transpose-friendly layout."""
    physical_page0 = direct_physical_page0
    physical_page1 = direct_physical_page1
    if cutlass.const_expr(use_smem_page_plan):
        physical_page0, physical_page1 = _load_staged_page_native_tile_pair(
            sPageIdPlan, kv_tile_idx
        )
    raw_packed_dims_per_n_tile: ctm.Constexpr = SMEM_P4_V_N_PER_CTA // 2
    raw_n_tile_bytes: ctm.Constexpr = TRTLLM_PAGE_SIZE * raw_packed_dims_per_n_tile
    raw_page_bytes: ctm.Constexpr = raw_n_tile_bytes * SMEM_P4_N_OUT_TILES
    if prims.elect_sync():
        prims.mbarrier_arrive_expect_tx(raw_v_tma_mbar, SMEM_P4_RAW_V_STAGE_BYTES)
        for logical_page_idx in ctm.range_constexpr(SMEM_P4_PAGES_PER_KV_TILE):
            physical_page = physical_page0
            if cutlass.const_expr(logical_page_idx == 1):
                physical_page = physical_page1
            for n_tile_idx in ctm.range_constexpr(SMEM_P4_N_OUT_TILES):
                packed_dim_begin = cta_rank * ctm.Int32(raw_packed_dims_per_n_tile) + ctm.Int32(
                    n_tile_idx * SMEM_P4_BMM2_N // 2
                )
                raw_stage_offset = ctm.Int32(
                    logical_page_idx * raw_page_bytes + n_tile_idx * raw_n_tile_bytes
                )
                prims.cp_async_bulk_tensor_shared_cta_global(
                    sRawV.subview(raw_stage_offset),
                    tma_k_v_raw_ptr,
                    (packed_dim_begin, ctm.Int32(0), physical_page),
                    raw_v_tma_mbar,
                )


@cute.jit
def _pack_raw_v_word_to_pv_word(raw_word: ctm.Int32) -> ctm.Int32:
    """Pack two token pairs from raw token-major E2M1 bytes."""
    low_nibbles = (raw_word & ctm.Int32(0x000F000F)) | (
        (raw_word >> ctm.Int32(4)) & ctm.Int32(0x00F000F0)
    )
    high_nibbles = ((raw_word >> ctm.Int32(4)) & ctm.Int32(0x000F000F)) | (
        (raw_word >> ctm.Int32(8)) & ctm.Int32(0x00F000F0)
    )
    return low_nibbles | (high_nibbles << ctm.Int32(8))


@cute.jit
def _transpose_raw_k_staging_to_v_stage(
    raw_v_tma_mbar,
    sRawV,
    sV,
    tidx: ctm.Int32,
    warp_idx: ctm.Int32,
    kv_tile_idx: ctm.Int32,
    v_stage: ctm.Int32,
) -> None:
    """Repack one staged K prefix into PV layout with all eight ALU warps."""
    raw_packed_dims_per_n_tile: ctm.Constexpr = SMEM_P4_V_N_PER_CTA // 2
    raw_n_tile_bytes: ctm.Constexpr = TRTLLM_PAGE_SIZE * raw_packed_dims_per_n_tile
    raw_page_bytes: ctm.Constexpr = raw_n_tile_bytes * SMEM_P4_N_OUT_TILES
    raw_phase = (
        kv_tile_idx // ctm.Int32(SMEM_P4_V_PIPELINE_STAGES) % ctm.Int32(SMEM_P4_MBAR_PARITY_PHASES)
    )
    while not prims.mbarrier_try_wait_parity(raw_v_tma_mbar, raw_phase, time_limit=10000000):
        pass
    prims.fence_proxy(kind=prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)

    raw_ptr = sRawV.data_ptr()
    v_word_ptr = sV.data_ptr()
    token_pairs: ctm.Constexpr = TRTLLM_PAGE_SIZE // 2
    token_blocks_per_page: ctm.Constexpr = TRTLLM_PAGE_SIZE // 32
    packed_dim_blocks_per_n_tile: ctm.Constexpr = raw_packed_dims_per_n_tile // 16
    warp_tiles_per_n_tile: ctm.Constexpr = packed_dim_blocks_per_n_tile * token_blocks_per_page
    warp_tiles_per_page: ctm.Constexpr = warp_tiles_per_n_tile * SMEM_P4_N_OUT_TILES
    total_warp_tiles: ctm.Constexpr = warp_tiles_per_page * SMEM_P4_PAGES_PER_KV_TILE
    warp_tile_iters: ctm.Constexpr = total_warp_tiles // SMEM_P4_TOTAL_PRODUCER_WARPS

    stsm_atom = cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix16x8x8bOp(transpose=True, num_matrices=4),
        cutlass.Uint8,
    )
    ldsm_atom = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix16x8x8bOp(transpose=True, num_matrices=4),
        cutlass.Uint8,
    )
    lane_idx = tidx & ctm.Int32(31)
    lane_source_row = (
        (lane_idx & ctm.Int32(1))
        + ((lane_idx >> ctm.Int32(1)) & ctm.Int32(1)) * ctm.Int32(8)
        + ((lane_idx >> ctm.Int32(2)) & ctm.Int32(3)) * ctm.Int32(2)
        + ((lane_idx >> ctm.Int32(4)) & ctm.Int32(1)) * ctm.Int32(16)
    )
    lane_low = lane_idx & ctm.Int32(15)
    source_lane = ((lane_low & ctm.Int32(3)) << ctm.Int32(2)) | (
        (lane_low >> ctm.Int32(2)) & ctm.Int32(3)
    )
    lane_layout = cute.make_layout((16, 1), stride=(1, 0))
    stage_base = v_stage * ctm.Int32(SMEM_P4_V_DATA_STAGE_BYTES)
    for warp_tile_iter in cutlass.range(0, warp_tile_iters, 1, unroll=1):
        warp_tile_idx = warp_idx + warp_tile_iter * ctm.Int32(SMEM_P4_TOTAL_PRODUCER_WARPS)
        logical_page_idx = warp_tile_idx // ctm.Int32(warp_tiles_per_page)
        page_tile_idx = warp_tile_idx - logical_page_idx * ctm.Int32(warp_tiles_per_page)
        n_tile_idx = page_tile_idx // ctm.Int32(warp_tiles_per_n_tile)
        n_tile_warp_idx = page_tile_idx - n_tile_idx * ctm.Int32(warp_tiles_per_n_tile)
        packed_dim_block = n_tile_warp_idx // ctm.Int32(token_blocks_per_page)
        token_block = n_tile_warp_idx - packed_dim_block * ctm.Int32(token_blocks_per_page)

        raw_tile_offset = (
            logical_page_idx * ctm.Int32(raw_page_bytes)
            + n_tile_idx * ctm.Int32(raw_n_tile_bytes)
            + token_block * ctm.Int32(32 * raw_packed_dims_per_n_tile)
            + packed_dim_block * ctm.Int32(16)
        )
        pv_tile_offset = (
            stage_base
            + logical_page_idx * ctm.Int32(B_STAGE_BYTES)
            + n_tile_idx * ctm.Int32(SMEM_P4_V_N_PER_CTA * token_pairs)
            + packed_dim_block * ctm.Int32(32 * token_pairs)
            + token_block * ctm.Int32(16)
        )
        raw_lane_offset = raw_tile_offset + lane_source_row * ctm.Int32(64)
        pv_lane_offset = pv_tile_offset + lane_idx * ctm.Int32(token_pairs)
        pv_lane_offset = pv_lane_offset ^ (
            ((pv_lane_offset >> ctm.Int32(7)) & ctm.Int32(3)) << ctm.Int32(4)
        )
        raw_lane_ptr = cute.make_ptr(
            cutlass.Uint8,
            cute.assume(raw_ptr.toint() + raw_lane_offset, divby=16),
            cute.AddressSpace.smem,
            assumed_align=16,
        )
        pv_lane_ptr = cute.make_ptr(
            cutlass.Uint8,
            cute.assume(v_word_ptr.toint() + pv_lane_offset, divby=16),
            cute.AddressSpace.smem,
            assumed_align=16,
        )
        raw_lane = cute.make_tensor(raw_lane_ptr, lane_layout)
        pv_lane = cute.make_tensor(pv_lane_ptr, lane_layout)
        raw_fragment = cute.make_rmem_tensor(lane_layout, cutlass.Uint8)
        cute.copy(ldsm_atom, raw_lane, raw_fragment)
        raw_words = raw_fragment.load().bitcast(ctm.Int32)
        pair_a0 = (raw_words[0] & ctm.Int32(0xFFFF)) | (
            (raw_words[2] & ctm.Int32(0xFFFF)) << ctm.Int32(16)
        )
        pair_a1 = ((raw_words[0] >> ctm.Int32(16)) & ctm.Int32(0xFFFF)) | (
            raw_words[2] & ctm.Int32(0xFFFF0000)
        )
        pair_b0 = (raw_words[1] & ctm.Int32(0xFFFF)) | (
            (raw_words[3] & ctm.Int32(0xFFFF)) << ctm.Int32(16)
        )
        pair_b1 = ((raw_words[1] >> ctm.Int32(16)) & ctm.Int32(0xFFFF)) | (
            raw_words[3] & ctm.Int32(0xFFFF0000)
        )
        source_lane_hi = source_lane + ctm.Int32(16)
        a0_lo = cute.arch.shuffle_sync(pair_a0, source_lane)
        a0_hi = cute.arch.shuffle_sync(pair_a0, source_lane_hi)
        a1_lo = cute.arch.shuffle_sync(pair_a1, source_lane)
        a1_hi = cute.arch.shuffle_sync(pair_a1, source_lane_hi)
        b0_lo = cute.arch.shuffle_sync(pair_b0, source_lane)
        b0_hi = cute.arch.shuffle_sync(pair_b0, source_lane_hi)
        b1_lo = cute.arch.shuffle_sync(pair_b1, source_lane)
        b1_hi = cute.arch.shuffle_sync(pair_b1, source_lane_hi)
        upper_lane_mask = ctm.Int32(0) - (lane_idx >> ctm.Int32(4))
        lower_lane_mask = ~upper_lane_mask
        first_pair0 = (a0_lo & lower_lane_mask) | (b0_lo & upper_lane_mask)
        second_pair0 = (a0_hi & lower_lane_mask) | (b0_hi & upper_lane_mask)
        first_pair1 = (a1_lo & lower_lane_mask) | (b1_lo & upper_lane_mask)
        second_pair1 = (a1_hi & lower_lane_mask) | (b1_hi & upper_lane_mask)
        packed_words = ctm.Vector.from_elements(
            (
                _pack_raw_v_word_to_pv_word(first_pair0),
                _pack_raw_v_word_to_pv_word(second_pair0),
                _pack_raw_v_word_to_pv_word(first_pair1),
                _pack_raw_v_word_to_pv_word(second_pair1),
            ),
            dtype=ctm.Int32,
        )
        packed_fragment = cute.TensorSSA.from_vector(
            packed_words.ir_value(), dtype=ctm.Int32, shape=(4,)
        ).bitcast(cutlass.Uint8)
        raw_fragment.store(packed_fragment)
        cute.copy(stsm_atom, raw_fragment, pv_lane)
    prims.fence_proxy(kind=prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
    prims.barrier(
        barrier_id=SMEM_P4_RAW_V_READY_BAR_ID,
        number_of_threads=SMEM_P4_RAW_V_READY_BAR_THREADS,
    )


@cute.jit
def _load_vsf_tile_stage_only(
    sPageIdPlan,
    tma_vsf_ptr,
    tma_vsf_stage_ptr,
    v_tma_mbar,
    sVSF,
    cta_rank: ctm.Int32,
    kv_tile_idx: ctm.Int32,
    direct_physical_page0: ctm.Int32,
    direct_physical_page1: ctm.Int32,
    v_page_offset: ctm.Int32,
    stage: ctm.Int32,
    use_smem_page_plan: ctm.Constexpr,
    use_consecutive_page_pair: ctm.Constexpr = False,
) -> None:
    sSFB_stage = sVSF.subview(SMEM_P4_V_SFB_TMA_STAGE_BYTES * stage)
    physical_page0 = direct_physical_page0
    physical_page1 = direct_physical_page1
    if cutlass.const_expr(use_smem_page_plan):
        physical_page0, physical_page1 = _load_staged_page_native_tile_pair(
            sPageIdPlan, kv_tile_idx
        )
    if cutlass.const_expr(use_consecutive_page_pair):
        physical_page1 = physical_page0 + ctm.Int32(1)
    if cta_rank == ctm.Int32(0):
        if prims.elect_sync():
            if cutlass.const_expr(use_consecutive_page_pair):
                prims.cp_async_bulk_tensor_shared_cluster_global(
                    sSFB_stage,
                    tma_vsf_ptr,
                    (
                        ctm.Int32(0),
                        ctm.Int32(0),
                        ctm.Int32(0),
                        physical_page0 + v_page_offset,
                    ),
                    _as_tma_completion_mbar(v_tma_mbar),
                    [],
                    multicast_mask=cutlass.Int16(3),
                    group=prims.CTAGroup.CTA_2,
                )
            else:
                for logical_page_idx in ctm.range_constexpr(SMEM_P4_PAGES_PER_KV_TILE):
                    physical_page = physical_page0
                    if cutlass.const_expr(logical_page_idx == 1):
                        physical_page = physical_page1
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        sSFB_stage.subview(ctm.Int32(logical_page_idx * SMEM_P4_VSF_PAGE_BYTES)),
                        tma_vsf_stage_ptr,
                        (
                            ctm.Int32(0),
                            ctm.Int32(0),
                            ctm.Int32(0),
                            physical_page + v_page_offset,
                        ),
                        _as_tma_completion_mbar(v_tma_mbar),
                        [],
                        multicast_mask=cutlass.Int16(3),
                        group=prims.CTAGroup.CTA_2,
                    )


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
    p_sum_pair = (ctm.Float32(0.0), ctm.Float32(0.0))
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
            p0_f32, p1_f32 = _unpack_f16x2_to_f32_pair(p_h2)
            if cutlass.const_expr(word_part == 0 and pair_idx == 0):
                p_sum_pair = (p0_f32, p1_f32)
            else:
                p_sum_pair = nvvm_add_packed_f32x2(p_sum_pair, (p0_f32, p1_f32))
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
    p_sum = p_sum_pair[0] + p_sum_pair[1]
    return (words[0], words[1], p_sum)


@cute.jit
def _issue_smem_p4_qk_ksf_stage_from_desc(
    desc_ksf_s2t,
    base_col_id: ctm.Int32,
    base_row_id: ctm.Int32,
    sfb_col_offset: ctm.Constexpr,
    qk_stage_idx: ctm.Constexpr,
    page_major_ksf: ctm.Constexpr = False,
) -> None:
    if cutlass.const_expr(SMEM_P4_QK_SFB_WARPX2_MODE == 2):
        sfb_s2t_shape, sfb_s2t_multicast = prims.S2TCopyMode.S2T_64x128b_WARPX2_02_13
    else:
        sfb_s2t_shape, sfb_s2t_multicast = prims.S2TCopyMode.S2T_64x128b_WARPX2_01_23
    sfb_tmem_addr_base = base_row_id << ctm.Int32(16) | base_col_id + ctm.Int32(sfb_col_offset)
    sfb_cols_per_scale_chunk: ctm.Constexpr = _compact_sfb_mma_cols_per_scale_chunk(
        SMEM_P4_CTA_GROUP_M, SMEM_P4_QK_SCORE_SLOT_N
    )
    stage_kblocks: ctm.Constexpr = (
        SMEM_P4_QK_DATA_STAGE_KBLOCKS
        if qk_stage_idx < SMEM_P4_QK_FULL_DATA_STAGES
        else QK_TAIL_DATA_STAGE_K_DIM // SMEM_P4_QK_MMA_KBLOCK_DIM
    )
    for local_kblock_idx in ctm.range_constexpr(stage_kblocks):
        global_kblock_idx: ctm.Constexpr = (
            qk_stage_idx * SMEM_P4_QK_DATA_STAGE_KBLOCKS + local_kblock_idx
        )
        for scale_chunk_idx in ctm.range_constexpr(SMEM_P4_QK_MMA_SCALE_CHUNKS_PER_KBLOCK):
            logical_local_scale_word_idx: ctm.Constexpr = (
                local_kblock_idx * SMEM_P4_QK_MMA_SCALE_CHUNKS_PER_KBLOCK + scale_chunk_idx
            )
            physical_local_scale_word_idx: ctm.Constexpr = logical_local_scale_word_idx
            global_scale_word_idx: ctm.Constexpr = (
                global_kblock_idx * SMEM_P4_QK_MMA_SCALE_CHUNKS_PER_KBLOCK + scale_chunk_idx
            )
            sfb_tmem_ptr = ctm.inttoptr(
                sfb_tmem_addr_base + ctm.Int32(global_scale_word_idx * sfb_cols_per_scale_chunk),
                6,
                ctm.Int32,
            )
            source_scale_word_stride: ctm.Constexpr = (
                1 if page_major_ksf else SMEM_P4_QK_SFB_N_CHUNKS
            )
            source_byte_offset: ctm.Constexpr = (
                physical_local_scale_word_idx * source_scale_word_stride * (512 // 16)
                + SMEM_P4_QK_COMPACT_SFB_S2T_BYTE_OFFSET
            )
            if prims.elect_sync():
                prims.tcgen05_cp(
                    sfb_s2t_shape,
                    sfb_tmem_ptr,
                    desc_ksf_s2t + ctm.Int64(source_byte_offset),
                    group=prims.CTAGroup.CTA_2,
                    multicast=sfb_s2t_multicast,
                )


@cute.jit
def _complete_smem_p4_qk_ksf_store() -> None:
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    cute.arch.fence_view_async_tmem_store()
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def _issue_smem_p4_qk_ksf_range_from_base(
    sKSF,
    base_col_id: ctm.Int32,
    base_row_id: ctm.Int32,
    sfb_col_offset: ctm.Constexpr,
    sfb_stage_base: ctm.Int32,
    qk_stage_start: ctm.Constexpr,
    qk_stage_count: ctm.Constexpr,
    sksf_is_compact_slot_base: ctm.Constexpr = False,
    page_major_ksf: ctm.Constexpr = False,
) -> None:
    desc_ksf_s2t_root = _make_smem_p4_qk_ksf_stage_desc(
        sKSF, sfb_stage_base, qk_stage_start, sksf_is_compact_slot_base, page_major_ksf
    )
    for qk_stage_idx in ctm.range_constexpr(qk_stage_start, qk_stage_start + qk_stage_count):
        desc_ksf_s2t = desc_ksf_s2t_root.advance_start_address(
            (qk_stage_idx - qk_stage_start) * QK_SFB_STAGE_BYTES
        )
        _issue_smem_p4_qk_ksf_stage_from_desc(
            desc_ksf_s2t,
            base_col_id,
            base_row_id,
            sfb_col_offset,
            qk_stage_idx,
            page_major_ksf,
        )


@cute.jit
def _stage_smem_p4_qk_ksf_range_from_base(
    sKSF,
    base_col_id: ctm.Int32,
    base_row_id: ctm.Int32,
    sfb_col_offset: ctm.Constexpr,
    sfb_stage_base: ctm.Int32,
    qk_stage_start: ctm.Constexpr,
    qk_stage_count: ctm.Constexpr,
    sksf_is_compact_slot_base: ctm.Constexpr = False,
    page_major_ksf: ctm.Constexpr = False,
) -> None:
    _issue_smem_p4_qk_ksf_range_from_base(
        sKSF,
        base_col_id,
        base_row_id,
        sfb_col_offset,
        sfb_stage_base,
        qk_stage_start,
        qk_stage_count,
        sksf_is_compact_slot_base,
        page_major_ksf,
    )
    _complete_smem_p4_qk_ksf_store()


@cute.jit
def _stage_smem_p4_qk_full_ksf_from_base(
    sKSF,
    base_col_id: ctm.Int32,
    base_row_id: ctm.Int32,
    sfb_col_offset: ctm.Constexpr,
    sfb_stage_base: ctm.Constexpr,
    page_major_ksf: ctm.Constexpr = False,
) -> None:
    if cutlass.const_expr(SMEM_P4_QK_SFB_WARPX2_MODE == 2):
        sfb_s2t_shape, sfb_s2t_multicast = prims.S2TCopyMode.S2T_64x128b_WARPX2_02_13
    else:
        sfb_s2t_shape, sfb_s2t_multicast = prims.S2TCopyMode.S2T_64x128b_WARPX2_01_23
    sfb_tmem_addr_base = base_row_id << ctm.Int32(16) | base_col_id + ctm.Int32(sfb_col_offset)
    sfb_cols_per_scale_chunk: ctm.Constexpr = _compact_sfb_mma_cols_per_scale_chunk(
        SMEM_P4_CTA_GROUP_M, SMEM_P4_QK_SCORE_SLOT_N
    )
    full_root_byte_offset: ctm.Constexpr = _qk_compact_ring_stage_byte_offset(
        sfb_stage_base, QK_SFB_STAGE_BYTES, SMEM_P4_KSF_COMPACT_SLOT_BYTES
    )
    tail_byte_offset: ctm.Constexpr = _qk_compact_ring_stage_byte_offset(
        sfb_stage_base + SMEM_P4_QK_FULL_DATA_STAGES,
        QK_SFB_STAGE_BYTES,
        SMEM_P4_KSF_COMPACT_SLOT_BYTES,
    )
    full_s2t_stride_bytes: ctm.Constexpr = (
        SMEM_P4_QK_SFB_S2T_STRIDE_BYTES * SMEM_P4_QK_DATA_STAGE_SF_CHUNKS
        if page_major_ksf
        else SMEM_P4_QK_SFB_S2T_STRIDE_BYTES
    )
    tail_s2t_stride_bytes: ctm.Constexpr = (
        SMEM_P4_QK_SFB_S2T_STRIDE_BYTES * SMEM_P4_QK_K_TAIL_DATA_STAGE_SF_CHUNKS
        if page_major_ksf
        else SMEM_P4_QK_SFB_S2T_STRIDE_BYTES
    )
    desc_ksf_s2t_full_root = prims.Tcgen05SmemDesc.build(
        sKSF.subview(full_root_byte_offset),
        leading_byte_offset=16,
        stride_byte_offset=full_s2t_stride_bytes,
        layout=0,
    )
    desc_ksf_s2t_tail = prims.Tcgen05SmemDesc.build(
        sKSF.subview(tail_byte_offset),
        leading_byte_offset=16,
        stride_byte_offset=tail_s2t_stride_bytes,
        layout=0,
    )
    for qk_stage_idx in ctm.range_constexpr(SMEM_P4_QK_DATA_STAGES):
        if cutlass.const_expr(qk_stage_idx < SMEM_P4_QK_FULL_DATA_STAGES):
            desc_ksf_s2t = desc_ksf_s2t_full_root.advance_start_address(
                qk_stage_idx * QK_SFB_STAGE_BYTES
            )
        else:
            desc_ksf_s2t = desc_ksf_s2t_tail
        stage_kblocks: ctm.Constexpr = (
            SMEM_P4_QK_DATA_STAGE_KBLOCKS
            if qk_stage_idx < SMEM_P4_QK_FULL_DATA_STAGES
            else QK_TAIL_DATA_STAGE_K_DIM // SMEM_P4_QK_MMA_KBLOCK_DIM
        )
        for local_kblock_idx in ctm.range_constexpr(stage_kblocks):
            global_kblock_idx: ctm.Constexpr = (
                qk_stage_idx * SMEM_P4_QK_DATA_STAGE_KBLOCKS + local_kblock_idx
            )
            for scale_chunk_idx in ctm.range_constexpr(SMEM_P4_QK_MMA_SCALE_CHUNKS_PER_KBLOCK):
                logical_local_scale_word_idx: ctm.Constexpr = (
                    local_kblock_idx * SMEM_P4_QK_MMA_SCALE_CHUNKS_PER_KBLOCK + scale_chunk_idx
                )
                physical_local_scale_word_idx: ctm.Constexpr = logical_local_scale_word_idx
                global_scale_word_idx: ctm.Constexpr = (
                    global_kblock_idx * SMEM_P4_QK_MMA_SCALE_CHUNKS_PER_KBLOCK + scale_chunk_idx
                )
                sfb_tmem_ptr = ctm.inttoptr(
                    sfb_tmem_addr_base
                    + ctm.Int32(global_scale_word_idx * sfb_cols_per_scale_chunk),
                    6,
                    ctm.Int32,
                )
                source_scale_word_stride: ctm.Constexpr = (
                    1 if page_major_ksf else SMEM_P4_QK_SFB_N_CHUNKS
                )
                source_byte_offset: ctm.Constexpr = (
                    physical_local_scale_word_idx * source_scale_word_stride * (512 // 16)
                    + SMEM_P4_QK_COMPACT_SFB_S2T_BYTE_OFFSET
                )
                if prims.elect_sync():
                    prims.tcgen05_cp(
                        sfb_s2t_shape,
                        sfb_tmem_ptr,
                        desc_ksf_s2t + ctm.Int64(source_byte_offset),
                        group=prims.CTAGroup.CTA_2,
                        multicast=sfb_s2t_multicast,
                    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    cute.arch.fence_view_async_tmem_store()
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def _stage_smem_p4_pv_vsf_bank_from_base(
    sVSF, base_col_id: ctm.Int32, base_row_id: ctm.Int32, v_stage: ctm.Int32
) -> None:
    sfb_s2t_shape, sfb_s2t_multicast = prims.S2TCopyMode.S2T_64x128b_WARPX2_01_23
    sVSF_stage = sVSF.subview(ctm.Int32(SMEM_P4_V_SFB_TMA_STAGE_BYTES) * v_stage)
    desc_vsf_s2t = prims.Tcgen05SmemDesc.build(
        sVSF_stage,
        leading_byte_offset=16,
        stride_byte_offset=SMEM_P4_PV_SFB_S2T_STRIDE_BYTES,
        layout=0,
    )
    vsf_tmem_addr_base = base_row_id << ctm.Int32(16) | base_col_id + ctm.Int32(
        SMEM_P4_PV_VSF_TAIL_COL_OFFSET
    )
    total_n_chunks: ctm.Constexpr = SMEM_P4_PV_SFB_N_CHUNKS * SMEM_P4_N_OUT_TILES
    for n_tile_idx in ctm.range_constexpr(SMEM_P4_N_OUT_TILES):
        for global_kblock_idx in ctm.range_constexpr(SMEM_P4_PV_KBLOCKS_PER_LI):
            vsf_col_delta: ctm.Constexpr = (
                n_tile_idx * SMEM_P4_PV_VSF_N_TILE_COLS
                + global_kblock_idx * SMEM_P4_PV_SFB_COLS_PER_KBLOCK
            )
            vsf_tmem_addr = vsf_tmem_addr_base + ctm.Int32(vsf_col_delta)
            vsf_tmem_ptr = ctm.inttoptr(vsf_tmem_addr, 6, ctm.Int32)
            source_byte_offset: ctm.Constexpr = (
                global_kblock_idx * total_n_chunks + n_tile_idx * SMEM_P4_PV_SFB_N_CHUNKS
            ) * (512 // 16)
            if prims.elect_sync():
                prims.tcgen05_cp(
                    sfb_s2t_shape,
                    vsf_tmem_ptr,
                    desc_vsf_s2t + ctm.Int64(source_byte_offset),
                    group=prims.CTAGroup.CTA_2,
                    multicast=sfb_s2t_multicast,
                )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    cute.arch.fence_view_async_tmem_store()


@cute.jit
def _issue_smem_p4_qk_stage_burst_from_base(
    sQ,
    sK,
    sQSF,
    sKSF,
    base_col_id: ctm.Int32,
    base_row_id: ctm.Int32,
    score_slot_idx: ctm.Int32,
    qk_kblock_start: ctm.Constexpr,
    qk_kblocks: ctm.Constexpr,
    acc_col_offset: ctm.Int32,
    sfa_col_offset_base: ctm.Constexpr,
    sfb_col_offset_base: ctm.Constexpr,
    use_split_operand_stages: ctm.Constexpr = False,
    use_resident_q_stages: ctm.Constexpr = False,
    b_stage_base: ctm.Int32 = 0,
    split_b_stage_base: ctm.Constexpr = NUM_QK_AB_STAGE,
    fence_after: ctm.Constexpr = True,
    b_smem_is_compact_slot_base: ctm.Constexpr = False,
) -> None:
    qk_sfb_scale_stride_cols: ctm.Constexpr = SMEM_P4_QK_SFB_COLS_PER_KBLOCK
    if cutlass.const_expr(use_resident_q_stages):
        issue_stage_limit: ctm.Constexpr = SMEM_P4_QK_DATA_STAGES
        issue_stage_start: ctm.Constexpr = qk_kblock_start // SMEM_P4_QK_DATA_STAGE_KBLOCKS
        requested_stage_count: ctm.Constexpr = (
            qk_kblocks + SMEM_P4_QK_DATA_STAGE_KBLOCKS - 1
        ) // SMEM_P4_QK_DATA_STAGE_KBLOCKS
        issue_stage_count: ctm.Constexpr = min(
            requested_stage_count, issue_stage_limit - issue_stage_start
        )
        for stage in ctm.range_constexpr(issue_stage_start, issue_stage_start + issue_stage_count):
            stage_kblock_start: ctm.Constexpr = stage * SMEM_P4_QK_DATA_STAGE_KBLOCKS
            stage_kblocks: ctm.Constexpr = (
                SMEM_P4_QK_DATA_STAGE_KBLOCKS
                if stage < SMEM_P4_QK_FULL_DATA_STAGES
                else QK_TAIL_EFFECTIVE_MMA_KBLOCKS
            )
            first_order_residual_tail: ctm.Constexpr = stage == SMEM_P4_QK_FULL_DATA_STAGES
            _issue_smem_p4_qk_score_slot_from_base(
                sQ,
                sK,
                sQSF,
                sKSF,
                base_col_id,
                base_row_id,
                stage,
                score_slot_idx=score_slot_idx,
                clear_accumulator=stage_kblock_start == 0,
                qk_kblock_idx=stage_kblock_start,
                qk_kblocks=stage_kblocks,
                acc_col_offset=acc_col_offset,
                sfa_col_offset=sfa_col_offset_base,
                sfb_col_offset=sfb_col_offset_base + stage_kblock_start * qk_sfb_scale_stride_cols,
                stage_sfa_from_smem=False,
                stage_sfb_from_smem=False,
                use_split_operand_stages=True,
                a_stage=stage,
                b_stage=b_stage_base + stage,
                use_stage_local_operand_kblock_start=True,
                stage_local_operand_kblock_start=0,
                stage_local_scale_kblock_start=stage_kblock_start,
                reuse_b_first_kblock=False,
                first_order_residual_tail=first_order_residual_tail,
                b_smem_is_compact_slot_base=b_smem_is_compact_slot_base,
            )
    else:
        for burst_kblock_idx in ctm.range_constexpr(qk_kblocks):
            qk_kblock_idx: ctm.Constexpr = qk_kblock_start + burst_kblock_idx
            stage: ctm.Constexpr = qk_kblock_idx % NUM_QK_AB_STAGE
            if cutlass.const_expr(use_split_operand_stages):
                a_stage: ctm.Constexpr = stage
                b_stage: ctm.Constexpr = split_b_stage_base + stage
            else:
                a_stage: ctm.Constexpr = 0
                b_stage: ctm.Constexpr = 0
            _issue_smem_p4_qk_score_slot_from_base(
                sQ,
                sK,
                sQSF,
                sKSF,
                base_col_id,
                base_row_id,
                stage,
                score_slot_idx=score_slot_idx,
                clear_accumulator=qk_kblock_idx == 0,
                qk_kblock_idx=qk_kblock_idx,
                acc_col_offset=acc_col_offset,
                sfa_col_offset=sfa_col_offset_base,
                sfb_col_offset=sfb_col_offset_base + burst_kblock_idx * qk_sfb_scale_stride_cols,
                stage_sfa_from_smem=False,
                stage_sfb_from_smem=False,
                use_split_operand_stages=use_split_operand_stages,
                a_stage=a_stage,
                b_stage=b_stage,
            )
    if cutlass.const_expr(fence_after):
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)


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
def _cross_arrive_smem_p4_cta2_mbar(mbar) -> None:
    for peer_cta_rank in ctm.range_constexpr(CLUSTER_SHAPE_MNK[0]):
        peer_mbar = _mapa_shared_cluster(mbar, peer_cta_rank)
        _mbarrier_arrive_shared_cluster(peer_mbar)


@cute.jit
def _arrive_smem_p4_mapped_leader_mbar(leader_mbar) -> None:
    _mbarrier_arrive_shared_cluster(leader_mbar)


@cute.jit
def _store_smem_p4_n256_rowsum_part(
    sRowSumParts,
    warp_idx: ctm.Int32,
    tidx: ctm.Int32,
    producer_warp_base: ctm.Int32,
    bank_idx: ctm.Int32,
    score_half_idx: ctm.Int32,
    value: ctm.Float32,
) -> None:
    local_warp_idx = warp_idx - producer_warp_base
    lane_idx = tidx & ctm.Int32(31)
    row_band = local_warp_idx & ctm.Int32(1)
    col_band = local_warp_idx >> ctm.Int32(1)
    local_row = row_band * ctm.Int32(32) + lane_idx
    slot_idx = bank_idx * ctm.Int32(SMEM_P4_SCORE_SLOTS_PER_KV_TILE) + score_half_idx
    row_state_offset = slot_idx * ctm.Int32(
        SMEM_P4_SCORE_SLOT_ROW_STATE_STRIDE
    ) + col_band * ctm.Int32(SMEM_P4_BMM1_M)
    sRowSumParts[row_state_offset + local_row] = value


@cute.jit
def _load_smem_p4_n256_rowsum_part(
    sRowSumParts,
    local_row: ctm.Int32,
    bank_idx: ctm.Int32,
    score_half_idx: ctm.Constexpr,
    col_band_idx: ctm.Constexpr,
) -> ctm.Float32:
    slot_idx = bank_idx * ctm.Int32(SMEM_P4_SCORE_SLOTS_PER_KV_TILE) + ctm.Int32(score_half_idx)
    row_state_offset = slot_idx * ctm.Int32(SMEM_P4_SCORE_SLOT_ROW_STATE_STRIDE) + ctm.Int32(
        col_band_idx * SMEM_P4_BMM1_M
    )
    return sRowSumParts[row_state_offset + local_row]


@cute.jit
def _load_p4_n256_score_half_from_tmem(
    sAtomicRunningRowMax,
    qk_full_mbar,
    score_tmem_addresses,
    producer_col_band: ctm.Int32,
    warp_idx: ctm.Int32,
    tidx: ctm.Int32,
    kv_tile_idx: ctm.Int32,
    valid_k: ctm.Int32,
    mask_valid_k: ctm.Constexpr,
    qk_phase: ctm.Int32,
    wait_for_qk_full: ctm.Constexpr,
    producer_warp_base: ctm.Int32,
    score_half_idx: ctm.Int32,
) -> tuple:
    col_band = producer_col_band
    if cutlass.const_expr(wait_for_qk_full):
        while not prims.mbarrier_try_wait_parity(qk_full_mbar, qk_phase, time_limit=10000000):
            pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    score_half_n: ctm.Constexpr = SMEM_P4_BMM1_N // SMEM_P4_N256_SCORE_HALVES
    score_half_groups_per_col_band: ctm.Constexpr = (
        score_half_n // SF_VEC_SIZE // SMEM_P4_TMEM_WARP_N
    )
    tile_base = kv_tile_idx * ctm.Int32(SMEM_P4_BMM1_N)
    pending_score_groups = []
    pending_group_stats = []
    for local_group_idx in ctm.range_constexpr(score_half_groups_per_col_band):
        tmem = ctm.inttoptr(score_tmem_addresses[local_group_idx], 6, ctm.Float32)
        if cutlass.const_expr(mask_valid_k):
            group_scores = prims.tcgen05_ld(
                prims.Tcgen05LdStShape.SHAPE_32X32B, tmem, num=SF_VEC_SIZE
            )
            pending_score_groups.append(group_scores)
            pending_group_stats.append(None)
        else:
            regs = _tcgen05_ld_red_x16_max_f32(tmem)
            pending_score_groups.append(regs)
            pending_group_stats.append(regs[SF_VEC_SIZE])
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
    cute.arch.fence_view_async_tmem_load()
    score_values = []
    group_max_values = []
    group_valid_values = []
    tile_row_max = ctm.Float32(-ctm.Float32.inf)
    for local_group_idx in ctm.range_constexpr(score_half_groups_per_col_band):
        pending_scores = pending_score_groups[local_group_idx]
        if cutlass.const_expr(mask_valid_k):
            group_scores = pending_scores
            group_max = ctm.Float32(-ctm.Float32.inf)
            group_base = (
                tile_base
                + col_band * ctm.Int32(SMEM_P4_N256_SCORE_N_PER_COL_BAND)
                + score_half_idx * ctm.Int32(SMEM_P4_N256_SCORE_N_PER_PRODUCER)
                + ctm.Int32(local_group_idx * SF_VEC_SIZE)
            )
            group_valid = ctm.Boolean(group_base < valid_k)
            for elem in ctm.range_constexpr(SF_VEC_SIZE):
                global_k_idx = group_base + ctm.Int32(elem)
                score = _mask_invalid_score(group_scores[elem], global_k_idx, valid_k)
                score_values.append(score)
                group_max = ctm.cute.arch.fmax(group_max, score)
        else:
            group_scores = ctm.Vector.from_elements(
                tuple((pending_scores[elem].bitcast(ctm.Float32) for elem in range(SF_VEC_SIZE))),
                dtype=ctm.Float32,
            )
            group_max = pending_group_stats[local_group_idx].bitcast(ctm.Float32)
            for elem in ctm.range_constexpr(SF_VEC_SIZE):
                score_values.append(group_scores[elem])
            group_valid = ctm.Boolean(True)
        group_max_values.append(group_max)
        group_valid_values.append(group_valid)
        tile_row_max = ctm.cute.arch.fmax(tile_row_max, group_max)
    if cutlass.const_expr(mask_valid_k):
        if not ctm.Boolean(tile_base < valid_k):
            tile_row_max = ctm.Float32(-ctm.Float32.inf)
    _publish_p4_n256_atomic_rowmax(
        sAtomicRunningRowMax, tile_row_max, warp_idx, tidx, producer_warp_base
    )
    return (
        ctm.Vector.from_elements(tuple(score_values), dtype=ctm.Float32),
        ctm.Vector.from_elements(tuple(group_max_values), dtype=ctm.Float32),
        ctm.Vector.from_elements(tuple(group_valid_values), dtype=ctm.Boolean),
    )


@cute.jit
def _select_smem_p4_lazy_anchor_with_validity(
    new_row_max: ctm.Float32,
    new_row_valid: ctm.Boolean,
    running_row_anchor: ctm.Float32,
    softmax_scale_log2: ctm.Float32,
) -> tuple:
    new_row_anchor = running_row_anchor
    lane_rebase = ctm.Boolean(False)
    anchor_delta_log2 = ctm.Float32(0.0)
    if new_row_valid:
        negative_infinity = ctm.Float32(-ctm.Float32.inf)
        if running_row_anchor == negative_infinity:
            new_row_anchor = new_row_max
        else:
            anchor_delta_log2 = (new_row_max - running_row_anchor) * softmax_scale_log2
            if anchor_delta_log2 > ctm.Float32(SMEM_P4_LAZY_ANCHOR_REBASE_LOG2):
                new_row_anchor = new_row_max
                lane_rebase = ctm.Boolean(True)
    return (new_row_anchor, lane_rebase, anchor_delta_log2)


@cute.jit
def _select_smem_p4_lazy_anchor_runtime(
    new_row_max: ctm.Float32,
    running_row_anchor: ctm.Float32,
    softmax_scale_log2: ctm.Float32,
    has_previous_tile: ctm.Boolean,
) -> tuple:
    anchor_delta_log2 = (new_row_max - running_row_anchor) * softmax_scale_log2
    lane_rebase = has_previous_tile & (
        anchor_delta_log2 > ctm.Float32(SMEM_P4_LAZY_ANCHOR_REBASE_LOG2)
    )
    new_row_anchor = running_row_anchor
    if ~has_previous_tile | lane_rebase:
        new_row_anchor = new_row_max
    return (new_row_anchor, lane_rebase, anchor_delta_log2)


@cute.jit
def _prepack_p4_n256_score_half_before_rowmax(
    score_values,
    group_max_values,
    group_valid_values,
    warp_idx: ctm.Int32,
    softmax_scale_log2: ctm.Float32,
    kv_tile_idx: ctm.Int32,
    valid_k: ctm.Int32,
    producer_warp_base: ctm.Int32,
    score_half_idx: ctm.Int32,
    mask_valid_k: ctm.Constexpr = False,
    use_mixed_imlp: ctm.Constexpr = False,
) -> tuple:
    local_warp_idx = warp_idx - producer_warp_base
    col_band = local_warp_idx >> ctm.Int32(1)
    score_half_n: ctm.Constexpr = SMEM_P4_BMM1_N // SMEM_P4_N256_SCORE_HALVES
    score_half_groups_per_col_band: ctm.Constexpr = (
        score_half_n // SF_VEC_SIZE // SMEM_P4_TMEM_WARP_N
    )
    tile_base = kv_tile_idx * ctm.Int32(SMEM_P4_BMM1_N)
    packed_p4_words = []
    group_p_sums = []
    for local_group_idx in ctm.range_constexpr(score_half_groups_per_col_band):
        score_value_base: ctm.Constexpr = local_group_idx * SF_VEC_SIZE
        group_scores = ctm.Vector.from_elements(
            tuple((score_values[score_value_base + elem] for elem in range(SF_VEC_SIZE))),
            dtype=ctm.Float32,
        )
        group_base = (
            tile_base
            + col_band * ctm.Int32(SMEM_P4_N256_SCORE_N_PER_COL_BAND)
            + score_half_idx * ctm.Int32(SMEM_P4_N256_SCORE_N_PER_PRODUCER)
            + ctm.Int32(local_group_idx * SF_VEC_SIZE)
        )
        group_valid = group_valid_values[local_group_idx]
        group_max = group_max_values[local_group_idx]
        if cutlass.const_expr(use_mixed_imlp):
            p_bias = ctm.Float32(LOG2_6) - group_max * softmax_scale_log2
            word0, word1, p_sum = _pack_p4_16_scores_from_group_packed_f16x2(
                group_scores,
                group_base,
                valid_k,
                group_valid,
                mask_valid_k,
                softmax_scale_log2,
                p_bias,
                elem_base=0,
            )
        else:
            p_bias = ctm.Float32(LOG2_6) - group_max * softmax_scale_log2
            word0, word1, p_sum = _pack_p4_16_scores_from_group(
                group_scores,
                group_base,
                valid_k,
                group_valid,
                mask_valid_k,
                use_mixed_imlp,
                softmax_scale_log2,
                p_bias,
                elem_base=0,
            )
        packed_p4_words.append(word0)
        packed_p4_words.append(word1)
        group_p_sums.append(p_sum)
    return (
        ctm.Vector.from_elements(tuple(packed_p4_words), dtype=ctm.Int32),
        ctm.Vector.from_elements(tuple(group_p_sums), dtype=ctm.Float32),
    )


@cute.jit
def _make_p4_n256_psf_bank_to_tmem_desc(
    sPSF, base_col_id: ctm.Int32, base_row_id: ctm.Int32, bank_idx: ctm.Int32
) -> tuple:
    sPSF_bank = sPSF.subview(bank_idx * ctm.Int32(SMEM_P4_N256_PSF_S2T_BANK_WORDS))
    desc_psf_s2t = prims.Tcgen05SmemDesc.build(
        sPSF_bank, leading_byte_offset=16, stride_byte_offset=128, layout=0
    )
    p_sfa_col_id = (
        base_col_id
        + ctm.Int32(SMEM_P4_P_SFA_BANK_REL_COL_OFFSET)
        + bank_idx * ctm.Int32(SMEM_P4_QK_PIPELINE_SLOT_STRIDE)
    )
    p_sfa_tmem_addr = base_row_id << ctm.Int32(16) | p_sfa_col_id
    p_sfa_tmem = ctm.inttoptr(p_sfa_tmem_addr, 6, ctm.Int32)
    return (desc_psf_s2t, p_sfa_tmem)


@cute.jit
def _load_smem_p4_n256_quadrant_rowsum(
    sRowSumParts,
    sP,
    local_row: ctm.Int32,
    bank_idx: ctm.Int32,
    score_half_idx: ctm.Constexpr,
    col_band_idx: ctm.Constexpr,
) -> ctm.Float32:
    return _load_smem_p4_n256_rowsum_part(
        sRowSumParts, local_row, bank_idx, score_half_idx, col_band_idx
    )


@cute.jit
def _issue_smem_p4_pv_score_slot_from_base(
    sP,
    sV,
    sPSF,
    sVSF,
    base_col_id: ctm.Int32,
    base_row_id: ctm.Int32,
    p_stage: ctm.Int32,
    is_first_kv_tile: ctm.Boolean,
    score_slot_idx: ctm.Constexpr = -1,
    v_stage: ctm.Int32 = 0,
    acc_col_offset: ctm.Constexpr = SMEM_P4_O_ACC_COL_OFFSET,
    p_tmem_col_offset: ctm.Int32 = -1,
    n_tile_idx: ctm.Constexpr = 0,
    p_layout_score_slot_idx: ctm.Constexpr = -1,
    pv_kblock_start_delta: ctm.Constexpr = 0,
    pv_kblocks: ctm.Constexpr = -1,
    clear_accumulator_override: ctm.Constexpr = -1,
    bank_idx: ctm.Int32 = -1,
    p_half_idx: ctm.Constexpr = -1,
    local_kblock_idx: ctm.Constexpr = -1,
    global_kblock_idx: ctm.Constexpr = -1,
    p_sfa_col_offset: ctm.Int32 = -1,
    a_collector_op: ctm.Constexpr = None,
) -> None:
    pv_stage = v_stage
    p_layout_half_idx: ctm.Constexpr = (
        score_slot_idx if p_layout_score_slot_idx < 0 else p_layout_score_slot_idx
    )
    physical_p_half_idx: ctm.Constexpr = p_layout_half_idx if p_half_idx < 0 else p_half_idx
    physical_local_kblock_idx: ctm.Constexpr = (
        pv_kblock_start_delta if local_kblock_idx < 0 else local_kblock_idx
    )
    physical_global_kblock_idx: ctm.Constexpr = (
        score_slot_idx * SMEM_P4_PV_KBLOCKS_PER_SCORE_SLOT + physical_local_kblock_idx
        if global_kblock_idx < 0
        else global_kblock_idx
    )
    p_bank_col_offset = p_tmem_col_offset
    default_clear_pv_accumulator = ctm.Boolean(False)
    if cutlass.const_expr(physical_global_kblock_idx == 0):
        default_clear_pv_accumulator = is_first_kv_tile
    clear_pv_accumulator = default_clear_pv_accumulator
    if cutlass.const_expr(clear_accumulator_override >= 0):
        clear_pv_accumulator = ctm.Boolean(clear_accumulator_override != 0)
    sfb_col_offset: ctm.Constexpr = (
        SMEM_P4_PV_VSF_TAIL_COL_OFFSET
        + n_tile_idx * SMEM_P4_PV_VSF_N_TILE_COLS
        + physical_global_kblock_idx * SMEM_P4_PV_SFB_COLS_PER_KBLOCK
    )
    p_stage_slot_col_offset = p_bank_col_offset + ctm.Int32(
        physical_p_half_idx * SMEM_P4_P_TMEM_SLOT_STRIDE
    )
    effective_p_sfa_col_offset = p_sfa_col_offset
    pv_mma_kblocks: ctm.Constexpr = (
        pv_kblocks if pv_kblocks > 0 else SMEM_P4_PV_KBLOCKS_PER_SCORE_SLOT
    )
    pv_mma_kblock_start: ctm.Constexpr = physical_global_kblock_idx
    pv_a_tmem_col_offset = p_stage_slot_col_offset
    pv_m_dim: ctm.Constexpr = SMEM_P4_CTA_GROUP_M
    pv_b_major: ctm.Constexpr = 0
    pv_page_idx: ctm.Constexpr = physical_global_kblock_idx // (TRTLLM_PAGE_SIZE // MMA_KBLOCK_DIM)
    pv_page_local_kblock_idx: ctm.Constexpr = physical_global_kblock_idx % (
        TRTLLM_PAGE_SIZE // MMA_KBLOCK_DIM
    )
    pv_b_mma_smem_word_delta: ctm.Constexpr = pv_page_idx * (B_STAGE_BYTES // 4) + n_tile_idx * (
        SMEM_P4_V_PAGE_N_TILE_BYTES // 4
    )
    pv_b_mma_kblock_start: ctm.Constexpr = pv_page_local_kblock_idx
    pv_b_mma_leading_bytes: ctm.Constexpr = SMEM_P4_PV_PAGE_B_MMA_LEADING_BYTES
    pv_b_mma_stride_bytes: ctm.Constexpr = SMEM_P4_PV_PAGE_B_MMA_STRIDE_BYTES
    pv_b_mma_layout: ctm.Constexpr = SMEM_P4_PV_PAGE_B_MMA_LAYOUT
    _issue_mxf4nvf4_mma_tile_from_base(
        sP,
        sV,
        sPSF,
        sVSF,
        base_col_id,
        base_row_id,
        pv_stage,
        clear_accumulator=clear_pv_accumulator,
        acc_col_offset=acc_col_offset,
        sfa_col_offset=effective_p_sfa_col_offset,
        sfb_col_offset=sfb_col_offset,
        n_dim=SMEM_P4_BMM2_N,
        m_dim=pv_m_dim,
        mma_kblock_start=pv_mma_kblock_start,
        mma_kblocks=pv_mma_kblocks,
        mma_kblock_dim=MMA_KBLOCK_DIM,
        mma_kblock_idesc_k_dim=MMA_KBLOCK_IDESC_K_DIM,
        mma_scale_chunks_per_kblock=MMA_SCALE_CHUNKS_PER_KBLOCK,
        mma_scale_vec_size=MMA_SCALE_VEC_SIZE,
        a_tmem_kblock_cols=SMEM_P4_P_TMEM_KBLOCK_COLS,
        stage_sfa_from_smem=False,
        stage_sfb_from_smem=False,
        compact_sfb_words_per_row=MMA_SCALE_CHUNKS_PER_TILE,
        compact_sfb_extra_row_stride=SMEM_P4_PV_SFB_EXTRA_ROW_STRIDE,
        compact_sfb_extra_col_stride=SMEM_P4_PV_SFB_EXTRA_COL_STRIDE,
        compact_sfb_cp_group_one=SMEM_P4_PV_SFB_CP_GROUP_ONE,
        compact_sfb_warpx2_mode=SMEM_P4_PV_SFB_WARPX2_MODE,
        sfb_mma_col_delta=SMEM_P4_PV_SFB_MMA_COL_DELTA,
        sfb_mma_kblock_stride_cols=SMEM_P4_PV_SFB_COLS_PER_KBLOCK,
        use_split_operand_stages=True,
        a_stage=p_stage,
        b_stage=pv_stage,
        a_stage_words=SMEM_P4_P4_SMEM_STAGE_WORDS,
        b_stage_words=SMEM_P4_V_DATA_STAGE_BYTES // 4,
        sfb_stage_bytes=SMEM_P4_V_SFB_TMA_STAGE_BYTES,
        a_from_tmem=False,
        a_tmem_col_offset=pv_a_tmem_col_offset,
        b_major=pv_b_major,
        b_mma_smem_word_delta=pv_b_mma_smem_word_delta,
        b_mma_leading_byte_offset=pv_b_mma_leading_bytes,
        b_mma_stride_byte_offset=pv_b_mma_stride_bytes,
        b_mma_layout=pv_b_mma_layout,
        b_mma_k_segment_offset=SMEM_P4_PV_B_MMA_K_SEGMENT_OFFSET,
        use_split_mma_kblock_start=True,
        a_mma_kblock_start=physical_local_kblock_idx,
        b_mma_kblock_start=pv_b_mma_kblock_start,
        a_mma_leading_byte_offset=SMEM_P4_P4_SMEM_MMA_LEADING_BYTE_OFFSET,
        a_mma_stride_byte_offset=SMEM_P4_P4_SMEM_MMA_STRIDE_BYTE_OFFSET,
        a_mma_layout=SMEM_P4_P4_SMEM_MMA_LAYOUT,
        scale_mma_kblock_start=physical_global_kblock_idx,
        sfb_s2t_stride_byte_offset=SMEM_P4_PV_SFB_S2T_STRIDE_BYTES,
        a_sf_layout=SMEM_P4_P_SFA_LAYOUT,
        a_collector_op=a_collector_op,
    )


@cute.jit
def _store_final_o_from_tmem(
    mC_mnl: cute.Tensor,
    mAccum_mnl: cute.Tensor,
    mRowMax_ml: cute.Tensor,
    mRowSum_ml: cute.Tensor,
    tmem_holding_buf,
    warp_idx: ctm.Int32,
    tidx: ctm.Int32,
    bidx: ctm.Int32,
    bidy: ctm.Int32,
    bidz: ctm.Int32,
    m: ctm.Int32,
    n: ctm.Int32,
    final_row_max: ctm.Float32,
    final_row_sum: ctm.Float32,
    final_stat_scale: ctm.Float32,
    output_normalizer: ctm.Float32,
    write_softmax_stats: ctm.Constexpr = False,
    producer_warp_base: ctm.Constexpr = 0,
    store_split: ctm.Constexpr = False,
) -> None:
    gC_arr = ctm.make_array_view(mC_mnl)
    gAccum_arr = ctm.make_array_view(mAccum_mnl)
    vsize: ctm.Constexpr = 16
    tmem_raw_addr = tmem_holding_buf.load()
    base_col_id = tmem_raw_addr & ctm.Int32(65535)
    base_row_id = tmem_raw_addr >> ctm.Int32(16)
    cta_rank = cute.arch.block_idx_in_cluster()
    local_warp_idx = warp_idx - ctm.Int32(producer_warp_base)
    row_band = local_warp_idx & ctm.Int32(1)
    col_band = local_warp_idx >> ctm.Int32(1)
    local_row = row_band * ctm.Int32(32) + (tidx & ctm.Int32(31))
    row_id_with_warp_offset = base_row_id + row_band * ctm.Int32(64) + col_band * ctm.Int32(32)
    m_tile_idx = bidx // ctm.Int32(CLUSTER_SHAPE_MNK[0])
    row = (
        m_tile_idx * ctm.Int32(SMEM_P4_CTA_GROUP_M)
        + cta_rank * ctm.Int32(SMEM_P4_BMM1_M)
        + local_row
    )
    if cutlass.const_expr(store_split):
        # Both column bands have the same row state; export each row once.
        if col_band == ctm.Int32(0):
            mRowMax_ml[row, bidz] = final_row_max
            mRowSum_ml[row, bidz] = final_row_sum
    batch_offset = bidz * m * n
    if cutlass.const_expr(write_softmax_stats):
        if col_band == ctm.Int32(0) and bidy == ctm.Int32(0):
            mRowMax_ml[row, bidz] = final_row_max * final_stat_scale
            mRowSum_ml[row, bidz] = final_row_sum
    n_tile_total = n // ctm.Int32(OUT_DIM)
    subtile_cols: ctm.Constexpr = 32
    subtiles_per_n_tile: ctm.Constexpr = SMEM_P4_BMM2_N // SMEM_P4_TMEM_WARP_N // subtile_cols
    output_subtile_total = n_tile_total * ctm.Int32(subtiles_per_n_tile)
    for output_subtile_idx in cutlass.range(0, output_subtile_total, 1, unroll=1):
        n_tile_idx = output_subtile_idx // ctm.Int32(subtiles_per_n_tile)
        subtile_in_n_tile = output_subtile_idx - n_tile_idx * ctm.Int32(subtiles_per_n_tile)
        col_chunk = subtile_in_n_tile * ctm.Int32(subtile_cols)
        o_acc_col_offset = ctm.Int32(SMEM_P4_O_ACC_COL_OFFSETS[0]) + n_tile_idx * ctm.Int32(
            SMEM_P4_O_ACC_TILE_COLS
        )
        col_base = (
            bidy * ctm.Int32(MMA_TILER_MNK[1])
            + n_tile_idx * ctm.Int32(OUT_DIM)
            + col_band * ctm.Int32(SMEM_P4_BMM2_N // SMEM_P4_TMEM_WARP_N)
            + col_chunk
        )
        col_id = base_col_id + o_acc_col_offset + col_chunk
        tmem_addr = row_id_with_warp_offset << ctm.Int32(16) | col_id
        tmem = ctm.inttoptr(tmem_addr, 6, ctm.Float32)
        c_rmem = prims.tcgen05_ld(prims.Tcgen05LdStShape.SHAPE_32X32B, tmem, num=subtile_cols)
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        for vec_idx in ctm.range_constexpr(subtile_cols // vsize):
            linear_idx = row * n + col_base + vec_idx * vsize + batch_offset
            vec_f32 = c_rmem[vec_idx * vsize : vec_idx * vsize + vsize]
            if cutlass.const_expr(store_split):
                gAccum_arr.store(
                    vec_f32 * final_stat_scale,
                    idx=linear_idx,
                    vector_size=vsize,
                    alignment=32,
                )
        if cutlass.const_expr(not store_split):
            output_linear_idx = row * n + col_base + batch_offset
            output_vec_f32 = c_rmem[0:subtile_cols]
            output_vec = (output_vec_f32 * output_normalizer).to(mC_mnl.element_type)
            # The decode output is consumed after this kernel and has no in-kernel
            # reuse.  Avoid allocating its streaming write in L1; on GR100 this
            # preserves the 256-bit vector store while lowering it to
            # STG.E.NA.ENL2.256.
            (gC_arr.subview(output_linear_idx)).data_ptr().nvvm_store_ext(
                output_vec,
                evict=EvictPriority.NOALLOCATE,
            )


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
        _mbarrier_arrive_shared_cluster(peer_mbar)
        while not prims.mbarrier_try_wait_parity(tmem_dealloc_mbar, 0, time_limit=10000000):
            pass
        prims.tcgen05_dealloc(
            acc_tmem_ptr,
            num_tmem_alloc_cols,
            is_exclusive=SMEM_P4_TMEM_ALLOC_EXCLUSIVE,
            group=prims.CTAGroup.CTA_2,
        )


@cute.jit
def _runtime_issue_pv_tile_slot(
    sP,
    sV,
    sPSF,
    sVSF,
    tmem_base_col_id: ctm.Int32,
    tmem_base_row_id: ctm.Int32,
    qk_full_mbar,
    pair_overlap_sync_mbars,
    p4_smem_slot0_mbars,
    pv_full_mbar,
    v_smem_mbars,
    stream_li_idx: ctm.Int32,
    stream_li_total: ctm.Int32,
    bank_idx: ctm.Int32,
    initial_tile: ctm.Boolean,
) -> None:
    v_stage = stream_li_idx % ctm.Int32(SMEM_P4_V_PIPELINE_STAGES)
    qk_phase = (
        stream_li_idx
        // ctm.Int32(SMEM_P4_QK_PIPELINE_SLOTS)
        % ctm.Int32(SMEM_P4_MBAR_PARITY_PHASES)
    )
    while not prims.mbarrier_try_wait_parity(
        qk_full_mbar.subview(bank_idx), qk_phase, time_limit=10000000
    ):
        pass
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    _stage_smem_p4_pv_vsf_bank_from_base(sVSF, tmem_base_col_id, tmem_base_row_id, v_stage)
    prefetched_psf_desc, prefetched_psf_tmem = _make_p4_n256_psf_bank_to_tmem_desc(
        sPSF, tmem_base_col_id, tmem_base_row_id, bank_idx
    )
    pair_phase = (
        stream_li_idx
        // ctm.Int32(SMEM_P4_QK_PIPELINE_SLOTS)
        % ctm.Int32(SMEM_P4_MBAR_PARITY_PHASES)
    )
    p_source_ready_mbar = pair_overlap_sync_mbars.subview(
        SMEM_P4_PAIR_P_SOURCE_READY_MBAR_OFFSET + (bank_idx)
    )
    while not prims.mbarrier_try_wait_parity(p_source_ready_mbar, pair_phase, time_limit=10000000):
        pass
    _fence_async_shared_cta()
    _issue_p4_n256_psf_bank_to_tmem_from_desc(prefetched_psf_desc, prefetched_psf_tmem)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    p_archive_col_offset = ctm.Int32(SMEM_P4_P_ARCHIVE_TMEM_COLS[0]) + bank_idx * ctm.Int32(
        SMEM_P4_P_ARCHIVE_SLOT_STRIDE
    )
    p_half0_smem_stage = bank_idx * ctm.Int32(SMEM_P4_SCORE_SLOTS_PER_KV_TILE)
    p_half1_smem_stage = p_half0_smem_stage + ctm.Int32(1)
    p_sfa_col_offset = ctm.Int32(SMEM_P4_P_SFA_BANK_REL_COL_OFFSET) + bank_idx * ctm.Int32(
        SMEM_P4_QK_PIPELINE_SLOT_STRIDE
    )
    for local_kblock_idx in ctm.range_constexpr(SMEM_P4_PV_KBLOCKS_PER_SCORE_SLOT):
        for n_tile in cutlass.range_constexpr(SMEM_P4_N_OUT_TILES):
            a_collector_op: ctm.Constexpr = (
                prims.Tcgen05MMACollectorOp.FILL
                if n_tile == 0
                else prims.Tcgen05MMACollectorOp.LASTUSE
            )
            _issue_smem_p4_pv_score_slot_from_base(
                sP,
                sV,
                sPSF,
                sVSF,
                tmem_base_col_id,
                tmem_base_row_id,
                p_half0_smem_stage,
                initial_tile,
                v_stage=v_stage,
                acc_col_offset=SMEM_P4_O_ACC_COL_OFFSETS[n_tile],
                n_tile_idx=n_tile,
                p_tmem_col_offset=p_archive_col_offset,
                bank_idx=bank_idx,
                p_half_idx=0,
                local_kblock_idx=local_kblock_idx,
                global_kblock_idx=local_kblock_idx,
                pv_kblocks=1,
                p_sfa_col_offset=p_sfa_col_offset,
                a_collector_op=a_collector_op,
            )
    for local_kblock_idx in ctm.range_constexpr(SMEM_P4_PV_KBLOCKS_PER_SCORE_SLOT):
        global_kblock_idx: ctm.Constexpr = SMEM_P4_PV_KBLOCKS_PER_SCORE_SLOT + local_kblock_idx
        for n_tile in cutlass.range_constexpr(SMEM_P4_N_OUT_TILES):
            a_collector_op: ctm.Constexpr = (
                prims.Tcgen05MMACollectorOp.FILL
                if n_tile == 0
                else prims.Tcgen05MMACollectorOp.LASTUSE
            )
            _issue_smem_p4_pv_score_slot_from_base(
                sP,
                sV,
                sPSF,
                sVSF,
                tmem_base_col_id,
                tmem_base_row_id,
                p_half1_smem_stage,
                initial_tile,
                v_stage=v_stage,
                acc_col_offset=SMEM_P4_O_ACC_COL_OFFSETS[n_tile],
                n_tile_idx=n_tile,
                p_tmem_col_offset=p_archive_col_offset,
                bank_idx=bank_idx,
                p_half_idx=1,
                local_kblock_idx=local_kblock_idx,
                global_kblock_idx=global_kblock_idx,
                pv_kblocks=1,
                p_sfa_col_offset=p_sfa_col_offset,
                a_collector_op=a_collector_op,
            )
    if stream_li_idx > ctm.Int32(0) and stream_li_idx + ctm.Int32(1) == stream_li_total:
        if prims.elect_sync():
            _cross_arrive_smem_p4_cta2_mbar(
                pair_overlap_sync_mbars.subview(
                    SMEM_P4_PAIR_FINAL_PV_ISSUED_MBAR_OFFSET + (bank_idx)
                )
            )
    p4_phase = (
        stream_li_idx
        // ctm.Int32(SMEM_P4_P4_PIPELINE_STAGES)
        % ctm.Int32(SMEM_P4_MBAR_PARITY_PHASES)
    )
    while not prims.mbarrier_try_wait_parity(
        p4_smem_slot0_mbars.subview(bank_idx), p4_phase, time_limit=10000000
    ):
        pass
    if prims.elect_sync():
        prims.tcgen05_commit(
            pv_full_mbar.subview(bank_idx),
            multicast_mask=ctm.Int16(3),
            group=prims.CTAGroup.CTA_2,
        )
        prims.tcgen05_commit(
            p4_smem_slot0_mbars.subview(SMEM_P4_P4_PIPELINE_STAGES + (bank_idx)),
            multicast_mask=ctm.Int16(3),
            group=prims.CTAGroup.CTA_2,
        )
        prims.tcgen05_commit(
            v_smem_mbars.subview(SMEM_P4_V_PIPELINE_STAGES + (v_stage)),
            multicast_mask=ctm.Int16(3),
            group=prims.CTAGroup.CTA_2,
        )


@cute.jit
def _runtime_issue_pv_tile_dispatch(
    sP,
    sV,
    sPSF,
    sVSF,
    tmem_base_col_id: ctm.Int32,
    tmem_base_row_id: ctm.Int32,
    qk_full_mbar,
    pair_overlap_sync_mbars,
    p4_smem_slot0_mbars,
    pv_full_mbar,
    v_smem_mbars,
    stream_li_idx: ctm.Int32,
    stream_li_total: ctm.Int32,
    initial_tile: ctm.Boolean,
) -> None:
    bank_idx = stream_li_idx % ctm.Int32(SMEM_P4_TMEM_BANKS)
    _runtime_issue_pv_tile_slot(
        sP,
        sV,
        sPSF,
        sVSF,
        tmem_base_col_id,
        tmem_base_row_id,
        qk_full_mbar,
        pair_overlap_sync_mbars,
        p4_smem_slot0_mbars,
        pv_full_mbar,
        v_smem_mbars,
        stream_li_idx,
        stream_li_total,
        bank_idx,
        initial_tile,
    )


@cute.jit
def _runtime_refill_qk_tile_state(
    sA,
    sB,
    sSFA,
    sSFB,
    tmem_base_col_id: ctm.Int32,
    tmem_base_row_id: ctm.Int32,
    qk_bulk_ready_mbars,
    qk_full_mbar,
    pair_overlap_sync_mbars,
    tmem_score_slot0_producer,
    qk_smem_consumer,
    qk_next_li_idx: ctm.Int32,
    qk_bulk_slot_idx: ctm.Int32,
    qk_bulk_phase: ctm.Int32,
    qk_consumer_phase: ctm.Int32,
    qk_score_ring_idx: ctm.Int32,
    split_qk15_stage1: ctm.Boolean,
    page_major_ksf: ctm.Constexpr,
) -> None:
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    score_slot0_handle = tmem_score_slot0_producer.acquire_and_advance()
    while not prims.mbarrier_try_wait_parity(
        qk_bulk_ready_mbars.subview(qk_bulk_slot_idx), qk_bulk_phase, time_limit=10000000
    ):
        pass
    _fence_async_shared_cta()
    sK_slot = sB.subview(qk_bulk_slot_idx * ctm.Int32(SMEM_P4_K_COMPACT_SLOT_BYTES // 4))
    sKSF_slot = sSFB.subview(qk_bulk_slot_idx * ctm.Int32(SMEM_P4_KSF_COMPACT_SLOT_BYTES))
    desc_ksf_s2t_root = _make_smem_p4_qk_ksf_stage_desc(
        sKSF_slot, ctm.Int32(0), 0, True, page_major_ksf
    )
    _issue_smem_p4_qk_ksf_stage_from_desc(
        desc_ksf_s2t_root,
        tmem_base_col_id,
        tmem_base_row_id,
        SMEM_P4_SCALE_B_COL_OFFSET,
        0,
        page_major_ksf,
    )
    if not split_qk15_stage1:
        desc_ksf_s2t_stage1 = desc_ksf_s2t_root.advance_start_address(QK_SFB_STAGE_BYTES)
        _issue_smem_p4_qk_ksf_stage_from_desc(
            desc_ksf_s2t_stage1,
            tmem_base_col_id,
            tmem_base_row_id,
            SMEM_P4_SCALE_B_COL_OFFSET,
            1,
            page_major_ksf,
        )
    _complete_smem_p4_qk_ksf_store()
    _issue_smem_p4_qk_stage_burst_from_base(
        sA,
        sK_slot,
        sSFA,
        sKSF_slot,
        tmem_base_col_id,
        tmem_base_row_id,
        score_slot_idx=qk_score_ring_idx,
        qk_kblock_start=0,
        qk_kblocks=SMEM_P4_QK_DATA_STAGE_KBLOCKS,
        acc_col_offset=SMEM_P4_QK_ACC_COL_OFFSET,
        sfa_col_offset_base=SMEM_P4_QK_SFA_COL_OFFSET,
        sfb_col_offset_base=SMEM_P4_SCALE_B_COL_OFFSET,
        use_resident_q_stages=True,
        b_stage_base=ctm.Int32(0),
        fence_after=False,
        b_smem_is_compact_slot_base=True,
    )
    if split_qk15_stage1:
        qk15_stage1_ready_mbar = pair_overlap_sync_mbars.subview(SMEM_P4_QK15_STAGE1_MBAR_OFFSET)
        while not prims.mbarrier_try_wait_parity(qk15_stage1_ready_mbar, 0, time_limit=10000000):
            pass
        _fence_async_shared_cta()
        _stage_smem_p4_qk_ksf_range_from_base(
            sKSF_slot,
            tmem_base_col_id,
            tmem_base_row_id,
            SMEM_P4_SCALE_B_COL_OFFSET,
            ctm.Int32(0),
            1,
            1,
            True,
            page_major_ksf,
        )
    _issue_smem_p4_qk_stage_burst_from_base(
        sA,
        sK_slot,
        sSFA,
        sKSF_slot,
        tmem_base_col_id,
        tmem_base_row_id,
        score_slot_idx=qk_score_ring_idx,
        qk_kblock_start=SMEM_P4_QK_DATA_STAGE_KBLOCKS,
        qk_kblocks=SMEM_P4_QK_DATA_STAGE_KBLOCKS,
        acc_col_offset=SMEM_P4_QK_ACC_COL_OFFSET,
        sfa_col_offset_base=SMEM_P4_QK_SFA_COL_OFFSET,
        sfb_col_offset_base=SMEM_P4_SCALE_B_COL_OFFSET,
        use_resident_q_stages=True,
        b_stage_base=ctm.Int32(0),
        fence_after=False,
        b_smem_is_compact_slot_base=True,
    )
    qk_handle = qk_smem_consumer.wait()
    qk_smem_consumer.advance()
    _fence_async_shared_cta()
    _stage_smem_p4_qk_ksf_range_from_base(
        sKSF_slot,
        tmem_base_col_id,
        tmem_base_row_id,
        SMEM_P4_SCALE_B_COL_OFFSET,
        ctm.Int32(0),
        SMEM_P4_QK_FULL_DATA_STAGES,
        1,
        True,
        page_major_ksf,
    )
    _issue_smem_p4_qk_stage_burst_from_base(
        sA,
        sK_slot,
        sSFA,
        sKSF_slot,
        tmem_base_col_id,
        tmem_base_row_id,
        score_slot_idx=qk_score_ring_idx,
        qk_kblock_start=SMEM_P4_QK_TAIL_STAGE_KBLOCK_START,
        qk_kblocks=QK_TAIL_DATA_STAGE_K_DIM // SMEM_P4_QK_MMA_KBLOCK_DIM,
        acc_col_offset=SMEM_P4_QK_ACC_COL_OFFSET,
        sfa_col_offset_base=SMEM_P4_QK_SFA_COL_OFFSET,
        sfb_col_offset_base=SMEM_P4_SCALE_B_COL_OFFSET,
        use_resident_q_stages=True,
        b_stage_base=ctm.Int32(0),
        fence_after=True,
        b_smem_is_compact_slot_base=True,
    )
    if prims.elect_sync():
        prims.tcgen05_commit(
            qk_full_mbar.subview(qk_score_ring_idx),
            multicast_mask=ctm.Int16(3),
            group=prims.CTAGroup.CTA_2,
        )
    score_slot0_handle.commit()
    qk_handle.release()


@cute.jit
def _runtime_refill_qk_tile_dispatch(
    sA,
    sB,
    sSFA,
    sSFB,
    tmem_base_col_id: ctm.Int32,
    tmem_base_row_id: ctm.Int32,
    qk_bulk_ready_mbars,
    qk_full_mbar,
    pair_overlap_sync_mbars,
    tmem_score_slot0_producer,
    qk_smem_consumer,
    qk_next_li_idx: ctm.Int32,
    qk_next_stage_idx: ctm.Int32,
    qk_bulk_phase: ctm.Int32,
    qk_consumer_phase: ctm.Int32,
    split_qk15_stage1: ctm.Boolean,
    page_major_ksf: ctm.Constexpr,
) -> None:
    _runtime_refill_qk_tile_state(
        sA,
        sB,
        sSFA,
        sSFB,
        tmem_base_col_id,
        tmem_base_row_id,
        qk_bulk_ready_mbars,
        qk_full_mbar,
        pair_overlap_sync_mbars,
        tmem_score_slot0_producer,
        qk_smem_consumer,
        qk_next_li_idx,
        qk_next_stage_idx,
        qk_bulk_phase,
        qk_consumer_phase,
        qk_next_li_idx % ctm.Int32(SMEM_P4_QK_SCORE_RING_SLOTS),
        split_qk15_stage1,
        page_major_ksf,
    )


@cute.jit
def _runtime_t336_prepare_p_source(
    sAtomicRunningRowMax,
    sP,
    sPSF,
    qk_full_mbar,
    pair_rowmax_read_done_mbar,
    direct_p_ready_mbar,
    p4_smem_slot0_producer,
    tmem_score_slot0_consumer,
    warp_idx: ctm.Int32,
    tidx: ctm.Int32,
    p4_tmem_base_col_id: ctm.Int32,
    p4_tmem_base_row_id: ctm.Int32,
    valid_k_for_l: ctm.Int32,
    softmax_scale_log2: ctm.Float32,
    pv_psf_rescale: ctm.Float32,
    running_row_anchor: ctm.Float32,
    stream_li_idx: ctm.Int32,
    has_previous_tile: ctm.Boolean,
    mask_valid_k: ctm.Constexpr,
    use_mixed_imlp: ctm.Constexpr,
) -> tuple:
    """Prepare one score half and return only scalar state across tail dispatch.

    Keeping score vectors, packed P words, and packed group sums inside this
    constexpr helper avoids runtime PHIs with ambiguous packed operand types.
    The caller dispatches this helper with a CTA-uniform tail predicate and
    merges only FP32/Boolean scalar state.
    """
    bank_idx = stream_li_idx % ctm.Int32(SMEM_P4_TMEM_BANKS)
    qk_phase = (
        stream_li_idx
        // ctm.Int32(SMEM_P4_QK_PIPELINE_SLOTS)
        % ctm.Int32(SMEM_P4_MBAR_PARITY_PHASES)
    )
    qk_score_chunk_col_offset = ctm.Int32(SMEM_P4_QK_ACC_COL_OFFSET) + bank_idx * ctm.Int32(
        SMEM_P4_QK_PIPELINE_SLOT_STRIDE
    )
    p_stage_base = bank_idx * ctm.Int32(SMEM_P4_SCORE_SLOTS_PER_KV_TILE)
    score_half_idx = warp_idx >> ctm.Int32(2)
    producer_warp_base = score_half_idx << ctm.Int32(2)
    producer_local_warp_idx = warp_idx & ctm.Int32(3)
    producer_col_band = producer_local_warp_idx >> ctm.Int32(1)
    is_compute_warp = score_half_idx == ctm.Int32(0)
    score_tmem_addresses = _prepare_p4_n256_score_half_tmem_addresses(
        warp_idx,
        p4_tmem_base_col_id,
        p4_tmem_base_row_id,
        qk_score_chunk_col_offset,
    )
    if is_compute_warp:
        tmem_score_slot0_consumer.wait()
    score_values, group_max_values, group_valid_values = _load_p4_n256_score_half_from_tmem(
        sAtomicRunningRowMax,
        qk_full_mbar.subview(bank_idx),
        score_tmem_addresses,
        producer_col_band,
        warp_idx,
        tidx,
        kv_tile_idx=stream_li_idx,
        valid_k=valid_k_for_l,
        mask_valid_k=mask_valid_k,
        qk_phase=qk_phase,
        wait_for_qk_full=True,
        producer_warp_base=producer_warp_base,
        score_half_idx=score_half_idx,
    )
    packed_p4_words, group_p_sums = _prepack_p4_n256_score_half_before_rowmax(
        score_values,
        group_max_values,
        group_valid_values,
        warp_idx,
        softmax_scale_log2,
        kv_tile_idx=stream_li_idx,
        valid_k=valid_k_for_l,
        producer_warp_base=producer_warp_base,
        score_half_idx=score_half_idx,
        mask_valid_k=mask_valid_k,
        use_mixed_imlp=use_mixed_imlp,
    )
    if is_compute_warp:
        p4_smem_slot0_producer.acquire()
    prims.barrier(
        barrier_id=SMEM_P4_DUAL_HALF_BAR_ID,
        number_of_threads=SMEM_P4_DUAL_HALF_BAR_THREADS,
    )
    if cutlass.const_expr(mask_valid_k):
        new_row_max, new_row_valid = _load_p4_n256_atomic_running_rowmax_with_validity(
            sAtomicRunningRowMax,
            warp_idx,
            tidx,
            producer_warp_base=producer_warp_base,
        )
        next_running_row_anchor, lane_rebase, anchor_delta_log2 = (
            _select_smem_p4_lazy_anchor_with_validity(
                new_row_max,
                new_row_valid,
                running_row_anchor,
                softmax_scale_log2,
            )
        )
    else:
        new_row_max = _load_p4_n256_atomic_running_rowmax(
            sAtomicRunningRowMax,
            warp_idx,
            tidx,
            producer_warp_base=producer_warp_base,
        )
        next_running_row_anchor, lane_rebase, anchor_delta_log2 = (
            _select_smem_p4_lazy_anchor_runtime(
                new_row_max,
                running_row_anchor,
                softmax_scale_log2,
                has_previous_tile,
            )
        )
    if is_compute_warp:
        if prims.elect_sync():
            prims.mbarrier_arrive(pair_rowmax_read_done_mbar)
    _store_prepacked_p4_n256_score_half_to_direct_a(
        sP,
        packed_p4_words,
        warp_idx,
        tidx,
        producer_warp_base=producer_warp_base,
        p_stage_base=p_stage_base,
        score_half_idx=score_half_idx,
    )
    if prims.elect_sync():
        prims.mbarrier_arrive(direct_p_ready_mbar)
    owned_psf_word, prequant_row_sum = _finalize_p4_n256_score_half_psf_source(
        sPSF,
        group_max_values,
        group_p_sums,
        next_running_row_anchor,
        warp_idx,
        tidx,
        softmax_scale_log2,
        pv_psf_rescale,
        kv_tile_idx=stream_li_idx,
        valid_k=valid_k_for_l,
        mask_valid_k=mask_valid_k,
        producer_warp_base=producer_warp_base,
        score_half_idx=score_half_idx,
        bank_idx=bank_idx,
        use_mixed_imlp=use_mixed_imlp,
    )
    row_sum = prequant_row_sum
    return (
        new_row_max,
        next_running_row_anchor,
        lane_rebase,
        anchor_delta_log2,
        row_sum,
    )


@cute.jit
def _runtime_t336_producer_tile_v23(
    sAtomicRunningRowMax,
    sP,
    sPSF,
    sScoreSlotRowMax,
    sRowScaleRing,
    raw_v_tma_mbar,
    sRawV,
    sV,
    qk_full_mbar,
    pair_overlap_sync_mbars,
    leader_pair_overlap_sync_mbars,
    leader_pair_overlap_sync_mbars_dsmem,
    p4_smem_slot0_producer,
    p4_smem_slot0_mbar_ptr,
    tmem_score_slot0_consumer,
    tmem_score_slot0_mbar_ptr,
    pv_full_mbar,
    corr_o_acc_tmem,
    warp_idx: ctm.Int32,
    tidx: ctm.Int32,
    p4_tmem_base_col_id: ctm.Int32,
    p4_tmem_base_row_id: ctm.Int32,
    valid_k_for_l: ctm.Int32,
    softmax_scale_log2: ctm.Float32,
    pv_psf_rescale: ctm.Float32,
    running_row_max: ctm.Float32,
    running_row_anchor: ctm.Float32,
    carried_dq1024: ctm.Int32,
    carried_old_scale: ctm.Float32,
    stream_li_idx: ctm.Int32,
    has_previous_tile: ctm.Boolean,
    tail_needs_mask: ctm.Boolean,
    is_final_tile: ctm.Constexpr,
    use_mixed_imlp: ctm.Constexpr,
) -> tuple:
    """Run one producer tile with a scalar-only runtime final-tail dispatch."""
    bank_idx = stream_li_idx % ctm.Int32(SMEM_P4_TMEM_BANKS)
    v_stage = stream_li_idx % ctm.Int32(SMEM_P4_V_PIPELINE_STAGES)
    _transpose_raw_k_staging_to_v_stage(
        raw_v_tma_mbar.subview(v_stage),
        sRawV,
        sV,
        tidx,
        warp_idx,
        stream_li_idx,
        v_stage,
    )
    qk_phase = (
        stream_li_idx
        // ctm.Int32(SMEM_P4_QK_PIPELINE_SLOTS)
        % ctm.Int32(SMEM_P4_MBAR_PARITY_PHASES)
    )
    rowsum_full_mbar = pair_overlap_sync_mbars.subview(
        SMEM_P4_PAIR_P4_READY_MBAR_OFFSET + (bank_idx)
    )
    pair_rowmax_read_done_mbar = pair_overlap_sync_mbars.subview(
        SMEM_P4_PAIR_ROWMAX_READ_DONE_MBAR_OFFSET + (bank_idx)
    )
    direct_p_ready_mbar = pair_overlap_sync_mbars.subview(
        SMEM_P4_PAIR_DIRECT_P_READY_MBAR_OFFSET + (bank_idx)
    )
    leader_pair_p_source_ready_mbar = leader_pair_overlap_sync_mbars.subview(
        SMEM_P4_PAIR_P_SOURCE_READY_MBAR_OFFSET + (bank_idx)
    )
    leader_pair_p_source_ready_dsmem_mbar = leader_pair_overlap_sync_mbars_dsmem.subview(
        SMEM_P4_PAIR_P_SOURCE_READY_MBAR_OFFSET + (bank_idx)
    )
    score_half_idx = warp_idx >> ctm.Int32(2)
    producer_warp_base = score_half_idx << ctm.Int32(2)
    producer_local_warp_idx = warp_idx & ctm.Int32(3)
    producer_row_band = producer_local_warp_idx & ctm.Int32(1)
    producer_col_band = producer_local_warp_idx >> ctm.Int32(1)
    is_compute_warp = score_half_idx == ctm.Int32(0)

    next_running_row_max = running_row_max
    next_running_row_anchor = running_row_anchor
    next_carried_dq1024 = carried_dq1024
    next_carried_old_scale = carried_old_scale
    lane_rebase = ctm.Boolean(False)
    anchor_delta_log2 = ctm.Float32(0.0)
    row_sum = ctm.Float32(0.0)
    if cutlass.const_expr(is_final_tile):
        if tail_needs_mask:
            (
                next_running_row_max,
                next_running_row_anchor,
                lane_rebase,
                anchor_delta_log2,
                row_sum,
            ) = _runtime_t336_prepare_p_source(
                sAtomicRunningRowMax,
                sP,
                sPSF,
                qk_full_mbar,
                pair_rowmax_read_done_mbar,
                direct_p_ready_mbar,
                p4_smem_slot0_producer,
                tmem_score_slot0_consumer,
                warp_idx,
                tidx,
                p4_tmem_base_col_id,
                p4_tmem_base_row_id,
                valid_k_for_l,
                softmax_scale_log2,
                pv_psf_rescale,
                running_row_anchor,
                stream_li_idx,
                has_previous_tile,
                True,
                use_mixed_imlp,
            )
        else:
            (
                next_running_row_max,
                next_running_row_anchor,
                lane_rebase,
                anchor_delta_log2,
                row_sum,
            ) = _runtime_t336_prepare_p_source(
                sAtomicRunningRowMax,
                sP,
                sPSF,
                qk_full_mbar,
                pair_rowmax_read_done_mbar,
                direct_p_ready_mbar,
                p4_smem_slot0_producer,
                tmem_score_slot0_consumer,
                warp_idx,
                tidx,
                p4_tmem_base_col_id,
                p4_tmem_base_row_id,
                valid_k_for_l,
                softmax_scale_log2,
                pv_psf_rescale,
                running_row_anchor,
                stream_li_idx,
                has_previous_tile,
                False,
                use_mixed_imlp,
            )
    else:
        (
            next_running_row_max,
            next_running_row_anchor,
            lane_rebase,
            anchor_delta_log2,
            row_sum,
        ) = _runtime_t336_prepare_p_source(
            sAtomicRunningRowMax,
            sP,
            sPSF,
            qk_full_mbar,
            pair_rowmax_read_done_mbar,
            direct_p_ready_mbar,
            p4_smem_slot0_producer,
            tmem_score_slot0_consumer,
            warp_idx,
            tidx,
            p4_tmem_base_col_id,
            p4_tmem_base_row_id,
            valid_k_for_l,
            softmax_scale_log2,
            pv_psf_rescale,
            running_row_anchor,
            stream_li_idx,
            has_previous_tile,
            False,
            use_mixed_imlp,
        )

    current_old_scale = ctm.Float32(1.0)
    warp_rebase = cute.arch.vote_any_sync(lane_rebase)
    if has_previous_tile:
        common_source_ready_pred = prims.elect_sync() & ~warp_rebase
        if common_source_ready_pred:
            prims.mbarrier_arrive(
                leader_pair_p_source_ready_dsmem_mbar,
                count=1,
                scope=prims.MemScope.CLUSTER,
                relaxed=True,
            )
        if warp_rebase:
            pv_done_li_idx = stream_li_idx - ctm.Int32(1)
            pv_done_slot = pv_done_li_idx % ctm.Int32(SMEM_P4_QK_PIPELINE_SLOTS)
            pv_done_phase = (
                pv_done_li_idx
                // ctm.Int32(SMEM_P4_QK_PIPELINE_SLOTS)
                % ctm.Int32(SMEM_P4_MBAR_PARITY_PHASES)
            )
            while not prims.mbarrier_try_wait_parity(
                pv_full_mbar.subview(pv_done_slot), pv_done_phase, time_limit=10000000
            ):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            current_old_scale = _compute_smem_p4_lazy_rebase_scale(lane_rebase, anchor_delta_log2)
            _rescale_smem_p4_resident_o_in_tmem(corr_o_acc_tmem, current_old_scale)
            if prims.elect_sync():
                _arrive_smem_p4_mapped_leader_mbar(leader_pair_p_source_ready_mbar)
    elif prims.elect_sync():
        _arrive_smem_p4_mapped_leader_mbar(leader_pair_p_source_ready_mbar)
    if is_compute_warp:
        if cutlass.const_expr(not is_final_tile):
            while not prims.mbarrier_try_wait_parity(
                direct_p_ready_mbar, qk_phase, time_limit=10000000
            ):
                pass
            cute.arch.mbarrier_arrive(
                p4_smem_slot0_mbar_ptr + p4_smem_slot0_producer.current_handle().index,
                cutlass.Int32(0),
            )
            p4_smem_slot0_producer.advance()
            cute.arch.mbarrier_arrive(
                tmem_score_slot0_mbar_ptr
                + SMEM_P4_TMEM_SCORE_PIPELINE_STAGES
                + tmem_score_slot0_consumer.current_handle().index,
                cutlass.Int32(0),
            )
            tmem_score_slot0_consumer.advance()
    if is_compute_warp:
        prims.barrier(
            barrier_id=SMEM_P4_P4_SLOT0_BAR_ID,
            number_of_threads=SMEM_P4_SCORE_SLOT_PRODUCER_THREADS,
        )
    else:
        prims.barrier(
            barrier_id=SMEM_P4_P4_SLOT1_BAR_ID,
            number_of_threads=SMEM_P4_CORRECTION_THREADS,
        )
    rowsum_empty_mbar = pair_overlap_sync_mbars.subview(
        SMEM_P4_PAIR_HALF1_ROWSUM_READY_MBAR_OFFSET + (bank_idx)
    )
    if stream_li_idx >= ctm.Int32(SMEM_P4_TMEM_BANKS):
        previous_rowsum_phase = (
            (stream_li_idx - ctm.Int32(SMEM_P4_TMEM_BANKS))
            // ctm.Int32(SMEM_P4_TMEM_BANKS)
            % ctm.Int32(SMEM_P4_MBAR_PARITY_PHASES)
        )
        while not prims.mbarrier_try_wait_parity(
            rowsum_empty_mbar, previous_rowsum_phase, time_limit=10000000
        ):
            pass
    _store_smem_p4_n256_rowsum_part(
        sScoreSlotRowMax,
        warp_idx,
        tidx,
        producer_warp_base=producer_warp_base,
        bank_idx=bank_idx,
        score_half_idx=score_half_idx,
        value=row_sum,
    )
    if is_compute_warp & (producer_col_band == ctm.Int32(0)):
        row_state_local_row = producer_row_band * ctm.Int32(32) + (tidx & ctm.Int32(31))
        sRowScaleRing[bank_idx * ctm.Int32(SMEM_P4_BMM1_M) + row_state_local_row] = (
            current_old_scale
        )
    if prims.elect_sync():
        prims.mbarrier_arrive(rowsum_full_mbar)
    if not is_compute_warp:
        while not prims.mbarrier_try_wait_parity(
            pair_rowmax_read_done_mbar, qk_phase, time_limit=10000000
        ):
            pass
    return (
        next_running_row_max,
        next_running_row_anchor,
        next_carried_dq1024,
        next_carried_old_scale,
    )


@cute.jit
def _stage_smem_p4_page_id_plan(
    mPageTable_pl: cute.Tensor,
    mPageIndptr_s: cute.Tensor,
    sPageIdPlan,
    tidx: ctm.Int32,
    page_batch: ctm.Int32,
    planned_page_count: ctm.Int32,
) -> None:
    page_begin = cute.arch.make_warp_uniform(ctm.Int32(mPageIndptr_s[page_batch]))
    page_count = cute.arch.make_warp_uniform(
        ctm.Int32(mPageIndptr_s[page_batch + ctm.Int32(1)]) - page_begin
    )
    logical_page_idx = tidx
    while logical_page_idx < planned_page_count:
        physical_page = _lookup_physical_page(
            mPageTable_pl, logical_page_idx, page_batch, page_begin, page_count
        )
        (sPageIdPlan.data_ptr() + logical_page_idx).store(physical_page)
        logical_page_idx += ctm.Int32(THREADS_PER_CTA)


@cute.jit
def _run_mla_decode_body(
    mPageTable_pl: cute.Tensor,
    mPageIndptr_s: cute.Tensor,
    mValidK_l: cute.Tensor,
    mHelixKvBounds_l: cute.Tensor,
    mQGlobalScale: cute.Tensor,
    mKvGlobalScale: cute.Tensor,
    tma_q_ptr,
    tma_k_page_ptr,
    tma_k_v_raw_ptr,
    tma_q_tail_ptr,
    tma_k_tail_page_ptr,
    tma_v_tile_ptr,
    tma_qsf_ptr,
    tma_qsf_tail_ptr,
    tma_ksf_ptr,
    tma_ksf_stage_ptr,
    tma_ksf_tail_stage_ptr,
    tma_vsf_ptr,
    tma_vsf_stage_ptr,
    tma_v_pair_ptr,
    tma_sfb_ptr,
    mC_mnl: cute.Tensor,
    mAccum_mnl: cute.Tensor,
    mRowMax_ml: cute.Tensor,
    mRowSum_ml: cute.Tensor,
    problem_size: tuple,
    runtime_m: ctm.Int32,
    v_page_offset: ctm.Int32,
    softmax_scale_log2: ctm.Float32,
    pv_output_scale: ctm.Float32,
    page_size: ctm.Constexpr,
    query_len_per_seq: ctm.Constexpr,
    use_helix_kv_bounds: ctm.Constexpr,
    use_smem_page_plan: ctm.Constexpr,
    use_mixed_imlp: ctm.Constexpr = False,
    use_consecutive_page_pair: ctm.Constexpr = False,
    use_ksf_gather4: ctm.Constexpr = False,
    write_softmax_stats: ctm.Constexpr = False,
    store_split: ctm.Constexpr = False,
) -> None:
    pv_psf_rescale = ctm.Float32(FP4_MLA_P_GLOBAL_SCALE) * pv_output_scale
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    n, k = problem_size
    m = runtime_m
    page_batch = ctm.Int32(0)
    valid_k_for_l = ctm.Int32(0)
    page_batch = cute.arch.make_warp_uniform(bidz // ctm.Int32(query_len_per_seq))
    if cutlass.const_expr(use_helix_kv_bounds):
        valid_k_for_l = mHelixKvBounds_l[bidz]
    else:
        query_offset = bidz - page_batch * ctm.Int32(query_len_per_seq)
        valid_k_for_l = ctm.max(
            mValidK_l[page_batch] - (ctm.Int32(query_len_per_seq - 1) - query_offset),
            ctm.Int32(0),
        )
    valid_k_for_l = cute.arch.make_warp_uniform(valid_k_for_l)
    csr_page_begin = ctm.Int32(0)
    csr_page_count = ctm.Int32(0)
    csr_page_begin = cute.arch.make_warp_uniform(ctm.Int32(mPageIndptr_s[page_batch]))
    csr_page_count = cute.arch.make_warp_uniform(
        ctm.Int32(mPageIndptr_s[page_batch + ctm.Int32(1)]) - csr_page_begin
    )
    raw_kv_tiles = (valid_k_for_l + ctm.Int32(KV_TILE - 1)) // ctm.Int32(KV_TILE)
    kv_tiles = ctm.max(raw_kv_tiles, ctm.Int32(1))
    planned_page_count = kv_tiles * ctm.Int32(SMEM_P4_PAGES_PER_KV_TILE)
    if cutlass.const_expr(page_size != TRTLLM_PAGE_SIZE):
        raise ValueError(
            f"FP4 MLA decode requires TensorRT-LLM page_size={TRTLLM_PAGE_SIZE}, got {page_size}"
        )
    physical_sfb_blocks_per_l = k // ctm.Int32(SMEM_P4_SFB_ATOM_ROWS)
    num_tmem_alloc_cols: ctm.Constexpr = NUM_TMEM_ALLOC_COLS
    q_tma_bytes: ctm.Constexpr = SMEM_P4_Q_ONLY_STAGE_BYTES * CLUSTER_SHAPE_MNK[0]
    qk_tma_bytes: ctm.Constexpr = SMEM_P4_QK_KONLY_TILE_BYTES * CLUSTER_SHAPE_MNK[0]
    v_tma_bytes: ctm.Constexpr = SMEM_P4_V_SFB_TMA_STAGE_BYTES * CLUSTER_SHAPE_MNK[0]
    cta_rank = cute.arch.block_idx_in_cluster()
    is_leader_cta = cta_rank == 0
    qk_full_mbar = ctm.Array(ctm.Int64, SMEM_P4_QK_COMPLETION_MBARS, space=ctm.AddressSpace.smem)
    pv_full_mbar = ctm.Array(
        ctm.Int64, SMEM_P4_QK_PIPELINE_SLOTS, space=ctm.AddressSpace.smem, alignment=8
    )
    tmem_dealloc_mbar = ctm.Array(ctm.Int64, 1, space=ctm.AddressSpace.smem)
    tmem_holding_buf = ctm.Array(ctm.Int32, 1, space=ctm.AddressSpace.smem)
    q_smem_mbars = ctm.Array(
        ctm.Int64,
        NUM_QK_Q_PIPELINE_STAGE * 2,
        space=ctm.AddressSpace.smem,
        alignment=8,
    )
    qk_smem_mbars = ctm.Array(
        ctm.Int64,
        SMEM_P4_QK_BARRIER_STAGES * 2,
        space=ctm.AddressSpace.smem,
        alignment=8,
    )
    v_smem_mbars = ctm.Array(
        ctm.Int64,
        SMEM_P4_V_PIPELINE_STAGES * 2,
        space=ctm.AddressSpace.smem,
        alignment=8,
    )
    raw_v_tma_mbar = ctm.Array(
        ctm.Int64,
        SMEM_P4_V_PIPELINE_STAGES,
        space=ctm.AddressSpace.smem,
        alignment=8,
    )
    tmem_score_slot0_mbars = ctm.Array(
        ctm.Int64,
        SMEM_P4_TMEM_SCORE_PIPELINE_STAGES * 2,
        space=ctm.AddressSpace.smem,
        alignment=8,
    )
    p4_smem_slot0_mbars = ctm.Array(
        ctm.Int64,
        SMEM_P4_P4_CONTROL_MBAR_STAGES * 2,
        space=ctm.AddressSpace.smem,
        alignment=8,
    )
    pair_overlap_sync_mbars = ctm.Array(
        ctm.Int64,
        SMEM_P4_PAIR_OVERLAP_MBAR_COUNT,
        space=ctm.AddressSpace.smem,
        alignment=8,
    )
    qsf_owner_ready_mbar = ctm.Array(ctm.Int64, 1, space=ctm.AddressSpace.smem, alignment=8)
    qsf_owner_done_mbar = ctm.Array(ctm.Int64, 1, space=ctm.AddressSpace.smem, alignment=8)
    qk_bulk_ready_mbars = ctm.Array(
        ctm.Int64,
        SMEM_P4_QK_BULK_READY_MBARS,
        space=ctm.AddressSpace.smem,
        alignment=8,
    )
    qk_dual_arm_mbars = ctm.Array(
        ctm.Int64,
        SMEM_P4_QK_DUAL_ARM_MBARS,
        space=ctm.AddressSpace.smem,
        alignment=8,
    )
    sA = ctm.Array(
        ctm.Int32,
        SMEM_P4_Q_COMPACT_TILE_BYTES // 4,
        space=ctm.AddressSpace.smem,
        alignment=128,
    )
    sB = ctm.Array(
        ctm.Int32,
        SMEM_P4_K_COMPACT_RING_BYTES // 4,
        space=ctm.AddressSpace.smem,
        alignment=128,
    )
    sSFA = ctm.Array(
        ctm.Uint8,
        SMEM_P4_QSF_COMPACT_TILE_BYTES,
        space=ctm.AddressSpace.smem,
        alignment=128,
    )
    sSFB = ctm.Array(
        ctm.Uint8,
        SMEM_P4_KSF_COMPACT_RING_BYTES,
        space=ctm.AddressSpace.smem,
        alignment=128,
    )
    sP = ctm.Array(ctm.Int32, SMEM_P4_P4_SMEM_WORDS, space=ctm.AddressSpace.smem, alignment=128)
    sV = ctm.Array(
        ctm.Int32,
        SMEM_P4_V_PIPELINE_STAGES * SMEM_P4_V_DATA_STAGE_BYTES // 4,
        space=ctm.AddressSpace.smem,
        alignment=128,
    )
    sRawV = ctm.Array(
        ctm.Uint8,
        SMEM_P4_RAW_V_STAGE_BYTES,
        space=ctm.AddressSpace.smem,
        alignment=128,
    )
    sPSF = ctm.Array(
        ctm.Int32,
        SMEM_P4_P4_OWNER_SMEM_SF_MAILBOX_WORDS,
        space=ctm.AddressSpace.smem,
        alignment=128,
    )
    sVSF = ctm.Array(
        ctm.Uint8,
        SMEM_P4_V_PIPELINE_STAGES * SMEM_P4_V_SFB_TMA_STAGE_BYTES,
        space=ctm.AddressSpace.smem,
        alignment=128,
    )
    sScoreSlotRowMax = ctm.Array(
        ctm.Float32,
        SMEM_P4_STAGE_ROW_STATE_SMEM_FLOATS,
        space=ctm.AddressSpace.smem,
        alignment=128,
    )
    sRowScaleRing = ctm.Array(
        ctm.Float32,
        SMEM_P4_ROW_SCALE_RING_FLOATS,
        space=ctm.AddressSpace.smem,
        alignment=128,
    )
    sFinalAnchorRowSum = ctm.Array(
        ctm.Float32,
        SMEM_P4_FINAL_ANCHOR_ROWSUM_FLOATS,
        space=ctm.AddressSpace.smem,
        alignment=128,
    )
    sAtomicRunningRowMax = ctm.Array(
        ctm.Uint32,
        SMEM_P4_ATOMIC_RUNNING_ROWMAX_SMEM_ROWS,
        space=ctm.AddressSpace.smem,
        alignment=128,
    )
    sPageIdPlan = ctm.Array(
        ctm.Int32,
        SMEM_P4_PAGE_ID_PLAN_INTS,
        space=ctm.AddressSpace.smem,
        alignment=128,
    )
    sRuntimeScalePair = ctm.Array(
        ctm.Float32,
        SMEM_P4_RUNTIME_SCALE_PAIR_FLOATS,
        space=ctm.AddressSpace.smem,
        alignment=8,
    )
    cta_layout_vmnk = cute.make_layout((CLUSTER_SHAPE_MNK[0], 1, 1, 1))
    tma_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    umma_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    p4_stage_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        SMEM_P4_SCORE_SLOT_PRODUCER_THREADS * CLUSTER_SHAPE_MNK[0],
    )
    q_smem_mbar_ptr = cute.make_ptr(
        ctm.Int64, q_smem_mbars.data_ptr().toint(), cute.AddressSpace.smem
    )
    qk_smem_mbar_ptr = cute.make_ptr(
        ctm.Int64, qk_smem_mbars.data_ptr().toint(), cute.AddressSpace.smem
    )
    v_smem_mbar_ptr = cute.make_ptr(
        ctm.Int64, v_smem_mbars.data_ptr().toint(), cute.AddressSpace.smem
    )
    tmem_score_slot0_mbar_ptr = cute.make_ptr(
        ctm.Int64, tmem_score_slot0_mbars.data_ptr().toint(), cute.AddressSpace.smem
    )
    p4_smem_slot0_mbar_ptr = cute.make_ptr(
        ctm.Int64, p4_smem_slot0_mbars.data_ptr().toint(), cute.AddressSpace.smem
    )
    qk_bulk_ready_mbar_ptr = cute.make_ptr(
        ctm.Int64, qk_bulk_ready_mbars.data_ptr().toint(), cute.AddressSpace.smem
    )
    q_smem_producer, q_smem_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=NUM_QK_Q_PIPELINE_STAGE,
        producer_group=tma_group,
        consumer_group=umma_group,
        tx_count=q_tma_bytes,
        barrier_storage=q_smem_mbar_ptr,
        cta_layout_vmnk=cta_layout_vmnk,
        defer_sync=True,
        name="fp4_q_smem",
    ).make_participants()
    qk_smem_producer, qk_smem_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=SMEM_P4_QK_BARRIER_STAGES,
        producer_group=tma_group,
        consumer_group=umma_group,
        tx_count=qk_tma_bytes,
        barrier_storage=qk_smem_mbar_ptr,
        cta_layout_vmnk=cta_layout_vmnk,
        defer_sync=True,
        name="fp4_qk_smem",
    ).make_participants()
    v_smem_producer, v_smem_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=SMEM_P4_V_PIPELINE_STAGES,
        producer_group=tma_group,
        consumer_group=umma_group,
        tx_count=v_tma_bytes,
        barrier_storage=v_smem_mbar_ptr,
        cta_layout_vmnk=cta_layout_vmnk,
        defer_sync=True,
        name="fp4_v_smem",
    ).make_participants()
    tmem_score_slot0_producer, tmem_score_slot0_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=SMEM_P4_TMEM_SCORE_PIPELINE_STAGES,
        producer_group=umma_group,
        consumer_group=p4_stage_group,
        barrier_storage=tmem_score_slot0_mbar_ptr,
        cta_layout_vmnk=cta_layout_vmnk,
        defer_sync=True,
        name="fp4_tmem_score_slot0",
    ).make_participants()
    p4_smem_slot0_producer, _ = pipeline.PipelineAsyncUmma.create(
        num_stages=SMEM_P4_P4_PIPELINE_STAGES,
        producer_group=p4_stage_group,
        consumer_group=umma_group,
        barrier_storage=p4_smem_slot0_mbar_ptr,
        cta_layout_vmnk=cta_layout_vmnk,
        defer_sync=True,
        name="fp4_p4_smem_slot0",
    ).make_participants()
    if warp_idx == 0:
        if prims.elect_sync():
            for qk_completion_mbar_idx in ctm.range_constexpr(SMEM_P4_QK_COMPLETION_MBARS):
                prims.mbarrier_init(qk_full_mbar.subview(qk_completion_mbar_idx), 1)
            for pv_full_mbar_idx in ctm.range_constexpr(SMEM_P4_QK_PIPELINE_SLOTS):
                prims.mbarrier_init(pv_full_mbar.subview(pv_full_mbar_idx), 1)
            for pair_slot_idx in ctm.range_constexpr(SMEM_P4_QK_PIPELINE_SLOTS):
                prims.mbarrier_init(
                    pair_overlap_sync_mbars.subview(
                        SMEM_P4_PAIR_P4_READY_MBAR_OFFSET + (pair_slot_idx)
                    ),
                    SMEM_P4_TOTAL_PRODUCER_WARPS,
                )
                prims.mbarrier_init(
                    pair_overlap_sync_mbars.subview(
                        SMEM_P4_PAIR_HALF1_ROWSUM_READY_MBAR_OFFSET + (pair_slot_idx)
                    ),
                    1,
                )
                prims.mbarrier_init(
                    pair_overlap_sync_mbars.subview(
                        SMEM_P4_PAIR_FINAL_PV_ISSUED_MBAR_OFFSET + (pair_slot_idx)
                    ),
                    1,
                )
                prims.mbarrier_init(
                    pair_overlap_sync_mbars.subview(
                        SMEM_P4_PAIR_ROWMAX_READ_DONE_MBAR_OFFSET + (pair_slot_idx)
                    ),
                    SMEM_P4_COMPUTE_WARPS,
                )
                prims.mbarrier_init(
                    pair_overlap_sync_mbars.subview(
                        SMEM_P4_PAIR_DIRECT_P_READY_MBAR_OFFSET + (pair_slot_idx)
                    ),
                    SMEM_P4_TOTAL_PRODUCER_WARPS,
                )
                prims.mbarrier_init(
                    pair_overlap_sync_mbars.subview(
                        SMEM_P4_PAIR_P_SOURCE_READY_MBAR_OFFSET + (pair_slot_idx)
                    ),
                    SMEM_P4_TOTAL_PRODUCER_WARPS * CLUSTER_SHAPE_MNK[0],
                )
            prims.mbarrier_init(qsf_owner_ready_mbar, 1)
            prims.mbarrier_init(qsf_owner_done_mbar, CLUSTER_SHAPE_MNK[0])
            for qk_bulk_slot_idx in ctm.range_constexpr(SMEM_P4_QK_BULK_READY_MBARS):
                prims.mbarrier_init(qk_bulk_ready_mbars.subview(qk_bulk_slot_idx), 1)
            for dual_arm_idx in ctm.range_constexpr(SMEM_P4_QK_DUAL_ARM_MBARS):
                prims.mbarrier_init(qk_dual_arm_mbars.subview(dual_arm_idx), 1)
            for raw_v_stage in ctm.range_constexpr(SMEM_P4_V_PIPELINE_STAGES):
                prims.mbarrier_init(raw_v_tma_mbar.subview(raw_v_stage), 1)
            prims.mbarrier_init(pair_overlap_sync_mbars.subview(SMEM_P4_QK15_STAGE1_MBAR_OFFSET), 1)
            prims.mbarrier_init(tmem_dealloc_mbar, 32)
    prims.fence_mbarrier_init()
    if warp_idx == ctm.Int32(TMA_QK_WARP_ID):
        _stage_smem_p4_runtime_scale_pair(
            mQGlobalScale, mKvGlobalScale, sRuntimeScalePair, softmax_scale_log2
        )
    if cutlass.const_expr(use_smem_page_plan):
        _stage_smem_p4_page_id_plan(
            mPageTable_pl, mPageIndptr_s, sPageIdPlan, tidx, page_batch, planned_page_count
        )
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    runtime_scale_pair = sRuntimeScalePair.data_ptr().load(count=2, alignment=8)
    softmax_scale_log2 = runtime_scale_pair[0]
    pv_psf_rescale = runtime_scale_pair[1]
    # Both exact and mixed-IMLP modes feed the amplified PV-only PSF to MMA.
    # Keep the denominator in its original coordinate and cancel the PV
    # amplification exactly once at the final output.
    pv_output_scale = ctm.Float32(1.0 / FP4_MLA_P_GLOBAL_SCALE)
    if warp_idx >= ctm.Int32(MMA_WARP_ID):
        prims.setmaxregister(SMEM_P4_AUX_REGISTER_BUDGET, prims.SetMaxRegisterAction.DECREASE)
        if warp_idx == MMA_WARP_ID:
            prims.tcgen05_alloc(
                tmem_holding_buf,
                num_tmem_alloc_cols,
                is_exclusive=SMEM_P4_TMEM_ALLOC_EXCLUSIVE,
                group=prims.CTAGroup.CTA_2,
            )
            prims.barrier(barrier_id=TMEM_BAR_ID, number_of_threads=TMEM_BAR_THREADS)
        if warp_idx == TMA_QK_WARP_ID:
            for q_stage_idx in ctm.range_constexpr(NUM_QK_Q_STAGE):
                q_stage_kblock_idx: ctm.Constexpr = q_stage_idx * SMEM_P4_QK_DATA_STAGE_KBLOCKS
                if cutlass.const_expr(q_stage_kblock_idx >= SMEM_P4_QK_TAIL_STAGE_KBLOCK_START):
                    q_handle = q_smem_producer.acquire_and_advance(
                        expected_tx=SMEM_P4_Q_ONLY_TAIL_STAGE_BYTES * CLUSTER_SHAPE_MNK[0]
                    )
                else:
                    q_handle = q_smem_producer.acquire_and_advance()
                q_tma_mbar = ctm.Array(q_handle.barrier, shape=1)
                _load_qk_qonly_kblock_stage(
                    tma_q_ptr,
                    tma_q_tail_ptr,
                    tma_qsf_ptr,
                    tma_qsf_tail_ptr,
                    q_tma_mbar,
                    sA,
                    sSFA,
                    tidx,
                    bidx,
                    bidz,
                    cta_rank,
                    qk_kblock_idx=q_stage_kblock_idx,
                    stage=q_stage_idx,
                    q_tma_phase=q_stage_idx,
                    manage_mbarrier=False,
                )
            qk0_handle = qk_smem_producer.acquire_and_advance()
            qk0_tma_mbar = ctm.Array(qk0_handle.barrier, shape=1)
            if is_leader_cta:
                if cutlass.const_expr(use_smem_page_plan):
                    qk0_page0, qk0_page1 = _load_staged_page_native_tile_pair(
                        sPageIdPlan, ctm.Int32(0)
                    )
                else:
                    qk0_page0, qk0_page1 = _load_global_page_native_tile_pair(
                        mPageTable_pl, ctm.Int32(0), csr_page_begin, csr_page_count
                    )
                _load_runtime_t336_qk_tile(
                    mPageTable_pl,
                    tma_k_page_ptr,
                    tma_k_tail_page_ptr,
                    tma_ksf_ptr,
                    tma_ksf_stage_ptr,
                    tma_ksf_tail_stage_ptr,
                    qk0_tma_mbar,
                    qk_bulk_ready_mbar_ptr,
                    pair_overlap_sync_mbars,
                    sB,
                    sSFB,
                    tidx,
                    bidz,
                    physical_sfb_blocks_per_l,
                    csr_page_begin,
                    csr_page_count,
                    qk0_page0,
                    qk0_page1,
                    ctm.Int32(0),
                    ctm.Int32(0),
                    page_size,
                    initial_tile=True,
                    use_consecutive_page_pair=use_consecutive_page_pair,
                    use_ksf_gather4=use_ksf_gather4,
                )
            qk_prefix_end = min(kv_tiles, ctm.Int32(3))
            for qk_prefix_li_idx in cutlass.range(1, qk_prefix_end, 1, unroll=1):
                qk_prefix_handle = qk_smem_producer.acquire_and_advance(
                    expected_tx=SMEM_P4_QK_KONLY_TAIL_STAGE_BYTES * CLUSTER_SHAPE_MNK[0]
                )
                qk_prefix_tma_mbar = ctm.Array(qk_prefix_handle.barrier, shape=1)
                if is_leader_cta:
                    if cutlass.const_expr(use_smem_page_plan):
                        qk_prefix_page0, qk_prefix_page1 = _load_staged_page_native_tile_pair(
                            sPageIdPlan, qk_prefix_li_idx
                        )
                    else:
                        qk_prefix_page0, qk_prefix_page1 = _load_global_page_native_tile_pair(
                            mPageTable_pl, qk_prefix_li_idx, csr_page_begin, csr_page_count
                        )
                    _load_runtime_t336_qk_tile(
                        mPageTable_pl,
                        tma_k_page_ptr,
                        tma_k_tail_page_ptr,
                        tma_ksf_ptr,
                        tma_ksf_stage_ptr,
                        tma_ksf_tail_stage_ptr,
                        qk_prefix_tma_mbar,
                        qk_bulk_ready_mbar_ptr,
                        pair_overlap_sync_mbars,
                        sB,
                        sSFB,
                        tidx,
                        bidz,
                        physical_sfb_blocks_per_l,
                        csr_page_begin,
                        csr_page_count,
                        qk_prefix_page0,
                        qk_prefix_page1,
                        qk_prefix_li_idx,
                        qk_prefix_li_idx,
                        page_size,
                        split_prefix_owner=True,
                        use_consecutive_page_pair=use_consecutive_page_pair,
                        use_ksf_gather4=use_ksf_gather4,
                    )
            qk_steady_end = kv_tiles
            qk15_final_enabled = kv_tiles == ctm.Int32(16)
            if qk15_final_enabled:
                qk_steady_end = ctm.Int32(15)
            for kv_tile_idx in cutlass.range(3, qk_steady_end, 1, unroll=1):
                qk_handle = qk_smem_producer.acquire_and_advance(
                    expected_tx=SMEM_P4_QK_KONLY_TAIL_STAGE_BYTES * CLUSTER_SHAPE_MNK[0]
                )
                qk_tma_mbar = ctm.Array(qk_handle.barrier, shape=1)
                if cutlass.const_expr(use_smem_page_plan):
                    qk_steady_page0, qk_steady_page1 = _load_staged_page_native_tile_pair(
                        sPageIdPlan, kv_tile_idx
                    )
                else:
                    qk_steady_page0, qk_steady_page1 = _load_global_page_native_tile_pair(
                        mPageTable_pl, kv_tile_idx, csr_page_begin, csr_page_count
                    )
                _load_runtime_t336_qk_tile_dual_steady(
                    mPageTable_pl,
                    tma_k_page_ptr,
                    tma_k_tail_page_ptr,
                    tma_ksf_ptr,
                    tma_ksf_stage_ptr,
                    tma_ksf_tail_stage_ptr,
                    qk_tma_mbar,
                    qk_bulk_ready_mbar_ptr,
                    qk_dual_arm_mbars,
                    sB,
                    sSFB,
                    tidx,
                    bidz,
                    cta_rank,
                    physical_sfb_blocks_per_l,
                    csr_page_begin,
                    csr_page_count,
                    qk_steady_page0,
                    qk_steady_page1,
                    kv_tile_idx,
                    kv_tile_idx % ctm.Int32(SMEM_P4_QK_SMEM_PIPELINE_SLOTS),
                    page_size,
                    use_consecutive_page_pair=use_consecutive_page_pair,
                    use_ksf_gather4=use_ksf_gather4,
                )
            if qk15_final_enabled:
                qk15_handle = qk_smem_producer.acquire_and_advance(
                    expected_tx=SMEM_P4_QK_KONLY_TAIL_STAGE_BYTES * CLUSTER_SHAPE_MNK[0]
                )
                qk15_tma_mbar = ctm.Array(qk15_handle.barrier, shape=1)
                if is_leader_cta:
                    if cutlass.const_expr(use_smem_page_plan):
                        qk15_page0, qk15_page1 = _load_staged_page_native_tile_pair(
                            sPageIdPlan, ctm.Int32(15)
                        )
                    else:
                        qk15_page0, qk15_page1 = _load_global_page_native_tile_pair(
                            mPageTable_pl, ctm.Int32(15), csr_page_begin, csr_page_count
                        )
                    _load_runtime_t336_qk_tile(
                        mPageTable_pl,
                        tma_k_page_ptr,
                        tma_k_tail_page_ptr,
                        tma_ksf_ptr,
                        tma_ksf_stage_ptr,
                        tma_ksf_tail_stage_ptr,
                        qk15_tma_mbar,
                        qk_bulk_ready_mbar_ptr,
                        pair_overlap_sync_mbars,
                        sB,
                        sSFB,
                        tidx,
                        bidz,
                        physical_sfb_blocks_per_l,
                        csr_page_begin,
                        csr_page_count,
                        qk15_page0,
                        qk15_page1,
                        ctm.Int32(15),
                        ctm.Int32(0),
                        page_size,
                        qk15_final=True,
                        use_consecutive_page_pair=use_consecutive_page_pair,
                        use_ksf_gather4=use_ksf_gather4,
                    )
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            q_smem_producer.tail()
        if warp_idx == TMA_V_WARP_ID:
            for kv_tile_idx in cutlass.range(0, kv_tiles, 1, unroll=1):
                v_handle = v_smem_producer.acquire_and_advance()
                v_tma_mbar = ctm.Array(v_handle.barrier, shape=1)
                v_stage = kv_tile_idx % ctm.Int32(SMEM_P4_V_PIPELINE_STAGES)
                raw_v_stage_mbar = raw_v_tma_mbar.subview(v_stage)
                v_physical_page0 = ctm.Int32(0)
                v_physical_page1 = ctm.Int32(0)
                if cutlass.const_expr(not use_smem_page_plan):
                    v_physical_page0, v_physical_page1 = _load_global_page_native_tile_pair(
                        mPageTable_pl, kv_tile_idx, csr_page_begin, csr_page_count
                    )
                _issue_raw_k_pages_to_v_staging(
                    tma_k_v_raw_ptr,
                    raw_v_stage_mbar,
                    sRawV,
                    sPageIdPlan,
                    cta_rank,
                    kv_tile_idx,
                    v_physical_page0,
                    v_physical_page1,
                    use_smem_page_plan,
                )
                prims.barrier(
                    barrier_id=SMEM_P4_RAW_V_READY_BAR_ID,
                    number_of_threads=SMEM_P4_RAW_V_READY_BAR_THREADS,
                )
                prims.fence_proxy(
                    kind=prims.Proxy.ASYNC_SHARED,
                    space=prims.SharedSpace.shared_cta,
                )
                _load_vsf_tile_stage_only(
                    sPageIdPlan,
                    tma_vsf_ptr,
                    tma_vsf_stage_ptr,
                    v_tma_mbar,
                    sVSF,
                    cta_rank,
                    kv_tile_idx,
                    v_physical_page0,
                    v_physical_page1,
                    v_page_offset,
                    v_stage,
                    use_smem_page_plan,
                    use_consecutive_page_pair=use_consecutive_page_pair,
                )
            v_smem_producer.tail()
        if warp_idx == ROWMETA_WARP_ID:
            if is_leader_cta:
                qk_prefix_end = min(kv_tiles, ctm.Int32(3))
                for qk_prefix_li_idx in cutlass.range(1, qk_prefix_end, 1, unroll=1):
                    if cutlass.const_expr(use_smem_page_plan):
                        rowmeta_page0, rowmeta_page1 = _load_staged_page_native_tile_pair(
                            sPageIdPlan, qk_prefix_li_idx
                        )
                    else:
                        rowmeta_page0, rowmeta_page1 = _load_global_page_native_tile_pair(
                            mPageTable_pl, qk_prefix_li_idx, csr_page_begin, csr_page_count
                        )
                    _issue_runtime_t336_qk_prefix_rank1(
                        mPageTable_pl,
                        tma_k_page_ptr,
                        tma_k_tail_page_ptr,
                        tma_ksf_ptr,
                        tma_ksf_stage_ptr,
                        tma_ksf_tail_stage_ptr,
                        qk_bulk_ready_mbar_ptr,
                        sB,
                        sSFB,
                        tidx,
                        bidz,
                        physical_sfb_blocks_per_l,
                        csr_page_begin,
                        csr_page_count,
                        rowmeta_page0,
                        rowmeta_page1,
                        qk_prefix_li_idx,
                        qk_prefix_li_idx,
                        page_size,
                    )
            rowmeta_lane = tidx & ctm.Int32(31)
            running_anchor_row_sum0 = ctm.Float32(0.0)
            running_anchor_row_sum1 = ctm.Float32(0.0)
            for rowmeta_li_idx in cutlass.range(0, kv_tiles, 1, unroll=1):
                rowmeta_bank_idx = rowmeta_li_idx % ctm.Int32(SMEM_P4_TMEM_BANKS)
                rowmeta_phase = (
                    rowmeta_li_idx
                    // ctm.Int32(SMEM_P4_TMEM_BANKS)
                    % ctm.Int32(SMEM_P4_MBAR_PARITY_PHASES)
                )
                rowsum_full_mbar = pair_overlap_sync_mbars.subview(
                    SMEM_P4_PAIR_P4_READY_MBAR_OFFSET + (rowmeta_bank_idx)
                )
                while not prims.mbarrier_try_wait_parity(
                    rowsum_full_mbar, rowmeta_phase, time_limit=10000000
                ):
                    pass
                row0 = rowmeta_lane
                h0c0_row0 = _load_smem_p4_n256_quadrant_rowsum(
                    sScoreSlotRowMax, sP, row0, rowmeta_bank_idx, 0, 0
                )
                h0c1_row0 = _load_smem_p4_n256_quadrant_rowsum(
                    sScoreSlotRowMax, sP, row0, rowmeta_bank_idx, 0, 1
                )
                h1c0_row0 = _load_smem_p4_n256_quadrant_rowsum(
                    sScoreSlotRowMax, sP, row0, rowmeta_bank_idx, 1, 0
                )
                h1c1_row0 = _load_smem_p4_n256_quadrant_rowsum(
                    sScoreSlotRowMax, sP, row0, rowmeta_bank_idx, 1, 1
                )
                stage_row_sum0 = h0c0_row0 + h1c0_row0 + (h0c1_row0 + h1c1_row0)
                old_scale0 = sRowScaleRing[ctm.Int32(rowmeta_bank_idx * SMEM_P4_BMM1_M) + row0]
                running_anchor_row_sum0 = running_anchor_row_sum0 * old_scale0 + stage_row_sum0
                row1 = rowmeta_lane + ctm.Int32(32)
                h0c0_row1 = _load_smem_p4_n256_quadrant_rowsum(
                    sScoreSlotRowMax, sP, row1, rowmeta_bank_idx, 0, 0
                )
                h0c1_row1 = _load_smem_p4_n256_quadrant_rowsum(
                    sScoreSlotRowMax, sP, row1, rowmeta_bank_idx, 0, 1
                )
                h1c0_row1 = _load_smem_p4_n256_quadrant_rowsum(
                    sScoreSlotRowMax, sP, row1, rowmeta_bank_idx, 1, 0
                )
                h1c1_row1 = _load_smem_p4_n256_quadrant_rowsum(
                    sScoreSlotRowMax, sP, row1, rowmeta_bank_idx, 1, 1
                )
                stage_row_sum1 = h0c0_row1 + h1c0_row1 + (h0c1_row1 + h1c1_row1)
                old_scale1 = sRowScaleRing[ctm.Int32(rowmeta_bank_idx * SMEM_P4_BMM1_M) + row1]
                running_anchor_row_sum1 = running_anchor_row_sum1 * old_scale1 + stage_row_sum1
                if rowmeta_li_idx + ctm.Int32(1) == kv_tiles:
                    sFinalAnchorRowSum[row0] = running_anchor_row_sum0
                    sFinalAnchorRowSum[row1] = running_anchor_row_sum1
                if prims.elect_sync():
                    rowsum_empty_mbar = pair_overlap_sync_mbars.subview(
                        SMEM_P4_PAIR_HALF1_ROWSUM_READY_MBAR_OFFSET + (rowmeta_bank_idx)
                    )
                    prims.mbarrier_arrive(rowsum_empty_mbar)
        if warp_idx == MMA_WARP_ID:
            if is_leader_cta:
                tmem_raw_addr = tmem_holding_buf.load()
                tmem_base_col_id = tmem_raw_addr & ctm.Int32(65535)
                tmem_base_row_id = tmem_raw_addr >> ctm.Int32(16)
                q_res_handle0 = q_smem_consumer.wait()
                q_smem_consumer.advance()
                q_res_handle1 = q_smem_consumer.wait()
                q_smem_consumer.advance()
                q_res_handle2 = q_smem_consumer.wait()
                q_smem_consumer.advance()
                _fence_async_shared_cta()
                if prims.elect_sync():
                    for qsf_ready_cta_rank in ctm.range_constexpr(CLUSTER_SHAPE_MNK[0]):
                        qsf_ready_peer_mbar = _mapa_shared_cluster(
                            qsf_owner_ready_mbar, qsf_ready_cta_rank
                        )
                        _mbarrier_arrive_shared_cluster(qsf_ready_peer_mbar)
                prims.barrier(
                    barrier_id=SMEM_P4_QSF_OWNER_BAR_ID,
                    number_of_threads=SMEM_P4_QSF_OWNER_BAR_THREADS,
                )
                prims.barrier(
                    barrier_id=SMEM_P4_QSF_OWNER_BAR_ID,
                    number_of_threads=SMEM_P4_QSF_OWNER_BAR_THREADS,
                )
                while not prims.mbarrier_try_wait_parity(
                    qsf_owner_done_mbar, 0, time_limit=10000000
                ):
                    pass
                stream_li_total = kv_tiles
                qk0_handle = qk_smem_consumer.wait()
                qk_smem_consumer.advance()
                qk0_score_handle = tmem_score_slot0_producer.acquire_and_advance()
                for qk_burst_start in ctm.range_constexpr(
                    0, QK_MMA_KBLOCKS, SMEM_P4_QK_SCALE_BURST_KBLOCKS
                ):
                    _fence_async_shared_cta()
                    _stage_smem_p4_qk_full_ksf_from_base(
                        sSFB,
                        tmem_base_col_id,
                        tmem_base_row_id,
                        sfb_col_offset=SMEM_P4_SCALE_B_COL_OFFSET,
                        sfb_stage_base=0,
                        page_major_ksf=False,
                    )
                    _issue_smem_p4_qk_stage_burst_from_base(
                        sA,
                        sB,
                        sSFA,
                        sSFB,
                        tmem_base_col_id,
                        tmem_base_row_id,
                        score_slot_idx=0,
                        qk_kblock_start=qk_burst_start,
                        qk_kblocks=SMEM_P4_QK_SCALE_BURST_KBLOCKS,
                        acc_col_offset=SMEM_P4_QK_ACC_COL_OFFSET,
                        sfa_col_offset_base=SMEM_P4_QK_SFA_COL_OFFSET,
                        sfb_col_offset_base=SMEM_P4_SCALE_B_COL_OFFSET,
                        use_resident_q_stages=True,
                        b_stage_base=0,
                    )
                if prims.elect_sync():
                    prims.tcgen05_commit(
                        qk_full_mbar,
                        multicast_mask=ctm.Int16(3),
                        group=prims.CTAGroup.CTA_2,
                    )
                qk0_score_handle.commit()
                qk0_handle.release()
                for stream_li_idx in cutlass.range(0, stream_li_total, 1, unroll=1):
                    qk_next_li_idx = stream_li_idx + ctm.Int32(1)
                    qk_next_stage_idx = qk_next_li_idx % ctm.Int32(SMEM_P4_QK_SMEM_PIPELINE_SLOTS)
                    qk_consumer_phase = (
                        qk_next_li_idx
                        // ctm.Int32(SMEM_P4_QK_SMEM_PIPELINE_SLOTS)
                        % ctm.Int32(SMEM_P4_MBAR_PARITY_PHASES)
                    )
                    qk_stage_wrapped = ctm.Int32(qk_next_stage_idx == ctm.Int32(0))
                    qk_bulk_phase = qk_consumer_phase ^ qk_stage_wrapped
                    if qk_next_li_idx < stream_li_total:
                        split_qk15_stage1 = ctm.Boolean(
                            stream_li_total == ctm.Int32(16)
                        ) & ctm.Boolean(qk_next_li_idx == ctm.Int32(15))
                        _runtime_refill_qk_tile_dispatch(
                            sA,
                            sB,
                            sSFA,
                            sSFB,
                            tmem_base_col_id,
                            tmem_base_row_id,
                            qk_bulk_ready_mbars,
                            qk_full_mbar,
                            pair_overlap_sync_mbars,
                            tmem_score_slot0_producer,
                            qk_smem_consumer,
                            qk_next_li_idx,
                            qk_next_stage_idx,
                            qk_bulk_phase,
                            qk_consumer_phase,
                            split_qk15_stage1,
                            False,
                        )
                    v_smem_consumer.wait_and_advance()
                    _fence_async_shared_cta()
                    _runtime_issue_pv_tile_dispatch(
                        sP,
                        sV,
                        sPSF,
                        sVSF,
                        tmem_base_col_id,
                        tmem_base_row_id,
                        qk_full_mbar,
                        pair_overlap_sync_mbars,
                        p4_smem_slot0_mbars,
                        pv_full_mbar,
                        v_smem_mbars,
                        stream_li_idx,
                        stream_li_total,
                        ctm.Boolean(stream_li_idx == ctm.Int32(0)),
                    )
                q_res_handle0.release()
                q_res_handle1.release()
                q_res_handle2.release()
                tmem_score_slot0_producer.tail()
        if warp_idx == MMA_WARP_ID and (not is_leader_cta):
            while not prims.mbarrier_try_wait_parity(qsf_owner_ready_mbar, 0, time_limit=10000000):
                pass
            _fence_async_shared_cta()
            prims.barrier(
                barrier_id=SMEM_P4_QSF_OWNER_BAR_ID,
                number_of_threads=SMEM_P4_QSF_OWNER_BAR_THREADS,
            )
            prims.barrier(
                barrier_id=SMEM_P4_QSF_OWNER_BAR_ID,
                number_of_threads=SMEM_P4_QSF_OWNER_BAR_THREADS,
            )
    if warp_idx < MMA_WARP_ID:
        prims.setmaxregister(SMEM_P4_ALU_REGISTER_BUDGET, prims.SetMaxRegisterAction.INCREASE)
        if warp_idx < ctm.Int32(SMEM_P4_ATOMIC_RUNNING_ROWMAX_SMEM_ROWS // 32):
            atomic_rowmax_row = warp_idx * ctm.Int32(32) + (tidx & ctm.Int32(31))
            sAtomicRunningRowMax[atomic_rowmax_row] = _float_to_ordered_u32_for_atomic_max(
                ctm.Float32(-ctm.Float32.inf)
            )
        prims.barrier(barrier_id=TMEM_BAR_ID, number_of_threads=TMEM_BAR_THREADS)
        is_compute_warp = warp_idx < ctm.Int32(SMEM_P4_CORRECTION_WARP_ID_BEGIN)
        is_correction_warp = warp_idx >= ctm.Int32(SMEM_P4_CORRECTION_WARP_ID_BEGIN)
        if warp_idx < ctm.Int32(SMEM_P4_COMPUTE_WARPS):
            qsf_tmem_raw_addr = tmem_holding_buf.load()
            qsf_tmem_base_col_id = qsf_tmem_raw_addr & ctm.Int32(65535)
            qsf_tmem_base_row_id = qsf_tmem_raw_addr >> ctm.Int32(16)
            prims.barrier(
                barrier_id=SMEM_P4_QSF_OWNER_BAR_ID,
                number_of_threads=SMEM_P4_QSF_OWNER_BAR_THREADS,
            )
            if warp_idx < ctm.Int32(SMEM_P4_BMM1_M // 32):
                _permute_raw_trtllm_q_tail_in_smem(sA, warp_idx, tidx & ctm.Int32(31))
                _fence_async_shared_cta()
            _store_raw_trtllm_qsf_128dp_unique_owner_to_tmem_from_smem(
                sSFA,
                qsf_tmem_base_col_id,
                qsf_tmem_base_row_id,
                cta_rank,
                warp_idx,
                tidx & ctm.Int32(31),
            )
            prims.barrier(
                barrier_id=SMEM_P4_QSF_OWNER_BAR_ID,
                number_of_threads=SMEM_P4_QSF_OWNER_BAR_THREADS,
            )
            if warp_idx == 0:
                if prims.elect_sync():
                    qsf_done_peer_mbar = _mapa_shared_cluster(qsf_owner_done_mbar, ctm.Int32(0))
                    _mbarrier_arrive_shared_cluster(qsf_done_peer_mbar)
        stream_li_total = kv_tiles
        running_row_max = ctm.Float32(-ctm.Float32.inf)
        running_row_anchor = ctm.Float32(-ctm.Float32.inf)
        p4_tmem_raw_addr = tmem_holding_buf.load()
        p4_tmem_base_col_id = p4_tmem_raw_addr & ctm.Int32(65535)
        p4_tmem_base_row_id = p4_tmem_raw_addr >> ctm.Int32(16)
        corr_local_warp_idx = warp_idx & ctm.Int32(3)
        corr_row_band = corr_local_warp_idx & ctm.Int32(1)
        corr_col_band = corr_local_warp_idx >> ctm.Int32(1)
        corr_row_id = (
            p4_tmem_base_row_id + corr_row_band * ctm.Int32(64) + corr_col_band * ctm.Int32(32)
        )
        corr_n_tile_idx = ctm.Int32(1) - (warp_idx >> ctm.Int32(2))
        corr_col_id = (
            p4_tmem_base_col_id
            + ctm.Int32(SMEM_P4_O_ACC_COL_OFFSETS[0])
            + corr_n_tile_idx * ctm.Int32(SMEM_P4_O_ACC_TILE_COLS)
        )
        corr_tmem_addr = corr_row_id << ctm.Int32(16) | corr_col_id
        corr_o_acc_tmem = ctm.inttoptr(corr_tmem_addr, 6, ctm.Float32)
        leader_pair_overlap_sync_mbars_dsmem = _mapa_shared_cluster(
            pair_overlap_sync_mbars, ctm.Int32(0)
        )
        leader_pair_overlap_sync_mbars = leader_pair_overlap_sync_mbars_dsmem
        zero_packed_words = ctm.Vector.from_elements(
            tuple((ctm.Int32(0) for _ in range(8))), dtype=ctm.Int32
        )
        half0_carried_packed_words = zero_packed_words
        half0_carried_psf_word = ctm.Int32(0)
        half0_carried_old_scale = ctm.Float32(1.0)
        half1_carried_packed_words = zero_packed_words
        half1_carried_psf_word = ctm.Int32(0)
        carried_dq1024 = ctm.Int32(0)
        carried_old_scale = ctm.Float32(1.0)
        stream_li_idx = ctm.Int32(0)
        stream_li_remaining = stream_li_total
        while stream_li_remaining > ctm.Int32(1):
            running_row_max, running_row_anchor, carried_dq1024, carried_old_scale = (
                _runtime_t336_producer_tile_v23(
                    sAtomicRunningRowMax,
                    sP,
                    sPSF,
                    sScoreSlotRowMax,
                    sRowScaleRing,
                    raw_v_tma_mbar,
                    sRawV,
                    sV,
                    qk_full_mbar,
                    pair_overlap_sync_mbars,
                    leader_pair_overlap_sync_mbars,
                    leader_pair_overlap_sync_mbars_dsmem,
                    p4_smem_slot0_producer,
                    p4_smem_slot0_mbar_ptr,
                    tmem_score_slot0_consumer,
                    tmem_score_slot0_mbar_ptr,
                    pv_full_mbar,
                    corr_o_acc_tmem,
                    warp_idx,
                    tidx,
                    p4_tmem_base_col_id,
                    p4_tmem_base_row_id,
                    valid_k_for_l,
                    softmax_scale_log2,
                    pv_psf_rescale,
                    running_row_max,
                    running_row_anchor,
                    carried_dq1024,
                    carried_old_scale,
                    stream_li_idx,
                    stream_li_idx > ctm.Int32(0),
                    ctm.Boolean(False),
                    False,
                    use_mixed_imlp,
                )
            )
            stream_li_idx = stream_li_idx + ctm.Int32(1)
            stream_li_remaining = stream_li_remaining - ctm.Int32(1)
        final_tile_k_end = (stream_li_idx + ctm.Int32(1)) * ctm.Int32(KV_TILE)
        tail_needs_mask = valid_k_for_l < final_tile_k_end
        running_row_max, running_row_anchor, carried_dq1024, carried_old_scale = (
            _runtime_t336_producer_tile_v23(
                sAtomicRunningRowMax,
                sP,
                sPSF,
                sScoreSlotRowMax,
                sRowScaleRing,
                raw_v_tma_mbar,
                sRawV,
                sV,
                qk_full_mbar,
                pair_overlap_sync_mbars,
                leader_pair_overlap_sync_mbars,
                leader_pair_overlap_sync_mbars_dsmem,
                p4_smem_slot0_producer,
                p4_smem_slot0_mbar_ptr,
                tmem_score_slot0_consumer,
                tmem_score_slot0_mbar_ptr,
                pv_full_mbar,
                corr_o_acc_tmem,
                warp_idx,
                tidx,
                p4_tmem_base_col_id,
                p4_tmem_base_row_id,
                valid_k_for_l,
                softmax_scale_log2,
                pv_psf_rescale,
                running_row_max,
                running_row_anchor,
                carried_dq1024,
                carried_old_scale,
                stream_li_idx,
                stream_li_idx > ctm.Int32(0),
                tail_needs_mask,
                True,
                use_mixed_imlp,
            )
        )
        if is_compute_warp:
            final_direct_p_li_idx = stream_li_total - ctm.Int32(1)
            final_direct_p_slot_idx = final_direct_p_li_idx % ctm.Int32(SMEM_P4_QK_PIPELINE_SLOTS)
            final_direct_p_phase = (
                final_direct_p_li_idx
                // ctm.Int32(SMEM_P4_QK_PIPELINE_SLOTS)
                % ctm.Int32(SMEM_P4_MBAR_PARITY_PHASES)
            )
            final_direct_p_ready_mbar = pair_overlap_sync_mbars.subview(
                SMEM_P4_PAIR_DIRECT_P_READY_MBAR_OFFSET + (final_direct_p_slot_idx)
            )
            while not prims.mbarrier_try_wait_parity(
                final_direct_p_ready_mbar, final_direct_p_phase, time_limit=10000000
            ):
                pass
            cute.arch.mbarrier_arrive(
                p4_smem_slot0_mbar_ptr + p4_smem_slot0_producer.current_handle().index,
                cutlass.Int32(0),
            )
            p4_smem_slot0_producer.advance()
            cute.arch.mbarrier_arrive(
                tmem_score_slot0_mbar_ptr
                + SMEM_P4_TMEM_SCORE_PIPELINE_STAGES
                + tmem_score_slot0_consumer.current_handle().index,
                cutlass.Int32(0),
            )
            tmem_score_slot0_consumer.advance()
        if is_correction_warp:
            last_rowsum_li_idx = stream_li_total - ctm.Int32(1)
            last_rowsum_qk_slot_idx = last_rowsum_li_idx % ctm.Int32(SMEM_P4_QK_PIPELINE_SLOTS)
            last_rowsum_empty_mbar = pair_overlap_sync_mbars.subview(
                SMEM_P4_PAIR_HALF1_ROWSUM_READY_MBAR_OFFSET + (last_rowsum_qk_slot_idx)
            )
            last_rowsum_empty_phase = (
                last_rowsum_li_idx
                // ctm.Int32(SMEM_P4_QK_PIPELINE_SLOTS)
                % ctm.Int32(SMEM_P4_MBAR_PARITY_PHASES)
            )
            while not prims.mbarrier_try_wait_parity(
                last_rowsum_empty_mbar, last_rowsum_empty_phase, time_limit=10000000
            ):
                pass
            final_row_state_local_row = corr_row_band * ctm.Int32(32) + (tidx & ctm.Int32(31))
            final_anchor_row_sum = sFinalAnchorRowSum[final_row_state_local_row]
            final_stat_scale = ctm.Float32(1.0)
            final_row_sum = final_anchor_row_sum
            final_row_max = running_row_max
            if cutlass.const_expr(write_softmax_stats):
                final_stat_scale = softmax_scale_log2 / ctm.Float32(LOG2_E)
                final_row_sum = final_anchor_row_sum * cute.exp2(
                    (running_row_anchor - running_row_max) * softmax_scale_log2,
                    fastmath=True,
                )
                if valid_k_for_l == ctm.Int32(0):
                    final_stat_scale = ctm.Float32(1.0)
                    final_row_max = ctm.Float32(-ctm.Float32.inf)
                    final_row_sum = ctm.Float32(0.0)
            if cutlass.const_expr(store_split):
                # Convert the lazy-anchor numerator and denominator to the
                # true row maximum expected by the shared reduction kernel.
                # Empty splits avoid evaluating -inf - -inf.
                if final_anchor_row_sum != ctm.Float32(0.0):
                    final_stat_scale = cute.exp2(
                        (running_row_anchor - running_row_max) * softmax_scale_log2,
                        fastmath=True,
                    )
                    final_row_sum = final_anchor_row_sum * final_stat_scale
            output_normalizer = ctm.Float32(0.0)
            if valid_k_for_l != ctm.Int32(0) and final_anchor_row_sum != ctm.Float32(0.0):
                output_normalizer = cute.arch.rcp_approx(final_anchor_row_sum) * pv_output_scale
            last_pv_li_idx = stream_li_total - ctm.Int32(1)
            last_pv_slot = last_pv_li_idx % ctm.Int32(SMEM_P4_QK_PIPELINE_SLOTS)
            last_pv_phase = (
                last_pv_li_idx
                // ctm.Int32(SMEM_P4_QK_PIPELINE_SLOTS)
                % ctm.Int32(SMEM_P4_MBAR_PARITY_PHASES)
            )
            while not prims.mbarrier_try_wait_parity(
                pv_full_mbar.subview(last_pv_slot), last_pv_phase, time_limit=10000000
            ):
                pass
            _store_final_o_from_tmem(
                mC_mnl,
                mAccum_mnl,
                mRowMax_ml,
                mRowSum_ml,
                tmem_holding_buf,
                warp_idx,
                tidx,
                bidx,
                bidy,
                bidz,
                m,
                n,
                final_row_max=final_row_max,
                final_row_sum=final_row_sum,
                final_stat_scale=final_stat_scale,
                output_normalizer=output_normalizer,
                write_softmax_stats=write_softmax_stats,
                store_split=store_split,
                producer_warp_base=SMEM_P4_CORRECTION_WARP_ID_BEGIN,
            )
            prims.barrier(barrier_id=O_STORE_BAR_ID, number_of_threads=O_STORE_BAR_THREADS)
            _dealloc_tmem_cluster(
                tmem_dealloc_mbar,
                tmem_holding_buf,
                warp_idx,
                cta_rank,
                num_tmem_alloc_cols=num_tmem_alloc_cols,
                dealloc_warp_id=SMEM_P4_CORRECTION_WARP_ID_BEGIN,
            )


@cute.kernel
def kernel(
    mPageTable_pl: cute.Tensor,
    mPageIndptr_s: cute.Tensor,
    mValidK_l: cute.Tensor,
    mHelixKvBounds_l: cute.Tensor,
    mQGlobalScale: cute.Tensor,
    mKvGlobalScale: cute.Tensor,
    tma_q_desc: ctm.GridConstant[cuda_tma.TensorMap],
    tma_k_page_desc: ctm.GridConstant[cuda_tma.TensorMap],
    tma_k_v_raw_desc: ctm.GridConstant[cuda_tma.TensorMap],
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
    mAccum_mnl: cute.Tensor,
    mRowMax_ml: cute.Tensor,
    mRowSum_ml: cute.Tensor,
    problem_size: tuple,
    runtime_m: ctm.Int32,
    v_page_offset: ctm.Int32,
    softmax_scale_log2: ctm.Float32,
    pv_output_scale: ctm.Float32,
    page_size: ctm.Constexpr,
    query_len_per_seq: ctm.Constexpr,
    use_helix_kv_bounds: ctm.Constexpr,
    use_smem_page_plan: ctm.Constexpr,
    use_mixed_imlp: ctm.Constexpr,
    use_consecutive_page_pair: ctm.Constexpr,
    use_ksf_gather4: ctm.Constexpr,
    write_softmax_stats: ctm.Constexpr,
    store_split: ctm.Constexpr = False,
) -> None:
    _run_mla_decode_body(
        mPageTable_pl,
        mPageIndptr_s,
        mValidK_l,
        mHelixKvBounds_l,
        mQGlobalScale,
        mKvGlobalScale,
        tma_q_desc.get_ptr(),
        tma_k_page_desc.get_ptr(),
        tma_k_v_raw_desc.get_ptr(),
        tma_q_tail_desc.get_ptr(),
        tma_k_tail_page_desc.get_ptr(),
        tma_v_tile_desc.get_ptr(),
        tma_qsf_desc.get_ptr(),
        tma_qsf_tail_desc.get_ptr(),
        tma_ksf_desc.get_ptr(),
        tma_ksf_stage_desc.get_ptr(),
        tma_ksf_tail_stage_desc.get_ptr(),
        tma_vsf_desc.get_ptr(),
        tma_vsf_stage_desc.get_ptr(),
        tma_v_pair_desc.get_ptr(),
        tma_sfb_desc.get_ptr(),
        mC_mnl,
        mAccum_mnl,
        mRowMax_ml,
        mRowSum_ml,
        problem_size,
        runtime_m,
        v_page_offset,
        softmax_scale_log2,
        pv_output_scale,
        page_size,
        query_len_per_seq,
        use_helix_kv_bounds,
        use_smem_page_plan,
        use_mixed_imlp,
        use_consecutive_page_pair,
        use_ksf_gather4,
        write_softmax_stats,
        store_split,
    )


def _make_fused_ptrs(
    q_data_ptr: int,
    k_data_ptr: int,
    q_sf_data_ptr: int,
    k_sf_data_ptr: int,
    b_data_ptr: int,
    sfa_data_ptr: int,
    sfb_data_ptr: int,
    page_table_data_ptr: int,
    valid_k_data_ptr: int,
    helix_kv_bounds_data_ptr: int,
    c_data_ptr: int,
    accum_data_ptr: int,
    row_max_data_ptr: int,
    row_sum_data_ptr: int,
    page_indptr_data_ptr: int,
    q_global_scale_data_ptr: int,
    kv_global_scale_data_ptr: int,
    output_dtype: torch.dtype = torch.float16,
) -> tuple[cute.Pointer, ...]:
    return (
        make_ptr(cutlass.Float4E2M1FN, q_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float4E2M1FN, k_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, q_sf_data_ptr, cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(cutlass.Uint8, k_sf_data_ptr, cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(cutlass.Float4E2M1FN, b_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, sfa_data_ptr, cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(cutlass.Uint8, sfb_data_ptr, cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(cutlass.Int32, page_table_data_ptr, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, valid_k_data_ptr, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, helix_kv_bounds_data_ptr, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(
            _cutlass_output_dtype(output_dtype),
            c_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=32,
        ),
        make_ptr(cutlass.Float32, accum_data_ptr, cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(cutlass.Float32, row_max_data_ptr, cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(cutlass.Float32, row_sum_data_ptr, cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(cutlass.Int32, page_indptr_data_ptr, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(
            cutlass.Float32,
            q_global_scale_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            cutlass.Float32,
            kv_global_scale_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
    )


_FUSED_COMPILE_CACHE: dict[tuple[object, ...], Callable] = {}
_PREPARED_FUSED_CALL_CACHE_CAPACITY = 256
_FUSED_EXECUTOR_CACHE_CAPACITY = 256


_PREPARED_FUSED_CALL_CACHE: LruCache[_PreparedFusedCall] = LruCache(
    _PREPARED_FUSED_CALL_CACHE_CAPACITY
)
_FUSED_EXECUTOR_CACHE: LruCache[Any] = LruCache(_FUSED_EXECUTOR_CACHE_CAPACITY)


def _compile_fused(
    q_data_ptr: int,
    k_data_ptr: int,
    q_sf_data_ptr: int,
    k_sf_data_ptr: int,
    b_data_ptr: int,
    sfa_data_ptr: int,
    sfb_data_ptr: int,
    page_table_data_ptr: int,
    valid_k_data_ptr: int,
    helix_kv_bounds_data_ptr: int,
    c_data_ptr: int,
    accum_data_ptr: int,
    row_max_data_ptr: int,
    row_sum_data_ptr: int,
    page_indptr_data_ptr: int,
    q_global_scale_data_ptr: int,
    kv_global_scale_data_ptr: int,
    m: int,
    n: int,
    kv: int,
    l: int,
    page_size: int,
    use_mixed_imlp: bool,
    output_dtype: torch.dtype,
    stream: cuda.CUstream,
    num_cache_pages: int = 0,
    query_len_per_seq: int = 1,
    kv_page_stride_bytes: int = 0,
    ksf_page_stride_bytes: int = 0,
    vsf_page_stride_bytes: int = 0,
    use_consecutive_page_pair: bool = False,
    write_softmax_stats: bool = False,
    use_helix_kv_bounds: bool = False,
    store_split: bool = False,
) -> Callable:
    if type(kv) is not int:
        raise TypeError(f"runtime-KV compile K must be an int, got {type(kv).__name__}")
    is_bucketed_runtime_profile = (
        SMEM_P4_PAGE_PLAN_PROFILE_KV < kv <= SMEM_P4_RUNTIME_MAX_KV
        and _select_runtime_kv_profile(kv) == kv
    )
    if kv != SMEM_P4_PAGE_PLAN_PROFILE_KV and not is_bucketed_runtime_profile:
        raise ValueError(
            "runtime-KV compile requires the fixed page-plan profile or a geometric runtime "
            f"profile in ({SMEM_P4_PAGE_PLAN_PROFILE_KV}, {SMEM_P4_RUNTIME_MAX_KV}], got {kv}"
        )
    use_ksf_gather4 = (
        not use_consecutive_page_pair and ksf_page_stride_bytes == page_size * TRTLLM_K_SF_GROUPS
    )
    cache_key = (
        kv,
        n,
        page_size,
        use_mixed_imlp,
        output_dtype,
        query_len_per_seq,
        use_consecutive_page_pair,
        use_ksf_gather4,
        write_softmax_stats,
        use_helix_kv_bounds,
        store_split,
    )
    cached = _FUSED_COMPILE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    ptrs = _make_fused_ptrs(
        q_data_ptr,
        k_data_ptr,
        q_sf_data_ptr,
        k_sf_data_ptr,
        b_data_ptr,
        sfa_data_ptr,
        sfb_data_ptr,
        page_table_data_ptr,
        valid_k_data_ptr,
        helix_kv_bounds_data_ptr,
        c_data_ptr,
        accum_data_ptr,
        row_max_data_ptr,
        row_sum_data_ptr,
        page_indptr_data_ptr,
        q_global_scale_data_ptr,
        kv_global_scale_data_ptr,
        output_dtype,
    )
    compiled = _compile_cutedsl(
        fused_fp4_mla_decode_ctm,
        *ptrs,
        (n, kv),
        ctm.Int32(SMEM_P4_CTA_GROUP_M),
        ctm.Int32(1),
        ctm.Int32(1),
        ctm.Int32(1),
        ctm.Int32(1),
        ctm.Int32(0),
        ctm.Int64(1),
        ctm.Int64(1),
        ctm.Int64(1),
        ctm.Float32(1.0),
        ctm.Float32(1.0),
        stream,
        page_size=page_size,
        use_mixed_imlp=use_mixed_imlp,
        query_len_per_seq=query_len_per_seq,
        use_helix_kv_bounds=use_helix_kv_bounds,
        use_smem_page_plan=kv == SMEM_P4_PAGE_PLAN_PROFILE_KV,
        use_consecutive_page_pair=use_consecutive_page_pair,
        use_ksf_gather4=use_ksf_gather4,
        write_softmax_stats=write_softmax_stats,
        store_split=store_split,
        options="--opt-level 2 --ptxas-options '--uumn'",
    )
    _FUSED_COMPILE_CACHE[cache_key] = compiled
    return compiled


def run_trtllm_fp4_mla_decode_page_native(
    q_internal: torch.Tensor,
    q_sf_internal: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_packed: torch.Tensor | None,
    v_sf: torch.Tensor,
    src_page_ids: torch.Tensor,
    paged_kv_indptr_decode: torch.Tensor,
    valid_k: torch.Tensor,
    output: torch.Tensor,
    *,
    max_kv_len: int,
    sm_scale: float,
    q_global_scale: torch.Tensor,
    kv_global_scale: torch.Tensor,
    page_size: int = TRTLLM_PAGE_SIZE,
    query_len_per_seq: int = 1,
    v_pack_block: int = SMEM_P4_V_N_PER_CTA,
    v_page_offset: int = 0,
    q_batch_capacity: int | None = None,
    assume_valid_k_prefix_tiles: int = 0,
    assume_consecutive_page_prefix_tiles: int = 0,
    partition_runtime_valid_k: bool = False,
    enable_mxi_imlp: bool = True,
    softmax_row_max: torch.Tensor | None = None,
    softmax_row_sum: torch.Tensor | None = None,
    helix_kv_bounds: torch.Tensor | None = None,
    split_kv_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> None:
    if type(v_pack_block) is not int:
        raise TypeError(f"v_pack_block must be an int, got {type(v_pack_block).__name__}")
    if v_pack_block not in (128, 256):
        raise ValueError(
            f"fused-V FP4 MLA decode requires v_pack_block in (128, 256), got {v_pack_block}"
        )
    if not math.isfinite(sm_scale) or sm_scale <= 0.0:
        raise ValueError(f"sm_scale must be finite and positive, got {sm_scale}")
    _validate_output_dtype(output.dtype)
    if page_size != TRTLLM_PAGE_SIZE:
        raise ValueError(
            f"page-native decode requires page_size={TRTLLM_PAGE_SIZE}, got {page_size}"
        )
    physical_k = _select_runtime_kv_profile(max_kv_len)
    for name, value in (
        ("assume_valid_k_prefix_tiles", assume_valid_k_prefix_tiles),
        (
            "assume_consecutive_page_prefix_tiles",
            assume_consecutive_page_prefix_tiles,
        ),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative, got {value}")
    if type(partition_runtime_valid_k) is not bool:
        raise TypeError(
            "partition_runtime_valid_k must be a bool, got "
            f"{type(partition_runtime_valid_k).__name__}"
        )
    if type(enable_mxi_imlp) is not bool:
        raise TypeError(f"enable_mxi_imlp must be a bool, got {type(enable_mxi_imlp).__name__}")
    # The valid-K and partition hints remain interface-compatible no-ops.  A
    # consecutive-page promise is used only when it covers every tile that
    # ``max_kv_len`` permits; partial promises fall back to the arbitrary-page
    # specialization rather than adding a runtime branch to the producer.
    del assume_valid_k_prefix_tiles, partition_runtime_valid_k
    required_consecutive_tiles = _ceil_div(max_kv_len, KV_TILE)
    use_consecutive_page_pair = assume_consecutive_page_prefix_tiles >= required_consecutive_tiles
    if q_internal.dtype != torch.uint8 or q_internal.dim() != 3:
        raise TypeError(
            f"q_internal must be a uint8 [M, Q640/2, L] tensor, got dtype={q_internal.dtype} shape={tuple(q_internal.shape)}"
        )
    physical_m, q_bytes, l_batch = q_internal.shape
    if (softmax_row_max is None) != (softmax_row_sum is None):
        raise ValueError("softmax_row_max and softmax_row_sum must be provided together")
    write_softmax_stats = softmax_row_max is not None
    if write_softmax_stats:
        expected_stats_shape = (l_batch, physical_m)
        for name, tensor in (
            ("softmax_row_max", softmax_row_max),
            ("softmax_row_sum", softmax_row_sum),
        ):
            if (
                tensor.dtype != torch.float32
                or tensor.shape != expected_stats_shape
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous float32 with shape {expected_stats_shape}"
                )
            _validate_tensor_pointer_alignment(name, tensor, alignment_bytes=32)
    if l_batch <= 0 or l_batch > CUDA_GRID_Z_MAX:
        raise ValueError(f"queries must be in [1, {CUDA_GRID_Z_MAX}], got {l_batch}")
    if q_batch_capacity is None:
        q_batch_capacity = l_batch
    if not isinstance(q_batch_capacity, int):
        raise TypeError(
            f"q_batch_capacity must be an int or None, got {type(q_batch_capacity).__name__}"
        )
    if q_batch_capacity < l_batch or q_batch_capacity > CUDA_GRID_Z_MAX:
        raise ValueError(
            f"q_batch_capacity must cover active queries and fit grid.z: queries={l_batch}, capacity={q_batch_capacity}, limit={CUDA_GRID_Z_MAX}"
        )
    if query_len_per_seq <= 0 or l_batch % query_len_per_seq != 0:
        raise ValueError(f"query_len_per_seq={query_len_per_seq} must divide queries={l_batch}")
    num_sequences = l_batch // query_len_per_seq
    if q_bytes != QK_LOGICAL_DIM // 2:
        raise ValueError(f"q_internal dimension 1 must be {QK_LOGICAL_DIM // 2}, got {q_bytes}")
    expected_q_l_stride = physical_m * (QK_LOGICAL_DIM // 2)
    if q_internal.stride()[:2] != (QK_LOGICAL_DIM // 2, 1) or (
        l_batch > 1 and q_internal.stride(2) != expected_q_l_stride
    ):
        raise ValueError(
            f"q_internal must use batch-slow [M,320,L] storage, got strides {q_internal.stride()}"
        )
    if physical_m not in SMEM_P4_PIPELINE_DIRECT_TARGET_M_OPTIONS:
        raise ValueError(
            f"page-native decode M must be in {SMEM_P4_PIPELINE_DIRECT_TARGET_M_OPTIONS}, got {physical_m}"
        )
    if q_batch_capacity > l_batch:
        q_storage_bytes = (
            q_internal.untyped_storage().nbytes()
            - q_internal.storage_offset() * q_internal.element_size()
        )
        required_q_storage_bytes = q_batch_capacity * expected_q_l_stride
        if q_storage_bytes < required_q_storage_bytes:
            raise ValueError(
                f"q_internal storage has {q_storage_bytes}B, requires {required_q_storage_bytes}B"
            )
    if output.shape != (physical_m, TRTLLM_V_HEAD_DIM, l_batch):
        raise ValueError(
            f"output must use [M,512,L] batch-slow metadata, got {tuple(output.shape)}"
        )
    expected_output_l_stride = physical_m * TRTLLM_V_HEAD_DIM
    if output.stride()[:2] != (TRTLLM_V_HEAD_DIM, 1) or (
        l_batch > 1 and output.stride(2) != expected_output_l_stride
    ):
        raise ValueError(f"output must be batch-slow contiguous, got strides {output.stride()}")
    if src_page_ids.dtype != torch.int32 or src_page_ids.dim() != 1:
        raise ValueError(
            f"src_page_ids must be a 1D int32 physical-page list, got dtype={src_page_ids.dtype} shape={tuple(src_page_ids.shape)}"
        )
    src_page_id_count = src_page_ids.numel()
    if src_page_id_count == 0 or src_page_ids.stride(0) != 1:
        raise ValueError(
            f"src_page_ids must be non-empty and contiguous, got numel={src_page_id_count} stride={src_page_ids.stride()}"
        )
    if src_page_id_count > INT32_MAX:
        raise ValueError(
            f"src_page_ids exceeds the Int32 element limit: {src_page_id_count} > {INT32_MAX}"
        )
    if (
        paged_kv_indptr_decode.dtype != torch.int32
        or paged_kv_indptr_decode.shape != (num_sequences + 1,)
        or paged_kv_indptr_decode.stride(0) != 1
    ):
        raise ValueError(
            f"paged_kv_indptr_decode must be contiguous int32 [{num_sequences + 1}], got dtype={paged_kv_indptr_decode.dtype} shape={tuple(paged_kv_indptr_decode.shape)}"
        )
    if valid_k.dtype != torch.int32 or valid_k.shape != (num_sequences,) or valid_k.stride(0) != 1:
        raise ValueError(
            f"valid_k must be contiguous int32 [{num_sequences}], got dtype={valid_k.dtype} shape={tuple(valid_k.shape)} stride={valid_k.stride()}"
        )
    if helix_kv_bounds is not None and (
        helix_kv_bounds.dtype != torch.int32
        or helix_kv_bounds.shape != (l_batch,)
        or helix_kv_bounds.stride(0) != 1
    ):
        raise ValueError(
            "helix_kv_bounds must be contiguous int32 with shape "
            f"[{l_batch}], got dtype={helix_kv_bounds.dtype} "
            f"shape={tuple(helix_kv_bounds.shape)} stride={helix_kv_bounds.stride()}"
        )
    cache_layout = _kv_cache_3d_layout(kv_cache, page_size)
    _validate_tensor_pointer_alignment("kv_cache", kv_cache, alignment_bytes=16)
    if (
        cache_layout.packed_dim < TRTLLM_K_STORAGE_DIM // 2
        or cache_layout.stride_packed_dim != 1
        or cache_layout.stride_token != TRTLLM_K_STORAGE_DIM // 2
        or (cache_layout.stride_page < page_size * (TRTLLM_K_STORAGE_DIM // 2))
        or (cache_layout.stride_page % 16 != 0)
    ):
        raise ValueError(
            "kv_cache must expose physical [page,128,K640/2] storage with a contiguous token payload and an optional padded page stride"
        )
    num_cache_pages = cache_layout.num_pages
    page_table_capacity = num_sequences * (physical_k // page_size)
    if page_table_capacity > INT32_MAX:
        raise ValueError(
            "bucketed CSR capacity exceeds the Int32 layout limit: "
            f"{page_table_capacity} > {INT32_MAX}"
        )
    if src_page_id_count > page_table_capacity:
        raise ValueError(
            f"src_page_ids exceeds the bucketed CSR capacity, got {src_page_id_count} > {page_table_capacity}"
        )
    if not isinstance(v_page_offset, int):
        raise TypeError(f"v_page_offset must be an int, got {type(v_page_offset).__name__}")
    if v_sf.dim() == 0:
        raise ValueError("v_sf must expose a physical-page dimension")
    num_v_cache_pages = int(v_sf.shape[0])
    if v_page_offset < 0 or v_page_offset > INT32_MAX:
        raise ValueError(f"v_page_offset must be in [0, {INT32_MAX}], got {v_page_offset}")
    if num_v_cache_pages > INT32_MAX:
        raise ValueError(
            f"v_sf exceeds the Int32 physical-page limit: {num_v_cache_pages} > {INT32_MAX}"
        )
    if v_page_offset + num_cache_pages > num_v_cache_pages:
        raise ValueError(
            "v_sf does not cover the requested physical-page range: "
            f"offset={v_page_offset}, pages={num_cache_pages}, "
            f"available={num_v_cache_pages}"
        )
    if (
        q_sf_internal.element_size() != 1
        or not q_sf_internal.is_contiguous()
        or q_sf_internal.numel() < physical_m * QK_SF_GROUPS * l_batch
    ):
        raise ValueError("q_sf_internal must be a contiguous byte-sized QSF buffer")
    ksf_payload_bytes = page_size * TRTLLM_K_SF_GROUPS
    if q_batch_capacity > l_batch:
        q_sf_storage_bytes = (
            q_sf_internal.untyped_storage().nbytes()
            - q_sf_internal.storage_offset() * q_sf_internal.element_size()
        )
        required_q_sf_storage_bytes = q_batch_capacity * physical_m * QK_SF_GROUPS
        if q_sf_storage_bytes < required_q_sf_storage_bytes:
            raise ValueError(
                f"q_sf_internal storage has {q_sf_storage_bytes}B, requires {required_q_sf_storage_bytes}B"
            )
    vsf_payload_bytes = TRTLLM_V_HEAD_DIM * (page_size // SF_VEC_SIZE)
    for name, tensor, payload_bytes, expected_pages in (
        ("sf_cache", sf_cache, ksf_payload_bytes, num_cache_pages),
        ("v_sf", v_sf, vsf_payload_bytes, num_v_cache_pages),
    ):
        if tensor.element_size() != 1 or tensor.dim() == 0:
            raise ValueError(f"{name} must be a byte-sized paged tensor")
        if tensor.shape[0] != expected_pages or tensor.stride(-1) != 1:
            raise ValueError(
                f"{name} must expose {expected_pages} physical pages with a contiguous innermost dimension, got shape={tuple(tensor.shape)} strides={tensor.stride()}"
            )
        if tensor.numel() < expected_pages * payload_bytes:
            raise ValueError(
                f"{name} has {tensor.numel()} logical bytes, requires at least {expected_pages * payload_bytes}"
            )
        if tensor.stride(0) < payload_bytes:
            raise ValueError(
                f"{name} page stride={tensor.stride(0)}B is smaller than its {payload_bytes}B payload"
            )
        if tensor.stride(0) % 16 != 0:
            raise ValueError(f"{name} page stride={tensor.stride(0)}B must be 16B aligned")
    device = q_internal.device
    tensors = (
        q_sf_internal,
        kv_cache,
        sf_cache,
        v_sf,
        src_page_ids,
        paged_kv_indptr_decode,
        valid_k,
        output,
        q_global_scale,
        kv_global_scale,
    )
    if write_softmax_stats:
        tensors += (softmax_row_max, softmax_row_sum)
    if helix_kv_bounds is not None:
        tensors += (helix_kv_bounds,)
    if device.type != "cuda" or any((tensor.device != device for tensor in tensors)):
        raise ValueError("all page-native decode tensors must share one CUDA device")
    if q_global_scale.dtype != torch.float32 or q_global_scale.numel() != 1:
        raise TypeError("q_global_scale must be a scalar FP32 CUDA tensor")
    if kv_global_scale.dtype != torch.float32 or kv_global_scale.numel() != 1:
        raise TypeError("kv_global_scale must be a scalar FP32 CUDA tensor")
    if split_kv_buffers is not None:
        if write_softmax_stats:
            raise ValueError("split-KV buffers and softmax stats outputs are mutually exclusive")
        expected_shapes = (
            (l_batch, physical_m, TRTLLM_V_HEAD_DIM),
            (l_batch, physical_m),
            (l_batch, physical_m),
        )
        if len(split_kv_buffers) != 3:
            raise ValueError("split_kv_buffers requires numerator, maxima, and sums")
        for tensor, shape in zip(split_kv_buffers, expected_shapes):
            if (
                tensor.shape != shape
                or tensor.dtype != torch.float32
                or tensor.device != device
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"split-KV buffer must be contiguous FP32 {shape} on {device}")
            _validate_tensor_pointer_alignment("split-KV buffer", tensor, alignment_bytes=32)
    stream = _current_cu_stream()
    device_index = torch.cuda.current_device()
    context_result, current_context = cuda.cuCtxGetCurrent()
    if context_result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"Failed to query the current CUDA context for FP4 MLA launch: {context_result}."
        )
    q_data_ptr = q_internal.data_ptr()
    k_data_ptr = kv_cache.data_ptr()
    q_sf_data_ptr = q_sf_internal.data_ptr()
    k_sf_data_ptr = sf_cache.data_ptr()
    # Legacy V tensor-map operands are dead in this specialization. Reuse the
    # canonical KV pointer so the compiled call carries no sidecar allocation.
    b_data_ptr = kv_cache.data_ptr()
    scratch_ptr = output.data_ptr()
    row_max_data_ptr = softmax_row_max.data_ptr() if write_softmax_stats else scratch_ptr
    row_sum_data_ptr = softmax_row_sum.data_ptr() if write_softmax_stats else scratch_ptr
    sfb_data_ptr = v_sf.data_ptr()
    page_table_data_ptr = src_page_ids.data_ptr()
    valid_k_data_ptr = valid_k.data_ptr()
    helix_kv_bounds_data_ptr = (
        helix_kv_bounds.data_ptr() if helix_kv_bounds is not None else valid_k_data_ptr
    )
    c_data_ptr = output.data_ptr()
    split_o_data_ptr, split_max_data_ptr, split_sum_data_ptr = (
        tuple(tensor.data_ptr() for tensor in split_kv_buffers)
        if split_kv_buffers is not None
        else (scratch_ptr, row_max_data_ptr, row_sum_data_ptr)
    )
    page_indptr_data_ptr = paged_kv_indptr_decode.data_ptr()
    q_global_scale_data_ptr = q_global_scale.data_ptr()
    kv_global_scale_data_ptr = kv_global_scale.data_ptr()
    kv_page_stride_bytes = cache_layout.stride_page
    ksf_page_stride_bytes = int(sf_cache.stride(0))
    vsf_page_stride_bytes = int(v_sf.stride(0))
    softmax_scale_log2 = sm_scale * LOG2_E
    fused = _compile_fused(
        q_data_ptr,
        k_data_ptr,
        q_sf_data_ptr,
        k_sf_data_ptr,
        b_data_ptr,
        scratch_ptr,
        sfb_data_ptr,
        page_table_data_ptr,
        valid_k_data_ptr,
        helix_kv_bounds_data_ptr,
        c_data_ptr,
        split_o_data_ptr,
        split_max_data_ptr,
        split_sum_data_ptr,
        page_indptr_data_ptr,
        q_global_scale_data_ptr,
        kv_global_scale_data_ptr,
        physical_m,
        TRTLLM_V_HEAD_DIM,
        physical_k,
        l_batch,
        page_size,
        use_mixed_imlp=enable_mxi_imlp,
        output_dtype=output.dtype,
        stream=stream,
        num_cache_pages=num_cache_pages,
        query_len_per_seq=query_len_per_seq,
        kv_page_stride_bytes=kv_page_stride_bytes,
        ksf_page_stride_bytes=ksf_page_stride_bytes,
        vsf_page_stride_bytes=vsf_page_stride_bytes,
        use_consecutive_page_pair=use_consecutive_page_pair,
        write_softmax_stats=write_softmax_stats,
        use_helix_kv_bounds=helix_kv_bounds is not None,
        store_split=split_kv_buffers is not None,
    )
    supports_prepared = class_defines_callables(
        fused, "to", "generate_execution_args", "run_compiled_program"
    )
    prepared_key = (
        fused,
        device_index,
        int(current_context),
        int(stream),
        q_data_ptr,
        k_data_ptr,
        q_sf_data_ptr,
        k_sf_data_ptr,
        b_data_ptr,
        sfb_data_ptr,
        page_table_data_ptr,
        valid_k_data_ptr,
        helix_kv_bounds_data_ptr,
        c_data_ptr,
        split_o_data_ptr,
        split_max_data_ptr,
        split_sum_data_ptr,
        page_indptr_data_ptr,
        q_global_scale_data_ptr,
        kv_global_scale_data_ptr,
        physical_m,
        l_batch,
        q_batch_capacity,
        num_cache_pages,
        num_v_cache_pages,
        v_page_offset,
        kv_page_stride_bytes,
        ksf_page_stride_bytes,
        vsf_page_stride_bytes,
        softmax_scale_log2,
        output.dtype,
        enable_mxi_imlp,
        use_consecutive_page_pair,
    )
    if supports_prepared:
        prepared = _PREPARED_FUSED_CALL_CACHE.get(prepared_key)
        if prepared is not None:
            prepared.run()
            return
    ptrs = _make_fused_ptrs(
        q_data_ptr,
        k_data_ptr,
        q_sf_data_ptr,
        k_sf_data_ptr,
        b_data_ptr,
        scratch_ptr,
        sfb_data_ptr,
        page_table_data_ptr,
        valid_k_data_ptr,
        helix_kv_bounds_data_ptr,
        c_data_ptr,
        split_o_data_ptr,
        split_max_data_ptr,
        split_sum_data_ptr,
        page_indptr_data_ptr,
        q_global_scale_data_ptr,
        kv_global_scale_data_ptr,
        output.dtype,
    )
    runtime_args = (
        *ptrs,
        (TRTLLM_V_HEAD_DIM, physical_k),
        ctm.Int32(physical_m),
        ctm.Int32(l_batch),
        ctm.Int32(q_batch_capacity),
        ctm.Int32(num_cache_pages),
        ctm.Int32(num_v_cache_pages),
        ctm.Int32(v_page_offset),
        ctm.Int64(kv_page_stride_bytes),
        ctm.Int64(ksf_page_stride_bytes),
        ctm.Int64(vsf_page_stride_bytes),
        ctm.Float32(softmax_scale_log2),
        ctm.Float32(1.0),
        stream,
    )
    if not supports_prepared:
        fused(*runtime_args)
        return
    executor_key = (fused, device_index, int(current_context))
    executor = _FUSED_EXECUTOR_CACHE.get(executor_key)
    if executor is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FP4 MLA CUDA Graph capture requires an eager warmup for the "
                "compiled kernel, device, and CUDA context."
            )
        candidate = fused.to(device_index)
        if not class_defines_callables(
            candidate, "generate_execution_args", "run_compiled_program"
        ):
            fused(*runtime_args)
            return
        executor = _FUSED_EXECUTOR_CACHE.put(executor_key, candidate)
    execution_args, adapted_args = executor.generate_execution_args(*runtime_args)
    prepared = _PREPARED_FUSED_CALL_CACHE.put(
        prepared_key,
        _PreparedFusedCall(
            executor=executor,
            runtime_args=runtime_args,
            execution_args=execution_args,
            adapted_args=adapted_args,
        ),
    )
    prepared.run()


def run_trtllm_fp4_mla_decode_page_native_from_raw(
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_packed: torch.Tensor | None,
    v_sf: torch.Tensor,
    global_scale: torch.Tensor,
    src_page_ids: torch.Tensor,
    paged_kv_indptr_decode: torch.Tensor,
    kv_lens: torch.Tensor,
    output: torch.Tensor,
    *,
    max_kv_len: int,
    sm_scale: float,
    num_heads: int,
    q_global_scale: torch.Tensor | None = None,
    page_size: int = TRTLLM_PAGE_SIZE,
    query_len_per_seq: int = 1,
    v_pack_block: int = SMEM_P4_V_N_PER_CTA,
    v_page_offset: int = 0,
    q_batch_capacity: int | None = None,
    assume_valid_k_prefix_tiles: int = 0,
    assume_consecutive_page_prefix_tiles: int = 0,
    partition_runtime_valid_k: bool = False,
    enable_mxi_imlp: bool = True,
    softmax_row_max: torch.Tensor | None = None,
    softmax_row_sum: torch.Tensor | None = None,
    helix_kv_bounds: torch.Tensor | None = None,
    split_kv_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> None:
    if type(v_pack_block) is not int:
        raise TypeError(f"v_pack_block must be an int, got {type(v_pack_block).__name__}")
    if v_pack_block not in (128, 256):
        raise ValueError(
            f"fused-V FP4 MLA decode requires v_pack_block in (128, 256), got {v_pack_block}"
        )
    if num_heads != SMEM_P4_CTA_GROUP_M:
        raise ValueError(
            f"raw page-native decode requires {SMEM_P4_CTA_GROUP_M} physical heads, got {num_heads}"
        )
    if q_fp4.dtype != torch.uint8 or not q_fp4.is_contiguous():
        raise ValueError("q_fp4 must be a contiguous uint8 tensor")
    num_queries = int(output.shape[0])
    expected_q_bytes = num_queries * num_heads * (QK_LOGICAL_DIM // 2)
    if q_fp4.numel() != expected_q_bytes:
        raise ValueError(f"q_fp4 must contain {expected_q_bytes} bytes, got {q_fp4.numel()}")
    if q_sf.element_size() != 1 or not q_sf.is_contiguous():
        raise ValueError("q_sf must be a contiguous byte-sized tensor")
    expected_q_sf_bytes = num_queries * num_heads * QK_SF_GROUPS
    if q_sf.numel() != expected_q_sf_bytes:
        raise ValueError(f"q_sf must contain {expected_q_sf_bytes} bytes, got {q_sf.numel()}")
    _validate_tensor_pointer_alignment("q_fp4", q_fp4, alignment_bytes=16)
    _validate_tensor_pointer_alignment("q_sf", q_sf, alignment_bytes=16)
    if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
        raise TypeError("global_scale must be a scalar FP32 CUDA tensor")
    q_batch_slow = q_fp4.view(num_queries, num_heads, QK_LOGICAL_DIM // 2).permute(1, 2, 0)
    run_trtllm_fp4_mla_decode_page_native(
        q_batch_slow,
        q_sf,
        kv_cache,
        sf_cache,
        v_packed,
        v_sf,
        src_page_ids,
        paged_kv_indptr_decode,
        kv_lens,
        output.permute(1, 2, 0),
        max_kv_len=max_kv_len,
        sm_scale=sm_scale,
        q_global_scale=global_scale if q_global_scale is None else q_global_scale,
        kv_global_scale=global_scale,
        page_size=page_size,
        query_len_per_seq=query_len_per_seq,
        v_pack_block=v_pack_block,
        v_page_offset=v_page_offset,
        q_batch_capacity=q_batch_capacity,
        assume_valid_k_prefix_tiles=assume_valid_k_prefix_tiles,
        assume_consecutive_page_prefix_tiles=assume_consecutive_page_prefix_tiles,
        partition_runtime_valid_k=partition_runtime_valid_k,
        enable_mxi_imlp=enable_mxi_imlp,
        softmax_row_max=softmax_row_max,
        softmax_row_sum=softmax_row_sum,
        helix_kv_bounds=helix_kv_bounds,
        split_kv_buffers=split_kv_buffers,
    )
