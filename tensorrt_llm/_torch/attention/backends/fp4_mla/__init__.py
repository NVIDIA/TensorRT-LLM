# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA entry points; implementation lives in the focused sibling modules."""

from .cache_gather import load_fp4_mla_chunked_kv_cache
from .cache_update import (
    _get_fp4_mla_context_start_positions as _get_fp4_mla_context_start_positions,
)
from .cache_update import _prepare_fp4_mla_q_buffers as _prepare_fp4_mla_q_buffers
from .cache_update import (
    _scatter_fp4_mla_kv_cache_2d_context as _scatter_fp4_mla_kv_cache_2d_context,
)
from .cache_update import (
    _scatter_fp4_mla_kv_cache_2d_generation as _scatter_fp4_mla_kv_cache_2d_generation,
)
from .cache_update import _validate_fp4_mla_context_rope as _validate_fp4_mla_context_rope
from .cache_update import (
    _validate_fp4_mla_context_start_alignment as _validate_fp4_mla_context_start_alignment,
)
from .cache_update import can_fuse_fp4_mla_q_quant, scatter_fp4_mla_kv_cache
from .config import _FP4_MLA_CUTEDSL_BACKEND as _FP4_MLA_CUTEDSL_BACKEND
from .config import _FP4_MLA_K_RESIDUAL_BACKENDS as _FP4_MLA_K_RESIDUAL_BACKENDS
from .config import _FP4_MLA_MAX_GRID_Z as _FP4_MLA_MAX_GRID_Z
from .config import _FP4_MLA_PAGE_TABLE_TILE_SIZE as _FP4_MLA_PAGE_TABLE_TILE_SIZE
from .config import _FP4_MLA_Q1_KV_BLOCKS_LARGE_BATCH as _FP4_MLA_Q1_KV_BLOCKS_LARGE_BATCH
from .config import _FP4_MLA_Q1_KV_BLOCKS_MEDIUM_BATCH as _FP4_MLA_Q1_KV_BLOCKS_MEDIUM_BATCH
from .config import _FP4_MLA_Q1_KV_BLOCKS_SMALL_BATCH as _FP4_MLA_Q1_KV_BLOCKS_SMALL_BATCH
from .config import _FP4_MLA_Q1_KV_LARGE_BATCH_THRESHOLD as _FP4_MLA_Q1_KV_LARGE_BATCH_THRESHOLD
from .config import _FP4_MLA_Q1_KV_MEDIUM_BATCH_THRESHOLD as _FP4_MLA_Q1_KV_MEDIUM_BATCH_THRESHOLD
from .config import (
    _FP4_MLA_Q1_PREFIX_GROUP4_BATCH_THRESHOLD as _FP4_MLA_Q1_PREFIX_GROUP4_BATCH_THRESHOLD,
)
from .config import (
    _FP4_MLA_Q1_PREFIX_PAIR_BATCH_THRESHOLD as _FP4_MLA_Q1_PREFIX_PAIR_BATCH_THRESHOLD,
)
from .config import _FP4_MLA_TRITON_PRELOAD_KEYS as _FP4_MLA_TRITON_PRELOAD_KEYS
from .config import (
    FP4_BLOCK_SIZE,
    FP4_MLA_ATTENTION_BACKEND_ENV,
    FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV,
    FP4_MLA_E4M3_MAX,
    FP4_MLA_K_RESIDUAL_DIM,
    FP4_MLA_KV_GLOBAL_SCALE,
    FP4_MLA_KV_STATIC_AMAX,
    FP4_MLA_P_GLOBAL_SCALE,
    FP4_MLA_Q1_PREFIX_BLOCK_DIM,
    FP4_MLA_Q_GLOBAL_SCALE,
    FP4_MLA_Q_LOGICAL_DIM,
    FP4_MLA_Q_PACKED_DIM,
    FP4_MLA_Q_PREFIX_BLOCK_DIM,
    FP4_MLA_Q_PREFIX_DIM,
    FP4_MLA_Q_RESIDUAL_DIM,
    FP4_MLA_Q_SF_GROUPS,
    FP4_MLA_Q_STATIC_AMAX,
    FP4_MLA_SCALE_COL_GROUP,
    FP4_MLA_SCALE_ROW_GROUP,
    FP4_MLA_TOKENS_PER_BLOCK,
    HP_BLOCK_SIZE,
)
from .config import _ceil_div as _ceil_div
from .config import _cutedsl_backend_available as _cutedsl_backend_available
from .config import _env_enabled_default as _env_enabled_default
from .config import _env_int as _env_int
from .config import _fp4_mla_attention_backend as _fp4_mla_attention_backend
from .config import (
    _fp4_mla_cutedsl_fused_v_transpose_enabled as _fp4_mla_cutedsl_fused_v_transpose_enabled,
)
from .config import _fp4_mla_cutedsl_kernel_module as _fp4_mla_cutedsl_kernel_module
from .config import _fp4_mla_q1_kv_blocks_per_program as _fp4_mla_q1_kv_blocks_per_program
from .config import _fp4_mla_q1_prefix_blocks_per_program as _fp4_mla_q1_prefix_blocks_per_program
from .config import _fp4_mla_q1_preload_variants as _fp4_mla_q1_preload_variants
from .config import _fp4_mla_triton_preload_key_set as _fp4_mla_triton_preload_key_set
from .config import _HPUpdatePhase as _HPUpdatePhase
from .decode import _SM_COUNT_CACHE as _SM_COUNT_CACHE
from .decode import _cutedsl_pad_q_and_sf_kernel as _cutedsl_pad_q_and_sf_kernel
from .decode import _cutedsl_swizzled_sf_offset as _cutedsl_swizzled_sf_offset
from .decode import _get_sm_count as _get_sm_count
from .decode import _run_triton_attention_decode as _run_triton_attention_decode
from .decode import run_fp4_mla_attention_decode
from .layout import _ensure_workspace_tensor as _ensure_workspace_tensor
from .layout import _get_fp4_mla_global_scale as _get_fp4_mla_global_scale
from .layout import _get_fp4_mla_hp_pool_layout as _get_fp4_mla_hp_pool_layout
from .layout import _get_fp4_mla_kv_cache_tensors as _get_fp4_mla_kv_cache_tensors
from .layout import _get_fp4_mla_q_global_scale as _get_fp4_mla_q_global_scale
from .layout import _get_fp4_mla_swizzled_scale_size as _get_fp4_mla_swizzled_scale_size
from .layout import _validate_fp4_mla_attention_q_shape as _validate_fp4_mla_attention_q_shape
from .layout import _validate_fp4_mla_cache_shape as _validate_fp4_mla_cache_shape
from .layout import _validate_fp4_mla_hp_generation_width as _validate_fp4_mla_hp_generation_width
from .layout import _validate_fp4_mla_kv_storage_shape as _validate_fp4_mla_kv_storage_shape
from .layout import (
    get_fp4_mla_v_scale_pool_shape,
    get_fp4_mla_v_scale_pool_size,
    get_fp4_mla_v_scale_pool_view,
)
from .metadata import _fp4_mla_append_metadata_kernel as _fp4_mla_append_metadata_kernel
from .metadata import _fp4_mla_generation_hp_page_ids as _fp4_mla_generation_hp_page_ids
from .metadata import _fp4_mla_generation_lengths_kernel as _fp4_mla_generation_lengths_kernel
from .metadata import _fp4_mla_generation_num_blocks_device as _fp4_mla_generation_num_blocks_device
from .metadata import _fp4_mla_generation_page_ids as _fp4_mla_generation_page_ids
from .metadata import (
    _fp4_mla_materialize_page_table_kernel as _fp4_mla_materialize_page_table_kernel,
)
from .metadata import _fp4_mla_page_table_spec as _fp4_mla_page_table_spec
from .metadata import (
    _fp4_mla_store_sequence_append_metadata as _fp4_mla_store_sequence_append_metadata,
)
from .metadata import _fp4_mla_uniform_generation_lengths as _fp4_mla_uniform_generation_lengths
from .metadata import _get_linear_mtp_query_len_per_seq as _get_linear_mtp_query_len_per_seq
from .metadata import _host_int_list as _host_int_list
from .metadata import _host_int_list_during_forward as _host_int_list_during_forward
from .metadata import _infer_assume_full_pages as _infer_assume_full_pages
from .metadata import (
    _materialize_fp4_mla_device_page_table_for_forward as _materialize_fp4_mla_device_page_table_for_forward,
)
from .metadata import _max_generation_pages as _max_generation_pages
from .metadata import (
    configure_fp4_mla_device_page_table,
    materialize_fp4_mla_device_page_table,
    populate_fp4_mla_append_metadata,
    populate_fp4_mla_generation_lengths,
)
from .v_cache import (
    _get_cutedsl_persistent_v_packed_cache as _get_cutedsl_persistent_v_packed_cache,
)
from .v_cache import _get_fp4_mla_v_packed_pool as _get_fp4_mla_v_packed_pool
from .v_cache import _get_fp4_mla_v_packed_pool_base as _get_fp4_mla_v_packed_pool_base
from .v_cache import _get_fp4_mla_v_scale_pool_base as _get_fp4_mla_v_scale_pool_base
from .v_cache import _get_triton_v_packed_cache as _get_triton_v_packed_cache
from .v_cache import _is_triton_v_packed_cache_valid as _is_triton_v_packed_cache_valid
from .v_cache import _maybe_update_triton_v_packed_cache as _maybe_update_triton_v_packed_cache
from .v_cache import _repack_cutedsl_v_packed_cache as _repack_cutedsl_v_packed_cache
from .v_cache import _select_triton_block_v as _select_triton_block_v
from .v_cache import _set_triton_v_packed_cache_valid as _set_triton_v_packed_cache_valid
from .v_cache import _shared_v_pack_storage_enabled as _shared_v_pack_storage_enabled
from .v_cache import _triton_can_prepack_v as _triton_can_prepack_v
from .v_cache import _triton_prepack_v_enabled as _triton_prepack_v_enabled
from .v_cache import _triton_shared_v_packed_valid_attr as _triton_shared_v_packed_valid_attr
from .v_cache import _triton_v_packed_attr as _triton_v_packed_attr
from .v_cache import _triton_v_packed_cache_tag as _triton_v_packed_cache_tag
from .v_cache import _triton_v_packed_valid_attr as _triton_v_packed_valid_attr
from .v_cache import _update_triton_v_packed_cache as _update_triton_v_packed_cache
from .v_cache import _v_packed_cache_tag as _v_packed_cache_tag
from .v_cache import _v_packed_shape as _v_packed_shape
from .v_cache import rebuild_fp4_mla_disagg_imported_cache

__all__ = [
    "load_fp4_mla_chunked_kv_cache",
    "rebuild_fp4_mla_disagg_imported_cache",
    "FP4_BLOCK_SIZE",
    "FP4_MLA_ATTENTION_BACKEND_ENV",
    "FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV",
    "FP4_MLA_E4M3_MAX",
    "FP4_MLA_KV_GLOBAL_SCALE",
    "FP4_MLA_KV_STATIC_AMAX",
    "FP4_MLA_K_RESIDUAL_DIM",
    "FP4_MLA_P_GLOBAL_SCALE",
    "FP4_MLA_Q1_PREFIX_BLOCK_DIM",
    "FP4_MLA_Q_GLOBAL_SCALE",
    "FP4_MLA_Q_LOGICAL_DIM",
    "FP4_MLA_Q_PACKED_DIM",
    "FP4_MLA_Q_PREFIX_BLOCK_DIM",
    "FP4_MLA_Q_PREFIX_DIM",
    "FP4_MLA_Q_RESIDUAL_DIM",
    "FP4_MLA_Q_SF_GROUPS",
    "FP4_MLA_Q_STATIC_AMAX",
    "FP4_MLA_SCALE_COL_GROUP",
    "FP4_MLA_SCALE_ROW_GROUP",
    "FP4_MLA_TOKENS_PER_BLOCK",
    "HP_BLOCK_SIZE",
    "can_fuse_fp4_mla_q_quant",
    "configure_fp4_mla_device_page_table",
    "get_fp4_mla_v_scale_pool_shape",
    "get_fp4_mla_v_scale_pool_size",
    "get_fp4_mla_v_scale_pool_view",
    "materialize_fp4_mla_device_page_table",
    "populate_fp4_mla_append_metadata",
    "populate_fp4_mla_generation_lengths",
    "run_fp4_mla_attention_decode",
    "scatter_fp4_mla_kv_cache",
]
