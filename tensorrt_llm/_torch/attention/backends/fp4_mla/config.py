# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA storage constants, backend selection, and kernel tuning policy."""

import importlib.util
import os
from typing import Any, Literal, Optional

from tensorrt_llm._utils import get_sm_version

HP_BLOCK_SIZE: int = 16


FP4_BLOCK_SIZE: int = 16


FP4_MLA_TOKENS_PER_BLOCK: int = 128


FP4_MLA_SCALE_ROW_GROUP: int = 128


FP4_MLA_SCALE_COL_GROUP: int = 4


FP4_MLA_P_GLOBAL_SCALE: float = 448.0 * 6.0


FP4_MLA_Q_STATIC_AMAX: float = 400.0


FP4_MLA_KV_STATIC_AMAX: float = 30.0


FP4_MLA_Q_GLOBAL_SCALE: float = FP4_MLA_P_GLOBAL_SCALE / FP4_MLA_Q_STATIC_AMAX


FP4_MLA_KV_GLOBAL_SCALE: float = FP4_MLA_P_GLOBAL_SCALE / FP4_MLA_KV_STATIC_AMAX


# Max finite e4m3 magnitude for FP4 MLA block-scale clamping.
FP4_MLA_E4M3_MAX: float = 448.0


FP4_MLA_Q_RESIDUAL_DIM: int = 64


FP4_MLA_K_RESIDUAL_DIM: int = FP4_MLA_Q_RESIDUAL_DIM


FP4_MLA_Q_PREFIX_DIM: int = 512


FP4_MLA_Q_PREFIX_BLOCK_DIM: int = 256


FP4_MLA_Q1_PREFIX_BLOCK_DIM: int = 512


FP4_MLA_Q_LOGICAL_DIM: int = FP4_MLA_Q_PREFIX_DIM + 2 * FP4_MLA_Q_RESIDUAL_DIM


FP4_MLA_Q_PACKED_DIM: int = FP4_MLA_Q_LOGICAL_DIM // 2


FP4_MLA_Q_SF_GROUPS: int = FP4_MLA_Q_LOGICAL_DIM // FP4_BLOCK_SIZE


FP4_MLA_ATTENTION_BACKEND_ENV = "TRTLLM_FP4_MLA_ATTENTION_BACKEND"


FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV = "TRTLLM_FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE"


_FP4_MLA_CUTEDSL_BACKEND = "cutedsl"


_FP4_MLA_K_RESIDUAL_BACKENDS = ("triton", _FP4_MLA_CUTEDSL_BACKEND)


_FP4_MLA_Q1_KV_BLOCKS_SMALL_BATCH = 4


_FP4_MLA_Q1_KV_BLOCKS_MEDIUM_BATCH = 16


_FP4_MLA_Q1_KV_BLOCKS_LARGE_BATCH = 32


_FP4_MLA_Q1_KV_MEDIUM_BATCH_THRESHOLD = 256


_FP4_MLA_Q1_KV_LARGE_BATCH_THRESHOLD = 512


_FP4_MLA_Q1_PREFIX_PAIR_BATCH_THRESHOLD = 640


_FP4_MLA_Q1_PREFIX_GROUP4_BATCH_THRESHOLD = 768


_HPUpdatePhase = Literal["context", "generation"]


_FP4_MLA_PAGE_TABLE_TILE_SIZE = 128


_FP4_MLA_MAX_GRID_Z = 65_535


# Environment helpers


def _env_enabled_default(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _fp4_mla_cutedsl_fused_v_transpose_enabled() -> bool:
    return _env_enabled_default(FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, False)


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def _fp4_mla_attention_backend() -> str:
    backend = os.getenv(FP4_MLA_ATTENTION_BACKEND_ENV)
    if backend:
        return backend.lower()
    return _FP4_MLA_CUTEDSL_BACKEND if get_sm_version() == 107 else "triton"


def _cutedsl_backend_available() -> bool:
    try:
        return all(
            importlib.util.find_spec(module) is not None
            for module in ("ctm", "cutlass", "cuda.bindings.driver")
        )
    except ModuleNotFoundError:
        return False


def _fp4_mla_cutedsl_kernel_module() -> Any:
    if _fp4_mla_cutedsl_fused_v_transpose_enabled():
        from . import fp4_mla_cutedsl_mufu16_fused_v_transpose

        return fp4_mla_cutedsl_mufu16_fused_v_transpose

    from . import fp4_mla_cutedsl_mufu16

    return fp4_mla_cutedsl_mufu16


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _fp4_mla_q1_kv_blocks_per_program(num_gen: int, v_head_dim: int) -> int:
    """Select the Q1 KV work per program without changing the launch boundary."""
    large_batch_block_dim = _FP4_MLA_Q1_KV_BLOCKS_LARGE_BATCH * FP4_BLOCK_SIZE
    if num_gen >= _FP4_MLA_Q1_KV_LARGE_BATCH_THRESHOLD and v_head_dim % large_batch_block_dim == 0:
        return _FP4_MLA_Q1_KV_BLOCKS_LARGE_BATCH
    medium_batch_block_dim = _FP4_MLA_Q1_KV_BLOCKS_MEDIUM_BATCH * FP4_BLOCK_SIZE
    if (
        num_gen >= _FP4_MLA_Q1_KV_MEDIUM_BATCH_THRESHOLD
        and v_head_dim % medium_batch_block_dim == 0
    ):
        return _FP4_MLA_Q1_KV_BLOCKS_MEDIUM_BATCH
    small_batch_block_dim = _FP4_MLA_Q1_KV_BLOCKS_SMALL_BATCH * FP4_BLOCK_SIZE
    if v_head_dim % small_batch_block_dim == 0:
        return _FP4_MLA_Q1_KV_BLOCKS_SMALL_BATCH
    return 1


def _fp4_mla_q1_prefix_blocks_per_program(
    num_gen: int,
    q1_kv_blocks_per_program: int,
) -> int:
    """Select rolled Q-prefix work only when the batch keeps it efficient."""
    max_prefix_blocks = FP4_MLA_Q_PREFIX_DIM // FP4_MLA_Q1_PREFIX_BLOCK_DIM
    if (
        num_gen >= _FP4_MLA_Q1_PREFIX_PAIR_BATCH_THRESHOLD
        and q1_kv_blocks_per_program == _FP4_MLA_Q1_KV_BLOCKS_LARGE_BATCH
    ):
        if num_gen >= _FP4_MLA_Q1_PREFIX_GROUP4_BATCH_THRESHOLD:
            return min(4, max_prefix_blocks)
        return min(2, max_prefix_blocks)
    return 1
