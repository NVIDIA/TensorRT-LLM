# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA storage constants, backend selection, and kernel tuning policy."""

import functools
import importlib.util
import inspect
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


FP4_MLA_PACKED_Q_ENV = "TRTLLM_FP4_MLA_PACKED_Q"
# Packed decode: each request owns a 128-row-padded block of [query token, head]
# Q rows, the kernel splits KV across grid.z tasks and reads per-row causal
# bounds, so no Q padding/copies, no Q replication, and Helix stays split.
_FP4_MLA_PACKED_MAX_M = 1024
_FP4_MLA_PACKED_MAX_SPLITS = 16
_FP4_MLA_PACKED_MIN_TILES_PER_SPLIT = 2
_FP4_MLA_SPLIT_KV_ENV = "TRTLLM_FP4_MLA_SPLIT_KV"
# Opt-in (Kimi-K3 ULL): the packed decode is a programmatic dependent of the kernel that
# writes its query (k3_mla_q_fused, which then releases its dependents at entry). The
# decode's prologue (barriers, TMEM, plan, page table, the first KV tile) runs before its
# griddepcontrol.wait, the query load right after it. Applies when the query rows come from that
# kernel and everything else the decode reads was written by kernels that completed before it
# was launched (the KV update on the aux stream; no plan / lengths / page-table launch in the
# decode call itself).
ULL2_MLA_DECODE_PDL_ENV = "TRTLLM_K3_ULL2_MLA_DEC_PDL"


def ull2_mla_decode_pdl_enabled() -> bool:
    return _env_enabled_default(ULL2_MLA_DECODE_PDL_ENV, False)


# Opt-in (Kimi-K3 ULL3): where a packed decode whose split reduction a consumer folds
# (``DeferredPackedReduce(release_early=False)``: the fused Helix tail of a small batch) releases
# that consumer. "epilogue": once its producer warps have finished the last KV tile, so the
# consumer stages what does not depend on the decode (W_UV, the task plan) during the decode's
# partial store instead of after the decode's completion. Unset / "0": only the completion
# releases it (the ULL2 late tail).
ULL3_MLA_TAIL_TRIGGER_ENV = "TRTLLM_K3_ULL3_MLA_TAIL_TRIGGER"


def ull3_mla_tail_trigger() -> int:
    """The decode's ``launch_dependents`` mode for a late-released fold consumer (0: none)."""
    from .fp4_mla_cutedsl_mufu16 import LAUNCH_DEPENDENTS_EPILOGUE

    value = os.environ.get(ULL3_MLA_TAIL_TRIGGER_ENV, "0").strip().lower()
    if value == "epilogue":
        return LAUNCH_DEPENDENTS_EPILOGUE
    if value not in ("", "0"):
        raise ValueError(f"{ULL3_MLA_TAIL_TRIGGER_ENV}={value!r}: expected 0 or epilogue")
    return 0


def _count_fp4_mla_metadata_launch(metadata: Any) -> None:
    """Count a kernel that writes decode metadata (plan, lengths, page table) on the current
    stream; a decode launched right after one must not read that metadata before its wait."""
    metadata.fp4_mla_state.metadata_launches += 1


def mark_fp4_mla_query_launch(metadata: Any) -> None:
    """Record that the kernel writing this step's query rows was just launched (the fused
    query, ``TRTLLM_K3_ULL2_MLA_DEC_PDL``): a decode launched as its programmatic dependent
    requires that no metadata kernel was launched since."""
    state = metadata.fp4_mla_state
    state.query_launch_mark = state.metadata_launches


def _fp4_mla_packed_q_enabled() -> bool:
    """Whether decode uses the packed per-request Q layout (mufu16 CuTe DSL only)."""
    return (
        _env_enabled_default(FP4_MLA_PACKED_Q_ENV, True)
        and _fp4_mla_attention_backend() == _FP4_MLA_CUTEDSL_BACKEND
        and not _fp4_mla_cutedsl_fused_v_transpose_enabled()
    )


def _fp4_mla_packed_rows_per_seq(num_heads: int, query_len: int) -> int:
    """Rows one request owns in the packed Q layout, or 0 when packing is off."""
    if not _fp4_mla_packed_q_enabled() or num_heads <= 0 or query_len <= 0:
        return 0
    rows = _ceil_div(num_heads * query_len, FP4_MLA_SCALE_ROW_GROUP) * FP4_MLA_SCALE_ROW_GROUP
    return rows if rows <= _FP4_MLA_PACKED_MAX_M else 0


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def _fp4_mla_attention_backend() -> str:
    backend = os.getenv(FP4_MLA_ATTENTION_BACKEND_ENV)
    if backend:
        return backend.lower()
    if get_sm_version() == 107 and _cutedsl_backend_available():
        return _FP4_MLA_CUTEDSL_BACKEND
    return "triton"


# Modules the FP4 MLA CuTe DSL kernels import.
_CUTEDSL_REQUIRED_MODULES = (
    "cutlass.cute",
    "cutlass.experimental",
    "cuda.bindings.driver",
)


def _cutedsl_dsl_supports_fp4_mla_kernels() -> bool:
    """Whether the installed CuTe DSL has the descriptor API the kernels use.

    The kernels build tcgen05 shared-memory descriptors with a K-segment
    offset, which the public nvidia-cutlass-dsl release does not provide.
    """
    from cutlass.experimental.primitives import Tcgen05SmemDesc

    return "k_segment_offset" in inspect.signature(Tcgen05SmemDesc.build).parameters


@functools.cache
def _cutedsl_backend_available() -> bool:
    try:
        if not all(
            importlib.util.find_spec(module) is not None for module in _CUTEDSL_REQUIRED_MODULES
        ):
            return False
        return _cutedsl_dsl_supports_fp4_mla_kernels()
    except ImportError:
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
