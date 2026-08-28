# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Storage-agnostic configuration for PrimTS block-sparse attention."""

from dataclasses import dataclass, replace
import functools
from typing import TYPE_CHECKING, Literal, cast

import torch

from .._utils import ceil_div
from ..decode import _dtype_key, _validate_mask, _validate_positive_int
from .common import (
    _PREPARED_KV_ROUTE_SIZE,
    _SIGNED_INT32_MAX,
    _select_block_sparse_q_tile_size,
    _validate_sparse_kv_block_size,
)
from .prepared import _BlockSparseRouteLayout

if TYPE_CHECKING:
    from ..kernels.fmha_decode.fmha_decode_config import FmhaDecodeConfig


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
# KV128 is the general prepared-route geometry. The qualified Q64/D128
# 16-bit profiles consume one native KV256 route for coarse KV blocks.
_BLOCK_SPARSE_DEFAULT_KV_ROUTE_SIZE = _PREPARED_KV_ROUTE_SIZE
_BLOCK_SPARSE_KV256_ROUTE_SIZE = 256
# CLC work stealing needs roughly two waves of independent sparse rows to
# amortize its request/response path. Below that point, a direct static grid
# avoids scheduler overhead without sacrificing useful device parallelism.
_BLOCK_SPARSE_CLC_MIN_WAVES = 2
_CUDA_GRID_YZ_MAX = 65_535
# Causal CLC needs more work per launch and per row to amortize dequeue cost.
# These B200-qualified thresholds affect scheduling only, not correctness.
_CAUSAL_CLC_WAVE_THRESHOLD = 5
_CAUSAL_CLC_MIN_MAX_ROW_ROUTES = 4
# Fine-Q CLC limits are conservative outside the measured B8/B16 geometries.
_Q8_CONSERVATIVE_CLC_MAX_ROW_ROUTES = 8
_Q8_B8_MASKED_CLC_MAX_ROW_ROUTES = 12
_Q8_FINE_KV_CLC_MAX_QUALIFIED_ROW_ROUTES = 128
# A SWAPAB loop schedules two KV128 route records at a time, padding an odd
# capacity. Representative B200 Q8/B8 and Q16/B16 sweeps place the crossover
# at three pairs for masked B8 and four pairs for B16; unmasked B8 benefits
# immediately. Reuse those measurements as KV-side defaults for every Swaps Q
# tile so cross-geometries do not create additional codegen policy variants.
_B8_MASKED_MIN_PARALLEL_ROUTE_PAIRS = 3
_B16_MIN_PARALLEL_ROUTE_PAIRS = 4


@dataclass(frozen=True)
class _BlockSparseCompileKey:
    """Named, hashable inputs that determine one compiled adapter."""

    device_index: int
    batch_size: int
    seq_len_q: int
    seq_len_kv: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    q_block_size: int
    kv_block_size: int
    kv_route_size: int
    dtype_key: str
    mask_type: Literal["dense", "causal"]
    use_kv_valid_bits: bool
    use_persistent_scheduler: bool
    use_parallel_sparse_kv_loads: bool
    page_size: int | None = None
    use_variable_seqlens_kv: bool = False


@dataclass(frozen=True)
class _BlockSparseLaunchSpec:
    """Resolved launch policy and compiler key for one sparse plan."""

    policy: tuple[tuple[str, object], ...]
    compile_key: _BlockSparseCompileKey


_CAPACITY_UNSET = object()


@dataclass(frozen=True)
class _BlockSparseStaticProfile:
    """Pure, device-independent facts shared by plan and one-shot paths."""

    batch_size: int
    seq_len_q: int
    seq_len_kv: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    q_block_size: int
    kv_block_size: int
    q_tile_size: int
    kv_route_size: int
    dtype_key: str
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    output_dtype: torch.dtype
    mask_type: Literal["dense", "causal"]
    use_kv_valid_bits: bool
    max_blocks_per_row: int | None
    page_size: int | None = None


def _select_block_sparse_kv_route_size(
    *,
    q_tile_size: int,
    kv_block_size: int,
) -> int:
    """Choose a route width from the immutable compile-time profile."""

    if q_tile_size == 64 and kv_block_size % 64 == 0:
        return _BLOCK_SPARSE_KV256_ROUTE_SIZE
    return _BLOCK_SPARSE_DEFAULT_KV_ROUTE_SIZE


def _should_consider_clc(
    *,
    q_tile_size: int,
    kv_block_size: int,
    mask_type: Literal["dense", "causal"],
    max_row_route_capacity: int,
    use_kv_valid_bits: bool,
) -> bool:
    """Return whether the common selector may choose CLC for this task."""

    if q_tile_size != 8:
        return True
    if mask_type == "causal":
        max_qualified_routes = _Q8_CONSERVATIVE_CLC_MAX_ROW_ROUTES
    elif kv_block_size == 8 and use_kv_valid_bits:
        max_qualified_routes = _Q8_B8_MASKED_CLC_MAX_ROW_ROUTES
    elif kv_block_size in (8, 16):
        max_qualified_routes = _Q8_FINE_KV_CLC_MAX_QUALIFIED_ROW_ROUTES
    else:
        max_qualified_routes = _Q8_CONSERVATIVE_CLC_MAX_ROW_ROUTES
    return max_row_route_capacity <= max_qualified_routes


def _select_parallel_sparse_kv_loads(
    *,
    kv_block_size: int,
    use_kv_valid_bits: bool,
    max_row_route_capacity: int,
    use_persistent_scheduler: bool,
) -> bool:
    """Choose two K/V issuer warps from KV-side TMA issue pressure."""

    if kv_block_size not in (8, 16):
        return False
    if not use_persistent_scheduler:
        return True

    # Base the crossover on paired route capacity so R(2n - 1) and R(2n),
    # whose final pair differs only by one padded route, share one path.
    if kv_block_size == 8 and use_kv_valid_bits:
        min_route_pairs = _B8_MASKED_MIN_PARALLEL_ROUTE_PAIRS
    elif kv_block_size == 16:
        min_route_pairs = _B16_MIN_PARALLEL_ROUTE_PAIRS
    else:
        return True
    route_capacity_pairs = (max_row_route_capacity + 1) // 2
    return route_capacity_pairs >= min_route_pairs


def _select_block_sparse_scheduler(
    *,
    device_index: int,
    batch_size: int,
    seq_len_q: int,
    num_qo_heads: int,
    num_kv_heads: int,
    q_block_size: int,
    kv_block_size: int,
    kv_route_size: int,
    mask_type: Literal["dense", "causal"],
    use_kv_valid_bits: bool,
    max_row_route_capacity: int,
) -> tuple[int, bool]:
    """Select the Q tile and scheduler without depending on KV storage."""

    from ..kernels.fmha_decode.fmha_decode_config import (
        _select_auto_launch_mode,
        make_q_tile_geometry,
    )

    heads_q_per_kv = num_qo_heads // num_kv_heads
    q_tile_size = _select_block_sparse_q_tile_size(
        q_block_size=q_block_size,
        heads_q_per_kv=heads_q_per_kv,
        kv_block_size=kv_block_size,
    )
    if not _should_consider_clc(
        q_tile_size=q_tile_size,
        kv_block_size=kv_block_size,
        mask_type=mask_type,
        max_row_route_capacity=max_row_route_capacity,
        use_kv_valid_bits=use_kv_valid_bits,
    ):
        return q_tile_size, False

    q_geometry = make_q_tile_geometry(
        rows_per_cta=q_tile_size,
        heads_q_per_kv=heads_q_per_kv,
        groups_tokens_heads_q=True,
    )
    scheduler_kv_capacity_tokens = max_row_route_capacity * kv_route_size
    with torch.cuda.device(device_index):
        mode = _select_auto_launch_mode(
            batch_size=batch_size,
            num_heads_kv=num_kv_heads,
            seq_len_kv=scheduler_kv_capacity_tokens,
            num_q_tiles=q_geometry.num_q_ctas(seq_len_q),
            tile_size_kv=kv_route_size,
            persistent_min_waves=(
                _CAUSAL_CLC_WAVE_THRESHOLD
                if mask_type == "causal"
                else _BLOCK_SPARSE_CLC_MIN_WAVES
            ),
            persistent_min_tiles_per_cta=(
                _CAUSAL_CLC_MIN_MAX_ROW_ROUTES if mask_type == "causal" else 1
            ),
        )
    return q_tile_size, mode == "persistent"


def _validate_matching_dtypes(
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> str:
    if not (q_dtype == kv_dtype == output_dtype):
        raise ValueError("block-sparse requires matching Q, K/V, and output dtypes")
    if q_dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(
            "block-sparse supports only torch.float16 and torch.bfloat16"
        )
    return _dtype_key(q_dtype)


def _validate_max_blocks_per_row(
    max_blocks_per_row: int,
    *,
    seq_len_kv: int,
    kv_block_size: int,
) -> int:
    """Validate the required semantic BSR row-capacity declaration."""

    if isinstance(max_blocks_per_row, bool) or not isinstance(max_blocks_per_row, int):
        raise TypeError("max_blocks_per_row must be a Python integer")
    if max_blocks_per_row < 0:
        raise ValueError("max_blocks_per_row must be non-negative")
    num_kv_blocks = ceil_div(seq_len_kv, kv_block_size)
    if max_blocks_per_row > num_kv_blocks:
        raise ValueError(
            "max_blocks_per_row cannot exceed the number of semantic KV blocks "
            f"({num_kv_blocks})"
        )
    return max_blocks_per_row


def _validate_block_sparse_static_profile(
    *,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_block_size: int,
    kv_block_size: int,
    use_kv_valid_bits: bool,
    mask_type: Literal["dense", "causal"],
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype | None,
    output_dtype: torch.dtype | None,
    max_blocks_per_row: object = _CAPACITY_UNSET,
    page_size: int | None = None,
) -> _BlockSparseStaticProfile:
    """Validate static policy before any device work or BSR inspection."""

    batch_size = _validate_positive_int(batch_size, "batch_size")
    seq_len_q = _validate_positive_int(seq_len_q, "seq_len_q")
    seq_len_kv = _validate_positive_int(seq_len_kv, "seq_len_kv")
    num_qo_heads = _validate_positive_int(num_qo_heads, "num_qo_heads")
    num_kv_heads = _validate_positive_int(num_kv_heads, "num_kv_heads")
    head_dim = _validate_positive_int(head_dim, "head_dim")
    for extent, name in (
        (seq_len_q, "seq_len_q"),
        (seq_len_kv, "seq_len_kv"),
    ):
        if extent > _SIGNED_INT32_MAX:
            raise OverflowError(f"{name} must fit in signed int32")
    # Direct and CLC launches both preserve heads/batch in grid Y/Z.
    for dimension, argument, extent in (
        ("y", "num_kv_heads", num_kv_heads),
        ("z", "batch_size", batch_size),
    ):
        if extent > _CUDA_GRID_YZ_MAX:
            raise ValueError(
                f"block-sparse grid.{dimension} exceeds the CUDA limit "
                f"{_CUDA_GRID_YZ_MAX} ({argument}={extent})"
            )
    if not isinstance(use_kv_valid_bits, bool):
        raise TypeError("use_kv_valid_bits must be a bool")
    kv_block_size = _validate_sparse_kv_block_size(kv_block_size)
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    q_tile_size = _select_block_sparse_q_tile_size(
        q_block_size=q_block_size,
        heads_q_per_kv=num_qo_heads // num_kv_heads,
        kv_block_size=kv_block_size,
    )
    validated_max_blocks_per_row = None
    if max_blocks_per_row is not _CAPACITY_UNSET:
        validated_max_blocks_per_row = _validate_max_blocks_per_row(
            cast(int, max_blocks_per_row),
            seq_len_kv=seq_len_kv,
            kv_block_size=kv_block_size,
        )
    if kv_block_size < 64 and q_tile_size >= 64:
        raise ValueError("fine KV blocks require a SwapsMmaAb Q tile")
    _validate_mask(mask_type)
    if head_dim != 128:
        raise ValueError("block-sparse requires head_dim=128")
    if mask_type == "causal" and seq_len_q > seq_len_kv:
        raise ValueError("causal block-sparse requires seq_len_q <= seq_len_kv")
    if kv_dtype is None:
        kv_dtype = q_dtype
    if output_dtype is None:
        output_dtype = q_dtype
    dtype_key = _validate_matching_dtypes(q_dtype, kv_dtype, output_dtype)
    kv_route_size = _select_block_sparse_kv_route_size(
        q_tile_size=q_tile_size,
        kv_block_size=kv_block_size,
    )
    if page_size is not None:
        _BlockSparseRouteLayout.create(
            kv_route_size=kv_route_size,
            kv_block_size=kv_block_size,
            page_size=page_size,
            has_token_bits=use_kv_valid_bits,
            route_metadata_capacity=0,
            num_rows=1,
        )
    return _BlockSparseStaticProfile(
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        q_tile_size=q_tile_size,
        kv_route_size=kv_route_size,
        dtype_key=dtype_key,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        output_dtype=output_dtype,
        mask_type=mask_type,
        use_kv_valid_bits=use_kv_valid_bits,
        max_blocks_per_row=validated_max_blocks_per_row,
        page_size=page_size,
    )


def _make_block_sparse_config(key: _BlockSparseCompileKey) -> "FmhaDecodeConfig":
    """Build one decode configuration from its exact compile cache key."""

    import cutlass

    from ..kernels.fmha_decode.fmha_decode_config import make_decode_config

    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
    }
    dtype = dtype_map[key.dtype_key]
    q_tile_size = _select_block_sparse_q_tile_size(
        q_block_size=key.q_block_size,
        heads_q_per_kv=key.num_qo_heads // key.num_kv_heads,
        kv_block_size=key.kv_block_size,
    )
    use_keeps_mma_ab = q_tile_size >= 64
    config_args: dict[str, object] = {
        "use_keeps_mma_ab": use_keeps_mma_ab,
        "tile_size_q": q_tile_size,
        "tile_size_kv": key.kv_route_size,
        "groups_tokens_heads_q": True,
        "use_block_sparse": True,
        "q_block_size": key.q_block_size,
        "kv_block_size": key.kv_block_size,
        "use_kv_valid_bits": key.use_kv_valid_bits,
        "use_parallel_sparse_kv_loads": key.use_parallel_sparse_kv_loads,
    }
    if key.use_persistent_scheduler:
        config_args["use_persistent_scheduler"] = True
    layout_args: dict[str, object]
    if key.page_size is None:
        layout_args = {"qkv_layout": "contiguousKv"}
    else:
        layout_args = {
            "qkv_layout": "pagedKv",
            "num_tokens_per_page": key.page_size,
        }
    return make_decode_config(
        headdim=key.head_dim,
        args=config_args,
        seq_len_q=key.seq_len_q,
        seq_len_kv=key.seq_len_kv,
        batch_size=key.batch_size,
        num_heads_q=key.num_qo_heads,
        num_heads_kv=key.num_kv_heads,
        qkv_dtype=dtype,
        o_dtype=dtype,
        split_kv_mode="disabled",
        splits_kv=1,
        mask_type=key.mask_type,
        auto_tuner=False,
        **layout_args,
    )


@functools.cache
def _resolve_block_sparse_launch_spec(
    device_index: int,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_block_size: int,
    kv_block_size: int,
    kv_route_size: int,
    dtype_key: str,
    mask_type: Literal["dense", "causal"],
    use_kv_valid_bits: bool,
    max_row_route_capacity: int,
    page_size: int | None = None,
    use_variable_seqlens_kv: bool = False,
) -> _BlockSparseLaunchSpec:
    """Resolve and cache one validated static or CLC launch.

    ``max_row_route_capacity`` is a conservative prepared-route bound. Live
    index values and physical-tail morphology never specialize this cache
    entry. If the selected persistent profile is unsupported, retain the valid
    static profile instead.
    """

    q_tile_size, use_persistent_scheduler = _select_block_sparse_scheduler(
        device_index=device_index,
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        kv_route_size=kv_route_size,
        mask_type=mask_type,
        use_kv_valid_bits=use_kv_valid_bits,
        max_row_route_capacity=max_row_route_capacity,
    )
    compile_key = _BlockSparseCompileKey(
        device_index=device_index,
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        kv_route_size=kv_route_size,
        dtype_key=dtype_key,
        mask_type=mask_type,
        use_kv_valid_bits=use_kv_valid_bits,
        use_persistent_scheduler=use_persistent_scheduler,
        use_parallel_sparse_kv_loads=_select_parallel_sparse_kv_loads(
            kv_block_size=kv_block_size,
            use_kv_valid_bits=use_kv_valid_bits,
            max_row_route_capacity=max_row_route_capacity,
            use_persistent_scheduler=use_persistent_scheduler,
        ),
        page_size=page_size,
        use_variable_seqlens_kv=use_variable_seqlens_kv,
    )
    try:
        _make_block_sparse_config(compile_key)
    except ValueError:
        if not compile_key.use_persistent_scheduler:
            raise
        compile_key = replace(
            compile_key,
            use_persistent_scheduler=False,
            use_parallel_sparse_kv_loads=_select_parallel_sparse_kv_loads(
                kv_block_size=kv_block_size,
                use_kv_valid_bits=use_kv_valid_bits,
                max_row_route_capacity=max_row_route_capacity,
                use_persistent_scheduler=False,
            ),
        )
        _make_block_sparse_config(compile_key)

    policy_entries: list[tuple[str, object]] = [
        ("tile_size_q", q_tile_size),
        ("tile_size_kv", kv_route_size),
    ]
    if page_size is not None:
        policy_entries.extend(
            (
                ("page_size", page_size),
                ("use_variable_seqlens_kv", use_variable_seqlens_kv),
            )
        )
    policy_entries.extend(
        (
            ("use_persistent_scheduler", compile_key.use_persistent_scheduler),
            ("max_row_route_capacity", max_row_route_capacity),
            ("use_kv_valid_bits", use_kv_valid_bits),
            (
                "use_parallel_sparse_kv_loads",
                compile_key.use_parallel_sparse_kv_loads,
            ),
        )
    )
    return _BlockSparseLaunchSpec(
        policy=tuple(policy_entries),
        compile_key=compile_key,
    )


__all__ = [
    "_BlockSparseCompileKey",
    "_BlockSparseLaunchSpec",
    "_BlockSparseStaticProfile",
    "_make_block_sparse_config",
    "_resolve_block_sparse_launch_spec",
    "_select_block_sparse_kv_route_size",
    "_select_parallel_sparse_kv_loads",
    "_should_consider_clc",
    "_validate_block_sparse_static_profile",
    "_validate_matching_dtypes",
    "_validate_max_blocks_per_row",
]
