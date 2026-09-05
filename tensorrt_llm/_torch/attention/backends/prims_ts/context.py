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

"""Task-scheduled contiguous and paged context attention.

The public surface intentionally exposes attention semantics and optional
compile-time metadata contracts, not scheduler choices. Reusable contiguous
and paged plans compile from static capacities; packed request lengths and
metadata remain per-run inputs. Both wrappers select scheduling from static
topology while consuming the current request extents at launch. Paged plans
conservatively treat all request lengths as variable unless the caller opts
into an exact-uniform or zero-causal-offset contract.
Single-instance dense paged plans use a static persistent launch when their
logical CTA grid exceeds one resident wave; smaller grids launch one CTA per
tile. Causal and paired persistent paged plans use CLC. The private policy is
query-paired unless a positive left window requires head-paired GQA.
Causal windows are bottom-right aligned: for row ``q``, the inclusive right
position is ``q + (S_kv - S_q)`` and ``window_left`` is measured from that
position.

PrimTS context entry points are intentionally excluded from ``fi_trace`` for
now; their ``@flashinfer_api`` decorators do not register trace templates.
"""

from dataclasses import dataclass
import functools
import itertools
import math
import numbers
import struct
from typing import TYPE_CHECKING, Callable, Literal, Optional

import torch

from flashinfer.api_logging import flashinfer_api

from ._tensor_aliasing import _validate_out_does_not_overlap_inputs


if TYPE_CHECKING:
    from .kernels.fmha_context.fmha_resources import FmhaConfig


_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"
_SUPPORTED_HEAD_DIMS = (128, 256)
_DEFAULT_PAGED_KV_PAGE_SIZE = 32
_SUPPORTED_PAGED_KV_PAGE_SIZES = (16, 32, 64, 128)
_SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
)
_SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
_INT32_MAX = 2**31 - 1
_CUDA_GRID_YZ_MAX = 65_535
_CONTEXT_KV_TILE_N = 128
_CONTEXT_TILE_SIZE_Q = 128
# Query-paired D128 represents two 128-row Q tiles in one work tile.  Kernel
# coordinates include the padded tail of that 256-row span and, for masking,
# its exclusive right boundary.  Reserve the maximum 255-row tail padding in
# every packed logical data extent so all such coordinates remain signed
# Int32 even when the final request ends at the plan's packed extent.
_CONTEXT_MAX_Q_ROWS_PER_WORK_TILE = 256
_CONTEXT_PADDED_EXTENT_MAX = _INT32_MAX - (_CONTEXT_MAX_Q_ROWS_PER_WORK_TILE - 1)


@dataclass(frozen=True)
class _ContextGeometry:
    """Validated request geometry used by the contiguous one-shot adapter."""

    device: torch.device
    device_index: int
    packed: bool
    batch_size: int
    total_q: int
    total_k: int
    max_seq_len_q: int
    max_seq_len_k: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    q_dtype: torch.dtype
    output_dtype: torch.dtype
    mask_type: str
    window_left: int
    head_paired: bool
    uniform_packed_lengths: bool
    has_q_offset: bool
    causal_single_kv_tile: bool
    packed_dense_k_mask: bool
    q_shape: tuple[int, ...]
    kv_shape: tuple[int, ...]


@dataclass(frozen=True)
class _ContextPlanGeometry:
    """Static compile geometry for one reusable contiguous context plan."""

    device: torch.device
    device_index: int
    packed: bool
    batch_size: int
    max_seq_len_q: int
    max_seq_len_k: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    output_dtype: torch.dtype
    mask_type: str
    window_left: int
    head_paired: bool
    uniform_packed_lengths: bool
    has_q_offset: bool
    causal_single_kv_tile: bool
    packed_dense_k_mask: bool


@dataclass(frozen=True)
class _ContextPlanState:
    """Atomically published state for one compiled contiguous context plan."""

    geometry: _ContextPlanGeometry
    scale_softmax_log2: torch.Tensor
    output_scale: torch.Tensor
    empty_i32: torch.Tensor
    variable_window_padded_starts: Optional[torch.Tensor]
    variable_window_cta_starts: torch.Tensor
    compiled: Callable[..., None]
    policy: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class _PagedContextPlanGeometry:
    """Static compile geometry for one reusable paged plan."""

    device: torch.device
    device_index: int
    batch_size: int
    max_seq_len_q: int
    max_kv_len: int
    page_size: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    output_dtype: torch.dtype
    mask_type: str
    window_left: int
    head_paired: bool
    uniform_packed_lengths: bool
    has_q_offset: bool
    packed_dense_k_mask: bool


@dataclass(frozen=True)
class _PagedContextPlanState:
    """Atomically published state for one compiled paged-context plan."""

    geometry: _PagedContextPlanGeometry
    scale_softmax_log2: torch.Tensor
    output_scale: torch.Tensor
    compiled: Callable[..., None]
    policy: tuple[tuple[str, object], ...]


_ContextScheduler = Literal[
    "nonpersistent",
    "static_persistent",
    "clc_dynamic_persistent",
]


@dataclass(frozen=True)
class _ContextSchedulerProbe:
    """Static kernel traits and device capacity used by scheduler policy."""

    single_qkv_instance: bool
    logical_work_tiles: int
    max_active_clusters: int


@dataclass(frozen=True)
class _ContextCompileSpec:
    """Batch-independent identity for one contiguous context compile."""

    device_index: int
    max_seq_len_q: int
    max_seq_len_k: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    q_dtype_key: str
    output_dtype_key: str
    mask_type: str
    window_left: int
    packed: bool
    head_paired: bool
    uniform_packed_lengths: bool
    has_q_offset: bool
    causal_single_kv_tile: bool
    packed_dense_k_mask: bool
    scheduler: _ContextScheduler


@dataclass(frozen=True)
class _PagedContextCompileSpec:
    """Batch-independent identity for one paged context compile."""

    device_index: int
    max_seq_len_q: int
    max_kv_len: int
    page_size: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    q_dtype_key: str
    output_dtype_key: str
    mask_type: str
    window_left: int
    head_paired: bool
    uniform_packed_lengths: bool
    has_q_offset: bool
    packed_dense_k_mask: bool
    scheduler: _ContextScheduler


def _make_context_kernel(
    *,
    input_dtype,
    output_dtype,
    head_dim: int,
    mask_type: str,
    window_left: int,
    head_paired: bool,
    num_qo_heads: int,
    num_kv_heads: int,
    has_q_offset: bool,
    causal_single_kv_tile: bool,
    scheduler: _ContextScheduler,
    page_size: int | None = None,
    max_kv_len: int | None = None,
):
    """Build one context kernel from its batch-independent static topology."""

    import cutlass

    from .kernels.fmha_context.fmha_kernel import FmhaTs

    use_paged_kv = page_size is not None
    if use_paged_kv != (max_kv_len is not None):
        raise ValueError("paged context requires both page geometry values")
    is_persistent = scheduler != "nonpersistent"
    is_clc_dynamic = scheduler == "clc_dynamic_persistent"
    paged_kwargs = (
        {
            "use_paged_kv": True,
            "num_tokens_per_page": page_size,
            "max_kv_len": max_kv_len,
        }
        if use_paged_kv
        else {}
    )
    return FmhaTs(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        in_dtype=input_dtype,
        out_dtype=output_dtype,
        d=head_dim,
        is_persistent=is_persistent,
        is_causal=mask_type == "causal",
        has_variable_window=mask_type == "variable_window",
        balance_causal_workload=(
            scheduler == "static_persistent"
            and _uses_heavy_first_static_causal_raster(
                mask_type=mask_type,
                window_left=window_left,
                has_q_offset=has_q_offset,
            )
        ),
        is_clc_dynamic=is_clc_dynamic,
        head_paired=head_paired,
        window_size_left=window_left if window_left > 0 else 0,
        h_r=num_qo_heads // num_kv_heads,
        enable_skip_correction=True,
        causal_single_kv_tile=(causal_single_kv_tile and not use_paged_kv),
        **paged_kwargs,
    )


def _validate_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")


def _compact_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= int(extent)
    return tuple(reversed(strides))


def _validate_compact(tensor: torch.Tensor, name: str, layout: str) -> None:
    expected = _compact_strides(tuple(tensor.shape))
    if tensor.stride() != expected:
        raise ValueError(
            f"{name} must have compact {layout} strides {expected}, "
            f"but has {tensor.stride()}"
        )


def _validate_alignment(tensor: torch.Tensor, name: str, alignment: int) -> None:
    if tensor.data_ptr() % alignment != 0:
        raise ValueError(f"{name} data pointer must be {alignment}-byte aligned")


def _dtype_key(dtype: torch.dtype) -> str:
    keys = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float8_e4m3fn: "float8_e4m3fn",
    }
    try:
        return keys[dtype]
    except KeyError as error:
        raise NotImplementedError(
            "attention-ts context supports torch.float16, torch.bfloat16, "
            f"and torch.float8_e4m3fn; got {dtype}"
        ) from error


def _validate_qkv_dtype(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    _dtype_key(q.dtype)
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise NotImplementedError(
            "attention-ts context requires Q, K, and V to use the same dtype; "
            f"got Q {q.dtype}, K {k.dtype}, and V {v.dtype}"
        )


def _validate_output_dtype(output_dtype: torch.dtype) -> None:
    if not isinstance(output_dtype, torch.dtype):
        raise TypeError("out_dtype must be a torch.dtype")
    _dtype_key(output_dtype)


def _validate_context_dtype_pair(
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> None:
    """Validate the explicit dtypes of a contiguous-context specialization."""

    for dtype, name in (
        (q_dtype, "q_dtype"),
        (kv_dtype, "kv_dtype"),
        (output_dtype, "out_dtype"),
    ):
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"{name} must be a torch.dtype")
        _dtype_key(dtype)
    if kv_dtype != q_dtype:
        raise NotImplementedError(
            "attention-ts context requires Q and K/V to use the same dtype; "
            f"got q_dtype={q_dtype} and kv_dtype={kv_dtype}"
        )


def _validate_paged_dtype_pair(
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> None:
    """Validate the explicit dtypes of a paged-context specialization."""

    for dtype, name in (
        (q_dtype, "q_dtype"),
        (kv_dtype, "kv_dtype"),
        (output_dtype, "out_dtype"),
    ):
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"{name} must be a torch.dtype")
        _dtype_key(dtype)
    if kv_dtype != q_dtype:
        raise NotImplementedError(
            "attention-ts paged context requires Q and K/V to use the same "
            f"dtype; got q_dtype={q_dtype} and kv_dtype={kv_dtype}"
        )


def _device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _validate_device(device: torch.device) -> int:
    device_index = _device_index(device)
    with torch.cuda.device(device_index):
        capability = torch.cuda.get_device_capability(device_index)
    if capability not in _SUPPORTED_COMPUTE_CAPABILITIES:
        raise NotImplementedError(
            "attention-ts context requires an SM100a/B200, SM103a/B300 or "
            "SM107a/Rubin GPU; "
            f"device cuda:{device_index} has compute capability {capability}"
        )
    # Rubin runs through the sm_100f family target; a CuTe DSL older than 4.8
    # cannot emit for it unless CUTE_DSL_ARCH=sm_100f is set before import.
    if capability == (10, 7):
        from flashinfer.cute_dsl.utils import require_cute_dsl_arch

        require_cute_dsl_arch(device_index)
    return device_index


def _resolve_cuda_device(
    device: int | str | torch.device,
) -> tuple[torch.device, int]:
    """Resolve one explicit CUDA plan device and validate its architecture."""

    if isinstance(device, bool):
        raise TypeError("device must identify a CUDA device")
    try:
        resolved = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise TypeError("device must identify a CUDA device") from error
    if resolved.type != "cuda":
        raise ValueError(f"device must be a CUDA device, got {resolved}")
    device_index = _validate_device(resolved)
    return torch.device("cuda", device_index), device_index


def _validate_mask(mask_type: str) -> None:
    if not isinstance(mask_type, str):
        raise TypeError("mask_type must be a string")
    if mask_type not in ("dense", "causal", "variable_window"):
        raise ValueError(
            "mask_type must be exactly 'dense', 'causal', or "
            f"'variable_window', got {mask_type!r}"
        )


def _validate_window_left(window_left: int, mask_type: str) -> int:
    if isinstance(window_left, bool) or not isinstance(window_left, int):
        raise TypeError("window_left must be an integer")
    if window_left == 0:
        raise ValueError("window_left=0 is unsupported; use -1 to disable the window")
    if window_left < -1:
        raise ValueError("window_left must be -1 (disabled) or positive")
    if window_left > _INT32_MAX - 1:
        raise ValueError(f"window_left must be no larger than {_INT32_MAX - 1}")
    if window_left > 0 and mask_type != "causal":
        raise ValueError("a positive window_left requires mask_type='causal'")
    return window_left


def _validate_variable_window_bounds(
    starts: Optional[torch.Tensor],
    ends: Optional[torch.Tensor],
    *,
    geometry: _ContextGeometry | _ContextPlanGeometry,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate inclusive per-query K bounds for contiguous attention."""
    if starts is None or ends is None:
        raise ValueError(
            "variable_window_token_starts and variable_window_token_ends are "
            "required for mask_type='variable_window'"
        )
    if geometry.packed:
        raise NotImplementedError(
            "variable_window currently requires fixed [B, S, H, D] tensors"
        )
    expected_shape = (geometry.batch_size, geometry.max_seq_len_q)
    for tensor, name in (
        (starts, "variable_window_token_starts"),
        (ends, "variable_window_token_ends"),
    ):
        _validate_tensor(tensor, name)
        if tensor.device != geometry.device:
            raise ValueError(
                f"{name} must be on {geometry.device}, got {tensor.device}"
            )
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
            )
        _validate_compact(tensor, name, "[B, Sq]")
        _validate_alignment(tensor, name, 4)
    start_values = tuple(int(value) for value in starts.flatten().tolist())
    end_values = tuple(int(value) for value in ends.flatten().tolist())
    for offset, (start, end) in enumerate(zip(start_values, end_values, strict=True)):
        if start < 0 or start > end or end >= geometry.max_seq_len_k:
            raise ValueError(
                "variable-window bounds must satisfy "
                "0 <= start <= end < max_kv_len; flattened row "
                f"{offset} has start={start}, end={end}, and "
                f"max_kv_len={geometry.max_seq_len_k}"
            )
    return starts.flatten(), ends.flatten()


def _refresh_variable_window_cta_starts(
    starts: torch.Tensor, *, state: _ContextPlanState
) -> torch.Tensor:
    """Refresh plan-owned CTA minima from the current variable-window bounds."""

    geometry = state.geometry
    tile_size_q = (
        _CONTEXT_MAX_Q_ROWS_PER_WORK_TILE
        if geometry.head_dim == _CONTEXT_TILE_SIZE_Q
        else _CONTEXT_TILE_SIZE_Q
    )
    num_seq_tiles = (geometry.max_seq_len_q + tile_size_q - 1) // tile_size_q
    padded_rows = num_seq_tiles * tile_size_q
    starts_2d = starts.view(geometry.batch_size, geometry.max_seq_len_q)
    padded = state.variable_window_padded_starts
    if padded is not None:
        padded.fill_(_INT32_MAX)
        padded[:, : geometry.max_seq_len_q].copy_(starts_2d)
        starts_2d = padded
    elif padded_rows != geometry.max_seq_len_q:
        raise RuntimeError("variable-window padded-start scratch is missing")
    torch.amin(
        starts_2d.view(geometry.batch_size, num_seq_tiles, tile_size_q),
        dim=-1,
        out=state.variable_window_cta_starts.view(geometry.batch_size, num_seq_tiles),
    )
    return state.variable_window_cta_starts


def _validate_scale(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a positive Python scalar")
    try:
        as_float = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a positive Python scalar") from error
    if not math.isfinite(as_float) or as_float <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    try:
        as_float32 = struct.unpack("=f", struct.pack("=f", as_float))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(
            f"{name} must be representable as a positive float32"
        ) from error
    if not math.isfinite(as_float32) or as_float32 <= 0.0:
        raise ValueError(f"{name} must be representable as a positive float32")
    return as_float32


def _validate_extent(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > _INT32_MAX:
        raise NotImplementedError(f"{name} must fit in a signed int32")
    return value


def _validate_static_extent(value: object, name: str) -> int:
    """Validate one explicit positive plan bound."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return _validate_extent(value, name)


def _validate_padded_data_extent(value: int, name: str) -> int:
    """Validate an Int32 data extent while reserving work-tile tail padding."""
    _validate_extent(value, name)
    if value > _CONTEXT_PADDED_EXTENT_MAX:
        raise NotImplementedError(
            f"{name} must be <= {_CONTEXT_PADDED_EXTENT_MAX} so padded "
            "context work-tile coordinates fit in a signed int32"
        )
    return value


def _validate_query_work_tile_span(config: "FmhaConfig") -> None:
    """Keep the public Int32 padding guard coupled to generated topology."""
    q_rows = int(config.q_tile_m) * int(config.work_tile_q_seq_tiles)
    if q_rows > _CONTEXT_MAX_Q_ROWS_PER_WORK_TILE:
        raise RuntimeError(
            "context Int32 extent safety assumes at most "
            f"{_CONTEXT_MAX_Q_ROWS_PER_WORK_TILE} Q rows per work tile, got {q_rows}"
        )


def _validate_indptr_tensor(
    indptr: torch.Tensor,
    name: str,
    *,
    device: torch.device,
) -> None:
    _validate_tensor(indptr, name)
    if indptr.device != device:
        raise ValueError(f"{name} must be on {device}, got {indptr.device}")
    if indptr.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if indptr.ndim != 1:
        raise ValueError(f"{name} must be rank 1, got rank {indptr.ndim}")
    if indptr.numel() < 2:
        raise ValueError(f"{name} must contain at least start and end offsets")
    _validate_compact(indptr, name, "[B+1]")
    _validate_alignment(indptr, name, 4)


def _read_indptr(
    indptr: torch.Tensor,
    name: str,
    *,
    expected_total: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Read cumulative offsets and validate strictly positive row lengths."""

    values = tuple(int(value) for value in indptr.tolist())
    if values[0] != 0:
        raise ValueError(f"{name} must start at 0")
    if values[-1] != expected_total:
        raise ValueError(
            f"the final {name} offset must equal the packed tensor extent; "
            f"expected {expected_total}, got {values[-1]}"
        )
    lengths = tuple(curr - prev for prev, curr in itertools.pairwise(values))
    if any(length <= 0 for length in lengths):
        raise ValueError(f"{name} offsets must be strictly increasing")
    return values, lengths


def _read_int32_values(
    tensor: torch.Tensor,
    name: str,
    *,
    expected_count: int,
) -> tuple[int, ...]:
    """Read one metadata vector after validating its extent."""

    if tensor.numel() != expected_count:
        raise ValueError(
            f"{name} must contain {expected_count} elements, got {tensor.numel()}"
        )
    return tuple(int(value) for value in tensor.tolist())


def _validate_paged_metadata_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    device: torch.device,
) -> None:
    _validate_tensor(tensor, name)
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be rank 1, got rank {tensor.ndim}")
    _validate_compact(tensor, name, "one-dimensional")
    _validate_alignment(tensor, name, 4)


def _validate_block_tables_tensor(
    block_tables: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    required_page_columns: int,
) -> None:
    """Validate a fixed page table while preserving its runtime row stride."""

    _validate_tensor(block_tables, "block_tables")
    if block_tables.device != device:
        raise ValueError(f"block_tables must be on {device}, got {block_tables.device}")
    if block_tables.dtype != torch.int32:
        raise TypeError("block_tables must have dtype torch.int32")
    if block_tables.ndim != 2:
        raise ValueError(f"block_tables must be rank 2, got rank {block_tables.ndim}")
    if int(block_tables.shape[0]) != batch_size:
        raise ValueError(
            "block_tables must have one row per request; expected "
            f"{batch_size}, got {block_tables.shape[0]}"
        )
    page_columns = int(block_tables.shape[1])
    if page_columns < required_page_columns:
        raise ValueError(
            "block_tables must have at least ceil(max_kv_len / page_size) "
            f"columns ({required_page_columns}), got {page_columns}"
        )
    row_stride = int(block_tables.stride(0))
    if block_tables.stride(1) != 1 or row_stride < page_columns:
        raise ValueError(
            "block_tables must have row-strided [B, C] storage with "
            f"stride(1) == 1 and stride(0) >= C; got shape "
            f"{tuple(block_tables.shape)} and strides {block_tables.stride()}"
        )
    _validate_extent(row_stride, "block_tables row stride")
    row_span = (batch_size - 1) * row_stride + page_columns
    _validate_extent(row_span, "block_tables addressable row span")
    _validate_alignment(block_tables, "block_tables", 4)


def _validate_page_size(page_size: int) -> int:
    if isinstance(page_size, bool) or not isinstance(page_size, int):
        raise TypeError("page_size must be an integer")
    if page_size not in _SUPPORTED_PAGED_KV_PAGE_SIZES:
        raise NotImplementedError(
            "attention-ts paged context supports page_size in "
            f"{_SUPPORTED_PAGED_KV_PAGE_SIZES}; got {page_size}"
        )
    return page_size


def _validate_kv_layout(kv_layout: str) -> None:
    if not isinstance(kv_layout, str):
        raise TypeError("kv_layout must be a string")
    if kv_layout != "HND":
        raise NotImplementedError(
            f"attention-ts paged context supports only kv_layout='HND'; got {kv_layout!r}"
        )


def _validate_base_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    for tensor, name in ((q, "q"), (k, "k"), (v, "v")):
        _validate_tensor(tensor, name)
        _validate_alignment(tensor, name, 16)
    if k.device != q.device or v.device != q.device:
        raise ValueError(
            "Q, K, and V must be on one CUDA device; "
            f"got {q.device}, {k.device}, and {v.device}"
        )
    _validate_qkv_dtype(q, k, v)
    if tuple(v.shape) != tuple(k.shape):
        raise ValueError(
            f"v must have the same shape as k; got {tuple(v.shape)} and {tuple(k.shape)}"
        )


def _validate_head_geometry(num_qo_heads: int, num_kv_heads: int) -> int:
    if num_qo_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("Q and KV head counts must be positive")
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            "the Q head count must be divisible by the KV head count; "
            f"got {num_qo_heads} and {num_kv_heads}"
        )
    return num_qo_heads // num_kv_heads


def _validate_head_dim(q_head_dim: int, kv_head_dim: int) -> int:
    if q_head_dim != kv_head_dim:
        raise ValueError(
            "Q and K/V head dimensions must match; "
            f"got Q {q_head_dim} and K/V {kv_head_dim}"
        )
    if q_head_dim not in _SUPPORTED_HEAD_DIMS:
        raise NotImplementedError(
            "attention-ts context supports head_dim in "
            f"{_SUPPORTED_HEAD_DIMS}; got {q_head_dim}"
        )
    return q_head_dim


def _derive_has_q_offset(
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    mask_type: str,
) -> bool:
    """Return whether any request has a bottom-right causal Q offset."""

    if mask_type != "causal":
        return False
    return any(
        k_length != q_length
        for q_length, k_length in zip(q_lengths, k_lengths, strict=True)
    )


def _uses_heavy_first_static_causal_raster(
    *,
    mask_type: str,
    window_left: int,
    has_q_offset: bool,
) -> bool:
    """Return whether a static paged plan has triangular causal tile costs.

    Zero-offset, unwindowed causal attention grows from short to long K-tile
    domains across Q tiles, so reversing the sequence raster retires the
    heaviest work first. A bottom-right offset makes the short-Q context rows
    similarly heavy, while a finite left window bounds their work; both keep
    the ordinary sequence-local raster.
    """

    return mask_type == "causal" and window_left < 0 and not has_q_offset


def _paged_context_uses_clc_scheduler(
    *,
    is_persistent: bool,
    single_qkv_instance: bool,
    is_causal: bool,
    uniform_packed_lengths: bool,
) -> bool:
    """Return whether a persistent paged plan needs the dynamic CLC queue."""

    return is_persistent and (
        not single_qkv_instance or (is_causal and not uniform_packed_lengths)
    )


def _contiguous_context_uses_clc_scheduler(
    *,
    single_qkv_instance: bool,
    head_paired: bool,
    packed: bool,
    uniform_packed_lengths: bool,
    is_causal: bool,
    has_q_offset: bool,
) -> bool:
    """Return whether contiguous persistence needs dynamic work discovery.

    Paired and zero-offset triangular task graphs use CLC to distribute their
    heavier work tiles. A bottom-right-offset single-instance fixed or
    uniform-packed plan has a near-uniform immutable domain, so its persistent
    CTAs can advance through the static queue without a CLC request/response
    on every tile. General packed plans retain CLC because live cumulative
    offsets can change the active query-tile domain.
    """

    return (
        head_paired
        or not single_qkv_instance
        or (packed and not uniform_packed_lengths)
        or (is_causal and not has_q_offset)
    )


def _contiguous_context_uses_persistent_scheduler(
    *,
    single_qkv_instance: bool,
    head_paired: bool,
    packed: bool,
    uniform_packed_lengths: bool,
    logical_work_tiles: int,
    max_active_clusters: int,
    batch_size: int,
    num_qo_heads: int,
    is_causal: bool,
    has_q_offset: bool,
) -> bool:
    """Return whether contiguous work needs a persistent launch.

    Paired, live-ragged, and zero-offset triangular task graphs keep dynamic
    persistence. Near-uniform bottom-right-offset single-instance domains
    launch directly when they fit in one resident wave and use static
    persistence only when more work remains. Oversized logical Y/Z dimensions
    also require the flattened persistent grid.
    """

    return (
        head_paired
        or not single_qkv_instance
        or (packed and not uniform_packed_lengths)
        or (is_causal and not has_q_offset)
        or logical_work_tiles > max_active_clusters
        or batch_size > _CUDA_GRID_YZ_MAX
        or num_qo_heads > _CUDA_GRID_YZ_MAX
    )


def _needs_packed_dense_k_mask(
    *,
    packed: bool,
    mask_type: str,
    k_lengths: tuple[int, ...],
) -> bool:
    """Return whether packed-contiguous or paged dense K needs right masking.

    Uniform K lengths that are exact K-tile multiples give every request the
    same full-tile domain.  That specialization needs no runtime ``seqlen_k``
    mask; mixed lengths or a partial final tile retain the general mask.
    """

    return (
        packed
        and mask_type == "dense"
        and (
            any(length != k_lengths[0] for length in k_lengths[1:])
            or k_lengths[0] % _CONTEXT_KV_TILE_N != 0
        )
    )


def _paged_context_uses_persistent_scheduler(
    *,
    mask_type: str,
    head_paired: bool,
    logical_work_tiles: int,
    max_active_clusters: int,
    batch_size: int,
    num_qo_heads: int,
) -> bool:
    """Return whether paged context should use the static persistent grid.

    A one-wave dense query-paired grid has no inter-wave launch imbalance to
    amortize, so it launches one CTA per logical tile.  Larger grids use the
    persistent scheduler to balance work across resident CTAs.  CUDA limits
    grid Y and Z to 65,535; retaining persistent scheduling for oversized
    public geometries keeps every otherwise-valid int32 plan launchable.
    """

    return (
        mask_type == "causal"
        or head_paired
        or logical_work_tiles > max_active_clusters
        or batch_size > _CUDA_GRID_YZ_MAX
        or num_qo_heads > _CUDA_GRID_YZ_MAX
    )


def _resolve_geometry(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    qo_indptr: Optional[torch.Tensor],
    kv_indptr: Optional[torch.Tensor],
    mask_type: str,
    window_left: int,
    output_dtype: torch.dtype,
) -> _ContextGeometry:
    """Validate one-shot inputs and derive their static wrapper bounds."""

    _validate_base_tensors(q, k, v)
    _validate_output_dtype(output_dtype)
    _validate_mask(mask_type)
    window_left = _validate_window_left(window_left, mask_type)
    device_index = _validate_device(q.device)
    device = torch.device("cuda", device_index)
    if (qo_indptr is None) != (kv_indptr is None):
        raise ValueError("qo_indptr and kv_indptr must be provided together")
    packed = qo_indptr is not None

    if packed:
        if q.ndim != 3 or k.ndim != 3:
            raise ValueError(
                "packed Q/K/V must use [total_tokens, H, D] storage; "
                f"got q rank {q.ndim} and k rank {k.ndim}"
            )
        total_q, num_qo_heads, q_head_dim = map(int, q.shape)
        total_k, num_kv_heads, kv_head_dim = map(int, k.shape)
        _validate_padded_data_extent(total_q, "total_q")
        _validate_padded_data_extent(total_k, "total_k")
        _validate_compact(q, "q", "[total_q, Hq, D]")
        _validate_compact(k, "k", "[total_k, Hkv, D]")
        _validate_compact(v, "v", "[total_k, Hkv, D]")
        assert qo_indptr is not None and kv_indptr is not None
        _validate_indptr_tensor(qo_indptr, "qo_indptr", device=device)
        _validate_indptr_tensor(kv_indptr, "kv_indptr", device=device)
        if qo_indptr.numel() != kv_indptr.numel():
            raise ValueError(
                "qo_indptr and kv_indptr must describe the same batch; "
                f"got {qo_indptr.numel() - 1} and {kv_indptr.numel() - 1} rows"
            )
        _, q_lengths = _read_indptr(qo_indptr, "qo_indptr", expected_total=total_q)
        _, k_lengths = _read_indptr(kv_indptr, "kv_indptr", expected_total=total_k)
        batch_size = len(q_lengths)
        _validate_extent(batch_size, "batch_size")
        max_seq_len_q = max(q_lengths)
        max_seq_len_k = max(k_lengths)
        q_shape = tuple(q.shape)
        kv_shape = tuple(k.shape)
    else:
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError(
                "fixed Q/K/V must use [B, S, H, D] storage; "
                f"got q rank {q.ndim} and k rank {k.ndim}"
            )
        batch_size, max_seq_len_q, num_qo_heads, q_head_dim = map(int, q.shape)
        k_batch, max_seq_len_k, num_kv_heads, kv_head_dim = map(int, k.shape)
        if batch_size != k_batch:
            raise ValueError(
                f"q and k batch dimensions must match; got {batch_size} and {k_batch}"
            )
        _validate_extent(batch_size, "batch_size")
        _validate_extent(max_seq_len_q, "seq_len_q")
        _validate_extent(max_seq_len_k, "seq_len_k")
        total_q = batch_size * max_seq_len_q
        total_k = batch_size * max_seq_len_k
        _validate_padded_data_extent(total_q, "B * seq_len_q")
        _validate_padded_data_extent(total_k, "B * seq_len_k")
        _validate_compact(q, "q", "[B, Sq, Hq, D]")
        _validate_compact(k, "k", "[B, Sk, Hkv, D]")
        _validate_compact(v, "v", "[B, Sk, Hkv, D]")
        q_lengths = (max_seq_len_q,) * batch_size
        k_lengths = (max_seq_len_k,) * batch_size
        q_shape = tuple(q.shape)
        kv_shape = tuple(k.shape)

    _validate_head_dim(q_head_dim, kv_head_dim)
    head_ratio = _validate_head_geometry(num_qo_heads, num_kv_heads)
    if mask_type == "causal":
        for batch_idx, (q_length, k_length) in enumerate(
            zip(q_lengths, k_lengths, strict=True)
        ):
            if q_length > k_length:
                raise ValueError(
                    "bottom-right causal context requires Sq <= Sk for each "
                    f"request; got batch {batch_idx}: Sq={q_length}, Sk={k_length}"
                )

    head_paired = window_left > 0
    if head_paired and (head_ratio <= 1 or head_ratio % 2 != 0):
        raise NotImplementedError(
            "a positive left window requires grouped-query attention with an "
            f"even Hq/Hkv ratio greater than one; got {head_ratio}"
        )
    has_q_offset = _derive_has_q_offset(q_lengths, k_lengths, mask_type)
    uniform_packed_lengths = (
        packed
        and all(length == q_lengths[0] for length in q_lengths[1:])
        and all(length == k_lengths[0] for length in k_lengths[1:])
    )
    causal_single_kv_tile = (
        mask_type == "causal"
        and not packed
        and not head_paired
        and max_seq_len_q == max_seq_len_k
        and max_seq_len_k <= 128
    )
    packed_dense_k_mask = _needs_packed_dense_k_mask(
        packed=packed,
        mask_type=mask_type,
        k_lengths=k_lengths,
    )
    return _ContextGeometry(
        device=device,
        device_index=device_index,
        packed=packed,
        batch_size=batch_size,
        total_q=total_q,
        total_k=total_k,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_k=max_seq_len_k,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=q_head_dim,
        q_dtype=q.dtype,
        output_dtype=output_dtype,
        mask_type=mask_type,
        window_left=window_left,
        head_paired=head_paired,
        uniform_packed_lengths=uniform_packed_lengths,
        has_q_offset=has_q_offset,
        causal_single_kv_tile=causal_single_kv_tile,
        packed_dense_k_mask=packed_dense_k_mask,
        q_shape=q_shape,
        kv_shape=kv_shape,
    )


def _resolve_context_plan_geometry(
    *,
    device: int | str | torch.device,
    batch_size: int,
    max_seq_len_q: int,
    max_kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    packed: bool,
    mask_type: str,
    window_left: int,
    output_dtype: torch.dtype,
) -> _ContextPlanGeometry:
    """Validate explicit static bounds for a reusable contiguous plan."""

    _validate_context_dtype_pair(q_dtype, kv_dtype, output_dtype)
    _validate_mask(mask_type)
    window_left = _validate_window_left(window_left, mask_type)
    if not isinstance(packed, bool):
        raise TypeError("packed must be a bool")
    if packed and mask_type == "variable_window":
        raise NotImplementedError(
            "mask_type='variable_window' requires fixed [B, S, H, D] tensors"
        )

    batch_size = _validate_static_extent(batch_size, "batch_size")
    max_seq_len_q = _validate_static_extent(max_seq_len_q, "max_seq_len_q")
    max_kv_len = _validate_static_extent(max_kv_len, "max_kv_len")
    num_qo_heads = _validate_static_extent(num_qo_heads, "num_qo_heads")
    num_kv_heads = _validate_static_extent(num_kv_heads, "num_kv_heads")
    head_dim = _validate_static_extent(head_dim, "head_dim")
    device, device_index = _resolve_cuda_device(device)

    _validate_padded_data_extent(
        batch_size * max_seq_len_q, "batch_size * max_seq_len_q"
    )
    _validate_padded_data_extent(batch_size * max_kv_len, "batch_size * max_kv_len")
    head_ratio = _validate_head_geometry(num_qo_heads, num_kv_heads)
    _validate_head_dim(head_dim, head_dim)
    if not packed and mask_type == "causal" and max_seq_len_q > max_kv_len:
        raise ValueError(
            "bottom-right causal context requires max_seq_len_q <= max_kv_len; "
            f"got {max_seq_len_q} and {max_kv_len}"
        )
    head_paired = window_left > 0
    if head_paired and (head_ratio <= 1 or head_ratio % 2 != 0):
        raise NotImplementedError(
            "a positive left window requires grouped-query attention with an "
            f"even Hq/Hkv ratio greater than one; got {head_ratio}"
        )

    # Packed request lengths are run inputs, so select conservative compile
    # facts that remain valid for every request within the plan capacities.
    has_q_offset = mask_type == "causal" and (packed or max_seq_len_q != max_kv_len)
    return _ContextPlanGeometry(
        device=device,
        device_index=device_index,
        packed=packed,
        batch_size=batch_size,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_k=max_kv_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        output_dtype=output_dtype,
        mask_type=mask_type,
        window_left=window_left,
        head_paired=head_paired,
        uniform_packed_lengths=False,
        has_q_offset=has_q_offset,
        causal_single_kv_tile=(
            mask_type == "causal"
            and not packed
            and not head_paired
            and max_seq_len_q == max_kv_len
            and max_kv_len <= _CONTEXT_KV_TILE_N
        ),
        packed_dense_k_mask=packed and mask_type == "dense",
    )


def _resolve_paged_geometry(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    qo_indptr: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    page_size: int,
    mask_type: str,
    window_left: int,
    output_dtype: torch.dtype,
) -> _PagedContextPlanGeometry:
    """Validate one fixed-table paged launch and derive its plan bounds."""

    _validate_mask(mask_type)
    if mask_type == "variable_window":
        raise NotImplementedError(
            "mask_type='variable_window' is not supported for paged context"
        )
    _validate_base_tensors(q, k_cache, v_cache)
    window_left = _validate_window_left(window_left, mask_type)
    page_size = _validate_page_size(page_size)
    device_index = _validate_device(q.device)
    device = torch.device("cuda", device_index)

    if q.ndim != 3:
        raise ValueError(
            "paged context Q must use packed [total_q, Hq, D] storage; "
            f"got rank {q.ndim}"
        )
    if k_cache.ndim != 4:
        raise ValueError(
            "paged K/V caches must use [num_pages, Hkv, page_size, D] "
            f"storage; got rank {k_cache.ndim}"
        )
    total_q, num_qo_heads, q_head_dim = map(int, q.shape)
    num_physical_pages, num_kv_heads, cache_page_size, kv_head_dim = map(
        int, k_cache.shape
    )
    _validate_padded_data_extent(total_q, "total_q")
    _validate_extent(num_physical_pages, "num_physical_pages")
    if cache_page_size != page_size:
        raise ValueError(
            f"K/V cache page extent must equal page_size={page_size}; "
            f"got {cache_page_size}"
        )
    _validate_head_dim(q_head_dim, kv_head_dim)
    _validate_compact(q, "q", "[total_q, Hq, D]")
    _validate_compact(k_cache, "k_cache", "[num_pages, Hkv, page_size, D]")
    _validate_compact(v_cache, "v_cache", "[num_pages, Hkv, page_size, D]")

    _validate_indptr_tensor(qo_indptr, "qo_indptr", device=device)
    batch_size = int(qo_indptr.numel()) - 1
    _validate_extent(batch_size, "batch_size")
    _, q_lengths = _read_indptr(qo_indptr, "qo_indptr", expected_total=total_q)
    _validate_paged_metadata_tensor(seq_lens_kv, "seq_lens_kv", device=device)
    k_lengths = _read_int32_values(
        seq_lens_kv,
        "seq_lens_kv",
        expected_count=batch_size,
    )
    for batch_idx, k_length in enumerate(k_lengths):
        if k_length <= 0:
            raise ValueError(
                "seq_lens_kv entries must be positive; "
                f"batch {batch_idx} has {k_length}"
            )
    max_seq_len_q = max(q_lengths)
    max_kv_len = max(k_lengths)
    uniform_packed_lengths = all(
        length == max_seq_len_q for length in q_lengths
    ) and all(length == max_kv_len for length in k_lengths)
    has_q_offset = _derive_has_q_offset(q_lengths, k_lengths, mask_type)

    geometry = _resolve_paged_plan_geometry(
        device=device,
        batch_size=batch_size,
        max_seq_len_q=max_seq_len_q,
        max_kv_len=max_kv_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=q_head_dim,
        q_dtype=q.dtype,
        kv_dtype=k_cache.dtype,
        page_size=page_size,
        mask_type=mask_type,
        window_left=window_left,
        output_dtype=output_dtype,
        uniform_packed_lengths=uniform_packed_lengths,
        has_q_offset=has_q_offset,
    )
    _validate_paged_runtime_inputs(q, k_cache, v_cache, geometry)
    _validate_paged_runtime_metadata(
        qo_indptr,
        block_tables,
        seq_lens_kv,
        geometry,
        total_q=total_q,
        num_physical_pages=num_physical_pages,
    )
    return geometry


def _make_context_scheduler_probe(
    geometry: _ContextPlanGeometry | _PagedContextPlanGeometry,
    *,
    causal_single_kv_tile: bool,
    page_size: int | None = None,
    max_kv_len: int | None = None,
) -> _ContextSchedulerProbe:
    """Build the common static probe and logical work count for policy."""

    import cutlass
    import cutlass.utils as utils

    dtype_map = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    }
    probe = _make_context_kernel(
        input_dtype=dtype_map[geometry.q_dtype],
        output_dtype=dtype_map[geometry.output_dtype],
        head_dim=geometry.head_dim,
        mask_type=geometry.mask_type,
        window_left=geometry.window_left,
        head_paired=geometry.head_paired,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        has_q_offset=geometry.has_q_offset,
        causal_single_kv_tile=causal_single_kv_tile,
        scheduler="static_persistent",
        page_size=page_size,
        max_kv_len=max_kv_len,
    )
    with torch.cuda.device(geometry.device_index):
        max_active_clusters = int(utils.HardwareInfo().get_max_active_clusters(1))
    num_seq_tiles = (
        geometry.max_seq_len_q + probe.cfg.cta_tiler[0] - 1
    ) // probe.cfg.cta_tiler[0]
    num_head_tiles = geometry.num_qo_heads // probe.cfg.work_tile_q_heads
    logical_work_tiles = geometry.batch_size * num_seq_tiles * num_head_tiles
    return _ContextSchedulerProbe(
        single_qkv_instance=bool(probe.cfg.single_qkv_instance),
        logical_work_tiles=int(logical_work_tiles),
        max_active_clusters=max_active_clusters,
    )


def _resolve_paged_plan_geometry(
    *,
    device: int | str | torch.device,
    batch_size: int,
    max_seq_len_q: int,
    max_kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    page_size: int,
    mask_type: str,
    window_left: int,
    output_dtype: torch.dtype,
    uniform_packed_lengths: bool = False,
    has_q_offset: bool = True,
) -> _PagedContextPlanGeometry:
    """Validate explicit static bounds for a reusable paged specialization."""

    _validate_paged_dtype_pair(q_dtype, kv_dtype, output_dtype)
    _validate_mask(mask_type)
    if not isinstance(uniform_packed_lengths, bool):
        raise TypeError("uniform_packed_lengths must be a bool")
    if not isinstance(has_q_offset, bool):
        raise TypeError("has_q_offset must be a bool")
    if mask_type == "variable_window":
        raise NotImplementedError(
            "mask_type='variable_window' is not supported for paged context"
        )
    window_left = _validate_window_left(window_left, mask_type)
    page_size = _validate_page_size(page_size)
    batch_size = _validate_static_extent(batch_size, "batch_size")
    max_seq_len_q = _validate_static_extent(max_seq_len_q, "max_seq_len_q")
    max_kv_len = _validate_static_extent(max_kv_len, "max_kv_len")
    num_qo_heads = _validate_static_extent(num_qo_heads, "num_qo_heads")
    num_kv_heads = _validate_static_extent(num_kv_heads, "num_kv_heads")
    head_dim = _validate_static_extent(head_dim, "head_dim")
    device, device_index = _resolve_cuda_device(device)

    _validate_padded_data_extent(
        batch_size * max_seq_len_q, "batch_size * max_seq_len_q"
    )
    head_ratio = _validate_head_geometry(num_qo_heads, num_kv_heads)
    _validate_head_dim(head_dim, head_dim)
    head_paired = window_left > 0
    if head_paired and (head_ratio <= 1 or head_ratio % 2 != 0):
        raise NotImplementedError(
            "a positive left window requires grouped-query attention with an "
            f"even Hq/Hkv ratio greater than one; got {head_ratio}"
        )

    # A causal Q offset is not part of dense attention semantics. Canonicalize
    # it out of the compile identity even when the conservative public default
    # was used, so dense callers keep one cache entry and the leaner kernel.
    has_q_offset = mask_type == "causal" and has_q_offset

    return _PagedContextPlanGeometry(
        device=device,
        device_index=device_index,
        batch_size=batch_size,
        max_seq_len_q=max_seq_len_q,
        max_kv_len=max_kv_len,
        page_size=page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        output_dtype=output_dtype,
        mask_type=mask_type,
        window_left=window_left,
        head_paired=head_paired,
        uniform_packed_lengths=uniform_packed_lengths,
        has_q_offset=has_q_offset,
        packed_dense_k_mask=(
            mask_type == "dense"
            and (not uniform_packed_lengths or max_kv_len % _CONTEXT_KV_TILE_N != 0)
        ),
    )


def _resolve_context_scheduler(geometry: _ContextPlanGeometry) -> _ContextScheduler:
    """Select a scheduler from plan-time work while keeping batch out of JIT."""

    probe = _make_context_scheduler_probe(
        geometry,
        causal_single_kv_tile=geometry.causal_single_kv_tile,
    )
    is_persistent = _contiguous_context_uses_persistent_scheduler(
        single_qkv_instance=probe.single_qkv_instance,
        head_paired=geometry.head_paired,
        packed=geometry.packed,
        uniform_packed_lengths=geometry.uniform_packed_lengths,
        logical_work_tiles=probe.logical_work_tiles,
        max_active_clusters=probe.max_active_clusters,
        batch_size=geometry.batch_size,
        num_qo_heads=geometry.num_qo_heads,
        is_causal=geometry.mask_type == "causal",
        has_q_offset=geometry.has_q_offset,
    )
    if not is_persistent:
        return "nonpersistent"
    if _contiguous_context_uses_clc_scheduler(
        single_qkv_instance=probe.single_qkv_instance,
        head_paired=geometry.head_paired,
        packed=geometry.packed,
        uniform_packed_lengths=geometry.uniform_packed_lengths,
        is_causal=geometry.mask_type == "causal",
        has_q_offset=geometry.has_q_offset,
    ):
        return "clc_dynamic_persistent"
    return "static_persistent"


def _resolve_paged_context_scheduler(
    geometry: _PagedContextPlanGeometry,
) -> _ContextScheduler:
    """Select paged-context scheduling from plan work, not JIT identity."""

    probe = _make_context_scheduler_probe(
        geometry,
        causal_single_kv_tile=False,
        page_size=geometry.page_size,
        max_kv_len=geometry.max_kv_len,
    )
    is_persistent = _paged_context_uses_persistent_scheduler(
        mask_type=geometry.mask_type,
        head_paired=geometry.head_paired,
        logical_work_tiles=probe.logical_work_tiles,
        max_active_clusters=probe.max_active_clusters,
        batch_size=geometry.batch_size,
        num_qo_heads=geometry.num_qo_heads,
    )
    if not is_persistent:
        return "nonpersistent"
    if _paged_context_uses_clc_scheduler(
        is_persistent=True,
        single_qkv_instance=probe.single_qkv_instance,
        is_causal=geometry.mask_type == "causal",
        uniform_packed_lengths=geometry.uniform_packed_lengths,
    ):
        return "clc_dynamic_persistent"
    return "static_persistent"


def _context_compile_spec(geometry: _ContextPlanGeometry) -> _ContextCompileSpec:
    return _ContextCompileSpec(
        device_index=geometry.device_index,
        max_seq_len_q=geometry.max_seq_len_q,
        max_seq_len_k=geometry.max_seq_len_k,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        q_dtype_key=_dtype_key(geometry.q_dtype),
        output_dtype_key=_dtype_key(geometry.output_dtype),
        mask_type=geometry.mask_type,
        window_left=geometry.window_left,
        packed=geometry.packed,
        head_paired=geometry.head_paired,
        uniform_packed_lengths=geometry.uniform_packed_lengths,
        has_q_offset=geometry.has_q_offset,
        causal_single_kv_tile=geometry.causal_single_kv_tile,
        packed_dense_k_mask=geometry.packed_dense_k_mask,
        scheduler=_resolve_context_scheduler(geometry),
    )


def _paged_context_compile_spec(
    geometry: _PagedContextPlanGeometry,
) -> _PagedContextCompileSpec:
    return _PagedContextCompileSpec(
        device_index=geometry.device_index,
        max_seq_len_q=geometry.max_seq_len_q,
        max_kv_len=geometry.max_kv_len,
        page_size=geometry.page_size,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        q_dtype_key=_dtype_key(geometry.q_dtype),
        output_dtype_key=_dtype_key(geometry.output_dtype),
        mask_type=geometry.mask_type,
        window_left=geometry.window_left,
        head_paired=geometry.head_paired,
        uniform_packed_lengths=geometry.uniform_packed_lengths,
        has_q_offset=geometry.has_q_offset,
        packed_dense_k_mask=geometry.packed_dense_k_mask,
        scheduler=_resolve_paged_context_scheduler(geometry),
    )


@functools.cache
def _get_compiled_context(
    compile_spec: _ContextCompileSpec,
):
    """Compile one batch-dynamic context specialization and cache by topology."""

    device_index = compile_spec.device_index
    max_seq_len_q = compile_spec.max_seq_len_q
    max_seq_len_k = compile_spec.max_seq_len_k
    num_qo_heads = compile_spec.num_qo_heads
    num_kv_heads = compile_spec.num_kv_heads
    head_dim = compile_spec.head_dim
    q_dtype_key = compile_spec.q_dtype_key
    output_dtype_key = compile_spec.output_dtype_key
    mask_type = compile_spec.mask_type
    window_left = compile_spec.window_left
    packed = compile_spec.packed
    head_paired = compile_spec.head_paired
    uniform_packed_lengths = compile_spec.uniform_packed_lengths
    has_q_offset = compile_spec.has_q_offset
    causal_single_kv_tile = compile_spec.causal_single_kv_tile
    packed_dense_k_mask = compile_spec.packed_dense_k_mask
    scheduler = compile_spec.scheduler

    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda_drv
    import cutlass.utils as utils

    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
    }
    input_dtype = dtype_map[q_dtype_key]
    output_dtype = dtype_map[output_dtype_key]
    has_variable_window = mask_type == "variable_window"
    with torch.cuda.device(device_index):
        max_active_clusters = int(utils.HardwareInfo().get_max_active_clusters(1))
    fmha = _make_context_kernel(
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        head_dim=head_dim,
        mask_type=mask_type,
        window_left=window_left,
        head_paired=head_paired,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        has_q_offset=has_q_offset,
        causal_single_kv_tile=causal_single_kv_tile,
        scheduler=scheduler,
    )
    fmha.cfg.has_varlen = packed
    fmha.cfg.has_uniform_varlen = uniform_packed_lengths
    if uniform_packed_lengths:
        fmha.cfg.uniform_seq_len_q = max_seq_len_q
        fmha.cfg.uniform_seq_len_k = max_seq_len_k
    fmha.cfg.has_q_offset = has_q_offset
    if fmha.cfg.kv_tile_n != _CONTEXT_KV_TILE_N:
        raise RuntimeError(
            "context packed-K specialization assumes kv_tile_n="
            f"{_CONTEXT_KV_TILE_N}, got {fmha.cfg.kv_tile_n}"
        )
    _validate_query_work_tile_span(fmha.cfg)
    fmha.cfg.packed_dense_k_mask = packed_dense_k_mask
    if mask_type != "causal" and not packed:
        fmha.cfg.fixed_dense_k_tail = max_seq_len_k % fmha.cfg.kv_tile_n

    @cute.jit
    def tensor_adapter(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        out: cute.Tensor,
        scale_softmax_log2: cute.Tensor,
        output_scale: cute.Tensor,
        qo_indptr: cute.Tensor,
        kv_indptr: cute.Tensor,
        variable_window_token_starts: cute.Tensor,
        variable_window_token_ends: cute.Tensor,
        variable_window_cta_starts: cute.Tensor,
        stream: cuda_drv.CUstream,
        static_max_active_clusters: cutlass.Constexpr[int],
        static_packed: cutlass.Constexpr[bool],
        static_max_seq_len_q: cutlass.Constexpr[int],
        static_max_seq_len_k: cutlass.Constexpr[int],
    ) -> None:
        """Adapt torch TVM-FFI tensors to the FmhaTs host entry point."""

        if cutlass.const_expr(static_packed):
            fmha(
                q,
                k,
                v,
                out,
                scale_softmax_log2,
                output_scale,
                static_max_active_clusters,
                stream,
                qo_indptr,
                kv_indptr,
                cutlass.Int32(static_max_seq_len_q),
                cutlass.Int32(static_max_seq_len_k),
                variable_window_token_starts=variable_window_token_starts,
                variable_window_token_ends=variable_window_token_ends,
                variable_window_cta_starts=variable_window_cta_starts,
            )
        else:
            fmha(
                q,
                k,
                v,
                out,
                scale_softmax_log2,
                output_scale,
                static_max_active_clusters,
                stream,
                variable_window_token_starts=variable_window_token_starts,
                variable_window_token_ends=variable_window_token_ends,
                variable_window_cta_starts=variable_window_cta_starts,
            )

    def fake_compact(dtype, shape, assumed_align):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=assumed_align,
        )

    q_shape: tuple[object, ...]
    kv_shape: tuple[object, ...]
    out_shape: tuple[object, ...]
    if packed:
        runtime_total_q = cute.sym_int()
        runtime_total_k = cute.sym_int()
        runtime_num_q_offsets = cute.sym_int()
        runtime_num_k_offsets = cute.sym_int()
        q_shape = (runtime_total_q, num_qo_heads, head_dim)
        kv_shape = (runtime_total_k, num_kv_heads, head_dim)
        out_shape = (runtime_total_q, num_qo_heads, head_dim)
        qo_indptr_shape = (runtime_num_q_offsets,)
        kv_indptr_shape = (runtime_num_k_offsets,)
    else:
        batch_size = cute.sym_int()
        q_shape = (batch_size, max_seq_len_q, num_qo_heads, head_dim)
        kv_shape = (batch_size, max_seq_len_k, num_kv_heads, head_dim)
        out_shape = q_shape
        qo_indptr_shape = (1,)
        kv_indptr_shape = (1,)
    q_fake = fake_compact(input_dtype, q_shape, 16)
    k_fake = fake_compact(input_dtype, kv_shape, 16)
    v_fake = fake_compact(input_dtype, kv_shape, 16)
    out_fake = fake_compact(output_dtype, out_shape, 16)
    scale_fake = fake_compact(cutlass.Float32, (1,), 4)
    output_scale_fake = fake_compact(cutlass.Float32, (1,), 4)
    qo_indptr_fake = fake_compact(cutlass.Int32, qo_indptr_shape, 4)
    kv_indptr_fake = fake_compact(cutlass.Int32, kv_indptr_shape, 4)
    if has_variable_window:
        if packed:
            raise RuntimeError("variable-window context requires fixed tensors")
        variable_window_shape = (batch_size * max_seq_len_q,)
        variable_window_tile_size_q = (
            _CONTEXT_MAX_Q_ROWS_PER_WORK_TILE
            if head_dim == _CONTEXT_TILE_SIZE_Q
            else _CONTEXT_TILE_SIZE_Q
        )
        variable_window_cta_shape = (
            batch_size * cute.ceil_div(max_seq_len_q, variable_window_tile_size_q),
        )
    else:
        variable_window_shape = (1,)
        variable_window_cta_shape = (1,)
    variable_window_starts_fake = fake_compact(cutlass.Int32, variable_window_shape, 4)
    variable_window_ends_fake = fake_compact(cutlass.Int32, variable_window_shape, 4)
    variable_window_cta_starts_fake = fake_compact(
        cutlass.Int32, variable_window_cta_shape, 4
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Task objects carry loop-local state through generated control flow, so
    # select the public staged frontend for this compilation.
    with torch.cuda.device(device_index):
        compiled = cute.compile[cute.FrontendNext](
            tensor_adapter,
            q_fake,
            k_fake,
            v_fake,
            out_fake,
            scale_fake,
            output_scale_fake,
            qo_indptr_fake,
            kv_indptr_fake,
            variable_window_starts_fake,
            variable_window_ends_fake,
            variable_window_cta_starts_fake,
            stream_fake,
            max_active_clusters,
            packed,
            max_seq_len_q,
            max_seq_len_k,
            options=_COMPILE_OPTIONS,
        )
    policy = (
        ("scheduler", scheduler),
        ("pairing", "head" if head_paired else "query"),
        ("uniform_packed_lengths", uniform_packed_lengths),
        ("causal_single_kv_tile", causal_single_kv_tile),
        ("packed_dense_k_mask", packed_dense_k_mask),
    )
    return compiled, policy


@functools.cache
def _get_compiled_paged_context(
    compile_spec: _PagedContextCompileSpec,
):
    """Compile one batch-dynamic packed-Q, paged-K/V specialization."""

    device_index = compile_spec.device_index
    max_seq_len_q = compile_spec.max_seq_len_q
    max_kv_len = compile_spec.max_kv_len
    page_size = compile_spec.page_size
    num_qo_heads = compile_spec.num_qo_heads
    num_kv_heads = compile_spec.num_kv_heads
    head_dim = compile_spec.head_dim
    q_dtype_key = compile_spec.q_dtype_key
    output_dtype_key = compile_spec.output_dtype_key
    mask_type = compile_spec.mask_type
    window_left = compile_spec.window_left
    head_paired = compile_spec.head_paired
    uniform_packed_lengths = compile_spec.uniform_packed_lengths
    has_q_offset = compile_spec.has_q_offset
    packed_dense_k_mask = compile_spec.packed_dense_k_mask
    scheduler = compile_spec.scheduler

    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda_drv
    import cutlass.utils as utils

    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
    }
    input_dtype = dtype_map[q_dtype_key]
    output_dtype = dtype_map[output_dtype_key]
    with torch.cuda.device(device_index):
        max_active_clusters = int(utils.HardwareInfo().get_max_active_clusters(1))
    fmha = _make_context_kernel(
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        head_dim=head_dim,
        mask_type=mask_type,
        window_left=window_left,
        head_paired=head_paired,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        has_q_offset=has_q_offset,
        causal_single_kv_tile=False,
        scheduler=scheduler,
        page_size=page_size,
        max_kv_len=max_kv_len,
    )
    fmha.cfg.has_varlen = True
    fmha.cfg.has_uniform_varlen = uniform_packed_lengths
    if uniform_packed_lengths:
        fmha.cfg.uniform_seq_len_q = max_seq_len_q
        fmha.cfg.uniform_seq_len_k = max_kv_len
    fmha.cfg.has_q_offset = has_q_offset
    if fmha.cfg.kv_tile_n != _CONTEXT_KV_TILE_N:
        raise RuntimeError(
            "context packed-K specialization assumes kv_tile_n="
            f"{_CONTEXT_KV_TILE_N}, got {fmha.cfg.kv_tile_n}"
        )
    _validate_query_work_tile_span(fmha.cfg)
    fmha.cfg.packed_dense_k_mask = packed_dense_k_mask

    @cute.jit
    def tensor_adapter(
        q: cute.Tensor,
        k_cache: cute.Tensor,
        v_cache: cute.Tensor,
        out: cute.Tensor,
        scale_softmax_log2: cute.Tensor,
        output_scale: cute.Tensor,
        qo_indptr: cute.Tensor,
        block_tables: cute.Tensor,
        seq_lens_kv: cute.Tensor,
        stream: cuda_drv.CUstream,
        static_max_active_clusters: cutlass.Constexpr[int],
        static_max_seq_len_q: cutlass.Constexpr[int],
        static_max_kv_len: cutlass.Constexpr[int],
    ) -> None:
        fmha(
            q,
            k_cache,
            v_cache,
            out,
            scale_softmax_log2,
            output_scale,
            static_max_active_clusters,
            stream,
            cum_seqlen_q=qo_indptr,
            max_seqlen_q=cutlass.Int32(static_max_seq_len_q),
            max_seqlen_k=cutlass.Int32(static_max_kv_len),
            block_tables=block_tables,
            seq_lens_kv=seq_lens_kv,
        )

    def fake_compact(dtype, shape, assumed_align):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=assumed_align,
        )

    runtime_total_q = cute.sym_int()
    runtime_num_pages = cute.sym_int()
    batch_size = cute.sym_int()
    runtime_num_q_offsets = cute.sym_int()
    q_fake = fake_compact(input_dtype, (runtime_total_q, num_qo_heads, head_dim), 16)
    kv_shape = (
        runtime_num_pages,
        num_kv_heads,
        page_size,
        head_dim,
    )
    k_fake = fake_compact(input_dtype, kv_shape, 16)
    v_fake = fake_compact(input_dtype, kv_shape, 16)
    out_fake = fake_compact(output_dtype, (runtime_total_q, num_qo_heads, head_dim), 16)
    scale_fake = fake_compact(cutlass.Float32, (1,), 4)
    output_scale_fake = fake_compact(cutlass.Float32, (1,), 4)
    qo_indptr_fake = fake_compact(cutlass.Int32, (runtime_num_q_offsets,), 4)
    runtime_page_columns = cute.sym_int()
    runtime_block_table_row_stride = cute.sym_int()
    block_tables_fake = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (batch_size, runtime_page_columns),
        stride=(runtime_block_table_row_stride, 1),
        assumed_align=4,
    )
    seq_lens_fake = fake_compact(cutlass.Int32, (batch_size,), 4)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    with torch.cuda.device(device_index):
        compiled = cute.compile[cute.FrontendNext](
            tensor_adapter,
            q_fake,
            k_fake,
            v_fake,
            out_fake,
            scale_fake,
            output_scale_fake,
            qo_indptr_fake,
            block_tables_fake,
            seq_lens_fake,
            stream_fake,
            max_active_clusters,
            max_seq_len_q,
            max_kv_len,
            options=_COMPILE_OPTIONS,
        )
    policy = (
        ("scheduler", scheduler),
        ("pairing", "head" if head_paired else "query"),
        ("kv_layout", "paged_hnd"),
        ("page_size", page_size),
        ("uniform_packed_lengths", uniform_packed_lengths),
        ("has_q_offset", has_q_offset),
        ("causal_single_kv_tile", False),
        ("packed_dense_k_mask", packed_dense_k_mask),
    )
    return compiled, policy


def _validate_runtime_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    geometry: _ContextPlanGeometry,
    *,
    qo_indptr: Optional[torch.Tensor],
    kv_indptr: Optional[torch.Tensor],
) -> None:
    """Validate contiguous data and request metadata against one plan."""

    _validate_base_tensors(q, k, v)
    if q.device != geometry.device:
        raise ValueError(f"q must be on {geometry.device}, got {q.device}")
    if q.dtype != geometry.q_dtype:
        raise ValueError(f"q must have dtype {geometry.q_dtype}, got {q.dtype}")
    if k.dtype != geometry.kv_dtype:
        raise ValueError(f"k and v must have dtype {geometry.kv_dtype}, got {k.dtype}")

    if geometry.packed:
        if q.ndim != 3 or tuple(q.shape[1:]) != (
            geometry.num_qo_heads,
            geometry.head_dim,
        ):
            raise ValueError(
                "q must have packed shape [total_q, Hq, D] with "
                f"Hq/D=({geometry.num_qo_heads}, {geometry.head_dim}), got "
                f"{tuple(q.shape)}"
            )
        if k.ndim != 3 or tuple(k.shape[1:]) != (
            geometry.num_kv_heads,
            geometry.head_dim,
        ):
            raise ValueError(
                "k and v must have packed shape [total_k, Hkv, D] with "
                f"Hkv/D=({geometry.num_kv_heads}, {geometry.head_dim}), got "
                f"{tuple(k.shape)}"
            )
        total_q = int(q.shape[0])
        total_k = int(k.shape[0])
        _validate_padded_data_extent(total_q, "total_q")
        _validate_padded_data_extent(total_k, "total_k")
        if total_q < geometry.batch_size or total_q > (
            geometry.batch_size * geometry.max_seq_len_q
        ):
            raise ValueError(
                "q must contain between batch_size and "
                "batch_size * max_seq_len_q rows; got "
                f"{total_q} for batch_size={geometry.batch_size} and "
                f"max_seq_len_q={geometry.max_seq_len_q}"
            )
        if total_k < geometry.batch_size or total_k > (
            geometry.batch_size * geometry.max_seq_len_k
        ):
            raise ValueError(
                "k and v must contain between batch_size and "
                "batch_size * max_kv_len rows; got "
                f"{total_k} for batch_size={geometry.batch_size} and "
                f"max_kv_len={geometry.max_seq_len_k}"
            )
        if qo_indptr is None or kv_indptr is None:
            raise ValueError("packed plans require qo_indptr and kv_indptr in run()")
        request_lengths: dict[str, tuple[int, ...]] = {}
        for indptr, name, expected_total in (
            (qo_indptr, "qo_indptr", total_q),
            (kv_indptr, "kv_indptr", total_k),
        ):
            _validate_indptr_tensor(indptr, name, device=geometry.device)
            if indptr.numel() != geometry.batch_size + 1:
                raise ValueError(
                    f"{name} must contain batch_size + 1 elements "
                    f"({geometry.batch_size + 1}), got {indptr.numel()}"
                )
            _, lengths = _read_indptr(
                indptr,
                name,
                expected_total=expected_total,
            )
            request_lengths[name] = lengths
            capacity = (
                geometry.max_seq_len_q
                if name == "qo_indptr"
                else geometry.max_seq_len_k
            )
            for batch_idx, length in enumerate(lengths):
                if length > capacity:
                    raise ValueError(
                        f"{name} deltas must not exceed the planned capacity "
                        f"{capacity}; batch {batch_idx} has {length}"
                    )
        q_lengths = request_lengths["qo_indptr"]
        k_lengths = request_lengths["kv_indptr"]
        if geometry.mask_type == "causal":
            for batch_idx, (q_length, k_length) in enumerate(
                zip(q_lengths, k_lengths, strict=True)
            ):
                if q_length > k_length:
                    raise ValueError(
                        "bottom-right causal context requires Sq <= Sk for each "
                        f"request; got batch {batch_idx}: Sq={q_length}, "
                        f"Sk={k_length}"
                    )
        _validate_compact(q, "q", "[total_q, Hq, D]")
        _validate_compact(k, "k", "[total_k, Hkv, D]")
        _validate_compact(v, "v", "[total_k, Hkv, D]")
    else:
        if qo_indptr is not None or kv_indptr is not None:
            raise ValueError("qo_indptr and kv_indptr require packed=True in plan()")
        q_shape = (
            geometry.batch_size,
            geometry.max_seq_len_q,
            geometry.num_qo_heads,
            geometry.head_dim,
        )
        kv_shape = (
            geometry.batch_size,
            geometry.max_seq_len_k,
            geometry.num_kv_heads,
            geometry.head_dim,
        )
        if tuple(q.shape) != q_shape:
            raise ValueError(f"q must have shape {q_shape}, got {tuple(q.shape)}")
        if tuple(k.shape) != kv_shape:
            raise ValueError(
                f"k and v must have shape {kv_shape}, got {tuple(k.shape)}"
            )
        _validate_compact(q, "q", "[B, Sq, Hq, D]")
        _validate_compact(k, "k", "[B, Sk, Hkv, D]")
        _validate_compact(v, "v", "[B, Sk, Hkv, D]")


def _validate_paged_runtime_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    geometry: _PagedContextPlanGeometry,
) -> None:
    """Validate per-run data tensors without fixing packed/page extents."""

    _validate_base_tensors(q, k_cache, v_cache)
    if q.device != geometry.device:
        raise ValueError(f"q must be on {geometry.device}, got {q.device}")
    if q.dtype != geometry.q_dtype:
        raise ValueError(f"q must have dtype {geometry.q_dtype}, got {q.dtype}")
    if k_cache.dtype != geometry.kv_dtype:
        raise ValueError(
            f"k_cache must have dtype {geometry.kv_dtype}, got {k_cache.dtype}"
        )
    if q.ndim != 3 or tuple(q.shape[1:]) != (
        geometry.num_qo_heads,
        geometry.head_dim,
    ):
        raise ValueError(
            "q must have shape [total_q, Hq, D] with "
            f"Hq/D=({geometry.num_qo_heads}, {geometry.head_dim}), got {tuple(q.shape)}"
        )
    expected_kv_tail = (
        geometry.num_kv_heads,
        geometry.page_size,
        geometry.head_dim,
    )
    if k_cache.ndim != 4 or tuple(k_cache.shape[1:]) != expected_kv_tail:
        raise ValueError(
            "k_cache must have shape [num_pages, Hkv, page_size, D] with "
            f"Hkv/page/D={expected_kv_tail}, got {tuple(k_cache.shape)}"
        )
    total_q = int(q.shape[0])
    _validate_padded_data_extent(total_q, "total_q")
    if total_q < geometry.batch_size or total_q > (
        geometry.batch_size * geometry.max_seq_len_q
    ):
        raise ValueError(
            "q must contain between batch_size and batch_size * max_seq_len_q "
            f"rows; got {total_q} rows for batch_size={geometry.batch_size} and "
            f"max_seq_len_q={geometry.max_seq_len_q}"
        )
    _validate_extent(int(k_cache.shape[0]), "num_physical_pages")
    _validate_compact(q, "q", "[total_q, Hq, D]")
    _validate_compact(k_cache, "k_cache", "[num_pages, Hkv, page_size, D]")
    _validate_compact(v_cache, "v_cache", "[num_pages, Hkv, page_size, D]")


def _validate_paged_runtime_metadata(
    qo_indptr: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    geometry: _PagedContextPlanGeometry,
    *,
    total_q: int,
    num_physical_pages: int,
) -> None:
    """Validate per-run fixed-table storage and active metadata values."""

    _validate_indptr_tensor(qo_indptr, "qo_indptr", device=geometry.device)
    if qo_indptr.numel() != geometry.batch_size + 1:
        raise ValueError(
            "qo_indptr must contain batch_size + 1 elements "
            f"({geometry.batch_size + 1}), got {qo_indptr.numel()}"
        )

    required_page_columns = (
        geometry.max_kv_len + geometry.page_size - 1
    ) // geometry.page_size
    _validate_block_tables_tensor(
        block_tables,
        device=geometry.device,
        batch_size=geometry.batch_size,
        required_page_columns=required_page_columns,
    )

    _validate_paged_metadata_tensor(seq_lens_kv, "seq_lens_kv", device=geometry.device)
    if seq_lens_kv.numel() != geometry.batch_size:
        raise ValueError(
            "seq_lens_kv must contain one value per request; expected "
            f"{geometry.batch_size}, got {seq_lens_kv.numel()}"
        )

    _, q_lengths = _read_indptr(qo_indptr, "qo_indptr", expected_total=total_q)
    seq_lens = _read_int32_values(
        seq_lens_kv,
        "seq_lens_kv",
        expected_count=geometry.batch_size,
    )
    block_table_rows = block_tables.tolist()
    for batch_idx, (q_length, kv_length, block_table_row) in enumerate(
        zip(q_lengths, seq_lens, block_table_rows, strict=True)
    ):
        if q_length > geometry.max_seq_len_q:
            raise ValueError(
                "qo_indptr deltas must not exceed max_seq_len_q; "
                f"batch {batch_idx} has {q_length}, maximum is "
                f"{geometry.max_seq_len_q}"
            )
        if geometry.uniform_packed_lengths and q_length != geometry.max_seq_len_q:
            raise ValueError(
                "uniform_packed_lengths=True requires every qo_indptr delta "
                f"to equal max_seq_len_q={geometry.max_seq_len_q}; batch "
                f"{batch_idx} has {q_length}"
            )
        if kv_length <= 0 or kv_length > geometry.max_kv_len:
            raise ValueError(
                "seq_lens_kv entries must be positive and not exceed "
                f"max_kv_len={geometry.max_kv_len}; batch {batch_idx} has "
                f"{kv_length}"
            )
        if geometry.uniform_packed_lengths and kv_length != geometry.max_kv_len:
            raise ValueError(
                "uniform_packed_lengths=True requires every seq_lens_kv entry "
                f"to equal max_kv_len={geometry.max_kv_len}; batch "
                f"{batch_idx} has {kv_length}"
            )
        required_pages = (kv_length + geometry.page_size - 1) // geometry.page_size
        if geometry.mask_type == "causal" and q_length > kv_length:
            raise ValueError(
                "bottom-right causal context requires Sq <= Sk for each "
                f"request; got batch {batch_idx}: Sq={q_length}, Sk={kv_length}"
            )
        if (
            geometry.mask_type == "causal"
            and not geometry.has_q_offset
            and q_length != kv_length
        ):
            raise ValueError(
                "has_q_offset=False requires Sq == Sk for every causal "
                f"request; got batch {batch_idx}: Sq={q_length}, Sk={kv_length}"
            )
        for row_offset, page_idx in enumerate(block_table_row[:required_pages]):
            if page_idx < 0 or page_idx >= num_physical_pages:
                raise ValueError(
                    "active block_tables entries must index the physical "
                    f"page pool; batch {batch_idx}, row entry {row_offset} is "
                    f"{page_idx}, pool has {num_physical_pages} pages"
                )


def _validate_runtime_scale_tensor(
    scale: torch.Tensor,
    name: str,
    *,
    device: torch.device,
) -> None:
    """Validate one allocation-free per-run scale input."""

    _validate_tensor(scale, name)
    if scale.device != device:
        raise ValueError(f"{name} must be on {device}, got {scale.device}")
    if scale.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if tuple(scale.shape) != (1,):
        raise ValueError(f"{name} must have shape [1], got {tuple(scale.shape)}")
    _validate_compact(scale, name, "[1]")
    _validate_alignment(scale, name, 4)
    _validate_scale(scale.item(), name)


def _prepare_out(
    out: Optional[torch.Tensor],
    *,
    q: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if out is None:
        return torch.empty(tuple(q.shape), dtype=output_dtype, device=q.device)
    _validate_tensor(out, "out")
    if tuple(out.shape) != tuple(q.shape):
        raise ValueError(
            f"out must have shape {tuple(q.shape)}, got {tuple(out.shape)}"
        )
    if out.dtype != output_dtype:
        raise ValueError(f"out must have dtype {output_dtype}, got {out.dtype}")
    if out.device != q.device:
        raise ValueError(f"out must be on {q.device}, got {out.device}")
    layout = "[total_q, Hq, D]" if q.ndim == 3 else "[B, Sq, Hq, D]"
    _validate_compact(out, "out", layout)
    _validate_alignment(out, "out", 16)
    return out


class BatchPrefillTSWrapper:
    """Compile and reuse fixed or packed-ragged contiguous context attention.

    ``plan`` accepts only static compilation geometry. Q/K/V tensors, packed
    cumulative offsets, variable-window bounds, and optional scale overrides
    are supplied to ``run``. Packed offset values may change between launches
    while their request lengths stay within the planned capacities.

    Plan-time scalar defaults are stored as one-element device tensors.
    Variable-window plans also own private reduction scratch refreshed from
    the current bounds on every run, so launches through one such wrapper must
    not overlap across streams or captured graphs. With caller-owned output,
    ``validate=False`` performs no allocation or metadata readback and is
    CUDA-graph-capturable when the caller guarantees the complete runtime
    contract and stable addresses. Replanning replaces plan-owned tensors and
    invalidates graphs captured from the previous plan; all prior launches and
    replays must finish before ``plan`` is called again. Keep the wrapper and
    all captured runtime tensors alive until every graph using the current
    plan is destroyed. Before running on a CUDA stream that is not already
    ordered after the planning stream, the caller must establish that
    dependency.
    """

    @flashinfer_api
    def __init__(self) -> None:
        """Initialize an unplanned task-scheduled context-attention wrapper."""
        self._plan_state: Optional[_ContextPlanState] = None

    @flashinfer_api
    def plan(
        self,
        *,
        device: int | str | torch.device,
        batch_size: int,
        max_seq_len_q: int,
        max_kv_len: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        out_dtype: Optional[torch.dtype] = None,
        packed: bool = False,
        mask_type: Literal["dense", "causal", "variable_window"] = "dense",
        window_left: int = -1,
        sm_scale: Optional[float] = None,
        output_scale: float = 1.0,
    ) -> None:
        """Compile one reusable specialization from explicit static geometry.

        Packed plans conservatively support any positive per-request Q and K/V
        lengths bounded by ``max_seq_len_q`` and ``max_kv_len``. Fixed plans
        use those bounds as their exact tensor extents. ``batch_size`` is exact.
        Calling ``plan`` again replaces all plan-owned tensors and invalidates
        CUDA graphs captured from the previous plan. Complete every launch and
        replay that uses the previous plan before replanning. Planning
        allocates and compiles; complete it before CUDA Graph capture.

        Parameters
        ----------
        device : int, str, or torch.device
            CUDA device on which the specialization is compiled and run.
        batch_size : int
            Exact number of requests in every run.
        max_seq_len_q : int
            Maximum query length of any request.
        max_kv_len : int
            Maximum K/V length of any request.
        num_qo_heads : int
            Number of query/output heads.
        num_kv_heads : int
            Number of key/value heads.
        head_dim : int
            Query, key, value, and output head dimension.
        q_dtype : torch.dtype
            Query dtype.
        kv_dtype : torch.dtype
            Key/value dtype. It must currently equal ``q_dtype``.
        out_dtype : torch.dtype, optional
            Output dtype; defaults to ``q_dtype``.
        packed : bool
            Whether run tensors use packed ``[total_tokens, H, D]`` storage.
        mask_type : {"dense", "causal", "variable_window"}
            Attention mask mode. ``variable_window`` is supported only for
            fixed-shape inputs.
        window_left : int
            Left sliding-window extent, or ``-1`` to disable the window.
        sm_scale : float, optional
            Softmax scale; defaults to the inverse square root of head size.
        output_scale : float
            Scale applied to the attention output.
        """

        resolved_out_dtype = q_dtype if out_dtype is None else out_dtype
        geometry = _resolve_context_plan_geometry(
            device=device,
            batch_size=batch_size,
            max_seq_len_q=max_seq_len_q,
            max_kv_len=max_kv_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            packed=packed,
            mask_type=mask_type,
            window_left=window_left,
            output_dtype=resolved_out_dtype,
        )
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(geometry.head_dim)
        sm_scale = _validate_scale(sm_scale, "sm_scale")
        output_scale = _validate_scale(output_scale, "output_scale")
        scale_softmax_log2 = _validate_scale(
            sm_scale * math.log2(math.e), "sm_scale * log2(e)"
        )
        scale_tensor = torch.tensor(
            [scale_softmax_log2], dtype=torch.float32, device=geometry.device
        )
        output_scale_tensor = torch.tensor(
            [output_scale], dtype=torch.float32, device=geometry.device
        )
        empty_i32 = torch.empty(1, dtype=torch.int32, device=geometry.device)
        if geometry.mask_type == "variable_window":
            tile_size_q = (
                _CONTEXT_MAX_Q_ROWS_PER_WORK_TILE
                if geometry.head_dim == _CONTEXT_TILE_SIZE_Q
                else _CONTEXT_TILE_SIZE_Q
            )
            num_seq_tiles = (geometry.max_seq_len_q + tile_size_q - 1) // tile_size_q
            padded_rows = num_seq_tiles * tile_size_q
            variable_window_cta_starts = torch.empty(
                geometry.batch_size * num_seq_tiles,
                dtype=torch.int32,
                device=geometry.device,
            )
            variable_window_padded_starts = (
                torch.empty(
                    (geometry.batch_size, padded_rows),
                    dtype=torch.int32,
                    device=geometry.device,
                )
                if padded_rows != geometry.max_seq_len_q
                else None
            )
        else:
            variable_window_cta_starts = empty_i32
            variable_window_padded_starts = None
        # Keep all runtime tensor allocation ahead of CUTLASS JIT. Besides
        # making plan publication atomic, this lets compute-sanitizer patch the
        # generated attention kernel without interleaving later PyTorch setup
        # launches with the DSL compiler/runtime callbacks.
        compiled, policy = _get_compiled_context(_context_compile_spec(geometry))
        self._plan_state = _ContextPlanState(
            geometry=geometry,
            scale_softmax_log2=scale_tensor,
            output_scale=output_scale_tensor,
            empty_i32=empty_i32,
            variable_window_padded_starts=variable_window_padded_starts,
            variable_window_cta_starts=variable_window_cta_starts,
            compiled=compiled,
            policy=policy,
        )

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qo_indptr: Optional[torch.Tensor] = None,
        kv_indptr: Optional[torch.Tensor] = None,
        *,
        variable_window_token_starts: Optional[torch.Tensor] = None,
        variable_window_token_ends: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        scale_softmax_log2: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        validate: bool = True,
    ) -> torch.Tensor:
        """Launch the compiled specialization with per-request tensors.

        Parameters
        ----------
        q, k, v : torch.Tensor
            Runtime query, key, and value tensors matching the plan.
        qo_indptr, kv_indptr : torch.Tensor, optional
            Cumulative packed Q and K/V offsets. Required by packed plans and
            forbidden by fixed plans.
        variable_window_token_starts, variable_window_token_ends : torch.Tensor, optional
            Inclusive K-token bounds for every fixed Q row. Required only for
            ``mask_type='variable_window'`` and shaped ``[B, max_seq_len_q]``.
        out : torch.Tensor, optional
            Caller-owned output tensor. A new tensor is allocated when omitted.
        scale_softmax_log2 : torch.Tensor, optional
            Per-run one-element float32 softmax scale in base-2 form. Defaults
            to the scale retained by the plan.
        output_scale : torch.Tensor, optional
            Per-run one-element float32 output scale. Defaults to the plan.
        validate : bool
            Validate the complete runtime contract. Disable only when the
            caller guarantees dtype, device, shape, stride, alignment, values,
            aliasing, and lifetime. Defaults to ``True``.
        """

        state = self._plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool")
        geometry = state.geometry
        if geometry.packed:
            if qo_indptr is None or kv_indptr is None:
                raise ValueError(
                    "packed plans require qo_indptr and kv_indptr in run()"
                )
            runtime_qo_indptr = qo_indptr
            runtime_kv_indptr = kv_indptr
        else:
            if qo_indptr is not None or kv_indptr is not None:
                raise ValueError(
                    "qo_indptr and kv_indptr require packed=True in plan()"
                )
            runtime_qo_indptr = state.empty_i32
            runtime_kv_indptr = state.empty_i32

        if validate:
            _validate_runtime_inputs(
                q,
                k,
                v,
                geometry,
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
            )
        if geometry.mask_type == "variable_window":
            if (
                variable_window_token_starts is None
                or variable_window_token_ends is None
            ):
                raise ValueError(
                    "variable-window plans require start and end bounds in run()"
                )
            if validate:
                runtime_window_starts, runtime_window_ends = (
                    _validate_variable_window_bounds(
                        variable_window_token_starts,
                        variable_window_token_ends,
                        geometry=geometry,
                    )
                )
            else:
                runtime_window_starts = variable_window_token_starts.flatten()
                runtime_window_ends = variable_window_token_ends.flatten()
            runtime_window_cta_starts = _refresh_variable_window_cta_starts(
                runtime_window_starts,
                state=state,
            )
        else:
            if (
                variable_window_token_starts is not None
                or variable_window_token_ends is not None
            ):
                raise ValueError(
                    "variable-window bounds require mask_type='variable_window'"
                )
            runtime_window_starts = state.empty_i32
            runtime_window_ends = state.empty_i32
            runtime_window_cta_starts = state.empty_i32
        if scale_softmax_log2 is None:
            scale_softmax_log2 = state.scale_softmax_log2
        elif validate:
            _validate_runtime_scale_tensor(
                scale_softmax_log2,
                "scale_softmax_log2",
                device=geometry.device,
            )
        if output_scale is None:
            output_scale = state.output_scale
        elif validate:
            _validate_runtime_scale_tensor(
                output_scale,
                "output_scale",
                device=geometry.device,
            )

        caller_provided_out = out is not None
        if out is None:
            out = torch.empty(
                tuple(q.shape), dtype=geometry.output_dtype, device=q.device
            )
        elif validate:
            out = _prepare_out(out, q=q, output_dtype=geometry.output_dtype)
        if validate and caller_provided_out:
            alias_inputs = [
                ("q", q),
                ("k", k),
                ("v", v),
                ("scale_softmax_log2", scale_softmax_log2),
                ("output_scale", output_scale),
            ]
            if geometry.packed:
                alias_inputs.extend(
                    (("qo_indptr", runtime_qo_indptr), ("kv_indptr", runtime_kv_indptr))
                )
            if geometry.mask_type == "variable_window":
                alias_inputs.extend(
                    (
                        ("variable_window_token_starts", runtime_window_starts),
                        ("variable_window_token_ends", runtime_window_ends),
                        ("variable_window_cta_starts", runtime_window_cta_starts),
                    )
                )
                if state.variable_window_padded_starts is not None:
                    alias_inputs.append(
                        (
                            "variable_window_padded_starts",
                            state.variable_window_padded_starts,
                        )
                    )
            _validate_out_does_not_overlap_inputs(
                out,
                *alias_inputs,
            )
        state.compiled(
            q,
            k,
            v,
            out,
            scale_softmax_log2,
            output_scale,
            runtime_qo_indptr,
            runtime_kv_indptr,
            runtime_window_starts,
            runtime_window_ends,
            runtime_window_cta_starts,
        )
        return out


class BatchPrefillPagedTSWrapper:
    """Compile and reuse packed-Q context attention over HND paged K/V.

    ``plan`` accepts only static compilation geometry. All request metadata is
    supplied to ``run``: cumulative Q offsets, a fixed row-strided K/V page
    table, and logical K/V lengths. Metadata values may change between launches
    while staying within the planned capacities. The wrapper owns no context
    workspace and does not transform the fixed page table.

    Plan-time scalar defaults are stored as one-element device tensors.
    ``run`` may replace either scale tensor without changing the compiled
    specialization. Plans default to a conservative dynamic-length contract;
    callers may instead promise exact-uniform packed lengths or no causal Q
    offset to compile one narrower specialization. Validation reads request
    metadata back to the host, checks those promises, and may synchronize. With
    caller-owned output, ``validate=False`` performs no allocation, metadata
    readback, or synchronization and is suitable for CUDA graph capture only
    when the caller already enforces the selected contract.
    Replanning invalidates captured graphs; prior launches and replays must
    finish before ``plan`` is called again. Keep the wrapper and all captured
    runtime tensors alive until every graph using the current plan is destroyed.
    Before running on a CUDA stream that is not already ordered after the
    planning stream, the caller must establish that dependency.
    """

    @flashinfer_api
    def __init__(self, kv_layout: Literal["HND"] = "HND") -> None:
        """Create an unplanned paged-context wrapper.

        Parameters
        ----------
        kv_layout : {"HND"}
            K/V layout. Only ``"HND"`` is supported.
        """

        _validate_kv_layout(kv_layout)
        self._kv_layout = kv_layout
        self._plan_state: Optional[_PagedContextPlanState] = None

    @flashinfer_api
    def plan(
        self,
        *,
        device: int | str | torch.device,
        batch_size: int,
        max_seq_len_q: int,
        max_kv_len: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        out_dtype: Optional[torch.dtype] = None,
        page_size: int = _DEFAULT_PAGED_KV_PAGE_SIZE,
        mask_type: Literal["dense", "causal"] = "dense",
        window_left: int = -1,
        sm_scale: Optional[float] = None,
        output_scale: float = 1.0,
        uniform_packed_lengths: bool = False,
        has_q_offset: bool = True,
    ) -> None:
        """Compile one reusable specialization from explicit static geometry.

        Sequence lengths and page indices are deliberately absent. By default,
        the plan supports dynamic positive lengths bounded by
        ``max_seq_len_q`` and ``max_kv_len``. ``uniform_packed_lengths=True``
        instead promises that every runtime Q length equals ``max_seq_len_q``
        and every runtime K/V length equals ``max_kv_len``.
        ``has_q_offset=False`` promises that every causal request has
        ``Sq == Sk``. Dense attention ignores and canonicalizes the latter
        flag. The selected contract compiles exactly one specialization and is
        checked by ``run(validate=True)``. ``validate=False`` skips the checks,
        so violating either promise can produce incorrect results or invalid
        memory accesses.

        ``batch_size`` is exact. Runtime page-table rows must expose at least
        ``ceil(max_kv_len / page_size)`` columns, including inactive padding
        columns. Calling ``plan`` again replaces plan-owned tensors and
        invalidates CUDA graphs captured from the previous plan. Complete every
        prior launch and replay before replanning. Planning allocates and
        compiles; complete it before CUDA Graph capture.

        Parameters
        ----------
        device : int, str, or torch.device
            CUDA device on which the specialization is compiled and run.
        batch_size : int
            Exact number of requests in every run.
        max_seq_len_q : int
            Maximum query length of any request.
        max_kv_len : int
            Maximum logical K/V length of any request.
        num_qo_heads : int
            Number of query/output heads.
        num_kv_heads : int
            Number of key/value heads. ``num_qo_heads`` must be divisible by
            this value.
        head_dim : int
            Query, key, value, and output head dimension.
        q_dtype : torch.dtype
            Query dtype.
        kv_dtype : torch.dtype
            Key/value cache dtype. It must currently equal ``q_dtype``.
        out_dtype : torch.dtype, optional
            Output dtype; defaults to ``q_dtype``.
        page_size : int
            Number of K/V tokens stored in each page.
        mask_type : {"dense", "causal"}
            Attention mask mode.
        window_left : int
            Left sliding-window extent, or ``-1`` to disable the window.
        sm_scale : float, optional
            Softmax scale; defaults to the inverse square root of head size.
        output_scale : float
            Scale applied to the attention output.
        uniform_packed_lengths : bool
            Whether every run has Q lengths exactly ``max_seq_len_q`` and K/V
            lengths exactly ``max_kv_len``. Defaults to ``False``.
        has_q_offset : bool
            Whether causal runs may have a nonzero bottom-right Q offset
            ``Sk - Sq``. Setting this to ``False`` promises ``Sq == Sk`` for
            every request. Dense attention ignores this flag. Defaults to
            ``True``.
        """

        resolved_out_dtype = q_dtype if out_dtype is None else out_dtype
        geometry = _resolve_paged_plan_geometry(
            device=device,
            batch_size=batch_size,
            max_seq_len_q=max_seq_len_q,
            max_kv_len=max_kv_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            page_size=page_size,
            mask_type=mask_type,
            window_left=window_left,
            output_dtype=resolved_out_dtype,
            uniform_packed_lengths=uniform_packed_lengths,
            has_q_offset=has_q_offset,
        )
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(geometry.head_dim)
        sm_scale = _validate_scale(sm_scale, "sm_scale")
        output_scale = _validate_scale(output_scale, "output_scale")
        scale_softmax_log2 = _validate_scale(
            sm_scale * math.log2(math.e), "sm_scale * log2(e)"
        )
        scale_tensor = torch.tensor(
            [scale_softmax_log2], dtype=torch.float32, device=geometry.device
        )
        output_scale_tensor = torch.tensor(
            [output_scale], dtype=torch.float32, device=geometry.device
        )

        # Keep the two plan-owned scalar allocations ahead of CUTLASS JIT,
        # matching the compute-sanitizer-safe ordering of the other wrapper.
        compiled, policy = _get_compiled_paged_context(
            _paged_context_compile_spec(geometry)
        )
        # Publish one immutable state object only after every fallible step.
        self._plan_state = _PagedContextPlanState(
            geometry=geometry,
            scale_softmax_log2=scale_tensor,
            output_scale=output_scale_tensor,
            compiled=compiled,
            policy=policy,
        )

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        scale_softmax_log2: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        validate: bool = True,
    ) -> torch.Tensor:
        """Launch the compiled specialization with per-run request metadata.

        Metadata must use int32 CUDA storage. ``qo_indptr`` is compact with
        shape ``[B + 1]``; ``block_tables`` has shape ``[B, C]``, unit column
        stride, and row stride at least ``C``; and ``seq_lens_kv`` is compact
        with shape ``[B]``. Their values must satisfy the capacities documented
        by ``plan``:

        * ``qo_indptr`` starts at zero, increases strictly, ends at the packed
          Q extent, and has deltas no greater than ``max_seq_len_q``;
        * ``C`` is at least ``ceil(max_kv_len / page_size)``;
        * ``seq_lens_kv`` entries are positive and do not exceed
          ``max_kv_len``;
        * causal launches additionally satisfy ``Sq[b] <= Sk[b]``; and
        * every active page-table entry is a valid physical page ID in both
          separate, isomorphic K and V pools. Padding entries beyond the
          logical K/V length are not dereferenced.

        When ``plan`` selected ``uniform_packed_lengths=True``, every Q delta
        must equal ``max_seq_len_q`` and every K/V length must equal
        ``max_kv_len``. For a causal plan with ``has_q_offset=False``, every
        request must additionally satisfy ``Sq[b] == Sk[b]``.

        With ``validate=True``, the host reads these values and may synchronize.
        Metadata may change only between completed launches or graph replays;
        CUDA graph capture additionally requires stable tensor shapes, strides,
        and addresses plus ``validate=False``. That mode does not verify the
        plan-time metadata promises; the caller is responsible for preserving
        them across every replay.

        Parameters
        ----------
        q : torch.Tensor
            Runtime packed query tensor matching the plan.
        k_cache : torch.Tensor
            Runtime HND key page pool matching the plan.
        v_cache : torch.Tensor
            Runtime HND value page pool matching the plan.
        qo_indptr : torch.Tensor
            Cumulative packed-query offsets with shape ``[B + 1]``.
        block_tables : torch.Tensor
            Physical page IDs with shape ``[B, C]``. Column stride must be one
            and row stride may exceed ``C``.
        seq_lens_kv : torch.Tensor
            Logical K/V lengths with shape ``[B]``.
        out : torch.Tensor, optional
            Caller-owned output tensor. A new tensor is allocated when omitted.
        scale_softmax_log2 : torch.Tensor, optional
            Per-run one-element float32 softmax scale in base-2 form. Defaults
            to the scale retained by the plan.
        output_scale : torch.Tensor, optional
            Per-run one-element float32 output scale. Defaults to the scale
            retained by the plan.
        validate : bool
            Run explicit storage, shape, dtype, device, scale, output, and
            aliasing validators. Disable only when the caller guarantees the
            complete runtime contract. Defaults to ``True``.

        Returns
        -------
        torch.Tensor
            The packed attention output.
        """

        state = self._plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool")
        geometry = state.geometry
        if validate:
            _validate_paged_runtime_inputs(q, k_cache, v_cache, geometry)
            _validate_paged_runtime_metadata(
                qo_indptr,
                block_tables,
                seq_lens_kv,
                geometry,
                total_q=int(q.shape[0]),
                num_physical_pages=int(k_cache.shape[0]),
            )
        if scale_softmax_log2 is None:
            scale_softmax_log2 = state.scale_softmax_log2
        elif validate:
            _validate_runtime_scale_tensor(
                scale_softmax_log2,
                "scale_softmax_log2",
                device=geometry.device,
            )
        if output_scale is None:
            output_scale = state.output_scale
        elif validate:
            _validate_runtime_scale_tensor(
                output_scale, "output_scale", device=geometry.device
            )

        caller_provided_out = out is not None
        if out is None:
            out = torch.empty(
                tuple(q.shape), dtype=geometry.output_dtype, device=q.device
            )
        elif validate:
            out = _prepare_out(out, q=q, output_dtype=geometry.output_dtype)
        if validate and caller_provided_out:
            _validate_out_does_not_overlap_inputs(
                out,
                ("q", q),
                ("k_cache", k_cache),
                ("v_cache", v_cache),
                ("qo_indptr", qo_indptr),
                ("block_tables", block_tables),
                ("seq_lens_kv", seq_lens_kv),
                ("scale_softmax_log2", scale_softmax_log2),
                ("output_scale", output_scale),
            )
        state.compiled(
            q,
            k_cache,
            v_cache,
            out,
            scale_softmax_log2,
            output_scale,
            qo_indptr,
            block_tables,
            seq_lens_kv,
        )
        return out


@flashinfer_api
def batch_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    qo_indptr: Optional[torch.Tensor] = None,
    kv_indptr: Optional[torch.Tensor] = None,
    mask_type: Literal["dense", "causal", "variable_window"] = "dense",
    window_left: int = -1,
    variable_window_token_starts: Optional[torch.Tensor] = None,
    variable_window_token_ends: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    output_scale: float = 1.0,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run one-shot fixed or packed-ragged task-scheduled context attention.

    Fixed tensors use ``[B, S, H, D]`` storage. Providing both cumulative
    int32 offset tensors selects packed ``[total_tokens, H, D]`` storage.
    ``D`` may be 128 or 256.
    Causal masking is bottom-right aligned.  ``window_left=-1`` disables the
    left window; a positive value selects the private head-paired GQA policy
    and retains at most ``window_left + 1`` keys at each causal row, including
    when ``S_q < S_kv``.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Fixed or packed query, key, and value tensors.
    qo_indptr, kv_indptr : torch.Tensor, optional
        Cumulative query and K/V offsets for packed-ragged input.
    mask_type : {"dense", "causal", "variable_window"}
        Attention mask mode. ``variable_window`` is supported only for
        fixed-shape inputs.
    window_left : int
        Left sliding-window extent, or ``-1`` to disable the window.
    variable_window_token_starts, variable_window_token_ends : torch.Tensor, optional
        Inclusive per-query K bounds required for ``variable_window``. Both
        must be CUDA int32 tensors shaped ``[B, Sq]`` and satisfy
        ``0 <= starts[b, q] <= ends[b, q] < Sk``.
    sm_scale : float, optional
        Softmax scale; defaults to the inverse square root of head size.
    output_scale : float
        Scale applied to the attention output.
    out_dtype : torch.dtype, optional
        Output dtype; defaults to ``out.dtype`` when ``out`` is provided,
        otherwise the query dtype.
    out : torch.Tensor, optional
        Caller-owned output tensor.

    Returns
    -------
    torch.Tensor
        The fixed or packed attention output.
    """

    resolved_out_dtype = (
        out.dtype if out_dtype is None and isinstance(out, torch.Tensor) else out_dtype
    )
    if resolved_out_dtype is None:
        if not isinstance(q, torch.Tensor):
            raise TypeError("q must be a torch.Tensor")
        resolved_out_dtype = q.dtype
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "batch_prefill cannot derive host plan bounds during CUDA graph "
            "capture; plan BatchPrefillTSWrapper before capture"
        )
    geometry = _resolve_geometry(
        q,
        k,
        v,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        mask_type=mask_type,
        window_left=window_left,
        output_dtype=resolved_out_dtype,
    )
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        device=geometry.device,
        batch_size=geometry.batch_size,
        max_seq_len_q=geometry.max_seq_len_q,
        max_kv_len=geometry.max_seq_len_k,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        q_dtype=geometry.q_dtype,
        kv_dtype=geometry.q_dtype,
        out_dtype=geometry.output_dtype,
        packed=geometry.packed,
        mask_type=mask_type,
        window_left=window_left,
        sm_scale=sm_scale,
        output_scale=output_scale,
    )
    return wrapper.run(
        q,
        k,
        v,
        qo_indptr,
        kv_indptr,
        variable_window_token_starts=variable_window_token_starts,
        variable_window_token_ends=variable_window_token_ends,
        out=out,
    )


@flashinfer_api
def batch_prefill_with_paged_kv_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    *,
    page_size: int = _DEFAULT_PAGED_KV_PAGE_SIZE,
    kv_layout: Literal["HND"] = "HND",
    mask_type: Literal["dense", "causal"] = "dense",
    window_left: int = -1,
    sm_scale: Optional[float] = None,
    output_scale: float = 1.0,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run one-shot packed-Q context attention over separate HND page pools.

    Q/output use ``[total_q, Hq, D]`` storage and K/V each use
    ``[num_pages, Hkv, page_size, D]`` with page size 16, 32, 64, or 128.
    ``qo_indptr`` describes Q rows, ``block_tables`` supplies one fixed
    row-strided page table, and ``seq_lens_kv`` supplies logical K/V lengths.
    Physical page indices need not be identity ordered. ``D`` may be 128 or
    256; Q, K, and V must share one supported dtype.

    This convenience API reads request metadata on the host to derive exact
    plan bounds and is not CUDA-graph-capturable. Capture-sensitive callers
    should plan :class:`BatchPrefillPagedTSWrapper` before capture and bind the
    same metadata tensors directly in ``run``.

    Parameters
    ----------
    q : torch.Tensor
        Packed query tensor.
    k_cache : torch.Tensor
        HND key page pool.
    v_cache : torch.Tensor
        HND value page pool isomorphic to ``k_cache``.
    qo_indptr : torch.Tensor
        Cumulative packed-query offsets.
    block_tables : torch.Tensor
        Physical page IDs with row-strided shape ``[B, C]``.
    seq_lens_kv : torch.Tensor
        Logical K/V lengths with shape ``[B]``.
    page_size : int
        Number of K/V tokens stored in each page.
    kv_layout : {"HND"}
        Layout of the separate K and V page pools.
    mask_type : {"dense", "causal"}
        Attention mask mode.
    window_left : int
        Left sliding-window extent, or ``-1`` to disable the window.
    sm_scale : float, optional
        Softmax scale; defaults to the inverse square root of head size.
    output_scale : float
        Scale applied to the attention output.
    out_dtype : torch.dtype, optional
        Output dtype; defaults to ``out.dtype`` when ``out`` is provided,
        otherwise the query dtype.
    out : torch.Tensor, optional
        Caller-owned output tensor.

    Returns
    -------
    torch.Tensor
        The packed attention output.
    """

    resolved_out_dtype = (
        out.dtype if out_dtype is None and isinstance(out, torch.Tensor) else out_dtype
    )
    if resolved_out_dtype is None:
        if not isinstance(q, torch.Tensor):
            raise TypeError("q must be a torch.Tensor")
        resolved_out_dtype = q.dtype
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "batch_prefill_with_paged_kv_cache cannot derive host plan bounds "
            "during CUDA graph capture; plan BatchPrefillPagedTSWrapper before "
            "capture"
        )
    geometry = _resolve_paged_geometry(
        q,
        k_cache,
        v_cache,
        qo_indptr=qo_indptr,
        block_tables=block_tables,
        seq_lens_kv=seq_lens_kv,
        page_size=page_size,
        mask_type=mask_type,
        window_left=window_left,
        output_dtype=resolved_out_dtype,
    )
    wrapper = BatchPrefillPagedTSWrapper(kv_layout=kv_layout)
    wrapper.plan(
        device=geometry.device,
        batch_size=geometry.batch_size,
        max_seq_len_q=geometry.max_seq_len_q,
        max_kv_len=geometry.max_kv_len,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        q_dtype=geometry.q_dtype,
        kv_dtype=geometry.kv_dtype,
        out_dtype=geometry.output_dtype,
        page_size=page_size,
        mask_type=mask_type,
        window_left=window_left,
        sm_scale=sm_scale,
        output_scale=output_scale,
        uniform_packed_lengths=geometry.uniform_packed_lengths,
        has_q_offset=geometry.has_q_offset,
    )
    return wrapper.run(
        q,
        k_cache,
        v_cache,
        qo_indptr,
        block_tables,
        seq_lens_kv,
        out=out,
    )


__all__ = [
    "BatchPrefillPagedTSWrapper",
    "BatchPrefillTSWrapper",
    "batch_prefill",
    "batch_prefill_with_paged_kv_cache",
]
