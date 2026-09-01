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

The public surface intentionally exposes attention semantics, not scheduler
choices. Contiguous K/V uses persistent CLC scheduling. Paged plans compile
from static capacities and conservatively treat all request lengths as live.
Single-instance dense paged plans use a static persistent launch when their
logical CTA grid exceeds one resident wave; smaller grids launch one CTA per
tile. Causal and paired persistent paged plans use CLC. The private policy is
query-paired unless a positive left window requires head-paired GQA.
Causal windows are bottom-right aligned: for row ``q``, the inclusive right
position is ``q + (S_kv - S_q)`` and ``window_left`` is measured from that
position.

PrimTS context entry points are intentionally excluded from ``fi_trace`` for
now; unlike the decode APIs, their ``@flashinfer_api`` decorators do not
register trace templates.
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
_SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3))
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
    """Validated semantic and storage geometry for one reusable plan."""

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
class _PagedContextOneShotGeometry:
    """Validated geometry used only by the FlashInfer-CSR one-shot adapter."""

    device: torch.device
    batch_size: int
    max_seq_len_q: int
    max_seq_len_k: int
    page_size: int
    max_num_pages_per_seq_kv: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    q_dtype: torch.dtype
    output_dtype: torch.dtype


@dataclass(frozen=True)
class _PagedContextOneShotMetadata:
    """Host-validated CSR translation for one immediate paged launch."""

    kv_indptr: tuple[int, ...]
    seq_lens: tuple[int, ...]
    dense_page_indices: tuple[int, ...]


@dataclass(frozen=True)
class _PagedContextPlanGeometry:
    """Static compile geometry for one live-metadata paged plan."""

    device: torch.device
    device_index: int
    batch_size: int
    max_seq_len_q: int
    max_seq_len_k: int
    page_size: int
    max_num_pages_per_seq_kv: int
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
            "attention-ts context requires an SM100a/B200 or SM103a/B300 GPU; "
            f"device cuda:{device_index} has compute capability {capability}"
        )
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
    geometry: _ContextGeometry,
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
    return starts.flatten(), ends.flatten()


def _build_variable_window_cta_starts(
    starts: torch.Tensor, *, geometry: _ContextGeometry
) -> torch.Tensor:
    """Reduce fixed per-row starts into one minimum for each kernel Q CTA."""
    tile_size_q = (
        _CONTEXT_MAX_Q_ROWS_PER_WORK_TILE
        if geometry.head_dim == _CONTEXT_TILE_SIZE_Q
        else _CONTEXT_TILE_SIZE_Q
    )
    num_seq_tiles = (geometry.max_seq_len_q + tile_size_q - 1) // tile_size_q
    padded_rows = num_seq_tiles * tile_size_q
    starts_2d = starts.view(geometry.batch_size, geometry.max_seq_len_q)
    if padded_rows != geometry.max_seq_len_q:
        padded = torch.full(
            (geometry.batch_size, padded_rows),
            _INT32_MAX,
            dtype=torch.int32,
            device=geometry.device,
        )
        padded[:, : geometry.max_seq_len_q] = starts_2d
        starts_2d = padded
    return (
        starts_2d.view(geometry.batch_size, num_seq_tiles, tile_size_q)
        .amin(dim=-1)
        .flatten()
    )


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
    """Copy plan metadata once and validate strictly positive row lengths."""

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
    """Copy one plan-time metadata vector after validating its extent."""

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
    """Validate a plan and derive its semantic compile key.

    Packed cumulative offsets are copied to the host only here.  A successful
    plan owns their tensor storage and never synchronizes on the run path.
    """

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


def _resolve_paged_one_shot_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    page_size: int,
    mask_type: str,
    window_left: int,
    output_dtype: torch.dtype,
) -> tuple[_PagedContextOneShotGeometry, _PagedContextOneShotMetadata]:
    """Validate and translate one paged-context convenience launch.

    This adapter bridges the public FlashInfer CSR representation accepted by
    :func:`batch_prefill_with_paged_kv_cache` to the static geometry consumed
    by the wrapper's compile-only :meth:`BatchPrefillPagedTSWrapper.plan` and
    the dense native metadata consumed by its
    :meth:`~BatchPrefillPagedTSWrapper.run`.
    """

    _validate_mask(mask_type)
    if mask_type == "variable_window":
        raise NotImplementedError(
            "mask_type='variable_window' is not supported for paged context"
        )
    _validate_base_tensors(q, k_cache, v_cache)
    _validate_output_dtype(output_dtype)
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
    _validate_indptr_tensor(paged_kv_indptr, "paged_kv_indptr", device=device)
    for tensor, name in (
        (paged_kv_indices, "paged_kv_indices"),
        (paged_kv_last_page_len, "paged_kv_last_page_len"),
    ):
        _validate_paged_metadata_tensor(tensor, name, device=device)

    batch_size = int(qo_indptr.numel()) - 1
    _validate_extent(batch_size, "batch_size")
    if paged_kv_indptr.numel() != batch_size + 1:
        raise ValueError(
            "paged_kv_indptr and qo_indptr must describe the same batch; "
            f"got {paged_kv_indptr.numel() - 1} and {batch_size} rows"
        )
    _, q_lengths = _read_indptr(qo_indptr, "qo_indptr", expected_total=total_q)
    page_indptr_values, page_counts = _read_indptr(
        paged_kv_indptr,
        "paged_kv_indptr",
        expected_total=int(paged_kv_indices.numel()),
    )
    page_indices = _read_int32_values(
        paged_kv_indices,
        "paged_kv_indices",
        expected_count=page_indptr_values[-1],
    )
    last_page_lens = _read_int32_values(
        paged_kv_last_page_len,
        "paged_kv_last_page_len",
        expected_count=batch_size,
    )
    for offset, page_idx in enumerate(page_indices):
        if page_idx < 0 or page_idx >= num_physical_pages:
            raise ValueError(
                "paged_kv_indices entries must index the physical page pool; "
                f"entry {offset} is {page_idx}, pool has {num_physical_pages} pages"
            )
    for batch_idx, last_page_len in enumerate(last_page_lens):
        if last_page_len < 1 or last_page_len > page_size:
            raise ValueError(
                "paged_kv_last_page_len entries must be in [1, page_size]; "
                f"batch {batch_idx} has {last_page_len}, page_size={page_size}"
            )

    k_lengths = tuple(
        (page_count - 1) * page_size + last_page_len
        for page_count, last_page_len in zip(page_counts, last_page_lens, strict=True)
    )
    max_seq_len_q = max(q_lengths)
    max_seq_len_k = max(k_lengths)
    # Paged TMA issues one complete KV-tile fragment at a time. Keep every
    # dense page-table row fragment-aligned so the elected direct-load lane
    # cannot cross into the peer K/V row on a partial final fragment.
    pages_per_kv_tile = _CONTEXT_KV_TILE_N // page_size
    max_page_count = max(page_counts)
    max_num_pages_per_seq_kv = (
        (max_page_count + pages_per_kv_tile - 1) // pages_per_kv_tile
    ) * pages_per_kv_tile
    _validate_extent(max_seq_len_q, "max_seq_len_q")
    _validate_extent(max_seq_len_k, "max_seq_len_k")
    _validate_extent(max_num_pages_per_seq_kv, "max_num_pages_per_seq_kv")
    _validate_extent(
        2 * batch_size * max_num_pages_per_seq_kv,
        "dense page-table elements",
    )

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
    logical_kv_indptr = [0]
    for k_length in k_lengths:
        logical_kv_indptr.append(logical_kv_indptr[-1] + k_length)
    _validate_padded_data_extent(logical_kv_indptr[-1], "sum_seq_len_k")
    # The upstream ABI is [B, 2, max_pages], so batch is the outer dimension.
    # Preserve arbitrary and repeated physical page indices from the
    # FlashInfer CSR representation.
    dense_page_indices: list[int] = []
    for batch_idx, page_count in enumerate(page_counts):
        page_start = page_indptr_values[batch_idx]
        row = page_indices[page_start : page_start + page_count]
        # Invalid tail tokens are masked before softmax, but their TMA page
        # fragments must still resolve to valid storage. Repeating this
        # request's last page is safe for both direct and SMEM-cached loads.
        padded_row = row + (row[-1],) * (max_num_pages_per_seq_kv - page_count)
        dense_page_indices.extend(padded_row)
        dense_page_indices.extend(padded_row)

    geometry = _PagedContextOneShotGeometry(
        device=device,
        batch_size=batch_size,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_k=max_seq_len_k,
        page_size=page_size,
        max_num_pages_per_seq_kv=max_num_pages_per_seq_kv,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=q_head_dim,
        q_dtype=q.dtype,
        output_dtype=output_dtype,
    )
    metadata = _PagedContextOneShotMetadata(
        kv_indptr=tuple(logical_kv_indptr),
        seq_lens=k_lengths,
        dense_page_indices=tuple(dense_page_indices),
    )
    return geometry, metadata


def _resolve_paged_plan_geometry(
    *,
    device: int | str | torch.device,
    batch_size: int,
    max_seq_len_q: int,
    max_seq_len_k: int,
    max_num_pages_per_seq_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    page_size: int,
    mask_type: str,
    window_left: int,
    output_dtype: torch.dtype,
) -> _PagedContextPlanGeometry:
    """Validate explicit static bounds for a reusable paged specialization."""

    _validate_paged_dtype_pair(q_dtype, kv_dtype, output_dtype)
    _validate_mask(mask_type)
    if mask_type == "variable_window":
        raise NotImplementedError(
            "mask_type='variable_window' is not supported for paged context"
        )
    window_left = _validate_window_left(window_left, mask_type)
    page_size = _validate_page_size(page_size)
    batch_size = _validate_static_extent(batch_size, "batch_size")
    max_seq_len_q = _validate_static_extent(max_seq_len_q, "max_seq_len_q")
    max_seq_len_k = _validate_static_extent(max_seq_len_k, "max_seq_len_k")
    max_num_pages_per_seq_kv = _validate_static_extent(
        max_num_pages_per_seq_kv, "max_num_pages_per_seq_kv"
    )
    num_qo_heads = _validate_static_extent(num_qo_heads, "num_qo_heads")
    num_kv_heads = _validate_static_extent(num_kv_heads, "num_kv_heads")
    head_dim = _validate_static_extent(head_dim, "head_dim")
    device, device_index = _resolve_cuda_device(device)

    _validate_padded_data_extent(
        batch_size * max_seq_len_q, "batch_size * max_seq_len_q"
    )
    _validate_padded_data_extent(
        batch_size * max_seq_len_k, "batch_size * max_seq_len_k"
    )
    pages_per_kv_tile = _CONTEXT_KV_TILE_N // page_size
    if max_num_pages_per_seq_kv % pages_per_kv_tile != 0:
        raise ValueError(
            "max_num_pages_per_seq_kv must pad each page-table row to a "
            f"multiple of {pages_per_kv_tile} pages for page_size={page_size}"
        )
    required_page_columns = (max_seq_len_k + page_size - 1) // page_size
    if max_num_pages_per_seq_kv < required_page_columns:
        raise ValueError(
            "max_num_pages_per_seq_kv must cover max_seq_len_k: requires at "
            f"least {required_page_columns} columns, got {max_num_pages_per_seq_kv}"
        )
    _validate_extent(
        2 * batch_size * max_num_pages_per_seq_kv,
        "dense page-table elements",
    )

    head_ratio = _validate_head_geometry(num_qo_heads, num_kv_heads)
    _validate_head_dim(head_dim, head_dim)
    head_paired = window_left > 0
    if head_paired and (head_ratio <= 1 or head_ratio % 2 != 0):
        raise NotImplementedError(
            "a positive left window requires grouped-query attention with an "
            f"even Hq/Hkv ratio greater than one; got {head_ratio}"
        )

    return _PagedContextPlanGeometry(
        device=device,
        device_index=device_index,
        batch_size=batch_size,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_k=max_seq_len_k,
        page_size=page_size,
        max_num_pages_per_seq_kv=max_num_pages_per_seq_kv,
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
        has_q_offset=mask_type == "causal",
        packed_dense_k_mask=mask_type == "dense",
    )


def _semantic_key(geometry: _ContextGeometry) -> tuple[object, ...]:
    return (
        geometry.device_index,
        geometry.batch_size,
        geometry.max_seq_len_q,
        geometry.max_seq_len_k,
        geometry.num_qo_heads,
        geometry.num_kv_heads,
        geometry.head_dim,
        _dtype_key(geometry.q_dtype),
        _dtype_key(geometry.output_dtype),
        geometry.mask_type,
        geometry.window_left,
        geometry.packed,
        geometry.head_paired,
        geometry.uniform_packed_lengths,
        geometry.has_q_offset,
        geometry.causal_single_kv_tile,
        geometry.packed_dense_k_mask,
    )


def _paged_semantic_key(
    geometry: _PagedContextPlanGeometry,
) -> tuple[object, ...]:
    return (
        geometry.device_index,
        geometry.batch_size,
        geometry.max_seq_len_q,
        geometry.max_seq_len_k,
        geometry.page_size,
        geometry.max_num_pages_per_seq_kv,
        geometry.num_qo_heads,
        geometry.num_kv_heads,
        geometry.head_dim,
        _dtype_key(geometry.q_dtype),
        _dtype_key(geometry.kv_dtype),
        _dtype_key(geometry.output_dtype),
        geometry.mask_type,
        geometry.window_left,
        geometry.head_paired,
        geometry.uniform_packed_lengths,
        geometry.has_q_offset,
        geometry.packed_dense_k_mask,
    )


@functools.cache
def _get_compiled_context(
    device_index: int,
    batch_size: int,
    max_seq_len_q: int,
    max_seq_len_k: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype_key: str,
    output_dtype_key: str,
    mask_type: str,
    window_left: int,
    packed: bool,
    head_paired: bool,
    uniform_packed_lengths: bool,
    has_q_offset: bool,
    causal_single_kv_tile: bool,
    packed_dense_k_mask: bool,
):
    """Compile and cache one exact semantic context-attention specialization."""

    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda_drv
    import cutlass.utils as utils

    from .kernels.fmha_context.fmha_kernel import FmhaTs

    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
    }
    input_dtype = dtype_map[q_dtype_key]
    output_dtype = dtype_map[output_dtype_key]
    is_causal = mask_type == "causal"
    has_variable_window = mask_type == "variable_window"
    # Construct the scheduler-independent topology first. Immutable
    # single-instance domains can use the lower-overhead static persistent
    # queue; paired or live-ragged domains are reconstructed with CLC.
    fmha = FmhaTs(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        in_dtype=input_dtype,
        out_dtype=output_dtype,
        d=head_dim,
        is_persistent=True,
        is_causal=is_causal,
        has_variable_window=has_variable_window,
        balance_causal_workload=_uses_heavy_first_static_causal_raster(
            mask_type=mask_type,
            window_left=window_left,
            has_q_offset=has_q_offset,
        ),
        is_clc_dynamic=False,
        head_paired=head_paired,
        window_size_left=window_left if window_left > 0 else 0,
        h_r=num_qo_heads // num_kv_heads,
        enable_skip_correction=True,
        causal_single_kv_tile=causal_single_kv_tile,
    )
    with torch.cuda.device(device_index):
        max_active_clusters = int(utils.HardwareInfo().get_max_active_clusters(1))
    num_seq_tiles = (max_seq_len_q + fmha.cfg.cta_tiler[0] - 1) // fmha.cfg.cta_tiler[0]
    num_head_tiles = num_qo_heads // fmha.cfg.work_tile_q_heads
    logical_work_tiles = batch_size * num_seq_tiles * num_head_tiles
    is_persistent = _contiguous_context_uses_persistent_scheduler(
        single_qkv_instance=fmha.cfg.single_qkv_instance,
        head_paired=head_paired,
        packed=packed,
        uniform_packed_lengths=uniform_packed_lengths,
        logical_work_tiles=logical_work_tiles,
        max_active_clusters=max_active_clusters,
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        is_causal=is_causal,
        has_q_offset=has_q_offset,
    )
    uses_clc = is_persistent and _contiguous_context_uses_clc_scheduler(
        single_qkv_instance=fmha.cfg.single_qkv_instance,
        head_paired=head_paired,
        packed=packed,
        uniform_packed_lengths=uniform_packed_lengths,
        is_causal=is_causal,
        has_q_offset=has_q_offset,
    )
    if uses_clc:
        fmha = FmhaTs(
            qk_acc_dtype=cutlass.Float32,
            pv_acc_dtype=cutlass.Float32,
            in_dtype=input_dtype,
            out_dtype=output_dtype,
            d=head_dim,
            is_persistent=True,
            is_causal=is_causal,
            has_variable_window=has_variable_window,
            is_clc_dynamic=True,
            head_paired=head_paired,
            window_size_left=window_left if window_left > 0 else 0,
            h_r=num_qo_heads // num_kv_heads,
            enable_skip_correction=True,
            causal_single_kv_tile=causal_single_kv_tile,
        )
    elif not is_persistent:
        fmha = FmhaTs(
            qk_acc_dtype=cutlass.Float32,
            pv_acc_dtype=cutlass.Float32,
            in_dtype=input_dtype,
            out_dtype=output_dtype,
            d=head_dim,
            is_persistent=False,
            is_causal=is_causal,
            has_variable_window=has_variable_window,
            is_clc_dynamic=False,
            head_paired=head_paired,
            window_size_left=window_left if window_left > 0 else 0,
            h_r=num_qo_heads // num_kv_heads,
            enable_skip_correction=True,
            causal_single_kv_tile=causal_single_kv_tile,
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
    if not is_causal and not packed:
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
        q_shape = (runtime_total_q, num_qo_heads, head_dim)
        kv_shape = (runtime_total_k, num_kv_heads, head_dim)
        out_shape = (runtime_total_q, num_qo_heads, head_dim)
        indptr_shape = (batch_size + 1,)
    else:
        q_shape = (batch_size, max_seq_len_q, num_qo_heads, head_dim)
        kv_shape = (batch_size, max_seq_len_k, num_kv_heads, head_dim)
        out_shape = q_shape
        indptr_shape = (1,)
    q_fake = fake_compact(input_dtype, q_shape, 16)
    k_fake = fake_compact(input_dtype, kv_shape, 16)
    v_fake = fake_compact(input_dtype, kv_shape, 16)
    out_fake = fake_compact(output_dtype, out_shape, 16)
    scale_fake = fake_compact(cutlass.Float32, (1,), 4)
    output_scale_fake = fake_compact(cutlass.Float32, (1,), 4)
    qo_indptr_fake = fake_compact(cutlass.Int32, indptr_shape, 4)
    kv_indptr_fake = fake_compact(cutlass.Int32, indptr_shape, 4)
    variable_window_shape = (
        (batch_size * max_seq_len_q,) if has_variable_window else (1,)
    )
    variable_window_starts_fake = fake_compact(cutlass.Int32, variable_window_shape, 4)
    variable_window_ends_fake = fake_compact(cutlass.Int32, variable_window_shape, 4)
    variable_window_tile_size_q = (
        _CONTEXT_MAX_Q_ROWS_PER_WORK_TILE
        if head_dim == _CONTEXT_TILE_SIZE_Q
        else _CONTEXT_TILE_SIZE_Q
    )
    variable_window_cta_shape = (
        (batch_size * cute.ceil_div(max_seq_len_q, variable_window_tile_size_q),)
        if has_variable_window
        else (1,)
    )
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
        (
            "scheduler",
            ("clc_dynamic_persistent" if fmha.is_clc_dynamic else "static_persistent"),
        ),
        ("pairing", "head" if head_paired else "query"),
        ("uniform_packed_lengths", uniform_packed_lengths),
        ("causal_single_kv_tile", causal_single_kv_tile),
        ("packed_dense_k_mask", packed_dense_k_mask),
    )
    return compiled, policy


@functools.cache
def _get_compiled_paged_context(
    device_index: int,
    batch_size: int,
    max_seq_len_q: int,
    max_seq_len_k: int,
    page_size: int,
    max_num_pages_per_seq_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype_key: str,
    kv_dtype_key: str,
    output_dtype_key: str,
    mask_type: str,
    window_left: int,
    head_paired: bool,
    uniform_packed_lengths: bool,
    has_q_offset: bool,
    packed_dense_k_mask: bool,
):
    """Compile one packed-Q, paged-K/V context specialization."""

    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda_drv
    import cutlass.utils as utils

    from .kernels.fmha_context.fmha_kernel import FmhaTs

    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
    }
    if kv_dtype_key != q_dtype_key:
        raise RuntimeError(
            "paged context compilation requires identical Q and K/V dtypes"
        )
    input_dtype = dtype_map[q_dtype_key]
    output_dtype = dtype_map[output_dtype_key]
    is_causal = mask_type == "causal"
    with torch.cuda.device(device_index):
        max_active_clusters = int(utils.HardwareInfo().get_max_active_clusters(1))

    # Build the persistent topology first to obtain its actual work-tile shape.
    # Scheduler selection then follows the logical CTA wave count rather than a
    # head-dimension or sequence-length performance threshold.  The tiler is
    # scheduler-independent; reconstruct only the one-wave nonpersistent case.
    fmha = FmhaTs(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        in_dtype=input_dtype,
        out_dtype=output_dtype,
        d=head_dim,
        is_persistent=True,
        is_causal=is_causal,
        # This topology probe is scheduler-independent. Persistent paired
        # plans are reconstructed below with CLC after their logical wave
        # count is known; single-instance plans retain the staged page-offset
        # producer and static scheduling.
        balance_causal_workload=_uses_heavy_first_static_causal_raster(
            mask_type=mask_type,
            window_left=window_left,
            has_q_offset=has_q_offset,
        ),
        is_clc_dynamic=False,
        head_paired=head_paired,
        window_size_left=window_left if window_left > 0 else 0,
        h_r=num_qo_heads // num_kv_heads,
        enable_skip_correction=True,
        use_paged_kv=True,
        num_tokens_per_page=page_size,
        max_num_pages_per_seq_kv=max_num_pages_per_seq_kv,
        # The fixed one-tile shortcut assumes contiguous fixed-shape K/V and
        # is intentionally disabled for the page-table path.
        causal_single_kv_tile=False,
    )
    num_seq_tiles = (max_seq_len_q + fmha.cfg.cta_tiler[0] - 1) // fmha.cfg.cta_tiler[0]
    num_head_tiles = num_qo_heads // fmha.cfg.work_tile_q_heads
    logical_work_tiles = batch_size * num_seq_tiles * num_head_tiles
    paged_is_persistent = _paged_context_uses_persistent_scheduler(
        mask_type=mask_type,
        head_paired=head_paired,
        logical_work_tiles=logical_work_tiles,
        max_active_clusters=max_active_clusters,
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
    )
    # A fixed/uniform causal plan has an immutable task domain. Its static
    # queue avoids CLC response overhead; causal live-ragged plans retain CLC
    # so request-local work changes are redistributed at runtime. Dense work
    # does not have the causal request-dependent K-tile domain and stays static.
    paged_uses_clc = _paged_context_uses_clc_scheduler(
        is_persistent=paged_is_persistent,
        single_qkv_instance=fmha.cfg.single_qkv_instance,
        is_causal=is_causal,
        uniform_packed_lengths=uniform_packed_lengths,
    )
    if paged_uses_clc:
        fmha = FmhaTs(
            qk_acc_dtype=cutlass.Float32,
            pv_acc_dtype=cutlass.Float32,
            in_dtype=input_dtype,
            out_dtype=output_dtype,
            d=head_dim,
            is_persistent=True,
            is_causal=is_causal,
            is_clc_dynamic=True,
            head_paired=head_paired,
            window_size_left=window_left if window_left > 0 else 0,
            h_r=num_qo_heads // num_kv_heads,
            enable_skip_correction=True,
            use_paged_kv=True,
            num_tokens_per_page=page_size,
            max_num_pages_per_seq_kv=max_num_pages_per_seq_kv,
            causal_single_kv_tile=False,
        )
    elif not paged_is_persistent:
        fmha = FmhaTs(
            qk_acc_dtype=cutlass.Float32,
            pv_acc_dtype=cutlass.Float32,
            in_dtype=input_dtype,
            out_dtype=output_dtype,
            d=head_dim,
            is_persistent=False,
            is_causal=is_causal,
            is_clc_dynamic=False,
            head_paired=head_paired,
            window_size_left=window_left if window_left > 0 else 0,
            h_r=num_qo_heads // num_kv_heads,
            enable_skip_correction=True,
            use_paged_kv=True,
            num_tokens_per_page=page_size,
            max_num_pages_per_seq_kv=max_num_pages_per_seq_kv,
            causal_single_kv_tile=False,
        )
    fmha.cfg.has_varlen = True
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

    @cute.jit
    def tensor_adapter(
        q: cute.Tensor,
        k_cache: cute.Tensor,
        v_cache: cute.Tensor,
        out: cute.Tensor,
        scale_softmax_log2: cute.Tensor,
        output_scale: cute.Tensor,
        qo_indptr: cute.Tensor,
        kv_indptr: cute.Tensor,
        page_idx_kv: cute.Tensor,
        seq_lens_kv: cute.Tensor,
        stream: cuda_drv.CUstream,
        static_max_active_clusters: cutlass.Constexpr[int],
        static_max_seq_len_q: cutlass.Constexpr[int],
        static_max_seq_len_k: cutlass.Constexpr[int],
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
            qo_indptr,
            kv_indptr,
            cutlass.Int32(static_max_seq_len_q),
            cutlass.Int32(static_max_seq_len_k),
            page_idx_kv,
            seq_lens_kv,
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
    qo_indptr_fake = fake_compact(cutlass.Int32, (batch_size + 1,), 4)
    kv_indptr_fake = fake_compact(cutlass.Int32, (batch_size + 1,), 4)
    page_idx_fake = fake_compact(
        cutlass.Int32,
        (batch_size, 2, max_num_pages_per_seq_kv),
        4,
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
            kv_indptr_fake,
            page_idx_fake,
            seq_lens_fake,
            stream_fake,
            max_active_clusters,
            max_seq_len_q,
            max_seq_len_k,
            options=_COMPILE_OPTIONS,
        )
    policy = (
        (
            "scheduler",
            (
                "clc_dynamic_persistent"
                if paged_uses_clc
                else "static_persistent"
                if paged_is_persistent
                else "nonpersistent"
            ),
        ),
        ("pairing", "head" if head_paired else "query"),
        ("kv_layout", "paged_hnd"),
        ("page_size", page_size),
        ("causal_single_kv_tile", False),
        ("packed_dense_k_mask", packed_dense_k_mask),
    )
    return compiled, policy


def _validate_runtime_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    geometry: _ContextGeometry,
) -> None:
    """Validate run tensors without allocation, synchronization, or metadata reads."""

    _validate_base_tensors(q, k, v)
    if q.device != geometry.device:
        raise ValueError(f"q must be on {geometry.device}, got {q.device}")
    if q.dtype != geometry.q_dtype:
        raise ValueError(f"q must have dtype {geometry.q_dtype}, got {q.dtype}")
    if tuple(q.shape) != geometry.q_shape:
        raise ValueError(f"q must have shape {geometry.q_shape}, got {tuple(q.shape)}")
    if tuple(k.shape) != geometry.kv_shape:
        raise ValueError(f"k must have shape {geometry.kv_shape}, got {tuple(k.shape)}")
    q_layout = "[total_q, Hq, D]" if geometry.packed else "[B, Sq, Hq, D]"
    kv_layout = "[total_k, Hkv, D]" if geometry.packed else "[B, Sk, Hkv, D]"
    _validate_compact(q, "q", q_layout)
    _validate_compact(k, "k", kv_layout)
    _validate_compact(v, "v", kv_layout)


def _validate_live_paged_runtime_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    geometry: _PagedContextPlanGeometry,
) -> None:
    """Validate live data tensors without fixing their packed/page extents."""

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


def _validate_live_paged_metadata(
    qo_indptr: torch.Tensor,
    logical_kv_indptr: torch.Tensor,
    dense_page_idx_kv: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    geometry: _PagedContextPlanGeometry,
) -> None:
    """Validate live metadata storage without reading any device values."""

    for indptr, name in (
        (qo_indptr, "qo_indptr"),
        (logical_kv_indptr, "logical_kv_indptr"),
    ):
        _validate_indptr_tensor(indptr, name, device=geometry.device)
        if indptr.numel() != geometry.batch_size + 1:
            raise ValueError(
                f"{name} must contain batch_size + 1 elements "
                f"({geometry.batch_size + 1}), got {indptr.numel()}"
            )

    _validate_paged_metadata_tensor(seq_lens_kv, "seq_lens_kv", device=geometry.device)
    if seq_lens_kv.numel() != geometry.batch_size:
        raise ValueError(
            "seq_lens_kv must contain one value per request; expected "
            f"{geometry.batch_size}, got {seq_lens_kv.numel()}"
        )

    _validate_tensor(dense_page_idx_kv, "dense_page_idx_kv")
    if dense_page_idx_kv.device != geometry.device:
        raise ValueError(
            f"dense_page_idx_kv must be on {geometry.device}, got {dense_page_idx_kv.device}"
        )
    if dense_page_idx_kv.dtype != torch.int32:
        raise TypeError("dense_page_idx_kv must have dtype torch.int32")
    expected_page_shape = (
        geometry.batch_size,
        2,
        geometry.max_num_pages_per_seq_kv,
    )
    if tuple(dense_page_idx_kv.shape) != expected_page_shape:
        raise ValueError(
            f"dense_page_idx_kv must have shape {expected_page_shape}, "
            f"got {tuple(dense_page_idx_kv.shape)}"
        )
    _validate_compact(
        dense_page_idx_kv,
        "dense_page_idx_kv",
        "[B, 2, max_num_pages_per_seq_kv]",
    )
    _validate_alignment(dense_page_idx_kv, "dense_page_idx_kv", 4)


def _validate_live_scale_tensor(
    scale: torch.Tensor,
    name: str,
    *,
    device: torch.device,
) -> None:
    """Validate one allocation-free runtime scale input."""

    _validate_tensor(scale, name)
    if scale.device != device:
        raise ValueError(f"{name} must be on {device}, got {scale.device}")
    if scale.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if tuple(scale.shape) != (1,):
        raise ValueError(f"{name} must have shape [1], got {tuple(scale.shape)}")
    _validate_compact(scale, name, "[1]")
    _validate_alignment(scale, name, 4)


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
    """Plan and reuse task-scheduled fixed or packed-ragged context attention.

    ``plan`` may compile, allocate two one-element scale tensors, and copy
    packed cumulative offsets to the host for validation. The ``run`` host
    path performs no metadata read or synchronization. With caller-provided
    ``out``, its Python path allocates no tensors and is suitable for CUDA
    graph capture. ``out`` must not overlap any Q, K, V, packed-offset, or
    scale input storage.

    Packed ``qo_indptr`` and ``kv_indptr`` storage is retained as live plan
    input, so it must remain alive and at stable addresses. General ragged
    kernels reread the values; a uniform packed plan may compile its fixed
    offsets into the specialization. Values may change while preserving the
    planned batch, zero starting offsets, final packed extents, strictly
    positive deltas, and the plan-time global Q/K capacities: every runtime
    ``Sq[b]`` must be no greater than the planned maximum Q length and every
    ``Sk[b]`` no greater than the planned maximum K length. Every causal replay
    must also satisfy ``Sq[b] <= Sk[b]``. The request-local bottom-right offset
    ``Sk[b] - Sq[b]`` may change between runs. Fixed totals plus the
    per-request capacity bounds force plan-time uniform Q or K lengths to
    remain unchanged, preserving a dense aligned-K specialization that
    compiled away request-local K-tail masking. The ``run`` host path trusts
    live offset values; violating this contract can produce incorrect results
    or out-of-bounds access.
    """

    @flashinfer_api
    def __init__(self) -> None:
        """Initialize an unplanned task-scheduled context-attention wrapper."""
        self._planned = False

    @flashinfer_api
    def plan(
        self,
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
    ) -> None:
        """Validate semantics, establish Q/K capacities, and compile once.

        Packed cumulative offsets remain live device inputs. Their runtime
        values must follow the replay contract documented on this wrapper.

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
            Inclusive per-query K bounds required for ``variable_window``.
            Both must be CUDA int32 tensors shaped ``[B, Sq]`` and satisfy
            ``0 <= starts[b, q] <= ends[b, q] < Sk``.
        sm_scale : float, optional
            Softmax scale; defaults to the inverse square root of head size.
        output_scale : float
            Scale applied to the attention output.
        out_dtype : torch.dtype, optional
            Output dtype; defaults to the query dtype.
        """

        if out_dtype is None:
            if not isinstance(q, torch.Tensor):
                raise TypeError("q must be a torch.Tensor")
            resolved_out_dtype = q.dtype
        else:
            resolved_out_dtype = out_dtype
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
        if mask_type == "variable_window":
            validated_window_starts, validated_window_ends = (
                _validate_variable_window_bounds(
                    variable_window_token_starts,
                    variable_window_token_ends,
                    geometry=geometry,
                )
            )
            # Variable-window bounds are plan metadata. Keep one internal
            # snapshot so row masks and the reduced CTA origins cannot diverge
            # if the caller later modifies its tensors.
            planned_window_starts = validated_window_starts.clone()
            planned_window_ends = validated_window_ends.clone()
            planned_window_cta_starts = _build_variable_window_cta_starts(
                planned_window_starts, geometry=geometry
            )
        else:
            if (
                variable_window_token_starts is not None
                or variable_window_token_ends is not None
            ):
                raise ValueError(
                    "variable-window bounds require mask_type='variable_window'"
                )
            planned_window_starts = torch.empty(
                1, dtype=torch.int32, device=geometry.device
            )
            planned_window_ends = torch.empty(
                1, dtype=torch.int32, device=geometry.device
            )
            planned_window_cta_starts = torch.empty(
                1, dtype=torch.int32, device=geometry.device
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
        if geometry.packed:
            assert qo_indptr is not None and kv_indptr is not None
            planned_qo_indptr = qo_indptr
            planned_kv_indptr = kv_indptr
        else:
            # Uniform TVM-FFI signature; fixed specializations compile these
            # arguments away but still keep stable runtime placeholders.
            planned_qo_indptr = torch.empty(
                1, dtype=torch.int32, device=geometry.device
            )
            planned_kv_indptr = torch.empty(
                1, dtype=torch.int32, device=geometry.device
            )
        # Keep all runtime tensor allocation ahead of CUTLASS JIT. Besides
        # making plan publication atomic, this lets compute-sanitizer patch the
        # generated attention kernel without interleaving later PyTorch setup
        # launches with the DSL compiler/runtime callbacks.
        compiled, policy = _get_compiled_context(*_semantic_key(geometry))

        # Publish only after validation, compilation, and allocation succeed.
        self._geometry = geometry
        self._qo_indptr = planned_qo_indptr
        self._kv_indptr = planned_kv_indptr
        self._scale_softmax_log2 = scale_tensor
        self._output_scale = output_scale_tensor
        self._variable_window_token_starts = planned_window_starts
        self._variable_window_token_ends = planned_window_ends
        self._variable_window_cta_starts = planned_window_cta_starts
        self._compiled = compiled
        self._policy = policy
        self._planned = True

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Launch on the current stream into output disjoint from Q, K, and V.

        Parameters
        ----------
        q, k, v : torch.Tensor
            Runtime query, key, and value tensors matching the plan.
        out : torch.Tensor, optional
            Caller-owned output tensor. A new tensor is allocated when omitted.
        """

        if not self._planned:
            raise RuntimeError("plan() must be called before run()")
        _validate_runtime_inputs(q, k, v, self._geometry)
        caller_provided_out = out is not None
        out = _prepare_out(out, q=q, output_dtype=self._geometry.output_dtype)
        if caller_provided_out:
            _validate_out_does_not_overlap_inputs(
                out,
                ("q", q),
                ("k", k),
                ("v", v),
                ("qo_indptr", self._qo_indptr),
                ("kv_indptr", self._kv_indptr),
                ("scale_softmax_log2", self._scale_softmax_log2),
                ("output_scale", self._output_scale),
                ("variable_window_token_starts", self._variable_window_token_starts),
                ("variable_window_token_ends", self._variable_window_token_ends),
                ("variable_window_cta_starts", self._variable_window_cta_starts),
            )
        self._compiled(
            q,
            k,
            v,
            out,
            self._scale_softmax_log2,
            self._output_scale,
            self._qo_indptr,
            self._kv_indptr,
            self._variable_window_token_starts,
            self._variable_window_token_ends,
            self._variable_window_cta_starts,
        )
        return out


class BatchPrefillPagedTSWrapper:
    """Compile and reuse packed-Q context attention over HND paged K/V.

    .. warning::
        This wrapper is experimental. Its API may change without a
        compatibility or deprecation period.

    ``plan`` accepts only static compilation geometry. All request metadata is
    native kernel metadata supplied live to ``run``: cumulative Q and logical
    K/V offsets, logical K/V lengths, and the dense padded K/V page table.
    Metadata values may change between launches while staying within the
    planned capacities. The wrapper owns no context workspace.

    Plan-time scalar defaults are stored as one-element device tensors.
    ``run`` may replace either scale tensor without changing the compiled
    specialization. With caller-owned output, the validated run path performs
    no allocation, metadata readback, or synchronization and is suitable for
    CUDA graph capture. Set ``validate=False`` only when the caller already
    enforces the documented storage contract.
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
        max_seq_len_k: int,
        max_num_pages_per_seq_kv: int,
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
    ) -> None:
        """Compile one reusable specialization from explicit static geometry.

        Sequence lengths and page indices are deliberately absent. The plan
        conservatively supports dynamic positive lengths bounded by
        ``max_seq_len_q`` and ``max_seq_len_k``. ``batch_size`` is exact.
        ``max_num_pages_per_seq_kv`` is the final dimension of the native dense
        page table and must be a multiple of ``128 / page_size`` large enough
        to cover ``max_seq_len_k``.

        Parameters
        ----------
        device : int, str, or torch.device
            CUDA device on which the specialization is compiled and run.
        batch_size : int
            Exact number of requests in every run.
        max_seq_len_q, max_seq_len_k : int
            Per-request Q and logical K/V length capacities.
        max_num_pages_per_seq_kv : int
            Padded page-table row capacity.
        num_qo_heads, num_kv_heads, head_dim : int
            Static attention head geometry.
        q_dtype, kv_dtype : torch.dtype
            Query and K/V dtypes. They must currently be equal.
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
        """

        resolved_out_dtype = q_dtype if out_dtype is None else out_dtype
        geometry = _resolve_paged_plan_geometry(
            device=device,
            batch_size=batch_size,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_k=max_seq_len_k,
            max_num_pages_per_seq_kv=max_num_pages_per_seq_kv,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            page_size=page_size,
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

        # Keep the two plan-owned scalar allocations ahead of CUTLASS JIT,
        # matching the compute-sanitizer-safe ordering of the other wrapper.
        compiled, policy = _get_compiled_paged_context(*_paged_semantic_key(geometry))

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
        logical_kv_indptr: torch.Tensor,
        dense_page_idx_kv: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        scale_softmax_log2: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        validate: bool = True,
    ) -> torch.Tensor:
        """Launch the compiled specialization with native live metadata.

        Metadata must be compact int32 CUDA storage. ``qo_indptr`` and
        ``logical_kv_indptr`` have shape ``[B + 1]``; ``seq_lens_kv`` has
        shape ``[B]``; and ``dense_page_idx_kv`` has shape
        ``[B, 2, max_num_pages_per_seq_kv]``. Their values must satisfy the
        capacities documented by ``plan``:

        * both cumulative-offset tensors start at zero and increase strictly;
          ``qo_indptr`` ends at the packed Q extent and its deltas do not
          exceed ``max_seq_len_q``;
        * ``logical_kv_indptr`` deltas equal ``seq_lens_kv``, whose entries are
          positive and do not exceed ``max_seq_len_k``;
        * causal launches additionally satisfy ``Sq[b] <= Sk[b]``; and
        * every dense page-table entry is a valid physical page ID. K and V
          planes describe the same logical page row for separate isomorphic
          pools, and padded tail entries still resolve to valid storage.

        These value contracts are never read back by the host. Metadata may
        change only between completed launches or graph replays; CUDA graph
        capture additionally requires stable tensor addresses.

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
        logical_kv_indptr : torch.Tensor
            Cumulative logical K/V token offsets with shape ``[B + 1]``.
        dense_page_idx_kv : torch.Tensor
            Dense native K/V page table with shape
            ``[B, 2, max_num_pages_per_seq_kv]``.
        seq_lens_kv : torch.Tensor
            Logical K/V lengths with shape ``[B]``.
        out : torch.Tensor, optional
            Caller-owned output tensor. A new tensor is allocated when omitted.
        scale_softmax_log2 : torch.Tensor, optional
            Live one-element float32 softmax scale in base-2 form. Defaults to
            the scale retained by the plan.
        output_scale : torch.Tensor, optional
            Live one-element float32 output scale. Defaults to the scale
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
            _validate_live_paged_runtime_inputs(q, k_cache, v_cache, geometry)
            _validate_live_paged_metadata(
                qo_indptr,
                logical_kv_indptr,
                dense_page_idx_kv,
                seq_lens_kv,
                geometry,
            )

        if scale_softmax_log2 is None:
            scale_softmax_log2 = state.scale_softmax_log2
        elif validate:
            _validate_live_scale_tensor(
                scale_softmax_log2,
                "scale_softmax_log2",
                device=geometry.device,
            )
        if output_scale is None:
            output_scale = state.output_scale
        elif validate:
            _validate_live_scale_tensor(
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
                ("logical_kv_indptr", logical_kv_indptr),
                ("seq_lens_kv", seq_lens_kv),
                ("dense_page_idx_kv", dense_page_idx_kv),
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
            logical_kv_indptr,
            dense_page_idx_kv,
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
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        q,
        k,
        v,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        mask_type=mask_type,
        window_left=window_left,
        variable_window_token_starts=variable_window_token_starts,
        variable_window_token_ends=variable_window_token_ends,
        sm_scale=sm_scale,
        output_scale=output_scale,
        out_dtype=resolved_out_dtype,
    )
    return wrapper.run(q, k, v, out=out)


@flashinfer_api
def batch_prefill_with_paged_kv_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
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
    ``qo_indptr`` describes Q rows while the three paged-KV metadata tensors
    use FlashInfer's CSR representation.
    Physical page indices need not be identity ordered. ``D`` may be 128 or
    256; Q, K, and V must share one supported dtype.

    Parameters
    ----------
    q : torch.Tensor
        Packed query tensor.
    k_cache, v_cache : torch.Tensor
        Separate HND key and value page pools.
    qo_indptr : torch.Tensor
        Cumulative packed-query offsets.
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len : torch.Tensor
        FlashInfer CSR page metadata.
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
    geometry, metadata = _resolve_paged_one_shot_inputs(
        q,
        k_cache,
        v_cache,
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        page_size=page_size,
        mask_type=mask_type,
        window_left=window_left,
        output_dtype=resolved_out_dtype,
    )
    logical_kv_indptr = torch.tensor(
        metadata.kv_indptr, dtype=torch.int32, device=geometry.device
    )
    seq_lens_kv = torch.tensor(
        metadata.seq_lens, dtype=torch.int32, device=geometry.device
    )
    dense_page_idx_kv = torch.tensor(
        metadata.dense_page_indices,
        dtype=torch.int32,
        device=geometry.device,
    ).view(
        geometry.batch_size,
        2,
        geometry.max_num_pages_per_seq_kv,
    )
    wrapper = BatchPrefillPagedTSWrapper(kv_layout=kv_layout)
    wrapper.plan(
        device=geometry.device,
        batch_size=geometry.batch_size,
        max_seq_len_q=geometry.max_seq_len_q,
        max_seq_len_k=geometry.max_seq_len_k,
        max_num_pages_per_seq_kv=geometry.max_num_pages_per_seq_kv,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        q_dtype=geometry.q_dtype,
        kv_dtype=geometry.q_dtype,
        out_dtype=geometry.output_dtype,
        page_size=page_size,
        mask_type=mask_type,
        window_left=window_left,
        sm_scale=sm_scale,
        output_scale=output_scale,
    )
    return wrapper.run(
        q,
        k_cache,
        v_cache,
        qo_indptr,
        logical_kv_indptr,
        dense_page_idx_kv,
        seq_lens_kv,
        out=out,
    )


__all__ = [
    "BatchPrefillPagedTSWrapper",
    "BatchPrefillTSWrapper",
    "batch_prefill",
    "batch_prefill_with_paged_kv_cache",
]
