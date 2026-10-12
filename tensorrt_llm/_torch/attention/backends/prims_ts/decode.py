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

"""Task-scheduled paged decode with a FlashInfer-style plan/run lifecycle."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import functools
import math
import numbers
import struct
from typing import TYPE_CHECKING, Literal, NamedTuple, Optional, Union, cast

import torch

from flashinfer.api_logging import flashinfer_api as flashinfer_experimental_api

from . import _q_token_kv_block_sparse_policy as _sparse_policy
from ._block_sparse.common import _num_sparse_pattern_heads

PagedKVCache = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
PagedKVScaleFactors = tuple[torch.Tensor, torch.Tensor]

if TYPE_CHECKING:
    from .kernels.fmha_decode.fmha_decode_config import FmhaDecodeConfig

_SUPPORTED_HEAD_DIMS = (64, 128, 256)
_SUPPORTED_PAGE_SIZES = (4, 8, 16, 32, 64, 128)
_MAX_INT32 = 2**31 - 1
# Decode K/V masks form an exclusive tile endpoint as
# ``tile_offset_k + tile_size_kv`` in signed Int32.  Public policies use at
# most a 256-token K/V tile, so reserve its full 255-token padded tail.
_DECODE_MAX_KV_TILE_SIZE = 256
_DECODE_MAX_KV_LEN = _MAX_INT32 - (_DECODE_MAX_KV_TILE_SIZE - 1)
_SUPPORTED_INPUT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
)
_SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"
_WORKSPACE_ALIGNMENT = 256
_WORKSPACE_DTYPES = (torch.int8, torch.uint8)
_MAX_HEAD_RATIO = 128


@dataclass(frozen=True)
class _WorkspaceSection:
    """One typed tensor view owned by a caller-provided byte workspace."""

    byte_offset: int
    byte_size: int
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class _DecodeWorkspaceLayout:
    """Private FMHA scratch layout; only ``total_bytes`` is public."""

    partial_o: _WorkspaceSection
    partial_stats: _WorkspaceSection
    split_kv_counter: _WorkspaceSection
    cu_seqlens_q: _WorkspaceSection
    attention_sinks: _WorkspaceSection
    uses_split_kv: bool
    total_bytes: int


@dataclass(frozen=True)
class _DecodeWorkspaceViews:
    """Typed zero-copy views bound to one validated workspace buffer."""

    partial_o: torch.Tensor
    partial_stats: torch.Tensor
    split_kv_counter: torch.Tensor
    cu_seqlens_q: torch.Tensor
    attention_sinks: torch.Tensor


@dataclass(frozen=True)
class _DecodeLaunchSpec:
    """Automatic policy and scratch geometry for one planned shape."""

    config: "FmhaDecodeConfig"
    max_active_clusters: int
    policy: tuple[tuple[str, object], ...]
    scratch_shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


class _DirectQ1CompileSpec(NamedTuple):
    """Static raw-index geometry; tensor values and strides remain runtime inputs."""

    position_dtype_key: str
    max_seq_len_kv: int
    block_topk: int
    kv_block_size: int
    page_capacity: int


@dataclass(frozen=True)
class _DecodeCompileSpec:
    """Batch-independent static identity for one compiled decode callable."""

    device_index: int
    config_items: tuple[tuple[str, object], ...]
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    max_kv_len: int
    seq_len_q: int
    q_dtype_key: str
    k_dtype_key: str
    v_dtype_key: str
    output_dtype_key: str
    use_packed_q: bool
    max_active_clusters: int
    kv_prefix_mode: Literal["dynamic", "planned_full"]
    kv_lengths_mode: Literal["dynamic", "planned_uniform_max"]
    direct_q1_spec: Optional[_DirectQ1CompileSpec] = None


@dataclass(frozen=True)
class _DecodeRuntime:
    """Validated runtime tensors and scalar arguments for one launch."""

    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_sf_cache: torch.Tensor
    v_sf_cache: torch.Tensor
    out: torch.Tensor
    num_physical_pages: int
    k_page_stride: int
    k_head_stride: int
    k_token_stride: int
    v_page_stride: int
    v_head_stride: int
    v_token_stride: int
    bmm1_scale: float
    bmm2_scale: float


@dataclass(frozen=True)
class _DecodePlanState:
    """One complete static paged-decode plan published atomically.

    Sequence lengths are owned either by the plan or by every run. The page
    table and (for packed Q) query offsets remain per-run request metadata.
    """

    device: torch.device
    device_index: int
    batch_size: int
    seq_len_q: int
    use_packed_q: bool
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    max_kv_len: int
    q_dtype: torch.dtype
    k_dtype: torch.dtype
    v_dtype: torch.dtype
    output_dtype: torch.dtype
    mask_type: str
    window_left: int
    config: "FmhaDecodeConfig"
    workspace_buffer: torch.Tensor
    workspace_layout: _DecodeWorkspaceLayout
    workspace: _DecodeWorkspaceViews
    compiled_main: Callable[..., object]
    compiled_reducer: Optional[Callable[..., object]]
    kv_prefix_mode: Literal["dynamic", "planned_full"]
    kv_lengths_mode: Literal["dynamic", "planned_uniform_max"]
    planned_seq_lens_host: Optional[tuple[int, ...]]
    planned_seq_lens_device: Optional[torch.Tensor]
    policy: tuple[tuple[str, object], ...]
    storage_page_size: Optional[int] = None


def _decode_policy_from_config(
    config: "FmhaDecodeConfig",
) -> tuple[tuple[str, object], ...]:
    """Return the stable private policy record for one resolved FMHA config."""

    seq_len_q = int(config.max_seq_len_q)
    uses_packed_q = bool(config.use_variable_seqlens_q)
    query_layout = (
        "TOTAL_Q_Hq_D"
        if uses_packed_q
        else ("B_Hq_D" if seq_len_q == 1 else "B_SQ_Hq_D")
    )
    return (
        ("seq_len_q", seq_len_q),
        ("max_seq_len_q", seq_len_q),
        ("use_packed_q", uses_packed_q),
        ("query_layout", query_layout),
        ("output_layout", query_layout),
        (
            "window_left",
            int(config.attention_window_size) - 1
            if config.use_sliding_window_causal
            else -1,
        ),
        (
            "mma_variant",
            "keeps_mma_ab" if config.use_keeps_mma_ab else "swaps_mma_ab",
        ),
        ("tile_size_q", int(config.tile_size_q)),
        ("tile_size_kv", int(config.tile_size_kv)),
        ("num_insts_kv", int(config.num_insts_kv)),
        ("use_split_kv", bool(config.use_split_kv)),
        ("splits_kv", int(config.splits_kv)),
        ("max_splits_kv", int(config.max_splits_kv)),
        (
            "use_separate_reduction_kernel",
            bool(config.use_separate_reduction_kernel),
        ),
        ("use_cluster_smem_reduction", bool(config.use_cluster_smem_reduction)),
        ("use_persistent_scheduler", bool(config.use_persistent_scheduler)),
        ("groups_tokens_heads_q", bool(config.groups_tokens_heads_q)),
    )


def _planned_full_split_prefix(
    config: "FmhaDecodeConfig",
    seq_lens: tuple[int, ...],
    *,
    seq_len_q: int,
    max_kv_len: int,
    mask_type: str,
) -> bool:
    """Prove that host length evidence uses every configured split CTA.

    A successful proof permits a private JIT specialization that removes only
    the no-op runtime split-prefix branch. The plan retains the evidenced
    lengths and supplies an owned device copy to every launch.

    Q groups are enumerated with the same token-base/union rule as the device
    helper.  Every batch/group pair must prove the configured fanout; otherwise
    the general runtime-pruning kernel is retained.
    """

    if (
        not bool(config.use_split_kv)
        or int(config.splits_kv) <= 1
        or bool(config.use_variable_seqlens_q)
        or bool(config.use_sliding_window_causal)
    ):
        return False
    from .kernels.fmha_decode.fmha_decode_config import (
        compute_runtime_active_splits_kv,
    )

    configured_splits = int(config.splits_kv)
    if bool(config.uses_nontrivial_grouped_q_layout):
        q_group_token_bases = range(0, seq_len_q, int(config.q_tokens_per_cta))
        q_tokens_per_group = int(config.q_tokens_per_cta)
    else:
        q_group_token_bases = range(seq_len_q)
        q_tokens_per_group = 1
    for seq_len_kv in seq_lens:
        if seq_len_kv <= 0 or seq_len_kv > max_kv_len:
            return False
        for q_token_base in q_group_token_bases:
            valid_k = seq_len_kv
            if mask_type == "causal":
                q_token_end = min(q_token_base + q_tokens_per_group, seq_len_q)
                valid_k = max(seq_len_kv - seq_len_q + q_token_end, 0)
            if (
                compute_runtime_active_splits_kv(
                    valid_k=valid_k,
                    tile_size_kv=int(config.tile_size_kv),
                    num_insts_kv=int(config.num_insts_kv),
                    configured_splits_kv=configured_splits,
                )
                != configured_splits
            ):
                return False
    return True


def _planned_kv_lengths_mode(
    seq_lens: tuple[int, ...],
    *,
    max_kv_len: int,
) -> Literal["dynamic", "planned_uniform_max"]:
    """Classify host length evidence for fixed-length kernel scheduling.

    When every evidenced request is exactly the compiled maximum, native page
    addressing still uses the runtime block table while task domains and
    masks can use the compile-time length. The plan retains the evidenced
    lengths and supplies an owned device copy to every launch.
    """

    if not seq_lens or max_kv_len <= 0:
        return "dynamic"
    if all(seq_len == max_kv_len for seq_len in seq_lens):
        return "planned_uniform_max"
    return "dynamic"


def _planned_kv_domain_has_unpaired_tail(
    config: "FmhaDecodeConfig", max_kv_len: int
) -> bool:
    """Return whether the planned K domain ends with one inactive KV instance."""

    tile_size_kv = int(config.tile_size_kv)
    num_insts_kv = int(config.num_insts_kv)
    total_kv_tiles = (max_kv_len + tile_size_kv - 1) // tile_size_kv
    return total_kv_tiles % num_insts_kv != 0


def _align_up(value: int, alignment: int = _WORKSPACE_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _dtype_itemsize(dtype: torch.dtype) -> int:
    itemsize = {
        torch.int8: 1,
        torch.uint8: 1,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int32: 4,
        torch.float32: 4,
    }
    try:
        return itemsize[dtype]
    except KeyError as error:
        raise TypeError(f"unsupported workspace section dtype {dtype}") from error


def _append_workspace_section(
    byte_end: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> tuple[_WorkspaceSection, int]:
    byte_offset = _align_up(byte_end)
    byte_size = math.prod(shape) * _dtype_itemsize(dtype)
    return (
        _WorkspaceSection(byte_offset, byte_size, shape, dtype),
        byte_offset + byte_size,
    )


def _make_decode_workspace_layout(
    scratch_shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    output_dtype: torch.dtype,
    *,
    use_separate_reduction_kernel: bool,
    use_split_kv: bool = False,
) -> _DecodeWorkspaceLayout:
    partial_o_shape, partial_stats_shape, counter_shape = scratch_shapes
    partial_o_dtype = output_dtype
    if use_separate_reduction_kernel and output_dtype in (
        torch.bfloat16,
        torch.float8_e4m3fn,
    ):
        partial_o_dtype = torch.bfloat16
    elif output_dtype == torch.float8_e4m3fn or partial_o_shape == (1, 1, 1, 1, 1):
        partial_o_dtype = torch.float16

    byte_end = 0
    partial_o, byte_end = _append_workspace_section(
        byte_end, partial_o_shape, partial_o_dtype
    )
    partial_stats, byte_end = _append_workspace_section(
        byte_end, partial_stats_shape, torch.float32
    )
    split_kv_counter, byte_end = _append_workspace_section(
        byte_end, counter_shape, torch.int32
    )
    cu_seqlens_q, byte_end = _append_workspace_section(byte_end, (1,), torch.int32)
    attention_sinks, byte_end = _append_workspace_section(byte_end, (1,), torch.float32)
    return _DecodeWorkspaceLayout(
        partial_o=partial_o,
        partial_stats=partial_stats,
        split_kv_counter=split_kv_counter,
        cu_seqlens_q=cu_seqlens_q,
        attention_sinks=attention_sinks,
        uses_split_kv=use_split_kv,
        total_bytes=_align_up(byte_end),
    )


def _validate_workspace_buffer(
    workspace_buffer: torch.Tensor,
    *,
    device: torch.device,
    required_bytes: int,
) -> None:
    if not isinstance(workspace_buffer, torch.Tensor):
        raise TypeError("workspace_buffer must be a torch.Tensor")
    if workspace_buffer.dtype not in _WORKSPACE_DTYPES:
        raise TypeError("workspace_buffer must have dtype torch.int8 or torch.uint8")
    if workspace_buffer.device != device:
        raise ValueError(
            f"workspace_buffer must be on {device}, got {workspace_buffer.device}"
        )
    if not workspace_buffer.is_contiguous():
        raise ValueError("workspace_buffer must be contiguous")
    if workspace_buffer.data_ptr() % 32 != 0:
        raise ValueError("workspace_buffer data pointer must be 32-byte aligned")
    available_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    if available_bytes < required_bytes:
        raise ValueError(
            "workspace_buffer is too small: requires at least "
            f"{required_bytes} bytes, got {available_bytes}"
        )


def _workspace_section_view(
    workspace_buffer: torch.Tensor, section: _WorkspaceSection
) -> torch.Tensor:
    workspace_bytes = workspace_buffer.reshape(-1).view(torch.uint8)
    section_bytes = workspace_bytes[
        section.byte_offset : section.byte_offset + section.byte_size
    ]
    return section_bytes.view(section.dtype).view(section.shape)


def _bind_decode_workspace(
    workspace_buffer: torch.Tensor, layout: _DecodeWorkspaceLayout
) -> _DecodeWorkspaceViews:
    return _DecodeWorkspaceViews(
        partial_o=_workspace_section_view(workspace_buffer, layout.partial_o),
        partial_stats=_workspace_section_view(workspace_buffer, layout.partial_stats),
        split_kv_counter=_workspace_section_view(
            workspace_buffer, layout.split_kv_counter
        ),
        cu_seqlens_q=_workspace_section_view(workspace_buffer, layout.cu_seqlens_q),
        attention_sinks=_workspace_section_view(
            workspace_buffer, layout.attention_sinks
        ),
    )


def _validate_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_head_dim(head_dim: int) -> int:
    head_dim = _validate_positive_int(head_dim, "head_dim")
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(
            "attention-ts decode requires head_dim in "
            f"{_SUPPORTED_HEAD_DIMS}, got {head_dim}"
        )
    return head_dim


def _validate_seq_len_q(seq_len_q: int) -> int:
    return _validate_positive_int(seq_len_q, "seq_len_q")


def _validate_window_left(window_left: int, mask_type: str) -> int:
    if isinstance(window_left, bool) or not isinstance(window_left, int):
        raise TypeError("window_left must be an integer")
    if window_left < -1:
        raise ValueError("window_left must be -1 (disabled) or non-negative")
    if window_left > 2**31 - 2:
        raise ValueError("window_left must be no larger than 2**31 - 2")
    if window_left >= 0 and mask_type != "causal":
        raise ValueError("window_left requires mask_type='causal'")
    return window_left


def _resolve_q_mode(
    *,
    seq_len_q: int,
    qo_indptr: Optional[torch.Tensor],
    max_seq_len_q: Optional[int],
    require_packed_max: bool,
) -> tuple[bool, Optional[int]]:
    """Resolve fixed versus packed Q without exposing an internal mode knob."""

    seq_len_q = _validate_seq_len_q(seq_len_q)
    if qo_indptr is None:
        if max_seq_len_q is not None:
            max_seq_len_q = _validate_seq_len_q(max_seq_len_q)
            if max_seq_len_q != seq_len_q:
                raise ValueError(
                    "fixed seq_len_q and max_seq_len_q must agree: "
                    f"got {seq_len_q} and {max_seq_len_q}"
                )
        return False, seq_len_q
    if max_seq_len_q is None and seq_len_q != 1:
        # Preserve the legacy name as a packed static-bound alias.  The
        # nullable qo_indptr alone still selects fixed versus packed storage.
        return True, seq_len_q
    if max_seq_len_q is None:
        if require_packed_max:
            raise ValueError(
                "max_seq_len_q is required with qo_indptr for the standalone "
                "workspace/JIT interface"
            )
        return True, None
    max_seq_len_q = _validate_seq_len_q(max_seq_len_q)
    if seq_len_q != 1 and seq_len_q != max_seq_len_q:
        raise ValueError(
            "seq_len_q and max_seq_len_q must agree when both provide the "
            f"packed static bound: got {seq_len_q} and {max_seq_len_q}"
        )
    return True, max_seq_len_q


def _validate_max_kv_len(value: int, name: str) -> int:
    """Reserve the largest padded decode K/V tile in signed Int32."""

    value = _validate_positive_int(value, name)
    if value > _DECODE_MAX_KV_LEN:
        raise NotImplementedError(
            f"{name} must be <= {_DECODE_MAX_KV_LEN} so padded FMHA decode "
            "K/V coordinates fit in a signed int32"
        )
    return value


def _validate_decode_policy_kv_tile_size(config: "FmhaDecodeConfig") -> None:
    """Keep the public K/V bound coupled to generated decode policies."""

    tile_size_kv = int(config.tile_size_kv)
    if tile_size_kv > _DECODE_MAX_KV_TILE_SIZE:
        raise RuntimeError(
            "FMHA decode Int32 extent safety assumes a K/V tile no larger "
            f"than {_DECODE_MAX_KV_TILE_SIZE}, got {tile_size_kv}"
        )


def _validate_decode_query_head_extent(
    *,
    batch_size: int,
    num_qo_heads: int,
    max_seq_len_q: int,
) -> None:
    """Keep every fixed-capacity or packed Q/head coordinate in Int32."""

    batch_size = _validate_positive_int(batch_size, "batch_size")
    num_qo_heads = _validate_positive_int(num_qo_heads, "num_qo_heads")
    max_seq_len_q = _validate_seq_len_q(max_seq_len_q)
    extent = batch_size * max_seq_len_q * num_qo_heads
    if extent > _MAX_INT32:
        raise NotImplementedError(
            "batch_size * max_seq_len_q * num_qo_heads must fit in a signed int32"
        )


def _compact_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= int(extent)
    return tuple(reversed(strides))


def _validate_exact_compact_strides(
    tensor: torch.Tensor,
    name: str,
    layout: str,
) -> None:
    expected_strides = _compact_strides(tuple(tensor.shape))
    if tensor.stride() != expected_strides:
        raise ValueError(
            f"{name} must have compact {layout} strides "
            f"{expected_strides}, but has {tensor.stride()}"
        )


def _validate_16byte_alignment(tensor: torch.Tensor, name: str) -> None:
    if tensor.data_ptr() % 16 != 0:
        raise ValueError(f"{name} data pointer must be 16-byte aligned")


def _validate_layout(kv_layout: str) -> None:
    if not isinstance(kv_layout, str):
        raise TypeError("kv_layout must be a string")
    if kv_layout == "NHD":
        raise NotImplementedError(
            "attention-ts decode currently supports kv_layout='HND' only"
        )
    if kv_layout != "HND":
        raise ValueError(f"kv_layout must be exactly 'HND', got {kv_layout!r}")


def _validate_mask(mask_type: str) -> None:
    if not isinstance(mask_type, str):
        raise TypeError("mask_type must be a string")
    if mask_type not in ("dense", "causal"):
        raise ValueError(
            f"mask_type must be exactly 'dense' or 'causal', got {mask_type!r}"
        )


def _validate_page_size(page_size: int) -> int:
    page_size = _validate_positive_int(page_size, "page_size")
    if page_size not in _SUPPORTED_PAGE_SIZES:
        raise ValueError(
            "attention-ts decode requires page_size in "
            f"{_SUPPORTED_PAGE_SIZES}, got {page_size}"
        )
    return page_size


def _validate_head_geometry(num_qo_heads: int, num_kv_heads: int) -> None:
    num_qo_heads = _validate_positive_int(num_qo_heads, "num_qo_heads")
    num_kv_heads = _validate_positive_int(num_kv_heads, "num_kv_heads")
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            "num_qo_heads must be divisible by num_kv_heads, got "
            f"{num_qo_heads} and {num_kv_heads}"
        )
    head_ratio = num_qo_heads // num_kv_heads
    if head_ratio > _MAX_HEAD_RATIO:
        raise ValueError(
            "attention-ts decode requires "
            f"1 <= Hq/Hkv <= {_MAX_HEAD_RATIO}, got {head_ratio}"
        )


def _validate_scale(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a positive Python scalar")
    try:
        value_as_float = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a positive Python scalar") from error
    if not math.isfinite(value_as_float) or value_as_float <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    try:
        value_as_float32 = struct.unpack("=f", struct.pack("=f", value_as_float))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(
            f"{name} must be representable as a positive float32"
        ) from error
    if not math.isfinite(value_as_float32) or value_as_float32 <= 0.0:
        raise ValueError(f"{name} must be representable as a positive float32")
    return value_as_float32


def _dtype_key(dtype: torch.dtype) -> str:
    if not isinstance(dtype, torch.dtype):
        raise TypeError("attention-ts dtypes must be torch.dtype values")
    keys = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float8_e4m3fn: "float8_e4m3fn",
        torch.uint8: "float4_e2m1fn",
    }
    try:
        return keys[dtype]
    except KeyError as error:
        raise NotImplementedError(
            "attention-ts decode supports torch.float16, torch.bfloat16, "
            "torch.float8_e4m3fn, and packed NVFP4 K/V in torch.uint8; "
            f"got {dtype}"
        ) from error


def _cutlass_dtype(dtype_key: str):
    """Return the cutlass numeric type named by a dtype key."""

    import cutlass

    return {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
        "float4_e2m1fn": cutlass.Float4E2M1FN,
        "int8": cutlass.Int8,
    }[dtype_key]


def _validate_dtype_pair(
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    output_dtype: torch.dtype,
    *,
    allow_fp8_bf16_output: bool = False,
) -> None:
    _dtype_key(q_dtype)
    _dtype_key(k_dtype)
    _dtype_key(v_dtype)
    _dtype_key(output_dtype)
    matching_kv = k_dtype == v_dtype
    supported = (
        (
            q_dtype == torch.float16
            and k_dtype == v_dtype == torch.float16
            and output_dtype == torch.float16
        )
        or (
            q_dtype == torch.bfloat16
            and output_dtype == torch.bfloat16
            and (
                (
                    matching_kv
                    and k_dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.uint8)
                )
                or (k_dtype == torch.bfloat16 and v_dtype == torch.float8_e4m3fn)
            )
        )
        or (
            q_dtype == torch.float8_e4m3fn
            and matching_kv
            and k_dtype in (torch.float8_e4m3fn, torch.uint8)
            and output_dtype
            in (
                (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
                if allow_fp8_bf16_output
                else (torch.float16, torch.float8_e4m3fn)
            )
        )
    )
    if not supported:
        raise NotImplementedError(
            "attention-ts decode supports FP16->FP16, BF16->BF16, "
            + (
                "FP8-E4M3->FP16/BF16, and FP8-E4M3->FP8-E4M3; "
                if allow_fp8_bf16_output
                else "FP8-E4M3->FP16, and FP8-E4M3->FP8-E4M3; "
            )
            + "BF16 Q + FP8 K/V, BF16/FP8-E4M3 Q + NVFP4 K/V, "
            + "and BF16 Q/K + FP8 V; got "
            + f"Q_{q_dtype}_K_{k_dtype}_V_{v_dtype}->{output_dtype}"
        )


def _resolve_kv_dtypes(
    q_dtype: torch.dtype,
    k_dtype: Optional[torch.dtype],
    v_dtype: Optional[torch.dtype],
    kv_dtype: Optional[torch.dtype],
) -> tuple[torch.dtype, torch.dtype]:
    """Resolve the common-KV compatibility alias without hiding disagreement."""
    if kv_dtype is not None:
        if (k_dtype is not None and k_dtype != kv_dtype) or (
            v_dtype is not None and v_dtype != kv_dtype
        ):
            raise ValueError("explicit K/V dtypes must agree with the common KV dtype")
        k_dtype = v_dtype = kv_dtype
    if k_dtype is None:
        k_dtype = q_dtype
    if v_dtype is None:
        v_dtype = k_dtype
    return k_dtype, v_dtype


def _device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _validate_runtime_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("attention-ts decode tensors must be CUDA tensors")
    device_index = device.index
    with torch.cuda.device(device_index):
        capability = torch.cuda.get_device_capability(device_index)
    if capability not in _SUPPORTED_COMPUTE_CAPABILITIES:
        raise NotImplementedError(
            "attention-ts decode requires an SM100a/B200, SM103a/B300 or "
            "SM107a/Rubin GPU; "
            f"device cuda:{device_index} has compute capability {capability}"
        )
    # Rubin runs through the sm_100f family target; a CuTe DSL older than 4.8
    # cannot emit for it unless CUTE_DSL_ARCH=sm_100f is set before import.
    if capability == (10, 7):
        from flashinfer.cute_dsl.utils import require_cute_dsl_arch

        require_cute_dsl_arch(device_index)


def _resolve_cuda_device(
    device: Optional[Union[int, str, torch.device]],
) -> tuple[torch.device, int]:
    """Normalize a device specifier without checking CUDA support."""

    if device is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, int) and not isinstance(device, bool):
        resolved = torch.device("cuda", device)
    else:
        try:
            resolved = torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise TypeError("device must identify one CUDA device") from error
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
    # Preserve non-CUDA device types for the caller's explicit validation.
    device_index = (
        _device_index(resolved) if resolved.type == "cuda" else (resolved.index or 0)
    )
    return torch.device(resolved.type, device_index), device_index


def _validate_q(
    q: torch.Tensor,
    *,
    seq_len_q: int = 1,
    use_packed_q: bool = False,
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
    num_qo_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    q_dtype: Optional[torch.dtype] = None,
) -> None:
    seq_len_q = _validate_seq_len_q(seq_len_q)
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    expected_rank = 3 if use_packed_q or seq_len_q == 1 else 4
    if q.ndim != expected_rank:
        expected_layout = (
            "[total_q, Hq, D]"
            if use_packed_q
            else ("[B, Hq, D]" if seq_len_q == 1 else "[B, SQ, Hq, D]")
        )
        raise ValueError(f"q must have shape {expected_layout} for this Q layout")
    if not use_packed_q and seq_len_q > 1 and q.shape[1] != seq_len_q:
        raise ValueError(
            f"q sequence length must match seq_len_q ({seq_len_q}), got {q.shape[1]}"
        )
    num_heads = int(q.shape[-2])
    if q.shape[0] <= 0 or num_heads <= 0:
        leading_name = "total Q token count" if use_packed_q else "batch size"
        raise ValueError(f"q {leading_name} and head count must be positive")
    _validate_head_dim(int(q.shape[-1]))
    if q.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise NotImplementedError(f"unsupported attention-ts q dtype {q.dtype}")
    if q.device.type != "cuda":
        raise ValueError("q must be a CUDA tensor")
    if device is not None and q.device != device:
        raise ValueError(f"q must be on planned device {device}, got {q.device}")
    if not use_packed_q and batch_size is not None and q.shape[0] != batch_size:
        raise ValueError(
            f"q batch size must match the plan ({batch_size}), got {q.shape[0]}"
        )
    if use_packed_q and batch_size is not None:
        total_q = int(q.shape[0])
        max_total_q = batch_size * seq_len_q
        if total_q < batch_size or total_q > max_total_q:
            raise ValueError(
                "packed q token count must be within "
                f"[{batch_size}, {max_total_q}], got {total_q}"
            )
    if num_qo_heads is not None and num_heads != num_qo_heads:
        raise ValueError(
            f"q head count must match the plan ({num_qo_heads}), got {num_heads}"
        )
    if head_dim is not None and q.shape[-1] != head_dim:
        raise ValueError(
            f"q head dimension must match the plan ({head_dim}), got {q.shape[-1]}"
        )
    if q_dtype is not None and q.dtype != q_dtype:
        raise ValueError(f"q dtype must match the plan ({q_dtype}), got {q.dtype}")
    layout = (
        "[total_q, Hq, D]"
        if use_packed_q
        else ("[B, Hq, D]" if seq_len_q == 1 else "[B, SQ, Hq, D]")
    )
    _validate_exact_compact_strides(q, "q", layout)
    _validate_16byte_alignment(q, "q")


def _validate_qo_indptr(
    qo_indptr: torch.Tensor,
    *,
    expected_device: torch.device,
    batch_size: int,
) -> None:
    """Validate packed-Q metadata without synchronizing device values."""

    if not isinstance(qo_indptr, torch.Tensor):
        raise TypeError("qo_indptr must be a torch.Tensor")
    if qo_indptr.ndim != 1:
        raise ValueError("qo_indptr must be one-dimensional")
    if qo_indptr.dtype != torch.int32:
        raise TypeError("qo_indptr must have dtype torch.int32")
    if qo_indptr.device != expected_device:
        raise ValueError(
            f"qo_indptr must be on {expected_device}, got {qo_indptr.device}"
        )
    if qo_indptr.numel() != batch_size + 1:
        raise ValueError(
            "qo_indptr must have B + 1 elements: expected "
            f"{batch_size + 1}, got {qo_indptr.numel()}"
        )
    if not qo_indptr.is_contiguous():
        raise ValueError("qo_indptr must be contiguous")
    if qo_indptr.data_ptr() % 4 != 0:
        raise ValueError("qo_indptr data pointer must be 4-byte aligned")


def _read_packed_q_plan_metadata(
    qo_indptr: torch.Tensor,
) -> tuple[int, int, tuple[int, ...]]:
    """Validate Q offsets at plan time and return max, total, and row lengths."""

    offsets = tuple(int(value) for value in qo_indptr.tolist())
    if offsets[0] != 0:
        raise ValueError("qo_indptr must start at zero")
    q_lengths = tuple(
        end - begin for begin, end in zip(offsets[:-1], offsets[1:], strict=True)
    )
    if any(length <= 0 for length in q_lengths):
        raise ValueError("qo_indptr must be strictly increasing")
    exact_max_seq_len_q = max(q_lengths, default=0)
    if exact_max_seq_len_q <= 0:
        raise ValueError("a packed-Q plan must contain at least one query token")
    return exact_max_seq_len_q, offsets[-1], q_lengths


def _validate_packed_q_plan_values(
    qo_indptr: torch.Tensor,
    *,
    max_seq_len_q: int,
    expected_total_q: Optional[int] = None,
) -> tuple[int, int]:
    """Synchronize once to validate the packed-Q values against a static bound."""

    derived_max_seq_len_q, total_q, _ = _read_packed_q_plan_metadata(qo_indptr)
    if derived_max_seq_len_q > max_seq_len_q:
        raise ValueError(
            "qo_indptr contains a per-request Q length larger than "
            f"max_seq_len_q ({max_seq_len_q}): got {derived_max_seq_len_q}"
        )
    if expected_total_q is not None and total_q != expected_total_q:
        raise ValueError(
            "the final qo_indptr offset must equal the packed q token count: "
            f"expected {expected_total_q}, got {total_q}"
        )
    return derived_max_seq_len_q, total_q


def _validate_hnd_inner_strides(tensor: torch.Tensor, name: str) -> int:
    _, num_kv_heads, page_size, head_dim = tensor.shape
    expected_inner = (page_size * head_dim, head_dim, 1)
    if tensor.stride()[1:] != expected_inner:
        raise ValueError(
            f"{name} must have compact HND inner strides {expected_inner}, "
            f"got {tensor.stride()[1:]}"
        )
    page_stride = int(tensor.stride(0))
    compact_page_elements = num_kv_heads * page_size * head_dim
    if page_stride < compact_page_elements:
        raise ValueError(
            f"{name} pages overlap: outer stride {page_stride} is smaller than "
            f"{compact_page_elements}"
        )
    if page_stride > 2**63 - 1:
        raise ValueError(f"{name} outer page stride exceeds signed int64")
    _validate_16byte_alignment(tensor, name)
    if page_stride * tensor.element_size() % 16 != 0:
        raise ValueError(f"{name} outer page stride must be 16-byte aligned")
    return page_stride


def _normalize_paged_kv_cache(
    paged_kv_cache: PagedKVCache,
    *,
    expected_device: torch.device,
    logical_head_dim: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int, int, int]:
    """Return compact HND views for block-sparse and legacy paged paths."""

    views = _normalize_paged_kv_cache_views(
        paged_kv_cache,
        expected_device=expected_device,
    )
    k_cache, v_cache = views[:2]
    inferred_head_dim = views[5]
    if logical_head_dim is not None and inferred_head_dim != logical_head_dim:
        raise ValueError(
            "paged_kv_cache storage head dimension does not match the logical "
            f"head dimension {logical_head_dim}: got {int(k_cache.shape[-1])}"
        )
    k_page_stride = _validate_hnd_inner_strides(k_cache, "K cache")
    v_page_stride = _validate_hnd_inner_strides(v_cache, "V cache")
    return (*views, k_page_stride, v_page_stride)


def _normalize_paged_kv_scale_factors(
    kv_scale_factors: Optional[PagedKVScaleFactors],
    *,
    k_cache: torch.Tensor,
    logical_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate NVFP4 scale tensors or create homogeneous-mode placeholders."""

    if k_cache.dtype != torch.uint8:
        if kv_scale_factors is not None:
            raise ValueError(
                "kv_scale_factors are accepted only with packed NVFP4 torch.uint8 K/V"
            )
        placeholder = k_cache[0, 0, 0, :1].view(torch.uint8)[:1]
        return placeholder, placeholder
    if kv_scale_factors is None:
        raise ValueError(
            "packed NVFP4 torch.uint8 K/V requires kv_scale_factors=(K_SF, V_SF)"
        )
    if not isinstance(kv_scale_factors, tuple) or len(kv_scale_factors) != 2:
        raise TypeError("kv_scale_factors must be a (K_SF, V_SF) tensor tuple")
    k_sf_cache, v_sf_cache = kv_scale_factors
    if not isinstance(k_sf_cache, torch.Tensor) or not isinstance(
        v_sf_cache, torch.Tensor
    ):
        raise TypeError("kv_scale_factors tuple members must be torch.Tensor")
    expected_shape = (*k_cache.shape[:-1], logical_head_dim // 16)
    for scale, name in (
        (k_sf_cache, "K scale factors"),
        (v_sf_cache, "V scale factors"),
    ):
        if tuple(scale.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(scale.shape)}"
            )
        if scale.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"{name} must have dtype torch.float8_e4m3fn, got {scale.dtype}"
            )
        if scale.device != k_cache.device:
            raise ValueError(f"{name} must be on {k_cache.device}, got {scale.device}")
        _validate_exact_compact_strides(
            scale,
            name,
            "[pages, Hkv, page_size, D/16]",
        )
        _validate_16byte_alignment(scale, name)
    return k_sf_cache, v_sf_cache


def _validate_block_tables(
    block_tables: torch.Tensor,
) -> tuple[torch.device, int, int]:
    """Validate a fixed page table without requiring sequence-length storage."""

    if not isinstance(block_tables, torch.Tensor):
        raise TypeError("block_tables must be a torch.Tensor")
    if block_tables.ndim != 2:
        raise ValueError("block_tables must have shape [B, C]")
    if block_tables.dtype != torch.int32:
        raise TypeError("block_tables must have dtype torch.int32")
    if block_tables.device.type != "cuda":
        raise ValueError("block_tables must be a CUDA tensor")
    if block_tables.data_ptr() % 4 != 0:
        raise ValueError("block_tables data pointer must be 4-byte aligned")

    batch_size = int(block_tables.shape[0])
    if batch_size <= 0:
        raise ValueError("block_tables must contain at least one request row")
    table_capacity = int(block_tables.shape[1])
    if table_capacity <= 0:
        raise ValueError("block_tables must contain at least one page column")
    if table_capacity > _MAX_INT32:
        raise ValueError("block_tables column count exceeds signed int32")
    if block_tables.stride(1) != 1:
        raise ValueError("block_tables must be contiguous within each row")
    if block_tables.stride(0) < table_capacity:
        raise ValueError(
            "block_tables rows must not overlap: row stride must be at least "
            f"the column count ({table_capacity}), got {block_tables.stride(0)}"
        )
    return block_tables.device, batch_size, table_capacity


def _validate_block_table_metadata(
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[torch.device, int, int]:
    """Validate fixed page-table and sequence-length tensor structure."""

    block_table_device, block_table_batch_size, table_capacity = _validate_block_tables(
        block_tables
    )

    if not isinstance(seq_lens, torch.Tensor):
        raise TypeError("seq_lens must be a torch.Tensor")
    if seq_lens.ndim != 1:
        raise ValueError("seq_lens must be one-dimensional")
    if seq_lens.dtype != torch.int32:
        raise TypeError("seq_lens must have dtype torch.int32")
    if seq_lens.device.type != "cuda":
        raise ValueError("seq_lens must be a CUDA tensor")
    if not seq_lens.is_contiguous():
        raise ValueError("seq_lens must be contiguous")
    if seq_lens.data_ptr() % 4 != 0:
        raise ValueError("seq_lens data pointer must be 4-byte aligned")
    if block_table_device != seq_lens.device:
        raise ValueError("block_tables and seq_lens must be on the same device")

    batch_size = int(seq_lens.numel())
    if batch_size <= 0:
        raise ValueError("seq_lens must contain at least one request")
    if block_table_batch_size != batch_size:
        raise ValueError(
            "block_tables must have one row per request: expected "
            f"{batch_size}, got {block_table_batch_size}"
        )
    return seq_lens.device, batch_size, table_capacity


def _decode_output_shape(
    *,
    batch_size: int,
    num_qo_heads: int,
    seq_len_q: int,
    head_dim: int,
    total_q_tokens: Optional[int] = None,
) -> tuple[int, ...]:
    if total_q_tokens is not None:
        return (total_q_tokens, num_qo_heads, head_dim)
    if seq_len_q == 1:
        return (batch_size, num_qo_heads, head_dim)
    return (batch_size, seq_len_q, num_qo_heads, head_dim)


def _validate_out(
    out: torch.Tensor,
    *,
    q: torch.Tensor,
    expected_shape: tuple[int, ...],
    seq_len_q: int,
    use_packed_q: bool,
    output_dtype: torch.dtype,
) -> None:
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if tuple(out.shape) != expected_shape:
        raise ValueError(
            f"out must have shape {expected_shape}, got {tuple(out.shape)}"
        )
    if out.dtype != output_dtype:
        raise ValueError(f"out must have dtype {output_dtype}, got {out.dtype}")
    if out.device != q.device:
        raise ValueError(f"out must be on {q.device}, got {out.device}")
    layout = (
        "[total_q, Hq, D]"
        if use_packed_q
        else ("[B, Hq, D]" if seq_len_q == 1 else "[B, SQ, Hq, D]")
    )
    _validate_exact_compact_strides(out, "out", layout)
    _validate_16byte_alignment(out, "out")


def _decode_scratch_shapes(
    cfg: "FmhaDecodeConfig",
    *,
    batch_size,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len_q: int,
):
    """Return split-workspace shapes for a static config and batch extent."""

    if not cfg.use_split_kv:
        return (
            (1, 1, 1, 1, 1),
            (1, 1, 1, 1, 2),
            (1, 1, 1),
        )

    from .kernels.fmha_decode.fmha_decode_config import make_q_tile_geometry

    head_ratio = num_qo_heads // num_kv_heads
    geometry = make_q_tile_geometry(
        rows_per_cta=cfg.tile_size_q,
        heads_q_per_kv=head_ratio,
        groups_tokens_heads_q=cfg.groups_tokens_heads_q,
    )
    num_q_groups = max(int(geometry.num_q_ctas(seq_len_q)), 1)
    partial_o_shape = (
        batch_size,
        num_kv_heads,
        int(cfg.max_splits_kv),
        head_ratio * seq_len_q,
        head_dim,
    )
    partial_stats_shape = (
        partial_o_shape[:-1]
        if cfg.use_separate_reduction_kernel
        else partial_o_shape[:-1] + (2,)
    )
    counter_shape = (batch_size, num_kv_heads, num_q_groups)
    return partial_o_shape, partial_stats_shape, counter_shape


def _decode_launch_spec_from_config(
    cfg: "FmhaDecodeConfig",
    *,
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len_q: int,
    max_active_clusters: int,
) -> _DecodeLaunchSpec:
    """Derive policy and scratch geometry from one finalized FMHA config."""

    scratch_shapes = _decode_scratch_shapes(
        cfg,
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len_q=seq_len_q,
    )

    return _DecodeLaunchSpec(
        config=cfg,
        max_active_clusters=int(max_active_clusters),
        policy=_decode_policy_from_config(cfg),
        scratch_shapes=scratch_shapes,
    )


@functools.cache
def _resolve_decode_launch_spec(
    device_index: int,
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    max_kv_len: int,
    seq_len_q: int,
    q_dtype_key: str,
    k_dtype_key: str,
    v_dtype_key: str,
    output_dtype_key: str,
    kv_layout: str,
    mask_type: str,
    use_packed_q: bool,
    window_left: int,
    storage_page_size: Optional[int] = None,
    use_q_token_kv_block_sparse_route: bool = False,
    use_pdl: bool = False,
    split_kv: bool = True,
    share_pattern_across_kv_heads: bool = True,
) -> _DecodeLaunchSpec:
    """Resolve automatic policy and workspace geometry without compiling."""

    seq_len_q = _validate_seq_len_q(seq_len_q)
    _validate_head_geometry(num_qo_heads, num_kv_heads)
    _validate_decode_query_head_extent(
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        max_seq_len_q=seq_len_q,
    )
    max_kv_len = _validate_max_kv_len(max_kv_len, "max_kv_len")
    window_left = _validate_window_left(window_left, mask_type)
    storage_page_size = _validate_storage_page_size(
        page_size,
        page_size if storage_page_size is None else storage_page_size,
    )

    from .kernels.fmha_decode.fmha_decode_config import (
        MIN_LOOP_ITERS_PER_SPLIT,
        get_max_active_clusters_for_cluster_size,
        make_decode_config,
        make_q_tile_geometry,
    )

    if kv_layout != "HND":
        raise ValueError("the cached TS decode compiler accepts HND only")
    q_dtype = _cutlass_dtype(q_dtype_key)
    k_dtype = _cutlass_dtype(k_dtype_key)
    v_dtype = _cutlass_dtype(v_dtype_key)
    output_dtype = _cutlass_dtype(output_dtype_key)

    def make_config(
        args: object | None = None,
        *,
        split_kv_mode: str = "disabled",
        splits_kv: int = -1,
        max_splits_kv: int | None = None,
        min_loop_iters_per_split: int = MIN_LOOP_ITERS_PER_SPLIT,
    ) -> "FmhaDecodeConfig":
        return make_decode_config(
            headdim=head_dim,
            args=args,
            seq_len_q=seq_len_q,
            seq_len_kv=max_kv_len,
            batch_size=batch_size,
            num_heads_q=num_qo_heads,
            num_heads_kv=num_kv_heads,
            q_dtype=q_dtype,
            k_dtype=k_dtype,
            v_dtype=v_dtype,
            o_dtype=output_dtype,
            qkv_layout="pagedKv",
            num_tokens_per_page=page_size,
            storage_tokens_per_page=storage_page_size,
            split_kv_mode=split_kv_mode,
            splits_kv=splits_kv,
            max_splits_kv=max_splits_kv,
            min_loop_iters_per_split=min_loop_iters_per_split,
            mask_type=mask_type,
            sliding_window_causal=window_left >= 0,
            attention_window_size=window_left + 1 if window_left >= 0 else 0,
            auto_tuner=True,
            split_kv=split_kv,
        )

    def q_ctas(config: "FmhaDecodeConfig") -> int:
        geometry = make_q_tile_geometry(
            rows_per_cta=config.tile_size_q,
            heads_q_per_kv=num_qo_heads // num_kv_heads,
            groups_tokens_heads_q=config.groups_tokens_heads_q,
        )
        return max(int(geometry.num_q_ctas(seq_len_q)), 1)

    def fits_one_service_wave(config: "FmhaDecodeConfig", num_q_ctas: int) -> bool:
        logical_grid = batch_size * num_kv_heads * num_q_ctas
        if config.use_persistent_scheduler:
            return False
        if config.use_cluster_smem_reduction:
            cluster_capacity = get_max_active_clusters_for_cluster_size(
                int(config.splits_kv)
            )
            return cluster_capacity > 0 and logical_grid <= cluster_capacity
        split_fanout = int(config.splits_kv) if config.use_split_kv else 1
        return logical_grid * split_fanout <= max_active_clusters

    # Device capacity participates in automatic selection. Resolve it in the
    # target device context without introducing caller-visible policy knobs.
    with torch.cuda.device(device_index):
        max_active_clusters = get_max_active_clusters_for_cluster_size(1)
        if use_q_token_kv_block_sparse_route:
            cfg = _resolve_q_token_kv_block_sparse_decode_config(
                make_config,
                batch_size=batch_size,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                max_kv_len=max_kv_len,
                seq_len_q=seq_len_q,
                q_dtype_key=q_dtype_key,
                k_dtype_key=k_dtype_key,
                v_dtype_key=v_dtype_key,
                output_dtype_key=output_dtype_key,
                mask_type=mask_type,
                use_packed_q=use_packed_q,
                window_left=window_left,
                max_active_clusters=max_active_clusters,
                use_pdl=use_pdl,
                split_kv=split_kv,
                share_pattern_across_kv_heads=share_pattern_across_kv_heads,
            )
        else:
            config_overrides = {"use_pdl": use_pdl}
            if use_packed_q:
                config_overrides["use_variable_seqlens_q"] = True
            cfg = make_config(config_overrides)

            # A grouped fixed-Q launch can leave most of the first service wave
            # idle. In that regime, evaluate the narrowest supported Swaps head
            # band. Keep it only when the extra head-band CTAs fit in the same
            # resident wave without reducing KV fanout or changing topology.
            if (
                seq_len_q == 1
                and not use_packed_q
                and cfg.groups_tokens_heads_q
                and not cfg.use_keeps_mma_ab
            ):
                grouped_q_ctas = q_ctas(cfg)
                head_band_geometry = make_q_tile_geometry(
                    rows_per_cta=8,
                    heads_q_per_kv=num_qo_heads // num_kv_heads,
                    groups_tokens_heads_q=False,
                )
                head_band_q_ctas = max(int(head_band_geometry.num_q_ctas(seq_len_q)), 1)
                head_band_cfg = None
                if head_band_q_ctas > grouped_q_ctas and fits_one_service_wave(
                    cfg, head_band_q_ctas
                ):
                    try:
                        head_band_cfg = make_config(
                            {
                                "groups_tokens_heads_q": False,
                                "tile_size_q": 8,
                                "use_pdl": use_pdl,
                            }
                        )
                    except ValueError:
                        head_band_cfg = None
                if head_band_cfg is not None:
                    same_launch_topology = all(
                        getattr(cfg, field) == getattr(head_band_cfg, field)
                        for field in (
                            "use_split_kv",
                            "splits_kv",
                            "max_splits_kv",
                            "use_cluster_smem_reduction",
                            "use_separate_reduction_kernel",
                            "use_persistent_scheduler",
                        )
                    )
                    if same_launch_topology and fits_one_service_wave(
                        head_band_cfg, q_ctas(head_band_cfg)
                    ):
                        cfg = head_band_cfg

            q_token_kv_block_sparse_scattered_fp8 = (
                page_size == 4
                and storage_page_size > page_size
                and q_dtype_key == "float8_e4m3fn"
            )
            unsafe_direct_swaps = not cfg.use_split_kv and not cfg.use_keeps_mma_ab
            unsafe_split_publisher = (
                cfg.use_split_kv and not cfg.use_separate_reduction_kernel
            )
            if q_token_kv_block_sparse_scattered_fp8 and (
                unsafe_direct_swaps or unsafe_split_publisher
            ):
                if not split_kv:
                    raise NotImplementedError(
                        "this FP8 encoded-subpage profile requires split-KV; "
                        "no qualified direct recipe is available with split_kv=False"
                    )
                # Direct, fused-GMEM, and cluster-SMEM publication are not
                # qualified with encoded subpage locators on the FP8 pipeline.
                safe_splits = int(cfg.splits_kv) if cfg.use_split_kv else 2
                safe_tile_size_q = max(int(cfg.tile_size_q), 16)
                cfg = make_config(
                    {
                        "groups_tokens_heads_q": cfg.groups_tokens_heads_q,
                        "tile_size_q": safe_tile_size_q,
                        "use_variable_seqlens_q": use_packed_q,
                        "use_pdl": use_pdl,
                    },
                    split_kv_mode="gmem_reduction_with_separate_kernel",
                    splits_kv=safe_splits,
                    max_splits_kv=safe_splits,
                )

    _validate_decode_policy_kv_tile_size(cfg)
    return _decode_launch_spec_from_config(
        cfg,
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len_q=seq_len_q,
        max_active_clusters=int(max_active_clusters),
    )


def _make_decode_compile_spec(
    launch_spec: _DecodeLaunchSpec,
    *,
    device_index: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    max_kv_len: int,
    seq_len_q: int,
    q_dtype_key: str,
    k_dtype_key: str,
    v_dtype_key: str,
    output_dtype_key: str,
    use_packed_q: bool,
    kv_prefix_mode: Literal["dynamic", "planned_full"],
    kv_lengths_mode: Literal["dynamic", "planned_uniform_max"],
    direct_q1_spec: Optional[_DirectQ1CompileSpec] = None,
) -> _DecodeCompileSpec:
    """Freeze the resolved topology while leaving batch in runtime tensors."""

    return _DecodeCompileSpec(
        device_index=device_index,
        config_items=launch_spec.config.compile_signature(),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        max_kv_len=max_kv_len,
        seq_len_q=seq_len_q,
        q_dtype_key=q_dtype_key,
        k_dtype_key=k_dtype_key,
        v_dtype_key=v_dtype_key,
        output_dtype_key=output_dtype_key,
        use_packed_q=use_packed_q,
        max_active_clusters=launch_spec.max_active_clusters,
        kv_prefix_mode=kv_prefix_mode,
        kv_lengths_mode=kv_lengths_mode,
        direct_q1_spec=direct_q1_spec,
    )


@functools.cache
def _get_compiled_decode(
    compile_spec: _DecodeCompileSpec,
):
    """Compile and cache one batch-dynamic TS decode topology."""

    device_index = compile_spec.device_index
    num_qo_heads = compile_spec.num_qo_heads
    num_kv_heads = compile_spec.num_kv_heads
    head_dim = compile_spec.head_dim
    max_kv_len = compile_spec.max_kv_len
    seq_len_q = compile_spec.seq_len_q
    q_dtype_key = compile_spec.q_dtype_key
    k_dtype_key = compile_spec.k_dtype_key
    v_dtype_key = compile_spec.v_dtype_key
    output_dtype_key = compile_spec.output_dtype_key
    use_packed_q = compile_spec.use_packed_q
    max_active_clusters = compile_spec.max_active_clusters
    kv_prefix_mode = compile_spec.kv_prefix_mode
    kv_lengths_mode = compile_spec.kv_lengths_mode

    if kv_prefix_mode not in ("dynamic", "planned_full"):
        raise ValueError(f"unsupported KV-prefix compile mode {kv_prefix_mode!r}")
    if kv_lengths_mode not in ("dynamic", "planned_uniform_max"):
        raise ValueError(f"unsupported KV-length compile mode {kv_lengths_mode!r}")
    static_full_split_prefix = kv_prefix_mode == "planned_full"
    static_native_uniform_kv = kv_lengths_mode == "planned_uniform_max"

    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda_drv

    from .kernels.fmha_decode.fmha_decode_config import FmhaDecodeConfig
    from .kernels.fmha_decode.fmha_decode_kernel import fmha_decode_launch
    from .kernels.fmha_decode.direct_sparse_metadata import DirectSparseMetadataView

    direct_q1_spec = compile_spec.direct_q1_spec

    q_dtype = _cutlass_dtype(q_dtype_key)
    k_dtype = _cutlass_dtype(k_dtype_key)
    v_dtype = _cutlass_dtype(v_dtype_key)
    output_dtype = _cutlass_dtype(output_dtype_key)
    cfg = FmhaDecodeConfig(**dict(compile_spec.config_items))
    k_storage_dtype = cutlass.Uint8 if k_dtype == cutlass.Float4E2M1FN else k_dtype
    v_storage_dtype = cutlass.Uint8 if v_dtype == cutlass.Float4E2M1FN else v_dtype
    storage_page_size = int(cfg.effective_storage_tokens_per_page)
    partial_dtype = output_dtype
    if cfg.use_separate_reduction_kernel and output_dtype in (
        cutlass.BFloat16,
        cutlass.Float8E4M3FN,
    ):
        partial_dtype = cutlass.BFloat16
    elif output_dtype == cutlass.Float8E4M3FN or not cfg.use_split_kv:
        partial_dtype = cutlass.Float16

    Int32 = cutlass.Int32
    Int64 = cutlass.Int64
    Float32 = cutlass.Float32

    @cute.jit
    def main_tensor_adapter(
        q: cute.Tensor,
        k_cache: cute.Tensor,
        v_cache: cute.Tensor,
        k_sf_cache: cute.Tensor,
        v_sf_cache: cute.Tensor,
        out: cute.Tensor,
        seq_lens: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        block_table: cute.Tensor,
        q_token_kv_block_sparse_page_memberships: cute.Tensor,
        partial_o: cute.Tensor,
        partial_stats: cute.Tensor,
        split_kv_counter: cute.Tensor,
        attention_sinks: cute.Tensor,
        num_physical_kv_pages: cutlass.Int64,
        k_page_stride: cutlass.Int64,
        k_head_stride: cutlass.Int64,
        k_token_stride: cutlass.Int64,
        v_page_stride: cutlass.Int64,
        v_head_stride: cutlass.Int64,
        v_token_stride: cutlass.Int64,
        bmm1_scale: cutlass.Float32,
        bmm2_scale: cutlass.Float32,
        direct_q1_inputs: tuple,
        stream: cuda_drv.CUstream,
        static_cfg: cutlass.Constexpr[FmhaDecodeConfig],
        static_seq_len_q: cutlass.Constexpr[int],
        static_num_qo_heads: cutlass.Constexpr[int],
        static_num_kv_heads: cutlass.Constexpr[int],
        static_head_dim: cutlass.Constexpr[int],
        static_max_kv_len: cutlass.Constexpr[int],
        static_max_active_clusters: cutlass.Constexpr[int],
        static_full_split_prefix: cutlass.Constexpr[bool],
        static_native_uniform_kv: cutlass.Constexpr[bool],
    ) -> None:
        """Adapt TVM-FFI tensors to the dense block-table pointer launcher."""

        pattern_heads = _num_sparse_pattern_heads(
            static_num_kv_heads, static_cfg.shares_sparse_pattern
        )
        batch_size = Int32(cute.size(seq_lens)) // Int32(pattern_heads)
        q_offsets_iter = cu_seqlens_q.iterator
        total_q_tokens = batch_size * Int32(static_seq_len_q)
        if cutlass.const_expr(not static_cfg.use_variable_seqlens_q):
            # Fixed-Q is a distinct specialization. Keep a uniform TVM-FFI
            # wrapper signature, but pass a real null pointer to the kernel so
            # fixed launches have no Q-offset metadata semantics.
            q_offsets_iter = cute.make_ptr(Int32, 0)
        else:
            total_q_tokens = Int32(q.shape[0])

        lengths_iter = seq_lens.iterator
        pages_iter = block_table.iterator
        if cutlass.const_expr(direct_q1_spec is not None):
            lengths_iter = DirectSparseMetadataView(
                direct_q1_inputs,
                cu_seqlens_q,
                packed=static_cfg.use_variable_seqlens_q,
                model_len=direct_q1_spec.max_seq_len_kv,
                page_size=storage_page_size,
                page_capacity=direct_q1_spec.page_capacity,
                sparse_block_size=direct_q1_spec.kv_block_size,
                fragment_size=static_cfg.num_tokens_per_page,
                lengths=True,
            )
            pages_iter = DirectSparseMetadataView(
                direct_q1_inputs,
                cu_seqlens_q,
                packed=static_cfg.use_variable_seqlens_q,
                model_len=direct_q1_spec.max_seq_len_kv,
                page_size=storage_page_size,
                page_capacity=direct_q1_spec.page_capacity,
                sparse_block_size=direct_q1_spec.kv_block_size,
                fragment_size=static_cfg.num_tokens_per_page,
                lengths=False,
            )

        fmha_decode_launch(
            (
                batch_size,
                Int32(static_num_qo_heads),
                Int32(static_num_kv_heads),
                Int32(static_max_kv_len),
                Int32(static_head_dim),
            ),
            q.iterator,
            k_cache.iterator,
            v_cache.iterator,
            k_sf_cache.iterator,
            v_sf_cache.iterator,
            out.iterator,
            lengths_iter,
            q_offsets_iter,
            total_q_tokens,
            pages_iter,
            q_token_kv_block_sparse_page_memberships.iterator,
            partial_o.iterator,
            partial_stats.iterator,
            split_kv_counter.iterator,
            attention_sinks.iterator,
            bmm1_scale,
            bmm2_scale,
            Int32(0),
            Int32(static_max_active_clusters),
            stream,
            static_cfg,
            static_max_kv_len,
            False,
            True,
            Int64(block_table.stride[0]),
            Int32(block_table.shape[1]),
            Int32(q_token_kv_block_sparse_page_memberships.stride[0]),
            num_physical_kv_pages,
            k_page_stride,
            k_head_stride,
            k_token_stride,
            v_page_stride,
            v_head_stride,
            v_token_stride,
            static_full_split_prefix,
            static_native_uniform_kv,
        )

    reduction_tensor_adapter = None
    if cfg.use_separate_reduction_kernel:
        from .kernels.fmha_decode.reduction import (
            fmha_decode_separate_reduction_launch,
        )

        @cute.jit
        def reduction_tensor_adapter(
            out: cute.Tensor,
            seq_lens: cute.Tensor,
            cu_seqlens_q: cute.Tensor,
            partial_o: cute.Tensor,
            partial_stats: cute.Tensor,
            attention_sinks: cute.Tensor,
            bmm1_scale: cutlass.Float32,
            bmm2_scale: cutlass.Float32,
            direct_q1_inputs: tuple,
            stream: cuda_drv.CUstream,
            static_cfg: cutlass.Constexpr[FmhaDecodeConfig],
            static_num_qo_heads: cutlass.Constexpr[int],
            static_num_kv_heads: cutlass.Constexpr[int],
            static_head_dim: cutlass.Constexpr[int],
            static_max_kv_len: cutlass.Constexpr[int],
            static_full_split_prefix: cutlass.Constexpr[bool],
        ) -> None:
            """Adapt TVM-FFI tensors to the raw standalone split reducer."""

            pattern_heads = _num_sparse_pattern_heads(
                static_num_kv_heads, static_cfg.shares_sparse_pattern
            )
            batch_size = Int32(cute.size(seq_lens)) // Int32(pattern_heads)
            q_offsets_iter = cu_seqlens_q.iterator
            if cutlass.const_expr(not static_cfg.use_variable_seqlens_q):
                q_offsets_iter = cute.make_ptr(Int32, 0)

            lengths_iter = seq_lens.iterator
            if cutlass.const_expr(direct_q1_spec is not None):
                lengths_iter = DirectSparseMetadataView(
                    direct_q1_inputs,
                    cu_seqlens_q,
                    packed=static_cfg.use_variable_seqlens_q,
                    model_len=direct_q1_spec.max_seq_len_kv,
                    page_size=storage_page_size,
                    page_capacity=direct_q1_spec.page_capacity,
                    sparse_block_size=direct_q1_spec.kv_block_size,
                    fragment_size=static_cfg.num_tokens_per_page,
                    lengths=True,
                )
            fmha_decode_separate_reduction_launch(
                (
                    batch_size,
                    Int32(static_num_qo_heads),
                    Int32(static_num_kv_heads),
                    Int32(static_max_kv_len),
                    Int32(static_head_dim),
                ),
                out.iterator,
                lengths_iter,
                q_offsets_iter,
                partial_o.iterator,
                partial_stats.iterator,
                attention_sinks.iterator,
                bmm1_scale,
                bmm2_scale,
                stream,
                static_cfg,
                static_full_split_prefix,
            )

    physical_pages = cute.sym_int()
    logical_pages = cute.sym_int()
    block_table_row_stride = cute.sym_int64(divisibility=1)
    membership_words = cute.sym_int()
    k_outer_stride = cute.sym_int64(divisibility=1)
    k_head_stride = cute.sym_int64(divisibility=1)
    k_token_stride = cute.sym_int64(divisibility=1)
    v_outer_stride = cute.sym_int64(divisibility=1)
    v_head_stride = cute.sym_int64(divisibility=1)
    v_token_stride = cute.sym_int64(divisibility=1)
    batch_size = cute.sym_int()
    total_q_tokens = cute.sym_int()
    runtime_num_q_offsets = cute.sym_int()
    q_shape = (
        (total_q_tokens, num_qo_heads, head_dim)
        if use_packed_q
        else (
            (batch_size, num_qo_heads, head_dim)
            if seq_len_q == 1
            else (batch_size, seq_len_q, num_qo_heads, head_dim)
        )
    )
    q_fake = cute.runtime.make_fake_compact_tensor(
        q_dtype,
        q_shape,
        stride_order=tuple(reversed(range(len(q_shape)))),
        assumed_align=16,
    )
    k_storage_head_dim = head_dim // 2 if k_dtype == cutlass.Float4E2M1FN else head_dim
    v_storage_head_dim = head_dim // 2 if v_dtype == cutlass.Float4E2M1FN else head_dim
    k_fake = cute.runtime.make_fake_tensor(
        k_storage_dtype,
        (physical_pages, num_kv_heads, storage_page_size, k_storage_head_dim),
        stride=(
            k_outer_stride,
            k_head_stride,
            k_token_stride,
            1,
        ),
        assumed_align=16,
    )
    v_fake = cute.runtime.make_fake_tensor(
        v_storage_dtype,
        (physical_pages, num_kv_heads, storage_page_size, v_storage_head_dim),
        stride=(
            v_outer_stride,
            v_head_stride,
            v_token_stride,
            1,
        ),
        assumed_align=16,
    )
    out_shape = _decode_output_shape(
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        seq_len_q=seq_len_q,
        head_dim=head_dim,
        total_q_tokens=total_q_tokens if use_packed_q else None,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        output_dtype,
        out_shape,
        stride_order=tuple(reversed(range(len(out_shape)))),
        assumed_align=16,
    )

    def fake_compact(dtype, shape, assumed_align):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=assumed_align,
        )

    pattern_heads = _num_sparse_pattern_heads(num_kv_heads, cfg.shares_sparse_pattern)
    metadata_rows = batch_size * pattern_heads
    seq_lens_fake = fake_compact(Int32, (metadata_rows,), 4)
    cu_seqlens_q_fake = fake_compact(
        Int32, (runtime_num_q_offsets,) if use_packed_q else (1,), 4
    )
    block_table_fake = cute.runtime.make_fake_tensor(
        Int32,
        (metadata_rows, logical_pages),
        stride=(block_table_row_stride, 1),
        assumed_align=4,
    )
    q_token_kv_block_sparse_page_memberships_fake = fake_compact(
        Int32, (metadata_rows, membership_words), 4
    )
    partial_o_shape, partial_stats_shape, counter_shape = _decode_scratch_shapes(
        cfg,
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len_q=seq_len_q,
    )
    partial_o_fake = fake_compact(partial_dtype, partial_o_shape, 16)
    partial_stats_fake = fake_compact(Float32, partial_stats_shape, 16)
    counter_fake = fake_compact(Int32, counter_shape, 4)
    attention_sinks_fake = fake_compact(Float32, (1,), 4)
    if cfg.use_nvfp4_kv:
        sf_shape = (physical_pages, num_kv_heads, storage_page_size, head_dim // 16)
        k_sf_fake = fake_compact(cutlass.Float8E4M3FN, sf_shape, 16)
        v_sf_fake = fake_compact(cutlass.Float8E4M3FN, sf_shape, 16)
    else:
        k_sf_fake = fake_compact(cutlass.Uint8, (1,), 1)
        v_sf_fake = fake_compact(cutlass.Uint8, (1,), 1)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    direct_q1_fake: tuple = ()
    if direct_q1_spec is not None:
        rows = cute.sym_int()
        direct_q1_fake = (
            cute.runtime.make_fake_tensor(
                Int32,
                (
                    (rows, direct_q1_spec.block_topk)
                    if cfg.shares_sparse_pattern
                    else (rows, num_kv_heads, direct_q1_spec.block_topk)
                ),
                stride=(
                    (cute.sym_int64(), 1)
                    if cfg.shares_sparse_pattern
                    else (cute.sym_int64(), cute.sym_int64(), 1)
                ),
                assumed_align=4,
            ),
            cute.runtime.make_fake_tensor(
                Int32,
                (cute.sym_int(), cute.sym_int()),
                stride=(cute.sym_int64(), 1),
                assumed_align=4,
            ),
            fake_compact(Int32, (rows,), 4),
            fake_compact(
                Int64 if direct_q1_spec.position_dtype_key == "int64" else Int32,
                (rows,),
                4,
            ),
        )

    with torch.cuda.device(device_index):
        compiled_main = cute.compile(
            main_tensor_adapter,
            q_fake,
            k_fake,
            v_fake,
            k_sf_fake,
            v_sf_fake,
            out_fake,
            seq_lens_fake,
            cu_seqlens_q_fake,
            block_table_fake,
            q_token_kv_block_sparse_page_memberships_fake,
            partial_o_fake,
            partial_stats_fake,
            counter_fake,
            attention_sinks_fake,
            Int64(1),
            Int64(1),
            Int64(1),
            Int64(1),
            Int64(1),
            Int64(1),
            Int64(1),
            Float32(1.0),
            Float32(1.0),
            direct_q1_fake,
            stream_fake,
            cfg,
            seq_len_q,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            max_kv_len,
            max_active_clusters,
            static_full_split_prefix,
            static_native_uniform_kv,
            options=_COMPILE_OPTIONS,
        )
        compiled_reducer = None
        if cfg.use_separate_reduction_kernel:
            assert reduction_tensor_adapter is not None
            compiled_reducer = cute.compile(
                reduction_tensor_adapter,
                out_fake,
                seq_lens_fake,
                cu_seqlens_q_fake,
                partial_o_fake,
                partial_stats_fake,
                attention_sinks_fake,
                Float32(1.0),
                Float32(1.0),
                direct_q1_fake,
                stream_fake,
                cfg,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                max_kv_len,
                static_full_split_prefix,
                options=_COMPILE_OPTIONS,
            )

    return compiled_main, compiled_reducer


def get_prims_ts_batch_decode_workspace_size(
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    max_seq_len: int,
    *,
    seq_len_q: int = 1,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
    q_dtype: torch.dtype = torch.float16,
    k_dtype: Optional[torch.dtype] = None,
    v_dtype: Optional[torch.dtype] = None,
    kv_dtype: Optional[torch.dtype] = None,
    out_dtype: Optional[torch.dtype] = None,
    mask_type: Literal["dense", "causal"] = "dense",
    window_left: int = -1,
    kv_layout: Literal["HND"] = "HND",
    storage_page_size: Optional[int] = None,
    device: Optional[Union[int, str, torch.device]] = None,
    split_kv: bool = True,
) -> int:
    """Return caller-workspace bytes for one automatic FMHA policy.

    The arguments resolve the same policy and scratch layout as
    :func:`batch_decode_with_paged_kv_cache`, without compiling a kernel.
    Allocate at least the returned number of bytes as a contiguous
    ``torch.int8`` or ``torch.uint8`` CUDA tensor and zero it before its first
    FMHA launch. Re-zero a reused buffer whenever any workspace-layout input,
    including ``batch_size``, changes because the internal section offsets can
    move even when the compiled callable is reused. Fixed-Q launches use
    ``seq_len_q``. Packed-Q launches provide ``qo_indptr`` and the explicit
    static ``max_seq_len_q`` bound used for workspace geometry and JIT policy.
    ``max_seq_len`` must be no larger than ``2,147,483,392`` so the padded
    256-token K/V tile endpoint remains representable as signed Int32.
    ``storage_page_size`` defaults to the semantic ``page_size``. A larger
    physical page is supported only for semantic page size four and activates
    encoded ``(physical page, subpage)`` locators.
    This sizing helper validates that every cumulative-offset delta is positive
    and no larger than the bound. If ``device`` is omitted, it is inferred from
    ``qo_indptr`` for a packed launch. ``k_dtype`` defaults to ``q_dtype`` and
    ``v_dtype`` to ``k_dtype``; pass ``v_dtype=torch.float8_e4m3fn`` with
    BF16 Q/K to size the QK-BF16/PV-FP8 launch. ``kv_dtype`` is a
    compatibility alias for both K and V and must agree with explicit dtypes.

    ``split_kv`` permits automatic split fanout when True (default), or forces
    nonsplit execution when False. It is independent of Q layout and must
    match the value supplied to attention planning/launch.
    """

    batch_size = _validate_positive_int(batch_size, "batch_size")
    use_packed_q, resolved_seq_len_q = _resolve_q_mode(
        seq_len_q=seq_len_q,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
        require_packed_max=True,
    )
    assert resolved_seq_len_q is not None
    seq_len_q = resolved_seq_len_q
    _validate_head_geometry(num_qo_heads, num_kv_heads)
    _validate_decode_query_head_extent(
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        max_seq_len_q=seq_len_q,
    )
    head_dim = _validate_head_dim(head_dim)
    page_size = _validate_page_size(page_size)
    storage_page_size = _validate_storage_page_size(
        page_size,
        page_size if storage_page_size is None else storage_page_size,
    )
    max_seq_len = _validate_max_kv_len(max_seq_len, "max_seq_len")
    _validate_layout(kv_layout)
    _validate_mask(mask_type)
    window_left = _validate_window_left(window_left, mask_type)
    k_dtype, v_dtype = _resolve_kv_dtypes(q_dtype, k_dtype, v_dtype, kv_dtype)
    if out_dtype is None:
        out_dtype = q_dtype
    _validate_dtype_pair(q_dtype, k_dtype, v_dtype, out_dtype)
    inferred_device = (
        qo_indptr.device
        if device is None and isinstance(qo_indptr, torch.Tensor)
        else device
    )
    resolved_device, _ = _resolve_cuda_device(inferred_device)
    if qo_indptr is not None:
        _validate_qo_indptr(
            qo_indptr,
            expected_device=resolved_device,
            batch_size=batch_size,
        )
        _validate_packed_q_plan_values(
            qo_indptr,
            max_seq_len_q=seq_len_q,
        )

    return _resolve_decode_workspace_layout(
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        max_seq_len,
        seq_len_q,
        q_dtype,
        k_dtype,
        v_dtype,
        out_dtype,
        kv_layout,
        mask_type,
        use_packed_q,
        window_left,
        storage_page_size,
        resolved_device,
        split_kv=split_kv,
    ).total_bytes


def _prepare_decode_runtime(
    q: torch.Tensor,
    paged_kv_cache: PagedKVCache,
    kv_scale_factors: Optional[PagedKVScaleFactors],
    *,
    device: torch.device,
    batch_size: int,
    seq_len_q: int,
    use_packed_q: bool,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    output_dtype: torch.dtype,
    bmm1_scale: Optional[float],
    bmm2_scale: float,
    out: Optional[torch.Tensor],
) -> _DecodeRuntime:
    """Validate runtime tensors and normalize zero-copy K/V views."""

    storage_page_size = page_size

    _validate_q(
        q,
        seq_len_q=seq_len_q,
        use_packed_q=use_packed_q,
        device=device,
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        head_dim=head_dim,
        q_dtype=q_dtype,
    )
    normalized_cache = _normalize_native_paged_kv_cache(
        paged_kv_cache,
        expected_device=device,
    )
    k_cache = normalized_cache.k_cache
    v_cache = normalized_cache.v_cache
    if (
        normalized_cache.num_kv_heads != num_kv_heads
        or normalized_cache.storage_page_size != storage_page_size
        or normalized_cache.head_dim != head_dim
    ):
        raise ValueError(
            "paged_kv_cache geometry does not match the launch: expected "
            f"Hkv/storage_page/D=({num_kv_heads}, {storage_page_size}, "
            f"{head_dim}), got "
            f"({normalized_cache.num_kv_heads}, "
            f"{normalized_cache.storage_page_size}, {normalized_cache.head_dim})"
        )
    if k_cache.dtype != k_dtype or v_cache.dtype != v_dtype:
        raise ValueError(
            f"K/V dtype must match the launch (K {k_dtype}, V {v_dtype}), got K "
            f"{k_cache.dtype} and V {v_cache.dtype}"
        )
    k_sf_cache, v_sf_cache = _normalize_paged_kv_scale_factors(
        kv_scale_factors,
        k_cache=k_cache,
        logical_head_dim=head_dim,
    )
    effective_bmm1_scale = _validate_scale(
        1.0 / math.sqrt(head_dim) if bmm1_scale is None else bmm1_scale,
        "bmm1_scale",
    )
    effective_bmm2_scale = _validate_scale(bmm2_scale, "bmm2_scale")
    output_shape = _decode_output_shape(
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        seq_len_q=seq_len_q,
        head_dim=head_dim,
        total_q_tokens=int(q.shape[0]) if use_packed_q else None,
    )
    if out is None:
        out = torch.empty(output_shape, device=device, dtype=output_dtype)
    else:
        _validate_out(
            out,
            q=q,
            expected_shape=output_shape,
            seq_len_q=seq_len_q,
            use_packed_q=use_packed_q,
            output_dtype=output_dtype,
        )
    return _DecodeRuntime(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k_sf_cache=k_sf_cache,
        v_sf_cache=v_sf_cache,
        out=out,
        num_physical_pages=normalized_cache.num_physical_pages,
        k_page_stride=normalized_cache.k_page_stride,
        k_head_stride=normalized_cache.k_head_stride,
        k_token_stride=normalized_cache.k_token_stride,
        v_page_stride=normalized_cache.v_page_stride,
        v_head_stride=normalized_cache.v_head_stride,
        v_token_stride=normalized_cache.v_token_stride,
        bmm1_scale=effective_bmm1_scale,
        bmm2_scale=effective_bmm2_scale,
    )


def _launch_decode(
    runtime: _DecodeRuntime,
    *,
    seq_lens: torch.Tensor,
    qo_indptr: Optional[torch.Tensor],
    block_tables: torch.Tensor,
    workspace: _DecodeWorkspaceViews,
    compiled_main: Callable[..., object],
    compiled_reducer: Optional[Callable[..., object]],
) -> torch.Tensor:
    """Launch the compiled main kernel and its optional standalone reducer."""

    q_offsets = workspace.cu_seqlens_q if qo_indptr is None else qo_indptr
    compiled_main(
        runtime.q,
        runtime.k_cache,
        runtime.v_cache,
        runtime.k_sf_cache,
        runtime.v_sf_cache,
        runtime.out,
        seq_lens,
        q_offsets,
        block_tables,
        block_tables,  # Dense decode constexpr-elides the membership table.
        workspace.partial_o,
        workspace.partial_stats,
        workspace.split_kv_counter,
        workspace.attention_sinks,
        runtime.num_physical_pages,
        runtime.k_page_stride,
        runtime.k_head_stride,
        runtime.k_token_stride,
        runtime.v_page_stride,
        runtime.v_head_stride,
        runtime.v_token_stride,
        runtime.bmm1_scale,
        runtime.bmm2_scale,
        (),
    )
    if compiled_reducer is not None:
        compiled_reducer(
            runtime.out,
            seq_lens,
            q_offsets,
            workspace.partial_o,
            workspace.partial_stats,
            workspace.attention_sinks,
            runtime.bmm1_scale,
            runtime.bmm2_scale,
            (),
        )
    return runtime.out


def _normalize_plan_seq_lens(
    seq_lens: Optional[Union[Sequence[int], torch.Tensor]],
    *,
    batch_size: int,
    max_kv_len: int,
) -> Optional[tuple[int, ...]]:
    """Validate optional host-only lengths retained by the plan."""

    if seq_lens is None:
        return None
    if isinstance(seq_lens, torch.Tensor):
        if seq_lens.device.type != "cpu":
            raise ValueError("plan seq_lens must be on CPU")
        if seq_lens.ndim != 1:
            raise ValueError("plan seq_lens must be 1D")
        if seq_lens.dtype not in (torch.int32, torch.int64):
            raise TypeError("plan seq_lens must have int32 or int64 dtype")
        raw_values: Sequence[object] = seq_lens.tolist()
    elif isinstance(seq_lens, Sequence) and not isinstance(
        seq_lens, (str, bytes, bytearray)
    ):
        raw_values = seq_lens
    else:
        raise TypeError(
            "plan seq_lens must be a host sequence of integers or a CPU tensor"
        )

    if len(raw_values) != batch_size:
        raise ValueError(
            "plan seq_lens must contain exactly "
            f"batch_size ({batch_size}) values, got {len(raw_values)}"
        )
    normalized = []
    for request_idx, value in enumerate(raw_values):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(
                "plan seq_lens must contain integers; "
                f"request {request_idx} has {value!r}"
            )
        seq_len = int(value)
        if seq_len <= 0 or seq_len > max_kv_len:
            raise ValueError(
                "plan seq_lens values must be within "
                f"[1, {max_kv_len}]; request {request_idx} has {seq_len}"
            )
        normalized.append(seq_len)
    return tuple(normalized)


def _prepare_decode_runtime_unchecked(
    q: torch.Tensor,
    paged_kv_cache: PagedKVCache,
    kv_scale_factors: Optional[PagedKVScaleFactors],
    *,
    output_dtype: torch.dtype,
    bmm1_scale: Optional[float],
    bmm2_scale: float,
    out: Optional[torch.Tensor],
) -> _DecodeRuntime:
    """Canonicalize one trusted run without invoking explicit validators."""

    if isinstance(paged_kv_cache, torch.Tensor):
        k_cache = paged_kv_cache[:, 0]
        v_cache = paged_kv_cache[:, 1]
    else:
        k_cache, v_cache = paged_kv_cache
    if k_cache.dtype == torch.uint8:
        k_sf_cache, v_sf_cache = cast(PagedKVScaleFactors, kv_scale_factors)
    else:
        placeholder = k_cache[0, 0, 0, :1].view(torch.uint8)[:1]
        k_sf_cache, v_sf_cache = placeholder, placeholder
    if out is None:
        out = torch.empty(
            q.shape,
            device=q.device,
            dtype=output_dtype,
        )
    return _DecodeRuntime(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k_sf_cache=k_sf_cache,
        v_sf_cache=v_sf_cache,
        out=out,
        num_physical_pages=int(k_cache.shape[0]),
        k_page_stride=int(k_cache.stride(0)),
        k_head_stride=int(k_cache.stride(1)),
        k_token_stride=int(k_cache.stride(2)),
        v_page_stride=int(v_cache.stride(0)),
        v_head_stride=int(v_cache.stride(1)),
        v_token_stride=int(v_cache.stride(2)),
        bmm1_scale=(
            1.0 / math.sqrt(int(q.shape[-1]))
            if bmm1_scale is None
            else float(bmm1_scale)
        ),
        bmm2_scale=float(bmm2_scale),
    )


def _validate_decode_run_metadata_values(
    runtime: _DecodeRuntime,
    *,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    qo_indptr: Optional[torch.Tensor],
    planned_seq_lens_host: Optional[tuple[int, ...]],
    max_kv_len: int,
    page_size: int,
    use_packed_q: bool,
    seq_len_q: int,
    batch_size: int,
    mask_type: str,
) -> None:
    """Synchronously validate per-run metadata values."""

    runtime_seq_lens = planned_seq_lens_host
    if runtime_seq_lens is None:
        runtime_seq_lens = tuple(int(value) for value in seq_lens.tolist())
    table_capacity = int(block_tables.shape[1])
    for request_idx, seq_len in enumerate(runtime_seq_lens):
        if seq_len <= 0 or seq_len > max_kv_len:
            raise ValueError(
                f"seq_lens values must be within [1, {max_kv_len}]; "
                f"request {request_idx} has {seq_len}"
            )
        required_pages = (seq_len + page_size - 1) // page_size
        if table_capacity < required_pages:
            raise ValueError(
                "block_tables does not have enough columns for "
                f"seq_lens[{request_idx}]={seq_len}: requires {required_pages}, "
                f"got {table_capacity}"
            )

    block_table_rows = block_tables.tolist()
    locator_capacity = runtime.num_physical_pages * (
        int(runtime.k_cache.shape[2]) // page_size
    )
    for request_idx, (row, seq_len) in enumerate(
        zip(block_table_rows, runtime_seq_lens, strict=True)
    ):
        required_pages = (seq_len + page_size - 1) // page_size
        if any(
            int(page_id) < 0 or int(page_id) >= locator_capacity
            for page_id in row[:required_pages]
        ):
            raise ValueError(
                "block_tables values for active pages must index the physical "
                f"K/V cache through locators in [0, {locator_capacity}); request "
                f"{request_idx} contains an invalid page ID"
            )

    if use_packed_q:
        assert qo_indptr is not None
        _, _, q_lengths = _read_packed_q_plan_metadata(qo_indptr)
        if max(q_lengths) > seq_len_q:
            raise ValueError(
                "qo_indptr contains a per-request Q length larger than "
                f"max_seq_len_q ({seq_len_q}): got {max(q_lengths)}"
            )
        total_q = sum(q_lengths)
        if total_q != int(runtime.q.shape[0]):
            raise ValueError(
                "the final qo_indptr offset must equal the packed q token count: "
                f"expected {runtime.q.shape[0]}, got {total_q}"
            )
    else:
        q_lengths = (seq_len_q,) * batch_size

    if mask_type == "causal":
        for request_idx, (q_len, kv_len) in enumerate(
            zip(q_lengths, runtime_seq_lens, strict=True)
        ):
            if q_len > kv_len:
                raise ValueError(
                    "causal decode requires every per-request Q length to be "
                    "no greater than its K/V length; request "
                    f"{request_idx} has Q={q_len} and K/V={kv_len}"
                )


@dataclass(frozen=True)
class _NativePagedKVCache:
    """Zero-copy logical HND views and TensorMap strides for native decode."""

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    num_physical_pages: int
    num_kv_heads: int
    storage_page_size: int
    head_dim: int
    k_page_stride: int
    k_head_stride: int
    k_token_stride: int
    v_page_stride: int
    v_head_stride: int
    v_token_stride: int


@dataclass(frozen=True)
class PrimsTSBatchDecodePlan:
    """Validated dense-page-table PrimTS state for framework hot paths.

    The plan retains the K/V cache and scale tensors, dense block-table storage, compiled
    callables, and typed workspace views validated at construction. Block-table
    and sequence-length values may change between completed launches, which is
    the contract needed by QToken-KvBlock-Sparse-Attention metadata builders. Query and output storage may
    also change, but must preserve the exact shape, dtype, device, and strides
    proven by the representative tensors passed to
    :func:`prepare_prims_ts_batch_decode_with_kv_cache`.

    ``run`` deliberately omits repeated cache-stride validation, workspace
    rebinding, and semantic-policy resolution. Storage overlap is not checked
    during preparation or execution. The framework must keep retained tensors
    alive, keep output/workspace disjoint from the inputs and each other, and
    not mutate metadata concurrently with a launch or replay that reads it.
    """

    _query_shape: tuple[int, ...]
    _query_stride: tuple[int, ...]
    _output_shape: tuple[int, ...]
    _output_stride: tuple[int, ...]
    _device: torch.device
    _q_dtype: torch.dtype
    _output_dtype: torch.dtype
    _head_dim: int
    _cache: _NativePagedKVCache
    _k_sf_cache: torch.Tensor
    _v_sf_cache: torch.Tensor
    _seq_lens: torch.Tensor
    _qo_indptr: Optional[torch.Tensor]
    _block_table: torch.Tensor
    _q_token_kv_block_sparse_page_memberships: torch.Tensor
    _workspace: _DecodeWorkspaceViews
    _compiled_main: Callable[..., object]
    _compiled_reducer: Optional[Callable[..., object]]
    _direct_q1_inputs: tuple[torch.Tensor, ...] = ()

    def run(
        self,
        query: torch.Tensor,
        *,
        out: torch.Tensor,
        bmm1_scale: Optional[float] = None,
        bmm2_scale: float = 1.0,
    ) -> torch.Tensor:
        """Launch with lightweight checks against the validated plan."""

        if (
            query.shape != self._query_shape
            or query.stride() != self._query_stride
            or query.device != self._device
            or query.dtype != self._q_dtype
        ):
            raise ValueError(
                "query must preserve the shape, strides, device, and dtype "
                "validated by the PrimTS plan"
            )
        if (
            out.shape != self._output_shape
            or out.stride() != self._output_stride
            or out.device != self._device
            or out.dtype != self._output_dtype
        ):
            raise ValueError(
                "out must preserve the shape, strides, device, and dtype "
                "validated by the PrimTS plan"
            )

        scale_qk = _validate_scale(
            1.0 / math.sqrt(self._head_dim) if bmm1_scale is None else bmm1_scale,
            "bmm1_scale",
        )
        scale_v = _validate_scale(bmm2_scale, "bmm2_scale")
        return self._run_unchecked(query, out, scale_qk, scale_v)

    def _run_unchecked(
        self,
        query: torch.Tensor,
        out: torch.Tensor,
        scale_qk: float,
        scale_v: float,
        direct_q1_inputs: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        """Launch state already proven by a framework-owned outer plan."""

        q_offsets = (
            self._workspace.cu_seqlens_q if self._qo_indptr is None else self._qo_indptr
        )
        direct_inputs = (
            self._direct_q1_inputs if direct_q1_inputs is None else direct_q1_inputs
        )
        self._compiled_main(
            query,
            self._cache.k_cache,
            self._cache.v_cache,
            self._k_sf_cache,
            self._v_sf_cache,
            out,
            self._seq_lens,
            q_offsets,
            self._block_table,
            self._q_token_kv_block_sparse_page_memberships,
            self._workspace.partial_o,
            self._workspace.partial_stats,
            self._workspace.split_kv_counter,
            self._workspace.attention_sinks,
            self._cache.num_physical_pages,
            self._cache.k_page_stride,
            self._cache.k_head_stride,
            self._cache.k_token_stride,
            self._cache.v_page_stride,
            self._cache.v_head_stride,
            self._cache.v_token_stride,
            scale_qk,
            scale_v,
            direct_inputs,
        )
        if self._compiled_reducer is not None:
            self._compiled_reducer(
                out,
                self._seq_lens,
                q_offsets,
                self._workspace.partial_o,
                self._workspace.partial_stats,
                self._workspace.attention_sinks,
                scale_qk,
                scale_v,
                direct_inputs,
            )
        return out


def _resolve_decode_workspace_layout(
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    max_seq_len: int,
    seq_len_q: int,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    out_dtype: torch.dtype,
    kv_layout: str,
    mask_type: str,
    use_packed_q: bool,
    window_left: int,
    storage_page_size: int,
    device: Optional[Union[int, str, torch.device]],
    use_q_token_kv_block_sparse_route: bool = False,
    use_pdl: bool = False,
    split_kv: bool = True,
    share_pattern_across_kv_heads: bool = True,
) -> _DecodeWorkspaceLayout:
    """Resolve the byte layout for one already-validated semantic key."""

    resolved_device, device_index = _resolve_cuda_device(device)
    _validate_runtime_device(resolved_device)
    spec = _resolve_decode_launch_spec(
        device_index,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        max_seq_len,
        seq_len_q,
        _dtype_key(q_dtype),
        _dtype_key(k_dtype),
        _dtype_key(v_dtype),
        _dtype_key(out_dtype),
        kv_layout,
        mask_type,
        use_packed_q,
        window_left,
        storage_page_size,
        use_q_token_kv_block_sparse_route,
        use_pdl,
        split_kv,
        share_pattern_across_kv_heads,
    )
    return _make_decode_workspace_layout(
        spec.scratch_shapes,
        out_dtype,
        use_separate_reduction_kernel=spec.config.use_separate_reduction_kernel,
        use_split_kv=spec.config.use_split_kv,
    )


def _validate_storage_page_size(page_size: int, storage_page_size: int) -> int:
    """Require physical cache pages to contain whole logical fragments."""
    storage_page_size = _validate_positive_int(storage_page_size, "storage_page_size")
    if storage_page_size % page_size != 0:
        raise ValueError("storage_page_size must be divisible by page_size")
    return storage_page_size


def _normalize_paged_kv_cache_views(
    paged_kv_cache: PagedKVCache,
    *,
    expected_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
    """Return validated zero-copy HND views and the logical head dimension."""

    if isinstance(paged_kv_cache, torch.Tensor):
        if paged_kv_cache.ndim != 5 or paged_kv_cache.shape[1] != 2:
            raise ValueError(
                "combined paged_kv_cache must have shape "
                "[num_pages, 2, Hkv, page_size, head_dim]"
            )
        if paged_kv_cache.device != expected_device:
            raise ValueError(
                f"paged_kv_cache must be on {expected_device}, "
                f"got {paged_kv_cache.device}"
            )
        k_cache = paged_kv_cache[:, 0]
        v_cache = paged_kv_cache[:, 1]
    elif isinstance(paged_kv_cache, tuple):
        if len(paged_kv_cache) != 2:
            raise ValueError("paged_kv_cache tuple must contain exactly (K, V)")
        k_cache, v_cache = paged_kv_cache
        if not isinstance(k_cache, torch.Tensor) or not isinstance(
            v_cache, torch.Tensor
        ):
            raise TypeError("paged_kv_cache tuple members must be torch.Tensor")
        if k_cache.ndim != 4 or v_cache.ndim != 4:
            raise ValueError(
                "tuple K/V caches must each have shape "
                "[num_pages, Hkv, page_size, head_dim]"
            )
        if k_cache.device != expected_device or v_cache.device != expected_device:
            raise ValueError(f"tuple K/V caches must be on {expected_device}")
    else:
        raise TypeError(
            "paged_kv_cache must be a combined torch.Tensor or a (K, V) tuple"
        )

    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("K/V cache views must be rank-4 HND tensors")
    if k_cache.shape != v_cache.shape:
        raise ValueError("K and V cache views must have identical logical shapes")
    if k_cache.device != v_cache.device:
        raise ValueError("K and V cache views must be on the same device")

    num_pages, num_kv_heads, page_size, storage_head_dim = map(int, k_cache.shape)
    if min(num_pages, num_kv_heads, page_size, storage_head_dim) <= 0:
        raise ValueError("paged_kv_cache dimensions must be positive")
    # This API reserves uint8 K/V storage for two packed NVFP4 values per byte.
    head_dim = (
        storage_head_dim * 2 if k_cache.dtype == torch.uint8 else storage_head_dim
    )
    return (
        k_cache,
        v_cache,
        num_pages,
        num_kv_heads,
        page_size,
        head_dim,
    )


def _validate_native_hnd_tma_strides(
    tensor: torch.Tensor,
    name: str,
) -> tuple[int, int, int]:
    """Validate a non-overlapping logical HND view for TensorMap addressing."""

    num_pages, num_kv_heads, storage_page_size, head_dim = map(int, tensor.shape)
    page_stride, head_stride, token_stride, dim_stride = map(int, tensor.stride())
    if dim_stride != 1:
        raise ValueError(f"{name} head-dimension stride must be one")
    named_strides = (
        ("page", page_stride),
        ("head", head_stride),
        ("token", token_stride),
    )
    if any(stride <= 0 for _, stride in named_strides):
        raise ValueError(f"{name} TensorMap strides must be positive")
    max_int64 = 2**63 - 1
    if any(stride > max_int64 for _, stride in named_strides):
        raise ValueError(f"{name} TensorMap stride exceeds signed int64")

    element_size = tensor.element_size()
    for stride_name, stride in named_strides:
        if stride * element_size % 16 != 0:
            raise ValueError(f"{name} {stride_name} stride must be 16-byte aligned")
    _validate_16byte_alignment(tensor, name)

    # A gapped K or V slice of a packed 2D cache is not dense, so PyTorch's
    # dense-layout predicates cannot establish safety. Prove non-overlap from
    # the actual logical dimensions and strides instead.
    span = 1
    dimensions = sorted(
        (
            (1, head_dim, "head dimension"),
            (token_stride, storage_page_size, "token"),
            (head_stride, num_kv_heads, "head"),
            (page_stride, num_pages, "page"),
        ),
        key=lambda item: item[0],
    )
    for stride, extent, dimension_name in dimensions:
        if extent <= 1:
            continue
        if stride < span:
            raise ValueError(
                f"{name} {dimension_name} dimension overlaps another dimension"
            )
        span += (extent - 1) * stride
        if span - 1 > max_int64:
            raise ValueError(f"{name} address span exceeds signed int64")
    return page_stride, head_stride, token_stride


def _normalize_native_paged_kv_cache(
    paged_kv_cache: PagedKVCache,
    *,
    expected_device: torch.device,
) -> _NativePagedKVCache:
    """Return native decode views supporting HND- or NHD-physical storage."""

    (
        k_cache,
        v_cache,
        num_pages,
        num_kv_heads,
        storage_page_size,
        head_dim,
    ) = _normalize_paged_kv_cache_views(
        paged_kv_cache,
        expected_device=expected_device,
    )
    k_page_stride, k_head_stride, k_token_stride = _validate_native_hnd_tma_strides(
        k_cache, "K cache"
    )
    v_page_stride, v_head_stride, v_token_stride = _validate_native_hnd_tma_strides(
        v_cache, "V cache"
    )
    return _NativePagedKVCache(
        k_cache=k_cache,
        v_cache=v_cache,
        num_physical_pages=num_pages,
        num_kv_heads=num_kv_heads,
        storage_page_size=storage_page_size,
        head_dim=head_dim,
        k_page_stride=k_page_stride,
        k_head_stride=k_head_stride,
        k_token_stride=k_token_stride,
        v_page_stride=v_page_stride,
        v_head_stride=v_head_stride,
        v_token_stride=v_token_stride,
    )


def _validate_q_token_kv_block_sparse_page_memberships(
    q_token_kv_block_sparse_page_memberships: torch.Tensor,
    *,
    expected_device: torch.device,
    batch_size: int,
    max_num_pages: int,
) -> None:
    """Validate the private grouped-QToken-KvBlock-Sparse-Attention packed-membership table."""

    if not isinstance(q_token_kv_block_sparse_page_memberships, torch.Tensor):
        raise TypeError(
            "q_token_kv_block_sparse_page_memberships must be a torch.Tensor"
        )
    if q_token_kv_block_sparse_page_memberships.ndim != 2:
        raise ValueError(
            f"q_token_kv_block_sparse_page_memberships must be rank 2, got rank {q_token_kv_block_sparse_page_memberships.ndim}"
        )
    if q_token_kv_block_sparse_page_memberships.dtype != torch.int32:
        raise TypeError(
            "q_token_kv_block_sparse_page_memberships must have dtype torch.int32"
        )
    if q_token_kv_block_sparse_page_memberships.device != expected_device:
        raise ValueError(
            f"q_token_kv_block_sparse_page_memberships must be on {expected_device}, got "
            f"{q_token_kv_block_sparse_page_memberships.device}"
        )
    if not q_token_kv_block_sparse_page_memberships.is_contiguous():
        raise ValueError("q_token_kv_block_sparse_page_memberships must be contiguous")
    expected_shape = (batch_size, (max_num_pages + 3) // 4)
    if tuple(q_token_kv_block_sparse_page_memberships.shape) != expected_shape:
        raise ValueError(
            "q_token_kv_block_sparse_page_memberships must have shape "
            f"{expected_shape}, got {tuple(q_token_kv_block_sparse_page_memberships.shape)}"
        )
    if q_token_kv_block_sparse_page_memberships.data_ptr() % 4 != 0:
        raise ValueError(
            "q_token_kv_block_sparse_page_memberships data pointer must be 4-byte aligned"
        )


def _resolve_q_token_kv_block_sparse_decode_config(
    make_config: Callable[..., "FmhaDecodeConfig"],
    *,
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    max_kv_len: int,
    seq_len_q: int,
    q_dtype_key: str,
    k_dtype_key: str,
    v_dtype_key: str,
    output_dtype_key: str,
    mask_type: str,
    use_packed_q: bool,
    window_left: int,
    max_active_clusters: int,
    use_pdl: bool,
    split_kv: bool = True,
    share_pattern_across_kv_heads: bool = True,
) -> "FmhaDecodeConfig":
    """Resolve the private encoded-page QToken-KvBlock-Sparse-Attention profile without compiling it."""

    from .kernels.fmha_decode.fmha_decode_config import (
        MIN_LOOP_ITERS_PER_SPLIT,
    )

    heads_q_per_kv = num_qo_heads // num_kv_heads
    _validate_prims_ts_q_token_kv_block_sparse_group_capacity(
        seq_len_q,
        num_qo_heads,
        num_kv_heads,
    )
    # This route has no QK-BF16/PV-FP8 kernel, so Q, K, and V must match.
    q_token_kv_block_sparse_dtype_supported = (
        q_dtype_key == k_dtype_key == v_dtype_key
        and (
            q_dtype_key == output_dtype_key
            and q_dtype_key in ("float16", "bfloat16")
            or (
                q_dtype_key == "float8_e4m3fn"
                and output_dtype_key in ("float16", "bfloat16")
            )
        )
    )
    if not (
        page_size in _SUPPORTED_PAGE_SIZES
        and head_dim in _SUPPORTED_HEAD_DIMS
        and q_token_kv_block_sparse_dtype_supported
        and mask_type == "causal"
        and window_left < 0
    ):
        raise ValueError(
            "PrimTS QToken-KvBlock-Sparse-Attention requires a supported sparse-fragment size, head_dim in {64,128,256}, "
            "FP16/BF16 Q/K/V with matching output or FP8 Q/K/V with FP16/BF16 output, "
            "and a causal non-windowed mask"
        )

    launch = _sparse_policy.select_sparse_launch(
        group_size=seq_len_q,
        heads_q_per_kv=heads_q_per_kv,
        head_dim=head_dim,
        q_dtype_key=q_dtype_key,
        num_routes=batch_size,
        num_kv_heads=num_kv_heads,
        route_kv_tokens=max_kv_len,
        multi_processor_count=max_active_clusters,
        split_kv=split_kv,
    )

    q_token_kv_block_sparse_profile = {
        "use_variable_seqlens_q": use_packed_q,
        "use_q_token_kv_block_sparse_route": True,
        "share_pattern_across_kv_heads": share_pattern_across_kv_heads,
        "use_pdl": use_pdl,
        "use_keeps_mma_ab": launch.use_keeps_mma_ab,
        "groups_tokens_heads_q": True,
        "tile_size_q": launch.tile_size_q,
        "tile_size_kv": _sparse_policy.TILE_SIZE_KV,
        "head_dim_per_stage_kv": launch.head_dim_per_stage_kv,
        "num_insts_kv": launch.num_insts_kv,
        "use_persistent_scheduler": launch.use_persistent_scheduler,
        "correction_num_warps": 4,
        "mma_warp_idx": 12,
        # CLC uses the control slot beside MMA; page producers occupy the
        # remaining slots. TMA issuers keep their own complete warp groups.
        "scheduler_warp_idx": 13,
        "clc_load_warp_idx": 16,
        "page_offsets_warp_idx": 14,
        "page_offsets_num_warps": 2 if seq_len_q == 1 else 1,
        "load_warp_idx": 16,
        # FP8's predicated TMA helper constructs coordinates in every load
        # lane, so Q1 needs only one producer warpgroup. BF16 distributes page
        # fragments across two producer warpgroups. Grouped routes keep
        # their qualified two-warpgroup schedule for both dtypes.
        "load_num_warps": (
            4 if seq_len_q == 1 and q_dtype_key == "float8_e4m3fn" else 8
        ),
    }
    if launch.use_keeps_mma_ab:
        q_token_kv_block_sparse_profile["o_stages"] = launch.o_stages
    return make_config(
        q_token_kv_block_sparse_profile,
        split_kv_mode=(
            "gmem_reduction_with_separate_kernel"
            if launch.splits_kv > 1
            else "disabled"
        ),
        splits_kv=launch.splits_kv,
        max_splits_kv=launch.splits_kv,
        min_loop_iters_per_split=(1 if seq_len_q == 1 else MIN_LOOP_ITERS_PER_SPLIT),
    )


def _validate_prims_ts_q_token_kv_block_sparse_group_value(group_size: int) -> int:
    """Validate one caller-selected QToken-KvBlock-Sparse-Attention grouping value."""

    group_size = _validate_positive_int(group_size, "group_size")
    if group_size not in _sparse_policy.SUPPORTED_GROUP_SIZES:
        raise ValueError(
            "QToken-KvBlock-Sparse-Attention group_size must be one of "
            f"{_sparse_policy.SUPPORTED_GROUP_SIZES}, got {group_size}"
        )
    return group_size


def _validate_prims_ts_q_token_kv_block_sparse_group_capacity(
    group_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
) -> int:
    """Validate that one caller-selected sparse Q group fits TileQ128 and a membership byte."""

    from .kernels.fmha_decode.fmha_decode_constants import (
        Q_TOKEN_KV_BLOCK_SPARSE_PAGE_MEMBERSHIP_BITS,
    )

    group_size = _validate_prims_ts_q_token_kv_block_sparse_group_value(group_size)
    _validate_head_geometry(num_qo_heads, num_kv_heads)
    heads_q_per_kv = num_qo_heads // num_kv_heads
    max_group_size = min(
        Q_TOKEN_KV_BLOCK_SPARSE_PAGE_MEMBERSHIP_BITS,
        _sparse_policy.MAX_TILE_SIZE_Q // heads_q_per_kv,
    )
    if group_size > max_group_size:
        raise ValueError(
            f"QToken-KvBlock-Sparse-Attention group_size={group_size} exceeds the TileQ128/head capacity "
            f"of {max_group_size}"
        )

    return group_size


@flashinfer_experimental_api
def suggest_q_token_kv_block_sparse_group_size(
    batch_size: int,
    seq_len_q: int,
    selected_seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    multi_processor_count: int,
    *,
    head_dim: int = 256,
    q_data_type: torch.dtype = torch.bfloat16,
    split_kv: bool = True,
    kv_block_size: int = 4,
) -> int:
    """Suggest a Q1--Q8 QToken-KvBlock-Sparse-Attention group.

    This pure host-side policy uses the caller-provided SM count; it never
    queries device properties or reads tensors. It starts with the largest
    legal group and the smallest fitting dense Q-tile recipe, then adds
    useful one-wave split-KV parallelism. It considers the largest group for
    each dense Q tile and further ungrouping at the smallest tile. If the
    largest group cannot fill a wave, prefer better active SM coverage after
    instruction-aligned split rounding. Ties prefer less Q padding, then the
    larger group. Explicit G passed to plan/run is never overridden.

    Query lengths need not be divisible by the result. Packed prefill may end
    each request with a shorter route. Fixed-shape decode may pad its final
    ``[B, num_query_groups, G, Hq, D]`` route with consecutive semantic dummy
    rows and discard their outputs.

    ``selected_seq_len_kv`` is the maximum candidate K/V-token count for one
    query, including its causal tail. Configured split fanout uses union
    capacity to accommodate non-shared KV. The recommendation scores useful
    active splits using a fully-shared per-query proxy, rather than counting
    every capacity-based split as active. Actual union-overlap and scattered-load
    costs remain a TODO.

    Parameters
    ----------
    batch_size : int
        Number of requests in the launch.
    seq_len_q : int
        Maximum number of live query tokens per request.
    selected_seq_len_kv : int
        Maximum selected candidate K/V tokens per query, including the causal
        tail. This is not the original context length or global cache capacity.
    num_qo_heads : int
        Number of query/output heads.
    num_kv_heads : int
        Number of key/value heads.
    multi_processor_count : int
        Number of SMs available to the launch. Frameworks should query it once
        and pass the cached value; this function performs no device query.
    head_dim : int
        Common Q/K/V head dimension: 64, 128, or 256.
    q_data_type : torch.dtype
        Q/K/V dtype used to select the matching dense MMA recipe.
    split_kv : bool
        Allow automatic split-KV fanout when True (default); False forces
        one KV partition. Pass False for prefill and True for decode,
        independently of packed or fixed Q layout. Match this value in
        workspace sizing and attention planning. Ragged workloads use maximum
        Q length as a capacity estimate; no GPU sequence lengths are inspected.

    kv_block_size : int
        Semantic sparse block size: 4/8/16/32/64/128, default four. Used to
        bound grouped union work; storage-page size does not affect this bound.

    Returns
    -------
    int
        One of Q1--Q8, subject to the TileQ128 head-capacity bound.
    """

    from .q_token_kv_block_sparse_metadata import _validate_sparse_block_size

    kv_block_size = _validate_sparse_block_size(kv_block_size)
    batch_size = _validate_positive_int(batch_size, "batch_size")
    seq_len_q = _validate_positive_int(seq_len_q, "seq_len_q")
    selected_seq_len_kv = _validate_positive_int(
        selected_seq_len_kv, "selected_seq_len_kv"
    )
    multi_processor_count = _validate_positive_int(
        multi_processor_count, "multi_processor_count"
    )
    _validate_head_geometry(num_qo_heads, num_kv_heads)
    heads_q_per_kv = num_qo_heads // num_kv_heads

    head_dim = _validate_head_dim(head_dim)
    if q_data_type not in _SUPPORTED_INPUT_DTYPES:
        raise ValueError("q_data_type must be float16, bfloat16, or float8_e4m3fn")
    if not isinstance(split_kv, bool):
        raise TypeError("split_kv must be a bool")
    return _sparse_policy.suggest_sparse_group(
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        selected_kv_tokens=selected_seq_len_kv,
        heads_q_per_kv=heads_q_per_kv,
        num_kv_heads=num_kv_heads,
        multi_processor_count=multi_processor_count,
        head_dim=head_dim,
        q_dtype_key=_dtype_key(q_data_type),
        split_kv=split_kv,
        block_size=kv_block_size,
    )


def _validate_prims_ts_q_token_kv_block_sparse_group_layout(
    group_size: int,
    query_start_loc_cpu: Optional[torch.Tensor],
    num_query_tokens: int,
    num_qo_heads: int,
    num_kv_heads: int,
) -> int:
    """Validate one caller-selected QToken-KvBlock-Sparse-Attention group against the live query rows."""

    group_size = _validate_prims_ts_q_token_kv_block_sparse_group_capacity(
        group_size,
        num_qo_heads,
        num_kv_heads,
    )
    num_query_tokens = _validate_positive_int(num_query_tokens, "num_query_tokens")
    if group_size == 1:
        return 1
    if (
        query_start_loc_cpu is None
        or not isinstance(query_start_loc_cpu, torch.Tensor)
        or query_start_loc_cpu.device.type != "cpu"
        or query_start_loc_cpu.ndim != 1
        or query_start_loc_cpu.numel() < 2
        or query_start_loc_cpu.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "grouped QToken-KvBlock-Sparse-Attention requires CPU int32/int64 query_start_loc with at "
            "least two entries"
        )

    query_starts = [int(value) for value in query_start_loc_cpu.tolist()]
    num_mapped_tokens = query_starts[-1]
    if (
        query_starts[0] != 0
        or num_mapped_tokens < 0
        or num_mapped_tokens > num_query_tokens
    ):
        raise ValueError(
            "grouped QToken-KvBlock-Sparse-Attention query boundaries do not cover valid rows"
        )
    query_lengths = [
        end - begin for begin, end in zip(query_starts, query_starts[1:], strict=False)
    ]
    if any(length < 0 for length in query_lengths):
        raise ValueError(
            "grouped QToken-KvBlock-Sparse-Attention query boundaries must be nondecreasing"
        )
    query_lengths = [length for length in query_lengths if length > 0]
    if not query_lengths:
        raise ValueError(
            "grouped QToken-KvBlock-Sparse-Attention requires at least one nonempty request"
        )

    return group_size


@flashinfer_experimental_api
def validate_q_token_kv_block_sparse_group_size(
    query_start_loc_cpu: Optional[torch.Tensor],
    num_query_tokens: int,
    num_qo_heads: int,
    num_kv_heads: int,
    *,
    group_size: int,
) -> int:
    """Validate a QToken-KvBlock-Sparse-Attention Q1--Q8 group.

    ``query_start_loc_cpu`` contains cumulative flattened-query offsets for
    real requests. ``num_query_tokens`` includes any inert CUDA-graph padding
    rows. Validation caps the maximum grouped query by both the kernel
    membership representation and TileQ128 capacity. Request lengths do not
    have to be divisible by ``group_size``: frameworks can partition each
    request into packed routes of at most that size without crossing a request
    or real/padding boundary.

    This function never chooses a group from workload size or occupancy. The
    framework owns ``group_size``; attention split-KV fanout is selected
    separately after the grouped launch grid is known.

    Parameters
    ----------
    query_start_loc_cpu : torch.Tensor or None
        CPU Int32 or Int64 cumulative offsets for the real requests. It is
        required when ``group_size`` is greater than one; ``None`` is accepted
        for Q1.
    num_query_tokens : int
        Total flattened query-row count, including any inert CUDA-graph
        padding rows.
    num_qo_heads : int
        Number of query/output heads.
    num_kv_heads : int
        Number of key/value heads.
    group_size : int
        Caller-selected number of query rows per QToken-KvBlock-Sparse-Attention route. Supported values
        are Q1--Q8, subject to the TileQ128 head-capacity bound.

    Returns
    -------
    int
        The validated ``group_size``.
    """

    return _validate_prims_ts_q_token_kv_block_sparse_group_layout(
        group_size,
        query_start_loc_cpu,
        num_query_tokens,
        num_qo_heads,
        num_kv_heads,
    )


@flashinfer_experimental_api
def make_q_token_kv_block_sparse_qo_indptr(
    query_start_loc_cpu: torch.Tensor,
    num_query_tokens: int,
    *,
    group_size: int,
    device: Optional[Union[int, str, torch.device]] = None,
) -> torch.Tensor:
    """Build request-safe QToken-KvBlock-Sparse-Attention route offsets.

    ``query_start_loc_cpu`` contains cumulative offsets for real requests.
    Every request is chunked independently, so its final route may contain
    fewer than ``group_size`` rows. Any inert CUDA-graph padding suffix is
    partitioned separately. The returned Int32 ``qo_indptr`` can be passed to
    the packed-Q wrapper together with planned ``seq_len_q=group_size``.

    Parameters
    ----------
    query_start_loc_cpu : torch.Tensor
        One-dimensional CPU Int32 or Int64 cumulative offsets for the real
        requests. The first entry must be zero and the offsets must be
        nondecreasing.
    num_query_tokens : int
        Total flattened query-row count, including any inert CUDA-graph
        padding suffix.
    group_size : int
        Maximum number of rows in each request-safe QToken-KvBlock-Sparse-Attention route.
    device : int, str, torch.device, or None
        Destination device for the returned offsets. By default, PyTorch's
        default tensor device is used.

    Returns
    -------
    torch.Tensor
        Contiguous Int32 cumulative route offsets. Every nonempty route has at
        most ``group_size`` rows and no route crosses a request boundary.
    """

    group_size = _validate_prims_ts_q_token_kv_block_sparse_group_value(group_size)
    num_query_tokens = _validate_positive_int(num_query_tokens, "num_query_tokens")
    if (
        not isinstance(query_start_loc_cpu, torch.Tensor)
        or query_start_loc_cpu.device.type != "cpu"
        or query_start_loc_cpu.ndim != 1
        or query_start_loc_cpu.numel() < 2
        or query_start_loc_cpu.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "grouped QToken-KvBlock-Sparse-Attention requires CPU int32/int64 query_start_loc with at "
            "least two entries"
        )

    request_offsets = [int(value) for value in query_start_loc_cpu.tolist()]
    if (
        request_offsets[0] != 0
        or request_offsets[-1] < 0
        or request_offsets[-1] > num_query_tokens
        or any(
            end < begin
            for begin, end in zip(request_offsets, request_offsets[1:], strict=False)
        )
    ):
        raise ValueError(
            "grouped QToken-KvBlock-Sparse-Attention query boundaries do not cover valid rows"
        )

    route_offsets = [0]
    for begin, end in zip(request_offsets, request_offsets[1:], strict=False):
        next_offset = begin
        while next_offset < end:
            next_offset = min(next_offset + group_size, end)
            route_offsets.append(next_offset)

    # Keep graph-padding rows out of the final real request's route even when
    # that request ends with a partial group.
    next_offset = request_offsets[-1]
    while next_offset < num_query_tokens:
        next_offset = min(next_offset + group_size, num_query_tokens)
        route_offsets.append(next_offset)

    return torch.tensor(
        _validate_q_token_kv_block_sparse_route_offsets_cpu(
            route_offsets,
            num_query_tokens=num_query_tokens,
            group_size=group_size,
        ),
        dtype=torch.int32,
        device=device,
    )


def _validate_q_token_kv_block_sparse_route_offsets_cpu(
    route_offsets: list[int],
    *,
    num_query_tokens: int,
    group_size: int,
) -> tuple[int, ...]:
    """Validate CPU-generated QToken-KvBlock-Sparse-Attention routes before copying them to device storage."""

    offsets = tuple(route_offsets)
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != num_query_tokens:
        raise ValueError(
            "QToken-KvBlock-Sparse-Attention route offsets must cover exactly all query tokens"
        )
    lengths = tuple(
        end - begin for begin, end in zip(offsets, offsets[1:], strict=False)
    )
    if any(length <= 0 or length > group_size for length in lengths):
        raise ValueError(
            "QToken-KvBlock-Sparse-Attention route offsets must describe nonempty routes no longer than group_size"
        )
    return offsets


def _prepare_prims_ts_batch_decode_plan(
    query: torch.Tensor,
    kv_cache: PagedKVCache,
    workspace_buffer: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    *,
    seq_len_q: int,
    qo_indptr: Optional[torch.Tensor],
    max_seq_len_q: Optional[int],
    out: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
    mask_type: Literal["dense", "causal"],
    window_left: int,
    kv_layout: Literal["HND"],
    page_size: Optional[int],
    kv_scale_factors: Optional[PagedKVScaleFactors] = None,
    q_token_kv_block_sparse_page_memberships: Optional[torch.Tensor] = None,
    use_q_token_kv_block_sparse_route: bool = False,
    use_pdl: bool = False,
    split_kv: bool = True,
    direct_q1_inputs: tuple[torch.Tensor, ...] = (),
    direct_q1_max_model_len: Optional[int] = None,
    direct_q1_sparse_block_size: int = 4,
    share_pattern_across_kv_heads: bool = True,
) -> tuple[PrimsTSBatchDecodePlan, torch.Tensor]:
    """Validate and freeze one dense-block-table PrimTS launch contract."""

    _validate_layout(kv_layout)
    _validate_mask(mask_type)
    window_left = _validate_window_left(window_left, mask_type)
    use_packed_q, resolved_seq_len_q = _resolve_q_mode(
        seq_len_q=seq_len_q,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
        require_packed_max=True,
    )
    assert resolved_seq_len_q is not None
    seq_len_q = resolved_seq_len_q
    metadata_device, metadata_rows, max_num_pages = _validate_block_table_metadata(
        block_table,
        seq_lens,
    )
    normalized_cache = _normalize_native_paged_kv_cache(
        kv_cache, expected_device=query.device
    )
    pattern_heads = _num_sparse_pattern_heads(
        normalized_cache.num_kv_heads, share_pattern_across_kv_heads
    )
    if metadata_rows % pattern_heads:
        raise ValueError("metadata rows must be divisible by the pattern head count")
    batch_size = metadata_rows // pattern_heads
    _validate_q(
        query,
        seq_len_q=seq_len_q,
        use_packed_q=use_packed_q,
        device=metadata_device,
        batch_size=batch_size,
    )
    if qo_indptr is not None:
        _validate_qo_indptr(
            qo_indptr,
            expected_device=query.device,
            batch_size=batch_size,
        )
    k_cache = normalized_cache.k_cache
    v_cache = normalized_cache.v_cache
    num_kv_heads = normalized_cache.num_kv_heads
    storage_page_size = normalized_cache.storage_page_size
    head_dim = normalized_cache.head_dim
    if int(query.shape[-1]) != head_dim:
        raise ValueError(
            "paged_kv_cache logical head dimension must match the query: "
            f"expected {int(query.shape[-1])}, got {head_dim}"
        )
    num_qo_heads = int(query.shape[-2])
    _validate_head_geometry(num_qo_heads, num_kv_heads)
    page_size = _validate_page_size(
        storage_page_size if page_size is None else page_size
    )
    storage_page_size = _validate_storage_page_size(page_size, storage_page_size)
    max_seq_len = _validate_max_kv_len(max_seq_len, "max_seq_len")
    required_page_columns = (max_seq_len + page_size - 1) // page_size
    if max_num_pages < required_page_columns:
        raise ValueError(
            "block_table must have at least ceil(max_seq_len / page_size) "
            f"columns ({required_page_columns}), got {max_num_pages}"
        )
    output_dtype = out_dtype
    if output_dtype is None:
        if out is not None and not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        output_dtype = out.dtype if out is not None else query.dtype
    elif not isinstance(output_dtype, torch.dtype):
        raise TypeError("out_dtype must be a torch.dtype")
    _validate_dtype_pair(
        query.dtype,
        k_cache.dtype,
        v_cache.dtype,
        output_dtype,
        allow_fp8_bf16_output=use_q_token_kv_block_sparse_route,
    )
    k_sf_cache, v_sf_cache = _normalize_paged_kv_scale_factors(
        kv_scale_factors,
        k_cache=k_cache,
        logical_head_dim=head_dim,
    )
    resolved_device, device_index = _resolve_cuda_device(query.device)
    _validate_runtime_device(resolved_device)

    policy_args = (
        device_index,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        max_seq_len,
        seq_len_q,
        _dtype_key(query.dtype),
        _dtype_key(k_cache.dtype),
        _dtype_key(v_cache.dtype),
        _dtype_key(output_dtype),
        kv_layout,
        mask_type,
        use_packed_q,
        window_left,
        storage_page_size,
        use_q_token_kv_block_sparse_route,
        use_pdl,
        split_kv,
        share_pattern_across_kv_heads,
    )
    spec = _resolve_decode_launch_spec(*policy_args)
    if spec.config.uses_q_token_kv_block_sparse_page_membership:
        if q_token_kv_block_sparse_page_memberships is None:
            raise ValueError(
                "grouped QToken-KvBlock-Sparse-Attention requires a separate q_token_kv_block_sparse_page_memberships tensor"
            )
        _validate_q_token_kv_block_sparse_page_memberships(
            q_token_kv_block_sparse_page_memberships,
            expected_device=query.device,
            batch_size=metadata_rows,
            max_num_pages=max_num_pages,
        )
    else:
        if q_token_kv_block_sparse_page_memberships is not None:
            raise ValueError(
                "q_token_kv_block_sparse_page_memberships is valid only for grouped QToken-KvBlock-Sparse-Attention launches"
            )
        # Keep one compiled tensor ABI without requiring Q1 or dense callers to
        # allocate an unused table. The kernel constexpr-elides all reads.
        q_token_kv_block_sparse_page_memberships = block_table
    layout = _make_decode_workspace_layout(
        spec.scratch_shapes,
        output_dtype,
        use_separate_reduction_kernel=spec.config.use_separate_reduction_kernel,
        use_split_kv=spec.config.use_split_kv,
    )
    _validate_workspace_buffer(
        workspace_buffer,
        device=query.device,
        required_bytes=layout.total_bytes,
    )
    output_shape = _decode_output_shape(
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        seq_len_q=seq_len_q,
        head_dim=head_dim,
        total_q_tokens=int(query.shape[0]) if use_packed_q else None,
    )
    if out is None:
        out = torch.empty(output_shape, device=query.device, dtype=output_dtype)
    else:
        _validate_out(
            out,
            q=query,
            expected_shape=output_shape,
            seq_len_q=seq_len_q,
            use_packed_q=use_packed_q,
            output_dtype=output_dtype,
        )
    direct_q1_spec = None
    if direct_q1_inputs:
        assert use_q_token_kv_block_sparse_route and seq_len_q == 1
        assert direct_q1_max_model_len is not None
        direct_q1_spec = _DirectQ1CompileSpec(
            position_dtype_key=str(direct_q1_inputs[3].dtype).removeprefix("torch."),
            max_seq_len_kv=direct_q1_max_model_len,
            block_topk=int(direct_q1_inputs[0].shape[-1]),
            kv_block_size=direct_q1_sparse_block_size,
            page_capacity=max_num_pages,
        )
    compile_spec = _make_decode_compile_spec(
        spec,
        device_index=device_index,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        max_kv_len=max_seq_len,
        seq_len_q=seq_len_q,
        q_dtype_key=_dtype_key(query.dtype),
        k_dtype_key=_dtype_key(k_cache.dtype),
        v_dtype_key=_dtype_key(v_cache.dtype),
        output_dtype_key=_dtype_key(output_dtype),
        use_packed_q=use_packed_q,
        kv_prefix_mode="dynamic",
        kv_lengths_mode="dynamic",
        direct_q1_spec=direct_q1_spec,
    )
    compiled_main, compiled_reducer = _get_compiled_decode(compile_spec)
    workspace = _bind_decode_workspace(workspace_buffer, layout)
    # A fused split-KV launch uses a wrapping global completion counter.  The
    # final arriving CTA restores it to zero, so initialization belongs here,
    # once, rather than on every prepared-plan run.  The standalone reducer
    # does not read this counter, but the same initialization is harmless and
    # keeps one replay contract for every split policy.
    if layout.uses_split_kv:
        workspace.split_kv_counter.zero_()
    plan = PrimsTSBatchDecodePlan(
        _query_shape=tuple(query.shape),
        _query_stride=tuple(query.stride()),
        _output_shape=tuple(out.shape),
        _output_stride=tuple(out.stride()),
        _device=query.device,
        _q_dtype=query.dtype,
        _output_dtype=output_dtype,
        _head_dim=head_dim,
        _cache=normalized_cache,
        _k_sf_cache=k_sf_cache,
        _v_sf_cache=v_sf_cache,
        _seq_lens=seq_lens,
        _qo_indptr=qo_indptr,
        _block_table=block_table,
        _q_token_kv_block_sparse_page_memberships=q_token_kv_block_sparse_page_memberships,
        _workspace=workspace,
        _compiled_main=compiled_main,
        _compiled_reducer=compiled_reducer,
        _direct_q1_inputs=direct_q1_inputs,
    )
    return plan, out


@flashinfer_experimental_api
def prepare_prims_ts_batch_decode_with_kv_cache(
    query: torch.Tensor,
    kv_cache: PagedKVCache,
    workspace_buffer: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    *,
    out: torch.Tensor,
    kv_scale_factors: Optional[PagedKVScaleFactors] = None,
    seq_len_q: int = 1,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
    out_dtype: Optional[torch.dtype] = None,
    mask_type: Literal["dense", "causal"] = "dense",
    window_left: int = -1,
    kv_layout: Literal["HND"] = "HND",
    page_size: Optional[int] = None,
    split_kv: bool = True,
) -> PrimsTSBatchDecodePlan:
    """Validate and prepare a reusable dense-block-table PrimTS launch.

    This setup API is intended for frameworks that keep K/V, block-table
    metadata, and workspace storage stable while changing their values between launches.
    Call :meth:`PrimsTSBatchDecodePlan.run` on the hot path.  The returned plan
    is CUDA-graph compatible as long as captured tensor storage remains alive.

    Output and workspace must be disjoint from each other and from all inputs.
    Storage overlap is a caller precondition and is not checked.

    Parameters
    ----------
    query : torch.Tensor
        Fixed ``[B, Hq, D]`` or ``[B, SQ, Hq, D]`` query, or packed
        ``[total_q, Hq, D]`` query when ``qo_indptr`` is supplied.
    kv_cache : torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        Combined or separate HND paged K/V storage.
    workspace_buffer : torch.Tensor
        Caller-owned byte workspace, zero-initialized before the first launch
        for this semantic configuration.
    block_table : torch.Tensor
        Contiguous CUDA Int32 dense page table shaped ``[B, max_pages]``.
    seq_lens : torch.Tensor
        Positive live K/V sequence lengths, one per request.
    max_seq_len : int
        Static maximum K/V length used for policy selection and JIT caching.
    out : torch.Tensor
        Caller-owned output tensor whose storage remains stable across replays.
    kv_scale_factors : tuple[torch.Tensor, torch.Tensor], optional
        Required for packed NVFP4 K/V stored as uint8. K and V scales are
        FP8 tensors with width ``D / 16``; V scales use the TRTLLM-GEN
        4-token interleaved layout. The plan retains these tensors.
    seq_len_q : int
        Fixed query length when ``qo_indptr`` is omitted.
    qo_indptr : torch.Tensor, optional
        Cumulative Int32 query offsets selecting packed-query mode.
    max_seq_len_q : int, optional
        Static maximum packed-query route length.
    out_dtype : torch.dtype, optional
        Output dtype; defaults to ``out.dtype``.
    mask_type : {"dense", "causal"}
        Attention mask mode.
    window_left : int
        Left sliding-window extent, or ``-1`` to disable the window.
    kv_layout : {"HND"}
        Layout of the paged K/V cache.
    page_size : int, optional
        Semantic page size; defaults to the physical cache-page extent.
    split_kv : bool
        Permit automatic split fanout (True, default), or force S1 (False).
        Independent of packed/fixed Q. Match workspace sizing's value.

    Returns
    -------
    PrimsTSBatchDecodePlan
        Reusable prepared launch plan. Call :meth:`PrimsTSBatchDecodePlan.run`
        after updating values in the prepared tensor storage.
    """

    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a caller-owned torch.Tensor")
    plan, _ = _prepare_prims_ts_batch_decode_plan(
        query,
        kv_cache,
        workspace_buffer,
        block_table,
        seq_lens,
        max_seq_len,
        seq_len_q=seq_len_q,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
        out=out,
        out_dtype=out_dtype,
        mask_type=mask_type,
        window_left=window_left,
        kv_layout=kv_layout,
        page_size=page_size,
        kv_scale_factors=kv_scale_factors,
        split_kv=split_kv,
        use_q_token_kv_block_sparse_route=False,
    )
    return plan


class BatchDecodePagedTSWrapper:
    """Plan static paged-decode capacity and bind request metadata.

    A plan fixes device, batch size, geometry, dtypes, query storage mode, and
    maximum K/V capacity. Sequence lengths are supplied either once to
    :meth:`plan`, which retains an owned CUDA copy, or to every :meth:`run`.
    Every run supplies a row-strided page table; packed-query plans also consume
    per-run Q offsets.

    One wrapper supports one ordered execution lane. Concurrent streams or
    graph replays require separate wrappers and workspace buffers.
    """

    @flashinfer_experimental_api
    def __init__(self, kv_layout: Literal["HND"] = "HND") -> None:
        """Initialize an unplanned wrapper with one static K/V layout.

        Parameters
        ----------
        kv_layout : {"HND"}
            Layout of the paged K/V cache. Only ``"HND"`` is supported.
        """

        _validate_layout(kv_layout)
        self._kv_layout = kv_layout
        self._plan_state: Optional[_DecodePlanState] = None

    def _require_plan_state(self) -> _DecodePlanState:
        state = self._plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        return state

    @property
    def _policy(self) -> tuple[tuple[str, object], ...]:
        """Return the immutable policy record for the published plan."""

        return self._require_plan_state().policy

    @flashinfer_experimental_api
    def plan(
        self,
        device: Union[int, str, torch.device],
        batch_size: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        max_kv_len: int,
        *,
        max_seq_len_q: int = 1,
        packed_query: bool = False,
        q_data_type: torch.dtype = torch.float16,
        k_data_type: Optional[torch.dtype] = None,
        v_data_type: Optional[torch.dtype] = None,
        kv_data_type: Optional[torch.dtype] = None,
        o_data_type: Optional[torch.dtype] = None,
        mask_type: Literal["dense", "causal"] = "dense",
        window_left: int = -1,
        seq_lens: Optional[Union[Sequence[int], torch.Tensor]] = None,
        workspace_buffer: Optional[torch.Tensor] = None,
        storage_page_size: Optional[int] = None,
        split_kv: bool = True,
        validate: bool = True,
        initialize_workspace: bool = True,
    ) -> None:
        """Compile one static-capacity plan and optionally own sequence lengths.

        ``max_seq_len_q`` is the exact fixed Q length when ``packed_query`` is
        false and the per-request capacity when ``packed_query`` is true. A
        packed run supplies ``qo_indptr`` and compact ``[total_q, Hq, D]``
        storage.

        ``seq_lens`` optionally fixes the per-request lengths for the lifetime
        of this plan. Planning validates the host values, retains them, and
        creates an owned CUDA copy used by every launch. In this mode,
        :meth:`run` requires its ``seq_lens`` argument to be ``None``. When
        omitted, both K/V prefix and length handling compile dynamically and
        every run must supply a CUDA length tensor.

        ``workspace_buffer`` is caller-owned scratch for this plan. It is
        allocated when omitted, retained by the frozen plan state, and never
        reset by ``run``. By default planning initializes its control sections;
        ``initialize_workspace=False`` preserves a caller-initialized buffer.

        Parameters
        ----------
        device : int, str, or torch.device
            CUDA device on which the specialization is compiled and run.
        batch_size : int
            Exact number of requests in every run.
        num_qo_heads : int
            Number of query/output heads.
        num_kv_heads : int
            Number of K/V heads.
        head_dim : int
            Query, key, value, and output head dimension.
        page_size : int
            Number of K/V tokens addressed by each page-table entry.
        storage_page_size : int, optional
            Physical cache-page extent, defaulting to ``page_size``. Larger
            physical pages must be divisible by the logical ``page_size``.
        max_kv_len : int
            Per-request K/V length capacity used for policy selection,
            compilation, and workspace sizing.
        max_seq_len_q : int
            Exact fixed Q length, or per-request capacity for packed Q.
            Defaults to ``1``.
        packed_query : bool
            Select compact ``[total_q, Hq, D]`` query/output storage instead
            of fixed ``[B, SQ, Hq, D]`` storage. Fixed SQ1 storage is
            ``[B, Hq, D]``. Defaults to ``False``.
        q_data_type : torch.dtype
            Query dtype used to compile the plan. Defaults to
            ``torch.float16``.
        k_data_type : torch.dtype, optional
            K dtype used to compile the plan. Defaults to ``q_data_type``.
        v_data_type : torch.dtype, optional
            V dtype used to compile the plan. Defaults to ``k_data_type``;
            ``torch.float8_e4m3fn`` with BF16 Q/K selects the QK-BF16/PV-FP8
            path and requires separate ``(K, V)`` cache tensors.
        kv_data_type : torch.dtype, optional
            Compatibility alias setting both K and V storage dtypes. Explicit
            K/V dtypes must agree with it. Use ``torch.uint8`` for packed
            NVFP4; ``head_dim`` stays logical and runs supply scale tensors.
        o_data_type : torch.dtype, optional
            Output dtype used to compile the plan. Defaults to
            ``q_data_type``.
        mask_type : {"dense", "causal"}
            Attention mask mode. Defaults to ``"dense"``.
        window_left : int
            Left sliding-window extent, or ``-1`` to disable the window. A
            non-negative value requires causal masking.
        seq_lens : Sequence[int] or torch.Tensor, optional
            Host-only per-request K/V lengths owned by the resulting plan and
            used for specialization. A tensor must be a one-dimensional CPU
            int32 or int64 tensor. The sequence must contain exactly
            ``batch_size`` positive values no larger than ``max_kv_len``.
        workspace_buffer : torch.Tensor, optional
            Caller-owned contiguous int8 or uint8 scratch on ``device``. It
            must be 32-byte aligned and large enough for the selected plan.
            When omitted, planning allocates the buffer. The retained buffer
            is exclusive to one in-flight launch or graph replay.
        split_kv : bool
            Permit automatic useful split fanout (True, default), or force
            unsplit execution (False), independently of ``packed_query``.
            Typically False for prefill and True for decode. This choice is
            frozen for the lifetime of the plan.
        validate : bool
            Validate static geometry and caller scratch. Defaults to ``True``.
            Disable only for previously validated inputs and warmed topology.
        initialize_workspace : bool
            Initialize control sections during planning, default ``True``.
            Set to ``False`` only for caller-initialized scratch with unchanged
            layout. Newly allocated scratch is always initialized.
        """

        k_data_type, v_data_type = _resolve_kv_dtypes(
            q_data_type, k_data_type, v_data_type, kv_data_type
        )
        if o_data_type is None:
            o_data_type = q_data_type
        seq_len_q = max_seq_len_q
        storage_page_size = (
            page_size if storage_page_size is None else storage_page_size
        )
        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool")
        if not isinstance(initialize_workspace, bool):
            raise TypeError("initialize_workspace must be a bool")
        if validate:
            if not isinstance(packed_query, bool):
                raise TypeError("packed_query must be a bool")
            batch_size = _validate_positive_int(batch_size, "batch_size")
            head_dim = _validate_head_dim(head_dim)
            page_size = _validate_page_size(page_size)
            storage_page_size = _validate_storage_page_size(
                page_size, storage_page_size
            )
            max_kv_len = _validate_max_kv_len(max_kv_len, "max_kv_len")
            seq_len_q = _validate_seq_len_q(max_seq_len_q)
            _validate_head_geometry(num_qo_heads, num_kv_heads)
            _validate_decode_query_head_extent(
                batch_size=batch_size,
                num_qo_heads=num_qo_heads,
                max_seq_len_q=seq_len_q,
            )
            _validate_mask(mask_type)
            window_left = _validate_window_left(window_left, mask_type)

            _validate_dtype_pair(q_data_type, k_data_type, v_data_type, o_data_type)

            specialization_seq_lens = _normalize_plan_seq_lens(
                seq_lens,
                batch_size=batch_size,
                max_kv_len=max_kv_len,
            )
            if (
                specialization_seq_lens is not None
                and mask_type == "causal"
                and not packed_query
            ):
                for request_idx, kv_len in enumerate(specialization_seq_lens):
                    if seq_len_q > kv_len:
                        raise ValueError(
                            "causal decode requires every per-request Q length to be "
                            "no greater than its K/V length; request "
                            f"{request_idx} has Q={seq_len_q} and K/V={kv_len}"
                        )

        else:
            specialization_seq_lens = _normalize_plan_seq_lens(
                seq_lens, batch_size=batch_size, max_kv_len=max_kv_len
            )

        device, device_index = _resolve_cuda_device(device)
        if validate:
            _validate_runtime_device(device)

        policy_args = (
            device_index,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            max_kv_len,
            seq_len_q,
            _dtype_key(q_data_type),
            _dtype_key(k_data_type),
            _dtype_key(v_data_type),
            _dtype_key(o_data_type),
            self._kv_layout,
            mask_type,
            packed_query,
            window_left,
            storage_page_size,
        )
        spec = _resolve_decode_launch_spec(*policy_args, split_kv=split_kv)
        if specialization_seq_lens is None:
            kv_prefix_mode: Literal["dynamic", "planned_full"] = "dynamic"
            kv_lengths_mode: Literal["dynamic", "planned_uniform_max"] = "dynamic"
        else:
            static_full_split_prefix = _planned_full_split_prefix(
                spec.config,
                specialization_seq_lens,
                seq_len_q=seq_len_q,
                max_kv_len=max_kv_len,
                mask_type=mask_type,
            )
            kv_prefix_mode = "planned_full" if static_full_split_prefix else "dynamic"
            has_unpaired_kv_tail = _planned_kv_domain_has_unpaired_tail(
                spec.config,
                max_kv_len,
            )
            requires_runtime_kv_lengths = (
                has_unpaired_kv_tail
                or spec.config.use_sliding_window_causal
                or (
                    spec.config.use_persistent_scheduler
                    and spec.config.uses_runtime_q_kv_union
                )
            )
            kv_lengths_mode = (
                "dynamic"
                if requires_runtime_kv_lengths
                else _planned_kv_lengths_mode(
                    specialization_seq_lens,
                    max_kv_len=max_kv_len,
                )
            )
        compile_spec = _make_decode_compile_spec(
            spec,
            device_index=device_index,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            max_kv_len=max_kv_len,
            seq_len_q=seq_len_q,
            q_dtype_key=_dtype_key(q_data_type),
            k_dtype_key=_dtype_key(k_data_type),
            v_dtype_key=_dtype_key(v_data_type),
            output_dtype_key=_dtype_key(o_data_type),
            use_packed_q=packed_query,
            kv_prefix_mode=kv_prefix_mode,
            kv_lengths_mode=kv_lengths_mode,
        )
        compiled_main, compiled_reducer = _get_compiled_decode(compile_spec)
        policy = spec.policy + (
            ("kv_prefix_mode", kv_prefix_mode),
            ("kv_lengths_mode", kv_lengths_mode),
            ("split_kv_allowed", split_kv),
        )
        workspace_layout = _make_decode_workspace_layout(
            spec.scratch_shapes,
            o_data_type,
            use_separate_reduction_kernel=(spec.config.use_separate_reduction_kernel),
            use_split_kv=spec.config.use_split_kv,
        )
        if workspace_buffer is None:
            initialize_workspace = True
            workspace_buffer = torch.empty(
                workspace_layout.total_bytes,
                device=device,
                dtype=torch.int8,
            )
        elif validate:
            _validate_workspace_buffer(
                workspace_buffer,
                device=device,
                required_bytes=workspace_layout.total_bytes,
            )
        workspace = _bind_decode_workspace(workspace_buffer, workspace_layout)
        if initialize_workspace:
            workspace.split_kv_counter.zero_()
            workspace.cu_seqlens_q.zero_()
            workspace.attention_sinks.zero_()
        # Materialize from the normalized tuple so the plan never aliases
        # caller-owned host or device storage.
        planned_seq_lens_device = (
            None
            if specialization_seq_lens is None
            else torch.tensor(
                specialization_seq_lens,
                dtype=torch.int32,
                device=device,
            )
        )

        candidate = _DecodePlanState(
            device=device,
            device_index=device_index,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            use_packed_q=packed_query,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            max_kv_len=max_kv_len,
            q_dtype=q_data_type,
            k_dtype=k_data_type,
            v_dtype=v_data_type,
            output_dtype=o_data_type,
            mask_type=mask_type,
            window_left=window_left,
            config=spec.config,
            workspace_buffer=workspace_buffer,
            workspace_layout=workspace_layout,
            workspace=workspace,
            compiled_main=compiled_main,
            compiled_reducer=compiled_reducer,
            kv_prefix_mode=kv_prefix_mode,
            kv_lengths_mode=kv_lengths_mode,
            planned_seq_lens_host=specialization_seq_lens,
            planned_seq_lens_device=planned_seq_lens_device,
            policy=policy,
            storage_page_size=storage_page_size,
        )
        # This is the only wrapper mutation. Any failure above leaves the
        # previous complete plan revision usable.
        self._plan_state = candidate

    @flashinfer_experimental_api
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: PagedKVCache,
        seq_lens: Optional[torch.Tensor],
        block_tables: torch.Tensor,
        *,
        kv_scale_factors: Optional[PagedKVScaleFactors] = None,
        qo_indptr: Optional[torch.Tensor] = None,
        bmm1_scale: Optional[float] = None,
        bmm2_scale: float = 1.0,
        out: Optional[torch.Tensor] = None,
        validate: bool = True,
    ) -> torch.Tensor:
        """Launch the current plan with one sequence-length owner.

        When :meth:`plan` received sequence lengths, ``seq_lens`` must be
        ``None`` and the plan's owned CUDA copy is used. Otherwise, ``seq_lens``
        must be supplied on every run. This ownership check is unconditional.

        ``validate=True`` performs structural and value validation. It
        reads per-run metadata values back to the host.
        ``validate=False`` treats every run argument as a trusted binding,
        performs no explicit wrapper validation, and remains free of metadata
        device-to-host synchronization. K/V view selection, scale forwarding,
        and optional output allocation are unavoidable in both modes.

        Packed plans require ``qo_indptr`` with ``B + 1`` int32 offsets. Fixed
        plans require ``qo_indptr`` to be omitted. Per-run metadata tensors may
        change identity between ordered runs.

        In either validation mode, output and workspace must be disjoint from
        each other and from all inputs. Storage overlap is not checked.

        Parameters
        ----------
        q : torch.Tensor
            Runtime fixed or packed query tensor matching the plan.
        paged_kv_cache : torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            Runtime combined or separate paged K/V storage matching the plan.
        seq_lens : torch.Tensor, optional
            Per-run contiguous int32 CUDA K/V lengths with shape ``[B]``.
            Required when omitted from :meth:`plan` and otherwise required to
            be ``None``.
        block_tables : torch.Tensor
            Per-run int32 CUDA physical page IDs with shape ``[B, C]``. Entries
            must be contiguous within each row; the row stride may be any value
            at least ``C``. Inactive tail entries are ignored.
        kv_scale_factors : tuple[torch.Tensor, torch.Tensor], optional
            Required for packed NVFP4 K/V. Compact FP8 ``(K_SF, V_SF)`` tensors
            have shape ``[pages, Hkv, storage_page_size, D/16]``; K scales are token-major
            and V scales use the 4-token interleaved layout. Omit for other
            dtypes. Scale tensors must not overlap output or workspace storage.
        qo_indptr : torch.Tensor, optional
            Per-run cumulative query offsets with shape ``[B + 1]``. Required for
            a packed-query plan and rejected for a fixed-query plan.
        bmm1_scale : float, optional
            QK scaling factor. Defaults to the inverse square root of
            ``head_dim``.
        bmm2_scale : float
            Value/output scaling factor. Defaults to ``1.0``.
        out : torch.Tensor, optional
            Caller-owned output tensor. A new tensor is allocated when omitted.
        validate : bool
            Run explicit structural and value validation. Disable only
            when the caller guarantees the complete runtime contract. Sequence
            length ownership is enforced in either mode. Defaults to ``True``.

        Returns
        -------
        torch.Tensor
            The fixed or packed attention output.
        """

        state = self._require_plan_state()
        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool")
        planned_seq_lens = state.planned_seq_lens_device
        plan_owns_seq_lens = planned_seq_lens is not None
        if plan_owns_seq_lens:
            if seq_lens is not None:
                raise ValueError(
                    "seq_lens must be None when plan() owns sequence lengths"
                )
            effective_seq_lens = planned_seq_lens
        else:
            if seq_lens is None:
                raise ValueError(
                    "seq_lens is required when plan() does not own sequence lengths"
                )
            effective_seq_lens = seq_lens

        runtime_qo_indptr = qo_indptr if state.use_packed_q else None
        if validate:
            (
                metadata_device,
                metadata_batch_size,
                _,
            ) = (
                _validate_block_tables(block_tables)
                if plan_owns_seq_lens
                else _validate_block_table_metadata(
                    block_tables,
                    effective_seq_lens,
                )
            )
            if metadata_device != state.device:
                raise ValueError(
                    f"per-run metadata must be on {state.device}, got {metadata_device}"
                )
            if metadata_batch_size != state.batch_size:
                raise ValueError(
                    "per-run metadata batch size must match the plan "
                    f"({state.batch_size}), got {metadata_batch_size}"
                )
            if state.use_packed_q:
                if qo_indptr is None:
                    raise ValueError("qo_indptr is required for a packed-Q plan")
                _validate_qo_indptr(
                    qo_indptr,
                    expected_device=state.device,
                    batch_size=state.batch_size,
                )
            elif qo_indptr is not None:
                raise ValueError("qo_indptr cannot be used with a fixed-Q plan")

            runtime = _prepare_decode_runtime(
                q,
                paged_kv_cache,
                kv_scale_factors,
                device=state.device,
                batch_size=state.batch_size,
                seq_len_q=state.seq_len_q,
                use_packed_q=state.use_packed_q,
                num_qo_heads=state.num_qo_heads,
                num_kv_heads=state.num_kv_heads,
                head_dim=state.head_dim,
                page_size=state.storage_page_size or state.page_size,
                q_dtype=state.q_dtype,
                k_dtype=state.k_dtype,
                v_dtype=state.v_dtype,
                output_dtype=state.output_dtype,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                out=out,
            )
            _validate_decode_run_metadata_values(
                runtime,
                planned_seq_lens_host=state.planned_seq_lens_host,
                max_kv_len=state.max_kv_len,
                page_size=state.page_size,
                use_packed_q=state.use_packed_q,
                seq_len_q=state.seq_len_q,
                batch_size=state.batch_size,
                mask_type=state.mask_type,
                seq_lens=effective_seq_lens,
                block_tables=block_tables,
                qo_indptr=runtime_qo_indptr,
            )
        else:
            runtime = _prepare_decode_runtime_unchecked(
                q,
                paged_kv_cache,
                kv_scale_factors,
                output_dtype=state.output_dtype,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                out=out,
            )

        return _launch_decode(
            runtime,
            seq_lens=effective_seq_lens,
            qo_indptr=runtime_qo_indptr,
            block_tables=block_tables,
            workspace=state.workspace,
            compiled_main=state.compiled_main,
            compiled_reducer=state.compiled_reducer,
        )


@flashinfer_experimental_api
def batch_decode_with_paged_kv_cache(
    q: torch.Tensor,
    paged_kv_cache: PagedKVCache,
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    *,
    kv_scale_factors: Optional[PagedKVScaleFactors] = None,
    seq_len_q: int = 1,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
    mask_type: Literal["dense", "causal"] = "dense",
    window_left: int = -1,
    kv_layout: Literal["HND"] = "HND",
    bmm1_scale: Optional[float] = None,
    bmm2_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    page_size: Optional[int] = None,
    split_kv: bool = True,
    workspace_buffer: Optional[torch.Tensor] = None,
    max_kv_len: Optional[int] = None,
    validate: bool = True,
) -> torch.Tensor:
    """One-shot fixed or packed-Q paged decode from fixed page tables.

    SQ1 preserves the ``[B, Hq, D]`` query/output contract. For fixed
    ``seq_len_q>1``, query and output are both token-major
    ``[B, SQ, Hq, D]``. Providing cumulative ``qo_indptr`` selects packed
    ``[total_q, Hq, D]`` query/output; the wrapper derives ``max_seq_len_q``
    once when it is omitted. No transpose is hidden here.

    Without caller scratch, the convenience planner reads ``seq_lens_kv`` on
    the host and retains the existing length-specialized kernel path. This mode
    is not CUDA-graph-capturable. See Notes for the explicit capture-safe mode.

    Parameters
    ----------
    q : torch.Tensor
        Fixed or packed query tensor.
    paged_kv_cache : torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        Combined or separate paged K/V storage. K and V dtypes are taken from
        the tensors; a ``torch.float8_e4m3fn`` V with ``torch.bfloat16`` Q/K
        requires the separate ``(K, V)`` form.
    block_tables : torch.Tensor
        Fixed row-strided ``[B, C]`` page table. Rows may have padding between
        them, but each row must be contiguous.
    seq_lens_kv : torch.Tensor
        Per-request K/V sequence lengths with shape ``[B]``.
    kv_scale_factors : tuple[torch.Tensor, torch.Tensor], optional
        Required for packed NVFP4 K/V. Compact FP8 ``(K_SF, V_SF)`` tensors
        have shape ``[pages, Hkv, storage_page_size, D/16]``; K scales are token-major
        and V scales use the 4-token interleaved layout. Omit for other dtypes.
    seq_len_q : int
        Fixed query length when ``qo_indptr`` is omitted. In packed-query mode,
        a non-default value is a backward-compatible alias for
        ``max_seq_len_q`` and must agree with it when both are provided.
    qo_indptr : torch.Tensor, optional
        Cumulative query offsets selecting packed-query mode.
    max_seq_len_q : int, optional
        Per-request packed-query length capacity. When omitted for packed Q,
        it is derived from ``qo_indptr`` unless a non-default ``seq_len_q``
        supplies the bound. In fixed-query mode, it must equal ``seq_len_q``.
    mask_type : {"dense", "causal"}
        Attention mask mode.
    window_left : int
        Left sliding-window extent, or ``-1`` to disable the window.
    kv_layout : {"HND"}
        Layout of the paged K/V cache.
    bmm1_scale, bmm2_scale : float, optional
        QK and value/output scaling factors.
    out : torch.Tensor, optional
        Caller-owned output tensor.
    out_dtype : torch.dtype, optional
        Output dtype; defaults to ``out.dtype`` or the query dtype.
    page_size : int, optional
        Semantic page-table size, defaulting to the physical cache extent.
        A smaller supported value must divide the physical extent and uses
        encoded subpage locators.
    split_kv : bool
        Allow automatic split fanout (True, default), or force S1 (False),
        independently of packed/fixed query storage.
    workspace_buffer : torch.Tensor, optional
        Zero-initialized caller-owned byte scratch; re-zero when layout inputs
        (including batch size) change. Exclusive to one in-flight launch/graph.
    max_kv_len : int, optional
        Static K/V capacity; defaults to the metadata maximum with validation.
    validate : bool
        Validate tensors and metadata values (may synchronize), default True.
        False trusts the caller and requires workspace and explicit bounds;
        skips tensor and metadata validation. Invalid inputs have undefined behavior.

    Returns
    -------
    torch.Tensor
        The fixed or packed attention output.
    Notes
    -----
    Both owned and caller-provided scratch use the same wrapper plan/run path.
    For CUDA Graph capture, supply ``workspace_buffer``, ``max_kv_len``,
    ``out``, and ``validate=False``; packed Q also needs an explicit
    ``max_seq_len_q`` (or its non-default ``seq_len_q`` alias).
    Warm this exact topology outside capture first. Retain stable tensor storage
    and mutate metadata only between completed launches/replays. All live K/V
    lengths must be positive and within the static bound, active page IDs must
    index the cache, and causal per-request Q lengths must not exceed K/V lengths.
    Packed offsets must start at zero, end at the query token count, and have
    strictly positive deltas within the Q bound. Scratch/output must not alias
    any inputs or each other. No metadata is copied to the host on the trusted
    explicit path. Kernel policy, necessary control resets, and output layout
    are unchanged. Missing output may be allocated only outside capture.

    """

    if not isinstance(validate, bool):
        raise TypeError("validate must be a bool")
    if validate or workspace_buffer is None or out is None:
        if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CUDA graph capture requires workspace_buffer, max_kv_len, "
                "out, validate=False, and explicit packed-Q bounds; warm up first"
            )
    if not validate:
        if workspace_buffer is None:
            raise ValueError("validate=False requires workspace_buffer")
        if max_kv_len is None:
            raise ValueError("validate=False requires max_kv_len")
        if qo_indptr is not None and max_seq_len_q is None and seq_len_q == 1:
            raise ValueError("validate=False requires max_seq_len_q for packed Q")

    allocate_workspace = workspace_buffer is None
    if validate:
        use_packed_q, resolved_seq_len_q = _resolve_q_mode(
            seq_len_q=seq_len_q,
            qo_indptr=qo_indptr,
            max_seq_len_q=max_seq_len_q,
            require_packed_max=False,
        )
        metadata_device, batch_size, _ = _validate_block_table_metadata(
            block_tables, seq_lens_kv
        )
        if metadata_device != q.device:
            raise ValueError(
                f"paged-KV metadata must be on {q.device}, got {metadata_device}"
            )
        if qo_indptr is not None:
            _validate_qo_indptr(
                qo_indptr, expected_device=q.device, batch_size=batch_size
            )
            if resolved_seq_len_q is None:
                resolved_seq_len_q, _, _ = _read_packed_q_plan_metadata(qo_indptr)
        assert resolved_seq_len_q is not None
        _validate_q(
            q,
            seq_len_q=resolved_seq_len_q,
            use_packed_q=use_packed_q,
            device=q.device,
            batch_size=batch_size,
        )
        k_cache, v_cache, _, num_kv_heads, storage_page_size, head_dim = (
            _normalize_paged_kv_cache_views(paged_kv_cache, expected_device=q.device)
        )
    else:
        use_packed_q = qo_indptr is not None
        resolved_seq_len_q = (
            max_seq_len_q if use_packed_q and max_seq_len_q is not None else seq_len_q
        )
        batch_size = int(seq_lens_kv.shape[0])
        k_cache = (
            paged_kv_cache[:, 0]
            if isinstance(paged_kv_cache, torch.Tensor)
            else paged_kv_cache[0]
        )
        v_cache = (
            paged_kv_cache[:, 1]
            if isinstance(paged_kv_cache, torch.Tensor)
            else paged_kv_cache[1]
        )
        num_kv_heads, storage_page_size, storage_head_dim = map(int, k_cache.shape[1:])
        head_dim = (
            storage_head_dim * 2 if k_cache.dtype == torch.uint8 else storage_head_dim
        )

    page_size = storage_page_size if page_size is None else page_size
    num_qo_heads = int(q.shape[-2])
    if validate and out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    output_dtype = (
        out_dtype
        if out_dtype is not None
        else (out.dtype if out is not None else q.dtype)
    )
    # Only the allocating convenience call freezes lengths. Caller scratch
    # always uses dynamic metadata, including when validation is enabled.
    seq_lens_host = (
        tuple(int(value) for value in seq_lens_kv.tolist())
        if allocate_workspace
        else None
    )
    if max_kv_len is None:
        max_kv_len = (
            max(seq_lens_host)
            if seq_lens_host is not None
            else int(seq_lens_kv.max().item())
        )
    if validate and out is not None:
        _validate_out(
            out,
            q=q,
            expected_shape=_decode_output_shape(
                batch_size=batch_size,
                num_qo_heads=num_qo_heads,
                seq_len_q=resolved_seq_len_q,
                head_dim=head_dim,
                total_q_tokens=int(q.shape[0]) if use_packed_q else None,
            ),
            seq_len_q=resolved_seq_len_q,
            use_packed_q=use_packed_q,
            output_dtype=output_dtype,
        )
    if allocate_workspace:
        workspace_bytes = get_prims_ts_batch_decode_workspace_size(
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            max_kv_len,
            seq_len_q=seq_len_q,
            qo_indptr=qo_indptr,
            max_seq_len_q=resolved_seq_len_q,
            q_dtype=q.dtype,
            k_dtype=k_cache.dtype,
            v_dtype=v_cache.dtype,
            out_dtype=output_dtype,
            mask_type=mask_type,
            window_left=window_left,
            kv_layout=kv_layout,
            storage_page_size=storage_page_size,
            device=q.device,
            split_kv=split_kv,
        )
        workspace_buffer = torch.empty(
            workspace_bytes, dtype=torch.int8, device=q.device
        )

    wrapper = BatchDecodePagedTSWrapper(kv_layout=kv_layout)
    wrapper.plan(
        q.device,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        max_kv_len,
        max_seq_len_q=resolved_seq_len_q,
        storage_page_size=storage_page_size,
        split_kv=split_kv,
        packed_query=use_packed_q,
        q_data_type=q.dtype,
        k_data_type=k_cache.dtype,
        v_data_type=v_cache.dtype,
        o_data_type=output_dtype,
        mask_type=mask_type,
        window_left=window_left,
        seq_lens=seq_lens_host,
        workspace_buffer=workspace_buffer,
        validate=validate,
        initialize_workspace=allocate_workspace,
    )
    return wrapper.run(
        q,
        paged_kv_cache,
        None if seq_lens_host is not None else seq_lens_kv,
        block_tables,
        kv_scale_factors=kv_scale_factors,
        qo_indptr=qo_indptr,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        out=out,
        validate=validate,
    )


__all__ = [
    "BatchDecodePagedTSWrapper",
    "PrimsTSBatchDecodePlan",
    "batch_decode_with_paged_kv_cache",
    "get_prims_ts_batch_decode_workspace_size",
    "make_q_token_kv_block_sparse_qo_indptr",
    "suggest_q_token_kv_block_sparse_group_size",
    "validate_q_token_kv_block_sparse_group_size",
    "prepare_prims_ts_batch_decode_with_kv_cache",
]
