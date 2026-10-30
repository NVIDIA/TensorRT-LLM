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

"""Task-scheduled paged MLA decode with a plan/run lifecycle."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
import copy
import math
import struct
import functools
from typing import Any, Literal, Optional, cast

import torch

from ._compile_cache import build_and_load_attention_kernel
from ._hardware import get_max_active_clusters


from .kernels.mla_decode.helpers.nvfp4_kv import (
    cache_record_size,
    validate_nvfp4_kv_formats,
)

from flashinfer.api_logging import flashinfer_api as flashinfer_experimental_api
from flashinfer.utils import get_device_sm_count


from .decode import (
    prepare_lse,
    validate_lse_mode,
    _WorkspaceSection,
    _align_up,
    _append_workspace_section,
    _dtype_key,
    _resolve_cuda_device,
    _validate_16byte_alignment,
    _validate_mask,
    _validate_page_size,
    _validate_positive_int,
    _validate_scale,
    _validate_runtime_device,
    _validate_workspace_buffer,
    _workspace_section_view,
    resolve_max_sequence_length,
)


from .kernels.mla_decode.sparse_policy import (
    prefer_sparse_d576_reload,
    select_sparse_mla_profile,
)

_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 2"
_MLA_LATENT_DIM = 512
_MLA_ROPE_DIM = 64
_MLA_QUERY_DIM = _MLA_LATENT_DIM + _MLA_ROPE_DIM
_SUPPORTED_INPUT_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)
_SUPPORTED_OUTPUT_DTYPES = (torch.bfloat16,)
_INT32_MAX = 2**31 - 1
# Retain the existing broad input guard; after policy resolution, validate
# the complete padded endpoint from its actual capacity (including >128).
# This legacy coordinate reserve is not a split-count ceiling.
_MLA_LEGACY_KV_COORDINATE_RESERVE = 128 * 2 * 128
_MLA_MAX_KV_LEN = _INT32_MAX - (_MLA_LEGACY_KV_COORDINATE_RESERVE - 1)


@dataclass(frozen=True)
class _MLADecodeLaunchSpec:
    """Automatic MLA policy and scratch geometry for one plan."""

    kernel: Any
    policy: tuple[tuple[str, object], ...]
    kernel_workspace_bytes: int
    split_kv: int


@dataclass(frozen=True)
class _MLADecodeCompileSpec:
    """Batch-independent identity and implementation for one MLA compile."""

    device_index: int
    kernel_signature: tuple[object, ...]
    num_qo_heads: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    page_size: int
    q_dtype_key: str
    out_dtype_key: str
    max_q_len: int | None
    packed_query: bool
    has_kernel_workspace: bool
    # Active split count is a runtime operand; kernel_signature carries capacity.
    split_kv: int = field(compare=False, hash=False)
    kernel: Any = field(compare=False, hash=False, repr=False)
    write_final_lse: bool = True
    lse_mode: str = "none"


@dataclass(frozen=True)
class _MLAWorkspaceLayout:
    """Private MLA scratch layout; only ``total_bytes`` is public."""

    kernel_workspace: _WorkspaceSection
    lse: _WorkspaceSection
    total_bytes: int


@dataclass(frozen=True)
class _MLAWorkspaceViews:
    kernel_workspace: Optional[torch.Tensor]
    lse: Optional[torch.Tensor]


@dataclass(frozen=True)
class _MLADecodePlanState:
    """Immutable compile and workspace state for one reusable MLA plan."""

    device: torch.device
    batch_size: int
    num_qo_heads: int
    max_q_len: int
    packed_query: bool
    kv_lora_rank: int
    qk_rope_head_dim: int
    page_size: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    out_dtype: torch.dtype
    mask_type: Literal["dense", "causal"]
    max_kv_len: int
    required_page_columns: int
    workspace_buffer: torch.Tensor
    workspace_layout: _MLAWorkspaceLayout
    workspace_views: _MLAWorkspaceViews
    compiled: Callable[..., object]
    policy: tuple[tuple[str, object], ...]
    split_kv: int
    softmax_rescale_threshold_log2: float = 0.0
    lse_mode: str = "none"


@dataclass(frozen=True)
class MLAAttentionTuning:
    """Compile-time tuning of one PrimTS MLA attention plan.

    It applies to every query length the MLA wrappers serve, context
    (multi-token Q) as well as generation. ``softmax_rescale_threshold_log2``
    is in binades. ``None`` keeps the entry point's default: exact rescale for
    dense, and each schedule's existing deferral for sparse. An explicit
    value, including ``0.0``, is compiled into the kernel; see
    :meth:`BatchMLADecodePagedTSWrapper.plan`.
    """

    softmax_rescale_threshold_log2: Optional[float] = None


def _tuning_rescale_threshold(tuning: object) -> Optional[float]:
    if tuning is None:
        return None
    if not isinstance(tuning, MLAAttentionTuning):
        raise TypeError("tuning must be an MLAAttentionTuning or None")
    return tuning.softmax_rescale_threshold_log2


@dataclass(frozen=True)
class _MLARuntime:
    query: torch.Tensor
    normalized_cache: torch.Tensor
    out: torch.Tensor
    num_physical_pages: int
    softmax_scale: float
    output_scale: float
    extra_cache: Optional[torch.Tensor] = None
    sparse_inputs: Optional[tuple] = None
    output_quant_args: Optional[tuple] = None


# Pack kernel scratch bytes and [batch, max Q tokens, heads] FP32 LSE storage.
def _make_mla_workspace_layout(
    kernel_workspace_bytes: int,
    batch_size: int,
    num_qo_heads: int,
    max_q_len: int = 1,
    *,
    include_lse: bool = True,
) -> _MLAWorkspaceLayout:
    kernel_workspace, byte_end = _append_workspace_section(
        0, (kernel_workspace_bytes,), torch.int8
    )
    lse, byte_end = _append_workspace_section(
        byte_end,
        (batch_size, max_q_len, num_qo_heads) if include_lse else (0,),
        torch.float32,
    )
    return _MLAWorkspaceLayout(
        kernel_workspace=kernel_workspace,
        lse=lse,
        total_bytes=_align_up(byte_end),
    )


def _bind_mla_workspace(
    workspace_buffer: torch.Tensor, layout: _MLAWorkspaceLayout
) -> _MLAWorkspaceViews:
    kernel_workspace = None
    if layout.kernel_workspace.byte_size > 0:
        kernel_workspace = _workspace_section_view(
            workspace_buffer, layout.kernel_workspace
        )
    return _MLAWorkspaceViews(
        kernel_workspace=kernel_workspace,
        lse=_workspace_section_view(workspace_buffer, layout.lse)
        if layout.lse.byte_size
        else None,
    )


def _validate_mla_dims(kv_lora_rank: int, qk_rope_head_dim: int) -> None:
    kv_lora_rank = _validate_positive_int(kv_lora_rank, "kv_lora_rank")
    qk_rope_head_dim = _validate_positive_int(qk_rope_head_dim, "qk_rope_head_dim")
    if (kv_lora_rank, qk_rope_head_dim) != (_MLA_LATENT_DIM, _MLA_ROPE_DIM):
        raise NotImplementedError(
            "attention-ts MLA decode currently requires "
            f"kv_lora_rank={_MLA_LATENT_DIM} and "
            f"qk_rope_head_dim={_MLA_ROPE_DIM}; got "
            f"{kv_lora_rank} and {qk_rope_head_dim}"
        )


def _validate_mla_max_kv_len(value: int, name: str) -> int:
    """Reserve the largest padded split-KV coordinate span in signed Int32."""
    value = _validate_positive_int(value, name)
    if value > _MLA_MAX_KV_LEN:
        raise NotImplementedError(
            f"{name} must be <= {_MLA_MAX_KV_LEN} so padded MLA K/V "
            "coordinates fit in a signed int32"
        )
    return value


def _validate_mla_int32_extent(value: int, name: str) -> int:
    """Validate a flattened metadata/cache extent used by Int32 coordinates."""
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > _INT32_MAX:
        raise NotImplementedError(f"{name} must fit in a signed int32")
    return value


def _validate_mla_nonnegative_int32_extent(value: int, name: str) -> int:
    """Validate a possibly empty flattened extent used by Int32 coordinates."""
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    if value > _INT32_MAX:
        raise NotImplementedError(f"{name} must fit in a signed int32")
    return value


def _validate_mla_query_head_extent(
    *,
    batch_size: int,
    num_qo_heads: int,
    max_q_len: int,
    total_q: Optional[int] = None,
) -> None:
    """Keep fixed-capacity and packed query-head coordinates in signed Int32."""
    _validate_mla_int32_extent(
        batch_size * max_q_len * num_qo_heads,
        "batch_size * max_q_len * num_qo_heads",
    )
    if total_q is not None:
        _validate_mla_nonnegative_int32_extent(
            total_q * num_qo_heads,
            "total_q * num_qo_heads",
        )


def _validate_mla_policy_coordinate_span(
    policy: tuple[tuple[str, object], ...],
    *,
    max_kv_len: int = 1,
) -> None:
    """Keep the host K/V bound coupled to the automatically selected policy."""
    resolved = dict(policy)
    span = (
        int(cast(int, resolved["tile_size_kv"]))
        * int(cast(int, resolved["num_insts_kv"]))
        * max(int(cast(int, resolved.get("split_capacity", resolved["split_kv"]))), 1)
    )
    if max_kv_len + span - 1 > _INT32_MAX:
        raise NotImplementedError(
            f"MLA padded K/V extent must fit signed Int32: max_kv_len={max_kv_len}, span={span}"
        )


def _validate_mla_dtype_pair(
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> None:
    _dtype_key(q_dtype)
    _dtype_key(kv_dtype)
    _dtype_key(out_dtype)
    if q_dtype != kv_dtype:
        raise NotImplementedError(
            "attention-ts MLA decode requires query and KV cache to use the "
            f"same dtype; got {q_dtype} and {kv_dtype}"
        )
    if q_dtype not in _SUPPORTED_INPUT_DTYPES:
        raise NotImplementedError(
            f"attention-ts MLA decode supports BF16 and FP8-E4M3 input; got {q_dtype}"
        )
    if out_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise NotImplementedError(
            "attention-ts MLA decode currently supports BF16 output only; "
            f"got {out_dtype}"
        )


def _validate_int32_cuda_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    ndim: int,
    require_contiguous: bool = True,
    require_16byte_alignment: bool = True,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be rank {ndim}, got rank {tensor.ndim}")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if require_contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if require_16byte_alignment:
        _validate_16byte_alignment(tensor, name)


def _validate_mla_metadata(
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
) -> tuple[torch.device, int, int]:
    _validate_int32_cuda_tensor(
        block_tables,
        "block_tables",
        ndim=2,
        require_contiguous=False,
        require_16byte_alignment=False,
    )
    _validate_int32_cuda_tensor(
        seq_lens_kv,
        "seq_lens_kv",
        ndim=1,
        require_16byte_alignment=False,
    )
    if block_tables.device != seq_lens_kv.device:
        raise ValueError("block_tables and seq_lens_kv must be on the same device")
    batch_size = int(seq_lens_kv.numel())
    if batch_size <= 0:
        raise ValueError("seq_lens_kv must contain at least one request")
    if block_tables.shape[0] != batch_size:
        raise ValueError(
            "block_tables must have one row per request: expected "
            f"{batch_size}, got {block_tables.shape[0]}"
        )
    max_num_pages = int(block_tables.shape[1])
    if max_num_pages <= 0:
        raise ValueError("block_tables must contain at least one page column")
    if block_tables.stride(1) != 1:
        raise ValueError("block_tables must be contiguous within each row")
    if block_tables.stride(0) < max_num_pages:
        raise ValueError(
            "block_tables rows must not overlap: row stride must be at least "
            f"the column count ({max_num_pages}), got {block_tables.stride(0)}"
        )
    if block_tables.data_ptr() % 4 != 0:
        raise ValueError("block_tables must be 4-byte aligned")
    if seq_lens_kv.data_ptr() % 4 != 0:
        raise ValueError("seq_lens_kv must be 4-byte aligned")
    _validate_mla_int32_extent(batch_size, "batch_size")
    _validate_mla_int32_extent(int(block_tables.numel()), "block_tables elements")
    return seq_lens_kv.device, batch_size, max_num_pages


def _validate_qo_indptr(
    qo_indptr: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> None:
    """Validate the public packed-query metadata without synchronizing."""

    _validate_int32_cuda_tensor(
        qo_indptr,
        "qo_indptr",
        ndim=1,
        require_16byte_alignment=False,
    )
    if qo_indptr.device != device:
        raise ValueError(f"qo_indptr must be on {device}, got {qo_indptr.device}")
    expected_offsets = batch_size + 1
    if qo_indptr.numel() != expected_offsets:
        raise ValueError(
            "qo_indptr must contain batch_size + 1 cumulative offsets: "
            f"expected {expected_offsets}, got {qo_indptr.numel()}"
        )


def has_runtime_query_lengths(kernel) -> bool:
    """Whether the kernel takes its Q length at launch.

    Sparse MLA flattens Q rows into virtual batch items, so its physical Q length
    stays compile-time even when the KV lengths are runtime operands.
    """
    return not kernel.fixed_seqlen and getattr(kernel, "sparse_profile", None) is None


def _derive_max_seq_len_q(
    qo_indptr: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[int, int, tuple[int, ...]]:
    """Validate cumulative offsets and derive their maximum nonnegative delta.

    This helper intentionally synchronizes and is therefore used only by
    explicit runtime validation and the one-shot convenience path.
    """

    offsets = [int(value) for value in qo_indptr.tolist()]
    if len(offsets) != batch_size + 1:
        raise ValueError("qo_indptr must contain batch_size + 1 offsets")
    if offsets[0] != 0:
        raise ValueError("qo_indptr must start at 0")
    q_lengths = tuple(
        end - start for start, end in zip(offsets[:-1], offsets[1:], strict=True)
    )
    if any(length < 0 for length in q_lengths):
        raise ValueError("qo_indptr must be nondecreasing")
    return max(q_lengths), offsets[-1], q_lengths


def _validate_mla_run_metadata(
    runtime: _MLARuntime,
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    qo_indptr: Optional[torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    required_page_columns: int,
    packed_query: bool,
    max_q_len: int,
    max_kv_len: int,
    page_size: int,
    mask_type: str,
) -> int:
    """Validate per-run request metadata against one static MLA plan.

    Value checks synchronize CUDA metadata with the host. ``run(validate=False)``
    is the synchronization-free path for compilation and graph capture.
    """

    metadata_device, metadata_batch_size, max_num_pages = _validate_mla_metadata(
        block_tables, seq_lens_kv
    )
    if metadata_device != device:
        raise ValueError(
            f"MLA metadata must be on the planned device {device}, got {metadata_device}"
        )
    if metadata_batch_size != batch_size:
        raise ValueError(
            "MLA metadata batch size must match the plan "
            f"({batch_size}), got {metadata_batch_size}"
        )
    if max_num_pages < required_page_columns:
        raise ValueError(
            "block_tables must have at least ceil(max_kv_len / page_size) "
            f"columns ({required_page_columns}), got {max_num_pages}"
        )

    if packed_query:
        if qo_indptr is None:
            raise ValueError("packed-query MLA run requires qo_indptr")
        _validate_qo_indptr(
            qo_indptr,
            device=device,
            batch_size=batch_size,
        )
        runtime_max_seq_len_q, total_q, q_lengths = _derive_max_seq_len_q(
            qo_indptr,
            batch_size=batch_size,
        )
        if total_q != int(runtime.query.shape[0]):
            raise ValueError(
                "qo_indptr must end at the packed query row count "
                f"({runtime.query.shape[0]}), got {total_q}"
            )
        if runtime_max_seq_len_q > max_q_len:
            raise ValueError(
                "qo_indptr contains a per-request Q length larger than "
                f"max_q_len ({max_q_len}): got "
                f"{runtime_max_seq_len_q}"
            )
    else:
        if qo_indptr is not None:
            raise ValueError("fixed-query MLA plan does not accept qo_indptr")
        runtime_max_seq_len_q = int(runtime.query.shape[1])
        q_lengths = (runtime_max_seq_len_q,) * batch_size

    seq_lens_host = tuple(int(value) for value in seq_lens_kv.tolist())
    runtime_max_kv_len = max(seq_lens_host)
    if runtime_max_kv_len > max_kv_len:
        raise ValueError(
            "runtime KV metadata contains a request longer than "
            f"max_kv_len ({max_kv_len}): got {runtime_max_kv_len}"
        )
    block_table_rows = block_tables.tolist()
    for request_idx, (row, seq_len) in enumerate(
        zip(block_table_rows, seq_lens_host, strict=True)
    ):
        required_pages = _ceil_div(max(seq_len, 0), page_size)
        if any(
            int(page_id) < 0 or int(page_id) >= runtime.num_physical_pages
            for page_id in row[:required_pages]
        ):
            raise ValueError(
                "block_tables values for active pages must index the physical "
                f"K/V cache in [0, {runtime.num_physical_pages}); request "
                f"{request_idx} contains an invalid page ID"
            )
    if mask_type == "causal":
        for request_idx, (q_len, kv_len) in enumerate(
            zip(q_lengths, seq_lens_host, strict=True)
        ):
            if kv_len > 0 and q_len > kv_len:
                raise ValueError(
                    "causal MLA decode requires every per-request Q length "
                    "to be no greater than its K/V length; request "
                    f"{request_idx} has Q={q_len} and K/V={kv_len}"
                )
    return runtime_max_seq_len_q


def _resolve_max_seq_len_q_alias(
    *,
    seq_len_q: Optional[int],
    max_q_len: Optional[int],
    default: Optional[int],
) -> Optional[int]:
    """Resolve the legacy fixed-Q name and the explicit static-bound name."""

    legacy_bound = (
        _validate_positive_int(seq_len_q, "seq_len_q")
        if seq_len_q is not None
        else None
    )
    explicit_bound = (
        _validate_positive_int(max_q_len, "max_q_len")
        if max_q_len is not None
        else None
    )
    if (
        legacy_bound is not None
        and explicit_bound is not None
        and legacy_bound != explicit_bound
    ):
        raise ValueError(
            "seq_len_q and max_q_len must agree when both are provided: "
            f"got {legacy_bound} and {explicit_bound}"
        )
    if explicit_bound is not None:
        return explicit_bound
    if legacy_bound is not None:
        return legacy_bound
    return default


# Validate compact fixed or packed CUDA Q storage against optional plan bounds.
def _validate_query(
    query: torch.Tensor,
    *,
    packed_query: bool = False,
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
    num_qo_heads: Optional[int] = None,
    max_q_len: Optional[int] = None,
    q_dtype: Optional[torch.dtype] = None,
    query_dim: int = _MLA_QUERY_DIM,
) -> None:
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor")
    expected_rank = 3 if packed_query else 4
    if query.ndim != expected_rank:
        expected_shape = (
            f"[total_q, H, {query_dim}]" if packed_query else f"[B, SQ, H, {query_dim}]"
        )
        raise ValueError(f"query must have shape {expected_shape}")
    if packed_query:
        if int(query.shape[1]) <= 0:
            raise ValueError("query head extent must be positive")
    elif any(int(extent) <= 0 for extent in query.shape[:-1]):
        raise ValueError("query row and head extents must be positive")
    if query.shape[-1] != query_dim:
        raise ValueError(
            f"query last dimension must be {query_dim}, got {query.shape[-1]}"
        )
    if query.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise NotImplementedError(
            f"unsupported attention-ts MLA query dtype {query.dtype}"
        )
    if query.device.type != "cuda":
        raise ValueError("query must be a CUDA tensor")
    if device is not None and query.device != device:
        raise ValueError(
            f"query must be on the planned device {device}, got {query.device}"
        )
    if not packed_query and batch_size is not None and query.shape[0] != batch_size:
        raise ValueError(
            f"query batch size must match the plan ({batch_size}), got {query.shape[0]}"
        )
    head_axis = 1 if packed_query else 2
    if num_qo_heads is not None and query.shape[head_axis] != num_qo_heads:
        raise ValueError(
            "query head count must match the plan "
            f"({num_qo_heads}), got {query.shape[head_axis]}"
        )
    if max_q_len is not None:
        if packed_query:
            if batch_size is None:
                raise ValueError("batch_size is required to validate packed query")
            total_q = int(query.shape[0])
            if total_q > batch_size * max_q_len:
                raise ValueError(
                    "packed query total rows must be within "
                    f"[0, {batch_size * max_q_len}], got {total_q}"
                )
        elif query.shape[1] != max_q_len:
            raise ValueError(
                "fixed query length must equal the planned max_q_len "
                f"({max_q_len}), got {query.shape[1]}"
            )
        if batch_size is not None and num_qo_heads is not None:
            _validate_mla_query_head_extent(
                batch_size=batch_size,
                num_qo_heads=num_qo_heads,
                max_q_len=max_q_len,
                total_q=int(query.shape[0]) if packed_query else None,
            )
    if q_dtype is not None and query.dtype != q_dtype:
        raise ValueError(
            f"query dtype must match the plan ({q_dtype}), got {query.dtype}"
        )
    if not query.is_contiguous():
        layout = (
            f"[total_q, H, {query_dim}]" if packed_query else f"[B, SQ, H, {query_dim}]"
        )
        raise ValueError(f"query must be compact in {layout} layout")
    _validate_16byte_alignment(query, "query")


def _normalize_mla_kv_cache(
    kv_cache: torch.Tensor,
    *,
    expected_device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    if not isinstance(kv_cache, torch.Tensor):
        raise TypeError("kv_cache must be a torch.Tensor")
    if kv_cache.ndim == 4:
        if kv_cache.shape[1] != 1:
            raise ValueError(
                "rank-4 kv_cache must have shape [num_pages, 1, page_size, 576]"
            )
        if not kv_cache.is_contiguous():
            raise ValueError("rank-4 kv_cache must be compact")
        normalized = kv_cache[:, 0]
    elif kv_cache.ndim == 3:
        if not kv_cache.is_contiguous():
            raise ValueError("rank-3 kv_cache must be compact")
        normalized = kv_cache
    else:
        raise ValueError(
            "kv_cache must have shape [num_pages, page_size, 576] or "
            "[num_pages, 1, page_size, 576]"
        )
    if normalized.device != expected_device:
        raise ValueError(
            f"kv_cache must be on the planned device {expected_device}, "
            f"got {normalized.device}"
        )
    if normalized.shape[0] <= 0 or normalized.shape[1] <= 0:
        raise ValueError("kv_cache page count and page size must be positive")
    _validate_mla_int32_extent(int(normalized.shape[0]), "kv_cache physical pages")
    if normalized.shape[2] != _MLA_QUERY_DIM:
        raise ValueError(
            f"kv_cache last dimension must be {_MLA_QUERY_DIM}, "
            f"got {normalized.shape[2]}"
        )
    _validate_16byte_alignment(normalized, "kv_cache")
    return normalized, int(normalized.shape[0]), int(normalized.shape[1])


# Validate caller-owned output shape, dtype, device and 16-byte alignment.
def _validate_out(
    out: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    num_qo_heads: int,
    max_q_len: int,
    packed_query: bool,
    total_q: Optional[int] = None,
    out_dtype: torch.dtype,
) -> None:
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if packed_query:
        if total_q is None:
            raise ValueError("total_q is required to validate packed output")
        expected_shape: tuple[int, ...]
        expected_shape = (total_q, num_qo_heads, _MLA_LATENT_DIM)
    else:
        expected_shape = (batch_size, max_q_len, num_qo_heads, _MLA_LATENT_DIM)
    if out.shape != expected_shape:
        raise ValueError(
            f"out must have shape {expected_shape}, got {tuple(out.shape)}"
        )
    if out.dtype != out_dtype:
        raise ValueError(f"out must have dtype {out_dtype}, got {out.dtype}")
    if out.device != device:
        raise ValueError(f"out must be on {device}, got {out.device}")
    if not out.is_contiguous():
        layout = "[total_q, H, 512]" if packed_query else "[B, SQ, H, 512]"
        raise ValueError(f"out must be compact in {layout} layout")
    _validate_16byte_alignment(out, "out")


def _check_softmax_rescale_threshold_log2_range(
    threshold: object, *, q_dtype: torch.dtype
) -> float:
    """Return the threshold as float32-rounded ``float``; raise if out of range.

    Values outside ``[0, bound]`` (E4M3: 8, BF16: 32) raise instead of being
    clamped, as trtllm-gen's ``checkSkipCorrThreshold`` does.
    """

    from .kernels.mla_decode.helpers.math import (
        canonicalize_softmax_rescale_threshold_log2,
    )

    if threshold == 0.0 and type(threshold) in (int, float):
        # Zero selects the default kernel for every dtype, including ones
        # without a threshold bound.
        return 0.0
    return canonicalize_softmax_rescale_threshold_log2(
        threshold, qkv_dtype=_kernel_dtype_name(_dtype_key(q_dtype))
    )


def _kernel_dtype_name(dtype_key: str) -> str:
    names = {
        "bfloat16": "bf16",
        "float8_e4m3fn": "e4m3",
    }
    try:
        return names[dtype_key]
    except KeyError as error:
        raise NotImplementedError(
            f"unsupported attention-ts MLA dtype key {dtype_key!r}"
        ) from error


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _separate_reducer_provenance(
    kernel,
    *,
    split_kv: int,
    use_cluster_reduction: bool,
) -> tuple[str, Optional[int]]:
    """Describe the derived standalone reducer without exposing a knob."""

    if split_kv <= 1 or use_cluster_reduction:
        return "none", None
    if bool(getattr(kernel, "use_parallel_reduction", False)):
        topology = getattr(kernel, "parallel_reduction_topology", None)
        if topology is None:
            raise RuntimeError("parallel MLA reducer is missing its topology")
        return "parallel", int(topology.cluster_size)
    return "reference", 1


# CPU policies are small and tensor-free. Cache them across layers and decode
# steps; native compilation remains cached independently by its static signature.
@functools.cache
def _resolve_mla_decode_launch_spec(
    device_index: int,
    batch_size: int,
    num_qo_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    max_kv_len: int,
    q_dtype_key: str,
    kv_dtype_key: str,
    out_dtype_key: str,
    mask_type: str,
    seq_len_q: int = 1,
    softmax_rescale_threshold_log2: float = 0.0,
):
    """Resolve and cache MLA policy/workspace without compiling."""

    policy: tuple[tuple[str, Any], ...]
    max_kv_len = _validate_mla_max_kv_len(max_kv_len, "max_kv_len")

    import cutlass

    from .kernels.mla_decode.kernel_policy import (
        resolve_mla_kernel_policy,
        select_mla_ts_kernel,
    )
    from .kernels.mla_decode.helpers.query import FlatQueryTileLayout
    from .kernels.mla_decode.throughput_2cta.config import (
        compute_split_kv,
        compute_workspace_size as compute_2cta_workspace_size,
    )
    from .kernels.mla_decode.throughput_2cta.kernel import MlaDecodeTs
    from .kernels.mla_decode.throughput_latency_1cta.config import (
        compute_workspace_size as compute_1cta_workspace_size,
        q_tile_work_count,
        resolve_auto_flat_query_launch_shape,
        resolve_runtime_cluster_reduction_mode,
        resolve_auto_cluster_reduction_config,
        select_auto_split_kv,
    )
    from .kernels.mla_decode.throughput_latency_1cta.kernel import (
        ThroughputLatencyMlaDecodeTs,
    )

    if q_dtype_key != kv_dtype_key:
        raise ValueError("the cached TS MLA compiler requires one QKV dtype")
    seq_len_q = _validate_positive_int(seq_len_q, "seq_len_q")
    _validate_mla_query_head_extent(
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        max_q_len=seq_len_q,
    )
    _validate_mla_dims(kv_lora_rank, qk_rope_head_dim)
    qkv_dtype_name = _kernel_dtype_name(q_dtype_key)
    output_dtype_name = _kernel_dtype_name(out_dtype_key)

    with torch.cuda.device(device_index):
        max_active_one_cta_clusters = get_max_active_clusters(1)
        max_active_two_cta_clusters = get_max_active_clusters(2)
        one_cta_launch_shape = resolve_auto_flat_query_launch_shape(
            num_heads_q=num_qo_heads,
            seq_len_q=seq_len_q,
        )
        one_cta_base_work = q_tile_work_count(
            batch_size,
            one_cta_launch_shape.num_heads_q,
            one_cta_launch_shape.seq_len_q,
            one_cta_launch_shape.tile_size_q,
        )
        one_cta_split_kv = select_auto_split_kv(
            seq_len_kv=max_kv_len,
            tile_size_q=one_cta_launch_shape.tile_size_q,
            base_work=one_cta_base_work,
            target_work=max_active_one_cta_clusters,
        )
        two_cta_launch_shape = FlatQueryTileLayout.for_tile(
            num_qo_heads, seq_len_q, 128
        )
        two_cta_split_kv = compute_split_kv(
            batch_size=batch_size,
            num_q_tiles=two_cta_launch_shape.num_tiles,
            seq_len_kv=max_kv_len,
            mma_qk_tiler_mn=(128, 128),
            max_active_blocks=max_active_two_cta_clusters * 2,
        )
        sm_count = get_device_sm_count(torch.device("cuda", device_index))
        one_cta_split_kv = min(one_cta_split_kv, sm_count)
        two_cta_split_kv = min(two_cta_split_kv, sm_count)
        requested_policy, policy_source = resolve_mla_kernel_policy(
            None,
            num_qo_heads,
            seq_len_q,
            one_cta_split_kv=one_cta_split_kv,
            two_cta_split_kv=two_cta_split_kv,
        )
        use_throughput_latency = requested_policy == "throughput_latency_1cta"

        kernel: Any
        if use_throughput_latency:
            max_active_clusters = max_active_one_cta_clusters
            launch_shape = one_cta_launch_shape
            decision = select_mla_ts_kernel(
                requested_policy=requested_policy,
                batch_size=batch_size,
                num_heads=launch_shape.num_heads_q,
                seq_len_q=launch_shape.seq_len_q,
                seq_len_k=max_kv_len,
                latent_dim=kv_lora_rank,
                rope_dim=qk_rope_head_dim,
                page_size=page_size,
                dtype=qkv_dtype_name,
                out_dtype=output_dtype_name,
                throughput_latency_profile=None,
                throughput_latency_tile_size_q=launch_shape.tile_size_q,
                max_active_clusters=max_active_clusters,
                throughput_latency_split_kv=None,
                throughput_latency_persistent=None,
            )
            if not decision.implementation_ready or decision.config is None:
                if policy_source != "auto":
                    raise NotImplementedError(decision.reason)
                requested_policy = "throughput_2cta"
                use_throughput_latency = False

        if use_throughput_latency:
            assert decision.config is not None
            # Cluster fanout is physical topology, not a rounded scratch bound.
            prepared_cfg = resolve_auto_cluster_reduction_config(
                decision.config,
                reduction_mode=None,
            )
            reduction_mode = (
                "gmem_separate"
                if decision.config.use_cluster_reduction
                and not prepared_cfg.use_cluster_reduction
                else resolve_runtime_cluster_reduction_mode(
                    prepared_cfg, reduction_mode=None
                )
            )
            capacity = (
                one_cta_split_kv
                if prepared_cfg.use_cluster_reduction or one_cta_split_kv == 1
                else sm_count
            )
            kernel = ThroughputLatencyMlaDecodeTs(
                batch_size=batch_size,
                num_heads=launch_shape.num_heads_q,
                seq_len_q=launch_shape.seq_len_q,
                seq_len_k=max_kv_len,
                latent_dim=kv_lora_rank,
                rope_dim=qk_rope_head_dim,
                page_size=page_size,
                max_active_clusters=max_active_clusters,
                acc_dtype=cutlass.Float32,
                lse_dtype=cutlass.Float32,
                qkv_dtype=qkv_dtype_name,
                out_dtype=output_dtype_name,
                profile=decision.profile_name,
                reduction_mode=reduction_mode,
                logical_num_heads=num_qo_heads,
                logical_seq_len_q=seq_len_q,
                tile_size_q=launch_shape.tile_size_q,
                explicit_split_kv=None,
                split_kv_capacity=capacity,
                reduction_policy_splits=one_cta_split_kv,
                explicit_persistent=None,
                mask_type=mask_type,
                softmax_rescale_threshold_log2=softmax_rescale_threshold_log2,
            )
            kernel.runtime_split_kv = True
            final_cfg = kernel._make_config()
            split_kv = one_cta_split_kv
            workspace_size = compute_1cta_workspace_size(
                cfg=final_cfg,
                partial_o_dtype=cutlass.BFloat16,
                lse_dtype=cutlass.Float32,
            )
            separate_reducer_impl, reducer_cluster_size = _separate_reducer_provenance(
                kernel,
                split_kv=split_kv,
                use_cluster_reduction=bool(final_cfg.use_cluster_reduction),
            )
            policy = (
                ("kernel", decision.selected_kernel),
                ("source", policy_source),
                ("profile", decision.profile_name),
                ("tile_size_q", int(final_cfg.tile_size_q)),
                ("tile_size_kv", int(final_cfg.tile_size_kv)),
                ("num_insts_kv", int(final_cfg.num_insts_kv)),
                ("split_kv", split_kv),
                ("num_ctas_per_head_dim", int(final_cfg.num_ctas_per_head_dim)),
                ("head_dim_per_cta_v", int(final_cfg.head_dim_per_cta_v)),
                ("use_cluster_reduction", bool(final_cfg.use_cluster_reduction)),
                (
                    "use_persistent_scheduler",
                    bool(final_cfg.use_persistent_scheduler),
                ),
                (
                    "use_clc_dynamic_persistent_scheduler",
                    bool(final_cfg.use_clc_dynamic_persistent_scheduler),
                ),
                ("separate_reducer_impl", separate_reducer_impl),
                ("reducer_cluster_size", reducer_cluster_size),
            )
        else:
            max_active_clusters = max_active_two_cta_clusters
            launch_shape = two_cta_launch_shape
            decision = select_mla_ts_kernel(
                requested_policy=requested_policy,
                batch_size=batch_size,
                num_heads=num_qo_heads,
                seq_len_q=seq_len_q,
                seq_len_k=max_kv_len,
                latent_dim=kv_lora_rank,
                rope_dim=qk_rope_head_dim,
                page_size=page_size,
                dtype=qkv_dtype_name,
                out_dtype=output_dtype_name,
                throughput_latency_profile=None,
                throughput_latency_tile_size_q=None,
                max_active_clusters=max_active_clusters,
                throughput_latency_split_kv=None,
                throughput_latency_persistent=None,
            )
            if not decision.implementation_ready:
                raise NotImplementedError(decision.reason)
            split_kv = two_cta_split_kv
            capacity = 1 if split_kv == 1 else sm_count
            work_clusters = batch_size * launch_shape.num_tiles * max(split_kv, 1)
            # Dynamic cluster stealing only helps once logical work exceeds a
            # resident wave.  Within one wave every cluster already launches,
            # so the CLC producer/response pipeline is pure overhead.
            is_persistent = work_clusters > max_active_clusters
            kernel = MlaDecodeTs(
                acc_dtype=cutlass.Float32,
                lse_dtype=cutlass.Float32,
                mma_qk_tiler_mn=(128, 128),
                mma_pv_tiler_mn=(128, 256),
                max_active_clusters=max_active_clusters,
                page_size=page_size,
                is_persistent=is_persistent,
                is_var_seq=False,
                is_var_split_kv=False,
                static_split_kv=1 if capacity == 1 else None,
                split_kv_capacity=capacity,
                static_seq_len_k=None,
                qkv_dtype=qkv_dtype_name,
                out_dtype=output_dtype_name,
                rope_dim=qk_rope_head_dim,
                num_heads=num_qo_heads,
                seq_len_q=seq_len_q,
                batch_size=batch_size,
                mask_type=mask_type,
                softmax_rescale_threshold_log2=softmax_rescale_threshold_log2,
            )
            kernel._configure_parallel_reduction_topology(split_kv)
            workspace_size = compute_2cta_workspace_size(
                tile_size_q=int(launch_shape.tile_size_q),
                num_q_tiles=int(launch_shape.num_tiles),
                latent_dim=kv_lora_rank,
                batch_size=batch_size,
                split_kv=capacity,
                partial_o_dtype=cutlass.BFloat16,
                lse_dtype=cutlass.Float32,
            )
            separate_reducer_impl, reducer_cluster_size = _separate_reducer_provenance(
                kernel,
                split_kv=split_kv,
                use_cluster_reduction=False,
            )
            policy = (
                ("kernel", decision.selected_kernel),
                ("source", policy_source),
                ("profile", None),
                ("tile_size_q", 128),
                ("tile_size_kv", 128),
                ("num_insts_kv", 1),
                ("split_kv", int(split_kv)),
                ("num_ctas_per_head_dim", 2),
                ("head_dim_per_cta_v", 256),
                ("use_cluster_reduction", False),
                ("use_persistent_scheduler", bool(is_persistent)),
                (
                    "use_clc_dynamic_persistent_scheduler",
                    bool(is_persistent and qkv_dtype_name == "bf16"),
                ),
                ("separate_reducer_impl", separate_reducer_impl),
                ("reducer_cluster_size", reducer_cluster_size),
            )
        policy = (
            *policy,
            (
                "split_capacity",
                capacity,
            ),
            ("max_split_kv", sm_count),
        )
        _validate_mla_policy_coordinate_span(policy, max_kv_len=max_kv_len)

    return _MLADecodeLaunchSpec(
        kernel=kernel,
        policy=policy,
        kernel_workspace_bytes=int(workspace_size),
        split_kv=int(split_kv),
    )


def _mla_kernel_compile_signature(kernel: Any) -> tuple[object, ...]:
    """Return all static kernel state except the batch extent."""

    make_signature = getattr(kernel, "compile_signature", None)
    if not callable(make_signature):
        raise TypeError("MLA kernels must define compile_signature()")
    signature = (
        type(kernel).__module__,
        type(kernel).__qualname__,
        make_signature(),
    )
    try:
        hash(signature)
    except TypeError as error:
        raise TypeError("MLA kernel compile state must be hashable") from error
    return signature


def _make_mla_decode_compile_spec(
    launch_spec: _MLADecodeLaunchSpec,
    *,
    device_index: int,
    num_qo_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    q_dtype_key: str,
    out_dtype_key: str,
    max_q_len: int,
    packed_query: bool,
    lse_mode: str = "none",
    runtime_kv_lengths: bool = False,
    write_final_lse: bool = True,
) -> _MLADecodeCompileSpec:
    """Keep policy resolution plan-specific and JIT identity batch-free."""

    kernel = copy.copy(launch_spec.kernel)
    kernel.lse_base_on_e = False
    kernel.runtime_kv_lengths = runtime_kv_lengths
    return _MLADecodeCompileSpec(
        device_index=device_index,
        kernel_signature=_mla_kernel_compile_signature(kernel),
        num_qo_heads=num_qo_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=page_size,
        q_dtype_key=q_dtype_key,
        out_dtype_key=out_dtype_key,
        max_q_len=(
            None if has_runtime_query_lengths(launch_spec.kernel) else max_q_len
        ),
        packed_query=packed_query,
        has_kernel_workspace=launch_spec.kernel_workspace_bytes > 0,
        split_kv=launch_spec.split_kv,
        kernel=kernel,
        lse_mode=lse_mode,
        write_final_lse=write_final_lse,
    )


# Dense launch specs are cached; sparse plans construct transient kernel objects.
# Reuse dense signature construction without retaining each transient sparse plan.
_cached_mla_decode_compile_spec = functools.cache(_make_mla_decode_compile_spec)


_COMPILED_MLA_DECODE: dict = {}


def _get_compiled_mla_decode(compile_spec: _MLADecodeCompileSpec):
    """Compile and cache one batch-dynamic TS MLA topology.

    A manual memo rather than ``functools.cache``: under the warm-up precompile
    modes of ``_compile_cache`` the build returns a placeholder (probe / persist
    only), which must not be remembered for the serving launches that follow.
    """
    compiled = _COMPILED_MLA_DECODE.get(compile_spec)
    if compiled is None:
        compiled = _compile_mla_decode(compile_spec)
        if not _compile_cache_is_placeholder(compiled):
            _COMPILED_MLA_DECODE[compile_spec] = compiled
    return compiled


_get_compiled_mla_decode.cache_clear = _COMPILED_MLA_DECODE.clear  # type: ignore[attr-defined]


def _compile_cache_is_placeholder(compiled) -> bool:
    from ._compile_cache import is_placeholder

    return is_placeholder(compiled)


def _compile_mla_decode(
    compile_spec: _MLADecodeCompileSpec,
):
    """Compile and cache one batch-dynamic TS MLA topology."""

    import cutlass
    import cutlass.cute as cute

    device_index = compile_spec.device_index
    num_qo_heads = compile_spec.num_qo_heads
    kv_lora_rank = compile_spec.kv_lora_rank
    qk_rope_head_dim = compile_spec.qk_rope_head_dim
    query_dim = kv_lora_rank + qk_rope_head_dim
    # The D512 path has no separate RoPE stage. Use an unused full-width
    # alias in the callable ABI rather than a zero-extent tensor map.
    rope_view_dim = qk_rope_head_dim or kv_lora_rank
    page_size = compile_spec.page_size
    max_q_len = (
        cute.sym_int() if compile_spec.max_q_len is None else compile_spec.max_q_len
    )
    packed_query = compile_spec.packed_query
    # Keep the cached spec pristine if compiled-callable caches are cleared.
    kernel = copy.copy(compile_spec.kernel)
    sparse = getattr(kernel, "direct_sparse", False)
    nvfp4_kv = getattr(kernel, "has_nvfp4_kv", False)
    dtype_map = {
        "bfloat16": cutlass.BFloat16,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
    }
    qkv_dtype = dtype_map[compile_spec.q_dtype_key]
    out_dtype = dtype_map[compile_spec.out_dtype_key]
    physical_pages = cute.sym_int()
    batch_size = cute.sym_int()
    runtime_total_q = cute.sym_int()

    # These fake tensors pin the public ABI while allowing runtime page counts,
    # table widths/row strides, and batch metadata pointers to vary.
    q_stride_h = query_dim
    q_stride_q = num_qo_heads * query_dim
    q_latent_shape: tuple[int, ...]
    q_rope_shape: tuple[int, ...]
    q_stride: tuple[int, ...]
    if packed_query:
        q_latent_shape = (num_qo_heads, kv_lora_rank, runtime_total_q)
        q_rope_shape = (num_qo_heads, rope_view_dim, runtime_total_q)
        q_stride = (q_stride_h, 1, q_stride_q)
    else:
        q_stride_batch = max_q_len * q_stride_q
        q_latent_shape = (num_qo_heads, kv_lora_rank, max_q_len, batch_size)
        q_rope_shape = (
            num_qo_heads,
            rope_view_dim,
            max_q_len,
            batch_size,
        )
        q_stride = (q_stride_h, 1, q_stride_q, q_stride_batch)
    q_latent_fake = cute.runtime.make_fake_tensor(
        qkv_dtype, q_latent_shape, stride=q_stride, assumed_align=16
    )
    q_rope_fake = cute.runtime.make_fake_tensor(
        qkv_dtype, q_rope_shape, stride=q_stride, assumed_align=16
    )
    cache_dim = (
        cache_record_size(kernel.kv_cache_format, qk_rope_head_dim)
        if nvfp4_kv
        else query_dim
    )
    extra_cache_dim = (
        cache_record_size(kernel.extra_kv_cache_format, qk_rope_head_dim)
        if nvfp4_kv
        else query_dim
    )
    cache_dtype = cutlass.Uint8 if nvfp4_kv else qkv_dtype
    cache_token_stride = cache_dim
    cache_page_stride = page_size * cache_dim
    c_latent_fake = cute.runtime.make_fake_tensor(
        cache_dtype,
        (page_size, cache_dim if nvfp4_kv else kv_lora_rank, physical_pages),
        stride=(cache_token_stride, 1, cache_page_stride),
        assumed_align=16,
    )
    c_rope_fake = cute.runtime.make_fake_tensor(
        cache_dtype,
        (
            page_size,
            extra_cache_dim if sparse else rope_view_dim,
            cute.sym_int() if page_size == 1 else physical_pages,
        ),
        stride=(extra_cache_dim, 1, page_size * extra_cache_dim)
        if sparse
        else (cache_token_stride, 1, cache_page_stride),
        assumed_align=16,
    )
    runtime_page_columns = cute.sym_int()
    runtime_page_row_stride = cute.sym_int64(divisibility=1)
    page_offsets_fake = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (runtime_page_columns, batch_size),
        stride=(1, runtime_page_row_stride),
        assumed_align=4,
    )
    out_stride_row = num_qo_heads * kv_lora_rank
    out_shape: tuple[int, ...]
    out_stride: tuple[int, ...]
    lse_shape: tuple[int, ...]
    lse_stride: tuple[int, ...]
    if packed_query:
        out_shape = (num_qo_heads, kv_lora_rank, runtime_total_q)
        out_stride = (kv_lora_rank, 1, out_stride_row)
        lse_shape = (num_qo_heads, runtime_total_q)
        lse_stride = (1, num_qo_heads)
    else:
        out_stride_batch = max_q_len * out_stride_row
        out_shape = (num_qo_heads, kv_lora_rank, max_q_len, batch_size)
        out_stride = (kv_lora_rank, 1, out_stride_row, out_stride_batch)
        lse_shape = (num_qo_heads, max_q_len, batch_size)
        lse_stride = (1, num_qo_heads, max_q_len * num_qo_heads)
    out_fake = cute.runtime.make_fake_tensor(
        out_dtype, out_shape, stride=out_stride, assumed_align=16
    )
    lse_fake = (
        cute.runtime.make_fake_tensor(
            cutlass.Float32, lse_shape, stride=lse_stride, assumed_align=4
        )
        if compile_spec.write_final_lse
        else None
    )
    workspace_fake = None
    if compile_spec.has_kernel_workspace:
        workspace_bytes = cute.sym_int()
        workspace_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int8,
            (workspace_bytes,),
            stride_order=(0,),
            assumed_align=32,
        )
    cache_seqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (batch_size,),
        stride_order=(0,),
        assumed_align=4,
    )
    qo_indptr_fake = None
    if packed_query:
        runtime_num_q_offsets = cute.sym_int()
        qo_indptr_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (runtime_num_q_offsets,),
            stride_order=(0,),
            assumed_align=4,
        )
    sparse_inputs_fake = None
    if getattr(kernel, "direct_sparse", False):
        # Describe runtime index-list extents; the plan capacity does not specialize
        # this view.
        def sparse_indices(capacity):
            return cute.runtime.make_fake_tensor(
                cutlass.Int32,
                (batch_size, cute.sym_int()),
                stride=(cute.sym_int64(), 1),
                assumed_align=4,
            )

        sink = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (num_qo_heads,), stride_order=(0,), assumed_align=4
        )
        sparse_inputs_fake = (
            sparse_indices(kernel.direct_sparse_capacities[0]),
            sparse_indices(kernel.direct_sparse_capacities[1]),
            cache_seqs_fake,
            cache_seqs_fake,
            sink,
            cutlass.Float32(1.0),
        )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    output_quant_args = _fake_output_quant_args(getattr(kernel, "output_quant", None))

    # Task objects carry loop-local state through generated control flow, so
    # select the public staged frontend for this compilation.
    def compile_kernel():
        """Compile MLA with opt-in task-builder advisory diagnostics."""
        from .kernels.mla_decode.helpers.diagnostics import mla_build_diagnostics

        with mla_build_diagnostics():
            return cute.compile[cute.FrontendNext](
                kernel,
                q_latent_fake,
                q_rope_fake,
                c_latent_fake,
                c_rope_fake,
                page_offsets_fake,
                out_fake,
                lse_fake,
                workspace_fake,
                cutlass.Int32(compile_spec.split_kv),
                cache_seqs_fake,
                qo_indptr_fake,
                sparse_inputs_fake,
                cutlass.Float32(1.0),
                cutlass.Float32(1.0),
                stream_fake,
                output_quant_args=output_quant_args,
                max_q_len=cutlass.Int32(0),
                options=_COMPILE_OPTIONS,
            )

    with torch.cuda.device(device_index):
        compiled = build_and_load_attention_kernel(
            "mla_decode",
            "main",
            compile_spec,
            _COMPILE_OPTIONS,
            compile_kernel,
            extra=(),
        )
    return compiled


def get_prims_ts_batch_mla_decode_workspace_size(
    batch_size: int,
    num_qo_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    max_kv_len: int,
    *,
    seq_len_q: Optional[int] = None,
    max_q_len: Optional[int] = None,
    q_dtype: torch.dtype = torch.bfloat16,
    kv_dtype: Optional[torch.dtype] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    mask_type: Literal["dense", "causal"] = "causal",
    device=None,
) -> int:
    """Return caller-workspace bytes for one automatic MLA policy.

    The arguments define the static geometry used to resolve the same automatic
    policy and private scratch layout as
    :func:`batch_mla_decode_with_paged_kv_cache`, without compiling a kernel.
    ``max_q_len`` is the static per-request Q bound for both fixed and
    packed-query launches;
    ``seq_len_q`` remains a backward-compatible fixed-Q alias. If neither is
    supplied, the bound is one. The returned byte count covers split-KV
    scratch. Optional final LSE uses a separate caller-owned output. Allocate
    a contiguous ``torch.int8`` or ``torch.uint8`` CUDA buffer; MLA does not require its
    contents to be initialized before first use.
    """

    batch_size = _validate_positive_int(batch_size, "batch_size")
    num_qo_heads = _validate_positive_int(num_qo_heads, "num_qo_heads")
    _validate_mla_dims(kv_lora_rank, qk_rope_head_dim)
    page_size = _validate_page_size(page_size)
    max_kv_len = _validate_mla_max_kv_len(max_kv_len, "max_kv_len")
    max_q_len = _resolve_max_seq_len_q_alias(
        seq_len_q=seq_len_q,
        max_q_len=max_q_len,
        default=1,
    )
    assert max_q_len is not None
    _validate_mla_query_head_extent(
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        max_q_len=max_q_len,
    )
    _validate_mask(mask_type)
    if kv_dtype is None:
        kv_dtype = q_dtype
    _validate_mla_dtype_pair(q_dtype, kv_dtype, out_dtype)
    resolved_device, device_index = _resolve_cuda_device(device)
    _validate_runtime_device(resolved_device)

    spec = _resolve_mla_decode_launch_spec(
        device_index,
        batch_size,
        num_qo_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        page_size,
        max_kv_len,
        _dtype_key(q_dtype),
        _dtype_key(kv_dtype),
        _dtype_key(out_dtype),
        mask_type,
        max_q_len,
        # The skip-correction threshold does not change the workspace; pass the default
        # positionally so a default plan() hits the same cached spec.
        0.0,
    )
    return _make_mla_workspace_layout(
        spec.kernel_workspace_bytes,
        batch_size,
        num_qo_heads,
        max_q_len,
        include_lse=False,
    ).total_bytes


def _prepare_mla_runtime(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    num_qo_heads: int,
    max_q_len: int,
    packed_query: bool,
    qo_indptr: Optional[torch.Tensor],
    page_size: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    out_dtype: torch.dtype,
    softmax_scale: float,
    output_scale: float,
    out: Optional[torch.Tensor],
    validate: bool,
) -> _MLARuntime:
    """Normalize one launch, optionally validating its public arguments."""

    if validate:
        if packed_query:
            if qo_indptr is None:
                raise ValueError("packed-query MLA run requires qo_indptr")
            _validate_qo_indptr(
                qo_indptr,
                device=device,
                batch_size=batch_size,
            )
        elif qo_indptr is not None:
            raise ValueError("fixed-query MLA plan does not accept qo_indptr")
        _validate_query(
            query,
            packed_query=packed_query,
            device=device,
            batch_size=batch_size,
            num_qo_heads=num_qo_heads,
            max_q_len=max_q_len,
            q_dtype=q_dtype,
        )
        (
            normalized_cache,
            num_physical_pages,
            runtime_page_size,
        ) = _normalize_mla_kv_cache(kv_cache, expected_device=device)
        if runtime_page_size != page_size:
            raise ValueError(
                "kv_cache page size does not match the launch: expected "
                f"{page_size}, got {runtime_page_size}"
            )
        if normalized_cache.dtype != kv_dtype:
            raise ValueError(
                f"kv_cache dtype must match the launch ({kv_dtype}), "
                f"got {normalized_cache.dtype}"
            )
        effective_softmax_scale = _validate_scale(softmax_scale, "softmax_scale")
        effective_output_scale = _validate_scale(output_scale, "output_scale")
    else:
        normalized_cache = kv_cache[:, 0] if kv_cache.ndim == 4 else kv_cache
        num_physical_pages = int(normalized_cache.shape[0])
        effective_softmax_scale = softmax_scale
        effective_output_scale = output_scale
    total_q = int(query.shape[0]) if packed_query else None
    if out is None:
        out_shape = (
            (total_q, num_qo_heads, _MLA_LATENT_DIM)
            if packed_query
            else (batch_size, max_q_len, num_qo_heads, _MLA_LATENT_DIM)
        )
        out = torch.empty(out_shape, device=device, dtype=out_dtype)
    elif validate:
        _validate_out(
            out,
            device=device,
            batch_size=batch_size,
            num_qo_heads=num_qo_heads,
            max_q_len=max_q_len,
            packed_query=packed_query,
            total_q=total_q,
            out_dtype=out_dtype,
        )
    return _MLARuntime(
        query=query,
        normalized_cache=normalized_cache,
        out=out,
        num_physical_pages=num_physical_pages,
        softmax_scale=effective_softmax_scale,
        output_scale=effective_output_scale,
    )


def _launch_mla_decode(
    runtime: _MLARuntime,
    *,
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    qo_indptr: Optional[torch.Tensor],
    packed_query: bool,
    kv_lora_rank: int,
    split_kv: int,
    workspace: _MLAWorkspaceViews,
    compiled: Callable[..., object],
    lse: Optional[torch.Tensor] = None,
    max_q_len: int,
) -> torch.Tensor:
    """Form the dimension-first views and launch one compiled MLA kernel."""

    lse_storage = workspace.lse if lse is None else lse
    if packed_query and int(runtime.query.shape[0]) == 0:
        return runtime.out
    if packed_query:
        q_latent = runtime.query[..., :kv_lora_rank].permute(1, 2, 0)
        q_rope = runtime.query[..., kv_lora_rank:].permute(1, 2, 0)
        out_kernel = runtime.out.permute(1, 2, 0)
        total_q = int(runtime.query.shape[0])
        lse_kernel = (
            lse_storage.view(-1, lse_storage.shape[-1])[:total_q].transpose(0, 1)
            if lse_storage is not None
            else None
        )
    else:
        q_latent = runtime.query[..., :kv_lora_rank].permute(2, 3, 1, 0)
        q_rope = runtime.query[..., kv_lora_rank:].permute(2, 3, 1, 0)
        out_kernel = runtime.out.permute(2, 3, 1, 0)
        lse_kernel = lse_storage.permute(2, 1, 0) if lse_storage is not None else None
    c_latent = (
        runtime.normalized_cache
        if runtime.normalized_cache.dtype == torch.uint8
        else runtime.normalized_cache[..., :kv_lora_rank]
    ).permute(1, 2, 0)
    c_rope = runtime.normalized_cache[..., kv_lora_rank:].permute(1, 2, 0)
    if runtime.query.shape[-1] == kv_lora_rank:
        q_rope = q_latent
        c_rope = c_latent
    if runtime.extra_cache is not None:
        c_rope = runtime.extra_cache.permute(1, 2, 0)
    page_offsets = block_tables.transpose(0, 1)
    # Persisted FFI callables require positional arguments, including None.
    compiled(
        q_latent,
        q_rope,
        c_latent,
        c_rope,
        page_offsets,
        out_kernel,
        lse_kernel,
        workspace.kernel_workspace,
        split_kv,
        seq_lens_kv,
        qo_indptr,
        runtime.sparse_inputs,
        runtime.softmax_scale,
        runtime.output_scale,
        runtime.output_quant_args,
        max_q_len,
    )
    return runtime.out


class BatchMLADecodePagedTSWrapper:
    """Reusable paged MLA with eager planning and automatic runtime splits.

    Plans select cached kernels from explicit problem bounds and retain scratch.
    Each run supplies live tensors and request metadata. Replanning invalidates
    graphs; concurrent execution requires independent wrappers/workspaces."""

    @flashinfer_experimental_api
    def __init__(self) -> None:
        """Initialize an unplanned task-scheduled paged-MLA wrapper."""
        self._plan_state: Optional[_MLADecodePlanState] = None

    @flashinfer_experimental_api
    def plan(
        self,
        device: int | str | torch.device,
        batch_size: int,
        num_qo_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        page_size: int,
        max_kv_len: int,
        *,
        max_q_len: int,
        packed_query: bool,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        out_dtype: torch.dtype,
        mask_type: Literal["dense", "causal"] = "causal",
        lse_mode: Literal["none", "base2"] = "none",
        workspace_buffer: Optional[torch.Tensor] = None,
        validate: bool = True,
        tuning: Optional[MLAAttentionTuning] = None,
    ) -> None:
        """Compile one static MLA shape and bind its reusable workspace.

        Planning consumes only compile-time geometry, capacity, dtype, and
        storage-mode inputs. Request metadata belongs exclusively to
        :meth:`run` and is never retained by the wrapper. A successful re-plan
        atomically replaces the previous immutable state; a failed re-plan
        leaves that state usable.

        ``packed_query=False`` selects fixed ``[B, SQ, H, 576]`` query storage,
        where ``SQ`` is exactly ``max_q_len``. ``packed_query=True`` selects
        ``[total_q, H, 576]`` storage with per-run cumulative offsets supplied to
        every run. ``max_q_len`` is then the per-request capacity.
        Individual packed requests may be empty; an all-empty run returns its
        empty output without dispatching a GPU kernel.

        The request count B remains fixed in both modes. Fixed mode also fixes
        the total query rows B*SQ; packed mode accepts changing per-request
        lengths within the planned capacity. Sparse independent rows can use
        the capacity-plan recipe on :class:`BatchSparseMLADecodePagedTSWrapper`.
        Q quantization and RoPE are caller-owned preparation; the kernel uses
        Q in ``q_data_type`` directly.

        If ``workspace_buffer`` is omitted, the plan allocates private scratch.
        A workspace is mutable and exclusive to one in-flight launch or graph
        replay. Warm the plan before graph capture, and call ``run`` with
        ``validate=False`` inside compiled or captured regions.

        Equivalent plans/layers reuse CPU policy and compiled-kernel caches.
        The active split count remains a runtime attention/reduction operand.

        Parameters
        ----------
        device : int, str, or torch.device
            CUDA device on which the plan will execute.
        batch_size : int
            Exact runtime request count.
        num_qo_heads, kv_lora_rank, qk_rope_head_dim, page_size : int
            Static MLA head geometry and K/V page size.
        max_kv_len, max_q_len : int
            Static per-request K/V and Q capacities.
        packed_query : bool
            Select packed rather than fixed query storage.
        q_dtype, kv_dtype, out_dtype : torch.dtype
            Query, K/V, and output dtypes used to compile the plan.
        mask_type : {"dense", "causal"}
            Attention mask mode.
        workspace_buffer : torch.Tensor, optional
            Caller-owned contiguous int8 or uint8 scratch on ``device``. It
            must be 32-byte aligned and large enough for the selected plan.
            When omitted, planning allocates the buffer. The retained buffer
            is exclusive to one in-flight launch or graph replay.
        validate : bool
            Validate static geometry and caller scratch. Defaults to ``True``.
            Disable only for previously validated inputs and warmed topology.
        tuning : MLAAttentionTuning, optional
            Compile-time tuning; ``None`` keeps every default.
            ``tuning.softmax_rescale_threshold_log2`` is the opt-in skip of
            the online-softmax O rescale (correction), in binades: 0 to 8 for
            FP8 QKV, 0 to 32 for BF16 QKV. The name and the FP8 P scale
            follow the Prims-TS FMHA decode option of the same name; the
            bounds follow the trtllm-gen FMHA ``skipCorrThreshold``.
            ``None`` or ``0.0`` rescales on every new row maximum. A positive
            value T keeps a row's softmax anchor while a K tile raises the row
            maximum by at most T binades, which skips that tile's rescale of
            the accumulated output. Probabilities then reach ``2**T``, so FP8
            P is quantized at ``448 * 2**-T`` instead of 448 (1.75 at T = 8,
            the trtllm-gen scale). E4M3 P then loses precision from about
            ``10.3 - 0.69 * T`` nats below the anchor and flushes to zero
            below about ``13 - 0.69 * T`` nats (4.7 and 7.5 at T = 8); the
            FP32 row sum keeps the exact values, so the tail's weight is
            biased low. The loss covers the KV range
            that shares the anchor's split, which is the whole row when
            split-KV is not used (typically large batches). Rows whose largest
            scores come first, such as attention sinks, with a long
            low-probability tail are the most exposed, so check model accuracy
            when enabling it for FP8. BF16 P is not scaled. The value is
            compiled into the kernel, so each distinct value builds its own
            kernel. It is range-checked even when ``validate`` is False.
        """

        lse_mode = validate_lse_mode(lse_mode)
        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool")
        if validate:
            if not isinstance(packed_query, bool):
                raise TypeError("packed_query must be a bool")
            _validate_mask(mask_type)
            batch_size = _validate_positive_int(batch_size, "batch_size")
            _validate_mla_int32_extent(batch_size, "batch_size")
            num_qo_heads = _validate_positive_int(num_qo_heads, "num_qo_heads")
            _validate_mla_dims(kv_lora_rank, qk_rope_head_dim)
            page_size = _validate_page_size(page_size)
            max_kv_len = _validate_mla_max_kv_len(max_kv_len, "max_kv_len")
            max_q_len = _validate_positive_int(max_q_len, "max_q_len")
            _validate_mla_query_head_extent(
                batch_size=batch_size,
                num_qo_heads=num_qo_heads,
                max_q_len=max_q_len,
            )
            _validate_mla_dtype_pair(q_dtype, kv_dtype, out_dtype)
        # Range-check the threshold even when validate is False: it selects a
        # compiled kernel, and past the per-dtype bound FP8 P (scaled by
        # 448 * 2**-T) would lose most of the row to E4M3 subnormals.
        threshold = _tuning_rescale_threshold(tuning)
        softmax_rescale_threshold_log2 = _check_softmax_rescale_threshold_log2_range(
            0.0 if threshold is None else threshold, q_dtype=q_dtype
        )

        device, device_index = _resolve_cuda_device(device)
        if validate:
            _validate_runtime_device(device)

        required_page_columns = _ceil_div(max_kv_len, page_size)

        spec_key = (
            device_index,
            batch_size,
            num_qo_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            page_size,
            max_kv_len,
            _dtype_key(q_dtype),
            _dtype_key(kv_dtype),
            _dtype_key(out_dtype),
            mask_type,
            max_q_len,
        )
        spec = _resolve_mla_decode_launch_spec(
            *spec_key, softmax_rescale_threshold_log2
        )
        compile_spec = _cached_mla_decode_compile_spec(
            spec,
            device_index=device_index,
            num_qo_heads=num_qo_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            page_size=page_size,
            q_dtype_key=_dtype_key(q_dtype),
            out_dtype_key=_dtype_key(out_dtype),
            max_q_len=max_q_len,
            packed_query=packed_query,
            lse_mode=lse_mode,
            runtime_kv_lengths=True,
            write_final_lse=lse_mode != "none",
        )
        policy = spec.policy
        workspace_layout = _make_mla_workspace_layout(
            spec.kernel_workspace_bytes,
            batch_size,
            num_qo_heads,
            max_q_len,
            include_lse=False,
        )
        if workspace_buffer is None:
            workspace_buffer = torch.empty(
                workspace_layout.total_bytes, device=device, dtype=torch.int8
            )
        elif validate:
            _validate_workspace_buffer(
                workspace_buffer,
                device=device,
                required_bytes=workspace_layout.total_bytes,
            )
        workspace = _bind_mla_workspace(workspace_buffer, workspace_layout)
        compiled = _get_compiled_mla_decode(compile_spec)

        # Publish only after validation, compilation, allocation, and binding
        # succeed, so a failed re-plan leaves the previous plan usable.
        self._plan_state = _MLADecodePlanState(
            device=device,
            batch_size=batch_size,
            num_qo_heads=num_qo_heads,
            max_q_len=max_q_len,
            packed_query=packed_query,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            page_size=page_size,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            out_dtype=out_dtype,
            mask_type=mask_type,
            max_kv_len=max_kv_len,
            required_page_columns=required_page_columns,
            workspace_buffer=workspace_buffer,
            workspace_layout=workspace_layout,
            workspace_views=workspace,
            compiled=compiled,
            policy=policy,
            split_kv=spec.split_kv,
            lse_mode=lse_mode,
            softmax_rescale_threshold_log2=softmax_rescale_threshold_log2,
        )

    @flashinfer_experimental_api
    def run(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        *,
        qo_indptr: Optional[torch.Tensor] = None,
        max_q_len: int | None = None,
        softmax_scale: float = 1.0,
        output_scale: float = 1.0,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        validate: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Launch MLA using live query/cache, page tables and KV lengths.

        Uses the policy and compiled kernels prepared by plan(). Batch and query
        shape must match that plan; live K/V lengths stay within its capacity.
        Supply output/LSE buffers and validate=False during graph capture, and
        retain the wrapper and captured buffers for every replay.

        An enabled lse_mode="base2" plan writes compact FP32 log2(sum(exp(scores)))
        with the query shape minus its final dimension. return_lse controls tuple
        return only; an omitted enabled buffer is allocated. Output/LSE/workspace must
        be disjoint from inputs and live buffers. One wrapper serves one ordered lane.

        Fixed queries use [B, Sq, H, 576]; packed queries use [total_q, H, 576] and
        qo_indptr with nonnegative per-request deltas. All-empty packed Q is a no-op.
        softmax_scale and output_scale are finite positive host QK and V/output scales.
        The dense MLA interface does not expose attention sinks.
        max_q_len covers the live Q lengths within plan capacity. Validation
        infers an omitted bound from existing metadata; trusted packed calls
        default to plan capacity without a readback. This bound is runtime."""

        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool")

        state = self._plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        runtime_qo_indptr = qo_indptr if state.packed_query else None
        runtime = _prepare_mla_runtime(
            query,
            kv_cache,
            device=state.device,
            batch_size=state.batch_size,
            num_qo_heads=state.num_qo_heads,
            max_q_len=state.max_q_len,
            packed_query=state.packed_query,
            qo_indptr=runtime_qo_indptr,
            page_size=state.page_size,
            q_dtype=state.q_dtype,
            kv_dtype=state.kv_dtype,
            out_dtype=state.out_dtype,
            softmax_scale=softmax_scale,
            output_scale=output_scale,
            out=out,
            validate=validate,
        )
        if validate:
            actual_q = _validate_mla_run_metadata(
                runtime,
                block_tables,
                seq_lens_kv,
                qo_indptr,
                device=state.device,
                batch_size=state.batch_size,
                required_page_columns=state.required_page_columns,
                packed_query=state.packed_query,
                max_q_len=state.max_q_len,
                max_kv_len=state.max_kv_len,
                page_size=state.page_size,
                mask_type=state.mask_type,
            )
            max_q_len = resolve_max_sequence_length(
                max_q_len, actual_q, "max_q_len", state.max_q_len
            )
        elif not state.packed_query:
            max_q_len = int(query.shape[1])
        elif max_q_len is None:
            max_q_len = state.max_q_len
        lse = prepare_lse(
            lse,
            q=query,
            mode=state.lse_mode,
            return_lse=return_lse,
            validate=validate,
        )
        result = _launch_mla_decode(
            runtime,
            block_tables=block_tables,
            seq_lens_kv=seq_lens_kv,
            qo_indptr=runtime_qo_indptr,
            packed_query=state.packed_query,
            kv_lora_rank=state.kv_lora_rank,
            split_kv=state.split_kv,
            workspace=state.workspace_views,
            compiled=state.compiled,
            lse=lse,
            max_q_len=max_q_len,
        )
        return (result, lse) if return_lse else result


@flashinfer_experimental_api
def batch_mla_decode_with_paged_kv_cache(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    *,
    qo_indptr: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    kv_lora_rank: int = _MLA_LATENT_DIM,
    qk_rope_head_dim: int = _MLA_ROPE_DIM,
    mask_type: Literal["dense", "causal"] = "causal",
    max_kv_len: Optional[int] = None,
    softmax_scale: float = 1.0,
    output_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    lse_mode: Literal["base2"] = "base2",
    out_dtype: torch.dtype = torch.bfloat16,
    workspace_buffer: Optional[torch.Tensor] = None,
    validate: bool = True,
    tuning: Optional[MLAAttentionTuning] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """One-shot convenience wrapper for fixed or packed-query MLA decode.

    Without caller scratch, this helper may read ``seq_lens_kv`` on the host
    to derive the K/V bound and constructs a temporary wrapper outside capture.
    Packed Q requires a caller-provided ``max_q_len`` in either mode.
    See Notes for the explicit capture-safe mode.

    Parameters
    ----------
    query : torch.Tensor
        Fixed or packed query tensor with concatenated latent and RoPE heads.
    kv_cache : torch.Tensor
        Compact paged latent K/V cache.
    block_tables : torch.Tensor
        Dense physical-page table for each request. Rows must be inner
        contiguous and non-overlapping, but may have padding between them.
    seq_lens_kv : torch.Tensor
        Live K/V sequence lengths.
    qo_indptr : torch.Tensor, optional
        Cumulative query offsets selecting packed-query mode.
    max_q_len : int, optional
        Per-request packed-query length capacity. Required for packed Q;
        it is never inferred from ``qo_indptr`` and may exceed the maximum
        request length.
        For fixed Q it defaults to the query's sequence extent and, when
        provided, must equal that extent.
    kv_lora_rank, qk_rope_head_dim : int
        Latent and RoPE dimensions.
    mask_type : {"dense", "causal"}
        Attention mask mode.
    max_kv_len : int, optional
        Static K/V length bound; defaults to the metadata maximum.
    softmax_scale, output_scale : float
        QK and value/output scaling factors.
    out : torch.Tensor, optional
        Caller-owned output tensor.
    out_dtype : torch.dtype
        Output dtype.
    workspace_buffer : torch.Tensor, optional
        Caller-owned byte scratch, exclusive to one in-flight launch/graph.
    validate : bool
        Validate tensors and metadata values (may synchronize), default True.
        False trusts the caller and requires workspace and explicit bounds;
        skips tensor and metadata validation. Invalid inputs have undefined behavior.
    tuning : MLAAttentionTuning, optional
        Compile-time tuning; see :meth:`BatchMLADecodePagedTSWrapper.plan`.
        ``None`` (default) keeps the exact rescale. A positive
        ``softmax_rescale_threshold_log2`` T quantizes FP8 P at
        ``448 * 2**-T`` instead of 448, which costs E4M3 precision for
        probabilities far below the anchor.

    Returns
    -------
    torch.Tensor
        The fixed or packed MLA attention output.
    Notes
    -----
    Both owned and caller-provided scratch use the same wrapper plan/run path.
    For CUDA Graph capture, supply ``workspace_buffer``, ``max_kv_len``,
    ``out``, and ``validate=False``; packed Q also needs an explicit
    ``max_q_len``.
    Warm this exact topology outside capture first. Retain stable tensor storage
    and mutate metadata only between completed launches/replays. All live K/V
    positive lengths must be within the static bound; nonpositive lengths mark
    inactive requests whose output is untouched. Active page IDs must
    index the cache, and causal per-request Q lengths must not exceed K/V lengths.
    Packed offsets must start at zero, end at the query token count, and have
    nonnegative deltas within the Q bound. Scratch/output must not alias
    any inputs or each other. No metadata is copied to the host on the trusted
    explicit path. Kernel policy, necessary control resets, and output layout
    are unchanged. Missing output may be allocated only outside capture.


    ``return_lse=True`` returns ``(output, lse)``. ``lse_mode`` selects
    ``"base2"`` (the only supported base); it is used only when returning LSE
    or writing a caller-provided ``lse`` buffer.

    ``lse`` is an optional independent, compact float32 output with shape
    ``query.shape[:-1]`` (``q.shape[:-1]`` for FMHA). An enabled LSE plan
    allocates it when omitted, including when ``return_lse=False``. A supplied
    buffer is written regardless of ``return_lse``; the flag only controls
    the Python return type. LSE includes any sink mass and excludes output
    scaling. Keep LSE disjoint from output, inputs, and workspace. Supply
    both output buffers and disable validation for allocation-free capture.
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
        if qo_indptr is not None and max_q_len is None:
            raise ValueError("validate=False requires max_q_len for packed Q")

    packed_query = qo_indptr is not None
    if packed_query and max_q_len is None:
        raise ValueError("packed Q requires max_q_len")
    if validate:
        _validate_query(query, packed_query=packed_query)
        metadata_device, batch_size, _ = _validate_mla_metadata(
            block_tables, seq_lens_kv
        )
        if metadata_device != query.device:
            raise ValueError(
                f"MLA metadata must be on {query.device}, got {metadata_device}"
            )
        normalized_cache, _, page_size = _normalize_mla_kv_cache(
            kv_cache, expected_device=query.device
        )
        if packed_query:
            _validate_qo_indptr(qo_indptr, device=query.device, batch_size=batch_size)
        elif max_q_len is not None and max_q_len != int(query.shape[1]):
            raise ValueError(
                "fixed query length must equal max_q_len: "
                f"got SQ={query.shape[1]} and max_q_len={max_q_len}"
            )
    else:
        batch_size = int(seq_lens_kv.shape[0])
        normalized_cache = kv_cache[:, 0] if kv_cache.ndim == 4 else kv_cache
        page_size = int(normalized_cache.shape[1])
    num_qo_heads = int(query.shape[-2])
    if not packed_query and max_q_len is None:
        max_q_len = int(query.shape[1])
    if max_kv_len is None:
        max_kv_len = int(seq_lens_kv.max().item())
    assert max_q_len is not None
    if validate and out is not None:
        _validate_out(
            out,
            device=query.device,
            batch_size=batch_size,
            num_qo_heads=num_qo_heads,
            max_q_len=max_q_len,
            packed_query=packed_query,
            total_q=int(query.shape[0]) if packed_query else None,
            out_dtype=out_dtype,
        )
    wrapper = BatchMLADecodePagedTSWrapper()
    wrapper.plan(
        query.device,
        batch_size=batch_size,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
        num_qo_heads=num_qo_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=page_size,
        packed_query=packed_query,
        q_dtype=query.dtype,
        kv_dtype=normalized_cache.dtype,
        out_dtype=out_dtype,
        mask_type=mask_type,
        workspace_buffer=workspace_buffer,
        validate=validate,
        tuning=tuning,
        lse_mode=validate_lse_mode(lse_mode)
        if return_lse or lse is not None
        else "none",
    )
    return wrapper.run(
        query,
        normalized_cache,
        block_tables,
        seq_lens_kv,
        qo_indptr=qo_indptr,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        out=out,
        lse=lse,
        return_lse=return_lse,
        validate=validate,
    )


__all__ = [
    "BatchMLADecodePagedTSWrapper",
    "batch_mla_decode_with_paged_kv_cache",
    "get_prims_ts_batch_mla_decode_workspace_size",
]


def _make_sparse_output_quant(heads, block_size, scale_format, groups):
    if block_size is None:
        if groups is not None or scale_format != "ue8m0":
            raise ValueError(
                "output quantization options require output_quant_block_size"
            )
        return None
    block_size = _validate_positive_int(block_size, "output_quant_block_size")
    if block_size not in (32, 128):
        raise ValueError("output_quant_block_size must be 32, 128, or None")
    if scale_format not in ("fp32", "ue8m0"):
        raise ValueError("out_scale_format must be 'fp32' or 'ue8m0'")
    groups = _validate_positive_int(groups, "num_output_groups")
    if heads % groups:
        raise ValueError("num_output_groups must divide num_qo_heads")
    from .kernels.mla_decode.helpers.output_quant import MlaOutputQuant

    return MlaOutputQuant(block_size, scale_format == "ue8m0", heads, groups)


def _fake_output_quant_args(quant):
    if quant is None:
        return None
    import cutlass
    import cutlass.cute as cute

    rows = cute.sym_int()
    padded_rows = cute.sym_int(divisibility=4)
    columns = quant.num_heads // quant.num_groups * 512 // quant.block_size
    if quant.ue8m0:
        columns //= 4
    return (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float8E4M3FN,
            (cute.sym_int64(),),
            stride_order=(0,),
            assumed_align=16,
        ),
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (rows,), stride_order=(0,), assumed_align=4
        ),
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (cute.sym_int(), 64), stride_order=(1, 0), assumed_align=16
        ),
        cute.runtime.make_fake_tensor(
            cutlass.Int32 if quant.ue8m0 else cutlass.Float32,
            (rows, quant.num_groups, columns),
            stride=(1, columns * padded_rows, padded_rows),
            assumed_align=16,
        ),
    )


def _prepare_sparse_output_quant(
    quant, query, positions, cos_sin, out, scales, validate
):
    """Bind the token-major output and column-major block scales."""
    prefix = tuple(query.shape[:-2])
    rows, device = math.prod(prefix), query.device
    channels = quant.num_heads // quant.num_groups * 512
    columns = channels // quant.block_size // (4 if quant.ue8m0 else 1)
    padded_rows = (rows + 3) // 4 * 4
    if (
        positions is None
        or positions.shape != prefix
        or positions.dtype != torch.int32
        or positions.device != device
        or not positions.is_contiguous()
    ):
        raise ValueError(
            "token_positions must be contiguous CUDA INT32 matching the query prefix"
        )
    if (
        cos_sin is None
        or cos_sin.ndim != 2
        or cos_sin.shape[0] == 0
        or cos_sin.shape[1] != 64
        or cos_sin.dtype != torch.float32
        or cos_sin.device != device
        or not cos_sin.is_contiguous()
        or cos_sin.data_ptr() % 16
    ):
        raise ValueError(
            "cos_sin_cache must be 16-byte aligned contiguous CUDA FP32[P,64]"
        )
    if (
        validate
        and rows
        and ((positions < 0) | (positions >= cos_sin.shape[0])).any().item()
    ):
        raise ValueError("token_positions must index cos_sin_cache")
    shape = (rows, quant.num_groups, channels)
    # Physical storage is group-major [G, T, C] (as in vLLM and TensorRT-LLM);
    # the public tensor is its token-major [T, G, C] view.
    if out is None:
        out = torch.empty(
            (quant.num_groups, rows, channels), dtype=torch.float8_e4m3fn, device=device
        ).transpose(0, 1)
    if (
        out.shape != shape
        or out.dtype != torch.float8_e4m3fn
        or out.device != device
        or out.data_ptr() % 16
    ):
        raise ValueError(
            f"out must be the [T, G, C] view of a contiguous group-major CUDA "
            f"E4M3 [G, T, C] tensor ({shape}), aligned to 16 bytes"
        )
    physical_out = out.transpose(0, 1)
    if rows and quant.num_groups > 1:
        # A live prefix can retain a larger graph bucket's group stride.
        capacity, remainder = divmod(physical_out.stride(0), channels)
        if remainder or capacity < rows or capacity * quant.num_heads >= 2**31:
            raise ValueError("out must have valid group-major token capacity")
        try:
            physical_out = physical_out.as_strided(
                (quant.num_groups, capacity, channels), physical_out.stride()
            )
        except RuntimeError as exc:
            raise ValueError(
                "out must view complete group-major backing storage"
            ) from exc
    if not physical_out.is_contiguous():
        raise ValueError("out must have contiguous physical group-major storage")
    scale_dtype = torch.int32 if quant.ue8m0 else torch.float32
    scale_shape = (rows, quant.num_groups, columns)
    scale_stride = (1, columns * padded_rows, padded_rows)
    if scales is None:
        scales = torch.empty(
            (quant.num_groups, columns, padded_rows), dtype=scale_dtype, device=device
        ).permute(2, 0, 1)[:rows]
        if rows == 0:
            scales = scales.as_strided(scale_shape, scale_stride)
    if (
        scales.shape != scale_shape
        or scales.stride() != scale_stride
        or scales.dtype != scale_dtype
        or scales.device != device
        or scales.data_ptr() % 16
    ):
        raise ValueError(
            f"out_scale must be 16-byte aligned CUDA {scale_dtype}{scale_shape} "
            f"with strides {scale_stride}"
        )
    return (
        out,
        scales,
        (physical_out.view(-1), positions.view(-1), cos_sin, scales),
    )


@functools.lru_cache(maxsize=1024)
def _resolve_sparse_mla_plan(
    device,
    batch_size,
    num_qo_heads,
    *,
    max_topk,
    max_extra_topk=0,
    max_q_len=1,
    packed_query=False,
    q_dtype=torch.bfloat16,
    qk_rope_head_dim=0,
    kv_cache_format="native",
    extra_kv_cache_format=None,
    kv_layout="NHD",
    use_attention_sinks=False,
    return_lse=False,
    assume_valid_prefix=False,
    output_quant_block_size=None,
    out_scale_format="ue8m0",
    num_output_groups=None,
    tuning=None,
    kv_lora_rank=512,
):
    """Resolve sparse geometry and scratch sizing without allocation/compilation."""
    import cutlass
    from .kernels.mla_decode.throughput_latency_1cta.config import (
        MlaProfile,
        compute_workspace_size,
    )
    from .kernels.mla_decode.throughput_latency_1cta.kernel import (
        ThroughputLatencyMlaDecodeTs,
    )
    from .kernels.mla_decode.throughput_2cta.config import (
        compute_workspace_size as workspace_2cta,
    )
    from .kernels.mla_decode.throughput_2cta.kernel import (
        MlaDecodeTs,
    )

    if isinstance(kv_lora_rank, bool) or kv_lora_rank != 512:
        raise ValueError("sparse MLA requires kv_lora_rank=512")
    if not isinstance(assume_valid_prefix, bool):
        raise TypeError("assume_valid_prefix must be bool")
    device, _ = _resolve_cuda_device(device)
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        raise NotImplementedError("sparse TS MLA requires SM100 or SM103")
    for name, value in (
        ("batch_size", batch_size),
        ("num_qo_heads", num_qo_heads),
        ("max_q_len", max_q_len),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if num_qo_heads > 128:
        raise ValueError("num_qo_heads must be at most 128")
    if isinstance(qk_rope_head_dim, bool) or qk_rope_head_dim not in (0, 64):
        raise ValueError("qk_rope_head_dim must be 0 or 64")
    if qk_rope_head_dim and output_quant_block_size is not None:
        raise ValueError("fused output RoPE requires D512 attention")
    output_quant = _make_sparse_output_quant(
        num_qo_heads, output_quant_block_size, out_scale_format, num_output_groups
    )
    for value in (max_topk, max_extra_topk):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("source capacities must be nonnegative integers")
    if max_topk == 0:
        raise ValueError("the primary selected-slot capacity must be positive")
    if q_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("native BF16 or E4M3 is required")
    if extra_kv_cache_format is None:
        extra_kv_cache_format = "nvfp4" if kv_cache_format == "fp8" else kv_cache_format
    validate_nvfp4_kv_formats(
        kv_cache_format,
        extra_kv_cache_format,
        qk_rope_head_dim,
        "e4m3" if q_dtype == torch.float8_e4m3fn else "bf16",
        1,
    )
    if kv_layout not in ("NHD", "HND"):
        raise ValueError("kv_layout must be NHD or HND")
    softmax_rescale_threshold_log2 = _tuning_rescale_threshold(tuning)
    if softmax_rescale_threshold_log2 is not None:
        softmax_rescale_threshold_log2 = _check_softmax_rescale_threshold_log2_range(
            softmax_rescale_threshold_log2, q_dtype=q_dtype
        )
    max_rows = batch_size * max_q_len
    capacity = max(
        256,
        ((max_topk + 127) // 128 + (max_extra_topk + 127) // 128) * 128,
    )
    if max_rows * num_qo_heads >= 2**31 or capacity >= 2**31 - 32768:
        raise ValueError("query/route extent exceeds the int32 kernel domain")
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    profile = select_sparse_mla_profile(
        rows=max_rows,
        heads=num_qo_heads,
        query_length=max_q_len,
        capacity=capacity,
        dtype="bf16" if q_dtype == torch.bfloat16 else "fp8",
        sm_count=sm_count,
        rope_dim=qk_rope_head_dim,
        kv_cache_format=kv_cache_format,
    )
    # Reloading V from L2 shortens native FP8 decode's producer lifetime.
    # Sustained grids amortize the retained-KV/CLC pipeline instead; packed
    # caches and two-source/sink plans retain their format-specific schedules.
    # The general path also exploits a hole-free prefix promise; the reload
    # loader keeps arbitrary-hole masks and does not use that optimization.
    if (
        qk_rope_head_dim == 64
        and kv_cache_format == "native"
        and q_dtype == torch.float8_e4m3fn
        and num_qo_heads in _D576_HEADS
        and not max_extra_topk
        and not use_attention_sinks
        and not assume_valid_prefix
        and prefer_sparse_d576_reload(
            rows=max_rows,
            heads=num_qo_heads,
            query_length=max_q_len,
            capacity=max_topk,
            sm_count=sm_count,
            general_profile=profile,
        )
    ):
        return resolve_sparse_d576_reload_plan(
            device,
            max_rows,
            num_qo_heads,
            max_topk=max_topk,
            max_extra_topk=max_extra_topk,
            q_dtype=q_dtype,
            use_attention_sinks=use_attention_sinks,
            assume_valid_prefix=assume_valid_prefix,
            threshold=softmax_rescale_threshold_log2,
        )
    kernel_threshold = 0.0
    if softmax_rescale_threshold_log2 is not None:
        # An explicit threshold replaces the schedule's fixed six-binade
        # deferral; zero restores exact maxima for every schedule.
        profile = replace(profile, defer_max_update=False)
        kernel_threshold = softmax_rescale_threshold_log2
    family, tile = profile.family, profile.tile_size_q
    splits, head_dim_ctas = profile.split_kv, profile.head_dim_ctas
    dtype_name = "bf16" if q_dtype == torch.bfloat16 else "e4m3"
    if family == "2cta":
        active_clusters = sm_count // 2
        # Keep the physical device limit batch-independent. Quantized
        # persistent launches balance their grid from live tensor extents.
        kernel = MlaDecodeTs(
            page_size=1,
            max_active_clusters=active_clusters,
            is_persistent=profile.scheduler != "nonpersistent",
            is_var_seq=False,
            is_var_split_kv=False,
            static_split_kv=splits,
            qkv_dtype=dtype_name,
            out_dtype="bf16",
            rope_dim=qk_rope_head_dim,
            num_heads=num_qo_heads,
            seq_len_q=1,
            batch_size=max_rows,
            mask_type="dense",
            softmax_rescale_threshold_log2=kernel_threshold,
            # Sparse MLA flattens Q rows into virtual batch items, so the
            # physical Q length stays compile-time.
            fixed_seqlen=True,
        )
        kernel.kv_cache_format = kv_cache_format
        kernel.extra_kv_cache_format = extra_kv_cache_format
        kernel.sparse_gather_warps = profile.gather_issue_warps
        kernel.sparse_kv_stages = profile.kv_pipeline_stages
        kernel.sparse_uniform_pages = profile.uniform_offset_cache
        kernel.sparse_balanced_registers = profile.balanced_registers
        core_bytes = workspace_2cta(
            tile_size_q=128,
            num_q_tiles=1,
            latent_dim=512,
            batch_size=max_rows,
            split_kv=splits,
            partial_o_dtype=cutlass.BFloat16,
            lse_dtype=cutlass.Float32,
        )
    else:
        kernel_profile = MlaProfile(
            name="sparse",
            kernel_variant="swaps_mma_ab" if family == "swap" else "keeps_mma_ab",
            tile_size_q=tile,
            num_ctas_per_head_dim=head_dim_ctas,
            num_ctas_per_seq_kv=splits,
            use_multi_ctas_kv=int(splits > 1),
            use_cluster_reduction=int(profile.reduction == "cluster"),
            use_persistent_scheduler=int(profile.scheduler != "nonpersistent"),
            use_clc_dynamic_persistent_scheduler=int(profile.scheduler == "clc"),
        )
        kernel = ThroughputLatencyMlaDecodeTs(
            batch_size=max_rows,
            num_heads=tile,
            seq_len_q=(num_qo_heads + tile - 1) // tile,
            seq_len_k=capacity,
            rope_dim=qk_rope_head_dim,
            page_size=1,
            max_active_clusters=sm_count,
            qkv_dtype=dtype_name,
            out_dtype="bf16",
            profile=kernel_profile,
            reduction_mode=profile.reduction,
            logical_num_heads=num_qo_heads,
            logical_seq_len_q=1,
            tile_size_q=tile,
            sparse_profile=profile,
            mask_type="dense",
            softmax_rescale_threshold_log2=kernel_threshold,
            kv_cache_format=kv_cache_format,
            extra_kv_cache_format=extra_kv_cache_format,
        )
        core_bytes = compute_workspace_size(
            cfg=kernel._make_config(),
            partial_o_dtype=cutlass.BFloat16,
            lse_dtype=cutlass.Float32,
        )
    spec = _make_mla_decode_compile_spec(
        _MLADecodeLaunchSpec(kernel, (), core_bytes, splits),
        device_index=device.index,
        num_qo_heads=num_qo_heads,
        kv_lora_rank=512,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=1,
        q_dtype_key=str(q_dtype).removeprefix("torch."),
        out_dtype_key="float8_e4m3fn" if output_quant is not None else "bfloat16",
        max_q_len=1,
        packed_query=False,
    )
    sections = {}
    byte_end = 0
    for name, shape, dtype in (
        ("core", (core_bytes,), torch.int8),
        ("public_lse", (max_rows, num_qo_heads), torch.float32),
    ):
        sections[name], byte_end = _append_workspace_section(byte_end, shape, dtype)
    return device, profile, spec, sections, byte_end, capacity, output_quant


_D576_HEAD_DIM = 576


_D576_V_DIM = 512


_D576_HEADS = (64, 128)


def _sparse_d576_workspace_bytes(max_rows: int, num_heads: int, split_kv: int) -> int:
    """Split-KV partials of a D576 plan: BF16 O and FP32 LSE per split row."""
    if split_kv <= 1:
        return 0
    return max_rows * num_heads * split_kv * (_D576_V_DIM * 2 + 4)


def resolve_sparse_d576_reload_plan(
    device,
    max_rows,
    num_heads,
    *,
    max_topk,
    max_extra_topk,
    q_dtype,
    use_attention_sinks,
    assume_valid_prefix,
    threshold,
):
    """Plan the native D576 reload schedule and its split-KV scratch.

    One CTA per (token, 64 heads) on the 1CTA keeps-MMA-AB q64 schedule; 128
    heads run as two physical query positions of one token. The kernel is
    compiled for the top-k capacity rounded up to whole 128-key tiles (it uses
    the capacity only through its tile count). Below one wave the plan splits
    each top-k list over ``split_kv`` CTAs (and the 512 V columns over
    ``head_dim_ctas``); the parallel GMEM reducer, launched with PDL, merges
    the partials and publishes natural-log LSE. The split is fixed per plan,
    so a CUDA graph captured from it uses one compiled kernel.
    """
    import cutlass

    from ._hardware import get_max_active_clusters
    from .kernels.mla_decode.sparse_policy import (
        SparseMlaProfile,
        select_sparse_mla_d576_split,
    )
    from .kernels.mla_decode.throughput_latency_1cta.config import MlaProfile
    from .kernels.mla_decode.throughput_latency_1cta.kernel import (
        ThroughputLatencyMlaDecodeTs,
    )

    if q_dtype != torch.float8_e4m3fn:
        raise ValueError("D576 sparse MLA (qk_rope_head_dim=64) requires E4M3 Q / KV")
    if num_heads not in _D576_HEADS:
        raise ValueError(
            f"D576 sparse MLA supports 64 or 128 query heads (64-row tiles), got {num_heads}"
        )
    if max_extra_topk:
        raise ValueError("D576 sparse MLA attends to one KV pool (max_extra_topk=0)")
    if use_attention_sinks:
        raise ValueError("D576 sparse MLA does not support attention sinks")
    if max_rows > 65535:
        # Each query token is one grid-z slice of the attention and reducer
        # launches; CUDA caps grid z at 65535.
        raise ValueError(
            "D576 sparse MLA supports at most 65535 query tokens per plan "
            f"(batch_size * max_q_len), got {max_rows}"
        )
    split_kv, head_dim_ctas = select_sparse_mla_d576_split(
        max_rows * (num_heads // 64),
        max_topk,
        torch.cuda.get_device_properties(device).multi_processor_count,
    )
    profile = SparseMlaProfile(
        family="keep",
        tile_size_q=64,
        split_kv=split_kv,
        head_dim_ctas=head_dim_ctas,
        uniform_offset_cache=True,
        kv_pipeline_stages=10,
    )
    capacity = -(-max_topk // 128) * 128
    phys_sq = num_heads // 64
    kernel_profile = "h64_keeps_mma_ab"
    if split_kv > 1:
        hdim = "" if head_dim_ctas == 1 else f"_hdim{_D576_V_DIM // head_dim_ctas}"
        kernel_profile = MlaProfile(
            name=f"h64_keeps_mma_ab_splitkv{split_kv}{hdim}_gmem",
            num_ctas_per_seq_kv=split_kv,
            num_ctas_per_head_dim=head_dim_ctas,
            use_persistent_scheduler=0,
            use_clc_dynamic_persistent_scheduler=0,
            use_multi_ctas_kv=1,
            use_cluster_reduction=0,
            kernel_variant="keeps_mma_ab",
            tile_size_q=64,
        )
    with torch.cuda.device(device):
        max_active_clusters = get_max_active_clusters(1)
    kernel = ThroughputLatencyMlaDecodeTs(
        batch_size=1,
        num_heads=64,
        seq_len_q=phys_sq,
        seq_len_k=capacity,
        latent_dim=512,
        rope_dim=64,
        page_size=1,
        max_active_clusters=max_active_clusters,
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        qkv_dtype="e4m3",
        out_dtype="bf16",
        profile=kernel_profile,
        reduction_mode=None if split_kv == 1 else "gmem_separate",
        logical_num_heads=num_heads,
        logical_seq_len_q=1,
        fixed_seqlen=True,
        tile_size_q=64,
        explicit_split_kv=None,
        explicit_persistent=None,
        mask_type="dense",
        softmax_rescale_threshold_log2=0.0 if threshold is None else threshold,
    )
    # Preserve the sparse wrapper's natural-log LSE contract.
    kernel.lse_base_on_e = True
    cfg = kernel._make_config()
    if (
        int(cfg.num_ctas_per_seq_kv) != split_kv
        or int(cfg.num_ctas_per_head_dim) != head_dim_ctas
        or int(cfg.use_cluster_reduction) != 0
    ):
        raise NotImplementedError(
            f"D576 sparse MLA expects {split_kv} KV CTAs, {head_dim_ctas} V "
            f"CTAs and GMEM reduction; got {cfg.num_ctas_per_seq_kv}, "
            f"{cfg.num_ctas_per_head_dim}, cluster={cfg.use_cluster_reduction}"
        )
    if split_kv > 1 and not kernel.use_parallel_reduction:
        # Only the parallel GMEM reducer publishes the D576 natural-log LSE.
        raise NotImplementedError("D576 split-KV requires the parallel GMEM reducer")
    spec = _MLADecodeCompileSpec(
        device_index=device.index,
        kernel_signature=_mla_kernel_compile_signature(kernel),
        num_qo_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        page_size=1,
        q_dtype_key="float8_e4m3fn",
        out_dtype_key="bfloat16",
        max_q_len=phys_sq,
        packed_query=False,
        has_kernel_workspace=split_kv > 1,
        split_kv=split_kv,
        kernel=kernel,
    )
    sections = {}
    byte_end = 0
    for name, shape, dtype in (
        (
            "core",
            (_sparse_d576_workspace_bytes(max_rows, num_heads, split_kv),),
            torch.int8,
        ),
        ("public_lse", (max_rows, num_heads), torch.float32),
    ):
        sections[name], byte_end = _append_workspace_section(byte_end, shape, dtype)
    return device, profile, spec, sections, byte_end, capacity, None


_PREPARED_LAUNCH_LIMIT = 256


class BatchSparseMLADecodePagedTSWrapper:
    """Plan native BF16/E4M3 attention, then bind one or two source index lists.

    Q is [B,Sq,H,D] or packed [total_q,H,D], where D is 512 or 576.
    D576 adds 64 QK-only RoPE channels; V and output remain D512.
    Each source supplies int32
    [rows, capacity] storage-row indices and int32 [rows] live lengths directly
    to run(). Indices include physical page strides; -1 is masked. Both source
    lists participate in one attention distribution. The caller folds the
    shared KV descale into the effective softmax and output scales.

    ``output_quant_block_size=32`` or ``128`` fuses conjugate output RoPE on
    D[448:512] and E4M3 quantization after the final attention reduction. Q/KV
    remain prepared native tensors. Supply the local ``num_output_groups`` G
    and ``out_scale_format="ue8m0"`` (packed exponent bytes) or ``"fp32"``.
    This option does not change attention dispatch or softmax correction.

    Planning with ``assume_valid_prefix=True`` promises that every index before
    each live source length is valid (no -1 holes). Kernels can then derive
    masks from lengths and omit per-index hole checks. This does not imply
    that the physical KV rows are consecutive.
    ``run(validate=True)`` checks the promise; graph replay with validation
    disabled must preserve it. Tail entries beyond the lengths are ignored.

    A wrapper/workspace permits one in-flight run. Graphs require a completed
    warmup, stable addresses, preallocated outputs, and validate=False.
    Input/output/workspace storage must not overlap; this is an unchecked
    caller precondition. Separate wrappers/workspaces are needed per stream.

    Planning compiles one kernel with host-runtime FP32 scale arguments.
    Scale values may change between launches without recompiling. CUDA graphs
    capture their values, so new values require graph parameter updates or
    recapture. For a
    fixed dispatch profile over a range of independent sparse query rows,
    plan with ``batch_size=1``, ``packed_query=True`` and
    ``max_q_len=row_capacity``. Run with ``[rows,H,D]`` queries and
    ``qo_indptr=[0,rows]`` for any ``rows <= row_capacity``.

    Q quantization and query RoPE belong to the caller. For a serving hot path,
    establish valid metadata upstream and pass ``run(validate=False)``.
    The default ``validate=True`` reads CUDA metadata values on the host and
    therefore synchronizes. Kernel-builder advisories are controlled separately
    at plan/JIT time by ``FLASHINFER_PRIMS_TS_DEBUG_CHECKS=1``.
    """

    @flashinfer_experimental_api
    def __init__(self, workspace_buffer=None):
        self._workspace_buffer = workspace_buffer
        self._state = None
        self._prepared: dict[tuple, dict] = {}
        self.workspace_size_bytes = 0

    def plan(
        self,
        device,
        batch_size,
        num_qo_heads,
        *,
        max_topk,
        max_extra_topk=0,
        max_q_len=1,
        packed_query=False,
        q_dtype=torch.bfloat16,
        qk_rope_head_dim=0,
        kv_cache_format="native",
        extra_kv_cache_format=None,
        kv_layout="NHD",
        use_attention_sinks=False,
        return_lse=False,
        assume_valid_prefix=False,
        output_quant_block_size=None,
        out_scale_format="ue8m0",
        num_output_groups=None,
        tuning=None,
        kv_lora_rank=512,
    ):
        """Plan D512/D576 attention over one primary and an optional extra pool.

        ``kv_cache_format="nvfp4"`` uses the DS v4.1 D512 format: 256 packed
        E2M1 bytes followed by 32 E4M3 scales in linear block16 order. Every
        QK/V channel, including rotated channels, uses the same storage.
        Keep ``qk_rope_head_dim=0``: there is no additional QK-only tail.

        Set ``qk_rope_head_dim=64`` and ``kv_cache_format="nvfp4.fp8rope"``
        for D576: 256 packed E2M1 NoPE bytes, 64 unscaled E4M3 RoPE bytes,
        then 32 E4M3 scales, with block b stored at 320 + 8*(b%4) + b//4.
        ``kv_cache_format="fp8"`` selects D512 per-tensor E4M3 storage:
        512 bytes per token with no scale trailer. Pass its positive tensor
        descale as ``kv_cache_scale`` to run(). Pair it with
        ``extra_kv_cache_format="nvfp4"`` for FP8 SWA and compressed NVFP4 KV.
        The extra format defaults to NVFP4 for an FP8 primary, otherwise to
        the primary format. D512 supports NVFP4 in both pools or FP8 primary
        with NVFP4 extra; native and mixed-RoPE formats require matching pools.
        NVFP4-containing recipes require E4M3 Q. NVFP4 block scales are
        applied in the loader before ordinary FP8 MMA, without a global
        storage scale. Per-tensor FP8 bytes feed MMA directly; its scale is
        applied to QK scores and FP32 output accumulation, avoiding another
        E4M3 rounding. Caches are uint8 records; FP8 accepts E4M3 tensors too.
        In NVFP4 bytes, even-indexed values occupy the low nibble. NVFP4
        expands to E4M3 in the loader; BF16 Q must be converted upstream for
        these recipes.

        ``max_topk`` describes arbitrary selected primary entries, not an SWA
        window. Prepared indices already include the physical page stride, so
        the core uses page size one regardless of the pools' external pages.
        Planning compiles no metadata-preparation kernel.

        Fixed queries use exactly ``batch_size * max_q_len`` rows. To reuse
        one dispatch profile over a range of independent query-row counts,
        use ``batch_size=1``, ``packed_query=True`` and
        ``max_q_len=row_capacity``. Supply ``[rows,H,D]`` Q and CUDA INT32
        ``qo_indptr=[0,rows]`` to each run, with ``rows <= row_capacity``.
        Index/length tensors and outputs use the same live row count. The
        profile is chosen for the planned capacity; each captured graph keeps
        its captured views and addresses. For example::

            wrapper.plan("cuda", 1, heads, max_topk=topk,
                         max_q_len=row_capacity, packed_query=True,
                         q_data_type=torch.float8_e4m3fn,
                         kv_cache_format="nvfp4", return_lse=True)
            # Q is already quantized/rotated; update metadata upstream.
            wrapper.run(q, cache, indices, lengths, qo_indptr=qo_indptr,
                        out=out, lse=lse, validate=False)

        Whole-tile K/V reuse supports nonpersistent split-KV schedules.
        The 1CTA persistent/CLC scheduler requires ``split_kv=1`` because its
        work queue enumerates query/head tiles without a KV-split coordinate.
        Low-level profile overrides must use a nonpersistent scheduler with
        multiple KV splits; this restriction does not apply to 2CTA profiles.

        Routine task-builder overlap and unused-routing advisories are quiet
        by default. Set ``FLASHINFER_PRIMS_TS_DEBUG_CHECKS=1`` before a cold
        plan/JIT build to show them. This controls build diagnostics separately
        from :meth:`run`'s runtime input validation; other warnings/errors are
        retained.

        Output quantization is disabled by default. When enabled, G must divide
        the local query-head count H. Blocks of 32 with UE8M0 scales implement
        MXFP8; FP32 scales use unrounded amax/448. Both use an amax floor of
        1e-4. A block never crosses a head boundary.

        ``tuning`` (an :class:`MLAAttentionTuning`) has the meaning of
        :meth:`BatchMLADecodePagedTSWrapper.plan`; its
        ``softmax_rescale_threshold_log2`` takes 0 to 8 binades for E4M3 and
        0 to 32 for BF16. ``None`` (default) keeps each schedule's own
        behavior: some BF16 schedules keep the row anchor while a K tile
        raises it by at most 6 binades, and all others rescale on every new
        maximum. An explicit value applies to every schedule; ``0.0``
        disables the skip (slower than ``None`` on the deferred BF16
        schedules), and a positive value T also quantizes E4M3 P at
        ``448 * 2**-T`` instead of 448, with the precision cost described
        there.
        """
        (device, profile, spec, sections, byte_end, _, output_quant) = (
            _resolve_sparse_mla_plan(
                device,
                batch_size,
                num_qo_heads,
                max_topk=max_topk,
                max_extra_topk=max_extra_topk,
                max_q_len=max_q_len,
                packed_query=packed_query,
                q_dtype=q_dtype,
                qk_rope_head_dim=qk_rope_head_dim,
                kv_cache_format=kv_cache_format,
                extra_kv_cache_format=extra_kv_cache_format,
                kv_layout=kv_layout,
                use_attention_sinks=use_attention_sinks,
                return_lse=return_lse,
                assume_valid_prefix=assume_valid_prefix,
                output_quant_block_size=output_quant_block_size,
                out_scale_format=out_scale_format,
                num_output_groups=num_output_groups,
                tuning=tuning,
                kv_lora_rank=kv_lora_rank,
            )
        )
        kernel = spec.kernel
        reload_d576 = (
            profile.family == "keep" and kernel._make_config().sparse_d576_reload
        )
        family = profile.family
        splits = profile.split_kv
        max_rows = batch_size * max_q_len
        workspace = self._workspace_buffer
        if workspace is None:
            workspace = torch.empty(byte_end, device=device, dtype=torch.uint8)
        _validate_workspace_buffer(workspace, device=device, required_bytes=byte_end)
        buffers = {
            name: _workspace_section_view(workspace, section)
            for name, section in sections.items()
        }
        # Finalize the joint distribution in the main kernel or its existing
        # split-KV reducer. No per-source outputs or finishing launch are needed.
        if not reload_d576:
            kernel = copy.copy(kernel)
            kernel.output_quant = output_quant
            kernel.assume_valid_prefix = assume_valid_prefix
            if family == "2cta":
                kernel.fuse_sparse_epilogue = splits == 1
                kernel.fuse_sparse_reduction = splits > 1
            else:
                kernel.finalize_output = True
            kernel.direct_sparse = True
            kernel.direct_sparse_capacities = (max_topk, max_extra_topk)
            spec = replace(
                spec,
                kernel=kernel,
                kernel_signature=_mla_kernel_compile_signature(kernel),
            )
        compiled = _get_compiled_mla_decode(spec)
        self._prepared.clear()
        self._state = dict(
            reload_d576=reload_d576,
            device=device,
            batch_size=batch_size,
            num_qo_heads=num_qo_heads,
            max_q_len=max_q_len,
            max_rows=max_rows,
            packed_query=packed_query,
            q_dtype=q_dtype,
            query_dim=512 + qk_rope_head_dim,
            cache_dim=cache_record_size(kernel.kv_cache_format, qk_rope_head_dim),
            extra_cache_dim=cache_record_size(
                kernel.extra_kv_cache_format, qk_rope_head_dim
            ),
            cache_dtype=torch.uint8 if kernel.has_nvfp4_kv else q_dtype,
            kv_cache_format=kernel.kv_cache_format,
            extra_kv_cache_format=kernel.extra_kv_cache_format,
            max_topk=max_topk,
            max_extra_topk=max_extra_topk,
            kv_layout=kv_layout,
            use_attention_sinks=use_attention_sinks,
            return_lse=return_lse,
            output_quant=output_quant,
            buffers=buffers,
            workspace=workspace,
            compiled=compiled,
            assume_valid_prefix=assume_valid_prefix,
            requested_splits=splits,
            default_cl=torch.full(
                (max_rows,), max_extra_topk, device=device, dtype=torch.int32
            ),
            dummy_indices=torch.zeros((max_rows, 1), device=device, dtype=torch.int32),
            dummy_cache=torch.zeros(
                (
                    1,
                    1,
                    cache_record_size(kernel.extra_kv_cache_format, qk_rope_head_dim),
                ),
                device=device,
                dtype=torch.uint8 if kernel.has_nvfp4_kv else q_dtype,
            ),
            default_attention_sinks=torch.full(
                (num_qo_heads,), -torch.inf, device=device, dtype=torch.float32
            ),
        )

        self.workspace_size_bytes = byte_end

    @staticmethod
    def host_scale(value, name):
        """Validate and round a host scalar to the runtime kernel's FP32 ABI."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a Python int or float")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
        try:
            rounded = struct.unpack("f", struct.pack("f", float(value)))[0]
        except OverflowError as error:
            raise ValueError(f"{name} must be representable in float32") from error
        if not math.isfinite(rounded) or rounded <= 0:
            raise ValueError(f"{name} must be positive finite float32")
        return rounded

    def _pool(self, cache, name):
        """Validate one cache pool and expose its physical rows without repacking storage."""
        state = self._state
        dim = state["extra_cache_dim" if name == "extra_kv_cache" else "cache_dim"]
        if not isinstance(cache, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        fmt = state[
            "extra_kv_cache_format" if name == "extra_kv_cache" else "kv_cache_format"
        ]
        if fmt == "fp8" and cache.dtype == torch.float8_e4m3fn:
            cache = cache.view(torch.uint8)
        if cache.ndim == 2:
            cache = cache.unsqueeze(1)
        if cache.ndim == 4:
            axis = 1 if state["kv_layout"] == "HND" else 2
            if cache.shape[axis] != 1:
                raise ValueError(f"{name} requires one KV head")
            cache = cache.squeeze(axis)
        if (
            cache.ndim != 3
            or cache.shape[1] <= 0
            or cache.shape[2] != dim
            or cache.shape[0] <= 0
        ):
            raise ValueError(
                f"{name} must have shape [pages,page_size,{dim}] with nonempty pages"
            )
        page_size = cache.shape[1]
        if cache.device != state["device"] or cache.dtype != state["cache_dtype"]:
            raise ValueError(f"{name} must match the planned device and dtype")
        if (
            cache.stride(-1) != 1
            or cache.stride(1) != dim
            or cache.stride(0) < page_size * dim
            or cache.stride(0) % dim
            or cache.data_ptr() % 16
        ):
            raise ValueError(
                f"{name} requires aligned compact rows and a page stride divisible by {dim} elements"
            )
        row_stride = cache.stride(0) // dim
        rows = (cache.shape[0] - 1) * row_stride + page_size
        if rows >= 0x7FFFFFFF:
            raise ValueError(f"{name} exceeds signed gather coordinate range")
        return cache.as_strided((rows, 1, dim), (dim, dim, 1))

    def _source_metadata(
        self, indices, lengths, capacity, tokens, name, *, rows, validate
    ):
        state = self._state
        if capacity == 0:
            if indices is not None or lengths is not None:
                raise ValueError(f"{name} metadata is not expected by the plan")
            return state["dummy_indices"][:rows], state["default_cl"][:rows]
        if not isinstance(indices, torch.Tensor) or (
            indices.shape != (rows, capacity)
            or indices.dtype != torch.int32
            or indices.device != state["device"]
            or indices.stride(-1) != 1
        ):
            raise ValueError(f"invalid {name} indices shape/dtype/device/stride")
        if not isinstance(lengths, torch.Tensor) or (
            lengths.shape != (rows,)
            or lengths.dtype != torch.int32
            or lengths.device != state["device"]
            or not lengths.is_contiguous()
        ):
            raise ValueError(f"invalid {name} lengths")
        if validate:
            if ((lengths < 0) | (lengths > capacity)).any().item():
                raise ValueError(f"{name} length exceeds capacity")
            active = (
                torch.arange(capacity, device=state["device"])[None, :]
                < lengths[:, None]
            )
            if (active & ((indices < -1) | (indices >= tokens))).any().item():
                raise ValueError(f"{name} active index outside pool")
            if state["assume_valid_prefix"] and (active & (indices < 0)).any().item():
                raise ValueError(
                    f"{name} active prefix contains a hole with assume_valid_prefix=True"
                )
        return indices, lengths

    @flashinfer_experimental_api
    def run(
        self,
        query,
        kv_cache,
        indices,
        lengths,
        extra_kv_cache=None,
        *,
        extra_indices=None,
        extra_lengths=None,
        qo_indptr=None,
        softmax_scale=None,
        output_scale=1.0,
        kv_cache_scale=1.0,
        attention_sinks=None,
        out=None,
        lse=None,
        token_positions=None,
        cos_sin_cache=None,
        out_scale=None,
        validate=True,
    ):
        """Attend to caller-selected storage rows in one or two native KV pools.

        ``indices`` and ``extra_indices`` are CUDA INT32 [T,K] tensors, with
        T equal to the query token count and K equal to the corresponding
        planned max_topk/max_extra_topk. The last dimension is contiguous;
        row strides may include padding. Each index addresses its own pool's
        flattened storage, including physical page strides. The caller performs
        logical-token/page-table conversion and applies causality upstream.
        ``lengths`` and ``extra_lengths`` are contiguous CUDA INT32 [T]. Only
        entries before each length are considered; -1 entries are masked holes.
        Omit both extra tensors when no extra pool was planned. Input widths
        and live lengths need no 128-entry padding: the kernel aligns source
        execution spans internally and bounds all source reads.

        ``validate=True`` (default) checks CUDA metadata contents using host
        reads, which synchronize the device. Serving should establish index,
        length, offset and position bounds upstream and use ``validate=False``
        on the hot path and during CUDA graph capture. Shape, dtype, device,
        stride and alignment checks still run without reading CUDA values.
        Q must already use the planned dtype and include the required query
        RoPE; run() performs neither Q quantization nor query rotation.

        ``softmax_scale`` and ``output_scale`` are positive finite Python
        host scalars, passed as runtime FP32 kernel arguments. They are already combined:
        softmax_scale = attention_scale * q_descale * kv_descale;
        output_scale = output_multiplier * kv_descale. Both pools share that
        KV descale. These multiply QK scores and the weighted stored V values,
        respectively; sink logits are not multiplied by softmax_scale. The
        output multiplier applies before fused RoPE/quantization. ``out_scale``
        instead holds the generated per-block quantization scales.
        Combine scales on the host before each launch. Their values are not
        compilation keys. CUDA graphs capture the supplied values; changing
        them for replay requires graph parameter updates or recapture.

        For ``kv_cache_format="fp8"``, ``kv_cache_scale`` is the independent
        positive host-runtime per-tensor descale of the primary pool. It accepts
        a Python scalar; leave it at the default 1.0
        for other formats. The extra pool is NVFP4 and has no tensor descale.
        The primary scale is applied internally to both K and V; do not also
        fold it into softmax_scale/output_scale.

        Caller-owned inputs and plan workspace must outlive graph replay,
        including attention_sinks, and must not alias writable output
        or workspace storage. Q and native KV are used directly; flattening
        padded pages only creates a tensor view. Required attention reductions
        remain included. Caller-owned LSE must be contiguous FP32 and 16-byte
        aligned, including with validate=False.

        With output quantization enabled, ``token_positions`` is contiguous
        INT32 with the query's batch/token prefix; ``cos_sin_cache`` is aligned
        contiguous FP32[P,64] containing forward [cos(32), sin(32)] rows. Use
        absolute query positions and the same layer-specific table as Q RoPE.
        Positions must be in [0,P); validate=False leaves this to the caller.

        Return (out, out_scale), or (out, out_scale, lse) with return_lse=True.
        Out is the [T,G,C] view (usual strides (C,T*C,1)) of a 16-byte aligned
        contiguous group-major E4M3[G,T,C] tensor, where T is total query
        tokens and C=(H/G)*512, in natural head/channel order. Equivalent
        singleton-axis strides are accepted. A live prefix may retain a
        larger [G,T_capacity,C] allocation; only its live T rows are written. Pass
        ``torch.empty((G,T,C), dtype=torch.float8_e4m3fn).transpose(0,1)`` as
        ``out`` to preallocate. For K=output_quant_block_size
        and N=C/K, FP32 scales have logical shape [T,G,N]; UE8M0 scales have
        shape [T,G,N/4], INT32 packing four block exponent bytes, first in the
        least significant byte (scale=2**(byte-127)). Allocate scale backing as
        16-byte aligned [G,columns,pad4(T)] and use backing.permute(2,0,1)[:T],
        with strides (1,columns*pad4(T),pad4(T)). Padding is not written. The scalar
        ``output_scale`` multiplier remains applied before quantization.

        Preallocate out/out_scale/lse and warm up before graph capture. With
        fusion disabled, the existing BF16 output and return contract apply.
        """
        state = self._state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        if softmax_scale is None:
            softmax_scale = state["query_dim"] ** -0.5
        lse_arg = lse
        prepared_key = (
            kv_cache.data_ptr(),
            kv_cache.shape,
            kv_cache.stride(),
            kv_cache.dtype,
            None
            if extra_kv_cache is None
            else (
                extra_kv_cache.data_ptr(),
                extra_kv_cache.shape,
                extra_kv_cache.stride(),
                extra_kv_cache.dtype,
            ),
            None if attention_sinks is None else attention_sinks.data_ptr(),
            softmax_scale,
            output_scale,
        )
        if (
            not validate
            and out is not None
            and lse is None
            and qo_indptr is None
            and token_positions is None
            and cos_sin_cache is None
            and out_scale is None
            and type(kv_cache_scale) in (int, float)
            and kv_cache_scale == 1.0
        ):
            prepared = self._prepared.get(prepared_key)
            if prepared is not None:
                return self._run_prepared(
                    prepared, query, indices, lengths, extra_indices, extra_lengths, out
                )
        _validate_query(
            query,
            packed_query=state["packed_query"],
            device=state["device"],
            batch_size=state["batch_size"],
            num_qo_heads=state["num_qo_heads"],
            max_q_len=state["max_q_len"],
            q_dtype=state["q_dtype"],
            query_dim=state["query_dim"],
        )
        prefix = tuple(query.shape[:-2])
        rows = math.prod(prefix)
        if state["packed_query"]:
            _validate_qo_indptr(
                qo_indptr, device=state["device"], batch_size=state["batch_size"]
            )
            if validate:
                maximum, total, _ = _derive_max_seq_len_q(
                    qo_indptr, batch_size=state["batch_size"]
                )
                if total != rows or maximum > state["max_q_len"]:
                    raise ValueError("invalid packed query offsets")
        elif qo_indptr is not None:
            raise ValueError("fixed queries do not accept qo_indptr")

        primary = self._pool(kv_cache, "kv_cache")
        if extra_kv_cache is None:
            if (
                state["max_extra_topk"]
                or extra_indices is not None
                or extra_lengths is not None
            ):
                raise ValueError("extra pool and indices are required by the plan")
            extra = state["dummy_cache"]
        else:
            extra = self._pool(extra_kv_cache, "extra_kv_cache")

        si, sl = self._source_metadata(
            indices,
            lengths,
            state["max_topk"],
            primary.shape[0],
            "primary",
            rows=rows,
            validate=validate,
        )
        ci, cl = self._source_metadata(
            extra_indices,
            extra_lengths,
            state["max_extra_topk"],
            extra.shape[0],
            "extra",
            rows=rows,
            validate=validate,
        )
        if state["use_attention_sinks"] != (attention_sinks is not None):
            raise ValueError("attention_sinks presence must match the plan")
        if attention_sinks is None:
            attention_sinks = state["default_attention_sinks"]
        if (
            attention_sinks.shape != (state["num_qo_heads"],)
            or attention_sinks.dtype != torch.float32
            or attention_sinks.device != state["device"]
            or not attention_sinks.is_contiguous()
        ):
            raise ValueError("attention_sinks must be contiguous CUDA FP32[H]")
        if validate and torch.isnan(attention_sinks).any().item():
            raise ValueError("attention_sinks must not contain NaN")
        softmax_scale = self.host_scale(softmax_scale, "softmax_scale")
        output_scale = self.host_scale(output_scale, "output_scale")
        kv_cache_scale = self.host_scale(kv_cache_scale, "kv_cache_scale")
        if state["kv_cache_format"] != "fp8" and kv_cache_scale != 1.0:
            raise ValueError("kv_cache_scale is only supported for an FP8 primary pool")
        quant = state["output_quant"]
        output_quant_args = None
        if quant is not None:
            out, out_scale, output_quant_args = _prepare_sparse_output_quant(
                quant, query, token_positions, cos_sin_cache, out, out_scale, validate
            )
        elif any(v is not None for v in (token_positions, cos_sin_cache, out_scale)):
            raise ValueError(
                "plan output quantization before passing RoPE/scale tensors"
            )
        elif out is None:
            out = torch.empty(
                (*prefix, state["num_qo_heads"], 512),
                device=state["device"],
                dtype=torch.bfloat16,
            )
        if quant is None:
            _validate_out(
                out,
                device=state["device"],
                batch_size=state["batch_size"],
                num_qo_heads=state["num_qo_heads"],
                max_q_len=state["max_q_len"],
                packed_query=state["packed_query"],
                total_q=rows if state["packed_query"] else None,
                out_dtype=torch.bfloat16,
            )
        if lse is None:
            # A returned result must survive the next call on this wrapper.
            # Use plan-owned scratch only when the LSE is not exposed.
            lse = (
                torch.empty(
                    (*prefix, state["num_qo_heads"]),
                    device=state["device"],
                    dtype=torch.float32,
                )
                if state["return_lse"]
                else state["buffers"]["public_lse"][:rows].view(
                    *prefix, state["num_qo_heads"]
                )
            )
        if (
            lse.shape != (*prefix, state["num_qo_heads"])
            or lse.dtype != torch.float32
            or lse.device != state["device"]
            or not lse.is_contiguous()
            or lse.data_ptr() % 16
        ):
            raise ValueError(
                "lse must be contiguous, 16-byte-aligned FP32 with one value per query/head"
            )
        sparse_inputs = (
            si,
            ci,
            sl,
            cl,
            attention_sinks,
            kv_cache_scale,
        )
        if rows and state["reload_d576"]:
            core = state["buffers"]["core"]
            # One token per batch row through the 1CTA page-table ABI; 128 heads
            # are two physical query positions of 64 heads.
            phys = (rows, state["num_qo_heads"] // 64, 64)
            with torch.cuda.device(state["device"]):
                _launch_mla_decode(
                    _MLARuntime(
                        query.view(*phys, _D576_HEAD_DIM),
                        primary,
                        out.view(*phys, _D576_V_DIM),
                        primary.shape[0],
                        softmax_scale,
                        output_scale,
                    ),
                    block_tables=si,
                    seq_lens_kv=sl,
                    qo_indptr=None,
                    packed_query=False,
                    kv_lora_rank=512,
                    split_kv=state["requested_splits"],
                    workspace=_MLAWorkspaceViews(
                        core if core.numel() else None, lse.view(*phys)
                    ),
                    compiled=state["compiled"],
                    max_q_len=phys[1],
                )
        elif rows:
            buffers = state["buffers"]
            # Quantized stores use output_quant_args, including group/bucket
            # strides. The core O view supplies compact launch geometry only.
            core_out = (
                out.view(rows, 1, state["num_qo_heads"], 512)
                if quant is None
                else output_quant_args[0][: rows * state["num_qo_heads"] * 512].view(
                    rows, 1, state["num_qo_heads"], 512
                )
            )
            _launch_mla_decode(
                _MLARuntime(
                    query.view(rows, 1, state["num_qo_heads"], state["query_dim"]),
                    primary,
                    core_out,
                    primary.shape[0],
                    softmax_scale,
                    output_scale,
                    extra_cache=extra,
                    sparse_inputs=sparse_inputs,
                    output_quant_args=output_quant_args,
                ),
                block_tables=si,
                seq_lens_kv=sl,
                qo_indptr=None,
                packed_query=False,
                kv_lora_rank=512,
                split_kv=state["requested_splits"],
                workspace=_MLAWorkspaceViews(
                    buffers["core"] if buffers["core"].numel() else None,
                    lse.view(rows, 1, state["num_qo_heads"]),
                ),
                compiled=state["compiled"],
                max_q_len=1,
            )
        if (
            rows
            and quant is None
            and not state["packed_query"]
            and not state["return_lse"]
            and state["max_q_len"] == 1
            and state["query_dim"] == 512
            and state["kv_cache_format"] == "native"
            and state["extra_kv_cache_format"] == "native"
            and lse_arg is None
        ):
            # Static for this (plan, pools, scales): pool views, scalars, sinks, split,
            # workspace. The next validate=False launch with the same pools re-forms
            # only the query / index / output views (see _run_prepared).
            if len(self._prepared) >= _PREPARED_LAUNCH_LIMIT:
                self._prepared.clear()
            self._prepared[prepared_key] = dict(
                primary=primary,
                extra=extra,
                primary_rows=primary.shape[0],
                c_latent=primary.permute(1, 2, 0),
                c_rope=extra.permute(1, 2, 0),
                sinks=attention_sinks,
                bmm=(softmax_scale, output_scale),
                ks=state["max_topk"],
                kc=state["max_extra_topk"],
                dummy_indices=state["dummy_indices"],
                default_cl=state["default_cl"],
            )
        if quant is not None:
            return (out, out_scale, lse) if state["return_lse"] else (out, out_scale)
        return (out, lse) if state["return_lse"] else out

    def _run_prepared(self, prepared, query, indices, lengths, extra_indices, extra_lengths, out):
        """Launch with the static arguments recorded by the last full run().

        Only the per-call tensors are checked and viewed: query and output
        geometry, index/length shapes, dtypes and strides. Pool geometry,
        scales and sinks were validated by the full path that prepared this
        launch, and the caller passed validate=False. Byte-identical to the
        full path for the same arguments.
        """
        state = self._state
        rows, heads, device = state["batch_size"], state["num_qo_heads"], state["device"]
        if (
            query.shape != (rows, 1, heads, 512)
            or query.dtype != state["q_dtype"]
            or query.device != device
            or not query.is_contiguous()
            or query.data_ptr() % 16
        ):
            raise ValueError(
                "prepared launch: query must be a contiguous, 16-byte aligned "
                "[batch, 1, heads, 512] tensor of the planned dtype on the planned device"
            )
        if (
            out.numel() != rows * heads * 512
            or out.dtype != torch.bfloat16
            or out.device != device
            or not out.is_contiguous()
            or out.data_ptr() % 16
        ):
            raise ValueError(
                "prepared launch: out must be a contiguous, 16-byte aligned BF16 tensor "
                "holding one 512-wide row per query head"
            )
        if (
            indices.shape != (rows, prepared["ks"])
            or indices.dtype != torch.int32
            or indices.device != device
            or indices.stride(-1) != 1
            or lengths.shape != (rows,)
            or lengths.dtype != torch.int32
            or lengths.device != device
            or not lengths.is_contiguous()
        ):
            raise ValueError("prepared launch: invalid primary indices/lengths")
        if prepared["kc"]:
            if (
                not isinstance(extra_indices, torch.Tensor)
                or not isinstance(extra_lengths, torch.Tensor)
                or extra_indices.shape != (rows, prepared["kc"])
                or extra_indices.dtype != torch.int32
                or extra_indices.device != device
                or extra_indices.stride(-1) != 1
                or extra_lengths.shape != (rows,)
                or extra_lengths.dtype != torch.int32
                or extra_lengths.device != device
                or not extra_lengths.is_contiguous()
            ):
                raise ValueError("prepared launch: invalid extra indices/lengths")
            ci, cl = extra_indices, extra_lengths
        else:
            if extra_indices is not None or extra_lengths is not None:
                raise ValueError("prepared launch: extra indices without a planned extra pool")
            ci, cl = prepared["dummy_indices"][:rows], prepared["default_cl"][:rows]
        buffers = state["buffers"]
        sinks = prepared["sinks"]
        q_kernel = query.permute(2, 3, 1, 0)
        lse_kernel = buffers["public_lse"][:rows].view(rows, 1, heads).permute(2, 1, 0)
        bmm1, bmm2 = prepared["bmm"]
        state["compiled"](
            q_kernel,
            q_kernel,
            prepared["c_latent"],
            prepared["c_rope"],
            indices.transpose(0, 1),
            out.view(rows, 1, heads, 512).permute(2, 3, 1, 0),
            lse_kernel,
            buffers["core"] if buffers["core"].numel() else None,
            state["requested_splits"],
            lengths,
            None,
            (indices, ci, lengths, cl, sinks, 1.0),
            bmm1,
            bmm2,
            None,
            1,
        )
        return out


def get_prims_ts_sparse_mla_decode_workspace_size(*plan_args, **plan_kwargs):
    """Return workspace bytes using the same arguments as wrapper.plan().

    Resolves the default kernel geometry without allocating GPU scratch or
    compiling a kernel. Bind the resulting byte buffer to the wrapper constructor
    and call plan() before graph capture; run() is the prepared standalone launch.
    """
    _, _, _, _, size_bytes, _, _ = _resolve_sparse_mla_plan(*plan_args, **plan_kwargs)
    return size_bytes


@flashinfer_experimental_api
def batch_sparse_mla_decode_with_paged_kv_cache(
    query,
    kv_cache,
    indices,
    lengths,
    extra_kv_cache=None,
    *,
    extra_indices=None,
    extra_lengths=None,
    qo_indptr=None,
    max_q_len=None,
    kv_layout="NHD",
    kv_cache_format="native",
    extra_kv_cache_format=None,
    softmax_scale=None,
    output_scale=1.0,
    kv_cache_scale=1.0,
    attention_sinks=None,
    out=None,
    lse=None,
    return_lse=False,
    workspace_buffer=None,
    assume_valid_prefix=False,
    output_quant_block_size=None,
    out_scale_format="ue8m0",
    num_output_groups=None,
    token_positions=None,
    cos_sin_cache=None,
    out_scale=None,
    tuning=None,
):
    """Eager plan-and-run helper using separate source indices and lengths.

    Use a planned wrapper for CUDA Graph replay. Index conversion and scale
    composition remain external. softmax_scale includes attention/Q/K descales;
    output_scale includes the V descale and optional output multiplier.
    Source contracts, output quantization arguments, ``tuning`` and return
    layouts follow
    :class:`BatchSparseMLADecodePagedTSWrapper`.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "plan BatchSparseMLADecodePagedTSWrapper before CUDA Graph capture"
        )
    packed = query.ndim == 3
    if packed:
        if qo_indptr is None:
            raise ValueError("packed queries require qo_indptr")
        batch = qo_indptr.numel() - 1
        if max_q_len is None:
            max_q_len = max(1, int((qo_indptr[1:] - qo_indptr[:-1]).max().item()))
    else:
        batch = query.shape[0]
        max_q_len = query.shape[1] if max_q_len is None else max_q_len
    wrapper = BatchSparseMLADecodePagedTSWrapper(workspace_buffer)
    wrapper.plan(
        query.device,
        batch,
        query.shape[-2],
        max_topk=indices.shape[-1],
        max_extra_topk=0 if extra_indices is None else extra_indices.shape[-1],
        max_q_len=max_q_len,
        packed_query=packed,
        q_dtype=query.dtype,
        qk_rope_head_dim=query.shape[-1] - 512,
        kv_cache_format=kv_cache_format,
        extra_kv_cache_format=extra_kv_cache_format,
        kv_layout=kv_layout,
        use_attention_sinks=attention_sinks is not None,
        return_lse=return_lse,
        assume_valid_prefix=assume_valid_prefix,
        output_quant_block_size=output_quant_block_size,
        out_scale_format=out_scale_format,
        num_output_groups=num_output_groups,
        tuning=tuning,
    )
    return wrapper.run(
        query,
        kv_cache,
        indices,
        lengths,
        extra_kv_cache,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
        qo_indptr=qo_indptr,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        kv_cache_scale=kv_cache_scale,
        attention_sinks=attention_sinks,
        out=out,
        lse=lse,
        token_positions=token_positions,
        cos_sin_cache=cos_sin_cache,
        out_scale=out_scale,
    )
