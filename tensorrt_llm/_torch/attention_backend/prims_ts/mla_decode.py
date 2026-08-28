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
from dataclasses import dataclass
import functools
from typing import Any, Literal, Optional, cast

import torch

from flashinfer.api_logging import flashinfer_api

from ._tensor_aliasing import (
    _validate_out_does_not_overlap_inputs,
    _validate_tensor_does_not_overlap_inputs,
)
from .decode import (
    _WorkspaceSection,
    _align_up,
    _append_workspace_section,
    _dtype_key,
    _resolve_cuda_device,
    _validate_16byte_alignment,
    _validate_mask,
    _validate_page_size,
    _validate_positive_int,
    _validate_runtime_device,
    _validate_scale,
    _validate_workspace_buffer,
    _workspace_section_view,
)


_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 2"
_MLA_LATENT_DIM = 512
_MLA_ROPE_DIM = 64
_MLA_QUERY_DIM = _MLA_LATENT_DIM + _MLA_ROPE_DIM
_SUPPORTED_INPUT_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)
_SUPPORTED_OUTPUT_DTYPES = (torch.bfloat16,)
_INT32_MAX = 2**31 - 1
# The largest public 1CTA schedule can pad one K/V split group across 128
# splits, two 128-token K/V instructions apiece. Reserve that complete span so
# every padded tile boundary remains representable as signed Int32.
_MLA_MAX_KV_COORDINATE_SPAN = 128 * 2 * 128
_MLA_MAX_KV_LEN = _INT32_MAX - (_MLA_MAX_KV_COORDINATE_SPAN - 1)


@dataclass(frozen=True)
class _MLADecodeLaunchSpec:
    """Automatic MLA kernel selection and scratch for one semantic key."""

    kernel: Any
    qkv_dtype: object
    output_dtype: object
    policy: tuple[tuple[str, object], ...]
    kernel_workspace_bytes: int
    split_kv: int


@dataclass(frozen=True)
class _MLAWorkspaceLayout:
    """Private MLA scratch layout; only ``total_bytes`` is public."""

    kernel_workspace: _WorkspaceSection
    lse: _WorkspaceSection
    total_bytes: int


@dataclass(frozen=True)
class _MLAWorkspaceViews:
    kernel_workspace: Optional[torch.Tensor]
    lse: torch.Tensor


@dataclass(frozen=True)
class _MLARuntime:
    query: torch.Tensor
    normalized_cache: torch.Tensor
    out: torch.Tensor
    bmm1_scale: float
    bmm2_scale: float


def _make_mla_workspace_layout(
    kernel_workspace_bytes: int,
    batch_size: int,
    num_heads: int,
    max_seq_len_q: int = 1,
) -> _MLAWorkspaceLayout:
    kernel_workspace, byte_end = _append_workspace_section(
        0, (kernel_workspace_bytes,), torch.int8
    )
    lse, byte_end = _append_workspace_section(
        byte_end, (batch_size, max_seq_len_q, num_heads), torch.float32
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
        lse=_workspace_section_view(workspace_buffer, layout.lse),
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


def _validate_mla_query_head_extent(
    *,
    batch_size: int,
    num_heads: int,
    max_seq_len_q: int,
    total_q: Optional[int] = None,
) -> None:
    """Keep fixed-capacity and packed query-head coordinates in signed Int32."""
    _validate_mla_int32_extent(
        batch_size * max_seq_len_q * num_heads,
        "batch_size * max_seq_len_q * num_heads",
    )
    if total_q is not None:
        _validate_mla_int32_extent(
            total_q * num_heads,
            "total_q * num_heads",
        )


def _validate_mla_policy_coordinate_span(
    policy: tuple[tuple[str, object], ...],
) -> None:
    """Keep the host K/V bound coupled to the automatically selected policy."""
    resolved = dict(policy)
    span = (
        int(cast(int, resolved["tile_size_kv"]))
        * int(cast(int, resolved["num_insts_kv"]))
        * max(int(cast(int, resolved["split_kv"])), 1)
    )
    if span > _MLA_MAX_KV_COORDINATE_SPAN:
        raise RuntimeError(
            "MLA Int32 extent safety assumes a padded K/V coordinate span no "
            f"larger than {_MLA_MAX_KV_COORDINATE_SPAN}, got {span}"
        )


def _validate_mla_dtype_pair(
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> None:
    _dtype_key(q_dtype)
    _dtype_key(kv_dtype)
    _dtype_key(output_dtype)
    if q_dtype != kv_dtype:
        raise NotImplementedError(
            "attention-ts MLA decode requires query and KV cache to use the "
            f"same dtype; got {q_dtype} and {kv_dtype}"
        )
    if q_dtype not in _SUPPORTED_INPUT_DTYPES:
        raise NotImplementedError(
            f"attention-ts MLA decode supports BF16 and FP8-E4M3 input; got {q_dtype}"
        )
    if output_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise NotImplementedError(
            "attention-ts MLA decode currently supports BF16 output only; "
            f"got {output_dtype}"
        )


def _validate_int32_cuda_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    ndim: int,
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
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if require_16byte_alignment:
        _validate_16byte_alignment(tensor, name)


def _validate_mla_metadata(
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[torch.device, int, int]:
    _validate_int32_cuda_tensor(block_tables, "block_tables", ndim=2)
    _validate_int32_cuda_tensor(seq_lens, "seq_lens", ndim=1)
    if block_tables.device != seq_lens.device:
        raise ValueError("block_tables and seq_lens must be on the same device")
    batch_size = int(seq_lens.numel())
    if batch_size <= 0:
        raise ValueError("seq_lens must contain at least one request")
    if block_tables.shape[0] != batch_size:
        raise ValueError(
            "block_tables must have one row per request: expected "
            f"{batch_size}, got {block_tables.shape[0]}"
        )
    max_num_pages = int(block_tables.shape[1])
    if max_num_pages <= 0:
        raise ValueError("block_tables must contain at least one page column")
    _validate_mla_int32_extent(batch_size, "batch_size")
    _validate_mla_int32_extent(int(block_tables.numel()), "block_tables elements")
    return seq_lens.device, batch_size, max_num_pages


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


def _derive_max_seq_len_q(
    qo_indptr: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[int, int, tuple[int, ...]]:
    """Validate cumulative offsets and derive their maximum positive delta.

    This helper intentionally synchronizes and is therefore used only by the
    wrapper planning path when the caller omits an explicit static bound.
    """

    offsets = [int(value) for value in qo_indptr.tolist()]
    if len(offsets) != batch_size + 1:
        raise ValueError("qo_indptr must contain batch_size + 1 offsets")
    if offsets[0] != 0:
        raise ValueError("qo_indptr must start at 0")
    q_lengths = tuple(
        end - start for start, end in zip(offsets[:-1], offsets[1:], strict=True)
    )
    if any(length <= 0 for length in q_lengths):
        raise ValueError("qo_indptr must be strictly increasing")
    return max(q_lengths), offsets[-1], q_lengths


def _resolve_max_seq_len_q_alias(
    *,
    seq_len_q: Optional[int],
    max_seq_len_q: Optional[int],
    default: Optional[int],
) -> Optional[int]:
    """Resolve the legacy fixed-Q name and the explicit static-bound name."""

    legacy_bound = (
        _validate_positive_int(seq_len_q, "seq_len_q")
        if seq_len_q is not None
        else None
    )
    explicit_bound = (
        _validate_positive_int(max_seq_len_q, "max_seq_len_q")
        if max_seq_len_q is not None
        else None
    )
    if (
        legacy_bound is not None
        and explicit_bound is not None
        and legacy_bound != explicit_bound
    ):
        raise ValueError(
            "seq_len_q and max_seq_len_q must agree when both are provided: "
            f"got {legacy_bound} and {explicit_bound}"
        )
    if explicit_bound is not None:
        return explicit_bound
    if legacy_bound is not None:
        return legacy_bound
    return default


def _validate_query(
    query: torch.Tensor,
    *,
    packed_query: bool = False,
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
    num_heads: Optional[int] = None,
    max_seq_len_q: Optional[int] = None,
    q_dtype: Optional[torch.dtype] = None,
) -> None:
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor")
    expected_rank = 3 if packed_query else 4
    if query.ndim != expected_rank:
        expected_shape = "[total_q, H, 576]" if packed_query else "[B, SQ, H, 576]"
        raise ValueError(f"query must have shape {expected_shape}")
    if any(int(extent) <= 0 for extent in query.shape[:-1]):
        raise ValueError("query row and head extents must be positive")
    if query.shape[-1] != _MLA_QUERY_DIM:
        raise ValueError(
            f"query last dimension must be {_MLA_QUERY_DIM}, got {query.shape[-1]}"
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
    if num_heads is not None and query.shape[head_axis] != num_heads:
        raise ValueError(
            "query head count must match the plan "
            f"({num_heads}), got {query.shape[head_axis]}"
        )
    if max_seq_len_q is not None:
        if packed_query:
            if batch_size is None:
                raise ValueError("batch_size is required to validate packed query")
            total_q = int(query.shape[0])
            if total_q < batch_size or total_q > batch_size * max_seq_len_q:
                raise ValueError(
                    "packed query total rows must be within "
                    f"[{batch_size}, {batch_size * max_seq_len_q}], got {total_q}"
                )
        elif query.shape[1] != max_seq_len_q:
            raise ValueError(
                "fixed query length must equal the planned max_seq_len_q "
                f"({max_seq_len_q}), got {query.shape[1]}"
            )
        if batch_size is not None and num_heads is not None:
            _validate_mla_query_head_extent(
                batch_size=batch_size,
                num_heads=num_heads,
                max_seq_len_q=max_seq_len_q,
                total_q=int(query.shape[0]) if packed_query else None,
            )
    if q_dtype is not None and query.dtype != q_dtype:
        raise ValueError(
            f"query dtype must match the plan ({q_dtype}), got {query.dtype}"
        )
    if not query.is_contiguous():
        layout = "[total_q, H, 576]" if packed_query else "[B, SQ, H, 576]"
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


def _validate_out(
    out: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    num_heads: int,
    max_seq_len_q: int,
    packed_query: bool,
    total_q: Optional[int] = None,
    output_dtype: torch.dtype,
) -> None:
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if packed_query:
        if total_q is None:
            raise ValueError("total_q is required to validate packed output")
        expected_shape: tuple[int, ...]
        expected_shape = (total_q, num_heads, _MLA_LATENT_DIM)
    else:
        expected_shape = (batch_size, max_seq_len_q, num_heads, _MLA_LATENT_DIM)
    if out.shape != expected_shape:
        raise ValueError(
            f"out must have shape {expected_shape}, got {tuple(out.shape)}"
        )
    if out.dtype != output_dtype:
        raise ValueError(f"out must have dtype {output_dtype}, got {out.dtype}")
    if out.device != device:
        raise ValueError(f"out must be on {device}, got {out.device}")
    if not out.is_contiguous():
        layout = "[total_q, H, 512]" if packed_query else "[B, SQ, H, 512]"
        raise ValueError(f"out must be compact in {layout} layout")
    _validate_16byte_alignment(out, "out")


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


@functools.cache
def _resolve_mla_decode_launch_spec(
    device_index: int,
    batch_size: int,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    max_kv_len: int,
    q_dtype_key: str,
    kv_dtype_key: str,
    output_dtype_key: str,
    mask_type: str,
    seq_len_q: int = 1,
):
    """Resolve and cache MLA policy/workspace without compiling."""

    max_kv_len = _validate_mla_max_kv_len(max_kv_len, "max_kv_len")

    import cutlass
    import cutlass.utils as cutlass_utils
    from cuda.bindings import driver as cuda_drv

    from .kernels.mla_decode.kernel_policy import (
        resolve_mla_kernel_policy,
        select_mla_ts_kernel,
    )
    from .kernels.mla_decode.throughput_2cta.config import (
        compute_split_kv,
        compute_workspace_size as compute_2cta_workspace_size,
    )
    from .kernels.mla_decode.throughput_2cta.kernel import MlaDecodeTs
    from .kernels.mla_decode.throughput_latency_1cta.config import (
        GroupsTokensHeadsLaunchShape,
        auto_tile_size_q_for_mla_gen,
        compute_workspace_size as compute_1cta_workspace_size,
        fp8_q16_extended_family_probe_split_kv,
        resolve_auto_mla_gen_groups_tokens_heads_q_shape,
        resolve_runtime_cluster_reduction_mode,
        wave_fill_split_kv,
    )
    from .kernels.mla_decode.throughput_latency_1cta.kernel import (
        ThroughputLatencyMlaDecodeTs,
    )

    if q_dtype_key != kv_dtype_key:
        raise ValueError("the cached TS MLA compiler requires one QKV dtype")
    seq_len_q = _validate_positive_int(seq_len_q, "seq_len_q")
    _validate_mla_query_head_extent(
        batch_size=batch_size,
        num_heads=num_heads,
        max_seq_len_q=seq_len_q,
    )
    _validate_mla_dims(kv_lora_rank, qk_rope_head_dim)
    qkv_dtype_name = _kernel_dtype_name(q_dtype_key)
    output_dtype_name = _kernel_dtype_name(output_dtype_key)
    dtype_map = {
        "bf16": cutlass.BFloat16,
        "e4m3": cutlass.Float8E4M3FN,
    }
    qkv_dtype = dtype_map[qkv_dtype_name]
    output_dtype = dtype_map[output_dtype_name]

    with torch.cuda.device(device_index):
        plan_stream = cuda_drv.CUstream(
            torch.cuda.current_stream(device_index).cuda_stream
        )
        hardware_info = cutlass_utils.HardwareInfo(device_index)
        max_active_one_cta_clusters = hardware_info.get_max_active_clusters(
            1, plan_stream
        )
        max_active_two_cta_clusters = hardware_info.get_max_active_clusters(
            2, plan_stream
        )

        two_cta_launch_shape = GroupsTokensHeadsLaunchShape.for_tile(
            num_heads, seq_len_q, 128
        )
        two_cta_split_kv = compute_split_kv(
            batch_size=batch_size,
            seq_len_q=two_cta_launch_shape.seq_len_q,
            seq_len_kv=max_kv_len,
            mma_qk_tiler_mn=(128, 128),
            max_active_blocks=max_active_two_cta_clusters * 2,
        )
        two_cta_cluster_work = (
            batch_size * two_cta_launch_shape.seq_len_q * max(two_cta_split_kv, 1)
        )

        one_cta_launch_shape = resolve_auto_mla_gen_groups_tokens_heads_q_shape(
            batch_size=batch_size,
            num_heads_q=num_heads,
            seq_len_q=seq_len_q,
            seq_len_kv=max_kv_len,
            qkv_dtype=qkv_dtype_name,
            max_active_clusters=max_active_one_cta_clusters,
        )
        family_probe_decision = None
        family_probe_split_kv = None
        family_probe_launch_shape = None
        family_probe_is_extended_fp8_swaps = False
        one_cta_work = None
        use_established_family_probe = (
            num_heads * seq_len_q > 64
            and two_cta_cluster_work * 4 <= max_active_two_cta_clusters
        )
        if use_established_family_probe:
            family_probe_launch_shape = one_cta_launch_shape
            initial_probe = select_mla_ts_kernel(
                requested_policy="throughput_latency_1cta",
                batch_size=batch_size,
                num_heads=family_probe_launch_shape.num_heads_q,
                seq_len_q=family_probe_launch_shape.seq_len_q,
                seq_len_k=max_kv_len,
                latent_dim=kv_lora_rank,
                rope_dim=qk_rope_head_dim,
                page_size=page_size,
                dtype=qkv_dtype_name,
                out_dtype=output_dtype_name,
                throughput_latency_profile=None,
                throughput_latency_tile_size_q=(family_probe_launch_shape.tile_size_q),
                max_active_clusters=max_active_one_cta_clusters,
                throughput_latency_split_kv=None,
                throughput_latency_persistent=None,
            )
            if initial_probe.implementation_ready and initial_probe.config is not None:
                family_probe_split_kv = wave_fill_split_kv(
                    batch_size=batch_size,
                    num_heads_q=family_probe_launch_shape.num_heads_q,
                    seq_len_q=family_probe_launch_shape.seq_len_q,
                    seq_len_kv=max_kv_len,
                    latent_dim=kv_lora_rank,
                    max_active_clusters=max_active_one_cta_clusters,
                    tile_size_q=int(initial_probe.config.tile_size_q),
                )
                family_probe_decision = select_mla_ts_kernel(
                    requested_policy="throughput_latency_1cta",
                    batch_size=batch_size,
                    num_heads=family_probe_launch_shape.num_heads_q,
                    seq_len_q=family_probe_launch_shape.seq_len_q,
                    seq_len_k=max_kv_len,
                    latent_dim=kv_lora_rank,
                    rope_dim=qk_rope_head_dim,
                    page_size=page_size,
                    dtype=qkv_dtype_name,
                    out_dtype=output_dtype_name,
                    throughput_latency_profile=None,
                    throughput_latency_tile_size_q=(
                        family_probe_launch_shape.tile_size_q
                    ),
                    max_active_clusters=max_active_one_cta_clusters,
                    throughput_latency_split_kv=family_probe_split_kv,
                    throughput_latency_persistent=None,
                )
                family_cfg = family_probe_decision.config
                if family_cfg is not None:
                    one_cta_work = (
                        family_cfg.batch_size
                        * family_cfg.num_ctas_per_seq_q
                        * family_cfg.num_ctas_for_all_heads
                        * family_cfg.num_ctas_per_seq_kv
                        * family_cfg.num_ctas_per_head_dim
                    )
        elif (
            num_heads * seq_len_q > 64
            and qkv_dtype_name == "e4m3"
            and two_cta_cluster_work <= max_active_two_cta_clusters
        ):
            # The established probe intentionally covers only a severely
            # underfilled 2CTA grid.  A second FP8-only probe considers Q16
            # Swaps when both candidates fit one wave and the Q16 local K span
            # can be held to two steady-state steps.  This avoids switching
            # saturated or long-local-K shapes that benefit from the grouped
            # M128 2CTA schedule.
            probe_tile_size_q = auto_tile_size_q_for_mla_gen(
                batch_size=batch_size,
                num_heads_q=num_heads,
                seq_len_q=seq_len_q,
                seq_len_kv=max_kv_len,
                multi_processor_count=max_active_one_cta_clusters,
            )
            if probe_tile_size_q == 16:
                extended_launch_shape = GroupsTokensHeadsLaunchShape.for_tile(
                    num_heads,
                    seq_len_q,
                    probe_tile_size_q,
                )
                extended_split_kv = fp8_q16_extended_family_probe_split_kv(
                    batch_size=batch_size,
                    num_heads_q=extended_launch_shape.num_heads_q,
                    seq_len_q=extended_launch_shape.seq_len_q,
                    seq_len_kv=max_kv_len,
                    max_active_clusters=max_active_one_cta_clusters,
                )
                if extended_split_kv is not None:
                    extended_decision = select_mla_ts_kernel(
                        requested_policy="throughput_latency_1cta",
                        batch_size=batch_size,
                        num_heads=extended_launch_shape.num_heads_q,
                        seq_len_q=extended_launch_shape.seq_len_q,
                        seq_len_k=max_kv_len,
                        latent_dim=kv_lora_rank,
                        rope_dim=qk_rope_head_dim,
                        page_size=page_size,
                        dtype=qkv_dtype_name,
                        out_dtype=output_dtype_name,
                        throughput_latency_profile=None,
                        throughput_latency_tile_size_q=probe_tile_size_q,
                        max_active_clusters=max_active_one_cta_clusters,
                        throughput_latency_split_kv=extended_split_kv,
                        throughput_latency_persistent=None,
                    )
                    extended_cfg = extended_decision.config
                    if (
                        extended_decision.implementation_ready
                        and extended_cfg is not None
                        and extended_cfg.kernel_variant == "swaps_mma_ab"
                        and extended_cfg.tile_size_q == 16
                    ):
                        family_probe_decision = extended_decision
                        family_probe_split_kv = extended_split_kv
                        family_probe_launch_shape = extended_launch_shape
                        family_probe_is_extended_fp8_swaps = True
                        one_cta_work = (
                            extended_cfg.batch_size
                            * extended_cfg.num_ctas_per_seq_q
                            * extended_cfg.num_ctas_for_all_heads
                            * extended_cfg.num_ctas_per_seq_kv
                            * extended_cfg.num_ctas_per_head_dim
                        )

        requested_policy, policy_source = resolve_mla_kernel_policy(
            None,
            num_heads,
            seq_len_q,
            one_cta_work=one_cta_work,
            one_cta_capacity=(
                max_active_one_cta_clusters if one_cta_work is not None else None
            ),
            two_cta_cluster_work=(
                two_cta_cluster_work if one_cta_work is not None else None
            ),
            two_cta_cluster_capacity=(
                max_active_two_cta_clusters if one_cta_work is not None else None
            ),
            one_cta_is_extended_fp8_swaps=(family_probe_is_extended_fp8_swaps),
        )
        use_throughput_latency = requested_policy == "throughput_latency_1cta"

        kernel: Any
        if use_throughput_latency:
            max_active_clusters = max_active_one_cta_clusters
            launch_shape = family_probe_launch_shape or one_cta_launch_shape
            decision = family_probe_decision
            if decision is None:
                family_probe_split_kv = None
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
                raise NotImplementedError(decision.reason)
            reduction_mode = resolve_runtime_cluster_reduction_mode(
                decision.config,
                reduction_mode=None,
                hardware_info=hardware_info,
                stream=plan_stream,
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
                groups_tokens_heads_q_ratio=launch_shape.ratio,
                logical_num_heads=num_heads,
                logical_seq_len_q=seq_len_q,
                tile_size_q=launch_shape.tile_size_q,
                explicit_split_kv=family_probe_split_kv,
                explicit_persistent=None,
                mask_type=mask_type,
            )
            final_cfg = kernel._make_config()
            split_kv = int(final_cfg.num_ctas_per_seq_kv)
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
                num_heads=num_heads,
                seq_len_q=seq_len_q,
                seq_len_k=max_kv_len,
                latent_dim=kv_lora_rank,
                rope_dim=qk_rope_head_dim,
                page_size=page_size,
                dtype=qkv_dtype_name,
                out_dtype=output_dtype_name,
                throughput_latency_profile=None,
                throughput_latency_tile_size_q=None,
                max_active_clusters=max_active_one_cta_clusters,
                throughput_latency_split_kv=None,
                throughput_latency_persistent=None,
            )
            if not decision.implementation_ready:
                raise NotImplementedError(decision.reason)
            split_kv = two_cta_split_kv
            work_clusters = batch_size * launch_shape.seq_len_q * max(split_kv, 1)
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
                static_split_kv=split_kv,
                static_seq_len_k=None,
                qkv_dtype=qkv_dtype_name,
                out_dtype=output_dtype_name,
                rope_dim=qk_rope_head_dim,
                num_heads=num_heads,
                seq_len_q=seq_len_q,
                batch_size=batch_size,
                mask_type=mask_type,
            )
            workspace_size = compute_2cta_workspace_size(
                num_heads=int(launch_shape.num_heads_q),
                seq_len_q=int(launch_shape.seq_len_q),
                latent_dim=kv_lora_rank,
                batch_size=batch_size,
                split_kv=split_kv,
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
        _validate_mla_policy_coordinate_span(policy)

    return _MLADecodeLaunchSpec(
        kernel=kernel,
        qkv_dtype=qkv_dtype,
        output_dtype=output_dtype,
        policy=policy,
        kernel_workspace_bytes=int(workspace_size),
        split_kv=int(split_kv),
    )


@functools.cache
def _get_compiled_mla_decode(
    device_index: int,
    batch_size: int,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    max_kv_len: int,
    q_dtype_key: str,
    kv_dtype_key: str,
    output_dtype_key: str,
    mask_type: str,
    max_seq_len_q: int = 1,
    packed_query: bool = False,
):
    """Compile and cache one exact semantic TS MLA decode plan."""

    import cutlass
    import cutlass.cute as cute

    spec = _resolve_mla_decode_launch_spec(
        device_index,
        batch_size,
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        page_size,
        max_kv_len,
        q_dtype_key,
        kv_dtype_key,
        output_dtype_key,
        mask_type,
        max_seq_len_q,
    )
    physical_pages = cute.sym_int()
    runtime_batch = cute.sym_int()
    runtime_total_q = cute.sym_int()

    # These fake tensors pin the compact public ABI while allowing runtime
    # page counts, table widths, and batch metadata pointers to vary.
    q_stride_h = _MLA_QUERY_DIM
    q_stride_q = num_heads * _MLA_QUERY_DIM
    q_latent_shape: tuple[int, ...]
    q_rope_shape: tuple[int, ...]
    q_stride: tuple[int, ...]
    if packed_query:
        q_latent_shape = (num_heads, kv_lora_rank, runtime_total_q)
        q_rope_shape = (num_heads, qk_rope_head_dim, runtime_total_q)
        q_stride = (q_stride_h, 1, q_stride_q)
    else:
        q_stride_batch = max_seq_len_q * q_stride_q
        q_latent_shape = (num_heads, kv_lora_rank, max_seq_len_q, runtime_batch)
        q_rope_shape = (
            num_heads,
            qk_rope_head_dim,
            max_seq_len_q,
            runtime_batch,
        )
        q_stride = (q_stride_h, 1, q_stride_q, q_stride_batch)
    q_latent_fake = cute.runtime.make_fake_tensor(
        spec.qkv_dtype, q_latent_shape, stride=q_stride, assumed_align=16
    )
    q_rope_fake = cute.runtime.make_fake_tensor(
        spec.qkv_dtype, q_rope_shape, stride=q_stride, assumed_align=16
    )
    cache_token_stride = _MLA_QUERY_DIM
    cache_page_stride = page_size * _MLA_QUERY_DIM
    c_latent_fake = cute.runtime.make_fake_tensor(
        spec.qkv_dtype,
        (page_size, kv_lora_rank, physical_pages),
        stride=(cache_token_stride, 1, cache_page_stride),
        assumed_align=16,
    )
    c_rope_fake = cute.runtime.make_fake_tensor(
        spec.qkv_dtype,
        (page_size, qk_rope_head_dim, physical_pages),
        stride=(cache_token_stride, 1, cache_page_stride),
        assumed_align=16,
    )
    runtime_page_columns = cute.sym_int()
    page_offsets_fake = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (runtime_page_columns, runtime_batch),
        stride=(1, runtime_page_columns),
        assumed_align=16,
    )
    out_stride_row = num_heads * kv_lora_rank
    out_shape: tuple[int, ...]
    out_stride: tuple[int, ...]
    lse_shape: tuple[int, ...]
    lse_stride: tuple[int, ...]
    if packed_query:
        out_shape = (num_heads, kv_lora_rank, runtime_total_q)
        out_stride = (kv_lora_rank, 1, out_stride_row)
        lse_shape = (num_heads, runtime_total_q)
        lse_stride = (1, num_heads)
    else:
        out_stride_batch = max_seq_len_q * out_stride_row
        out_shape = (num_heads, kv_lora_rank, max_seq_len_q, runtime_batch)
        out_stride = (kv_lora_rank, 1, out_stride_row, out_stride_batch)
        lse_shape = (num_heads, max_seq_len_q, runtime_batch)
        lse_stride = (1, num_heads, max_seq_len_q * num_heads)
    out_fake = cute.runtime.make_fake_tensor(
        spec.output_dtype, out_shape, stride=out_stride, assumed_align=16
    )
    lse_fake = cute.runtime.make_fake_tensor(
        cutlass.Float32, lse_shape, stride=lse_stride, assumed_align=16
    )
    workspace_fake = None
    if spec.kernel_workspace_bytes > 0:
        workspace_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int8,
            (spec.kernel_workspace_bytes,),
            stride_order=(0,),
            assumed_align=32,
        )
    cache_seqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (runtime_batch,),
        stride_order=(0,),
        assumed_align=16,
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
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Refresh host-side grouping/reducer traits with a representative compact
    # extent before lowering. Runtime Q lengths still come only from qo_indptr.
    representative_total_q = batch_size * max_seq_len_q
    validation_q_shape: tuple[int, ...]
    validation_q_rope_shape: tuple[int, ...]
    validation_out_shape: tuple[int, ...]
    validation_lse_shape: tuple[int, ...]
    validation_qo_shape: tuple[int, ...] | None
    if packed_query:
        validation_q_shape = (num_heads, kv_lora_rank, representative_total_q)
        validation_q_rope_shape = (
            num_heads,
            qk_rope_head_dim,
            representative_total_q,
        )
        validation_out_shape = (num_heads, kv_lora_rank, representative_total_q)
        validation_lse_shape = (num_heads, representative_total_q)
        validation_qo_shape = (batch_size + 1,)
    else:
        validation_q_shape = (num_heads, kv_lora_rank, max_seq_len_q, batch_size)
        validation_q_rope_shape = (
            num_heads,
            qk_rope_head_dim,
            max_seq_len_q,
            batch_size,
        )
        validation_out_shape = (num_heads, kv_lora_rank, max_seq_len_q, batch_size)
        validation_lse_shape = (num_heads, max_seq_len_q, batch_size)
        validation_qo_shape = None
    if dict(spec.policy)["kernel"] == "throughput_latency_1cta":
        spec.kernel.validate_groups_tokens_heads_launch_shape(
            validation_q_shape,
            validation_q_rope_shape,
            validation_out_shape,
            validation_lse_shape,
            num_heads,
            max_seq_len_q,
            validation_qo_shape,
        )
    else:
        spec.kernel.validate_groups_tokens_heads_launch_shape(
            validation_q_shape,
            validation_q_rope_shape,
            validation_out_shape,
            validation_lse_shape,
            validation_qo_shape,
        )

    # Task objects carry loop-local state through generated control flow, so
    # select the public staged frontend for this compilation.
    with torch.cuda.device(device_index):
        compiled = cute.compile[cute.FrontendNext](
            spec.kernel,
            q_latent_fake,
            q_rope_fake,
            c_latent_fake,
            c_rope_fake,
            page_offsets_fake,
            out_fake,
            lse_fake,
            workspace_fake,
            cutlass.Int32(spec.split_kv),
            cache_seqs_fake,
            qo_indptr_fake,
            None,
            cutlass.Float32(1.0),
            cutlass.Float32(1.0),
            stream_fake,
            options=_COMPILE_OPTIONS,
        )
    return compiled, spec.policy, spec.kernel_workspace_bytes


def get_prims_ts_batch_decode_mla_workspace_size(
    batch_size: int,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    max_seq_len: int,
    *,
    seq_len_q: Optional[int] = None,
    max_seq_len_q: Optional[int] = None,
    q_dtype: torch.dtype = torch.bfloat16,
    kv_dtype: Optional[torch.dtype] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    mask_type: Literal["dense", "causal"] = "causal",
    device=None,
) -> int:
    """Return caller-workspace bytes for one automatic MLA policy.

    The arguments define the same semantic JIT key as
    :func:`prims_ts_batch_decode_with_kv_cache_mla`. Policy and private scratch
    layout are resolved without compiling a kernel. ``max_seq_len_q`` is the
    static per-request Q bound for both fixed and packed-query launches;
    ``seq_len_q`` remains a backward-compatible fixed-Q alias. If neither is
    supplied, the bound is one. The returned byte count includes both split-KV
    scratch and the internal FP32 LSE tensor. Allocate a contiguous
    ``torch.int8`` or ``torch.uint8`` CUDA buffer; MLA does not require its
    contents to be initialized before first use.
    """

    batch_size = _validate_positive_int(batch_size, "batch_size")
    num_heads = _validate_positive_int(num_heads, "num_heads")
    _validate_mla_dims(kv_lora_rank, qk_rope_head_dim)
    page_size = _validate_page_size(page_size)
    max_seq_len = _validate_mla_max_kv_len(max_seq_len, "max_seq_len")
    max_seq_len_q = _resolve_max_seq_len_q_alias(
        seq_len_q=seq_len_q,
        max_seq_len_q=max_seq_len_q,
        default=1,
    )
    assert max_seq_len_q is not None
    _validate_mla_query_head_extent(
        batch_size=batch_size,
        num_heads=num_heads,
        max_seq_len_q=max_seq_len_q,
    )
    _validate_mask(mask_type)
    if kv_dtype is None:
        kv_dtype = q_dtype
    _validate_mla_dtype_pair(q_dtype, kv_dtype, out_dtype)
    _, device_index = _resolve_cuda_device(device)

    spec = _resolve_mla_decode_launch_spec(
        device_index,
        batch_size,
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        page_size,
        max_seq_len,
        _dtype_key(q_dtype),
        _dtype_key(kv_dtype),
        _dtype_key(out_dtype),
        mask_type,
        max_seq_len_q,
    )
    return _make_mla_workspace_layout(
        spec.kernel_workspace_bytes, batch_size, num_heads, max_seq_len_q
    ).total_bytes


def _prepare_mla_runtime(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    num_heads: int,
    max_seq_len_q: int,
    qo_indptr: Optional[torch.Tensor],
    page_size: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    output_dtype: torch.dtype,
    bmm1_scale: float,
    bmm2_scale: float,
    out: Optional[torch.Tensor],
) -> _MLARuntime:
    """Validate public MLA tensors and normalize the cache without copies."""

    packed_query = qo_indptr is not None
    if packed_query:
        _validate_qo_indptr(
            qo_indptr,
            device=device,
            batch_size=batch_size,
        )
    _validate_query(
        query,
        packed_query=packed_query,
        device=device,
        batch_size=batch_size,
        num_heads=num_heads,
        max_seq_len_q=max_seq_len_q,
        q_dtype=q_dtype,
    )
    normalized_cache, _, runtime_page_size = _normalize_mla_kv_cache(
        kv_cache, expected_device=device
    )
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
    effective_bmm1_scale = _validate_scale(bmm1_scale, "bmm1_scale")
    effective_bmm2_scale = _validate_scale(bmm2_scale, "bmm2_scale")
    total_q = int(query.shape[0]) if packed_query else None
    if out is None:
        out_shape = (
            (total_q, num_heads, _MLA_LATENT_DIM)
            if packed_query
            else (batch_size, max_seq_len_q, num_heads, _MLA_LATENT_DIM)
        )
        out = torch.empty(out_shape, device=device, dtype=output_dtype)
    else:
        _validate_out(
            out,
            device=device,
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len_q=max_seq_len_q,
            packed_query=packed_query,
            total_q=total_q,
            output_dtype=output_dtype,
        )
    return _MLARuntime(
        query=query,
        normalized_cache=normalized_cache,
        out=out,
        bmm1_scale=effective_bmm1_scale,
        bmm2_scale=effective_bmm2_scale,
    )


def _validate_mla_output_aliasing(
    runtime: _MLARuntime,
    *,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    qo_indptr: Optional[torch.Tensor],
    workspace_buffer: torch.Tensor,
) -> None:
    """Keep output disjoint from every live MLA decode allocation."""

    _validate_out_does_not_overlap_inputs(
        runtime.out,
        ("query", runtime.query),
        ("kv_cache", runtime.normalized_cache),
        ("block_tables", block_tables),
        ("seq_lens", seq_lens),
        ("qo_indptr", qo_indptr),
        ("workspace_buffer", workspace_buffer),
    )


def _launch_mla_decode(
    runtime: _MLARuntime,
    *,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    qo_indptr: Optional[torch.Tensor],
    kv_lora_rank: int,
    split_kv: int,
    workspace: _MLAWorkspaceViews,
    compiled: Callable[..., object],
) -> torch.Tensor:
    """Form the dimension-first views and launch one compiled MLA kernel."""

    packed_query = qo_indptr is not None
    if packed_query:
        q_latent = runtime.query[..., :kv_lora_rank].permute(1, 2, 0)
        q_rope = runtime.query[..., kv_lora_rank:].permute(1, 2, 0)
        out_kernel = runtime.out.permute(1, 2, 0)
        total_q = int(runtime.query.shape[0])
        lse_kernel = workspace.lse.view(-1, workspace.lse.shape[-1])[
            :total_q
        ].transpose(0, 1)
    else:
        q_latent = runtime.query[..., :kv_lora_rank].permute(2, 3, 1, 0)
        q_rope = runtime.query[..., kv_lora_rank:].permute(2, 3, 1, 0)
        out_kernel = runtime.out.permute(2, 3, 1, 0)
        lse_kernel = workspace.lse.permute(2, 1, 0)
    c_latent = runtime.normalized_cache[..., :kv_lora_rank].permute(1, 2, 0)
    c_rope = runtime.normalized_cache[..., kv_lora_rank:].permute(1, 2, 0)
    page_offsets = block_tables.transpose(0, 1)
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
        seq_lens,
        qo_indptr,
        None,
        runtime.bmm1_scale,
        runtime.bmm2_scale,
    )
    return runtime.out


@flashinfer_api
def prims_ts_batch_decode_with_kv_cache_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    *,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    bmm1_scale: float = 1.0,
    bmm2_scale: float = 1.0,
    mask_type: Literal["dense", "causal"] = "causal",
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Launch fixed or packed-query paged MLA decode with caller-owned scratch.

    With ``qo_indptr=None``, ``query`` has fixed shape ``[B, SQ, H, 576]``.
    Otherwise ``query`` has compact shape ``[total_q, H, 576]`` and
    ``qo_indptr`` contains the ``B + 1`` cumulative Q offsets. Runtime Q
    lengths are exclusively ``qo_indptr[b + 1] - qo_indptr[b]``;
    ``max_seq_len_q`` is only the static policy, JIT, and workspace bound and
    is required for compact launches. The last query dimension concatenates
    the 512 latent and 64 RoPE dimensions. ``kv_cache`` accepts compact rank-3
    ``[pages, page_size, 576]`` or rank-4 ``[pages, 1, page_size, 576]``
    storage. ``block_tables`` and ``seq_lens`` follow FlashInfer's native dense
    paged-cache ABI; ``max_seq_len`` is the exact static policy/JIT maximum.
    Causal masking is bottom-right aligned: query row ``i`` can attend through
    KV row ``seq_lens[b] - q_len[b] + i`` for request ``b``.

    The workspace is exclusive to one in-flight launch or captured graph and
    must not overlap query, K/V cache, metadata, or output storage.
    Runtime lengths must remain positive and no larger than ``max_seq_len``;
    this hot path deliberately performs no device-to-host metadata reads. For
    packed launches, callers must ensure that offsets start at zero, are
    strictly increasing, end at ``query.shape[0]``, and have every delta no
    larger than ``max_seq_len_q``. For causal masking, every fixed or packed
    per-request Q length must also be no greater than the corresponding live
    ``seq_lens`` value. Warm the semantic key before CUDA graph
    capture and provide ``out`` to avoid an output allocation. Captured graphs
    must retain stable ``qo_indptr`` storage; its values may change only while
    that packed-offset contract and the captured query/output extent remain
    valid. No backend fallback or scheduling knob is exposed.

    Parameters
    ----------
    query : torch.Tensor
        Fixed or packed query tensor with concatenated latent and RoPE heads.
    kv_cache : torch.Tensor
        Compact paged latent K/V cache.
    workspace_buffer : torch.Tensor
        Caller-owned byte workspace for this semantic key.
    kv_lora_rank, qk_rope_head_dim : int
        Latent and RoPE dimensions.
    block_tables : torch.Tensor
        Dense physical-page table for each request.
    seq_lens : torch.Tensor
        Live K/V sequence lengths.
    max_seq_len : int
        Static maximum K/V length used for policy selection and JIT caching.
    qo_indptr : torch.Tensor, optional
        Cumulative query offsets selecting packed-query mode.
    max_seq_len_q : int, optional
        Static packed-query length bound.
    out : torch.Tensor, optional
        Caller-owned output tensor.
    bmm1_scale, bmm2_scale : float
        QK and value/output scaling factors.
    mask_type : {"dense", "causal"}
        Attention mask mode.
    out_dtype : torch.dtype
        Output dtype.
    """

    packed_query = qo_indptr is not None
    _validate_query(query, packed_query=packed_query)
    metadata_device, batch_size, max_num_pages = _validate_mla_metadata(
        block_tables, seq_lens
    )
    if metadata_device != query.device:
        raise ValueError(
            f"MLA metadata must be on {query.device}, got {metadata_device}"
        )
    normalized_cache, _, page_size = _normalize_mla_kv_cache(
        kv_cache, expected_device=query.device
    )
    if packed_query:
        _validate_qo_indptr(
            qo_indptr,
            device=query.device,
            batch_size=batch_size,
        )
        if max_seq_len_q is None:
            raise ValueError(
                "max_seq_len_q is required when qo_indptr selects packed query"
            )
        max_seq_len_q = _validate_positive_int(max_seq_len_q, "max_seq_len_q")
        num_heads = int(query.shape[1])
    else:
        fixed_seq_len_q = int(query.shape[1])
        if max_seq_len_q is None:
            max_seq_len_q = fixed_seq_len_q
        else:
            max_seq_len_q = _validate_positive_int(max_seq_len_q, "max_seq_len_q")
            if max_seq_len_q != fixed_seq_len_q:
                raise ValueError(
                    "fixed query length must equal max_seq_len_q: "
                    f"got SQ={fixed_seq_len_q} and max_seq_len_q={max_seq_len_q}"
                )
        num_heads = int(query.shape[2])
    _validate_mla_dims(kv_lora_rank, qk_rope_head_dim)
    _validate_page_size(page_size)
    max_seq_len = _validate_mla_max_kv_len(max_seq_len, "max_seq_len")
    required_page_columns = _ceil_div(max_seq_len, page_size)
    if max_num_pages < required_page_columns:
        raise ValueError(
            "block_tables must have at least ceil(max_seq_len / page_size) "
            f"columns ({required_page_columns}), got {max_num_pages}"
        )
    _validate_mask(mask_type)
    _validate_mla_dtype_pair(query.dtype, normalized_cache.dtype, out_dtype)
    device_index = _validate_runtime_device(query.device)
    spec_key = (
        device_index,
        batch_size,
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        page_size,
        max_seq_len,
        _dtype_key(query.dtype),
        _dtype_key(normalized_cache.dtype),
        _dtype_key(out_dtype),
        mask_type,
        max_seq_len_q,
    )
    spec = _resolve_mla_decode_launch_spec(*spec_key)
    layout = _make_mla_workspace_layout(
        spec.kernel_workspace_bytes, batch_size, num_heads, max_seq_len_q
    )
    _validate_workspace_buffer(
        workspace_buffer,
        device=query.device,
        required_bytes=layout.total_bytes,
    )
    caller_provided_out = out is not None
    runtime = _prepare_mla_runtime(
        query,
        normalized_cache,
        device=query.device,
        batch_size=batch_size,
        num_heads=num_heads,
        max_seq_len_q=max_seq_len_q,
        qo_indptr=qo_indptr,
        page_size=page_size,
        q_dtype=query.dtype,
        kv_dtype=normalized_cache.dtype,
        output_dtype=out_dtype,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        out=out,
    )
    _validate_tensor_does_not_overlap_inputs(
        workspace_buffer,
        "workspace_buffer",
        ("query", runtime.query),
        ("kv_cache", runtime.normalized_cache),
        ("block_tables", block_tables),
        ("seq_lens", seq_lens),
        ("qo_indptr", qo_indptr),
        ("out", runtime.out),
    )
    if caller_provided_out:
        _validate_mla_output_aliasing(
            runtime,
            block_tables=block_tables,
            seq_lens=seq_lens,
            qo_indptr=qo_indptr,
            workspace_buffer=workspace_buffer,
        )
    compiled, policy, kernel_workspace_bytes = _get_compiled_mla_decode(
        *spec_key, packed_query
    )
    if kernel_workspace_bytes != spec.kernel_workspace_bytes or policy != spec.policy:
        raise RuntimeError("MLA workspace policy changed during compilation")
    workspace = _bind_mla_workspace(workspace_buffer, layout)
    return _launch_mla_decode(
        runtime,
        block_tables=block_tables,
        seq_lens=seq_lens,
        qo_indptr=qo_indptr,
        kv_lora_rank=kv_lora_rank,
        split_kv=spec.split_kv,
        workspace=workspace,
        compiled=compiled,
    )


class BatchMLADecodePagedTSWrapper:
    """Plan and reuse task-scheduled paged MLA decode launches.

    Args:
        workspace_buffer: Optional caller-owned contiguous ``torch.int8`` or
            ``torch.uint8`` CUDA workspace. The buffer is validated and bound
            once by :meth:`plan`; its contents need not be initialized. It must
            remain alive, at the same address and size, and exclusive to one
            in-flight execution lane or captured-graph replay, and must not
            overlap query, K/V cache, metadata, or output storage. If omitted,
            each successful plan allocates private workspace with the same
            exclusivity requirement.
    """

    @flashinfer_api
    def __init__(self, workspace_buffer: Optional[torch.Tensor] = None) -> None:
        """Initialize an unplanned task-scheduled paged-MLA wrapper."""
        self._caller_workspace_buffer = workspace_buffer
        self._planned = False

    def _resolve_run_metadata(
        self,
        block_tables: Optional[torch.Tensor],
        seq_lens: Optional[torch.Tensor],
        qo_indptr: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Resolve retained or live metadata and validate its static contract."""

        if not self._live_metadata:
            if (
                block_tables is not None
                or seq_lens is not None
                or qo_indptr is not None
            ):
                raise ValueError(
                    "run-time MLA metadata requires plan(live_metadata=True)"
                )
            return self._block_tables, self._seq_lens, self._qo_indptr

        resolved_block_tables = (
            self._block_tables if block_tables is None else block_tables
        )
        resolved_seq_lens = self._seq_lens if seq_lens is None else seq_lens
        resolved_qo_indptr = self._qo_indptr if qo_indptr is None else qo_indptr
        device, batch_size, max_num_pages = _validate_mla_metadata(
            resolved_block_tables, resolved_seq_lens
        )
        if device != self._device:
            raise ValueError(
                f"MLA metadata must be on the planned device {self._device}, got {device}"
            )
        if batch_size != self._batch_size:
            raise ValueError(
                "MLA metadata batch size must match the plan "
                f"({self._batch_size}), got {batch_size}"
            )
        if max_num_pages < self._required_page_columns:
            raise ValueError(
                "block_tables must have at least ceil(max_kv_len / page_size) "
                f"columns ({self._required_page_columns}), got {max_num_pages}"
            )
        if self._packed_query:
            if resolved_qo_indptr is None:
                raise ValueError("packed-query MLA run requires qo_indptr")
        elif resolved_qo_indptr is not None:
            raise ValueError("fixed-query MLA plan does not accept qo_indptr")
        return resolved_block_tables, resolved_seq_lens, resolved_qo_indptr

    @flashinfer_api
    def plan(
        self,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        page_size: int,
        *,
        seq_len_q: Optional[int] = None,
        qo_indptr: Optional[torch.Tensor] = None,
        max_seq_len_q: Optional[int] = None,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: Optional[torch.dtype] = None,
        o_data_type: torch.dtype = torch.bfloat16,
        mask_type: Literal["dense", "causal"] = "causal",
        max_kv_len: Optional[int] = None,
        live_metadata: bool = False,
        workspace_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        """Prepare metadata, automatic policy, compiled callable, and scratch.

        With ``live_metadata=False``, planning preserves the legacy lifecycle:
        it reads and validates every K/V length, retains ``block_tables`` and
        ``seq_lens``, and, for packed query, reads cumulative ``qo_indptr``
        offsets to validate their deltas and fixed final total. If a Q or K/V
        bound is omitted, the corresponding metadata maximum becomes the exact
        static plan bound. An explicit bound may be larger but is still checked
        against every planned row.

        ``live_metadata=True`` makes the three metadata tensors runtime bindings.
        The plan-time tensors establish device, batch, page-table capacity, and
        fixed-versus-packed query mode, and remain defaults when a run omits an
        override. This mode requires an explicit ``max_kv_len`` and packed-query
        plans additionally require ``max_seq_len_q``. Planning therefore
        performs no device-to-host metadata read. Runtime metadata values are
        otherwise trusted. The planned batch size is exact: every run has ``B``
        sequence lengths, ``B`` block-table rows, and ``B + 1`` packed offsets.
        Lengths must remain positive and within their static bounds; packed
        offsets must start at zero, increase strictly, end at the runtime
        query/output extent, and have every delta within ``max_seq_len_q``.
        Every causal run must also preserve ``q_len[b] <= seq_lens[b]``.

        ``qo_indptr=None`` selects fixed query storage and the Q bound defaults
        to one. ``seq_len_q`` remains a backward-compatible alias for the same
        static bound. ``workspace_buffer`` overrides the constructor workspace
        for this successful plan; when both are omitted, the plan allocates
        private workspace. A wrapper workspace is mutable scratch and supports
        only one in-flight execution lane or captured-graph replay. Separate
        workspaces are required for concurrent streams or graph replays, even
        when they use separate wrapper instances. Warm the plan and retain stable
        workspace and metadata addresses before graph capture.

        Parameters
        ----------
        block_tables : torch.Tensor
            Dense physical-page table for each request.
        seq_lens : torch.Tensor
            Live K/V sequence lengths.
        num_heads, kv_lora_rank, qk_rope_head_dim, page_size : int
            MLA head geometry and K/V page size.
        seq_len_q : int, optional
            Backward-compatible fixed-query length alias.
        qo_indptr : torch.Tensor, optional
            Cumulative query offsets selecting packed-query mode.
        max_seq_len_q : int, optional
            Static packed-query length bound.
        q_data_type, kv_data_type, o_data_type : torch.dtype
            Query, K/V, and output dtypes used to compile the plan.
        mask_type : {"dense", "causal"}
            Attention mask mode.
        max_kv_len : int, optional
            Static K/V length bound; defaults to the metadata maximum.
        live_metadata : bool
            Select synchronization-free live-metadata planning.
        workspace_buffer : torch.Tensor, optional
            Caller-owned scratch for this plan.
        """

        if not isinstance(live_metadata, bool):
            raise TypeError("live_metadata must be a bool")
        _validate_mask(mask_type)
        num_heads = _validate_positive_int(num_heads, "num_heads")
        _validate_mla_dims(kv_lora_rank, qk_rope_head_dim)
        page_size = _validate_page_size(page_size)
        device, batch_size, max_num_pages = _validate_mla_metadata(
            block_tables, seq_lens
        )
        device_index = _validate_runtime_device(device)
        packed_query = qo_indptr is not None
        planned_total_q = None
        resolved_q_bound = _resolve_max_seq_len_q_alias(
            seq_len_q=seq_len_q,
            max_seq_len_q=max_seq_len_q,
            default=None,
        )
        if live_metadata and max_kv_len is None:
            raise ValueError("max_kv_len is required when live_metadata=True")
        if live_metadata and packed_query and resolved_q_bound is None:
            raise ValueError(
                "max_seq_len_q is required with qo_indptr when live_metadata=True"
            )
        if packed_query:
            _validate_qo_indptr(
                qo_indptr,
                device=device,
                batch_size=batch_size,
            )
            if live_metadata and resolved_q_bound is not None:
                max_seq_len_q = resolved_q_bound
                planned_q_lengths = None
            else:
                (
                    derived_q_bound,
                    derived_total_q,
                    planned_q_lengths,
                ) = _derive_max_seq_len_q(qo_indptr, batch_size=batch_size)
                if not live_metadata:
                    planned_total_q = derived_total_q
                if resolved_q_bound is None:
                    max_seq_len_q = derived_q_bound
                else:
                    max_seq_len_q = resolved_q_bound
                    if derived_q_bound > max_seq_len_q:
                        raise ValueError(
                            "qo_indptr contains a per-request Q length larger than "
                            f"max_seq_len_q ({max_seq_len_q}): got {derived_q_bound}"
                        )
        elif resolved_q_bound is None:
            max_seq_len_q = 1
            planned_q_lengths = (max_seq_len_q,) * batch_size
        else:
            max_seq_len_q = resolved_q_bound
            planned_q_lengths = (max_seq_len_q,) * batch_size

        _validate_mla_query_head_extent(
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len_q=max_seq_len_q,
            total_q=planned_total_q,
        )

        if kv_data_type is None:
            kv_data_type = q_data_type
        _validate_mla_dtype_pair(q_data_type, kv_data_type, o_data_type)

        seq_lens_host = None
        if not live_metadata:
            seq_lens_host = tuple(int(value) for value in seq_lens.tolist())
            if any(seq_len <= 0 for seq_len in seq_lens_host):
                raise ValueError(
                    "every planned request must contain at least one KV token"
                )
        if (
            mask_type == "causal"
            and planned_q_lengths is not None
            and seq_lens_host is not None
        ):
            for request_idx, (q_len, kv_len) in enumerate(
                zip(planned_q_lengths, seq_lens_host, strict=True)
            ):
                if q_len > kv_len:
                    raise ValueError(
                        "causal MLA decode requires every per-request Q length "
                        "to be no greater than its K/V length; request "
                        f"{request_idx} has Q={q_len} and K/V={kv_len}"
                    )
        if max_kv_len is None:
            assert seq_lens_host is not None
            exact_max_kv_len = max(seq_lens_host)
        else:
            exact_max_kv_len = _validate_mla_max_kv_len(max_kv_len, "max_kv_len")
            if seq_lens_host is not None:
                metadata_max_kv_len = max(seq_lens_host)
                if metadata_max_kv_len > exact_max_kv_len:
                    raise ValueError(
                        "planned KV metadata contains a request longer than "
                        f"max_kv_len ({exact_max_kv_len}): got {metadata_max_kv_len}"
                    )
        exact_max_kv_len = _validate_mla_max_kv_len(exact_max_kv_len, "max_kv_len")
        required_page_columns = _ceil_div(exact_max_kv_len, page_size)
        if max_num_pages < required_page_columns:
            raise ValueError(
                "block_tables must have at least ceil(max_kv_len / page_size) "
                f"columns ({required_page_columns}), got {max_num_pages}"
            )

        compiled, policy, workspace_size = _get_compiled_mla_decode(
            device_index,
            batch_size,
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            page_size,
            exact_max_kv_len,
            _dtype_key(q_data_type),
            _dtype_key(kv_data_type),
            _dtype_key(o_data_type),
            mask_type,
            max_seq_len_q,
            packed_query,
        )
        workspace_layout = _make_mla_workspace_layout(
            workspace_size, batch_size, num_heads, max_seq_len_q
        )
        if workspace_buffer is None:
            workspace_buffer = self._caller_workspace_buffer
        if workspace_buffer is None:
            workspace_buffer = torch.empty(
                workspace_layout.total_bytes, device=device, dtype=torch.int8
            )
        else:
            _validate_workspace_buffer(
                workspace_buffer,
                device=device,
                required_bytes=workspace_layout.total_bytes,
            )
            _validate_tensor_does_not_overlap_inputs(
                workspace_buffer,
                "workspace_buffer",
                ("block_tables", block_tables),
                ("seq_lens", seq_lens),
                ("qo_indptr", qo_indptr),
            )
        workspace = _bind_mla_workspace(workspace_buffer, workspace_layout)

        # Publish only after every validation, compilation, and allocation has
        # succeeded, so a failed re-plan leaves the previous plan usable.
        self._device = device
        self._device_index = device_index
        self._batch_size = batch_size
        self._num_heads = num_heads
        self._max_seq_len_q = max_seq_len_q
        self._qo_indptr = qo_indptr
        self._packed_query = packed_query
        self._planned_total_q = planned_total_q
        self._kv_lora_rank = kv_lora_rank
        self._qk_rope_head_dim = qk_rope_head_dim
        self._page_size = page_size
        self._q_dtype = q_data_type
        self._kv_dtype = kv_data_type
        self._output_dtype = o_data_type
        self._mask_type = mask_type
        self._max_kv_len = exact_max_kv_len
        self._required_page_columns = required_page_columns
        self._block_tables = block_tables
        self._seq_lens = seq_lens
        self._live_metadata = live_metadata
        self._workspace_buffer = workspace_buffer
        self._workspace_layout = workspace_layout
        self._workspace_views = workspace
        self._workspace = workspace.kernel_workspace
        self._lse = workspace.lse
        self._compiled = compiled
        self._policy = policy
        self._split_kv = int(dict(policy)["split_kv"])
        self._planned = True

    @flashinfer_api
    def run(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        *,
        block_tables: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        qo_indptr: Optional[torch.Tensor] = None,
        bmm1_scale: float = 1.0,
        bmm2_scale: float = 1.0,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Launch the most recently planned MLA decode on the current stream.

        In retained metadata mode, omit all three metadata arguments. In live
        mode, each argument optionally replaces its plan-time default for this
        run without changing the compiled plan. The caller must keep metadata
        values within the static plan contract documented by :meth:`plan`.

        Parameters
        ----------
        query : torch.Tensor
            Runtime fixed or packed query tensor matching the plan.
        kv_cache : torch.Tensor
            Runtime compact paged latent K/V cache.
        bmm1_scale, bmm2_scale : float
            QK and value/output scaling factors.
        out : torch.Tensor, optional
            Caller-owned output tensor. A new tensor is allocated when omitted.
        block_tables, seq_lens, qo_indptr : torch.Tensor, optional
            Live metadata bindings for a live-metadata plan.
        """

        if not self._planned:
            raise RuntimeError("plan() must be called before run()")
        block_tables, seq_lens, qo_indptr = self._resolve_run_metadata(
            block_tables, seq_lens, qo_indptr
        )
        if (
            self._planned_total_q is not None
            and int(query.shape[0]) != self._planned_total_q
        ):
            raise ValueError(
                "packed query rows must match the final planned qo_indptr "
                f"offset ({self._planned_total_q}), got {query.shape[0]}"
            )
        caller_provided_out = out is not None
        runtime = _prepare_mla_runtime(
            query,
            kv_cache,
            device=self._device,
            batch_size=self._batch_size,
            num_heads=self._num_heads,
            max_seq_len_q=self._max_seq_len_q,
            qo_indptr=qo_indptr,
            page_size=self._page_size,
            q_dtype=self._q_dtype,
            kv_dtype=self._kv_dtype,
            output_dtype=self._output_dtype,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            out=out,
        )
        _validate_tensor_does_not_overlap_inputs(
            self._workspace_buffer,
            "workspace_buffer",
            ("query", runtime.query),
            ("kv_cache", runtime.normalized_cache),
            ("block_tables", block_tables),
            ("seq_lens", seq_lens),
            ("qo_indptr", qo_indptr),
            ("out", runtime.out),
        )
        if caller_provided_out:
            _validate_mla_output_aliasing(
                runtime,
                block_tables=block_tables,
                seq_lens=seq_lens,
                qo_indptr=qo_indptr,
                workspace_buffer=self._workspace_buffer,
            )
        return _launch_mla_decode(
            runtime,
            block_tables=block_tables,
            seq_lens=seq_lens,
            qo_indptr=qo_indptr,
            kv_lora_rank=self._kv_lora_rank,
            split_kv=self._split_kv,
            workspace=self._workspace_views,
            compiled=self._compiled,
        )


@flashinfer_api
def batch_decode_mla_with_paged_kv_cache(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
    kv_lora_rank: int = _MLA_LATENT_DIM,
    qk_rope_head_dim: int = _MLA_ROPE_DIM,
    mask_type: Literal["dense", "causal"] = "causal",
    max_kv_len: Optional[int] = None,
    bmm1_scale: float = 1.0,
    bmm2_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """One-shot convenience wrapper for fixed or packed-query MLA decode.

    Parameters
    ----------
    query : torch.Tensor
        Fixed or packed query tensor with concatenated latent and RoPE heads.
    kv_cache : torch.Tensor
        Compact paged latent K/V cache.
    block_tables : torch.Tensor
        Dense physical-page table for each request.
    seq_lens : torch.Tensor
        Live K/V sequence lengths.
    qo_indptr : torch.Tensor, optional
        Cumulative query offsets selecting packed-query mode.
    max_seq_len_q : int, optional
        Static packed-query length bound.
    kv_lora_rank, qk_rope_head_dim : int
        Latent and RoPE dimensions.
    mask_type : {"dense", "causal"}
        Attention mask mode.
    max_kv_len : int, optional
        Static K/V length bound; defaults to the metadata maximum.
    bmm1_scale, bmm2_scale : float
        QK and value/output scaling factors.
    out : torch.Tensor, optional
        Caller-owned output tensor.
    out_dtype : torch.dtype
        Output dtype.
    """

    packed_query = qo_indptr is not None
    _validate_query(query, packed_query=packed_query)
    metadata_device, batch_size, _ = _validate_mla_metadata(block_tables, seq_lens)
    if metadata_device != query.device:
        raise ValueError(
            f"MLA metadata must be on {query.device}, got {metadata_device}"
        )
    normalized_cache, _, page_size = _normalize_mla_kv_cache(
        kv_cache, expected_device=query.device
    )
    _validate_mla_dims(kv_lora_rank, qk_rope_head_dim)
    _validate_page_size(page_size)
    _validate_mla_dtype_pair(query.dtype, normalized_cache.dtype, out_dtype)
    if packed_query:
        _validate_qo_indptr(
            qo_indptr,
            device=query.device,
            batch_size=batch_size,
        )
        num_heads = int(query.shape[1])
        if max_seq_len_q is None:
            _validate_mla_int32_extent(
                int(query.shape[0]) * num_heads,
                "total_q * num_heads",
            )
        else:
            max_seq_len_q = _validate_positive_int(max_seq_len_q, "max_seq_len_q")
            _validate_mla_query_head_extent(
                batch_size=batch_size,
                num_heads=num_heads,
                max_seq_len_q=max_seq_len_q,
                total_q=int(query.shape[0]),
            )
    else:
        num_heads = int(query.shape[2])
        fixed_seq_len_q = int(query.shape[1])
        if max_seq_len_q is None:
            max_seq_len_q = fixed_seq_len_q
        else:
            max_seq_len_q = _validate_positive_int(max_seq_len_q, "max_seq_len_q")
            if max_seq_len_q != fixed_seq_len_q:
                raise ValueError(
                    "fixed query length must equal max_seq_len_q: "
                    f"got SQ={fixed_seq_len_q} and "
                    f"max_seq_len_q={max_seq_len_q}"
                )
        _validate_mla_query_head_extent(
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len_q=max_seq_len_q,
        )
    if out is not None:
        _validate_out(
            out,
            device=query.device,
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len_q=(
                _validate_positive_int(max_seq_len_q, "max_seq_len_q")
                if max_seq_len_q is not None
                else int(query.shape[0])
            ),
            packed_query=packed_query,
            total_q=int(query.shape[0]) if packed_query else None,
            output_dtype=out_dtype,
        )

    wrapper = BatchMLADecodePagedTSWrapper()
    wrapper.plan(
        block_tables,
        seq_lens,
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        page_size,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
        q_data_type=query.dtype,
        kv_data_type=normalized_cache.dtype,
        o_data_type=out_dtype,
        mask_type=mask_type,
        max_kv_len=max_kv_len,
    )
    return wrapper.run(
        query,
        normalized_cache,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        out=out,
    )


__all__ = [
    "BatchMLADecodePagedTSWrapper",
    "batch_decode_mla_with_paged_kv_cache",
    "get_prims_ts_batch_decode_mla_workspace_size",
    "prims_ts_batch_decode_with_kv_cache_mla",
]
