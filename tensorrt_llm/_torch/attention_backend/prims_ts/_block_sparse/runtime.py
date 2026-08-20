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

"""Runtime validation and launch adapter for block-sparse attention."""

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Protocol

import torch

from .._tensor_aliasing import _validate_out_does_not_overlap_inputs
from ..decode import (
    PagedKVCache,
    _normalize_paged_kv_cache,
    _validate_16byte_alignment,
    _validate_exact_compact_strides,
    _validate_scale,
)
from .common import _SIGNED_INT32_MAX


@dataclass(frozen=True)
class _ContiguousKVStorage:
    """Compact BSHD K/V inputs validated directly against the fixed plan."""

    k: torch.Tensor
    v: torch.Tensor


@dataclass(frozen=True)
class _PagedKVStorage:
    """Live paged-K/V runtime storage validated against one frozen paged plan."""

    paged_kv_cache: PagedKVCache
    paged_kv_indices: torch.Tensor


@dataclass(frozen=True)
class _PagedKVLaunchPayload:
    """Launch-only live paged metadata derived during shared validation."""

    paged_kv_indices: torch.Tensor
    num_physical_kv_pages: int
    k_page_stride: int
    v_page_stride: int


@dataclass(frozen=True)
class _BlockSparseRunArgs:
    """Validated launch arguments with optional values materialized once."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    block_indptr: torch.Tensor
    block_indices: torch.Tensor
    kv_valid_bits: torch.Tensor
    kv_valid_bits_is_live: bool
    sm_scale: float
    paged_kv: _PagedKVLaunchPayload | None


class _PagedKVPlanMetadataLike(Protocol):
    """Minimal paged-plan metadata consumed by shared runtime validation."""

    @property
    def page_size(self) -> int: ...

    @property
    def paged_kv_indptr(self) -> torch.Tensor: ...

    @property
    def seq_lens_kv(self) -> torch.Tensor: ...

    @property
    def num_page_indices(self) -> int: ...


class _BlockSparsePlanStateLike(Protocol):
    """Structural launch state shared by contiguous and paged wrappers."""

    @property
    def device(self) -> torch.device: ...

    @property
    def batch_size(self) -> int: ...

    @property
    def seq_len_q(self) -> int: ...

    @property
    def seq_len_kv(self) -> int: ...

    @property
    def num_qo_heads(self) -> int: ...

    @property
    def num_kv_heads(self) -> int: ...

    @property
    def head_dim(self) -> int: ...

    @property
    def q_block_size(self) -> int: ...

    @property
    def q_dtype(self) -> torch.dtype: ...

    @property
    def kv_dtype(self) -> torch.dtype: ...

    @property
    def output_dtype(self) -> torch.dtype: ...

    @property
    def use_kv_valid_bits(self) -> bool: ...

    @property
    def dummy_kv_valid_bits(self) -> torch.Tensor | None: ...

    @property
    def row_route_offsets(self) -> torch.Tensor: ...

    @property
    def route_workspace(self) -> torch.Tensor: ...

    @property
    def max_blocks_per_row(self) -> int: ...

    @property
    def compiled(self) -> Callable[..., object]: ...

    @property
    def paged_kv(self) -> _PagedKVPlanMetadataLike | None: ...


def _validate_metadata_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    ndim: int,
    dtype: torch.dtype,
    expected_device: torch.device,
    expected_shape: tuple[int, ...] | None = None,
) -> None:
    """Validate one compact tensor in the raw sparse-routing ABI."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be rank {ndim}, got rank {tensor.ndim}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if tensor.device != expected_device:
        raise ValueError(
            f"{name} must be on planned device {expected_device}, got {tensor.device}"
        )
    if expected_shape is not None and tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
        )
    if tensor.numel() > _SIGNED_INT32_MAX:
        raise OverflowError(f"{name}.numel() must fit in signed int32")
    _validate_exact_compact_strides(tensor, name, f"rank-{ndim}")
    if tensor.data_ptr() % 4 != 0:
        raise ValueError(f"{name} data pointer must be 4-byte aligned")


def validate_block_sparse_metadata(
    block_indptr: torch.Tensor,
    block_indices: torch.Tensor,
    kv_valid_bits: torch.Tensor | None,
    *,
    device: torch.device,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_kv_heads: int,
    q_block_size: int,
    use_kv_valid_bits: bool,
) -> None:
    """Validate raw runtime routing without reading device-side values."""

    num_q_blocks = (seq_len_q + q_block_size - 1) // q_block_size
    _validate_metadata_tensor(
        block_indptr,
        "block_indptr",
        ndim=3,
        dtype=torch.int32,
        expected_device=device,
        expected_shape=(batch_size, num_kv_heads, num_q_blocks + 1),
    )
    _validate_metadata_tensor(
        block_indices,
        "block_indices",
        ndim=1,
        dtype=torch.int32,
        expected_device=device,
    )

    if use_kv_valid_bits:
        if kv_valid_bits is None:
            raise ValueError("kv_valid_bits is required when use_kv_valid_bits=True")
        _validate_metadata_tensor(
            kv_valid_bits,
            "kv_valid_bits",
            ndim=2,
            dtype=torch.uint32,
            expected_device=device,
            expected_shape=(batch_size, (seq_len_kv + 31) // 32),
        )
    elif kv_valid_bits is not None:
        raise ValueError("kv_valid_bits must be None when use_kv_valid_bits=False")


def _validate_bshd_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    expected_shape: tuple[int, int, int, int],
    expected_dtype: torch.dtype,
    expected_device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 4 or tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} must have compact BSHD shape {expected_shape}, "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.dtype != expected_dtype:
        raise ValueError(f"{name} must have dtype {expected_dtype}, got {tensor.dtype}")
    if tensor.device != expected_device:
        raise ValueError(
            f"{name} must be on planned device {expected_device}, got {tensor.device}"
        )
    _validate_exact_compact_strides(tensor, name, "BSHD")
    _validate_16byte_alignment(tensor, name)


def _resolve_effective_kv_valid_bits(
    *,
    kv_valid_bits: torch.Tensor | None,
    dummy_kv_valid_bits: torch.Tensor | None,
    use_kv_valid_bits: bool,
    missing_dummy_message: str,
) -> torch.Tensor:
    """Resolve the always-present runtime mask tensor for one launch."""

    if use_kv_valid_bits:
        assert kv_valid_bits is not None
        return kv_valid_bits
    if dummy_kv_valid_bits is None:
        raise RuntimeError(missing_dummy_message)
    return dummy_kv_valid_bits


def _require_contiguous_plan_state(
    state: _BlockSparsePlanStateLike,
) -> None:
    if state.paged_kv is not None:
        raise TypeError("contiguous K/V storage requires a contiguous plan state")


def _require_paged_plan_state(
    state: _BlockSparsePlanStateLike,
) -> _PagedKVPlanMetadataLike:
    paged_kv = state.paged_kv
    if paged_kv is None:
        raise TypeError("paged K/V storage requires a paged plan state")
    return paged_kv


def validate_block_sparse_run(
    q: torch.Tensor,
    kv_storage: _ContiguousKVStorage | _PagedKVStorage,
    *,
    state: _BlockSparsePlanStateLike,
    block_indptr: torch.Tensor,
    block_indices: torch.Tensor,
    kv_valid_bits: torch.Tensor | None,
    sm_scale: float | None,
    out: torch.Tensor | None,
) -> _BlockSparseRunArgs:
    """Validate one run and allocate only an omitted output tensor.

    Q/O use compact ``[B, Sq, Hq, D]`` on the planned device and dtype.
    Contiguous K/V use compact ``[B, Skv, Hkv, D]`` directly. Paged K/V are
    normalized once into zero-copy HND cache views plus launch metadata. An
    explicit output is returned by identity and may not overlap any live launch
    input. ``sm_scale=None`` is materialized as ``1 / sqrt(D)``.
    """

    validate_block_sparse_metadata(
        block_indptr,
        block_indices,
        kv_valid_bits,
        device=state.device,
        batch_size=state.batch_size,
        seq_len_q=state.seq_len_q,
        seq_len_kv=state.seq_len_kv,
        num_kv_heads=state.num_kv_heads,
        q_block_size=state.q_block_size,
        use_kv_valid_bits=state.use_kv_valid_bits,
    )
    effective_kv_valid_bits = _resolve_effective_kv_valid_bits(
        kv_valid_bits=kv_valid_bits,
        dummy_kv_valid_bits=state.dummy_kv_valid_bits,
        use_kv_valid_bits=state.use_kv_valid_bits,
        missing_dummy_message="unmasked block-sparse plan is missing its dummy mask",
    )

    q_shape = (state.batch_size, state.seq_len_q, state.num_qo_heads, state.head_dim)
    _validate_bshd_tensor(
        q,
        "q",
        expected_shape=q_shape,
        expected_dtype=state.q_dtype,
        expected_device=state.device,
    )
    paged_kv: _PagedKVLaunchPayload | None = None
    overlap_inputs: tuple[tuple[str, torch.Tensor], ...]
    if isinstance(kv_storage, _ContiguousKVStorage):
        _require_contiguous_plan_state(state)
        kv_shape = (
            state.batch_size,
            state.seq_len_kv,
            state.num_kv_heads,
            state.head_dim,
        )
        for tensor, name in ((kv_storage.k, "k"), (kv_storage.v, "v")):
            _validate_bshd_tensor(
                tensor,
                name,
                expected_shape=kv_shape,
                expected_dtype=state.kv_dtype,
                expected_device=state.device,
            )
        k = kv_storage.k
        v = kv_storage.v
        overlap_inputs = (
            ("q", q),
            ("k", k),
            ("v", v),
            ("block_indptr", block_indptr),
            ("block_indices", block_indices),
            ("kv_valid_bits", effective_kv_valid_bits),
            ("row_route_offsets", state.row_route_offsets),
            ("route_workspace", state.route_workspace),
        )
    elif isinstance(kv_storage, _PagedKVStorage):
        paged_plan = _require_paged_plan_state(state)
        (
            k,
            v,
            num_physical_kv_pages,
            runtime_num_kv_heads,
            runtime_page_size,
            runtime_head_dim,
            k_page_stride,
            v_page_stride,
        ) = _normalize_paged_kv_cache(
            kv_storage.paged_kv_cache,
            expected_device=state.device,
        )
        runtime_geometry = (
            runtime_num_kv_heads,
            runtime_page_size,
            runtime_head_dim,
        )
        expected_geometry = (
            state.num_kv_heads,
            paged_plan.page_size,
            state.head_dim,
        )
        if runtime_geometry != expected_geometry:
            raise ValueError(
                "paged_kv_cache geometry does not match the plan: expected "
                f"Hkv/page/D={expected_geometry}, got {runtime_geometry}"
            )
        if k.dtype != state.kv_dtype:
            raise ValueError(
                f"K/V dtype must match the plan ({state.kv_dtype}), got {k.dtype}"
            )
        _validate_metadata_tensor(
            kv_storage.paged_kv_indices,
            "paged_kv_indices",
            ndim=1,
            dtype=torch.int32,
            expected_device=state.device,
            expected_shape=(paged_plan.num_page_indices,),
        )
        paged_kv = _PagedKVLaunchPayload(
            paged_kv_indices=kv_storage.paged_kv_indices,
            num_physical_kv_pages=num_physical_kv_pages,
            k_page_stride=k_page_stride,
            v_page_stride=v_page_stride,
        )
        overlap_inputs = (
            ("q", q),
            ("k_cache", k),
            ("v_cache", v),
            ("block_indptr", block_indptr),
            ("block_indices", block_indices),
            ("kv_valid_bits", effective_kv_valid_bits),
            ("paged_kv_indptr", paged_plan.paged_kv_indptr),
            ("paged_kv_indices", kv_storage.paged_kv_indices),
            ("seq_lens_kv", paged_plan.seq_lens_kv),
            ("row_route_offsets", state.row_route_offsets),
            ("route_workspace", state.route_workspace),
        )
    else:
        raise TypeError("kv_storage must be _ContiguousKVStorage or _PagedKVStorage")

    effective_scale = _validate_scale(
        1.0 / math.sqrt(state.head_dim) if sm_scale is None else sm_scale,
        "sm_scale",
    )
    if out is None:
        out = torch.empty(q_shape, device=state.device, dtype=state.output_dtype)
    else:
        _validate_bshd_tensor(
            out,
            "out",
            expected_shape=q_shape,
            expected_dtype=state.output_dtype,
            expected_device=state.device,
        )
        _validate_out_does_not_overlap_inputs(out, *overlap_inputs)
    return _BlockSparseRunArgs(
        q=q,
        k=k,
        v=v,
        out=out,
        block_indptr=block_indptr,
        block_indices=block_indices,
        kv_valid_bits=effective_kv_valid_bits,
        kv_valid_bits_is_live=state.use_kv_valid_bits,
        sm_scale=effective_scale,
        paged_kv=paged_kv,
    )


def record_block_sparse_run_args(
    run_args: _BlockSparseRunArgs,
    stream: torch.cuda.Stream,
) -> None:
    """Extend tensor lifetimes for the asynchronous launch currently in flight."""

    run_args.q.record_stream(stream)
    run_args.k.record_stream(stream)
    run_args.v.record_stream(stream)
    run_args.out.record_stream(stream)
    run_args.block_indptr.record_stream(stream)
    run_args.block_indices.record_stream(stream)
    if run_args.kv_valid_bits_is_live:
        run_args.kv_valid_bits.record_stream(stream)
    if run_args.paged_kv is not None:
        run_args.paged_kv.paged_kv_indices.record_stream(stream)


def launch_block_sparse(
    run_args: _BlockSparseRunArgs,
    *,
    state: _BlockSparsePlanStateLike,
) -> torch.Tensor:
    """Invoke the exact contiguous or paged ABI chosen by validated payload."""

    if run_args.paged_kv is None:
        _require_contiguous_plan_state(state)
        state.compiled(
            run_args.q,
            run_args.k,
            run_args.v,
            run_args.out,
            run_args.block_indptr,
            run_args.block_indices,
            run_args.kv_valid_bits,
            state.row_route_offsets,
            state.route_workspace,
            state.max_blocks_per_row,
            run_args.sm_scale,
        )
    else:
        paged_plan = _require_paged_plan_state(state)
        state.compiled(
            run_args.q,
            run_args.k,
            run_args.v,
            run_args.out,
            run_args.block_indptr,
            run_args.block_indices,
            run_args.kv_valid_bits,
            paged_plan.paged_kv_indptr,
            run_args.paged_kv.paged_kv_indices,
            paged_plan.seq_lens_kv,
            state.row_route_offsets,
            state.route_workspace,
            state.max_blocks_per_row,
            run_args.paged_kv.num_physical_kv_pages,
            run_args.paged_kv.k_page_stride,
            run_args.paged_kv.v_page_stride,
            run_args.sm_scale,
        )
    return run_args.out


__all__ = [
    "_BlockSparseRunArgs",
    "_ContiguousKVStorage",
    "_PagedKVLaunchPayload",
    "_PagedKVStorage",
    "launch_block_sparse",
    "record_block_sparse_run_args",
    "validate_block_sparse_metadata",
    "validate_block_sparse_run",
]
