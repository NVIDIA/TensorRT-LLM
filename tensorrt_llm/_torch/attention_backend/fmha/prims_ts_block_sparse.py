# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PrimTS block-sparse FMHA for contiguous tensors and paged KV cache.

The runtime owns storage-specific PrimTS plans. The registered FMHA maps the
TRT-LLM request contract to the matching contiguous or paged runtime ABI.
"""

import importlib.metadata
import math
from collections.abc import MutableMapping
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, Literal, cast

import torch
from packaging.version import InvalidVersion, Version

from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention_backend.sparse.block_sparse import (
    BlockSparseForwardInputs,
    BlockSparseParams,
    BlockSparseRoutes,
    _is_current_stream_capturing,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.quantization.mode import QuantMode

from .interface import Fmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


_prims_ts_block_sparse_import_error: Exception | None = None
try:
    from tensorrt_llm._torch.attention_backend.prims_ts import (
        BlockSparsePagedTSWrapper as _BlockSparsePagedTSWrapper,
    )
    from tensorrt_llm._torch.attention_backend.prims_ts import (
        BlockSparseTSWrapper as _BlockSparseTSWrapper,
    )
    from tensorrt_llm._torch.attention_backend.prims_ts._block_sparse.config import (
        _validate_block_sparse_static_profile,
    )
    from tensorrt_llm._torch.attention_backend.prims_ts._block_sparse.runtime import (
        validate_block_sparse_metadata as _validate_block_sparse_metadata,
    )
except (ImportError, OSError) as error:
    _BlockSparseTSWrapper = None
    _BlockSparsePagedTSWrapper = None
    _validate_block_sparse_static_profile = None
    _validate_block_sparse_metadata = None
    _prims_ts_block_sparse_import_error = error

_SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3))
_MIN_CUTLASS_DSL_VERSION = Version("4.7.0")
_MIN_CUTLASS_COMPILER_VERSION = "13.3"

# Callers may provide one state per serialized execution lane to share plans
# across layers. Without such state, the FMHA retains per-layer ownership.
PRIMS_TS_BLOCK_SPARSE_RUNTIME_STATE_KEY = "prims_ts_block_sparse_runtime"


@dataclass(frozen=True, slots=True)
class _BlockSparsePlanKey:
    """Static traits selecting one reusable PrimTS execution plan."""

    device: torch.device
    batch_size: int
    seq_len_q: int
    seq_len_kv: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    q_block_size: int
    kv_block_size: int
    max_blocks_per_row: int
    mask_type: Literal["dense", "causal"]
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    output_dtype: torch.dtype
    use_kv_valid_bits: bool


@dataclass(frozen=True, slots=True)
class _PagedBlockSparsePlanKey:
    """Static traits selecting one reusable paged PrimTS plan."""

    device: torch.device
    batch_size: int
    seq_len_q: int
    max_seq_len_kv: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    q_block_size: int
    kv_block_size: int
    page_size: int
    max_blocks_per_row: int
    mask_type: Literal["dense", "causal"]
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    output_dtype: torch.dtype
    use_kv_valid_bits: bool


@cache
def _get_prims_ts_runtime_support_error() -> str | None:
    """Return a cached dependency error shared by every block-sparse plan."""

    try:
        installed_version = Version(importlib.metadata.version("nvidia-cutlass-dsl"))
    except (importlib.metadata.PackageNotFoundError, InvalidVersion):
        return f"nvidia-cutlass-dsl>={_MIN_CUTLASS_DSL_VERSION} is required"
    if installed_version < _MIN_CUTLASS_DSL_VERSION:
        return (
            f"nvidia-cutlass-dsl>={_MIN_CUTLASS_DSL_VERSION} is required, got {installed_version}"
        )
    try:
        cutlass = import_module("cutlass")
        compiler_version_supported = cutlass.target_version(
            min_version=_MIN_CUTLASS_COMPILER_VERSION
        )
    except Exception as error:  # noqa: BLE001 - availability probes must fail closed
        return f"could not query the active CUTLASS compiler version: {error}"
    if not compiler_version_supported:
        return f"the active CUTLASS compiler must target CUDA>={_MIN_CUTLASS_COMPILER_VERSION}"
    try:
        import_module("cutlass.experimental.task_scheduling")
    except ImportError:
        return "CUTLASS task scheduling is unavailable"
    return None


def get_prims_ts_block_sparse_unavailability_reason(
    device: torch.device | str | int | None = None,
) -> str | None:
    """Return why PrimTS block-sparse cannot run, optionally on ``device``."""

    if _BlockSparseTSWrapper is None:
        return f"the vendored wrapper could not be imported: {_prims_ts_block_sparse_import_error}"
    runtime_error = _get_prims_ts_runtime_support_error()
    if runtime_error is not None:
        return runtime_error
    if device is None:
        return None
    resolved_device = (
        torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    )
    if resolved_device.type != "cuda":
        return f"PrimTS block-sparse requires a CUDA device, got {resolved_device}"
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    try:
        capability = torch.cuda.get_device_capability(resolved_device)
    except (AssertionError, RuntimeError, ValueError) as error:
        return f"could not query CUDA device capability: {error}"
    if capability not in _SUPPORTED_COMPUTE_CAPABILITIES:
        return f"PrimTS block-sparse requires SM100 or SM103, got SM{capability[0]}{capability[1]}"
    return None


def get_prims_ts_block_sparse_contiguous_unsupported_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_block_size: int,
    kv_block_size: int,
    max_blocks_per_row: int,
    use_kv_valid_bits: bool = False,
    output_dtype: torch.dtype | None = None,
    mask_type: Literal["dense", "causal"] = "dense",
) -> str | None:
    """Return why a contiguous tensor profile cannot use PrimTS."""

    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        return "Q, K, and V must be torch.Tensor instances"
    if any(tensor.ndim != 4 for tensor in (q, k, v)):
        return "Q, K, and V must have BSHD rank-4 shapes"

    runtime_reason = get_prims_ts_block_sparse_unavailability_reason(q.device)
    if runtime_reason is not None:
        return runtime_reason
    assert _validate_block_sparse_static_profile is not None
    batch_size, seq_len_q, num_qo_heads, head_dim = map(int, q.shape)
    kv_batch_size, seq_len_kv, num_kv_heads, kv_head_dim = map(int, k.shape)
    if v.shape != k.shape:
        return "K and V must have identical shapes"
    if kv_batch_size != batch_size or kv_head_dim != head_dim:
        return "Q, K, and V batch/head dimensions must match"
    if any(tensor.device != q.device or tensor.dtype != q.dtype for tensor in (k, v)):
        return "Q, K, and V must share device and dtype"
    try:
        _validate_block_sparse_static_profile(
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            max_blocks_per_row=max_blocks_per_row,
            use_kv_valid_bits=use_kv_valid_bits,
            mask_type=mask_type,
            q_dtype=q.dtype,
            kv_dtype=k.dtype,
            output_dtype=output_dtype,
        )
    except (TypeError, ValueError, OverflowError, NotImplementedError) as error:
        return str(error)
    return None


def _get_prims_ts_block_sparse_metadata_unsupported_reason(
    routes: BlockSparseRoutes,
    kv_valid_bits: torch.Tensor | None,
    *,
    device: torch.device,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_kv_heads: int,
    q_block_size: int,
) -> str | None:
    """Return why live canonical BSR metadata cannot be dispatched."""

    runtime_reason = get_prims_ts_block_sparse_unavailability_reason(device)
    if runtime_reason is not None:
        return runtime_reason
    try:
        _validate_block_sparse_metadata(
            routes.block_indptr,
            routes.block_indices,
            kv_valid_bits,
            device=device,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            num_kv_heads=num_kv_heads,
            q_block_size=q_block_size,
            use_kv_valid_bits=kv_valid_bits is not None,
        )
    except (TypeError, ValueError, OverflowError) as error:
        return str(error)
    return None


# Common PrimTS runtime and plan caches.


class PrimsTSBlockSparseRuntime:
    """Own reusable PrimTS plans and their mutable route workspaces.

    Plans depend only on static tensor geometry, dtypes, mask mode, and route
    capacity. Route tensors and values remain live run inputs, so serialized
    model layers can share this runtime. Concurrent streams or graph replays
    need independent runtimes because each plan has one mutable workspace.
    """

    def __init__(self) -> None:
        self._plans: dict[_BlockSparsePlanKey, object] = {}
        self._paged_plans: dict[_PagedBlockSparsePlanKey, object] = {}

    def clear(self) -> None:
        """Release plans after graph teardown."""

        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("block-sparse state cannot be cleared during CUDA Graph capture")
        self._plans.clear()
        self._paged_plans.clear()

    @staticmethod
    def _make_plan_key(
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        routes: BlockSparseRoutes,
        q_block_size: int,
        kv_block_size: int,
        kv_valid_bits: torch.Tensor | None,
        mask_type: Literal["dense", "causal"],
        out: torch.Tensor | None,
    ) -> _BlockSparsePlanKey:
        if not isinstance(routes, BlockSparseRoutes):
            raise TypeError("routes must be a BlockSparseRoutes instance")
        if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor):
            raise TypeError("Q and K must be torch.Tensor instances")
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError("Q and K must have BSHD rank-4 shapes to select a plan")
        batch_size, seq_len_q, num_qo_heads, head_dim = map(int, q.shape)
        _kv_batch_size, seq_len_kv, num_kv_heads, _kv_head_dim = map(int, k.shape)
        output_dtype = out.dtype if isinstance(out, torch.Tensor) else q.dtype
        return _BlockSparsePlanKey(
            device=q.device,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            max_blocks_per_row=routes.max_blocks_per_row,
            mask_type=mask_type,
            q_dtype=q.dtype,
            kv_dtype=k.dtype,
            output_dtype=output_dtype,
            use_kv_valid_bits=kv_valid_bits is not None,
        )

    def _get_or_plan(self, key: _BlockSparsePlanKey) -> object:
        wrapper = self._plans.get(key)
        if wrapper is not None:
            return wrapper
        if _is_current_stream_capturing(key.device):
            raise RuntimeError(
                "PrimTS block-sparse plan cache miss during CUDA Graph capture; "
                "run an eager warmup with the same static profile first"
            )
        if _BlockSparseTSWrapper is None:
            raise RuntimeError(
                "PrimTS block-sparse attention is unavailable: "
                f"{_prims_ts_block_sparse_import_error}"
            )

        candidate = _BlockSparseTSWrapper()
        candidate.plan(
            key.batch_size,
            key.seq_len_q,
            key.seq_len_kv,
            key.num_qo_heads,
            key.num_kv_heads,
            key.head_dim,
            key.q_block_size,
            key.kv_block_size,
            device=key.device,
            max_blocks_per_row=key.max_blocks_per_row,
            use_kv_valid_bits=key.use_kv_valid_bits,
            mask_type=key.mask_type,
            q_data_type=key.q_dtype,
            kv_data_type=key.kv_dtype,
            o_data_type=key.output_dtype,
        )
        self._plans[key] = candidate
        return candidate

    @staticmethod
    def _make_paged_plan_key(
        *,
        device: torch.device,
        batch_size: int,
        seq_len_q: int,
        max_seq_len_kv: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        routes: BlockSparseRoutes,
        q_block_size: int,
        kv_block_size: int,
        page_size: int,
        use_kv_valid_bits: bool,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        output_dtype: torch.dtype,
        mask_type: Literal["dense", "causal"],
    ) -> _PagedBlockSparsePlanKey:
        if not isinstance(routes, BlockSparseRoutes):
            raise TypeError("routes must be a BlockSparseRoutes instance")
        return _PagedBlockSparsePlanKey(
            device=torch.device(device),
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            max_seq_len_kv=max_seq_len_kv,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            page_size=page_size,
            max_blocks_per_row=routes.max_blocks_per_row,
            mask_type=mask_type,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            output_dtype=output_dtype,
            use_kv_valid_bits=use_kv_valid_bits,
        )

    def _get_or_plan_paged(self, key: _PagedBlockSparsePlanKey) -> object:
        wrapper = self._paged_plans.get(key)
        if wrapper is not None:
            return wrapper
        if _is_current_stream_capturing(key.device):
            raise RuntimeError(
                "PrimTS paged block-sparse plan cache miss during CUDA Graph capture; "
                "run an eager warmup with the same static profile first"
            )
        if _BlockSparsePagedTSWrapper is None:
            raise RuntimeError(
                "PrimTS paged block-sparse attention is unavailable: "
                f"{_prims_ts_block_sparse_import_error}"
            )

        candidate = _BlockSparsePagedTSWrapper()
        candidate.plan(
            key.batch_size,
            key.seq_len_q,
            key.max_seq_len_kv,
            key.num_qo_heads,
            key.num_kv_heads,
            key.head_dim,
            key.q_block_size,
            key.kv_block_size,
            key.page_size,
            device=key.device,
            max_blocks_per_row=key.max_blocks_per_row,
            use_kv_valid_bits=key.use_kv_valid_bits,
            mask_type=key.mask_type,
            q_data_type=key.q_dtype,
            kv_data_type=key.kv_dtype,
            o_data_type=key.output_dtype,
        )
        self._paged_plans[key] = candidate
        return candidate

    def ensure_paged_plan(
        self,
        *,
        device: torch.device,
        batch_size: int,
        seq_len_q: int,
        max_seq_len_kv: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        routes: BlockSparseRoutes,
        q_block_size: int,
        kv_block_size: int,
        page_size: int,
        use_kv_valid_bits: bool,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        output_dtype: torch.dtype,
        mask_type: Literal["dense", "causal"] = "dense",
    ) -> object:
        """Compile or find the static paged plan before KV-cache mutation."""

        key = self._make_paged_plan_key(
            device=device,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            max_seq_len_kv=max_seq_len_kv,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            routes=routes,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            page_size=page_size,
            use_kv_valid_bits=use_kv_valid_bits,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            output_dtype=output_dtype,
            mask_type=mask_type,
        )
        return self._get_or_plan_paged(key)

    @torch.compiler.disable
    def run_contiguous(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        routes: BlockSparseRoutes,
        q_block_size: int,
        kv_block_size: int,
        kv_valid_bits: torch.Tensor | None = None,
        mask_type: Literal["dense", "causal"] = "dense",
        sm_scale: float | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one dynamic BSR pattern through a cached static PrimTS plan."""

        key = self._make_plan_key(
            q,
            k,
            routes=routes,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            kv_valid_bits=kv_valid_bits,
            mask_type=mask_type,
            out=out,
        )
        wrapper = self._get_or_plan(key)
        return wrapper.run(
            q,
            k,
            v,
            routes.block_indptr,
            routes.block_indices,
            kv_valid_bits=kv_valid_bits,
            sm_scale=sm_scale,
            out=out,
        )

    @staticmethod
    def _paged_cache_traits(
        paged_kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.dtype, int, int, int]:
        if isinstance(paged_kv_cache, tuple):
            if len(paged_kv_cache) != 2:
                raise ValueError("paged_kv_cache tuple must contain K and V")
            k_cache, v_cache = paged_kv_cache
            if not all(isinstance(cache, torch.Tensor) for cache in (k_cache, v_cache)):
                raise TypeError("paged K and V caches must be torch.Tensor instances")
            if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
                raise ValueError("paged K and V caches must share [P, Hkv, page, D]")
            return (
                k_cache.dtype,
                int(k_cache.shape[1]),
                int(k_cache.shape[2]),
                int(k_cache.shape[3]),
            )
        if not isinstance(paged_kv_cache, torch.Tensor):
            raise TypeError("paged_kv_cache must be a tensor or a (K, V) tuple")
        if paged_kv_cache.ndim != 5 or int(paged_kv_cache.shape[1]) != 2:
            raise ValueError("combined paged KV cache must have [P, 2, Hkv, page, D]")
        return (
            paged_kv_cache.dtype,
            int(paged_kv_cache.shape[2]),
            int(paged_kv_cache.shape[3]),
            int(paged_kv_cache.shape[4]),
        )

    @torch.compiler.disable
    def run_paged(
        self,
        q: torch.Tensor,
        paged_kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        *,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        routes: BlockSparseRoutes,
        q_block_size: int,
        kv_block_size: int,
        page_size: int,
        max_seq_len_kv: int,
        kv_valid_bits: torch.Tensor | None = None,
        mask_type: Literal["dense", "causal"] = "dense",
        sm_scale: float | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a pre-materialized paged cache through a capacity-only plan.

        This direct API is for callers that already own paged K/V tensors and
        live page metadata. The TRT-LLM FMHA adapter prepares its plan before
        THOP appends K/V, then runs that exact wrapper to preserve transactional
        ordering.
        """

        if not isinstance(q, torch.Tensor) or q.ndim != 4:
            raise ValueError("paged block-sparse Q must have BSHD rank 4")
        kv_dtype, num_kv_heads, cache_page_size, cache_head_dim = self._paged_cache_traits(
            paged_kv_cache
        )
        batch_size, seq_len_q, num_qo_heads, head_dim = map(int, q.shape)
        if cache_page_size != page_size or cache_head_dim != head_dim:
            raise ValueError("paged KV cache geometry does not match Q and page_size")
        output_dtype = out.dtype if isinstance(out, torch.Tensor) else q.dtype
        wrapper = self.ensure_paged_plan(
            device=q.device,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            max_seq_len_kv=max_seq_len_kv,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            routes=routes,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            page_size=page_size,
            use_kv_valid_bits=kv_valid_bits is not None,
            q_dtype=q.dtype,
            kv_dtype=kv_dtype,
            output_dtype=output_dtype,
            mask_type=mask_type,
        )
        return wrapper.run(
            q,
            paged_kv_cache,
            paged_kv_indptr,
            paged_kv_indices,
            seq_lens_kv,
            routes.block_indptr,
            routes.block_indices,
            kv_valid_bits=kv_valid_bits,
            sm_scale=sm_scale,
            out=out,
        )


def get_or_create_prims_ts_block_sparse_runtime(
    state: MutableMapping[str, object] | None,
) -> PrimsTSBlockSparseRuntime:
    """Return a per-layer runtime or the component-scoped shared runtime."""

    if state is None:
        return PrimsTSBlockSparseRuntime()
    runtime = state.get(PRIMS_TS_BLOCK_SPARSE_RUNTIME_STATE_KEY)
    if runtime is None:
        runtime = PrimsTSBlockSparseRuntime()
        state[PRIMS_TS_BLOCK_SPARSE_RUNTIME_STATE_KEY] = runtime
    if not isinstance(runtime, PrimsTSBlockSparseRuntime):
        raise TypeError(
            f"{PRIMS_TS_BLOCK_SPARSE_RUNTIME_STATE_KEY} must contain PrimsTSBlockSparseRuntime"
        )
    return runtime


# Registered block-sparse FMHA and TRT-LLM storage adaptation.


@dataclass(frozen=True, slots=True)
class _PagedProfile:
    """Side-effect-free, fixed-shape profile for one paged dispatch.

    ``max_seq_len_kv`` is the padded page-table capacity used for planning.
    """

    batch_size: int
    seq_len_q: int
    max_seq_len_kv: int
    max_past_kv_length: int
    page_size: int
    mask_type: Literal["dense", "causal"]
    local_layer_idx: int
    kv_page_offset: int
    total_num_blocks: int


class PrimsTSBlockSparseFmha(Fmha):
    """Generic block-sparse FMHA for contiguous Q/K/V and paged KV cache."""

    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

    def __init__(self, attn: "TrtllmAttention") -> None:
        super().__init__(attn)
        self._sparse_params = cast(BlockSparseParams, attn.sparse_params)
        self._num_heads = int(attn.num_heads)
        self._num_kv_heads = int(attn.num_kv_heads)
        self._head_dim = int(attn.head_dim)
        self._q_scaling = float(attn.q_scaling)
        self._sm_scale = 1.0 / (math.sqrt(self._head_dim) * self._q_scaling)
        self._quant_mode = attn.quant_mode
        self._predicted_tokens_per_seq = int(attn.predicted_tokens_per_seq)
        self._position_embedding_type = int(attn.position_embedding_type)
        self._attention_chunk_size = int(attn.attention_chunk_size or 0)
        self._is_mla_enable = bool(attn.is_mla_enable)
        self._runtime = get_or_create_prims_ts_block_sparse_runtime(attn.attention_metadata_state)

        # Paged-only staging remains layer-owned. The PrimTS plan may be shared
        # by a serialized component, but TRT-LLM metadata buffers belong to the
        # layer whose THOP preprocessing populates them.
        self._page_indices_buffer: torch.Tensor | None = None
        self._fixed_indptr_buffer: torch.Tensor | None = None
        self._page_row_capacity = 0
        self._page_column_capacity = 0
        self._retained_page_buffers: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._multi_processor_count: int | None = None

    @classmethod
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        return isinstance(attn.sparse_params, BlockSparseParams)

    @staticmethod
    def _mask_type(
        forward_args: AttentionForwardArgs,
    ) -> Literal["dense", "causal"] | None:
        if forward_args.attention_mask == PredefinedAttentionMask.FULL:
            return "dense"
        if forward_args.attention_mask == PredefinedAttentionMask.CAUSAL:
            return "causal"
        return None

    @staticmethod
    def _has_legacy_sparse_prediction(forward_args: AttentionForwardArgs) -> bool:
        prediction = forward_args.sparse_prediction
        return any(
            value is not None
            for value in (
                prediction.sparse_kv_indices,
                prediction.sparse_kv_offsets,
                prediction.sparse_attn_indices,
                prediction.sparse_attn_offsets,
                prediction.sparse_mla_topk_lens,
                prediction.compressed_kv_cache_pool_ptr,
            )
        ) or bool(prediction.sparse_attn_indices_block_size)

    def _common_unsupported_reason(
        self,
        q: torch.Tensor,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> str | None:
        """Validate semantics shared by both storage modes."""

        inputs = forward_args.block_sparse_inputs
        if not isinstance(inputs, BlockSparseForwardInputs):
            return "live block-sparse forward inputs are required"
        if not isinstance(inputs.routes, BlockSparseRoutes):
            return "block-sparse routes are required"
        if getattr(metadata, "is_cross", False):
            return "cross attention is not supported"
        output = forward_args.output
        if not isinstance(output, torch.Tensor):
            return "a caller-owned output tensor is required"
        if not isinstance(q, torch.Tensor) or q.ndim != 2 or not q.is_contiguous():
            return "attention input must be a contiguous rank-2 tensor"
        if output.ndim != 2 or not output.is_contiguous():
            return "output must be a contiguous rank-2 tensor"
        if output.device != q.device or output.dtype != q.dtype:
            return "Q and output must share device and dtype"
        if output.numel() != q.shape[0] * self._num_heads * self._head_dim:
            return "output has an incompatible extent"
        if q.dtype not in self._SUPPORTED_DTYPES:
            return f"query dtype {q.dtype} is unsupported"
        if not math.isfinite(self._q_scaling) or self._q_scaling <= 0:
            return "q_scaling must be finite and positive"
        if self._is_mla_enable:
            return "MLA is not supported"
        if getattr(metadata, "helix_position_offsets", None) is not None:
            return "Helix parallelism is not supported"
        if int(getattr(metadata, "num_sparse_topk", 0)) > 0:
            return "legacy sparse attention metadata is not supported"
        if self._has_legacy_sparse_prediction(forward_args):
            return "legacy sparse prediction cannot be combined with block-sparse inputs"
        if forward_args.enable_dsv4_epilogue_fusion:
            return "DSv4 epilogue fusion is not supported"
        if forward_args.sage_attn_qk_int8 or any(
            getattr(forward_args, name) > 0
            for name in (
                "sage_attn_num_elts_per_blk_q",
                "sage_attn_num_elts_per_blk_k",
                "sage_attn_num_elts_per_blk_v",
            )
        ):
            return "SageAttention is not supported"
        if forward_args.softmax_stats_tensor is not None:
            return "softmax statistics output is not supported"
        if (
            forward_args.output_sf is not None
            or forward_args.out_scale is not None
            or forward_args.out_scale_sf is not None
        ):
            return "quantized output is not supported"
        if (
            forward_args.attention_mask_data is not None
            or forward_args.relative_attention_bias is not None
            or forward_args.attention_sinks is not None
        ):
            return "custom attention masks, bias, and sinks are not supported"
        if self._mask_type(forward_args) is None:
            return "only full and causal masks are supported"
        return None

    # Paged storage: TRT-LLM QKV preprocessing, KV append, and page-table lowering.
    @staticmethod
    def _preprocess_workspace_reason(
        q: torch.Tensor,
        workspace: object,
    ) -> str | None:
        if not isinstance(workspace, torch.Tensor):
            return "TRT-LLM QKV preprocessing workspace is unavailable"
        if workspace.device != q.device:
            return "TRT-LLM QKV preprocessing workspace must be on the query device"
        if workspace.dtype not in (torch.int8, torch.uint8):
            return "TRT-LLM QKV preprocessing workspace must use byte storage"
        if workspace.ndim != 1 or not workspace.is_contiguous():
            return "TRT-LLM QKV preprocessing workspace must be compact rank-1 storage"
        if workspace.data_ptr() % 16:
            return "TRT-LLM QKV preprocessing workspace must be 16-byte aligned"
        return None

    @staticmethod
    def _cache_dtype(metadata: "TrtllmAttentionMetadata") -> torch.dtype | None:
        cache_dtype = getattr(metadata.kv_cache_manager, "dtype", None)
        if isinstance(cache_dtype, torch.dtype):
            return cache_dtype
        return {
            DataType.HALF: torch.float16,
            DataType.BF16: torch.bfloat16,
            DataType.FP8: torch.float8_e4m3fn,
        }.get(cache_dtype)

    def _get_local_layer_idx(self, metadata: "TrtllmAttentionMetadata") -> int:
        """Resolve the live cache-layer index after TRT-LLM primes it."""

        owner = self.attn
        local_layer_idx = getattr(owner, "local_layer_idx", None)
        if local_layer_idx is not None:
            return int(local_layer_idx)
        return int(owner.get_local_layer_idx(metadata))

    def _paged_profile(
        self,
        q: torch.Tensor,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> tuple[_PagedProfile | None, str | None]:
        common_reason = self._common_unsupported_reason(q, metadata, forward_args)
        if common_reason is not None:
            return None, common_reason
        if not all(
            callable(getattr(thop, name, None))
            for name in (
                "get_trtllm_gen_generation_workspace_layout",
                "trtllm_gen_generation_preprocess",
            )
        ):
            return None, "TRT-LLM generation preprocessing ops are unavailable"
        if not forward_args.is_fused_qkv:
            return None, "paged block-sparse attention requires fused QKV input"
        if forward_args.attention_input_type != AttentionInputType.generation_only:
            return None, "only generation-only paged requests are supported"
        if int(getattr(metadata, "num_contexts", 0)) != 0:
            return None, "paged block-sparse attention does not support context requests"
        batch_size = int(getattr(metadata, "num_generations", 0))
        if batch_size <= 0 or q.shape[0] % batch_size:
            return None, "query tokens must be uniformly divisible across generation requests"
        seq_len_q = int(q.shape[0]) // batch_size
        if seq_len_q <= 0:
            return None, "each generation request must contain at least one query token"
        query_lengths = getattr(metadata, "seq_lens", None)
        if (
            not isinstance(query_lengths, torch.Tensor)
            or query_lengths.device.type != "cpu"
            or query_lengths.dtype != torch.int32
            or not query_lengths.is_contiguous()
            or query_lengths.numel() < batch_size
        ):
            return (
                None,
                "host int32 query lengths are required to prove a uniform fixed query shape",
            )
        active_query_lengths = query_lengths[:batch_size]
        if not bool(active_query_lengths.eq(seq_len_q).all()):
            return None, "query lengths must be batch-uniform and match the fixed query shape"
        expected_width = (self._num_heads + 2 * self._num_kv_heads) * self._head_dim
        if int(q.shape[1]) != expected_width:
            return None, f"fused QKV width must be {expected_width}"
        if getattr(metadata, "kv_cache_manager", None) is None:
            return None, "a paged KV-cache manager is required"
        pool_pointers = getattr(metadata, "host_kv_cache_pool_pointers", None)
        pool_mapping = getattr(metadata, "host_kv_cache_pool_mapping", None)
        if pool_pointers is None or pool_mapping is None:
            return None, "KV-cache pool pointers and layer mapping are required"
        if (
            not isinstance(pool_pointers, torch.Tensor)
            or pool_pointers.dtype != torch.int64
            or pool_pointers.device.type != "cpu"
            or pool_pointers.ndim != 2
            or int(pool_pointers.shape[1]) != 2
            or not pool_pointers.is_contiguous()
        ):
            return None, "KV-cache pool pointers must be compact host int64 [pools, 2] metadata"
        local_layer_idx = self._get_local_layer_idx(metadata)
        if (
            not isinstance(pool_mapping, torch.Tensor)
            or pool_mapping.dtype != torch.int32
            or pool_mapping.device.type != "cpu"
            or not pool_mapping.is_contiguous()
            or pool_mapping.ndim != 2
            or int(pool_mapping.shape[1]) < 2
            or not 0 <= local_layer_idx < int(pool_mapping.shape[0])
        ):
            return None, "KV-cache layer-to-pool mapping is invalid"
        manager = metadata.kv_cache_manager
        is_v2 = callable(
            getattr(getattr(manager, "impl", None), "get_page_index_upper_bound", None)
        )
        if not is_v2 and int(getattr(manager, "num_pools", 1)) != 1:
            return None, "KVCacheManagerV1 with multiple pools is not supported"
        pool_index = int(pool_mapping[local_layer_idx, 0])
        layer_in_pool = int(pool_mapping[local_layer_idx, 1])
        if pool_index < 0 or layer_in_pool < 0 or pool_index >= int(pool_pointers.shape[0]):
            return None, "KV-cache layer-to-pool mapping is invalid"
        primary_pool_address = int(pool_pointers[pool_index, 0])
        if primary_pool_address <= 0:
            return None, "the selected primary KV-cache pool pointer must be positive"
        if not is_v2 and (
            pool_index != 0 or layer_in_pool >= int(getattr(manager, "num_local_layers", 1))
        ):
            return None, "KVCacheManagerV1 layer-to-pool mapping is invalid"
        kv_page_offset = self._kv_page_offset(metadata, local_layer_idx)
        if kv_page_offset is None:
            return None, "the paged K-to-V page displacement could not be resolved"
        total_num_blocks = self._total_num_blocks(metadata, local_layer_idx)
        if total_num_blocks <= 0:
            return None, "the KV-cache page-pool extent could not be resolved"
        block_tables = getattr(metadata, "kv_cache_block_offsets", None)
        if not isinstance(block_tables, torch.Tensor):
            return None, "paged KV-cache block offsets are required"
        if (
            block_tables.ndim != 4
            or int(block_tables.shape[2]) != 2
            or pool_index >= int(block_tables.shape[0])
            or int(block_tables.shape[1]) < batch_size
        ):
            return None, "KV-cache block offsets must have shape [pools, B, 2, pages]"
        if block_tables.dtype != torch.int32 or block_tables.device != q.device:
            return None, "KV-cache block offsets must be int32 on the query device"
        if not block_tables.is_contiguous() or int(block_tables.shape[-1]) <= 0:
            return None, "KV-cache block offsets do not cover the active batch"
        if getattr(metadata, "kv_layout", "HND") != "HND":
            return None, "only HND paged KV layout is supported"
        if int(getattr(metadata, "beam_width", 1)) != 1:
            return None, "beam search is not supported"
        if any(
            bool(getattr(metadata, name, False))
            for name in (
                "is_spec_decoding_enabled",
                "use_spec_decoding",
                "is_spec_dec_tree",
                "is_spec_dec_dynamic_tree",
            )
        ):
            return None, "speculative decoding is not supported"
        if self._predicted_tokens_per_seq != 1:
            return None, "predicted multi-token generation is not supported"
        if self._attention_chunk_size:
            return None, "chunked attention is not supported"
        if self._position_embedding_type in (4, 5, 6, 7, 10):
            return None, f"position embedding type {self._position_embedding_type} is not supported"
        if bool(getattr(metadata.kv_cache_manager, "enable_swa_scratch_reuse", False)):
            return None, "SWA scratch reuse is not supported"
        page_size = int(getattr(metadata, "tokens_per_block", 0))
        if page_size <= 0:
            return None, "page size must be positive"
        max_seq_len_kv = int(block_tables.shape[-1]) * page_size
        try:
            logical_max_seq_len = int(metadata.max_seq_len)
        except (AttributeError, TypeError, ValueError):
            return None, "a positive logical maximum sequence length is required"
        if logical_max_seq_len <= 0 or logical_max_seq_len > max_seq_len_kv:
            return None, "logical maximum sequence length must fit the page-table capacity"
        attention_window_size = forward_args.attention_window_size
        if (
            isinstance(attention_window_size, bool)
            or not isinstance(attention_window_size, int)
            or attention_window_size < logical_max_seq_len
        ):
            return None, "sliding-window/cyclic page tables are not supported"
        try:
            quant_mode = QuantMode(self._quant_mode)
        except (TypeError, ValueError):
            return None, "invalid quantization mode"
        if quant_mode.has_kv_cache_quant():
            return None, "quantized KV cache is not supported"
        cache_dtype = self._cache_dtype(metadata)
        if cache_dtype != q.dtype:
            return None, "query, paged KV cache, and output dtypes must match"
        host_seq_lens = getattr(metadata, "kv_lens_runtime", None)
        if (
            not isinstance(host_seq_lens, torch.Tensor)
            or host_seq_lens.dtype != torch.int32
            or host_seq_lens.device.type != "cpu"
            or host_seq_lens.numel() < batch_size
            or not host_seq_lens.is_contiguous()
        ):
            return None, "live host int32 KV lengths are required for safe policy selection"
        active_host_seq_lens = host_seq_lens[:batch_size]
        min_seq_len_kv = int(active_host_seq_lens.min())
        max_past_kv_length = int(active_host_seq_lens.max())
        if min_seq_len_kv <= 0:
            return None, "every active request must contain at least one KV token"
        mask_type = self._mask_type(forward_args)
        assert mask_type is not None
        if mask_type == "causal" and min_seq_len_kv < seq_len_q:
            return None, "causal KV lengths must be at least the fixed query length"
        if max_past_kv_length > logical_max_seq_len:
            return None, "an active KV length exceeds the logical maximum sequence length"
        seq_lens = getattr(metadata, "kv_lens_cuda_runtime", None)
        if (
            not isinstance(seq_lens, torch.Tensor)
            or seq_lens.dtype != torch.int32
            or seq_lens.device != q.device
            or seq_lens.numel() != batch_size
            or not seq_lens.is_contiguous()
            or seq_lens.data_ptr() % 4
        ):
            return None, "live int32 GPU KV lengths are required"
        inputs = forward_args.block_sparse_inputs
        assert isinstance(inputs, BlockSparseForwardInputs)
        route_reason = _get_prims_ts_block_sparse_metadata_unsupported_reason(
            inputs.routes,
            inputs.kv_valid_bits,
            device=q.device,
            batch_size=batch_size,
            num_kv_heads=self._num_kv_heads,
            seq_len_q=seq_len_q,
            seq_len_kv=max_seq_len_kv,
            q_block_size=self._sparse_params.q_block_size,
        )
        if route_reason is not None:
            return None, route_reason
        profile = _PagedProfile(
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            max_seq_len_kv=max_seq_len_kv,
            max_past_kv_length=max_past_kv_length,
            page_size=page_size,
            mask_type=mask_type,
            local_layer_idx=local_layer_idx,
            kv_page_offset=kv_page_offset,
            total_num_blocks=total_num_blocks,
        )
        return (profile, None)

    def _ensure_page_table_buffers(
        self,
        device: torch.device,
        *,
        row_capacity: int,
        column_capacity: int,
    ) -> None:
        needs_allocation = (
            self._page_indices_buffer is None
            or self._fixed_indptr_buffer is None
            or self._page_indices_buffer.device != device
            or self._page_row_capacity < row_capacity
            or self._page_column_capacity != column_capacity
        )
        if not needs_allocation:
            return
        if _is_current_stream_capturing(device):
            raise RuntimeError(
                "PrimTS block-sparse page-table buffers must be allocated before CUDA Graph capture"
            )
        if self._page_indices_buffer is not None and self._fixed_indptr_buffer is not None:
            # A captured graph may still reference a previous-capacity buffer.
            self._retained_page_buffers.append(
                (self._page_indices_buffer, self._fixed_indptr_buffer)
            )
        self._page_indices_buffer = torch.empty(
            (row_capacity, column_capacity),
            dtype=torch.int32,
            device=device,
        )
        self._fixed_indptr_buffer = torch.arange(
            row_capacity + 1,
            dtype=torch.int32,
            device=device,
        ).mul_(column_capacity)
        self._page_row_capacity = row_capacity
        self._page_column_capacity = column_capacity

    def _stage_page_table(
        self,
        block_tables: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._page_indices_buffer is None
            or self._fixed_indptr_buffer is None
            or block_tables.ndim != 3
            or int(block_tables.shape[0]) < batch_size
            or int(block_tables.shape[1]) != 2
            or int(block_tables.shape[2]) != self._page_column_capacity
            or block_tables.dtype != torch.int32
            or block_tables.device != self._page_indices_buffer.device
            or not block_tables.is_contiguous()
        ):
            raise RuntimeError("PrimTS block-sparse page-table storage was not prepared")
        page_table = self._page_indices_buffer[:batch_size]
        page_table.copy_(block_tables[:batch_size, 0, :])
        return self._fixed_indptr_buffer[: batch_size + 1], page_table.reshape(-1)

    @staticmethod
    def _kv_page_offset(
        metadata: "TrtllmAttentionMetadata",
        local_layer_idx: int,
    ) -> int | None:
        manager = metadata.kv_cache_manager
        pool_mapping = metadata.host_kv_cache_pool_mapping
        if manager is None or pool_mapping is None:
            return None
        pool_index = int(pool_mapping[local_layer_idx, 0])
        kv_offsets = getattr(manager, "kv_offset", None)
        if kv_offsets is not None:
            offset = int(kv_offsets[pool_index])
            if offset > 0:
                return offset
        host_offsets = getattr(manager, "host_kv_cache_block_offsets", None)
        if host_offsets is None or host_offsets.ndim != 4 or pool_index >= host_offsets.shape[0]:
            return None
        deltas = host_offsets[pool_index, :, 1] - host_offsets[pool_index, :, 0]
        positive = deltas[deltas > 0]
        return int(positive[0]) if positive.numel() else None

    @staticmethod
    def _kv_views(
        kv_pool: torch.Tensor,
        kv_page_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if kv_pool.ndim != 4:
            raise RuntimeError("TRT-LLM preprocessing returned an invalid KV page pool")
        usable_pages = int(kv_pool.shape[0]) - kv_page_offset
        if kv_page_offset <= 0 or usable_pages <= 0:
            raise RuntimeError("TRT-LLM preprocessing returned an invalid K-to-V page offset")
        return (
            kv_pool.narrow(0, 0, usable_pages),
            kv_pool.narrow(0, kv_page_offset, usable_pages),
        )

    @staticmethod
    def _total_num_blocks(
        metadata: "TrtllmAttentionMetadata",
        local_layer_idx: int,
    ) -> int:
        manager = metadata.kv_cache_manager
        if manager is None:
            return 0
        get_page_index_upper_bound = getattr(
            getattr(manager, "impl", None),
            "get_page_index_upper_bound",
            None,
        )
        if callable(get_page_index_upper_bound):
            from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import Role

            return int(get_page_index_upper_bound(local_layer_idx, Role.KEY))
        blocks = getattr(manager, "blocks_in_primary_pool", None)
        if blocks is None:
            blocks_per_window = getattr(manager, "blocks_per_window", None)
            if blocks_per_window:
                blocks = max(int(primary) for primary, _ in blocks_per_window.values())
        if blocks is None:
            return 0
        total = int(blocks) * int(getattr(manager, "num_local_layers", 1)) * 2
        mapping = metadata.host_kv_cache_pool_mapping
        if mapping is not None:
            layer_in_pool = int(mapping[local_layer_idx, 1])
            total -= layer_in_pool * 2
        return total

    def _ensure_preprocess_workspace(
        self,
        q: torch.Tensor,
        metadata: "TrtllmAttentionMetadata",
        *,
        batch_size: int,
    ) -> torch.Tensor:
        workspace = metadata.effective_workspace
        workspace_reason = self._preprocess_workspace_reason(q, workspace)
        if workspace_reason is not None:
            raise RuntimeError(workspace_reason)
        assert isinstance(workspace, torch.Tensor)
        # Unit-level CPU adapters mock the preprocessing op. Production calls
        # are CUDA-only and use the authoritative THOP workspace layout.
        if q.device.type != "cuda":
            return workspace
        layout = thop.get_trtllm_gen_generation_workspace_layout(
            q.dtype,
            batch_size,
            int(q.shape[0]),
            self._num_heads,
            self._head_dim,
            int(self.attn.rope_params.dim),
            self._num_kv_heads,
        )
        required_bytes = int(layout["total_size"])
        available_bytes = workspace.numel() * workspace.element_size()
        if available_bytes < required_bytes:
            if _is_current_stream_capturing(q.device):
                raise RuntimeError(
                    "TRT-LLM QKV preprocessing workspace must be sized before CUDA Graph capture"
                )
            workspace.resize_((math.ceil(required_bytes / workspace.element_size()),))
        return workspace

    def _forward_paged(
        self,
        q: torch.Tensor,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        profile: _PagedProfile,
    ) -> None:
        inputs = forward_args.block_sparse_inputs
        assert isinstance(inputs, BlockSparseForwardInputs)
        assert forward_args.output is not None
        batch_size = profile.batch_size
        seq_len_q = profile.seq_len_q
        page_size = profile.page_size
        max_seq_len_kv = profile.max_seq_len_kv
        mask_type = profile.mask_type

        # Finish adapter-owned planning and allocation before THOP appends K/V.
        # Keep this exact wrapper so the post-append path cannot plan again.
        prepared_wrapper = self._runtime.ensure_paged_plan(
            device=q.device,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            max_seq_len_kv=max_seq_len_kv,
            num_qo_heads=self._num_heads,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            routes=inputs.routes,
            q_block_size=self._sparse_params.q_block_size,
            kv_block_size=self._sparse_params.kv_block_size,
            page_size=page_size,
            use_kv_valid_bits=inputs.kv_valid_bits is not None,
            q_dtype=q.dtype,
            kv_dtype=q.dtype,
            output_dtype=forward_args.output.dtype,
            mask_type=mask_type,
        )
        block_offsets = metadata.kv_cache_block_offsets
        self._ensure_page_table_buffers(
            block_offsets.device,
            row_capacity=batch_size,
            column_capacity=int(block_offsets.shape[-1]),
        )
        workspace = self._ensure_preprocess_workspace(q, metadata, batch_size=batch_size)
        if self._multi_processor_count is None:
            if q.device.type != "cuda":
                self._multi_processor_count = 1
            elif _is_current_stream_capturing(q.device):
                raise RuntimeError("GPU properties must be prepared before CUDA Graph capture")
            else:
                self._multi_processor_count = torch.cuda.get_device_properties(
                    q.device
                ).multi_processor_count

        local_layer_idx = profile.local_layer_idx
        sequence_lengths = metadata.kv_lens_cuda_runtime
        max_past_kv_length = profile.max_past_kv_length
        total_num_blocks = profile.total_num_blocks
        owner = self.attn
        rope = owner.rope_params
        (
            q_processed,
            kv_pool,
            block_tables,
            _kv_scale_pool,
            _bmm1_scale,
            _bmm2_scale,
            _fmha_workspace,
            _cu_seqlens,
            _max_q_len,
            _max_kv_len,
            _window_left,
            _is_multi_token_gen,
        ) = thop.trtllm_gen_generation_preprocess(
            q,
            workspace,
            sequence_lengths,
            None,
            None,
            metadata.kv_cache_block_offsets,
            metadata.host_kv_cache_pool_pointers,
            metadata.host_kv_cache_pool_mapping,
            forward_args.kv_scale_orig_quant,
            forward_args.kv_scale_quant_orig,
            forward_args.out_scale,
            owner.rotary_inv_freq,
            owner.rotary_cos_sin,
            forward_args.mrope_position_deltas,
            local_layer_idx,
            0,
            self._num_heads,
            self._num_kv_heads,
            self._head_dim,
            page_size,
            self._quant_mode,
            int(forward_args.attention_window_size),
            int(forward_args.attention_window_size),
            int(q.shape[0]),
            batch_size,
            seq_len_q,
            max_past_kv_length,
            int(rope.dim),
            float(rope.theta),
            int(rope.scale_type),
            float(rope.scale),
            int(rope.max_positions),
            self._position_embedding_type,
            self._sm_scale,
            1.0,
            False,
            self._predicted_tokens_per_seq,
            self._attention_chunk_size,
            self._multi_processor_count,
            total_num_blocks,
            2,
            True,
            False,
        )
        if _is_multi_token_gen:
            raise RuntimeError(
                "TRT-LLM preprocessing reported an unexpected speculative or "
                "variable-query generation profile"
            )
        if q_processed is None or kv_pool is None or block_tables is None:
            raise RuntimeError("TRT-LLM preprocessing did not return paged PrimTS metadata")
        kv_page_offset = profile.kv_page_offset
        k_cache, v_cache = self._kv_views(kv_pool, kv_page_offset)
        paged_kv_indptr, paged_kv_indices = self._stage_page_table(
            block_tables,
            batch_size,
        )
        query = q_processed.view(
            batch_size,
            seq_len_q,
            self._num_heads,
            self._head_dim,
        )
        output = forward_args.output.view_as(query)
        prepared_wrapper.run(
            query,
            (k_cache, v_cache),
            paged_kv_indptr,
            paged_kv_indices,
            sequence_lengths,
            inputs.routes.block_indptr,
            inputs.routes.block_indices,
            kv_valid_bits=inputs.kv_valid_bits,
            sm_scale=self._sm_scale,
            out=output,
        )

    # Contiguous storage: reshape flat separate Q/K/V into the common BSHD ABI.

    def _contiguous_views(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        routes: BlockSparseRoutes,
        output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(routes.block_indptr.shape[0])
        if batch_size <= 0 or q.shape[0] % batch_size or k.shape[0] % batch_size:
            raise ValueError("flat Q and K token counts must be divisible by route batch size")
        seq_len_q = int(q.shape[0]) // batch_size
        seq_len_kv = int(k.shape[0]) // batch_size
        return (
            q.view(batch_size, seq_len_q, self._num_heads, self._head_dim),
            k.view(batch_size, seq_len_kv, self._num_kv_heads, self._head_dim),
            v.view(batch_size, seq_len_kv, self._num_kv_heads, self._head_dim),
            output.view(batch_size, seq_len_q, self._num_heads, self._head_dim),
        )

    def _contiguous_unsupported_reason(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> str | None:
        common_reason = self._common_unsupported_reason(q, metadata, forward_args)
        if common_reason is not None:
            return common_reason
        if self._position_embedding_type != 0 or forward_args.mrope_position_deltas is not None:
            return "contiguous Q/K/V must have position embedding applied before attention"
        if forward_args.is_fused_qkv or k is None or v is None:
            return "contiguous block-sparse attention requires separate Q, K, and V"
        if not all(tensor.ndim == 2 and tensor.is_contiguous() for tensor in (k, v)):
            return "K and V must be contiguous rank-2 tensors"
        if k.shape != v.shape:
            return "K and V must have identical shapes"
        if any(tensor.device != q.device or tensor.dtype != q.dtype for tensor in (k, v)):
            return "Q, K, and V must share device and dtype"
        if int(k.shape[1]) != self._num_kv_heads * self._head_dim:
            return "K and V hidden dimensions do not match the attention configuration"
        inputs = forward_args.block_sparse_inputs
        assert isinstance(inputs, BlockSparseForwardInputs)
        routes = inputs.routes
        try:
            q_view, k_view, v_view, _ = self._contiguous_views(q, k, v, routes, forward_args.output)
        except (RuntimeError, ValueError) as error:
            return str(error)
        batch_size, seq_len_q = map(int, q_view.shape[:2])
        query_lengths = getattr(metadata, "seq_lens", None)
        if (
            not isinstance(query_lengths, torch.Tensor)
            or query_lengths.device.type != "cpu"
            or query_lengths.dtype != torch.int32
            or not query_lengths.is_contiguous()
            or query_lengths.numel() < batch_size
        ):
            return "host int32 query lengths are required to prove a uniform fixed query shape"
        if not bool(query_lengths[:batch_size].eq(seq_len_q).all()):
            return "query lengths must be batch-uniform and match the fixed query shape"
        return None

    def _forward_contiguous(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_args: AttentionForwardArgs,
    ) -> None:
        inputs = forward_args.block_sparse_inputs
        assert isinstance(inputs, BlockSparseForwardInputs)
        assert forward_args.output is not None
        q_view, k_view, v_view, out_view = self._contiguous_views(
            q,
            k,
            v,
            inputs.routes,
            forward_args.output,
        )
        mask_type = self._mask_type(forward_args)
        assert mask_type is not None
        self._runtime.run_contiguous(
            q_view,
            k_view,
            v_view,
            routes=inputs.routes,
            q_block_size=self._sparse_params.q_block_size,
            kv_block_size=self._sparse_params.kv_block_size,
            kv_valid_bits=inputs.kv_valid_bits,
            mask_type=mask_type,
            sm_scale=self._sm_scale,
            out=out_view,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        if getattr(metadata, "kv_cache_manager", None) is None:
            reason = self._contiguous_unsupported_reason(q, k, v, metadata, forward_args)
            if reason is not None:
                raise RuntimeError(f"unsupported contiguous block-sparse request: {reason}")
            assert k is not None and v is not None
            self._forward_contiguous(q, k, v, forward_args)
            return

        if k is not None or v is not None:
            raise RuntimeError("paged block-sparse attention requires fused QKV input")
        profile, reason = self._paged_profile(q, metadata, forward_args)
        if reason is not None or profile is None:
            raise RuntimeError(f"unsupported paged block-sparse request: {reason}")
        self._forward_paged(q, metadata, forward_args, profile)


__all__ = [
    "PRIMS_TS_BLOCK_SPARSE_RUNTIME_STATE_KEY",
    "PrimsTSBlockSparseFmha",
    "PrimsTSBlockSparseRuntime",
    "get_or_create_prims_ts_block_sparse_runtime",
    "get_prims_ts_block_sparse_contiguous_unsupported_reason",
    "get_prims_ts_block_sparse_unavailability_reason",
]
