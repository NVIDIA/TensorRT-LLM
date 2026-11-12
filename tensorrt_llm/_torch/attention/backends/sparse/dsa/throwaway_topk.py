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
"""Fused MQA indexer + exact Top-K ("throwaway Top-K") for DSA prefill chunks.

The kernels never materialize the [Q, K] logits matrix, so a prefill chunk is
handled by one wrapper call per query tile instead of ``fp8_mqa_logits`` followed by a
separate Top-K. The wrapper supports one request per call, so only chunks that
hold a single request (``IndexerPrefillChunkMetadata.kv_prefix_len`` is set)
are eligible; everything else keeps using the existing path.
"""

from __future__ import annotations

import os
import weakref
from typing import Dict, Optional, Set, Tuple

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.logger import logger

THROWAWAY_TOPK_ENV = "TRTLLM_THROWAWAY_TOPK"
THROWAWAY_MAX_CHUNK_ENV = "TRTLLM_THROWAWAY_TOPK_MAX_CHUNK"
THROWAWAY_LOGITS_DTYPE_ENV = "TRTLLM_THROWAWAY_TOPK_LOGITS_DTYPE"
THROWAWAY_MIN_CONTEXT_ENV = "TRTLLM_THROWAWAY_TOPK_MIN_CONTEXT"

_HEAD_DIM = 128
# (heads, index_topk, compress_ratio) of the indexers the kernels are validated for: DS-V3.2,
# GLM-5.x, DSV4-Flash and DSV4-Pro. Other geometries select kernel variants that are not tested;
# the H4 kernels are validated only under Qwen3.8-Flash-Next's fixed-weight contract
# (``model_config``), which the DSA indexer does not use.
_VALIDATED_GEOMETRIES = frozenset({(64, 2048, 1), (32, 2048, 1), (64, 512, 4), (64, 1024, 4)})
VALIDATED_NUM_HEADS = tuple(sorted({heads for heads, _, _ in _VALIDATED_GEOMETRIES}))
_DEFAULT_MAX_CHUNK = 8192
# Below this many context tokens (history + chunk rows) the unfused DeepGEMM logits + GVR Top-K
# is as fast or faster; from 64K on the fused path wins for every supported indexer geometry
# (DS-V3.2, GLM-5.x, DSV4-Flash/Pro; FP32 and BF16 logits; 4K and 8K query rows).
_DEFAULT_MIN_CONTEXT = 65536
# Query rows of the synthetic chunk ``ThrowawayPrefillTopK.warmup`` runs.
_WARMUP_QUERY_ROWS = 128
_FP8_KEY_ALIGNMENT = 128
_MXFP4_KEY_ALIGNMENT = 256
_LOGITS_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

# Plan tensors per (kv_prefix_len, tile_begin, tile_end); owned by the chunk metadata so
# they are built once per step and reused by every layer.
PlanCache = Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]]


def throwaway_topk_requested() -> bool:
    """Returns True when the user opted in through ``TRTLLM_THROWAWAY_TOPK=1``."""
    return os.environ.get(THROWAWAY_TOPK_ENV, "0") == "1"


def throwaway_topk_min_context() -> int:
    """Returns the smallest chunk context (in tokens) that takes the fused path.

    ``TRTLLM_THROWAWAY_TOPK_MIN_CONTEXT`` overrides the default; 0 sends every eligible chunk
    to the fused path.
    """
    value = os.environ.get(THROWAWAY_MIN_CONTEXT_ENV, str(_DEFAULT_MIN_CONTEXT))
    try:
        min_context = int(value)
    except ValueError:
        min_context = -1
    if min_context < 0:
        raise ValueError(
            f"{THROWAWAY_MIN_CONTEXT_ENV} must be a non-negative integer, got {value!r}"
        )
    return min_context


def throwaway_topk_max_chunk() -> int:
    """Returns the most query rows one kernel launch handles.

    ``TRTLLM_THROWAWAY_TOPK_MAX_CHUNK`` overrides the default; larger chunks run in tiles.
    """
    value = os.environ.get(THROWAWAY_MAX_CHUNK_ENV, str(_DEFAULT_MAX_CHUNK))
    try:
        max_chunk = int(value)
    except ValueError:
        max_chunk = 0
    if max_chunk <= 0:
        raise ValueError(f"{THROWAWAY_MAX_CHUNK_ENV} must be a positive integer, got {value!r}")
    return max_chunk


def throwaway_topk_logits_dtype() -> torch.dtype:
    """Returns the score dtype from ``TRTLLM_THROWAWAY_TOPK_LOGITS_DTYPE`` (default fp32)."""
    name = os.environ.get(THROWAWAY_LOGITS_DTYPE_ENV, "fp32")
    if name not in _LOGITS_DTYPES:
        raise ValueError(
            f"{THROWAWAY_LOGITS_DTYPE_ENV} must be one of {sorted(_LOGITS_DTYPES)}, got {name!r}"
        )
    return _LOGITS_DTYPES[name]


def validate_throwaway_topk_env() -> None:
    """Parses the tuning variables up front so an invalid value fails at model construction
    instead of falling back to the unfused path at the first long chunk.

    Raises:
        ValueError: If a tuning variable holds an invalid value.
    """
    throwaway_topk_min_context()
    throwaway_topk_max_chunk()
    throwaway_topk_logits_dtype()


def throwaway_topk_supported(n_heads: int, head_dim: int) -> bool:
    """Returns True when the DSA indexer may use the fused kernels for this head layout.

    The head counts come from ``_VALIDATED_GEOMETRIES`` so this gate agrees with the one
    ``get_throwaway_prefill_topk`` applies when it builds the runner.
    """
    return IS_CUTLASS_DSL_AVAILABLE and head_dim == _HEAD_DIM and n_heads in VALIDATED_NUM_HEADS


class ThrowawayPrefillTopK:
    """Runs the fused indexer + Top-K for single-request prefill chunks.

    One instance is shared by all layers of a model. The kernel binds K by device
    pointer, so gathered keys are copied into one persistent staging buffer and a single
    binding serves every layer and chunk.
    """

    def __init__(
        self,
        *,
        n_heads: int,
        index_topk: int,
        compress_ratio: int,
        max_context_length: int,
        device: torch.device,
        logits_dtype: Optional[torch.dtype] = None,
        qkv_dtype: str = "fp8",
        model_config: Optional[str] = None,
    ):
        """Builds the wrapper and the K staging buffers.

        Args:
            n_heads: Number of indexer heads (4, 32 or 64).
            index_topk: Number of selected keys per query row.
            compress_ratio: KV compression ratio of the indexer K cache.
            max_context_length: Longest sequence in original (uncompressed) tokens,
                including the query rows.
            device: CUDA device that owns the buffers.
            logits_dtype: Storage dtype of the scores the kernels select over (float32,
                bfloat16 or float16). None reads ``TRTLLM_THROWAWAY_TOPK_LOGITS_DTYPE``
                (``fp32``, ``bf16`` or ``fp16``; default fp32).
            qkv_dtype: ``"fp8"`` for token-scaled FP8 E4M3 queries/keys, or ``"mxfp4"`` for
                packed E2M1 values with UE8M0 scales per 32 channels (the DSA FP4 indexer).
            model_config: Optional kernel model name. ``"Qwen3.8-Flash-Next"`` selects the fixed
                weight contract: every head weight must equal ``128 ** -0.5`` and token-scaled FP8
                key scales must be 1. The kernel policy is otherwise inferred from the geometry,
                and a 4-head geometry alone does not imply the Qwen contract.

        Raises:
            ValueError: If the kernels do not support this configuration.
        """
        if qkv_dtype not in ("fp8", "mxfp4"):
            raise ValueError(f"qkv_dtype must be 'fp8' or 'mxfp4', got {qkv_dtype!r}")
        self._mxfp4 = qkv_dtype == "mxfp4"
        from tensorrt_llm._torch.attention.kernels.blackwell.throwaway_topk.interface import (
            FP8MQATopKWrapper,
        )

        self.max_chunk_size = throwaway_topk_max_chunk()
        if logits_dtype is None:
            logits_dtype = throwaway_topk_logits_dtype()
        elif logits_dtype not in _LOGITS_DTYPES.values():
            raise ValueError(
                f"logits_dtype must be float32, bfloat16 or float16, got {logits_dtype}"
            )
        self.logits_dtype = logits_dtype
        # The K storage holds whole key tiles (FP8: 128 rows; block-scaled MXFP4: 256 rows);
        # plan() bounds the live part.
        alignment = _MXFP4_KEY_ALIGNMENT if self._mxfp4 else _FP8_KEY_ALIGNMENT
        self.num_keys = -(-(max_context_length // compress_ratio) // alignment) * alignment
        self._wrapper = FP8MQATopKWrapper(
            max_context_length=max_context_length,
            max_chunk_size=self.max_chunk_size,
            max_batch_size=1,
            device=device,
            num_heads=n_heads,
            head_dim=_HEAD_DIM,
            top_k=index_topk,
            math_mode="relu_fast",
            compression_ratio=compress_ratio,
            num_keys=self.num_keys,
            qkv_dtype=qkv_dtype,
            logits_dtype=logits_dtype,
            weights_dtype=torch.float32,
            model_config=model_config,
        )
        if self._mxfp4:
            # Packed E2M1 bytes [N, D/2] and four UE8M0 scale bytes per key [N, 4].
            self._k_stage = torch.empty(
                (self.num_keys, _HEAD_DIM // 2), dtype=torch.uint8, device=device
            )
            self._k_scale_stage = torch.empty((self.num_keys, 4), dtype=torch.uint8, device=device)
        else:
            self._k_stage = torch.empty(
                (self.num_keys, _HEAD_DIM), dtype=torch.float8_e4m3fn, device=device
            )
            self._k_scale_stage = torch.empty(self.num_keys, dtype=torch.float32, device=device)
        self._device = device
        self._n_heads = n_heads
        self._index_topk = index_topk
        self._compress_ratio = compress_ratio
        self._pool_prepared = False
        self._failed = False
        # (kv_prefix_len, rows) of the plan the wrapper currently holds, or None.
        self._planned_key: Optional[Tuple[int, int]] = None

    @property
    def compress_ratio(self) -> int:
        """KV compression ratio the kernels were built for."""
        return self._compress_ratio

    @property
    def ready(self) -> bool:
        """True once the kernels are built and their workspace is allocated."""
        return self._pool_prepared and not self._failed

    def can_run(self, num_k_rows: int) -> bool:
        """Returns True when ``num_k_rows`` compressed keys fit the staging buffer."""
        return 0 < num_k_rows <= self.num_keys

    def _plan_tensors(
        self, cache: PlanCache, kv_prefix_len: int, num_q: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns device prefixes ``[0, num_q]`` and ``[0, kv_prefix_len + num_q]``."""
        key = (kv_prefix_len, num_q)
        plan = cache.get(key)
        if plan is None:
            qo_indptr = torch.tensor([0, num_q], dtype=torch.int32)
            kv_indptr = torch.tensor([0, kv_prefix_len + num_q], dtype=torch.int32)
            plan = (
                maybe_pin_memory(qo_indptr).to(self._device, non_blocking=True),
                maybe_pin_memory(kv_indptr).to(self._device, non_blocking=True),
            )
            cache[key] = plan
        return plan

    def run_chunk(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        out: torch.Tensor,
        *,
        kv_prefix_len: int,
        plan_cache: PlanCache,
        q_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        """Selects Top-K indices for consecutive query rows of one request.

        Query row ``i`` attends to keys ``[0, (kv_prefix_len + i + 1) // compress_ratio)``,
        the same causal window the unfused path derives from ``cu_seqlen_ke``.

        Args:
            q: Queries, ``[num_q, n_heads, 128]`` float8_e4m3fn for FP8, or the packed E2M1
                bytes ``[num_q, n_heads, 64]`` (any one-byte dtype) for MXFP4.
            weights: Per-head weights, ``[num_q, n_heads]``, FP32 (BF16 is converted).
            k: Gathered compressed keys of the request, ``[num_k, 128]`` float8_e4m3fn for FP8
                or packed bytes ``[num_k, 64]`` for MXFP4.
            k_scale: FP8: per-key FP32 scales, ``[num_k]`` (any shape with ``num_k`` elements).
                MXFP4: four UE8M0 bytes per key (any dtype/shape with ``4 * num_k`` bytes).
            out: INT32 destination, ``[num_q, index_topk]``; receives request-local indices
                with ``-1`` padding.
            kv_prefix_len: Original tokens of the request preceding ``q[0]``.
            plan_cache: Cache of device prefix tensors shared across layers.
            q_scale: MXFP4 only: four UE8M0 bytes per query row and head (int32 ``[num_q,
                n_heads]``, or any dtype/shape with ``4 * num_q * n_heads`` bytes).

        Returns:
            True if ``out`` was written; False if the kernel workspace could not be allocated, in
            which case this runner is disabled and the caller must use the unfused path.
        """
        if self._failed:
            return False
        num_q = q.shape[0]
        num_k = k.shape[0]
        if weights.dtype != torch.float32:
            weights = weights.float()
        if self._mxfp4:
            if q_scale is None:
                raise ValueError("MXFP4 queries require q_scale")
            q = q.view(torch.uint8)
            q_scale = q_scale.contiguous().view(torch.int32).reshape(num_q, q.shape[1])
            self._k_stage[:num_k].copy_(k.view(torch.uint8))
            self._k_scale_stage[:num_k].copy_(k_scale.contiguous().view(torch.uint8).reshape(-1, 4))
        else:
            if q_scale is not None:
                raise ValueError("q_scale is only used with MXFP4")
            self._k_stage[:num_k].copy_(k)
            self._k_scale_stage[:num_k].copy_(k_scale.reshape(-1))

        for begin in range(0, num_q, self.max_chunk_size):
            end = min(begin + self.max_chunk_size, num_q)
            # The plan depends only on the two prefixes, not on K, Q or the layer, and one plan
            # serves every layer that runs after it. Layers of a step repeat the same plan, so
            # only a change of (prefix, rows) -- another chunk or tile -- needs a new one.
            plan_key = (kv_prefix_len + begin, end - begin)
            if plan_key != self._planned_key:
                qo_indptr, kv_indptr = self._plan_tensors(plan_cache, *plan_key)
                self._wrapper.plan(qo_indptr, kv_indptr, num_query_rows=end - begin)
                self._planned_key = plan_key
            if not self._pool_prepared:
                # The first call builds the kernels and their workspace, after the KV cache has
                # taken the memory it was sized for.
                try:
                    self._wrapper.prepare_pool(self._k_stage, k_scale=self._k_scale_stage)
                except torch.OutOfMemoryError as error:
                    self._disable(error)
                    return False
                self._pool_prepared = True
            self._wrapper.run(
                q[begin:end],
                weights[begin:end],
                self._k_stage,
                k_scale=self._k_scale_stage,
                out=out[begin:end],
                q_scale=None if q_scale is None else q_scale[begin:end],
            )
        return True

    def warmup(self) -> bool:
        """Compiles the kernels and allocates their workspace with one synthetic chunk.

        Otherwise the first chunk that reaches the min-context gate pays for both inside a live
        request. The chunk holds all-zero inputs with more keys than ``index_topk``, so it takes
        the full selection path.

        Returns:
            True if the kernels are built; False if this runner is disabled (e.g. the workspace
            did not fit).
        """
        if self._failed:
            return False
        if self._pool_prepared:
            return True
        num_k = min(self.num_keys, 4 * self._index_topk)
        num_q = min(_WARMUP_QUERY_ROWS, num_k * self._compress_ratio)
        kv_prefix_len = num_k * self._compress_ratio - num_q
        device = self._device
        if self._mxfp4:
            q = torch.zeros(
                (num_q, self._n_heads, _HEAD_DIM // 2), dtype=torch.uint8, device=device
            )
            q_scale = torch.zeros((num_q, self._n_heads), dtype=torch.int32, device=device)
            k = torch.zeros((num_k, _HEAD_DIM // 2), dtype=torch.uint8, device=device)
            k_scale = torch.zeros((num_k, 4), dtype=torch.uint8, device=device)
        else:
            q = torch.zeros(
                (num_q, self._n_heads, _HEAD_DIM), dtype=torch.float8_e4m3fn, device=device
            )
            q_scale = None
            k = torch.zeros((num_k, _HEAD_DIM), dtype=torch.float8_e4m3fn, device=device)
            k_scale = torch.ones(num_k, dtype=torch.float32, device=device)
        weights = torch.zeros((num_q, self._n_heads), dtype=torch.float32, device=device)
        out = torch.empty((num_q, self._index_topk), dtype=torch.int32, device=device)
        ran = self.run_chunk(
            q,
            weights,
            k,
            k_scale,
            out,
            kv_prefix_len=kv_prefix_len,
            plan_cache={},
            q_scale=q_scale,
        )
        # The plan tensors lived in the local cache above; make the next chunk plan again.
        self._planned_key = None
        return ran

    def _disable(self, error: torch.OutOfMemoryError) -> None:
        """Releases this runner's buffers so the unfused path gets the memory back."""
        logger.warning(
            f"The fused throwaway Top-K could not allocate its workspace ({error}); "
            "using the unfused indexer + Top-K path"
        )
        self._failed = True
        self._wrapper = self._k_stage = self._k_scale_stage = None
        torch.cuda.empty_cache()


# One runner per (geometry, max length, device), shared by every layer of a model. The runner owns
# the K staging buffers, the selected-index buffer and the kernel workspace; keeping one per layer
# would multiply that memory by the number of indexer layers. The layers' indexers hold the
# runner, so it is referenced weakly here and released with the model. Configurations that failed
# to build are remembered by key so the fallback warning is logged once.
_RunnerKey = Tuple[int, int, int, int, str, str]
_RUNNERS: "weakref.WeakValueDictionary[_RunnerKey, ThrowawayPrefillTopK]" = (
    weakref.WeakValueDictionary()
)
_FAILED: Set[_RunnerKey] = set()


def get_throwaway_prefill_topk(
    *,
    n_heads: int,
    index_topk: int,
    compress_ratio: int,
    max_context_length: int,
    device: torch.device,
    qkv_dtype: str = "fp8",
) -> Optional[ThrowawayPrefillTopK]:
    """Returns the runner shared by the live layers for this configuration, building it on
    first use.

    Returns None (after one warning) if the kernels do not support the configuration or its
    buffers do not fit in GPU memory.
    """
    key = (n_heads, index_topk, compress_ratio, max_context_length, str(device), qkv_dtype)
    if key in _FAILED:
        return None
    runner = _RUNNERS.get(key)
    if runner is not None:
        return runner
    if (n_heads, index_topk, compress_ratio) not in _VALIDATED_GEOMETRIES:
        logger.warning(
            f"{THROWAWAY_TOPK_ENV}=1 but the fused throwaway Top-K is not validated for "
            f"heads={n_heads}, top_k={index_topk}, compress_ratio={compress_ratio}; "
            "using the unfused indexer + Top-K path"
        )
        _FAILED.add(key)
        return None
    try:
        runner = ThrowawayPrefillTopK(
            n_heads=n_heads,
            index_topk=index_topk,
            compress_ratio=compress_ratio,
            max_context_length=max_context_length,
            device=device,
            qkv_dtype=qkv_dtype,
        )
    except (ValueError, ImportError, torch.OutOfMemoryError) as error:
        logger.warning(
            f"{THROWAWAY_TOPK_ENV}=1 but the fused throwaway Top-K is unavailable for "
            f"heads={n_heads}, top_k={index_topk}, compress_ratio={compress_ratio}, "
            f"qkv_dtype={qkv_dtype}: {error}; "
            "using the unfused indexer + Top-K path"
        )
        _FAILED.add(key)
        out_of_memory = isinstance(error, torch.OutOfMemoryError)
    else:
        _RUNNERS[key] = runner
        return runner
    if out_of_memory:
        # The partial build's tensors live until the exception (whose traceback holds them) is
        # gone, so their cached blocks go back to the unfused path only here.
        torch.cuda.empty_cache()
    return None
