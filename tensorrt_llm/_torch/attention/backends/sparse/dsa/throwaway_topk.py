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
from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.logger import logger

THROWAWAY_TOPK_ENV = "TRTLLM_THROWAWAY_TOPK"
THROWAWAY_MAX_CHUNK_ENV = "TRTLLM_THROWAWAY_TOPK_MAX_CHUNK"
THROWAWAY_LOGITS_DTYPE_ENV = "TRTLLM_THROWAWAY_TOPK_LOGITS_DTYPE"

_HEAD_DIM = 128
_SUPPORTED_NUM_HEADS = (4, 32, 64)
_DEFAULT_MAX_CHUNK = 8192
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


def throwaway_topk_supported(n_heads: int, head_dim: int) -> bool:
    """Returns True when this indexer geometry can use the fused kernels (FP8 or MXFP4 K)."""
    return IS_CUTLASS_DSL_AVAILABLE and head_dim == _HEAD_DIM and n_heads in _SUPPORTED_NUM_HEADS


class ThrowawayPrefillTopK:
    """Runs the fused indexer + Top-K for single-request prefill chunks.

    One instance is shared by all layers of a process. The kernel binds K by device
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

        Raises:
            ValueError: If the kernels do not support this configuration.
        """
        if qkv_dtype not in ("fp8", "mxfp4"):
            raise ValueError(f"qkv_dtype must be 'fp8' or 'mxfp4', got {qkv_dtype!r}")
        self._mxfp4 = qkv_dtype == "mxfp4"
        from tensorrt_llm._torch.attention.kernels.blackwell.throwaway_topk.interface import (
            FP8MQATopKWrapper,
        )

        self.max_chunk_size = int(os.environ.get(THROWAWAY_MAX_CHUNK_ENV, _DEFAULT_MAX_CHUNK))
        if logits_dtype is None:
            logits_name = os.environ.get(THROWAWAY_LOGITS_DTYPE_ENV, "fp32")
            if logits_name not in _LOGITS_DTYPES:
                raise ValueError(
                    f"{THROWAWAY_LOGITS_DTYPE_ENV} must be one of {sorted(_LOGITS_DTYPES)}, "
                    f"got {logits_name!r}"
                )
            logits_dtype = _LOGITS_DTYPES[logits_name]
        elif logits_dtype not in _LOGITS_DTYPES.values():
            raise ValueError(
                f"logits_dtype must be float32, bfloat16 or float16, got {logits_dtype}"
            )
        self.logits_dtype = logits_dtype
        self.num_keys = max_context_length // compress_ratio
        if self._mxfp4:
            # Block-scaled K storage must hold a multiple of 256 rows; plan() bounds the live part.
            self.num_keys = -(-self.num_keys // _MXFP4_KEY_ALIGNMENT) * _MXFP4_KEY_ALIGNMENT
        self._wrapper = FP8MQATopKWrapper(
            max_context_length=max_context_length,
            max_chunk_size=self.max_chunk_size,
            max_batch_size=1,
            device=device,
            num_heads=n_heads,
            head_dim=_HEAD_DIM,
            top_k=index_topk,
            compression_ratio=compress_ratio,
            num_keys=self.num_keys,
            qkv_dtype=qkv_dtype,
            logits_dtype=logits_dtype,
            weights_dtype=torch.float32,
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
        self._pool_prepared = False

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
    ) -> None:
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
        """
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
            qo_indptr, kv_indptr = self._plan_tensors(
                plan_cache, kv_prefix_len + begin, end - begin
            )
            self._wrapper.plan(qo_indptr, kv_indptr, num_query_rows=end - begin)
            if not self._pool_prepared:
                self._wrapper.prepare_pool(self._k_stage, k_scale=self._k_scale_stage)
                self._pool_prepared = True
            self._wrapper.run(
                q[begin:end],
                weights[begin:end],
                self._k_stage,
                k_scale=self._k_scale_stage,
                out=out[begin:end],
                q_scale=None if q_scale is None else q_scale[begin:end],
            )


def create_throwaway_prefill_topk(
    *,
    n_heads: int,
    index_topk: int,
    compress_ratio: int,
    max_context_length: int,
    device: torch.device,
    qkv_dtype: str = "fp8",
) -> Optional[ThrowawayPrefillTopK]:
    """Builds the shared runner, or returns None (with a warning) if it is unsupported."""
    try:
        return ThrowawayPrefillTopK(
            n_heads=n_heads,
            index_topk=index_topk,
            compress_ratio=compress_ratio,
            max_context_length=max_context_length,
            device=device,
            qkv_dtype=qkv_dtype,
        )
    except (ValueError, ImportError) as error:
        logger.warning(
            f"{THROWAWAY_TOPK_ENV}=1 but the fused throwaway Top-K is unavailable for "
            f"heads={n_heads}, top_k={index_topk}, compress_ratio={compress_ratio}, "
            f"qkv_dtype={qkv_dtype}: {error}; "
            "using the unfused indexer + Top-K path"
        )
        return None
