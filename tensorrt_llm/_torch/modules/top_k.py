# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable index-selection Top-K modules for sparse inference paths."""

from __future__ import annotations

import threading
from enum import Enum

import torch
import torch.nn as nn


class PrefillTopKImplementation(str, Enum):
    """Available segmented prefill Top-K implementations."""

    TORCH = "torch"
    TRTLLM = "trtllm"


class DecodeTopKPolicy(str, Enum):
    """Decode Top-K dispatch policies.

    ``CUTE_DSL_PREFERRED`` uses the CuTe DSL radix implementation when it
    supports the runtime shape and falls back to TRTLLM otherwise.
    """

    TORCH = "torch"
    TRTLLM = "trtllm"
    TRTLLM_HEURISTIC = "trtllm_heuristic"
    CUTE_DSL_PREFERRED = "cute_dsl_preferred"
    CUTE_DSL_GVR = "cute_dsl_gvr"


_PREPARE_LOCK = threading.Lock()
_PREPARED_DECODE_TOP_K: set[tuple[object, ...]] = set()
_HEURISTIC_WARMUP_COLUMNS = 4096
_RADIX_MAX_BLOCKS_PER_ROW = 10


def _cuda_device(device: torch.device) -> torch.device:
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"Top-K preparation requires a CUDA device, got {device}")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _validate_output(
    scores: torch.Tensor,
    output_indices: torch.Tensor,
    top_k: int,
) -> None:
    if scores.ndim != 2:
        raise ValueError(f"scores must be rank 2, got shape {tuple(scores.shape)}")
    if output_indices.ndim != 2:
        raise ValueError(f"output_indices must be rank 2, got shape {tuple(output_indices.shape)}")
    expected_shape = (scores.shape[0], top_k)
    if output_indices.shape != expected_shape:
        raise ValueError(
            f"output_indices must have shape {expected_shape}, got {tuple(output_indices.shape)}"
        )
    if output_indices.dtype != torch.int32:
        raise TypeError(f"output_indices must have dtype torch.int32, got {output_indices.dtype}")
    if output_indices.device != scores.device:
        raise ValueError(
            "scores and output_indices must be on the same device, got "
            f"{scores.device} and {output_indices.device}"
        )


def _validate_lengths(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    name: str,
    expected_size: int,
) -> None:
    if lengths.ndim != 1 or lengths.shape[0] != expected_size:
        raise ValueError(f"{name} must have shape ({expected_size},), got {tuple(lengths.shape)}")
    if lengths.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {lengths.dtype}")
    if lengths.device != scores.device:
        raise ValueError(
            f"scores and {name} must be on the same device, got {scores.device} and {lengths.device}"
        )


class PrefillTopK(nn.Module):
    """Select row-local Top-K indices from segmented prefill scores."""

    def __init__(
        self,
        top_k: int,
        implementation: PrefillTopKImplementation,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        self.top_k = top_k
        self.implementation = PrefillTopKImplementation(implementation)

    def prepare(self) -> None:
        """Prepare the implementation for execution.

        Segmented prefill implementations currently need no explicit warmup.
        """

    def forward(
        self,
        scores: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        output_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Write row-local Top-K indices into ``output_indices``.

        Args:
            scores: Score matrix with shape ``[num_rows, num_columns]``.
            row_starts: Inclusive valid-column starts with shape ``[num_rows]``.
            row_ends: Exclusive valid-column ends with shape ``[num_rows]``.
            output_indices: Caller-owned int32 output with shape
                ``[num_rows, top_k]``.
        """
        _validate_output(scores, output_indices, self.top_k)
        _validate_lengths(scores, row_starts, "row_starts", scores.shape[0])
        _validate_lengths(scores, row_ends, "row_ends", scores.shape[0])

        if self.implementation == PrefillTopKImplementation.TRTLLM:
            torch.ops.trtllm.indexer_topk_prefill(
                scores,
                row_starts,
                row_ends,
                output_indices,
                self.top_k,
            )
            return output_indices

        output_indices.fill_(-1)
        selected_count = min(self.top_k, scores.shape[1])
        if selected_count == 0:
            return output_indices
        columns = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0)
        valid = (columns >= row_starts.unsqueeze(1)) & (columns < row_ends.unsqueeze(1))
        selected = scores.masked_fill(~valid, float("-inf")).topk(selected_count, dim=-1).indices
        selected_valid = torch.gather(valid, 1, selected)
        selected = selected - row_starts.unsqueeze(1)
        selected = selected.masked_fill(~selected_valid, -1)
        output_indices[:, :selected_count].copy_(selected.to(torch.int32))
        return output_indices


class DecodeTopK(nn.Module):
    """Select Top-K indices from decode scores using a fixed dispatch policy."""

    def __init__(
        self,
        top_k: int,
        policy: DecodeTopKPolicy,
        *,
        compress_ratio: int = 1,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if compress_ratio <= 0:
            raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
        self.top_k = top_k
        self.policy = DecodeTopKPolicy(policy)
        self.compress_ratio = compress_ratio

    def prepare(
        self,
        *,
        device: torch.device,
        max_num_columns: int,
        next_n: int,
        input_dtype: torch.dtype,
        num_sms: int | None = None,
    ) -> None:
        """Warm up static implementation state for one deployment shape."""
        if max_num_columns <= 0 or next_n <= 0:
            return
        if self.policy in (
            DecodeTopKPolicy.TORCH,
            DecodeTopKPolicy.TRTLLM,
            DecodeTopKPolicy.CUTE_DSL_GVR,
        ):
            return
        if (
            self.policy == DecodeTopKPolicy.CUTE_DSL_PREFERRED
            and self.compress_ratio > 1
            and next_n > 1
        ):
            return

        device = _cuda_device(device)
        key = (
            self.policy,
            device.index,
            input_dtype,
            self.top_k,
            max_num_columns,
            next_n,
            num_sms,
            self.compress_ratio,
        )
        with _PREPARE_LOCK:
            if key in _PREPARED_DECODE_TOP_K:
                return
            with torch.cuda.device(device):
                if self.policy == DecodeTopKPolicy.TRTLLM_HEURISTIC:
                    self._warmup_trtllm_heuristic(input_dtype)
                else:
                    try:
                        from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
                            warmup_cute_dsl_radix_topk_decode,
                        )
                    except ImportError:
                        return
                    warmup_cute_dsl_radix_topk_decode(
                        top_k=self.top_k,
                        num_cols=max_num_columns,
                        next_n=next_n,
                        dtype=input_dtype,
                        num_sms=num_sms,
                    )
            _PREPARED_DECODE_TOP_K.add(key)

    def _warmup_trtllm_heuristic(self, input_dtype: torch.dtype) -> None:
        num_columns = max(_HEURISTIC_WARMUP_COLUMNS, self.top_k)
        device = torch.device("cuda")
        scores = torch.zeros((1, num_columns), dtype=input_dtype, device=device)
        sequence_lengths = torch.tensor([num_columns], dtype=torch.int32, device=device)
        output_indices = torch.empty((1, self.top_k), dtype=torch.int32, device=device)
        prior_indices = torch.zeros((1, self.top_k), dtype=torch.int32, device=device)
        heuristic_values = torch.empty((1, self.top_k), dtype=input_dtype, device=device)
        radix_indices = torch.empty(
            (1, _RADIX_MAX_BLOCKS_PER_ROW, self.top_k),
            dtype=torch.int32,
            device=device,
        )
        radix_values = torch.empty(
            (1, _RADIX_MAX_BLOCKS_PER_ROW, self.top_k),
            dtype=torch.float32,
            device=device,
        )
        torch.ops.trtllm.indexer_topk_decode(
            scores,
            sequence_lengths,
            output_indices,
            1,
            self.top_k,
            pre_idx=prior_indices,
            heuristic_scratch=heuristic_values,
            radix_aux_indices=radix_indices,
            radix_aux_logits=radix_values,
        )
        torch.cuda.synchronize()

    def forward(
        self,
        scores: torch.Tensor,
        sequence_lengths: torch.Tensor,
        scan_lengths: torch.Tensor,
        output_indices: torch.Tensor,
        *,
        next_n: int,
        prior_indices: torch.Tensor | None = None,
        heuristic_values: torch.Tensor | None = None,
        radix_indices: torch.Tensor | None = None,
        radix_values: torch.Tensor | None = None,
        max_num_columns: int | None = None,
        row_order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write decode Top-K indices into ``output_indices``.

        ``sequence_lengths`` are logical request KV lengths. ``scan_lengths``
        are request lengths in the score-column coordinate system.
        """
        _validate_output(scores, output_indices, self.top_k)
        if next_n <= 0:
            raise ValueError(f"next_n must be positive, got {next_n}")
        if scores.shape[0] % next_n != 0:
            raise ValueError(
                f"score rows ({scores.shape[0]}) must be divisible by next_n ({next_n})"
            )
        num_requests = scores.shape[0] // next_n
        for name, lengths in (
            ("sequence_lengths", sequence_lengths),
            ("scan_lengths", scan_lengths),
        ):
            _validate_lengths(scores, lengths, name, num_requests)

        if (radix_indices is None) != (radix_values is None):
            raise ValueError("radix_indices and radix_values must be provided together")

        if self.policy == DecodeTopKPolicy.TORCH:
            return self._forward_torch(scores, scan_lengths, output_indices, next_n)

        use_trtllm = self.policy in (
            DecodeTopKPolicy.TRTLLM,
            DecodeTopKPolicy.TRTLLM_HEURISTIC,
        ) or (
            self.policy == DecodeTopKPolicy.CUTE_DSL_PREFERRED
            and self.compress_ratio > 1
            and next_n > 1
        )
        if use_trtllm:
            if self.policy == DecodeTopKPolicy.TRTLLM_HEURISTIC:
                if prior_indices is None or heuristic_values is None:
                    raise ValueError("TRTLLM_HEURISTIC requires prior_indices and heuristic_values")
            torch.ops.trtllm.indexer_topk_decode(
                scores,
                sequence_lengths,
                output_indices,
                next_n,
                self.top_k,
                pre_idx=prior_indices,
                heuristic_scratch=heuristic_values,
                compress_ratio=self.compress_ratio,
                radix_aux_indices=radix_indices,
                radix_aux_logits=radix_values,
            )
            return output_indices

        if self.policy == DecodeTopKPolicy.CUTE_DSL_PREFERRED:
            torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                scores,
                scan_lengths,
                output_indices,
                self.top_k,
                next_n,
            )
            return output_indices

        if prior_indices is None:
            raise ValueError("CUTE_DSL_GVR requires prior_indices")
        if max_num_columns is None:
            raise ValueError("CUTE_DSL_GVR requires max_num_columns")
        torch.ops.trtllm.cute_dsl_gvr_topk_decode(
            scores,
            prior_indices,
            sequence_lengths,
            output_indices,
            self.top_k,
            next_n=next_n,
            compress_ratio=self.compress_ratio,
            max_seq_len=max_num_columns,
            order_row=row_order,
        )
        return output_indices

    def _forward_torch(
        self,
        scores: torch.Tensor,
        scan_lengths: torch.Tensor,
        output_indices: torch.Tensor,
        next_n: int,
    ) -> torch.Tensor:
        output_indices.fill_(-1)
        selected_count = min(self.top_k, scores.shape[1])
        if selected_count == 0:
            return output_indices
        positions = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0)
        row_indices = torch.arange(scores.shape[0], device=scores.device) // next_n
        next_n_offsets = torch.arange(scores.shape[0], device=scores.device) % next_n
        row_ends = scan_lengths[row_indices] - next_n + next_n_offsets + 1
        valid = positions < row_ends.unsqueeze(1)
        selected = scores.masked_fill(~valid, float("-inf")).topk(selected_count, dim=-1).indices
        selected_valid = torch.gather(valid, 1, selected)
        selected = selected.masked_fill(~selected_valid, -1)
        output_indices[:, :selected_count].copy_(selected.to(torch.int32))
        return output_indices
