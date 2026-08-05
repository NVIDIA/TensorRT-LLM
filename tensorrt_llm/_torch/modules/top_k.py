# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable index-selection Top-K module for sparse inference paths."""

from __future__ import annotations

import threading
from enum import Enum

import torch
import torch.nn as nn


class TopKImplementation(str, Enum):
    """Top-K implementations for prefill and decode.

    ``CUTE_DSL_PREFERRED`` uses the CuTe DSL radix implementation when it
    supports the runtime shape and falls back to TRTLLM otherwise.
    """

    TORCH = "torch"
    TRTLLM = "trtllm"
    TRTLLM_HEURISTIC = "trtllm_heuristic"
    CUTE_DSL_PREFERRED = "cute_dsl_preferred"
    CUTE_DSL_GVR = "cute_dsl_gvr"


_PREFILL_IMPLEMENTATIONS = {
    TopKImplementation.TORCH,
    TopKImplementation.TRTLLM,
}


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


def _require_tensor(tensor: torch.Tensor | None, name: str) -> torch.Tensor:
    if tensor is None:
        raise ValueError(f"{name} is required")
    return tensor


def _forward_prefill_torch(
    scores: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    output_indices: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    output_indices.fill_(-1)
    selected_count = min(top_k, scores.shape[1])
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


class TopK(nn.Module):
    """Select Top-K indices for sparse prefill and decode paths."""

    def __init__(
        self,
        top_k: int,
        *,
        prefill_implementation: TopKImplementation | None = None,
        decode_implementation: TopKImplementation | None = None,
        compress_ratio: int = 1,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if compress_ratio <= 0:
            raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
        if prefill_implementation is None and decode_implementation is None:
            raise ValueError("at least one Top-K implementation must be configured")
        self.top_k = top_k
        self.prefill_implementation = (
            TopKImplementation(prefill_implementation)
            if prefill_implementation is not None
            else None
        )
        if (
            self.prefill_implementation is not None
            and self.prefill_implementation not in _PREFILL_IMPLEMENTATIONS
        ):
            raise ValueError(
                f"{self.prefill_implementation.value} is not supported for prefill Top-K"
            )
        self.decode_implementation = (
            TopKImplementation(decode_implementation) if decode_implementation is not None else None
        )
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
        """Warm up decode implementation state for one deployment shape."""
        implementation = self.decode_implementation
        if implementation is None:
            raise ValueError("decode Top-K is not configured")
        if max_num_columns <= 0 or next_n <= 0:
            return
        if implementation in (
            TopKImplementation.TORCH,
            TopKImplementation.TRTLLM,
            TopKImplementation.CUTE_DSL_GVR,
        ):
            return
        if (
            implementation == TopKImplementation.CUTE_DSL_PREFERRED
            and self.compress_ratio > 1
            and next_n > 1
        ):
            return

        device = _cuda_device(device)
        key = (
            implementation,
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
                if implementation == TopKImplementation.TRTLLM_HEURISTIC:
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
        output_indices: torch.Tensor,
        *,
        is_prefill: bool,
        row_starts: torch.Tensor | None = None,
        row_ends: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
        scan_lengths: torch.Tensor | None = None,
        next_n: int = 1,
        prior_indices: torch.Tensor | None = None,
        heuristic_values: torch.Tensor | None = None,
        radix_indices: torch.Tensor | None = None,
        radix_values: torch.Tensor | None = None,
        max_num_columns: int | None = None,
        row_order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write prefill or decode Top-K indices into ``output_indices``.

        Prefill uses ``row_starts`` and ``row_ends``. Decode uses logical
        ``sequence_lengths`` plus ``scan_lengths`` in score-column coordinates.
        """
        _validate_output(scores, output_indices, self.top_k)
        if is_prefill:
            return self._forward_prefill(
                scores,
                _require_tensor(row_starts, "row_starts"),
                _require_tensor(row_ends, "row_ends"),
                output_indices,
            )
        return self._forward_decode(
            scores,
            _require_tensor(sequence_lengths, "sequence_lengths"),
            _require_tensor(scan_lengths, "scan_lengths"),
            output_indices,
            next_n=next_n,
            prior_indices=prior_indices,
            heuristic_values=heuristic_values,
            radix_indices=radix_indices,
            radix_values=radix_values,
            max_num_columns=max_num_columns,
            row_order=row_order,
        )

    def _forward_prefill(
        self,
        scores: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        output_indices: torch.Tensor,
    ) -> torch.Tensor:
        implementation = self.prefill_implementation
        if implementation is None:
            raise ValueError("prefill Top-K is not configured")
        _validate_lengths(scores, row_starts, "row_starts", scores.shape[0])
        _validate_lengths(scores, row_ends, "row_ends", scores.shape[0])
        if implementation == TopKImplementation.TORCH:
            return _forward_prefill_torch(
                scores,
                row_starts,
                row_ends,
                output_indices,
                self.top_k,
            )
        torch.ops.trtllm.indexer_topk_prefill(
            scores,
            row_starts,
            row_ends,
            output_indices,
            self.top_k,
        )
        return output_indices

    def _forward_decode(
        self,
        scores: torch.Tensor,
        sequence_lengths: torch.Tensor,
        scan_lengths: torch.Tensor,
        output_indices: torch.Tensor,
        *,
        next_n: int,
        prior_indices: torch.Tensor | None,
        heuristic_values: torch.Tensor | None,
        radix_indices: torch.Tensor | None,
        radix_values: torch.Tensor | None,
        max_num_columns: int | None,
        row_order: torch.Tensor | None,
    ) -> torch.Tensor:
        implementation = self.decode_implementation
        if implementation is None:
            raise ValueError("decode Top-K is not configured")
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

        if implementation == TopKImplementation.TORCH:
            return self._forward_decode_torch(scores, scan_lengths, output_indices, next_n)

        use_trtllm = implementation in (
            TopKImplementation.TRTLLM,
            TopKImplementation.TRTLLM_HEURISTIC,
        ) or (
            implementation == TopKImplementation.CUTE_DSL_PREFERRED
            and self.compress_ratio > 1
            and next_n > 1
        )
        if use_trtllm:
            if implementation == TopKImplementation.TRTLLM_HEURISTIC:
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

        if implementation == TopKImplementation.CUTE_DSL_PREFERRED:
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

    def _forward_decode_torch(
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
