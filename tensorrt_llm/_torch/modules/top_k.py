# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable index-selection Top-K module for sparse inference paths."""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn


class TopKImplementation(str, Enum):
    """Top-K implementations grouped by backend and algorithm."""

    TORCH = "torch"
    CUDA_RADIX = "cuda_radix"
    CUTE_DSL_RADIX = "cute_dsl_radix"
    CUDA_GVR = "cuda_gvr"
    CUTE_DSL_GVR = "cute_dsl_gvr"


_GVR_IMPLEMENTATIONS = {
    TopKImplementation.CUDA_GVR,
    TopKImplementation.CUTE_DSL_GVR,
}


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
        self.top_k = top_k
        self.prefill_implementation = TopKImplementation(
            prefill_implementation or TopKImplementation.CUDA_RADIX
        )
        self.decode_implementation = TopKImplementation(
            decode_implementation or TopKImplementation.CUDA_RADIX
        )
        self.compress_ratio = compress_ratio
        self.register_buffer(
            "_gvr_prior_indices",
            torch.empty((0, top_k), dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "_cuda_gvr_scratch",
            torch.empty((0, top_k)),
            persistent=False,
        )
        self.register_buffer(
            "_gvr_row_order",
            torch.empty((0,), dtype=torch.int32),
            persistent=False,
        )
        self._num_sms = 0

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
        radix_indices: torch.Tensor | None = None,
        radix_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write prefill or decode Top-K indices into ``output_indices``."""
        if is_prefill:
            assert row_starts is not None and row_ends is not None
            return self._forward_prefill(scores, row_starts, row_ends, output_indices)

        assert sequence_lengths is not None and scan_lengths is not None
        return self._forward_decode(
            scores,
            sequence_lengths,
            scan_lengths,
            output_indices,
            next_n,
            radix_indices,
            radix_values,
        )

    def seed_from_prefill(
        self,
        output_indices: torch.Tensor,
        request_lengths: torch.Tensor,
        *,
        request_offset: int = 0,
    ) -> None:
        """Seed GVR from the last selected row of each prefill request."""
        if self.decode_implementation not in _GVR_IMPLEMENTATIONS:
            return
        last_rows = (torch.cumsum(request_lengths, dim=0) - 1).to(dtype=torch.long)
        num_requests = request_lengths.shape[0]
        required_rows = request_offset + num_requests
        if (
            self._gvr_prior_indices.device != output_indices.device
            or self._gvr_prior_indices.shape[0] < required_rows
        ):
            prior_indices = output_indices.new_zeros((required_rows, self.top_k), dtype=torch.int32)
            if self._gvr_prior_indices.device == output_indices.device:
                prior_indices[: self._gvr_prior_indices.shape[0]].copy_(self._gvr_prior_indices)
            self._gvr_prior_indices = prior_indices
        self._gvr_prior_indices[request_offset : request_offset + num_requests].copy_(
            output_indices[last_rows]
        )

    def _forward_prefill(
        self,
        scores: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        output_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.prefill_implementation == TopKImplementation.TORCH:
            return self._forward_prefill_torch(
                scores,
                row_starts,
                row_ends,
                output_indices,
            )
        if self.prefill_implementation != TopKImplementation.CUDA_RADIX:
            raise NotImplementedError(
                f"{self.prefill_implementation.value} does not support prefill Top-K"
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
        next_n: int,
        radix_indices: torch.Tensor | None,
        radix_values: torch.Tensor | None,
    ) -> torch.Tensor:
        implementation = self.decode_implementation
        if implementation == TopKImplementation.TORCH:
            return self._forward_decode_torch(scores, scan_lengths, output_indices, next_n)

        if implementation in _GVR_IMPLEMENTATIONS:
            self._ensure_gvr_state(scores, sequence_lengths, next_n, radix_indices)

        use_cuda = implementation in (
            TopKImplementation.CUDA_RADIX,
            TopKImplementation.CUDA_GVR,
        ) or (
            implementation == TopKImplementation.CUTE_DSL_RADIX
            and self.compress_ratio > 1
            and next_n > 1
        )
        if use_cuda:
            prior_indices = None
            heuristic_values = None
            if implementation == TopKImplementation.CUDA_GVR:
                prior_indices = self._gvr_prior_indices[: sequence_lengths.shape[0]]
                heuristic_values = self._cuda_gvr_scratch[: scores.shape[0]]
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
            if prior_indices is not None:
                prior_indices.copy_(output_indices[next_n - 1 :: next_n])
            return output_indices

        if implementation == TopKImplementation.CUTE_DSL_RADIX:
            torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                scores,
                scan_lengths,
                output_indices,
                self.top_k,
                next_n,
            )
            return output_indices

        prior_indices = self._gvr_prior_indices[: sequence_lengths.shape[0]]
        torch.ops.trtllm.cute_dsl_gvr_topk_decode(
            scores,
            prior_indices,
            sequence_lengths,
            output_indices,
            self.top_k,
            next_n=next_n,
            compress_ratio=self.compress_ratio,
            max_seq_len=scores.shape[1],
            order_row=self._prepare_row_order(sequence_lengths, next_n),
        )
        prior_indices.copy_(output_indices[next_n - 1 :: next_n])
        return output_indices

    def _ensure_gvr_state(
        self,
        scores: torch.Tensor,
        sequence_lengths: torch.Tensor,
        next_n: int,
        radix_indices: torch.Tensor | None,
    ) -> None:
        row_capacity = radix_indices.shape[0] if radix_indices is not None else scores.shape[0]
        request_capacity = max(sequence_lengths.shape[0], row_capacity // next_n)

        if (
            self._gvr_prior_indices.device != scores.device
            or self._gvr_prior_indices.shape[0] < request_capacity
        ):
            prior_indices = scores.new_zeros((request_capacity, self.top_k), dtype=torch.int32)
            if self._gvr_prior_indices.device == scores.device:
                prior_indices[: self._gvr_prior_indices.shape[0]].copy_(self._gvr_prior_indices)
            self._gvr_prior_indices = prior_indices

        if self.decode_implementation == TopKImplementation.CUDA_GVR:
            if (
                self._cuda_gvr_scratch.device != scores.device
                or self._cuda_gvr_scratch.dtype != scores.dtype
                or self._cuda_gvr_scratch.shape[0] < row_capacity
            ):
                self._cuda_gvr_scratch = scores.new_empty((row_capacity, self.top_k))
            return

        if (
            self._gvr_row_order.device != scores.device
            or self._gvr_row_order.shape[0] < request_capacity
        ):
            self._gvr_row_order = scores.new_empty((request_capacity,), dtype=torch.int32)
        if self._num_sms == 0:
            self._num_sms = (
                torch.cuda.get_device_properties(scores.device).multi_processor_count
                if scores.is_cuda
                else 1
            )

    def _prepare_row_order(
        self,
        sequence_lengths: torch.Tensor,
        next_n: int,
    ) -> torch.Tensor | None:
        num_requests = sequence_lengths.shape[0]
        if num_requests * next_n < 2 * self._num_sms:
            return None
        row_order = self._gvr_row_order[:num_requests]
        row_order.copy_(torch.argsort(sequence_lengths, descending=True).to(torch.int32))
        return row_order

    def _forward_prefill_torch(
        self,
        scores: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        output_indices: torch.Tensor,
    ) -> torch.Tensor:
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
