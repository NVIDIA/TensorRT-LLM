# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable index-selection Top-K module for sparse inference paths."""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn

from ..memory_buffer_utils import get_memory_buffers


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
_MAX_RADIX_BLOCKS_PER_ROW = 10


class TopK(nn.Module):
    """Select Top-K indices for sparse prefill and decode paths."""

    _memory_buffers = get_memory_buffers()

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
        gvr_prior_indices: torch.Tensor | None = None,
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
            gvr_prior_indices,
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
        gvr_prior_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.decode_implementation == TopKImplementation.TORCH:
            return self._forward_decode_torch(scores, scan_lengths, output_indices, next_n)

        if self.decode_implementation in _GVR_IMPLEMENTATIONS:
            return self._forward_decode_gvr(
                scores,
                sequence_lengths,
                output_indices,
                next_n,
                gvr_prior_indices,
            )

        return self._forward_decode_radix(
            scores,
            sequence_lengths,
            scan_lengths,
            output_indices,
            next_n,
        )

    def _forward_decode_radix(
        self,
        scores: torch.Tensor,
        sequence_lengths: torch.Tensor,
        scan_lengths: torch.Tensor,
        output_indices: torch.Tensor,
        next_n: int,
    ) -> torch.Tensor:
        use_cute_dsl = self.decode_implementation == TopKImplementation.CUTE_DSL_RADIX and not (
            self.compress_ratio > 1 and next_n > 1
        )
        if use_cute_dsl:
            torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                scores,
                scan_lengths,
                output_indices,
                self.top_k,
                next_n,
            )
            return output_indices

        radix_indices, radix_values = self._get_radix_workspace(scores)
        torch.ops.trtllm.indexer_topk_decode(
            scores,
            sequence_lengths,
            output_indices,
            next_n,
            self.top_k,
            pre_idx=None,
            heuristic_scratch=None,
            compress_ratio=self.compress_ratio,
            radix_aux_indices=radix_indices,
            radix_aux_logits=radix_values,
        )
        return output_indices

    def _get_radix_workspace(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if scores.dtype != torch.float32:
            return None, None

        shape = (scores.shape[0], _MAX_RADIX_BLOCKS_PER_ROW, self.top_k)
        capture_graph = scores.is_cuda and torch.cuda.is_current_stream_capturing()
        radix_indices = self._memory_buffers.get_buffer(
            shape,
            dtype=torch.int32,
            buffer_name="top_k_radix_indices_workspace",
            reserve_buffer=capture_graph,
        )
        radix_values = self._memory_buffers.get_buffer(
            shape,
            dtype=torch.float32,
            buffer_name="top_k_radix_values_workspace",
            reserve_buffer=capture_graph,
        )
        return radix_indices, radix_values

    def _forward_decode_gvr(
        self,
        scores: torch.Tensor,
        sequence_lengths: torch.Tensor,
        output_indices: torch.Tensor,
        next_n: int,
        gvr_prior_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        assert gvr_prior_indices is not None
        capture_graph = scores.is_cuda and torch.cuda.is_current_stream_capturing()
        num_requests = sequence_lengths.shape[0]
        if self.decode_implementation == TopKImplementation.CUDA_GVR:
            workspace = self._memory_buffers.get_buffer(
                (scores.shape[0], self.top_k),
                dtype=scores.dtype,
                buffer_name="top_k_cuda_gvr_workspace",
                reserve_buffer=capture_graph,
            )
            radix_indices, radix_values = self._get_radix_workspace(scores)
            torch.ops.trtllm.indexer_topk_decode(
                scores,
                sequence_lengths,
                output_indices,
                next_n,
                self.top_k,
                pre_idx=gvr_prior_indices,
                heuristic_scratch=workspace,
                compress_ratio=self.compress_ratio,
                radix_aux_indices=radix_indices,
                radix_aux_logits=radix_values,
            )
        else:
            if self._num_sms == 0:
                self._num_sms = (
                    torch.cuda.get_device_properties(scores.device).multi_processor_count
                    if scores.is_cuda
                    else 1
                )
            row_order = None
            if num_requests * next_n >= 2 * self._num_sms:
                row_order = self._memory_buffers.get_buffer(
                    (num_requests,),
                    dtype=torch.int32,
                    buffer_name="top_k_cute_dsl_gvr_row_order",
                    reserve_buffer=capture_graph,
                )
                row_order.copy_(torch.argsort(sequence_lengths, descending=True).to(torch.int32))
            torch.ops.trtllm.cute_dsl_gvr_topk_decode(
                scores,
                gvr_prior_indices,
                sequence_lengths,
                output_indices,
                self.top_k,
                next_n=next_n,
                compress_ratio=self.compress_ratio,
                max_seq_len=scores.shape[1],
                order_row=row_order,
            )
        gvr_prior_indices.copy_(output_indices[next_n - 1 :: next_n])
        return output_indices

    def update_gvr_prior_from_prefill(
        self,
        output_indices: torch.Tensor,
        request_lengths: torch.Tensor,
        gvr_prior_indices: torch.Tensor | None,
        *,
        request_offset: int = 0,
    ) -> None:
        """Update GVR prior indices from each prefill request's last selected row."""
        if self.decode_implementation not in _GVR_IMPLEMENTATIONS:
            return
        assert gvr_prior_indices is not None
        last_rows = (torch.cumsum(request_lengths, dim=0) - 1).to(dtype=torch.long)
        num_requests = request_lengths.shape[0]
        gvr_prior_indices[request_offset : request_offset + num_requests].copy_(
            output_indices[last_rows]
        )

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
