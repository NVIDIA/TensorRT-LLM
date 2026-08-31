# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable index-selection Top-K module for sparse inference paths."""

from __future__ import annotations

import os
from enum import Enum

import torch
import torch.nn as nn

from tensorrt_llm.logger import logger

from ..memory_buffer_utils import get_memory_buffers


class TopKImplementation(str, Enum):
    """Top-K implementations grouped by backend and algorithm."""

    TORCH = "torch"
    CUDA_RADIX = "cuda_radix"
    CUTE_DSL_RADIX = "cute_dsl_radix"
    CUDA_GVR = "cuda_gvr"
    CUTE_DSL_GVR = "cute_dsl_gvr"
    CUTE_DSL_GVR_V2 = "cute_dsl_gvr_v2"


_GVR_IMPLEMENTATIONS = {
    TopKImplementation.CUDA_GVR,
    TopKImplementation.CUTE_DSL_GVR,
    TopKImplementation.CUTE_DSL_GVR_V2,
}
_MAX_RADIX_BLOCKS_PER_ROW = 10


class TopK(nn.Module):
    """Select Top-K indices for sparse prefill and decode paths.

    GVR decode state is owned by the caller so it can be shared with the
    request metadata and retain a stable address across CUDA Graph replays.
    """

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
        # GVR V2 decode is hint-free by default; TRTLLM_GVR_V2_HINTED=1
        # restores prev-step hint consumption.
        self._gvr_v2_hinted = os.environ.get("TRTLLM_GVR_V2_HINTED", "0") == "1"

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
        max_seq_len: int | None = None,
        gvr_ext_kwargs: dict[str, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Write prefill or decode Top-K indices into ``output_indices``.

        Args:
            scores: Top-K input scores with shape ``[num_rows, num_columns]``.
            output_indices: Int32 output with shape ``[num_rows, top_k]``.
            is_prefill: Whether to run the prefill implementation.
            row_starts: Per-row inclusive starts for prefill.
            row_ends: Per-row exclusive ends for prefill.
            sequence_lengths: Per-request logical KV lengths for decode.
            scan_lengths: Per-request score-column lengths for decode.
            next_n: Number of decode rows per request.
            max_seq_len: Maximum decode score width used for GVR kernel tuning.
            gvr_ext_kwargs: GVR-only keyword arguments. ``gvr_prior_indices``
                is the required caller-owned int32 previous selection with
                shape ``[num_requests, top_k]`` on ``scores.device``
                (``CUTE_DSL_GVR_V2`` launches hint-free by default and reads
                it only under ``TRTLLM_GVR_V2_HINTED=1``).
                ``gvr_row_order`` is an optional int32 request ordering with
                shape ``[num_requests]`` on the same device.

        Returns:
            ``output_indices`` after the selected implementation writes it.
        """
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
            max_seq_len,
            gvr_ext_kwargs,
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
        max_seq_len: int | None,
        gvr_ext_kwargs: dict[str, torch.Tensor | None] | None,
    ) -> torch.Tensor:
        if self.decode_implementation == TopKImplementation.TORCH:
            return self._forward_decode_torch(scores, scan_lengths, output_indices, next_n)

        if self.decode_implementation in _GVR_IMPLEMENTATIONS:
            return self._forward_decode_gvr(
                scores,
                sequence_lengths,
                output_indices,
                next_n,
                max_seq_len=max_seq_len,
                **(gvr_ext_kwargs or {}),
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

    def _get_workspace(
        self,
        scores: torch.Tensor,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        buffer_name: str,
    ) -> torch.Tensor:
        device_buffer_name = f"{buffer_name}_{scores.device}"
        if scores.is_cuda:
            with torch.cuda.device(scores.device):
                capture_graph = torch.cuda.is_current_stream_capturing()
                return self._memory_buffers.get_buffer(
                    shape,
                    dtype=dtype,
                    buffer_name=device_buffer_name,
                    reserve_buffer=capture_graph,
                )
        return self._memory_buffers.get_buffer(
            shape,
            dtype=dtype,
            buffer_name=device_buffer_name,
            reserve_buffer=False,
        )

    def _get_radix_workspace(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if scores.dtype != torch.float32:
            # The C++ bf16/fp16 entry has no split-work tier or aux-buffer
            # arguments and rejects widths that would require split work.
            return None, None

        shape = (scores.shape[0], _MAX_RADIX_BLOCKS_PER_ROW, self.top_k)
        radix_indices = self._get_workspace(
            scores,
            shape,
            torch.int32,
            "top_k_radix_indices_workspace",
        )
        radix_values = self._get_workspace(
            scores,
            shape,
            torch.float32,
            "top_k_radix_values_workspace",
        )
        return radix_indices, radix_values

    def _forward_decode_gvr(
        self,
        scores: torch.Tensor,
        sequence_lengths: torch.Tensor,
        output_indices: torch.Tensor,
        next_n: int,
        max_seq_len: int | None,
        gvr_prior_indices: torch.Tensor | None = None,
        gvr_row_order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert gvr_prior_indices is not None
        if self.decode_implementation == TopKImplementation.CUTE_DSL_GVR_V2:
            assert max_seq_len is not None
            if (
                # engine hardware-format gate (falls through otherwise):
                # fp32 row-major scores with a float4-aligned row stride and
                # a 16B-aligned base (the DSL paged-MQA arena view — column-
                # sliced from a 256-aligned buffer — satisfies this; odd
                # max_seq_len DeepGEMM layouts do not). Single-row batches
                # derive their row window from shape[1] (arena last-row
                # safety), so that width must satisfy the same float4 rule —
                # otherwise run_varlen raises instead of falling through.
                scores.dtype == torch.float32
                and scores.stride(1) == 1
                and scores.stride(0) % 4 == 0
                and scores.data_ptr() % 16 == 0
                and (scores.shape[0] > 1 or scores.shape[1] % 4 == 0)
            ):
                # hint-free k derives from the output width; pin it to the module's k
                assert output_indices.shape[1] == self.top_k
                from ..cute_dsl_kernels.blackwell.top_k import selfsampling_topk_run_varlen

                logger.info_once(
                    "self-sampling GVR top-K engaged "
                    f"(K={self.top_k}, cr={self.compress_ratio}, "
                    f"next_n={next_n}, "
                    f"{'hinted' if self._gvr_v2_hinted else 'hint-free'}).",
                    key="selfsampling_topk_engaged",
                )
                # Self-sampling GVR varlen engine (TRTLLM_GVR_SELF_SAMPLING=1):
                # one launch for the batch; per-row n from device kv_lens,
                # capture-stable tuning from the max-seq-len engine constant
                # (no host reads — CUDA-graph safe). Hint-free by default
                # (pre_idx=None: the kernel brackets from the current row);
                # TRTLLM_GVR_V2_HINTED=1 restores raw prev-step hint
                # consumption (offset-free contract). The module receives
                # max_seq_len in COMPRESSED index space; run_varlen's
                # max_seq_len is in kv-token space like sequence_lengths —
                # multiply back.
                selfsampling_topk_run_varlen(
                    scores,
                    gvr_prior_indices if self._gvr_v2_hinted else None,
                    sequence_lengths,
                    output_indices,
                    next_n=next_n,
                    compress_ratio=self.compress_ratio,
                    max_seq_len=max_seq_len * self.compress_ratio,
                )
                return output_indices
            logger.warning_once(
                "TRTLLM_GVR_SELF_SAMPLING=1 but the decode scores do not "
                "satisfy the engine's hardware-format gate "
                f"(dtype={scores.dtype}, strides={tuple(scores.stride())}); "
                "falling through to the CUDA GVR top-K path.",
                key="selfsampling_topk_fallthrough",
            )
        if self.decode_implementation != TopKImplementation.CUTE_DSL_GVR:
            # CUDA_GVR, or the V2 hardware-format fall-through above
            workspace = self._get_workspace(
                scores,
                (scores.shape[0], self.top_k),
                scores.dtype,
                "top_k_cuda_gvr_workspace",
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
            assert max_seq_len is not None
            torch.ops.trtllm.cute_dsl_gvr_topk_decode(
                scores,
                gvr_prior_indices,
                sequence_lengths,
                output_indices,
                self.top_k,
                next_n=next_n,
                compress_ratio=self.compress_ratio,
                max_seq_len=max_seq_len,
                order_row=gvr_row_order,
            )
        return output_indices

    def update_gvr_prior_from_prefill(
        self,
        output_indices: torch.Tensor,
        request_lengths: torch.Tensor,
        gvr_prior_indices: torch.Tensor | None,
        *,
        request_offset: int = 0,
    ) -> None:
        """Update GVR prior indices from each prefill request's last row.

        Args:
            output_indices: Int32 prefill selections with shape
                ``[num_prefill_rows, top_k]``.
            request_lengths: Per-request prefill row counts.
            gvr_prior_indices: Int32 caller-owned state on
                ``output_indices.device`` with shape ``[capacity, top_k]``.
                The slice starting at ``request_offset`` is updated in place.
            request_offset: First request row to update in the prior state.
        """
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
