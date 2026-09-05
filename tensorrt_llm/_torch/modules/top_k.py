# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable index-selection Top-K module for sparse inference paths."""

from __future__ import annotations

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
_TEMPORAL_GVR_IMPLEMENTATIONS = {
    TopKImplementation.CUDA_GVR,
    TopKImplementation.CUTE_DSL_GVR,
}
_MAX_RADIX_BLOCKS_PER_ROW = 10
# One 16-byte vector per copy, matching the Blackwell prefill op's contract.
_CUTE_DSL_PREFILL_COPY_BITS = 128


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
        # emission-assisted GVR (opt-in via prepare_gvr_emission): the
        # module owns the closed-loop emission state; the caller passes
        # the returned kwargs to the scoring op, and the consume side is
        # injected into the GVR Top-K call while the step stays armed
        self._gvr_emission_state = None
        self._gvr_emission_route = None
        self._gvr_emission_armed = False

    @property
    def needs_gvr_prior(self) -> bool:
        """Return whether decode consumes previous-step Top-K indices."""
        return self.decode_implementation in _TEMPORAL_GVR_IMPLEMENTATIONS

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
                is required by the temporal CUDA and CuTe DSL GVR paths. It is
                caller-owned int32 previous selection with shape
                ``[num_requests, top_k]`` on ``scores.device``. GVR V2 does
                not consume this state.
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
        if self.prefill_implementation == TopKImplementation.CUTE_DSL_RADIX:
            # Keep the op's reread policy default; only its copy width is tuned.
            torch.ops.trtllm.cute_dsl_indexer_topk_prefill_blackwell(
                scores,
                row_starts,
                row_ends,
                output_indices,
                self.top_k,
                _CUTE_DSL_PREFILL_COPY_BITS,
            )
            return output_indices
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
                    f"next_n={next_n}, hint-free).",
                    key="selfsampling_topk_engaged",
                )
                # Self-sampling GVR varlen engine (TRTLLM_GVR_SELF_SAMPLING=1):
                # one launch for the batch; per-row n from device kv_lens,
                # capture-stable tuning from the max-seq-len engine constant
                # (no host reads — CUDA-graph safe). The module receives
                # max_seq_len in compressed index space; run_varlen's value
                # is in KV-token space like sequence_lengths, so multiply it
                # back by the compression ratio.
                selfsampling_topk_run_varlen(
                    scores,
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
                "falling back to the CUDA insertion/radix Top-K path.",
                key="selfsampling_topk_fallthrough",
            )
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

        assert gvr_prior_indices is not None
        if self.decode_implementation == TopKImplementation.CUDA_GVR:
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
        elif self.decode_implementation == TopKImplementation.CUTE_DSL_GVR:
            assert max_seq_len is not None
            emission_kwargs: dict = {}
            if self._gvr_emission_armed:
                state = self._gvr_emission_state
                num_rows = scores.shape[0]
                emission_kwargs = state.topk_ext_kwargs(
                    self._gvr_emission_route,
                    num_rows,
                    state.block_max[:num_rows] if state.block_max is not None else None,
                )
                self._gvr_emission_armed = False
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
                **emission_kwargs,
            )
        else:
            raise AssertionError(f"Unexpected GVR implementation: {self.decode_implementation}")
        return output_indices

    def prepare_gvr_emission(
        self,
        batch: int,
        n_comp: int,
        num_sms: int,
        gvr_prior_indices: torch.Tensor,
    ) -> dict:
        """Plan the emission-assisted GVR tier for this decode step.

        Returns the emission kwargs for the paged-MQA scoring op (empty
        when the planner declines this step); the matching consume-side
        kwargs are injected into the next GVR Top-K call automatically.
        Host arithmetic on engine-static shapes plus capturable device
        ops, so a captured graph bakes the tier and replays refresh the
        state buffers in place.

        Args:
            batch: Number of decode requests this step.
            n_comp: Engine-static compressed maximum sequence length.
            num_sms: Device SM count.
            gvr_prior_indices: Caller-owned previous-selection state;
                defines the emission state's row capacity and device.
        """
        # the emission/xstate writes are undeclared mutations (see the op's
        # schema note), so the tier is eager / CUDA-graph only
        if torch.compiler.is_dynamo_compiling():
            return {}
        from ..cute_dsl_kernels.blackwell.top_k.gvr_emission import (
            LIST_EMIT_MIN_N,
            GvrEmissionState,
        )

        if self._gvr_emission_state is None:
            self._gvr_emission_state = GvrEmissionState(
                max_rows=gvr_prior_indices.shape[0],
                top_k=self.top_k,
                device=gvr_prior_indices.device,
                enable_list_tier=n_comp >= LIST_EMIT_MIN_N,
                own_prior=False,
            )
        state = self._gvr_emission_state
        emit_tier, self._gvr_emission_route = state.plan(
            batch, n_comp, num_sms, compress_ratio=max(self.compress_ratio, 1)
        )
        self._gvr_emission_armed = self._gvr_emission_route.tier != "none"
        if emit_tier in ("counts", "list", "rungs"):
            state.update_seed_rows(batch, emit_tier)
        kwargs: dict = {}
        if emit_tier in ("counts", "list"):
            kwargs = state.indexer_emit_kwargs(emit_tier, batch)
        if self._gvr_emission_route.attach_block_max or emit_tier in ("counts", "list"):
            kwargs["block_max_out"] = state.ensure_block_max(n_comp)[:batch]
        return kwargs

    def reset_gvr_emission_rows(self, rows: slice) -> None:
        """Cold-start the emission closed-loop state for reused request
        slots (prefill-to-decode handoff): a zeroed xstate reads as
        invalid and routes those rows to the stock path in-kernel."""
        if self._gvr_emission_state is not None:
            self._gvr_emission_state.xstate[rows].zero_()

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
            request_lengths: Per-request prefill row counts on
                ``output_indices.device``; a host tensor here makes the row
                gather a synchronous host-to-device copy.
            gvr_prior_indices: Int32 caller-owned state on
                ``output_indices.device`` with shape ``[capacity, top_k]``.
                The slice starting at ``request_offset`` is updated in place.
            request_offset: First request row to update in the prior state.
        """
        if not self.needs_gvr_prior:
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
