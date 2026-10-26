# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Breakable CUDA graphs for the bounded-replay decoder pass of attention-DP prefill.

The model builds them only with breakable prefill CUDA graphs
(``prefill_cuda_graph_backend=breakable``). A prefill forward in a captured token bucket
replays the whole model body as one breakable graph instead; other prefill forwards run
eagerly with the decoder pass from these graphs.

With bounded replay the decoder layers of a prefill forward run only on the final-window
rows of each prompt, so their cost is host launch time. Each row bucket captures those
layers once: mHC boundaries, MoE (router, EP all-to-all, experts, shared expert) and the
row-wise attention projections are graph segments, and only the metadata-driven sparse
attention core runs eagerly on the live rows. Every attention-DP rank replays the bucket of
the largest rank's row count, so the captured collectives match; padding rows are zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from tensorrt_llm.logger import logger

from ..autotuner import AutoTuner
from ..modules.linear import _FLASHINFER_MXFP8_AUTOTUNE_ACTIVE
from ..modules.mhc.hyper_connection import HCState
from ..modules.multi_stream_utils import no_eager_forks
from ..pyexecutor.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    enable_breakable_cuda_graph,
)
from ..utils import make_weak_ref

if TYPE_CHECKING:
    from ..attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata


@dataclass
class DecoderPassRows:
    """What the eager attention breaks of a decoder-pass graph read at each replay."""

    rows: int = 0
    attn_metadata: Optional["CSA2TrtllmMetadata"] = None


def decoder_pass_buckets(window: int, max_rows: int) -> tuple[int, ...]:
    """Multiples of the replay window at most 1.5x apart, up to ``max_rows``."""
    buckets, units = [], 1
    while units * window < max_rows:
        buckets.append(units * window)
        units = units + 1 if units < 4 else units * 3 // 2
    return (*buckets, max_rows)


@dataclass
class _Captured:
    graph: BreakableCUDAGraph
    # Weak views into the shared pool, written by each replay of this graph.
    residual: torch.Tensor
    pre_mix: torch.Tensor


class DecoderPassGraph:
    """Per-bucket breakable CUDA graphs of a model's replayed decoder layers.

    ``body(position_ids, input_ids, hc_state, attn_metadata, row_split, image_mask)`` runs the
    layers.
    """

    def __init__(self, body: Callable[..., HCState], window: int):
        self._body = body
        self._window = window
        self.buckets: Optional[tuple[int, ...]] = None
        self._split = DecoderPassRows()
        self._graphs: dict[int, _Captured] = {}
        # Inputs of the largest bucket; smaller buckets capture views of them.
        self._inputs: Optional[tuple[torch.Tensor, ...]] = None
        self._pool = None
        self._stream: Optional[torch.cuda.Stream] = None

    def reset(self) -> None:
        for captured in self._graphs.values():
            captured.graph.reset()
        self._graphs.clear()
        self._inputs = None
        self._pool = None

    def select(self, attn_metadata: "CSA2TrtllmMetadata") -> Optional[int]:
        """The row bucket this forward replays, or ``None`` to run the layers eagerly.

        Decided only from state every attention-DP rank shares.
        """
        if AutoTuner.get().is_tuning_mode or _FLASHINFER_MXFP8_AUTOTUNE_ACTIVE.get():
            # Tuning may change tactics the graphs captured.
            self.reset()
            return None
        if self.buckets is None:
            # A compacting pass keeps at most one window of rows per request.
            self.buckets = decoder_pass_buckets(
                self._window,
                min(attn_metadata.max_num_tokens, attn_metadata.max_num_requests * self._window),
            )
        peak = max(attn_metadata.all_rank_num_tokens)
        return next((size for size in self.buckets if size >= peak), None) if peak else None

    def run(
        self,
        bucket: int,
        attn_metadata: "CSA2TrtllmMetadata",
        hc_state: HCState,
        position_ids: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        image_mask: Optional[torch.Tensor] = None,
    ) -> HCState:
        """Run the layers for the resolved ``hc_state`` at the shape of ``bucket``.

        Text forwards replay the bucket's graph; the returned state views the graph pool and
        is valid until that graph replays again. Image rows route experts with their own bias,
        so a forward with an image mask runs the same layers eagerly at the bucket's shape.
        """
        if not self._graphs:
            # In a server the first such forward is a warmup forward, inside the memory profile.
            self._capture_all(attn_metadata, hc_state, position_ids, input_ids)
        rows = hc_state.residual.shape[0]
        if attn_metadata.num_tokens > rows:
            raise ValueError("CSA2 query rows exceed the decoder-pass rows")
        live_inputs = (hc_state.residual, hc_state.pre_mix, position_ids, input_ids)
        for static, live in zip(self._inputs, live_inputs):
            if live is not None:
                static[:rows].copy_(live.reshape(static[:rows].shape))
            static[rows:bucket].zero_()
        self._split.rows, self._split.attn_metadata = attn_metadata.num_tokens, attn_metadata
        try:
            if image_mask is not None:
                padded_mask = image_mask.new_zeros(bucket)
                padded_mask[:rows].copy_(image_mask)
                output = self._run_bucket(bucket, attn_metadata, padded_mask)
                return HCState.resolved(output.residual[:rows], pre_mix=output.pre_mix[:rows])
            with enable_breakable_cuda_graph():
                self._graphs[bucket].graph.replay()
        finally:
            self._split.rows, self._split.attn_metadata = 0, None
        captured = self._graphs[bucket]
        return HCState.resolved(captured.residual[:rows], pre_mix=captured.pre_mix[:rows])

    def _capture_all(
        self,
        attn_metadata: "CSA2TrtllmMetadata",
        hc_state: HCState,
        position_ids: torch.Tensor,
        input_ids: Optional[torch.Tensor],
    ) -> None:
        largest = self.buckets[-1]
        residual = hc_state.residual
        self._inputs = (
            residual.new_zeros((largest, *residual.shape[1:])),
            hc_state.pre_mix.new_zeros((largest, *hc_state.pre_mix.shape[1:])),
            position_ids.new_zeros(largest),
            torch.zeros(largest, dtype=torch.int32, device=residual.device),
        )
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=residual.device)
        self._pool = torch.cuda.graph_pool_handle()
        # Imported here: the runner helpers import model modules.
        from ..pyexecutor.engine.runners.common import moe_a2a_steady_state_budget_for_capture

        current = torch.cuda.current_stream()
        self._stream.wait_stream(current)
        try:
            with torch.cuda.stream(self._stream), no_eager_forks():
                # First-use JIT and allocations at every shape, under the warmup budget.
                for bucket in reversed(self.buckets):
                    self._run_bucket(bucket, attn_metadata)
                # Replays keep the all-to-all budget they were captured with.
                with moe_a2a_steady_state_budget_for_capture():
                    # Largest first, so the smaller captures reuse the shared pool.
                    for bucket in reversed(self.buckets):
                        graph = BreakableCUDAGraph()
                        # Thread-local: other threads (KV transfer, offload) may launch work.
                        with BreakableCUDAGraphCapture(
                            graph,
                            pool=self._pool,
                            stream=self._stream,
                            capture_error_mode="thread_local",
                        ):
                            output = self._run_bucket(bucket, attn_metadata)
                        self._graphs[bucket] = _Captured(
                            graph, make_weak_ref(output.residual), make_weak_ref(output.pre_mix)
                        )
        except Exception:
            self.reset()
            raise
        finally:
            current.wait_stream(self._stream)
        logger.info(
            "DeepSeek-V4.1 decoder-pass CUDA graphs engaged: captured row buckets "
            f"{list(self.buckets)}"
        )

    def _run_bucket(
        self,
        bucket: int,
        attn_metadata: "CSA2TrtllmMetadata",
        image_mask: Optional[torch.Tensor] = None,
    ) -> HCState:
        """Run the layers on the bucket's static inputs, with every rank at the bucket's rows."""
        residual, pre_mix, position_ids, input_ids = (t[:bucket] for t in self._inputs)
        live_counts = attn_metadata.all_rank_num_tokens
        # The collectives run at the bucket's shape on every rank.
        attn_metadata.all_rank_num_tokens = [bucket] * len(live_counts)
        try:
            with enable_breakable_cuda_graph():
                return self._body(
                    position_ids,
                    input_ids,
                    HCState.resolved(residual, pre_mix=pre_mix),
                    attn_metadata,
                    self._split,
                    image_mask,
                )
        finally:
            attn_metadata.all_rank_num_tokens = live_counts
