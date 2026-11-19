# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Breakable CUDA graphs for the two passes of attention-DP bounded-replay prefill.

The model builds them only with breakable prefill CUDA graphs
(``prefill_cuda_graph_backend=breakable``). Every attention-DP prefill batch then carries a
decoder replay plan, so no prefill replays a whole-body graph: prefill forwards run their
passes from these graphs:

- the encoder pass, the layers before the replay boundary over every prompt row, per token
  bucket (host time is the cost of eager prefill at full token budgets);
- the decoder pass, the replayed layers over the final-window rows of each prompt, per row
  bucket (a few hundred rows per rank, whose cost is host launch time).

mHC boundaries, MoE (router, EP all-to-all, experts, shared expert) and the decoder pass's
row-wise attention projections are graph segments. Eager breaks run, on the live rows, the
metadata-driven attention (the whole CSA2 module in the encoder pass, its sparse core in the
decoder pass) and the encoder pass's Engram work. Every attention-DP rank replays the bucket of
the largest rank's row count, so the captured collectives match; padding rows are zero, and the MoE
routes them to no expert where it can (``routes_padding_nowhere``).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Sequence

import torch

from tensorrt_llm.logger import logger

from ..autotuner import AutoTuner
from ..custom_ops.torch_custom_ops import MXFP8GemmRunner
from ..modules.linear import _FLASHINFER_MXFP8_AUTOTUNE_ACTIVE
from ..modules.mhc.hyper_connection import HCState
from ..modules.multi_stream_utils import eager_forks_in_capture, no_eager_forks
from ..moe.fused_moe.moe_comm_timeout_guard import moe_comm_serving_timeouts
from ..pyexecutor.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    enable_breakable_cuda_graph,
)
from ..utils import make_weak_ref

if TYPE_CHECKING:
    from ..attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

# Token buckets of the encoder pass. A replay pads every rank to its bucket, and padding rows cost
# MoE work or EP traffic, so a step with more padding on a rank than ENCODER_MAX_PAD runs eagerly
# unless the graph pads freely (``max_pad=None``).
ENCODER_BUCKETS = (2048, 4096, 8192, 12288, 16384, 20480, 24576, 26624, 28672, 30720, 32768)
ENCODER_MAX_PAD = 2048
# Decoder passes above this many rows run eagerly: they are rare, and capturing them grows the
# shared graph pool.
DECODER_MAX_ROWS = 8064  # 63 x 128


@dataclass
class PassRows:
    """What the eager breaks of a pass graph read at each replay: the live rows and metadata, and
    for the encoder pass the forward's Engram lookups by layer, their events and the launcher of
    the lookups that start at a layer."""

    rows: int = 0
    attn_metadata: Optional["CSA2TrtllmMetadata"] = None
    all_rank_num_tokens: Optional[list[int]] = None
    engram: Optional[dict] = None
    engram_events: Optional[dict] = None
    launch_engram: Optional[Callable[[int], None]] = None

    def clear(self) -> None:
        """Nothing is live between replays."""
        self.rows, self.attn_metadata, self.all_rank_num_tokens = 0, None, None
        self.engram = self.engram_events = self.launch_engram = None

    @contextmanager
    def live_counts(self) -> Iterator[None]:
        """The live per-rank token counts in the metadata, which an eager run at a bucket's
        shape sets to the bucket's for the collectives it captures alike."""
        metadata = self.attn_metadata
        saved = metadata.all_rank_num_tokens
        metadata.all_rank_num_tokens = self.all_rank_num_tokens
        try:
            yield
        finally:
            metadata.all_rank_num_tokens = saved


def decoder_pass_buckets(window: int, max_rows: int) -> tuple[int, ...]:
    """Multiples of the replay window at most 1.5x apart, up to ``max_rows``."""
    buckets, units = [], 1
    while units * window < max_rows:
        buckets.append(units * window)
        units = units + 1 if units < 4 else units * 3 // 2
    return (*buckets, max_rows)


class PassGraphPool:
    """The CUDA graph memory pool, and the capture stream its blocks are reused on, that a model's
    pass graphs share. The passes never replay at once, and a pass copies its inputs out of the
    previous pass's outputs before it replays."""

    def __init__(self):
        self._handle = None
        self._holders = 0
        self._stream: Optional[torch.cuda.Stream] = None

    def acquire(self):
        """The pool to capture into, held until ``release``."""
        if not self._holders:
            # A pool whose graphs are all gone cannot be shared again.
            self._handle = torch.cuda.graph_pool_handle()
        self._holders += 1
        return self._handle

    def release(self) -> None:
        self._holders -= 1

    def stream(self, device: torch.device) -> torch.cuda.Stream:
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=device)
        return self._stream


@dataclass
class _Captured:
    graph: BreakableCUDAGraph
    # Weak views into the shared pool, written by each replay of this graph.
    residual: torch.Tensor
    pre_mix: torch.Tensor


class _PassGraph:
    """Per-bucket breakable CUDA graphs of a run of layers.

    ``body(*inputs, attn_metadata, live, image_mask, unrouted_rows)`` runs the layers on the first
    rows of each static input and returns the resolved state; ``unrouted_rows`` marks the padding
    rows.
    """

    # Captures keep the eager shared-expert fork, as a fork / join inside the graph.
    forks_in_capture = False
    # The inputs may be views of the previous pass's outputs, in the shared pool.
    inputs_from_pass = False

    def __init__(self, body: Callable[..., HCState], pool: PassGraphPool):
        self._body = body
        self._pool_source = pool
        self.buckets: Optional[tuple[int, ...]] = None
        self._live = PassRows()
        self._graphs: dict[int, _Captured] = {}
        # Inputs of the largest bucket; smaller buckets capture views of them.
        self._inputs: Optional[tuple[torch.Tensor, ...]] = None
        # Row indices of the largest bucket and the live row count the MoE routes.
        self._row_ids: Optional[torch.Tensor] = None
        self._live_rows: Optional[torch.Tensor] = None
        self._pool = None
        self._stream: Optional[torch.cuda.Stream] = None

    def reset(self) -> None:
        for captured in self._graphs.values():
            captured.graph.reset()
        self._graphs.clear()
        self._inputs = None
        self._row_ids = self._live_rows = None
        if self._pool is not None:
            self._pool_source.release()
        self._pool = None

    def _tuning(self) -> bool:
        if AutoTuner.get().is_tuning_mode or _FLASHINFER_MXFP8_AUTOTUNE_ACTIVE.get():
            # Tuning may change tactics the graphs captured.
            self.reset()
            return True
        return False

    def _static_inputs(self, largest: int, live_inputs: Sequence[Optional[torch.Tensor]]):
        raise NotImplementedError

    def _replay(
        self,
        bucket: int,
        attn_metadata: "CSA2TrtllmMetadata",
        live_inputs: Sequence[Optional[torch.Tensor]],
        image_mask: Optional[torch.Tensor] = None,
        engram: Optional[dict] = None,
        engram_events: Optional[dict] = None,
        launch_engram: Optional[Callable[[int], None]] = None,
    ) -> HCState:
        """Copy the live inputs into the bucket's static inputs and replay its graph.

        The returned state views the graph pool and is valid until that graph replays again. A
        forward with an image mask runs the same layers eagerly at the bucket's shape: image rows
        route experts with their own bias.
        """
        if not self._graphs:
            if self.inputs_from_pass:
                # A capture runs its eager breaks for real, in pool memory that still holds the
                # previous pass's outputs: take the inputs out of it first.
                live_inputs = [None if value is None else value.clone() for value in live_inputs]
            # In a server the first such forward is a warmup forward, inside the memory profile.
            self._capture_all(attn_metadata, live_inputs)
        rows = live_inputs[0].shape[0]
        if attn_metadata.num_tokens > rows:
            raise ValueError("CSA2 query rows exceed the pass rows")
        for static, value in zip(self._inputs, live_inputs):
            if value is not None:
                static[:rows].copy_(value.reshape(static[:rows].shape))
            static[rows:bucket].zero_()
        self._live_rows.fill_(rows)
        live = self._live
        live.rows, live.attn_metadata = attn_metadata.num_tokens, attn_metadata
        live.all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        live.engram, live.engram_events, live.launch_engram = engram, engram_events, launch_engram
        try:
            if image_mask is not None:
                padded_mask = image_mask.new_zeros(bucket)
                padded_mask[:rows].copy_(image_mask)
                output = self._run_bucket(bucket, attn_metadata, padded_mask)
                return HCState.resolved(output.residual[:rows], pre_mix=output.pre_mix[:rows])
            with enable_breakable_cuda_graph():
                self._graphs[bucket].graph.replay()
        finally:
            live.clear()
        captured = self._graphs[bucket]
        return HCState.resolved(captured.residual[:rows], pre_mix=captured.pre_mix[:rows])

    def _capture_all(
        self, attn_metadata: "CSA2TrtllmMetadata", live_inputs: Sequence[Optional[torch.Tensor]]
    ) -> None:
        self._inputs = self._static_inputs(self.buckets[-1], live_inputs)
        device = self._inputs[0].device
        self._row_ids = torch.arange(self.buckets[-1], dtype=torch.int32, device=device)[:, None]
        # Warm runs and captures route every row; each replay sets its own live rows.
        self._live_rows = torch.full((1,), self.buckets[-1], dtype=torch.int32, device=device)
        if self._stream is None:
            self._stream = self._pool_source.stream(device)
        self._pool = self._pool_source.acquire()
        forks = eager_forks_in_capture() if self.forks_in_capture else no_eager_forks()
        current = torch.cuda.current_stream()
        self._stream.wait_stream(current)
        try:
            with torch.cuda.stream(self._stream), forks:
                # First-use JIT and allocations at every shape, under the warmup budget.
                for bucket in reversed(self.buckets):
                    self._run_bucket(bucket, attn_metadata)
                # The warm runs leave their blocks cached, and a capture cannot free cache to grow
                # the graph pool: return it first.
                self._stream.synchronize()
                torch.cuda.empty_cache()
                # Replays keep the MoE communication timeouts they were captured with.
                with moe_comm_serving_timeouts():
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
            f"DeepSeek-V4.1 {type(self).__name__} CUDA graphs engaged: captured row buckets "
            f"{list(self.buckets)}"
        )

    def _run_bucket(
        self,
        bucket: int,
        attn_metadata: "CSA2TrtllmMetadata",
        image_mask: Optional[torch.Tensor] = None,
    ) -> HCState:
        """Run the layers on the bucket's static inputs, with every rank at the bucket's rows."""
        live_counts = attn_metadata.all_rank_num_tokens
        # The collectives run at the bucket's shape on every rank.
        attn_metadata.all_rank_num_tokens = [bucket] * len(live_counts)
        try:
            padding = self._row_ids[:bucket] >= self._live_rows
            with enable_breakable_cuda_graph():
                return self._body(
                    *(tensor[:bucket] for tensor in self._inputs),
                    attn_metadata,
                    self._live,
                    image_mask,
                    padding,
                )
        finally:
            attn_metadata.all_rank_num_tokens = live_counts


class DecoderPassGraph(_PassGraph):
    """Per-row-bucket graphs of a model's replayed decoder layers.

    ``body(residual, pre_mix, position_ids, input_ids, attn_metadata, row_split, image_mask,
    unrouted_rows)``.
    """

    inputs_from_pass = True

    def __init__(self, body: Callable[..., HCState], window: int, pool: PassGraphPool):
        super().__init__(body, pool)
        self._window = window

    def select(self, attn_metadata: "CSA2TrtllmMetadata") -> Optional[int]:
        """The row bucket this forward replays, or ``None`` to run the layers eagerly.

        Decided only from state every attention-DP rank shares.
        """
        if self.buckets is None:
            # A compacting pass keeps at most one window of rows per request.
            self.buckets = decoder_pass_buckets(
                self._window,
                min(
                    attn_metadata.max_num_tokens,
                    attn_metadata.max_num_requests * self._window,
                    DECODER_MAX_ROWS,
                ),
            )
            # The graphs' native MXFP8 GEMMs run at exactly these row counts.
            MXFP8GemmRunner.tune_exact_rows(self.buckets)
        if self._tuning():
            return None
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
        """Run the layers for the resolved ``hc_state`` at the shape of ``bucket``."""
        return self._replay(
            bucket,
            attn_metadata,
            (hc_state.residual, hc_state.pre_mix, position_ids, input_ids),
            image_mask,
        )

    def _static_inputs(self, largest, live_inputs):
        residual, pre_mix, position_ids, _ = live_inputs
        return (
            residual.new_zeros((largest, *residual.shape[1:])),
            pre_mix.new_zeros((largest, *pre_mix.shape[1:])),
            position_ids.new_zeros(largest),
            torch.zeros(largest, dtype=torch.int32, device=residual.device),
        )


class EncoderPassGraph(_PassGraph):
    """Per-token-bucket graphs of a model's layers before the decoder replay boundary.

    ``body(inputs_embeds, position_ids, input_ids, attn_metadata, live, image_mask, unrouted_rows)``.
    """

    # The shared expert overlaps the routed experts from 12K tokens, as in eager forwards.
    forks_in_capture = True

    def __init__(self, body: Callable[..., HCState], max_pad: Optional[int], pool: PassGraphPool):
        """``max_pad=None`` allows any padding: for MoEs that route padding rows to no expert."""
        super().__init__(body, pool)
        self.max_pad = max_pad

    def select(self, attn_metadata: "CSA2TrtllmMetadata") -> Optional[int]:
        """The token bucket every rank replays, or ``None`` to run the layers eagerly.

        Decided only from state every attention-DP rank shares.
        """
        if self._tuning():
            return None
        if self.buckets is None:
            limit = attn_metadata.max_num_tokens
            self.buckets = (*(size for size in ENCODER_BUCKETS if size < limit), limit)
        counts = attn_metadata.all_rank_num_tokens
        if not counts or min(counts) <= 0:
            return None
        bucket = next((size for size in self.buckets if size >= max(counts)), None)
        if bucket is None or (self.max_pad is not None and bucket - min(counts) > self.max_pad):
            return None
        return bucket

    def run(
        self,
        bucket: int,
        attn_metadata: "CSA2TrtllmMetadata",
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        input_ids: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        engram: Optional[dict] = None,
        engram_events: Optional[dict] = None,
        launch_engram: Optional[Callable[[int], None]] = None,
    ) -> HCState:
        """Run the layers from the step's embeddings at the shape of ``bucket``, with the forward's
        Engram lookups by layer, their events and the launcher of the lookups that start at a
        layer (``PassRows``)."""
        return self._replay(
            bucket,
            attn_metadata,
            (inputs_embeds, position_ids, input_ids),
            image_mask,
            engram=engram,
            engram_events=engram_events,
            launch_engram=launch_engram,
        )

    def _static_inputs(self, largest, live_inputs):
        inputs_embeds, position_ids, input_ids = live_inputs
        return (
            inputs_embeds.new_zeros((largest, *inputs_embeds.shape[1:])),
            position_ids.new_zeros(largest),
            input_ids.new_zeros(largest),
        )
