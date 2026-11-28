# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Breakable CUDA graphs of the two passes of a bounded-replay context step, as stock's `deepseekv41_decoder_graph.py`.

A context step with a decoder replay plan runs

- the encoder pass (the layers before `decoder_replay_split`, every row) from a graph per token bucket, and
- the decoder pass (the replayed layers, the plan's rows) from a graph per row bucket,

captured with the engine's breakable CUDA graphs (`BreakableCUDAGraph`, `BreakableCUDAGraphCapture`) in one pool shared
by both passes, on one capture stream. mHC boundaries, routing, the experts' all-to-all, the experts and the shared
expert are graph segments over the bucket's rows; attention and the engram run in eager breaks (`eager_on_graph`) over
the live rows only (`PassLive`), with the step's own metadata. Every attention-DP rank replays the bucket of the largest
rank's row count, a rank whose own rows are all generation rows included (stock's identity replay plans), so the
captured collectives match; padding rows are zero and the MoE routes them to no expert (`padding`, the
body's last argument, from the replay's live row count on the device), so they cost no expert work and no all-to-all
traffic and any padding is allowed (stock's `max_pad=None` for MoEs that route padding nowhere).

All buckets of a pass are captured on its first use (`_capture_all`): every bucket runs once eagerly (first-use
compilation and allocation at every shape), then each is captured, largest first so the smaller ones reuse the pool.
Breaks see no live rows during the warm runs and the captures, so they do no work there. In a server the first such
forward is a warm-up forward, inside the memory profile.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    enable_breakable_cuda_graph,
)
from tensorrt_llm._torch.utils import make_weak_ref
from tensorrt_llm.logger import logger

# Token buckets of the encoder pass (stock's ENCODER_BUCKETS), capped by the engine's max_num_tokens.
ENCODER_BUCKETS = (2048, 4096, 8192, 12288, 16384, 20480, 24576, 26624, 28672, 30720, 32768)
# Padding rows a rank may carry in the encoder pass (stock's
# ENCODER_MAX_PAD); None: any, as the MoE routes them nowhere.
ENCODER_MAX_PAD = None
# Decoder passes above this many rows run eagerly (stock's DECODER_MAX_ROWS).
DECODER_MAX_ROWS = 8064

_capture = threading.local()


def forks_in_capture() -> bool:
    """Whether the pass graph being captured keeps the eager side-stream forks (the shared expert beside the routed
    experts) as a fork and join inside the graph: the encoder pass, as stock's `eager_forks_in_capture`."""
    return getattr(_capture, "forks", False)


def decoder_pass_buckets(window: int, max_rows: int) -> Tuple[int, ...]:
    """Multiples of the replay window at most 1.5x apart, up to `max_rows` (stock's `decoder_pass_buckets`)."""
    buckets, units = [], 1
    while units * window < max_rows:
        buckets.append(units * window)
        units = units + 1 if units < 4 else units * 3 // 2
    return (*buckets, max_rows)


@dataclass
class PassLive:
    """What the eager breaks of a pass graph read at each replay: the live rows of this rank, the step context their
    attention runs with, the engram ids of those rows and the live per-rank row counts (the engram's collectives)."""

    rows: int = 0
    ctx: object = None
    engram_ids: Optional[torch.Tensor] = None
    counts: Optional[list] = None

    def clear(self) -> None:
        """Nothing is live between replays (and during the warm runs and captures)."""
        self.rows, self.ctx, self.engram_ids, self.counts = 0, None, None, None


class PassGraphPool:
    """The graph memory pool, and the capture stream its blocks are reused on, that a model's pass graphs share. The
    passes never replay at once, and a pass copies its inputs out of the previous pass's outputs before it replays."""

    def __init__(self):
        self._handle = None
        self._holders = 0
        self._stream: Optional[torch.cuda.Stream] = None

    def acquire(self):
        if not self._holders:  # a pool whose graphs are all gone cannot be shared again
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
    outputs: Tuple[
        torch.Tensor, ...
    ]  # weak views into the shared pool, written by each replay of this graph


class _PassGraph:
    """Per-bucket breakable CUDA graphs of a run of layers. `body(*inputs, live, counts, padding)` runs the layers on
    the first rows of each static input, with `counts` the per-rank rows its collectives run at and `padding` (bool
    `[rows, 1]`) the rows past the replay's live rows, and returns a tuple of tensors."""

    # The inputs may be views of the previous pass's outputs, in the shared pool.
    inputs_from_pass = False
    # Captures keep the eager side-stream forks (see `forks_in_capture`).
    forks_in_capture = False

    def __init__(
        self, body: Callable[..., Tuple[torch.Tensor, ...]], pool: PassGraphPool, world: int
    ):
        self._body = body
        self._pool_source = pool
        self._world = world
        self.buckets: Optional[Tuple[int, ...]] = None
        self.live = PassLive()
        self._graphs: dict = {}
        self._inputs: Optional[Tuple[torch.Tensor, ...]] = (
            None  # of the largest bucket; smaller ones capture views
        )
        self._row_ids: Optional[torch.Tensor] = (
            None  # row index of the largest bucket, int32 [rows, 1]
        )
        self._live_rows: Optional[torch.Tensor] = (
            None  # the replay's live rows (int32 [1]): the rest are padding
        )
        self._pool = None
        self._stream: Optional[torch.cuda.Stream] = None

    @property
    def captured(self) -> bool:
        return bool(self._graphs)

    def reset(self) -> None:
        for captured in self._graphs.values():
            captured.graph.reset()
        self._graphs.clear()
        self._inputs = self._row_ids = self._live_rows = None
        if self._pool is not None:
            self._pool_source.release()
        self._pool = None

    def _tuning(self) -> bool:
        if AutoTuner.get().is_tuning_mode:  # tuning may change tactics the graphs captured
            self.reset()
            return True
        return False

    def _static_inputs(
        self, largest: int, live_inputs: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(t.new_zeros((largest, *t.shape[1:])) for t in live_inputs)

    def replay(
        self, bucket: int, live_inputs: Sequence[torch.Tensor], rows: int, ctx, engram_ids, counts
    ):
        """Copy the live inputs into the bucket's static inputs and replay its graph with the eager breaks reading this
        rank's `rows` live rows; returns the outputs' first `rows` rows, views of the pool valid until the graph that
        wrote them replays again (or the other pass replays)."""
        if not self._graphs:
            if (
                self.inputs_from_pass
            ):  # a capture's warm runs write pool memory that may still hold these inputs
                live_inputs = [t.clone() for t in live_inputs]
            self._capture_all(live_inputs)
        for static, value in zip(self._inputs, live_inputs):
            static[: value.shape[0]].copy_(value)
            static[value.shape[0] : bucket].zero_()
        self._live_rows.fill_(rows)
        live = self.live
        live.rows, live.ctx, live.engram_ids, live.counts = rows, ctx, engram_ids, counts
        try:
            with enable_breakable_cuda_graph():
                self._graphs[bucket].graph.replay()
        finally:
            live.clear()
        return tuple(t[:rows] for t in self._graphs[bucket].outputs)

    def _capture_all(self, live_inputs: Sequence[torch.Tensor]) -> None:
        free0, reserved0, t0 = (
            torch.cuda.mem_get_info()[0],
            torch.cuda.memory_reserved(),
            time.perf_counter(),
        )
        self._inputs = self._static_inputs(self.buckets[-1], live_inputs)
        device = self._inputs[0].device
        self._row_ids = torch.arange(self.buckets[-1], dtype=torch.int32, device=device)[:, None]
        # the warm runs and the captures route every row; each replay sets its own live rows
        self._live_rows = torch.full((1,), self.buckets[-1], dtype=torch.int32, device=device)
        if self._stream is None:
            self._stream = self._pool_source.stream(device)
        self._pool = self._pool_source.acquire()
        current = torch.cuda.current_stream()
        self._stream.wait_stream(current)
        try:
            with torch.cuda.stream(self._stream):
                for bucket in reversed(
                    self.buckets
                ):  # first-use compilation and allocations at every shape
                    self._run_bucket(bucket)
                # the warm runs leave their blocks cached, and a capture cannot free cache to grow the pool
                self._stream.synchronize()
                torch.cuda.empty_cache()
                _capture.forks = self.forks_in_capture
                for bucket in reversed(
                    self.buckets
                ):  # largest first, so the smaller captures reuse the pool
                    graph = BreakableCUDAGraph()
                    # thread-local: other threads (KV transfer, offload) may launch work
                    with BreakableCUDAGraphCapture(
                        graph,
                        pool=self._pool,
                        stream=self._stream,
                        capture_error_mode="thread_local",
                    ):
                        outputs = self._run_bucket(bucket)
                    self._graphs[bucket] = _Captured(
                        graph, tuple(make_weak_ref(t) for t in outputs)
                    )
        except Exception:
            self.reset()
            raise
        finally:
            _capture.forks = False
            current.wait_stream(self._stream)
        logger.info(
            f"DeepSeek-V4.1 dep2 {type(self).__name__} CUDA graphs captured: buckets {list(self.buckets)} in "
            f"{time.perf_counter() - t0:.1f} s, "
            f"device memory used +{(free0 - torch.cuda.mem_get_info()[0]) / 2**30:.2f} GiB, "
            f"torch reserved +{(torch.cuda.memory_reserved() - reserved0) / 2**30:.2f} GiB"
        )

    def _run_bucket(self, bucket: int) -> Tuple[torch.Tensor, ...]:
        padding = self._row_ids[:bucket] >= self._live_rows
        with enable_breakable_cuda_graph():
            return self._body(
                *(t[:bucket] for t in self._inputs), self.live, [bucket] * self._world, padding
            )


class EncoderPassGraph(_PassGraph):
    """Per-token-bucket graphs of the layers before the decoder replay boundary: `body(embeddings, live, counts,
    padding)` -> (residual, pre_mix)."""

    # The shared expert overlaps the routed experts from 12K rows, as in eager steps (stock's encoder pass).
    forks_in_capture = True

    def select(self, counts, max_num_tokens: int) -> Optional[int]:
        """The token bucket every rank replays, or None to run eagerly; from state every attention-DP rank shares."""
        if self._tuning():
            return None
        if self.buckets is None:
            self.buckets = (*(b for b in ENCODER_BUCKETS if b < max_num_tokens), max_num_tokens)
        if not counts or min(counts) <= 0:
            return None
        bucket = next((b for b in self.buckets if b >= max(counts)), None)
        if bucket is None or (
            ENCODER_MAX_PAD is not None and bucket - min(counts) > ENCODER_MAX_PAD
        ):
            return None
        return bucket


class DecoderPassGraph(_PassGraph):
    """Per-row-bucket graphs of the replayed layers and the final norm: `body(residual, pre_mix, live, counts,
    padding)` -> (hidden,)."""

    inputs_from_pass = True

    def __init__(self, body, pool: PassGraphPool, world: int, window: int):
        super().__init__(body, pool, world)
        self._window = window

    def select(self, counts, max_num_tokens: int, max_num_requests: int) -> Optional[int]:
        """The row bucket every rank replays, or None to run eagerly; from state every attention-DP rank shares."""
        if self.buckets is None:  # a compacting pass keeps at most one window of rows per request
            self.buckets = decoder_pass_buckets(
                self._window, min(max_num_tokens, max_num_requests * self._window, DECODER_MAX_ROWS)
            )
        if self._tuning():
            return None
        peak = max(counts) if counts else 0
        return next((b for b in self.buckets if b >= peak), None) if peak else None
