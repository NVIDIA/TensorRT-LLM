# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance metrics manager for PyExecutor.

Encapsulates GPU/CPU timing instrumentation: event creation, recording,
and per-request metric bookkeeping.  Extracted from PyExecutor to improve
readability and separation of concerns.
"""

import os
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Optional, Union

import torch

from tensorrt_llm._utils import get_steady_clock_now_in_seconds, prefer_pinned
from tensorrt_llm.logger import logger

from .llm_request import PerfTimingInfo

# GPU timestamps for the executor's timing events (the per-request perf metrics, the iteration log's device step):
# a model opt-in armed process-wide (arm_gpu_timestamps; the MiniMax-M3.1 sm_107 routing arms it). Kill switch
# TRTLLM_M31_DISABLE_GPU_TIMESTAMPS=1. Needs the native stamp op (trtllm::minimax_m31_globaltimer_stamp).
_GPU_TIMESTAMPS = False
_STAMP_OP = None


def _native_stamp_op():
    try:
        return torch.ops.trtllm.minimax_m31_globaltimer_stamp
    except (AttributeError, RuntimeError):
        return None


def arm_gpu_timestamps() -> bool:
    """Time the executor with GpuTimestamp in this process, unless killed or the native stamp op is missing."""
    global _GPU_TIMESTAMPS, _STAMP_OP
    if os.environ.get("TRTLLM_M31_DISABLE_GPU_TIMESTAMPS", "0") == "1":
        _GPU_TIMESTAMPS = False
    else:
        _STAMP_OP = _native_stamp_op()
        # The stamp lands in pinned host memory with a non-blocking copy; without pinned memory (Confidential
        # Compute) that copy would block the host, so keep timed events there.
        _GPU_TIMESTAMPS = _STAMP_OP is not None and prefer_pinned()
    logger.info(
        f"[M31] executor timing: {'GPU timestamps' if _GPU_TIMESTAMPS else 'timed CUDA events'}"
    )
    return _GPU_TIMESTAMPS


def timing_event() -> Union[torch.cuda.Event, "GpuTimestamp"]:
    """A GPU timing event for the executor: a GpuTimestamp when armed, else a timed CUDA event."""
    if _GPU_TIMESTAMPS:
        return GpuTimestamp()
    return torch.cuda.Event(enable_timing=True)


class GpuTimestamp:
    """The torch.cuda.Event surface the executor's timing uses (record, query, synchronize, elapsed_time), measured
    with %globaltimer.

    A timed event record waits for its stream to drain before it takes its timestamp, and while one is in flight the
    next decode graph's launch waits behind it (a few microseconds per record on Rubin, at every step boundary).
    record() instead makes this timestamp's own non-blocking follower stream wait for the recording stream, then
    stores %globaltimer there into a device word, copies it to pinned host memory and records an untimed event, all
    on the follower. The stamp is taken when the follower passes the recording stream's position, as a timed record's
    timestamp is. Each timestamp has its own follower, so stamps recorded at one position do not queue behind each
    other. Under stream capture record() records nothing. Ported from dongfengy/tekit 5f03c8e89 (GLM-5), with the
    stamp as a native op.
    """

    def __init__(self):
        self._follower: Optional[torch.cuda.Stream] = None
        self._stamp: Optional[torch.Tensor] = None
        self._host: Optional[torch.Tensor] = None
        self._done: Optional[torch.cuda.Event] = None
        self._recorded = False

    @staticmethod
    def _new_stream(device: torch.device) -> torch.cuda.Stream:
        """A non-blocking stream of its own (not one of PyTorch's pooled streams, which can alias the timed one)."""
        from tensorrt_llm._utils import CUASSERT

        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            from cuda import cudart
        index = device.index if device.index is not None else torch.cuda.current_device()
        with torch.cuda.device(index):
            handle = CUASSERT(cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))[0]
        return torch.cuda.ExternalStream(int(handle), device=torch.device("cuda", index))

    @staticmethod
    def _write_stamp(out: torch.Tensor) -> None:
        (_STAMP_OP or _native_stamp_op())(out)

    def record(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        if stream is None:
            stream = torch.cuda.current_stream()
        if torch.cuda.is_current_stream_capturing():
            return
        if self._follower is None:
            self._follower = self._new_stream(stream.device)
        follower = self._follower
        follower.wait_stream(stream)
        with torch.cuda.stream(follower):
            if self._stamp is None:
                self._stamp = torch.empty(1, dtype=torch.int64, device=stream.device)
                self._host = torch.empty(1, dtype=torch.int64, pin_memory=prefer_pinned())
                self._done = torch.cuda.Event()
            self._write_stamp(self._stamp)
            self._host.copy_(self._stamp, non_blocking=True)
            self._done.record(follower)
        self._recorded = True

    def query(self) -> bool:
        return not self._recorded or self._done.query()

    def synchronize(self) -> None:
        if self._recorded:
            self._done.synchronize()

    def elapsed_time(self, end_event: "GpuTimestamp") -> float:
        """Milliseconds from this stamp to ``end_event``'s, once both landed."""
        if not (self._recorded and end_event._recorded):
            raise RuntimeError("GpuTimestamp.elapsed_time: a timestamp was not recorded")
        self.synchronize()
        end_event.synchronize()
        return (int(end_event._host[0]) - int(self._host[0])) / 1e6


class PerfMetricsManager:
    """Manages GPU/CPU timing instrumentation for PyExecutor iterations.

    Args:
        enabled: Whether performance metrics collection is turned on
            (mirrors ``LlmArgs.return_perf_metrics``).
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._perf_events = None
        self._perf_event_idx = 0
        self._forward_event_pool = []

    # ------------------------------------------------------------------
    # GPU event helpers
    # ------------------------------------------------------------------

    def create_timing_events(self):
        """Get GPU timing events for performance measurement.

        Uses ping-pong pattern (two sets of events, alternating per
        iteration) to avoid creating new events every step.  Each set
        persists until the next same-parity iteration, which is safe
        because :meth:`compute_batch_gpu_times` reads the previous
        iteration's events before they are reused.

        Returns:
            Tuple of ``(gpu_forward_start, gpu_forward_end,
            gpu_sample_end)`` or ``(None, None, None)`` if per-request perf
            metrics are disabled.
        """
        if not self.enabled:
            return None, None, None
        if self._perf_events is None:
            self._perf_events = [
                tuple(timing_event() for _ in range(3)),
                tuple(timing_event() for _ in range(3)),
            ]
            self._perf_event_idx = 0
        events = self._perf_events[self._perf_event_idx % 2]
        self._perf_event_idx += 1
        return events

    def borrow_forward_timing_events(self):
        """Borrow a forward-only pair when the ping-pong perf events are unavailable."""
        if self._forward_event_pool:
            return self._forward_event_pool.pop()
        return (timing_event(), timing_event())

    def release_forward_timing_events(self, start_event, end_event) -> None:
        if start_event is not None and end_event is not None:
            self._forward_event_pool.append((start_event, end_event))

    @contextmanager
    def record_perf_events(
        self, start_event: Optional[torch.cuda.Event], end_event: Optional[torch.cuda.Event]
    ):
        """Context manager to record GPU events and CPU timestamps around a
        code block.

        Usage::

            with perf_manager.record_perf_events(gpu_start, gpu_end) as timing:
                do_work()
            # timing.start_time / timing.end_time hold CPU timestamps (or None)

        Args:
            start_event: GPU event to record before the block, or None.
            end_event: GPU event to record after the block, or None.

        Yields:
            A :class:`~types.SimpleNamespace` with ``start_time`` and
            ``end_time`` fields (populated only when metrics are enabled).
        """
        timing = SimpleNamespace(start_time=None, end_time=None)

        # --- Pre-execution: record start ---
        if start_event is not None:
            start_event.record()
        if self.enabled:
            timing.start_time = get_steady_clock_now_in_seconds()

        yield timing

        # --- Post-execution: record end ---
        if end_event is not None:
            end_event.record()
        if self.enabled:
            timing.end_time = get_steady_clock_now_in_seconds()

    # ------------------------------------------------------------------
    # Saving / computing timing data
    # ------------------------------------------------------------------

    def get_timestamp(self) -> Optional[float]:
        """Return a CPU timestamp if metrics are enabled, else ``None``."""
        return get_steady_clock_now_in_seconds() if self.enabled else None

    @staticmethod
    def try_compute_gpu_elapsed_time_ms(
        start_event: Optional[torch.cuda.Event],
        end_event: Optional[torch.cuda.Event],
    ) -> Optional[float]:
        """Return CUDA-event elapsed time if ready, without synchronizing."""
        if start_event is None or end_event is None:
            return None
        try:
            if not end_event.query():
                return None
            return float(start_event.elapsed_time(end_event))
        except RuntimeError as e:
            logger.warning("Failed to compute GPU event elapsed_time: %s", e)
            return None

    @staticmethod
    def save_timing_to_requests(
        requests,
        gpu_forward_start,
        gpu_forward_end,
        gpu_sample_end,
        forward_start_time,
        forward_end_time,
        sample_start_time,
        sample_end_time,
    ):
        """Save current iteration's timing info to all requests in the batch."""
        for req in requests:
            # Lazily create PerfTimingInfo only when perf metrics are enabled
            if req.py_perf_timing is None:
                req.py_perf_timing = PerfTimingInfo()
            req.py_perf_timing.gpu_forward_start_event = gpu_forward_start
            req.py_perf_timing.gpu_forward_end_event = gpu_forward_end
            req.py_perf_timing.gpu_sample_end_event = gpu_sample_end
            req.py_perf_timing.forward_start_time = forward_start_time
            req.py_perf_timing.forward_end_time = forward_end_time
            req.py_perf_timing.sample_start_time = sample_start_time
            req.py_perf_timing.sample_end_time = sample_end_time

    def compute_batch_gpu_times(self, requests, gpu_times_cache=None):
        """Compute GPU times once per batch for the last ctx chunk or gen step.

        Reads events from perf fields, computes elapsed_time once per batch,
        and fills in gpu_forward_time / gpu_sample_time for the last entry in
        either ``ctx_chunk_metrics`` or ``step_metrics``.
        For ctx chunks, also accumulates ``ctx_gpu_forward_time`` across all
        chunks.

        Args:
            requests: Requests whose latest timing entry should be updated.
            gpu_times_cache: Optional cache shared by multiple calls within one
                response-processing pass. Its lifetime must not span executor
                iterations because the timing events are reused.
        """
        if not self.enabled:
            return
        if gpu_times_cache is None:
            gpu_times_cache = {}
        for req in requests:
            perf = req.py_perf_timing
            if perf is None or perf.gpu_forward_start_event is None:
                continue

            # Find the last metric entry with gpu_forward_time == 0
            # Check ctx_chunk_metrics first, then step_metrics
            target = None
            is_ctx = False
            if (
                perf.ctx_chunk_metrics
                and perf.ctx_chunk_metrics[-1].get("gpu_forward_time", 0) == 0
            ):
                target = perf.ctx_chunk_metrics[-1]
                is_ctx = True
            elif perf.step_metrics and perf.step_metrics[-1].get("gpu_forward_time", 0) == 0:
                target = perf.step_metrics[-1]
            if target is None:
                continue

            # Ping-pong events are reused across iterations, so the CPU start
            # timestamp is part of the batch identity.
            cache_key = (
                id(perf.gpu_forward_start_event),
                id(perf.gpu_forward_end_event),
                id(perf.gpu_sample_end_event),
                perf.forward_start_time,
            )
            if cache_key not in gpu_times_cache:
                if not perf.gpu_forward_end_event.query():
                    perf.gpu_forward_end_event.synchronize()
                if perf.gpu_sample_end_event and not perf.gpu_sample_end_event.query():
                    perf.gpu_sample_end_event.synchronize()
                try:
                    batch_gpu_forward_time = perf.gpu_forward_start_event.elapsed_time(
                        perf.gpu_forward_end_event
                    )
                    batch_gpu_sample_time = (
                        perf.gpu_forward_end_event.elapsed_time(perf.gpu_sample_end_event)
                        if perf.gpu_sample_end_event
                        else 0.0
                    )
                except RuntimeError as e:
                    # CUDA event timing can fail if events were not recorded
                    # on the current stream. Skip metrics for this batch rather
                    # than crashing the executor thread.
                    logger.warning(
                        "Failed to compute GPU event elapsed_time: %s. "
                        "Setting batch GPU times to 0.0. This may indicate "
                        "an issue with the forward pass or stream synchronization.",
                        e,
                    )
                    batch_gpu_forward_time = 0.0
                    batch_gpu_sample_time = 0.0
                gpu_times_cache[cache_key] = (
                    batch_gpu_forward_time,
                    batch_gpu_sample_time,
                )

            batch_gpu_forward_time, batch_gpu_sample_time = gpu_times_cache[cache_key]

            target["gpu_forward_time"] = batch_gpu_forward_time
            target["gpu_sample_time"] = batch_gpu_sample_time

            # Accumulate total context GPU times across chunks
            if is_ctx:
                if perf.ctx_gpu_forward_time is None:
                    perf.ctx_gpu_forward_time = 0.0
                    perf.ctx_gpu_sample_time = 0.0
                perf.ctx_gpu_forward_time += batch_gpu_forward_time
                perf.ctx_gpu_sample_time += batch_gpu_sample_time

    def append_step_metrics(self, request, iter_counter: int, batch_token_time=None):
        """Append per-iteration metrics for a request (ctx chunk or gen step).

        For context phase (``py_decoding_iter < 1``): saves to
        ``ctx_chunk_metrics``.
        For generation phase (``py_decoding_iter >= 1``): saves to
        ``step_metrics``.

        Args:
            request: The :class:`LlmRequest` to update.
            iter_counter: Current iteration number from ``PyExecutor``.
            batch_token_time: Optional pre-computed batch token timestamp.
        """
        perf = request.py_perf_timing
        if not self.enabled or perf is None or perf.forward_start_time is None:
            return

        # Determine ctx vs gen:
        # - py_decoding_iter == 0: intermediate chunk (sampler skipped)
        # - py_decoding_iter == 1 and not yet marked complete: last/only chunk
        # - Gen-only requests (disagg gen server) are never ctx
        is_ctx = (
            not request.is_generation_only_request
            and not perf.ctx_chunks_complete
            and request.py_decoding_iter <= 1
        )

        # Skip if timing hasn't changed (request not scheduled this iteration)
        for metrics_list in (perf.step_metrics, perf.ctx_chunk_metrics):
            if metrics_list and metrics_list[-1]["forward_start_time"] == perf.forward_start_time:
                return

        # Common fields for both ctx chunk and gen step
        metric = {
            "forward_start_time": perf.forward_start_time,
            "forward_end_time": perf.forward_end_time,
            "sample_start_time": perf.sample_start_time,
            "sample_end_time": perf.sample_end_time,
            "gpu_forward_time": 0,
            "gpu_sample_time": 0,
        }

        step_token_time = batch_token_time or get_steady_clock_now_in_seconds()
        metric["token_time"] = step_token_time

        if is_ctx:
            # Mark complete when context is done (remaining == 0 after move_to_next_chunk)
            if request.context_remaining_length == 0:
                perf.ctx_chunks_complete = True
            perf.ctx_chunk_metrics.append(metric)
        else:
            metric["iter"] = request.py_decoding_iter
            perf.step_metrics.append(metric)
