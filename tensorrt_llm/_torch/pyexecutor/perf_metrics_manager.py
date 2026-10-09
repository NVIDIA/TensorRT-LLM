# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance metrics manager for PyExecutor.

Encapsulates GPU/CPU timing instrumentation: event creation, recording,
and per-request metric bookkeeping.  Extracted from PyExecutor to improve
readability and separation of concerns.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Optional

import torch

from tensorrt_llm._utils import get_steady_clock_now_in_seconds
from tensorrt_llm.logger import logger

from .llm_request import PerfTimingInfo


class PerfMetricsManager:
    """Manages GPU/CPU timing instrumentation for PyExecutor iterations.

    Args:
        enabled: Whether performance metrics collection is turned on
            (mirrors ``LlmArgs.return_perf_metrics``).
    """

    # Give up on a CUDA event that never becomes ready rather than growing the
    # deferred list without bound (a lost span is preferable to a leak).
    _MAX_DEFERRED_GPU_READS = 8

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._perf_events = None
        self._perf_event_idx = 0
        self._forward_event_pool = []
        # One shared per-iteration step record. Every field of a step/chunk
        # entry is batch-level, so the batch needs one dict, not one per
        # request; ``save_timing_to_requests`` opens a new iteration by
        # clearing this. See :meth:`append_step_metrics`.
        self._batch_record = None
        # (start_event, end_event, sample_end_event, record) triples whose GPU
        # events were not ready yet; drained on the next call rather than
        # blocking the executor loop. See :meth:`compute_batch_gpu_times`.
        self._deferred_gpu_reads = []

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
                tuple(torch.cuda.Event(enable_timing=True) for _ in range(3)),
                tuple(torch.cuda.Event(enable_timing=True) for _ in range(3)),
            ]
            self._perf_event_idx = 0
        events = self._perf_events[self._perf_event_idx % 2]
        self._perf_event_idx += 1
        return events

    def borrow_forward_timing_events(self):
        """Borrow a forward-only pair when the ping-pong perf events are unavailable."""
        if self._forward_event_pool:
            return self._forward_event_pool.pop()
        return (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))

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

    def save_timing_to_requests(
        self,
        requests,
        gpu_forward_start,
        gpu_forward_end,
        gpu_sample_end,
        forward_start_time,
        forward_end_time,
        sample_start_time,
        sample_end_time,
    ):
        """Save current iteration's timing info to all requests in the batch.

        Also opens a new iteration for the shared step record: the previous
        iteration's entry is complete once new forward timings land, so the next
        :meth:`append_step_metrics` builds a fresh one.
        """
        self._batch_record = None
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

    def _read_gpu_times(self, start_event, end_event, sample_end_event):
        """Read the batch's GPU spans without blocking.

        Returns ``(gpu_forward_time, gpu_sample_time)``, or ``None`` when the
        events have not completed yet and the caller should retry later.
        """
        try:
            if not end_event.query():
                return None
            if sample_end_event is not None and not sample_end_event.query():
                return None
            gpu_forward_time = start_event.elapsed_time(end_event)
            gpu_sample_time = (
                end_event.elapsed_time(sample_end_event) if sample_end_event else 0.0
            )
        except RuntimeError as e:
            # CUDA event timing can fail if events were not recorded on the
            # current stream. Skip metrics for this batch rather than crashing
            # the executor thread.
            logger.warning(
                "Failed to compute GPU event elapsed_time: %s. "
                "Setting batch GPU times to 0.0. This may indicate "
                "an issue with the forward pass or stream synchronization.",
                e,
            )
            return 0.0, 0.0
        return gpu_forward_time, gpu_sample_time

    @staticmethod
    def _accumulate_ctx_gpu_times(perf, gpu_forward_time, gpu_sample_time):
        """Add one chunk's GPU times to a request's context-phase totals."""
        if perf.ctx_gpu_forward_time is None:
            perf.ctx_gpu_forward_time = 0.0
            perf.ctx_gpu_sample_time = 0.0
        perf.ctx_gpu_forward_time += gpu_forward_time
        perf.ctx_gpu_sample_time += gpu_sample_time

    def _drain_deferred_gpu_reads(self):
        """Retry GPU reads whose events were still in flight on an earlier call."""
        if not self._deferred_gpu_reads:
            return
        still_pending = []
        for entry in self._deferred_gpu_reads:
            start_event, end_event, sample_end_event, record, ctx_perf, attempts = entry
            times = self._read_gpu_times(start_event, end_event, sample_end_event)
            if times is None:
                if attempts + 1 < self._MAX_DEFERRED_GPU_READS:
                    still_pending.append(
                        (
                            start_event,
                            end_event,
                            sample_end_event,
                            record,
                            ctx_perf,
                            attempts + 1,
                        )
                    )
                continue
            record["gpu_forward_time"], record["gpu_sample_time"] = times
            if ctx_perf is not None:
                self._accumulate_ctx_gpu_times(ctx_perf, *times)
        self._deferred_gpu_reads = still_pending

    def compute_batch_gpu_times(self, requests):
        """Fill GPU times into the batch's ctx-chunk / gen-step entry.

        Reads ``elapsed_time`` once per batch -- not once per request -- because
        the entry is shared by every request in the batch (see
        :meth:`append_step_metrics`). For ctx chunks the per-request
        ``ctx_gpu_forward_time`` totals are still accumulated individually.

        Never synchronizes on a CUDA event: an event that has not completed is
        retried on a later call instead, so instrumentation cannot cost the
        executor its host run-ahead under the overlap scheduler.

        Args:
            requests: Requests whose latest timing entry should be updated.
        """
        if not self.enabled:
            return
        self._drain_deferred_gpu_reads()
        for req in requests:
            perf = req.py_perf_timing
            if perf is None:
                continue
            record = perf.pending_gpu_record
            if record is None or perf.gpu_forward_start_event is None:
                continue
            # Claim the entry so a shared record is accounted once per request,
            # even when compute_batch_gpu_times is called per request.
            perf.pending_gpu_record = None
            is_ctx = perf.pending_gpu_is_ctx
            perf.pending_gpu_is_ctx = False

            # Only the first request of the batch pays for the event read; the
            # rest see the times already written into the shared entry.
            if record["gpu_forward_time"] == 0 and record["gpu_sample_time"] == 0:
                times = self._read_gpu_times(
                    perf.gpu_forward_start_event,
                    perf.gpu_forward_end_event,
                    perf.gpu_sample_end_event,
                )
                if times is None:
                    # Retry on a later call rather than blocking here. Carries
                    # the ctx accumulation along so the totals are not short by
                    # this chunk. Bounded by _MAX_DEFERRED_GPU_READS.
                    self._deferred_gpu_reads.append(
                        (
                            perf.gpu_forward_start_event,
                            perf.gpu_forward_end_event,
                            perf.gpu_sample_end_event,
                            record,
                            perf if is_ctx else None,
                            0,
                        )
                    )
                    continue
                record["gpu_forward_time"], record["gpu_sample_time"] = times

            if is_ctx:
                self._accumulate_ctx_gpu_times(
                    perf, record["gpu_forward_time"], record["gpu_sample_time"]
                )

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

        # Every field below is batch-level: the forward/sample timestamps come
        # from this iteration's events and ``batch_token_time`` is the batch's
        # token timestamp. So the whole batch shares one dict instead of
        # allocating an identical one per request -- at concurrency 666 that is
        # 665 dicts per iteration that no longer get built. The entry stays
        # shared through serialization; nothing downstream mutates it except
        # compute_batch_gpu_times, which writes the batch's GPU times once.
        metric = self._batch_record
        if metric is None or metric["forward_start_time"] != perf.forward_start_time:
            metric = {
                "forward_start_time": perf.forward_start_time,
                "forward_end_time": perf.forward_end_time,
                "sample_start_time": perf.sample_start_time,
                "sample_end_time": perf.sample_end_time,
                "gpu_forward_time": 0,
                "gpu_sample_time": 0,
                # batch_token_time is None only on paths that have no batch-wide
                # token timestamp; then the read is per request and unshareable.
                "token_time": batch_token_time or get_steady_clock_now_in_seconds(),
            }
            self._batch_record = metric if batch_token_time is not None else None

        perf.pending_gpu_record = metric
        if is_ctx:
            # Mark complete when context is done (remaining == 0 after move_to_next_chunk)
            if request.context_remaining_length == 0:
                perf.ctx_chunks_complete = True
            perf.pending_gpu_is_ctx = True
            perf.ctx_chunk_metrics.append(metric)
        else:
            perf.pending_gpu_is_ctx = False
            # The absolute iteration of each step is step_iter_base + index, so
            # a single int per request replaces one "iter" field per step.
            if perf.step_iter_base is None:
                perf.step_iter_base = request.py_decoding_iter
            perf.step_metrics.append(metric)
