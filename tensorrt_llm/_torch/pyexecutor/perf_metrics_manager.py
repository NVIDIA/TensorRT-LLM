"""Performance metrics manager for PyExecutor.

Encapsulates GPU/CPU timing instrumentation: event creation, recording,
and per-request metric bookkeeping.  Extracted from PyExecutor to improve
readability and separation of concerns.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Optional

import torch

from tensorrt_llm.serve.responses_utils import get_steady_clock_now_in_seconds

from .llm_request import PerfTimingInfo


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
            gpu_sample_end)`` or ``(None, None, None)`` if metrics are
            disabled.
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

    def compute_batch_gpu_times(self, requests):
        """Compute GPU times once per batch for the last ctx chunk or gen step.

        Reads events from perf fields, computes elapsed_time once per batch,
        and fills in gpu_forward_time / gpu_sample_time for the last entry in
        either ``ctx_chunk_metrics`` or ``step_metrics``.
        For ctx chunks, also accumulates ``ctx_gpu_forward_time`` across all
        chunks.
        """
        if not self.enabled:
            return
        batch_gpu_forward_time = None
        batch_gpu_sample_time = None
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

            # Compute once per batch, reuse for all requests
            if batch_gpu_forward_time is None:
                batch_gpu_forward_time = perf.gpu_forward_start_event.elapsed_time(
                    perf.gpu_forward_end_event
                )
                batch_gpu_sample_time = (
                    perf.gpu_forward_end_event.elapsed_time(perf.gpu_sample_end_event)
                    if perf.gpu_sample_end_event
                    else 0.0
                )

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
            not request.is_generation_only_request()
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
