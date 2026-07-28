"""Performance metrics manager for PyExecutor.

Encapsulates GPU/CPU timing instrumentation: event creation, recording,
and per-request metric bookkeeping.  Extracted from PyExecutor to improve
readability and separation of concerns.
"""

import json
import os
import queue
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Optional

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.responses_utils import get_steady_clock_now_in_seconds

from .llm_request import PerfTimingInfo, get_draft_token_length

# Master switch env var. When set to a directory, capture is forced on
# (independent of ``LlmArgs.return_perf_metrics``), extended per-iteration
# batch-context fields are recorded, and each rank writes its own
# ``time_events_rank{N}_pid{P}.jsonl`` live into that directory. Symmetric
# with ``TRTLLM_KVCACHE_TIME_OUTPUT_PATH`` used by the disaggregation perf
# logger.
PERF_TIME_EVENTS_PATH_ENV = "TRTLLM_PERF_TIME_EVENTS_PATH"

# Sentinel enqueued by ``close()`` to stop the writer thread.
_WRITER_STOP = object()


class PerfMetricsManager:
    """Manages GPU/CPU timing instrumentation for PyExecutor iterations.

    Args:
        enabled: Whether performance metrics collection is turned on
            (mirrors ``LlmArgs.return_perf_metrics``).
        capture_extended: Whether to record extended per-iteration
            batch-context / starvation fields and write per-rank time-event
            files. ``None`` (default) derives it from the
            ``TRTLLM_PERF_TIME_EVENTS_PATH`` env var, which also force-enables
            ``enabled``. Pass an explicit bool in unit tests.
    """

    def __init__(self, enabled: bool, capture_extended: Optional[bool] = None):
        events_dir = os.getenv(PERF_TIME_EVENTS_PATH_ENV, "")
        env_on = len(events_dir) > 0
        if capture_extended is None:
            capture_extended = env_on
        # The env var alone turns the whole capture path on, independent of
        # LlmArgs.return_perf_metrics.
        self.enabled = bool(enabled or env_on)
        # Extended fields / per-rank writer only when capture is on AND
        # extended capture was requested (env or explicit).
        self.capture_extended = bool(self.enabled and capture_extended)
        self._perf_events = None
        self._perf_event_idx = 0
        self._forward_event_pool = []

        # Per-rank live writer (off the executor critical path).
        self._events_dir = events_dir or None
        self._events_file = None  # opened lazily by the writer thread
        self._writer_queue: Optional[queue.Queue] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_lock = threading.Lock()

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
        iter_batch_context=None,
    ):
        """Save current iteration's timing info to all requests in the batch.

        ``iter_batch_context`` (when provided) is a dict of per-iteration
        batch-context / starvation fields shared by every request scheduled
        this iteration; it is merged into each request's per-iteration metric
        dict by :meth:`append_step_metrics`. Callers pass it only when
        ``capture_extended`` is on, so the default keeps the base timing path
        untouched.
        """
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
            req.py_perf_timing.iter_batch_context = iter_batch_context

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

        When ``capture_extended`` is on, the shared per-iteration
        ``iter_batch_context`` is merged into each entry, so batch-context and
        the ``scheduled_time`` admission timestamp ride onto every ctx chunk and
        every decode step without any per-request bookkeeping here.

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

        # Extended per-iteration batch context (only when capture_extended is
        # on and the executor stashed a context dict for this iteration).
        # The batch-level fields ride inside each per-iteration metric dict, so
        # they flow through create_response's existing .copy() into
        # time_breakdown_metrics with no payload-shape change.
        if self.capture_extended and perf.iter_batch_context is not None:
            metric.update(perf.iter_batch_context)
            if is_ctx:
                # Tokens processed for THIS request in THIS ctx chunk.
                try:
                    metric["req_context_token_number"] = request.context_chunk_size
                except RuntimeError:
                    last_chunk = getattr(request, "py_last_context_chunk", None)
                    if last_chunk is not None and last_chunk[0] is not None:
                        metric["req_context_token_number"] = last_chunk[1] - last_chunk[0]
            else:
                # Tokens this request emits this gen step (1 + speculative draft).
                metric["req_generation_token_number"] = 1 + get_draft_token_length(request)

        if is_ctx:
            # Mark complete when context is done (remaining == 0 after move_to_next_chunk)
            if request.context_remaining_length == 0:
                perf.ctx_chunks_complete = True
            perf.ctx_chunk_metrics.append(metric)
        else:
            metric["iter"] = request.py_decoding_iter
            perf.step_metrics.append(metric)

    # ------------------------------------------------------------------
    # Per-rank live time-event writer (off the executor critical path)
    # ------------------------------------------------------------------

    def maybe_write_request_events(self, response, rank: int, ctx_request_id=None) -> None:
        """Enqueue a finished request's time-event record for the writer thread.

        Hard-gated by ``capture_extended`` and by the presence of
        ``time_breakdown_metrics`` on the response result -- create_response
        populates that field only on the FINAL response, so this fires exactly
        once per request and naturally skips non-final (first-token / streaming)
        responses.

        ``ctx_request_id`` (the disagg context request id, when the caller has a
        request with ``py_disaggregated_params``) is recorded to let the offline
        aggregator correlate ctx-server and gen-server records. ``None`` on
        non-disagg runs.

        The record is also enriched with the request-level lifecycle scalars from
        the C++ ``RequestPerfMetrics.timing_metrics`` (arrival / first-scheduled /
        first-token / last-token / kv-cache-transfer start+end, and kv_cache_size)
        via :func:`tensorrt_llm.executor.result.get_metrics_dict`. Those fields are
        populated only when the request carried ``return_perf_metrics=True`` -- the
        trtllm-serve entrypoints force that on whenever
        ``TRTLLM_PERF_TIME_EVENTS_PATH`` is set, so on the serve/disagg path the
        scalars are present; a raw ``LLM``-API run that leaves the flag off yields
        an empty dict and the key is simply omitted. These scalars are the steady-
        clock lifecycle anchors the per-iteration ``time_breakdown_metrics`` (a
        relative interior view) cannot express, and they share the server steady
        clock with the disagg-router dispatch file for cross-process joins.

        The on-loop cost is only a dict build + a non-blocking
        ``queue.Queue.put_nowait``; all file I/O (json.dumps + write + flush)
        happens on a lazily-started daemon thread so the executor loop never
        blocks on disk.
        """
        if not self.capture_extended or self._events_dir is None:
            return
        result = getattr(response, "result", None)
        time_breakdown_metrics = getattr(result, "time_breakdown_metrics", None)
        if time_breakdown_metrics is None:
            return

        record = {
            "request_id": getattr(response, "request_id", None),
            "rank": rank,
            # Disagg context id when the runtime exposes one; enables ctx<->gen
            # correlation in the offline aggregator. Absent on non-disagg runs.
            "ctx_request_id": ctx_request_id,
            "time_breakdown_metrics": time_breakdown_metrics,
        }

        # Request-level lifecycle timestamps (steady-clock seconds). Empty unless
        # return_perf_metrics was on for the request (forced by the serve layer
        # under TRTLLM_PERF_TIME_EVENTS_PATH). Lazily imported to avoid a module
        # import cycle at load time. Enum keys are stringified to their stable
        # ``.value`` (arrival_time, first_scheduled_time, ...).
        timing_metrics = self._extract_request_timing_metrics(response)
        if timing_metrics:
            record["request_timing_metrics"] = timing_metrics

        self._ensure_writer(rank)
        try:
            self._writer_queue.put_nowait(record)
        except queue.Full:
            # Bounded queue guards against unbounded memory growth if the
            # writer thread stalls; dropping a record is preferable to
            # blocking the executor loop.
            logger.warning("perf time-events queue full; dropping one record")

    @staticmethod
    def _extract_request_timing_metrics(response) -> dict:
        """Return the C++ request-lifecycle timestamps as a plain str->float dict.

        Reuses ``executor.result.get_metrics_dict`` (which handles the
        ``timedelta.total_seconds()`` conversion and returns ``{}`` when perf
        metrics are absent). Keys are the ``RequestEventTiming`` enum's stable
        ``.value`` strings. Any failure degrades to an empty dict -- this is an
        enrichment, never load-bearing for the primary time-breakdown record.
        """
        try:
            from tensorrt_llm.executor.result import get_metrics_dict

            metrics = get_metrics_dict(response)
        except Exception as e:  # noqa: BLE001 - enrichment must never crash the writer
            logger.debug("perf time-events: request timing enrichment skipped: %s", e)
            return {}
        if not metrics:
            return {}
        return {(k.value if hasattr(k, "value") else str(k)): v for k, v in metrics.items()}

    def _ensure_writer(self, rank: int) -> None:
        """Lazily create the bounded queue + daemon writer thread (once)."""
        if self._writer_thread is not None:
            return
        with self._writer_lock:
            if self._writer_thread is not None:
                return
            self._writer_queue = queue.Queue(maxsize=100000)
            thread = threading.Thread(
                target=self._writer_loop,
                args=(rank,),
                name=f"perf-time-events-writer-rank{rank}",
                daemon=True,
            )
            self._writer_thread = thread
            thread.start()

    def _writer_loop(self, rank: int) -> None:
        """Daemon loop: drain the queue and append one JSON line per record."""
        path = os.path.join(
            self._events_dir,
            f"time_events_rank{rank}_pid{os.getpid()}.jsonl",
        )
        try:
            os.makedirs(self._events_dir, exist_ok=True)
            self._events_file = open(path, "a")
        except OSError as e:
            logger.warning("Failed to open perf time-events file %s: %s", path, e)
            return
        while True:
            record = self._writer_queue.get()
            if record is _WRITER_STOP:
                break
            try:
                self._events_file.write(json.dumps(record) + "\n")
                self._events_file.flush()
            except (OSError, TypeError) as e:
                logger.warning("Failed to write perf time-event record: %s", e)
        try:
            self._events_file.close()
        except OSError:
            pass

    def close(self) -> None:
        """Flush and stop the writer thread; safe to call when disabled."""
        if self._writer_thread is None:
            return
        try:
            self._writer_queue.put(_WRITER_STOP)
        except Exception:  # noqa: BLE001 - best-effort on teardown
            pass
        self._writer_thread.join(timeout=30)
        self._writer_thread = None
