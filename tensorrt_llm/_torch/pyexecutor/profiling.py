# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runtime profiling controller for ``PyExecutor``.

The ``PyExecutorProfileManager`` owns the entire profile-state machine
that backs the ``trtllm-serve`` ``/start_profile`` / ``/stop_profile``
HTTP endpoints (and the equivalent programmatic
``LLM.start_profile`` / ``LLM.stop_profile`` calls):

* ``start_profile`` / ``stop_profile`` — caller-thread entry points
  that schedule a profile window via the executor's request queue.
  These run on the FastAPI thread-pool worker (or a user thread for
  the in-process / RPC paths).
* ``apply_start_config`` / ``apply_stop_config`` — replay the
  broadcasted config on every rank so all ranks fire the same start
  and stop iterations.
* ``profile_step`` — context manager driving the per-iteration
  ``torch.profiler`` / ``cudaProfilerStart`` / ``Stop`` calls. Runs
  on the executor thread because torch.profiler / Kineto require
  the start and stop calls to happen on the same thread.
* ``cleanup`` — best-effort tear-down on executor shutdown.

The profile state itself (``profile_start_iters``, ``profile_stop_iters``,
``_runtime_profile_*`` fields, ``_profile_state_lock``,
``_profile_enabled``) lives on the ``PyExecutor`` instance so callers
and unit tests can introspect it directly. The manager holds a weak
reference back to the executor and operates on those fields. This
keeps ``py_executor.py`` mostly orchestration / thin forwarding while
isolating the lifecycle logic in this module for review.
"""

import os
import time
import uuid
import weakref
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.tools.layer_wise_benchmarks import get_calibrator
from tensorrt_llm.tools.profiler.host_profile_tools.host_profiler import get_global_profiler

# Environment-variable names used by both the env-var-driven legacy
# path and the runtime HTTP-endpoint path. They are duplicated as
# module-level constants here so callers that only need the manager
# do not have to import from ``py_executor.py``.
PROFILE_START_STOP_ENV_VAR_NAME = "TLLM_PROFILE_START_STOP"
PROFILE_TRACE_ENV_VAR_NAME = "TLLM_TORCH_PROFILE_TRACE"
PROFILE_LOG_RANKS_ENV_VAR_NAME = "TLLM_PROFILE_LOG_RANKS"


def load_iteration_indexes(env_var: str):
    """Parse an iteration-range environment variable.

    Format: ``"start1-stop1,start2-stop2,..."`` for ranges or
    ``"iter1,iter2,..."`` for single iterations.

    Returns:
        Two ``frozenset[int]`` -- (start_iters, stop_iters).
    """
    spans = os.environ.get(env_var, None)
    starts, stops = [], []

    if spans:
        spans = spans.split(",")

        for span in spans:
            try:
                if "-" in span:
                    start, stop = span.strip().split("-")
                    starts.append(int(start))
                    stops.append(int(stop))
                else:
                    it = int(span.strip())
                    starts.append(it)
                    stops.append(it)
            except ValueError as e:
                raise ValueError(
                    f"Cannot parse span in environment variable `{env_var}`: {e}"
                ) from None

    return frozenset(starts), frozenset(stops)


class PyExecutorProfileManager:
    """Owns the runtime-profiling state machine for ``PyExecutor``.

    Instantiated from ``PyExecutor.__init__`` after the executor has
    populated ``profile_start_iters`` / ``profile_stop_iters`` /
    ``_runtime_profile_*`` / ``_profile_state_lock`` /
    ``_profile_enabled``. The manager keeps a ``weakref`` back to the
    executor so it can read ``iter_counter``, ``is_warmup``,
    ``executor_request_queue``, ``dist``, and ``global_rank`` on demand
    without creating a cycle that defeats garbage collection.

    All methods follow the same threading invariants as the inline
    code they replace; see the per-method docstrings for details.
    """

    def __init__(self, executor: "PyExecutor") -> None:  # noqa: F821
        self._executor_ref = weakref.ref(executor)

    # ---- helpers ---------------------------------------------------

    @property
    def _executor(self) -> "PyExecutor":  # noqa: F821
        ex = self._executor_ref()
        if ex is None:
            raise RuntimeError(
                "PyExecutorProfileManager: owning PyExecutor has been "
                "garbage collected; profile control calls are no longer "
                "valid."
            )
        return ex

    @staticmethod
    def _activities_from_names(
        names: Optional[List[str]],
    ) -> List["torch.profiler.ProfilerActivity"]:
        if not names:
            return [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.XPU,
            ]
        mapping = {
            "CPU": torch.profiler.ProfilerActivity.CPU,
            "GPU": torch.profiler.ProfilerActivity.CUDA,
            "CUDA": torch.profiler.ProfilerActivity.CUDA,
            "XPU": torch.profiler.ProfilerActivity.XPU,
        }
        resolved: List[torch.profiler.ProfilerActivity] = []
        for name in names:
            key = name.upper()
            if key == "CUDA_PROFILER":
                # CUDA_PROFILER controls cudaProfilerStart/Stop only;
                # it is not a torch.profiler activity.
                continue
            if key in mapping and mapping[key] not in resolved:
                resolved.append(mapping[key])
        return resolved

    # ---- HTTP-callable entry points --------------------------------

    def start_profile(
        self,
        output_dir: Optional[str] = None,
        num_steps: Optional[int] = None,
        start_step: int = 0,
        activities: Optional[List[str]] = None,
    ) -> None:
        """Schedule a runtime profile window on every rank.

        Mirrors the env-var driven ``TLLM_PROFILE_START_STOP`` /
        ``TLLM_TORCH_PROFILE_TRACE`` path but is callable at runtime
        (e.g. from the ``trtllm-serve /start_profile`` HTTP handler).

        Runs on the caller thread (FastAPI worker thread / user
        thread). Marks ``_runtime_profile_pending_start_iter`` locally
        so a concurrent second call is rejected immediately, then
        broadcasts the config on rank 0 via the executor's request
        queue so every rank sees the same start.
        """
        executor = self._executor
        # Reject overlapping profile windows. Check locally on the
        # caller thread so the HTTP handler gets a clean error instead
        # of the executor thread crashing on ``assert not enabled``
        # inside ``profile_step``.
        with executor._profile_state_lock:
            already_active = executor._profile_enabled
        if already_active or executor._runtime_profile_pending_start_iter is not None:
            # Use ``RequestError`` (a ``RuntimeError`` subclass) so the
            # ``GenerationExecutor`` error monitor treats this as a
            # per-call rejection rather than a fatal engine error.
            from tensorrt_llm.executor.utils import RequestError

            raise RequestError(
                "Profiling is already in progress (or a pending start has "
                "been scheduled); call stop_profile first before starting "
                "a new window. (state: _profile_enabled="
                f"{already_active}, _runtime_profile_pending_start_iter="
                f"{executor._runtime_profile_pending_start_iter}, "
                f"profile_start_iters="
                f"{sorted(executor.profile_start_iters)[:5]}, "
                f"profile_stop_iters="
                f"{sorted(executor.profile_stop_iters)[:5]})"
            )

        # Reject ``num_steps <= 0``. The HTTP layer already rejects
        # this via the ``StartProfileRequest`` Pydantic schema, but
        # programmatic callers (``LLM.start_profile``, env-var-driven
        # paths, unit tests) bypass that schema. With
        # ``num_steps == 0``, ``stop_iter`` would equal ``start_iter``
        # in ``apply_start_config`` below, ``profile_step`` would
        # discard the stop marker as stale, and the profile window
        # would run forever until an explicit ``stop_profile()``.
        if num_steps is not None and num_steps <= 0:
            raise ValueError(
                f"start_profile: num_steps must be >= 1 if provided "
                f"(got num_steps={num_steps!r}). Pass num_steps=None to "
                "run until an explicit stop_profile() call."
            )

        if activities is None:
            activities = ["CPU", "GPU"]

        if output_dir is None:
            # ``/tmp`` is the documented developer default for the
            # profile trace directory (matches the server docs); the
            # user-facing override is the
            # ``TLLM_TORCH_PROFILER_DIR`` env var or the
            # ``output_dir`` request body field.
            output_dir = os.environ.get(
                "TLLM_TORCH_PROFILER_DIR",
                "/tmp",  # nosec B108
            )

        # Fail fast if the directory cannot be created. Returning
        # success here would schedule a profile window that later
        # blows up in ``export_chrome_trace()`` on the executor
        # thread, after the HTTP caller has already been told the
        # request was accepted. ``RequestError`` (a ``RuntimeError``
        # subclass) routes through the same per-call rejection
        # plumbing as the "already in progress" guard above, so the
        # HTTP handler can map it to a 5xx with a meaningful body.
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            from tensorrt_llm.executor.utils import RequestError

            raise RequestError(f"Failed to create profile output_dir {output_dir}: {e}") from e

        # Include a unique profile id (monotonic timestamp + short
        # uuid fragment) so successive start/stop cycles under the
        # same output_dir don't silently overwrite each other's
        # traces.
        profile_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"

        # Build the per-request config that every rank consumes. We
        # do NOT pre-compute start_iter here: each rank must derive
        # it from its own ``iter_counter`` at the moment the
        # broadcast is processed (inside
        # ``_handle_special_queue_items``), so that rank 0 doesn't
        # get a head start by pre-applying with its caller-thread
        # ``iter_counter`` while rank 1's executor loop has already
        # advanced past the start point. All ranks are in sync on
        # iter_counter at the broadcast point, so they all compute
        # the same start_iter/stop_iter.
        profile_config = {
            "output_dir": output_dir,
            "profile_id": profile_id,
            "activities": list(activities),
            "start_step": int(max(0, start_step)),
            "num_steps": num_steps,
        }

        # Mark pending on the caller thread too so a second
        # concurrent ``start_profile()`` is rejected by the guard
        # above. The actual ``profile_start_iters`` / trace_path
        # state is only populated when the broadcast reaches
        # ``_handle_special_queue_items``. Use a sentinel (0) to
        # mean "pending but not yet scheduled".
        executor._runtime_profile_pending_start_iter = 0
        executor._runtime_profile_activities = list(activities)

        # Broadcast to every rank via the request queue so non-zero
        # ranks update their own ``profile_start_iters`` in
        # lockstep. This is also what wakes the idle executor loop
        # (the PROFILE_START_REQUEST_ID item gets broadcasted via
        # ``RequestBroadcaster`` inside
        # ``_fetch_and_enqueue_requests``, then dropped inside
        # ``_handle_special_queue_items`` — no forward pass runs for
        # this iteration, so the chrome trace captures only real
        # user workload).
        rank = getattr(getattr(executor, "dist", None), "rank", 0)
        eq = getattr(executor, "executor_request_queue", None)
        if rank == 0 and eq is not None:
            try:
                eq.enqueue_profile_start_request(profile_config)
            except (RuntimeError, OSError, ValueError, AttributeError) as e:
                # Roll back the pending markers we just set so a
                # subsequent ``start_profile()`` is not rejected as
                # "already pending", and so non-zero ranks (which
                # never received the broadcast) don't drift out of
                # sync with rank 0. Then surface the failure to the
                # HTTP layer rather than silently logging and
                # continuing.
                executor._runtime_profile_pending_start_iter = None
                executor._runtime_profile_activities = None
                raise RuntimeError(
                    f"start_profile: failed to enqueue profile-start broadcast item: {e}"
                ) from e

    def stop_profile(self) -> None:
        """Schedule the in-progress runtime profile to stop.

        The actual ``torch.profiler.stop()`` +
        ``export_chrome_trace()`` call happens inside
        ``profile_step`` on the executor thread, so torch.profiler /
        Kineto's "stop must run on the same thread that called
        start" invariant is respected.

        Two extra cases are handled so no profile window is leaked:

        * If ``start_profile()`` was called but the engine has not
          yet reached the scheduled ``start_iter`` (e.g. the caller
          stopped before sending any requests), cancel the pending
          start so profiling does not begin later.
        * Otherwise schedule a stop iteration. When called via an
          HTTP handler the server layer is responsible for tickling
          the executor loop so the stop iteration actually runs even
          when the engine would otherwise be idle (see
          ``OpenAIServer.stop_profile``).
        """
        executor = self._executor
        with executor._profile_state_lock:
            currently_enabled = executor._profile_enabled

        if not currently_enabled and executor._runtime_profile_pending_start_iter is not None:
            # Pending runtime start has not fired yet; cancel it
            # cleanly on this rank.  Also broadcast the stop so
            # non-zero ranks clear their own pending starts —
            # otherwise a ``start_profile``-then-``stop_profile``
            # cycle on an idle multi-rank deployment would leave
            # stale pending starts on the subordinates.
            self.apply_stop_config()
            rank = getattr(getattr(executor, "dist", None), "rank", 0)
            eq = getattr(executor, "executor_request_queue", None)
            if rank == 0 and eq is not None:
                try:
                    eq.enqueue_profile_stop_request()
                except (RuntimeError, OSError, ValueError, AttributeError) as e:
                    # Local cancel already happened on this rank,
                    # but subordinate ranks would still be carrying
                    # the pending start. Surface the broadcast
                    # failure so the caller can retry rather than
                    # silently leaving the cluster out of sync.
                    raise RuntimeError(
                        f"stop_profile: failed to enqueue profile-stop broadcast item: {e}"
                    ) from e
            return

        # Apply locally so diagnostic state (profile_stop_iters,
        # pending_stop_iter) is visible on the caller thread
        # immediately.
        self.apply_stop_config()

        # Broadcast to every rank so non-zero ranks add the same
        # stop_iter to their own profile_stop_iters. Also wakes the
        # idle executor loop on rank 0; the broadcast then
        # propagates the wake to all ranks inside
        # ``_fetch_and_enqueue_requests``.
        rank = getattr(getattr(executor, "dist", None), "rank", 0)
        eq = getattr(executor, "executor_request_queue", None)
        if rank == 0 and eq is not None:
            try:
                eq.enqueue_profile_stop_request()
            except (RuntimeError, OSError, ValueError, AttributeError) as e:
                # Subordinate ranks won't see the stop. Bail out
                # with a real error rather than silently waiting on
                # a stop that will never fire.
                raise RuntimeError(
                    f"stop_profile: failed to enqueue profile-stop broadcast item: {e}"
                ) from e

        # Wait (with timeout) for the stop to actually fire so that
        # by the time stop_profile() returns the chrome trace has
        # been exported to disk. Without this wait, callers that hit
        # the HTTP /stop_profile endpoint and then immediately try
        # to read the trace file would see an empty directory.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            with executor._profile_state_lock:
                if not executor._profile_enabled:
                    return
            time.sleep(0.005)
        logger.warning(
            "stop_profile: executor loop did not fire the scheduled stop "
            "within 30s; the trace may not be fully flushed yet."
        )

    # ---- broadcast handlers (called on every rank) -----------------

    def apply_start_config(self, config: Dict) -> None:
        """Apply a broadcasted profile-start config to this rank.

        Called from two places: directly on the caller thread from
        ``start_profile()`` (so
        ``_runtime_profile_pending_start_iter`` becomes visible for
        re-entrancy checks) AND from ``_handle_special_queue_items``
        when the broadcasted ``PROFILE_START_REQUEST_ID`` item
        arrives. The operation is idempotent: the second call on
        rank 0 sees the same state and is a no-op.
        """
        executor = self._executor
        activities = config.get("activities") or ["CPU", "GPU"]
        # ``/tmp`` matches the documented developer default; see
        # ``start_profile`` for the rationale.
        output_dir = config.get("output_dir") or "/tmp"  # nosec B108
        profile_id = config.get("profile_id") or f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        # Prefer the legacy absolute ``start_iter`` form (older unit
        # tests and env-var driven paths set this directly). For the
        # broadcast path the caller sends ``start_step`` instead so
        # every rank computes its own ``start_iter`` from its
        # current ``iter_counter`` — this avoids a race where rank
        # 0's local apply sets a past start_iter that rank 1 never
        # reaches in time to fire.
        if "start_iter" in config:
            start_iter = int(config["start_iter"])
        else:
            start_step = int(config.get("start_step", 0))
            current_iter = int(getattr(executor, "iter_counter", 0))
            start_iter = current_iter + 1 + start_step
        num_steps = config.get("num_steps")
        stop_iter = config.get("stop_iter")
        if stop_iter is None and num_steps is not None:
            stop_iter = start_iter + int(num_steps)

        # Rank-specific trace path so every rank writes its own file
        # when running with TP/PP > 1.
        trace_filename = f"trtllm-trace-{profile_id}-rank-{executor.global_rank}.json"
        trace_path = os.path.join(output_dir, trace_filename)

        # First apply on this rank? If the start_iter is already in
        # profile_start_iters it means either rank 0's local apply
        # ran (before the broadcast echo-back) or a prior broadcast
        # already applied — in both cases the pending marker
        # belongs to the original apply, and re-setting it here
        # would overwrite a subsequent clear by ``profile_step()``
        # and leak stale state.
        is_first_apply = start_iter not in executor.profile_start_iters

        executor._runtime_profile_trace_path = trace_path
        executor._runtime_profile_activities = list(activities)
        executor._runtime_profile_cuda_only = (
            len(activities) == 1 and activities[0].upper() == "CUDA_PROFILER"
        )

        executor.profile_start_iters.add(start_iter)
        if is_first_apply:
            executor._runtime_profile_pending_start_iter = start_iter
            if stop_iter is not None:
                executor.profile_stop_iters.add(int(stop_iter))
                executor._runtime_profile_pending_stop_iter = int(stop_iter)
            # ``num_steps is None`` branch: leave pending_stop_iter
            # alone — caller set it to None in start_profile.
        elif stop_iter is not None:
            # Broadcast echoes back to rank 0 after its local apply,
            # or after the start has already fired. Only ensure the
            # stop iter is in the set on this rank (idempotent);
            # don't reset pending markers.
            executor.profile_stop_iters.add(int(stop_iter))

        logger.info(
            f"_apply_profile_start_config[rank={executor.global_rank}]: "
            f"start_iter={start_iter}, stop_iter={stop_iter}, "
            f"activities={activities}, trace_path={trace_path}, "
            f"is_first_apply={is_first_apply}"
        )

    def apply_stop_config(self) -> None:
        """Apply a broadcasted profile-stop to this rank.

        Handles both the cancel-pending-start case and the
        schedule-a-stop case so the broadcast from rank 0 replays
        correctly on every rank.
        """
        executor = self._executor
        with executor._profile_state_lock:
            currently_enabled = executor._profile_enabled

        if not currently_enabled and executor._runtime_profile_pending_start_iter is not None:
            pending_start = executor._runtime_profile_pending_start_iter
            pending_stop = executor._runtime_profile_pending_stop_iter
            executor.profile_start_iters.discard(pending_start)
            if pending_stop is not None:
                executor.profile_stop_iters.discard(pending_stop)
            executor._runtime_profile_pending_start_iter = None
            executor._runtime_profile_pending_stop_iter = None
            logger.info(
                f"_apply_profile_stop_config[rank={executor.global_rank}]: "
                f"cancelled pending start at iteration {pending_start} "
                "before it fired"
            )
            return

        # Fire condition inside profile_step(): stop iter matches
        # ``self.iter_counter`` at the top of the NEXT executor loop
        # iteration. ``iter_counter`` is incremented at the END of
        # the current body, so setting ``stop_iter = iter_counter +
        # 1`` fires the stop on the very next profile_step call
        # whether this function runs mid-body (pre-increment) or
        # between iterations (post-increment). One cancel-wake (or
        # the profile-stop broadcast item acting as a wake) is
        # enough to flush.
        current_iter = getattr(executor, "iter_counter", 0)
        stop_iter = current_iter + 1
        executor.profile_stop_iters.add(stop_iter)
        executor._runtime_profile_pending_stop_iter = stop_iter
        logger.info(
            f"_apply_profile_stop_config[rank={executor.global_rank}]: "
            f"scheduled stop at iteration {stop_iter}"
        )

    # ---- per-iteration driver (executor thread) --------------------

    @contextmanager
    def profile_step(self):
        """Context manager driving per-iteration profile bookkeeping.

        Replaces the inline ``_profiler()`` context manager that used
        to live on ``PyExecutor``. Yields a callable
        (``profile_step``) the executor loop invokes once per
        iteration.

        Runs on the executor thread; ``torch.profiler.start`` /
        ``stop`` MUST happen on the same thread, so all of the
        per-iteration ``torch.profiler`` activations live here
        rather than on the HTTP-caller path.
        """
        executor = self._executor
        it = -1
        enabled = False
        start_time = None

        # These events are used to record the time of the previous
        # batch. We need two sets of start-end events so that the
        # ping-pong pattern works with the overlap scheduler.
        start_event_1 = None
        end_event_1 = torch.cuda.Event(enable_timing=True)
        start_event_2 = None
        end_event_2 = torch.cuda.Event(enable_timing=True)
        prev_device_step_time = None

        env_torch_trace_path = os.environ.get(PROFILE_TRACE_ENV_VAR_NAME, None)
        if env_torch_trace_path is not None:
            # Append the rank so each rank writes to its own file.
            # Without this, TP/PP/DP > 1 runs have every rank calling
            # ``torch_profiler.export_chrome_trace()`` on the same
            # path concurrently, producing interleaved output that
            # fails to parse in Chrome tracing / Perfetto.
            trace_base, trace_ext = os.path.splitext(env_torch_trace_path)
            env_torch_trace_path = f"{trace_base}-rank-{executor.global_rank}{trace_ext}"
        profile_start_stop = os.environ.get(PROFILE_START_STOP_ENV_VAR_NAME, None)
        env_enable_torch_trace = bool(env_torch_trace_path and profile_start_stop)
        if env_torch_trace_path and profile_start_stop is None:
            logger.warning(
                f"{PROFILE_START_STOP_ENV_VAR_NAME} environment variable "
                "needs to be set to enable the torch trace. Example to "
                f"profile iteration 10-20: export "
                f"{PROFILE_START_STOP_ENV_VAR_NAME}=10-20"
            )

        # ``torch_profiler`` is created lazily on the first start
        # iteration so it can pick up runtime-configured trace paths
        # / activity lists set via ``start_profile()``. The following
        # variables track the active configuration for the current
        # profiling window.
        torch_profiler = None
        active_torch_trace_path: Optional[str] = None
        active_enable_torch_trace: bool = False
        # Make sure ``_profile_enabled`` starts cleared for a fresh
        # executor run (in case the instance is re-entered).
        with executor._profile_state_lock:
            executor._profile_enabled = False

        log_ranks_str = os.environ.get(PROFILE_LOG_RANKS_ENV_VAR_NAME, "0")
        if log_ranks_str.strip().lower() == "all":
            log_all_ranks = True
            log_ranks = set()
        else:
            log_all_ranks = False
            log_ranks = {int(r) for r in log_ranks_str.split(",")}

        calibrator = get_calibrator()

        # Local helper so the inner closure does not need to call
        # ``self._activities_from_names`` which would force a
        # method-name lookup per iteration.
        activities_from_names = self._activities_from_names

        def profile_step_fn():
            nonlocal it, enabled, start_time
            nonlocal start_event_1, end_event_1, start_event_2, end_event_2
            nonlocal prev_device_step_time
            nonlocal torch_profiler, active_torch_trace_path
            nonlocal active_enable_torch_trace
            calibrator.post_step(it)
            if executor.iter_counter in executor.profile_stop_iters and not executor.is_warmup:
                if not enabled:
                    # Happens when stop_profile() was called before
                    # the scheduled start_iter was reached (e.g.
                    # the user calls /start_profile then
                    # /stop_profile quickly while the engine is
                    # still warming up).  Silently drop the stale
                    # stop marker so we do not crash the event
                    # loop.
                    logger.warning(
                        f"profile stop scheduled at iter "
                        f"{executor.iter_counter} but no active "
                        "profiling window - skipping"
                    )
                    executor.profile_stop_iters.discard(executor.iter_counter)
                else:
                    if active_enable_torch_trace and torch_profiler is not None:
                        torch_profiler.stop()
                        torch_profiler.export_chrome_trace(active_torch_trace_path)
                        logger.info(
                            f"Profiling stopped at iteration "
                            f"{executor.iter_counter}, "
                            f"trace saved to {active_torch_trace_path}"
                        )
                    torch.cuda.cudart().cudaProfilerStop()
                    calibrator.stop()
                    enabled = False
                    with executor._profile_state_lock:
                        executor._profile_enabled = False
                    # Reset lazy state so a subsequent start/stop
                    # window can pick up a fresh runtime
                    # configuration.
                    torch_profiler = None
                    active_torch_trace_path = None
                    # Clear the pending-stop marker so
                    # ``start_profile()``'s guard no longer sees a
                    # stale window.
                    if executor._runtime_profile_pending_stop_iter == executor.iter_counter:
                        executor._runtime_profile_pending_stop_iter = None
                active_enable_torch_trace = False

            # Capture per-loop timing whenever stats or the iter log
            # are enabled. The reading of the OTHER parity's event
            # pair (the ping-pong) is what keeps synchronize() from
            # blocking the GPU — the events being read have already
            # passed by the time we read them. Stashing on
            # ``self`` lets the /metrics serializer pick up the
            # values without going through the log line.
            should_capture_timing = start_time is not None and (
                executor.print_log or executor.enable_iter_perf_stats
            )
            if should_capture_timing:
                end_time = time.time()
                if it % 2 == 0:
                    end_event_1.record()
                    if start_event_2 is not None:
                        end_event_2.synchronize()
                        prev_device_step_time = start_event_2.elapsed_time(end_event_2)
                else:
                    end_event_2.record()
                    if start_event_1 is not None:
                        end_event_1.synchronize()
                        prev_device_step_time = start_event_1.elapsed_time(end_event_1)

                host_step_time = (end_time - start_time) * 1000  # ms
                executor._latest_host_step_time_ms = host_step_time
                executor._latest_prev_device_step_time_ms = prev_device_step_time

                if executor.print_log and (log_all_ranks or executor.dist.rank in log_ranks):
                    if prev_device_step_time is None:
                        prev_device_step_time_str = "N/A"  # first iteration
                    else:
                        prev_device_step_time_str = f"{prev_device_step_time}ms"
                    kv_util_str = "N/A"
                    if executor.kv_cache_manager is not None:
                        kv_stats = executor.kv_cache_manager.get_kv_cache_stats()
                        if kv_stats.max_num_blocks > 0:
                            kv_util_str = (
                                f"{1.0 - kv_stats.free_num_blocks / kv_stats.max_num_blocks:.3f}"
                            )
                    import datetime

                    formatted_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    logger.info(
                        f"iter = {executor.iter_counter}, "
                        f"global_rank = {executor.global_rank}, "
                        f"rank = {executor.dist.rank}, "
                        f"num_scheduled_requests = {executor.num_scheduled_requests}, "
                        f"kv_cache_util = {kv_util_str}, "
                        f"currank_total_requests = {executor.num_fetch_requests_cur_rank}/"
                        f"{executor.num_fetch_requests}, "
                        f"host_step_time = {host_step_time}ms, "
                        f"prev_device_step_time = {prev_device_step_time_str}, "
                        f"timestamp = {formatted_timestamp}, "
                        f"states = {executor.model_engine.iter_states}"
                    )

            it += 1

            if executor.iter_counter in executor.profile_start_iters and not executor.is_warmup:
                assert not enabled, "Inconsistent CUDA profiling state"
                # Resolve active trace path + torch.profiler
                # configuration at start time so runtime overrides
                # from ``start_profile()`` take precedence over the
                # environment variables.
                runtime_trace_path = executor._runtime_profile_trace_path
                runtime_cuda_only = executor._runtime_profile_cuda_only
                runtime_activities = executor._runtime_profile_activities
                if runtime_trace_path is not None:
                    active_torch_trace_path = runtime_trace_path
                    # CUDA_PROFILER only: skip torch.profiler so we
                    # do not pay its overhead and so nsys traces
                    # stay clean.
                    active_enable_torch_trace = not runtime_cuda_only
                    torch_activities = activities_from_names(runtime_activities)
                    if active_enable_torch_trace and not torch_activities:
                        # Runtime requested only CUDA_PROFILER even
                        # though cuda_only was False; treat as
                        # cuda-only.
                        active_enable_torch_trace = False
                else:
                    active_torch_trace_path = env_torch_trace_path
                    active_enable_torch_trace = env_enable_torch_trace
                    torch_activities = [
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                        torch.profiler.ProfilerActivity.XPU,
                    ]

                calibrator.start()
                torch.cuda.cudart().cudaProfilerStart()
                if active_enable_torch_trace:
                    if torch_profiler is None:
                        torch_profiler = torch.profiler.profile(
                            activities=torch_activities, record_shapes=True, with_modules=True
                        )
                    torch_profiler.start()
                logger.info(f"Profiling started at iteration {executor.iter_counter}.")
                enabled = True
                with executor._profile_state_lock:
                    executor._profile_enabled = True
                # The scheduled start has arrived; clear the pending
                # marker so the next ``start_profile()`` after this
                # window ends is not rejected as "already pending".
                if executor._runtime_profile_pending_start_iter == executor.iter_counter:
                    executor._runtime_profile_pending_start_iter = None

            # Notify host line profiler of iteration for
            # iteration-aware profiling.
            host_profiler = get_global_profiler()
            if host_profiler is not None:
                host_profiler.notify_iteration(executor.iter_counter)

            calibrator.pre_step(it)
            start_time = time.time()
            if it % 2 == 0:
                if start_event_1 is None:
                    start_event_1 = torch.cuda.Event(enable_timing=True)
                start_event_1.record()
            else:
                if start_event_2 is None:
                    start_event_2 = torch.cuda.Event(enable_timing=True)
                start_event_2.record()

        try:
            yield profile_step_fn
        finally:
            if enabled:
                # Stop on early exit / exception
                if active_enable_torch_trace and torch_profiler is not None:
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace(active_torch_trace_path)
                    logger.info(
                        f"Profiling stopped at iteration "
                        f"{executor.iter_counter}, "
                        f"trace saved to {active_torch_trace_path}"
                    )
                torch.cuda.cudart().cudaProfilerStop()
                calibrator.stop()
                with executor._profile_state_lock:
                    executor._profile_enabled = False

    # ---- shutdown --------------------------------------------------

    def cleanup(self) -> None:
        """Best-effort shutdown of any active profile window.

        Called during ``PyExecutor.shutdown()`` when an HTTP caller
        forgot to issue a matching ``stop_profile()``. Idempotent
        and never raises -- shutdown must continue even if the
        underlying ``torch.profiler`` / ``cudaProfilerStop`` fails.
        """
        executor = self._executor_ref()
        if executor is None:
            return
        try:
            with executor._profile_state_lock:
                still_active = executor._profile_enabled
                executor._profile_enabled = False
        except Exception:  # pragma: no cover - defensive
            still_active = False
        if not still_active:
            return
        try:
            torch.cuda.cudart().cudaProfilerStop()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"profile cleanup: cudaProfilerStop failed: {e!r}")
