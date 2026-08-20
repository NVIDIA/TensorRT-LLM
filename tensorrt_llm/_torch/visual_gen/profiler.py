# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture-window management for VisualGen pipeline profiling.

A single :class:`VisualGenProfiler` owns every profiling decision for one
pipeline: which windows open, when they close, and which collectors they
drive. ``BasePipeline`` holds one instance and exposes thin hooks over it, so
a pipeline with a hand-written denoise loop opts in without re-implementing
any lifecycle logic.

Two collectors share the same windows:

* ``cudaProfilerStart``/``cudaProfilerStop`` — the capture gate an attached
  Nsight Systems collector honours (``nsys profile -c cudaProfilerApi ...``).
  These calls do **not** start Nsight on their own; with no collector
  attached they are cheap no-ops.
* ``torch.profiler`` — a Kineto trace, exported as a Chrome trace when
  ``TLLM_TORCH_PROFILE_TRACE`` is set.

Both collectors use CUPTI, so an Nsight run and a torch-trace run should be
separate invocations.
"""

import os
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Optional, Tuple, Union

import torch

from tensorrt_llm.logger import logger

PROFILE_START_STOP_ENV_VAR_NAME = "TLLM_PROFILE_VISUAL_GEN_START_STOP"
PROFILE_TRACE_ENV_VAR_NAME = "TLLM_TORCH_PROFILE_TRACE"

# Parsed form of PROFILE_START_STOP_ENV_VAR_NAME: a keyword mode, or
# (step starts, step stops) for numeric ranges.
ProfileRange = Union[str, Tuple[frozenset, frozenset], None]


def parse_profile_range() -> ProfileRange:
    """Parse ``TLLM_PROFILE_VISUAL_GEN_START_STOP`` for profiler scoping.

    Visual-gen-specific env var (separate from the LLM path's
    ``TLLM_PROFILE_START_STOP``). Use with ``nsys profile -c cudaProfilerApi ...``.

    Supported formats:

    * ``A-B``            – profile denoise steps A through B
    * ``A-B,C-D,...``    – multiple ranges; profiler toggles on/off per range
    * ``A,B,...``        – individual steps treated as single-step ranges
    * ``predenoise``     – profile from request start through denoise-loop
                           setup, including text encoding and latent
                           preparation. Single-shot.
    * ``postdenoise``    – profile from the end of a denoise loop to request
                           completion, covering VAE decode. Single-shot.
    * ``all``            – profile the full request, from text encoding through
                           VAE decode; skip warmup
    * (unset)            – no profiler API calls; plain ``nsys profile`` captures everything

    Returns ``None`` when unset, one of ``"all"`` / ``"predenoise"`` /
    ``"postdenoise"`` for keyword modes, or ``(frozenset(starts), frozenset(stops))``
    for numeric ranges.

    .. note::
       Step indices are **per-denoise-loop**: each loop resets the counter to
       0, so e.g. ``0-4`` profiles steps 0-4 of *every* request. This differs
       from the LLM path's ``TLLM_PROFILE_START_STOP``, which indexes a global
       executor iteration counter (one forward pass services all in-flight
       requests, so there is no "per request" index). A multi-stage pipeline
       runs more than one denoise loop per request, so a numeric range opens
       one window per stage. A window never outlives its loop: a stop index
       past the loop's last step closes at the last step instead, so a
       numeric range only ever covers denoise work.

       ``predenoise`` and ``postdenoise`` are **single-shot per process**:
       they fire once around the first user request after warmup and do not
       re-arm on subsequent requests. Known limitation: ``postdenoise`` arms
       after the *first* denoise loop, so on a multi-stage pipeline its
       window is a superset of VAE decode — on LTX-2 two-stage it also holds
       the spatial upsample and all of stage 2. Isolating decode there needs
       the profiler to know which loop is last; no mode does it today. Pair
       either mode with
       ``nsys --capture-range-end=stop`` (keeps the app running cleanly after
       collection ends). ``all`` and numeric ranges re-arm for each request;
       use ``--capture-range-end=repeat:N`` to collect multiple requests.
    """
    val = os.environ.get(PROFILE_START_STOP_ENV_VAR_NAME)
    if not val:
        return None
    val = val.strip()
    if val.lower() in ("all", "predenoise", "postdenoise"):
        return val.lower()
    # Parse comma-separated ranges: "A-B,C-D,..." or single steps "A,B,..."
    # Same format as the LLM path (PyExecutor._load_iteration_indexes).
    starts, stops = [], []
    for span in val.split(","):
        span = span.strip()
        if "-" in span:
            start, stop = span.split("-", 1)
            starts.append(int(start))
            stops.append(int(stop))
        else:
            v = int(span)
            starts.append(v)
            stops.append(v)
    return frozenset(starts), frozenset(stops)


class VisualGenProfiler:
    """Opens and closes profiling windows over a VisualGen request.

    Window boundaries, in the order a request hits them:

    * :meth:`request_scope` wraps the whole request. It opens the ``all`` or
      ``predenoise`` window and guarantees every window is closed on the way
      out, including after an exception.
    * :meth:`steps` wraps one denoise loop's iterator. It closes the
      ``predenoise`` window before the first step, opens and closes windows
      on the step indices a numeric range names, and arms the
      ``postdenoise`` window after the loop's last step.

    Callers are responsible for skipping warmup passes. Each closed window
    exports its own trace file, so repeated windows in one process never
    overwrite each other.
    """

    def __init__(self, rank: int = 0) -> None:
        self.range: ProfileRange = parse_profile_range()
        self.rank = rank
        self._active = False
        self._torch_profiler = None
        self._trace_path: Optional[str] = None
        self._window: int = 0
        # Single-shot guards: fire once around the first non-warmup request,
        # then disarm.
        self._predenoise_pending: bool = self.range == "predenoise"
        self._postdenoise_pending: bool = self.range == "postdenoise"
        self._setup_torch_profiler()

    @property
    def enabled(self) -> bool:
        """Whether any profiling window can open."""
        return self.range is not None

    @property
    def active(self) -> bool:
        """Whether a window is currently open."""
        return self._active

    # ------------------------------------------------------------------
    # Window boundaries a pipeline drives
    # ------------------------------------------------------------------

    @contextmanager
    def request_scope(self) -> Iterator[None]:
        """Bracket one request with its request-level profiling windows."""
        if self.range == "all":
            self.open_window()
        elif self._predenoise_pending:
            self._predenoise_pending = False
            self.open_window()
        try:
            yield
        finally:
            # Ends all/postdenoise windows, and safely closes a numeric range
            # if inference raises before the range's own stop index.
            self.close_window()

    def steps(self, timesteps: Iterable[Any]) -> Iterator[Tuple[int, Any]]:
        """Enumerate one denoise loop's steps, driving every window it owns.

        Wrapping the iterator rather than exposing separate
        before-loop/per-step/after-loop hooks means a pipeline cannot
        instrument half a loop: the phase boundaries ride along with the
        enumeration a denoise loop already needs.
        """
        if self.range == "predenoise":
            self.close_window()

        starts, stops = self.range if isinstance(self.range, tuple) else (frozenset(), frozenset())
        for i, t in enumerate(timesteps):
            if i in starts:
                self.open_window()
            yield i, t
            if i in stops:
                self.close_window()

        # Everything below is reached only when the loop runs to completion.
        # A loop that raised or broke out early has no post-denoise work
        # worth capturing, and ``request_scope`` closes whatever window is
        # still open.
        if starts:
            # A numeric range selects denoise steps, so it must not outlive
            # the loop. Without this, a range whose stop index is past the
            # last step -- 0-4 against LTX-2 stage 2's three steps -- would
            # stay open through VAE decode and, on a multi-stage pipeline,
            # swallow the following stage as well.
            self.close_window()
        if self._postdenoise_pending:
            self.open_window()
            self._postdenoise_pending = False

    # ------------------------------------------------------------------
    # Window primitives
    # ------------------------------------------------------------------

    def open_window(self) -> None:
        """Open a capture window if configured and not already open."""
        if not self.enabled or self._active:
            return
        if self._trace_path is not None and self._torch_profiler is None:
            self._create_torch_profiler()
        cudart = torch.cuda.cudart()
        try:
            cudart.cudaProfilerStart()
            if self._torch_profiler is not None:
                self._torch_profiler.start()
        except RuntimeError:
            cudart.cudaProfilerStop()
            self._torch_profiler = None
            raise
        self._active = True
        if self.rank == 0:
            logger.info("CUDA profiler started")

    def close_window(self) -> None:
        """Close the open capture window, exporting its trace."""
        if not self._active:
            return
        torch_profiler = self._torch_profiler
        try:
            # End the window on an idle device. A collector may stop
            # collecting — or end the process, as nsys
            # --capture-range-end=stop-shutdown does — the moment the range
            # closes, so no async work may still be in flight. Mirrors
            # PyExecutor's profile_step().
            #
            # This is also where a sticky CUDA error from an earlier kernel
            # surfaces, which is exactly when the collectors below must still
            # come down: a half-closed window leaves the Nsight capture range
            # open and this profiler wedged for the rest of the process.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        finally:
            self._shutdown_collectors(torch_profiler)

    def _shutdown_collectors(self, torch_profiler: Any) -> None:
        """Stop both collectors and clear window state, whatever else failed."""
        try:
            if torch_profiler is not None:
                torch_profiler.stop()
                trace_path = self._torch_profile_output_path()
                torch_profiler.export_chrome_trace(trace_path)
                self._window += 1
                if self.rank == 0:
                    logger.info(f"PyTorch profiler trace saved to {trace_path}")
        finally:
            self._torch_profiler = None
            try:
                torch.cuda.cudart().cudaProfilerStop()
            finally:
                self._active = False
        if self.rank == 0:
            logger.info("CUDA profiler stopped")

    # ------------------------------------------------------------------
    # torch.profiler plumbing
    # ------------------------------------------------------------------

    def _setup_torch_profiler(self) -> None:
        """Configure PyTorch tracing for the VisualGen profiler range."""
        torch_trace_path = os.environ.get(PROFILE_TRACE_ENV_VAR_NAME)
        if not torch_trace_path:
            return
        if self.range is None:
            logger.warning(
                f"{PROFILE_START_STOP_ENV_VAR_NAME} environment variable "
                "needs to be set to enable the torch trace. Example to profile "
                f"denoise steps 0-4: export {PROFILE_START_STOP_ENV_VAR_NAME}=0-4"
            )
            return

        # Append the rank so each rank writes its own file. Without this,
        # multi-rank runs have every rank exporting to the same path
        # concurrently, producing output that fails to parse.
        trace_base, trace_ext = os.path.splitext(torch_trace_path)
        self._trace_path = f"{trace_base}-rank-{self.rank}{trace_ext}"
        self._create_torch_profiler()

    def _create_torch_profiler(self) -> None:
        """Create a fresh PyTorch profiler for the next capture window."""
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        elif (
            hasattr(torch, "xpu")
            and torch.xpu.is_available()
            and hasattr(torch.profiler.ProfilerActivity, "XPU")
        ):
            activities.append(torch.profiler.ProfilerActivity.XPU)
        self._torch_profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
        )

    def _torch_profile_output_path(self) -> str:
        """Return a non-overwriting path for the current capture window."""
        if self._trace_path is None:
            raise RuntimeError("PyTorch profiler trace path is not configured")
        if self._window == 0:
            return self._trace_path
        trace_base, trace_ext = os.path.splitext(self._trace_path)
        return f"{trace_base}-window-{self._window}{trace_ext}"
