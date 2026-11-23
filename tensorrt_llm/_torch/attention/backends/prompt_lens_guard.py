# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Opt-in guard for the reuse of the pinned ``prompt_lens`` host buffer.

``TrtllmAttentionMetadata.prepare()`` writes the step's prompt lengths into the
persistent pinned buffer ``prompt_lens_cpu`` and enqueues
``prompt_lens_cuda.copy_(prompt_lens_cpu, non_blocking=True)``. A pinned
host-to-device copy is a DMA that reads its source when it executes, not when
it is enqueued, and nothing keeps this buffer alive per call. If the next
``prepare()`` of the same metadata (for example for the next pipeline-parallel
micro-batch) rewrites the buffer before the pending copy has run, the pending
step's device ``prompt_lens_cuda`` (the attention op's ``context_lengths``)
gets the next step's lengths, while its ``kv_lens_cuda`` (copied from a fresh
pinned block) keeps its own. Where the new prompt length of a row is larger
than the pending step's KV length of that row, the context KV append computes a
negative past length and indexes the block table before the row; other
mismatches shift the K/V writes silently.

``TLLM_ATTN_PROMPT_LENS_GUARD`` (read once at import, default off):

- unset or ``0``: no change.
- ``probe``: detect only. A CUDA event is recorded on the current stream right
  after the native ``prompt_lens`` copy. At the start of the next ``prepare()``,
  before the buffer is rewritten, an event that has not completed means the
  copy may not have run yet; this is counted and logged (WARNING, rate
  limited) with the pending step's prompt and KV lengths, the new prompt
  lengths and a per-row prediction (``negative past`` / ``mismatch`` /
  ``same``).
- ``fix``: the same detection, then the host waits on that event before the
  rewrite. It blocks only when the rewrite would have raced the copy.
- ``fresh``: the copy reads a fresh pinned block holding this step's prompt
  lengths (the pattern ``kv_lens`` already uses); the caching host allocator
  keeps that block until the copy has run. ``prompt_lens_cpu`` is still written
  for its host consumers. The probe detection also runs (no blocking), so the
  log shows how often the window opened while the copy was protected.

No value the model computes changes. This module depends on
torch only, so it can be unit-tested without the native bindings.
"""

import os
from typing import Any, Callable, List, Mapping, Optional, Sequence

import torch

PROMPT_LENS_GUARD_ENV = "TLLM_ATTN_PROMPT_LENS_GUARD"
GUARD_TAG = "ATTN_PROMPT_LENS_GUARD"

MODE_PROBE = "probe"
MODE_FIX = "fix"
MODE_FRESH = "fresh"
GUARD_MODES = (MODE_PROBE, MODE_FIX, MODE_FRESH)
_OFF_VALUES = ("", "0")

ROW_SAME = "same"
ROW_MISMATCH = "mismatch"
ROW_NEGATIVE_PAST = "negative past"
ROW_UNKNOWN = "unknown"

# WARNING lines: the first _LOG_FIRST events, then every _LOG_EVERY-th.
_LOG_FIRST = 20
_LOG_EVERY = 100
# INFO summary: the first prepare() of a guard, then every _SUMMARY_EVERY-th.
_SUMMARY_EVERY = 1000
_MAX_VALUES_SHOWN = 16

LogFn = Callable[[str], None]


def parse_guard_mode(value: Optional[str]) -> Optional[str]:
    """Return the guard mode for an environment value, or None when off.

    Raises ``ValueError`` for an unknown value, so that a misspelled mode does
    not silently run without the guard.
    """
    if value is None:
        return None
    mode = value.strip().lower()
    if mode in _OFF_VALUES:
        return None
    if mode in GUARD_MODES:
        return mode
    raise ValueError(
        f"{PROMPT_LENS_GUARD_ENV}={value!r}: expected unset, "
        f"'0', {', '.join(repr(m) for m in GUARD_MODES)}"
    )


def guard_mode_from_env(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The guard mode set in ``environ`` (default ``os.environ``)."""
    environ = os.environ if environ is None else environ
    return parse_guard_mode(environ.get(PROMPT_LENS_GUARD_ENV))


def predict_rows(
    old_prompt_lens: Sequence[int],
    old_kv_lens: Optional[Sequence[int]],
    new_prompt_lens: Sequence[int],
) -> List[str]:
    """Predict what the pending copy hands the device if the rewrite lands first.

    The pending copy reads rows ``[0, len(old_prompt_lens))``; the rewrite
    writes rows ``[0, len(new_prompt_lens))``. Per row the pending copy reads:

    - ``negative past``: the row is rewritten with a prompt length larger than
      the pending step's KV length of that row (past = kv - q < 0);
    - ``mismatch``: rewritten with a different length, past still >= 0;
    - ``same``: not rewritten, or rewritten with the same length;
    - ``unknown``: rewritten with a different length, and the pending step's
      KV length of the row was not recorded.
    """
    rows = []
    for i, old_q in enumerate(old_prompt_lens):
        if i >= len(new_prompt_lens):
            rows.append(ROW_SAME)
            continue
        new_q = int(new_prompt_lens[i])
        if new_q == int(old_q):
            rows.append(ROW_SAME)
        elif old_kv_lens is None or i >= len(old_kv_lens):
            rows.append(ROW_UNKNOWN)
        elif new_q > int(old_kv_lens[i]):
            rows.append(ROW_NEGATIVE_PAST)
        else:
            rows.append(ROW_MISMATCH)
    return rows


def worst_row(rows: Sequence[str]) -> str:
    """The most severe prediction among ``rows``."""
    for label in (ROW_NEGATIVE_PAST, ROW_UNKNOWN, ROW_MISMATCH):
        if label in rows:
            return label
    return ROW_SAME


def _shown(values: Sequence[int]) -> str:
    values = [int(v) for v in values]
    if len(values) <= _MAX_VALUES_SHOWN:
        return str(values)
    return f"{values[:_MAX_VALUES_SHOWN]} (+{len(values) - _MAX_VALUES_SHOWN} more)"


def describe_rows(
    old_prompt_lens: Sequence[int],
    old_kv_lens: Optional[Sequence[int]],
    new_prompt_lens: Sequence[int],
    rows: Sequence[str],
) -> str:
    """Per-row text for the rows predicted to differ."""
    parts = []
    for i, label in enumerate(rows):
        if label == ROW_SAME:
            continue
        if len(parts) >= _MAX_VALUES_SHOWN:
            parts.append("...")
            break
        new_q = int(new_prompt_lens[i])
        text = f"row {i}: q {int(old_prompt_lens[i])} -> {new_q}"
        if old_kv_lens is not None and i < len(old_kv_lens):
            kv = int(old_kv_lens[i])
            text += f", kv {kv}, past {kv - new_q}"
        parts.append(f"{text} ({label})")
    return "; ".join(parts) if parts else "no row differs"


class PromptLensH2DGuard:
    """Per-metadata state of ``TLLM_ATTN_PROMPT_LENS_GUARD`` (see the module doc).

    The metadata calls, in ``prepare()`` order: ``before_rewrite`` (before
    ``prompt_lens_cpu`` is written), ``copy_source`` (the source of the native
    copy; differs from ``prompt_lens_cpu`` only in ``fresh`` mode),
    ``after_native_copy`` (right after the native ``prompt_lens`` copy; not
    when a step-input stager took the copy), and ``note_kv_lens`` (with the KV
    lengths copied to the device by the same ``prepare()``). While the current
    stream is capturing a CUDA graph, all four leave everything as it is: no
    event is queried or recorded, and the copy keeps ``prompt_lens_cpu`` as its
    source (a captured copy must read a buffer that outlives the graph).
    """

    def __init__(
        self,
        mode: str,
        name: str,
        warn: LogFn,
        info: LogFn,
        pin_memory: bool = True,
        event_factory: Optional[Callable[[], Any]] = None,
        is_capturing: Optional[Callable[[], bool]] = None,
    ):
        if mode not in GUARD_MODES:
            raise ValueError(f"unknown {PROMPT_LENS_GUARD_ENV} mode {mode!r}")
        self.mode = mode
        self.name = name
        self._warn = warn
        self._info = info
        self._pin_memory = pin_memory
        self._event_factory = event_factory or torch.cuda.Event
        self._is_capturing = is_capturing or torch.cuda.is_current_stream_capturing
        self._event = None
        # The rows the last native copy reads, and that step's KV lengths.
        self._pending_prompt_lens: Optional[torch.Tensor] = None
        self._pending_kv_lens: Optional[torch.Tensor] = None
        self._awaiting_kv_lens = False
        self.num_prepares = 0
        self.num_native_copies = 0
        self.num_events = 0
        self.num_negative_past = 0
        self.num_mismatch = 0
        self.num_waits = 0

    def before_rewrite(self, new_prompt_lens: torch.Tensor) -> Optional[str]:
        """Check the last native copy before ``prompt_lens_cpu`` is rewritten.

        Returns the worst row prediction when the copy has not run yet (the
        race window is open), else None. In ``fix`` mode, then waits for it.
        """
        if self._is_capturing():
            return None
        self.num_prepares += 1
        outcome = None
        event = self._event
        if event is not None and not event.query():
            outcome = self._on_window_open(new_prompt_lens)
            if self.mode == MODE_FIX:
                event.synchronize()
                self.num_waits += 1
        if self.num_prepares == 1 or self.num_prepares % _SUMMARY_EVERY == 0:
            self._info(self.summary())
        return outcome

    def copy_source(self, prompt_lens_cpu: torch.Tensor) -> torch.Tensor:
        """The source of the native copy: ``prompt_lens_cpu`` itself, or in
        ``fresh`` mode a fresh (pinned) block holding the same values."""
        if self.mode != MODE_FRESH or self._is_capturing():
            return prompt_lens_cpu
        fresh = torch.empty(
            prompt_lens_cpu.shape,
            dtype=prompt_lens_cpu.dtype,
            device="cpu",
            pin_memory=self._pin_memory,
        )
        fresh.copy_(prompt_lens_cpu)
        return fresh

    def after_native_copy(self, prompt_lens: torch.Tensor) -> None:
        """Record the event of the native copy just enqueued with ``prompt_lens``."""
        if self._is_capturing():
            return
        if self._event is None:
            self._event = self._event_factory()
        self._event.record()
        self.num_native_copies += 1
        self._pending_prompt_lens = prompt_lens.clone()
        self._pending_kv_lens = None
        self._awaiting_kv_lens = True

    def note_kv_lens(self, kv_lens: torch.Tensor) -> None:
        """The KV lengths copied to the device by the ``prepare()`` whose
        native copy was recorded last (ignored after a staged copy)."""
        if self._awaiting_kv_lens:
            self._pending_kv_lens = kv_lens.clone()
            self._awaiting_kv_lens = False

    def _on_window_open(self, new_prompt_lens: torch.Tensor) -> str:
        self.num_events += 1
        old_q = [] if self._pending_prompt_lens is None else self._pending_prompt_lens.tolist()
        old_kv = None if self._pending_kv_lens is None else self._pending_kv_lens.tolist()
        new_q = new_prompt_lens.tolist()
        rows = predict_rows(old_q, old_kv, new_q)
        outcome = worst_row(rows)
        if outcome == ROW_NEGATIVE_PAST:
            self.num_negative_past += 1
        elif outcome != ROW_SAME:
            self.num_mismatch += 1
        if self.num_events <= _LOG_FIRST or self.num_events % _LOG_EVERY == 0:
            if self.mode == MODE_FIX:
                action = "Waiting for the copy before the rewrite (fix)"
            elif self.mode == MODE_FRESH:
                action = (
                    "Not blocking: the copy reads a private pinned block "
                    "(fresh), so the rewrite does not reach it"
                )
            else:
                action = "Not blocking (probe)"
            num_same = self.num_events - self.num_negative_past - self.num_mismatch
            pending_kv = "not recorded" if old_kv is None else _shown(old_kv)
            self._warn(
                f"{GUARD_TAG}: {self.name}: the prompt_lens copy of the previous "
                f"prepare() has not run yet and this prepare() is about to "
                f"rewrite its pinned source (open window {self.num_events}, "
                f"prepare {self.num_prepares}). Prediction for the pending step "
                f"if the rewrite lands first: {outcome}. Pending prompt_lens "
                f"{_shown(old_q)}, pending kv_lens {pending_kv}, new prompt_lens "
                f"{_shown(new_q)}; {describe_rows(old_q, old_kv, new_q, rows)}. "
                f"{action}. Totals: {self.num_negative_past} negative past, "
                f"{self.num_mismatch} mismatch, {num_same} same."
            )
        return outcome

    def summary(self) -> str:
        """One line of counters, logged at INFO periodically."""
        return (
            f"{GUARD_TAG} ({PROMPT_LENS_GUARD_ENV}={self.mode}) {self.name}: "
            f"{self.num_prepares} prepares, {self.num_native_copies} native "
            f"copies, {self.num_events} open windows "
            f"({self.num_negative_past} negative past, {self.num_mismatch} "
            f"mismatch), {self.num_waits} waits"
        )


#: The mode of this process, read once at import (None: off).
PROMPT_LENS_GUARD_MODE: Optional[str] = guard_mode_from_env()
