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
"""Verification mode selection and per-step observability for DSpark.

Why this module exists
----------------------
Every failure mode of confidence-scheduled verification is *silent*. The
scheduler can decline to trim, the planner can fall back on a flat cost model,
a batch can miss its captured shape and drop out of graph replay, and a
partially-windowed batch can be refused -- and in every one of those cases the
run still produces correct output at the baseline accuracy. A GSM8K pass
therefore proves nothing about whether the feature was ever active.

So the two questions a reviewer actually has -- *did the ragged path run?* and
*did it trim anything?* -- are made into counters rather than left to
inference, and :meth:`DSparkRaggedStats.assert_ragged_active` turns them into a
test assertion.

Verification modes
------------------
Mirrors SGLang's ``SGLANG_RAGGED_VERIFY_MODE``, and for the same reason: it
separates "is the ragged layout computed correctly" from "does the planner
choose to trim", which otherwise fail together and are indistinguishable.

``static``
    Verify the whole drafted block for every request. The baseline, and what
    every non-DSpark path already does. Scheduling is off.

``cap-accept``
    Compute the per-request windows and carry them through the whole layout,
    but still submit the full block to the target and only *commit* the window.
    The expensive path is unchanged, so this buys no throughput -- it exists
    because its output must be **bit-identical** to ``static``. Any divergence
    is a scheduling bug isolated from the kernel-side packing, which is the one
    comparison that separates the two.

``compact``
    Submit only each request's window. The production path, and the only one
    that trims tokens.

Cost-table dependence
---------------------
``compact`` needs a profiled :class:`~.dspark_planner.SpsCostTable`. Without
one the planner's cost model is flat, every extra verified token looks free,
and the budget degenerates to verify-all -- ragged silently becomes uniform.
:meth:`DSparkRaggedStats.assert_ragged_active` catches exactly that.
"""

import os
from collections import Counter
from enum import Enum
from typing import Dict, List, Optional, Sequence

from ...logger import logger

__all__ = [
    "RaggedVerifyMode",
    "read_ragged_verify_mode",
    "DSparkRaggedStats",
    "forced_verify_lens",
]

#: Overrides ``DSparkDecodingConfig`` so a mode can be selected without editing
#: a test or a serving config. Reading it per call (rather than caching at
#: import) is deliberate: tests flip it with ``monkeypatch.setenv``.
RAGGED_VERIFY_MODE_ENV = "TLLM_DSPARK_RAGGED_VERIFY_MODE"


class RaggedVerifyMode(str, Enum):
    """How much of the drafted block reaches the target. See module docstring."""

    STATIC = "static"
    CAP_ACCEPT = "cap-accept"
    COMPACT = "compact"

    @property
    def computes_windows(self) -> bool:
        """Whether per-request windows are computed at all.

        ``cap-accept`` computes them and commits against them without trimming
        the submitted batch, so it shares every host-side code path with
        ``compact`` except the one that shrinks the token axis.
        """
        return self is not RaggedVerifyMode.STATIC

    @property
    def trims_submitted_tokens(self) -> bool:
        """Whether the token axis handed to the target actually shrinks."""
        return self is RaggedVerifyMode.COMPACT


def read_ragged_verify_mode(
        default: RaggedVerifyMode = RaggedVerifyMode.STATIC
) -> RaggedVerifyMode:
    """Resolve the verification mode from the environment.

    Raises:
        ValueError: if the variable is set to something that is not a mode.
            Silently falling back to ``static`` on a typo would turn "I asked
            for compact" into "the feature never ran", which is the one
            outcome this module exists to make impossible.
    """
    value = os.environ.get(RAGGED_VERIFY_MODE_ENV)
    if value is None or value == "":
        return default
    for mode in RaggedVerifyMode:
        if value == mode.value:
            return mode
    raise ValueError(
        f"invalid {RAGGED_VERIFY_MODE_ENV}={value!r}; expected one of "
        f"{', '.join(repr(m.value) for m in RaggedVerifyMode)}")


#: Debug-only override that forces a deterministic non-uniform split. See
#: :func:`forced_verify_lens`.
FORCE_VERIFY_LENS_ENV = "TLLM_DSPARK_FORCE_VERIFY_LENS"


def forced_verify_lens(*, num_gen_requests: int, tiers: Sequence[int],
                       min_verify_len: int) -> Optional[List[int]]:
    """A deterministic ragged split for correctness testing, or None.

    Set ``TLLM_DSPARK_FORCE_VERIFY_LENS=1`` to have every generation request
    take a window from the captured tier ladder, rotating by batch position.

    This exists because the two failure modes are otherwise inseparable. The
    planner refuses to trim without a profiled cost table -- correctly, since a
    flat model makes every extra token look free -- so a run without one keeps
    the ragged path dark no matter what the config says. Testing that the packing
    is correct would then require first producing a cost table, which requires a
    working run. This breaks that circle.

    It is not a shortcut around the planner: it decides only *how many* drafted
    positions reach the target, exactly the quantity the planner decides.
    Acceptance is unchanged, so a forced split must produce the same answers as
    ``static`` -- which is what makes it a usable correctness gate.

    Returns:
        One window per request, each drawn from ``tiers`` (so every choice has a
        captured CUDA graph), or None when the override is off.
    """
    if os.environ.get(FORCE_VERIFY_LENS_ENV, "") not in ("1", "true", "True"):
        return None
    ladder = sorted({int(t) for t in tiers if int(t) >= int(min_verify_len)})
    if not ladder:
        return None
    # Rotate through the ladder by position: deterministic, identical on every
    # rank for a given batch position, and guaranteed to be non-uniform as soon
    # as the batch is larger than one and the ladder has more than one entry.
    return [ladder[i % len(ladder)] for i in range(int(num_gen_requests))]


class DSparkRaggedStats:
    """Per-step counters for the ragged verification path.

    Cheap enough to leave on: one integer histogram update per decode step, no
    device work and no synchronization. Nothing here reads a device tensor --
    every value recorded is already a host value by the time the scheduler has
    made its decision.
    """

    def __init__(self, *, mode: RaggedVerifyMode, max_draft_len: int):
        self.mode = mode
        self.max_draft_len = int(max_draft_len)

        self.steps_total = 0
        #: Steps where per-request windows were produced AND differed from each
        #: other. A step where every request got the same window is counted as
        #: uniform: it exercises none of the ragged packing.
        self.steps_ragged = 0
        #: Windows were produced, but every request got the same one.
        self.steps_uniform_windows = 0
        #: No windows at all -- the planner declined or the shape did not fit.
        self.steps_no_windows = 0

        #: Tokens the target would have scored with no trimming at all.
        self.ceiling_tokens = 0
        #: Tokens the windows asked for, before bucket rounding.
        self.window_tokens = 0
        #: Tokens actually submitted, after rounding up to a captured bucket.
        self.delivered_tokens = 0

        self.verify_len_hist: Counter = Counter()
        self.bucket_hist: Counter = Counter()
        self.padded_bs_hist: Counter = Counter()

        #: Graph replay is the whole point of landing on a bucket; an eager
        #: step costs far more than the tokens it saved.
        self.graph_replays = 0
        self.graph_eager = 0

        #: Reasons the step declined to go ragged, keyed by a short slug.
        self.fallbacks: Counter = Counter()

    def record_step(
        self,
        *,
        num_gen_requests: int,
        verify_lens: Optional[Sequence[int]] = None,
        bucket: Optional[int] = None,
        padded_bs: Optional[int] = None,
        fallback: Optional[str] = None,
    ) -> None:
        """Record one decode step's scheduling decision.

        Args:
            num_gen_requests: generation requests in the batch.
            verify_lens: per-request drafted positions verified, or None when
                the step stayed uniform. Excludes the bonus token, matching
                ``py_verify_len``.
            bucket: the captured token bucket the batch was fitted to, if any.
            padded_bs: the row count the batch was padded to, if any.
            fallback: short reason the step did not go ragged.
        """
        self.steps_total += 1
        # What a no-trim step would have submitted: every request sends its
        # bonus token plus the full block.
        self.ceiling_tokens += int(num_gen_requests) * (1 + self.max_draft_len)

        if fallback:
            self.fallbacks[fallback] += 1

        if not verify_lens:
            self.steps_no_windows += 1
            self.window_tokens += int(num_gen_requests) * (1 +
                                                           self.max_draft_len)
            self.delivered_tokens += int(num_gen_requests) * (
                1 + self.max_draft_len)
            return

        lens = [int(v) for v in verify_lens]
        self.verify_len_hist.update(lens)
        self.window_tokens += sum(1 + v for v in lens)
        # A bucket is what the graph actually ran; without one the step ran on
        # exactly the windows it asked for.
        self.delivered_tokens += int(bucket) if bucket else sum(1 + v
                                                                for v in lens)
        if bucket:
            self.bucket_hist[int(bucket)] += 1
        if padded_bs:
            self.padded_bs_hist[int(padded_bs)] += 1

        if len(set(lens)) > 1:
            self.steps_ragged += 1
        else:
            self.steps_uniform_windows += 1

    def record_graph(self, *, replayed: bool) -> None:
        """Record whether this step replayed a CUDA graph or ran eager."""
        if replayed:
            self.graph_replays += 1
        else:
            self.graph_eager += 1

    def merge_planner_stats(self, planner_stats: Dict[str, int]) -> None:
        """Fold in the planner's own fallback counters."""
        for key, value in planner_stats.items():
            if key.startswith("fallback_") and value:
                self.fallbacks[key] = int(value)

    @property
    def trim_ratio(self) -> float:
        """Fraction of ceiling tokens the scheduler actually removed.

        ``0.0`` means the feature ran but saved nothing, which is the signature
        of an unprofiled cost table.
        """
        if self.ceiling_tokens <= 0:
            return 0.0
        return 1.0 - (self.delivered_tokens / self.ceiling_tokens)

    @property
    def distinct_verify_lens(self) -> int:
        """How many different window sizes were ever handed out."""
        return len(self.verify_len_hist)

    def summary(self) -> Dict[str, object]:
        """A JSON-safe snapshot, for logging or assertions."""
        return {
            "mode": self.mode.value,
            "steps_total": self.steps_total,
            "steps_ragged": self.steps_ragged,
            "steps_uniform_windows": self.steps_uniform_windows,
            "steps_no_windows": self.steps_no_windows,
            "ceiling_tokens": self.ceiling_tokens,
            "window_tokens": self.window_tokens,
            "delivered_tokens": self.delivered_tokens,
            "trim_ratio": round(self.trim_ratio, 4),
            "distinct_verify_lens": self.distinct_verify_lens,
            "verify_len_hist": dict(sorted(self.verify_len_hist.items())),
            "bucket_hist": dict(sorted(self.bucket_hist.items())),
            "padded_bs_hist": dict(sorted(self.padded_bs_hist.items())),
            "graph_replays": self.graph_replays,
            "graph_eager": self.graph_eager,
            "fallbacks": dict(self.fallbacks),
        }

    def log_summary(self, *, prefix: str = "DSpark ragged verify") -> None:
        summary = self.summary()
        logger.info(f"{prefix}: {summary}")

    def assert_ragged_active(self, *, require_trim: bool = True) -> None:
        """Fail loudly if the ragged path never actually ran.

        This is the check that makes an accuracy run meaningful. Passing GSM8K
        with ``compact`` selected proves nothing on its own -- the scheduler
        may have declined on every step and the run may have been a plain
        uniform baseline wearing a different config.

        Args:
            require_trim: also require that the delivered token count came in
                under the no-trim ceiling. Leave it on unless deliberately
                measuring a workload where every draft survives.

        Raises:
            AssertionError: with the full counter summary, so the reason is in
                the failure rather than in a log the CI dropped.
        """
        summary = self.summary()
        if self.steps_total == 0:
            raise AssertionError(
                f"DSpark ragged verify recorded no decode steps at all: {summary}"
            )
        if self.steps_ragged == 0:
            raise AssertionError(
                f"DSpark ragged verify never produced differing per-request "
                f"windows, so the ragged path was not exercised. The usual "
                f"cause is a missing profiled SPS cost table, under which the "
                f"planner's budget degenerates to verify-all: {summary}")
        if self.distinct_verify_lens < 2:
            raise AssertionError(
                f"DSpark ragged verify handed out only one distinct window "
                f"size, so the batch was uniform in all but name: {summary}")
        if require_trim and self.trim_ratio <= 0.0:
            raise AssertionError(
                f"DSpark ragged verify delivered as many tokens as a no-trim "
                f"run, so nothing was saved: {summary}")
        if self.graph_eager > 0:
            raise AssertionError(
                f"DSpark ragged verify dropped out of CUDA graph replay on "
                f"{self.graph_eager} step(s); a ragged step without a captured "
                f"shape costs far more than the tokens it trims: {summary}")


def format_verify_len_histogram(lens: List[int]) -> str:
    """One-line ``len:count`` rendering, for per-step debug logging."""
    hist = Counter(int(v) for v in lens)
    return " ".join(f"{k}:{v}" for k, v in sorted(hist.items()))
