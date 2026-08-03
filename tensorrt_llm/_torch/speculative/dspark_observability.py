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
from typing import Dict, List, Optional, Sequence, Tuple

from ...logger import logger

__all__ = [
    "RaggedVerifyMode",
    "read_ragged_verify_mode",
    "DSparkRaggedStats",
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


def resolve_ragged_verify_mode(spec_config) -> RaggedVerifyMode:
    """The mode a config plus the environment actually select.

    The config flag alone is not the answer, and using it as one is a bug with
    a long fuse: ``enable_ragged_verify`` is what ``cap-accept`` is configured
    with too, but cap-accept submits the FULL block. Anything that shapes the
    batch -- the CUDA-graph capture set, the graph key -- has to ask whether
    the token axis actually shrinks (:attr:`RaggedVerifyMode.trims_submitted_tokens`),
    not whether ragged was requested.

    Resolved identically to the DSpark worker's own resolution, and separately
    from it, because the capture set is built before any batch exists.
    """
    configured = (RaggedVerifyMode.COMPACT if getattr(
        spec_config, "enable_ragged_verify", False) else RaggedVerifyMode.STATIC)
    return read_ragged_verify_mode(default=configured)


def trims_submitted_tokens(spec_config) -> bool:
    """Whether this config+environment shrinks the token axis. See above."""
    if spec_config is None:
        return False
    return resolve_ragged_verify_mode(spec_config).trims_submitted_tokens


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
        #: The planner's own decision counters, attached by the worker once the
        #: planner exists (goal doc A6). They answer a question these counters
        #: cannot: *why* a step declined to trim. `fallback_flat_cost` in
        #: particular means the run had no profiled SPS table, under which the
        #: budget search is constructionally incapable of trimming -- so a run
        #: can look entirely healthy here and still have delivered nothing.
        self.planner_stats: Optional[Dict[str, int]] = None

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
        #: why the graph runner declined -> count. The reasons mean opposite
        #: things: "peer_shape_mismatch" is one rank declining to go ragged and
        #: dragging the world eager, "key_not_captured" is the bucket grid being
        #: wrong, "peer_not_gen_only" is ordinary continuous batching.
        #: Steps whose batch held at least two generation requests. The gate
        #: counts these rather than all steps: trtllm-bench ramps concurrency,
        #: so step 64 can still be a batch of two.
        self.steps_multi_request = 0
        self.graph_miss_reasons: Dict[str, int] = {}
        #: the actual graph key, kept only for the reasons where it localizes
        #: the bug.
        self.graph_miss_shapes: Dict[str, int] = {}

        #: Acceptance, split by whether the request's window was shortened.
        #: This is the term the trim/ceiling ratio needs and the one TRT-LLM
        #: did not have: `ceiling_tokens` above counts tokens *submitted* with
        #: no trim, which answers "how much compute did we save" and never
        #: "how much acceptance did we destroy".
        self.accepted_tokens = 0
        self.requests_scored = 0
        #: Requests whose window was shorter than the full block.
        self.requests_trimmed = 0
        #: Trimmed requests that accepted their ENTIRE window. Those are the
        #: ones where the draft was still alive at the cut, so trimming
        #: certainly cost acceptance -- a trimmed request that accepted fewer
        #: than its window would have died on its own and cost nothing. This is
        #: an exact, unbiased count, which is why it is preferred here over
        #: estimating the uncapped acceptance length: no model, no extrapolation
        #: from the untrimmed subpopulation (which the planner selected, so it
        #: is not a fair sample of the trimmed one).
        self.trimmed_hit_ceiling = 0
        #: Positions the target accepted and the window then discarded --
        #: the exact acceptance the schedule cost. Only ``cap-accept`` can
        #: populate this, because only there are those positions scored at all;
        #: it stays 0 under ``compact``, where the honest answer is that the
        #: number is unknowable without re-running.
        self.cap_trim_tokens = 0
        #: The distribution, not just the total. "Every request gave up a
        #: little" and "a few were cut to pieces" sum to the same
        #: cap_trim_tokens and mean opposite things about the planner: the
        #: first is a tuning knob, the second is the scheduler concentrating
        #: its whole cost on a subpopulation it selected. A total cannot tell
        #: them apart, which is the reason this is kept per request at all.
        self.requests_cap_trimmed = 0
        self.cap_trim_max = 0
        self.cap_trim_hist: Counter = Counter()

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
        delivered: Optional[int] = None,
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
            delivered: tokens actually handed to the target, when that is not
                derivable from ``verify_lens`` and ``bucket``. Required by
                ``cap-accept``, which schedules windows but submits the full
                block: deriving it there would report the windows as if they
                had been submitted, i.e. credit the mode with a compute saving
                it deliberately does not take, and ``trim_ratio`` would then
                describe a run that never happened.
        """
        self.steps_total += 1
        # Steps where raggedness is even definable. A window can only differ
        # *between* requests, so a batch of one is uniform by construction --
        # counting it toward the gate's floor makes the gate fire during ramp-up
        # and blame the scheduler for a batch that had not filled yet.
        if int(num_gen_requests) >= 2:
            self.steps_multi_request += 1
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
        # exactly the windows it asked for -- unless the caller says otherwise,
        # which is how cap-accept reports submitting the full block.
        if delivered is not None:
            self.delivered_tokens += int(delivered)
        else:
            self.delivered_tokens += int(bucket) if bucket else sum(1 + v
                                                                    for v in
                                                                    lens)
        if bucket:
            self.bucket_hist[int(bucket)] += 1
        if padded_bs:
            self.padded_bs_hist[int(padded_bs)] += 1

        if len(set(lens)) > 1:
            self.steps_ragged += 1
        else:
            self.steps_uniform_windows += 1

    def record_graph(self, *, replayed: bool, shape) -> None:
        """Record whether this step replayed a CUDA graph or ran eager.

        ``shape`` is the ``(num_rows, total_verify_tokens)`` the step actually
        submitted, or None on a replay. It is kept only for misses, and only
        the distinct ones: a miss means some shape was not captured, and
        knowing *which* is the whole diagnosis. Counting misses without
        recording the shape leaves only guessing or re-running with a profiler.

        Required rather than defaulted, deliberately. A run reported
        ``graph_eager: 9`` alongside an empty ``graph_miss_shapes`` because the
        caller was loaded from a build that predated this argument -- with a
        default that reads as "no misses had shapes", which is indistinguishable
        from "the probe works and found nothing". Without a default it is a
        TypeError instead.
        """
        if replayed:
            self.graph_replays += 1
            return
        self.graph_eager += 1
        if shape is not None:
            reason, key = shape
            self.graph_miss_reasons[reason] = (
                self.graph_miss_reasons.get(reason, 0) + 1)
            if key is not None:
                text = str(key)
                self.graph_miss_shapes[text] = (
                    self.graph_miss_shapes.get(text, 0) + 1)

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

    def record_acceptance(self,
                          *,
                          accepted: int,
                          window: int,
                          cap_trim: int = 0) -> None:
        """Record one request's acceptance against the window it was given.

        Args:
            accepted: drafted positions accepted (excludes the bonus token).
            window: drafted positions the request was allowed to verify.
            cap_trim: positions this request accepted that its window then
                discarded. Non-zero only under ``cap-accept``, the one mode
                that scores them; elsewhere 0 means "not measured", not "no
                loss".
        """
        self.requests_scored += 1
        self.accepted_tokens += int(accepted)
        if int(window) < self.max_draft_len:
            self.requests_trimmed += 1
            if int(accepted) >= int(window):
                self.trimmed_hit_ceiling += 1
        cap_trim = int(cap_trim)
        if cap_trim > 0:
            self.cap_trim_tokens += cap_trim
            self.requests_cap_trimmed += 1
            self.cap_trim_hist[cap_trim] += 1
            if cap_trim > self.cap_trim_max:
                self.cap_trim_max = cap_trim

    @property
    def accept_len(self) -> float:
        """Mean accepted drafted positions per request. 0.0 with no data."""
        if not self.requests_scored:
            return 0.0
        return self.accepted_tokens / self.requests_scored

    @property
    def accept_loss_per_request(self) -> float:
        """Mean accepted positions the schedule discarded, per request.

        The exact counterpart to :attr:`accept_len`: add the two and you get
        the acceptance a no-trim run would have achieved on the same drafts.
        Meaningful only under ``cap-accept``; 0.0 elsewhere, where it means
        "not measured", not "no loss" -- ``trim_regret_rate`` is the bound
        available under ``compact``.
        """
        if not self.requests_scored:
            return 0.0
        return self.cap_trim_tokens / self.requests_scored

    @property
    def cap_trim_concentration(self) -> float:
        """Share of scored requests that lost anything at all. ``cap-accept``.

        Read it together with :attr:`accept_loss_per_request`. The same mean
        loss at 0.9 concentration is a broad, mild trim; at 0.02 it is the
        scheduler paying for its throughput out of two percent of the traffic.
        Only the second is a problem, and the mean alone cannot see it.
        """
        if not self.requests_scored:
            return 0.0
        return self.requests_cap_trimmed / self.requests_scored

    @property
    def trim_regret_rate(self) -> float:
        """Share of trimmed requests whose draft was still alive at the cut.

        The operational reading: 0.0 means every trim was free -- those drafts
        would have died anyway -- and the scheduler is taking compute back for
        nothing. Climbing toward 1.0 means trimming is buying its throughput by
        throwing away acceptance, which is the regression a delivered-only
        acceptance metric cannot distinguish from a harder workload.
        """
        if not self.requests_trimmed:
            return 0.0
        return self.trimmed_hit_ceiling / self.requests_trimmed

    def summary(self) -> Dict[str, object]:
        """A JSON-safe snapshot, for logging or assertions."""
        return {
            "mode": self.mode.value,
            "steps_total": self.steps_total,
            "steps_ragged": self.steps_ragged,
            "steps_multi_request": self.steps_multi_request,
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
            "graph_miss_reasons": dict(sorted(self.graph_miss_reasons.items())),
            "graph_miss_keys": dict(sorted(self.graph_miss_shapes.items())),
            "accept_len": round(self.accept_len, 4),
            "requests_scored": self.requests_scored,
            "requests_trimmed": self.requests_trimmed,
            "trim_regret_rate": round(self.trim_regret_rate, 4),
            "cap_trim_tokens": self.cap_trim_tokens,
            "accept_loss_per_request": round(self.accept_loss_per_request, 4),
            "requests_cap_trimmed": self.requests_cap_trimmed,
            "cap_trim_max": self.cap_trim_max,
            "cap_trim_hist": dict(sorted(self.cap_trim_hist.items())),
            "cap_trim_concentration": round(self.cap_trim_concentration, 4),
            "graph_replays": self.graph_replays,
            "graph_eager": self.graph_eager,
            "fallbacks": dict(self.fallbacks),
            "planner": dict(self.planner_stats) if self.planner_stats else {},
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
        # `cap-accept` submits the full block on purpose -- it measures the
        # commitment policy, not a compute saving -- so a zero trim ratio is
        # the expected outcome there, not a silent no-op. Requiring one would
        # make the gate unsatisfiable in the one mode whose entire job is to be
        # the trusted reference the other modes are compared against.
        if (require_trim and self.mode.trims_submitted_tokens
                and self.trim_ratio <= 0.0):
            raise AssertionError(
                f"DSpark ragged verify delivered as many tokens as a no-trim "
                f"run, so nothing was saved: {summary}")
        # Only the misses ragged is responsible for. A batch some rank filled
        # with context requests falls out of replay on the uniform path too --
        # the uniform GSM8K run sat at ~1.7% eager for exactly that reason --
        # so failing on it makes this check unsatisfiable on any workload with
        # continuous batching, which is all of them. The three below are
        # different: "key_not_captured" and "bs_not_supported" mean the batch
        # was graphable, every rank agreed, and the shape was still missing --
        # i.e. the token bucket grid is wrong. "peer_shape_mismatch" means one
        # rank chose a different ragged bucket and dragged the whole world
        # eager, which is the attention-DP divergence this feature has to avoid.
        blamed = {"key_not_captured", "bs_not_supported", "peer_shape_mismatch"}
        attributable = {
            reason: n
            for reason, n in self.graph_miss_reasons.items() if reason in blamed
        }
        if attributable:
            raise AssertionError(
                f"DSpark ragged verify dropped out of CUDA graph replay for "
                f"reasons attributable to ragged scheduling "
                f"({attributable}); a ragged step without a captured shape "
                f"costs far more than the tokens it trims: {summary}")
        if self.graph_eager > 0 and not self.graph_miss_reasons:
            raise AssertionError(
                f"DSpark ragged verify dropped out of CUDA graph replay on "
                f"{self.graph_eager} step(s) but recorded no reason for any of "
                f"them, so the misses cannot be attributed. That is a broken "
                f"probe, not a clean run: {summary}")


def format_verify_len_histogram(lens: List[int]) -> str:
    """One-line ``len:count`` rendering, for per-step debug logging."""
    hist = Counter(int(v) for v in lens)
    return " ".join(f"{k}:{v}" for k, v in sorted(hist.items()))
