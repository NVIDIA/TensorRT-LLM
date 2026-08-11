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
"""Verification-mode selection and per-step counters for DSpark ragged verify.

Failures on this path are silent (output stays correct at baseline accuracy),
so activity is counted rather than inferred. Modes: ``static`` verifies the full block; ``cap-accept`` computes and
commits per-request windows but submits the full block, so its output must
stay bit-identical to ``static``; ``compact`` submits only each request's
window and needs a profiled :class:`~.dspark_planner.SpsCostTable` -- with a
flat cost model the budget degenerates to verify-all.
"""

import os
from collections import Counter
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple

from ...logger import logger

__all__ = [
    "RaggedVerifyMode",
    "read_ragged_verify_mode",
    "DSparkRaggedStats",
]

#: Overrides ``DSparkDecodingConfig``. Read per call, not cached at import:
#: tests flip it with ``monkeypatch.setenv``.
RAGGED_VERIFY_MODE_ENV = "TLLM_DSPARK_RAGGED_VERIFY_MODE"


class RaggedVerifyMode(str, Enum):
    """How much of the drafted block reaches the target. See module docstring."""

    STATIC = "static"
    CAP_ACCEPT = "cap-accept"
    COMPACT = "compact"

    @property
    def computes_windows(self) -> bool:
        """Whether per-request windows are computed at all (``cap-accept``
        shares every host-side path with ``compact`` except the token-axis
        trim)."""
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
        ValueError: on an unrecognized value; a silent fallback to ``static``
            would turn a typo into "the feature never ran".
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

    ``enable_ragged_verify`` alone is not the answer: ``cap-accept`` sets it
    too but submits the FULL block, so batch-shaping code (capture set, graph
    key) must ask :attr:`RaggedVerifyMode.trims_submitted_tokens`. Must stay
    identical to the DSpark worker's own resolution.
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

    Host-only and cheap enough to leave on: no device reads, no
    synchronization.
    """

    def __init__(self, *, mode: RaggedVerifyMode, max_draft_len: int):
        self.mode = mode
        self.max_draft_len = int(max_draft_len)
        #: Planner decision counters, attached by the worker once the planner
        #: exists; the only record of WHY a step declined to trim
        #: (`fallback_flat_cost` = no profiled SPS table, so no trimming).
        self.planner_stats: Optional[Dict[str, int]] = None

        self.steps_total = 0
        #: Steps where per-request windows were produced AND differed; a step
        #: where every request got the same window counts as uniform.
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
        #: why the graph runner declined -> count. "peer_shape_mismatch" = one
        #: rank declined ragged and dragged the world eager; "key_not_captured"
        #: = wrong bucket grid; "peer_not_gen_only" = ordinary continuous
        #: batching.
        #: Steps whose batch held at least two generation requests; the gate's
        #: floor counts these, not all steps (concurrency ramps up).
        self.steps_multi_request = 0
        self.graph_miss_reasons: Dict[str, int] = {}
        #: the actual graph key, kept only for the reasons where it localizes
        #: the bug.
        self.graph_miss_shapes: Dict[str, int] = {}

        #: Acceptance, split by whether the request's window was shortened.
        self.accepted_tokens = 0
        self.requests_scored = 0
        #: Requests whose window was shorter than the full block.
        self.requests_trimmed = 0
        #: Trimmed requests that accepted their ENTIRE window: the draft was
        #: still alive at the cut, so the trim certainly cost acceptance (one
        #: that accepted fewer would have died on its own and cost nothing).
        self.trimmed_hit_ceiling = 0
        #: Positions the target accepted and the window then discarded. Only
        #: ``cap-accept`` scores these; under ``compact`` 0 means "unknowable",
        #: not "no loss".
        self.cap_trim_tokens = 0
        #: Kept per request because a broad mild trim and a concentrated one
        #: sum to the same total and mean opposite things about the planner.
        self.requests_cap_trimmed = 0
        self.cap_trim_max = 0
        self.cap_trim_hist: Counter = Counter()

        #: An eager step costs far more than the tokens the trim saved.
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
                block.
        """
        self.steps_total += 1
        # A batch of one is uniform by construction; counting it toward the
        # gate's floor would make the gate fire during ramp-up.
        if int(num_gen_requests) >= 2:
            self.steps_multi_request += 1
        # Ceiling = what a no-trim step would have submitted: each ROW sends
        # its bonus token plus the full block. ROWS, not requests: on a fitted
        # bucket `delivered` books `padded_bs * (tier + 1)` where `padded_bs`
        # is the CUDA-graph-rounded max row count ACROSS RANKS, so the ceiling
        # must count the same rows or trim_ratio goes negative. Only that case
        # rebases; every other path books `delivered` over the real rows, and
        # widening their ceiling would understate the trim.
        ceiling_rows = int(num_gen_requests)
        if bucket and delivered is None and padded_bs:
            ceiling_rows = max(ceiling_rows, int(padded_bs))
        self.ceiling_tokens += ceiling_rows * (1 + self.max_draft_len)

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
        submitted, or None on a replay; kept only for misses. Deliberately
        required, not defaulted: a default would make "probe not wired"
        indistinguishable from "no misses had shapes".
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

        :attr:`accept_len` plus this is the no-trim acceptance on the same
        drafts. Meaningful only under ``cap-accept``; elsewhere 0.0 means "not
        measured", not "no loss".
        """
        if not self.requests_scored:
            return 0.0
        return self.cap_trim_tokens / self.requests_scored

    @property
    def cap_trim_concentration(self) -> float:
        """Share of scored requests that lost anything at all. ``cap-accept``.

        Distinguishes a broad mild trim from the same mean loss concentrated
        on a few requests; only the second is a problem.
        """
        if not self.requests_scored:
            return 0.0
        return self.requests_cap_trimmed / self.requests_scored

    @property
    def trim_regret_rate(self) -> float:
        """Share of trimmed requests whose draft was still alive at the cut.

        0.0 means every trim was free (those drafts would have died anyway);
        toward 1.0 the trims are buying throughput with acceptance.
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
