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
"""Opt-in size-aware FIFO fit gate for disaggregated generation-init admission.

A disaggregated generation server admits a received request (``DISAGG_GENERATION_INIT``) by
creating its KV cache and allocating pages for the whole prompt. The only utilization check
on that path is the resume gate of the fresh cache: it refuses when the GPU utilization of
some pool group is above ``max_util_for_resume`` *before* the request's own pages are
allocated. Admission therefore stops at a utilization that depends on the size of the last
admitted request, and a long prompt can push the pool well past the threshold.

``DisaggGenFitGate`` checks the request's own demand instead: a generation-init is refused
when, for any pool group, ``(used + pages the request needs) / total > U_hi``. Admitted
requests then fill the pool up to ``U_hi`` and never beyond it.

A size-aware check alone would favour short prompts: the scheduler skips a refused
generation-init and tries the next one, so a short prompt behind a refused long one would get
in first, every pass. The gate therefore keeps arrival order: once one generation-init is
refused in a scheduling pass, every later generation-init of that pass is refused too.
Generation requests are not affected. The next pass starts unblocked. Two exceptions:

* A generation-init that can never be allocated (some pool group needs more pages than it
  has, or a request needing more than ``U_hi`` of a group alone is refused while no other
  request holds GPU pages) does not block the pass: it would block every later one forever.
  It is logged once and keeps being retried, as without the gate.
* A pass that starts while a resident generation request has no active cache (it was
  suspended or gave its pages up) refuses every generation-init, so that the residents get
  their pages back before newer requests take them. ``used`` counts only pages that are
  neither free nor evictable, so without this a new request could be admitted into the pages
  of a suspended resident.

``U_hi`` is in ``[MIN_U_HI, 1 - MIN_RESERVE]``: right after an admission at least 1 % of
every pool group is free, for the decode growth of the residents and the CUDA-graph padding
extension (``extend_capacity_for_tokens``, at most one page per request and life cycle). The
reserve is not held back afterwards: decode growth of the residents can use it up, and the
padding extension still raises if it finds no page. The gate records the fewest free pages
of every pool group per log window, so a run shows how much of the margin it used.

Enabled by ``TLLM_KV_DISAGG_GEN_FIT_GATE=<U_hi>`` (unset, empty or 0: off, and nothing is
built). Pair it with ``kv_cache_config.max_util_for_resume`` at the same value: the resume
gate still runs after the fit gate and refuses above ``max_util_for_resume``.

``TLLM_KV_DISAGG_GEN_FIT_GATE_MIN_RESERVE=<share>`` (opt-in; unset or empty: ``MIN_RESERVE``)
lowers the reserve to ``share`` in ``[RESERVE_FLOOR, MIN_RESERVE]``, which raises the largest
accepted ``U_hi`` to ``1 - share``. Only the admission headroom changes; a run shows whether
the decode growth reached the smaller margin in the gate's ``window_min_free_pages`` and in
the resident suspends.
"""

import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Sequence, Set

from tensorrt_llm.logger import logger

FIT_GATE_ENV = "TLLM_KV_DISAGG_GEN_FIT_GATE"
# Opt-in override of the reserve below (``reserve_of``).
MIN_RESERVE_ENV = "TLLM_KV_DISAGG_GEN_FIT_GATE_MIN_RESERVE"
# Share of every pool group that an admission never hands out (default).
MIN_RESERVE = 0.01
MAX_U_HI = 1.0 - MIN_RESERVE
# Smallest reserve ``MIN_RESERVE_ENV`` accepts.
RESERVE_FLOOR = 0.002
# Below this, a request needing more than U_hi of a pool group alone (and therefore left to
# the resume gate and the allocation, see FitDecision.oversize) would be an ordinary one.
MIN_U_HI = 0.5
DEFAULT_LOG_EVERY_PASSES = 2048
LOG_PREFIX = "[disagg-gen-fit-gate]"
# Admissions that end above this utilization are counted: it is the default
# max_util_for_resume, so the counter shows how much the gate admits beyond the default gate.
REPORT_UTIL = 0.95
# Bound on the request ids remembered for the once-per-request never-admissible warning.
MAX_WARNED_REQUESTS = 4096

# Why the rest of a pass is blocked.
_BLOCK_REFUSED = "refused"
_BLOCK_RESIDENT = "resident"


def reserve_of(env: Optional[Mapping[str, str]] = None) -> float:
    """Return the reserve: ``MIN_RESERVE_ENV`` if set, else ``MIN_RESERVE``.

    Raises:
        ValueError: the value is not a number in ``[RESERVE_FLOOR, MIN_RESERVE]``.
    """
    env = os.environ if env is None else env
    raw = env.get(MIN_RESERVE_ENV, "").strip()
    if not raw:
        return MIN_RESERVE
    try:
        reserve = float(raw)
    except ValueError as e:
        raise ValueError(
            f"{MIN_RESERVE_ENV} must be a number in [{RESERVE_FLOOR}, {MIN_RESERVE}], got {raw!r}"
        ) from e
    if not math.isfinite(reserve) or not RESERVE_FLOOR <= reserve <= MIN_RESERVE:
        raise ValueError(
            f"{MIN_RESERVE_ENV} must be a number in [{RESERVE_FLOOR}, {MIN_RESERVE}], got {raw!r}"
        )
    return reserve


def parse_fit_gate(env: Optional[Mapping[str, str]] = None) -> Optional[float]:
    """Return the configured ``U_hi``, or None when the gate is off.

    Raises:
        ValueError: the value is not a number in ``[MIN_U_HI, 1 - reserve_of(env)]`` (or 0 for
            off), or the reserve override is invalid.
    """
    env = os.environ if env is None else env
    raw = env.get(FIT_GATE_ENV, "").strip()
    if not raw:
        return None
    reserve = reserve_of(env)
    max_u_hi = 1.0 - reserve
    try:
        u_hi = float(raw)
    except ValueError as e:
        raise ValueError(
            f"{FIT_GATE_ENV} must be a number in [{MIN_U_HI}, {max_u_hi}], got {raw!r}"
        ) from e
    if u_hi == 0.0:
        return None
    if not math.isfinite(u_hi) or u_hi < MIN_U_HI or u_hi > max_u_hi:
        raise ValueError(
            f"{FIT_GATE_ENV} must be a number in [{MIN_U_HI}, {max_u_hi}] (at least "
            f"{reserve:.1%} of every pool group stays free), got {raw!r}"
        )
    return u_hi


@dataclass(frozen=True)
class FitDecision:
    """Outcome of one fit check.

    Attributes:
        admit: Whether the request may be admitted.
        util_after: Largest ``(used + need) / total`` over the pool groups.
        pool_group: The pool group that ``util_after`` belongs to.
        oversize: The request alone needs more than ``U_hi`` of some pool group, so the fit gate
            can never admit it; it is left to the resume gate and the allocation instead.
        never_fits: The request alone needs more pages than some pool group has: no
            allocation can ever succeed (implies ``oversize``).
        nothing_held: No pool group had a used page when the check ran.
    """

    admit: bool
    util_after: float
    pool_group: int
    oversize: bool = False
    never_fits: bool = False
    nothing_held: bool = False

    @property
    def never_admissible(self) -> bool:
        """Whether a refusal after this check means the request can never be admitted."""
        return self.never_fits or (self.oversize and self.nothing_held)


class DisaggGenFitGate:
    """Size-aware fit check for generation-init requests, FIFO within a scheduling pass.

    The KV cache manager calls ``check`` with the GPU page counts of every pool group and
    ``note_admitted`` once the request's pages are allocated. The scheduler brackets each pass
    with ``begin_pass`` / ``end_pass``, reports every refused generation-init through
    ``note_refused`` and, while ``pass_blocked`` is set, refuses later generation-inits itself
    and reports them through ``note_fifo_blocked``.
    """

    def __init__(
        self,
        u_hi: float,
        name: str = "",
        log_every_passes: int = DEFAULT_LOG_EVERY_PASSES,
        reserve: float = MIN_RESERVE,
    ) -> None:
        if not RESERVE_FLOOR <= reserve <= MIN_RESERVE:
            raise ValueError(f"reserve must be in [{RESERVE_FLOOR}, {MIN_RESERVE}], got {reserve}")
        if not MIN_U_HI <= u_hi <= 1.0 - reserve:
            raise ValueError(f"u_hi must be in [{MIN_U_HI}, {1.0 - reserve}], got {u_hi}")
        self.u_hi = u_hi
        self.name = name
        self.log_every_passes = log_every_passes
        self.stats: Counter = Counter()
        self.max_util_after_admit = 0.0
        #: Free (neither used nor evictable) GPU pages per pool group, read once per pass by
        #: ``end_pass``; set by the KV cache manager. None: not recorded.
        self.free_probe: Optional[Callable[[], Sequence[int]]] = None
        self._window_min_free: Optional[List[int]] = None
        self._pass_blocked: Optional[str] = None
        self._pass_fifo_blocked = 0
        self._passes_since_log = 0
        # (request id, decision) of the last check, until note_admitted / note_refused.
        self._pending: Optional[tuple] = None
        self._warned: Set[object] = set()
        self._warned_suspend = False

    # ---- fit check ----

    def decide(self, used: Sequence[int], need: Sequence[int], total: Sequence[int]) -> FitDecision:
        """Decide without touching the counters.

        Args:
            used: Unavailable (neither free nor evictable) GPU pages per pool group.
            need: Pages the request's full prompt allocation takes per pool group.
            total: GPU pages per pool group.
        """
        if not len(used) == len(need) == len(total):
            raise ValueError(
                f"pool group counts disagree: used {len(used)}, need {len(need)}, "
                f"total {len(total)}"
            )
        util_after = 0.0
        worst = -1
        oversize = False
        never_fits = False
        for pool_group, (u, n, t) in enumerate(zip(used, need, total)):
            if t <= 0:
                continue
            if n > self.u_hi * t:
                oversize = True
            if n > t:
                never_fits = True
            util = (u + n) / t
            if util > util_after or worst < 0:
                util_after = util
                worst = pool_group
        nothing_held = all(u <= 0 for u in used)
        if oversize:
            return FitDecision(
                True,
                util_after,
                worst,
                oversize=True,
                never_fits=never_fits,
                nothing_held=nothing_held,
            )
        return FitDecision(util_after <= self.u_hi, util_after, worst, nothing_held=nothing_held)

    def check(
        self,
        used: Sequence[int],
        need: Sequence[int],
        total: Sequence[int],
        request_id: object = None,
    ) -> bool:
        """Decide and count; True lets the request on to its allocation."""
        decision = self.decide(used, need, total)
        self._pending = (request_id, decision)
        if decision.oversize:
            self.stats["oversize_passed"] += 1
            return True
        if not decision.admit:
            self.stats["fit_refused"] += 1
            return False
        self.stats["fit_passed"] += 1
        return True

    def note_admitted(self, request_id: object = None) -> None:
        """The request ``check`` let through got its pages: count the admission."""
        pending = self._take_pending(request_id)
        if pending is None:
            return
        self.stats["fit_admitted"] += 1
        if pending.oversize:
            self.stats["oversize_admitted"] += 1
            return
        if pending.util_after > REPORT_UTIL:
            self.stats["admitted_above_report_util"] += 1
        self.max_util_after_admit = max(self.max_util_after_admit, pending.util_after)

    def _take_pending(self, request_id: object) -> Optional[FitDecision]:
        pending, self._pending = self._pending, None
        if pending is None or pending[0] != request_id:
            return None
        return pending[1]

    # ---- FIFO within a scheduling pass ----

    def begin_pass(self, residents_without_cache: int = 0) -> None:
        """Start a scheduling pass.

        Args:
            residents_without_cache: Started generation requests without an active cache
                (suspended, or their pages given up). Any of them blocks every
                generation-init of the pass, so that they get their pages back first.
        """
        self._pass_fifo_blocked = 0
        self._pending = None
        if residents_without_cache > 0:
            self._pass_blocked = _BLOCK_RESIDENT
            self.stats["resident_blocked_passes"] += 1
        else:
            self._pass_blocked = None

    @property
    def pass_blocked(self) -> bool:
        """Whether later generation-inits of this pass are refused."""
        return self._pass_blocked is not None

    def note_refused(self, request_id: object = None) -> None:
        """A generation-init was refused this pass; refuse every later one.

        Unless the request can never be admitted (``FitDecision.never_admissible``): it is
        logged once and the pass stays open, so it cannot block every later request forever.
        """
        pending = self._take_pending(request_id)
        if pending is not None:
            if pending.never_admissible:
                self.stats["never_admissible"] += 1
                self._warn_never_admissible(request_id, pending)
                return
            # The fit check let it through; the index slots, the resume gate or the
            # allocation refused it.
            self.stats["refused_after_fit"] += 1
        if self._pass_blocked is None:
            self._pass_blocked = _BLOCK_REFUSED

    def _warn_never_admissible(self, request_id: object, decision: FitDecision) -> None:
        if request_id in self._warned:
            return
        if len(self._warned) >= MAX_WARNED_REQUESTS:
            self._warned.clear()
        self._warned.add(request_id)
        why = (
            "it needs more pages than a pool group has"
            if decision.never_fits
            else f"it alone needs more than U_hi={self.u_hi:.4f} of a pool group and was "
            "refused while no other request held GPU pages"
        )
        logger.warning(
            f"{LOG_PREFIX}{self.name} generation-init {request_id} can never be admitted ({why}; "
            f"pool group {decision.pool_group} at {decision.util_after:.4f} after it): it does "
            "not hold back later generation-inits"
        )

    def note_fifo_blocked(self) -> None:
        """A generation-init was refused because the pass is blocked."""
        self._pass_fifo_blocked += 1
        if self._pass_blocked == _BLOCK_RESIDENT:
            self.stats["resident_blocked"] += 1
        else:
            self.stats["fifo_blocked"] += 1

    def end_pass(self) -> None:
        """Close the pass and log the counters every ``log_every_passes`` passes."""
        self.stats["passes"] += 1
        if self._pass_fifo_blocked and self._pass_blocked == _BLOCK_REFUSED:
            self.stats["fifo_blocked_passes"] += 1
        self._pass_blocked = None
        self._pass_fifo_blocked = 0
        self._pending = None
        if self.free_probe is not None:
            self.observe_free(self.free_probe())
        if self.log_every_passes > 0:
            self._passes_since_log += 1
            if self._passes_since_log >= self.log_every_passes:
                self._passes_since_log = 0
                self.log_stats()

    # ---- resident generation requests ----

    def note_resident_suspended(self) -> None:
        """A started generation request's cache was suspended (its pages may leave the GPU)."""
        self.stats["resident_suspends"] += 1
        if not self._warned_suspend:
            self._warned_suspend = True
            logger.warning(
                f"{LOG_PREFIX}{self.name} a generation request's KV cache was suspended while "
                "the fit gate is on: its pages may move to a lower cache tier and back"
            )

    def note_resident_resumed(self, ok: bool) -> None:
        """A suspended generation request's cache was resumed (``ok``) or refused."""
        self.stats["resident_resumes" if ok else "resident_resume_refused"] += 1

    # ---- free-page margin ----

    def observe_free(self, free: Sequence[int]) -> None:
        """Record the free pages per pool group; the window minimum is logged."""
        free = [int(f) for f in free]
        if self._window_min_free is None or len(self._window_min_free) != len(free):
            self._window_min_free = free
        else:
            self._window_min_free = [min(a, b) for a, b in zip(self._window_min_free, free)]

    def log_stats(self) -> None:
        s = self.stats
        min_free = "n/a" if self._window_min_free is None else self._window_min_free
        logger.info(
            f"{LOG_PREFIX}{self.name} U_hi={self.u_hi:.4f} passes={s['passes']} "
            f"fit_passed={s['fit_passed']} fit_refused={s['fit_refused']} "
            f"fit_admitted={s['fit_admitted']} "
            f"fit_admitted_above_{REPORT_UTIL}={s['admitted_above_report_util']} "
            f"max_util_after_admit={self.max_util_after_admit:.4f} "
            f"refused_after_fit={s['refused_after_fit']} "
            f"fifo_blocked={s['fifo_blocked']} fifo_blocked_passes={s['fifo_blocked_passes']} "
            f"resident_blocked={s['resident_blocked']} "
            f"resident_blocked_passes={s['resident_blocked_passes']} "
            f"resident_suspends={s['resident_suspends']} "
            f"resident_resumes={s['resident_resumes']} "
            f"resident_resume_refused={s['resident_resume_refused']} "
            f"oversize_passed={s['oversize_passed']} oversize_admitted={s['oversize_admitted']} "
            f"never_admissible={s['never_admissible']} "
            f"window_min_free_pages={min_free}"
        )
        self._window_min_free = None
