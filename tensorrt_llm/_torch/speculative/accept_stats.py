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
"""Opt-in DFlash/DSpark acceptance-statistics recorder.

Collects, per rank, from the DFlash one-engine accept/reject site:

- a histogram of accepted-draft-token counts per target verify step
  (position-k acceptance rates and AL derive from it, see
  :func:`summarize_hist`),
- per-request step / accepted-draft totals,
- optional per-position (confidence, accepted) calibration counts for
  DSpark confidence-scheduled verification (binned; used to calibrate
  ``confidence_threshold`` and Sequential Temperature Scaling).

Default OFF and zero-overhead: :func:`maybe_create_recorder` returns
``None`` unless ``TLLM_DFLASH_ACCEPT_STATS_DIR`` is set, and the DFlash
worker only calls into the recorder when it exists. When enabled, the
recorder performs a small device->host copy per step, so it is intended
for measurement runs (eager mode), never perf-reference runs.

Environment variables:
    TLLM_DFLASH_ACCEPT_STATS_DIR: directory to write per-rank JSON stats
        (``dflash_accept_stats_rank{rank}.json``). Enables recording.
    TLLM_DFLASH_ACCEPT_STATS_FLUSH_EVERY: flush period in verify steps
        (default 50; also flushed at interpreter exit).

Calibration caveat: with ``confidence_threshold`` > 0 the runtime trims
low-confidence draft positions to the mask sentinel, which forces their
rejection — pairs recorded for trimmed positions are biased. Collect
calibration data with the confidence head present but
``confidence_threshold`` unset/0 so every drafted position is genuinely
verified.

Confidence provider: the recorder itself never computes confidences; it
owns an optional ``confidence_provider`` callable

    provider(draft_model, gen_hidden, first_prev_tokens, gen_draft_tokens)
        -> Optional[Sequence[Sequence[float]]]   # [num_gens][K] host rows

which the DFlash worker invokes at draft time (via
:meth:`DFlashAcceptStatsRecorder.record_draft_confidence`). With the
provider ``None`` (this MR), no calibration rows are collected and the
calibration table stays empty. The DSpark confidence-scheduled
verification MR ships ``tensorrt_llm._torch.speculative.dspark_confidence``
with the real provider (built on its ``compute_draft_confidence`` /
``build_confidence_prev_tokens`` helpers); when that module is importable,
:func:`maybe_create_recorder` picks it up automatically.

This module is import-light on purpose (no torch import) so the
aggregation math is unit-testable on CPU-only hosts.
"""

import atexit
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from tensorrt_llm.logger import logger

ENV_STATS_DIR = "TLLM_DFLASH_ACCEPT_STATS_DIR"
ENV_FLUSH_EVERY = "TLLM_DFLASH_ACCEPT_STATS_FLUSH_EVERY"
DEFAULT_NUM_CONF_BINS = 20
STATS_FILE_PREFIX = "dflash_accept_stats_rank"

# provider(draft_model, gen_hidden, first_prev_tokens, gen_draft_tokens)
#   -> per-request per-position confidence rows (host floats), or None to
#      skip this step (e.g. the drafter has no confidence head).
ConfidenceProvider = Callable[..., Optional[Sequence[Sequence[float]]]]


def _resolve_default_confidence_provider() -> Optional[ConfidenceProvider]:
    """Return the DSpark confidence provider if its module is in the tree.

    The module ships with the DSpark confidence-scheduled verification MR;
    without it (this MR alone) calibration rows are simply not collected.
    """
    try:
        from .dspark_confidence import dspark_confidence_provider
    except ImportError:
        return None
    return dspark_confidence_provider


def maybe_create_recorder(max_draft_len: int, rank: int) -> Optional["DFlashAcceptStatsRecorder"]:
    """Create a recorder iff TLLM_DFLASH_ACCEPT_STATS_DIR is set (else None)."""
    stats_dir = os.environ.get(ENV_STATS_DIR)
    if not stats_dir:
        return None
    logger.warning(
        "DFlash acceptance-stats recording is EXPERIMENTAL and specific to "
        "the DFlash/DSpark drafter: enabling %s with other speculative "
        "decoding methods records nothing and is unsupported.",
        ENV_STATS_DIR,
    )
    flush_every = int(os.environ.get(ENV_FLUSH_EVERY, "50"))
    return DFlashAcceptStatsRecorder(
        stats_dir,
        max_draft_len,
        rank,
        flush_every=flush_every,
        confidence_provider=_resolve_default_confidence_provider(),
    )


class DFlashAcceptStatsRecorder:
    """Accumulates DFlash acceptance statistics and dumps them to JSON.

    All inputs arrive as host lists (the caller does the single small
    ``.tolist()`` device->host copy); the accumulation itself is pure
    Python so it is unit-testable without a GPU.
    """

    # Request id 0 is never a real request in the py executor (ids are
    # assigned from 1): it is the id carried by executor-warmup dummies and
    # by the padding requests idle attention-DP ranks step in lock-step
    # with active ranks. Without this filter, idle DEP ranks
    # log hundreds of spurious all-K "acceptances" of padding blocks.
    DEFAULT_EXCLUDE_REQUEST_IDS = frozenset({0})

    def __init__(
        self,
        stats_dir: str,
        max_draft_len: int,
        rank: int,
        flush_every: int = 50,
        num_conf_bins: int = DEFAULT_NUM_CONF_BINS,
        exclude_request_ids=DEFAULT_EXCLUDE_REQUEST_IDS,
        confidence_provider: Optional[ConfidenceProvider] = None,
    ):
        os.makedirs(stats_dir, exist_ok=True)
        self.path = os.path.join(stats_dir, f"{STATS_FILE_PREFIX}{rank}.json")
        self.max_draft_len = max_draft_len
        self.rank = rank
        self.flush_every = max(1, flush_every)
        self.num_conf_bins = num_conf_bins
        self.exclude_request_ids = frozenset(int(r) for r in exclude_request_ids)
        # Optional confidence source (see module docstring); None means
        # no calibration rows are collected.
        self.confidence_provider = confidence_provider

        # hist[a] = number of verify steps that accepted exactly `a` draft
        # tokens (0..K); the bonus token is not counted here.
        self.hist: List[int] = [0] * (max_draft_len + 1)
        # request_id -> [verify_steps, accepted_draft_total]
        self.per_request: Dict[int, List[int]] = {}
        self.num_steps = 0
        # Confidence calibration, binned: position k (0-based), bin b covers
        # confidence in [b/nbins, (b+1)/nbins).
        self.calib_attempts = [[0] * num_conf_bins for _ in range(max_draft_len)]
        self.calib_accepted = [[0] * num_conf_bins for _ in range(max_draft_len)]
        # request_id -> per-position confidences of the in-flight draft block
        # (recorded at draft time, joined with the accept outcome next step).
        self._pending_conf: Dict[int, List[float]] = {}

        atexit.register(self.flush)

    # ---- recording -------------------------------------------------------

    def on_accept(self, request_ids: Sequence[int], num_accepted_tokens: Sequence[int]) -> None:
        """Record one verify step for the gen requests.

        Args:
            request_ids: gen-request ids, batch order.
            num_accepted_tokens: per gen request, bonus token + accepted
                draft tokens (the runtime's ``new_tokens_lens`` semantics).
        """
        K = self.max_draft_len
        for rid, n in zip(request_ids, num_accepted_tokens):
            if int(rid) in self.exclude_request_ids:
                continue  # executor-warmup / DEP-padding dummy
            accepted_draft = min(max(int(n) - 1, 0), K)
            self.hist[accepted_draft] += 1
            entry = self.per_request.setdefault(int(rid), [0, 0])
            entry[0] += 1
            entry[1] += accepted_draft
            conf = self._pending_conf.pop(int(rid), None)
            if conf is not None:
                for k, c in enumerate(conf[:K]):
                    b = _conf_bin(c, self.num_conf_bins)
                    self.calib_attempts[k][b] += 1
                    if accepted_draft > k:
                        self.calib_accepted[k][b] += 1
        self.num_steps += 1
        if self.num_steps % self.flush_every == 0:
            self.flush()

    def on_draft_confidence(
        self, request_ids: Sequence[int], confidence_rows: Sequence[Sequence[float]]
    ) -> None:
        """Stash per-position confidences of the block drafted this step.

        They are joined with the accept outcome when the same request is
        verified on the next step (dropped if the request finishes first).
        """
        for rid, row in zip(request_ids, confidence_rows):
            if int(rid) in self.exclude_request_ids:
                continue  # executor-warmup / DEP-padding dummy
            self._pending_conf[int(rid)] = [float(c) for c in row]

    def record_draft_confidence(
        self,
        request_ids: Sequence[int],
        draft_model,
        gen_hidden,
        first_prev_tokens,
        gen_draft_tokens,
    ) -> None:
        """Ask the confidence provider for this step's rows and stash them.

        No-op when no provider is configured or the provider declines
        (returns None). The provider arguments are opaque to the recorder;
        they are forwarded verbatim from the DFlash draft site.
        """
        if self.confidence_provider is None:
            return
        rows = self.confidence_provider(
            draft_model, gen_hidden, first_prev_tokens, gen_draft_tokens
        )
        if rows is None:
            return
        self.on_draft_confidence(request_ids, rows)

    # ---- output ----------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "max_draft_len": self.max_draft_len,
            "num_steps": self.num_steps,
            "accepted_draft_hist": list(self.hist),
            "per_request": {
                str(rid): {"steps": v[0], "accepted_draft": v[1]}
                for rid, v in self.per_request.items()
            },
            "confidence_calibration": {
                "num_bins": self.num_conf_bins,
                "attempts": [list(r) for r in self.calib_attempts],
                "accepted": [list(r) for r in self.calib_accepted],
            },
        }

    def flush(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.snapshot(), f)
        os.replace(tmp, self.path)


def _conf_bin(c: float, num_bins: int) -> int:
    return min(max(int(float(c) * num_bins), 0), num_bins - 1)


# ---- aggregation math (pure functions, CPU unit-tested) -------------------


def summarize_hist(hist: Sequence[int]) -> Dict[str, Any]:
    """Summary statistics from an accepted-draft-count histogram.

    Args:
        hist: hist[a] = verify steps that accepted exactly ``a`` draft
            tokens, a in 0..K.

    Returns dict with:
        num_steps, mean_accepted_draft,
        al: mean accepted tokens per target step INCLUDING the bonus token
            (mean_accepted_draft + 1) — the AL figure of merit,
        ar_per_position: [K] where entry k-1 = P(draft position k accepted)
            = P(accepted_draft >= k). Prefix acceptance makes this exact.
    """
    K = len(hist) - 1
    num_steps = sum(hist)
    if num_steps == 0:
        return {
            "num_steps": 0,
            "mean_accepted_draft": 0.0,
            "al": 0.0,
            "ar_per_position": [0.0] * K,
        }
    total_accepted = sum(a * n for a, n in enumerate(hist))
    ar = []
    tail = num_steps
    for k in range(1, K + 1):
        tail -= hist[k - 1]  # steps with accepted_draft < k drop out
        ar.append(tail / num_steps)
    return {
        "num_steps": num_steps,
        "mean_accepted_draft": total_accepted / num_steps,
        "al": total_accepted / num_steps + 1.0,
        "ar_per_position": ar,
    }


def merge_snapshots(snapshots: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge per-rank recorder snapshots (attention-DP: ranks hold disjoint
    request sets; TP-replicated ranks should pass a single rank's file)."""
    if not snapshots:
        raise ValueError("no snapshots to merge")
    K = snapshots[0]["max_draft_len"]
    nbins = snapshots[0]["confidence_calibration"]["num_bins"]
    merged = {
        "max_draft_len": K,
        "num_steps": 0,
        "accepted_draft_hist": [0] * (K + 1),
        "per_request": {},
        "confidence_calibration": {
            "num_bins": nbins,
            "attempts": [[0] * nbins for _ in range(K)],
            "accepted": [[0] * nbins for _ in range(K)],
        },
    }
    for s in snapshots:
        if s["max_draft_len"] != K:
            raise ValueError("mismatched max_draft_len across snapshots")
        if s["confidence_calibration"]["num_bins"] != nbins:
            raise ValueError("mismatched num_bins across snapshots")
        merged["num_steps"] += s["num_steps"]
        for a, n in enumerate(s["accepted_draft_hist"]):
            merged["accepted_draft_hist"][a] += n
        for rid, v in s["per_request"].items():
            entry = merged["per_request"].setdefault(rid, {"steps": 0, "accepted_draft": 0})
            entry["steps"] += v["steps"]
            entry["accepted_draft"] += v["accepted_draft"]
        cc = s["confidence_calibration"]
        for k in range(K):
            for b in range(nbins):
                merged["confidence_calibration"]["attempts"][k][b] += cc["attempts"][k][b]
                merged["confidence_calibration"]["accepted"][k][b] += cc["accepted"][k][b]
    return merged


def calibration_table(
    attempts: Sequence[Sequence[int]], accepted: Sequence[Sequence[int]]
) -> Dict[str, Any]:
    """Empirical acceptance rate per (position, confidence bin).

    Returns bin centers plus, per position, the empirical acceptance rate
    in each bin (None where the bin has no samples) and per-position
    expected calibration error (ECE, attempts-weighted |confidence -
    empirical acceptance|). A well-calibrated confidence head has the
    empirical rate tracking the bin center (ECE ~ 0); a monotone but
    shifted/scaled curve is what Sequential Temperature Scaling
    corrects.
    """
    K = len(attempts)
    nbins = len(attempts[0]) if K else 0
    centers = [(b + 0.5) / nbins for b in range(nbins)]
    per_position = []
    for k in range(K):
        rates: List[Optional[float]] = []
        ece_num = 0.0
        n_total = 0
        for b in range(nbins):
            n = attempts[k][b]
            if n == 0:
                rates.append(None)
                continue
            rate = accepted[k][b] / n
            rates.append(rate)
            ece_num += n * abs(centers[b] - rate)
            n_total += n
        per_position.append(
            {
                "empirical_acceptance": rates,
                "num_samples": list(attempts[k]),
                "ece": (ece_num / n_total) if n_total else None,
            }
        )
    return {"bin_centers": centers, "per_position": per_position}


def load_rank_snapshots(stats_dir: str) -> List[Dict[str, Any]]:
    """Load every per-rank stats JSON found in ``stats_dir``."""
    snaps = []
    for name in sorted(os.listdir(stats_dir)):
        if name.startswith(STATS_FILE_PREFIX) and name.endswith(".json"):
            with open(os.path.join(stats_dir, name)) as f:
                snaps.append(json.load(f))
    return snaps
