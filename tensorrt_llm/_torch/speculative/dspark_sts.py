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
"""STS calibration: collecting the data, and reading the fitted table.

The confidence head emits a raw logit per drafted position. ``apply_sts`` turns
it into a probability with a *per-position* temperature::

    confidence[r][j] = sigmoid(logit[r][j] / T[j])
    survival[r][j]   = prod_{i <= j} confidence[r][i]

``T`` defaults to all ones, which makes that a plain sigmoid -- i.e. no
calibration at all. That default is not neutral, because the planner does not
consume ``confidence``; it consumes the **cumulative product**. A per-position
bias of x compounds to x^(j+1) by position j, so a head that is 5% over-confident
per position over-states the depth-5 survival by more than 30%, and the budget
argmax divides by a cost while multiplying that error into its numerator.

So the table has to be fitted against the survival, not against the per-position
probability -- which is what makes the fitting objective (see
``tests/microbenchmarks/dspark_fit_sts.py``) a cumulative-product ECE rather than
a per-position one. The recorder here produces exactly the pairs that objective
needs.

Collection is opt-in via ``TLLM_DSPARK_STS_COLLECT_PATH`` and is **not** a
serving path: it reads the confidence buffer back to the host once per decode
step. That is a real synchronization, deliberately accepted because a
calibration run is not a throughput run.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch

from ..._utils import prefer_pinned
from ...logger import logger

__all__ = [
    "STS_COLLECT_ENV",
    "DSparkStsCalibration",
    "DSparkStsRecorder",
    "load_sts_temperatures_from_path",
    "make_recorder_from_env",
]

#: Set to a path stem to collect calibration data. Shards land at
#: ``<stem>.<rank>.<n>.pt``.
STS_COLLECT_ENV = "TLLM_DSPARK_STS_COLLECT_PATH"

#: Flush after this many recorded steps, so a killed run still leaves data.
_DEFAULT_FLUSH_EVERY = 64


@dataclass
class DSparkStsCalibration:
    """A fitted per-position temperature vector, plus how well it did."""

    temperatures: List[float]
    ece_before: List[float] = field(default_factory=list)
    ece_after: List[float] = field(default_factory=list)
    dataset: str = ""
    num_samples: int = 0

    def to_json(self) -> dict:
        # Both key spellings are written. ``sts_temperatures`` is what this repo
        # has always read; ``temperatures`` is what SGLang's DSparkStsCalibration
        # emits, so a table produced here is loadable there and vice versa.
        return {
            "sts_temperatures": list(self.temperatures),
            "temperatures": list(self.temperatures),
            "ece_before": list(self.ece_before),
            "ece_after": list(self.ece_after),
            "dataset": self.dataset,
            "num_samples": int(self.num_samples),
        }


def load_sts_temperatures_from_path(path: str) -> List[float]:
    """Read a temperature vector, accepting either spelling of the key.

    This repo writes ``sts_temperatures``; SGLang's ``DSparkStsCalibration``
    writes ``temperatures``. The vectors are interchangeable -- both are fitted
    against the cumulative product with the same sigmoid -- so refusing the
    other spelling would reject a usable table for no reason.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    for key in ("sts_temperatures", "temperatures"):
        if key in payload:
            temps = payload[key]
            break
    else:
        raise KeyError(
            f"{path} has neither 'sts_temperatures' nor 'temperatures'; "
            f"found keys {sorted(payload)}")
    if not temps:
        raise ValueError(f"{path} carries an empty temperature vector")
    return [float(t) for t in temps]


def resolve_confidence_head(draft_model):
    """Find the confidence head across the known draft-model layouts.

    Two layouts exist: the bare ``DSparkDraftModel`` (stages under
    ``.mtp_layers``, head on the last stage) and the ``DSparkForCausalLM``
    wrapper (bare model under ``.dspark_model``). A chained ``getattr`` with
    soft defaults conflated "wrapper" with "no head" and silently disabled
    calibration for an entire measurement campaign; this helper exists so the
    resolution is written once and unit-tested against both layouts.
    """
    inner = getattr(draft_model, "dspark_model", draft_model)
    stages = getattr(inner, "mtp_layers", None)
    if not stages:
        return None
    return getattr(stages[-1], "confidence_head", None)


class DSparkStsRecorder:
    """Pair each drafted block's logits with how much of it was accepted.

    The two halves live in different places and on different clocks: the
    logits sit in the worker's persistent slot-indexed device buffer, written
    by draft pass ``i``; the accepted count arrives in the sampler while pass
    ``i+1`` (or ``i+2``, under the overlap scheduler) has already overwritten
    that buffer. The join is therefore made on TWO axes, neither of which is
    wall-clock arrival:

    * **time** -- ``stage_snapshot`` keeps a small ring of host copies keyed
      by the worker's draft sequence counter, and ``record`` selects the ring
      entry whose per-row stamp equals the pass that drafted the block being
      verified;
    * **row** -- the caller resolves the request's buffer row through the
      worker's own allocator (``confidence_row_for``), never through
      ``py_seq_slot``, which belongs to a different allocator and only
      coincides until the first request completes.

    A pair that cannot be made exactly is dropped and counted in ``stats``
    rather than approximated: a mispaired shard is strictly worse than a
    smaller one, and looks identical downstream.
    """

    #: Ring depth. The pairing lag is one draft pass (two under the overlap
    #: scheduler); four snapshots is comfortably past both without letting a
    #: wrong target_seq silently match ancient content.
    _RING_DEPTH = 4

    def __init__(self, *, path_stem: str, block_size: int, rank: int = 0,
                 flush_every: int = _DEFAULT_FLUSH_EVERY) -> None:
        self.path_stem = str(path_stem)
        self.block_size = int(block_size)
        self.rank = int(rank)
        self.flush_every = max(1, int(flush_every))
        self._logits: List[torch.Tensor] = []
        self._accepted: List[int] = []
        self._shard = 0
        self._steps_since_flush = 0
        # Snapshot ring, indexed by staged_seq % _RING_DEPTH. Each entry is
        # (host_logits, host_stamps, cuda_event, staged_seq): a copy of the
        # whole slot-indexed confidence buffer plus the draft-pass stamp of
        # every row, taken on the host path each step -- the same relay the
        # planner trusts. `record` selects the entry whose stamp for the
        # request's row equals the pass that drafted the block being verified,
        # so the pairing is by draft-pass identity, not by wall-clock arrival:
        # the single mutable stash this replaces was overwritten by the very
        # forward that produced the label, and NO execution order paired it
        # correctly (off by one plain, off by two under overlap).
        self._ring: List[Optional[tuple]] = [None] * self._RING_DEPTH
        # Every decline is counted: this feature's failures are all silent,
        # and an empty or thin shard with healthy-looking stats was exactly
        # how the mispaired fit shipped.
        self.stats: dict = {"recorded": 0, "no_snapshot": 0,
                            "stale_stamp": 0, "row_out_of_range": 0,
                            "no_row": 0, "snapshots_staged": 0}

    def stage_snapshot(self, *, device_logits: torch.Tensor,
                       device_stamps: Optional[torch.Tensor],
                       staged_seq: Optional[int]) -> None:
        """Snapshot the confidence buffer + stamps, keyed by draft pass.

        Called from the executor's host path once per step (next to the
        planner's own staging), never from inside the captured graph: Python
        in a captured region runs at capture time only, which is how the old
        stash silently replayed stale rows on every graphed step.
        """
        if device_stamps is None or staged_seq is None:
            return
        idx = int(staged_seq) % self._RING_DEPTH
        entry = self._ring[idx]
        if (entry is None or entry[0].shape != device_logits.shape
                or entry[1].shape != device_stamps.shape):
            on_gpu = device_logits.is_cuda
            entry = (
                torch.empty(device_logits.shape, dtype=torch.float32,
                            device="cpu",
                            pin_memory=on_gpu and prefer_pinned()),
                torch.empty(device_stamps.shape, dtype=torch.int32,
                            device="cpu",
                            pin_memory=on_gpu and prefer_pinned()),
                torch.cuda.Event() if on_gpu else None,
                None,
            )
        entry[0].copy_(device_logits.detach().to(torch.float32),
                       non_blocking=True)
        entry[1].copy_(device_stamps, non_blocking=True)
        if entry[2] is not None:
            entry[2].record()
        self._ring[idx] = (entry[0], entry[1], entry[2], int(staged_seq))
        self.stats["snapshots_staged"] += 1

    def record(self, *, row: Optional[int], accepted: int,
               target_seq: Optional[int]) -> None:
        """Pair one request's accepted count with the logits that drafted it.

        Args:
            row: the request's row in the confidence buffer, resolved through
                the worker's own allocator (``confidence_row_for``). The
                previous join key -- ``py_seq_slot`` -- came from a DIFFERENT
                allocator that only coincides until the first request
                completes; every pair after roster churn was request A's
                label against request B's logits, and the shards looked
                healthy.
            accepted: drafted positions accepted, excluding the bonus token.
            target_seq: the draft pass that produced the block this label
                verifies, snapshotted into the SampleState at sampling time
                (the ``verify_lens_snapshot`` pattern; reading it live at
                update time rewinds wrong under the overlap scheduler).
        """
        if row is None:
            self.stats["no_row"] += 1
            return
        if target_seq is None:
            self.stats["no_snapshot"] += 1
            return
        entry = self._ring[int(target_seq) % self._RING_DEPTH]
        if entry is None:
            self.stats["no_snapshot"] += 1
            return
        logits_host, stamps_host, event, staged_seq = entry
        if not 0 <= int(row) < logits_host.shape[0]:
            self.stats["row_out_of_range"] += 1
            return
        if event is not None:
            event.synchronize()
        if int(stamps_host[int(row)]) != int(target_seq):
            # The buffer content for this row is from a different draft pass
            # than the one this label verifies -- the ring slot was reused, or
            # the row was never written for that pass. Refusing IS the fix:
            # appending it anyway is the mispairing this design replaces.
            self.stats["stale_stamp"] += 1
            return
        self._logits.append(logits_host[int(row)].clone())
        self._accepted.append(int(accepted))
        self.stats["recorded"] += 1

    def end_step(self) -> None:
        self._steps_since_flush += 1
        if self._steps_since_flush >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Write a shard. Cheap to call with nothing buffered."""
        self._steps_since_flush = 0
        if not self._logits:
            return
        logits = torch.stack(self._logits, dim=0)
        counts = torch.tensor(self._accepted, dtype=torch.int64).view(-1, 1)
        positions = torch.arange(self.block_size).view(1, -1)
        # prefix_mask[i, k] == 1 iff draft positions 0..k were ALL accepted,
        # which is the event whose probability `survival[k]` estimates. Any
        # other label (e.g. "position k accepted") would fit a different
        # quantity than the planner consumes.
        prefix_mask = (positions < counts).to(torch.float32)

        path = Path(f"{self.path_stem}.r{self.rank}.{self._shard}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        # The pairing marker lets the fitter reject shards from the pre-ring
        # recorder, whose pairs were mislabeled in ways the tensors themselves
        # cannot reveal. The running decline counters ride along so a thin
        # shard explains itself.
        payload = {
            "logits": logits,
            "prefix_mask": prefix_mask,
            "meta": {
                "pairing": "draft_seq_ring",
                "block_size": self.block_size,
                "stats": dict(self.stats),
            },
        }
        torch.save(payload, path)
        logger.info(
            f"DSpark STS: wrote {logits.shape[0]} samples to {path} "
            f"(stats={self.stats})")
        self._logits.clear()
        self._accepted.clear()
        self._shard += 1


def make_recorder_from_env(*, block_size: int, rank: int = 0,
                           has_cost_table: bool = False,
                           ragged_mode: Optional[str] = None
                           ) -> Optional[DSparkStsRecorder]:
    """Build a recorder iff ``TLLM_DSPARK_STS_COLLECT_PATH`` is set.

    Refuses the two regimes in which the collected pairs describe the scheduler
    rather than the head. Both are silent failures otherwise: the run completes,
    the shards look normal, and the fitted temperatures are wrong in a way no
    downstream check can see. SGLang guards the first of these next to its own
    collector (dspark_planner.py:100-108) and names the second in the sibling
    ConfidenceMetricsProbe ("padded verify rows corrupt the per-position prefix
    label", dspark_observability.py:688-697).
    """
    stem = os.environ.get(STS_COLLECT_ENV, "").strip()
    if not stem:
        return None

    forced = os.environ.get("TLLM_DSPARK_FORCE_VERIFY_LEN", "").strip()
    forced_full = False
    if forced:
        try:
            forced_full = int(forced) >= int(block_size)
        except ValueError:
            forced_full = False
    if has_cost_table and not forced_full:
        # A loaded table means the planner trims, and a trimmed window clips
        # `py_num_accepted_draft_tokens`: a request granted 2 positions can show
        # at most 2 accepted, however well the head predicted. Fitting that
        # calibrates the scheduler. Pinning the window to the full block makes
        # the label uncensored by construction.
        raise ValueError(
            f"{STS_COLLECT_ENV} is set while a cost table is loaded. The planner "
            f"will trim, and a trimmed window censors the acceptance label -- the "
            f"fit would describe the scheduler, not the confidence head. Either "
            f"collect without a cost table, or pin the window with "
            f"TLLM_DSPARK_FORCE_VERIFY_LEN=<block_size> (a pin below the "
            f"block still censors the label and is refused).")

    if (ragged_mode or "").lower() == "compact":
        # Compact packs the token axis and fills the bucket slack with padding
        # rows. Those rows carry positions that were never drafted for a real
        # request, so the per-position prefix label they contribute is not a
        # measurement of anything.
        raise ValueError(
            f"{STS_COLLECT_ENV} is set with ragged verify mode 'compact'. Padded "
            f"verify rows corrupt the per-position prefix label. Collect in "
            f"static mode (TLLM_DSPARK_RAGGED_VERIFY_MODE=static).")
    logger.warning(
        f"DSpark STS collection ON ({STS_COLLECT_ENV}={stem!r}). This adds one "
        f"device->host copy per decode step; it is a calibration mode, not a "
        f"serving configuration.")
    return DSparkStsRecorder(path_stem=stem, block_size=block_size, rank=rank)
