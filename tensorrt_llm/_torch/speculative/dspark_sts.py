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

    The two halves live in different places and are joined by sequence slot:
    the logits are in the worker's persistent ``[max_batch + 2, K]`` device
    buffer, written by the draft pass; the accepted count is host-side in the
    sampler. Both are indexed by ``py_seq_slot``, so the join is exact.

    One host copy per step, not per request. ``record`` is called once per
    request inside the sampler's loop, and copying a row each time would be a
    synchronization per request; instead the whole buffer is pulled once when
    the step's first request arrives and reused for the rest.
    """

    def __init__(self, *, path_stem: str, block_size: int, rank: int = 0,
                 flush_every: int = _DEFAULT_FLUSH_EVERY) -> None:
        self.path_stem = str(path_stem)
        self.block_size = int(block_size)
        self.rank = int(rank)
        self.flush_every = max(1, int(flush_every))
        self._logits: List[torch.Tensor] = []
        self._stashed_logits: List[torch.Tensor] = []
        self._accepted: List[int] = []
        self._stash_logits = None
        self._stash_slots = None
        self._stash_host = None
        self._shard = 0
        self._steps_since_flush = 0
        # Host mirror of the device buffer, refreshed once per step.
        self._host_logits: Optional[torch.Tensor] = None
        self._host_step = -1
        self._stash_capture_warned = False

    def stash_draft_confidence(self, *, logits, slots) -> None:
        """Keep what THIS step's draft produced, indexed by slot.

        The persistent buffer is shared and the sampler reads it later; under
        the overlap scheduler "later" can be after the next draft has written
        it. SGLang sidesteps this by recording the draft's own output rather
        than re-reading the buffer, and so does this once the diagnostic below
        confirms the two differ.
        """
        # DEVICE-side clone only. This runs inside the captured draft graph, so
        # a .to("cpu") here raises "Cannot copy between CPU and CUDA tensors
        # during CUDA graph capture" -- the same constraint that forces
        # _confidence_logits to be written in place. SGLang keeps a device
        # reference for exactly this reason. The host copy is deferred to the
        # sampler, which is outside the graph.
        #
        # EAGER-ONLY diagnostic: Python inside a captured region executes at
        # capture time only, so under graph replay these clones would keep
        # pointing at the last CAPTURE's tensors (dummy-slot content) and the
        # late-read comparison would diff against garbage while looking
        # perfectly healthy. Refuse to arm the stash during capture and say so
        # once, rather than let a replayed collection run silently measure the
        # wrong thing.
        if torch.cuda.is_current_stream_capturing():
            if not self._stash_capture_warned:
                self._stash_capture_warned = True
                logger.warning(
                    "DSpark STS stash skipped under CUDA-graph capture: the "
                    "late-read diagnostic is eager-only. Run collection with "
                    "cuda_graph_config disabled to exercise it.")
            self._stash_logits = None
            self._stash_slots = None
            self._stash_host = None
            return
        self._stash_logits = logits.detach().clone()
        self._stash_slots = slots.detach().clone()
        self._stash_host = None

    def begin_step(self, step: int) -> None:
        """Invalidate the host mirror; the next ``record`` refreshes it."""
        if step != self._host_step:
            self._host_logits = None
            self._stash_host = None
            self._host_step = step

    def record(self, *, slot: int, accepted: int,
               device_logits: torch.Tensor) -> None:
        """Record one request's ``(logits, accepted)`` pair.

        Args:
            slot: ``py_seq_slot``; indexes both the device buffer and nothing
                else, which is why it is the join key.
            accepted: drafted positions accepted, excluding the bonus token.
                This is exactly the ``num_correct_drafts`` the prefix mask is
                built from.
            device_logits: the worker's ``[rows, K]`` confidence buffer.
        """
        if self._host_logits is None:
            # The single sync of the step. Deliberate: see the module docstring.
            self._host_logits = device_logits.detach().to(
                device="cpu", dtype=torch.float32, copy=True)
        if slot < 0 or slot >= self._host_logits.shape[0]:
            return
        buffered = self._host_logits[slot].clone()
        self._logits.append(buffered)
        self._accepted.append(int(accepted))
        # Diagnostic: the same row as the draft produced it. If these two agree
        # everywhere, reading the shared buffer late is safe; if they diverge,
        # every pair recorded so far was confidence(t+k) against accepted(t).
        # Materialize the draft-time snapshot once per step, here, outside the
        # captured region.
        if self._stash_host is None and self._stash_logits is not None:
            rows = self._stash_logits.to(device="cpu", dtype=torch.float32,
                                         copy=True)
            idx = self._stash_slots.to(device="cpu").tolist()
            self._stash_host = {int(sl): rows[i] for i, sl in enumerate(idx)}
        stashed = (self._stash_host or {}).get(int(slot))
        self._stashed_logits.append(
            stashed.clone() if stashed is not None
            else torch.full_like(buffered, float("nan")))

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
        payload = {"logits": logits, "prefix_mask": prefix_mask}
        if self._stashed_logits:
            payload["logits_at_draft"] = torch.stack(self._stashed_logits, dim=0)
        torch.save(payload, path)
        logger.info(
            f"DSpark STS: wrote {logits.shape[0]} samples to {path}")
        self._logits.clear()
        self._stashed_logits.clear()
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
