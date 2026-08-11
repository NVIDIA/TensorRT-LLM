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

``apply_sts`` computes ``confidence[r][j] = sigmoid(logit[r][j] / T[j])``. The
planner consumes the cumulative product ``survival[r][j]``, in which
per-position calibration error compounds geometrically, so the table must be
fitted against the survival (see ``tests/microbenchmarks/dspark_fit_sts.py``).
Collection is opt-in via ``TLLM_DSPARK_STS_COLLECT_PATH`` and is not a serving
path: it reads the confidence buffer back to the host once per decode step.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import torch

from ..._utils import prefer_pinned
from ...logger import logger

__all__ = [
    "STS_COLLECT_ENV",
    "DSparkStsRecorder",
    "load_sts_temperatures_from_path",
    "make_recorder_from_env",
]

#: Set to a path stem to collect calibration data. Shards land at
#: ``<stem>.<rank>.<n>.pt``.
STS_COLLECT_ENV = "TLLM_DSPARK_STS_COLLECT_PATH"

#: Flush after this many recorded steps, so a killed run still leaves data.
_DEFAULT_FLUSH_EVERY = 64


def load_sts_temperatures_from_path(path: str) -> List[float]:
    """Read a temperature vector, accepting either spelling of the key.

    This repo writes ``sts_temperatures``; SGLang writes ``temperatures``.
    The vectors are interchangeable.
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
    wrapper (bare model under ``.dspark_model``). Written once and unit-tested
    against both; returns None when no head is found.
    """
    inner = getattr(draft_model, "dspark_model", draft_model)
    stages = getattr(inner, "mtp_layers", None)
    if not stages:
        return None
    return getattr(stages[-1], "confidence_head", None)


class DSparkStsRecorder:
    """Pair each drafted block's logits with how much of it was accepted.

    The confidence buffer is overwritten by later draft passes before the
    accepted count arrives, so the join is made on two axes, never wall-clock
    arrival: by draft-pass stamp (``stage_snapshot`` keeps a ring of host
    copies keyed by the worker's draft sequence counter) and by buffer row
    resolved through the worker's own allocator (``confidence_row_for`` --
    never ``py_seq_slot``, which belongs to a different allocator). A pair
    that cannot be made exactly is dropped and counted in ``stats``, never
    approximated.
    """

    #: Pairing lag is one draft pass (two under the overlap scheduler); four
    #: clears both without letting a wrong target_seq match ancient content.
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
        # (host_logits, host_stamps, cuda_event, staged_seq). `record` selects
        # the entry whose stamp for the request's row equals the pass that
        # drafted the block being verified: pairing is by draft-pass identity,
        # never wall-clock arrival.
        self._ring: List[Optional[tuple]] = [None] * self._RING_DEPTH
        # Every decline is counted; a thin shard must explain itself.
        self.stats: dict = {"recorded": 0, "no_snapshot": 0,
                            "stale_stamp": 0, "row_out_of_range": 0,
                            "no_row": 0, "snapshots_staged": 0}

    def stage_snapshot(self, *, device_logits: torch.Tensor,
                       device_stamps: Optional[torch.Tensor],
                       staged_seq: Optional[int]) -> None:
        """Snapshot the confidence buffer + stamps, keyed by draft pass.

        Must be called from the executor's host path once per step, never from
        inside the captured graph: Python in a captured region runs at capture
        time only.
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
                the worker's own allocator (``confidence_row_for``), never
                ``py_seq_slot``, which belongs to a different allocator.
            accepted: drafted positions accepted, excluding the bonus token.
            target_seq: the draft pass that produced the block this label
                verifies, snapshotted into the SampleState at sampling time
                (reading it live rewinds wrong under the overlap scheduler).
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
            # This row's content is from a different draft pass than the label
            # verifies; refusing is the point -- appending would mispair.
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
        # The pairing marker lets the fitter reject mispaired legacy shards;
        # the decline counters ride along so a thin shard explains itself.
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

    Refuses the two regimes in which the collected pairs describe the
    scheduler rather than the head; both would otherwise fail silently.
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
        raise ValueError(
            f"{STS_COLLECT_ENV} is set while a cost table is loaded. The planner "
            f"will trim, and a trimmed window censors the acceptance label -- the "
            f"fit would describe the scheduler, not the confidence head. Either "
            f"collect without a cost table, or pin the window with "
            f"TLLM_DSPARK_FORCE_VERIFY_LEN=<block_size> (a pin below the "
            f"block still censors the label and is refused).")

    if (ragged_mode or "").lower() == "compact":
        raise ValueError(
            f"{STS_COLLECT_ENV} is set with ragged verify mode 'compact'. Padded "
            f"verify rows corrupt the per-position prefix label. Collect in "
            f"static mode (TLLM_DSPARK_RAGGED_VERIFY_MODE=static).")

    if os.environ.get("TLLM_DSPARK_DEVICE_WINDOWS", "0") == "1":
        raise ValueError(
            f"{STS_COLLECT_ENV} is set with TLLM_DSPARK_DEVICE_WINDOWS=1. "
            f"Device-selected windows trim per request on device, so every "
            f"collected acceptance label is censored by a window the host "
            f"never sees. Collect with device windows off.")

    frac = os.environ.get("TLLM_DSPARK_FORCE_BUDGET_FRAC", "").strip()
    if frac:
        try:
            frac_full = float(frac) >= 1.0
        except ValueError:
            frac_full = False
        if not frac_full:
            raise ValueError(
                f"{STS_COLLECT_ENV} is set while TLLM_DSPARK_FORCE_BUDGET_FRAC="
                f"{frac!r} trims the verify budget; a trimmed window censors "
                f"the acceptance label for every collected sample. Collect "
                f"with the fraction unset or at 1.0.")
    logger.warning(
        f"DSpark STS collection ON ({STS_COLLECT_ENV}={stem!r}). This adds one "
        f"device->host copy per decode step; it is a calibration mode, not a "
        f"serving configuration.")
    return DSparkStsRecorder(path_stem=stem, block_size=block_size, rank=rank)
