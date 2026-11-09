# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MuseSparkOpen static per-layer expert placement, and an expert-count census.

Static placement (``MUSESPARK_OPEN_EXPERT_PLACEMENT=<json>``, default off).
The experts of a layer are expert-parallel over the TP group: rank ``r`` holds the
experts in slots ``[r * 64, (r + 1) * 64)``, and by default slot ``s`` holds expert
``s``. With routing skew a layer's hottest experts can sit on one rank, which then
runs the most expert rows while the other ranks wait for it at the experts'
reduce-scatter (lockstep). The placement file names, per layer, which expert each
slot holds::

    {"layers": {"7": [e_0, e_1, ..., e_255], ...}}

where ``[e_0..e_255]`` is a permutation of the layer's 256 experts; a layer that is
not listed keeps the identity. Two things follow from it, and nothing else changes:

* the layer's experts module loads expert ``e_s`` into local slot ``s - 64 r``
  (``initial_local_expert_ids``, which the weight loader and every per-expert scale
  copy already index by), and
* the routing method maps every selected expert id ``e`` to its slot
  (``slot_of(e)``) after selection, so the expert kernels, the sort and the
  finalize see slot ids, exactly as they saw expert ids before. Selection itself
  (scores, bias, top-8 order and ties, weights) runs on the expert ids, unchanged.

Numerics: a permutation inside one rank's slots is bitwise (every token's top-8
slots, weights and their order are unchanged, the grouped GEMM rows are
independent and the finalize sums each token's slots in top-8 order). Moving an
expert to another rank changes which experts each rank's partial sum holds, so the
cross-rank reduction (reduce-scatter / all-reduce of BF16 partials) groups the terms
differently: a reduction-order change, gated with the accuracy pair.

Census (``MUSESPARK_OPEN_EXPERT_COUNTS=<dir>``, default off): every sequence-parallel
context step adds its shard's top-8 expert ids (before placement) into a per-layer
``[256]`` device counter (``index_add_``, graph-safe); a daemon thread writes the
counters to ``<dir>/expert_counts_rank<r>.json`` every 60 s. Summed over the ranks they
are the step's routing histogram, which the placement solver reads.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Dict, List, Optional

import torch

from tensorrt_llm.logger import logger

_PLACEMENT_ENV = "MUSESPARK_OPEN_EXPERT_PLACEMENT"
_COUNTS_ENV = "MUSESPARK_OPEN_EXPERT_COUNTS"
_placement_cache: Dict[str, Optional[Dict[int, List[int]]]] = {}


def placement_table(num_experts: int) -> Optional[Dict[int, List[int]]]:
    """``{layer: [expert id held by slot s]}`` from ``MUSESPARK_OPEN_EXPERT_PLACEMENT``, or None.

    Read once per process. Raises ValueError when a listed layer is not a
    permutation of ``range(num_experts)``.
    """
    path = os.getenv(_PLACEMENT_ENV, "")
    if not path:
        return None
    if path in _placement_cache:
        return _placement_cache[path]
    with open(path) as fh:
        raw = json.load(fh)
    table: Dict[int, List[int]] = {}
    for key, ids in raw["layers"].items():
        ids = [int(v) for v in ids]
        if sorted(ids) != list(range(num_experts)):
            raise ValueError(
                f"[musespark_open] {_PLACEMENT_ENV}: layer {key} is not a permutation of "
                f"{num_experts} experts"
            )
        if ids != list(range(num_experts)):
            table[int(key)] = ids
    _placement_cache[path] = table
    logger.info(
        f"[musespark_open] expert placement {path}: {len(table)} layers placed "
        f"(sha {raw.get('sha', '?')})"
    )
    return table


class SlotMap:
    """The ``expert id -> slot`` map of one layer, on the device of the first call."""

    def __init__(self, slot_expert: List[int]) -> None:
        slot_of = [0] * len(slot_expert)
        for slot, expert in enumerate(slot_expert):
            slot_of[expert] = slot
        self.cpu = torch.tensor(slot_of, dtype=torch.int32)
        self.dev: Optional[torch.Tensor] = None

    def __call__(self, selected: torch.Tensor) -> torch.Tensor:
        """``selected`` (int32 expert ids, any shape) mapped to slots, same shape and dtype."""
        dev = self.dev
        if dev is None or dev.device != selected.device:
            if selected.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "[musespark_open] expert placement: the slot map was first needed while "
                    "capturing a CUDA graph; it must be built on an eager step first"
                )
            dev = self.cpu.to(selected.device)
            self.dev = dev
        return dev.index_select(0, selected.reshape(-1)).view(selected.shape)


def apply_to_experts(experts, slot_expert: List[int], ep_rank: int, ep_size: int) -> List[int]:
    """Point the experts module (and its backend) at this rank's placed experts; returns them."""
    per = len(slot_expert) // ep_size
    local = list(slot_expert[ep_rank * per : (ep_rank + 1) * per])
    targets = [experts]
    backend = getattr(experts, "backend", None)
    if backend is not None and backend is not experts:
        targets.append(backend)
    for module in targets:
        if not hasattr(module, "initial_local_expert_ids"):
            continue
        if len(module.initial_local_expert_ids) != per:
            raise ValueError(
                f"[musespark_open] expert placement: {type(module).__name__} holds "
                f"{len(module.initial_local_expert_ids)} local experts, expected {per}"
            )
        if getattr(module, "layer_load_balancer", None) is not None:
            raise ValueError("[musespark_open] expert placement cannot run with the MoE load balancer")
        module.initial_local_expert_ids = list(local)
        module.initial_global_assignments = list(slot_expert)
    return local


# ---------------------------------------------------------------------------------- census
class _Census:
    def __init__(self, path: str) -> None:
        self.path = path
        self.counts: Dict[int, torch.Tensor] = {}
        self.ones: Optional[torch.Tensor] = None
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.device: Optional[torch.device] = None

    def add(self, layer: int, selected: torch.Tensor) -> None:
        counts = self.counts.get(layer)
        if counts is None:
            if torch.cuda.is_current_stream_capturing():
                return  # first seen while capturing: skip (the eager warmup creates it)
            with self.lock:
                counts = torch.zeros(256, dtype=torch.int64, device=selected.device)
                self.counts[layer] = counts
            self.device = selected.device
            self._start()
        flat = selected.reshape(-1).long()
        ones = self.ones
        if ones is None or ones.numel() < flat.numel():
            if torch.cuda.is_current_stream_capturing():
                ones = None
            else:
                ones = torch.ones(max(flat.numel(), 8192 * 8), dtype=torch.int64, device=selected.device)
                self.ones = ones
        if ones is None:
            return
        counts.index_add_(0, flat, ones[: flat.numel()])

    def _start(self) -> None:
        if self.thread is not None:
            return
        os.makedirs(self.path, exist_ok=True)
        self.thread = threading.Thread(target=self._loop, name="mso-expert-census", daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        torch.cuda.set_device(self.device)
        stream = torch.cuda.Stream(device=self.device)
        rank = int(os.getenv("OMPI_COMM_WORLD_RANK", os.getenv("RANK", "0")))
        out = os.path.join(self.path, f"expert_counts_rank{rank}.json")
        while True:
            time.sleep(60)
            try:
                with self.lock:
                    items = sorted(self.counts.items())
                with torch.cuda.stream(stream):
                    host = {k: v.to("cpu", non_blocking=True) for k, v in items}
                stream.synchronize()
                tmp = out + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump({"rank": rank, "time": time.time(),
                               "layers": {str(k): v.tolist() for k, v in host.items()}}, fh)
                os.replace(tmp, out)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[musespark_open] expert census write failed: {e}")


_census: Optional[_Census] = None
_census_checked = False


def census() -> Optional[_Census]:
    global _census, _census_checked
    if not _census_checked:
        _census_checked = True
        path = os.getenv(_COUNTS_ENV, "")
        if path:
            _census = _Census(path)
            logger.info(f"[musespark_open] expert census on: {path}")
    return _census
