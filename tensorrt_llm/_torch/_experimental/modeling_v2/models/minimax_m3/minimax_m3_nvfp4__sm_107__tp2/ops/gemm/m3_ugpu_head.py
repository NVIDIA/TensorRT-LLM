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
"""The target lm_head shard GEMM of a verify step on a two-die GPU, each die reading only its own memory.

A copy of the head shard [N, K] (BF16) is placed once as one virtual range whose rows [0, S) live in locality domain 0
and rows [S, N) in domain 1 (:meth:`LocalityMap.place_split`; S on a 2 MiB granule). A covered verify computes
``F.linear`` of each row half of the copy on a green-context stream holding that domain's SMs, and concatenates: each
logit is the in-tree head GEMM's own logit (the same cuBLAS GEMM per output column) at the row counts the cells
certify. The module's own weight is never re-placed: a kernel reading die-placed memory from all SMs runs 1.5-1.7x
slower (SpecTail 769262), so every other reader (uncovered row counts, prefill logits, warmup, fallbacks) keeps the
ordinary weight, and only this op reads the placed copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.logger import logger

__all__ = ["UgpuHead", "m3_ugpu_head", "prepare_head"]

TARGET_SHARD = 100032  # the TP2 target head shard rows (vocabulary 200064 / 2)
HIDDEN = 6144
_ENGAGED: set = set()


@dataclass
class UgpuHead:
    """A placed head: ``weight`` the die-placed copy (read only by :data:`m3_ugpu_head`), ``source`` the module's own
    weight it copies and ``version`` that weight's version counter at the copy (a later in-place change makes the copy
    stale: see :func:`current`), ``split`` the first domain-1 row, ``streams`` the two domains' green-context
    streams."""

    weight: torch.Tensor
    source: torch.Tensor
    version: int
    split: int
    streams: List[torch.cuda.ExternalStream]


def current(head: Optional["UgpuHead"], weight: torch.Tensor) -> bool:
    """Whether ``head`` is a placed copy of ``weight`` as it is now (same tensor, not modified since the copy)."""
    return head is not None and head.source is weight and weight._version == head.version


def _split_row(n: int, row_bytes: int, granule: int) -> Optional[int]:
    """The granule-aligned row nearest the middle (the halves' bytes differ by < one granule), or None."""
    import math

    step = granule // math.gcd(granule, row_bytes)  # rows per granule-aligned boundary
    s = round(n / 2 / step) * step
    return s if 0 < s < n else None


def prepare_head(lm_head: torch.nn.Module) -> Optional[UgpuHead]:
    """A die-placed copy of ``lm_head.weight`` for :data:`m3_ugpu_head` (the module is not changed), or None when the
    head does not take the plain ``F.linear`` path this op reproduces, or the device has no two locality domains.
    Before capture."""
    from tensorrt_llm._torch.modules import low_m_gemm

    from ..m3_locality import LocalityMap

    w = getattr(lm_head, "weight", None)
    if (
        w is None
        or w.dtype != torch.bfloat16
        or w.dim() != 2
        or not w.is_contiguous()
        or getattr(lm_head, "bias", None) is not None
        or getattr(lm_head, "use_cute_dsl_bf16_gemm", False)
        or getattr(lm_head, "use_custom_cublas_mm", False)
        or low_m_gemm.LOW_M_GEMM_ACTIVE
        or type(getattr(lm_head, "quant_method", None)).__name__ != "UnquantizedLinearMethod"
    ):
        return None
    try:
        lm = LocalityMap.get(w.device)
    except (ImportError, RuntimeError) as err:
        logger.warning(f"[M3 sm_107 tp2] uGPU head: no locality map ({err}); the head stays as is")
        return None
    s = _split_row(w.shape[0], w.shape[1] * w.element_size(), lm.granularity)
    if s is None:
        return None
    with torch.no_grad():
        placed = lm.place_split(w.detach(), s)
    return UgpuHead(
        weight=placed, source=w, version=w._version, split=s, streams=lm.domain_streams()
    )


class M3UgpuHead(OpWrapper):
    """``logits`` [M, N] (BF16) = x [M, K] (BF16) times the placed head W [N, K] transposed, as
    ``cat(F.linear(x, W[:S]), F.linear(x, W[S:]))`` with the halves on the two domains' green-context streams
    (forked from and joined back into the caller's stream)."""

    ARCHS = frozenset({Arch.SM_107})

    # The verify rows of one to four requests at draft length 3 (the TP2 GEN batch is 1: every E2E step is 4 rows).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="4 verify rows: one request, draft length 3 (the b1 point)",
            spec=dict(m=4, n=TARGET_SHARD, k=HIDDEN),
        ),
        Cell(
            why="8 verify rows: two requests, draft length 3",
            spec=dict(m=8, n=TARGET_SHARD, k=HIDDEN),
        ),
        Cell(why="12 verify rows: three requests", spec=dict(m=12, n=TARGET_SHARD, k=HIDDEN)),
        Cell(
            why="16 verify rows: four requests (the c4 point)",
            spec=dict(m=16, n=TARGET_SHARD, k=HIDDEN),
        ),
    )

    note = """
    Each output column is computed by cuBLAS over the same weight row and the same input as the in-tree head GEMM; at
    the certified row counts the half-N call picks the full call's kernel, so the logits are the in-tree head's bit
    for bit (the cells check it on SM107). At 1 and 2 rows cuBLAS picks a different kernel for the half shard: not
    covered. Placement only moves pages between the dies; every other reader of the weight is unchanged.
    """

    def __call__(self, x: torch.Tensor, head: UgpuHead) -> torch.Tensor:
        key = (x.device, x.shape[0])
        if key not in _ENGAGED:
            _ENGAGED.add(key)
            logger.info(
                f"[M3 sm_107 tp2] fast path engaged: m3_ugpu_head ({x.shape[0]} rows, split {head.split} of "
                f"{head.weight.shape[0]}, on {x.device})"
            )
        main = torch.cuda.current_stream(x.device)
        fork = torch.cuda.Event()
        fork.record(main)
        parts = []
        for stream, w in zip(head.streams, (head.weight[: head.split], head.weight[head.split :])):
            stream.wait_event(fork)
            with torch.cuda.stream(stream):
                parts.append(F.linear(x, w))
        for stream in head.streams:
            done = torch.cuda.Event()
            done.record(stream)
            main.wait_event(done)
        for p in parts:
            p.record_stream(main)
        return torch.cat(parts, dim=1)

    def reference(self, x: torch.Tensor, head: UgpuHead) -> torch.Tensor:
        """The in-tree head GEMM over the whole (placed) weight."""
        return F.linear(x, head.weight)


m3_ugpu_head = M3UgpuHead()
