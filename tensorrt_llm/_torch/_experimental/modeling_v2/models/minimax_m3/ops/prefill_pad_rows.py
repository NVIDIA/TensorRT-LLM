# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefill breakable-graph buckets above 4096 rows and their padding rows (K3), shared by the M3.1 targets.

A breakable prefill graph (BCG) step runs at its bucket's row count: every attention-DP rank pads its flattened tokens
to the bucket of the largest rank (engine/runners/common.py get_padding_params), and the pad rows (input id 0,
position 0) run every token-major op, the routed experts included. The MegaMoE local backend can send rows to no expert
(ids -1, weights 0) through ``set_pad_mask``; the dep2ctx route-stage seat hands a pending mask to its route-stage
kernel, the fused route-quant and the fallback staging take it too.

* ``DEP2CTX_BCG_GRID``: the buckets a target adds above 4096 (``RuntimeHooks.prefill_capture_extra_num_tokens``).
  A live count L pads to the smallest bucket >= L, and replay at the bucket runs the kernel configuration eager runs
  at L only when no configuration edge lies in (L, bucket]. The 1K grain keeps both sides of the 12288 edge
  (12287 and 12288) so no bucket above 4096 straddles an edge in ``DEP2CTX_CONFIG_EDGES`` (``band_straddles``). The
  M-keyed tactics dep2ctx does not pin (``DEP2CTX_UNPINNED_EDGES``) are straddled, so replay against eager at the live
  rows is not guaranteed bitwise there; the O2 oracle's off-bucket cells cover those live ranges.
* ``PrefillPadRows``: one persistent uint8 row mask of ``capacity`` (= max_num_tokens) rows, allocated once and never
  reallocated. ``begin`` writes ``mask[:rows] = arange(rows) >= live`` with one launch from the model's outer forward,
  which BCG never captures (the runner swaps only the decoder body), on every padded step and every graph replay. The
  MoE block hands ``mask[:rows]`` to its backend (``hand_over``) when the launch has exactly this forward's rows and the
  step is padded or being captured; under capture the kernels bake the view's pointer and read each replay's mask.

Eager steps that are not padded hand None (as before). The target owns the switches.
No new kernel: the writer is one ``torch.ge`` launch; the masking is the existing route-stage kernel.
"""

from __future__ import annotations

import functools
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch

from tensorrt_llm.logger import logger

# The dep2ctx buckets above the bucket list's 4096 (1K grain plus the 12287 / 12288 band edge pair).
DEP2CTX_BCG_GRID: Tuple[int, ...] = (
    5120,
    6144,
    7168,
    8192,
    9216,
    10240,
    11264,
    12287,
    12288,
    13312,
    14336,
    15360,
    16384,
)
# Row counts where a dep2ctx prefill step changes kernel configuration: the M2 cuBLASLt pin bands (1024-4095,
# 4096-12287, 12288-16384), the MegaMoE launch ladder's step from the 4096 rung to the full bucket, and C1's 12288.
DEP2CTX_CONFIG_EDGES: Tuple[int, ...] = (1024, 4096, 4097, 12288)
# Row counts where an M-keyed tactic dep2ctx does not pin changes above 4096: the native MXFP8 CUTLASS autotuner's
# large-M bands start at 6553 and 13106 (MXFP8GemmRunner._LARGE_M_BANDS; the cuBLASLt pins fall back to it when a
# band's self-check fails or a shape is unpinned), and the power-of-two tuning buckets (last_positive_power_of_2) step
# at 8192 and 16384. The grid straddles each one (live 6145-6552, 7169-8191, 12289-13105, 15361-16383), so a replay
# there may run another tactic than eager at the live rows: not guaranteed bitwise, decided by the O2 oracle.
DEP2CTX_UNPINNED_EDGES: Tuple[int, ...] = (6553, 8192, 13106, 16384)


def band_straddles(
    buckets: Sequence[int], edges: Sequence[int], floor: int = 0
) -> List[Tuple[int, int, int]]:
    """The (previous bucket, bucket, edge) triples where a live count in (previous, bucket] and the bucket itself fall
    on different sides of ``edge`` (a band starting at ``edge``). Only buckets above ``floor`` are checked."""
    out = []
    prev = 0
    for b in sorted({int(x) for x in buckets}):
        if b > floor:
            for e in edges:
                if prev + 1 < e <= b:
                    out.append((prev, b, int(e)))
        prev = b
    return out


class PrefillPadRows:
    """The current forward's padding rows as a persistent uint8 mask (nonzero = pad row)."""

    def __init__(
        self,
        capacity: int,
        enabled: bool = True,
        tag: str = "PREFILL_PAD_ROWS",
        in_graph: Callable[[], bool] = lambda: False,
    ):
        self.capacity = int(capacity)
        self.enabled = bool(enabled) and self.capacity > 0
        self.tag = tag
        # Whether the current call is inside a breakable graph capture or replay (is_in_breakable_cuda_graph). Only
        # such a capture takes the mask: another capture (a decode graph) records no mask write before its reads.
        self._in_graph = in_graph
        #: MoE blocks ``wire`` connected (log-only).
        self.layers = 0
        #: Host hand-offs of a mask (padded eager calls and captures; replays run no host code). Log-only.
        self.handed = 0
        #: Mask writes (padded steps and graph replays). Log-only.
        self.writes = 0
        self._iota: Optional[torch.Tensor] = None
        self._mask: Optional[torch.Tensor] = None
        self._rows = 0
        self._padded = False
        self._logged: set = set()

    @property
    def allocated(self) -> bool:
        return self._mask is not None

    def allocate(self, device) -> None:
        """Allocate the mask and its row index once, outside any capture. Idempotent."""
        if not self.enabled or self._mask is not None:
            return
        self._iota = torch.arange(self.capacity, dtype=torch.int32, device=device)
        self._mask = torch.zeros(self.capacity, dtype=torch.uint8, device=device)

    def begin(self, live: int, rows: int, *, padded: bool, graphed: bool, device=None) -> None:
        """Arm this forward's mask for ``rows`` rows of which the first ``live`` are real.

        The mask is written (one launch) when the step is padded or replays a graph (``graphed``: a replay reads the
        mask whatever its live count); a capture only needs the armed row count, since its kernels do not run."""
        self._rows, self._padded = 0, False
        rows = int(rows)
        if not self.enabled or not 0 < rows <= self.capacity:
            return
        if self._mask is None:
            if device is None or torch.cuda.is_current_stream_capturing():
                return
            self.allocate(device)
        if padded or graphed:
            torch.ge(self._iota[:rows], int(live), out=self._mask[:rows].view(torch.bool))
            self.writes += 1
        self._rows, self._padded = rows, bool(padded)

    def mask_for(self, rows: int, capturing: bool) -> Optional[torch.Tensor]:
        """``mask[:rows]`` for a MoE launch of exactly this forward's rows on a padded or captured step, else None."""
        if self._rows == 0 or int(rows) != self._rows or not (capturing or self._padded):
            return None
        self.handed += 1
        kind = "capture" if capturing else "padded eager"
        if kind not in self._logged:
            self._logged.add(kind)
            logger.info(f"{self.tag} handed to the routed experts ({kind}, rows {self._rows})")
        return self._mask[: self._rows]

    def hand_over(self, backend, max_rows: int, rows: int) -> None:
        """Give ``backend``'s next launch this forward's mask, or None (which also drops a mask a launch never took).

        ``max_rows`` > 0 is the MoE's chunk size: a chunked launch cannot take a whole-step mask. Only a breakable
        prefill capture takes the mask under capture: another capture (a decode graph) records no mask write. An
        unpadded eager call (the common eager case) decides without querying the stream."""
        if torch.compiler.is_compiling():
            return
        rows = int(rows)
        mask = None
        if self._rows and not (max_rows and rows > max_rows):
            if self._in_graph():
                mask = self.mask_for(rows, torch.cuda.is_current_stream_capturing())
            elif self._padded and not torch.cuda.is_current_stream_capturing():
                mask = self.mask_for(rows, False)
        backend.set_pad_mask(mask)

    def end(self) -> None:
        self._rows, self._padded = 0, False


def moe_backend(layer) -> Tuple[Optional[object], Optional[str]]:
    """(backend, None) when ``layer``'s MoE launch takes a pad mask, else (None, why not)."""
    moe = getattr(layer, "block_sparse_moe", None)
    if moe is None:
        return None, "no MoE"
    experts = getattr(moe, "experts", None)
    backend = getattr(experts, "backend", experts)
    if not hasattr(backend, "set_pad_mask"):
        return None, f"backend {type(backend).__name__} takes no pad mask"
    reason = backend.pad_mask_decline_reason()
    return (backend, None) if reason is None else (None, reason)


def wire(layers: Iterable, state: PrefillPadRows) -> Tuple[int, Optional[str]]:
    """Set ``pad_rows_hand_over`` on every MoE block whose backend takes a pad mask (idempotent; None when ``state`` is
    disabled). The block calls it with its input rows first thing in its forward, before its router seats the stage.
    Returns the wired count and, when a MoE layer is left out, the first such layer's reason."""
    wired, left_out = 0, None
    for layer in layers:
        backend, reason = moe_backend(layer)
        if backend is None:
            if reason != "no MoE" and left_out is None:
                left_out = reason
            continue
        moe = layer.block_sparse_moe
        if not state.enabled:
            moe.pad_rows_hand_over = None
            continue
        max_rows = int(getattr(moe.experts, "moe_max_num_tokens", 0) or 0)
        moe.pad_rows_hand_over = functools.partial(state.hand_over, backend, max_rows)
        wired += 1
    state.layers = wired
    return wired, left_out
