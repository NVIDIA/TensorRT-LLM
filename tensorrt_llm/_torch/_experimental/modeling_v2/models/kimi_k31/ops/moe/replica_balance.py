# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 per-layer dynamic EPLB: the sampled load of every MoE layer before and after its replica plan.

Under per-layer dynamic EPLB every routed chunk of every MoE layer is planned again (``ops/moe/routed_experts``): a
replica plan, one HALO-Q generation. :class:`ReplicaBalanceLog` measures what the plans do. On every
``TLLM_K31_EPLB_BALANCE_EVERY``-th forward (default 8; 0 records nothing) each rank counts, per MoE layer, where its
routed rows go: to the rank owning their logical expert (the load without EPLB, ``before``) and to the rank of the
physical slot the plan chose (``after``), and how many land on helper slots. Every :data:`WINDOW` sampled forwards the
counts of every rank are summed over the EP group (one host all-gather on the EPLB manager's EP communicator, at the
start of the next forward, after the counts were copied out without a wait) and EP rank 0 logs the window: the routed
rows, the busiest rank's load over the mean (``max / mean``) before and after, row-weighted and per layer-forward, and
the share of rows on helper slots. Every rank takes the same forwards and windows (the forwards of an attention-DP
group run in lockstep: the routed exchange is collective).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

#: Forwards between two sampled ones (``TLLM_K31_EPLB_BALANCE_EVERY``; 0 disables the record).
EVERY_ENV = "TLLM_K31_EPLB_BALANCE_EVERY"
DEFAULT_EVERY = 8
#: Sampled forwards per logged window.
WINDOW = 8
#: Layer-forwards with fewer routed tokens than this are left out of the per layer-forward percentiles (a near-empty
#: step's ratio says nothing about the MoE time).
MIN_TOKENS = 1024


def sampling_stride() -> int:
    """``TLLM_K31_EPLB_BALANCE_EVERY`` as a non-negative int (default :data:`DEFAULT_EVERY`)."""
    value = os.environ.get(EVERY_ENV, "").strip()
    if not value:
        return DEFAULT_EVERY
    try:
        every = int(value)
    except ValueError as error:
        raise ValueError(f"{EVERY_ENV}={value!r} is not an integer") from error
    if every < 0:
        raise ValueError(f"{EVERY_ENV}={value!r} must be 0 (off) or positive")
    return every


def balance_summary(counts: np.ndarray, ep_size: int, top_k: int) -> Dict[str, Any]:
    """The window statistics of the group's counts ``[samples, layers, 2 * ep_size + 1]`` (per sampled forward and
    layer: the routed rows per rank before the plan, after it, and the rows on helper slots; a token routes ``top_k``
    rows).

    Returns the routed rows, the layer-forwards with rows, the row-weighted ``sum(max) / sum(mean)`` before and after,
    their sums (``max_before``, ``max_after``, ``mean``: they add across windows), the helper share and, over the
    layer-forwards of at least :data:`MIN_TOKENS` tokens, the p50 / p90 / max of ``max / mean`` before and after."""
    counts = np.asarray(counts, dtype=np.int64)
    before = counts[..., :ep_size]
    after = counts[..., ep_size : 2 * ep_size]
    helper = counts[..., 2 * ep_size]
    rows = before.sum(-1)
    used = rows > 0
    mean = rows / ep_size
    max_before = before.max(-1)
    max_after = after.max(-1)
    summary: Dict[str, Any] = dict(
        rows=int(rows.sum()),
        layer_forwards=int(used.sum()),
        max_before=float(max_before[used].sum()),
        max_after=float(max_after[used].sum()),
        mean=float(mean[used].sum()),
        helper_rows=int(helper.sum()),
    )
    summary["before"] = summary["max_before"] / summary["mean"] if summary["mean"] else 0.0
    summary["after"] = summary["max_after"] / summary["mean"] if summary["mean"] else 0.0
    summary["helper_share"] = summary["helper_rows"] / summary["rows"] if summary["rows"] else 0.0
    big = rows >= MIN_TOKENS * top_k
    summary["large_layer_forwards"] = int(big.sum())
    for name, top in (("before", max_before), ("after", max_after)):
        ratio = top[big] / mean[big] if big.any() else np.zeros(0)
        summary[f"{name}_p50"] = float(np.percentile(ratio, 50)) if ratio.size else 0.0
        summary[f"{name}_p90"] = float(np.percentile(ratio, 90)) if ratio.size else 0.0
        summary[f"{name}_max"] = float(ratio.max()) if ratio.size else 0.0
    return summary


class ReplicaBalanceLog:
    """The sampled per-layer load record of a model's per-layer dynamic EPLB layers; see the module docstring.

    ``num_layers``: the model's EPLB layers (their handles' ``layer_idx`` index the record); ``home_experts`` /
    ``compute_slots``: a rank's resident experts and its resident + helper slots; ``top_k``: the routed rows of a
    token; ``ep_comm``: returns the EP group's communicator (``allgather``), called at the first flush. The model
    calls :meth:`begin_forward` / :meth:`end_forward` around every forward and the layers :meth:`record` after each
    chunk's dispatch.
    """

    def __init__(
        self,
        *,
        ep_rank: int,
        ep_size: int,
        home_experts: int,
        compute_slots: int,
        num_layers: int,
        top_k: int,
        ep_comm: Callable[[], Any],
        every: Optional[int] = None,
        window: int = WINDOW,
    ) -> None:
        self.top_k = int(top_k)
        self.ep_rank = int(ep_rank)
        self.ep_size = int(ep_size)
        self.home_experts = int(home_experts)
        self.compute_slots = int(compute_slots)
        self.num_layers = int(num_layers)
        self.every = sampling_stride() if every is None else int(every)
        self.window = int(window)
        self._ep_comm = ep_comm
        self._comm = None
        #: Forwards started, replica plans recorded (every chunk of every layer: the HALO-Q generations of this rank).
        self.forwards = 0
        self.plans = 0
        self._sampled = False
        self._sample = 0
        self._counts: Optional[torch.Tensor] = None
        self._ones: Optional[torch.Tensor] = None
        self._host: Optional[torch.Tensor] = None
        self._pending = None
        #: Summed over every flushed window (EP rank 0): what the per-window lines add up to.
        self.totals: Dict[str, float] = dict(max_before=0.0, max_after=0.0, mean=0.0, rows=0)

    @property
    def enabled(self) -> bool:
        return self.every > 0

    def begin_forward(self) -> None:
        """Start a forward: flush a window whose counts the previous forwards copied out, then decide whether this
        forward is sampled (the same forwards on every rank)."""
        if not self.enabled:
            return
        if self._pending is not None:
            self._flush()
        if self._counts is None:
            self._counts = torch.zeros(
                (self.window, self.num_layers, 2 * self.ep_size + 1),
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
        self._sampled = self.forwards % self.every == 0
        self.forwards += 1

    def record(self, layer: int, logical: torch.Tensor, physical: torch.Tensor) -> None:
        """Count one chunk of EPLB layer ``layer``: its logical routes ``[rows, top_k]`` (expert ids) and their
        physical slots (same shape; read after the plan's route wait). Queued on the current stream, no host wait."""
        self.plans += 1
        if not self._sampled or logical.numel() == 0:
            return
        n = logical.numel()
        if self._ones is None or self._ones.numel() < n:
            self._ones = torch.ones(n, dtype=torch.int32, device=logical.device)
        ones = self._ones[:n]
        ep = self.ep_size
        row = self._counts[self._sample, layer]
        row[:ep].index_add_(
            0, torch.div(logical.reshape(-1), self.home_experts, rounding_mode="floor").long(), ones
        )
        slots = physical.reshape(-1)
        rank = torch.div(slots, self.compute_slots, rounding_mode="floor")
        row[ep : 2 * ep].index_add_(0, rank.long(), ones)
        on_helper = (slots - rank * self.compute_slots) >= self.home_experts
        row[2 * ep :].add_(on_helper.sum(dtype=torch.int32))

    def end_forward(self) -> None:
        """End a forward; after the window's last sampled forward copy its counts out (no wait) and clear them."""
        if not self._sampled:
            return
        self._sampled = False
        self._sample += 1
        if self._sample < self.window:
            return
        self._sample = 0
        if self._host is None:
            self._host = torch.empty(
                self._counts.shape, dtype=torch.int32, pin_memory=prefer_pinned()
            )
        self._host.copy_(self._counts, non_blocking=True)
        done = torch.cuda.Event()
        done.record()
        self._counts.zero_()
        self._pending = (done, self.forwards, self.plans)

    def _flush(self) -> None:
        """Sum the copied window over the EP group (collective: every rank flushes the same window at the start of
        the same forward) and log it on EP rank 0."""
        done, forwards, plans = self._pending
        self._pending = None
        done.synchronize()
        local = self._host.numpy().copy()
        if self._comm is None:
            self._comm = self._ep_comm()
        gathered = self._comm.allgather(local)
        if self.ep_rank != 0:
            return
        summary = balance_summary(np.sum(np.stack(gathered), axis=0), self.ep_size, self.top_k)
        for key in self.totals:
            self.totals[key] += summary[key]
        totals = self.totals
        overall_before = totals["max_before"] / totals["mean"] if totals["mean"] else 0.0
        overall_after = totals["max_after"] / totals["mean"] if totals["mean"] else 0.0
        logger.info(
            f"Kimi K3.1 per-layer dynamic EPLB: {forwards} forwards, {plans} replica plans (HALO-Q generations "
            f"of {self.num_layers} layers) on this rank; window of {self.window} forwards sampled every "
            f"{self.every}: {summary['rows']} routed rows over {summary['layer_forwards']} layer-forwards, rank load "
            f"max/mean before {summary['before']:.3f} after {summary['after']:.3f} (row-weighted; "
            f"{summary['large_layer_forwards']} layer-forwards of >= {MIN_TOKENS} tokens: p50/p90/max before "
            f"{summary['before_p50']:.3f}/{summary['before_p90']:.3f}/{summary['before_max']:.3f} after "
            f"{summary['after_p50']:.3f}/{summary['after_p90']:.3f}/{summary['after_max']:.3f}), "
            f"{100 * summary['helper_share']:.1f}% of the rows on helper slots; since start before "
            f"{overall_before:.3f} after {overall_after:.3f} [sums max_before={summary['max_before']:.0f} "
            f"max_after={summary['max_after']:.0f} mean={summary['mean']:.1f} rows={summary['rows']}]"
        )


__all__ = [
    "EVERY_ENV",
    "MIN_TOKENS",
    "ReplicaBalanceLog",
    "WINDOW",
    "balance_summary",
    "sampling_stride",
]
