# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 o_proj -> attention residual through a Lamport mailbox (opt-in).

``TRTLLM_K3_OPROJ_LAMPORT=1`` (SM107 with the CuTe DSL Rubin kernels, through ``kimi_k3_mk1_gate``)
replaces, at the post-attention site of eligible decoder layers of a pure generation batch of at
most ``TRTLLM_K3_OPROJ_LAMPORT_MAX_M`` tokens, the chain

    o_proj GEMM (row-parallel partial) -> all-reduce -> k3_attn_res(+prefix add)+RMSNorm

by

    o_proj GEMM whose epilogue publishes the partial into this rank's slot of a one-shot region
    [M][TP][7168] of a ``K3LamportArena`` -> k3_attn_res that sums the TP slots itself

(``cute_dsl_mxfp8_gemm_rubin_lamport``, ``run_k3_attn_res_slots``). The GEMM runs the stock
op's tactic (a split-K or mixed-cluster tactic, which has no mailbox epilogue, runs as the stock
GEMM and a copy kernel publishes its output), the sum is the one-shot MNNVL order and every
kernel keeps its arithmetic, so the layer outputs are bit-identical to the stock path. The
all-reduce kernel and its launch disappear; the attention-residual kernel polls the slots after
its ``griddepcontrol.wait``.

A layer is eligible when the attention-residual PDL kernel would run there anyway
(``TRTLLM_K3_ATTN_RES_PDL``, ``M <= TRTLLM_K3_ATTN_RES_PDL_MAX_M``, a supported candidate
count), its o_proj is the Rubin FP8 weight-read GEMM, and it is neither layer 0 nor a layer
whose pre-attention mix is a DSpark/DFlash tap. The decision depends only on the batch, the
layer and static settings, so it is the same on every rank; each forward takes its mailbox
parities from the arena in launch order, and every new batch size runs the plan check
(``check_mk1_plan``) before its first use.

Default ``TRTLLM_K3_OPROJ_LAMPORT_MAX_M``: ``min(64, 256 // TP)`` (64 at TP4, 32 at TP8; larger
batches keep the all-reduce). Above TP8 it is also at most the MNNVL all-reduce's one-shot range
(``TRTLLM_MNNVL_AR_ONESHOT_MAX_BYTES // (7168 * TP * 2)`` tokens, 13 at TP16 with 3 MiB): beyond it
that all-reduce runs two-shot, which sums the ranks in one chunk of 16 instead of two chunks of 8,
so the slots sum would no longer be bit-identical. Up to TP8 both forms sum in one rank chunk.

A forward that raises outside graph capture while it holds mailbox uses leaves the mailbox in an
unknown state (a peer may have published into this rank's copy for a use it never polled): the
arena then refuses every later forward, and the error is re-raised as :data:`FORWARD_FAILED`,
which the executor treats as immediately fatal, because CUDA-graph replays would not notice. A
dead or stalled peer leaves the polls spinning, unless ``TRTLLM_K3_LAMPORT_SPIN_LIMIT`` bounds them.
"""

from __future__ import annotations

import os
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np
import torch

from ...logger import logger
from .kimi_k3_mk1_gate import check_mk1_plan, mk1_enabled, mk1_plan_hash

OPROJ_LAMPORT_ENV = "TRTLLM_K3_OPROJ_LAMPORT"
#: The error of a forward that failed while it held mailbox uses (module docstring). "unrecoverable
#: error in the engine" is an immediate-fatal pattern of ``pyexecutor/error_classification.py``.
FORWARD_FAILED = (
    "Kimi K3 o_proj Lamport: unrecoverable error in the engine: a forward failed while it held "
    "mailbox uses, so the TP group's mailbox state is unknown"
)
OPROJ_LAMPORT_MAX_M_ENV = "TRTLLM_K3_OPROJ_LAMPORT_MAX_M"
#: Largest batch of the one-shot slots (the attn_res kernel's poll registers and the region).
MAX_M_LIMIT = 64
HIDDEN_SIZE = 7168
_SLOT_BYTES = HIDDEN_SIZE * 2


class K3LamportDelta(NamedTuple):
    """What an o_proj that published into the mailbox returns instead of its output tensor.

    ``mailbox`` is this rank's address of the region (``K3LamportArena.local_addr``), ``world``
    the TP size and ``m`` the rows every rank published.
    """

    mailbox: int
    world: int
    m: int


class K3LamportTarget:
    """One mailbox use: where an o_proj publishes and the handle it returns."""

    __slots__ = ("slot_address", "leading_dim", "delta", "published")

    def __init__(self, slot_address: int, leading_dim: int, delta: K3LamportDelta):
        self.slot_address = slot_address
        self.leading_dim = leading_dim
        self.delta = delta
        self.published = False

    def publish(
        self,
        act_fp8: torch.Tensor,
        act_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> K3LamportDelta:
        """Run the o_proj GEMM into this rank's slot (once) and return the handle."""
        from ..custom_ops.cute_dsl_custom_ops import cute_dsl_mxfp8_gemm_rubin_lamport

        if self.published:
            raise RuntimeError("an o_proj Lamport target was published twice in one use")
        if act_fp8.shape[0] != self.delta.m or weight.shape[0] != HIDDEN_SIZE:
            raise RuntimeError(
                f"o_proj Lamport publish of {tuple(act_fp8.shape[:1])} x {weight.shape[0]} "
                f"does not match the planned {self.delta.m} x {HIDDEN_SIZE}"
            )
        cute_dsl_mxfp8_gemm_rubin_lamport(
            act_fp8, weight, act_scale, weight_scale, self.slot_address, self.leading_dim
        )
        self.published = True
        return self.delta


#: Largest TP group whose MNNVL one-shot and two-shot all-reduce sum the ranks in the same order.
_SAME_ORDER_MAX_TP = 8


def _oneshot_tokens(tp_size: int) -> int:
    """Most tokens the MNNVL all-reduce of a TP group reduces one-shot."""
    from ..distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES

    return _MNNVL_ONE_SHOT_THRESHOLD_BYTES // (HIDDEN_SIZE * tp_size * 2)


def _max_m(tp_size: int) -> int:
    raw = os.environ.get(OPROJ_LAMPORT_MAX_M_ENV, "").strip()
    limit = MAX_M_LIMIT
    if tp_size > _SAME_ORDER_MAX_TP:
        limit = min(limit, _oneshot_tokens(tp_size))
    if not raw:
        return min(limit, 256 // tp_size)
    value = int(raw)
    if not 1 <= value <= MAX_M_LIMIT:
        raise ValueError(f"{OPROJ_LAMPORT_MAX_M_ENV} must be in [1, {MAX_M_LIMIT}], got {raw!r}")
    if value > limit:
        logger.warning(
            f"{OPROJ_LAMPORT_MAX_M_ENV}={value} is above the MNNVL one-shot range at TP{tp_size} "
            f"({limit} tokens; the two-shot all-reduce sums in another rank order): using {limit}."
        )
        return limit
    return value


class K3OprojLamport:
    """The TP group's arena and per-forward plan of the Lamport o_proj (module docstring)."""

    def __init__(self, mapping: Any, max_m: int):
        from ..distributed.k3_lamport_mailbox import DEFAULT_REGION_BYTES, K3LamportArena

        self.world = mapping.tp_size
        self.max_m = max_m
        need = max_m * self.world * _SLOT_BYTES
        # The default region size unless more is needed: users of the TP group's arena share it.
        self.arena = K3LamportArena.get(mapping, region_bytes=max(need, DEFAULT_REGION_BYTES))
        self.arena.require_fit(need, "the o_proj Lamport region")
        self._checked: set = set()
        logger.info(f"[K3 o_proj Lamport] TP{self.world}, up to {max_m} tokens, {need} B per use")

    @classmethod
    def create(cls, mapping: Any) -> Optional["K3OprojLamport"]:
        """The instance when ``TRTLLM_K3_OPROJ_LAMPORT=1`` applies, else None (collective: every
        rank of the model calls it at construction with the same settings)."""
        if not mk1_enabled(OPROJ_LAMPORT_ENV):
            return None
        from ..._utils import mpi_comm, mpi_disabled
        from ..distributed.k3_lamport_mailbox import k3_lamport_supported

        reason = None
        max_m = _max_m(mapping.tp_size)
        if mapping.tp_size < 2 or mapping.enable_attention_dp:
            reason = "it needs tensor-parallel attention (TP >= 2, no attention DP)"
        elif mapping.tp_size > 8 and mapping.tp_size != 16:
            # The one-shot sum order is one chunk up to 8 ranks and chunks of 8 above.
            reason = f"the slots sum supports TP 2 to 8 or 16, not TP{mapping.tp_size}"
        elif max_m < 1:
            reason = "the MNNVL all-reduce's one-shot range holds no token at this TP size"
        elif mpi_disabled():
            reason = "the mailbox arena needs MPI"
        elif not all(mpi_comm().allgather(k3_lamport_supported())):
            # Collective, so that no rank enters the arena allocation alone.
            reason = "multicast memory is unavailable on a rank (or TRTLLM_K3_LAMPORT_DISABLE=1)"
        if reason is not None:
            logger.warning(f"{OPROJ_LAMPORT_ENV}=1 is ignored: {reason}.")
            return None
        return cls(mapping, max_m)

    def plan(self, m: int, layer_sites: Sequence[int]) -> Optional["K3OprojLamportForward"]:
        """This forward's uses (``layer_sites``: indices of the eligible layers in launch order),
        or None when the batch keeps the all-reduce. Runs the plan check for a new batch size."""
        if m > self.max_m or len(layer_sites) < 2 or torch.compiler.is_compiling():
            return None
        key = (m, tuple(layer_sites))
        if key not in self._checked:
            plan_hash = mk1_plan_hash(
                m, 0, {"oproj_lamport": list(layer_sites), "world": self.world, "max_m": self.max_m}
            )
            comm = self.arena.comm
            check_mk1_plan(
                plan_hash,
                lambda values: np.asarray(comm.allgather(list(values)), dtype=np.int64),
                "the Kimi K3 o_proj Lamport sites",
            )
            self._checked.add(key)
        return K3OprojLamportForward(self, m, layer_sites)


class K3OprojLamportForward:
    """The mailbox uses of one forward, handed out in launch order."""

    def __init__(self, owner: K3OprojLamport, m: int, layer_sites: Sequence[int]):
        self.owner = owner
        self.m = m
        self.sites = frozenset(layer_sites)
        self.schedule = owner.arena.begin_forward(len(layer_sites))

    def take(self) -> K3LamportTarget:
        arena = self.owner.arena
        parity = self.schedule.take()
        world = self.owner.world
        delta = K3LamportDelta(arena.local_addr(parity), world, self.m)
        return K3LamportTarget(
            arena.mc_addr(parity) + arena.tp_rank * _SLOT_BYTES, world * HIDDEN_SIZE, delta
        )

    def finish(self) -> None:
        self.schedule.finish()

    def abort(self) -> bool:
        """Close this forward after an exception (module docstring). Returns True when the arena
        stays usable (under graph capture nothing ran), False when it was poisoned."""
        if torch.cuda.is_current_stream_capturing():
            self.schedule.cancel()
            return True
        self.owner.arena.poison("a forward raised while it held o_proj Lamport mailbox uses")
        return False


def run_attn_res_lamport(
    delta: K3LamportDelta,
    prefix_sum: Optional[torch.Tensor],
    block_residual: torch.Tensor,
    weights: Sequence[torch.Tensor],
    eps: float,
    out_eps: float,
    fp8_output: bool,
):
    """``(updated prefix, normed mix, (fp8 act, scale) or None)`` of the post-attention mix whose
    addend (``prefix_sum`` given) or layer row (``prefix_sum`` None) is the TP sum of the o_proj
    partials in ``delta``'s region (``run_k3_attn_res_slots``; with no prefix the summed row is
    the updated prefix)."""
    from ..cute_dsl_kernels.blackwell import k3_attn_res

    m = delta.m
    device = block_residual.device
    output = torch.empty(m, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    updated = torch.empty(m, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    act = scale = None
    if fp8_output:
        act = torch.empty(m, HIDDEN_SIZE, dtype=torch.float8_e4m3fn, device=device)
        scale = torch.empty(k3_attn_res.fp8_scale_numel(m), dtype=torch.uint8, device=device)
    k3_attn_res.run_k3_attn_res_slots(
        prefix_sum,
        delta.mailbox,
        delta.world,
        block_residual,
        *weights,
        eps,
        out_eps,
        output,
        updated,
        fp8_output=act,
        fp8_scale=scale,
    )
    return updated, output, (None if act is None else (act, scale))
