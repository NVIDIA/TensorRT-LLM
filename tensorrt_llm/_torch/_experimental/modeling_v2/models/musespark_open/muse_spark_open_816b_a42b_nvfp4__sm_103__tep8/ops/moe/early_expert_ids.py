# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The expert ids of a decode MoE step, published to the persistent expert kernel early (``early_expert_ids``).

``ops/comm/pool_gather_route`` (``moe_prologue``) computes each decode step's
routing and ring-buddy placement inside the pool gather + quantization kernel,
and ``ops/moe/persistent_experts`` streams the selected experts' weights.
Without this module the expert kernel reads the placed ids only after its
dependency wait, i.e. after the routed kernel has completed and flushed, so
its weight stream starts only after its launch, wait, ids load, routing and
first TMA.

With this module the routed kernel's first CTA also stores the placed ids
as 64 tagged words, ``(id, tag)`` in one 64-bit word each, as soon as its
routing warps have them, and the kernel triggers its dependents after its
logit exchange (after its own dependency wait in any case). The expert kernel,
launched early behind it, polls those words, routes, starts its weight
stream, and waits for the routed kernel only before it reads the
activations or writes anything (see ``_persistent_experts/kernel.py``).

The tag is the launch epoch of the expert kernel's workspace that the next
expert launch will use: the routed kernel reads the workspace's counter (CTA
0's word, after its own dependency wait, when the previous expert launch has
completed) and adds one, skipping 0. A word is therefore fresh only for the
expert launch that directly follows a publication, provided an expert launch
happens between any two publications; ``EarlyExpertIds`` enforces that on the
host:

* ``pool_gather_route`` publishes only when the buffer is not armed, then
  arms it with the call's ``local_ids`` tensor;
* ``persistent_experts`` runs its early variant only when it is called with
  exactly that tensor as its ids (the same routed call), and every launch on
  the workspace disarms the buffer.

All 64 words are rewritten by each publication (rows past the step's last
get id -1), so a stale word carries the previous publication's tag, which is
the epoch of a launch that has happened. The buffer starts zeroed (tag 0 is
never an epoch).

Values: the published ids are the placement the routed kernel also stores
in ``local_ids`` (the same expression), so the expert kernel's inputs are
unchanged bit for bit. If a word is not fresh within the deadline, the expert
kernel waits and reads ``local_ids`` instead (and counts a fallback).

The fallback counter lives in pinned host memory that the kernel adds to
directly (unified addressing), so a server can read it at any time without
synchronizing the device: ``fallbacks_seen`` (logged at process exit).

The routed kernel triggers its dependents after its logit exchange
(``TRIGGER``): earlier, the expert kernel would be resident while the
cross-rank exchange is still open and fall back (one deadline per layer)
whenever a peer rank's routed kernel starts late.
"""

from __future__ import annotations

from typing import List, Optional

import torch

#: Tagged words: one per (row, slot) of an 8-row step.
WORDS = 64

#: When the routed kernel triggers its dependents: after its own exit-time
#: triggers only (the built-in behaviour), after its logit exchange, or right
#: after its dependency wait.
TRIGGER_EXIT = 0
TRIGGER_EXCHANGE = 1
TRIGGER_ENTRY = 2
TRIGGERS = {"exit": TRIGGER_EXIT, "exchange": TRIGGER_EXCHANGE, "entry": TRIGGER_ENTRY}
#: The trigger point every buffer is created with.
TRIGGER = TRIGGER_EXCHANGE
#: How long the expert kernel polls for a fresh word before it reads ``local_ids``.
DEADLINE_NS = 100_000
#: The expert kernel's back-off between two polls.
POLL_NS = 32

STAMP_WORDS = 8
STAMP_CTAS = 4
#: Timeline stamps: a buffer whose ``stamps`` is a tensor (None by default) has the
#: routed kernel store global-timer stamps there, ``STAMP_WORDS`` per CTA.
#: Stamp slots of the routed kernel's CTA ``b`` at ``b * STAMP_WORDS + k``.
ST_ROUTE_WAITED = 0  # routing warps' dependency wait returned
ST_EXCHANGED = 1  # routing warps' logit exchange done (CTA barrier)
ST_PUBLISHED = 2  # CTA 0: the tagged ids stored
ST_ROUTE_DONE = 3  # routing warps' last store
ST_QUANT_DONE = 4  # quantization warps' last store

#: Every buffer created in this process (``fallbacks_seen``).
_BUFFERS: List["EarlyExpertIds"] = []


def fallbacks_seen() -> Optional[int]:
    """Polls that hit the deadline so far, over every buffer of this process; None if none.

    Reads the pinned host counters without synchronizing: launches still running
    may add to them later.
    """
    if not _BUFFERS:
        return None
    return sum(int(buffer.fallbacks[0]) for buffer in _BUFFERS)


class EarlyExpertIds:
    """One device's publication buffer, bound to one persistent-expert workspace.

    Create it with ``persistent_experts.early_expert_ids(device)`` (outside graph
    capture), which binds it to the workspace the kernel's launches use and
    registers it there.
    """

    def __init__(self, device: torch.device, epoch: torch.Tensor, workspace_key: tuple):
        self.device = device
        self.trigger = TRIGGER
        self.deadline_ns = DEADLINE_NS
        self.poll_ns = POLL_NS
        self.words = torch.zeros(WORDS, dtype=torch.int64, device=device)
        # Pinned host memory: the kernel's atomic add reaches it through unified
        # addressing (same pointer), and the host reads it without a device sync.
        self.fallbacks = torch.zeros(1, dtype=torch.int32, pin_memory=device.type == "cuda")
        #: The workspace's per-CTA launch counters (word 0 is the one read).
        self.epoch = epoch
        self.workspace_key = workspace_key
        self.stamps: Optional[torch.Tensor] = None
        #: The ``local_ids`` tensor of the routed call whose ids the words hold,
        #: until the next expert launch on the workspace; None when not armed.
        self.armed_for: Optional[torch.Tensor] = None
        _BUFFERS.append(self)

    # ---- addresses the kernels are compiled with ---------------------------

    @property
    def words_address(self) -> int:
        return self.words.data_ptr()

    @property
    def epoch_address(self) -> int:
        return self.epoch.data_ptr()

    @property
    def fallbacks_address(self) -> int:
        return self.fallbacks.data_ptr()

    @property
    def stamps_address(self) -> int:
        return 0 if self.stamps is None else self.stamps.data_ptr()

    # ---- host-side pairing -----------------------------------------------------

    def may_publish(self) -> bool:
        """Whether a routed call may publish now (no unconsumed publication)."""
        return self.armed_for is None

    def arm(self, local_ids: torch.Tensor) -> None:
        """Record that the words now hold ``local_ids``'s placement."""
        assert self.armed_for is None, "two publications without an expert launch between"
        self.armed_for = local_ids

    def take(self, topk_ids: torch.Tensor) -> bool:
        """An expert launch on the workspace: whether it may use the words; disarms."""
        early = self.armed_for is not None and self.armed_for is topk_ids
        self.armed_for = None
        return early

    def fallback_count(self) -> int:
        """Polls that hit the deadline so far, after the device's work so far (synchronizes)."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return int(self.fallbacks[0])


def launches_early(early: EarlyExpertIds, topk_ids: torch.Tensor) -> bool:
    """An expert launch on ``early``'s workspace with ``topk_ids``: whether it runs early.

    The one call site (``persistent_experts``) goes through this module function so
    that tests can count the early launches.
    """
    return early.take(topk_ids)
