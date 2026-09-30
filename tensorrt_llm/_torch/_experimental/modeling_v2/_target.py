# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What runs a step, separated from what a model statically is.

`ModelingV2Core` carries the static structure -- routing identity, weights,
parallel and quantization configuration. A `Target` carries one way of running
a forward over that structure. A core has two, and the runtime picks between
them per call.

The split exists because the two phases are not one computation with different
arguments. On MLA the generation path works in latent space (absorption) while
the context path materializes K and V; `TrtllmAttention` refuses `mixed` for
MLA outright, which is what forces the choice up into Python. gpt_oss is not
MLA and so has no such branch -- this layer lands there first for the mechanism,
not for a speedup.

Two rules the base class exists to hold:

  A target reads the core, it does not copy it. The core replaces `_rope`
  when a contract check finds the table too short for the engine's admitted
  `max_seq_len`, and parameters are still meta tensors when the shell is
  constructed. Both have already cost this tree a bug; a target that snapshots
  operands at construction reintroduces them.

  The phase is read once per routable unit and never cached. The trunk reads
  it once per forward. A speculative draft loop rewrites `num_contexts` in
  place between draft steps, so a drafter has to read it per step -- a value
  computed on the first call is wrong on the rest, and under graph capture
  would be frozen wrong at every position.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import Any


class Phase(enum.Enum):
    """Which target a step routes to.

    Two members, not three: a mixed batch -- context and generation requests
    in one step, which in-flight batching produces routinely -- goes to
    PREFILL. That makes PREFILL the general case and DECODE a specialization
    that may assume it has no context rows at all.
    """

    PREFILL = "prefill"
    DECODE = "decode"


def phase_of(md: Any) -> Phase:
    """The routing predicate, in full.

    One host-side integer read, no synchronization, and nothing else consulted:
    a decode-only CUDA graph is a per-capture constant environment, and a
    predicate that reached for a second field would be reading one this call
    is not guaranteed to carry.

    `num_contexts == 0` is reachable two ways and both mean the same thing --
    `_apply_steady_gen_fast_prepare` writes it explicitly for an unchanged
    generation-only batch, and an ordinary prepare arrives at it whenever the
    scheduled batch holds no context request.
    """
    return Phase.DECODE if md.num_contexts == 0 else Phase.PREFILL


class Target(ABC):
    """One way of running a forward over a core's weights.

    Instantiated after weights load, never in `__init__` -- see the module
    docstring. Holds the core by reference; every operand is resolved at read
    time through `self.core`.
    """

    def __init__(self, core: Any) -> None:
        self.core = core

    @abstractmethod
    def step_args(self, md: Any) -> dict:
        """The per-step batch state this target hands the attention op.

        Separate from `forward` because it is the one projection whose *content*
        differs by phase even when the surrounding computation does not: a
        decode target states the values its routing already guarantees rather
        than reading them back off the metadata.
        """

    @abstractmethod
    def forward(self, attn_metadata: Any, *args: Any, **kwargs: Any) -> Any:
        """Run one step. The core's `forward` dispatches here after running the
        phase-independent contract check once, rather than duplicating it into
        every target."""
