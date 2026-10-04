# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What runs a step, separated from what a model statically is.

A model's core -- deriving the shared `ModelingV2Core` base in `_core.py`,
under its own model-specific name -- carries the static structure: routing
identity, weights, parallel and quantization configuration. A `Target`
carries one way of running a forward over that structure. A core has two,
and the runtime picks between them per call.

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

from tensorrt_llm._torch._experimental.modeling_v2._router_index import step_contract_enabled


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

    Also owns the opt-in step-contract check (`_check_step_contract` below):
    for a model split into targets, `step_args` is itself the probe -- the
    one projection whose content differs by phase -- so the check belongs
    beside it, not on the core it was split out of. A core with no target
    split (deepseek, today) keeps the equivalent skeleton on
    `ModelingV2Core` in `_core.py` instead; see the comment there for why the
    two models carry it in different places.
    """

    def __init__(self, core: Any) -> None:
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it: read once here
        # rather than per forward, and False for the whole life of a served
        # engine. "pending" rather than "enabled" because the check runs once
        # -- everything it looks at is fixed at engine construction. Each
        # target owns its own flag: the two targets' `step_args` projections
        # differ, so probing one does not prove the other's fields exist, and
        # a decode-only engine run must still clear decode's flag on its own
        # first step rather than inherit prefill's.
        self._contract_pending = step_contract_enabled()

    def _check_step_contract(self, md: Any) -> None:
        """Opt-in first-forward fail-fast, run when TRTLLM_MODELING_V2_VALIDATE
        asks for it: the metadata fields this target consumes must exist (they
        are private trtllm surface). Everything checked is fixed at engine
        construction -- once per target instance is sound, and off in a served
        engine, where the only thing this could still do is fail.

        Called unconditionally every forward; the early return below is the
        gate. That costs one Python call per forward, not per layer, against
        a decode step measured in milliseconds -- accepted in exchange for
        not scattering the `_contract_pending` check across every call site.
        """
        if not self._contract_pending:
            return
        # Calling the projection is the check: it reads every metadata field
        # this target consumes, so a rename or removal upstream surfaces here
        # rather than mid-forward. Deriving it this way is the point -- a
        # hand-kept list of the same names drifts silently the first time
        # `step_args` gains a field and nobody updates the copy.
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    @abstractmethod
    def step_args(self, md: Any) -> dict:
        """The per-step batch state this target hands the attention op.

        Separate from `forward` because it is the one projection whose *content*
        differs by phase even when the surrounding computation does not: a
        decode target states the values its routing already guarantees rather
        than reading them back off the metadata. Also the probe
        `_check_step_contract` calls -- see above.
        """

    @abstractmethod
    def forward(self, attn_metadata: Any, *args: Any, **kwargs: Any) -> Any:
        """Run one step. Each target runs its own `_check_step_contract` first
        -- see above -- rather than the core running one check for both."""
