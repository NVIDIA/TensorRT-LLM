# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PdlSpec, Spin and Footprint: the per-stage launch/sync declaration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Relation = Literal["N-1", "N-k", "later", "side", "peer", "same-grid"]
Residency = Literal["PDL", "OCC", "PEER"]
Wait = Literal["entry", "after_poll", "none"]
Trigger = Literal["entry", "after_wait", "after_push", "exit"]

RELATIONS = ("N-1", "N-k", "later", "side", "peer", "same-grid")
RESIDENCIES = ("PDL", "OCC", "PEER")
WAITS = ("entry", "after_poll", "none")
TRIGGERS = ("entry", "after_wait", "after_push", "exit")


@dataclass(frozen=True)
class Spin:
    """One spin-wait inside a stage: whom it waits on, why the producer is guaranteed resident, and its deadline."""

    relation: Relation
    residency: Residency
    deadline_ns: int
    on: str = ""  # producer stage or Lamport bucket name

    def __post_init__(self):
        if self.relation not in RELATIONS:
            raise ValueError(f"Spin.relation {self.relation!r} not in {RELATIONS}")
        if self.residency not in RESIDENCIES:
            raise ValueError(f"Spin.residency {self.residency!r} not in {RESIDENCIES}")
        if not isinstance(self.deadline_ns, int) or self.deadline_ns <= 0:
            raise ValueError(
                "Spin.deadline_ns must be a positive int (rule 8: every poll needs a deadline)"
            )


@dataclass(frozen=True)
class PdlSpec:
    wait: Wait
    trigger: Trigger
    spins: tuple[Spin, ...] = ()
    grid: int = 0  # CTAs; 0 = data-dependent, then smid_roles must be False
    ctas_per_sm: int = 1
    smid_roles: bool = False
    tmem_before_wait: bool = False
    lamport_buckets: tuple[
        str, ...
    ] = ()  # names this stage produces or consumes, "<name>:p" / "<name>:c"
    push: bool = False  # the stage is a push-mode producer (rule 6)
    why: str = ""

    def __post_init__(self):
        if self.wait not in WAITS:
            raise ValueError(f"PdlSpec.wait {self.wait!r} not in {WAITS}")
        if self.trigger not in TRIGGERS:
            raise ValueError(f"PdlSpec.trigger {self.trigger!r} not in {TRIGGERS}")
        if self.wait == "none" and not self.why:
            raise ValueError(
                "PdlSpec.wait='none' needs why (rule 2: every kernel waits, or says why not)"
            )
        if not isinstance(self.spins, tuple) or not all(isinstance(s, Spin) for s in self.spins):
            raise TypeError("PdlSpec.spins must be a tuple of Spin")
        for b in self.lamport_buckets:
            if not (b.endswith(":p") or b.endswith(":c")):
                raise ValueError(
                    f"lamport bucket {b!r} must end in ':p' (producer) or ':c' (consumer)"
                )


@dataclass(frozen=True)
class Footprint:
    """Bytes a stage carves from each arena."""

    local: int = 0
    comm: int = 0
    counters: int = 0  # persistent counters, zeroed once at load
    measured: bool = True  # False = the stage allocates outside the arenas and its size is unknown
    in_arena: bool = (
        True  # False = bytes allocated by make() outside the arenas (EAGLE head/PASS copies)
    )

    def __add__(self, o: "Footprint") -> "Footprint":
        return Footprint(
            self.local + o.local,
            self.comm + o.comm,
            self.counters + o.counters,
            self.measured and o.measured,
            self.in_arena and o.in_arena,
        )

    def covers(self, o: "Footprint") -> bool:
        return self.local >= o.local and self.comm >= o.comm and self.counters >= o.counters


ZERO = Footprint()

__all__ = ["Spin", "PdlSpec", "Footprint", "ZERO", "RELATIONS", "RESIDENCIES", "WAITS", "TRIGGERS"]
