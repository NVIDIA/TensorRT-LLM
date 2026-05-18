# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal ordered pass pipeline for the visgen-auto FX rewrites.

Designed to be deliberately small — the visgen-auto path has 5-6 passes with
one canonical ordering; the value of a full pass-manager framework
(`BaseTransform`, registry, fixpoint loop, dependency graph, per-pass
serialization) doesn't pay for itself at this scale. What this module
provides is the *one* feature that's hard to add later: a stable composition
surface for family adapters that need to splice in family-specific passes.

Contract:

* A pass is `Pass(name, fn)` where `fn(gm) -> Any`. Return value is logged
  but not interpreted (counts vary across passes — some return the number
  of rewrites, some return a `GraphModule`, some return None).
* The manager is ordered. No registry, no priorities, no implicit ordering
  by name. The order is the order passes appear in the list.
* No fixpoint. Each pass runs exactly once per `run()` call. Today's passes
  are independent — A doesn't create patterns B can re-rewrite.
* Composition: `append`, `insert_before`, `insert_after`, `replace`,
  `remove`. All mutate in place + return self for chaining.

When a family adapter wants to slot in a custom pass, it overrides
`VisGenFamilyAdapter.customize_passes(pm)` on the way through `apply_rewrites`.
That's the only public extension point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List

import torch.fx as fx

from tensorrt_llm.logger import logger


@dataclass(frozen=True)
class Pass:
    """One FX rewrite, applied once in pipeline order.

    `fn(gm)` may return an int (#nodes rewritten), the (mutated) GraphModule,
    or `None` — the manager doesn't interpret it, only logs it.
    """

    name: str
    fn: Callable[[fx.GraphModule], Any]


class PassManager:
    """Ordered list of passes with insert/replace/remove primitives."""

    def __init__(self, passes: Iterable[Pass] = ()) -> None:
        self._passes: List[Pass] = list(passes)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def names(self) -> List[str]:
        return [p.name for p in self._passes]

    def __contains__(self, name: str) -> bool:
        return any(p.name == name for p in self._passes)

    def __len__(self) -> int:
        return len(self._passes)

    def _index(self, name: str) -> int:
        for i, p in enumerate(self._passes):
            if p.name == name:
                return i
        raise KeyError(f"PassManager: no pass named {name!r} (have {self.names()})")

    # ------------------------------------------------------------------
    # Composition primitives (each returns self for chaining)
    # ------------------------------------------------------------------

    def append(self, p: Pass) -> "PassManager":
        if p.name in self:
            raise ValueError(f"PassManager: pass {p.name!r} already present")
        self._passes.append(p)
        return self

    def insert_before(self, anchor: str, p: Pass) -> "PassManager":
        if p.name in self:
            raise ValueError(f"PassManager: pass {p.name!r} already present")
        self._passes.insert(self._index(anchor), p)
        return self

    def insert_after(self, anchor: str, p: Pass) -> "PassManager":
        if p.name in self:
            raise ValueError(f"PassManager: pass {p.name!r} already present")
        self._passes.insert(self._index(anchor) + 1, p)
        return self

    def replace(self, name: str, p: Pass) -> "PassManager":
        # Allow same-name replacement; bypass the "already present" check.
        i = self._index(name)
        self._passes[i] = p
        return self

    def remove(self, name: str) -> "PassManager":
        del self._passes[self._index(name)]
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        gm: fx.GraphModule,
        *,
        skip: Iterable[str] = (),
    ) -> Dict[str, Any]:
        """Run each pass in order, logging name + wall-time + return value.

        Returns a `{pass_name: return_value}` dict. Exceptions propagate
        (one bad pass aborts the pipeline — debugging a partial rewrite is
        worse than a clean failure).
        """
        skip_set = set(skip)
        results: Dict[str, Any] = {}
        for p in self._passes:
            if p.name in skip_set:
                logger.info(f"VisGen-Auto PassManager: skip {p.name!r}")
                continue
            t0 = time.perf_counter()
            try:
                rv = p.fn(gm)
            except Exception:
                dt_ms = 1000.0 * (time.perf_counter() - t0)
                logger.error(f"VisGen-Auto PassManager: pass {p.name!r} raised after {dt_ms:.1f}ms")
                raise
            dt_ms = 1000.0 * (time.perf_counter() - t0)
            results[p.name] = rv
            logger.info(f"VisGen-Auto PassManager: {p.name} → {rv!r} in {dt_ms:.1f}ms")
        return results
