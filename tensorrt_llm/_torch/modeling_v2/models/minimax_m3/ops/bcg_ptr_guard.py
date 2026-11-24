# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static-pointer guard for buffers a captured breakable prefill graph writes through (K3 S4), shared by the M3.1
targets.

A producer captured inside a graph segment bakes the addresses of the buffers it reads and writes (the KV pool views,
the persistent out-cache-loc slots). The step metadata keeps those buffers for as long as the graphs live, so the
addresses never move under a live graph; if one ever did, every replay would write the old memory.
``StaticPointerGuard`` records the addresses at capture (``note``). The owner compares every noted key's addresses
(``check``) before every replay, ahead of the first captured segment: a mismatch raises before any captured producer
writes through a stale address. Host only: a few ``data_ptr`` reads per noted key per step.

The notes live exactly as long as the graphs that baked them: the owner calls ``reset`` when the graphs are released
(RuntimeHooks.prefill_graphs_released). The engine releases them before it drops the KV managers and the metadata,
for example between the KV-estimation executor and the final one, so the final executor's captures note the new
pool's addresses afresh instead of comparing them against the released graphs' ones.
"""

from __future__ import annotations

from typing import Callable, Dict, Hashable, Sequence, Tuple


class StaticPointerGuard:
    """Addresses noted at capture, all checked before every replay."""

    def __init__(self, what: str):
        self.what = what
        self._noted: Dict[Hashable, Tuple[int, ...]] = {}
        #: Replays checked (log-only).
        self.checks = 0

    @property
    def armed(self) -> bool:
        return bool(self._noted)

    def reset(self) -> None:
        """Forget every note (the graphs that baked them were released); the next capture arms the guard again."""
        self._noted.clear()

    def note(self, key: Hashable, ptrs: Sequence[int]) -> None:
        """Record ``key``'s addresses at capture. A later capture that sees other addresses for the same key raises:
        the earlier graph baked the old ones."""
        ptrs = tuple(int(p) for p in ptrs)
        seen = self._noted.get(key)
        if seen is not None and seen != ptrs:
            raise RuntimeError(
                f"{self.what}: {key!r} moved between captures ({seen} -> {ptrs}); a captured graph would write "
                "the old buffers"
            )
        self._noted[key] = ptrs

    def check(self, ptrs: Callable[[Hashable], Sequence[int]]) -> bool:
        """Before a replay: compare every noted key's current addresses (``ptrs(key)``) with the noted ones. Returns
        whether a comparison ran (False while unarmed); raises on the first mismatch."""
        if not self._noted:
            return False
        for key, want in self._noted.items():
            got = tuple(int(p) for p in ptrs(key))
            if got != want:
                raise RuntimeError(
                    f"{self.what}: {key!r} buffers moved since capture (captured {want}, now {got}); the replay "
                    "would write the captured addresses"
                )
        self.checks += 1
        return True
