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
"""Keys this rank recently confirmed in the pool, so the save path can skip probing them again (ldfix F3).

`TRTLLM_MOONCAKE_STORE_LOCAL_STORED_SET=1`. Most pages a save batch lists are already stored (86 % in ctx-only
S150: the RESAVE re-listing of a long prompt's full-attention blocks, written by this rank a few chunks earlier),
and each batch pays a `batch_is_exist` RPC on the save thread to find that out. The set remembers keys this rank
wrote successfully or saw present, for `TRTLLM_MOONCAKE_STORE_LOCAL_STORED_TTL` seconds (default 60) and at most
`TRTLLM_MOONCAKE_STORE_LOCAL_STORED_CAP` keys (default 1,000,000, least recently confirmed evicted first).

Only the save path consults it, and only to skip a probe-and-rewrite: a key the pool has evicted since costs, at
worst, one re-save skipped within the TTL, i.e. a later cache miss (what a dropped save costs today). Lookups and
loads never read it, so correctness does not depend on it. Mooncake has no eviction notification, so the
nearest signal is used instead: any probe on this rank that reports one of its keys absent removes that key.
"""

import os
import threading
import time
from collections import OrderedDict
from typing import Callable, Iterable, List, Optional, Sequence

__all__ = ["LOCAL_STORED_SET_ENV", "LocalStoredSet"]

LOCAL_STORED_SET_ENV = "TRTLLM_MOONCAKE_STORE_LOCAL_STORED_SET"
LOCAL_STORED_TTL_ENV = "TRTLLM_MOONCAKE_STORE_LOCAL_STORED_TTL"
LOCAL_STORED_CAP_ENV = "TRTLLM_MOONCAKE_STORE_LOCAL_STORED_CAP"
DEFAULT_TTL = 60.0
DEFAULT_CAP = 1_000_000


class LocalStoredSet:
    """A bounded, expiring set of store keys; thread-safe (save thread adds, probing threads discard)."""

    def __init__(
        self,
        capacity: int = DEFAULT_CAP,
        ttl: float = DEFAULT_TTL,
        clock: Callable[[], float] = time.monotonic,
    ):
        if capacity < 1:
            raise ValueError(f"{LOCAL_STORED_CAP_ENV} must be at least 1, got {capacity}")
        self._capacity = int(capacity)
        self._ttl = float(ttl)
        self._clock = clock
        self._entries: "OrderedDict[str, float]" = OrderedDict()  # key -> confirmed at, oldest first
        self._lock = threading.Lock()
        #: Keys a save skipped probing, keys confirmed, keys dropped on an absent probe, and expired lookups.
        self.hits = 0
        self.added = 0
        self.invalidated = 0
        self.expired = 0

    @classmethod
    def from_env(cls) -> Optional["LocalStoredSet"]:
        """The set when `LOCAL_STORED_SET_ENV` is 1, else None (the switch is off)."""
        if os.environ.get(LOCAL_STORED_SET_ENV, "0") != "1":
            return None
        return cls(
            capacity=int(os.environ.get(LOCAL_STORED_CAP_ENV, DEFAULT_CAP)),
            ttl=float(os.environ.get(LOCAL_STORED_TTL_ENV, DEFAULT_TTL)),
        )

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def add_many(self, keys: Iterable[str]) -> None:
        """Record keys as confirmed in the pool now."""
        now = self._clock()
        with self._lock:
            for key in keys:
                if key in self._entries:
                    self._entries.move_to_end(key)
                self._entries[key] = now
                self.added += 1
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def discard_many(self, keys: Iterable[str]) -> None:
        """Forget keys a probe found absent (the pool evicted them, or a write never landed)."""
        with self._lock:
            for key in keys:
                if self._entries.pop(key, None) is not None:
                    self.invalidated += 1

    def known(self, keys: Sequence[str]) -> List[bool]:
        """Per key, whether it was confirmed within the TTL. Expired entries are dropped on the way."""
        now = self._clock()
        out: List[bool] = []
        with self._lock:
            for key in keys:
                confirmed = self._entries.get(key)
                if confirmed is None:
                    out.append(False)
                elif self._ttl > 0 and now - confirmed > self._ttl:
                    del self._entries[key]
                    self.expired += 1
                    out.append(False)
                else:
                    out.append(True)
            self.hits += sum(out)
        return out

    def describe(self) -> str:
        with self._lock:
            size = len(self._entries)
        return (
            f"local-stored set {size} keys (cap {self._capacity}, ttl {self._ttl:g}s): skipped {self.hits}, "
            f"added {self.added}, invalidated {self.invalidated}, expired {self.expired}"
        )
