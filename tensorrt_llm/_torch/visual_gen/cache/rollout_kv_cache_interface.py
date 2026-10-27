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
"""The interface a K/V cache for an autoregressive video rollout offers its pipeline and
attention backends, with the rollout spec and the forward plan it is driven by.

How an implementation stores K/V and how its attention backends read them is the
implementation's own business; this is only the lifecycle every one of them shares.
"""

from __future__ import annotations

import abc
import collections.abc
import itertools
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional, Tuple


@dataclass(frozen=True)
class RolloutSpec:
    """The geometry of a rollout: what ``open`` lays out in the pool and what every
    forward of the rollout may ask for. The cache keeps device state per spec, so CUDA
    graphs captured over a spec replay whenever it is opened again.

    Every causal block sees the fixed region, the ``window_tokens`` tokens before its own
    start, and itself. The window is the same for every causal block and every forward
    of a rollout; a window that changes between forwards of one rollout is not
    supported.

    Args:
        window_tokens: history tokens each causal block sees before its own start.
        max_staged_tokens: most tokens a forward may write before a commit; a forward
            may write fewer.
        causal_blocks: every causal block size a forward may use, mapped to the most
            causal blocks of that size one forward cuts its tokens into. Each size gets
            private pages for that many causal blocks.
        max_pin_tokens: most tokens any ``open`` of this spec pins.
    """

    window_tokens: int
    max_staged_tokens: int
    causal_blocks: collections.abc.Mapping[int, int]
    max_pin_tokens: int = 0

    def __post_init__(self) -> None:
        blocks = dict(sorted((int(size), int(n)) for size, n in dict(self.causal_blocks).items()))
        object.__setattr__(self, "causal_blocks", MappingProxyType(blocks))
        if min(self.window_tokens, self.max_staged_tokens) <= 0 or self.max_pin_tokens < 0:
            raise ValueError(
                "window_tokens and max_staged_tokens must be positive and max_pin_tokens "
                "non-negative"
            )
        if not blocks or any(
            size <= 0 or n <= 0 or size * n > self.max_staged_tokens for size, n in blocks.items()
        ):
            raise ValueError(
                f"causal_blocks {blocks} must map positive sizes to positive counts whose "
                f"tokens fit max_staged_tokens {self.max_staged_tokens}"
            )

    def __hash__(self) -> int:
        return hash(
            (
                self.window_tokens,
                self.max_staged_tokens,
                tuple(self.causal_blocks.items()),
                self.max_pin_tokens,
            )
        )

    @property
    def causal_block_sizes(self) -> Tuple[int, ...]:
        """The declared causal block sizes, ascending."""
        return tuple(self.causal_blocks)


@dataclass(frozen=True)
class ForwardPlan:
    """One forward over the cache, the way attention takes it (``kv=``): the real new
    tokens and how they are cut into causal blocks. Built by
    ``RolloutKVCacheInterface.plan`` for the open rollout.

    ``num_tokens`` counts the real tokens of the forward across all ranks, before
    sequence-parallel padding; rows past it are padding, neither cached nor attended.
    ``graph_key`` holds every host value a forward captured in a CUDA graph bakes in,
    so a graph runner keys on it; commits only rewrite device tensors and are not part
    of it.
    """

    cache: "RolloutKVCacheInterface"
    spec: RolloutSpec
    num_tokens: int
    causal_block_size: int
    num_causal_blocks: int

    @property
    def graph_key(self) -> Tuple:
        return (self.cache.instance_id, self.spec, self.num_tokens, self.causal_block_size)

    def validate(self) -> None:
        """Raise unless the plan's cache is open with the spec the plan was made for."""
        if not self.cache.is_open or self.cache.spec != self.spec:
            raise RuntimeError(
                "this ForwardPlan was made for a rollout that is no longer open; make a new "
                "one with cache.plan()"
            )


class RolloutKVCacheInterface(abc.ABC):
    """A K/V cache holding one autoregressive video rollout at a time.

    A pipeline drives it: ``open`` a rollout for a ``RolloutSpec``; for every forward,
    ``plan`` it and hand the plan to attention as ``kv=``; ``commit`` once the chunk's
    K/V are final; ``close`` the rollout and finally ``shutdown`` the cache. How
    attention backends write and read the K/V is up to each implementation.
    """

    _instance_ids = itertools.count()

    @property
    def instance_id(self) -> int:
        """An id unique to this cache for the life of the process; unlike ``id()``, it
        is never reused by a later cache."""
        if "_rollout_instance_id" not in self.__dict__:
            self._rollout_instance_id = next(RolloutKVCacheInterface._instance_ids)
        return self._rollout_instance_id

    @property
    @abc.abstractmethod
    def spec(self) -> Optional[RolloutSpec]:
        """The open rollout's spec; ``None`` before the first ``open``."""

    @property
    @abc.abstractmethod
    def is_open(self) -> bool:
        """Whether a rollout is open."""

    @abc.abstractmethod
    def open(self, spec: RolloutSpec, *, pin_tokens: int = 0) -> "RolloutKVCacheInterface":
        """Start a rollout of ``spec``; the first ``pin_tokens`` committed tokens are
        pinned for the whole rollout. Returns the cache, which is a context manager
        over the open rollout: leaving the ``with`` block closes it."""

    @abc.abstractmethod
    def __enter__(self) -> "RolloutKVCacheInterface":
        """The cache, with its rollout open."""

    @abc.abstractmethod
    def __exit__(self, *exc) -> None:
        """Close the rollout, whether the block ended normally or by an exception."""

    @abc.abstractmethod
    def close(self) -> None:
        """End the open rollout."""

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release the cache; it cannot be opened afterwards."""

    @abc.abstractmethod
    def plan(self, num_tokens: int, causal_block_size: Optional[int] = None) -> ForwardPlan:
        """The plan of a forward of ``num_tokens`` real new tokens cut into causal blocks
        of ``causal_block_size`` (default: one causal block of all of them)."""

    @abc.abstractmethod
    def commit(self, num_tokens: Optional[int] = None) -> None:
        """Make the staged tokens final history (default: as many as were planned)."""
