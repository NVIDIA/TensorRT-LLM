# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy profiling scratch borrowed from the custom op's process-wide cache."""

from __future__ import annotations

import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.distributed as dist

    from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
        MegaMoeSymmMemProvider,
        MegaMoeSymmRegions,
    )

__all__ = ["MegaMoeProfilingScratch", "MegaMoeScratchRequest"]


@dataclass(frozen=True)
class MegaMoeScratchRequest:
    """Full-bucket allocation facts; creating a request performs no collective."""

    process_group: dist.ProcessGroup
    world_size: int
    rank: int
    hidden_size: int
    max_tokens_per_rank: int
    num_topk: int
    output_dtype: torch.dtype
    combine_format: str
    shared_workspace_bytes: int

    def acquire(self) -> MegaMoeSymmMemProvider:
        """Get the process-cached provider during a lockstep profiling miss."""
        from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
            get_megamoe_profiling_scratch,
        )

        return get_megamoe_profiling_scratch(
            process_group=self.process_group,
            world_size=self.world_size,
            rank=self.rank,
            hidden_size=self.hidden_size,
            max_tokens_per_rank=self.max_tokens_per_rank,
            num_topk=self.num_topk,
            output_dtype=self.output_dtype,
            combine_format=self.combine_format,
            shared_workspace_bytes=self.shared_workspace_bytes,
        )


class MegaMoeProfilingScratch:
    """Keep allocation intent without extending the cached provider's lifetime."""

    def __init__(self) -> None:
        self._request: MegaMoeScratchRequest | None = None
        self._provider_ref: weakref.ReferenceType[MegaMoeSymmMemProvider] | None = None

    def reset_provider(self) -> None:
        """Forget a borrowed provider when rebuilding the backend's workspaces."""
        self._provider_ref = None

    def configure(self, request: MegaMoeScratchRequest) -> None:
        """Record an allocation request without allocating device memory."""
        self._request = request

    def _acquire_regions(self) -> MegaMoeSymmRegions:
        assert self._request is not None
        provider = self._request.acquire()
        self._provider_ref = weakref.ref(provider)
        return provider.get_regions()

    @contextmanager
    def activate(self, *, world_size: int, tuning: bool) -> Iterator[None]:
        """Expose scratch for one op, allocating only if its profiling hook asks.

        A tuning-mode cache hit never calls the factory. The op cache owns any
        allocated provider; clearing it makes our weak reference expire as well.
        """
        from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
            set_active_megamoe_profiling_scratch,
            set_active_megamoe_profiling_scratch_factory,
        )

        provider = self._provider_ref() if self._provider_ref is not None else None
        factory = (
            self._acquire_regions
            if provider is None and self._request is not None and world_size > 1 and tuning
            else None
        )
        set_active_megamoe_profiling_scratch(
            provider.get_regions() if provider is not None and world_size > 1 else None
        )
        set_active_megamoe_profiling_scratch_factory(factory)
        try:
            yield
        finally:
            set_active_megamoe_profiling_scratch(None)
            set_active_megamoe_profiling_scratch_factory(None)
