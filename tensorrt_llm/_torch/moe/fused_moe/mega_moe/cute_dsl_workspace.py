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
"""Per-backend input workspace and staging for MegaMoE CuteDSL."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
    megamoe_activation_sf_bytes_per_row,
)

from .cute_dsl_runtime import MegaMoeCuteDslUnavailable

if TYPE_CHECKING:
    from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import MegaMoeSymmMemProvider

__all__ = ["MegaMoeBuffers", "MegaMoeWorkspace"]


@dataclass(frozen=True)
class MegaMoeBuffers:
    """Bucket-sized kernel buffers; routes stay local even with peer-mapped payloads."""

    activation: torch.Tensor  # (max_T, hidden // 2) uint8 (NVFP4 packed)
    activation_sf: torch.Tensor  # (max_T, sf_bytes_per_row) uint8 (FP8 SF)
    topk_weights: torch.Tensor  # (max_T, top_k) float32
    combine_output: torch.Tensor  # (max_T, 1, hidden) bf16
    shared_workspace: torch.Tensor  # (shared_ws_bytes,) uint8
    peer_offsets: list[int]  # length == world_size; [0] for single-rank
    topk_idx_local: torch.Tensor  # (max_T, top_k), int32 on Rubin, otherwise int64


class MegaMoeWorkspace:
    """Own local staging and borrow process-cached symmetric providers.

    Providers are registered during weight creation, before any forward or capture.
    The kernel-property callback is used only on a single-rank allocation miss.
    """

    def __init__(self, *, world_size: int, topk_index_dtype: torch.dtype, full_bucket: int) -> None:
        self.world_size = world_size
        self.topk_index_dtype = topk_index_dtype
        self.full_bucket = full_bucket
        self._providers: dict[int, MegaMoeSymmMemProvider] = {}
        self._direct_input_buckets: set[int] = set()
        self._local_staging_cache: dict[tuple, dict[str, torch.Tensor]] = {}
        self._last_staged_T: dict[int, int] = {}

    @property
    def full_provider(self) -> MegaMoeSymmMemProvider | None:
        """Return the full-bucket provider, or None before multi-rank allocation."""
        return self._providers.get(self.full_bucket)

    def register_provider(
        self,
        max_tokens: int,
        provider: MegaMoeSymmMemProvider,
        *,
        supports_direct_inputs: bool,
    ) -> None:
        """Attach a build-time provider and its kernel's input capability."""
        self._providers[max_tokens] = provider
        if supports_direct_inputs:
            self._direct_input_buckets.add(max_tokens)
        else:
            self._direct_input_buckets.discard(max_tokens)

    def supports_direct_inputs(self, max_tokens: int) -> bool:
        """Whether this bucket's kernel can consume the caller's input buffers."""
        return max_tokens in self._direct_input_buckets

    def _ensure_local_staging(
        self,
        *,
        top_k: int,
        hidden: int,
        device: torch.device,
        output_dtype: torch.dtype,
        max_T: int,
        query_local_kernel: Callable[[int, int, int], tuple[int, bool]],
    ) -> dict[str, torch.Tensor]:
        """Cache local tensors per bucket; multi-rank payloads come from the provider."""
        max_T = int(max_T)
        cache_key = (max_T, top_k, hidden, str(device), output_dtype)
        cached = self._local_staging_cache
        if cache_key in cached:
            return cached[cache_key]

        # ``topk_idx`` defaults to -1 so dispatch_prep skips padded tail
        # rows (``if expert_id >= Int32(0):`` in dispatch_kernel.py).
        staging = {
            "topk_idx": torch.full(
                (max_T, top_k),
                -1,
                dtype=self.topk_index_dtype,
                device=device,
            ),
        }
        if self.world_size == 1:
            sf_bytes_per_row = megamoe_activation_sf_bytes_per_row(hidden)
            # ``topk_weights`` defaults to 0 so stale combine rows
            # contribute nothing. Multi-rank uses the symm provider's
            # topk_weights region instead.
            staging["topk_weights"] = torch.zeros(
                (max_T, top_k), dtype=torch.float32, device=device
            )
            staging["activation"] = torch.empty(
                (max_T, hidden // 2), dtype=torch.uint8, device=device
            )
            staging["activation_sf"] = torch.empty(
                (max_T, sf_bytes_per_row), dtype=torch.uint8, device=device
            )
            # Always 1: the kernel collapses the top-k axis in-op (TopkReduce
            # form-A / REDG form-B); the op aliases this as the 2D output.
            combine_k = 1
            staging["combine_output"] = torch.empty(
                (max_T, combine_k, hidden),
                dtype=torch.bfloat16,
                device=device,
            )
            shared_bytes, supports_direct_inputs = query_local_kernel(max_T, top_k, hidden)
            if supports_direct_inputs:
                self._direct_input_buckets.add(max_T)
            # zeros (not empty): the leading counter prefix is an atomic-add
            # target the kernel only tail-resets for the NEXT launch, so the
            # FIRST launch needs it pre-zeroed (one-time; cached per bucket).
            staging["shared_workspace"] = torch.zeros(
                shared_bytes, dtype=torch.uint8, device=device
            )
        cached[cache_key] = staging
        return staging

    def acquire(
        self,
        *,
        top_k: int,
        hidden: int,
        device: torch.device,
        output_dtype: torch.dtype,
        launch_max_T: int,
        query_local_kernel: Callable[[int, int, int], tuple[int, bool]],
    ) -> MegaMoeBuffers:
        """Borrow one bucket of local or preallocated symmetric buffers."""
        staging = self._ensure_local_staging(
            top_k=top_k,
            hidden=hidden,
            device=device,
            output_dtype=output_dtype,
            max_T=launch_max_T,
            query_local_kernel=query_local_kernel,
        )
        if self.world_size == 1:
            return MegaMoeBuffers(
                activation=staging["activation"],
                activation_sf=staging["activation_sf"],
                topk_weights=staging["topk_weights"],
                combine_output=staging["combine_output"],
                shared_workspace=staging["shared_workspace"],
                peer_offsets=[0],
                topk_idx_local=staging["topk_idx"],
            )
        # Multi-rank: the provider must have been rendezvous'd at
        # build time (``create_weights``) -- doing it at forward time
        # would violate the build-time collective rule and deadlock
        # under PP / layer-skip.
        if self.full_provider is None:
            raise MegaMoeCuteDslUnavailable(
                f"MegaMoECuteDsl multi-rank (ep_size={self.world_size}) "
                f"requires a symmetric-memory provider built at "
                f"create_weights time, but none is registered -- "
                f"check that the EP ProcessGroup was resolvable when the "
                f"backend was constructed (mapping.moe_ep_group_pg or a "
                f"named dist.new_group), or that "
                f"model_config.skip_create_weights_in_init was not set "
                f"without a follow-up create_weights() call."
            )
        # ``launch_max_T`` comes from ``_select_launch_max_tokens``, so it is
        # always a ladder member.
        provider = self._providers[launch_max_T]
        if provider.num_topk != top_k:
            raise MegaMoeCuteDslUnavailable(
                f"MegaMoECuteDsl symm provider was built for top_k="
                f"{provider.num_topk} but run_moe called with "
                f"top_k={top_k}; recreate the backend."
            )
        regions = provider.get_regions()
        return MegaMoeBuffers(
            activation=regions.activation,
            activation_sf=regions.activation_sf,
            topk_weights=regions.topk_weights,
            combine_output=regions.combine_output,
            shared_workspace=regions.shared_workspace,
            peer_offsets=regions.peer_offsets,
            topk_idx_local=staging["topk_idx"],
        )

    def stage_inputs(
        self,
        *,
        bufs: MegaMoeBuffers,
        x: torch.Tensor | None,
        x_sf: torch.Tensor | None,
        topk_idx: torch.Tensor | None,
        topk_weights: torch.Tensor | None,
        num_tokens: int,
        top_k: int,
        stage_activation: bool,
    ) -> None:
        """Copy live inputs and clear the padded tail.

        Only previously live route rows need resetting: untouched tails start at -1.
        Routing weights must be zero over the full padded tail for the combine sum.
        """
        max_T = bufs.topk_idx_local.shape[0]
        last_T = self._last_staged_T.get(max_T)
        if last_T is not None and last_T > num_tokens:
            bufs.topk_idx_local[num_tokens:last_T].fill_(-1)
        if num_tokens > 0:
            bufs.topk_idx_local[:num_tokens].copy_(topk_idx, non_blocking=True)
            if stage_activation:
                bufs.activation[:num_tokens].copy_(x.view(torch.uint8), non_blocking=True)
                bufs.activation_sf[:num_tokens].copy_(x_sf.view(torch.uint8), non_blocking=True)
            bufs.topk_weights[:num_tokens, :top_k].copy_(topk_weights, non_blocking=True)
        if num_tokens < max_T:
            bufs.topk_weights[num_tokens:max_T, :top_k].zero_()
        self._last_staged_T[max_T] = num_tokens
