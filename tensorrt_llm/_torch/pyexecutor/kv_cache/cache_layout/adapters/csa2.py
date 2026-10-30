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
"""``CSA2CacheManager``'s layout, and what its remote-tail replay leaves out of a transfer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheRole

from ..adapter import CacheLayoutAdapter
from ..page_layout import Tensor, replicated

if TYPE_CHECKING:
    from ....llm_request import LlmRequest


class CSA2LayoutAdapter(CacheLayoutAdapter):
    """CSA2's caches, alike on every rank, and its remote-tail replay, under which the generation
    worker recomputes the prompt from the remote tail's start."""

    def buffer_tensors(
        self, layer_id: int, role: str, size: int, tokens: int, state: bool
    ) -> tuple[Tensor, ...]:
        """Every CSA2 cache holds the same bytes on every rank."""
        return replicated(size)

    def request_rebuilt_roles(self) -> frozenset[str]:
        """Under bounded replay on generation the generation worker rebuilds the sliding-window
        caches past a request's split, which a context-only worker keeps none of."""
        if getattr(self.manager, "bounded_replay_on_generation", False):
            return frozenset({str(CSA2CacheRole.SWA.role)})
        return frozenset()

    def transfer_length(self, request: LlmRequest) -> int:
        """Under remote-tail replay the generation worker recomputes the prompt from the remote
        tail's start, so the transfer moves the pages before it."""
        start = getattr(request, "py_csa2_remote_tail_start", None)
        return request.prompt_len if start is None else start

    def untransferred_buffers(self, request: LlmRequest) -> frozenset[tuple[int, str]]:
        """Under remote-tail replay the generation worker rebuilds the sliding-window caches of the
        decoder and draft layers (model layers from the split on), so the transfer leaves them."""
        split = getattr(request, "py_csa2_remote_tail_split", None)
        if split is None:
            return frozenset()
        global_of = {lid: ids[0] for lid, ids in self.layer_ids().items()}
        return frozenset(
            (global_of[int(layer_id)], str(role))
            for (layer_id, role), (model_layer, _ratio, cache_role) in (
                self.manager._physical_roles.items()
            )
            if cache_role == CSA2CacheRole.SWA and model_layer >= split
        )

    def sends_first_tokens(self, request: LlmRequest) -> bool:
        """A remote-tail source sends no first tokens: the destination generates them."""
        return getattr(request, "py_csa2_remote_tail_mode", None) != "source"
