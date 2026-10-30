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
"""``MambaHybridCacheManagerV2``'s layout: recurrent state beside the attention caches."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

from ...mamba_cache_manager import _KDA_REPLAY_COUNT_ROLE, MambaRole, _dtype_name
from ..adapter import CacheLayoutAdapter
from ..page_layout import Tensor, grid_place, replicated, sectioned
from ..page_table import rank_topology

if TYPE_CHECKING:
    from ....llm_request import LlmRequest


class MambaLayoutAdapter(CacheLayoutAdapter):
    """Recurrent state, which every pipeline stage of a hybrid model may lack while the model keeps
    it, and the generation-local KDA replay caches the importer seeds."""

    def buffer_tensors(
        self, layer_id: int, role: str, size: int, tokens: int, state: bool
    ) -> tuple[Tensor, ...]:
        """Recurrent roles and how they spread over ranks.

        Convolution state is sectioned: its flat per-layer buffer is a concatenation of sections
        (Mamba2 ``[x | B | C]``, GDN ``[Q | K | V]``) each split by byte over the ranks recurrent
        state splits over. SSM state is split by head. Both name their element types, which their
        dtypes configure apart from the KV cache's. PLE state is computed from replicated inputs, so
        every rank holds identical bytes, as it does the accepted-draft count. The
        generation-local KDA replay caches follow the rank's heads; as derived roles they are this
        rank's own (the base declaration).
        """
        m = self.manager
        if role == MambaRole.CONV_STATE:
            d_conv_m1 = int(m.conv_state_shape[1])
            conv_elem_size = m.conv_state_dtype.itemsize
            split = self.state_split()
            ranks, _ = grid_place(rank_topology(m.mapping), split)
            return sectioned(
                [int(dim) * d_conv_m1 * conv_elem_size for dim in m.conv_section_dims],
                ranks,
                split,
                dtype=_dtype_name(m.conv_state_dtype),
            )
        if role == MambaRole.SSM_STATE:
            _, head_dim, d_state = m.ssm_state_shape
            return self.head_split(
                layer_id,
                size,
                tokens,
                state,
                head_bytes=int(head_dim) * int(d_state) * m.ssm_state_dtype.itemsize,
                dtype=_dtype_name(m.ssm_state_dtype),
            )
        if role in (
            MambaRole.PLE_CONV_STATE,
            MambaRole.PLE_NGRAM_CONTEXT,
            str(_KDA_REPLAY_COUNT_ROLE),
        ):
            return replicated(size)
        return super().buffer_tensors(layer_id, role, size, tokens, state)

    def derived_roles(self) -> frozenset[str]:
        """Keep the KDA replay buffers out of transfers. They are generation-local: a context
        server allocates none, and a generation server seeds them from the transferred conv state
        (``rebuild_imported``)."""
        replay_roles = {str(name) for name in self.manager._kda_replay_layout()}
        if replay_roles:
            replay_roles.add(str(_KDA_REPLAY_COUNT_ROLE))
        return super().derived_roles() | frozenset(replay_roles)

    def state_slot(self, request: LlmRequest, layer_group: int) -> Optional[int]:
        """The request's recurrent-state slot, while this rank holds recurrent layers."""
        m = self.manager
        if m.local_num_mamba_layers == 0:
            return None
        return m._request_id_to_state_index[request.py_request_id]

    @property
    def model_has_recurrent_state(self) -> bool:
        """A hybrid model keeps recurrent state, on whichever pipeline stages its recurrent layers
        lie."""
        return True

    def rebuild_imported(self, requests: Iterable[LlmRequest]) -> None:
        """Seed the KDA replay caches, which no transfer carries, from the imported conv state."""
        requests = list(requests)
        super().rebuild_imported(requests)
        self.manager.seed_kda_replay_caches_for_disagg_gen(
            [request.py_request_id for request in requests]
        )
