# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
AllGather + ReduceScatter Communication Strategy

This module implements the AllGather + ReduceScatter communication method for MoE.
This is the default fallback strategy that always works.

AllGather ALWAYS supports post-quant dispatch (quantize → allgather)
"""

from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.distributed import allgather, reducescatter
from tensorrt_llm.mapping import Mapping

from .base import Communication


class AllGatherReduceScatter(Communication):
    def __init__(
        self,
        mapping: Mapping,
    ):
        super().__init__(mapping)

        # Initialize dispatch state
        self._dispatch_state = {}

    @staticmethod
    def is_platform_supported() -> bool:
        """
        AllGather + ReduceScatter is always supported as the fallback strategy
        """
        return True

    def is_workload_feasible(self, all_rank_num_tokens: List[int], num_chunks: int) -> bool:
        """
        Check if AllGather is feasible for the given workload at runtime.

        AllGather is always available as fallback, so this always returns True.
        """
        return True

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        hidden_states_sf: Optional[torch.Tensor],
        token_selected_slots: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        all_rank_num_tokens: List[int],
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        AllGather dispatch (always post-quant dispatch)
        """
        sizes = None if use_dp_padding else all_rank_num_tokens

        hidden_states, hidden_states_sf, token_selected_slots, token_final_scales = allgather(
            [hidden_states, hidden_states_sf, token_selected_slots, token_final_scales],
            self.mapping,
            dim=0,
            sizes=sizes,
        )

        # Store sizes for combine
        self._dispatch_state["sizes"] = sizes

        return hidden_states, hidden_states_sf, token_selected_slots, token_final_scales

    def combine(
        self,
        final_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        ReduceScatter combine phase
        """
        outputs = reducescatter(
            final_hidden_states, self.mapping, dim=0, sizes=self._dispatch_state.get("sizes")
        )
        return outputs
