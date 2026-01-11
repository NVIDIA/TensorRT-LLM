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
NVLINK Two-Sided AllToAll Communication Strategy

This module implements the NVLINK two-sided comm AllToAll communication method for MoE.

NVLINK Two-Sided supports post-quant dispatch for all quantization modes.
"""

import os
from typing import List, Optional, Tuple

import torch

from tensorrt_llm._mnnvl_utils import MnnvlMemory, MnnvlMoe
from tensorrt_llm.mapping import Mapping

from .base import Communication


class NVLinkTwoSided(Communication):
    """
    NVLINK two-sided comm AllToAll strategy.
    This implementation utilizes symmetric memory to enable peer-to-peer access between GPUs over NVLink.
    The kernel takes the role as both sender and receiver: as the sender, it puts the data into a FIFO
    quene in peer ranks' symmetric memory; as the receiver, it gets the data from the FIFO quene to the
    local buffer. This communication model is akin to NCCL's collective operations.
    The required symmetric memory size is proportional to the communication channels opened.
    """

    def __init__(
        self,
        mapping: Mapping,
        num_experts: int,
        num_slots: int,
        top_k: int = 1,
        use_low_precision_combine: bool = False,
        alltoall_result_do_sum: bool = False,
    ):
        super().__init__(mapping)

        # Store needed parameters
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.top_k = top_k

        self.use_low_precision_combine = use_low_precision_combine
        self.alltoall_result_do_sum = alltoall_result_do_sum
        # Read from environment variable, same as wideEP
        self.enable_postquant_alltoall = (
            os.environ.get("TRTLLM_MOE_POST_QUANT_ALLTOALLV", "1") == "1"
        )

        # Invalid token expert ID (default to -1), the kernels in TRTLLM-gen is hard-coded to support -1 only.
        # CutlassFusedMoE kernels support any invalid value.
        self.invalid_token_expert_id: int = -1

        # Initialize NVLINK workspaces
        MnnvlMemory.initialize()
        self.alltoall_workspace = MnnvlMoe.get_moe_workspaces(mapping)
        self.alltoall_prepare_workspace = MnnvlMoe.get_moe_prepare_workspace(mapping)

        # Initialize dispatch state
        self._dispatch_state = {}

    @staticmethod
    def is_platform_supported() -> bool:
        """
        Check if NVLINK two-sided comm is supported on current hardware.
        """
        return MnnvlMemory.supports_mnnvl()

    def supports_post_quant_dispatch(self) -> bool:
        """
        NVLINK two-sided comm supports post-quant for all modes.
        """
        return self.enable_postquant_alltoall

    def is_workload_feasible(self, all_rank_num_tokens: List[int], num_chunks: int) -> bool:
        """
        Check if NVLINK two-sided comm is feasible for the given workload at runtime.

        This method performs runtime checks based on workload characteristics such as
        token counts, number of chunks, and other runtime parameters.
        """
        return True

    def prepare_dispatch(
        self,
        token_selected_slots: torch.Tensor,
        all_rank_num_tokens: List[int],
        local_statistic_tensor: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        NVLINK two-sided comm prepare dispatch: gather EPLB statistics and prepare alltoall_info.
        """
        all_rank_max_num_tokens = max(all_rank_num_tokens)
        top_k = token_selected_slots.shape[1]

        # Call NVLINK prepare to get alltoall_info and gather EPLB statistics
        alltoall_info, gathered_local_statistic_tensor = (
            MnnvlMoe.mnnvl_moe_alltoallv_prepare_without_allgather(
                token_selected_slots,
                local_statistic_tensor,
                self.alltoall_prepare_workspace,
                all_rank_max_num_tokens,
                self.ep_rank,
                self.ep_size,
                self.num_experts,
                self.num_slots,
                top_k,
            )
        )

        # Store alltoall_info in dispatch_state for use in dispatch()
        self._dispatch_state["alltoall_info"] = alltoall_info

        return gathered_local_statistic_tensor

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
        NVLINK two-sided comm dispatch (post-quant, uses alltoall_info from prepare_dispatch).
        """
        # Read alltoall_info from dispatch_state (set by prepare_dispatch)
        alltoall_info = self._dispatch_state.get("alltoall_info")
        if alltoall_info is None:
            raise ValueError(
                "NVLinkTwoSided dispatch requires prepare_dispatch() to be called first"
            )

        all_rank_max_num_tokens = max(all_rank_num_tokens)
        original_token_count = hidden_states.shape[0]  # Store for combine
        top_k = token_selected_slots.shape[1]

        # Dispatch quantized data using AllToAll
        hidden_states, hidden_states_sf, token_selected_slots, token_final_scales = (
            MnnvlMoe.mnnvl_moe_alltoallv(
                [hidden_states, hidden_states_sf, token_selected_slots, token_final_scales],
                alltoall_info,
                self.alltoall_workspace,
                self.ep_rank,
                self.ep_size,
            )
        )

        # Set expert IDs after alltoall
        torch.ops.trtllm.memset_expert_ids(
            token_selected_slots,
            alltoall_info.recv_rank_count_cumsum,
            all_rank_max_num_tokens,
            top_k,
            self.invalid_token_expert_id,
            self.ep_size,
        )

        # Store original_token_count for combine (alltoall_info already stored in prepare_dispatch)
        self._dispatch_state["original_token_count"] = original_token_count

        return hidden_states, hidden_states_sf, token_selected_slots, token_final_scales

    def combine(
        self,
        final_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        NVLINK two-sided comm combine - reads from self._dispatch_state.
        """
        if isinstance(final_hidden_states, list):
            final_hidden_states = final_hidden_states[0]

        final_hidden_states = MnnvlMoe.mnnvl_moe_alltoallv_combine(
            final_hidden_states,
            self._dispatch_state["alltoall_info"],
            self.alltoall_workspace,
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
            top_k=self.top_k,
            token_count=self._dispatch_state["original_token_count"],
            use_low_precision_combine=self.use_low_precision_combine,
            do_reduce=self.alltoall_result_do_sum,
        )

        return final_hidden_states
