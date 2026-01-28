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
Base Classes for MoE Communication Strategies

This module defines the abstract base class and common types for MoE communication methods.

Key Design: Communication dispatch can happen BEFORE or AFTER quantization
- Pre-quant dispatch: dispatch → quantize → allgather (DeepEP pre-quant)
- Post-quant dispatch: quantize → allgather → dispatch (MNNVL, DeepEP post-quant)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch

from tensorrt_llm.mapping import Mapping


class Communication(ABC):
    """
    Abstract base class for MoE communication methods

    Key Design: Supports both pre-quant and post-quant dispatch
    - Pre-quant: dispatch() called BEFORE quantization
    - Post-quant: dispatch() called AFTER quantization

    The communication method declares which mode(s) it supports via supports_post_quant_dispatch()
    """

    def __init__(
        self,
        mapping: Mapping,
    ):
        self.mapping = mapping
        self.ep_size = mapping.moe_ep_size
        self.ep_rank = mapping.moe_ep_rank

        # Check platform support and raise error if not supported
        if not self.is_platform_supported():
            raise RuntimeError(
                f"Communication strategy {self.__class__.__name__} "
                f"is not supported on this platform."
            )
        self._is_platform_supported = True

    @staticmethod
    @abstractmethod
    def is_platform_supported() -> bool:
        """
        Check if this communication strategy is supported on the current platform.

        This method performs platform/hardware checks to determine if the strategy
        can be used on the current system.

        Returns:
            True if platform is supported, False otherwise

        Note: This is a static method that can be called before instantiation
              to check compatibility without creating an instance.
        """
        raise NotImplementedError

    @abstractmethod
    def is_workload_feasible(
        self,
        all_rank_num_tokens: List[int],
        num_chunks: int,
    ) -> bool:
        """
        Check if this communication strategy is feasible for the given workload at runtime.

        This method performs runtime checks based on workload characteristics such as
        token counts, number of chunks, and other runtime parameters.
        """
        raise NotImplementedError

    def supports_post_quant_dispatch(self) -> bool:
        """
        Check if this strategy supports post-quantization dispatch

        Returns:
            True: Dispatch should happen AFTER quantization
            False: Dispatch should happen BEFORE quantization

        Default: True for most strategies (post-quant is more common)
        """
        return True

    def prepare_dispatch(
        self,
        token_selected_slots: torch.Tensor,  # [local_num_tokens, top_k]
        all_rank_num_tokens: List[int],  # [ep_size]
        local_statistic_tensor: Optional[torch.Tensor] = None,  # [num_experts], for EPLB
    ) -> Optional[torch.Tensor]:
        """
        Prepare dispatch metadata and information (BEFORE quantization)

        This method is called before quantization to:
        1. Gather EPLB statistics (if needed, e.g., MNNVL)
        2. Prepare communication metadata (stored internally for later use in dispatch)

        Args:
            token_selected_slots: Selected expert slots [local_num_tokens, top_k]
            all_rank_num_tokens: Token counts per rank [ep_size]
            local_statistic_tensor: Local EPLB statistics [num_experts] (optional)

        Returns:
            gathered_stats: Gathered EPLB statistics across all ranks (optional)

        Side effects:
            May store internal state for use in dispatch() (e.g., MNNVL stores alltoall_info)
        """
        # Default: No preparation needed (AllGather, DeepEP, DeepEPLowLatency)
        return None

    @abstractmethod
    def dispatch(
        self,
        # Core data
        hidden_states: torch.Tensor,  # [local_num_tokens, hidden_size]
        hidden_states_sf: Optional[torch.Tensor],  # [local_num_tokens, sf_size]
        token_selected_slots: torch.Tensor,  # [local_num_tokens, top_k]
        token_final_scales: Optional[torch.Tensor],  # [local_num_tokens, top_k]
        # Metadata
        all_rank_num_tokens: List[int],  # [ep_size]
        # Optional parameters for flexibility
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Dispatch phase: scatter/send data to different ranks

        This method may read internal state from prepare_dispatch() and stores
        dispatch metadata in self._dispatch_state for later use in combine().

        Args:
            hidden_states: Input tensor [local_num_tokens, hidden_size]
            hidden_states_sf: Input scaling factor [local_num_tokens, sf_size]
            token_selected_slots: Selected expert slots [local_num_tokens, top_k]
            token_final_scales: Router weights [local_num_tokens, top_k]
            all_rank_num_tokens: Token counts per rank [ep_size]
            use_dp_padding: Whether to use DP padding (optional)
            **kwargs: Strategy-specific arguments

        Returns:
            Tuple of (hidden_states, hidden_states_sf, token_selected_slots, token_final_scales)

        Side effects:
            May read from internal state set by prepare_dispatch() (e.g., MNNVL reads alltoall_info)
            Stores dispatch state in self._dispatch_state for combine()
        """
        raise NotImplementedError

    @abstractmethod
    def combine(
        self,
        final_hidden_states: torch.Tensor,  # MoE computation output
        **kwargs,
    ) -> torch.Tensor:
        """
        Combine phase: gather/receive data from different ranks

        This method reads dispatch metadata from self._dispatch_state that was set by dispatch().

        Args:
            final_hidden_states: Output from MoE computation
            **kwargs: Strategy-specific arguments

        Returns:
            Combined output tensor [local_num_tokens, hidden_size]
        """
        raise NotImplementedError

    def get_eplb_gathered_statistics(self) -> Optional[torch.Tensor]:
        """
        Return gathered EPLB statistics from the last dispatch, if available.
        """
        raise NotImplementedError
