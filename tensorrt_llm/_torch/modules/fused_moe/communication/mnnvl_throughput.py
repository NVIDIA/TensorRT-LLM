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
MNNVL AllToAll Throughput Communication Strategy

This module implements the MNNVL AllToAll throughput communication method for MoE.
MNNVL Throughput uses Python-based AllToAll operations for high throughput scenarios.

MNNVL Throughput supports post-quant dispatch
"""

import os
from typing import List, Optional, Tuple

import torch

from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm.bindings import internal as _tllm_internal
from tensorrt_llm.logger import logger as tllm_logger
from tensorrt_llm.mapping import Mapping

from .base import Communication


class MNNVLThroughput(Communication):
    """
    MNNVL AllToAll strategy for throughput scenarios

    This class uses Python-based AllToAll operations for high throughput scenarios.
    It manages workspace allocation and synchronization for cross-GPU communication.
    """

    # Constants from C++ (must match moeAlltoAllKernels.h)
    MAX_RANKS = 64
    MAX_TOP_K = 8
    MAX_PAYLOADS = 8

    # Single shared workspace/memory across the process
    _WORKSPACE: dict | None = None

    # MetaInfo indices - initialized from C++ constants
    FLAG_VAL_OFFSET_INDEX = None
    LOCAL_TOKEN_COUNTER_OFFSET_INDEX = None
    SEND_COUNTERS_OFFSET_INDEX = None
    RECV_COUNTERS_OFFSET_INDEX = None
    DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX = None
    COMBINE_COMPLETION_FLAGS_OFFSET_INDEX = None
    PAYLOAD_DATA_OFFSET_INDEX = None

    @classmethod
    def _init_constants(cls):
        """Initialize constants from C++ if not already done."""
        if cls.FLAG_VAL_OFFSET_INDEX is None:
            thop = _tllm_internal.thop
            cls.FLAG_VAL_OFFSET_INDEX = int(thop.MOE_A2A_FLAG_VAL_OFFSET_INDEX)
            cls.LOCAL_TOKEN_COUNTER_OFFSET_INDEX = int(
                thop.MOE_A2A_LOCAL_TOKEN_COUNTER_OFFSET_INDEX
            )
            cls.SEND_COUNTERS_OFFSET_INDEX = int(thop.MOE_A2A_SEND_COUNTERS_OFFSET_INDEX)
            cls.RECV_COUNTERS_OFFSET_INDEX = int(thop.MOE_A2A_RECV_COUNTERS_OFFSET_INDEX)
            cls.DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX = int(
                thop.MOE_A2A_DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX
            )
            cls.COMBINE_COMPLETION_FLAGS_OFFSET_INDEX = int(
                thop.MOE_A2A_COMBINE_COMPLETION_FLAGS_OFFSET_INDEX
            )
            cls.PAYLOAD_DATA_OFFSET_INDEX = int(thop.MOE_A2A_PAYLOAD_DATA_OFFSET_INDEX)

    def __init__(
        self,
        mapping: Mapping,
        num_experts: int,
        top_k: int = 1,
        max_num_tokens_per_rank: Optional[int] = None,
        payload_in_workspace: bool = False,
    ):
        """
        Initialize MNNVLThroughput with workspace allocation.

        Args:
            mapping: TensorRT-LLM Mapping object containing rank information
            num_experts: Total number of experts
            top_k: Number of experts per token
            max_num_tokens_per_rank: Maximum number of tokens per rank (for workspace allocation)
            payload_in_workspace: If True, final_hidden_states is already in workspace
        """
        super().__init__(mapping)

        # Store needed parameters
        self.num_experts = num_experts
        self.top_k = top_k

        self.max_num_tokens_per_rank = max_num_tokens_per_rank
        self.payload_in_workspace = payload_in_workspace

        # Initialize constants from C++
        self._init_constants()

        # Get workspace size from environment variable (default 512MB)
        workspace_mb = int(os.environ.get("TRTLLM_MOE_A2A_WORKSPACE_MB", "512"))
        self.workspace_size_per_rank = workspace_mb * 1024 * 1024
        # Initialize or reuse workspace
        MnnvlMemory.initialize()

        if self._WORKSPACE is None:
            tllm_logger.info(
                f"MoE AlltoAll: Allocating workspace with size {self.workspace_size_per_rank} bytes. "
                f"ep_rank: {self.ep_rank}, ep_size: {self.ep_size}, "
                f"max_num_tokens_per_rank: {self.max_num_tokens_per_rank}"
            )
            mnnvl_mem = MnnvlMemory(mapping, self.workspace_size_per_rank)
            workspace = mnnvl_mem.as_torch_strided_tensor(torch.uint8)
            metainfo = torch.ops.trtllm.moe_a2a_initialize(
                workspace,
                self.ep_rank,
                self.ep_size,
                self.max_num_tokens_per_rank,
            )
            MNNVLThroughput._WORKSPACE = {
                "workspace_size_per_rank": self.workspace_size_per_rank,
                "max_num_tokens_per_rank": self.max_num_tokens_per_rank,
                "ep_rank": self.ep_rank,
                "ep_size": self.ep_size,
                "mnnvl_mem": mnnvl_mem,
                "workspace": workspace,
                "metainfo": metainfo,
            }
        else:
            assert self._WORKSPACE["workspace_size_per_rank"] == self.workspace_size_per_rank, (
                "reuse workspace with different workspace_size_per_rank"
            )
            assert self._WORKSPACE["max_num_tokens_per_rank"] == self.max_num_tokens_per_rank, (
                "reuse workspace with different max_num_tokens_per_rank"
            )
            assert self._WORKSPACE["ep_rank"] == self.ep_rank, (
                "reuse workspace with different ep_rank"
            )
            assert self._WORKSPACE["ep_size"] == self.ep_size, (
                "reuse workspace with different ep_size"
            )

        self.mnnvl_mem = self._WORKSPACE["mnnvl_mem"]
        self.workspace = self._WORKSPACE["workspace"]
        self.moe_a2a_metainfo = self._WORKSPACE["metainfo"]
        self.max_num_tokens_per_rank = self._WORKSPACE["max_num_tokens_per_rank"]

        # Initialize dispatch state
        self._dispatch_state = {}

        # Internal state
        self._state: str = "idle"  # idle | dispatched

        # Invalid token expert ID (default to num_experts)
        self.invalid_token_expert_id: int = self.num_experts

    @staticmethod
    def is_platform_supported() -> bool:
        """
        Check if MNNVL is supported on current hardware
        """
        return MnnvlMemory.supports_mnnvl()

    def supports_post_quant_dispatch(self) -> bool:
        """
        MNNVL Throughput supports post-quant dispatch
        """
        return True

    def is_workload_feasible(self, all_rank_num_tokens: List[int], num_chunks: int) -> bool:
        """
        Check if MNNVL Throughput is feasible for the given workload at runtime.

        This method performs runtime checks based on workload characteristics such as
        token counts, number of chunks, and other runtime parameters.
        """
        return self.is_platform_supported()

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
        Dispatch phase: scatter/send data to different ranks

        Args:
            hidden_states: Input tensor [local_num_tokens, hidden_size]
            hidden_states_sf: Input scaling factor [local_num_tokens, sf_size]
            token_selected_slots: Selected expert slots [local_num_tokens, top_k]
            token_final_scales: Router weights [local_num_tokens, top_k]
            all_rank_num_tokens: Token counts per rank [ep_size]
            use_dp_padding: Whether to use DP padding (optional)
            **kwargs: Strategy-specific arguments (unused)

        Returns:
            Tuple of (hidden_states, hidden_states_sf, token_selected_slots, token_final_scales)
            Each tensor has shape [ep_size, max_tokens_per_rank, ...]
        """
        if self._state == "dispatched":
            raise RuntimeError("dispatch called twice without an intervening combine")

        # Build payloads list - token_selected_slots is always first
        payloads = []
        payloads.append(token_selected_slots)
        payloads.append(hidden_states)
        if hidden_states_sf is not None:
            payloads.append(hidden_states_sf)
        if token_final_scales is not None:
            payloads.append(token_final_scales)

        # Call AllToAll dispatch
        (
            recv_buffers,
            send_counters,
            recv_counters,
            topk_target_ranks,
            topk_send_indices,
            combine_payload_offset,
        ) = torch.ops.trtllm.moe_a2a_dispatch(
            token_selected_slots,
            payloads,
            self.workspace,
            self.max_num_tokens_per_rank,
            self.ep_rank,
            self.ep_size,
            self.top_k,
            self.num_experts,
        )

        self._state = "dispatched"

        # Store all dispatch state for combine (no class variables)
        self._dispatch_state["topk_target_ranks"] = topk_target_ranks
        self._dispatch_state["topk_send_indices"] = topk_send_indices
        self._dispatch_state["send_counters"] = send_counters
        self._dispatch_state["recv_counters"] = recv_counters
        self._dispatch_state["combine_payload_offset"] = int(combine_payload_offset)

        # Sanitize expert IDs for invalid tokens if needed
        # token_selected_slots is always at index 0 in recv_buffers
        recv_token_selected_slots = recv_buffers[0]
        torch.ops.trtllm.moe_a2a_sanitize_expert_ids(
            recv_token_selected_slots,
            recv_counters,
            int(self.invalid_token_expert_id),
        )

        # Extract results from recv_buffers
        # Payload order: [token_selected_slots, hidden_states, hidden_states_sf (optional),
        #                  token_final_scales (optional)]
        token_selected_slots_recv = recv_buffers[0]
        hidden_states_recv = recv_buffers[1]
        if hidden_states_sf is not None:
            hidden_states_sf_recv = recv_buffers[2]
            token_final_scales_recv = recv_buffers[3] if token_final_scales is not None else None
        else:
            hidden_states_sf_recv = None
            token_final_scales_recv = recv_buffers[2] if token_final_scales is not None else None

        return (
            hidden_states_recv,
            hidden_states_sf_recv,
            token_selected_slots_recv,
            token_final_scales_recv,
        )

    def combine(
        self,
        final_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Combine phase: gather/receive data from different ranks

        Args:
            final_hidden_states: Output from MoE computation
                Shape: [ep_size, max_tokens_per_rank, hidden_size] or
                       [ep_size * max_tokens_per_rank, hidden_size] (will be reshaped)

        Returns:
            Combined output tensor [local_num_tokens, hidden_size]

        """
        if self._state != "dispatched":
            raise RuntimeError("combine called before a successful dispatch")

        # Read dispatch state
        topk_target_ranks = self._dispatch_state.get("topk_target_ranks")
        topk_send_indices = self._dispatch_state.get("topk_send_indices")
        recv_counters = self._dispatch_state.get("recv_counters")
        combine_payload_offset = self._dispatch_state.get("combine_payload_offset")

        if topk_target_ranks is None or topk_send_indices is None or recv_counters is None:
            raise RuntimeError("combine called but dispatch state is missing")

        # Reshape if needed (handle case where input is flattened)
        if final_hidden_states.dim() == 2:
            # Flattened: [ep_size * max_tokens_per_rank, hidden_size]
            # Reshape to: [ep_size, max_tokens_per_rank, hidden_size]
            hidden_size = final_hidden_states.shape[-1]
            final_hidden_states = final_hidden_states.view(
                self.ep_size, self.max_num_tokens_per_rank, hidden_size
            )
        elif final_hidden_states.dim() == 3:
            # Already shaped: [ep_size, max_tokens_per_rank, hidden_size]
            pass
        else:
            raise ValueError(
                f"final_hidden_states must be 2D or 3D, got {final_hidden_states.dim()}D"
            )

        # Call AllToAll combine
        output = torch.ops.trtllm.moe_a2a_combine(
            topk_target_ranks,
            topk_send_indices,
            recv_counters,
            final_hidden_states,
            self.workspace,
            self.max_num_tokens_per_rank,
            self.ep_rank,
            self.ep_size,
            self.top_k,
            int(combine_payload_offset),
            bool(self.payload_in_workspace),
        )

        # Reset state for next round
        self._state = "idle"
        self._dispatch_state.clear()

        return output

    def get_combine_payload_tensor_in_workspace(
        self, hidden_size: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Return the combine payload tensor in the workspace, which could be used
        as the output of MoE kernel to avoid extra copy.
        See "payload_in_workspace" in combine method.

        Args:
            hidden_size: Hidden dimension size
            dtype: Data type

        Returns:
            Tensor view into workspace [ep_size, max_tokens_per_rank, hidden_size]
        """
        if self._state != "dispatched":
            raise RuntimeError(
                "get_combine_payload_tensor_in_workspace called before a successful dispatch"
            )

        combine_payload_offset = self._dispatch_state.get("combine_payload_offset")
        if combine_payload_offset is None:
            raise RuntimeError("combine_payload_offset not found in dispatch state")

        return torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
            self.workspace,
            int(self.ep_rank),
            int(self.ep_size),
            int(self.max_num_tokens_per_rank),
            int(combine_payload_offset),
            dtype,
            int(hidden_size),
        )
