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

# ruff: noqa: E501


"""
NVLINK One-Sided AllToAll Communication Strategy

This module implements the NVLINK one-sided comm AllToAll throughput communication method for MoE.

NVLINK One-Sided supports post-quant dispatch.
"""

import os
from typing import List, Optional, Tuple

import torch

from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm.bindings import internal as _tllm_internal
from tensorrt_llm.logger import logger as tllm_logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import pad_up

from .base import Communication


class NVLinkOneSided(Communication):
    """
    NVLINK one-sided comm AllToAll strategy for throughput scenarios.

    This implementation utilizes symmetric memory to enable peer-to-peer access between GPUs over NVLink.
    The kernels only take the role as one side of the communication: the dispatch kernel puts the data
    into peer ranks' symmetric memory from local buffer, while the combine kernel gets the data from peer
    ranks' symmetric memory and reduces the data into local buffer. It is the most efficient implementation
    by now, but requires symmetric memory size proportional to `max_num_tokens * n_ranks`, which may not
    scale well for very large-scale parallelization.
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
    EPLB_GATHERED_STATS_OFFSET_INDEX = None
    PAYLOAD_DATA_OFFSET_INDEX = None

    @staticmethod
    def get_aux_data_size(
        ep_size: int,
        max_num_tokens: int,
        eplb_stats_num_experts: Optional[int] = None,
    ) -> int:
        return torch.ops.trtllm.moe_a2a_get_aux_data_size(
            ep_size, max_num_tokens, eplb_stats_num_experts
        )

    @staticmethod
    def calculate_required_workspace_size(
        ep_size: int,
        top_k: int,
        max_num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        eplb_stats_num_experts: Optional[int] = None,
        extra_payload_bytes_per_token: int = 0,
    ) -> int:
        element_size = dtype.itemsize

        # Auxiliary data size
        workspace_size = NVLinkOneSided.get_aux_data_size(
            ep_size, max_num_tokens, eplb_stats_num_experts
        )

        # Dispatch needs workspace for [ep_size, max_tokens] tokens,
        # but due to the variety of quantization recipes, we cannot know the exact size, so we conservatively estimate assuming no quantization.
        # Meanwhile, we consider the alignment requirement as in moeA2ADispatchOp and moeA2ACombineOp.
        # (Unquantized) token hidden states
        workspace_size += ep_size * max_num_tokens * hidden_size * element_size
        workspace_size = pad_up(workspace_size, 128)
        # token_selected_experts
        workspace_size += ep_size * max_num_tokens * top_k * 4
        workspace_size = pad_up(workspace_size, 128)
        # token_final_scales
        workspace_size += ep_size * max_num_tokens * top_k * 4
        workspace_size = pad_up(workspace_size, 128)
        # Required workspace for combine [ep_size, max_tokens] tokens
        workspace_size += ep_size * max_num_tokens * hidden_size * element_size
        workspace_size = pad_up(workspace_size, 128)
        # extra payload bytes per token
        workspace_size += ep_size * max_num_tokens * extra_payload_bytes_per_token
        workspace_size = pad_up(workspace_size, 128)

        return workspace_size

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
            cls.EPLB_GATHERED_STATS_OFFSET_INDEX = int(
                thop.MOE_A2A_EPLB_GATHERED_STATS_OFFSET_INDEX
            )
            cls.PAYLOAD_DATA_OFFSET_INDEX = int(thop.MOE_A2A_PAYLOAD_DATA_OFFSET_INDEX)

    def __init__(
        self,
        mapping: Mapping,
        num_slots: int,
        top_k: int,
        max_num_tokens_per_rank: int,
        payload_in_workspace: bool = False,
        hidden_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        num_experts: Optional[int] = None,
    ):
        """
        Initialize NVLinkOneSided with workspace allocation.

        Args:
            mapping: TensorRT-LLM Mapping object containing rank information
            num_slots: Number of routing slots (token_selected_experts values are in [0, num_slots)).
                Note: The terminology is mapped to `num_experts` in this class and the kernels.
            top_k: Number of experts per token
            max_num_tokens_per_rank: Maximum number of tokens per rank (for workspace allocation)
            payload_in_workspace: If True, final_hidden_states is already in workspace
            hidden_size: Hidden dimension size (optional, for auto workspace calculation)
            dtype: Data type (optional, for auto workspace calculation)
            num_experts: (Optional) Number of experts for EPLB stats (must be <= num_slots). DO NOT provide this parameter if EPLB is not enabled.
                Note: The terminology is mapped to `eplb_stats_num_experts` in this class and the kernels.
        """
        super().__init__(mapping)

        if self.mapping.world_size != self.ep_size:
            raise RuntimeError("Currently NVLinkOneSided only supports pure EP for MoE.")

        # Store needed parameters
        self.num_experts = num_slots
        self.top_k = top_k
        self.max_num_tokens_per_rank = max_num_tokens_per_rank
        self.payload_in_workspace = payload_in_workspace
        if num_experts is not None:
            assert num_experts > 0 and num_experts <= num_slots, (
                "num_experts must be in (0, num_slots]"
            )
            tllm_logger.info(
                "NVLinkOneSided AlltoAll: EPLB is enabled, with num_slots="
                f"{num_slots} and num_experts={num_experts}"
            )
        self.enable_eplb = num_experts is not None
        self.eplb_stats_num_experts = num_experts

        # Initialize constants from C++
        self._init_constants()

        # Get workspace size
        auto_workspace_size = None
        if hidden_size is not None and dtype is not None:
            auto_workspace_size = self.calculate_required_workspace_size(
                self.ep_size,
                self.top_k,
                max_num_tokens_per_rank,
                hidden_size,
                dtype,
                eplb_stats_num_experts=self.eplb_stats_num_experts,
            )
        workspace_mb_env = os.environ.get("TRTLLM_MOE_A2A_WORKSPACE_MB")
        if workspace_mb_env:
            self.workspace_size_per_rank = int(workspace_mb_env) * 1024 * 1024
            msg = f"NVLinkOneSided: Forcing workspace size to {self.workspace_size_per_rank} bytes (TRTLLM_MOE_A2A_WORKSPACE_MB={workspace_mb_env})."
            if auto_workspace_size is not None:
                msg += f"Automatically calculated workspace size is {auto_workspace_size} bytes."
                msg += "Auto calculation is conservative, so only consider overriding it if you have a specific reason."
            tllm_logger.warning(msg)
        elif auto_workspace_size is not None:
            self.workspace_size_per_rank = auto_workspace_size
        else:
            tllm_logger.warning(
                "NVLinkOneSided: hidden_size and dtype are not provided (which are required for calculating workspace size)."
                "Using default workspace size 2048MB."
            )
            self.workspace_size_per_rank = 2048 * 1024 * 1024

        # Initialize or reuse workspace
        MnnvlMemory.initialize()

        if self._WORKSPACE is None:
            tllm_logger.info(
                f"NVLinkOneSided: Allocating workspace with size {self.workspace_size_per_rank} bytes."
                f"ep_rank: {self.ep_rank}, ep_size: {self.ep_size}, top_k: {self.top_k}, max_num_tokens_per_rank: {self.max_num_tokens_per_rank}"
            )
            mnnvl_mem = MnnvlMemory(mapping, self.workspace_size_per_rank)
            workspace = mnnvl_mem.as_torch_strided_tensor(torch.uint8)
            metainfo = torch.ops.trtllm.moe_a2a_initialize(
                workspace,
                self.ep_rank,
                self.ep_size,
                self.max_num_tokens_per_rank,
                self.eplb_stats_num_experts,
            )
            NVLinkOneSided._WORKSPACE = {
                "workspace_size_per_rank": self.workspace_size_per_rank,
                "max_num_tokens_per_rank": self.max_num_tokens_per_rank,
                "ep_rank": self.ep_rank,
                "ep_size": self.ep_size,
                "eplb_stats_num_experts": self.eplb_stats_num_experts,
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
            assert self._WORKSPACE["eplb_stats_num_experts"] == self.eplb_stats_num_experts, (
                "reuse workspace with different eplb_stats_num_experts"
            )

        self.mnnvl_mem = self._WORKSPACE["mnnvl_mem"]
        self.workspace = self._WORKSPACE["workspace"]
        self.moe_a2a_metainfo = self._WORKSPACE["metainfo"]
        self.max_num_tokens_per_rank = self._WORKSPACE["max_num_tokens_per_rank"]

        # Initialize dispatch state
        self._dispatch_state = {"phase": "idle"}

        # Invalid token expert ID (default to -1), the kernels in TRTLLM-gen is hard-code to support -1 only.
        self.invalid_token_expert_id: int = -1

    @staticmethod
    def is_platform_supported() -> bool:
        """
        Check if NVLINK one-sided comm is supported on current hardware.
        """
        return MnnvlMemory.supports_mnnvl()

    def supports_post_quant_dispatch(self) -> bool:
        """
        NVLINK one-sided comm supports post-quant dispatch.
        """
        return True

    def is_workload_feasible(self, all_rank_num_tokens: List[int], num_chunks: int) -> bool:
        """
        Check if NVLINK one-sided comm is feasible for the given workload at runtime.

        This method performs runtime checks based on workload characteristics such as
        token counts, number of chunks, and other runtime parameters.
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
        if self._dispatch_state.get("phase") == "dispatched":
            raise RuntimeError("dispatch called twice without an intervening combine")

        # Calculate runtime_max_tokens_per_rank from all_rank_num_tokens
        runtime_max_tokens_per_rank = max(all_rank_num_tokens)

        # Build payloads list - match TRTLLMGen baseline order for optimal performance
        # Order: [hidden_states, hidden_states_sf (optional), token_selected_slots, token_final_scales (optional)]

        payloads = []
        payloads.append(hidden_states)
        if hidden_states_sf is not None:
            payloads.append(hidden_states_sf)

        payloads.append(token_selected_slots)
        if token_final_scales is not None:
            payloads.append(token_final_scales)

        eplb_local_stats = kwargs.get("eplb_local_stats")
        if eplb_local_stats is not None:
            assert self.enable_eplb, "eplb_local_stats provided but enable_eplb is False"
            assert eplb_local_stats.dim() == 1, "eplb_local_stats must be a 1D tensor"
            assert eplb_local_stats.size(0) == self.eplb_stats_num_experts, (
                "eplb_local_stats size must match eplb_stats_num_experts"
            )

        recv_buffers, combine_payload_offset, eplb_gathered_stats = (
            torch.ops.trtllm.moe_a2a_dispatch(
                token_selected_slots,
                payloads,
                self.workspace,
                self.moe_a2a_metainfo,
                runtime_max_tokens_per_rank,
                self.ep_rank,
                self.ep_size,
                self.top_k,
                self.num_experts,
                eplb_local_stats,
            )
        )
        if eplb_gathered_stats.numel() == 0:
            eplb_gathered_stats = None
        self._dispatch_state["eplb_gathered_stats"] = eplb_gathered_stats
        self._dispatch_state["combine_payload_offset"] = int(combine_payload_offset)
        self._dispatch_state["local_num_tokens"] = token_selected_slots.size(0)
        self._dispatch_state["runtime_max_tokens_per_rank"] = runtime_max_tokens_per_rank
        self._dispatch_state["phase"] = "dispatched"

        # Extract results from recv_buffers
        # Payload order matches input:
        # [hidden_states, hidden_states_sf (optional), token_selected_slots, token_final_scales (optional)]
        hidden_states_recv = recv_buffers[0]
        if hidden_states_sf is not None:
            hidden_states_sf_recv = recv_buffers[1]
            token_selected_slots_recv = recv_buffers[2]
            token_final_scales_recv = recv_buffers[3] if token_final_scales is not None else None
        else:
            hidden_states_sf_recv = None
            token_selected_slots_recv = recv_buffers[1]
            token_final_scales_recv = recv_buffers[2] if token_final_scales is not None else None

        torch.ops.trtllm.moe_a2a_sanitize_expert_ids(
            token_selected_slots_recv,
            self.workspace,
            self.moe_a2a_metainfo,
            self.ep_rank,
            int(self.invalid_token_expert_id),
        )

        # Flatten 3D tensors to 2D for compatibility with MoE backends
        # recv_buffers have shape [ep_size, max_tokens_per_rank, ...], flatten to [ep_size * max_tokens_per_rank, ...]
        hidden_states_recv = hidden_states_recv.view(-1, hidden_states_recv.shape[-1])
        if hidden_states_sf_recv is not None:
            hidden_states_sf_recv = hidden_states_sf_recv.view(-1, hidden_states_sf_recv.shape[-1])
        token_selected_slots_recv = token_selected_slots_recv.view(
            -1, token_selected_slots_recv.shape[-1]
        )
        if token_final_scales_recv is not None:
            token_final_scales_recv = token_final_scales_recv.view(
                -1, token_final_scales_recv.shape[-1]
            )

        return (
            hidden_states_recv,
            hidden_states_sf_recv,
            token_selected_slots_recv,
            token_final_scales_recv,
        )

    def get_eplb_gathered_statistics(self) -> Optional[torch.Tensor]:
        """
        Return gathered EPLB statistics from the last dispatch, if available.
        """
        assert self.enable_eplb, "EPLB is not enabled"
        return self._dispatch_state.get("eplb_gathered_stats")

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
        if self._dispatch_state.get("phase") != "dispatched":
            raise RuntimeError("combine called before a successful dispatch")

        local_num_tokens = self._dispatch_state.get("local_num_tokens")
        combine_payload_offset = self._dispatch_state.get("combine_payload_offset")
        runtime_max_tokens_per_rank = self._dispatch_state.get("runtime_max_tokens_per_rank")

        if (
            local_num_tokens is None
            or combine_payload_offset is None
            or runtime_max_tokens_per_rank is None
        ):
            raise RuntimeError("combine called but dispatch state is missing")

        # Reshape if needed (handle case where input is flattened)
        if final_hidden_states.dim() == 2:
            # Flattened: [ep_size * max_tokens_per_rank, hidden_size]
            # Reshape to: [ep_size, max_tokens_per_rank, hidden_size]
            hidden_size = final_hidden_states.shape[-1]
            final_hidden_states = final_hidden_states.view(
                self.ep_size, runtime_max_tokens_per_rank, hidden_size
            )
        elif final_hidden_states.dim() == 3:
            # Already shaped: [ep_size, max_tokens_per_rank, hidden_size]
            pass
        else:
            raise ValueError(
                f"final_hidden_states must be 2D or 3D, got {final_hidden_states.dim()}D"
            )
        output = torch.ops.trtllm.moe_a2a_combine(
            final_hidden_states,
            int(local_num_tokens),
            self.workspace,
            self.moe_a2a_metainfo,
            int(runtime_max_tokens_per_rank),
            self.ep_rank,
            self.ep_size,
            self.top_k,
            int(combine_payload_offset),
            bool(self.payload_in_workspace),
        )

        # Reset state for next round
        self._dispatch_state = {"phase": "idle"}

        return output

    def get_combine_payload_tensor_in_workspace(
        self, runtime_max_tokens_per_rank: int, hidden_size: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Return the combine payload tensor in the workspace, which could be used
        as the output of MoE kernel to avoid extra copy.
        See "payload_in_workspace" in combine method.

        Args:
            runtime_max_tokens_per_rank: Runtime max tokens per rank
            hidden_size: Hidden dimension size
            dtype: Data type

        Returns:
            Tensor view into workspace [ep_size, max_tokens_per_rank, hidden_size]
        """
        if self._dispatch_state.get("phase") != "dispatched":
            raise RuntimeError(
                "get_combine_payload_tensor_in_workspace called before a successful dispatch"
            )

        combine_payload_offset = self._dispatch_state.get("combine_payload_offset")
        if combine_payload_offset is None:
            raise RuntimeError("combine_payload_offset not found in dispatch state")

        result = torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
            self.workspace,
            int(self.ep_rank),
            int(self.ep_size),
            int(runtime_max_tokens_per_rank),
            int(combine_payload_offset),
            dtype,
            int(hidden_size),
        )

        return result
