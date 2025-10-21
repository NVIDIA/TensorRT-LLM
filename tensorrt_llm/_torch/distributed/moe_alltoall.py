"""
MoE All-to-All Operations

This module provides a high-level interface for MoE all-to-all dispatch and combine operations
with proper workspace management and synchronization.
"""

from typing import Optional

import torch

from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm.logger import logger as tllm_logger
from tensorrt_llm.mapping import Mapping


class MoeAlltoAll:
    """
    Manages MoE All-to-All operations with proper workspace allocation and synchronization.

    This class encapsulates the dispatch and combine operations, managing workspace memory
    and auxiliary data structures needed for cross-GPU communication.
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
            cls.FLAG_VAL_OFFSET_INDEX = torch.ops.trtllm.MOE_A2A_FLAG_VAL_OFFSET_INDEX(
            )
            cls.LOCAL_TOKEN_COUNTER_OFFSET_INDEX = torch.ops.trtllm.MOE_A2A_LOCAL_TOKEN_COUNTER_OFFSET_INDEX(
            )
            cls.SEND_COUNTERS_OFFSET_INDEX = torch.ops.trtllm.MOE_A2A_SEND_COUNTERS_OFFSET_INDEX(
            )
            cls.RECV_COUNTERS_OFFSET_INDEX = torch.ops.trtllm.MOE_A2A_RECV_COUNTERS_OFFSET_INDEX(
            )
            cls.DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX = torch.ops.trtllm.MOE_A2A_DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX(
            )
            cls.COMBINE_COMPLETION_FLAGS_OFFSET_INDEX = torch.ops.trtllm.MOE_A2A_COMBINE_COMPLETION_FLAGS_OFFSET_INDEX(
            )
            cls.PAYLOAD_DATA_OFFSET_INDEX = torch.ops.trtllm.MOE_A2A_PAYLOAD_DATA_OFFSET_INDEX(
            )

    def __init__(
        self,
        mapping: Mapping,
        max_num_tokens_per_rank: int,
        top_k: int,
        num_experts: int,
        workspace_size_per_rank: int = 256 * 1024 * 1024,
    ):
        """
        Initialize MoeAlltoAll with workspace allocation.

        Args:
            mapping: TensorRT-LLM Mapping object containing rank information
            max_num_tokens_per_rank: Maximum number of tokens per rank
            workspace_size_per_rank: Size of workspace per rank in bytes
        """
        # Initialize constants from C++
        self._init_constants()

        self.mapping = mapping
        self.ep_size = mapping.moe_ep_size  # Expert parallel size
        self.ep_rank = mapping.moe_ep_rank
        self.max_num_tokens_per_rank = max_num_tokens_per_rank
        self.top_k = top_k
        self.num_experts = num_experts
        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("top_k must be a positive int")
        if not isinstance(self.num_experts, int) or self.num_experts <= 0:
            raise ValueError("num_experts must be a positive int")
        self.workspace_size_per_rank = workspace_size_per_rank

        # Initialize or reuse workspace
        MnnvlMemory.initialize()

        if self._WORKSPACE is None:
            tllm_logger.info(
                f"MoE AlltoAll: Allocating workspace with size {workspace_size_per_rank} bytes. ep_rank: {self.ep_rank}, ep_size: {self.ep_size}, max_num_tokens_per_rank: {self.max_num_tokens_per_rank}"
            )
            mnnvl_mem = MnnvlMemory(mapping, workspace_size_per_rank)
            workspace = mnnvl_mem.as_torch_strided_tensor(torch.uint8)
            metainfo = torch.ops.trtllm.moe_a2a_initialize(
                workspace,
                self.ep_rank,
                self.ep_size,
                self.max_num_tokens_per_rank,
            )
            MoeAlltoAll._WORKSPACE = {
                "workspace_size_per_rank": workspace_size_per_rank,
                "max_num_tokens_per_rank": self.max_num_tokens_per_rank,
                "ep_rank": self.ep_rank,
                "ep_size": self.ep_size,
                "mnnvl_mem": mnnvl_mem,
                "workspace": workspace,
                "metainfo": metainfo,
            }
        else:
            assert self._WORKSPACE[
                "workspace_size_per_rank"] == workspace_size_per_rank, "reuse workspace with different workspace_size_per_rank"
            assert self._WORKSPACE[
                "max_num_tokens_per_rank"] == self.max_num_tokens_per_rank, "reuse workspace with different max_num_tokens_per_rank"
            assert self._WORKSPACE[
                "ep_rank"] == self.ep_rank, "reuse workspace with different ep_rank"
            assert self._WORKSPACE[
                "ep_size"] == self.ep_size, "reuse workspace with different ep_size"

        self.mnnvl_mem = self._WORKSPACE["mnnvl_mem"]
        self.workspace = self._WORKSPACE["workspace"]
        self.moe_a2a_metainfo = self._WORKSPACE["metainfo"]
        self.max_num_tokens_per_rank = self._WORKSPACE[
            "max_num_tokens_per_rank"]
        # Internal state and aux data
        self.send_counters: torch.Tensor | None = None
        self.recv_counters: torch.Tensor | None = None
        self._state: str = "idle"  # idle | dispatched

    def dispatch(self,
                 token_selected_experts: torch.Tensor,
                 input_payloads: list[torch.Tensor],
                 invalid_token_expert_id: Optional[int] = None,
                 expert_id_payload_index: Optional[int] = None):
        """
        Perform MoE all-to-all dispatch operation.

        Args:
            token_selected_experts: [local_num_tokens, top_k] tensor of expert indices
            input_payloads: List of tensors to dispatch, each has shape [local_num_tokens, payload_num_elements_per_token]
            invalid_token_expert_id: If not None, set the token_selected_experts of the invalid tokens to this expert id. This is used to notify the MoE to skip these tokens for GroupGEMM.
            expert_id_payload_index: The index of token_selected_experts in the input_payloads. Must be provided if invalid_token_expert_id is not None.

        Returns:
            recv_buffers: List of tensors received, each has shape [ep_size, max_tokens_per_rank, payload_num_elements_per_token]
        """
        if self._state == "dispatched":
            raise RuntimeError(
                "dispatch called twice without an intervening combine")

        recv_buffers, send_counters, recv_counters, topk_target_ranks, topk_send_indices, combine_payload_offset = torch.ops.trtllm.moe_a2a_dispatch(
            token_selected_experts, input_payloads, self.workspace,
            self.max_num_tokens_per_rank, self.ep_rank, self.ep_size,
            self.top_k, self.num_experts)
        self._state = "dispatched"
        self.send_counters = send_counters
        self.recv_counters = recv_counters
        self.topk_target_ranks = topk_target_ranks
        self.topk_send_indices = topk_send_indices
        self.combine_payload_offset = int(combine_payload_offset)

        if invalid_token_expert_id is not None:
            assert expert_id_payload_index is not None, "expert_id_payload_index must be provided if invalid_token_expert_id is not None"
            # Sanitize expert IDs for invalid tokens directly on the recv buffer payload
            recv_token_selected_experts = recv_buffers[expert_id_payload_index]
            torch.ops.trtllm.moe_a2a_sanitize_expert_ids(
                recv_token_selected_experts,
                self.recv_counters,
                int(invalid_token_expert_id),
            )

        return recv_buffers

    def combine(self, payload, payload_in_workspace: bool = False):
        """
        Perform MoE all-to-all combine operation.

        Args:
            payload: [ep_size, max_tokens_per_rank, num_elements_per_token] tensor to combine. The dtype must be float32, bfloat16 or float16.
            payload_in_workspace: If True, 'payload' is a view into 'workspace' at 'combine_payload_offset' and no staging copy is needed. If False, the op stages 'payload' into the workspace region before combining.

        Returns:
            combined_output: [local_num_tokens, num_elements_per_token] tensor of combined results
        """
        if self._state != "dispatched":
            raise RuntimeError("combine called before a successful dispatch")

        output = torch.ops.trtllm.moe_a2a_combine(
            self.topk_target_ranks, self.topk_send_indices, self.recv_counters,
            payload, self.workspace, self.max_num_tokens_per_rank, self.ep_rank,
            self.ep_size, self.top_k, int(self.combine_payload_offset),
            bool(payload_in_workspace))
        # Reset state for next round
        self._state = "idle"
        self.send_counters = None
        self.recv_counters = None
        self.topk_target_ranks = None
        self.topk_send_indices = None
        self.combine_payload_offset = None
        return output

    def get_combine_payload_tensor_in_workspace(
            self, hidden_size: int, dtype: torch.dtype) -> torch.Tensor:
        """
        Return the combine payload tensor in the workspace, which could be used as the output of MoE kernel to avoid extra copy.
        See "payload_in_workspace" in combine method.
        """
        if self._state != "dispatched":
            raise RuntimeError(
                "get_combine_payload_tensor_in_workspace called before a successful dispatch"
            )

        return torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
            self.workspace,
            int(self.ep_rank),
            int(self.ep_size),
            int(self.max_num_tokens_per_rank),
            int(self.combine_payload_offset),
            dtype,
            int(hidden_size),
        )
