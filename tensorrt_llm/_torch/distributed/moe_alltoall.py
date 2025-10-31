"""
MoE All-to-All Operations

This module provides a high-level interface for MoE all-to-all dispatch and combine operations
with proper workspace management and synchronization.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm.bindings import internal as _tllm_internal
from tensorrt_llm.logger import logger as tllm_logger
from tensorrt_llm.mapping import Mapping


@dataclass
class _A2AState:
    phase: str = "idle"  # idle | dispatched
    local_num_tokens: int | None = None
    combine_payload_offset: int | None = None


class MoeAlltoAll:
    """
    Manages MoE All-to-All operations with proper workspace allocation and synchronization.

    This class encapsulates the dispatch and combine operations, managing workspace memory
    and auxiliary data structures needed for cross-GPU communication.
    """

    # Single shared workspace/memory across the process
    _WORKSPACE: dict | None = None

    _METAINFO_INDEX: Dict[str, int] | None = None

    @classmethod
    def _init_constants(cls):
        """Initialize constants from C++ if not already done."""
        # TODO: Can we avoid such code duplication?
        if cls._METAINFO_INDEX is None:
            thop = _tllm_internal.thop
            cls._METAINFO_INDEX = {
                "FLAG_VAL_OFFSET_INDEX":
                int(thop.MOE_A2A_FLAG_VAL_OFFSET_INDEX),
                "LOCAL_TOKEN_COUNTER_OFFSET_INDEX":
                int(thop.MOE_A2A_LOCAL_TOKEN_COUNTER_OFFSET_INDEX),
                "SEND_COUNTERS_OFFSET_INDEX":
                int(thop.MOE_A2A_SEND_COUNTERS_OFFSET_INDEX),
                "RECV_COUNTERS_OFFSET_INDEX":
                int(thop.MOE_A2A_RECV_COUNTERS_OFFSET_INDEX),
                "DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX":
                int(thop.MOE_A2A_DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX),
                "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX":
                int(thop.MOE_A2A_COMBINE_COMPLETION_FLAGS_OFFSET_INDEX),
                "TOPK_TARGET_RANKS_OFFSET_INDEX":
                int(thop.MOE_A2A_TOPK_TARGET_RANKS_OFFSET_INDEX),
                "TOPK_SEND_INDICES_OFFSET_INDEX":
                int(thop.MOE_A2A_TOPK_SEND_INDICES_OFFSET_INDEX),
                "PAYLOAD_DATA_OFFSET_INDEX":
                int(thop.MOE_A2A_PAYLOAD_DATA_OFFSET_INDEX),
                "NUM_METAINFO_FIELDS":
                int(thop.MOE_A2A_NUM_METAINFO_FIELDS),
            }

    def __init__(
        self,
        mapping: Mapping,
        max_num_tokens: int,
        top_k: int,
        num_experts: int,
        # TODO: WE should be able to know the required workspace size if knowing max_num_tokens, ep_size and hidden_size
        workspace_size_per_rank: int = 256 * 1024 * 1024,
    ):
        """
        Initialize MoeAlltoAll with workspace allocation.

        Args:
            mapping: TensorRT-LLM Mapping object containing rank information
            max_num_tokens: Maximum number of tokens supported. Should be ModelConfig.max_num_tokens.
            workspace_size_per_rank: Size of workspace per rank in bytes
        """
        # Initialize constants from C++
        self._init_constants()

        # Initialize or reuse workspace
        MnnvlMemory.initialize()

        self.workspace_size_per_rank = workspace_size_per_rank
        self.max_num_tokens = max_num_tokens
        self.ep_size = mapping.moe_ep_size
        self.ep_rank = mapping.moe_ep_rank

        self.top_k = top_k
        self.num_experts = num_experts
        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("top_k must be a positive int")
        if not isinstance(self.num_experts, int) or self.num_experts <= 0:
            raise ValueError("num_experts must be a positive int")

        if self._WORKSPACE is None:
            tllm_logger.info(
                f"MoE AlltoAll: Allocating workspace with size {workspace_size_per_rank} bytes. ep_rank: {self.ep_rank}, ep_size: {self.ep_size}, max_num_tokens: {self.max_num_tokens}"
            )
            mnnvl_mem = MnnvlMemory(mapping, workspace_size_per_rank)
            workspace = mnnvl_mem.as_torch_strided_tensor(torch.uint8)
            metainfo = torch.ops.trtllm.moe_a2a_initialize(
                workspace,
                self.ep_rank,
                self.ep_size,
                self.max_num_tokens,
            )
            MoeAlltoAll._WORKSPACE = {
                "workspace_size_per_rank": workspace_size_per_rank,
                "max_num_tokens": self.max_num_tokens,
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
                "max_num_tokens"] == self.max_num_tokens, "reuse workspace with different max_num_tokens"
            assert self._WORKSPACE[
                "ep_rank"] == self.ep_rank, "reuse workspace with different ep_rank"
            assert self._WORKSPACE[
                "ep_size"] == self.ep_size, "reuse workspace with different ep_size"

        self.mnnvl_mem = self._WORKSPACE["mnnvl_mem"]
        self.workspace = self._WORKSPACE["workspace"]
        self.metainfo = self._WORKSPACE["metainfo"]
        # Internal state
        self._state: _A2AState = _A2AState()

    def dispatch(self,
                 token_selected_experts: torch.Tensor,
                 input_payloads: list[torch.Tensor],
                 runtime_max_tokens_per_rank: int,
                 invalid_token_expert_id: Optional[int] = None,
                 expert_id_payload_index: Optional[int] = None):
        """
        Perform MoE all-to-all dispatch operation.

        Args:
            token_selected_experts: [local_num_tokens, top_k] tensor of expert indices
            input_payloads: List of tensors to dispatch, each has shape [local_num_tokens, payload_num_elements_per_token]
            runtime_max_tokens_per_rank: Maximum of the number of tokens of each DP rank's local batch.
            invalid_token_expert_id: If not None, set the token_selected_experts of the invalid tokens to this expert id. This is used to notify the MoE to skip these tokens for GroupGEMM.
            expert_id_payload_index: The index of token_selected_experts in the input_payloads. Must be provided if invalid_token_expert_id is not None.

        Returns:
            recv_buffers: List of tensors received, each has shape [ep_size, max_tokens_per_rank, payload_num_elements_per_token]
        """
        assert self._state.phase == "idle", "dispatch called twice without an intervening combine"
        assert runtime_max_tokens_per_rank <= self.max_num_tokens, "runtime_max_tokens_per_rank must not exceed max_num_tokens"
        recv_tensors, combine_payload_offset = torch.ops.trtllm.moe_a2a_dispatch(
            token_selected_experts, input_payloads, self.workspace,
            self.metainfo, runtime_max_tokens_per_rank, self.ep_rank,
            self.ep_size, self.top_k, self.num_experts)
        # Update state together after successful dispatch
        self._state.local_num_tokens = token_selected_experts.size(0)
        self._state.combine_payload_offset = combine_payload_offset
        self._state.phase = "dispatched"

        if invalid_token_expert_id is not None:
            assert expert_id_payload_index is not None, "expert_id_payload_index must be provided if invalid_token_expert_id is not None"
            # Sanitize expert IDs for invalid tokens directly on the recv tensor payload
            recv_token_selected_experts = recv_tensors[expert_id_payload_index]
            torch.ops.trtllm.moe_a2a_sanitize_expert_ids(
                recv_token_selected_experts,
                self.workspace,
                self.metainfo,
                self.ep_rank,
                invalid_token_expert_id,
            )

        return recv_tensors

    def combine(
        self,
        payload,
        runtime_max_tokens_per_rank: int,
        payload_in_workspace: bool = False,
    ):
        """
        Perform MoE all-to-all combine operation.

        Args:
            payload: [ep_size, max_tokens_per_rank, num_elements_per_token] tensor to combine. The dtype must be float32, bfloat16 or float16.
            runtime_max_tokens_per_rank: Maximum of the number of tokens of each DP rank's local batch.
            payload_in_workspace: If True, 'payload' is a view into 'workspace' at 'combine_payload_offset' and no staging copy is needed. If False, the op stages 'payload' into the workspace region before combining.

        Returns:
            combined_output: [local_num_tokens, num_elements_per_token] tensor of combined results
        """
        assert self._state.phase == "dispatched", "combine called before a successful dispatch"
        assert runtime_max_tokens_per_rank <= self.max_num_tokens, "runtime_max_tokens_per_rank must not exceed max_num_tokens"

        output = torch.ops.trtllm.moe_a2a_combine(
            payload, self._state.local_num_tokens, self.workspace,
            self.metainfo, runtime_max_tokens_per_rank, self.ep_rank,
            self.ep_size, self.top_k, self._state.combine_payload_offset,
            payload_in_workspace)

        # Reset state for next round
        self._state = _A2AState()

        return output

    def get_combine_payload_tensor_in_workspace(
            self, runtime_max_tokens_per_rank: int, hidden_size: int,
            dtype: torch.dtype) -> torch.Tensor:
        """
        Return the combine payload tensor in the workspace, which could be used as the output of MoE kernel to avoid extra copy.
        See "payload_in_workspace" in combine method.
        """
        if self._state.phase != "dispatched":
            raise RuntimeError(
                "get_combine_payload_tensor_in_workspace called before a successful dispatch"
            )

        return torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
            self.workspace,
            self.ep_rank,
            self.ep_size,
            runtime_max_tokens_per_rank,
            self._state.combine_payload_offset,
            dtype,
            hidden_size,
        )
