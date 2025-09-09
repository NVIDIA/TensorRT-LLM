"""
MoE All-to-All Operations

This module provides a high-level interface for MoE all-to-all dispatch and combine operations
with proper workspace management and synchronization.
"""

import torch
from tensorrt_llm._mnnvl_utils import MnnvlMemory
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
    
    # MetaInfo indices - initialized from C++ constants
    FLAG_VAL_OFFSET_INDEX = None
    LOCAL_TOKEN_COUNTER_OFFSET_INDEX = None
    SEND_COUNTERS_OFFSET_INDEX = None
    COMPLETION_FLAGS_OFFSET_INDEX = None
    SEND_INDICES_OFFSET_INDEX = None
    PAYLOAD_DATA_OFFSET_INDEX = None
    
    @classmethod
    def _init_constants(cls):
        """Initialize constants from C++ if not already done."""
        if cls.FLAG_VAL_OFFSET_INDEX is None:
            cls.FLAG_VAL_OFFSET_INDEX = torch.ops.trtllm.MOE_A2A_FLAG_VAL_OFFSET_INDEX()
            cls.LOCAL_TOKEN_COUNTER_OFFSET_INDEX = torch.ops.trtllm.MOE_A2A_LOCAL_TOKEN_COUNTER_OFFSET_INDEX()
            cls.SEND_COUNTERS_OFFSET_INDEX = torch.ops.trtllm.MOE_A2A_SEND_COUNTERS_OFFSET_INDEX()
            cls.COMPLETION_FLAGS_OFFSET_INDEX = torch.ops.trtllm.MOE_A2A_COMPLETION_FLAGS_OFFSET_INDEX()
            cls.SEND_INDICES_OFFSET_INDEX = torch.ops.trtllm.MOE_A2A_SEND_INDICES_OFFSET_INDEX()
            cls.PAYLOAD_DATA_OFFSET_INDEX = torch.ops.trtllm.MOE_A2A_PAYLOAD_DATA_OFFSET_INDEX()
    
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
        
        # Initialize MNNVL memory
        MnnvlMemory.initialize()
        self.mnnvl_mem = MnnvlMemory(mapping, workspace_size_per_rank)
        self.workspace = self.mnnvl_mem.as_torch_strided_tensor(torch.uint8)
        
        # Initialize and get metainfo
        # moe_a2a_metainfo contains offsets indexed by class constants:
        # FLAG_VAL_OFFSET_INDEX, LOCAL_TOKEN_COUNTER_OFFSET_INDEX, SEND_COUNTERS_OFFSET_INDEX,
        # COMPLETION_FLAGS_OFFSET_INDEX, SEND_INDICES_OFFSET_INDEX, PAYLOAD_DATA_OFFSET_INDEX
        # This provides robust binding between Python and C++ without hardcoded indices
        self.moe_a2a_metainfo = torch.ops.trtllm.moe_a2a_initialize(
            self.workspace,
            self.ep_rank,
            self.ep_size,
            self.max_num_tokens_per_rank
        )
        # Internal state and aux data
        self.send_indices: torch.Tensor | None = None
        self.send_counters: torch.Tensor | None = None
        self.recv_counters: torch.Tensor | None = None
        self._state: str = "idle"  # idle | dispatched
            
    def dispatch(self, token_selected_experts, input_payloads):
        """
        Perform MoE all-to-all dispatch operation.
        
        Args:
            token_selected_experts: [local_num_tokens, top_k] tensor of expert indices
            input_payloads: List of tensors to dispatch, each has shape [local_num_tokens, payload_num_elements_per_token]
            
        Returns:
            recv_buffers: List of tensors received, each has shape [ep_size, max_tokens_per_rank, payload_num_elements_per_token]
        """
        if self._state == "dispatched":
            raise RuntimeError("dispatch called twice without an intervening combine")

        recv_buffers, send_counters, send_indices, recv_counters = torch.ops.trtllm.moe_a2a_dispatch(
            token_selected_experts,
            input_payloads,
            self.workspace,
            self.max_num_tokens_per_rank,
            self.ep_rank,
            self.ep_size,
            self.top_k,
            self.num_experts
        )
        self._state = "dispatched"
        self.send_indices = send_indices
        self.send_counters = send_counters
        self.recv_counters = recv_counters
        return recv_buffers
        
    def combine(self, payload):
        """
        Perform MoE all-to-all combine operation.
        
        Args:
            payload: [ep_size, max_tokens_per_rank, num_elements_per_token] tensor to combine. The dtype must be float32, bfloat16 or float16.
            
        Returns:
            combined_output: [local_num_tokens, num_elements_per_token] tensor of combined results
        """
        if self._state != "dispatched" or self.send_indices is None:
            raise RuntimeError("combine called before a successful dispatch")

        output = torch.ops.trtllm.moe_a2a_combine(
            self.send_indices,
            payload,
            self.workspace,
            self.max_num_tokens_per_rank,
            self.ep_rank,
            self.ep_size,
            self.top_k
        )
        # Reset state for next round
        self.send_indices = None
        self._state = "idle"
        return output
