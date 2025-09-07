"""
MoE All-to-All Operations

This module provides a high-level interface for MoE all-to-all dispatch and combine operations
with proper workspace management and synchronization.
"""

import torch
from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._utils import str_dtype_to_torch
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
    
    def __init__(self, mapping: Mapping, max_num_tokens_per_rank: int, workspace_size_per_rank: int = 256 * 1024 * 1024):
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
            
    def dispatch(self, token_selected_experts, input_payloads, max_tokens_per_rank, top_k, num_experts):
        """
        Perform MoE all-to-all dispatch operation.
        
        Args:
            token_selected_experts: [local_num_tokens, top_k] tensor of expert indices
            input_payloads: List of tensors to dispatch
            max_tokens_per_rank: Maximum tokens per rank
            top_k: Number of experts per token
            num_experts: Total number of experts
            
        Returns:
            tuple: (recv_buffers, send_counters, send_indices)
        """
        # The C++ implementation already handles auxiliary data allocation
        # based on the workspace layout we defined. Just call the operation.
        return torch.ops.trtllm.moe_a2a_dispatch(
            token_selected_experts,
            input_payloads,
            self.workspace,
            max_tokens_per_rank,
            self.ep_rank,
            self.ep_size,
            top_k,
            num_experts
        )
        
    def combine(self, send_indices, payload, max_tokens_per_rank, top_k):
        """
        Perform MoE all-to-all combine operation.
        
        Args:
            send_indices: [local_num_tokens, ep_size] tensor from dispatch
            payload: [ep_size, max_tokens_per_rank, elements] tensor to combine
            max_tokens_per_rank: Maximum tokens per rank
            top_k: Number of experts per token
            
        Returns:
            Combined output tensor
        """
        # The C++ implementation handles counter resets and synchronization
        return torch.ops.trtllm.moe_a2a_combine(
            send_indices,
            payload,
            self.workspace,
            max_tokens_per_rank,
            self.ep_rank,
            self.ep_size,
            top_k
        )
        
