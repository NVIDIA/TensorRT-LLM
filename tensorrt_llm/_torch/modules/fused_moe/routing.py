import math
from enum import IntEnum
from typing import Optional

import torch
from torch import nn

# Global cache for perfect router logits to share across all MLP blocks
_PERFECT_ROUTER_LOGITS_CACHE = {}


def get_perfect_router_cache_stats():
    """Get statistics about the perfect router cache."""
    global _PERFECT_ROUTER_LOGITS_CACHE

    if not _PERFECT_ROUTER_LOGITS_CACHE:
        return {
            "cache_size": 0,
            "memory_usage_mb": 0.0,
            "cached_batch_sizes": []
        }

    total_memory = 0
    cached_batch_sizes = []

    for (num_tokens, num_experts, experts_per_token,
         moe_ep_size), logits in _PERFECT_ROUTER_LOGITS_CACHE.items():
        total_memory += logits.numel() * logits.element_size()
        cached_batch_sizes.append(num_tokens)

    return {
        "cache_size": len(_PERFECT_ROUTER_LOGITS_CACHE),
        "memory_usage_mb": total_memory / (1024 * 1024),
        "cached_batch_sizes": sorted(list(set(cached_batch_sizes)))
    }


def precompute_common_perfect_router_logits(num_experts: int,
                                            experts_per_token: int,
                                            moe_ep_size: int,
                                            dtype: torch.dtype):
    """
    Pre-compute logits for common batch sizes to avoid first-time computation overhead.
    Only precomputes if cache is empty (avoids redundant work across multiple MLPBlock instances).
    """
    # Check if cache is already populated (avoid redundant work)
    cache_stats = get_perfect_router_cache_stats()
    if cache_stats["cache_size"] > 0:
        return

    # Common batch sizes for different scenarios
    common_batch_sizes = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        384,
        512,
        640,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
        5120,
        6144,
        7168,
        8192  # Powers of 2 and common sizes
    ]

    print(
        f"Precomputing perfect router logits for {len(common_batch_sizes)} common batch sizes..."
    )

    # Precompute logits for common batch sizes using global cache
    for num_tokens in common_batch_sizes:
        try:
            # Use the global cache function which will handle CPU computation and caching
            get_cached_perfect_router_logits(
                num_tokens=num_tokens,
                num_experts=num_experts,
                experts_per_token=experts_per_token,
                moe_ep_size=moe_ep_size,
                device=torch.device('cpu'),  # Precompute on CPU
                dtype=dtype)

        except Exception as e:
            # Skip this batch size if computation fails
            print(
                f"Warning: Failed to precompute logits for batch size {num_tokens}: {e}"
            )
            continue

    # Print cache statistics
    final_stats = get_perfect_router_cache_stats()
    print(
        f"Perfect router cache initialized: {final_stats['cache_size']} entries, "
        f"{final_stats['memory_usage_mb']:.2f} MB memory usage")


def get_cached_perfect_router_logits(num_tokens: int, num_experts: int,
                                     experts_per_token: int, moe_ep_size: int,
                                     device: torch.device,
                                     dtype: torch.dtype) -> torch.Tensor:
    """
    Get cached perfect router logits, computing and caching if not found.
    Uses global cache to share across all MLP blocks.
    """
    global _PERFECT_ROUTER_LOGITS_CACHE

    cache_key = (num_tokens, num_experts, experts_per_token, moe_ep_size)

    if cache_key in _PERFECT_ROUTER_LOGITS_CACHE:
        # Return cached logits moved to the correct device
        cached_logits = _PERFECT_ROUTER_LOGITS_CACHE[cache_key]
        if cached_logits.device != device:
            cached_logits = cached_logits.to(device)
            # Update cache with device-specific version for future use
            _PERFECT_ROUTER_LOGITS_CACHE[cache_key] = cached_logits
        return cached_logits
    else:
        # Compute and cache new logits
        logits = create_renormalize_expert_load_balanced_logits(
            num_tokens=num_tokens,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            moe_ep_size=moe_ep_size,
            device=device,
            dtype=dtype)

        _PERFECT_ROUTER_LOGITS_CACHE[cache_key] = logits
        return logits


# The type of method in top-K routing, for use in torch custom op
# Please keep this in sync with the counterpart defined in cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h
class RoutingMethodType(IntEnum):
    # Default: Softmax -> TopK
    Default = 0,
    # Renormalize: TopK -> Softmax
    Renormalize = 1,
    # DeepSeekV3: Sigmoid -> RoutingBiasAdd -> Top2 in group -> Top4 groups -> Top8 experts from the Top4 groups
    DeepSeekV3 = 2,
    # Llama4: Top1 -> Sigmoid
    Llama4 = 3,
    # Qwen3: Softmax -> TopK -> Renormalize
    RenormalizeNaive = 4,
    # Unspecified
    Unspecified = 5.


class BaseMoeRoutingMethod(nn.Module):

    def apply(self, _router_logits) -> (torch.Tensor, torch.Tensor):
        """
        Applies the routing method to the router logits.
        Router logits are usually the output of the router Linear layer, but can be any type for more complex routing methods.
        Returns (token_selected_experts: torch.Tensor<int32>, token_final_scales: torch.Tensor<float32>):
            token_selected_experts: shape (num_tokens, experts_per_token).
                It is a list of selected expert indices for each token
            token_final_scales: shape (num_tokens, experts_per_token). May be None
                It contains a final scaling/weighting factor applied to the output of each selected expert before summing the results
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_experts_per_token(self) -> int:
        return self.top_k

    @property
    def experts_per_token(self) -> int:
        return self.get_experts_per_token()

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.Unspecified


class DefaultMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(self,
                 top_k: int,
                 output_dtype: torch.dtype = torch.float32,
                 force_enable_pytorch_op: bool = False):
        super().__init__()
        self.top_k = top_k
        self.force_enable_pytorch_op = force_enable_pytorch_op
        self.output_dtype = output_dtype

    def apply_pytorch(
            self, router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        topk_values, topk_indices = torch.topk(torch.nn.functional.softmax(
            router_logits.to(self.output_dtype), dim=-1),
                                               k=self.top_k,
                                               dim=-1)
        return topk_indices.to(torch.int32), topk_values

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        num_experts = router_logits.shape[-1]
        if self.force_enable_pytorch_op or num_experts > 128 or self.top_k > 8:
            return self.apply_pytorch(router_logits)
        else:
            return torch.ops.trtllm.default_moe_routing_op(
                router_logits, self.top_k, self.output_dtype)

    @property
    def routing_method_type(self):
        return RoutingMethodType.Default


class DeepSeekV3MoeRoutingMethod(BaseMoeRoutingMethod):

    # Intentionally leave apply() unimplemented.
    # See comments in DeepseekV3Gate on why routing is done by DeepseekV3Gate.
    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    @property
    def routing_method_type(self):
        return RoutingMethodType.DeepSeekV3


class RenormalizeMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(
        self,
        top_k: int,
        output_dtype: torch.dtype = torch.float32,
        force_enable_pytorch_op: bool = False,
    ):
        super().__init__()
        self.top_k = top_k
        self.force_enable_pytorch_op = force_enable_pytorch_op
        self.output_dtype = output_dtype

    def apply_pytorch(
            self, router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        topk_values, topk_indices = torch.topk(router_logits,
                                               k=self.top_k,
                                               dim=-1)
        return topk_indices.to(torch.int32), torch.nn.functional.softmax(
            topk_values.to(self.output_dtype), dim=-1)

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        num_experts = router_logits.shape[-1]
        if self.force_enable_pytorch_op or num_experts > 128 or self.top_k > 8:
            return self.apply_pytorch(router_logits)
        else:
            return torch.ops.trtllm.renorm_moe_routing_op(
                router_logits, self.top_k, self.output_dtype)

    @property
    def routing_method_type(self):
        return RoutingMethodType.Renormalize


class Llama4RenormalizeMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(self, top_k: int, output_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.top_k = top_k
        self.output_dtype = output_dtype

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        topk_values, topk_indices = torch.topk(router_logits,
                                               k=self.top_k,
                                               dim=-1)
        return topk_indices.to(torch.int32), torch.sigmoid(
            topk_values.float()).to(self.output_dtype)

    @property
    def routing_method_type(self):
        return RoutingMethodType.Llama4


# TODO: re-enable this once the custom op is working.
# class Llama4RenormalizeMoeRoutingMethod(BaseMoeRoutingMethod):

#     def __init__(self, top_k: int, num_experts_total: int, ep_size: int,
#                  ep_rank: int):
#         super().__init__()
#         self.top_k = top_k
#         self.num_experts_total = num_experts_total
#         self.num_experts_per_node = self.num_experts_total // ep_size
#         self.start_expert = self.num_experts_per_node * ep_rank
#         self.end_expert = self.start_expert + self.num_experts_per_node

#     def apply(self,
#               router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
#         unpermuted_scales, indices = torch.ops.trtllm.fused_topk_softmax(
#             router_logits, self.top_k, self.num_experts_total,
#             self.start_expert, self.end_expert)
#         return indices, unpermuted_scales


# TODO Test this for Phi models
class SparseMixerMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(self, top_k: int, eps: float):
        super().__init__()
        self.top_k = top_k
        self.eps = eps

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        router_logits = router_logits.float()
        topk_values = torch.empty(router_logits.shape[0],
                                  self.top_k,
                                  device=router_logits.device,
                                  dtype=torch.float32)
        topk_indices = torch.empty(router_logits.shape[0],
                                   self.top_k,
                                   device=router_logits.device,
                                   dtype=torch.int32)
        for i in range(self.top_k):
            if i > 0:
                max_elem = torch.argmax(router_logits, dim=-1)
                # Mask out the previously selected indices to negative infinity
                router_logits.scatter_(-1, max_elem.unsqueeze(-1),
                                       -float('inf'))
            # Get the max value of the remaining indices
            max_values, max_indices = torch.max(router_logits,
                                                dim=-1,
                                                keepdim=True)
            assert torch.all(max_values != -float('inf'))

            topk_indices[:, i] = max_indices.squeeze(-1)

            # Mask out any values that fail the condition '(max - value) / std::max(abs(value), max) > 2 * epsilon'
            mask = (
                (max_values - router_logits) /
                torch.max(torch.abs(router_logits), max_values)) > 2 * self.eps
            masked_logits = torch.where(mask, -float('inf'), router_logits)
            softmax_masked_logits = torch.nn.functional.softmax(masked_logits,
                                                                dim=-1)
            selected_values = torch.gather(softmax_masked_logits, -1,
                                           max_indices)
            topk_values[:, i] = selected_values.squeeze(-1)

        return topk_indices.to(torch.int32), topk_values


class StaticMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(self,
                 routing_tensor: torch.Tensor,
                 routing_scales: Optional[torch.Tensor] = None):
        super().__init__()
        assert routing_tensor.dtype == torch.int32
        if routing_scales is not None:
            assert routing_tensor.shape[0] == routing_scales.shape[0]
            assert routing_tensor.shape[1] == routing_scales.shape[1]
            assert routing_scales.dtype == torch.float32
        self.routing_tensor = routing_tensor
        self.routing_scales = routing_scales

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        return self.routing_tensor, self.routing_scales

    def get_experts_per_token(self):
        return self.routing_tensor.shape[1]


class LoadBalancedMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        balanced_values = torch.ones(router_logits.shape[0],
                                     self.top_k,
                                     device=router_logits.device,
                                     dtype=torch.float32)
        balanced_indices = torch.empty(router_logits.shape[0],
                                       self.top_k,
                                       device=router_logits.device,
                                       dtype=torch.int32)

        # Fill the balanced_indices with each expert in round-robin fashion
        final_size = router_logits.shape[0] * self.top_k
        repeat_count = math.ceil(final_size / router_logits.shape[1])
        indices = torch.arange(router_logits.shape[1],
                               device=router_logits.device,
                               dtype=torch.int32)
        indices = indices.repeat(repeat_count)
        indices = indices[:final_size]
        balanced_indices = indices.view(router_logits.shape[0],
                                        self.top_k).contiguous()

        return balanced_indices, balanced_values


class RenormalizeNaiveMoeRoutingMethod(RenormalizeMoeRoutingMethod):

    def __init__(self, top_k: int, output_dtype: torch.dtype = torch.float32):
        super().__init__(top_k, output_dtype)
        self.top_k = top_k
        self.output_dtype = output_dtype

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        #x = topk(softmax()); x /= x.sum() is mathematically equivalent to softmax(topk)
        topk_indices, topk_values = self.apply_pytorch(router_logits)
        return topk_indices.to(torch.int32), topk_values.to(self.output_dtype)

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.RenormalizeNaive


def create_renormalize_expert_load_balanced_logits(
        num_tokens: int,
        num_experts: int,
        experts_per_token: int,
        moe_ep_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Create ideal logits that produce GPU-aware expert load balanced assignment for RenormalizeMoeRoutingMethod.

    This function is specifically designed to work with RenormalizeMoeRoutingMethod, which applies
    TopK selection first, then Softmax normalization on the selected experts. The function generates
    logits with high values for the desired experts and low values for others, ensuring that the
    TopK selection picks the intended experts for perfect load balancing.

    This is a GPU-optimized version that avoids Python loops.

    The function creates routing logits that ensure perfect load balancing across GPUs
    by cycling through experts in a GPU-aware pattern. Each token is assigned to
    exactly k=experts_per_token experts, distributed evenly across all GPUs.

    Strategy:
    1. First cycle through one expert from each GPU (GPU representatives)
    2. Then move to the next expert on each GPU, and so on
    3. This ensures even distribution of work across all GPUs

    Example 1: num_gpus=4, num_experts=8, experts_per_token=2, tokens=3
    experts_per_gpu = 8 // 4 = 2
    gpu_representatives = [0, 2, 4, 6] (first expert from each GPU)
    final_size = 3 * 2 = 6 (total expert assignments needed)

    | i_tensor | gpu_idx | expert_offset | indices | Explanation |
    |----------|---------|---------------|---------|-------------|
    | 0        | 0       | 0             | 0       | GPU 0, expert 0 |
    | 1        | 1       | 0             | 2       | GPU 1, expert 0 |
    | 2        | 2       | 0             | 4       | GPU 2, expert 0 |
    | 3        | 3       | 0             | 6       | GPU 3, expert 0 |
    | 4        | 0       | 1             | 1       | GPU 0, expert 1 |
    | 5        | 1       | 1             | 3       | GPU 1, expert 1 |

    Reshaped to (3, 2): [[0, 2], [4, 6], [1, 3]]
    Token 0 -> experts [0, 2], Token 1 -> experts [4, 6], Token 2 -> experts [1, 3]

    Final GPU Load Balance (Example 1):
    - GPU 0: 2 expert calls (expert 0 from token 0, expert 1 from token 2)
    - GPU 1: 2 expert calls (expert 0 from token 0, expert 1 from token 2)
    - GPU 2: 1 expert call (expert 0 from token 1)
    - GPU 3: 1 expert call (expert 0 from token 1)
    Note: Slight imbalance due to (3 tokens * 2 experts = 6 total work units) not being divisible by EP size (4 GPUs)

    Example 2: num_gpus=4, num_experts=8, experts_per_token=2, tokens=4
    experts_per_gpu = 8 // 4 = 2
    gpu_representatives = [0, 2, 4, 6]
    final_size = 4 * 2 = 8

    | i_tensor | gpu_idx | expert_offset | indices | Explanation |
    |----------|---------|---------------|---------|-------------|
    | 0        | 0       | 0             | 0       | GPU 0, expert 0 |
    | 1        | 1       | 0             | 2       | GPU 1, expert 0 |
    | 2        | 2       | 0             | 4       | GPU 2, expert 0 |
    | 3        | 3       | 0             | 6       | GPU 3, expert 0 |
    | 4        | 0       | 1             | 1       | GPU 0, expert 1 |
    | 5        | 1       | 1             | 3       | GPU 1, expert 1 |
    | 6        | 2       | 1             | 5       | GPU 2, expert 1 |
    | 7        | 3       | 1             | 7       | GPU 3, expert 1 |

    Reshaped to (4, 2): [[0, 2], [4, 6], [1, 3], [5, 7]]
    Token 0 -> experts [0, 2], Token 1 -> experts [4, 6],
    Token 2 -> experts [1, 3], Token 3 -> experts [5, 7]

    Final GPU Load Balance (Example 2):
    - GPU 0: 2 expert calls (expert 0 from token 0, expert 1 from token 2)
    - GPU 1: 2 expert calls (expert 0 from token 0, expert 1 from token 2)
    - GPU 2: 2 expert calls (expert 0 from token 1, expert 1 from token 3)
    - GPU 3: 2 expert calls (expert 0 from token 1, expert 1 from token 3)
    Perfect balance: Each GPU handles exactly 2 expert calls

    Args:
        num_tokens: Number of tokens to route
        num_experts: Total number of experts
        experts_per_token: Number of experts each token should be routed to (top-k)
        moe_ep_size: Number of GPUs for MoE expert parallelism
        device: Device to create tensors on
        dtype: Data type for the logits tensor

    Returns:
        torch.Tensor: Logits tensor of shape [num_tokens, num_experts] with softmax-applied probabilities

    Raises:
        ValueError: If num_experts is not divisible by moe_ep_size or if moe_ep_size is zero
    """
    k = experts_per_token
    experts_per_gpu = num_experts // moe_ep_size
    # For expert load balance, only moe_ep_size is relevant. System could have multiple TP/gpus sharding each group of experts
    num_gpus = moe_ep_size

    # Validation checks
    if num_experts % moe_ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by moe_ep_size ({moe_ep_size})"
        )

    if moe_ep_size == 0:
        raise ValueError("moe_ep_size cannot be zero")

    # Create logits tensor on the same device and dtype as input
    # Shape: [num_tokens, num_experts] - will hold routing probabilities
    logits = torch.zeros(num_tokens, num_experts, device=device, dtype=dtype)

    # GPU-aware expert assignment: cycle through one expert from each GPU first
    final_size = num_tokens * k  # Total number of expert assignments needed

    # Create GPU representatives (first expert from each GPU): [0, 8, 16, 24, ...]
    # These are the starting expert indices for each GPU
    gpu_representatives = torch.arange(0,
                                       num_experts,
                                       experts_per_gpu,
                                       device=device)

    # Generate indices using GPU-aware pattern (vectorized)
    # i_tensor: sequential indices from 0 to final_size-1
    i_tensor = torch.arange(final_size, device=device)

    # gpu_idx: which GPU this assignment should go to (cycles through 0,1,2,3,0,1,2,3,...)
    gpu_idx = i_tensor % num_gpus

    # expert_offset: which expert within the GPU (0,0,0,0,1,1,1,1,2,2,2,2,...)
    # This ensures we use all experts from each GPU before moving to next expert
    expert_offset = (i_tensor // num_gpus) % experts_per_gpu

    # indices: actual expert indices by combining GPU base + offset
    indices = gpu_representatives[gpu_idx] + expert_offset

    # Reshape to (num_tokens, k) - each row contains k expert indices for that token
    expert_indices = indices.view(num_tokens, k)

    # Generate large values for selected experts (5-10 range)
    # These high values ensure the selected experts have high probability after softmax
    large_values = torch.full((num_tokens, k), 7.5, device=device, dtype=dtype)

    # Assign large values to selected expert positions
    # token_indices: [[0,0],[1,1],[2,2],...] for indexing tokens
    token_indices = torch.arange(num_tokens,
                                 device=device).unsqueeze(1).expand(-1, k)
    logits[token_indices, expert_indices] = large_values

    # Fill remaining positions with small values (0-1 range)
    # This ensures non-selected experts have low but non-zero probability
    mask = (logits == 0)
    logits[mask] = 0.5

    # Apply softmax to get probabilities
    # After softmax, selected experts will have high probability (~0.99)
    # while non-selected experts will have very low probability
    logits = torch.nn.functional.softmax(logits, dim=-1)

    return logits
