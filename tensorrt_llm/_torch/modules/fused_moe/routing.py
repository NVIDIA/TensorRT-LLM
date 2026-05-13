import math
import warnings
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Dict, Optional, Type

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm.llmapi.utils import enable_llm_debug

# Global cache for perfect router logits shared across MoE modules
_PERFECT_ROUTER_LOGITS_CACHE = {}


def _get_perfect_router_cache_signature(routing_method):
    """Return the planner-specific fields that affect cached perfect-router logits."""
    planner = _get_perfect_router_planner(routing_method)
    return planner.get_cache_signature(routing_method)


def _get_perfect_router_cache_key(num_tokens, num_experts, experts_per_token,
                                  moe_ep_size, ep_rank, dtype, routing_method):
    """Build a stable cache key for synthetic perfect-router logits."""
    return (num_tokens, num_experts, experts_per_token, moe_ep_size, ep_rank,
            dtype, *_get_perfect_router_cache_signature(routing_method))


@dataclass(frozen=True)
class PerfectRouterRequest:
    """Bundle the inputs needed to synthesize perfect-router logits.

    Attributes:
        num_tokens: Number of local tokens on the current EP rank.
        num_experts: Total number of experts across all EP ranks.
        experts_per_token: Number of experts selected for each token.
        moe_ep_size: Number of EP ranks participating in the all-to-all.
        ep_rank: Sender EP rank. Used to rotate the ideal receiver timeline.
        device: Device where the synthesized logits should live.
        dtype: Output dtype for the synthesized logits tensor.
    """
    num_tokens: int
    num_experts: int
    experts_per_token: int
    moe_ep_size: int
    ep_rank: int
    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True)
class _TokenGroupBeamState:
    """One beam-search candidate for DeepSeekV3 group planning."""

    assignments: tuple[int, ...]
    active_groups: tuple[int, ...]
    group_counts: dict[int, int]
    mismatch_total: int
    preference_total: int


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

    for cache_key, logits in _PERFECT_ROUTER_LOGITS_CACHE.items():
        total_memory += logits.numel() * logits.element_size()
        cached_batch_sizes.append(cache_key[0])

    return {
        "cache_size": len(_PERFECT_ROUTER_LOGITS_CACHE),
        "memory_usage_mb": total_memory / (1024 * 1024),
        "cached_batch_sizes": sorted(list(set(cached_batch_sizes)))
    }


def precompute_common_perfect_router_logits(num_experts: int,
                                            experts_per_token: int,
                                            moe_ep_size: int,
                                            dtype: torch.dtype,
                                            routing_method,
                                            ep_rank: int = 0):
    """
    Pre-compute logits for common batch sizes to avoid first-time computation overhead.

    The cache is keyed by routing signature, dtype, EP size, and sender rank, so
    this helper only fills in entries that are missing for the current setup.

    Args:
        num_experts: Total number of experts across all EP ranks.
        experts_per_token: Number of experts selected for each token.
        moe_ep_size: Number of EP ranks participating in routing.
        dtype: Dtype used for the synthesized logits.
        routing_method: Routing method whose perfect-router planner will be used.
        ep_rank: Sender EP rank used to rotate the routing assignment stream
            across ranks.
    """
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
        8192  # Common sizes
    ]

    missing_batch_sizes = [
        num_tokens
        for num_tokens in common_batch_sizes if _get_perfect_router_cache_key(
            num_tokens, num_experts, experts_per_token, moe_ep_size, ep_rank,
            dtype, routing_method) not in _PERFECT_ROUTER_LOGITS_CACHE
    ]
    if not missing_batch_sizes:
        return

    print(
        f"Precomputing perfect router logits for {len(missing_batch_sizes)} common batch sizes (ep_rank={ep_rank})..."
    )

    # Precompute logits for common batch sizes using global cache
    for num_tokens in missing_batch_sizes:
        try:
            # Use the global cache function which will handle CPU computation and caching
            get_cached_perfect_router_logits(
                num_tokens=num_tokens,
                num_experts=num_experts,
                experts_per_token=experts_per_token,
                moe_ep_size=moe_ep_size,
                ep_rank=ep_rank,
                device=torch.device('cpu'),  # Precompute on CPU
                dtype=dtype,
                routing_method=routing_method)

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
                                     ep_rank: int, device: torch.device,
                                     dtype: torch.dtype,
                                     routing_method) -> torch.Tensor:
    """
    Return cached perfect-router logits, computing and caching them on demand.

    The same cache is shared across MoE modules. A request is uniquely identified
    by the token count, expert geometry, dtype, sender rank, and the planner
    signature derived from the routing method.

    Args:
        num_tokens: Number of local tokens on the current rank.
        num_experts: Total number of experts across all EP ranks.
        experts_per_token: Number of experts selected for each token.
        moe_ep_size: Number of EP ranks.
        ep_rank: Sender EP rank used to rotate the routing assignment stream.
        device: Device where the caller wants to consume the logits.
        dtype: Dtype for the synthesized logits.
        routing_method: Routing method whose planner defines the synthetic logits.
    """
    global _PERFECT_ROUTER_LOGITS_CACHE

    # Planners may mutate the routing method in place (for example by replacing
    # the learned bias callable with a zero-bias version). Keep that mutation
    # active even on cache hits.
    _get_perfect_router_planner(routing_method).prepare_routing_method_in_place(
        routing_method, device)

    cache_key = _get_perfect_router_cache_key(num_tokens, num_experts,
                                              experts_per_token, moe_ep_size,
                                              ep_rank, dtype, routing_method)

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
        logits = create_load_balanced_logits(
            num_tokens=num_tokens,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            moe_ep_size=moe_ep_size,
            ep_rank=ep_rank,
            device=device,
            dtype=dtype,
            routing_method=routing_method)

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
    # MiniMaxM2: Sigmoid -> RoutingBiasAdd -> TopK -> Renormalize(without bias)
    MiniMax2 = 5,
    # SigmoidRenorm: Sigmoid -> TopK -> Renormalize
    SigmoidRenorm = 6,
    # Unspecified
    Unspecified = 7,


class BaseMoeRoutingMethod(nn.Module):

    def apply(self, _router_logits) -> tuple[torch.Tensor, torch.Tensor]:
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
            self,
            router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        topk_values, topk_indices = torch.topk(torch.nn.functional.softmax(
            router_logits.to(self.output_dtype), dim=-1),
                                               k=self.top_k,
                                               dim=-1)
        return topk_indices.to(torch.int32), topk_values

    def apply(self,
              router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_experts = router_logits.shape[-1]
        if self.force_enable_pytorch_op or num_experts > 512 or self.top_k > 16:
            return self.apply_pytorch(router_logits)
        else:
            return torch.ops.trtllm.default_moe_routing_op(
                router_logits, self.top_k, self.output_dtype)

    @property
    def routing_method_type(self):
        return RoutingMethodType.Default


class Deepseekv3RoutingImpl:

    def __init__(
            self,
            top_k: int,
            n_group: int,
            topk_group: int,
            routed_scaling_factor: float,
            is_fused: bool = True,  # fuse_routing_kernel
    ):
        super().__init__()
        self.top_k = top_k
        self.topk_group = topk_group
        self.n_group = n_group
        self.routed_scaling_factor = routed_scaling_factor
        self.is_fused = is_fused

    @staticmethod
    @torch.compile(options={"max-autotune": True})
    def get_scores(logits, e_score_correction_bias):
        scores = F.sigmoid(logits)
        scores_with_bias = scores + e_score_correction_bias
        if enable_llm_debug():
            has_nan = torch.isnan(scores_with_bias).any()
            if has_nan:
                warnings.warn(
                    "Detected NAN in the tensor scores_with_bias. Please check if it matches the expectation."
                )

        return scores, scores_with_bias

    def noaux_tc(self, logits, e_score_correction_bias):
        n_group = self.n_group

        _, num_experts = logits.shape
        if self.n_group > 1:
            experts_per_group = num_experts // n_group
            if (self.top_k > 8 or num_experts > 256 or experts_per_group > 32
                    or experts_per_group * self.topk_group > 256):
                if self.is_fused:
                    warnings.warn(
                        "The configuration is not supported by the fused routing kernel. We have to use the original pytorch implementation."
                    )
                self.is_fused = False
        elif num_experts > 1024 or self.top_k > 32:
            if self.is_fused:
                warnings.warn(
                    "The configuration is not supported by the fused routing kernel. We have to use the original pytorch implementation."
                )
            self.is_fused = False

        if not self.is_fused:
            # Short path for n_group == 1 and topk_group == 1.
            if self.n_group == 1 and self.topk_group == 1:
                scores, scores_with_bias = self.get_scores(
                    logits, e_score_correction_bias)
                _, topk_indices = torch.topk(scores_with_bias,
                                             k=self.top_k,
                                             dim=1)
                topk_values = torch.gather(scores, dim=1,
                                           index=topk_indices).type_as(scores)

                # Normalize and scale.
                topk_values_sum = torch.sum(topk_values, dim=-1,
                                            keepdim=True) + 1e-20
                topk_values = topk_values / topk_values_sum * self.routed_scaling_factor
                return topk_values, topk_indices

            # General case with pytorch implementation.
            scores, scores_with_bias = self.get_scores(logits,
                                                       e_score_correction_bias)
            scores_shape = list(scores_with_bias.shape)
            group_scores = torch.sum(torch.topk(
                scores_with_bias.view(scores_shape[:-1] +
                                      [n_group, scores_shape[-1] // n_group]),
                k=2,
                dim=-1,
                largest=True,
                sorted=True)[0],
                                     dim=-1)
            _, group_idx = torch.topk(group_scores,
                                      k=self.topk_group,
                                      dim=-1,
                                      largest=True,
                                      sorted=True)
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(-1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(
                scores_shape[:-1] +
                [n_group, scores_shape[-1] // n_group]).reshape(scores_shape)
            scores_with_bias = torch.where(
                score_mask.bool(), scores_with_bias,
                torch.tensor(float('-inf'),
                             dtype=scores_with_bias.dtype,
                             device=scores_with_bias.device))
            _, topk_idx = torch.topk(scores_with_bias,
                                     k=self.top_k,
                                     dim=-1,
                                     largest=True,
                                     sorted=True)
            new_mask = torch.zeros_like(scores)
            new_mask.scatter_(-1, topk_idx, 1)
            scores = scores * new_mask
            score_sum = torch.sum(scores, dim=-1, keepdim=True) + 1e-20
            scores = scores / score_sum * self.routed_scaling_factor
            topk_values, topk_indices = torch.topk(scores,
                                                   k=self.top_k,
                                                   dim=-1,
                                                   largest=True)
            return topk_values, topk_indices
        else:
            topk_values, topk_indices = torch.ops.trtllm.noaux_tc_op(
                logits, e_score_correction_bias, n_group, self.topk_group,
                self.top_k, self.routed_scaling_factor)
            return topk_values, topk_indices

    def apply(
        self, logits: torch.Tensor, e_score_correction_bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_values, topk_indices = self.noaux_tc(logits,
                                                  e_score_correction_bias)
        return topk_indices.to(torch.int32), topk_values.to(torch.float32)


class DeepSeekV3MoeRoutingMethod(BaseMoeRoutingMethod):

    # See comments in DeepseekV3Gate on why routing is done by DeepseekV3Gate.
    def __init__(
            self,
            top_k: int,
            n_group: int,
            topk_group: int,
            routed_scaling_factor: float,
            callable_e_score_correction_bias: Callable[[], torch.Tensor],
            is_fused: bool = True,  # fuse_routing_kernel
    ):
        super().__init__()
        self.routing_impl = Deepseekv3RoutingImpl(
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            is_fused=is_fused,
        )
        # Pass a callable to fetch the tensor from DeepseekV3Gate at runtime, ensuring it is on the correct device
        assert callable(callable_e_score_correction_bias)
        self.callable_e_score_correction_bias = callable_e_score_correction_bias

    def apply(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.routing_impl.apply(logits, self.e_score_correction_bias)

    @property
    def e_score_correction_bias(self) -> torch.Tensor:
        return self.callable_e_score_correction_bias()

    @property
    def top_k(self):
        return self.routing_impl.top_k

    @property
    def routing_method_type(self):
        return RoutingMethodType.DeepSeekV3


class MiniMaxM2MoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(
        self,
        top_k: int,
        num_experts: int,
        callable_e_score_correction_bias: Callable[[], torch.Tensor],
        output_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        assert callable(callable_e_score_correction_bias)
        self.callable_e_score_correction_bias = callable_e_score_correction_bias
        self.output_dtype = output_dtype

    @staticmethod
    @torch.compile(options={"max-autotune": True})
    def get_scores(logits, e_score_correction_bias):
        scores = F.sigmoid(logits)
        scores_with_bias = scores + e_score_correction_bias
        if enable_llm_debug():
            has_nan = torch.isnan(scores_with_bias).any()
            if has_nan:
                warnings.warn(
                    "Detected NAN in the tensor scores_with_bias. Please check if it matches the expectation."
                )

        return scores, scores_with_bias

    def apply(self,
              router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores, scores_with_bias = self.get_scores(router_logits,
                                                   self.e_score_correction_bias)
        _, topk_idx = torch.topk(scores_with_bias,
                                 k=self.top_k,
                                 dim=-1,
                                 sorted=False)
        top_k_weights = scores.gather(1, topk_idx)
        top_k_weights /= (top_k_weights.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx.to(torch.int32), top_k_weights.to(self.output_dtype)

    @property
    def e_score_correction_bias(self) -> torch.Tensor:
        return self.callable_e_score_correction_bias()

    @property
    def routing_method_type(self):
        return RoutingMethodType.MiniMax2


class SigmoidRenormMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(
        self,
        top_k: int,
        num_experts: int,
        renormalize: bool = True,
        output_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.renormalize = renormalize
        self.output_dtype = output_dtype

    def apply(self,
              router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.sigmoid(router_logits)
        topk_weights, topk_idx = torch.topk(scores,
                                            k=self.top_k,
                                            dim=-1,
                                            sorted=False)
        if self.renormalize:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx.to(torch.int32), topk_weights.to(self.output_dtype)

    @property
    def routing_method_type(self):
        return RoutingMethodType.SigmoidRenorm


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
            self,
            router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        topk_values, topk_indices = torch.topk(router_logits,
                                               k=self.top_k,
                                               dim=-1)
        return topk_indices.to(torch.int32), torch.nn.functional.softmax(
            topk_values.to(self.output_dtype), dim=-1)

    def apply(self,
              router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_experts = router_logits.shape[-1]
        if self.force_enable_pytorch_op or num_experts > 512 or self.top_k > 16:
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
              router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
#               router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
              router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
              router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.routing_tensor, self.routing_scales

    def get_experts_per_token(self):
        return self.routing_tensor.shape[1]


class LoadBalancedMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    def apply(self,
              router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
              router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        #x = topk(softmax()); x /= x.sum() is mathematically equivalent to softmax(topk)
        topk_indices, topk_values = self.apply_pytorch(router_logits)
        return topk_indices.to(torch.int32), topk_values.to(self.output_dtype)

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.RenormalizeNaive


ROUTING_METHOD_TYPE_TO_CLASS: Dict[RoutingMethodType,
                                   Type[BaseMoeRoutingMethod]] = {
                                       RoutingMethodType.Default:
                                       DefaultMoeRoutingMethod,
                                       RoutingMethodType.Renormalize:
                                       RenormalizeMoeRoutingMethod,
                                       RoutingMethodType.DeepSeekV3:
                                       DeepSeekV3MoeRoutingMethod,
                                       RoutingMethodType.Llama4:
                                       Llama4RenormalizeMoeRoutingMethod,
                                       RoutingMethodType.RenormalizeNaive:
                                       RenormalizeNaiveMoeRoutingMethod,
                                       RoutingMethodType.Unspecified:
                                       BaseMoeRoutingMethod,
                                       RoutingMethodType.MiniMax2:
                                       MiniMaxM2MoeRoutingMethod,
                                       RoutingMethodType.SigmoidRenorm:
                                       SigmoidRenormMoeRoutingMethod,
                                   }


class BasePerfectRouterPlanner:
    """Planner interface for synthesizing logits used by the perfect-router path.

    Runtime routing already lives behind ``BaseMoeRoutingMethod.apply()``. This
    companion abstraction does the inverse job: given a routing method, synthesize
    a logits tensor that will make that routing method produce a target
    load-balanced assignment.
    """

    def get_cache_signature(
            self,
            routing_method: BaseMoeRoutingMethod) -> tuple[RoutingMethodType]:
        """Return routing-method fields that influence cached synthetic logits."""
        return (routing_method.routing_method_type, )

    def prepare_routing_method_in_place(self,
                                        routing_method: BaseMoeRoutingMethod,
                                        device: torch.device) -> None:
        """Mutate ``routing_method`` in place before synthesizing perfect-router logits."""
        del routing_method, device

    def create_logits(self, routing_method: BaseMoeRoutingMethod,
                      request: PerfectRouterRequest) -> torch.Tensor:
        """Build a synthetic logits tensor for the given routing method."""
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def _force_zero_routing_bias_in_place(
        routing_method: BaseMoeRoutingMethod,
        device: torch.device,
    ) -> None:
        """Replace the routing bias callable on ``routing_method`` in place.

        Bias-aware routers add a learned correction bias after the base score
        transform. Perfect-router logits already encode the desired ranking, so
        the learned bias must be masked out before those logits are consumed.
        This helper mutates the routing-method instance so subsequent routing
        calls observe the zero-bias callable.
        """
        bias = routing_method.e_score_correction_bias
        zero_bias = torch.zeros_like(bias, device=device)
        routing_method.callable_e_score_correction_bias = lambda zero_bias=zero_bias: zero_bias

    @staticmethod
    def _scatter_target_logits(
        target_experts: torch.Tensor,
        target_values: torch.Tensor,
        num_experts: int,
        device: torch.device,
        dtype: torch.dtype,
        background_value: float,
    ) -> torch.Tensor:
        """Scatter per-token target values into a dense logits tensor."""
        num_tokens, experts_per_token = target_experts.shape
        logits = torch.full((num_tokens, num_experts),
                            background_value,
                            device=device,
                            dtype=dtype)
        token_indices = torch.arange(num_tokens,
                                     device=device).unsqueeze(1).expand(
                                         -1, experts_per_token)
        logits[token_indices, target_experts] = target_values.to(dtype=dtype)
        return logits


class FlatTopKPerfectRouterPlanner(BasePerfectRouterPlanner):
    """Planner for routing methods whose balanced targets form a flat expert stream.

    This covers routers where the ideal assignment can be described as a simple
    per-rank rotation over expert ids without any grouped-routing constraint.
    """

    def __init__(self,
                 *,
                 zero_routing_bias: bool = False,
                 high_value: float = 10.0,
                 low_value: float = -10.0) -> None:
        self._zero_routing_bias = zero_routing_bias
        self._high_value = high_value
        self._low_value = low_value

    def prepare_routing_method_in_place(self,
                                        routing_method: BaseMoeRoutingMethod,
                                        device: torch.device) -> None:
        """Mutate ``routing_method`` in place for flat perfect-router synthesis."""
        if self._zero_routing_bias:
            self._force_zero_routing_bias_in_place(routing_method, device)

    def create_logits(self, routing_method: BaseMoeRoutingMethod,
                      request: PerfectRouterRequest) -> torch.Tensor:
        """Create flat top-k logits that preserve the desired EP rotation."""
        del routing_method
        target_experts = self._get_balanced_expert_indices(
            request.num_tokens, request.num_experts, request.experts_per_token,
            request.moe_ep_size, request.ep_rank, request.device)
        return self._create_topk_based_logits(target_experts,
                                              request.num_experts,
                                              request.device, request.dtype,
                                              self._high_value, self._low_value)

    @staticmethod
    def _get_balanced_expert_indices(
        num_tokens: int,
        num_experts: int,
        experts_per_token: int,
        moe_ep_size: int,
        ep_rank: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate GPU-aware load balanced expert indices with rank-aware scheduling.

        Dispatch kernel background
        --------------------------
        In an EP (expert parallel) MoE, the dispatch kernel performs an all-to-all:
        every sender rank ships each token to the GPU that owns the chosen expert.
        The peak bandwidth of that all-to-all is bound by the *hottest* receiver.
        If naive top-k picks expert 0 first for every token, all sender ranks
        target GPU 0 in the same communication step -- a classic hotspot that
        serializes the transfer over a single NVLink/NIC pair.

        Balancing strategy
        ------------------
        Produce an assignment stream that cycles across GPUs first, experts second:

            step 0: GPU 0, step 1: GPU 1, step 2: GPU 2, step 3: GPU 3,
            step 4: GPU 0 (next expert on it), step 5: GPU 1 ...

        and then *rotate* this stream by ``ep_rank`` so every sender hits a
        different receiver at the same step. With R = num_gpus sender ranks, the
        receivers selected at step t form the permutation
        ``{(t + r) mod R : r in 0..R-1}`` -- a Latin square that saturates all R
        links in parallel.

        Variables
        ---------
        - ``slot``         : linear assignment index, 0..final_size-1
                             (final_size = num_tokens * k).
        - ``rotated_slot`` : ``slot + ep_rank``; the slot viewed after the
                             per-rank rotation. Drives the target GPU and expert.
        - ``gpu_idx``      : ``rotated_slot % num_gpus`` -- the receiver GPU.
        - ``expert_off``   : ``(rotated_slot // num_gpus) % experts_per_gpu``
                             -- which expert within that GPU.
        - ``expert``       : ``gpu_representatives[gpu_idx] + expert_off`` --
                             the global expert id written into the top-k output.

        Example: num_gpus=4, num_experts=8, experts_per_token=2, tokens=4
                 experts_per_gpu = 2, gpu_representatives = [0, 2, 4, 6]

        ep_rank=0 (no rotation)              ep_rank=1 (rotated by +1)
        slot rotated gpu expert_off expert   slot rotated gpu expert_off expert
          0     0    0      0        0        0     1    1      0        2
          1     1    1      0        2        1     2    2      0        4
          2     2    2      0        4        2     3    3      0        6
          3     3    3      0        6        3     4    0      1        1
          4     4    0      1        1        4     5    1      1        3
          5     5    1      1        3        5     6    2      1        5
          6     6    2      1        5        6     7    3      1        7
          7     7    3      1        7        7     8    0      0        0

        Receiver timeline (GPU each rank targets at every communication step):

            step :      0    1    2    3    4    5    6    7
            rank 0 ->  GPU0 GPU1 GPU2 GPU3 GPU0 GPU1 GPU2 GPU3
            rank 1 ->  GPU1 GPU2 GPU3 GPU0 GPU1 GPU2 GPU3 GPU0
            rank 2 ->  GPU2 GPU3 GPU0 GPU1 GPU2 GPU3 GPU0 GPU1
            rank 3 ->  GPU3 GPU0 GPU1 GPU2 GPU3 GPU0 GPU1 GPU2

        The flattened stream is the intended receiver schedule. Flat routers use
        equal high values for all selected experts, so the final top-k output
        keeps this receiver set but may permute the intra-token slot order.

        Args:
            num_tokens: Number of tokens to route
            num_experts: Total number of experts
            experts_per_token: Number of experts each token is routed to (top-k)
            moe_ep_size: Number of GPUs for MoE expert parallelism
            ep_rank: Sender EP rank used to rotate the assignment stream
                     (0 to moe_ep_size-1)
            device: Device to create tensors on

        Returns:
            torch.Tensor: Expert indices of shape [num_tokens, experts_per_token]

        Raises:
            ValueError: If moe_ep_size <= 0, num_experts not divisible by moe_ep_size,
                        or ep_rank out of range
        """
        if moe_ep_size <= 0:
            raise ValueError(
                f"moe_ep_size must be positive (cannot be zero or negative), got {moe_ep_size}"
            )
        if num_experts % moe_ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by moe_ep_size ({moe_ep_size})"
            )
        if not 0 <= ep_rank < moe_ep_size:
            raise ValueError(
                f"ep_rank ({ep_rank}) must be in [0, {moe_ep_size})")

        k = experts_per_token
        experts_per_gpu = num_experts // moe_ep_size
        num_gpus = moe_ep_size
        final_size = num_tokens * k

        # First expert id on each GPU: [0, experts_per_gpu, 2*experts_per_gpu, ...]
        gpu_representatives = torch.arange(0,
                                           num_experts,
                                           experts_per_gpu,
                                           device=device)

        # Raw assignment slots, one per (token, k) pair.
        slot = torch.arange(final_size, device=device)

        # Rotate the stream by ep_rank so every sender rank targets a distinct
        # receiver GPU at the same communication step (Latin-square pattern).
        rotated_slot = slot + ep_rank

        gpu_idx = rotated_slot % num_gpus  # receiver GPU
        expert_off = (rotated_slot //
                      num_gpus) % experts_per_gpu  # expert within GPU
        indices = gpu_representatives[gpu_idx] + expert_off  # global expert id

        return indices.view(num_tokens, k)

    @staticmethod
    def _create_topk_based_logits(
        target_experts: torch.Tensor,
        num_experts: int,
        device: torch.device,
        dtype: torch.dtype,
        high_value: float = 10.0,
        low_value: float = 0.0,
    ) -> torch.Tensor:
        """Create binary logits with high values for target experts and low values for others.

        With a sufficient gap between ``high_value`` and ``low_value``, this
        pattern preserves the desired expert set for flat routers.
        """
        num_tokens = target_experts.shape[0]
        experts_per_token = target_experts.shape[1]

        logits = torch.full((num_tokens, num_experts),
                            low_value,
                            device=device,
                            dtype=dtype)
        token_indices = torch.arange(num_tokens,
                                     device=device).unsqueeze(1).expand(
                                         -1, experts_per_token)
        logits[token_indices, target_experts] = high_value
        return logits


class DeepSeekV3PerfectRouterPlanner(BasePerfectRouterPlanner):
    """Planner for DeepSeekV3 grouped routing.

    DeepSeekV3 cannot consume the flat expert stream directly because the router
    first selects groups and only then selects experts within those groups. The
    planner therefore projects the ideal GPU timeline onto a valid per-token
    group plan before writing ranked logits for the selected experts.
    """

    def get_cache_signature(
        self, routing_method: BaseMoeRoutingMethod
    ) -> tuple[RoutingMethodType, Optional[int], Optional[int]]:
        """Include the effective group configuration in the cache signature."""
        actual_n_group, actual_topk_group = self._resolve_group_config(
            routing_method)
        return (routing_method.routing_method_type, actual_n_group,
                actual_topk_group)

    def prepare_routing_method_in_place(self,
                                        routing_method: BaseMoeRoutingMethod,
                                        device: torch.device) -> None:
        """Mutate ``routing_method`` in place for grouped perfect-router synthesis."""
        self._force_zero_routing_bias_in_place(routing_method, device)

    def create_logits(self, routing_method: BaseMoeRoutingMethod,
                      request: PerfectRouterRequest) -> torch.Tensor:
        """Create ranked logits that survive DeepSeekV3 group selection."""
        actual_n_group, actual_topk_group = self._resolve_group_config(
            routing_method)
        target_experts, target_values = self._project_targets(
            num_tokens=request.num_tokens,
            num_experts=request.num_experts,
            experts_per_token=request.experts_per_token,
            moe_ep_size=request.moe_ep_size,
            ep_rank=request.ep_rank,
            device=request.device,
            n_group=actual_n_group,
            topk_group=actual_topk_group)
        return self._create_ranked_topk_logits(target_experts, target_values,
                                               request.num_experts,
                                               request.device, request.dtype,
                                               -10.0)

    def _resolve_group_config(
        self,
        routing_method: BaseMoeRoutingMethod,
    ) -> tuple[int, int]:
        """Resolve DeepSeekV3 group settings from the routing method.

        Example:
            If ``routing_method.routing_impl`` stores ``n_group=8`` and
            ``topk_group=4``, this helper returns ``(8, 4)``.
        """
        routing_impl = getattr(routing_method, "routing_impl", None)
        if routing_impl is None or not hasattr(routing_impl,
                                               "n_group") or not hasattr(
                                                   routing_impl, "topk_group"):
            raise ValueError(
                "DeepSeekV3 perfect-router planning requires routing_method.routing_impl "
                "to provide n_group and topk_group")
        return routing_impl.n_group, routing_impl.topk_group

    def _project_targets(
        self,
        num_tokens: int,
        num_experts: int,
        experts_per_token: int,
        moe_ep_size: int,
        ep_rank: int,
        device: torch.device,
        n_group: int,
        topk_group: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the ideal GPU receiver timeline onto a valid DeepSeekV3 plan.

        The planner runs in three stages:
          1. Build the group-to-GPU topology implied by expert placement.
          2. Choose a valid group for each per-token slot while respecting
             ``topk_group``.
          3. Pick a concrete expert inside each chosen group and assign a ranked
             score so the grouped router keeps the intended ordering.

        Example:
            With ``num_experts=8``, ``moe_ep_size=2``, ``n_group=4``,
            ``topk_group=2``, ``experts_per_token=4``, and ``ep_rank=1``, the
            first token's ideal GPU timeline is ``[1, 0, 1, 0]``. A valid
            projection for that token is ``target_experts=[7, 0, 6, 1]`` with
            ranked ``target_values=[4.0, 3.0, 2.0, 1.0]``.

        Raises:
            ValueError: If the DeepSeekV3 group constraints cannot supply enough
                unique experts for each token.
        """
        group_gpu_experts, groups_by_gpu = self._build_group_topology(
            num_experts, moe_ep_size, n_group)
        experts_per_group = num_experts // n_group
        max_unique_experts = min(topk_group, n_group) * experts_per_group
        if experts_per_token > max_unique_experts:
            raise ValueError(
                "DeepSeekV3 perfect-router planning could not assign unique "
                "experts: experts_per_token "
                f"({experts_per_token}) exceeds the {max_unique_experts} "
                "unique experts available from "
                f"topk_group={topk_group} group(s) with "
                f"experts_per_group={experts_per_group}")
        desired_gpu_timeline = (
            (torch.arange(num_tokens * experts_per_token, dtype=torch.int64) +
             ep_rank) % moe_ep_size).view(num_tokens,
                                          experts_per_token).tolist()
        target_values = torch.linspace(4.0,
                                       1.0,
                                       steps=experts_per_token,
                                       device=device,
                                       dtype=torch.float32)
        target_experts = []

        for token_idx, desired_gpus in enumerate(desired_gpu_timeline):
            slot_groups = self._plan_token_groups(desired_gpus,
                                                  group_gpu_experts,
                                                  groups_by_gpu, topk_group,
                                                  token_idx, ep_rank)
            used_experts = set()
            token_targets = []

            for slot_idx, (desired_gpu, group_idx) in enumerate(
                    zip(desired_gpus, slot_groups)):
                # Prefer experts from the desired receiver GPU. If the selected
                # group does not cover that GPU, fall back to any expert in the
                # group, but still require token-local uniqueness.
                candidates = list(group_gpu_experts[group_idx][desired_gpu])
                if not candidates:
                    candidates = [
                        expert_idx
                        for gpu_experts in group_gpu_experts[group_idx]
                        for expert_idx in gpu_experts
                    ]
                ordered_candidates = self._rotate_list(
                    candidates, token_idx + ep_rank + slot_idx)
                expert_idx = next((expert for expert in ordered_candidates
                                   if expert not in used_experts), None)
                if expert_idx is None:
                    # The preferred candidates may already be consumed by this
                    # token. Retry across the full group before concluding that
                    # this configuration cannot satisfy the token uniquely.
                    all_group_experts = [
                        expert for gpu_experts in group_gpu_experts[group_idx]
                        for expert in gpu_experts
                    ]
                    ordered_group_experts = self._rotate_list(
                        all_group_experts, token_idx + ep_rank + slot_idx)
                    expert_idx = next((expert
                                       for expert in ordered_group_experts
                                       if expert not in used_experts), None)
                    if expert_idx is None:
                        raise ValueError(
                            "DeepSeekV3 perfect-router planning could not "
                            "assign unique experts for "
                            f"token {token_idx}, slot {slot_idx}, "
                            f"group {group_idx}, desired_gpu {desired_gpu}. "
                            f"used_experts={sorted(used_experts)}; "
                            f"group_experts={ordered_group_experts}")
                used_experts.add(expert_idx)
                token_targets.append(expert_idx)

            target_experts.append(token_targets)

        return (torch.tensor(target_experts, device=device, dtype=torch.int64),
                target_values.unsqueeze(0).expand(num_tokens, -1))

    @staticmethod
    def _build_group_topology(num_experts: int, moe_ep_size: int, n_group: int):
        """Build the DeepSeekV3 mapping from groups to the experts they expose.

        Returns:
            A tuple ``(group_gpu_experts, groups_by_gpu)`` where:
                - ``group_gpu_experts[group_idx][gpu_idx]`` is the list of
                  experts from ``group_idx`` that physically live on
                  ``gpu_idx``.
                - ``groups_by_gpu[gpu_idx]`` lists the groups that can satisfy
                  a slot whose ideal receiver is ``gpu_idx``.

        Example:
            With ``num_experts=8``, ``moe_ep_size=2``, and ``n_group=4``, the
            groups are ``[0, 1]``, ``[2, 3]``, ``[4, 5]``, and ``[6, 7]``.
            The returned topology is:
            ``group_gpu_experts = [[[0, 1], []], [[2, 3], []], [[], [4, 5]], [[], [6, 7]]]``
            and ``groups_by_gpu = [[0, 1], [2, 3]]``.

        Raises:
            ValueError: If ``moe_ep_size`` is non-positive or the expert layout
                cannot be partitioned evenly across groups and GPUs.
        """
        if moe_ep_size <= 0:
            raise ValueError(
                f"moe_ep_size must be positive (cannot be zero or negative), got {moe_ep_size}"
            )
        if num_experts % n_group != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by n_group ({n_group})"
            )
        if num_experts % moe_ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by moe_ep_size ({moe_ep_size})"
            )

        experts_per_group = num_experts // n_group
        experts_per_gpu = num_experts // moe_ep_size
        groups_by_gpu = [[] for _ in range(moe_ep_size)]
        group_gpu_experts = []

        for group_idx in range(n_group):
            group_start = group_idx * experts_per_group
            gpu_experts = [[] for _ in range(moe_ep_size)]
            for expert_idx in range(group_start,
                                    group_start + experts_per_group):
                gpu_idx = expert_idx // experts_per_gpu
                gpu_experts[gpu_idx].append(expert_idx)
            for gpu_idx, experts in enumerate(gpu_experts):
                if experts:
                    groups_by_gpu[gpu_idx].append(group_idx)
            group_gpu_experts.append(gpu_experts)

        return group_gpu_experts, groups_by_gpu

    @staticmethod
    def _beam_state_sort_key(
            state: _TokenGroupBeamState) -> tuple[int, int, int, int]:
        """Rank beam states by mismatch count, group reuse, preference, and breadth."""
        return (
            state.mismatch_total,
            max(state.group_counts.values(), default=0),
            state.preference_total,
            -len(state.active_groups),
        )

    def _plan_token_groups(
        self,
        desired_gpus,
        group_gpu_experts,
        groups_by_gpu,
        topk_group: int,
        token_idx: int,
        ep_rank: int,
    ):
        """Project the ideal GPU timeline onto a valid DeepSeekV3 group plan.

        The beam search prefers plans that:
          1. Match the ideal receiver GPU as often as possible.
          2. Spread assignments across the active groups instead of overusing
             one.
          3. Follow the per-slot group preference order induced by the rotated
             receiver timeline.

        Example:
            Suppose ``desired_gpus=[1, 0, 1, 0]`` and ``topk_group=2`` while the
            topology only allows groups ``[2, 3]`` on GPU1 and ``[0, 1]`` on
            GPU0. The planner may choose ``slot_groups=(3, 0, 3, 0)``, which
            uses only two groups while staying aligned with the desired GPU
            sequence as much as possible.
        """
        active_group_limit = min(topk_group, len(desired_gpus),
                                 len(group_gpu_experts))
        beam = [
            _TokenGroupBeamState(assignments=tuple(),
                                 active_groups=tuple(),
                                 group_counts={},
                                 mismatch_total=0,
                                 preference_total=0)
        ]
        beam_width = 32

        for slot_idx, desired_gpu in enumerate(desired_gpus):
            next_beam = []
            for state in beam:
                preferred_groups = self._rotate_list(
                    groups_by_gpu[desired_gpu], token_idx + ep_rank + slot_idx)
                if len(state.active_groups) < active_group_limit:
                    candidates = preferred_groups + [
                        group_idx for group_idx in state.active_groups
                        if group_idx not in preferred_groups
                    ]
                else:
                    candidates = [
                        group_idx for group_idx in preferred_groups
                        if group_idx in state.active_groups
                    ]
                    if not candidates:
                        candidates = list(state.active_groups)

                for group_idx in candidates:
                    new_active_groups = (state.active_groups if group_idx
                                         in state.active_groups else
                                         state.active_groups + (group_idx, ))
                    if len(new_active_groups) > active_group_limit:
                        continue

                    new_group_counts = dict(state.group_counts)
                    new_group_counts[group_idx] = new_group_counts.get(
                        group_idx, 0) + 1
                    new_mismatch_total = state.mismatch_total + int(
                        not group_gpu_experts[group_idx][desired_gpu])
                    preference_rank = preferred_groups.index(
                        group_idx) if group_idx in preferred_groups else len(
                            preferred_groups)
                    next_beam.append(
                        _TokenGroupBeamState(
                            assignments=state.assignments + (group_idx, ),
                            active_groups=new_active_groups,
                            group_counts=new_group_counts,
                            mismatch_total=new_mismatch_total,
                            preference_total=state.preference_total +
                            preference_rank,
                        ))

            if not next_beam:
                raise RuntimeError(
                    "DeepSeekV3 projection planner failed to find a valid group assignment"
                )

            next_beam.sort(key=self._beam_state_sort_key)
            beam = next_beam[:beam_width]

        return min(beam, key=self._beam_state_sort_key).assignments

    @staticmethod
    def _rotate_list(values, offset: int):
        """Rotate a Python list by ``offset`` positions.

        Example:
            ``_rotate_list([4, 5, 6, 7], 1)`` returns ``[5, 6, 7, 4]``.
        """
        if not values:
            return []
        offset = offset % len(values)
        return values[offset:] + values[:offset]

    @staticmethod
    def _create_ranked_topk_logits(
        target_experts: torch.Tensor,
        target_values: torch.Tensor,
        num_experts: int,
        device: torch.device,
        dtype: torch.dtype,
        low_value: float = -10.0,
    ) -> torch.Tensor:
        """Create logits with per-slot scores for deterministic top-k ordering.

        ``DeepSeekV3`` needs a stronger signal than binary high/low logits
        because the grouped router first selects groups and then runs top-k
        inside the surviving groups. ``target_values`` therefore encodes an
        explicit ranking among the desired experts for each token.

        Example:
            If ``target_experts=[[7, 0, 6, 1]]`` and
            ``target_values=[[4.0, 3.0, 2.0, 1.0]]`` with ``num_experts=8``,
            this helper emits one logits row whose non-background entries are:
            ``expert 7 -> 4.0``, ``expert 0 -> 3.0``, ``expert 6 -> 2.0``, and
            ``expert 1 -> 1.0``. All other experts receive ``low_value``.
        """
        return BasePerfectRouterPlanner._scatter_target_logits(
            target_experts, target_values, num_experts, device, dtype,
            low_value)


PERFECT_ROUTER_PLANNERS: Dict[RoutingMethodType, BasePerfectRouterPlanner] = {
    RoutingMethodType.Default: FlatTopKPerfectRouterPlanner(),
    RoutingMethodType.Renormalize: FlatTopKPerfectRouterPlanner(),
    RoutingMethodType.RenormalizeNaive: FlatTopKPerfectRouterPlanner(),
    RoutingMethodType.Llama4: FlatTopKPerfectRouterPlanner(),
    RoutingMethodType.MiniMax2:
    FlatTopKPerfectRouterPlanner(zero_routing_bias=True),
    RoutingMethodType.DeepSeekV3: DeepSeekV3PerfectRouterPlanner(),
}


def _get_perfect_router_planner(
        routing_method: BaseMoeRoutingMethod) -> BasePerfectRouterPlanner:
    """Resolve the planner responsible for synthesizing logits for a router."""
    planner = PERFECT_ROUTER_PLANNERS.get(routing_method.routing_method_type)
    if planner is None:
        supported_types = ", ".join(
            routing_type.name
            for routing_type in PERFECT_ROUTER_PLANNERS.keys())
        raise ValueError(
            f"Unsupported routing method type: {routing_method.routing_method_type}. "
            f"Supported types: {supported_types}")
    return planner


def _build_perfect_router_logits(
    routing_method: BaseMoeRoutingMethod,
    request: PerfectRouterRequest,
) -> torch.Tensor:
    """Delegate perfect-router logit synthesis to the routing-specific planner."""
    planner = _get_perfect_router_planner(routing_method)
    planner.prepare_routing_method_in_place(routing_method, request.device)
    return planner.create_logits(routing_method, request)


def create_load_balanced_logits(
    num_tokens: int,
    num_experts: int,
    experts_per_token: int,
    moe_ep_size: int,
    ep_rank: int,
    device: torch.device,
    dtype: torch.dtype,
    routing_method: BaseMoeRoutingMethod,
) -> torch.Tensor:
    """
    Create logits that produce GPU-aware load balanced expert assignment.

    The heavy lifting is delegated to a routing-specific perfect-router planner.
    Each planner knows how to synthesize logits that survive that router's
    scoring pipeline while preserving the desired EP-aware expert assignment.

    For flat routers this is a simple high/low top-k construction. For grouped
    routers such as ``DeepSeekV3`` the planner must first project the ideal GPU
    receiver timeline onto a valid per-token group plan.

    The synthesized logits are balanced in two complementary ways:
      1. GPU balance: over the full local batch, the selected experts are spread
         as evenly as possible across the EP GPUs.
      2. Timeline balance: the receiver stream is rotated by ``ep_rank`` so
         sender rank ``r`` targets receiver GPU ``(t + r) % moe_ep_size`` at
         communication step ``t``. Different sender ranks therefore hit
         different GPUs at the same step instead of creating a hotspot on one
         receiver.

    Supported routing methods:
    - DefaultMoeRoutingMethod (softmax -> topk)
    - RenormalizeMoeRoutingMethod (topk -> softmax)
    - RenormalizeNaiveMoeRoutingMethod (softmax -> topk -> renorm)
    - Llama4RenormalizeMoeRoutingMethod (topk -> sigmoid)
    - MiniMaxM2MoeRoutingMethod (sigmoid -> topk -> renorm, forces bias=0)
    - DeepSeekV3MoeRoutingMethod (sigmoid -> group routing, forces bias=0)

    Args:
        num_tokens: Number of tokens to route
        num_experts: Total number of experts
        experts_per_token: Number of experts each token is routed to (top-k)
        moe_ep_size: Number of GPUs for MoE expert parallelism
        ep_rank: Sender EP rank used to rotate the assignment stream
                 (0 to moe_ep_size-1)
        device: Device to create tensors on. Passing ``torch.device('cuda:N')``
            returns a GPU-resident tensor directly on that CUDA device.
        dtype: Data type for the logits tensor
        routing_method: The routing method instance to generate logits for.
            Grouped routers such as ``DeepSeekV3`` must already carry their
            group configuration on the routing method itself.

    Returns:
        torch.Tensor: Logits of shape ``[num_tokens, num_experts]`` allocated on
        ``device``.

        Typical output forms:
          - Flat routers may emit rows such as
            ``[-10, -10, 10, -10, 10, -10, -10, -10]`` where the large entries
            mark the intended top-k experts for that token.
          - Grouped routers such as ``DeepSeekV3`` may emit ranked rows such as
            ``[3, -10, 4, -10, -10, -10, -10, -10]`` where the relative
            magnitudes preserve expert ordering after group selection.

    Raises:
        ValueError: If the routing method type is not supported

    Note:
        Bias-aware planners may replace the routing bias callable on
        ``routing_method`` in place with a zero tensor so the synthesized
        ranking is not
        perturbed by learned correction terms.

    Example:
        >>> routing = RenormalizeMoeRoutingMethod(top_k=2)
        >>> logits = create_load_balanced_logits(
        ...     num_tokens=4, num_experts=8, experts_per_token=2,
        ...     moe_ep_size=4, ep_rank=1, device=torch.device('cuda:0'),
        ...     dtype=torch.float32, routing_method=routing
        ... )
        >>> logits.device
        device(type='cuda', index=0)
        >>> logits.shape
        torch.Size([4, 8])
        >>> logits[0].tolist()
        [-10.0, -10.0, 10.0, -10.0, 10.0, -10.0, -10.0, -10.0]
        >>> indices, scales = routing.apply(logits)
        >>> indices.shape
        torch.Size([4, 2])

        In this example the first token prefers experts 2 and 4, which live on
        different GPUs. Across the full batch the chosen experts cycle through
        all 4 GPUs, and the ``ep_rank=1`` rotation shifts the same-step receiver
        order so rank 1 starts from GPU 1 instead of GPU 0.
    """
    request = PerfectRouterRequest(num_tokens=num_tokens,
                                   num_experts=num_experts,
                                   experts_per_token=experts_per_token,
                                   moe_ep_size=moe_ep_size,
                                   ep_rank=ep_rank,
                                   device=device,
                                   dtype=dtype)
    return _build_perfect_router_logits(routing_method, request)
