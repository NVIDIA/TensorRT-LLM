import math
from enum import IntEnum
from typing import Optional

import torch
from torch import nn


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

    def get_experts_per_token(self):
        return self.top_k

    @property
    def experts_per_token(self):
        return self.get_experts_per_token()

    @property
    def routing_method_type(self):
        return RoutingMethodType.Unspecified


class DefaultMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(self, top_k: int, force_enable_pytorch_op: bool = False):
        super().__init__()
        self.top_k = top_k
        self.force_enable_pytorch_op = force_enable_pytorch_op

    def apply_pytorch(
            self, router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        topk_values, topk_indices = torch.topk(torch.nn.functional.softmax(
            router_logits.float(), dim=-1),
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
                router_logits, self.top_k)

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
        force_enable_pytorch_op: bool = False,
    ):
        super().__init__()
        self.top_k = top_k
        self.force_enable_pytorch_op = force_enable_pytorch_op

    def apply_pytorch(
            self, router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        topk_values, topk_indices = torch.topk(router_logits,
                                               k=self.top_k,
                                               dim=-1)
        return topk_indices.to(torch.int32), torch.nn.functional.softmax(
            topk_values.float(), dim=-1)

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        num_experts = router_logits.shape[-1]
        if self.force_enable_pytorch_op or num_experts > 128 or self.top_k > 8:
            return self.apply_pytorch(router_logits)
        else:
            return torch.ops.trtllm.renorm_moe_routing_op(
                router_logits, self.top_k)

    @property
    def routing_method_type(self):
        return RoutingMethodType.Renormalize


class Llama4RenormalizeMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        topk_values, topk_indices = torch.topk(router_logits,
                                               k=self.top_k,
                                               dim=-1)
        return topk_indices.to(torch.int32), torch.sigmoid(topk_values.float())

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

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        #x = topk(softmax()); x /= x.sum() is mathematically equivalent to softmax(topk)
        topk_indices, topk_values = self.apply_pytorch(router_logits)
        return topk_indices.to(torch.int32), topk_values

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.RenormalizeNaive
