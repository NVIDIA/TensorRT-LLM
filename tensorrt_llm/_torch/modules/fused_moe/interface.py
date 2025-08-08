from abc import abstractmethod
from enum import Enum
from typing import Dict, List, Optional

import torch
from torch import nn

from ...distributed.ops import reducescatter
from ...model_config import ModelConfig
from .routing import BaseMoeRoutingMethod


class MoEWeightLoadingMode(Enum):
    VANILLA = 0
    FUSED_GATE_UP_PROJ = 1


class MoE(nn.Module):
    """
    Fused Mixture of Experts (MoE) Layer interface.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        aux_stream (Optional[torch.cuda.Stream]): Auxiliary CUDA stream to overlap chunks.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.
    """

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
        VANILLA,
        bias: bool = False,
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        swiglu_limit: Optional[torch.Tensor] = None,
    ):
        from ...distributed import AllReduce

        super().__init__()
        self.routing_method = routing_method
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.weight_loading_mode = weight_loading_mode
        self.bias = bias
        self.dtype = dtype
        self.reduce_results = reduce_results
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.swiglu_limit = swiglu_limit

        # could be modified later
        self.quant_config = model_config.quant_config

        self.cluster_rank = model_config.mapping.moe_cluster_rank
        self.cluster_size = model_config.mapping.moe_cluster_size
        self.smart_router = True if self.cluster_size > 1 else False

        self.rank = model_config.mapping.rank

        self.tp_rank = model_config.mapping.moe_tp_rank
        self.tp_size = model_config.mapping.moe_tp_size

        self.ep_size = model_config.mapping.moe_ep_size
        self.ep_rank = model_config.mapping.moe_ep_rank

        self.moe_backend = model_config.moe_backend
        self.use_dp = model_config.mapping.enable_attention_dp

        # All ranks participate in allreduce regardless of EP/TP combination
        self.mapping = model_config.mapping
        self.parallel_size = self.mapping.tp_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size

        self.all_reduce = AllReduce(mapping=self.mapping,
                                    strategy=model_config.allreduce_strategy,
                                    dtype=self.dtype)

    @abstractmethod
    def create_weights(self):
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, weights: List[Dict]):
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    @property
    def has_any_quant(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
            exclude_kv_cache=True)

    # The following three properties are common enough to warrant inclusion in the interface.
    @property
    def has_fp8_qdq(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_fp8_qdq(
        )

    @property
    def has_deepseek_fp8_block_scales(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_fp8_block_scales(
        )

    @property
    def has_nvfp4(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_nvfp4(
        )

    @property
    def has_w4a8_mxfp4_fp8(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_w4a8_mxfp4_fp8(
        )

    @property
    def has_w4a8_mxfp4_mxfp8(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_w4a8_mxfp4_mxfp8(
        )

    @property
    def has_w4a16_mxfp4(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_w4a16_mxfp4(
        )

    @property
    def enable_alltoall(self):
        """ enable_alltoall (bool): whether to enable alltoall instead of allgather/reducescatter
        """
        return False

    def reducescatter_or_allreduce(
        self,
        inputs,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
    ):
        """
        Common helper for TP and EP in subclasses of the MoE module.
        """
        outputs = inputs
        if self.parallel_size > 1 and not self.enable_alltoall:
            if self.use_dp:
                outputs = reducescatter(
                    inputs,
                    self.mapping,
                    dim=0,
                    sizes=None if use_dp_padding else all_rank_num_tokens)
            elif self.reduce_results:
                outputs = self.all_reduce(inputs)
        return outputs
