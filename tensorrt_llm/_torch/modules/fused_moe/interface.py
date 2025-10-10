import weakref
from abc import abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Union, final

import torch
from torch import nn

from ...distributed.ops import reducescatter
from ...model_config import ModelConfig
from ...utils import (Fp4QuantizedTensor, get_model_extra_attrs,
                      is_torch_compiling)
from .routing import BaseMoeRoutingMethod


class MoEWeightLoadingMode(Enum):
    # Gate and up projection are not fused
    VANILLA = 0
    # Gate and up projection are fused
    FUSED_GATE_UP_PROJ = 1
    # Custom W4A8 weights from examples/quantization/quantize_mixed_precision_moe.py
    W4A8_CUSTOM = 2


def extract_extra_attrs(layer_idx: str):
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs are not set"

    moe_layers = extra_attrs.get("moe_layers", None)
    assert moe_layers is not None, "No MoE layers registered"
    moe_layer_ref = moe_layers.get(layer_idx)
    assert moe_layer_ref is not None, f"Cannot find MoE layer for layer_idx={layer_idx}"
    moe_layer = moe_layer_ref() if callable(moe_layer_ref) else None
    assert moe_layer is not None, f"MoE layer for layer_idx={layer_idx!r} is no longer alive"

    return moe_layer


@torch.library.custom_op("trtllm::moe_custom_op", mutates_args=())
def moe_custom_op(
    layer_idx: str,
    x: torch.Tensor,
    x_sf: Optional[torch.Tensor],
    is_swizzled: bool,
    router_logits: torch.Tensor,
    do_finalize: bool,
    output_dtype: Optional[torch.dtype],
    all_rank_num_tokens: Optional[List[int]],
    use_dp_padding: Optional[bool],
) -> List[torch.Tensor]:
    moe_layer = extract_extra_attrs(layer_idx)

    hidden_states = x if x_sf is None else Fp4QuantizedTensor(
        x, x_sf, is_swizzled)

    res = moe_layer.forward_impl(
        hidden_states,
        router_logits,
        do_finalize=do_finalize,
        output_dtype=output_dtype,
        all_rank_num_tokens=all_rank_num_tokens,
        use_dp_padding=use_dp_padding,
    )

    if do_finalize:
        return [res]
    else:
        return res


@moe_custom_op.register_fake
def _(
    layer_idx,
    x,
    x_sf,
    is_swizzled,
    router_logits,
    do_finalize,
    output_dtype,
    all_rank_num_tokens,
    use_dp_padding,
):
    moe_layer = extract_extra_attrs(layer_idx)
    hidden_states = x if x_sf is None else Fp4QuantizedTensor(
        x, x_sf, is_swizzled)
    res = moe_layer.forward_fake(
        hidden_states,
        router_logits,
        do_finalize=do_finalize,
        output_dtype=output_dtype,
        all_rank_num_tokens=all_rank_num_tokens,
        use_dp_padding=use_dp_padding,
    )

    if do_finalize:
        return [res]
    else:
        return res


class MoE(nn.Module):
    """
    Fused Mixture of Experts (MoE) Layer interface.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
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
        layer_idx: Optional[int] = None,
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
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx) if layer_idx is not None else None

        self._register_layer(model_config)

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

    def _register_layer(self, model_config: ModelConfig):
        self.register_to_config = False
        if model_config is not None and self.layer_idx_str is not None:
            if "moe_layers" not in model_config.extra_attrs:
                model_config.extra_attrs["moe_layers"] = {}
            assert self.layer_idx_str not in model_config.extra_attrs["moe_layers"], \
                f"Duplicate MoE layer for layer_idx={self.layer_idx_str}"
            model_config.extra_attrs["moe_layers"][
                self.layer_idx_str] = weakref.ref(self)
            self.register_to_config = True

    @abstractmethod
    def create_weights(self):
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, weights: List[Dict]):
        raise NotImplementedError

    def post_load_weights(self):
        pass

    @abstractmethod
    def forward_impl(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        raise NotImplementedError

    def forward_fake(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        is_nvfp4_input = isinstance(x, Fp4QuantizedTensor)
        assert do_finalize, "Default forward_fake does not support do_finalize=False"
        data_type = output_dtype if is_nvfp4_input else x.dtype
        num_tokens = all_rank_num_tokens[
            self.mapping.tp_rank] if all_rank_num_tokens else x.shape[0]
        hidden_size = x.shape[1] * (2 if is_nvfp4_input else 1)
        return x.new_empty((num_tokens, hidden_size), dtype=data_type)

    # Sub class is not allowed to override forward.
    # This is universal interface for all MoE backends
    @final
    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        do_finalize: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.register_to_config and is_torch_compiling():
            hidden_states = x.fp4_tensor if isinstance(
                x, Fp4QuantizedTensor) else x
            x_sf = x.scaling_factor if isinstance(x,
                                                  Fp4QuantizedTensor) else None
            is_swizzled = x.is_sf_swizzled if isinstance(
                x, Fp4QuantizedTensor) else False

            res = moe_custom_op(
                self.layer_idx_str,
                hidden_states,
                x_sf,
                is_swizzled,
                router_logits,
                do_finalize,
                output_dtype,
                all_rank_num_tokens,
                use_dp_padding,
            )
            if do_finalize:
                return res[0]
            else:
                return res
        else:
            return self.forward_impl(
                x,
                router_logits,
                do_finalize=do_finalize,
                output_dtype=output_dtype,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=use_dp_padding,
            )

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
    def has_w4a8_nvfp4_fp8(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_w4a8_nvfp4_fp8(
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
