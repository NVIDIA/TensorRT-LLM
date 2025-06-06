from typing import Optional, Type

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig

from ...model_config import ModelConfig
from .fused_moe_cutlass import CutlassFusedMoE
from .fused_moe_flux import FluxFusedMoE
from .fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from .fused_moe_vanilla import VanillaMoE
from .interface import MoE, MoEWeightLoadingMode
from .moe_load_balancer import MoeLoadBalancer
from .routing import BaseMoeRoutingMethod


def get_moe_cls(
        model_config: ModelConfig,
        override_quant_config: Optional[QuantConfig] = None) -> Type[MoE]:
    moe_backend = model_config.moe_backend
    quant_config = model_config.quant_config
    if override_quant_config is not None:
        quant_config = override_quant_config
    if moe_backend.upper() == "CUTLASS":
        return CutlassFusedMoE
    elif moe_backend.upper() == "VANILLA":
        return VanillaMoE
    elif moe_backend.upper() == "TRTLLM":
        if quant_config is not None and (
                quant_config.quant_mode.has_fp8_block_scales()
                or quant_config.quant_mode.has_nvfp4()):
            return TRTLLMGenFusedMoE
        else:
            logger.warning(
                "TRTLLMGenFusedMoE only supports fp8_block_scales or nvfp4. "
                f"Check out details in quant_config: {quant_config}"
                "Using CutlassFusedMoE instead.")
            return CutlassFusedMoE
    elif moe_backend.upper() == "FLUX":
        return FluxFusedMoE
    else:
        raise ValueError(f"Unsupported moe backend: {moe_backend}")


def create_moe(
    routing_method: BaseMoeRoutingMethod,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: Optional[torch.dtype] = None,
    reduce_results: bool = False,
    model_config: ModelConfig = ModelConfig(),
    override_quant_config: Optional[QuantConfig] = None,
    aux_stream: Optional[torch.cuda.Stream] = None,
    weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA,
    apply_router_weight_on_input: bool = False,
    enable_alltoall: bool = False,
    moe_load_balancer: Optional[MoeLoadBalancer] = None,
    layer_idx: Optional[int] = None,
) -> MoE:
    moe_cls = get_moe_cls(model_config, override_quant_config)

    if moe_cls == TRTLLMGenFusedMoE:
        assert not apply_router_weight_on_input, "apply_router_weight_on_input is not supported in TRTLLMGenFusedMoE."
        assert not enable_alltoall, "enable_alltoall is not supported in TRTLLMGenFusedMoE."
        assert moe_load_balancer is None, "moe_load_balancer is not supported in TRTLLMGenFusedMoE."

        return moe_cls(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            weight_loading_mode=weight_loading_mode,
            layer_idx=layer_idx,
        )
    elif moe_cls == CutlassFusedMoE:
        return moe_cls(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            aux_stream=aux_stream,
            weight_loading_mode=weight_loading_mode,
            apply_router_weight_on_input=apply_router_weight_on_input,
            enable_alltoall=enable_alltoall,
            moe_load_balancer=moe_load_balancer,
            layer_idx=layer_idx,
        )
    elif moe_cls == VanillaMoE:
        assert not apply_router_weight_on_input, "apply_router_weight_on_input is not supported in VanillaMoE."
        assert not enable_alltoall, "enable_alltoall is not supported in VanillaMoE."
        assert moe_load_balancer is None, "moe_load_balancer is not supported in VanillaMoE."

        return moe_cls(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            weight_loading_mode=weight_loading_mode,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
    elif moe_cls == FluxFusedMoE:
        assert not apply_router_weight_on_input, "apply_router_weight_on_input is not supported in FluxFusedMoE."
        assert not enable_alltoall, "enable_alltoall is not supported in FluxFusedMoE."
        assert moe_load_balancer is None, "moe_load_balancer is not supported in FluxFusedMoE."
        assert model_config.mapping.enable_attention_dp, "FluxFusedMoE should be used with attention dp"

        pack_weights = True

        return moe_cls(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            aux_stream=aux_stream,
            weight_loading_mode=weight_loading_mode,
            apply_router_weight_on_input=apply_router_weight_on_input,
            enable_alltoall=enable_alltoall,
            pack_weights=pack_weights,
        )
    else:
        raise ValueError(f"Unsupported moe backend: {moe_cls}")
