from typing import Dict, Optional, Type

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig

from ...model_config import ModelConfig
from ...utils import AuxStreamType
from .fused_moe_cute_dsl import CuteDslFusedMoE
from .fused_moe_cutlass import CutlassFusedMoE
from .fused_moe_deepgemm import DeepGemmFusedMoE
from .fused_moe_triton import TritonFusedMoE
from .fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from .fused_moe_vanilla import VanillaMoE
from .fused_moe_wide_ep import WideEPMoE
from .interface import MoE, MoEWeightLoadingMode
from .moe_load_balancer import get_moe_load_balancer
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
    elif moe_backend.upper() == "CUTEDSL":
        return CuteDslFusedMoE
    elif moe_backend.upper() == "DEEPGEMM":
        return DeepGemmFusedMoE
    elif moe_backend.upper() == "TRTLLM":
        if quant_config is not None and (
                quant_config.quant_mode.has_fp8_block_scales()
                or quant_config.quant_mode.has_nvfp4()
                or quant_config.quant_mode.has_w4a16_mxfp4()
                or quant_config.quant_mode.has_w4a8_nvfp4_fp8()
                or quant_config.quant_mode.has_w4a8_mxfp4_fp8()
                or quant_config.quant_mode.has_w4a8_mxfp4_mxfp8()):
            return TRTLLMGenFusedMoE
        else:
            logger.warning(
                "TRTLLMGenFusedMoE only supports fp8_block_scales, nvfp4, w4a16_mxfp4, w4a8_mxfp4_fp8 and w4a8_mxfp4_mxfp8. "
                f"Check out details in quant_config: {quant_config}"
                "Using CutlassFusedMoE instead.")
            return CutlassFusedMoE
    elif moe_backend.upper() == "WIDEEP":
        return WideEPMoE
    elif moe_backend.upper() == "TRITON":
        return TritonFusedMoE
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
    aux_stream_dict: Optional[Dict[AuxStreamType, torch.cuda.Stream]] = None,
    weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA,
    bias: bool = False,
    apply_router_weight_on_input: bool = False,
    layer_idx: Optional[int] = None,
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[torch.Tensor] = None,
) -> MoE:
    moe_cls = get_moe_cls(model_config, override_quant_config)

    moe_load_balancer = get_moe_load_balancer()
    if moe_load_balancer is not None:
        assert moe_cls == WideEPMoE, "MoE Load Balance is only supported in WideEPMoE now."

    if bias:
        assert moe_cls in [CutlassFusedMoE, TritonFusedMoE, TRTLLMGenFusedMoE
                           ], f"bias not supported in {moe_cls.__name__}."

    if swiglu_alpha is not None or swiglu_beta is not None:
        assert moe_cls in [CutlassFusedMoE, TritonFusedMoE, TRTLLMGenFusedMoE], \
            f"swiglu_alpha and swiglu_beta are only supported in CutlassFusedMoE, TritonFusedMoE and TRTLLMGenFusedMoE, not in {moe_cls.__name__}."
        assert swiglu_alpha is not None and swiglu_beta is not None, \
            "Both swiglu_alpha and swiglu_beta must be provided."

    if swiglu_limit is not None:
        assert moe_cls in [CutlassFusedMoE, TritonFusedMoE, TRTLLMGenFusedMoE], \
            f"swiglu_limit is only supported in CutlassFusedMoE, TritonFusedMoE and TRTLLMGenFusedMoE, not in {moe_cls.__name__}."

    if moe_cls == TRTLLMGenFusedMoE:
        assert not apply_router_weight_on_input, "apply_router_weight_on_input is not supported in TRTLLMGenFusedMoE."

        return moe_cls(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            weight_loading_mode=weight_loading_mode,
            bias=bias,
            layer_idx=layer_idx,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
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
            aux_stream_dict=aux_stream_dict,
            weight_loading_mode=weight_loading_mode,
            bias=bias,
            apply_router_weight_on_input=apply_router_weight_on_input,
            layer_idx=layer_idx,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
        )
    elif moe_cls == WideEPMoE:
        return moe_cls(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            aux_stream_dict=aux_stream_dict,
            weight_loading_mode=weight_loading_mode,
            apply_router_weight_on_input=apply_router_weight_on_input,
            layer_idx=layer_idx,
        )
    elif moe_cls == VanillaMoE:
        assert not apply_router_weight_on_input, "apply_router_weight_on_input is not supported in VanillaMoE."

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
    elif moe_cls == CuteDslFusedMoE:
        return moe_cls(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            aux_stream_dict=aux_stream_dict,
            weight_loading_mode=weight_loading_mode,
            apply_router_weight_on_input=apply_router_weight_on_input,
            layer_idx=layer_idx,
        )
    elif moe_cls == DeepGemmFusedMoE:
        return moe_cls(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            aux_stream_dict=aux_stream_dict,
            weight_loading_mode=weight_loading_mode,
            apply_router_weight_on_input=apply_router_weight_on_input,
            layer_idx=layer_idx,
        )
    elif moe_cls == TritonFusedMoE:
        assert not apply_router_weight_on_input, "apply_router_weight_on_input is not supported in TritonFusedMoE."

        return moe_cls(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            weight_loading_mode=weight_loading_mode,
            bias=bias,
            layer_idx=layer_idx,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
        )
    else:
        raise ValueError(f"Unsupported moe backend: {moe_cls}")
