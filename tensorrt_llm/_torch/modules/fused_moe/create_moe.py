import os
from typing import Dict, Optional, Type

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig

from ...model_config import ModelConfig
from ...utils import ActivationType, AuxStreamType
from .configurable_moe import ConfigurableMoE
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
        if quant_config is not None and (
                quant_config.quant_mode.has_fp8_block_scales()
                or quant_config.quant_mode.has_nvfp4()):
            return CuteDslFusedMoE
        else:
            logger.warning(
                "CuteDslFusedMoE only supports fp8_block_scales and nvfp4. "
                f"Check out details in quant_config: {quant_config}. Using CutlassFusedMoE instead."
            )
            return CutlassFusedMoE
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
                f"Check out details in quant_config: {quant_config}. Using CutlassFusedMoE instead."
            )
            return CutlassFusedMoE
    elif moe_backend.upper() == "WIDEEP":
        return WideEPMoE
    elif moe_backend.upper() == "TRITON":
        return TritonFusedMoE
    else:
        raise ValueError(f"Unsupported moe backend: {moe_backend}")


def create_moe_backend(
    moe_cls: Type[MoE],
    routing_method: BaseMoeRoutingMethod,
    # TODO: remove num_experts, hidden_size, intermediate_size, dtype parameters
    # these parameters will be inferred from model_config.pretrained_config.
    num_experts: Optional[int] = None,
    hidden_size: Optional[int] = None,
    intermediate_size: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    reduce_results: bool = False,
    model_config: ModelConfig = ModelConfig(),
    aux_stream_dict: Optional[Dict[AuxStreamType, torch.cuda.Stream]] = None,
    weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA,
    bias: bool = False,
    apply_router_weight_on_input: bool = False,
    layer_idx: Optional[int] = None,
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[torch.Tensor] = None,
    init_load_balancer: bool = True,
    without_comm: bool = False,
    activation_type: ActivationType = ActivationType.Swiglu,
) -> MoE:
    """
    Create MoE backend instance with validation.

    Args:
        moe_cls: MoE backend class to instantiate
        routing_method: Routing method for token-to-expert assignment
        num_experts: Total number of experts (if None, get from model_config.pretrained_config)
        hidden_size: Hidden dimension size (if None, get from model_config.pretrained_config)
        intermediate_size: Intermediate dimension size (if None, get from model_config.pretrained_config)
        dtype: Data type for weights (if None, get from model_config.pretrained_config)
        reduce_results: Whether to reduce results
        model_config: Model configuration
        aux_stream_dict: Auxiliary CUDA streams for overlap
        weight_loading_mode: Weight loading mode
        bias: Whether to use bias
        apply_router_weight_on_input: Whether to apply router weight on input
        layer_idx: Layer index
        swiglu_alpha: SwiGLU alpha parameter
        swiglu_beta: SwiGLU beta parameter
        swiglu_limit: SwiGLU limit parameter
        activation_type: Activation type

    Returns:
        MoE: MoE backend instance
    """
    # Get parameters from pretrained_config if not explicitly provided
    pretrained_config = model_config.pretrained_config
    if num_experts is None:
        assert pretrained_config is not None, "num_experts must be provided or model_config.pretrained_config must be set"
        num_experts = pretrained_config.num_experts
    if hidden_size is None:
        assert pretrained_config is not None, "hidden_size must be provided or model_config.pretrained_config must be set"
        hidden_size = pretrained_config.hidden_size
    if intermediate_size is None:
        assert pretrained_config is not None, "intermediate_size must be provided or model_config.pretrained_config must be set"
        # For MoE models, prefer moe_intermediate_size if available
        if hasattr(pretrained_config, 'moe_intermediate_size'):
            intermediate_size = pretrained_config.moe_intermediate_size
        else:
            intermediate_size = pretrained_config.intermediate_size
    if dtype is None and pretrained_config is not None and hasattr(
            pretrained_config, 'torch_dtype'):
        dtype = pretrained_config.torch_dtype

    moe_load_balancer = get_moe_load_balancer()
    if moe_load_balancer is not None:
        assert moe_cls in [
            WideEPMoE, CutlassFusedMoE, TRTLLMGenFusedMoE, CuteDslFusedMoE,
            DeepGemmFusedMoE
        ], "MoE Load Balance is only supported in WideEPMoE, CutlassFusedMoE, TRTLLMGenFusedMoE and CuteDslFusedMoE, and DeepGemmFusedMoE."

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
            aux_stream_dict=aux_stream_dict,
            weight_loading_mode=weight_loading_mode,
            bias=bias,
            layer_idx=layer_idx,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            init_load_balancer=init_load_balancer,
            without_comm=without_comm,
            activation_type=activation_type,
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
            init_load_balancer=init_load_balancer,
            activation_type=activation_type,
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
            layer_idx=layer_idx)
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
            layer_idx=layer_idx,
            activation_type=activation_type,
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
            init_load_balancer=init_load_balancer,
            without_comm=without_comm,
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


def create_moe(
    routing_method: BaseMoeRoutingMethod,
    num_experts: Optional[int] = None,
    hidden_size: Optional[int] = None,
    intermediate_size: Optional[int] = None,
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
    activation_type: ActivationType = ActivationType.Swiglu,
) -> MoE:
    """
    Create MoE instance with automatic parameter inference from model_config.

    Args:
        routing_method: Routing method for token-to-expert assignment
        num_experts: Total number of experts (if None, get from model_config.pretrained_config)
        hidden_size: Hidden dimension size (if None, get from model_config.pretrained_config)
        intermediate_size: Intermediate dimension size (if None, get from model_config.pretrained_config)
        dtype: Data type for weights (if None, get from model_config.pretrained_config)
        reduce_results: Whether to reduce results
        model_config: Model configuration
        override_quant_config: Override quantization config
        aux_stream_dict: Auxiliary CUDA streams for overlap
        weight_loading_mode: Weight loading mode
        bias: Whether to use bias
        apply_router_weight_on_input: Whether to apply router weight on input
        layer_idx: Layer index
        swiglu_alpha: SwiGLU alpha parameter
        swiglu_beta: SwiGLU beta parameter
        swiglu_limit: SwiGLU limit parameter
        activation_type: Activation type

    Returns:
        MoE: MoE instance
    """
    # Get parameters from pretrained_config if not explicitly provided
    pretrained_config = model_config.pretrained_config
    if num_experts is None:
        assert pretrained_config is not None, "num_experts must be provided or model_config.pretrained_config must be set"
        num_experts = pretrained_config.num_experts
    if hidden_size is None:
        assert pretrained_config is not None, "hidden_size must be provided or model_config.pretrained_config must be set"
        hidden_size = pretrained_config.hidden_size
    if intermediate_size is None:
        assert pretrained_config is not None, "intermediate_size must be provided or model_config.pretrained_config must be set"
        # For MoE models, prefer moe_intermediate_size if available
        if hasattr(pretrained_config, 'moe_intermediate_size'):
            intermediate_size = pretrained_config.moe_intermediate_size
        else:
            intermediate_size = pretrained_config.intermediate_size
    if dtype is None and pretrained_config is not None and hasattr(
            pretrained_config, 'torch_dtype'):
        dtype = pretrained_config.torch_dtype

    moe_cls = get_moe_cls(model_config, override_quant_config)

    enable_configurable_moe = os.environ.get("ENABLE_CONFIGURABLE_MOE",
                                             "1") == "1"
    if enable_configurable_moe or moe_cls == CuteDslFusedMoE:
        if moe_cls in (DeepGemmFusedMoE, TRTLLMGenFusedMoE, CuteDslFusedMoE,
                       CutlassFusedMoE):
            return ConfigurableMoE(
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
                override_quant_config=override_quant_config,
                bias=bias,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
                swiglu_limit=swiglu_limit,
                activation_type=activation_type,
            )
        else:
            # Check if this is a TRTLLM backend request that fallback to CutlassFusedMoE
            requested_backend = model_config.moe_backend.upper()
            if requested_backend in ("TRTLLM",
                                     "CUTEDSL") and moe_cls == CutlassFusedMoE:
                # Workaround for test cases where TRTLLM backend fallbacks to CutlassFusedMoE due to quant_config incompatibility
                # Log warning and continue with the fallback backend
                logger.warning(
                    f"ENABLE_CONFIGURABLE_MOE is set but TRTLLM backend fallback to {moe_cls.__name__} due to quant_config. "
                    f"ConfigurableMoE only supports TRTLLMGenFusedMoE and CuteDslFusedMoE backends. "
                    f"Continuing with legacy MoE backend {moe_cls.__name__}.")
            else:
                # Other backends are not supported by ConfigurableMoE, fallback to legacy backend
                # This is a WAR to make sure all the CI test cases pass.
                # TODO: Remove this workaround when ConfigurableMoE is supported by all backends.
                logger.warning(
                    f"ENABLE_CONFIGURABLE_MOE is set but {moe_cls.__name__} is not supported by ConfigurableMoE. "
                    f"Continuing with legacy MoE backend {moe_cls.__name__}.")

    # Use legacy create_moe_backend for other backends or when ConfigurableMoE is disabled
    return create_moe_backend(
        moe_cls=moe_cls,
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
        activation_type=activation_type,
    )
