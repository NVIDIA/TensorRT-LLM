# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Optional

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm.models.modeling_utils import QuantConfig

from .activation import DEFAULT_MOE_ACTIVATION, MoEActivation
from .configurable_moe import ConfigurableMoE
from .fused_moe_cute_dsl import CuteDslFusedMoE
from .fused_moe_cute_dsl_b12x import CuteDslB12xFusedMoE
from .fused_moe_cute_dsl_fc12 import CuteDslFc12FusedMoE
from .fused_moe_cutlass import CutlassFusedMoE
from .fused_moe_deepgemm import DeepGemmFusedMoE
from .fused_moe_densegemm import DenseGEMMFusedMoE
from .fused_moe_marlin import MarlinFusedMoE
from .fused_moe_triton import TritonFusedMoE
from .fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from .fused_moe_vanilla import VanillaMoE
from .impl_base import MoEImplBase
from .interface import MoE, MoEWeightLoadingMode
from .mega_moe import MegaMoECuteDsl, MegaMoEDeepGemm
from .moe_load_balancer import get_moe_load_balancer
from .moe_resolution import (WIDEEP_DEPRECATION_MESSAGE, MoEImplClass,
                             derive_moe_layer_shapes, infer_swiglu_gptoss_style,
                             resolve_moe_cls, resolve_moe_impl)
from .routing import BaseMoeRoutingMethod

__all__ = [
    "create_moe",
    "create_moe_backend",
    "infer_swiglu_gptoss_style",
    "MoEImplClass",
    "resolve_moe_cls",
    "resolve_moe_impl",
    "WIDEEP_DEPRECATION_MESSAGE",
]


def create_moe_backend(
    moe_cls: MoEImplClass,
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
    init_load_balancer: bool = False,
    activation: MoEActivation = DEFAULT_MOE_ACTIVATION,
) -> MoE | MoEImplBase | VanillaMoE:
    """
    Create a MoE backend or a self-contained MoE layer.

    Execution units (``MoEImplBase`` subclasses) are only ever constructed as
    backends, so ``init_load_balancer`` defaults to ``False`` -- ``True`` asks
    for a standalone layer and the impl constructor rejects it with a
    ``TypeError``. The ``TritonFusedMoE`` and ``VanillaMoE`` branches are
    complete layers and do not take the parameter at all.

    Args:
        moe_cls: MoE backend or self-contained layer class to instantiate
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
        activation: The layer's activation kind and its constants. Whether a
            constant reaches the kernel per expert or as one baked scalar is the
            backend's declaration (``activation_support``), not a caller choice.

    Returns:
        A ``MoEImplBase`` execution unit, a self-contained ``MoE`` layer, or
        ``VanillaMoE``.
    """
    shapes = derive_moe_layer_shapes(
        model_config,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        routing=routing_method,
    )
    num_experts = shapes.num_experts
    hidden_size = shapes.hidden_size
    intermediate_size = shapes.intermediate_size
    dtype = shapes.dtype
    assert num_experts is not None, (
        "num_experts must be provided or model_config.pretrained_config must "
        "expose num_experts / n_routed_experts / num_local_experts")
    assert hidden_size is not None, (
        "hidden_size must be provided or model_config.pretrained_config must be set"
    )
    assert intermediate_size is not None, (
        "intermediate_size must be provided or model_config.pretrained_config "
        "must expose moe_intermediate_size / intermediate_size")

    # Backstop for the direct-``moe_cls`` callers that bypass resolution: a
    # backend that cannot run under a load balancer already declines in
    # ``can_implement``.
    eplb_enabled = get_moe_load_balancer() is not None
    if eplb_enabled and not moe_cls.capabilities.supports_eplb:
        raise ValueError(
            f"{moe_cls.__name__} does not support the MoE load balancer.")

    if bias and not moe_cls.capabilities.supports_expert_bias:
        raise ValueError(f"bias not supported in {moe_cls.__name__}.")

    if (apply_router_weight_on_input
            and not moe_cls.capabilities.supports_apply_router_weight_on_input):
        raise ValueError(
            f"apply_router_weight_on_input not supported in {moe_cls.__name__}."
        )

    if moe_cls == TRTLLMGenFusedMoE:
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
            init_load_balancer=init_load_balancer,
            activation=activation,
        )

    if moe_cls in (CutlassFusedMoE, MarlinFusedMoE):
        # The two whose constructor takes an expert-bias flag. Marlin declines
        # the flag itself, so the check above already rejected a True.
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
            init_load_balancer=init_load_balancer,
            activation=activation,
        )
    elif moe_cls == VanillaMoE:
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
            activation=activation,
        )
    elif moe_cls in (CuteDslFusedMoE, CuteDslB12xFusedMoE, CuteDslFc12FusedMoE):
        # Both are constructed through the narrower CuteDsl argument set (no
        # bias / swiglu_alpha-beta-limit). CuteDslB12xFusedMoE now delegates to
        # CutlassFusedMoE.__init__, which does accept those four, so widening
        # this branch would need the allow-lists above to admit b12x first.
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
            activation=activation,
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
            init_load_balancer=init_load_balancer,
            activation=activation,
        )
    elif moe_cls == TritonFusedMoE:
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
            activation=activation,
        )
    elif moe_cls == DenseGEMMFusedMoE:
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
            activation=activation,
        )
    elif moe_cls in (MegaMoEDeepGemm, MegaMoECuteDsl):
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
            activation=activation,
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
    activation: MoEActivation = DEFAULT_MOE_ACTIVATION,
    communication_method: Optional[str] = None,
    allow_backend_degradation: bool = True,
) -> MoE | VanillaMoE:
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
        activation: The layer's activation kind and its constants, e.g.
            ``SwigluActivation(clamp=7.0)`` or
            ``SiTuActivation(gate_softcap=..., linear_softcap=...)``
        communication_method: Optional ConfigurableMoE communication method
        allow_backend_degradation: When False, a requested backend that cannot
            serve this layer raises with the rejection trail instead of falling
            back. For callers that must know they got the backend they asked
            for, e.g. because they are measuring it.

    Returns:
        A complete MoE layer: a ``MoE`` (``ConfigurableMoE`` around an
        execution unit, or ``TritonFusedMoE``), or ``VanillaMoE``. Never a bare
        execution unit -- those only reach a model as ``ConfigurableMoE.backend``.
    """
    shapes = derive_moe_layer_shapes(
        model_config,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        routing=routing_method,
    )
    num_experts = shapes.num_experts
    hidden_size = shapes.hidden_size
    intermediate_size = shapes.intermediate_size
    dtype = shapes.dtype
    assert num_experts is not None, (
        "num_experts must be provided or model_config.pretrained_config must "
        "expose num_experts / n_routed_experts / num_local_experts")
    assert hidden_size is not None, (
        "hidden_size must be provided or model_config.pretrained_config must be set"
    )
    assert intermediate_size is not None, (
        "intermediate_size must be provided or model_config.pretrained_config "
        "must expose moe_intermediate_size / intermediate_size")

    # Pass the same shapes / activation package the layer will be built with.
    moe_cls = resolve_moe_cls(
        model_config,
        override_quant_config=override_quant_config,
        dtype=dtype,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        swiglu_gptoss_style=infer_swiglu_gptoss_style(
            bias=bias,
            activation_type=activation.kind,
        ),
        bias=bias,
        activation=activation,
        routing=routing_method,
        layer_idx=layer_idx,
        allow_degradation=allow_backend_degradation,
    )

    # This dispatch needs no per-class entry: inheriting ``MoEImplBase`` is
    # enough to be wrapped. Becoming *selectable* still needs the constructor
    # branch in ``create_moe_backend`` above plus ``moe_resolution``'s
    # ``IMPL_PRIORITY`` / ``BACKEND_FAMILY``.
    if issubclass(moe_cls, MoEImplBase):
        return ConfigurableMoE(
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
            apply_router_weight_on_input=apply_router_weight_on_input,
            layer_idx=layer_idx,
            override_quant_config=override_quant_config,
            bias=bias,
            activation=activation,
            communication_method=communication_method,
        )

    # TritonFusedMoE and VanillaMoE are not wrapped by ConfigurableMoE
    # and own their communication and forward paths.
    if communication_method is not None:
        raise ValueError("communication_method requires ConfigurableMoE.")
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
        activation=activation,
    )
