# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Optional, Type

import torch

from tensorrt_llm.models.modeling_utils import QuantConfig

from ...model_config import ModelConfig
from ...utils import ActivationType, ActType_TrtllmGen, AuxStreamType
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
from .fused_moe_wide_ep import WideEPMoE
from .interface import MoE, MoEWeightLoadingMode
from .mega_moe import MegaMoECuteDsl, MegaMoEDeepGemm
from .moe_load_balancer import get_moe_load_balancer
from .moe_resolution import (WIDEEP_DEPRECATION_MESSAGE,
                             derive_moe_layer_shapes, infer_swiglu_gptoss_style,
                             resolve_moe_cls, resolve_moe_impl)
from .routing import BaseMoeRoutingMethod

__all__ = [
    "create_moe",
    "create_moe_backend",
    "infer_swiglu_gptoss_style",
    "resolve_moe_cls",
    "resolve_moe_impl",
    "WIDEEP_DEPRECATION_MESSAGE",
]


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
    swiglu_limit_scalar: Optional[float] = None,
    init_load_balancer: bool = True,
    activation_type: ActivationType = ActivationType.Swiglu,
    activation: Optional[str] = None,
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    trtllm_gen_activation_type: Optional[ActType_TrtllmGen] = None,
    trtllm_gen_activation_alpha: Optional[float] = None,
    trtllm_gen_activation_beta: Optional[float] = None,
    override_quant_config: Optional[QuantConfig] = None,
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
        swiglu_limit: SwiGLU limit parameter (per-expert tensor; for NVFP4)
        swiglu_limit_scalar: SwiGLU limit scalar (uniform across experts; for FP8)
        activation_type: Activation type
        activation: Optional MegaMoE DeepGEMM activation name
        situ_beta: Optional MegaMoE DeepGEMM SiTU beta
        situ_linear_beta: Optional MegaMoE DeepGEMM SiTU linear beta
        trtllm_gen_activation_type: Optional TRTLLM-Gen backend-local activation type
        trtllm_gen_activation_alpha: Optional backend-local activation alpha
        trtllm_gen_activation_beta: Optional backend-local activation beta

    Returns:
        MoE: MoE backend instance
    """
    if moe_cls is WideEPMoE:
        raise ValueError(WIDEEP_DEPRECATION_MESSAGE)

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

    moe_load_balancer = get_moe_load_balancer()
    if moe_load_balancer is not None:
        supported_load_balancer_backends = (
            CutlassFusedMoE,
            TRTLLMGenFusedMoE,
            CuteDslFusedMoE,
            CuteDslFc12FusedMoE,
            DeepGemmFusedMoE,
            DenseGEMMFusedMoE,
            MegaMoEDeepGemm,
            MegaMoECuteDsl,
        )
        assert moe_cls in supported_load_balancer_backends, (
            "MoE Load Balance is only supported in "
            f"{', '.join(cls.__name__ for cls in supported_load_balancer_backends)}."
        )

    # ---- SiTU hand-off ---------------------------------------------------
    # SiTU is a model property, so a model names it once -- activation='situ'
    # plus both constants -- and this is the single place that translates that
    # into each backend's own constructor signature. The backends spell the
    # same two numbers three different ways: CUTLASS wants per-expert tensors
    # under the gpt-oss ``swiglu_alpha`` / ``swiglu_beta`` names, TRTLLM-Gen
    # wants its own enum plus scalars, CuteDSL wants ActivationType.SiTu with
    # scalars, and MegaMoE keeps the gated-FC1 geometry in ``activation_type``
    # while naming the elementwise function in ``activation``. Translating in
    # the model layer instead would put backend dialects -- and a CUDA
    # allocation shaped by moe_ep_size -- into every model that uses SiTU.
    if activation is not None and activation.lower() == "situ":
        if situ_beta is None or situ_linear_beta is None:
            raise ValueError(
                "activation='situ' requires explicit situ_beta and "
                "situ_linear_beta.")
        if situ_beta <= 0 or situ_linear_beta <= 0:
            raise ValueError(
                "situ_beta and situ_linear_beta must be positive; got "
                f"{situ_beta} and {situ_linear_beta}.")
        situ_beta = float(situ_beta)
        situ_linear_beta = float(situ_linear_beta)

        if moe_cls in (CutlassFusedMoE, MarlinFusedMoE):
            # SiTuAdaptor reads per-expert alpha/beta tensors, so the constants
            # are broadcast over this rank's expert shard.
            local_num_experts = num_experts // max(
                model_config.mapping.moe_ep_size, 1)
            device = torch.device("cuda", torch.cuda.current_device())
            swiglu_alpha = torch.full((local_num_experts, ),
                                      situ_beta,
                                      dtype=torch.float32,
                                      device=device)
            swiglu_beta = torch.full((local_num_experts, ),
                                     situ_linear_beta,
                                     dtype=torch.float32,
                                     device=device)
            activation_type = ActivationType.SiTu
            activation = situ_beta = situ_linear_beta = None
        elif moe_cls in (CuteDslFusedMoE, CuteDslB12xFusedMoE,
                         CuteDslFc12FusedMoE):
            # The CuTe DSL FC1 epilogue folds the constants in at trace time,
            # so they stay scalars and need no per-expert tensors.
            # CuteDslFc12FusedMoE inherits the parent's SiTU-capable __init__
            # and forwards the scalars into the fused kernel cache key.
            activation_type = ActivationType.SiTu
            activation = None
        elif moe_cls is TRTLLMGenFusedMoE:
            # Cubin alpha is the gate-side constant, cubin beta the
            # linear-side one. activation_type stays the FC1 geometry.
            trtllm_gen_activation_type = ActType_TrtllmGen.SiTu
            trtllm_gen_activation_alpha = situ_beta
            trtllm_gen_activation_beta = situ_linear_beta
            activation = situ_beta = situ_linear_beta = None
        elif moe_cls in (MegaMoEDeepGemm, MegaMoECuteDsl):
            # Already the MegaMoE dialect; activation_type stays Swiglu, which
            # is the gated FC1 geometry rather than the elementwise function.
            pass
        else:
            raise ValueError(
                f"activation='situ' is not supported by {moe_cls.__name__}.")
    elif situ_beta is not None or situ_linear_beta is not None:
        raise ValueError(
            "situ_beta / situ_linear_beta require activation='situ'.")

    if bias:
        assert moe_cls in [CutlassFusedMoE, TritonFusedMoE, TRTLLMGenFusedMoE
                           ], f"bias not supported in {moe_cls.__name__}."

    if swiglu_alpha is not None or swiglu_beta is not None:
        assert moe_cls in [CutlassFusedMoE, TritonFusedMoE, TRTLLMGenFusedMoE], \
            f"swiglu_alpha and swiglu_beta are only supported in CutlassFusedMoE, TritonFusedMoE and TRTLLMGenFusedMoE, not in {moe_cls.__name__}."
        assert swiglu_alpha is not None and swiglu_beta is not None, \
            "Both swiglu_alpha and swiglu_beta must be provided."

    if swiglu_limit is not None:
        assert moe_cls in [
            CutlassFusedMoE, TritonFusedMoE, TRTLLMGenFusedMoE,
            DeepGemmFusedMoE, MegaMoECuteDsl
        ], f"swiglu_limit is not supported in {moe_cls.__name__}."

    if swiglu_limit_scalar is not None:
        assert moe_cls in [
            CutlassFusedMoE, TRTLLMGenFusedMoE, DeepGemmFusedMoE,
            MegaMoEDeepGemm, CuteDslFusedMoE, CuteDslFc12FusedMoE,
            MegaMoECuteDsl
        ], f"swiglu_limit_scalar is not supported in {moe_cls.__name__}."

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
            swiglu_limit_scalar=swiglu_limit_scalar,
            init_load_balancer=init_load_balancer,
            activation_type=activation_type,
            trtllm_gen_activation_type=trtllm_gen_activation_type,
            trtllm_gen_activation_alpha=trtllm_gen_activation_alpha,
            trtllm_gen_activation_beta=trtllm_gen_activation_beta,
        )

    if any(value is not None for value in (trtllm_gen_activation_type,
                                           trtllm_gen_activation_alpha,
                                           trtllm_gen_activation_beta)):
        raise ValueError(
            "TRTLLM-Gen backend-local activation options are only supported "
            f"by TRTLLMGenFusedMoE, got {moe_cls.__name__}")

    if activation is not None and moe_cls not in (MegaMoEDeepGemm,
                                                  MegaMoECuteDsl):
        raise ValueError("activation= is only supported by MegaMoEDeepGemm and "
                         f"MegaMoECuteDsl, got {moe_cls.__name__}")
    if (situ_beta is not None or situ_linear_beta is not None
        ) and moe_cls not in (MegaMoEDeepGemm, MegaMoECuteDsl, CuteDslFusedMoE,
                              CuteDslB12xFusedMoE, CuteDslFc12FusedMoE):
        raise ValueError(
            "situ_beta / situ_linear_beta are only supported by MegaMoE and "
            f"CuteDSL backends, got {moe_cls.__name__}")

    if moe_cls in (CutlassFusedMoE, MarlinFusedMoE):
        # CuteDslFusedMoE, DeepGemmFusedMoE, and CuteDslB12xFusedMoE
        # also subclass CutlassFusedMoE but have narrower constructors, so
        # they take their own branches below.
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
            swiglu_limit_scalar=swiglu_limit_scalar,
            init_load_balancer=init_load_balancer,
            activation_type=activation_type,
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
            layer_idx=layer_idx,
            activation_type=activation_type,
        )
    elif moe_cls in (CuteDslFusedMoE, CuteDslB12xFusedMoE, CuteDslFc12FusedMoE):
        # CuteDslB12xFusedMoE subclasses CuteDslFusedMoE and shares
        # its narrower constructor (no bias / swiglu_alpha-beta-limit args).
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
            swiglu_limit_scalar=swiglu_limit_scalar,
            init_load_balancer=init_load_balancer,
            activation_type=activation_type,
            override_quant_config=override_quant_config,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
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
            swiglu_limit=swiglu_limit,
            swiglu_limit_scalar=swiglu_limit_scalar,
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
            activation_type=activation_type,
        )
    elif moe_cls in (MegaMoEDeepGemm, MegaMoECuteDsl):
        # MegaMoE fused-comm backends share the same construction surface.
        # ``mega_moe_deepgemm`` lazily resolves DG via ``_import_deep_gemm``
        # at runtime and ``mega_moe_cute_dsl`` lazily imports the CuteDSL
        # kernel package, so a top-level import here doesn't pull either
        # heavyweight dependency on boxes that don't use these backends.
        megamoe_kwargs = dict(
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
            activation_type=activation_type,
            activation=activation,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )
        if moe_cls is MegaMoECuteDsl:
            megamoe_kwargs["swiglu_limit"] = (swiglu_limit
                                              if swiglu_limit is not None else
                                              swiglu_limit_scalar)
        else:
            # activation / situ_beta / situ_linear_beta are already in the
            # shared megamoe_kwargs above: both MegaMoE backends take SiTU
            # explicitly now, not DeepGEMM alone. Only the DeepGEMM-specific
            # entry is added here.
            megamoe_kwargs.update(swiglu_limit_scalar=swiglu_limit_scalar)
        return moe_cls(**megamoe_kwargs)
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
    swiglu_limit_scalar: Optional[float] = None,
    activation_type: ActivationType = ActivationType.Swiglu,
    activation: Optional[str] = None,
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    trtllm_gen_activation_type: Optional[ActType_TrtllmGen] = None,
    trtllm_gen_activation_alpha: Optional[float] = None,
    trtllm_gen_activation_beta: Optional[float] = None,
    communication_method: Optional[str] = None,
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
        swiglu_limit: SwiGLU limit parameter (per-expert tensor; for NVFP4)
        swiglu_limit_scalar: SwiGLU limit scalar (uniform across experts; for FP8)
        activation_type: Activation type
        trtllm_gen_activation_type: Optional TRTLLM-Gen backend-local activation type
        trtllm_gen_activation_alpha: Optional backend-local activation alpha
        trtllm_gen_activation_beta: Optional backend-local activation beta
        communication_method: Optional ConfigurableMoE communication method
        activation: Optional MegaMoE elementwise activation (``'swiglu'`` / ``'situ'``)
        situ_beta: Optional SiTU gate beta (MegaMoE / CuteDSL)
        situ_linear_beta: Optional SiTU linear-side beta (MegaMoE / CuteDSL)

    Returns:
        MoE: MoE instance
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
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            activation_type=activation_type,
        ),
        bias=bias,
        activation_type=activation_type,
        routing=routing_method,
        layer_idx=layer_idx,
    )
    # Ownership of MegaMoE's activation / situ_* is decided by the two guards
    # in create_moe_backend, which accept both MegaMoEDeepGemm and
    # MegaMoECuteDsl. A redundant `moe_cls is not MegaMoEDeepGemm` test used to
    # sit here as well and would have rejected MegaMoECuteDsl outright; it goes
    # away with the f5950a3799 port.
    if (any(value is not None for value in (trtllm_gen_activation_type,
                                            trtllm_gen_activation_alpha,
                                            trtllm_gen_activation_beta))
            and moe_cls is not TRTLLMGenFusedMoE):
        raise ValueError(
            "A TRTLLM-Gen backend-local activation requires "
            "TRTLLMGenFusedMoE without backend fallback, but resolved "
            f"{moe_cls.__name__}.")

    if moe_cls in (DeepGemmFusedMoE, TRTLLMGenFusedMoE, CuteDslFusedMoE,
                   CuteDslB12xFusedMoE, CuteDslFc12FusedMoE, CutlassFusedMoE,
                   DenseGEMMFusedMoE, MegaMoEDeepGemm, MegaMoECuteDsl,
                   MarlinFusedMoE):
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
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            swiglu_limit_scalar=swiglu_limit_scalar,
            activation_type=activation_type,
            activation=activation,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            trtllm_gen_activation_type=trtllm_gen_activation_type,
            trtllm_gen_activation_alpha=trtllm_gen_activation_alpha,
            trtllm_gen_activation_beta=trtllm_gen_activation_beta,
            communication_method=communication_method,
        )

    # WideEPMoE, TritonFusedMoE and VanillaMoE are not wrapped by ConfigurableMoE
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
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        swiglu_limit_scalar=swiglu_limit_scalar,
        activation_type=activation_type,
        activation=activation,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        trtllm_gen_activation_type=trtllm_gen_activation_type,
        trtllm_gen_activation_alpha=trtllm_gen_activation_alpha,
        trtllm_gen_activation_beta=trtllm_gen_activation_beta,
        override_quant_config=override_quant_config,
    )
