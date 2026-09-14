# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Assembling the arguments a TRTLLM-Gen kernel call takes.

Free functions taking the impl as a parameter, not methods on
:class:`.TrtllmGenFusedMoEBase`: nothing here is part of what a MoE
implementation *is*, it is the prologue the five ``run_moe`` bodies share.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch

from ....utils import ActivationType, ActType_TrtllmGen, Fp4QuantizedTensor
from ..impl_contract import MoERunContext, require_comm_plan
from ..routing import (
    BaseMoeRoutingMethod,
    DeepSeekV3MoeRoutingMethod,
    DeepSeekV4MoeRoutingMethod,
    MiniMaxM2MoeRoutingMethod,
    MiniMaxM3MoeRoutingMethod,
)

# ActivationType -> the batched-GEMM ``ActType`` encoding the cubins are keyed
# by. SwigluBias shares the SwiGlu kernel (the gpt-oss constants travel as
# separate per-expert tensors, not as a distinct act type).
_TRTLLM_GEN_ACT_TYPE = {
    ActivationType.Swiglu: ActType_TrtllmGen.SwiGlu,
    ActivationType.SwigluBias: ActType_TrtllmGen.SwiGlu,
    ActivationType.Relu2: ActType_TrtllmGen.Relu2,
    ActivationType.Silu: ActType_TrtllmGen.Silu,
    ActivationType.SiTu: ActType_TrtllmGen.SiTu,
}


@dataclass
class RoutingParams:
    top_k: int
    routing_bias: Optional[torch.Tensor]
    n_group: Optional[int]
    topk_group: Optional[int]
    routed_scaling_factor: Optional[float]


@dataclass
class KernelInputs:
    """What every TRTLLM-Gen kernel call needs, derived once from a run context.

    The routing arguments each kernel takes positionally, plus the three
    context values every ``run_moe`` body reads. ``routing_bias`` is already
    ``None`` when routing happened outside the kernel, and ``top_k`` already
    reflects a caller-supplied width -- both are decisions, settled once here
    rather than in each body.
    """

    x: Union[torch.Tensor, Fp4QuantizedTensor]
    x_sf: Optional[torch.Tensor]  # flattened to 1D for the kernel ABI
    router_logits: Optional[torch.Tensor]  # None when top-k is precomputed
    token_selected_experts: Optional[torch.Tensor]
    token_final_scales: Optional[torch.Tensor]
    moe_output: Optional[torch.Tensor]
    do_finalize: bool
    top_k: int
    routing_bias: Optional[torch.Tensor]
    n_group: Optional[int]
    topk_group: Optional[int]
    routed_scaling_factor: Optional[float]


def to_trtllm_gen_act_type(activation_type: ActivationType) -> int:
    """Encode an activation kind the way the cubin table is keyed."""
    act_type = _TRTLLM_GEN_ACT_TYPE.get(ActivationType(activation_type))
    if act_type is None:
        raise ValueError(f"Unsupported activation type: {activation_type}")
    return int(act_type)


def get_data_or_none(module: torch.nn.Module, attr_name: str) -> Optional[torch.Tensor]:
    """``module.<attr>.data``, or ``None`` if the parameter was never created.

    Several scale tensors exist only for some quantization formats, and the
    kernel ABI takes ``None`` in their place.
    """
    attr = getattr(module, attr_name, None)
    return attr.data if attr is not None else None


def extract_routing_params(routing_method: BaseMoeRoutingMethod) -> RoutingParams:
    """The four fused-routing arguments, wherever this method happens to keep them.

    A pure function of the routing method: which of ``n_group`` / ``topk_group``
    / ``routed_scaling_factor`` a kernel receives depends on the algorithm, and
    the classes do not agree on where those live -- DeepSeek-V3 nests them under
    ``routing_impl``, the others hold them directly.
    """
    if isinstance(routing_method, DeepSeekV3MoeRoutingMethod):
        return RoutingParams(
            top_k=routing_method.routing_impl.top_k,
            routing_bias=routing_method.e_score_correction_bias,
            n_group=routing_method.routing_impl.n_group,
            topk_group=routing_method.routing_impl.topk_group,
            routed_scaling_factor=routing_method.routing_impl.routed_scaling_factor,
        )
    if isinstance(routing_method, MiniMaxM3MoeRoutingMethod):
        return RoutingParams(
            top_k=routing_method.top_k,
            routing_bias=routing_method.e_score_correction_bias,
            n_group=None,
            topk_group=None,
            routed_scaling_factor=routing_method.routed_scaling_factor,
        )
    if isinstance(routing_method, MiniMaxM2MoeRoutingMethod):
        return RoutingParams(
            top_k=routing_method.top_k,
            routing_bias=routing_method.e_score_correction_bias,
            n_group=None,
            topk_group=None,
            routed_scaling_factor=None,
        )
    if isinstance(routing_method, DeepSeekV4MoeRoutingMethod):
        return RoutingParams(
            top_k=routing_method.top_k,
            routing_bias=routing_method.e_score_correction_bias,
            n_group=routing_method.n_group,
            topk_group=routing_method.topk_group,
            routed_scaling_factor=routing_method.routed_scaling_factor,
        )
    return RoutingParams(
        top_k=routing_method.top_k,
        routing_bias=None,
        n_group=None,
        topk_group=None,
        routed_scaling_factor=None,
    )


def prepare_kernel_inputs(impl, ctx: MoERunContext) -> KernelInputs:
    """Resolve the arguments every leaf's kernel call shares.

    Takes the leaf rather than extracted fields because
    ``_routes_outside_the_kernel`` is a decision the framework also reads off
    the module, so it stays a method there.
    """
    plan = require_comm_plan(impl, ctx)

    if impl._routes_outside_the_kernel():
        if ctx.router_logits is not None and ctx.token_selected_experts is None:
            raise ValueError(
                f"{type(impl).__name__} requires separated routing for this "
                "config, so ctx.router_logits is ignored, but "
                "ctx.token_selected_experts is None -- there is nothing left "
                "to route with. Supply precomputed top-k ids and scales."
            )
        router_logits = None
    else:
        router_logits = ctx.router_logits

    routing_params = extract_routing_params(impl.routing_method)
    top_k = routing_params.top_k
    if ctx.token_selected_experts is not None:
        # for cases like deepep low latency where fake top_k=1 might be used
        top_k = ctx.token_selected_experts.shape[-1]

    x_sf = ctx.x_sf
    if x_sf is not None:
        # Ensure x_sf is 2D before flattening
        assert len(x_sf.shape) == 2, f"x_sf should be 2D tensor, got shape {x_sf.shape}"
        x_sf = x_sf.flatten()

    return KernelInputs(
        x=ctx.x,
        x_sf=x_sf,
        router_logits=router_logits,
        token_selected_experts=ctx.token_selected_experts,
        token_final_scales=ctx.token_final_scales,
        moe_output=plan.moe_output,
        do_finalize=ctx.do_finalize,
        top_k=top_k,
        routing_bias=(routing_params.routing_bias if router_logits is not None else None),
        n_group=routing_params.n_group,
        topk_group=routing_params.topk_group,
        routed_scaling_factor=routing_params.routed_scaling_factor,
    )
