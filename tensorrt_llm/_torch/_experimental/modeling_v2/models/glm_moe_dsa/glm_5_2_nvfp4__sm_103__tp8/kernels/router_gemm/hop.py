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
"""The router contract: the pre-MoE reduction/norm/router hop of package router_gemm.

The hop keeps the native all-reduce + residual + RMSNorm (one fused native
kernel, unchanged) and replaces only the native router GEMM that consumes its
normalized output. The router logits are bitwise identical to the native
``DeepseekV3Gate`` logits, so expert selection, which stays in the native MoE,
is unchanged.

Entry points (registered in ``router_gemm/__init__.py``):

* ``router_logits(hidden, gate_weight)``: router_gemm logits or ``None`` when the call
  is outside the support domain; on ``None`` the caller runs the native gate.
* ``pre_moe_hop(...)``: native fused AR/residual/RMSNorm followed by
  ``router_logits``; returns ``(hidden, residual, logits_or_None)``.
* ``make_moe_router(gate, layer_idx, model_config)``: per-layer capability
  decision made once at model construction; returns a callable for
  ``Deepseekv3MoE.compute_routed_output`` or ``None`` (native gate).
"""

from typing import Callable, Dict, Optional, Tuple

import torch

from ... import dispatch
from .router_kernel import pdl_enabled, router_gemm, router_gemm_supported

FEATURE_ID = "router_gemm"

# Host-side call counters for dispatch-engagement reporting. They count Python
# calls (eager runs and CUDA graph captures), not graph replays.
_counts: Dict[str, int] = {"engaged": 0, "native_fallback": 0}


def engagement_counts() -> Dict[str, int]:
    """Copy of the host-side engaged/fallback call counters."""
    return dict(_counts)


def router_logits(hidden_states: torch.Tensor, gate_weight: torch.Tensor) -> Optional[torch.Tensor]:
    """FP32 router logits ``[R, E]`` or ``None`` (caller uses native gate).

    Ineligible calls return before launching anything, so the native fallback
    happens before any side effect.
    """
    eligible, _ = router_gemm_supported(hidden_states, gate_weight)
    if not eligible:
        _counts["native_fallback"] += 1
        return None
    _counts["engaged"] += 1
    return router_gemm(hidden_states, gate_weight)


def pre_moe_hop(
    allreduce,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    gate_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Reduce -> residual -> RMSNorm (native) -> router (router_gemm).

    Args:
        allreduce: the layer's native ``AllReduce`` module.
        hidden_states: rank-local attention output partial ``[R, H]``.
        residual: residual stream ``[R, H]``.
        norm_weight, eps: ``post_attention_layernorm`` parameters.
        gate_weight: router weight ``[E, H]`` (BF16).

    Returns:
        ``(normed_hidden, residual, router_logits_or_None)``; the first two
        are exactly the native fused all-reduce outputs.
    """
    from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams

    hidden_states, residual = allreduce(
        hidden_states,
        all_reduce_params=AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=residual,
            norm_weight=norm_weight,
            eps=eps,
            trigger_completion_at_end=False,
        ),
    )
    return hidden_states, residual, router_logits(hidden_states, gate_weight)


def _layer_domain(gate, layer_idx: Optional[int], model_config) -> str:
    """Return '' when the layer is in the support domain, else the reason."""
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Gate
    from tensorrt_llm._utils import get_sm_version, is_sm_100f

    if type(gate) is not DeepseekV3Gate:
        return f"gate class {type(gate).__name__}"
    mapping = model_config.mapping
    if mapping.enable_attention_dp:
        return "attention DP"
    if mapping.tp_size <= 1:
        return "no tensor parallelism"
    num_target_layers = model_config.pretrained_config.num_hidden_layers
    if layer_idx is None or layer_idx >= num_target_layers:
        return "not a target layer"
    if gate.use_cute_dsl_bf16_gemm and is_sm_100f():
        return "native gate uses the CuTe DSL BF16 GEMM"
    if not 100 <= get_sm_version() < 110:
        return f"sm{get_sm_version()}"
    return ""


def make_moe_router(gate, layer_idx: Optional[int], model_config) -> Optional[Callable]:
    """Per-layer router factory used by the ``Deepseekv3MoE`` adapter.

    Returns ``None`` when the layer is outside the support domain (the adapter
    then never calls router_gemm for it), otherwise ``fn(hidden) -> logits or None``.
    The decision is the same on every rank: it depends only on the model and
    mapping configuration.
    """
    reason = _layer_domain(gate, layer_idx, model_config)
    dispatch.log_capability(
        FEATURE_ID,
        f"router:{reason or 'engaged'}",
        not reason,
        reason or f"first layer {layer_idx}, pdl={int(pdl_enabled())}",
    )
    if reason:
        return None

    def _route(hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        return router_logits(hidden_states, gate.weight)

    return _route
