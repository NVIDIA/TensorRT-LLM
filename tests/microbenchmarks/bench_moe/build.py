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

"""MoE module construction and introspection helpers.

This module owns the "build phase" of one timing case: it instantiates a
fresh :class:`ConfigurableMoE` via :func:`create_moe`, generates the
quantization-aware weights, loads them into the module, and provides the
small set of introspection helpers (``actual_backend`` / ``scheduler_kind``
/ ``comm_method`` / ``num_chunks``) used by the result schema.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch

from tensorrt_llm._torch.modules.fused_moe.interface import MoESchedulerKind, MoEWeightLoadingMode
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

from .backend import MoeBackendType, ensure_cute_dsl_importable_for_benchmark
from .mapping import _build_model_config, _create_routing_method
from .quantize import get_test_quant_params
from .specs import ConfigSpec, ModelSpec
from .utils import _ensure_dist_for_megamoe

# Map concrete MoE module class names to short backend identifiers used in
# results and the dashboard. Anything not in this table falls back to the
# upper-case class name.
_BACKEND_CLASS_TO_NAME: Dict[str, str] = {
    "CutlassFusedMoE": "CUTLASS",
    "TRTLLMGenFusedMoE": "TRTLLM",
    "CuteDslFusedMoE": "CUTEDSL",
    "DeepGemmFusedMoE": "DEEPGEMM",
    "DenseGEMMFusedMoE": "DENSEGEMM",
    "MegaMoEDeepGemm": "MEGAMOE_DEEPGEMM",
    "VanillaMoE": "VANILLA",
}


def _backend_name_from_module(moe) -> str:
    """Resolve ``actual_backend`` for both ConfigurableMoE and legacy modules."""
    backend_attr = getattr(moe, "backend", None)
    if backend_attr is not None and backend_attr is not moe:
        backend_cls = type(backend_attr).__name__
    else:
        backend_cls = type(moe).__name__
    return _BACKEND_CLASS_TO_NAME.get(backend_cls, backend_cls.upper())


def _scheduler_kind_name(moe) -> Optional[str]:
    """Return ``"EXTERNAL_COMM"`` / ``"FUSED_COMM"`` for the underlying backend."""
    backend = getattr(moe, "backend", None) or moe
    kind = getattr(backend, "scheduler_kind", None)
    if isinstance(kind, MoESchedulerKind):
        return kind.name
    return None


def _comm_method_name(moe) -> str:
    """Return the actual communication strategy class name, or ``"NONE"``."""
    if _scheduler_kind_name(moe) == "FUSED_COMM":
        return "NONE"
    comm = getattr(moe, "comm", None)
    if comm is None:
        return "NONE"
    return type(comm).__name__


def _calculate_num_chunks_safe(moe, all_rank_num_tokens: List[int]) -> Optional[int]:
    """Best-effort lookup of ``num_chunks`` for the case we are about to time."""
    scheduler = getattr(moe, "scheduler", None)
    if scheduler is None:
        return None
    fn = getattr(scheduler, "calculate_num_chunks", None)
    if fn is None:
        return None
    try:
        return int(fn(all_rank_num_tokens))
    except Exception:
        return None


def _create_moe_for_benchmark(**kwargs):
    ensure_cute_dsl_importable_for_benchmark()
    from tensorrt_llm._torch.modules.fused_moe.create_moe import create_moe

    return create_moe(**kwargs)


class _UnfusedSharedMoE(torch.nn.Module):
    """Pre-fusion baseline: routed MoE plus a separate shared-expert GatedMLP.

    The routed MoE is built without fused shared experts; a standalone shared-expert
    ``GatedMLP`` is summed onto its output, mirroring the pre-#11143 model path. The
    routed MoE and the shared GatedMLP are run with ``maybe_execute_in_parallel`` on
    a dedicated aux stream (same mechanism DeepseekV3MoE uses for the unfused path),
    so the routed grouped GEMM and the shared MLP can overlap. Multi-stream is
    enabled only under CUDA graph (``use_cuda_graph``), matching the real engine
    (cuda_graph_runner sets with_multi_stream(True) during capture); in eager the
    two run serially, since stream-switch host overhead makes overlap a net loss.
    Exposes the same ``forward(x, router_logits, **kwargs)`` signature as the MoE;
    introspection attributes (backend / routing_method / ...) delegate to the MoE.
    """

    def __init__(
        self,
        moe: torch.nn.Module,
        shared_mlp: torch.nn.Module,
        use_cuda_graph: bool = False,
    ):
        super().__init__()
        self.moe = moe
        self.shared_mlp = shared_mlp
        # Overlap only when the case is timed under CUDA graph (mirrors the real
        # model, which only enables multi-stream during graph capture).
        self._multi_stream = bool(use_cuda_graph)
        self.aux_stream = torch.cuda.Stream()
        self.event_main = torch.cuda.Event()
        self.event_shared = torch.cuda.Event()

    def __getattr__(self, name):
        # nn.Module.__getattr__ resolves registered submodules/params/buffers
        # (incl. ``moe``/``shared_mlp``); anything else (routing_method, backend,
        # comm, scheduler, ...) is delegated to the wrapped MoE for introspection.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("moe"), name)

    def forward(self, x, router_logits, **kwargs):
        from tensorrt_llm._torch.modules.multi_stream_utils import (
            maybe_execute_in_parallel,
            with_multi_stream,
        )

        with with_multi_stream(self._multi_stream):
            routed, shared = maybe_execute_in_parallel(
                lambda: self.moe.forward(x, router_logits, **kwargs),
                lambda: self.shared_mlp(x),
                self.event_main,
                self.event_shared,
                self.aux_stream,
            )
        return routed + shared


def _build_moe_module(
    *,
    model: ModelSpec,
    config: ConfigSpec,
    mapping: Mapping,
    moe_backend: str,
    use_cuda_graph: bool,
    max_num_tokens: int,
    use_low_precision_moe_combine: bool,
    enable_perfect_router: bool,
    dtype: torch.dtype,
    routing_logits_dtype: torch.dtype,
    device: torch.device,
):
    """Build a fresh ``ConfigurableMoE`` for one ``(backend, num_tokens)`` case.

    Returns ``(moe_module, routing_logits_dtype)``.
    """
    # Shared-expert support (PR #11143) currently only works in TTP: attention TP
    # (dp_size==1, so the fusion gate fires) + MoE TP (moe_ep_size==1; the EP fused
    # path is unimplemented). Reject other multi-GPU layouts (DEP/TEP/DTP) up front
    # instead of silently dropping the shared experts or hitting the dead EP path.
    # (Single GPU resolves to dp_size==1/moe_ep_size==1 for every mode, so it passes.)
    if model.n_shared_experts > 0 and (mapping.dp_size != 1 or mapping.moe_ep_size != 1):
        raise NotImplementedError(
            f"bench_moe shared-expert support (n_shared_experts={model.n_shared_experts}) "
            f"only supports TTP (attention TP + MoE TP: dp_size==1 and moe_ep_size==1). "
            f"Got parallel_mode={config.parallel_mode!r} -> dp_size={mapping.dp_size}, "
            f"moe_ep_size={mapping.moe_ep_size}. Re-run with --parallel_mode TTP "
            f"(or drop --n_shared_experts)."
        )

    if enable_perfect_router:
        os.environ["ENABLE_PERFECT_ROUTER"] = "1"
    else:
        os.environ.pop("ENABLE_PERFECT_ROUTER", None)

    mc = model.to_moe_model_config()
    swiglu_gptoss_style = model.swiglu_gptoss_style

    routing_method = _create_routing_method(
        model.routing_method_cls,
        top_k=mc.top_k,
        num_experts=mc.num_experts,
        bias_dtype=dtype,
        profile_model_config=mc,
    )

    model_config = _build_model_config(
        model=model,
        mapping=mapping,
        moe_backend=moe_backend,
        use_cuda_graph=use_cuda_graph,
        max_num_tokens=max_num_tokens,
        use_low_precision_moe_combine=use_low_precision_moe_combine,
        dtype=dtype,
    )

    _ensure_dist_for_megamoe(moe_backend, mapping.rank, mapping.world_size)

    probe_x = torch.randn(
        (max(1, mc.hidden_size // 32), mc.hidden_size), dtype=dtype, device=device
    )
    backend_type = MoeBackendType(moe_backend.upper())
    quant_algo = model.quant_algo_enum
    quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
        quant_algo, probe_x, backend_type
    )
    quant_kwargs.pop("ref_cls", None)

    num_local_experts = mc.num_experts // max(mapping.moe_ep_size, 1)
    quantize_util = quantize_util_cls(
        num_experts=mc.num_experts,
        dtype=dtype,
        intermediate_size=mc.intermediate_size,
        hidden_size=mc.hidden_size,
        quant_config=quant_config,
        bias=swiglu_gptoss_style,
        swiglu_gptoss_style=swiglu_gptoss_style,
        swiglu_alpha=model.swiglu_alpha if swiglu_gptoss_style else None,
        swiglu_beta=model.swiglu_beta if swiglu_gptoss_style else None,
        swiglu_limit=model.swiglu_limit if swiglu_gptoss_style else None,
        num_local_experts=num_local_experts,
    )

    weight_loading_mode = getattr(
        quantize_util, "weight_loading_mode", MoEWeightLoadingMode.VANILLA
    )

    swiglu_tensors = quantize_util.get_swiglu_tensors()

    moe = _create_moe_for_benchmark(
        routing_method=routing_method,
        num_experts=mc.num_experts,
        hidden_size=mc.hidden_size,
        intermediate_size=mc.intermediate_size,
        dtype=dtype,
        reduce_results=True,
        model_config=model_config,
        weight_loading_mode=weight_loading_mode,
        bias=swiglu_gptoss_style,
        swiglu_alpha=swiglu_tensors["swiglu_alpha"] if swiglu_tensors else None,
        swiglu_beta=swiglu_tensors["swiglu_beta"] if swiglu_tensors else None,
        swiglu_limit=swiglu_tensors["swiglu_limit"] if swiglu_tensors else None,
    )

    if quant_algo == QuantAlgo.W4A8_MXFP4_MXFP8:
        weights, _ref_weights, _ref_kwargs = quantize_util.prepare_weights_from_backend(
            moe, **quant_kwargs
        )
    else:
        weights = quantize_util.create_weights(**quant_kwargs)

    moe.load_weights([weights])
    moe.post_load_weights()
    moe.cuda(f"cuda:{torch.cuda.current_device()}")

    # "unfused" baseline: the routed MoE above was built without fused shared
    # experts (mapping.py forces pretrained_config.n_shared_experts=0 in this mode);
    # add a standalone shared-expert GatedMLP and sum it, mirroring the pre-fusion
    # model path so the benchmark can compare against the fused path.
    if model.n_shared_experts > 0 and model.shared_expert_mode == "unfused":
        from tensorrt_llm._torch.modules.gated_mlp import GatedMLP

        shared_mlp = GatedMLP(
            hidden_size=mc.hidden_size,
            intermediate_size=model.n_shared_experts * mc.intermediate_size,
            bias=False,
            dtype=dtype,
            config=model_config,
            reduce_output=False,
            # Route the shared GatedMLP's FP8 block-scale GEMM through cute_dsl
            # (cute_dsl_fp8_gemm_blackwell) instead of the SM100f default DeepGEMM
            # `fp8_swap_ab_gemm`. The DeepGEMM path hits an intermittent
            # cudaErrorIllegalAddress when the routed TRTLLM-Gen MoE and this shared
            # GatedMLP run in the same process across growing token counts.
            use_cute_dsl_blockscaling_mm=True,
        )
        shared_mlp.cuda(f"cuda:{torch.cuda.current_device()}")
        moe = _UnfusedSharedMoE(moe, shared_mlp, use_cuda_graph=use_cuda_graph)

    return moe, routing_logits_dtype
