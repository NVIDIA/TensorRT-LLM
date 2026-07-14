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

"""Construction of TRT-LLM Mapping / ModelConfig / RoutingMethod objects.

The benchmark plumbs three TRT-LLM-internal config objects into
``create_moe``:

* :class:`Mapping`  — TP / EP / attention-DP topology derived from the
  :class:`ConfigSpec` parallel-mode shortcut (``DEP`` / ``TEP`` / ``DTP``
  / ``TTP`` / ``CUSTOM``).
* :class:`ModelConfig` — ``Mapping`` + quant config + per-run knobs
  (CUDA graph, max num tokens, low-precision combine).
* A concrete routing-method instance — picked by the registry value
  recorded on :class:`ModelSpec`.

Centralising this construction in one module makes the per-case execution
path easier to follow and keeps the ``ConfigurableMoE`` glue out of
the worker and CLI layers.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe.routing import (
    DeepSeekV3MoeRoutingMethod,
    DefaultMoeRoutingMethod,
    Llama4RenormalizeMoeRoutingMethod,
    MiniMaxM2MoeRoutingMethod,
    RenormalizeMoeRoutingMethod,
    RenormalizeNaiveMoeRoutingMethod,
    SigmoidRenormMoeRoutingMethod,
)
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from .backend import MoeModelConfig, resolve_deepseek_group_config
from .specs import ConfigSpec, ModelSpec

_PARALLEL_MODE_LAYOUTS: Dict[str, Dict[str, Any]] = {
    "DEP": {"moe_ep_size": "world", "moe_tp_size": 1, "enable_attention_dp": True},
    "TEP": {"moe_ep_size": "world", "moe_tp_size": 1, "enable_attention_dp": False},
    "DTP": {"moe_ep_size": 1, "moe_tp_size": "world", "enable_attention_dp": True},
    "TTP": {"moe_ep_size": 1, "moe_tp_size": "world", "enable_attention_dp": False},
}


def _resolve_mapping_layout(config: ConfigSpec, world_size: int) -> Tuple[int, int, bool]:
    """Resolve ``(moe_ep_size, moe_tp_size, enable_attention_dp)`` for a ConfigSpec."""
    if config.parallel_mode == "CUSTOM":
        if config.moe_ep_size is None or config.moe_tp_size is None:
            raise ValueError("parallel_mode=CUSTOM requires explicit moe_ep_size and moe_tp_size")
        moe_ep = int(config.moe_ep_size)
        moe_tp = int(config.moe_tp_size)
        enable_dp = (
            bool(config.enable_attention_dp) if config.enable_attention_dp is not None else False
        )
    else:
        layout = _PARALLEL_MODE_LAYOUTS.get(config.parallel_mode)
        if layout is None:
            raise ValueError(f"Unknown parallel_mode={config.parallel_mode!r}")
        moe_ep = world_size if layout["moe_ep_size"] == "world" else int(layout["moe_ep_size"])
        moe_tp = world_size if layout["moe_tp_size"] == "world" else int(layout["moe_tp_size"])
        enable_dp = bool(layout["enable_attention_dp"])
    if moe_ep * moe_tp != world_size:
        raise ValueError(
            f"moe_ep_size * moe_tp_size = {moe_ep * moe_tp} must equal world_size={world_size}"
        )
    return moe_ep, moe_tp, enable_dp


def _build_mapping_from_config(config: ConfigSpec, world_size: int) -> Mapping:
    """Build ``Mapping`` from a ``ConfigSpec`` + world size; sets ``rank=mpi_rank()``."""
    moe_ep, moe_tp, enable_dp = _resolve_mapping_layout(config, world_size)
    # gpus_per_node must match actual visible GPUs per node so that
    # mapping.local_rank (= rank % gpus_per_node) gives the correct device index.
    # The Mapping default (8) is wrong for multi-node runs with fewer GPUs per node.
    gpus_per_node = torch.cuda.device_count()
    mapping = Mapping(
        world_size=world_size,
        tp_size=world_size,
        moe_ep_size=moe_ep,
        moe_tp_size=moe_tp,
        enable_attention_dp=enable_dp,
        gpus_per_node=gpus_per_node,
    )
    mapping.rank = mpi_rank()
    return mapping


def _create_routing_method(
    routing_method_cls,
    top_k: int,
    num_experts: int,
    bias_dtype: torch.dtype,
    profile_model_config: MoeModelConfig,
):
    """Create a routing-method instance mirroring ``test_moe_module._create_routing_method``."""
    if routing_method_cls in (RenormalizeMoeRoutingMethod, DefaultMoeRoutingMethod):
        return routing_method_cls(top_k=top_k, force_enable_pytorch_op=True)

    if routing_method_cls in (RenormalizeNaiveMoeRoutingMethod, Llama4RenormalizeMoeRoutingMethod):
        return routing_method_cls(top_k=top_k)

    if routing_method_cls is DeepSeekV3MoeRoutingMethod:
        n_group, topk_group = resolve_deepseek_group_config(profile_model_config)
        e_score_correction_bias = torch.zeros(num_experts, dtype=bias_dtype, device="cuda")
        return routing_method_cls(
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=1.0,
            callable_e_score_correction_bias=lambda: e_score_correction_bias,
            is_fused=False,
        )

    if routing_method_cls is MiniMaxM2MoeRoutingMethod:
        e_score_correction_bias = torch.zeros(num_experts, dtype=bias_dtype, device="cuda")
        return routing_method_cls(
            top_k=top_k,
            num_experts=num_experts,
            callable_e_score_correction_bias=lambda: e_score_correction_bias,
        )

    if routing_method_cls is SigmoidRenormMoeRoutingMethod:
        return routing_method_cls(top_k=top_k, num_experts=num_experts)

    return routing_method_cls(top_k=top_k)


def _build_pretrained_config(
    num_experts: int, hidden_size: int, intermediate_size: int, dtype: torch.dtype
) -> PretrainedConfig:
    """Construct a HF-style ``PretrainedConfig`` for ``ConfigurableMoE``."""
    pc = PretrainedConfig()
    pc.num_experts = num_experts
    pc.hidden_size = hidden_size
    pc.intermediate_size = intermediate_size
    pc.torch_dtype = dtype
    return pc


def _build_model_config(
    *,
    model: ModelSpec,
    mapping: Mapping,
    moe_backend: str,
    use_cuda_graph: bool,
    max_num_tokens: int,
    use_low_precision_moe_combine: bool,
    dtype: torch.dtype,
) -> ModelConfig:
    """Build ``ModelConfig`` plumbed into ``create_moe``."""
    pretrained_config = _build_pretrained_config(
        model.num_experts, model.hidden_size, model.intermediate_size, dtype
    )

    quant_algo = model.quant_algo_enum
    quant_config = (
        QuantConfig(quant_algo=None) if quant_algo is None else QuantConfig(quant_algo=quant_algo)
    )

    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        quant_config=quant_config,
        moe_backend=moe_backend,
        moe_disable_finalize_fusion=False,
        max_num_tokens=max(int(max_num_tokens), 1),
        use_cuda_graph=use_cuda_graph,
        use_low_precision_moe_combine=use_low_precision_moe_combine,
    )
