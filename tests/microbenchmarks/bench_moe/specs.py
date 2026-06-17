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

"""Structured specs used throughout the bench_moe pipeline.

Spec dataclasses are intentionally light: they hold values only, expose tiny
helpers for JSON serialisation (used by the dashboard schema), and rely on
sibling modules to convert them into runtime objects (Mappings, MoE modules,
RoutingMethods, etc.).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tensorrt_llm._torch.modules.fused_moe.routing import (
    DeepSeekV3MoeRoutingMethod,
    DefaultMoeRoutingMethod,
    Llama4RenormalizeMoeRoutingMethod,
    MiniMaxM2MoeRoutingMethod,
    RenormalizeMoeRoutingMethod,
    RenormalizeNaiveMoeRoutingMethod,
    SigmoidRenormMoeRoutingMethod,
)
from tensorrt_llm.models.modeling_utils import QuantAlgo

from .backend import MoeBackendType, MoeModelConfig

_ROUTING_METHODS: Dict[str, type] = {
    "DEFAULT": DefaultMoeRoutingMethod,
    "RENORMALIZE": RenormalizeMoeRoutingMethod,
    "RENORMALIZE_NAIVE": RenormalizeNaiveMoeRoutingMethod,
    "LLAMA4_RENORMALIZE": Llama4RenormalizeMoeRoutingMethod,
    "DEEPSEEK_V3": DeepSeekV3MoeRoutingMethod,
    "MINIMAX_M2": MiniMaxM2MoeRoutingMethod,
    "SIGMOID_RENORM": SigmoidRenormMoeRoutingMethod,
}

_ROUTING_NAME_BY_CLS: Dict[type, str] = {cls: name for name, cls in _ROUTING_METHODS.items()}
_ALL_BACKENDS: List[str] = [b.value for b in MoeBackendType]
_COMM_METHODS: Tuple[str, ...] = (
    "AUTO",
    "NVLINK_ONE_SIDED",
    "NVLINK_TWO_SIDED",
    "DEEPEP",
    "DEEPEPLOWLATENCY",
    "ALLGATHER",
)
_FORCED_COMM_ENV_VALUES: Tuple[str, ...] = (
    "NVLINK_ONE_SIDED",
    "NVLINK_TWO_SIDED",
    "DEEPEP",
    "DEEPEPLOWLATENCY",
    "ALLGATHER",
)


def _to_jsonable_dict(obj: Any) -> Dict[str, Any]:
    """``dataclasses.asdict`` with nested tuples converted to lists.

    Several specs use ``Tuple[...]`` fields for hashability/immutability.
    ``asdict`` preserves tuples; downstream consumers serialize the result
    to JSON, which treats tuples and lists identically, but list form is
    the historical wire format and avoids surprising callers that may
    later mutate.
    """

    def _walk(value: Any) -> Any:
        if isinstance(value, (tuple, list)):
            return [_walk(v) for v in value]
        if isinstance(value, dict):
            return {k: _walk(v) for k, v in value.items()}
        return value

    return _walk(asdict(obj))


@dataclass(frozen=True)
class ModelSpec:
    """Static MoE model description.

    A built-in name resolves to one of the entries in ``BUILT_IN_MODELS``.
    Custom shapes pass ``name="custom"`` and fill the remaining fields
    explicitly. ``routing_method`` is the registry key from
    ``_ROUTING_METHODS``; resolution to a concrete class happens lazily so the
    spec stays JSON-serializable.
    """

    name: str
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    quant_algo: Optional[str]
    routing_method: str
    n_group: Optional[int] = None
    topk_group: Optional[int] = None
    n_shared_experts: int = 0
    # How shared experts are realized when n_shared_experts > 0:
    #   "fused"   -> fold them into the routed-expert grouped GEMM (PR #11143).
    #   "unfused" -> routed MoE (no fusion) + a separate shared GatedMLP, summed
    #                (the pre-fusion baseline, for measuring fusion's net benefit).
    shared_expert_mode: str = "fused"
    swiglu_alpha: float = 1.0
    swiglu_beta: float = 0.0
    swiglu_limit: float = float("inf")

    @property
    def routing_method_cls(self) -> type:
        return _ROUTING_METHODS[self.routing_method]

    @property
    def quant_algo_enum(self) -> Optional[QuantAlgo]:
        return QuantAlgo[self.quant_algo] if self.quant_algo is not None else None

    @property
    def swiglu_gptoss_style(self) -> bool:
        return (
            self.swiglu_alpha != 1.0 or self.swiglu_beta != 0.0 or self.swiglu_limit != float("inf")
        )

    def to_moe_model_config(self) -> MoeModelConfig:
        return MoeModelConfig(
            num_experts=int(self.num_experts),
            top_k=int(self.top_k),
            hidden_size=int(self.hidden_size),
            intermediate_size=int(self.intermediate_size),
            n_group=self.n_group,
            topk_group=self.topk_group,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = _to_jsonable_dict(self)
        d["routing_method_class"] = self.routing_method_cls.__name__
        return d


@dataclass(frozen=True)
class RoutingControlSpec:
    """Advanced routing-control knobs for one workload.

    ``comm_pattern`` and ``expert_pattern`` describe the requested traffic
    shape; ``routing_mode`` picks between native logits realization and forced
    supplied-topk; ``projection_policy`` controls what happens when native
    logits cannot exactly express the requested pattern.

    By default, balanced patterns build a deterministic RoutingPlan that is
    projected to native logits or supplied directly, depending on
    ``routing_mode``. Set both ``comm_pattern=random`` and
    ``expert_pattern=random`` to leave routing uncontrolled and use random
    router logits.
    """

    routing_mode: str = "native"  # "native" | "forced"
    projection_policy: str = "project"  # "project" | "reject"
    comm_pattern: str = "balanced_alltoall"
    expert_pattern: str = "balanced"
    routing_pattern_file: Optional[str] = None
    per_rank_num_tokens: Optional[Tuple[int, ...]] = None
    routing_dump_matrix: bool = False
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable_dict(self)

    @property
    def is_active(self) -> bool:
        """True when this spec asks for planned routing instead of random logits.

        Used to decide whether to dispatch through routing-control planning or
        keep the normal benchmark path.
        """
        return (
            self.routing_mode != "native"
            or self.routing_pattern_file is not None
            or self.per_rank_num_tokens is not None
            or not (self.comm_pattern == "random" and self.expert_pattern == "random")
        )


@dataclass(frozen=True)
class WorkloadSpec:
    """Workload for one timing case after the model is fixed."""

    num_tokens: int
    routing_control: RoutingControlSpec = field(default_factory=RoutingControlSpec)

    def to_dict(self, per_rank_num_tokens: Optional[List[int]] = None) -> Dict[str, Any]:
        return {
            "num_tokens": int(self.num_tokens),
            "per_rank_num_tokens": (
                [int(v) for v in per_rank_num_tokens] if per_rank_num_tokens is not None else None
            ),
            "routing_control": self.routing_control.to_dict(),
        }


@dataclass(frozen=True)
class ConfigSpec:
    """One executable MoE runtime configuration."""

    backend: str
    parallel_mode: str  # "DEP" | "TEP" | "DTP" | "TTP" | "CUSTOM"
    moe_ep_size: Optional[int] = None
    moe_tp_size: Optional[int] = None
    enable_attention_dp: Optional[bool] = None
    comm_method: str = "AUTO"
    cuda_graph: bool = True
    use_low_precision_moe_combine: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable_dict(self)


@dataclass(frozen=True)
class SearchSpec:
    """Description of which ConfigSpec axes to expand into candidates."""

    mode: str = "none"  # "none" | "backend" | "comm" | "parallel" | "full" | comma-joined axes
    backends: Tuple[str, ...] = ()
    parallel_modes: Tuple[str, ...] = ()
    comm_methods: Tuple[str, ...] = ()
    cuda_graph_options: Tuple[bool, ...] = ()
    combine_precision_options: Tuple[bool, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable_dict(self)


@dataclass
class RunResult:
    """Result of timing a single ``(model, workload, config)`` triple.

    Stays a regular dataclass on the worker side so we can mutate fields while
    incrementally collecting data.
    """

    model: ModelSpec
    workload: WorkloadSpec
    config: ConfigSpec
    status: str = "success"  # "success" | "skipped" | "failed"
    skip_reason: Optional[str] = None
    actual_backend: Optional[str] = None
    actual_comm_method: Optional[str] = None
    actual_comm_fallback_reason: Optional[str] = None
    scheduler_kind: Optional[str] = None
    moe_ep_size: Optional[int] = None
    moe_tp_size: Optional[int] = None
    enable_attention_dp: Optional[bool] = None
    num_chunks: Optional[int] = None
    per_rank_num_tokens: List[int] = field(default_factory=list)
    status_per_rank: Dict[str, str] = field(default_factory=dict)
    instrumentation: Dict[str, Any] = field(default_factory=dict)
    latency_ms: Dict[str, Any] = field(default_factory=dict)
    phase_times_ms: Dict[str, Any] = field(default_factory=dict)
    kernel_breakdown: Dict[str, Any] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    overlap: Dict[str, Any] = field(default_factory=dict)
    bottleneck: Optional[str] = None
    routing_control: Dict[str, Any] = field(default_factory=dict)


# Built-in MoE model registry. Each entry is a ready-to-run ``ModelSpec`` for a
# publicly known MoE checkpoint.

BUILT_IN_MODELS: Dict[str, ModelSpec] = {
    "qwen1.5_moe": ModelSpec(
        name="qwen1.5_moe",
        num_experts=60,
        top_k=4,
        hidden_size=2048,
        intermediate_size=1408,
        quant_algo="FP8",
        routing_method="RENORMALIZE",
    ),
    "deepseek_v2_lite": ModelSpec(
        name="deepseek_v2_lite",
        num_experts=64,
        top_k=6,
        hidden_size=2048,
        intermediate_size=1408,
        quant_algo="FP8_BLOCK_SCALES",
        routing_method="DEEPSEEK_V3",
    ),
    "deepseek_v3": ModelSpec(
        name="deepseek_v3",
        num_experts=256,
        top_k=8,
        hidden_size=7168,
        intermediate_size=2048,
        quant_algo="FP8_BLOCK_SCALES",
        routing_method="DEEPSEEK_V3",
        n_group=8,
        topk_group=4,
    ),
    "kimi_k2": ModelSpec(
        name="kimi_k2",
        num_experts=384,
        top_k=8,
        hidden_size=7168,
        intermediate_size=2048,
        quant_algo="FP8_BLOCK_SCALES",
        routing_method="DEEPSEEK_V3",
    ),
    # DeepSeek-V4-Pro: 1.6T total / 49B activated. quant_algo intentionally
    # left None: pass --quant on the CLI to pin the mode (the released
    # checkpoint mixes FP4 experts with FP8 elsewhere which has no single
    # QuantAlgo match).
    "deepseek_v4_pro": ModelSpec(
        name="deepseek_v4_pro",
        num_experts=384,
        top_k=6,
        hidden_size=7168,
        intermediate_size=3072,
        quant_algo=None,
        routing_method="RENORMALIZE",
    ),
    # DeepSeek-V4-Flash: 284B total / 13B activated.
    "deepseek_v4_flash": ModelSpec(
        name="deepseek_v4_flash",
        num_experts=256,
        top_k=6,
        hidden_size=4096,
        intermediate_size=2048,
        quant_algo=None,
        routing_method="RENORMALIZE",
    ),
    "mixtral_8x7b": ModelSpec(
        name="mixtral_8x7b",
        num_experts=8,
        top_k=2,
        hidden_size=4096,
        intermediate_size=14336,
        quant_algo="FP8",
        routing_method="RENORMALIZE",
    ),
    "gpt_oss_120b": ModelSpec(
        name="gpt_oss_120b",
        num_experts=128,
        top_k=4,
        hidden_size=2880,
        intermediate_size=2880,
        quant_algo="W4A8_MXFP4_MXFP8",
        routing_method="RENORMALIZE",
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        swiglu_limit=7.0,
    ),
}
