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
"""Abstract base class for MoE execution units."""

from __future__ import annotations

import abc
import os
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from ...model_config import ModelConfig
from ...utils import ActivationType, AuxStreamType, is_gated_activation
from .impl_blocks import MoEEplbWeightLayoutMixin, MoEExecutionContractMixin, MoEWeightOwnerMixin
from .impl_contract import MoEDeployment, MoEEligibility, MoEEplbBinding, MoEProblem, MoERunContext
from .interface import MoESchedulerKind, MoEWeightLoadingMode, _compute_ep_partition
from .routing import BaseMoeRoutingMethod

if TYPE_CHECKING:
    from tensorrt_llm._torch.utils import Fp4QuantizedTensor

    from .impl_identity import MoEImplDescriptor


STANDALONE_MOE_IMPL_ERROR = (
    "{name} is an execution unit, not a complete MoE layer. "
    "Construct it through create_moe() / ConfigurableMoE."
)


def apply_moe_impl_construction_state(
    module: nn.Module,
    *,
    routing_method: BaseMoeRoutingMethod,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: Optional[torch.dtype] = None,
    reduce_results: bool = False,
    model_config: ModelConfig = ModelConfig(),
    aux_stream_dict: Optional[dict[AuxStreamType, torch.cuda.Stream]] = None,
    weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA,
    bias: bool = False,
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[torch.Tensor] = None,
    swiglu_limit_scalar: Optional[float] = None,
    layer_idx: Optional[int] = None,
    activation_type: ActivationType = ActivationType.Swiglu,
    init_load_balancer: bool = False,
) -> None:
    """Install the construction state backends used to get from ``MoE.__init__``.

    ``MoEImplBase`` does not inherit ``MoE``, so nothing else sets the attributes
    every execution unit reads off ``self``: ``hidden_size``, ``quant_config``,
    ``mapping``, the partition sizes, and a default EPLB layout the wrapper later
    overwrites. Layer-only work stays on ``MoE`` / ``ConfigurableMoE``
    (``_register_layer``, ``_init_load_balancer``, ``AllReduce``, DWDP layout).

    ``init_load_balancer`` defaults to ``False`` because an execution unit never
    owns a load balancer -- the wrapper does. Passing ``True`` asks for a
    standalone layer, which these classes no longer are, so it is rejected.
    """
    if init_load_balancer:
        raise TypeError(STANDALONE_MOE_IMPL_ERROR.format(name=type(module).__name__))

    # The EPLB fields written below are only defaults. A backend that already
    # passed a real binding to ``MoEImplBase.__init__`` would have it silently
    # re-derived into the non-EPLB partition (``num_slots`` collapsing to
    # ``num_experts``, ``layer_load_balancer`` back to None) -- the exact names
    # the quantization layer reads to size expert weights. Fail before any of
    # this state is written.
    if getattr(module, "eplb", None) is not None:
        raise ValueError(
            f"{type(module).__name__} installed an EPLB binding via "
            f"MoEImplBase.__init__ and then called "
            f"apply_moe_impl_construction_state(), which would overwrite it. "
            f"Call this first and let the binding land last, or drop the "
            f"binding."
        )

    module.routing_method = routing_method
    module.num_experts = num_experts
    module.hidden_size = hidden_size
    module.intermediate_size = intermediate_size
    module.weight_loading_mode = weight_loading_mode
    module.bias = bias
    module.dtype = dtype
    module.reduce_results = reduce_results
    module.swiglu_alpha = swiglu_alpha
    module.swiglu_beta = swiglu_beta
    module.swiglu_limit = swiglu_limit
    module.swiglu_limit_scalar = swiglu_limit_scalar
    module.layer_idx = layer_idx
    module.layer_idx_str = str(layer_idx) if layer_idx is not None else None
    module.activation_type = int(activation_type)
    module.is_gated_activation = is_gated_activation(activation_type)
    module.intermediate_size_expand_ratio = 2 if module.is_gated_activation else 1

    module.quant_config = model_config.quant_config
    module.force_dynamic_quantization = getattr(model_config, "force_dynamic_quantization", False)

    module.cluster_rank = model_config.mapping.moe_cluster_rank
    module.cluster_size = model_config.mapping.moe_cluster_size
    module.smart_router = module.cluster_size > 1
    module.rank = model_config.mapping.rank
    module.tp_rank = model_config.mapping.moe_tp_rank
    module.tp_size = model_config.mapping.moe_tp_size
    module.ep_size = model_config.mapping.moe_ep_size
    module.ep_rank = model_config.mapping.moe_ep_rank
    module.moe_backend = model_config.moe_backend
    module.use_dp = model_config.mapping.enable_attention_dp
    module.mapping = model_config.mapping
    module.parallel_rank = module.mapping.tp_rank
    module.parallel_size = module.mapping.tp_size
    module.intermediate_size_per_partition = intermediate_size // module.tp_size

    # AllReduce belongs to the layer, not to an execution unit. The attribute is
    # kept so an inherited ``MoE`` code path reading ``self.all_reduce`` sees the
    # same None it saw when ``reduce_results`` was False, rather than an
    # AttributeError from a completely different-looking place.
    module.all_reduce = None
    module.enable_dummy_allreduce = os.environ.get("TRTLLM_ENABLE_DUMMY_ALLREDUCE", "0") == "1"

    # Same defaults ``MoE.__init__`` used when ``init_load_balancer=False``.
    # ConfigurableMoE overwrites the EPLB fields via ``_BACKEND_SYNC_ATTRS``.
    module.aux_stream_dict = aux_stream_dict
    module.layer_load_balancer = None
    module.repeat_idx = 0
    module.repeat_count = 1
    expert_size, slot_start, slot_end = _compute_ep_partition(
        module.num_experts, module.ep_size, module.ep_rank
    )
    module.expert_size_per_partition = expert_size
    module.num_slots = module.num_experts
    module.slot_start = slot_start
    module.slot_end = slot_end
    module.initial_local_expert_ids = list(range(slot_start, slot_end))
    module.initial_global_assignments = list(range(module.num_experts))
    module.allreduce = None


class MoEImplBase(
    MoEExecutionContractMixin, MoEWeightOwnerMixin, MoEEplbWeightLayoutMixin, nn.Module, abc.ABC
):
    """An execution unit. NOT a complete MoE layer.

    Takes the concrete halves of the three blocks an expert-weight owner needs --
    what it declares to the scheduler, the weights themselves, and the
    weight-side EPLB layout -- from the mixins, which ``MoE`` includes as well.
    The abstract contract is restated here rather than shared, because the two
    bases do not promise the same thing: ``run_moe`` below is deliberately
    narrower than ``MoE.run_moe``, and only this class enforces the contract at
    construction.

    ``eplb`` is still optional: backends that have not moved to the binding yet
    pass ``None`` and let ``apply_moe_impl_construction_state`` install the same
    default layout ``MoE.__init__`` did. A later EPLB item makes it required.

    Deliberately does NOT inherit ``MoE``. Three consequences the design relies
    on:

    - no ``forward`` / ``forward_impl``, so it cannot be mistaken for a layer;
    - no ``_register_layer``, so double EPLB registration is impossible at the
      type level and needs no runtime guard;
    - ``ABCMeta`` is real, so a missing method fails at CONSTRUCTION -- unlike
      ``MoE`` today, whose ``@abstractmethod`` markers do not bite because
      ``MoE`` is declared without ``ABCMeta``.
    """

    descriptor: "MoEImplDescriptor"  # set by every concrete subclass

    # The other scheduler-facing defaults live in ``MoEExecutionContractMixin``.
    # This one cannot: its default needs ``MoESchedulerKind`` from ``interface``,
    # which ``impl_blocks`` cannot import without a cycle.
    scheduler_kind: MoESchedulerKind = MoESchedulerKind.EXTERNAL_COMM

    def __init__(self, *, eplb: Optional[MoEEplbBinding] = None) -> None:
        super().__init__()
        # Layout is known BEFORE create_weights, because weight shapes depend
        # on it. Passing it here is what makes post-hoc setattr unnecessary.
        self.eplb = eplb
        if eplb is None:
            return
        # Project the binding onto the attribute names the quantization layer
        # reads off the weight owner (``module.expert_size_per_partition`` and
        # friends). Plain attributes rather than properties, because the DWDP
        # fixup rewrites the layout in place after construction.
        self.layer_idx = eplb.layer_idx
        self.num_experts = eplb.num_experts
        self.num_slots = eplb.num_slots
        self.slot_start = eplb.slot_start
        self.slot_end = eplb.slot_end
        self.expert_size_per_partition = eplb.expert_size_per_partition
        # Lists, not tuples: call sites slice and index these.
        self.initial_local_expert_ids = list(eplb.initial_local_expert_ids)
        self.initial_global_assignments = list(eplb.initial_global_assignments)
        self.layer_load_balancer = eplb.layer_load_balancer

    # ---- selection (pure; no GPU, no env, no import probe) ----------------
    @classmethod
    @abc.abstractmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility: ...

    # ---- weight lifecycle -------------------------------------------------
    # ``create_weights`` / ``load_weights`` / ``_check_configs`` are working
    # defaults from ``MoEWeightOwnerMixin``; backends that need more (TRTLLMGen,
    # MegaMoE-CuteDsl) override them.

    @abc.abstractmethod
    def _get_quant_method(self) -> object:
        """Resolve the quantization method that owns this backend's weights.

        No default: a Cutlass-layout NVFP4 method is not interchangeable with a
        TRTLLMGen one.
        """
        ...

    # ---- execution --------------------------------------------------------
    @abc.abstractmethod
    def quantize_input(
        self, x: "torch.Tensor | Fp4QuantizedTensor", **kwargs: object
    ) -> "tuple[torch.Tensor, torch.Tensor | None] | dict": ...

    # Narrower than ``MoE.run_moe``, which also takes a keyword-only
    # ``workspace``. Not a drift: the impl already allocates that scratch
    # itself, through ``get_workspaces`` below. What the scheduler still owns is
    # its LIFETIME -- one allocation reused across chunks and alternated
    # between streams, so it outlives a single call and travels back in through
    # the signature. This signature is the state after that lifetime moves
    # inside the impl; impls arriving on this base drop the parameter as they
    # do.
    @abc.abstractmethod
    def run_moe(self, ctx: MoERunContext) -> torch.Tensor: ...

    # ---- impl-owned resources: produced here, never passed in -------------
    def get_workspaces(self, *args: object, **kwargs: object) -> "list[dict] | None":
        return None
