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

import os
import weakref
from abc import abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, final

import torch
from torch import nn

from ...distributed.ops import reducescatter
from .impl_blocks import (MoEEplbWeightLayoutMixin, MoEExecutionContractMixin,
                          MoEWeightOwnerMixin)
from .impl_contract import (MoEDeployment, MoEEligibility, MoEProblem,
                            MoERejectReason, MoERunContext)

# Route on the host (fused noaux_tc + post-topk pipeline) instead of inside
# the trtllm-gen cubin. The in-cubin top-k tier for large expert counts
# (896 experts / top-16) register-spills and costs ~33 us/layer at decode
# batch 5..64 vs ~10 us for the post-topk pipeline; the separated path is
# the same math the attention-DP deployments already run.
#
# Lives here rather than next to either reader because both need it: the
# scheduler decides whether to precompute top-k at all, and TRTLLMGenFusedMoE
# decides whether its kernel may route again.
FORCE_SEPARATED_ROUTING = os.environ.get(
    "TLLM_TRTLLMGEN_FORCE_SEPARATED_ROUTING", "0") == "1"


def _reject(reason: MoERejectReason, detail: str) -> MoEEligibility:
    """Create a silent ``can_implement`` rejection."""
    return MoEEligibility.no(reason, detail)


from ...model_config import ModelConfig
from ...utils import (ActivationType, AuxStreamType, Fp4QuantizedTensor,
                      get_model_extra_attrs, is_gated_activation,
                      is_torch_compiling)
from .routing import (BaseMoeRoutingMethod, RoutingMethodType,
                      get_cached_perfect_router_logits,
                      precompute_common_perfect_router_logits)


def _compute_ep_partition(num_experts: int, ep_size: int,
                          ep_rank: int) -> tuple:
    """Compute per-rank expert count and slot boundaries.

    Uses ceil/floor distribution: ranks 0..remainder-1 hold (base+1) experts
    and remaining ranks hold base. Covers all experts even when
    num_experts % ep_size != 0.

    Returns:
        (expert_size, slot_start, slot_end)
    """
    # Reject num_experts < ep_size: would yield zero-expert ranks, which no
    # MoE backend / comm strategy supports end-to-end. The downstream alltoall
    # op has a backstop check (numExperts >= epSize) but the AllGatherReduceScatter
    # fallback path bypasses it, so guard upfront here.
    if num_experts < ep_size:
        raise ValueError(
            f"num_experts ({num_experts}) must be >= ep_size ({ep_size}); "
            f"configurations producing ranks with zero local experts are not supported."
        )
    base = num_experts // ep_size
    remainder = num_experts % ep_size
    expert_size = base + (1 if ep_rank < remainder else 0)
    slot_start = ep_rank * base + min(ep_rank, remainder)
    return expert_size, slot_start, slot_start + expert_size


class MoEWeightLoadingMode(Enum):
    # Gate and up projection are not fused
    VANILLA = 0
    # Gate and up projection are fused
    FUSED_GATE_UP_PROJ = 1
    # Custom W4A8 weights from examples/quantization/quantize_mixed_precision_moe.py
    W4A8_CUSTOM = 2


class MoESchedulerKind(Enum):
    """Selects which forward-execution scheduler ConfigurableMoE picks for a backend.

    Backends declare this via the ``scheduler_kind`` class attribute on
    ``MoEImplBase`` (execution units) or ``MoE`` (self-contained layers).
    ``ConfigurableMoE`` reads it once at init time to construct the
    matching scheduler and to gate communication-strategy creation.

    The axis is whether the cross-rank EP exchange is fused into the MoE
    kernel or is a separate host-orchestrated step:

    - ``EXTERNAL_COMM``: comm lives outside the MoE kernel boundary; the
      scheduler issues ``Communication.dispatch`` / ``Communication.combine``
      from the host with per-chunk EPLB hooks and optional multi-stream
      chunk overlap (Cutlass, DeepGemm, CuteDSL, DenseGEMM, TRTLLMGen).
    - ``FUSED_COMM``: comm is fused into the backend's fused kernel via
      NVLink SymmBuffer (DeepGEMM ``fp8_fp4_mega_moe``). No host comm;
      lockstep chunk launches; EPLB statistic update with
      ``ignore_allreduce=False``.
    """

    EXTERNAL_COMM = "external_comm"
    FUSED_COMM = "fused_comm"


def extract_extra_attrs(layer_idx: str):
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs are not set"

    moe_layers = extra_attrs.get("moe_layers", None)
    assert moe_layers is not None, "No MoE layers registered"
    moe_layer_ref = moe_layers.get(layer_idx)
    assert moe_layer_ref is not None, f"Cannot find MoE layer for layer_idx={layer_idx}"
    moe_layer = moe_layer_ref() if callable(moe_layer_ref) else None
    assert moe_layer is not None, f"MoE layer for layer_idx={layer_idx!r} is no longer alive"

    return moe_layer


@torch.library.custom_op("trtllm::moe_custom_op", mutates_args=())
def moe_custom_op(
    layer_idx: str,
    x: torch.Tensor,
    x_sf: Optional[torch.Tensor],
    is_swizzled: bool,
    router_logits: torch.Tensor,
    do_finalize: bool,
    output_dtype: Optional[torch.dtype],
    all_rank_num_tokens: Optional[List[int]],
    use_dp_padding: Optional[bool],
) -> List[torch.Tensor]:
    moe_layer = extract_extra_attrs(layer_idx)

    hidden_states = x if x_sf is None else Fp4QuantizedTensor(
        x, x_sf, is_swizzled)

    res = moe_layer.forward_impl(
        hidden_states,
        router_logits,
        do_finalize=do_finalize,
        output_dtype=output_dtype,
        all_rank_num_tokens=all_rank_num_tokens,
        use_dp_padding=use_dp_padding,
    )

    if do_finalize:
        return [res]
    else:
        return res


@moe_custom_op.register_fake
def _(
    layer_idx,
    x,
    x_sf,
    is_swizzled,
    router_logits,
    do_finalize,
    output_dtype,
    all_rank_num_tokens,
    use_dp_padding,
):
    moe_layer = extract_extra_attrs(layer_idx)
    hidden_states = x if x_sf is None else Fp4QuantizedTensor(
        x, x_sf, is_swizzled)
    res = moe_layer.forward_fake(
        hidden_states,
        router_logits,
        do_finalize=do_finalize,
        output_dtype=output_dtype,
        all_rank_num_tokens=all_rank_num_tokens,
        use_dp_padding=use_dp_padding,
    )

    if do_finalize:
        return [res]
    else:
        return res


class MoE(MoEExecutionContractMixin, MoEWeightOwnerMixin,
          MoEEplbWeightLayoutMixin, nn.Module):
    """
    Fused Mixture of Experts (MoE) Layer interface.

    A complete layer that also happens to own expert weights, so it takes the
    scheduler-facing contract, weight-owner, and weight-side EPLB layout blocks
    from ``impl_blocks``. What is stated *here* is the complete-layer half --
    ``forward`` / ``forward_impl``, routing, reduce/allreduce, layer
    registration, and the EPLB forward-time orchestration the wrapper drives --
    plus this layer's own abstract contract, which ``MoEImplBase`` deliberately
    states differently.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.
        aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary CUDA streams for overlapping.
    """

    # The other scheduler-facing defaults come from
    # ``MoEExecutionContractMixin``, so this class and ``MoEImplBase`` cannot
    # drift apart. This one cannot move there: its default needs
    # ``MoESchedulerKind``, defined in this module, and ``impl_blocks`` is
    # imported *by* this module.
    scheduler_kind: MoESchedulerKind = MoESchedulerKind.EXTERNAL_COMM

    @classmethod
    @abstractmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        """Purely evaluate ``p`` and ``d`` without probing runtime state.

        Abstain rather than reject when a required problem field is unknown.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement can_implement method")

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        aux_stream_dict: Optional[Dict[AuxStreamType,
                                       torch.cuda.Stream]] = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
        VANILLA,
        bias: bool = False,
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        swiglu_limit: Optional[torch.Tensor] = None,
        swiglu_limit_scalar: Optional[float] = None,
        layer_idx: Optional[int] = None,
        activation_type: ActivationType = ActivationType.Swiglu,
        init_load_balancer: bool = True,
    ):
        from ...distributed import AllReduce

        super().__init__()
        self.routing_method = routing_method
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.weight_loading_mode = weight_loading_mode
        self.bias = bias
        self.dtype = dtype
        self.reduce_results = reduce_results
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.swiglu_limit = swiglu_limit
        # Uniform-across-experts scalar variant of swiglu_limit, consumed by
        # FP8 paths (DeepGEMM Triton kernel, TRTLLMGen FP8 separate-activation
        # kernel) that don't actually need a per-expert tensor. Distinct from
        # `swiglu_limit` (kept for NVFP4 fused-activation cubins that *do*
        # consume per-expert values via fc31_alpha rescaling).
        self.swiglu_limit_scalar = swiglu_limit_scalar
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx) if layer_idx is not None else None
        self.activation_type = int(activation_type)
        # Note:
        # - for gated activations, there should be with gate and up projections, so the intermediate size should be expanded by 2.
        # - for non-gated activations, there is only one up projection (no gate projection), so the intermediate size should not be expanded.
        self.is_gated_activation = is_gated_activation(activation_type)
        self.intermediate_size_expand_ratio = 2 if self.is_gated_activation else 1

        self._register_layer(model_config)

        # could be modified later
        self.quant_config = model_config.quant_config
        self.force_dynamic_quantization = getattr(model_config,
                                                  'force_dynamic_quantization',
                                                  False)

        self.cluster_rank = model_config.mapping.moe_cluster_rank
        self.cluster_size = model_config.mapping.moe_cluster_size
        self.smart_router = True if self.cluster_size > 1 else False

        self.rank = model_config.mapping.rank

        self.tp_rank = model_config.mapping.moe_tp_rank
        self.tp_size = model_config.mapping.moe_tp_size

        self.ep_size = model_config.mapping.moe_ep_size
        self.ep_rank = model_config.mapping.moe_ep_rank

        # Non-divisible EP divisibility is gated inside _init_load_balancer()
        # because the correct quantity to check depends on whether EPLB is
        # active (num_slots vs num_experts), and num_slots is only known
        # after the load balancer config is consulted.
        #
        # When init_load_balancer=False (the wrapper path, e.g. ConfigurableMoE
        # creating an inner backend), the wrapper is responsible for the
        # divisibility contract and we skip the check entirely — see
        # ConfigurableMoE._reject_non_divisible_ep_backend(), which runs it
        # against this backend's class once the backend exists. The wrapper's
        # own _init_load_balancer cannot: it runs before that.

        self.moe_backend = model_config.moe_backend
        self.use_dp = model_config.mapping.enable_attention_dp

        # All ranks participate in allreduce regardless of EP/TP combination
        self.mapping = model_config.mapping
        self.parallel_rank = self.mapping.tp_rank
        self.parallel_size = self.mapping.tp_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size

        self.all_reduce = None
        # Debug function for eliminating imbalance during performance analysis.
        self.enable_dummy_allreduce = os.environ.get(
            "TRTLLM_ENABLE_DUMMY_ALLREDUCE", "0") == "1"
        if not self.use_dp and self.mapping.tp_size > 1:
            self.all_reduce = AllReduce(
                mapping=self.mapping,
                strategy=model_config.allreduce_strategy,
                dtype=self.dtype)
        elif self.enable_dummy_allreduce:
            from tensorrt_llm.functional import AllReduceStrategy
            self.all_reduce = AllReduce(mapping=self.mapping,
                                        strategy=AllReduceStrategy.NCCL,
                                        dtype=self.dtype)

        # Initialize load balancer related attributes
        if init_load_balancer:
            self._init_load_balancer(model_config, aux_stream_dict)
        else:
            # When init_load_balancer=False, initialize minimal attributes
            # These will be synced from the parent wrapper (e.g., ConfigurableMoE) later
            self.aux_stream_dict = aux_stream_dict
            self.layer_load_balancer = None
            self.repeat_idx = 0
            self.repeat_count = 1
            _size, _start, _end = _compute_ep_partition(self.num_experts,
                                                        self.ep_size,
                                                        self.ep_rank)
            self.expert_size_per_partition = _size
            self.num_slots = self.num_experts
            self.slot_start = _start
            self.slot_end = _end
            self.initial_local_expert_ids = list(
                range(self.slot_start, self.slot_end))
            self.initial_global_assignments = list(range(self.num_experts))
            self.allreduce = None

        # Override expert layout when DWDP is enabled.  This must run before
        # create_weights() (which is invoked later by ConfigurableMoE) so the
        # fused MoE backend allocates ``num_experts_per_worker`` slots per
        # rank — not ``num_experts // ep_size``.  fixup_moe_backends() at
        # setup_dwdp() time later promotes these to the full composite-VA
        # view (``ep_size = 1``, ``slot_start = 0``,
        # ``slot_end = num_experts``); the earlier override is still
        # required because ``mapping.moe_ep_size`` is now 1, so the
        # un-overridden default would be ``expert_size_per_partition =
        # num_experts`` (each rank allocating storage for every expert).
        self._init_dwdp_expert_layout()
        self._init_perfect_router()

    def _init_dwdp_expert_layout(self):
        """Override expert layout when DWDP is enabled.

        Plumbs ``num_experts_per_worker`` (storage size) and
        ``start_expert_id`` (storage start) from the active
        ``DwdpManager``.  This is a no-op when DWDP is not enabled
        (``get_global_dwdp_manager()`` returns ``None``).

        For the uniform partition case (``num_prefetch_experts ==
        num_experts_per_worker == num_experts // dwdp_size``) this is
        mathematically equivalent to the legacy ``ep_size = dwdp_size``
        layout.  It additionally enables:

        * Non-uniform partition (``dwdp_size`` does not divide
          ``num_experts``): user picks ``num_prefetch_experts < num_experts_per_worker``
          so that ``(dwdp_size - 1) * num_prefetch_experts + num_experts_per_worker
          == num_experts`` exactly.  Adjacent ranks' valid ranges
          overlap by ``num_experts_per_worker - num_prefetch_experts``
          experts; ``_validate_partition_config`` rejects
          configurations whose last-rank end exceeds ``num_experts``
          because the GB200 cuMemMap-with-fabric-handle ABI requires
          ``mnnvl_size == handle_phys_size`` (no partial mapping), so
          tail-padded storage cannot be partially mapped into a
          ``num_experts``-sized composite VA.
        * Redundancy (``num_prefetch_experts < num_experts_per_worker``
          with ``(dwdp_size - 1) * stride + size == num_experts``):
          consecutive ranks' ranges overlap; peer-side
          ``lookup_owner`` (Phase 2) picks the lowest-rank owner so
          reads of shared experts are deterministic.

        DWDP and MoE EPLB are mutually exclusive — DWDP swaps
        ``param.data`` to a composite-VA tensor at runtime, which the
        EPLB rebalancer would clobber.
        """
        from tensorrt_llm._torch.pyexecutor.dwdp import get_global_dwdp_manager

        dwdp_manager = get_global_dwdp_manager()
        if dwdp_manager is None:
            return
        assert self.layer_load_balancer is None, (
            "DWDP and EPLB (MoE load balancer) cannot be used together. "
            "Disable one of dwdp_config or moe_load_balancer.")

        self.num_slots = self.num_experts
        self.expert_size_per_partition = dwdp_manager.num_experts_per_worker
        dwdp_size = dwdp_manager.dwdp_size
        # Routing-side expert assignment: distribute ``num_experts`` round-robin
        # across DWDP ranks.  Independent of the storage layout — the gate uses
        # this to map expert ids to ranks, while DWDP composite VA handles the
        # actual weight access.
        self.initial_global_assignments = [
            (ep_rank * self.num_experts // dwdp_size + local_slot_id) %
            self.num_experts for ep_rank in range(dwdp_size)
            for local_slot_id in range(self.expert_size_per_partition)
        ]
        # Storage range: ``[start, start + size)``.  Phase 2's strict
        # validation guarantees ``slot_end <= num_experts``, so every
        # storage slot maps to a valid expert id.  ``len(initial_local_expert_ids)
        # == expert_size_per_partition`` is preserved, which is required
        # by the per-slot weight-scale copy in
        # ``quantization.load_quant_scales``.
        self.slot_start = dwdp_manager.start_expert_id
        self.slot_end = self.slot_start + self.expert_size_per_partition
        self.initial_local_expert_ids = list(
            range(self.slot_start, self.slot_end))

    def _get_perfect_router_dtype(self) -> torch.dtype:
        if self.routing_method.routing_method_type in (
                RoutingMethodType.DeepSeekV3, RoutingMethodType.MiniMax2):
            return torch.float32
        return self.dtype if self.dtype is not None else torch.float32

    def _init_perfect_router(self):
        self._enable_perfect_router = os.environ.get("ENABLE_PERFECT_ROUTER",
                                                     "0") == "1"
        if not self._enable_perfect_router:
            return

        precompute_common_perfect_router_logits(
            num_experts=self.num_experts,
            experts_per_token=self.routing_method.experts_per_token,
            moe_ep_size=self.ep_size,
            dtype=self._get_perfect_router_dtype(),
            routing_method=self.routing_method,
            ep_rank=self.ep_rank)

    def _maybe_get_perfect_router_logits(
            self,
            router_logits: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if router_logits is None or not self._enable_perfect_router:
            return router_logits

        num_tokens, num_experts = router_logits.shape
        return get_cached_perfect_router_logits(
            num_tokens=num_tokens,
            num_experts=num_experts,
            experts_per_token=self.routing_method.experts_per_token,
            moe_ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            device=router_logits.device,
            dtype=router_logits.dtype,
            routing_method=self.routing_method)

    def _init_load_balancer(
        self,
        model_config: ModelConfig,
        aux_stream_dict: Optional[Dict[AuxStreamType,
                                       torch.cuda.Stream]] = None,
    ):
        """Initialize load balancer related attributes."""
        from .moe_load_balancer import get_moe_load_balancer

        # Store aux_stream_dict for load balancer
        self.aux_stream_dict = aux_stream_dict

        # Initialize load balancer attributes
        self.layer_load_balancer = None
        self.repeat_idx = 0
        self.repeat_count = 1

        # Get global load balancer instance
        moe_load_balancer = get_moe_load_balancer()
        moe_load_balancer_config = model_config.moe_load_balancer

        # Calculate initial expert assignments
        if moe_load_balancer_config:
            init_expert_size_per_partition = moe_load_balancer_config.num_local_slots
            self.initial_global_assignments = [
                (ep_rank * self.num_experts // self.ep_size + local_slot_id) %
                self.num_experts for ep_rank in range(self.ep_size)
                for local_slot_id in range(init_expert_size_per_partition)
            ]
        else:
            # Sequential mapping: expert i → slot i; covers all experts regardless of divisibility
            self.initial_global_assignments = list(range(self.num_experts))

        # Setup load balancer if available
        if moe_load_balancer:
            assert self._supports_load_balancer()
            assert self.use_dp and self.parallel_size > 1, "Load Balancer should be only used with ADP and EP > 1"
            assert moe_load_balancer_config is not None
            # EPLB provides uniform slot partition: every rank holds exactly
            # num_slots // ep_size slots, regardless of (num_experts % ep_size).
            # All backend kernels see uniform local-slot counts, so the
            # non-divisible-EP opt-in is irrelevant here. We only need
            # num_slots to divide ep_size for the slot partition to be uniform.
            if moe_load_balancer_config.num_slots % self.ep_size != 0:
                raise ValueError(
                    f"{type(self).__name__}: with EPLB enabled, num_slots "
                    f"({moe_load_balancer_config.num_slots}) must be divisible "
                    f"by ep_size ({self.ep_size}) so each rank holds the same "
                    f"number of slots.")
            top_k = self.routing_method.experts_per_token
            self.expert_size_per_partition = moe_load_balancer_config.num_local_slots

            # Add this layer to the load balancer
            aux_stream = getattr(self, '_get_load_balancer_aux_stream',
                                 lambda: None)()
            self.layer_load_balancer = moe_load_balancer.add_layer(
                self.num_experts,
                top_k,
                self.expert_size_per_partition,
                aux_stream=aux_stream)

            self.repeat_count = self.layer_load_balancer.get_repeat_count()

            # Handle initial global assignments
            loaded_initial_global_assignments = (
                moe_load_balancer_config.get_layer_initial_global_assignments(
                    self.layer_idx))
            self.num_slots = moe_load_balancer_config.num_slots

            if loaded_initial_global_assignments is not None:
                assert isinstance(loaded_initial_global_assignments, list)
                assert len(loaded_initial_global_assignments) == self.num_slots
                assert self.num_slots >= self.num_experts
                assert set(loaded_initial_global_assignments) == set(
                    range(self.num_experts))
                self.initial_global_assignments = loaded_initial_global_assignments

            self.layer_load_balancer.set_initial_weight_assignments(
                self.initial_global_assignments)

            from tensorrt_llm.logger import logger
            logger.info(
                f"MoE load balancer enabled. num_experts = {self.num_experts}, "
                f"num_slots = {self.num_slots}, ep_size = {self.ep_size}")
            logger.info(
                f"initial_global_assignments (layer {self.layer_idx}) = {self.initial_global_assignments}"
            )

            # Slot boundaries for EPLB (uniform: all ranks hold same num_local_slots)
            self.slot_start = self.ep_rank * self.expert_size_per_partition
            self.slot_end = self.slot_start + self.expert_size_per_partition
            self.initial_local_expert_ids = self.initial_global_assignments[
                self.slot_start:self.slot_end]
            assert len(
                self.initial_local_expert_ids) == self.expert_size_per_partition
        else:
            # No EPLB: each rank gets a ceil/floor slice of num_experts.
            # Only backends that have validated their dispatch/combine paths
            # for the ceil/floor partition may opt in; others fail fast with
            # an actionable error pointing at EPLB as the simpler escape.
            if (self.num_experts % self.ep_size != 0
                    and not type(self)._supports_non_divisible_ep):
                raise ValueError(
                    f"{type(self).__name__} does not support non-divisible EP: "
                    f"num_experts ({self.num_experts}) must be divisible by "
                    f"ep_size ({self.ep_size}). Enable EPLB with num_slots "
                    f"divisible by ep_size, or override "
                    f"`_supports_non_divisible_ep = True` on the subclass "
                    f"after verifying the kernel/comm path handles ceil/floor "
                    f"partitioning.")
            # Fallback: ceil/floor distribution across ranks.
            # Ranks 0..remainder-1 each hold (base+1) experts; remaining ranks hold base.
            _size, _start, _end = _compute_ep_partition(self.num_experts,
                                                        self.ep_size,
                                                        self.ep_rank)
            self.expert_size_per_partition = _size
            self.num_slots = self.num_experts
            self.slot_start = _start
            self.slot_end = _end
            self.initial_local_expert_ids = self.initial_global_assignments[
                self.slot_start:self.slot_end]
            assert len(
                self.initial_local_expert_ids) == self.expert_size_per_partition

        # Setup AllReduce for dynamic routing if needed
        if self._using_dynamic_load_balancer():
            from tensorrt_llm.functional import AllReduceStrategy

            from ...distributed import AllReduce
            self.allreduce = AllReduce(mapping=model_config.mapping,
                                       strategy=AllReduceStrategy.NCCL)
        else:
            self.allreduce = None

    def _get_load_balancer_aux_stream(self) -> Optional[torch.cuda.Stream]:
        """Get auxiliary stream for load balancer from aux_stream_dict.

        Returns the MoeBalancer stream from aux_stream_dict if available.
        """
        if self.aux_stream_dict is not None:
            return self.aux_stream_dict.get(AuxStreamType.MoeBalancer)
        return None

    def _load_balancer_start_wait_gpu_stage(self, is_first_call: bool):
        """Start waiting for GPU stage in load balancer."""
        if self._using_dynamic_load_balancer() and is_first_call:
            self.layer_load_balancer.start_wait_gpu_stage()

    def _load_balancer_done_wait_gpu_stage(self, is_first_call: bool):
        """Mark GPU wait stage as done in load balancer."""
        if self._using_dynamic_load_balancer() and is_first_call:
            self.layer_load_balancer.done_wait_gpu_stage()

    def _load_balancer_update_statistic(self,
                                        token_selected_experts: torch.Tensor,
                                        is_first_call: bool,
                                        is_last_call: bool,
                                        ignore_allreduce: bool = False):
        """
        Update load balancer statistics.

        Args:
            token_selected_experts: The selected experts of all tokens, has shape of [tokenCount * topK]
            is_first_call: Whether this is the first call for the same weights
            is_last_call: Whether this is the last call for the same weights
            ignore_allreduce: Whether to ignore allreduce, if True, only update local statistics, need call _load_balancer_get_local_statistic_tensor to get the local statistic tensor and then do external allgather and then call _load_balancer_update_statistic_with_gathered_statistic to update the global statistics. NVLINKTwoSided supports this.
        """
        if self._using_dynamic_load_balancer():
            if ignore_allreduce:
                self.layer_load_balancer.update_local_statistic(
                    token_selected_experts,
                    is_first_stage=is_first_call,
                    is_last_stage=is_last_call)
            else:
                self.layer_load_balancer.update_statistic_with_local_ids(
                    token_selected_experts,
                    is_first_stage=is_first_call,
                    is_last_stage=is_last_call,
                    allreduce=self.allreduce)

    def _load_balancer_route(self, token_selected_experts: torch.Tensor,
                             use_dp: bool) -> torch.Tensor:
        """Route tokens using load balancer."""
        if self.layer_load_balancer:
            return self.layer_load_balancer.route(token_selected_experts,
                                                  use_dp)
        else:
            return token_selected_experts

    def _load_balancer_start_set_cpu_stage(self, is_last_call: bool):
        """Start CPU stage in load balancer."""
        if self._using_dynamic_load_balancer() and is_last_call:
            self.layer_load_balancer.start_set_cpu_stage()

    def _load_balancer_done_set_cpu_stage(self, is_last_call: bool):
        """Mark CPU stage as done in load balancer."""
        if self._using_dynamic_load_balancer() and is_last_call:
            self.layer_load_balancer.done_set_cpu_stage()

    def _load_balancer_get_local_statistic_tensor(self):
        """Get local statistic tensor from load balancer."""
        if self._using_dynamic_load_balancer():
            return self.layer_load_balancer.get_local_statistic_tensor()
        return None

    def _load_balancer_update_statistic_with_gathered_statistic(
            self, gathered_statistic):
        """Update load balancer with gathered statistics."""
        if self._using_dynamic_load_balancer():
            self.layer_load_balancer.update_statistic_with_gathered_statistic(
                gathered_statistic)

    def _register_layer(self, model_config: ModelConfig):
        self.register_to_config = False
        if model_config is not None and self.layer_idx_str is not None:
            if "moe_layers" not in model_config.extra_attrs:
                model_config.extra_attrs["moe_layers"] = {}
            suffix = 0
            # ``layer_idx`` is local to a model stack, while one-model
            # speculative decoding shares this registry across target and
            # draft modules. Preserve every module under a stable unique key.
            while self.layer_idx_str in model_config.extra_attrs["moe_layers"]:
                self.layer_idx_str = str(self.layer_idx) + f"_{suffix}"
                suffix += 1
            model_config.extra_attrs["moe_layers"][
                self.layer_idx_str] = weakref.ref(self)
            self.register_to_config = True

    # ``create_weights`` / ``load_weights`` are implemented in
    # ``MoEWeightOwnerMixin``, shared with ``MoEImplBase``.

    @abstractmethod
    def quantize_input(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Dict]:
        """
        Quantize input tensor - unified interface for all MoE backends

        NOTE: This is a temporary interface. In the future, this method should be moved
        to the MoEBackend interface as part of the backend abstraction layer.

        This method handles quantization of input tensors before MoE computation.
        All MoE backend implementations must override this method to implement their
        specific quantization logic.

        Args:
            x: Input tensor [num_tokens, hidden_size] or Fp4QuantizedTensor
            **kwargs: Backend-specific arguments (e.g., token_selected_experts, workspace, etc.)

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]] or Dict:
                (quantized_x, scaling_factors)
                where scaling_factors should be reshaped to 2D if applicable

        Examples:
            Simple backends (Cutlass, WideEP, TRTLLMGen):
                return x_quantized, x_sf  # x_sf is 2D or None
        """
        raise NotImplementedError

    @abstractmethod
    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Unified MoE computation interface.

        Every value the caller genuinely produces travels in ``ctx``; every
        fact the comm layer decided for this forward travels in
        ``ctx.comm_plan``. Backends read only the fields they need, so adding a
        backend never requires touching the scheduler.

        Args:
            ctx: Inputs for this forward. ``token_selected_experts`` holds
                expert slots rather than expert IDs when EPLB is enabled.
            workspace: Scratch the backend itself allocated, via
                ``get_workspaces``; the scheduler owns only its lifetime,
                because one allocation is reused across chunks and alternated
                between streams and so outlives a single call. Only backends
                declaring ``requires_run_moe_workspace`` receive one.
                ``MoEImplBase.run_moe`` deliberately omits this parameter: it
                describes the state after that lifetime moves inside the impl,
                which happens as each impl moves onto that base
                (TRTLLM-14958, TRTLLM-14960..14969). Keyword-only here so that
                removing it is a mechanical change to named call sites rather
                than a silent re-binding of a positional argument.

        Returns:
            torch.Tensor: MoE computation result [num_tokens, hidden_size]
        """
        raise NotImplementedError

    @abstractmethod
    def forward_impl(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        raise NotImplementedError

    # ``forward_fake`` is implemented in ``MoEExecutionContractMixin``.

    # Sub class is not allowed to override forward.
    # This is universal interface for all MoE backends
    @final
    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        do_finalize: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        router_logits = self._maybe_get_perfect_router_logits(router_logits)
        if self.register_to_config and is_torch_compiling():
            # Routed-expert MoE LoRA is fused into torch.ops.trtllm.fused_moe
            # via lora_params, but the trtllm::moe_custom_op graph path used here
            # cannot carry lora_params. Dropping it silently would apply no LoRA,
            # so reject instead. _moe_lora_enabled is only set by backends that
            # fuse MoE LoRA, so other backends are unaffected.
            if getattr(self, "_moe_lora_enabled", False):
                raise RuntimeError(
                    "Routed-expert MoE LoRA is not supported together with "
                    "`register_to_config` + `torch.compile` (the "
                    "`trtllm::moe_custom_op` graph path cannot carry LoRA "
                    "adapter pointers). Disable `register_to_config`/"
                    "torch.compile for this model, or remove the MoE modules "
                    "from `lora_config.lora_target_modules`.")
            hidden_states = x.fp4_tensor if isinstance(
                x, Fp4QuantizedTensor) else x
            x_sf = x.scaling_factor if isinstance(x,
                                                  Fp4QuantizedTensor) else None
            is_swizzled = x.is_sf_swizzled if isinstance(
                x, Fp4QuantizedTensor) else False

            res = moe_custom_op(
                self.layer_idx_str,
                hidden_states,
                x_sf,
                is_swizzled,
                router_logits,
                do_finalize,
                output_dtype,
                all_rank_num_tokens,
                use_dp_padding,
            )
            if do_finalize:
                return res[0]
            else:
                return res
        else:
            return self.forward_impl(
                x,
                router_logits,
                do_finalize=do_finalize,
                output_dtype=output_dtype,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=use_dp_padding,
                **kwargs,
            )

    @property
    def enable_alltoall(self):
        """ enable_alltoall (bool): whether to enable alltoall instead of allgather/reducescatter

        Layer-level, not part of the execution-unit contract: the answer follows
        from the communication strategy the layer owns, and ``ConfigurableMoE``
        overrides it from ``self.comm``.
        """
        return False

    def reducescatter_or_allreduce(
        self,
        inputs,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
    ):
        """
        Common helper for TP and EP in subclasses of the MoE module.
        """
        outputs = inputs
        if self.parallel_size > 1 and not self.enable_alltoall:
            if self.use_dp:
                outputs = reducescatter(
                    inputs,
                    self.mapping,
                    dim=0,
                    sizes=None if use_dp_padding else all_rank_num_tokens)
            elif self.reduce_results:
                outputs = self.all_reduce(inputs)
        return outputs

    def dummy_allreduce(self):
        assert self.enable_dummy_allreduce and self.all_reduce is not None, "Dummy allreduce is not enabled"
        """
        Debug function for eliminating imbalance during performance analysis.
        Creates a small dummy tensor and performs allreduce to synchronize processes
        and eliminate timing imbalances for more accurate profiling measurements.
        """
        dummy_tensor = torch.zeros(4, dtype=torch.float32, device="cuda")
        dummy_tensor = self.all_reduce(dummy_tensor)
        return dummy_tensor
