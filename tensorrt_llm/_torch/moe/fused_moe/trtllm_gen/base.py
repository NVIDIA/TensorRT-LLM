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
"""The abstract root every TRTLLM-Gen leaf shares.

Carries only what every leaf needs: construction, the weight-creation skeleton,
the two routing predicates the framework reads off a module, and the fake-output
shapes. Anything varying by quantization format lives in
:mod:`.fp4_block_scale` or :mod:`.fp8_block_scale`; anything varying by provider
is read off the leaf's own identity. Both axes reach this file only as
attributes a leaf declares and hooks it overrides, never as a branch on
``quant_config`` or on the provider string. Abstract and unregistered: it
implements none of ``MoEImplBase``'s four abstract methods and publishes no
descriptor.
"""

from typing import ClassVar

import torch
from torch import nn

from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import (
    fp4_block_scale_fake_output_without_finalize,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.utils import ActivationType, AuxStreamType, Fp4QuantizedTensor
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from ..activation import (
    DEFAULT_MOE_ACTIVATION,
    ActivationParamShape,
    MoEActivation,
    MoEActivationSupport,
)
from ..impl_base import MoEImplBase, apply_moe_impl_construction_state
from ..impl_contract import canonical_quant, normalize_quant
from ..impl_identity import MOE_IMPL_REGISTRY
from ..interface import FORCE_SEPARATED_ROUTING, MoEWeightLoadingMode
from ..moe_op_backend import MoEOpBackend, get_op_backend
from ..routing import BaseMoeRoutingMethod, DeepSeekV3MoeRoutingMethod, MiniMaxM2MoeRoutingMethod
from .eligibility import identity_quant_of
from .identity import (
    PROVIDER_FLASHINFER,
    TECHNIQUE_TRTLLM_GEN,
    TRTLLM_GEN_CAPABILITIES,
    TRTLLM_GEN_INPUT_REQUIREMENT,
)


class TrtllmGenFusedMoEBase(MoEImplBase):
    """Abstract root of the ``*.trtllm_gen.fused_moe.*`` implementations.

    The split below this class is by ``quant`` x ``provider``, and the two axes
    own different things:

    - ``quant`` decides which kernel ``run_moe`` calls, how weights and inputs
      are prepared, and which activation ABI applies. Held by a per-quant class
      in the module named for that kernel ABI -- :mod:`.fp4_block_scale` or
      :mod:`.fp8_block_scale` -- one per format that two providers share. A
      format only one provider serves has no such class.
    - ``provider`` decides which op backend executes, which configurations are
      eligible, and whether the kernel writes into the caller's workspace. Read
      off the leaf's own descriptor by ``__init_subclass__`` below, so a leaf
      is just an identity and a ``can_implement``.

    One kernel call per forward: routing (unless a leaf routes
    outside it), scatter, gemm1, the fused activation, gemm2, and the combine
    reduction are a single cubin. Hence ``run_moe`` returns a finalized tensor
    by default; a leaf whose runner supports it can be asked to stop short
    and hand back per-expert outputs through ``_unfinalized``.

    No AllReduce is issued from here. With ``reduce_results=False`` the model
    definition owns it; ``ConfigurableMoE`` supplies it through the
    communication strategy otherwise.

    Slots, not experts, size the weight buffers: EPLB may give a hot expert
    several replicas, so ``num_slots >= num_experts`` and the kernel is told
    both. ``expert_size_per_partition`` and ``slot_start`` are this rank's
    window into the slot array.
    """

    # ---- identity-derived declarations, set by each registered leaf ------
    # ``ClassVar``, not instance state: each is answered by the class, several
    # of them at resolution time when there is no instance to ask.
    #: ``MoEImplId.provider``, and the ``moe_op_backend`` registry key
    #: ``__init__`` builds the op backend from. Derived in
    #: ``__init_subclass__``, not declared per leaf.
    provider: ClassVar[str]
    #: ``provider == PROVIDER_FLASHINFER``, read from outside as a plain
    #: attribute by ``ConfigurableMoE`` and the backend tests. Derived with
    #: ``provider``.
    use_flashinfer: ClassVar[bool]
    #: Whether this format has a fused cubin for the gpt-oss SwiGLU package
    #: (expert bias plus alpha/beta). Read off the *class* at resolution time,
    #: where there is no instance to ask.
    supports_gptoss_style: ClassVar[bool] = False
    #: Whether this leaf can run SiTu. Per-leaf and not per-format: the format
    #: needs a fused SiTu FC1 cubin *and* the provider has to call it, so a
    #: FlashInfer sibling of a supported format may still be False.
    supports_situ: ClassVar[bool] = False
    #: Whether this format's cubins always read an FC bias, so a model without
    #: one still needs a zeroed buffer. Declarative rather than an override
    #: because its four setters sit on two inheritance levels.
    needs_zero_expert_bias: ClassVar[bool] = False

    # Not ``ClassVar``: ``MoEExecutionContractMixin`` declares both as plain
    # attributes, and a subclass cannot narrow an instance variable to a class
    # variable. Their type comes from that declaration.
    capabilities = TRTLLM_GEN_CAPABILITIES
    input_requirement = TRTLLM_GEN_INPUT_REQUIREMENT

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Take the provider off the descriptor a leaf already publishes.

        A leaf stating its provider a second time can state it inconsistently,
        and the two statements feed different consumers: ``provider`` keys the
        op backend, the descriptor's copy keys the registry. Deriving leaves
        one source.

        Plain class attributes and not properties, because both are read off
        the *class* -- at resolution, and by ``fused_shared_expert_count``
        through ``type(self)`` -- where a property yields the descriptor
        object rather than its value.

        Intermediate bases publish no descriptor of their own, having no one
        identity to name; they inherit whatever their parent resolved to and
        are overwritten by each registered leaf below them.
        """
        super().__init_subclass__(**kwargs)
        if "descriptor" not in cls.__dict__:
            return
        cls.provider = cls.descriptor.identity.provider
        cls.use_flashinfer = cls.provider == PROVIDER_FLASHINFER

    @staticmethod
    def situ_supported_quant_algos() -> frozenset[QuantAlgo]:
        """The quant algos some leaf of this family has a fused SiTu cubin for.

        Public because a model handing SiTu to this family has to decide,
        before construction, whether the handoff can work at all. Read off the
        leaves' own ``supports_situ`` rather than restated, since restating it
        is what broke once: ``modeling_kimi_linear`` carried a copy, the NVFP4
        cubins landed, and the copy stayed at MXFP4-only.
        """
        situ_quants = {
            identity.quant
            for identity in MOE_IMPL_REGISTRY.identities()
            if identity.technique == TECHNIQUE_TRTLLM_GEN
            and getattr(MOE_IMPL_REGISTRY.lookup(identity), "supports_situ", False)
        }
        # Matched on the identity spelling rather than through
        # ``canonical_quant``, which folds the calibration recipes: NVFP4_AWQ
        # runs on the NVFP4 leaf, but naming it here would widen the set a
        # caller reports back to the user.
        return frozenset(algo for algo in QuantAlgo if algo.value.lower() in situ_quants)

    # The fused-activation cubins index alpha/beta/clamp by expert. FP8 block
    # scales run the clamp in a separate kernel taking a scalar, so that format
    # overrides ``resolve_activation_support`` instead of appearing here.
    activation_support = MoEActivationSupport(
        kinds=frozenset(
            {
                ActivationType.Swiglu,
                ActivationType.SwigluBias,
                ActivationType.Relu2,
                ActivationType.Silu,
                ActivationType.SiTu,
            }
        ),
        alpha_beta=ActivationParamShape.PER_EXPERT_TENSOR,
        limit=ActivationParamShape.PER_EXPERT_TENSOR,
    )

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype | None = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream] | None = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA,
        layer_idx: int | None = None,
        bias: bool = False,
        init_load_balancer: bool = False,
        activation: MoEActivation = DEFAULT_MOE_ACTIVATION,
    ):
        super().__init__(eplb=None)
        apply_moe_impl_construction_state(
            self,
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

        self._check_before_weights()

        # Cached for autotune profile sizing (forward path passes
        # tune_max_num_tokens to the MoE op).
        self.max_num_tokens = model_config.max_num_tokens

        self.op_backend: MoEOpBackend = get_op_backend(self.provider)

        self._weights_created = False
        # Read from outside on every leaf, so initialized here even though only
        # one format can raise it above zero.
        self.num_fused_shared_expert = 0
        self._configure_shared_expert_fusion(model_config)

        # create_weights must see the final fused-expert count so the fused shared
        # slots are allocated when fusion is enabled.
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    @property
    def is_situ_activation(self) -> bool:
        return self.activation.kind is ActivationType.SiTu

    def _check_before_weights(self) -> None:
        """Preconditions the SiTu cubins carry.

        Separate from ``_check_configs`` because of when it runs: this fires
        from ``__init__`` even when ``skip_create_weights_in_init`` defers
        weight creation, and it covers every leaf, including those that declare
        no ``_check_configs`` at all. Resolution rejects the same
        configurations earlier from ``supports_situ``; this catches a leaf
        constructed directly.
        """
        if not self.is_situ_activation:
            return
        if self.dtype != torch.bfloat16:
            raise ValueError(f"TRTLLM-Gen SiTu requires bfloat16 activations, got {self.dtype}.")
        if not is_sm_100f():
            raise ValueError(
                f"TRTLLM-Gen SiTu requires the SM100 family, got SM{get_sm_version()}."
            )
        if not self.supports_situ:
            quant_algo = None if self.quant_config is None else self.quant_config.quant_algo
            raise ValueError(
                f"{type(self).__name__} reaches no fused SiTu cubin "
                f"(quant_algo={quant_algo}, provider={self.provider})."
            )
        if self.tp_size > 1:
            # Intra-expert MoE TP: w1/w3 column-shard and w2 row-shard along
            # the intermediate dim (the stock MXFP4/NVFP4 quant-method loaders
            # slice the packed bytes and scales per rank). Require every rank
            # to own whole scale groups; the padded loaders then pad each
            # logical shard independently to the physical kernel alignment.
            alignment = self._situ_tp_weight_alignment()
            if (
                self.intermediate_size % self.tp_size != 0
                or self.intermediate_size_per_partition % alignment != 0
            ):
                raise ValueError(
                    "TRTLLM-Gen SiTu MoE TP requires intermediate_size "
                    f"({self.intermediate_size}) divisible by moe_tp_size "
                    f"({self.tp_size}) with the per-rank shard a multiple of "
                    f"{alignment}, got {self.intermediate_size_per_partition}."
                )
        if self.bias:
            raise ValueError(
                "TRTLLM-Gen SiTu does not support expert bias; the cubin adds "
                "no FC1 bias before the soft-caps."
            )

    def _situ_tp_weight_alignment(self) -> int:
        """The weight alignment a SiTu MoE-TP shard has to be a multiple of.

        The one format-dependent step of the check above, so the format base
        answers it. Only reachable with ``supports_situ`` set.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares supports_situ but no SiTu MoE-TP alignment."
        )

    @classmethod
    def fused_shared_expert_count(cls, model_config: ModelConfig) -> int:
        """How many shared experts this format folds into the routed GEMM.

        Zero here; only one format can append shared-expert slots, and that
        class knows the environment flag, parallel restrictions, and weight
        layout involved. A classmethod so a caller holding the resolved leaf
        can ask without building it.
        """
        del model_config  # a format that cannot fuse has nothing to read
        return 0

    def _configure_shared_expert_fusion(self, model_config: ModelConfig) -> None:
        """Adopt this format's fused-shared-expert count for this instance."""
        self.num_fused_shared_expert = type(self).fused_shared_expert_count(model_config)
        if self.num_fused_shared_expert > 0:
            logger.info_once(
                f"Shared-expert fusion enabled: folding "
                f"{self.num_fused_shared_expert} shared expert(s) into the "
                f"routed-expert grouped GEMM.",
                key="trtllm_gen_shared_expert_fusion",
            )

    def _requires_separated_routing(self) -> bool:
        """Whether this leaf's kernel takes top-k from the host, not the logits.

        False by default: these cubins route internally from the logits. A
        leaf whose kernel has no internal routing overrides this.
        """
        return False

    def _supports_load_balancer(self) -> bool:
        """Whether separated routing (top-k outside the kernel) is used.

        ConfigurableMoE uses this flag to decide whether routing is separated
        (top-k ids/scales computed outside backend) or fused inside the kernel.
        """
        if self._requires_separated_routing():
            return True
        return self.use_dp and self.parallel_size > 1

    def _routes_outside_the_kernel(self) -> bool:
        """Whether top-k is precomputed, so the kernel must not route again.

        Three independent triggers: a kernel or parallel layout that forces it
        (both folded into ``_supports_load_balancer``), a routing algorithm no
        C++ kernel implements, and the host-routing override.
        """
        return (
            self._supports_load_balancer()
            or self.routing_method.requires_separated_routing
            or FORCE_SEPARATED_ROUTING
        )

    def create_weights(self) -> None:
        """The allocation skeleton; format variation is a hook, not a test here.

        What differs is either an override of ``_create_quant_method_weights``
        or the declarative ``needs_zero_expert_bias``. ``_check_configs`` is
        deliberately not implemented at this level: each format or leaf asserts
        only its own.
        """
        if self._weights_created:
            return

        self._check_quant_config_is_my_format()
        self.quant_method = self._get_quant_method()
        self._create_quant_method_weights()

        self._weights_created = True
        self._check_configs()

        if self.needs_zero_expert_bias and not self.bias:
            self._allocate_zero_expert_bias()

    def _check_quant_config_is_my_format(self) -> None:
        """Fail loudly if this layer ended up on a leaf of the wrong format.

        Resolution keys on the model-level ``quant_algo``, but
        ``apply_layerwise_quant_config`` and
        ``apply_quant_config_exclude_modules`` can give a layer its own
        ``quant_config`` afterwards, so a layer can be admitted by a leaf whose
        format it no longer has. Nothing else catches this, and the mismatch
        surfaces here rather than at the pass that moved the layer, so the
        message has to name both. Unchecked it means weights the checkpoint
        cannot fill, or silently wrong numerics.
        """
        expected = identity_quant_of(type(self))
        actual = normalize_quant(
            canonical_quant(None if self.quant_config is None else self.quant_config.quant_algo)
        )
        if actual != expected:
            raise ValueError(
                f"{type(self).__name__} implements quant={expected}, but layer "
                f"{self.layer_idx}'s quant_config resolves to {actual}. The "
                f"implementation was picked from the model-level quant_algo; "
                f"layerwise quantization or a module exclusion moved this layer "
                f"afterwards, and the layer must be re-resolved for the format "
                f"it actually has."
            )

    def _create_quant_method_weights(self) -> None:
        """Hand the module to its quantization method, and settle the layout.

        A format may override to pass extra arguments its method takes, or to
        install backend-owned parameters that ``_check_configs`` validates;
        the latter must call ``super()`` first.
        """
        self.quant_method.create_weights(self)

    def _allocate_zero_expert_bias(self) -> None:
        """Give the FC epilogues a bias buffer the checkpoint does not carry.

        The cubins for these formats always read a bias, so a model without one
        still needs the registers filled.
        """
        self.w3_w1_bias = nn.Parameter(
            torch.zeros(
                (self.w3_w1_weight.shape[0], self.w3_w1_weight.shape[1]), dtype=torch.float32
            ),
            requires_grad=False,
        )
        self.register_parameter("w3_w1_bias", self.w3_w1_bias)
        self.w2_bias = nn.Parameter(
            torch.zeros((self.w2_weight.shape[0], self.w2_weight.shape[1]), dtype=torch.float32),
            requires_grad=False,
        )
        self.register_parameter("w2_bias", self.w2_bias)

    def supports_moe_output_in_alltoall_workspace(self) -> bool:
        """Whether ``run_moe`` fills a caller-supplied output buffer.

        Only the native provider's runners take an output tensor. A leaf that
        diverges from its provider overrides this.
        """
        return not self.use_flashinfer

    def _unfinalized(self, outputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Hand back per-expert outputs for the caller to combine."""
        assert not self.reduce_results, "reduce_results must be False when do_finalize is False"
        return outputs

    def forward_fake(
        self,
        x: torch.Tensor | Fp4QuantizedTensor,
        router_logits: torch.Tensor,
        *,
        do_finalize: bool = True,
        output_dtype: torch.dtype | None = None,
        all_rank_num_tokens: list[int] | None = None,
        use_dp_padding: bool | None = None,
        **kwargs,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Meta-tensor shapes for both the finalized and un-finalized ABIs.

        The finalized output is bf16 for every leaf because the shared
        eligibility gate admits only bfloat16 activations. The un-finalized
        layout is reached from more than one parent, so it is answered here
        rather than on any one of them.
        """
        if do_finalize:
            return super().forward_fake(
                x,
                router_logits,
                do_finalize=do_finalize,
                output_dtype=torch.bfloat16,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=use_dp_padding,
                **kwargs,
            )

        is_deepseek_v3_routing = isinstance(self.routing_method, DeepSeekV3MoeRoutingMethod)
        is_minimax_routing = isinstance(self.routing_method, MiniMaxM2MoeRoutingMethod)
        top_k = (
            self.routing_method.routing_impl.top_k
            if is_deepseek_v3_routing
            else self.routing_method.top_k
        )
        routing_bias = (
            self.routing_method.e_score_correction_bias
            if (is_deepseek_v3_routing or is_minimax_routing)
            else None
        )
        return fp4_block_scale_fake_output_without_finalize(
            x,
            self.num_experts,
            top_k,
            routing_bias,
        )
