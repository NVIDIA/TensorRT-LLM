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
"""The root of the CUTLASS-family leaves, and the ``torch.ops.trtllm.fused_moe``
call every leaf but the SM120 Triton one issues.

A leaf declares its kernel-selection flags once, as a frozen
:class:`FusedMoeOpFlags`. The two ``quantize_input`` implementations at the
bottom are the ones more than one format binds.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields

import torch

from tensorrt_llm.math_utils import pad_up

from ....model_config import ModelConfig
from ....utils import ActivationType, AuxStreamType, Fp4QuantizedTensor
from ..activation import (
    DEFAULT_MOE_ACTIVATION,
    ActivationParamShape,
    MoEActivation,
    MoEActivationSupport,
)
from ..impl_base import MoEImplBase, apply_moe_impl_construction_state
from ..impl_contract import MoERunContext, require_comm_plan
from ..impl_environment import collect_moe_environment
from ..quantization import MoEWeightLoadingMode
from ..routing import BaseMoeRoutingMethod
from .eligibility import SmSupport


@dataclass(frozen=True)
class FusedMoeOpFlags:
    """Kernel-selection flags for one quantization format.

    ``weight_dtype`` is how the weight buffers are viewed; every other field is
    an argument of ``torch.ops.trtllm.fused_moe`` and also part of the key that
    picks its cached runner. Defaults describe the unquantized path, so a leaf
    only states what makes it different.
    """

    #: Reinterpret the weight buffers as this dtype before the call. ``None``
    #: keeps ``w3_w1_weight.dtype``; W4A8_AWQ and W4A16_MXFP4 name the dtype
    #: the op reads their packed weights as.
    weight_dtype: torch.dtype | None = None
    #: Route FC1/FC2 through the DeepSeek block-scale GEMM runner instead of the
    #: CUTLASS grouped GEMM. Only the SM90 FP8-block-scale leaf sets this.
    use_deepseek_fp8_block_scale: bool = False
    #: Group-wise weight scales: with ``quint4x2`` weights the INT4 group-wise
    #: runner (W4A8_AWQ), with ``uint8`` weights the WFP4A16 one (W4A16_MXFP4).
    use_w4_group_scaling: bool = False
    use_int8_woq_per_channel: bool = False
    #: With packed MXFP4 weights, read ``quant_scales`` as the MXFP4 x MXFP8
    #: layout instead of MXFP4 x FP8. The activations themselves are quantized
    #: by the leaf's ``quantize_input``.
    use_mxfp8_act_scaling: bool = False
    #: Select the MXFP8xMXFP8 block-scaled path inside the <e4m3, e4m3> runner
    #: template; per-tensor FP8 otherwise.
    use_mxfp8_weight_scaling: bool = False

    def op_kwargs(self) -> dict[str, bool]:
        """The flags as ``torch.ops.trtllm.fused_moe`` keyword arguments."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "weight_dtype"}


class CutlassFusedMoEBase(MoEImplBase):
    """Abstract root of the Cutlass-family leaves.

    Owns construction, the layer-format check in ``create_weights``, and the
    default ``quantize_input`` (pass-through) and ``run_moe`` (one
    ``torch.ops.trtllm.fused_moe`` call); the weight lifecycle comes from
    ``MoEWeightOwnerMixin``. Unregistered, because a descriptor is exactly one
    identity and this class stands for twelve.

    A leaf declares:

    * ``descriptor`` -- its one identity. ``scheduler_kind``, ``capabilities``
      and ``input_requirement`` are taken off it when the class is created.
    * ``sm_support`` / ``supported_dtypes`` -- what its kernel is built for.
    * ``FUSED_MOE_OP_FLAGS`` -- the kernel-selection flags it hands the op.
    * ``supports_gptoss_style`` / ``rejects_gptoss_expert_bias`` -- whether its
      weight method can load a gpt-oss / MiniMax SwiGLU shape.
    * ``requires_oss_cutlass_build`` / ``uses_dynamic_fc2_scale`` -- whether its
      format, or its dynamic FC2 scale, needs the open-source CUTLASS kernels.
    * ``can_implement`` and ``_get_quant_method``.
    """

    # No default: a wrong SM set or dtype set silently admits a layer whose
    # kernel does not exist.
    sm_support: SmSupport
    supported_dtypes: frozenset
    # Read by ``eligibility.check_gptoss_activation``.
    supports_gptoss_style: bool = False
    rejects_gptoss_expert_bias: bool = False
    # Read by ``eligibility.check_oss_cutlass_build``.
    requires_oss_cutlass_build: bool = False
    uses_dynamic_fc2_scale: bool = False
    #: Whether ``run_moe`` issues ``torch.ops.trtllm.fused_moe``; only such a
    #: leaf is bound by the open-source CUTLASS build (its gate and all-to-all).
    issues_fused_moe_op: bool = True

    FUSED_MOE_OP_FLAGS: FusedMoeOpFlags = FusedMoeOpFlags()
    #: Round ``hidden_size`` and ``intermediate_size_per_partition`` up to a
    #: multiple of this at construction, before the weights are allocated.
    pad_hidden_and_intermediate_to: int | None = None
    #: As above for ``intermediate_size_per_partition`` alone, and only at MoE TP
    #: size 1: a TP shard would split a block from its scale.
    pad_intermediate_on_single_tp_rank_to: int | None = None
    #: Whether the layer is a routed-expert LoRA target; only
    #: ``CutlassMoELoraMixin._init_moe_lora`` sets it.
    _moe_lora_enabled: bool = False

    # The CUTLASS epilogue has an adaptor per kind (``moe_kernels.cuh``) and
    # takes all three constants as ``float*`` indexed by expert.
    activation_support = MoEActivationSupport(
        kinds=frozenset(
            {
                ActivationType.Swiglu,
                ActivationType.SwigluBias,
                ActivationType.Geglu,
                ActivationType.SiTu,
                ActivationType.Relu2,
                ActivationType.Silu,
            }
        ),
        alpha_beta=ActivationParamShape.PER_EXPERT_TENSOR,
        limit=ActivationParamShape.PER_EXPERT_TENSOR,
        clamp_after_silu=True,
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
        bias: bool = False,
        apply_router_weight_on_input: bool = False,
        layer_idx: int | None = None,
        activation: MoEActivation = DEFAULT_MOE_ACTIVATION,
        init_load_balancer: bool = False,
    ) -> None:
        """Construct the backend.

        Args:
            routing_method (BaseMoeRoutingMethod): Token-to-expert assignment.
            num_experts (int): Number of experts in the MoE layer.
            hidden_size (int): Size of the hidden state.
            intermediate_size (int): Size of the intermediate state.
            dtype (torch.dtype | None): Data type for the weights.
            reduce_results (bool): Whether to reduce the results across devices.
            model_config (ModelConfig): Configuration object for the model.
            aux_stream_dict (dict[AuxStreamType, torch.cuda.Stream] | None):
                Auxiliary CUDA streams for overlapping.
            weight_loading_mode (MoEWeightLoadingMode): How checkpoint weights
                map onto this backend's parameters.
            bias (bool): Whether the experts carry a bias.
            apply_router_weight_on_input (bool): Pre-scale the input by the
                routing weight instead of scaling the output. Requires top-1
                routing, which ``_check_configs`` asserts.
            layer_idx (int | None): Index of the layer this backend serves.
            activation (MoEActivation): The layer's activation kind and its
                constants.
            init_load_balancer (bool): Must stay False: an execution unit is
                always a ``ConfigurableMoE`` backend, never a standalone layer.
        """
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
            activation=activation,
            layer_idx=layer_idx,
            init_load_balancer=init_load_balancer,
        )

        # After apply_moe_impl_construction_state(), which has already captured
        # ``unpadded_hidden_size`` for the op.
        alignment = self.pad_hidden_and_intermediate_to
        if alignment is not None:
            self.hidden_size = pad_up(self.hidden_size, alignment)
            self.intermediate_size_per_partition = pad_up(
                self.intermediate_size_per_partition, alignment
            )
        alignment = self.pad_intermediate_on_single_tp_rank_to
        if alignment is not None and self.tp_size == 1:
            self.intermediate_size_per_partition = pad_up(
                self.intermediate_size_per_partition, alignment
            )

        # moe_max_num_tokens is set in ModelConfig.__post_init__ if not specified
        # The default value is max_num_tokens * dp_size
        self.moe_max_num_tokens = model_config.moe_max_num_tokens
        default_moe_max_num_tokens = model_config.max_num_tokens * model_config.mapping.dp_size

        # The profiler converges on the same best tactic when the number of tokens is large enough.
        # To avoid long profiling time, the max number of tokens used in the profiling is capped to
        # around 16k tokens per expert, which is well into the compute bound domain.
        self.tune_max_num_tokens = min(
            self.moe_max_num_tokens,
            16384 * self.num_slots // routing_method.get_experts_per_token(),
            # A forward can never present more tokens than this.
            default_moe_max_num_tokens,
        )

        # If True, the router weight will be multiplied on the input rather than at the end of FC2
        self.apply_router_weight_on_input = apply_router_weight_on_input

        # Finalize fusion should be disabled if Lora is used.
        self.use_fused_finalize = (
            not model_config.moe_disable_finalize_fusion and model_config.lora_config is None
        )

        self._init_moe_lora(model_config)

        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def _init_moe_lora(self, model_config: ModelConfig) -> None:
        """No-op; ``CutlassMoELoraMixin`` overrides it."""

    def create_weights(self) -> None:
        """Allocate expert weights once the layer's final format is confirmed."""
        if self._weights_created:
            return
        self._check_quant_config_is_my_format()
        super().create_weights()

    def _check_configs(self) -> None:
        assert self._weights_created
        if self.apply_router_weight_on_input:
            assert self.routing_method.top_k == 1, "Current walkaround only supports top-1 routing"

    def _supports_load_balancer(self) -> bool:
        """Every Cutlass leaf supports the load balancer."""
        return True

    def supports_moe_output_in_alltoall_workspace(self) -> bool:
        return True

    @property
    def supports_alltoall(self) -> bool:
        # The internal kernel archive's fused_moe op rejects all-to-all.
        return not self.issues_fused_moe_op or collect_moe_environment().oss_cutlass_moe

    def validate_configurable_moe(self, moe: torch.nn.Module) -> None:
        if moe.enable_alltoall and not self.supports_alltoall:
            raise ValueError(
                f"{type(self).__name__} all-to-all requires a build with "
                "USING_OSS_CUTLASS_MOE_GEMM=ON"
            )

    def quantize_input(
        self,
        x: torch.Tensor | Fp4QuantizedTensor,
        post_quant_comm: bool = True,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Pass activations through; right for every format whose kernel
        quantizes them itself or keeps them in high precision.

        Args:
            x: The activations, or FP4 activations quantized before dispatch.
            post_quant_comm: Whether communication follows. A format that
                returns scale factors then lays them out as one 2-D row per
                token; otherwise in the layout its kernel reads.
            **kwargs: Unused; accepted for the ``MoE.quantize_input`` interface.

        Returns:
            The activations to run and their scale factors, or None.
        """
        del post_quant_comm, kwargs
        return x, None

    def run_moe(self, ctx: MoERunContext, *, workspace: dict | None = None) -> torch.Tensor:
        del workspace  # Cutlass allocates its own intermediates.
        return self._run_fused_moe_op(ctx)

    def _run_fused_moe_op(
        self,
        ctx: MoERunContext,
        *,
        dequantized_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
        dynamic_fc2_scale: torch.Tensor | None = None,
        lora_kwargs: Mapping[str, object] | None = None,
    ) -> torch.Tensor:
        """Issue ``torch.ops.trtllm.fused_moe`` with ``FUSED_MOE_OP_FLAGS``.

        The op runs the whole MoE: permute, FC1, activation, FC2 and finalize.
        A leaf whose ``run_moe`` needs operands the ``run_moe`` contract does
        not carry passes them here.

        Args:
            ctx: The run context.
            dequantized_weights: ``(w3_w1, w2)`` already in the activation dtype.
                They replace the module's quantized buffers, and the op then runs
                unquantized: default flags, no quant scales and no input
                scaling factor.
            dynamic_fc2_scale: Appended to ``quant_scales``; the kernel then
                applies it as a per-forward FC2 scale (dynamic NVFP4).
            lora_kwargs: Routed-expert LoRA operands. Only a leaf mixing in
                ``CutlassMoELoraMixin`` can build them.

        Returns:
            The finalized MoE output.
        """
        plan = require_comm_plan(self, ctx)
        enable_alltoall = plan.enable_alltoall
        tuner_num_tokens, tuner_top_k = self._tuner_shapes(ctx, enable_alltoall)
        moe_output = plan.moe_output

        if dequantized_weights is None:
            flags = self.FUSED_MOE_OP_FLAGS
            weight_dtype = flags.weight_dtype or self.w3_w1_weight.dtype
            w3_w1 = self.w3_w1_weight.view(weight_dtype)
            w2 = self.w2_weight.view(weight_dtype)
            quant_scales = list(self.quant_scales)
            if dynamic_fc2_scale is not None:
                quant_scales.append(dynamic_fc2_scale)
            input_sf, swizzled_input_sf = ctx.x_sf, plan.input_sf_swizzled
        else:
            flags = FusedMoeOpFlags()
            w3_w1, w2 = dequantized_weights
            quant_scales, input_sf, swizzled_input_sf = [], None, False

        result = torch.ops.trtllm.fused_moe(
            ctx.x,
            ctx.token_selected_experts,
            ctx.token_final_scales,
            w3_w1,
            self.w3_w1_bias,
            w2,
            self.w2_bias,
            ctx.output_dtype,
            quant_scales=quant_scales,
            input_sf=input_sf,
            swizzled_input_sf=swizzled_input_sf,
            # ``swiglu_*`` are the moe_op schema's names for these registers
            # (``ActivationParams`` in moe_kernels.h); SiTU fills the same three
            # with tanh soft-caps.
            swiglu_alpha=self.act_alpha,
            swiglu_beta=self.act_beta,
            swiglu_limit=self.act_clamp,
            swiglu_clamp_after_silu=self.act_clamp_after_silu,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=self.cluster_size,
            cluster_rank=self.cluster_rank,
            enable_alltoall=enable_alltoall,
            **flags.op_kwargs(),
            min_latency_mode=False,
            use_fused_finalize=self.use_fused_finalize,
            tune_max_num_tokens=self.tune_max_num_tokens,
            tuner_num_tokens=tuner_num_tokens,
            tuner_top_k=tuner_top_k,
            activation_type=self.activation_type,
            unpadded_hidden_size=self.unpadded_hidden_size,
            out_tensor=moe_output,
            use_dynamic_fc2_scale=dynamic_fc2_scale is not None,
            **(lora_kwargs or {}),
        )
        # With moe_output supplied the result is written in place and the op
        # returns an empty list to avoid an aliasing-constraint violation;
        # otherwise the single output tensor comes back in a list.
        return moe_output if moe_output is not None else result[0]

    def _local_slot_ids(self, slot_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map global slot ids onto this rank's expert slots.

        Dispatch keeps the global numbering with or without alltoall and marks
        padding with an id no rank owns (-1, or ``num_slots`` under DeepEP low
        latency).

        Args:
            slot_ids: ``token_selected_experts``, int32 ``[num_tokens, top_k]``.

        Returns:
            ``(local_ids, offsets)``. ``offsets`` is ``slot_ids - slot_start``;
            ``local_ids`` clamps it into ``[0, expert_size_per_partition)`` so
            indexing local weights stays in bounds. This rank owns exactly the
            ids where the two are equal.
        """
        offsets = slot_ids - self.slot_start
        return offsets.clamp(0, self.expert_size_per_partition - 1), offsets

    def _tuner_shapes(
        self,
        ctx: MoERunContext,
        enable_alltoall: bool | None,
    ) -> tuple[int | None, int | None]:
        """Token/top-k shapes the profiling tuner should key on.

        Only meaningful under alltoall: the tuner must see pre-alltoall token
        counts so tactics cached during the no-alltoall warmup still apply at
        runtime. Without alltoall the kernel derives both from ``x`` itself.
        """
        if not enable_alltoall:
            return None, None
        if ctx.all_rank_num_tokens is not None:
            tuner_num_tokens = sum(ctx.all_rank_num_tokens)
        else:
            tuner_num_tokens = ctx.x.shape[0] * self.mapping.tp_size
        return tuner_num_tokens, self.routing_method.top_k


def quantize_input_static_e4m3(
    self: CutlassFusedMoEBase,
    x: torch.Tensor | Fp4QuantizedTensor,
    post_quant_comm: bool = True,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Per-tensor FP8: scale by the static FC1 input scale and cast to e4m3.

    A :meth:`CutlassFusedMoEBase.quantize_input` implementation; a leaf binds
    it as ``quantize_input = quantize_input_static_e4m3``.
    """
    del post_quant_comm, kwargs
    x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, self.fc31_input_dequant)
    return x, None


def quantize_input_mxfp8(
    self: CutlassFusedMoEBase,
    x: torch.Tensor | Fp4QuantizedTensor,
    post_quant_comm: bool = True,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dynamic MXFP8 activation quantization, for every MXFP8-activation format.

    A :meth:`CutlassFusedMoEBase.quantize_input` implementation; a leaf binds
    it as ``quantize_input = quantize_input_mxfp8``.
    """
    del kwargs
    alignment = self.quant_method.weight_alignment
    if not post_quant_comm:
        return torch.ops.trtllm.mxfp8_quantize(x, True, alignment=alignment)
    x, x_sf = torch.ops.trtllm.mxfp8_quantize(x, False, alignment=alignment)
    # Communication needs the scales as 2-D rows; x.shape[0] is already padded.
    if x_sf is not None:
        x_sf = x_sf.view((x.shape[0], -1))
    return x, x_sf
