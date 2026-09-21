# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""The abstract root both Marlin leaves share, and the kernel ABI under it.

Marlin has one kernel ABI for both formats it publishes -- the kernel is W4A16
and reads the weights the same way whichever algorithm the checkpoint declared
-- so there is no per-ABI layer between base and leaves as there is in
:mod:`..trtllm_gen`. Everything executable lives here: the two leaves add
nothing but a descriptor and a ``can_implement`` that defers straight to
:func:`.eligibility.check_marlin_leaf`, so they are one implementation under
two names, not two implementations.

They are two names because a leaf answers for exactly one checkpoint label and
both labels have to reach Marlin -- a single ``nvfp4`` leaf would turn an
NVFP4-declared checkpoint away with ``QUANT_UNSUPPORTED`` and let resolution
fall through to Cutlass. The labels differ only in whether the activations are
quantized, which on SM89-SM99 is not a choice: there is no FP4 MMA, so W4A16 is
the only plan either label can run.

Publishes no descriptor and leaves ``can_implement`` undefined, so no request
can reach it.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.utils import (
    ActivationType,
    AuxStreamType,
    Fp4QuantizedTensor,
    is_gated_activation,
    relu2,
)

from ..activation import DEFAULT_MOE_ACTIVATION, MoEActivation, MoEActivationSupport
from ..impl_base import MoEImplBase, apply_moe_impl_construction_state
from ..impl_contract import MoERunContext, identity_quant_of
from ..quantization import MoEWeightLoadingMode, NVFP4MarlinFusedMoEMethod
from ..routing import BaseMoeRoutingMethod
from .identity import MARLIN_CAPABILITIES, MARLIN_INPUT_REQUIREMENT

# Block size for moe_align_block_size — must match TILE_M in the kernel
_MOE_BLOCK_SIZE = 16
# Hidden-dimension tile for _sum_topk_kernel. 256 BF16 elements × 4 warps
# (128 threads) = 2 elements/thread per load — fills one 128B cache line and
# keeps all warps active on Ada/Hopper without register pressure.
_SUM_TOPK_BLOCK_H = 256


@triton.jit
def _sum_topk_kernel(
    expert_outputs,
    output,
    num_tokens,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    block_h: tl.constexpr,
):
    """Sum contiguous ``[token, top_k, hidden]`` Marlin expert outputs."""
    token_idx = tl.program_id(0)
    hidden_offsets = tl.program_id(1) * block_h + tl.arange(0, block_h)
    mask = (token_idx < num_tokens) & (hidden_offsets < hidden_size)
    accumulator = tl.zeros((block_h,), dtype=tl.float32)
    for top_k_idx in tl.static_range(top_k):
        offsets = (token_idx * top_k + top_k_idx) * hidden_size + hidden_offsets
        accumulator += tl.load(expert_outputs + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output + token_idx * hidden_size + hidden_offsets, accumulator, mask=mask)


def sum_topk_expert_outputs(
    expert_outputs: torch.Tensor,
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Reduce contiguous Marlin outputs from ``[num_tokens * top_k, hidden_size]``."""
    output = torch.empty(
        (num_tokens, hidden_size), dtype=output_dtype, device=expert_outputs.device
    )
    if num_tokens == 0:
        return output
    grid = (num_tokens, triton.cdiv(hidden_size, _SUM_TOPK_BLOCK_H))
    _sum_topk_kernel[grid](
        expert_outputs,
        output,
        num_tokens,
        hidden_size=hidden_size,
        top_k=top_k,
        block_h=_SUM_TOPK_BLOCK_H,
        num_warps=4,
    )
    return output


def _has_fused_moe_kernel() -> bool:
    return hasattr(torch.ops.trtllm, "marlin_nvfp4_moe_gemm")


class MarlinFusedMoEBase(MoEImplBase):
    """Abstract root of the ``marlin.cuda.fused_moe.*`` implementations.

    Uses ``marlin_nvfp4_moe_gemm`` with BF16 activations to process all experts
    in a single kernel launch via sorted token dispatch. W4A16: BF16
    activations plus FP4 weights, dequantized FP4->BF16 in registers, using
    BF16 m16n8k16 MMA, so there is no activation quantization overhead.
    In-kernel topk_weights multiplication eliminates a separate scatter-weight
    step. CUDA-graph compatible. Requires the fused kernel to be built (no
    fallback path).
    """

    # Not ``ClassVar``: ``MoEExecutionContractMixin`` declares both as plain
    # attributes, and a subclass cannot narrow an instance variable to a class one.
    capabilities = MARLIN_CAPABILITIES
    input_requirement = MARLIN_INPUT_REQUIREMENT

    # The Marlin epilogue takes no activation constants, and
    # ``_apply_activation`` only distinguishes these three -- any other gated
    # kind would silently run SiLU, any other non-gated kind ReLU.
    activation_support = MoEActivationSupport(
        kinds=frozenset({ActivationType.Swiglu, ActivationType.Geglu, ActivationType.Relu2})
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
            bias (bool): Always False here. Accepted because ``create_moe``
                passes it on the branch this family shares with
                CutlassFusedMoE; the factory rejects a True against
                ``capabilities.supports_expert_bias``.
            apply_router_weight_on_input (bool): Pre-scale the input by the
                routing weight instead of scaling the output. Requires top-1
                routing, which ``_check_configs`` asserts.
            layer_idx (int | None): Index of the layer this backend serves.
            activation (MoEActivation): The layer's activation kind and its
                constants.
            init_load_balancer (bool): Register with the EPLB load balancer at
                construction.
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
            layer_idx=layer_idx,
            activation=activation,
            init_load_balancer=init_load_balancer,
        )
        self.apply_router_weight_on_input = apply_router_weight_on_input

        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def _check_configs(self) -> None:
        assert self._weights_created
        if self.apply_router_weight_on_input:
            assert self.routing_method.top_k == 1, "Current walkaround only supports top-1 routing"

    def quantize_input(
        self, x: torch.Tensor | Fp4QuantizedTensor, post_quant_comm: bool = True, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del post_quant_comm, kwargs  # W4A16: the activations reach the kernel unchanged.
        return x, None

    def _get_quant_method(self) -> NVFP4MarlinFusedMoEMethod:
        """The Marlin NVFP4 weight layout, for either format this family serves.

        Not keyed off ``self.moe_backend``: a layer reached by a pinned
        ``impl_id`` never has to set that literal to MARLIN, so asserting on it
        would refuse a correctly-resolved layer.
        """
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_nvfp4():
            return NVFP4MarlinFusedMoEMethod()
        raise ValueError(
            f"{type(self).__name__} implements quant={identity_quant_of(type(self))}, "
            f"which needs an NVFP4 weight layout, got {self.quant_config}"
        )

    def _apply_activation(self, gemm1_out: torch.Tensor) -> torch.Tensor:
        """Apply the activation, returning [tokens, intermediate_size].

        ``gemm1_out`` is [tokens, 2 * intermediate_size] for a gated kind and
        [tokens, intermediate_size] otherwise.
        """
        inter_size = self.intermediate_size_per_partition
        if is_gated_activation(self.activation_type):
            gate = gemm1_out[:, :inter_size]
            up = gemm1_out[:, inter_size : 2 * inter_size]
            if self.activation_type == ActivationType.Geglu:
                return F.gelu(gate) * up
            else:
                return F.silu(gate) * up  # SwiGLU
        else:
            if self.activation_type == ActivationType.Relu2:
                return relu2(gemm1_out)
            else:
                return F.relu(gemm1_out)

    def _ensure_workspace(self, device: torch.device) -> torch.Tensor:
        if not hasattr(self, "_marlin_workspace") or self._marlin_workspace is None:
            props = torch.cuda.get_device_properties(device)
            sms = props.multi_processor_count
            max_blocks_per_sm = 4
            self._marlin_workspace = torch.zeros(
                sms * max_blocks_per_sm, dtype=torch.int32, device=device
            )
        return self._marlin_workspace

    def supports_moe_output_in_alltoall_workspace(self) -> bool:
        # This kernel always allocates and returns its own output tensor, so
        # nothing would fill a workspace-backed buffer that ``combine()`` reads.
        return False

    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: dict | None = None,
    ) -> torch.Tensor:
        del workspace  # Marlin owns its own scratch (see _marlin_workspace).
        x = ctx.x
        token_selected_experts = ctx.token_selected_experts
        token_final_scales = ctx.token_final_scales
        router_logits = ctx.router_logits
        output_dtype = ctx.output_dtype
        assert output_dtype is None or output_dtype == torch.bfloat16
        # Raised rather than asserted: whether the kernel was built is an
        # environment fact the message has to reach the user with, and ``-O``
        # drops asserts.
        if not _has_fused_moe_kernel():
            raise RuntimeError(
                "marlin_nvfp4_moe_gemm is not available. Rebuild TensorRT-LLM "
                "with the fused Marlin MoE kernel for NVFP4."
            )
        assert x.dtype == torch.bfloat16

        output_dtype = torch.bfloat16

        if token_selected_experts is None:
            assert router_logits is not None, (
                f"{type(self).__name__}.run_moe needs token_selected_experts or router_logits"
            )
            token_selected_experts, token_final_scales = self.routing_method.apply(router_logits)

        num_tokens = x.shape[0]
        top_k = token_selected_experts.shape[1]

        local_n = self.expert_size_per_partition
        if local_n != self.num_experts:
            # EP: non-local token-expert pairs are clamped to local expert 0
            # with a zero final scale, so they still run through both GEMMs
            # and are discarded at the combine.
            # TODO(perf): skip them instead — e.g. mark non-local pairs with a
            # sentinel expert id that moe_align_block_size drops. Requires
            # bounds guards in moeAlignKernels.cu (the histogram and
            # count_and_sort kernels index shared/cumsum buffers with the raw
            # id) and zero-initializing unscheduled GEMM output rows before
            # the index_add_ combine.
            slot_start = self.slot_start
            is_local = (token_selected_experts >= slot_start) & (
                token_selected_experts < slot_start + local_n
            )
            token_selected_experts = (token_selected_experts - slot_start).clamp(0, local_n - 1)
            if token_final_scales is None:
                token_final_scales = torch.ones(
                    num_tokens, top_k, dtype=torch.float32, device=x.device
                )
            token_final_scales = token_final_scales * is_local.to(token_final_scales.dtype)
        num_experts = local_n

        workspace = self._ensure_workspace(x.device)  # [num_sms * max_blocks_per_sm(4)] int32

        topk_ids = token_selected_experts.to(torch.int32).contiguous()
        max_num_tokens_padded = num_tokens * top_k + num_experts * _MOE_BLOCK_SIZE

        sorted_token_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=x.device)
        expert_ids_out = torch.empty(
            (max_num_tokens_padded + _MOE_BLOCK_SIZE - 1) // _MOE_BLOCK_SIZE,
            dtype=torch.int32,
            device=x.device,
        )
        num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=x.device)

        torch.ops.trtllm.moe_align_block_size(
            topk_ids,
            num_experts,
            _MOE_BLOCK_SIZE,
            sorted_token_ids,
            expert_ids_out,
            num_tokens_post_pad,
        )

        if token_final_scales is not None:
            topk_weights = token_final_scales.float().contiguous()
        else:
            topk_weights = torch.ones(num_tokens, top_k, dtype=torch.float32, device=x.device)

        hidden_size = x.shape[1]
        k1 = hidden_size
        n1 = self.expand_intermediate_size_per_partition

        gemm1_out = torch.ops.trtllm.marlin_nvfp4_moe_gemm(
            x.contiguous(),
            self.w3_w1_weight,
            b_scales=self.w3_w1_weight_scale,
            global_scale=self.fc31_alpha,
            workspace=workspace,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids_out,
            num_tokens_past_padded=num_tokens_post_pad,
            topk_weights=topk_weights,
            moe_block_size=_MOE_BLOCK_SIZE,
            top_k=top_k,
            mul_topk_weights=False,  # Don't multiply weights in gemm1
            size_n=n1,
            size_k=k1,
            out_dtype=output_dtype,
            use_fp32_reduce=False,
        )  # [num_tokens * top_k, expand_intermediate]

        hidden = self._apply_activation(gemm1_out)

        k2 = self.intermediate_size_per_partition
        n2 = self.unpadded_hidden_size

        # ``hidden`` rows are already independent token-expert pairs, so gemm2
        # re-sorts with top_k=1 to reach each row's expert without expanding a
        # second time.
        num_tokens_gemm2 = num_tokens * top_k
        gemm2_topk_ids = topk_ids.reshape(-1, 1)[:num_tokens_gemm2].contiguous()

        max_padded_g2 = num_tokens_gemm2 + num_experts * _MOE_BLOCK_SIZE
        sorted_ids_g2 = torch.empty(max_padded_g2, dtype=torch.int32, device=x.device)
        expert_ids_g2 = torch.empty(
            (max_padded_g2 + _MOE_BLOCK_SIZE - 1) // _MOE_BLOCK_SIZE,
            dtype=torch.int32,
            device=x.device,
        )
        num_post_pad_g2 = torch.empty(1, dtype=torch.int32, device=x.device)

        torch.ops.trtllm.moe_align_block_size(
            gemm2_topk_ids,
            num_experts,
            _MOE_BLOCK_SIZE,
            sorted_ids_g2,
            expert_ids_g2,
            num_post_pad_g2,
        )

        topk_weights_g2 = topk_weights.reshape(-1, 1)[:num_tokens_gemm2].contiguous()

        gemm2_out = torch.ops.trtllm.marlin_nvfp4_moe_gemm(
            hidden.contiguous(),
            self.w2_weight,
            b_scales=self.w2_weight_scale,
            global_scale=self.fc2_alpha,
            workspace=workspace,
            sorted_token_ids=sorted_ids_g2,
            expert_ids=expert_ids_g2,
            num_tokens_past_padded=num_post_pad_g2,
            topk_weights=topk_weights_g2,
            moe_block_size=_MOE_BLOCK_SIZE,
            top_k=1,
            mul_topk_weights=True,
            size_n=n2,
            size_k=k2,
            out_dtype=output_dtype,
            use_fp32_reduce=False,
        )  # [num_tokens_gemm2, hidden_size]

        # gemm2_out rows are [token_idx * top_k + k, hidden_size] — contiguous
        # because marlin_nvfp4_moe_gemm writes size_n == unpadded_hidden_size
        # with no hidden-dim padding.  The Triton kernel exploits this layout
        # directly, accumulating top_k rows per token in FP32 (more precise
        # than BF16 index_add_ which rounds after each expert).
        gemm2_out = gemm2_out[:num_tokens_gemm2, : self.unpadded_hidden_size]
        if gemm2_out.is_contiguous():
            return sum_topk_expert_outputs(
                gemm2_out,
                num_tokens,
                top_k,
                self.unpadded_hidden_size,
                output_dtype,
            )

        # Fallback: gemm2_out is non-contiguous (e.g. future kernel pads the
        # hidden dimension).  index_add_ handles arbitrary strides safely.
        row_indices = torch.arange(num_tokens_gemm2, device=x.device)
        orig_tokens = row_indices // top_k
        final_hidden_states = torch.zeros(
            (num_tokens, self.unpadded_hidden_size),
            dtype=output_dtype,
            device=x.device,
        )
        final_hidden_states.index_add_(0, orig_tokens, gemm2_out.to(output_dtype))
        return final_hidden_states
