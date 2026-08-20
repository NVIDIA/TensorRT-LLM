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

import inspect
from typing import Dict, List, Optional, Tuple, Union

import torch

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...model_config import ModelConfig
from ...peft.lora.layer import (MOE_LORA_MODULE_NAMES,
                                MOE_LORA_MODULE_TO_KERNEL_SLOT, LoraModuleType,
                                MoeLoraLayer)
from ...peft.lora.validation import has_moe_lora_targets
from ...utils import (ActivationType, AuxStreamType, EventType,
                      Fp4QuantizedTensor)
from .impl_contract import (MoEDeployment, MoEEligibility, MoEInputRequirement,
                            MoEProblem, MoERejectReason, MoERunContext,
                            MoEStaticCapability, require_comm_plan)
from .interface import MoE, _reject
from .quantization import UnquantizedFusedMoEMethod

# isort: off
from .quantization import (
    DeepSeekFP8BlockScalesFusedMoEMethod, FP8QDQFusedMoEMethod,
    MoEWeightLoadingMode, MXFP8CutlassFusedMoEMethod,
    NVFP4CutlassFusedMoEMethod, INT8WoqPerChannelFusedMoEMethod,
    W4A16NVFP4CutlassFusedMoEMethod, W4A8MXFP4FP8CutlassFusedMoEMethod,
    W4A8MXFP4MXFP8CutlassFusedMoEMethod, WFP4A16FusedMoEMethod,
    WInt4AFP8FusedMoEMethod)
# isort: on
from .routing import BaseMoeRoutingMethod


def raise_moe_lora_multichunk_unsupported(num_chunks: int) -> None:
    """Reject multi-chunk execution for routed-expert MoE LoRA.

    Routed-expert MoE LoRA passes per-request/slot adapter metadata that is not
    re-sliced per token-chunk, so multi-chunk execution would mismatch the
    kernel's per-token expansion. Shared by CutlassFusedMoE.forward_impl and the
    MoEScheduler so the message stays in one place.
    """
    raise NotImplementedError(
        f"Routed-expert MoE LoRA does not support multi-chunk execution "
        f"(num_chunks={num_chunks}). Reduce the per-forward token count or "
        f"increase `moe_max_num_tokens` so the MoE runs in a single chunk.")


class CutlassFusedMoE(MoE):
    """
    Fused Mixture of Experts (MoE) Layer with performance tuning.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary CUDA streams for overlapping.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.

    MoE torch custom op:
        In max-throughput mode:
        Quant:
            fp8 block scales (SM90 Hopper only):
                FusedMoE Op: dynamic quant + scatter + gemm1 + swiglu + gemm2 + finalizeMoeRoute (return one tensor)
            p8 qdq, nvfp4:
                FusedMoE Op: scatter + gemm1 + swiglu + gemm2 + finalizeMoeRoute (return one tensor)

    FusedMoE module:
        max-throughput mode:
            routing(topK, etc.) [+ dynamic quant for fp8 qdq and nvfp4 ] [+ fp4_allgather] + FusedMoe Op[no allreduce] + reducescatter, with AttentionDP on
            equals to: dynamic quant + routing(topK, etc.) [+ fp4_allgather] + scatter + gemm1 + swiglu + gemm2 + finalizeMoeRoute [no allreduce] + reducescatter
    """

    # Routed-expert MoE LoRA is fused into this backend's op only; the
    # subclasses below each restate ``supports_moe_lora=False``.
    capabilities = MoEStaticCapability(supports_moe_lora=True)

    # Inherited by every subclass, matching the isinstance check this replaces.
    input_requirement = MoEInputRequirement(routing_scales_dtype=torch.float32)

    # Quantization algorithm support table for can_implement()
    # Format: quant_algo -> {sm_constraint, dtypes}
    # sm_constraint types:
    #   - ("min", N): SM >= N
    #   - ("exact", N): SM == N
    #   - ("in", {N1, N2, ...}): SM in set
    _QUANT_SUPPORT_TABLE = {
        # Unquantized (FP16/BF16): SM >= 80
        None: {
            "sm_constraint": ("min", 80),
            "dtypes": {torch.float16, torch.bfloat16},
        },
        # FP8 per-tensor (QDQ): SM >= 89
        QuantAlgo.FP8: {
            "sm_constraint": ("min", 89),
            "dtypes": {torch.float16, torch.bfloat16, torch.float32},
        },
        # FP8_BLOCK_SCALES: SM in {90, 120}
        QuantAlgo.FP8_BLOCK_SCALES: {
            "sm_constraint": ("in", {90, 120}),
            "dtypes": {torch.bfloat16},
        },
        # NVFP4: SM in {100, 103, 120, 121}
        QuantAlgo.NVFP4: {
            "sm_constraint": ("in", {100, 103, 120, 121}),
            "dtypes": {torch.float16, torch.bfloat16, torch.float8_e4m3fn},
        },
        # W4A16_NVFP4: weights stay NVFP4 but are dequantized to the activation
        # dtype every forward, so what finally runs is the unquantized kernel --
        # this entry tracks that path's limits, not NVFP4 tensor-core support.
        QuantAlgo.W4A16_NVFP4: {
            "sm_constraint": ("min", 80),
            "dtypes": {torch.float16, torch.bfloat16},
        },
        # W4A8_AWQ: SM in {89, 90} only
        QuantAlgo.W4A8_AWQ: {
            "sm_constraint": ("in", {89, 90}),
            "dtypes": {torch.float16, torch.bfloat16},
        },
        # W8A16: SM >= 80
        QuantAlgo.W8A16: {
            "sm_constraint": ("min", 80),
            "dtypes": {torch.float16, torch.bfloat16},
        },
        # W4A16_MXFP4: SM == 90 only
        QuantAlgo.W4A16_MXFP4: {
            "sm_constraint": ("exact", 90),
            "dtypes": {torch.float16, torch.bfloat16},
        },
        # W4A8_MXFP4_FP8: SM in {100, 103}
        QuantAlgo.W4A8_MXFP4_FP8: {
            "sm_constraint": ("in", {100, 103}),
            "dtypes": {torch.float16, torch.bfloat16, torch.float32},
        },
        # W4A8_MXFP4_MXFP8: SM in {100, 103, 120, 121}
        QuantAlgo.W4A8_MXFP4_MXFP8: {
            "sm_constraint": ("in", {100, 103, 120, 121}),
            "dtypes": {torch.float16, torch.bfloat16},
        },
        # MXFP8 (W8A8 e4m3xe4m3 with UE8M0 1x32 block scales): SM in {100, 103}.
        # M3.1 enables construction/load; the fused kernel is M3.2.
        QuantAlgo.MXFP8: {
            "sm_constraint": ("in", {100, 103}),
            "dtypes": {torch.float16, torch.bfloat16},
        },
    }

    _GPTOSS_SUPPORTED_ALGOS: frozenset[Optional[QuantAlgo]] = frozenset({
        None,
        QuantAlgo.NVFP4,
        QuantAlgo.W4A16_MXFP4,
        QuantAlgo.W4A8_MXFP4_FP8,
        QuantAlgo.W4A8_MXFP4_MXFP8,
    })
    """Algorithms whose weight methods can serve gpt-oss / MiniMax SwiGLU.

    Unquantized and the MXFP4 family can load a 1-D gpt-oss expert bias.
    NVFP4 is included for MiniMax-style SwigluBias without expert bias.
    ``can_implement`` still rejects NVFP4 when ``p.bias is True`` because the
    NVFP4 weight pad only accepts 2-D tensors.
    """

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        """Cutlass grouped-GEMM MoE: the widest quant and SM coverage there is.

        Per-algorithm SM and dtype support lives in ``_QUANT_SUPPORT_TABLE``;
        this method is only the interpreter for it.
        """
        sm_version = d.env.sm
        quant_algo = p.quant_algo

        # Check minimum SM version for Cutlass backend
        if sm_version < 80:
            return _reject(
                MoERejectReason.SM_UNSUPPORTED,
                f"CutlassFusedMoE requires SM >= 80, got SM{sm_version}")

        if p.swiglu_gptoss_style and quant_algo not in cls._GPTOSS_SUPPORTED_ALGOS:
            supported = sorted("unquantized" if a is None else a.name
                               for a in cls._GPTOSS_SUPPORTED_ALGOS)
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"CutlassFusedMoE cannot load a gpt-oss bias for "
                f"quant_algo={quant_algo}; supported: {supported}")

        # NVFP4 can run SwigluBias, but cannot pad a 1-D gpt-oss expert bias.
        if (p.swiglu_gptoss_style and quant_algo == QuantAlgo.NVFP4
                and p.bias is True):
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                "CutlassFusedMoE NVFP4 cannot load a 1-D gpt-oss expert bias "
                "(weight-pad assert is 2-D); MiniMax-style SwigluBias without "
                "bias is eligible")

        # Check if quant_algo is supported
        if quant_algo not in cls._QUANT_SUPPORT_TABLE:
            return _reject(
                MoERejectReason.QUANT_UNSUPPORTED,
                f"CutlassFusedMoE does not support quant_algo={quant_algo}")

        support_info = cls._QUANT_SUPPORT_TABLE[quant_algo]

        # Check SM version constraint
        constraint_type, constraint_value = support_info["sm_constraint"]
        algo_name = "unquantized" if quant_algo is None else quant_algo.name

        if constraint_type == "min":
            if sm_version < constraint_value:
                return _reject(
                    MoERejectReason.SM_UNSUPPORTED,
                    f"CutlassFusedMoE {algo_name} requires SM >= {constraint_value}, "
                    f"got SM{sm_version}")
        elif constraint_type == "exact":
            if sm_version != constraint_value:
                return _reject(
                    MoERejectReason.SM_UNSUPPORTED,
                    f"CutlassFusedMoE {algo_name} only supports SM{constraint_value}, "
                    f"got SM{sm_version}")
        elif constraint_type == "in":
            if sm_version not in constraint_value:
                sm_list = "/".join(f"SM{v}" for v in sorted(constraint_value))
                return _reject(
                    MoERejectReason.SM_UNSUPPORTED,
                    f"CutlassFusedMoE {algo_name} only supports {sm_list}, "
                    f"got SM{sm_version}")

        # Check activation dtype
        supported_dtypes = support_info["dtypes"]
        if p.dtype_act not in supported_dtypes:
            dtype_list = ", ".join(str(dtype) for dtype in supported_dtypes)
            return _reject(
                MoERejectReason.DTYPE_UNSUPPORTED,
                f"CutlassFusedMoE {algo_name} requires {dtype_list}, "
                f"got {p.dtype_act}")

        # Routed-expert MoE LoRA supports unquantized fp16/bf16 or per-tensor FP8 only.
        if d.moe_lora_enabled and quant_algo not in (None, QuantAlgo.FP8):
            return _reject(
                MoERejectReason.LORA_UNSUPPORTED,
                "CutlassFusedMoE MoE LoRA only supports unquantized "
                f"fp16/bf16 or per-tensor FP8 (qdq); got quant_algo={quant_algo}"
            )

        return MoEEligibility.ok()

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
        apply_router_weight_on_input: bool = False,
        layer_idx: Optional[int] = None,
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        swiglu_limit: Optional[torch.Tensor] = None,
        swiglu_limit_scalar: Optional[float] = None,
        init_load_balancer: bool = True,
        activation_type: ActivationType = ActivationType.Swiglu,
    ):

        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            weight_loading_mode=weight_loading_mode,
            bias=bias,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            swiglu_limit_scalar=swiglu_limit_scalar,
            layer_idx=layer_idx,
            init_load_balancer=init_load_balancer,
            activation_type=activation_type,
        )

        # Store original hidden size before any potential padding
        self.unpadded_hidden_size = self.hidden_size

        if model_config.quant_config and model_config.quant_config.layer_quant_mode.has_w4a16_mxfp4(
        ):
            self.hidden_size = ((self.hidden_size + 127) // 128) * 128
            self.intermediate_size_per_partition = (
                (self.intermediate_size_per_partition + 127) // 128) * 128

        # Note: num_slots, expert_size_per_partition, initial_global_assignments,
        # slot_start, slot_end, initial_local_expert_ids are all initialized by
        # base class's _init_load_balancer() method

        # moe_max_num_tokens is set in ModelConfig.__post_init__ if not specified
        # The default value is max_num_tokens * dp_size
        self.moe_max_num_tokens = model_config.moe_max_num_tokens
        # The auxiliary CUDA stream and CUDA events are only used when MoE chunking is applied
        default_moe_max_num_tokens = model_config.max_num_tokens * model_config.mapping.dp_size
        if self.moe_max_num_tokens < default_moe_max_num_tokens:
            self.aux_stream = aux_stream_dict[
                AuxStreamType.
                MoeChunkingOverlap] if aux_stream_dict is not None else torch.cuda.Stream(
                )
            self.event_dict = {
                key: torch.cuda.Event()
                for key in [EventType.Main, EventType.MoeChunkingOverlap]
            }
        else:
            self.aux_stream = None
            self.event_dict = None

        # The profiler converges on the same best tactic when the number of tokens is large enough.
        # To avoid long profiling time, the max number of tokens used in the profiling is capped to
        # around 16k tokens per expert, which is well into the compute bound domain.
        self.tune_max_num_tokens = min(
            self.moe_max_num_tokens,
            16384 * self.num_slots // routing_method.get_experts_per_token(),
        )
        self.has_been_profiled = False
        self.has_been_profiled_min_latency = False

        # If True, the router weight will be multiplied on the input rather than at the end of FC2
        self.apply_router_weight_on_input = apply_router_weight_on_input

        # Finalize fusion should be disabled if Lora is used.
        self.use_fused_finalize = not model_config.moe_disable_finalize_fusion and model_config.lora_config is None

        # Routed-expert LoRA is fused inside torch.ops.trtllm.fused_moe. This
        # flag records whether the layer was configured with MoE LoRA targets,
        # so forward_impl can reject stray lora_params instead of ignoring them.
        self._moe_lora_enabled = self._has_moe_lora_targets(model_config)

        # Discovery-only marker submodule. The actual LoRA GEMMs are fused into
        # torch.ops.trtllm.fused_moe; MoeLoraLayer exists purely so that
        # CudaGraphLoraManager and the target-module validator can find this MoE
        # layer via isinstance(child, LoraLayer) traversal and read its
        # lora_module_types / output_hidden_sizes when building slot tables.
        self.lora = self._maybe_make_lora_marker(model_config)

        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    # ---- Routed-expert LoRA helpers ----

    def _has_moe_lora_targets(self, model_config: ModelConfig) -> bool:
        """Return True iff this MoE layer is in the routed-expert LoRA
        target-module set. The LoRA application itself is fused into
        `torch.ops.trtllm.fused_moe`; no submodule is registered.
        """
        return has_moe_lora_targets(getattr(model_config, "lora_config", None))

    def _maybe_make_lora_marker(
            self, model_config: ModelConfig) -> Optional[MoeLoraLayer]:
        """Construct a MoeLoraLayer marker iff this MoE layer is in the LoRA
        target-module set. The marker is a discovery-only submodule; the actual
        LoRA application is fused into torch.ops.trtllm.fused_moe.

        The output_hidden_sizes recorded here are the per-token outputs of the
        LoRA-side GEMM (not per-expert weight shapes): MOE_H_TO_4H / MOE_GATE
        produce intermediate_size, MOE_4H_TO_H produces hidden_size.
        """
        lora_config = getattr(model_config, "lora_config", None)
        if lora_config is None:
            return None
        # Normalize to lowercase to match has_moe_lora_targets (which lowercases
        # before comparing), so a mixed-case config marks the layer and builds
        # the discovery marker consistently.
        targets = {
            name.lower()
            for name in (getattr(lora_config, "lora_target_modules", []) or [])
        }
        active_modules: List[LoraModuleType] = []
        active_out_sizes: List[int] = []
        for name in MOE_LORA_MODULE_NAMES:
            if name not in targets:
                continue
            module_type = LoraModuleType.from_string(name)
            if name == "moe_4h_to_h":
                active_out_sizes.append(self.hidden_size)
            else:
                active_out_sizes.append(self.intermediate_size)
            active_modules.append(module_type)
        if not active_modules:
            return None
        return MoeLoraLayer(active_modules, active_out_sizes)

    def reserve_moe_lora_cuda_graph_workspace(self, max_num_tokens: int,
                                              max_lora_rank: int,
                                              max_lora_size: int) -> None:
        """Pre-size the C++ FusedMoeRunner's MoE-LoRA scratch to the engine's
        worst case so no (re)allocation happens during CUDA graph capture or
        replay (which would dangle addresses baked into earlier graphs).

        No-op for layers without MoE LoRA targets and for quantized layers (MoE
        LoRA requires unquantized fp16/bf16); idempotent and grow-only. Call
        during warmup, before any capture that exercises MoE LoRA;
        CudaGraphLoraManager does this automatically.

        Args:
            max_num_tokens: Worst-case tokens in a captured forward
                (max_batch_size * max_tokens_per_seq).
            max_lora_rank: Largest LoRA rank across adapters.
            max_lora_size: Adapter-slot pool size for the slot-indexed device tables.
        """
        if not self._moe_lora_enabled or max_num_tokens <= 0:
            return
        # MoE LoRA runs on the unquantized fp16/bf16 path or the per-tensor FP8
        # (qdq) path (see moeOp.cpp). Any other quant mode (FP8 block-scale,
        # NVFP4, MXFP8, integer WoQ) is rejected by the C++ op, so a layer in
        # those modes can never reach the LoRA scratch; skip and let the runtime
        # path error loudly.
        if getattr(self, "has_any_quant", False) and not self.has_fp8_qdq:
            return
        # Weights must exist to read the runner's weight dtype. If they have not
        # been created yet, skip; the lazy sizing + in-capture guard still
        # protect correctness.
        if getattr(self, "w3_w1_weight", None) is None:
            return

        # The reservation must cover the engine's worst case, otherwise the first
        # capture hits a lazy allocation that the C++ in-capture guard rejects.
        assert max_lora_rank > 0, (
            "reserve_moe_lora_cuda_graph_workspace requires max_lora_rank > 0 "
            f"(got {max_lora_rank}); set lora_config.max_lora_rank.")
        assert max_lora_size > 0, (
            "reserve_moe_lora_cuda_graph_workspace requires max_lora_size > 0 "
            f"(got {max_lora_size}).")

        # The reservation must land on the *same* cached C++ FusedMoeRunner that
        # the runtime torch.ops.trtllm.fused_moe op uses on this layer, so the
        # MoERunner instance key must match the runtime key exactly. The runtime
        # key uses the activation dtype the op sees:
        #   - per-tensor FP8 (qdq): quantize_input casts activations to e4m3, so
        #     x/weight are fp8 and the output (LoRA compute) dtype is self.dtype;
        #   - unquantized fp16/bf16: x/weight/output all equal self.dtype.
        # Every quant flag in the key is False for both (per-tensor FP8 is not
        # block-scaled / MXFP8 / W4). If a runtime call ever uses a different
        # key, the C++ capture guard surfaces a clear error rather than
        # corrupting replay.
        weight_dtype = self.w3_w1_weight.dtype
        if self.has_fp8_qdq:
            act_dtype = torch.float8_e4m3fn
            output_dtype = self.dtype
        else:
            assert self.dtype in (torch.float16, torch.bfloat16), (
                "MoE LoRA requires fp16/bf16 activations to reserve a "
                f"deterministic FusedMoeRunner key; got {self.dtype}.")
            act_dtype = self.dtype
            output_dtype = self.dtype

        from ...custom_ops.torch_custom_ops import MoERunner

        runner = MoERunner(
            x_dtype=act_dtype,
            weight_dtype=weight_dtype,
            output_dtype=output_dtype,
            top_k=self.routing_method.experts_per_token,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=self.cluster_size,
            cluster_rank=self.cluster_rank,
            use_deepseek_fp8_block_scale=False,
            use_w4_group_scaling=False,
            use_int8_woq_per_channel=False,
            use_mxfp8_act_scaling=False,
            min_latency_mode=False,
            use_fused_finalize=self.use_fused_finalize,
            activation_type=self.activation_type,
        )
        runner.fused_moe_runner.reserve_lora_host_buffers(
            int(max_num_tokens),
            int(self.routing_method.experts_per_token),
            int(max_lora_rank),
            int(max_lora_size),
            bool(self.is_gated_activation),
        )

    def _moe_lora_active(self, lora_params: Optional[Dict]) -> bool:
        """Return True when lora_params carries routed-expert MoE LoRA tensors
        for this layer, meaning run_moe would fuse a LoRA delta.
        """
        if not lora_params or self.layer_idx is None:
            return False
        # CUDA-graph slot-indexed mode carries MoE LoRA in cuda_graph_params
        # rather than a per-layer eager dict (mirrors _extract_moe_lora_tensors),
        # so consult the graph layer map to keep the stray-param and multi-chunk
        # guards effective during capture/replay.
        if lora_params.get("use_cuda_graph_mode", False):
            cuda_graph_params = lora_params.get("cuda_graph_params")
            if cuda_graph_params is None:
                return False
            layer_module2key = getattr(cuda_graph_params, "layer_module2key",
                                       {})
            return any(
                (self.layer_idx,
                 int(LoraModuleType.from_string(name))) in layer_module2key
                for name in MOE_LORA_MODULE_NAMES)
        layer_params = lora_params.get(self.layer_idx, {})
        if not layer_params:
            return False
        return any(
            int(LoraModuleType.from_string(name)) in layer_params
            for name in MOE_LORA_MODULE_NAMES)

    @staticmethod
    def _empty_kernel_slot_dict() -> Dict[str, Optional[torch.Tensor]]:
        return {"fc1": None, "fc2": None, "gated": None}

    def _gather_moe_lora_slots(self, source):
        """Gather per-kernel-slot (ranks, weight_ptrs) tensors.

        `source(module_type)` returns the (ranks, weight_ptrs) pair for an MoE
        LoRA module, or None if absent. Returns (ranks_by_slot, ptrs_by_slot)
        dicts keyed by the kernel slot ("fc1"/"gated"/"fc2"); see
        MOE_LORA_MODULE_TO_KERNEL_SLOT for the module->slot convention. Shared by
        the eager (per-request) and CUDA-graph (slot-indexed) extraction paths.
        """
        ranks = self._empty_kernel_slot_dict()
        ptrs = self._empty_kernel_slot_dict()
        for module_type, slot in MOE_LORA_MODULE_TO_KERNEL_SLOT.items():
            got = source(module_type)
            if got is None:
                continue
            ranks[slot], ptrs[slot] = got
        return ranks, ptrs

    @staticmethod
    def _require_fc1_fc2(ranks: Dict[str, Optional[torch.Tensor]]) -> None:
        """The kernel always dereferences the fc1 and fc2 rank/pointer arrays
        (see setupLoraWorkspace in moe_kernels.cu), so moe_h_to_4h (fc1/gate) and
        moe_4h_to_h (fc2/down) must both be present when MoE LoRA is active. The
        gated slot (moe_gate) is only read for gated activations.
        """
        if ranks["fc1"] is None or ranks["fc2"] is None:
            raise ValueError(
                "MoE LoRA requires both `moe_h_to_4h` (gate/SiLU) and "
                "`moe_4h_to_h` (down) in lora_target_modules.")

    def _extract_moe_lora_tensors(
            self, lora_params: Optional[Dict]) -> Optional[Dict[str, object]]:
        """Pick the MoE-side LoRA tensors out of the global `lora_params` dict
        for this layer. Returns a dict with the kwargs expected by
        `torch.ops.trtllm.fused_moe`, or None when no MoE LoRA applies.

        Each entry is a CPU tensor:
            *_lora_ranks         : int32  [num_seqs]
            *_lora_weight_ptrs   : int64  [num_seqs, 3]   (A, B, DoRA_unused)
            host_request_types   : int32  [num_seqs]      (0=CTX, 1=GEN)
            host_context_lengths : int32  [num_seqs]
            lora_max_low_rank    : int (max rank across the active modules)
        """
        if not lora_params:
            return None
        # Slot-indexed (CUDA-graph decode) path: the per-token expansion is
        # driven inside the op by token_to_slot indexed into stable slot
        # tables owned by CudaGraphLoraParams (see _extract_moe_lora_tensors_cuda_graph).
        if lora_params.get("use_cuda_graph_mode", False):
            return self._extract_moe_lora_tensors_cuda_graph(lora_params)
        layer_params = lora_params.get(
            self.layer_idx, {}) if self.layer_idx is not None else {}
        if not layer_params:
            return None

        # Gather (ranks, weight_ptrs) per kernel slot. weight_pointers is built
        # flat ([num_seqs * 3], row-major (A, B, DoRA) per seq) in
        # PyTorchModelEngine._build_lora_params; the op expects [num_seqs, 3].
        active_max_rank = 0

        def _source(module_type: LoraModuleType):
            nonlocal active_max_rank
            entry = layer_params.get(int(module_type))
            if entry is None:
                return None
            rank_t = entry["adapter_size"]
            if rank_t.numel() > 0:
                active_max_rank = max(active_max_rank, int(rank_t.max().item()))
            return rank_t, entry["weight_pointers"].reshape(-1, 3)

        ranks, ptrs = self._gather_moe_lora_slots(_source)
        if all(v is None for v in ranks.values()):
            return None
        self._require_fc1_fc2(ranks)

        num_seqs = lora_params["num_seqs"]

        def _slice(t):
            return t[:num_seqs].contiguous() if t is not None else None

        return {
            "fc1_lora_ranks": _slice(ranks["fc1"]),
            "fc1_lora_weight_ptrs": _slice(ptrs["fc1"]),
            "fc2_lora_ranks": _slice(ranks["fc2"]),
            "fc2_lora_weight_ptrs": _slice(ptrs["fc2"]),
            "gated_lora_ranks": _slice(ranks["gated"]),
            "gated_lora_weight_ptrs": _slice(ptrs["gated"]),
            "host_request_types": _slice(lora_params["host_request_types"]),
            "host_context_lengths": _slice(lora_params["prompt_lens_cpu"]),
            "lora_max_low_rank": active_max_rank,
        }

    def _extract_moe_lora_tensors_cuda_graph(
            self, lora_params: Dict) -> Optional[Dict[str, object]]:
        """CUDA-graph slot-indexed extraction for routed-expert MoE LoRA.

        Pulls per-module slot tables and token_to_slot out of
        CudaGraphLoraParams and returns the slot-indexed kwargs accepted by
        torch.ops.trtllm.fused_moe. Returns None when this layer does not
        carry any MoE LoRA modules in the graph layer map.

        Returned tensor addresses are stable across captures and replays: they
        come from persistent pinned host buffers owned by CudaGraphLoraParams
        and the per-module packed pointer cache. Uses the same module->kernel
        slot convention as the per-request path (moe_h_to_4h -> fc1,
        moe_gate -> gated, moe_4h_to_h -> fc2).
        """
        if self.layer_idx is None:
            return None
        cuda_graph_params = lora_params.get("cuda_graph_params")
        if cuda_graph_params is None:
            return None

        def _source(module_type: LoraModuleType):
            return cuda_graph_params.get_moe_slot_inputs(
                self.layer_idx, int(module_type))

        slot_ranks, slot_ptrs = self._gather_moe_lora_slots(_source)
        if slot_ranks["fc1"] is None or slot_ranks["fc2"] is None:
            return None

        num_seqs = lora_params["num_seqs"]
        tokens_per_seq = getattr(cuda_graph_params, "max_tokens_per_seq", 1)
        num_tokens = num_seqs * max(int(tokens_per_seq), 1)
        token_to_slot = cuda_graph_params.token_to_slot_host[:
                                                             num_tokens].contiguous(
                                                             )

        # Pass the global max LoRA rank, not the per-step active max: the device
        # path uses it only to size the low-rank workspace strides baked into the
        # captured graph, so the global max keeps them valid for any per-slot
        # rank across replays. The actual per-token rank is read on-device from
        # the slot table, so a smaller rank just runs a smaller GEMM.
        max_rank = int(getattr(cuda_graph_params, "max_rank", 0))
        if max_rank <= 0:
            return None

        return {
            "fc1_slot_lora_ranks":
            slot_ranks["fc1"].contiguous(),
            "fc1_slot_lora_weight_ptrs":
            slot_ptrs["fc1"].contiguous(),
            "fc2_slot_lora_ranks":
            slot_ranks["fc2"].contiguous(),
            "fc2_slot_lora_weight_ptrs":
            slot_ptrs["fc2"].contiguous(),
            "gated_slot_lora_ranks":
            (slot_ranks["gated"].contiguous()
             if slot_ranks["gated"] is not None else None),
            "gated_slot_lora_weight_ptrs":
            (slot_ptrs["gated"].contiguous()
             if slot_ptrs["gated"] is not None else None),
            "token_to_slot":
            token_to_slot,
            "lora_max_low_rank":
            max_rank,
        }

    def _check_configs(self):
        assert self._weights_created

        if self.apply_router_weight_on_input:
            assert self.routing_method.top_k == 1, "Current walkaround only supports top-1 routing"

        if self.quant_config and self.quant_config.quant_mode.has_any_quant(
                exclude_kv_cache=True):
            if not (self.quant_config.quant_mode.has_nvfp4()
                    | self.quant_config.quant_mode.has_fp8_block_scales()
                    | self.quant_config.quant_mode.has_fp8_qdq()
                    | self.quant_config.quant_mode.is_weight_only()
                    | self.quant_config.quant_mode.has_w4a8_mxfp4_fp8()
                    | self.quant_config.quant_mode.has_w4a16_mxfp4()
                    | self.quant_config.quant_mode.has_w4a8_mxfp4_mxfp8()
                    | self.quant_config.quant_mode.has_mxfp8()):
                raise ValueError(
                    f"unsupported quantization mode: {self.quant_config.quant_mode}"
                )

    @property
    def has_w4afp8(self):
        assert self._weights_created
        return self.quant_config and self.quant_config.quant_mode.is_int4_weight_only_per_group(
        )

    @property
    def has_int8_woq_per_channel(self):
        return self.quant_config and self.quant_config.layer_quant_mode.is_int8_weight_only(
        ) and not self.quant_config.layer_quant_mode.has_per_group_scaling()

    def quantize_input(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        post_quant_comm: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Quantize input tensor - CutlassFusedMoE implementation

        Handles all quantization cases for Cutlass backend.

        Args:
            x: Input tensor to quantize
            post_quant_comm: Whether this is for post-quantization communication
                           (allgather or alltoall). If True, x_sf will be reshaped to 2D.

        Returns:
            Tuple of (quantized_x, x_sf)
        """
        x_sf = None
        if self.has_any_quant:
            # W4A16 NVFP4 path keeps activations hp; skip FP4 quant below.
            if isinstance(self.quant_method, W4A16NVFP4CutlassFusedMoEMethod):
                return x, None
            if self.has_fp8_qdq or self.has_w4a8_mxfp4_fp8:
                x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    x, self.fc31_input_dequant)
            elif self.has_deepseek_fp8_block_scales:
                # No quantization needed here, handled in kernel
                pass
            elif self.has_w4afp8:
                # No quantization needed here, handled in kernel
                pass
            elif self.has_w4a16_mxfp4:
                # Padding deferred to run_moe so that dispatch sends
                # unpadded tensors (avoids NVLink workspace overallocation).
                pass
            elif self.has_int8_woq_per_channel:
                # No quantization needed here, handled in kernel
                pass
            elif self.has_nvfp4:
                if hasattr(
                        self,
                        'fc31_act_scale') and self.fc31_act_scale is not None:
                    assert not isinstance(
                        x, Fp4QuantizedTensor
                    ), "Fp4QuantizedTensor is not expected for AWQ quantization."
                    x = x * self.fc31_act_scale

                # Dynamic quantization: compute input_scale from current input
                # and update alpha in-place (same tensor addresses for CUDA graph).
                if self.force_dynamic_quantization and hasattr(
                        self, 'fc31_weight_scale_2'):
                    FP8_MAX, E2M1_MAX = 448.0, 6.0
                    amax_input = torch.amax(torch.abs(x)).float()
                    dyn_input_scale = FP8_MAX * E2M1_MAX / amax_input

                    # fc31_alpha[e] = weight_scale_2[e] / dyn_input_scale
                    self.fc31_alpha.data.copy_(self.fc31_weight_scale_2.data /
                                               dyn_input_scale)
                    self.fc31_input_scale.data.copy_(dyn_input_scale)

                # Quantize based on communication scenario
                if post_quant_comm:
                    if isinstance(x, Fp4QuantizedTensor):
                        assert not x.is_sf_swizzled, "Fp4QuantizedTensor should not be swizzled before communication"
                        x, x_sf = x.fp4_tensor, x.scaling_factor
                        x_row = x.shape[0]
                    else:
                        x_row = x.shape[0]
                        x, x_sf = torch.ops.trtllm.fp4_quantize(
                            x, self.fc31_input_scale, self.scaling_vector_size,
                            False, False)
                    # Reshape x_sf to 2D for post-quant communication
                    if x_sf is not None:
                        x_sf = x_sf.view((x_row, -1))
                else:
                    if not isinstance(x, Fp4QuantizedTensor):
                        x, x_sf = torch.ops.trtllm.fp4_quantize(
                            x, self.fc31_input_scale, self.scaling_vector_size,
                            False, True)
            elif self.has_w4a8_mxfp4_mxfp8 or self.has_mxfp8:
                # MXFP8 dynamic activation quantize. The MXFP8xMXFP8 path reuses
                # the same activation quant kernel as W4A8 MXFP4xMXFP8 -- only
                # the weight side differs (B element widens from fp4 to fp8).
                if post_quant_comm:
                    x, x_sf = torch.ops.trtllm.mxfp8_quantize(
                        x, False, alignment=self.quant_method.weight_alignment)
                    # Reshape x_sf to 2D for post-quant communication
                    # x.shape[0] is padded
                    if x_sf is not None:
                        x_sf = x_sf.view((x.shape[0], -1))
                else:
                    x, x_sf = torch.ops.trtllm.mxfp8_quantize(
                        x, True, alignment=self.quant_method.weight_alignment)
            else:
                raise ValueError(
                    f"unsupported quantization mode: {self.quant_config.quant_mode}"
                )

        return x, x_sf

    def _supports_load_balancer(self) -> bool:
        """CutlassFusedMoE supports load balancer."""
        return True

    def _get_quant_method(self):
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True):
            if self.quant_config.layer_quant_mode.has_fp8_qdq():
                return FP8QDQFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_fp8_block_scales():
                return DeepSeekFP8BlockScalesFusedMoEMethod()
            elif self.quant_config.quant_algo == QuantAlgo.W4A16_NVFP4:
                return W4A16NVFP4CutlassFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_nvfp4():
                return NVFP4CutlassFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.is_int4_weight_only_per_group(
            ):
                return WInt4AFP8FusedMoEMethod()
            elif self.has_int8_woq_per_channel:
                return INT8WoqPerChannelFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_w4a8_mxfp4_fp8():
                return W4A8MXFP4FP8CutlassFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_w4a16_mxfp4():
                return WFP4A16FusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_w4a8_mxfp4_mxfp8():
                return W4A8MXFP4MXFP8CutlassFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_mxfp8():
                return MXFP8CutlassFusedMoEMethod()
            else:
                raise ValueError(
                    f"Unsupported quantization mode: {self.quant_config.quant_mode}"
                )
        else:
            return UnquantizedFusedMoEMethod()

    def create_weights(self):
        if self._weights_created:
            return

        self.quant_method = self._get_quant_method()
        self.quant_method.create_weights(self)

        self._weights_created = True
        self._check_configs()

    def supports_moe_output_in_alltoall_workspace(self):
        return True

    def _tuner_shapes(
        self,
        ctx: MoERunContext,
        enable_alltoall: Optional[bool],
    ) -> Tuple[Optional[int], Optional[int]]:
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

    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Run MoE computation with Cutlass backend.

        This method encapsulates the core MoE computation logic, handling different
        quantization schemes.

        Returns:
            final_hidden_states: Output tensor from MoE computation
        """
        del workspace  # Cutlass allocates its own intermediates.
        plan = require_comm_plan(self, ctx)
        x = ctx.x
        token_selected_experts = ctx.token_selected_experts
        token_final_scales = ctx.token_final_scales
        x_sf = ctx.x_sf
        output_dtype = ctx.output_dtype
        lora_params = ctx.lora_params
        is_sf_swizzled = plan.input_sf_swizzled
        moe_output = plan.moe_output
        enable_alltoall = plan.enable_alltoall
        tuner_num_tokens, tuner_top_k = self._tuner_shapes(ctx, enable_alltoall)

        # W4A16 NVFP4 fallback (SM<100).
        if isinstance(self.quant_method, W4A16NVFP4CutlassFusedMoEMethod):
            return self._run_moe_w4a16_nvfp4(
                x,
                token_selected_experts,
                token_final_scales,
                output_dtype=output_dtype,
                tuner_num_tokens=tuner_num_tokens,
                tuner_top_k=tuner_top_k,
                moe_output=moe_output,
                enable_alltoall=enable_alltoall,
            )

        # SM120 + FP8 block scales: use Triton kernel (CUTLASS TMA fails on SM120
        # for large token counts due to cuTensorMapEncodeTiled limitations).
        if self.has_deepseek_fp8_block_scales and get_sm_version() == 120:
            from .fused_moe_triton_fp8_block_scale import \
                run_triton_fp8_block_scale_moe

            # forward_chunk sets token_final_scales=None when
            # apply_router_weight_on_input=True (weights already folded into x);
            # substitute ones so the Triton kernel's per-token scaling is a no-op.
            if token_final_scales is None:
                token_final_scales = torch.ones_like(token_selected_experts,
                                                     dtype=torch.float32)
            # token_selected_experts contains GLOBAL expert IDs in the non-alltoall
            # path (slot_start .. slot_end-1 for this rank's local experts, plus
            # IDs for other ranks).  The Triton kernel operates on LOCAL IDs
            # (0 .. expert_size_per_partition-1), so remap and zero-scale any
            # non-local token-expert pairs to suppress their contribution.
            local_n = self.expert_size_per_partition
            if enable_alltoall:
                # After alltoall dispatch, IDs are already local; padding = local_n
                local_ids = token_selected_experts.clamp(0, local_n - 1)
                is_local = token_selected_experts < local_n
            else:
                slot_start = self.slot_start
                local_ids = (token_selected_experts - slot_start).clamp(
                    0, local_n - 1)
                is_local = ((token_selected_experts >= slot_start)
                            & (token_selected_experts < slot_start + local_n))
            local_scales = token_final_scales * is_local.to(
                token_final_scales.dtype)
            result = run_triton_fp8_block_scale_moe(
                x,
                local_ids,
                local_scales,
                self.w3_w1_weight,
                self.quant_scales.fc_weight_scales,
                self.w2_weight,
                self.quant_scales.proj_weight_scales,
                activation_type=self.activation_type,
                output_dtype=output_dtype,
            )
            return result

        # Pad input for mxfp4 alignment (128-aligned hidden_size).
        # Done here rather than in quantize_input so that dispatch sends
        # unpadded tensors and avoids NVLink workspace overallocation.
        if self.has_w4a16_mxfp4:
            pad_size = self.hidden_size - x.shape[-1]
            if pad_size > 0:
                x = torch.nn.functional.pad(x, (0, pad_size))

        # Determine weight dtype based on quantization mode
        weight_dtype = self.w3_w1_weight.dtype
        if self.has_any_quant:
            if self.has_w4afp8:
                weight_dtype = torch.quint4x2
            elif self.has_w4a16_mxfp4:
                weight_dtype = torch.uint8

        use_dynamic_fc2_scale = (self.has_nvfp4 and getattr(
            self, 'force_dynamic_quantization', False)
                                 and hasattr(self, 'fc2_weight_scale_2'))

        lora_kwargs = self._extract_moe_lora_tensors(lora_params)
        if lora_kwargs is None:
            lora_kwargs = {}

        result = torch.ops.trtllm.fused_moe(
            x,
            token_selected_experts,
            token_final_scales,
            self.w3_w1_weight.view(weight_dtype),
            self.w3_w1_bias,
            self.w2_weight.view(weight_dtype),
            self.w2_bias,
            output_dtype,
            quant_scales=list(self.quant_scales) +
            ([self.fc2_weight_scale_2] if use_dynamic_fc2_scale else []),
            input_sf=x_sf,
            swizzled_input_sf=is_sf_swizzled,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
            swiglu_limit=self.swiglu_limit,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=self.cluster_size,
            cluster_rank=self.cluster_rank,
            enable_alltoall=enable_alltoall,
            use_deepseek_fp8_block_scale=self.has_deepseek_fp8_block_scales,
            use_w4_group_scaling=self.has_w4afp8 or self.has_w4a16_mxfp4,
            use_int8_woq_per_channel=self.has_int8_woq_per_channel,
            # use_mxfp8_act_scaling drives dynamic MXFP8 activation quantization
            # before the GEMM; required for both W4A8 MXFP4xMXFP8 and W8A8
            # MXFP8xMXFP8 paths.
            use_mxfp8_act_scaling=self.has_w4a8_mxfp4_mxfp8 or self.has_mxfp8,
            min_latency_mode=False,
            use_fused_finalize=self.use_fused_finalize,
            tune_max_num_tokens=self.tune_max_num_tokens,
            tuner_num_tokens=tuner_num_tokens,
            tuner_top_k=tuner_top_k,
            activation_type=self.activation_type,
            unpadded_hidden_size=self.unpadded_hidden_size,
            out_tensor=moe_output,
            use_dynamic_fc2_scale=use_dynamic_fc2_scale,
            # use_mxfp8_weight_scaling selects the MXFP8xMXFP8 block-scaled
            # kernel path within the <e4m3, e4m3> CutlassMoeFCRunner template
            # (per-tensor FP8 otherwise).
            use_mxfp8_weight_scaling=self.has_mxfp8,
            **lora_kwargs,
        )
        # When moe_output is provided, the result is written in-place and
        # fused_moe returns empty list to avoid aliasing constraint violation.
        # Otherwise, unpack the single tensor from the returned list.
        if moe_output is not None:
            final_hidden_states = moe_output
        else:
            final_hidden_states = result[0]

        return final_hidden_states

    def _run_moe_w4a16_nvfp4(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
        tuner_num_tokens: Optional[int] = None,
        tuner_top_k: Optional[int] = None,
        moe_output: Optional[torch.Tensor] = None,
        *,
        enable_alltoall: bool,
    ) -> torch.Tensor:
        """W4A16 fallback for NVFP4 MoE on SM<100. Active-mask dequant into
        a static [E_total, N, K] bf16 workspace, then bf16 fused_moe with the
        original (global) token_selected_experts. CUDA-graph capturable.

        ``enable_alltoall`` has no default because it picks the expert-id remap
        below, and either default is silently wrong for half the callers: the
        ids are local after an alltoall dispatch and global otherwise, so a
        wrong guess shifts every id by ``slot_start`` without failing.
        """
        assert isinstance(self.quant_method, W4A16NVFP4CutlassFusedMoEMethod)

        if output_dtype is None:
            output_dtype = x.dtype

        # Same EP id convention as the FP8 path above: global ids (or
        # ``local_n``-padded under alltoall). Clamp to local range so the
        # active-mask scatter is in-bounds; non-local tokens collapse onto a
        # boundary expert (1 extra dequant/rank). ``trtllm.fused_moe`` below
        # still gets the original global ids -- it does its own remap.
        local_n = self.expert_size_per_partition
        if enable_alltoall:
            local_ids = token_selected_experts.clamp(0, local_n - 1)
        else:
            local_ids = (token_selected_experts - self.slot_start).clamp(
                0, local_n - 1)

        w3_w1_hp, w2_hp = self.quant_method.dequant_active_experts_to_hp(
            self, local_ids, output_dtype)

        # bf16 fused_moe with empty quant_scales (matches unquantized path).
        result = torch.ops.trtllm.fused_moe(
            x,
            token_selected_experts,
            token_final_scales,
            w3_w1_hp,
            self.w3_w1_bias,
            w2_hp,
            self.w2_bias,
            output_dtype,
            quant_scales=[],
            input_sf=None,
            swizzled_input_sf=False,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
            swiglu_limit=self.swiglu_limit,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=self.cluster_size,
            cluster_rank=self.cluster_rank,
            enable_alltoall=enable_alltoall,
            use_deepseek_fp8_block_scale=False,
            use_w4_group_scaling=False,
            use_int8_woq_per_channel=False,
            use_mxfp8_act_scaling=False,
            min_latency_mode=False,
            use_fused_finalize=self.use_fused_finalize,
            tune_max_num_tokens=self.tune_max_num_tokens,
            tuner_num_tokens=tuner_num_tokens,
            tuner_top_k=tuner_top_k,
            activation_type=self.activation_type,
            unpadded_hidden_size=self.unpadded_hidden_size,
            out_tensor=moe_output,
            use_dynamic_fc2_scale=False,
        )
        if moe_output is not None:
            return moe_output
        return result[0]

    def forward_fake(
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
        return super().forward_fake(
            x,
            router_logits,
            do_finalize=do_finalize,
            output_dtype=output_dtype,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
            **kwargs,
        )

    def load_weights(self,
                     weights: List[Dict],
                     allow_partial_loading: bool = False):
        assert self._weights_created
        assert len(weights) == 1
        weights = weights[0]

        kargs = {}
        if "allow_partial_loading" in inspect.getfullargspec(
                self.quant_method.load_weights).args:
            kargs["allow_partial_loading"] = allow_partial_loading
        self.quant_method.load_weights(self, weights, self.weight_loading_mode,
                                       **kargs)
