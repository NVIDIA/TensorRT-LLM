import math
import os
import threading
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Union

import torch
from torch import nn

from tensorrt_llm._mnnvl_utils import MnnvlMoe, MoEAlltoallInfo
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.quantization.utils.fp4_utils import (
    reorder_rows_for_gated_act_gemm, shuffle_matrix_a, shuffle_matrix_sf_a)

from ...quantization.utils.fp4_utils import float4_sf_dtype
from ..distributed import allgather, reducescatter
from ..model_config import ModelConfig
from ..utils import (EventType, Fp4QuantizedTensor, disable_fp4_allgather,
                     reswizzle_sf, swizzle_sf, unswizzle_sf)
from .linear import TensorParallelMode, load_weight_shard

# The declarations aligns with moe_kernels.h
# pack inputs into int64, e.g. 4 x bf16 input values
FUSED_MOE_NVFP4_INPUT_DTYPE = torch.int64
# pack weights into int64, e.g. 16 x nvfp4 weight values
FUSED_MOE_NVFP4_WEIGHT_DTYPE = torch.int64
# pack weight block scales into int32, e.g. 4 x fp8 weight values
FUSED_MOE_NVFP4_WEIGHT_BLOCK_SCALE_DTYPE = torch.int32


class BaseMoeRoutingMethod(nn.Module):

    def apply(self, _router_logits) -> (torch.Tensor, torch.Tensor):
        """
        Applies the routing method to the router logits.
        Router logits are usually the output of the router Linear layer, but can be any type for more complex routing methods.
        Returns (token_selected_experts: torch.Tensor<int32>, token_final_scales: torch.Tensor<float32>):
            token_selected_experts: shape (num_tokens, experts_per_token).
                It is a list of selected expert indices for each token
            token_final_scales: shape (num_tokens, experts_per_token). May be None
                It contains a final scaling/weighting factor applied to the output of each selected expert before summing the results
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_experts_per_token(self):
        return self.top_k

    @property
    def experts_per_token(self):
        return self.get_experts_per_token()


class DefaultMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        topk_values, topk_indices = torch.topk(torch.nn.functional.softmax(
            router_logits.float(), dim=-1),
                                               k=self.top_k,
                                               dim=-1)
        return topk_indices.to(torch.int32), topk_values


class RenormalizeMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        topk_values, topk_indices = torch.topk(router_logits,
                                               k=self.top_k,
                                               dim=-1)
        return topk_indices.to(torch.int32), torch.nn.functional.softmax(
            topk_values.float(), dim=-1)


class Llama4RenormalizeMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        topk_values, topk_indices = torch.topk(router_logits,
                                               k=self.top_k,
                                               dim=-1)
        return topk_indices.to(torch.int32), torch.sigmoid(topk_values.float())


# TODO: re-enable this once the custom op is working.
# class Llama4RenormalizeMoeRoutingMethod(BaseMoeRoutingMethod):

#     def __init__(self, top_k: int, num_experts_total: int, ep_size: int,
#                  ep_rank: int):
#         super().__init__()
#         self.top_k = top_k
#         self.num_experts_total = num_experts_total
#         self.num_experts_per_node = self.num_experts_total // ep_size
#         self.start_expert = self.num_experts_per_node * ep_rank
#         self.end_expert = self.start_expert + self.num_experts_per_node

#     def apply(self,
#               router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
#         unpermuted_scales, indices = torch.ops.trtllm.fused_topk_softmax(
#             router_logits, self.top_k, self.num_experts_total,
#             self.start_expert, self.end_expert)
#         return indices, unpermuted_scales


# TODO Test this for Phi models
class SparseMixerMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(self, top_k: int, eps: float):
        super().__init__()
        self.top_k = top_k
        self.eps = eps

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        router_logits = router_logits.float()
        topk_values = torch.empty(router_logits.shape[0],
                                  self.top_k,
                                  device=router_logits.device,
                                  dtype=torch.float32)
        topk_indices = torch.empty(router_logits.shape[0],
                                   self.top_k,
                                   device=router_logits.device,
                                   dtype=torch.int32)
        for i in range(self.top_k):
            if i > 0:
                max_elem = torch.argmax(router_logits, dim=-1)
                # Mask out the previously selected indices to negative infinity
                router_logits.scatter_(-1, max_elem.unsqueeze(-1),
                                       -float('inf'))
            # Get the max value of the remaining indices
            max_values, max_indices = torch.max(router_logits,
                                                dim=-1,
                                                keepdim=True)
            assert torch.all(max_values != -float('inf'))

            topk_indices[:, i] = max_indices.squeeze(-1)

            # Mask out any values that fail the condition '(max - value) / std::max(abs(value), max) > 2 * epsilon'
            mask = (
                (max_values - router_logits) /
                torch.max(torch.abs(router_logits), max_values)) > 2 * self.eps
            masked_logits = torch.where(mask, -float('inf'), router_logits)
            softmax_masked_logits = torch.nn.functional.softmax(masked_logits,
                                                                dim=-1)
            selected_values = torch.gather(softmax_masked_logits, -1,
                                           max_indices)
            topk_values[:, i] = selected_values.squeeze(-1)

        return topk_indices.to(torch.int32), topk_values


class StaticMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(self,
                 routing_tensor: torch.Tensor,
                 routing_scales: Optional[torch.Tensor] = None):
        super().__init__()
        assert routing_tensor.dtype == torch.int32
        if routing_scales is not None:
            assert routing_tensor.shape[0] == routing_scales.shape[0]
            assert routing_tensor.shape[1] == routing_scales.shape[1]
            assert routing_scales.dtype == torch.float32
        self.routing_tensor = routing_tensor
        self.routing_scales = routing_scales

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        return self.routing_tensor, self.routing_scales

    def get_experts_per_token(self):
        return self.routing_tensor.shape[1]


class LoadBalancedMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        balanced_values = torch.ones(router_logits.shape[0],
                                     self.top_k,
                                     device=router_logits.device,
                                     dtype=torch.float32)
        balanced_indices = torch.empty(router_logits.shape[0],
                                       self.top_k,
                                       device=router_logits.device,
                                       dtype=torch.int32)

        # Fill the balanced_indices with each expert in round-robin fashion
        final_size = router_logits.shape[0] * self.top_k
        repeat_count = math.ceil(final_size / router_logits.shape[1])
        indices = torch.arange(router_logits.shape[1],
                               device=router_logits.device,
                               dtype=torch.int32)
        indices = indices.repeat(repeat_count)
        indices = indices[:final_size]
        balanced_indices = indices.view(router_logits.shape[0],
                                        self.top_k).contiguous()

        return balanced_indices, balanced_values


class MoEWeightLoadingMode(Enum):
    VANILLA = 0
    FUSED_GATE_UP_PROJ = 1


class FusedMoE(nn.Module):
    """
    Fused Mixture of Experts (MoE) Layer with performance tuning.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        aux_stream (Optional[torch.cuda.Stream]): Auxiliary CUDA stream to overlap chunks.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.
        enable_alltoall (bool): whether to enable alltoall instead of allgather/reducescatter

    MoE torch custom op:
        cutlass Backend
            In min-latency mode:
            Quant:
                fp8 block scales (SM90 Hopper only):
                    FusedMoE Op: dynamic quant + gemm1 + swiglu + gemm2 (return tensor list).
                fp8 qdq, nvfp4:
                    FusedMoE Op: gemm1 + swiglu + gemm2 (return tensor list).

            In max-throughput mode:
            Quant:
                fp8 block scales (SM90 Hopper only):
                    FusedMoE Op: dynamic quant + scatter + gemm1 + swiglu + gemm2 + finalizeMoeRoute (return one tensor)
                p8 qdq, nvfp4:
                    FusedMoE Op: scatter + gemm1 + swiglu + gemm2 + finalizeMoeRoute (return one tensor)

        trtllm_gen backend:
            Only support min-latency mode now (SM100 Blackwell only).
            Quant: fp8 block scales quant and nvfp4 quant
                FusedMoE Op: routing(topK, etc.) + scatter + gemm1 + swiglu + gemm2 + finalize MoeRoute

    FusedMoE module:
        cutlass Backend (moe_backend="CUTLASS"):
            min-latency mode:
                routing(topK, etc.) + FusedMoE Op
                equals to: routing(topK, etc.) [+ dynamic quant fp8 qdq | optional dynamic quant nvfp4] + gemm1 + swiglu + gemm2

            max-throughput mode:
                routing(topK, etc.) [+ dynamic quant for fp8 qdq and nvfp4 ] [+ fp4_allgather] + FusedMoe Op[no allreduce] + reducescatter, with AttentionDP on
                equals to: dynamic quant + routing(topK, etc.) [+ fp4_allgather] + scatter + gemm1 + swiglu + gemm2 + finalizeMoeRoute [no allreduce] + reducescatter

        trtllm_gen backend (moe_backend="TRTLLM"):
            min-latency mode (cutlass_min_latency_mode flag of forward has no effect when trtllm_gen is used):
                dynamic quant + FusedMoe Op
                equals to: dynamic quant + routing(topK, etc.) + scatter + gemm1 + swiglu + gemm2 + finalize MoeRoute

    In min-latency mode, setting `reduce_results=False` disables the AllReduce in the FusedMoE module, so any necessary AllReduce operations must be added explicitly in the model definition.
    AttentionDP should be turned off for min-latency mode.
    """

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
        aux_stream: Optional[torch.cuda.Stream] = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
        VANILLA,
        apply_router_weight_on_input: bool = False,
        enable_alltoall: bool = False,
    ):
        from ..distributed import AllReduce

        super().__init__()
        self.routing_method = routing_method
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.weight_loading_mode = weight_loading_mode

        self.dtype = dtype
        self.reduce_results = reduce_results
        # could be modified later
        self.quant_config = model_config.quant_config

        self.cluster_rank = model_config.mapping.moe_cluster_rank
        self.cluster_size = model_config.mapping.moe_cluster_size
        self.smart_router = True if self.cluster_size > 1 else False

        self.rank = model_config.mapping.rank

        self.tp_rank = model_config.mapping.moe_tp_rank
        self.tp_size = model_config.mapping.moe_tp_size

        self.ep_size = model_config.mapping.moe_ep_size
        self.ep_rank = model_config.mapping.moe_ep_rank
        self.moe_backend = model_config.moe_backend
        self.use_dp = model_config.mapping.enable_attention_dp

        # All ranks participate in allreduce regardless of EP/TP combination
        self.mapping = model_config.mapping
        self.parallel_size = self.mapping.tp_size

        self.all_reduce = AllReduce(self.mapping)

        self.intermediate_size_per_partition = intermediate_size // self.tp_size

        self.expert_size_per_partition = num_experts // self.ep_size
        self.expert_start = self.ep_rank * self.expert_size_per_partition
        self.expert_end = min(
            self.expert_start + self.expert_size_per_partition,
            self.num_experts)

        max_num_tokens = model_config.max_num_tokens
        # The maximum number of tokens in MoE are multiplied by DP size when attention DP is enabled
        if self.use_dp:
            max_num_tokens *= model_config.mapping.world_size
        self.moe_max_num_tokens = model_config.moe_max_num_tokens if model_config.moe_max_num_tokens is not None else max_num_tokens
        # The auxiliary CUDA stream and CUDA events are only used when MoE chunking is applied
        if self.moe_max_num_tokens < max_num_tokens:
            self.aux_stream = aux_stream if aux_stream is not None else torch.cuda.Stream(
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
            16384 * num_experts // routing_method.get_experts_per_token(),
        )
        self.has_been_profiled = False
        self.has_been_profiled_min_latency = False

        self.enable_alltoall = enable_alltoall
        self.use_postquant_alltoall = False
        if self.enable_alltoall:
            assert self.use_dp and self.parallel_size > 1,\
                "alltoall should only enabled with attention dp and parallel_size > 1"
            qm = self.quant_config.quant_mode
            self.use_postquant_alltoall = (os.environ.get(
                "TRTLLM_MOE_POST_QUANT_ALLTOALLV", "1")
                                           == "1") and qm.has_nvfp4()
        self.alltoall_workspace = MnnvlMoe.get_moe_workspaces(
            model_config.mapping) if enable_alltoall else None

        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

        # If True, the router weight will be multiplied on the input rather than at the end of FC2
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self._check_configs()

    @property
    def has_any_quant(self):
        return self.quant_config and self.quant_config.quant_mode.has_any_quant(
            exclude_kv_cache=True)

    def _check_configs(self):
        if self.enable_alltoall:
            assert self.use_dp and self.parallel_size > 1,\
                "alltoall should only enabled with attention dp and parallel_size > 1"

        if self.is_trtllm():
            # trtllm_gen backend only support min-latency mode now
            assert not self.reduce_results
            assert self.quant_config and (
                self.quant_config.quant_mode.has_nvfp4()
                | self.quant_config.quant_mode.has_fp8_block_scales()
            ), "The TRTLLM backend of FusedMoE only supports fp8_block_scaling and nvfp4 dtypes."
        else:
            if self.apply_router_weight_on_input:
                assert self.routing_method.top_k == 1, "Current walkaround only supports top-1 routing"
            if self.quant_config and self.quant_config.quant_mode.has_any_quant(
                    exclude_kv_cache=True):
                if not (self.quant_config.quant_mode.has_nvfp4()
                        | self.quant_config.quant_mode.has_fp8_block_scales()
                        | self.quant_config.quant_mode.has_fp8_qdq()
                        | self.quant_config.quant_mode.
                        is_int4_weight_only_per_group()):
                    raise ValueError(
                        f"unsupported quantization mode: {self.quant_config.quant_mode}"
                    )

    def setup_quant_scales(self):
        self.quant_scales = None
        if not self.has_any_quant:
            return
        if self.has_fp8_qdq:
            self.quant_scales = FusedMoEQuantScalesFP8(
                fc1_dequant=self.fc31_dequant,
                fc2_quant=self.fc2_quant,
                fc2_dequant=self.fc2_dequant,
                fc1_input_dequant=self.fc31_input_dequant,
            )
        elif self.has_fp8_block_scales:
            self.quant_scales = FusedMoEQuantScalesFP8BlockScales(
                fc_weight_scales=self.w3_w1_weight_scaling_factor,
                proj_weight_scales=self.w2_weight_scaling_factor,
            )
        elif self.has_nvfp4:
            self.quant_scales = FusedMoEQuantScalesNVFP4(
                fc1_act_global=self.fc31_input_scale,
                fc1_weight_block=self.w3_w1_weight_scale,
                fc1_global=self.fc31_alpha,
                fc2_act_global=self.fc2_input_scale,
                fc2_weight_block=self.w2_weight_scale,
                fc2_global=self.fc2_alpha,
            )
        elif self.has_w4afp8:
            self.quant_scales = FusedMoEQuantScalesW4A8(
                scale_1_interleaved=self.fc31_weight_scale,
                scale_2_interleaved=self.fc2_weight_scale,
                pre_quant_scale_1=self.fc31_act_scale,
                pre_quant_scale_2=self.fc2_act_scale,
                zero_1=torch.Tensor(),
                zero_2=torch.Tensor(),
                alpha_1=self.fc31_alpha,
                alpha_2=self.fc2_alpha,
            )

    def is_trtllm(self):
        return self.moe_backend == "TRTLLM" and self.has_any_quant

    def is_cutlass(self):
        return not self.is_trtllm()

    def get_quant_scales(self, expert_start, expert_end):
        assert self.smart_router

        if self.has_fp8_block_scales:
            return FusedMoEQuantScalesFP8BlockScales(
                fc_weight_scales=self.w3_w1_weight_scaling_factor.narrow(
                    0, expert_start, expert_end - expert_start),
                proj_weight_scales=self.w2_weight_scaling_factor.narrow(
                    0, expert_start, expert_end - expert_start),
            )
        elif self.has_nvfp4:
            return FusedMoEQuantScalesNVFP4(
                fc1_act_global=self.fc31_input_scale,
                fc1_weight_block=self.w3_w1_weight_scale.narrow(
                    0, expert_start, expert_end - expert_start),
                fc1_global=self.fc31_alpha.narrow(0, expert_start,
                                                  expert_end - expert_start),
                fc2_act_global=self.fc2_input_scale,
                fc2_weight_block=self.w2_weight_scale.narrow(
                    0, expert_start, expert_end - expert_start),
                fc2_global=self.fc2_alpha.narrow(0, expert_start,
                                                 expert_end - expert_start),
            )
        elif self.has_w4afp8:
            return FusedMoEQuantScalesW4A8(
                scale_1_interleaved=self.fc31_weight_scale.narrow(
                    0, expert_start, expert_end - expert_start),
                scale_2_interleaved=self.fc2_weight_scale.narrow(
                    0, expert_start, expert_end - expert_start),
                pre_quant_scale_1=self.fc31_act_scale.narrow(
                    0, expert_start, expert_end - expert_start),
                pre_quant_scale_2=self.fc2_act_scale.narrow(
                    0, expert_start, expert_end - expert_start),
                zero_1=torch.Tensor(),
                zero_2=torch.Tensor(),
                alpha_1=self.fc31_alpha.narrow(0, expert_start,
                                               expert_end - expert_start),
                alpha_2=self.fc2_alpha.narrow(0, expert_start,
                                              expert_end - expert_start),
            )
        else:
            return self.quant_scales

    def create_weights(self):
        if self._weights_created:
            return
        weight_dtype = self.dtype
        w3_w1_weight_shape = (self.expert_size_per_partition,
                              self.intermediate_size_per_partition * 2,
                              self.hidden_size)
        w2_weight_shape = (
            self.expert_size_per_partition,
            self.hidden_size,
            self.intermediate_size_per_partition,
        )

        self.quant_scales = []
        self.has_fp8_qdq = False
        self.has_fp8_block_scales = False
        self.has_nvfp4 = False
        self.has_w4afp8 = False
        if self.quant_config and self.quant_config.quant_mode.has_any_quant(
                exclude_kv_cache=True):
            qc = self.quant_config
            if qc.quant_mode.has_fp8_qdq():
                self.has_fp8_qdq = True
                weight_dtype = torch.float8_e4m3fn

                fc31_dequant = nn.Parameter(torch.empty(
                    self.expert_size_per_partition, dtype=torch.float32),
                                            requires_grad=False)
                self.register_parameter("fc31_dequant", fc31_dequant)

                fc2_dequant = nn.Parameter(torch.empty(
                    self.expert_size_per_partition, dtype=torch.float32),
                                           requires_grad=False)
                self.register_parameter("fc2_dequant", fc2_dequant)

                fc2_quant = nn.Parameter(torch.tensor(1., dtype=torch.float32),
                                         requires_grad=False)
                self.register_parameter("fc2_quant", fc2_quant)

                fc31_input_dequant = nn.Parameter(torch.tensor(
                    1., dtype=torch.float32),
                                                  requires_grad=False)
                self.register_parameter("fc31_input_dequant",
                                        fc31_input_dequant)
            elif qc.quant_mode.has_fp8_block_scales():
                self.has_fp8_block_scales = True
                weight_dtype = torch.float8_e4m3fn
                cell_div = lambda x, y: (x + y - 1) // y
                w3_w1_weight_scaling_factor = nn.Parameter(torch.empty(
                    (self.expert_size_per_partition,
                     cell_div(self.intermediate_size_per_partition, 128) * 2,
                     cell_div(w3_w1_weight_shape[2], 128)),
                    dtype=torch.float32),
                                                           requires_grad=False)
                self.register_parameter("w3_w1_weight_scaling_factor",
                                        w3_w1_weight_scaling_factor)

                w2_weight_scaling_factor = nn.Parameter(torch.empty(
                    (self.expert_size_per_partition,
                     cell_div(w2_weight_shape[1],
                              128), cell_div(w2_weight_shape[2], 128)),
                    dtype=torch.float32),
                                                        requires_grad=False)
                self.register_parameter("w2_weight_scaling_factor",
                                        w2_weight_scaling_factor)
            elif qc.quant_mode.is_int4_weight_only_per_group():
                self.has_w4afp8 = True
                self.sm_version = get_sm_version()
                if self.sm_version == 89:
                    self.interleave = [1, 1]
                elif self.sm_version == 90:
                    self.interleave = []
                    for k_shape in [
                            self.hidden_size,
                            self.intermediate_size_per_partition
                    ]:
                        if k_shape % 512 == 0:
                            self.interleave.append(4)
                        elif k_shape % 256 == 0:
                            self.interleave.append(2)
                        elif k_shape % 128 == 0:
                            self.interleave.append(1)
                        else:
                            raise NotImplementedError(
                                f"K shape is required to be multiple of 128, received {k_shape}."
                            )
                else:
                    raise NotImplementedError(
                        f"W4AFP8 MoE is unsupported on SM{self.sm_version}.")
                weight_dtype = torch.int8
                w3_w1_weight_shape = (self.expert_size_per_partition,
                                      self.intermediate_size_per_partition * 2,
                                      self.hidden_size // 2)
                w2_weight_shape = (self.expert_size_per_partition,
                                   self.hidden_size,
                                   self.intermediate_size_per_partition // 2)

                fc31_act_scale = nn.Parameter(torch.empty(
                    self.expert_size_per_partition, 1, dtype=self.dtype),
                                              requires_grad=False)
                self.register_parameter("fc31_act_scale", fc31_act_scale)

                fc2_act_scale = nn.Parameter(torch.empty(
                    self.expert_size_per_partition, 1, dtype=self.dtype),
                                             requires_grad=False)
                self.register_parameter("fc2_act_scale", fc2_act_scale)

                # col parallel
                fc31_weight_scale = nn.Parameter(
                    torch.empty(self.expert_size_per_partition,
                                self.hidden_size // (128 * self.interleave[0]),
                                self.intermediate_size_per_partition * 2 *
                                self.interleave[0],
                                dtype=self.dtype),
                    requires_grad=False)
                self.register_parameter("fc31_weight_scale", fc31_weight_scale)

                # row parallel
                fc2_weight_scale = nn.Parameter(
                    torch.empty(self.expert_size_per_partition,
                                self.intermediate_size_per_partition //
                                (128 * self.interleave[1]),
                                self.hidden_size * self.interleave[1],
                                dtype=self.dtype),
                    requires_grad=False)
                self.register_parameter("fc2_weight_scale", fc2_weight_scale)

                fc31_alpha = nn.Parameter(torch.empty(
                    self.expert_size_per_partition, 1, dtype=torch.float32),
                                          requires_grad=False)
                self.register_parameter("fc31_alpha", fc31_alpha)

                fc2_alpha = nn.Parameter(torch.empty(
                    self.expert_size_per_partition, 1, dtype=torch.float32),
                                         requires_grad=False)
                self.register_parameter("fc2_alpha", fc2_alpha)
            elif qc.quant_mode.has_nvfp4():
                self.has_nvfp4 = True
                if self.is_trtllm():
                    weight_dtype = float4_sf_dtype
                    weight_vec_size = torch.iinfo(weight_dtype).bits // 4
                    block_scales_dtype = torch.float8_e4m3fn
                    block_scales_vec_size = 1
                else:
                    weight_dtype = FUSED_MOE_NVFP4_WEIGHT_DTYPE
                    weight_vec_size = torch.iinfo(weight_dtype).bits // 4
                    block_scales_dtype = FUSED_MOE_NVFP4_WEIGHT_BLOCK_SCALE_DTYPE
                    block_scales_vec_size = torch.iinfo(
                        block_scales_dtype).bits // 8

                self.scaling_vector_size = 16
                # Divide by 16 because we use int64 to pack 16 fp4 values
                w3_w1_weight_shape = (self.expert_size_per_partition,
                                      self.intermediate_size_per_partition * 2,
                                      self.hidden_size // weight_vec_size)
                w2_weight_shape = (self.expert_size_per_partition,
                                   self.hidden_size,
                                   self.intermediate_size_per_partition //
                                   weight_vec_size)

                # Divide by 4 because we use int32 to pack 4 fp8 values
                # column parallel
                w3_w1_weight_scale = nn.Parameter(
                    torch.ones(self.expert_size_per_partition,
                               self.intermediate_size_per_partition * 2,
                               self.hidden_size // self.scaling_vector_size //
                               block_scales_vec_size,
                               dtype=block_scales_dtype),
                    requires_grad=False)
                self.register_parameter("w3_w1_weight_scale",
                                        w3_w1_weight_scale)

                # row parallel
                w2_weight_scale = nn.Parameter(torch.ones(
                    self.expert_size_per_partition,
                    self.hidden_size,
                    self.intermediate_size_per_partition //
                    self.scaling_vector_size // block_scales_vec_size,
                    dtype=block_scales_dtype),
                                               requires_grad=False)
                self.register_parameter("w2_weight_scale", w2_weight_scale)

                fc31_input_scale = nn.Parameter(torch.tensor(
                    1., dtype=torch.float32),
                                                requires_grad=False)
                self.register_parameter("fc31_input_scale", fc31_input_scale)

                fc2_input_scale = nn.Parameter(torch.tensor(
                    1., dtype=torch.float32),
                                               requires_grad=False)
                self.register_parameter("fc2_input_scale", fc2_input_scale)

                fc31_alpha = nn.Parameter(torch.ones(
                    self.expert_size_per_partition, dtype=torch.float32),
                                          requires_grad=False)
                self.register_parameter("fc31_alpha", fc31_alpha)

                fc2_alpha = nn.Parameter(torch.ones(
                    self.expert_size_per_partition, dtype=torch.float32),
                                         requires_grad=False)
                self.register_parameter("fc2_alpha", fc2_alpha)

                if self.is_trtllm():
                    fc31_scale_c = nn.Parameter(torch.ones(
                        self.expert_size_per_partition, dtype=torch.float32),
                                                requires_grad=False)
                    self.register_parameter("fc31_scale_c", fc31_scale_c)

            else:
                # TODO: support other quant mode
                raise ValueError(
                    f"unsupported quantization mode: {qc.quant_mode}")
            self.setup_quant_scales()

        # Fused gate_up_proj (column parallel)
        w3_w1_weight = nn.Parameter(torch.empty(w3_w1_weight_shape,
                                                dtype=weight_dtype),
                                    requires_grad=False)
        self.register_parameter("w3_w1_weight", w3_w1_weight)

        # down_proj (row parallel)
        w2_weight = nn.Parameter(torch.empty(w2_weight_shape,
                                             dtype=weight_dtype),
                                 requires_grad=False)
        self.register_parameter("w2_weight", w2_weight)
        self._weights_created = True

    def reducescatter_or_allreduce(
        self,
        inputs,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
    ):
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

    def forward_chunk(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        cutlass_min_latency_mode: bool = False,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
    ) -> torch.Tensor:
        if isinstance(x, Fp4QuantizedTensor):
            assert output_dtype is not None
            output_dtype = output_dtype
        else:
            output_dtype = x.dtype

        use_fp8_block_scaling = False
        use_w4a8_group_scaling = False
        weight_dtype = self.w3_w1_weight.dtype

        token_selected_experts, token_final_scales = self.routing_method.apply(
            router_logits)

        assert token_selected_experts.shape[
            1] == self.routing_method.experts_per_token
        assert token_selected_experts.shape == token_final_scales.shape
        assert token_selected_experts.shape[0] == router_logits.shape[0]
        assert token_final_scales.dtype == torch.float32
        assert token_selected_experts.dtype == torch.int32

        if self.apply_router_weight_on_input:
            x = x * token_final_scales.to(x.dtype)
            # TODO: remove this once we have correct fusedmoe kernel ready
            token_final_scales = None

        token_count = x.shape[0]

        alltoall_info = None

        if self.enable_alltoall:
            x, token_selected_experts, token_final_scales, alltoall_info = \
                self.alltoall_prepare_maybe_dispatch(all_rank_num_tokens,
                                                     x,
                                                     token_selected_experts,
                                                     token_final_scales)

        x_sf = None
        if self.has_any_quant:
            if self.has_fp8_qdq:
                x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    x, self.fc31_input_dequant)
            elif self.has_nvfp4:
                if not disable_fp4_allgather() or self.use_postquant_alltoall:
                    if isinstance(x, Fp4QuantizedTensor):
                        x, x_sf = x.fp4_tensor, x.scaling_factor
                        x_row = x.shape[0]
                        # note: we use uint8 to store 2 fp4 values
                        x_col = x.shape[1] * 2
                    else:
                        x_row = x.shape[0]
                        x_col = x.shape[1]
                        x, x_sf = torch.ops.trtllm.fp4_quantize(
                            x, self.fc31_input_scale, self.scaling_vector_size,
                            False)

            elif self.has_fp8_block_scales:
                use_fp8_block_scaling = True
            elif self.has_w4afp8:
                use_w4a8_group_scaling = True
                weight_dtype = torch.quint4x2
            else:
                raise ValueError(
                    f"unsupported quantization mode: {self.quant_config.quant_mode}"
                )

        if self.use_dp and self.parallel_size > 1 and not disable_fp4_allgather(
        ) and not self.enable_alltoall:
            if x_sf is None:
                x, token_selected_experts, token_final_scales = allgather(
                    [x, token_selected_experts, token_final_scales],
                    self.mapping,
                    dim=0,
                    sizes=None if use_dp_padding else all_rank_num_tokens)
            else:
                # Fp4 gemm has extra scaling factor
                x, x_sf, token_selected_experts, token_final_scales = allgather(
                    [x, x_sf, token_selected_experts, token_final_scales],
                    self.mapping,
                    dim=0,
                    sizes=None if use_dp_padding else all_rank_num_tokens)
                x_sf = reswizzle_sf(x_sf, x_row, x_col,
                                    self.scaling_vector_size)

        if self.smart_router and not cutlass_min_latency_mode:
            ep_size = self.cluster_size
            ep_rank = self.cluster_rank
            expert_start = ep_rank * self.num_experts // ep_size
            expert_end = min(self.num_experts,
                             (ep_rank + 1) * self.num_experts // ep_size)
            w3_w1_weight = self.w3_w1_weight.narrow(0, expert_start,
                                                    expert_end - expert_start)
            w2_weight = self.w2_weight.narrow(0, expert_start,
                                              expert_end - expert_start)
            cluster_size = self.ep_size
            cluster_rank = self.ep_rank
            quant_scales = self.get_quant_scales(expert_start, expert_end)
        else:
            ep_size = self.ep_size
            ep_rank = self.ep_rank
            w3_w1_weight = self.w3_w1_weight
            w2_weight = self.w2_weight
            cluster_size = self.cluster_size
            cluster_rank = self.cluster_rank
            quant_scales = self.quant_scales

        if self.use_postquant_alltoall:
            x, x_sf = self.alltoall_postquant_dispatch(x, x_sf, x_row, x_col,
                                                       alltoall_info)

        final_hidden_states = torch.ops.trtllm.fused_moe(
            x,
            token_selected_experts,
            token_final_scales,
            w3_w1_weight.view(weight_dtype),
            w2_weight.view(weight_dtype),
            output_dtype,
            quant_scales=quant_scales,
            input_sf=x_sf,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            cluster_size=cluster_size,
            cluster_rank=cluster_rank,
            use_fp8_block_scaling=use_fp8_block_scaling,
            use_w4a8_group_scaling=use_w4a8_group_scaling,
            min_latency_mode=cutlass_min_latency_mode,
        )

        if cutlass_min_latency_mode:
            assert not self.reduce_results
            return final_hidden_states
        else:
            # Custom op requires all inputs are in the same type.
            # Only in cutlass_min_latency_mode, the output is a list of tensors.
            # Otherwise, the output should be unpacked as a single tensor.
            final_hidden_states = final_hidden_states[0]

        if not self.enable_alltoall:
            return final_hidden_states
        else:
            return self.alltoall_combine(final_hidden_states, alltoall_info,
                                         token_count)

    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        cutlass_min_latency_mode: bool = False,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        cutlass_min_latency_mode has no effect when trtllm_gen backend is enabled.
        """
        if self.is_cutlass():
            return self.forward_cutlass(x, router_logits,
                                        cutlass_min_latency_mode, output_dtype,
                                        all_rank_num_tokens, use_dp_padding)
        elif self.is_trtllm():
            return self.forward_trtllmgen(x, router_logits)
        else:
            raise NotImplementedError(
                f"FusedMoE only supports CUTLASS or TRTLLM backends, not {self.moe_backend}"
            )

    def forward_cutlass(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        cutlass_min_latency_mode: bool = False,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
    ) -> torch.Tensor:
        assert self.is_cutlass()

        if self.use_dp:
            assert all_rank_num_tokens is not None
            assert use_dp_padding is not None
            num_rows = sum(all_rank_num_tokens)
        else:
            num_rows = x.shape[0]
        # in case of num_rows is larger than max_chunk_size, we need to split the input into multiple chunks
        num_chunks = (num_rows + self.moe_max_num_tokens -
                      1) // self.moe_max_num_tokens

        if cutlass_min_latency_mode:
            assert num_chunks == 1 and (
                not self.reduce_results
            ), "cutlass_min_latency_mode must be used with a single chunk and reduce_results must be False"

        if use_dp_padding:
            all_rank_num_tokens_padded = [max(all_rank_num_tokens)
                                          ] * len(all_rank_num_tokens)
        else:
            all_rank_num_tokens_padded = all_rank_num_tokens
        if num_chunks == 1:
            outputs = self.forward_chunk(
                x,
                router_logits,
                cutlass_min_latency_mode,
                output_dtype,
                all_rank_num_tokens=all_rank_num_tokens_padded,
                use_dp_padding=use_dp_padding)
            outputs = self.reducescatter_or_allreduce(
                outputs,
                all_rank_num_tokens=all_rank_num_tokens_padded,
                use_dp_padding=use_dp_padding)
        else:

            def split_chunk(split_token_num: int, split_num_chunks: int):
                val_div = split_token_num // split_num_chunks
                val_mod = split_token_num % split_num_chunks
                split_chunk_size_list = [val_div + 1] * val_mod + [val_div] * (
                    split_num_chunks - val_mod)
                return split_chunk_size_list

            if self.use_dp:
                all_rank_chunk_size_list = [
                    split_chunk(val, num_chunks)
                    for val in all_rank_num_tokens_padded
                ]
                all_rank_num_tokens_list = [[
                    val[idx_chunk] for val in all_rank_chunk_size_list
                ] for idx_chunk in range(num_chunks)]
                chunk_size_list = all_rank_chunk_size_list[self.rank]
                if self.enable_alltoall:
                    all_rank_num_tokens_list = [[
                        1 if val == 0 else val for val in val_list
                    ] for val_list in all_rank_num_tokens_list]
            else:
                all_rank_num_tokens_list = [None] * num_chunks
                chunk_size_list = split_chunk(x.shape[0], num_chunks)

            x_list = x.split(chunk_size_list)
            router_logits_list = router_logits.split(chunk_size_list)

            if not self.enable_alltoall:
                self.event_dict[EventType.Main].record()
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.Main].wait()

            outputs_list = []
            # Postpone reduce-scatter/all-reduce to the next iteration to achieve better overlap
            for idx_chunk, (x, router_logits) in enumerate(
                    zip(x_list, router_logits_list)):
                if not self.enable_alltoall:
                    if idx_chunk % 2 == 0:
                        with torch.cuda.stream(self.aux_stream):
                            outputs = self.forward_chunk(
                                x,
                                router_logits,
                                all_rank_num_tokens=all_rank_num_tokens_list[
                                    idx_chunk] if self.use_dp else None,
                                use_dp_padding=use_dp_padding)
                        if idx_chunk > 0:
                            outputs_list[-1] = self.reducescatter_or_allreduce(
                                outputs_list[-1],
                                all_rank_num_tokens=all_rank_num_tokens_list[
                                    idx_chunk - 1],
                                use_dp_padding=use_dp_padding)
                    else:
                        outputs = self.forward_chunk(
                            x,
                            router_logits,
                            all_rank_num_tokens=all_rank_num_tokens_list[
                                idx_chunk] if self.use_dp else None,
                            use_dp_padding=use_dp_padding)
                        with torch.cuda.stream(self.aux_stream):
                            outputs_list[-1] = self.reducescatter_or_allreduce(
                                outputs_list[-1],
                                all_rank_num_tokens=all_rank_num_tokens_list[
                                    idx_chunk - 1],
                                use_dp_padding=use_dp_padding)
                else:
                    outputs = self.forward_chunk(
                        x,
                        router_logits,
                        all_rank_num_tokens=all_rank_num_tokens_list[idx_chunk]
                        if self.use_dp else None)

                outputs_list.append(outputs)
            if not self.enable_alltoall:
                if num_chunks % 2 == 0:
                    outputs_list[-1] = self.reducescatter_or_allreduce(
                        outputs_list[-1],
                        all_rank_num_tokens=all_rank_num_tokens_list[-1],
                        use_dp_padding=use_dp_padding)
                else:
                    with torch.cuda.stream(self.aux_stream):
                        outputs_list[-1] = self.reducescatter_or_allreduce(
                            outputs_list[-1],
                            all_rank_num_tokens=all_rank_num_tokens_list[-1],
                            use_dp_padding=use_dp_padding)
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.MoeChunkingOverlap].record()
                self.event_dict[EventType.MoeChunkingOverlap].wait()
            outputs = torch.cat(outputs_list)
        if self.use_dp:
            rank = self.mapping.tp_rank
            outputs = outputs[:all_rank_num_tokens[rank]]
        return outputs

    def forward_trtllmgen(self, x: torch.Tensor,
                          router_logits: torch.Tensor) -> torch.Tensor:
        assert self.is_trtllm()
        assert x.dtype == torch.bfloat16

        # TODO: since routing kernel is integrated into moe_runner for fp8,
        #       here we just route the I/Os for moe_runner
        routing_bias = self.routing_method.e_score_correction_bias
        top_k = self.routing_method.routing_impl.top_k
        n_group = self.routing_method.routing_impl.n_group
        topk_group = self.routing_method.routing_impl.topk_group
        routed_scaling_factor = self.routing_method.routing_impl.routed_scaling_factor

        if self.quant_config and self.quant_config.quant_mode.has_fp8_block_scales(
        ):
            x_val, x_scale = torch.ops.trtllm.fp8_quantize_1x128(x)

            final_hidden_states = torch.ops.trtllm.fp8_block_scale_moe_runner(
                router_logits,
                routing_bias,
                x_val,
                x_scale,
                self.w3_w1_weight,
                self.w3_w1_weight_scaling_factor,
                self.w2_weight,
                self.w2_weight_scaling_factor,
                self.num_experts,
                top_k,
                n_group,
                topk_group,
                self.intermediate_size_per_partition,
                self.
                expert_start,  # local_expert_start;  use ep_rank if stride!=1
                self.expert_size_per_partition,  # local_expert_size
                routed_scaling_factor,
            )
        elif self.quant_config and self.quant_config.quant_mode.has_nvfp4():
            scale_factor_use_ue8m0 = False
            is_scale_factor_swizzled = False  # use linear layout here
            hidden_states_fp4, hidden_states_scale_linear_fp4 = torch.ops.trtllm.fp4_quantize(
                x, self.fc31_input_scale, 16, scale_factor_use_ue8m0,
                is_scale_factor_swizzled)

            final_hidden_states = torch.ops.trtllm.fp4_block_scale_moe_runner(
                router_logits,
                routing_bias,
                hidden_states_fp4,
                hidden_states_scale_linear_fp4.view(torch.float8_e4m3fn),
                self.w3_w1_weight,
                self.w3_w1_weight_scale.view(torch.float8_e4m3fn),
                self.w2_weight,
                self.w2_weight_scale.view(torch.float8_e4m3fn),
                self.fc31_scale_c.data,
                self.fc31_alpha.data,
                self.fc2_alpha.data,
                self.num_experts,
                top_k,
                n_group,
                topk_group,
                self.intermediate_size_per_partition,
                self.
                expert_start,  # local_expert_start;  use ep_rank if stride!=1
                self.expert_size_per_partition,  # local_expert_size
                routed_scaling_factor,
            )
        else:
            raise NotImplementedError(
                "The TRTLLM backend of FusedMoE only supports fp8_block_scaling and nvfp4 dtypes."
            )

        if self.reduce_results and self.parallel_size > 1:
            final_hidden_states = self.all_reduce(final_hidden_states)

        return final_hidden_states

    def alltoall_prepare_maybe_dispatch(self, all_rank_num_tokens: list,
                                        x: torch.Tensor,
                                        token_selected_experts: torch.Tensor,
                                        token_final_scales: torch.Tensor):
        top_k = self.routing_method.experts_per_token
        expert_count = self.num_experts
        # gather router info
        max_num_token = max(all_rank_num_tokens)
        token_selected_experts = torch.nn.functional.pad(
            token_selected_experts,
            (0, 0, 0, max_num_token - token_selected_experts.shape[0]),
            'constant', self.num_experts)
        token_final_scales = torch.nn.functional.pad(
            token_final_scales,
            (0, 0, 0, max_num_token - token_final_scales.shape[0]))
        gathered_token_selected_experts, gathered_token_final_scales = allgather(
            [token_selected_experts, token_final_scales], self.mapping, dim=0)
        gathered_token_selected_experts = torch.flatten(
            gathered_token_selected_experts.contiguous(),
            start_dim=0,
            end_dim=-2)
        gathered_token_final_scales = torch.flatten(
            gathered_token_final_scales.contiguous(), start_dim=0, end_dim=-2)
        gathered_target_rank_ids = MnnvlMoe.compute_target_rank_id(
            gathered_token_selected_experts, self.num_experts, self.ep_size)
        alltoall_info, token_selected_experts, token_final_scales = MnnvlMoe.mnnvl_moe_alltoallv_prepare(
            gathered_target_rank_ids, None, gathered_token_selected_experts,
            gathered_token_final_scales, max_num_token, expert_count, top_k,
            self.ep_rank, self.ep_size)

        if not self.use_postquant_alltoall:
            assert not isinstance(
                x, Fp4QuantizedTensor
            ), "pre-quant alltoall doesn't support fp4 tensor"
            x = MnnvlMoe.mnnvl_moe_alltoallv(x, alltoall_info,
                                             self.alltoall_workspace,
                                             self.ep_rank, self.ep_size)

        return x, token_selected_experts, token_final_scales, alltoall_info

    def alltoall_postquant_dispatch(self, x: torch.Tensor, x_sf: torch.Tensor,
                                    x_row: int, x_col: int,
                                    alltoall_info: MoEAlltoallInfo):
        x = MnnvlMoe.mnnvl_moe_alltoallv(x, alltoall_info,
                                         self.alltoall_workspace, self.ep_rank,
                                         self.ep_size)

        if x_sf is not None:
            if self.has_nvfp4:
                x_sf = unswizzle_sf(x_sf, x_row, x_col,
                                    self.scaling_vector_size)

            x_sf = MnnvlMoe.mnnvl_moe_alltoallv(x_sf, alltoall_info,
                                                self.alltoall_workspace,
                                                self.ep_rank, self.ep_size)

            if self.has_nvfp4:
                x_sf = swizzle_sf(x_sf, x.shape[0], x.shape[1] * 2,
                                  self.scaling_vector_size)

        return x, x_sf

    def alltoall_combine(self, final_hidden_states: torch.Tensor,
                         alltoall_info: MoEAlltoallInfo, token_count: int):
        top_k = self.routing_method.experts_per_token
        if isinstance(final_hidden_states, list):
            final_hidden_states = final_hidden_states[0]
        final_hidden_states = MnnvlMoe.mnnvl_moe_alltoallv_combine(
            final_hidden_states,
            alltoall_info,
            self.alltoall_workspace,
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
            top_k=top_k,
            token_count=token_count)

        return final_hidden_states

    def load_weights(self, weights: List[Dict]):
        assert self._weights_created
        assert len(weights) == 1
        weights = weights[0]

        def load_expert_w3_w1_weight(w1_weight,
                                     w3_weight,
                                     dst_w3_w1_weight: torch.Tensor,
                                     is_trtllm: bool = False):
            w1_weight_shard = load_weight_shard(w1_weight, self.tp_size,
                                                self.tp_rank,
                                                TensorParallelMode.COLUMN)
            w3_weight_shard = load_weight_shard(w3_weight, self.tp_size,
                                                self.tp_rank,
                                                TensorParallelMode.COLUMN)

            w31_weight_shard = torch.cat([w3_weight_shard, w1_weight_shard],
                                         dim=0)

            if is_trtllm:
                # FIXME: this depends on the kernel internals
                epilogue_tile_m = 128
                w31_weight_shard = reorder_rows_for_gated_act_gemm(
                    w31_weight_shard)
                w31_weight_shard = shuffle_matrix_a(w31_weight_shard,
                                                    epilogue_tile_m)
            if self.has_w4afp8 and self.sm_version == 89:
                import tensorrt_llm.quantization.functional
                preprocessor = tensorrt_llm.quantization.functional.preprocess_weights_for_mixed_gemm
                packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
                unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
                w31_weight_shard = packer(
                    unpacker(w31_weight_shard.cpu()).T.contiguous()).to(
                        w31_weight_shard.device)
                w31_weight_shard = preprocessor(w31_weight_shard,
                                                torch.quint4x2,
                                                torch.float8_e4m3fn,
                                                89).view(dst_w3_w1_weight.shape)

            dst_w3_w1_weight.copy_(w31_weight_shard.view(
                dst_w3_w1_weight.dtype))

        def load_expert_w2_weight(w2_weight,
                                  dst_w2_weight: torch.Tensor,
                                  is_trtllm: bool = False):
            w2_weight_shard = load_weight_shard(w2_weight, self.tp_size,
                                                self.tp_rank,
                                                TensorParallelMode.ROW)
            if is_trtllm:
                # FIXME: this depends on the kernel internals
                epilogue_tile_m = 128
                w2_weight_shard = shuffle_matrix_a(w2_weight_shard,
                                                   epilogue_tile_m)

            if self.has_w4afp8 and self.sm_version == 89:
                import tensorrt_llm.quantization.functional
                preprocessor = tensorrt_llm.quantization.functional.preprocess_weights_for_mixed_gemm
                packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
                unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
                w2_weight_shard = packer(
                    unpacker(w2_weight_shard.cpu()).T.contiguous()).to(
                        w2_weight_shard.device)
                w2_weight_shard = preprocessor(w2_weight_shard, torch.quint4x2,
                                               torch.float8_e4m3fn,
                                               89).view(dst_w2_weight.shape)

            dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype))

        # Use multi-threading to load expert weights in parallel.
        # Even though CPython has global interpreter lock (GIL),
        # it's still faster to load weights in parallel because it can utilize
        # CPU memory bandwidth better.
        threads = []

        for expert_id in range(self.expert_start, self.expert_end):
            expert_idx = expert_id - self.expert_start

            if self.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_weight = weights[f"{expert_id}.w1.weight"]
                w3_weight = weights[f"{expert_id}.w3.weight"]
                w2_weight = weights[f"{expert_id}.w2.weight"]
            elif self.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_w3_weight = weights["gate_up_proj"][expert_id].transpose(
                    0, 1)
                w1_weight, w3_weight = w1_w3_weight.chunk(2, dim=0)
                w2_weight = weights["down_proj"][expert_id].transpose(
                    0, 1).contiguous()
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {self.weight_loading_mode}"
                )

            is_trtllm_nvfp4 = self.is_trtllm(
            ) and self.quant_config.quant_mode.has_nvfp4()

            thread = threading.Thread(target=load_expert_w3_w1_weight,
                                      args=(w1_weight, w3_weight,
                                            self.w3_w1_weight.data[expert_idx],
                                            is_trtllm_nvfp4))
            thread.start()
            threads.append(thread)

            thread = threading.Thread(target=load_expert_w2_weight,
                                      args=(w2_weight,
                                            self.w2_weight.data[expert_idx],
                                            is_trtllm_nvfp4))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        if self.quant_config and self.quant_config.quant_mode.has_any_quant(
                exclude_kv_cache=True):
            if self.quant_config.quant_mode.has_fp8_qdq():
                self._load_fp8_qdq_scales(weights)
            elif self.quant_config.quant_mode.has_nvfp4():
                self._load_nvfp4_scales(weights)
            elif self.quant_config.quant_mode.has_fp8_block_scales():
                self._load_fp8_block_scales_scales(weights)
            elif self.quant_config.quant_mode.is_int4_weight_only_per_group():
                self._load_int4_groupwise_scales(weights)
            else:
                raise ValueError(
                    f"unsupported quantization mode: {self.quant_config.quant_mode}"
                )
            # Re-setup quant scales after loading weights as the tensors may have been modified.
            self.setup_quant_scales()

    def _load_fp8_block_scales_scales(self, weights: Dict):
        all_w2_scales = [
            load_weight_shard(weights[f"{expert_id}.w2.weight_scale_inv"],
                              self.tp_size, self.tp_rank,
                              TensorParallelMode.ROW)
            for expert_id in range(self.expert_start, self.expert_end)
        ]

        w2_scales = torch.stack(all_w2_scales)
        self.w2_weight_scaling_factor.data.copy_(w2_scales)

        all_w3_scales = [
            load_weight_shard(weights[f"{expert_id}.w3.weight_scale_inv"],
                              self.tp_size, self.tp_rank,
                              TensorParallelMode.COLUMN)
            for expert_id in range(self.expert_start, self.expert_end)
        ]

        all_w1_scales = [
            load_weight_shard(weights[f"{expert_id}.w1.weight_scale_inv"],
                              self.tp_size, self.tp_rank,
                              TensorParallelMode.COLUMN)
            for expert_id in range(self.expert_start, self.expert_end)
        ]

        w3_w1_scales = torch.cat(
            [torch.stack(all_w3_scales),
             torch.stack(all_w1_scales)], dim=-2)
        self.w3_w1_weight_scaling_factor.data.copy_(w3_w1_scales)

    def _load_fp8_qdq_scales(self, weights: Dict):
        # Step1: Load input scales.
        def load_expert_fc31_input_scale_fp8_qdq(
                w1_input_scale, w3_input_scale,
                dst_fc31_input_scale: torch.Tensor):
            dst_fc31_input_scale.copy_(
                max(w1_input_scale[...].reshape([]),
                    w3_input_scale[...].reshape([])))

        def load_expert_fc2_input_scale_fp8_qdq(
                w2_input_scale, dst_fc2_input_scale: torch.Tensor):
            dst_fc2_input_scale.copy_(w2_input_scale[...].reshape([]))

        tmp_fc31_input_scale = torch.empty(self.num_experts,
                                           dtype=torch.float32)
        tmp_fc2_input_scale = torch.empty(self.num_experts, dtype=torch.float32)
        for expert_id in range(self.num_experts):
            if self.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_input_scale = weights[f"{expert_id}.w1.input_scale"]
                w3_input_scale = weights[f"{expert_id}.w3.input_scale"]
                w2_input_scale = weights[f"{expert_id}.w2.input_scale"]
            elif self.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_input_scale = weights[f"gate_up_proj_input_scale"]
                w3_input_scale = weights[f"gate_up_proj_input_scale"]
                w2_input_scale = weights[f"down_proj_input_scale"]
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {self.weight_loading_mode}"
                )

            load_expert_fc31_input_scale_fp8_qdq(
                w1_input_scale, w3_input_scale, tmp_fc31_input_scale[expert_id])

            load_expert_fc2_input_scale_fp8_qdq(w2_input_scale,
                                                tmp_fc2_input_scale[expert_id])

        # max_fc31_input_scale is the maximum of all w1 input scales and w3 input scales.
        # It's used to quantize fc31 input inside the MOE op
        max_fc31_input_scale = tmp_fc31_input_scale.max()
        # max_fc2_input_scale is the maximum of all w2 input scales.
        max_fc2_input_scale = tmp_fc2_input_scale.max()

        # Step2: Load weight scales and requantize w3_w1_weight.
        tmp_w3_w1_weight_scale = torch.empty(self.expert_size_per_partition,
                                             dtype=torch.float32)
        tmp_w2_weight_scale = torch.empty(self.expert_size_per_partition,
                                          dtype=torch.float32)

        def load_expert_w3_w1_weight_scale_fp8_qdq(
                w1_weight_scale, w3_weight_scale,
                dst_w3_w1_weight_scale: torch.Tensor):
            w1_weight_scale = w1_weight_scale[...].reshape([])
            w3_weight_scale = w3_weight_scale[...].reshape([])
            dst_w3_w1_weight_scale.copy_(max(w1_weight_scale, w3_weight_scale))

        def requantize_expert_w3_w1_weight_fp8_qdq(
                w1_weight_scale, w3_weight_scale,
                dst_w3_w1_weight: torch.Tensor):
            w1_weight_scale = w1_weight_scale[...].reshape([])
            w3_weight_scale = w3_weight_scale[...].reshape([])
            max_w3_w1_weight_scale = max(w1_weight_scale, w3_weight_scale)

            w3_weight = dst_w3_w1_weight.narrow(
                dim=0, start=0, length=self.intermediate_size_per_partition).to(
                    dtype=self.dtype)
            w1_weight = dst_w3_w1_weight.narrow(
                dim=0,
                start=self.intermediate_size_per_partition,
                length=self.intermediate_size_per_partition).to(
                    dtype=self.dtype)
            dequant_w3_weight = w3_weight * w3_weight_scale
            dequant_w1_weight = w1_weight * w1_weight_scale
            requant_w3_weight = (dequant_w3_weight / max_w3_w1_weight_scale).to(
                torch.float8_e4m3fn)
            requant_w1_weight = (dequant_w1_weight / max_w3_w1_weight_scale).to(
                torch.float8_e4m3fn)

            dst_w3_w1_weight.narrow(
                dim=0, start=0,
                length=self.intermediate_size_per_partition).copy_(
                    requant_w3_weight)
            dst_w3_w1_weight.narrow(
                dim=0,
                start=self.intermediate_size_per_partition,
                length=self.intermediate_size_per_partition).copy_(
                    requant_w1_weight)

        def load_expert_w2_weight_scale_fp8(w2_weight_scale,
                                            dst_w2_weight_scale: torch.Tensor):
            dst_w2_weight_scale.copy_(w2_weight_scale[...].reshape([]))

        for expert_id in range(self.expert_start, self.expert_end):
            if self.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_weight_scale = weights[f"{expert_id}.w1.weight_scale"]
                w3_weight_scale = weights[f"{expert_id}.w3.weight_scale"]
                w2_weight_scale = weights[f"{expert_id}.w2.weight_scale"]
            elif self.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_weight_scale = weights[f"gate_up_proj_weight_scale"]
                w3_weight_scale = weights[f"gate_up_proj_weight_scale"]
                w2_weight_scale = weights[f"down_proj_weight_scale"]
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {self.weight_loading_mode}"
                )

            expert_idx = expert_id - self.expert_start

            load_expert_w3_w1_weight_scale_fp8_qdq(
                w1_weight_scale, w3_weight_scale,
                tmp_w3_w1_weight_scale[expert_idx])

            requantize_expert_w3_w1_weight_fp8_qdq(
                w1_weight_scale, w3_weight_scale,
                self.w3_w1_weight.data[expert_idx])

            load_expert_w2_weight_scale_fp8(w2_weight_scale,
                                            tmp_w2_weight_scale[expert_idx])

        # Step3: calculate and store final loaded weights
        self.fc31_dequant.data.copy_(tmp_w3_w1_weight_scale *
                                     max_fc31_input_scale)
        self.fc2_quant.data.copy_(max_fc2_input_scale.reciprocal())
        self.fc2_dequant.data.copy_(tmp_w2_weight_scale * max_fc2_input_scale)
        self.fc31_input_dequant.data.copy_(max_fc31_input_scale)

    def _load_nvfp4_scales(self, weights: Dict):
        # Step1: Load input scales.
        tmp_fc31_input_scale = torch.empty(self.num_experts,
                                           dtype=torch.float32)
        tmp_fc2_input_scale = torch.empty(self.num_experts, dtype=torch.float32)

        def load_expert_fc31_input_scale_nvfp4(
                w1_input_scale, w3_input_scale,
                dst_fc31_input_scale: torch.Tensor):
            w1_input_scale = w1_input_scale[...].reshape([])
            w3_input_scale = w3_input_scale[...].reshape([])
            assert torch.allclose(
                w1_input_scale,
                w3_input_scale), "w1_input_scale != w3_input_scale"
            dst_fc31_input_scale.copy_(w1_input_scale)

        def load_expert_fc2_input_scale_nvfp4(
                w2_input_scale, dst_fc2_input_scale: torch.Tensor):
            dst_fc2_input_scale.copy_(w2_input_scale[...].reshape([]))

        for expert_id in range(self.num_experts):
            w1_input_scale = weights[f"{expert_id}.w1.input_scale"]
            w3_input_scale = weights[f"{expert_id}.w3.input_scale"]
            w2_input_scale = weights[f"{expert_id}.w2.input_scale"]

            load_expert_fc31_input_scale_nvfp4(w1_input_scale, w3_input_scale,
                                               tmp_fc31_input_scale[expert_id])
            load_expert_fc2_input_scale_nvfp4(w2_input_scale,
                                              tmp_fc2_input_scale[expert_id])

        # fc31_input_scale is the reciprocal of the maximum of all w1 input scales and w3 input scales.
        self.fc31_input_scale.data.copy_(
            tmp_fc31_input_scale.max().reciprocal())
        # fc2_input_scale is the reciprocal of the maximum of all w2 input scales.
        self.fc2_input_scale.data.copy_(tmp_fc2_input_scale.max().reciprocal())

        if self.is_trtllm():
            block_scales_dtype = torch.float8_e4m3fn
        else:
            block_scales_dtype = FUSED_MOE_NVFP4_WEIGHT_BLOCK_SCALE_DTYPE

        # Step2: Load weight block scales and alphas.
        def load_expert_w3_w1_weight_scale_nvfp4(
                w1_weight_scale, w3_weight_scale,
                dst_w3_w1_weight_scale: torch.Tensor, is_trtllm: bool):
            w1_weight_scale = load_weight_shard(w1_weight_scale, self.tp_size,
                                                self.tp_rank,
                                                TensorParallelMode.COLUMN)
            w3_weight_scale = load_weight_shard(w3_weight_scale, self.tp_size,
                                                self.tp_rank,
                                                TensorParallelMode.COLUMN)
            # w3
            dst_w3_weight_scale = dst_w3_w1_weight_scale.narrow(
                dim=0, start=0, length=self.intermediate_size_per_partition)
            dst_w3_weight_scale.copy_(
                w3_weight_scale.view(dst_w3_weight_scale.dtype))

            # w1
            dst_w1_weight_scale = dst_w3_w1_weight_scale.narrow(
                dim=0,
                start=self.intermediate_size_per_partition,
                length=self.intermediate_size_per_partition)
            dst_w1_weight_scale.copy_(
                w1_weight_scale.view(dst_w1_weight_scale.dtype))

            orig_shape = dst_w3_w1_weight_scale.shape

            if is_trtllm:
                # FIXME
                epilogue_tile_m = 128
                dst_w3_w1_weight_scale.copy_(
                    shuffle_matrix_sf_a(
                        reorder_rows_for_gated_act_gemm(
                            dst_w3_w1_weight_scale.view(float4_sf_dtype)),
                        epilogue_tile_m,
                        16).view(block_scales_dtype).reshape(orig_shape))
            else:
                dst_w3_w1_weight_scale.copy_(
                    torch.ops.tensorrt_llm.nvfp4_block_scale_interleave(
                        dst_w3_w1_weight_scale.view(float4_sf_dtype)).view(
                            block_scales_dtype).reshape(orig_shape))

        def load_expert_w2_weight_scale_nvfp4(w2_weight_scale,
                                              dst_w2_weight_scale: torch.Tensor,
                                              is_trtllm: bool):
            w2_weight_scale = load_weight_shard(w2_weight_scale, self.tp_size,
                                                self.tp_rank,
                                                TensorParallelMode.ROW)
            dst_w2_weight_scale.copy_(
                w2_weight_scale.view(dst_w2_weight_scale.dtype))

            orig_shape = dst_w2_weight_scale.shape
            if is_trtllm:
                epilogue_tile_m = 128  # FIXME: read from kernel
                dst_w2_weight_scale.copy_(
                    shuffle_matrix_sf_a(
                        dst_w2_weight_scale.view(float4_sf_dtype),
                        epilogue_tile_m,
                        16).view(block_scales_dtype).reshape(orig_shape))
            else:
                dst_w2_weight_scale.copy_(
                    torch.ops.tensorrt_llm.nvfp4_block_scale_interleave(
                        dst_w2_weight_scale.view(float4_sf_dtype)).view(
                            block_scales_dtype).reshape(orig_shape))

        def load_expert_fc31_alpha_nvfp4(w1_weight_scale_2, w3_weight_scale_2,
                                         final_fc31_input_scale: torch.Tensor,
                                         dst_fc31_alpha: torch.Tensor):
            w1_weight_scale_2 = w1_weight_scale_2[...].reshape([])
            w3_weight_scale_2 = w3_weight_scale_2[...].reshape([])
            assert torch.allclose(
                w1_weight_scale_2,
                w3_weight_scale_2), "w1_weight_scale_2 != w3_weight_scale_2"

            w3_w1_weight_scale_2 = 1.0 / w1_weight_scale_2
            dst_fc31_alpha.copy_(
                1.0 / (final_fc31_input_scale * w3_w1_weight_scale_2))

        def load_expert_fc2_alpha_nvfp4(w2_weight_scale_2,
                                        final_fc2_input_scale: torch.Tensor,
                                        dst_w2_alpha: torch.Tensor):
            w2_weight_scale_2 = 1.0 / w2_weight_scale_2[...].reshape([])
            dst_w2_alpha.copy_(1.0 /
                               (final_fc2_input_scale * w2_weight_scale_2))

        for expert_id in range(self.expert_start, self.expert_end):
            if self.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_weight_scale = weights[f"{expert_id}.w1.weight_scale"]
                w3_weight_scale = weights[f"{expert_id}.w3.weight_scale"]
                w2_weight_scale = weights[f"{expert_id}.w2.weight_scale"]
                w1_weight_scale_2 = weights[f"{expert_id}.w1.weight_scale_2"]
                w3_weight_scale_2 = weights[f"{expert_id}.w3.weight_scale_2"]
                w2_weight_scale_2 = weights[f"{expert_id}.w2.weight_scale_2"]
            elif self.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_w3_weight_scale = weights["gate_up_proj_weight_scale"][
                    expert_id].transpose(0, 1).contiguous()
                w1_weight_scale, w3_weight_scale = w1_w3_weight_scale.chunk(
                    2, dim=0)
                w2_weight_scale = weights["down_proj_weight_scale"][
                    expert_id].transpose(0, 1).contiguous()
                w1_weight_scale_2 = weights["gate_up_proj_weight_scale_2"]
                w3_weight_scale_2 = weights["gate_up_proj_weight_scale_2"]
                w2_weight_scale_2 = weights["down_proj_weight_scale_2"]
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {self.weight_loading_mode}"
                )

            expert_idx = expert_id - self.expert_start

            load_expert_w3_w1_weight_scale_nvfp4(
                w1_weight_scale, w3_weight_scale,
                self.w3_w1_weight_scale.data[expert_idx], self.is_trtllm())
            load_expert_w2_weight_scale_nvfp4(
                w2_weight_scale, self.w2_weight_scale.data[expert_idx],
                self.is_trtllm())

            load_expert_fc31_alpha_nvfp4(w1_weight_scale_2, w3_weight_scale_2,
                                         self.fc31_input_scale.data,
                                         self.fc31_alpha.data[expert_idx])
            load_expert_fc2_alpha_nvfp4(w2_weight_scale_2,
                                        self.fc2_input_scale.data,
                                        self.fc2_alpha.data[expert_idx])
        if self.is_trtllm():
            self.fc31_scale_c.data.copy_(self.fc2_input_scale.data *
                                         self.fc31_alpha.data)

    def _load_int4_groupwise_scales(self, weights: Dict):
        # fc31 scales
        assert (len(self.interleave) == 2)
        all_w3_input_scales = [
            load_weight_shard(weights[f"{expert_id}.w3.input_scale"])
            for expert_id in range(self.expert_start, self.expert_end)
        ]
        all_w1_input_scales = [
            load_weight_shard(weights[f"{expert_id}.w1.input_scale"])
            for expert_id in range(self.expert_start, self.expert_end)
        ]
        all_w3_w1_input_scales = torch.max(torch.stack(all_w3_input_scales),
                                           torch.stack(all_w1_input_scales))
        all_w3_w1_input_scales = torch.ones_like(
            all_w3_w1_input_scales) * all_w3_w1_input_scales.max()
        self.fc31_act_scale.data.copy_(1 / all_w3_w1_input_scales)
        self.fc31_alpha.data.copy_(all_w3_w1_input_scales.float())

        all_w3_scales = [
            load_weight_shard(weights[f"{expert_id}.w3.weight_scale_inv"],
                              self.tp_size, self.tp_rank,
                              TensorParallelMode.COLUMN)
            for expert_id in range(self.expert_start, self.expert_end)
        ]
        all_w1_scales = [
            load_weight_shard(weights[f"{expert_id}.w1.weight_scale_inv"],
                              self.tp_size, self.tp_rank,
                              TensorParallelMode.COLUMN)
            for expert_id in range(self.expert_start, self.expert_end)
        ]
        all_w3_w1_scales = torch.cat(
            [torch.stack(all_w3_scales),
             torch.stack(all_w1_scales)], dim=-2)
        if self.sm_version == 89:
            w3_w1_scales = all_w3_w1_scales.to(torch.float16).view(self.dtype)
        else:
            w3_w1_scales = all_w3_w1_scales.to(torch.bfloat16).view(self.dtype)
        w3_w1_s_shape = w3_w1_scales.shape
        w3_w1_scales_interleaved = w3_w1_scales.reshape(
            w3_w1_s_shape[0], w3_w1_s_shape[1],
            (w3_w1_s_shape[2] // self.interleave[0]), self.interleave[0])
        w3_w1_scales_interleaved = w3_w1_scales_interleaved.permute(0, 2, 1, 3)
        w3_w1_scales_interleaved = w3_w1_scales_interleaved.reshape(
            w3_w1_s_shape[0], w3_w1_s_shape[2] // self.interleave[0],
            w3_w1_s_shape[1] * self.interleave[0])
        self.fc31_weight_scale.data.copy_(w3_w1_scales_interleaved.contiguous())

        # fc2 scales
        all_w2_input_scales = [
            load_weight_shard(weights[f"{expert_id}.w2.input_scale"])
            for expert_id in range(self.expert_start, self.expert_end)
        ]
        all_w2_input_scales = torch.stack(all_w2_input_scales).to(self.dtype)
        all_w2_input_scales = torch.ones_like(
            all_w2_input_scales) * all_w2_input_scales.max()
        self.fc2_act_scale.data.copy_(1 / all_w2_input_scales)
        self.fc2_alpha.data.copy_(all_w2_input_scales.float())

        all_w2_scales = [
            load_weight_shard(weights[f"{expert_id}.w2.weight_scale_inv"],
                              self.tp_size, self.tp_rank,
                              TensorParallelMode.ROW)
            for expert_id in range(self.expert_start, self.expert_end)
        ]
        if self.sm_version == 89:
            w2_scales = torch.stack(all_w2_scales).to(torch.float16).view(
                self.dtype)
        else:
            w2_scales = torch.stack(all_w2_scales).to(torch.bfloat16).view(
                self.dtype)
        w2_s_shape = w2_scales.shape
        w2_scales_interleaved = w2_scales.reshape(
            w2_s_shape[0], w2_s_shape[1], (w2_s_shape[2] // self.interleave[1]),
            self.interleave[1])
        w2_scales_interleaved = w2_scales_interleaved.permute(0, 2, 1, 3)
        w2_scales_interleaved = w2_scales_interleaved.reshape(
            w2_s_shape[0], w2_s_shape[2] // self.interleave[1],
            w2_s_shape[1] * self.interleave[1])
        self.fc2_weight_scale.data.copy_(w2_scales_interleaved.contiguous())


class FusedMoEQuantScalesFP8(NamedTuple):
    fc1_dequant: torch.Tensor
    fc2_quant: torch.Tensor
    fc2_dequant: torch.Tensor
    fc1_input_dequant: torch.Tensor


class FusedMoEQuantScalesNVFP4(NamedTuple):
    fc1_act_global: torch.Tensor
    fc1_weight_block: torch.Tensor
    # fc1_global_scale = 1.0 / (fc1_weight_global_scale * fc1_act_global_scale)
    fc1_global: torch.Tensor

    fc2_act_global: torch.Tensor
    fc2_weight_block: torch.Tensor
    # fc2_global_scale = 1.0 / (fc2_weight_global_scale * fc2_act_global_scale)
    fc2_global: torch.Tensor


class FusedMoEQuantScalesFP8BlockScales(NamedTuple):
    fc_weight_scales: torch.Tensor
    proj_weight_scales: torch.Tensor


class FusedMoEQuantScalesW4A8(NamedTuple):
    scale_1_interleaved: torch.Tensor
    scale_2_interleaved: torch.Tensor
    pre_quant_scale_1: torch.Tensor
    pre_quant_scale_2: torch.Tensor
    zero_1: torch.Tensor
    zero_2: torch.Tensor
    alpha_1: torch.Tensor
    alpha_2: torch.Tensor
