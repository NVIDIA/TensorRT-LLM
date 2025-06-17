import os
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union

import torch

from tensorrt_llm._mnnvl_utils import MnnvlMemory, MnnvlMoe, MoEAlltoallInfo
from tensorrt_llm._utils import logger
from tensorrt_llm.mapping import Mapping

from ...distributed import allgather, reducescatter
from ...expert_statistic import ExpertStatistic
from ...model_config import ModelConfig
from ...utils import (EventType, Fp4QuantizedTensor, disable_fp4_allgather,
                      reswizzle_sf, swizzle_sf, unswizzle_sf)
from .deep_ep_utils import buffer_pool, deep_ep_installed
from .interface import MoE
from .moe_load_balancer import get_moe_load_balancer
from .quantization import (DeepSeekFP8BlockScalesFusedMoEMethod,
                           FP8QDQFusedMoEMethod, MoEWeightLoadingMode,
                           NVFP4CutlassFusedMoEMethod,
                           UnquantizedFusedMoEMethod, WInt4AFP8FusedMoEMethod)
from .routing import BaseMoeRoutingMethod


# The type of alltoall method
class AlltoallMethodType(IntEnum):
    # Not available
    NotEnabled = 0
    # MNNVL
    MNNVL = 1
    # DeepEP intranode or internode: no CUDA Graphs support, IBGDA is required by internode
    DeepEP = 2
    # DeepEP low latency: CUDA Graphs are supported, IBGDA is required
    DeepEPLowLatency = 3


class CutlassFusedMoE(MoE):
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

    MoE torch custom op:
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

    FusedMoE module:
        min-latency mode:
            routing(topK, etc.) + FusedMoE Op
            equals to: routing(topK, etc.) [+ dynamic quant fp8 qdq | optional dynamic quant nvfp4] + gemm1 + swiglu + gemm2

        max-throughput mode:
            routing(topK, etc.) [+ dynamic quant for fp8 qdq and nvfp4 ] [+ fp4_allgather] + FusedMoe Op[no allreduce] + reducescatter, with AttentionDP on
            equals to: dynamic quant + routing(topK, etc.) [+ fp4_allgather] + scatter + gemm1 + swiglu + gemm2 + finalizeMoeRoute [no allreduce] + reducescatter

    In min-latency mode, setting `reduce_results=False` disables the AllReduce in the FusedMoE module, so any necessary AllReduce operations must be added explicitly in the model definition.
    AttentionDP should be turned off for min-latency mode.

    Large-scale EP:
    When we have redundant expert, we have more weight slots than `num_experts`, in that case, we separate the concepts of expert and slot.
    Expert is the concept from model's perspective while slot is the concept from model engine's perspective.
    There should be at least `num_experts` slots in the model engine. More than that is OK, in that case, some experts may have multiple replicas.
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
        layer_idx: Optional[int] = None,
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
        )

        self.layer_idx = layer_idx

        moe_load_balancer = get_moe_load_balancer()
        self.layer_load_balancer = None

        moe_load_balancer_config = model_config.moe_load_balancer
        init_expert_size_per_partition = moe_load_balancer_config.num_local_slots if moe_load_balancer_config else self.num_experts // self.ep_size
        self.initial_global_assignments = [
            (ep_rank * self.num_experts // self.ep_size + local_slot_id) %
            self.num_experts for ep_rank in range(self.ep_size)
            for local_slot_id in range(init_expert_size_per_partition)
        ]

        if moe_load_balancer:
            assert moe_load_balancer_config is not None
            top_k = self.routing_method.experts_per_token
            self.expert_size_per_partition = moe_load_balancer_config.num_local_slots
            self.layer_load_balancer = moe_load_balancer.add_layer(
                self.num_experts, top_k, self.expert_size_per_partition)
            loaded_initial_global_assignments = moe_load_balancer_config.get_layer_initial_global_assignments(
                self.layer_idx)
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
            logger.info(
                f"MoE load balancer enabled. num_experts = {num_experts}, num_slots = {self.num_slots}, ep_size = {self.ep_size}"
            )
            logger.info(
                f"initial_global_assignments (layer {self.layer_idx}) = {self.initial_global_assignments}"
            )
        else:
            assert num_experts % self.ep_size == 0
            self.expert_size_per_partition = num_experts // self.ep_size
            self.num_slots = num_experts

        if self.smart_router:
            assert self.num_slots == self.num_experts, "Smart router should not have redundant slots"

        self.slot_start = self.ep_rank * self.expert_size_per_partition
        self.slot_end = self.slot_start + self.expert_size_per_partition
        self.initial_local_expert_ids = self.initial_global_assignments[
            self.slot_start:self.slot_end]
        assert len(
            self.initial_local_expert_ids) == self.expert_size_per_partition

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
            16384 * self.num_slots // routing_method.get_experts_per_token(),
        )
        self.has_been_profiled = False
        self.has_been_profiled_min_latency = False

        self.alltoall_method_type = self.select_alltoall_method_type(
            model_config.mapping, routing_method.experts_per_token, dtype,
            model_config.use_cuda_graph)
        logger.info_once(
            f"CutlassFusedMoE selects alltoall_method_type {self.alltoall_method_type!r}",
            key="alltoall_method_type")
        self.use_postquant_alltoall = False
        if self.enable_alltoall:
            assert self.use_dp and self.parallel_size > 1,\
                "alltoall should only enabled with attention dp and parallel_size > 1"
            qm = self.quant_config.quant_mode
            self.use_postquant_alltoall = (os.environ.get(
                "TRTLLM_MOE_POST_QUANT_ALLTOALLV", "1")
                                           == "1") and qm.has_nvfp4()
            self.enable_alltoall_without_allgather = os.environ.get(
                "TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER", "0") == "1"
            if self.alltoall_method_type == AlltoallMethodType.MNNVL:
                MnnvlMemory.initialize()
                self.alltoall_workspace = MnnvlMoe.get_moe_workspaces(
                    model_config.mapping)
                if self.enable_alltoall_without_allgather:
                    self.alltoall_prepare_workspace = MnnvlMoe.get_moe_prepare_workspace(
                        model_config.mapping)
            elif self.alltoall_method_type == AlltoallMethodType.DeepEP:
                self.deep_ep_buffer = buffer_pool.get_buffer(
                    model_config.mapping)
                self.deep_ep_buffer.reserve(hidden_size, dtype)
            elif self.alltoall_method_type == AlltoallMethodType.DeepEPLowLatency:
                self.deep_ep_max_num_tokens = min(model_config.max_num_tokens,
                                                  self.moe_max_num_tokens)
                self.deep_ep_buffer = buffer_pool.get_low_latency_buffer(
                    model_config.mapping)
                self.deep_ep_buffer.reserve(self.deep_ep_max_num_tokens,
                                            hidden_size, self.num_slots)
            else:
                raise NotImplementedError(
                    f"Not available alltoall method type: {alltoall_method_type!r}"
                )

        # If True, the router weight will be multiplied on the input rather than at the end of FC2
        self.apply_router_weight_on_input = apply_router_weight_on_input

        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def _check_configs(self):
        assert self._weights_created

        if self.enable_alltoall:
            assert self.use_dp and self.parallel_size > 1,\
                "alltoall should only enabled with attention dp and parallel_size > 1"

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

    @staticmethod
    def select_alltoall_method_type(mapping: Mapping, top_k: int,
                                    dtype: torch.dtype,
                                    use_cuda_graph: bool) -> AlltoallMethodType:
        if not mapping.enable_attention_dp:
            return AlltoallMethodType.NotEnabled

        if mapping.tp_size == 1:
            return AlltoallMethodType.NotEnabled

        if os.environ.get("TRTLLM_MOE_DISABLE_ALLTOALLV", "0") == "1":
            return AlltoallMethodType.NotEnabled

        if mapping.moe_ep_size <= top_k:
            return AlltoallMethodType.NotEnabled

        if MnnvlMemory.supports_mnnvl():
            return AlltoallMethodType.MNNVL

        if os.environ.get("TRTLLM_CAN_USE_DEEP_EP", "0") == "1":
            if deep_ep_installed and dtype == torch.bfloat16:
                if use_cuda_graph:
                    # Here we can only choose DeepEPLowLatency since only this method supports CUDA Graphs.
                    return AlltoallMethodType.DeepEPLowLatency
                else:
                    # Here we can choose DeepEP or DeepEPLowLatency if both are available. Now DeepEP is faster.
                    return AlltoallMethodType.DeepEP

        return AlltoallMethodType.NotEnabled

    @property
    def has_w4afp8(self):
        assert self._weights_created
        return self.quant_config and self.quant_config.quant_mode.is_int4_weight_only_per_group(
        )

    @property
    def enable_alltoall(self):
        """ enable_alltoall (bool): whether to enable alltoall instead of allgather/reducescatter
        """
        return self.alltoall_method_type != AlltoallMethodType.NotEnabled

    def _get_quant_method(self):
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True):
            if self.quant_config.layer_quant_mode.has_fp8_qdq():
                return FP8QDQFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_fp8_block_scales():
                return DeepSeekFP8BlockScalesFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_nvfp4():
                return NVFP4CutlassFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.is_int4_weight_only_per_group(
            ):
                return WInt4AFP8FusedMoEMethod()
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
            repeating_info: Tuple = (True, True),
    ) -> torch.Tensor:
        if isinstance(x, Fp4QuantizedTensor):
            assert output_dtype is not None
            output_dtype = output_dtype
        else:
            output_dtype = x.dtype

        is_first_call, is_last_call = repeating_info

        if self.layer_load_balancer and not self.layer_load_balancer.is_static_routing(
        ) and is_first_call:
            self.layer_load_balancer.wait_for_gpu_stage()

        use_deepseek_fp8_block_scale = False
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
            assert self.routing_method.top_k == 1, "Current workaround only supports top-1 routing"
            assert x.dtype != torch.float8_e4m3fn, "Current workaround for apply_router_weight_on_input does not support fp8 input"
            x = x * token_final_scales.to(x.dtype)
            # TODO: remove this once we have correct fusedmoe kernel ready
            token_final_scales = None

        if self.layer_load_balancer and not self.layer_load_balancer.is_static_routing(
        ) and is_first_call:
            self.layer_load_balancer.maybe_cudagraph_done_wait()

        need_statistic = False
        if self.layer_load_balancer is None:
            token_selected_slots = token_selected_experts
        else:
            token_selected_slots = self.layer_load_balancer.route(
                token_selected_experts, self.use_dp)
            if not self.layer_load_balancer.is_static_routing():
                need_statistic = True

        # If load balancer is disabled, the statistics are collected from expert IDs.
        # If load balancer is enabled, the statistics are collected from expert slot IDs.
        ExpertStatistic.set_layer(self.layer_idx)
        ExpertStatistic.maybe_add_info(self.num_slots, token_selected_slots)

        token_selected_experts_for_statistic = token_selected_experts if need_statistic else None

        if self.enable_alltoall:
            if self.alltoall_method_type == AlltoallMethodType.MNNVL:
                token_count = x.shape[0]
                alltoall_info = None
                x, token_selected_slots, token_final_scales, token_selected_experts_for_statistic, alltoall_info = \
                    self.alltoall_prepare_maybe_dispatch(all_rank_num_tokens,
                                                         x,
                                                         token_selected_slots,
                                                         token_final_scales,
                                                         token_selected_experts_for_statistic)
            elif self.alltoall_method_type == AlltoallMethodType.DeepEP:
                if not self.use_postquant_alltoall:
                    x, recv_topk_idx, token_final_scales, num_recv_tokens_per_expert_list, deep_ep_handle = \
                        self.deep_ep_buffer.dispatch(x, token_selected_slots.to(torch.int64), token_final_scales, self.num_slots)
            elif self.alltoall_method_type == AlltoallMethodType.DeepEPLowLatency:
                if not self.use_postquant_alltoall:
                    deep_ep_topk_idx = token_selected_slots.to(torch.int64)
                    deep_ep_topk_weights = token_final_scales
                    x, recv_expert_count, deep_ep_handle = \
                        self.deep_ep_buffer.low_latency_dispatch(x, deep_ep_topk_idx, self.deep_ep_max_num_tokens, self.num_slots)
                    # x shape: [#local experts, #max recv tokens, hidden_size]
                    # recv_expert_count shape: [#local experts]

                    # Adapter between `torch.ops.trtllm.fused_moe` and DeepEP
                    # TODO: remove the adapter by changing `torch.ops.trtllm.fused_moe` API
                    mask = torch.arange(
                        x.shape[1], dtype=torch.int32, device=x.device).expand(
                            x.shape[0],
                            x.shape[1]) < recv_expert_count.unsqueeze(1)
                    token_selected_slots = torch.full(
                        (x.shape[0], x.shape[1], self.routing_method.top_k),
                        self.num_slots,
                        dtype=torch.int32,
                        device=x.device)
                    token_selected_slots[:, :, 0] = torch.where(
                        mask,
                        torch.arange(
                            x.shape[0] * self.mapping.moe_ep_rank,
                            x.shape[0] * (self.mapping.moe_ep_rank + 1),
                            dtype=torch.int32,
                            device=x.device).unsqueeze(1), self.num_slots)
                    x = x.view(x.shape[0] * x.shape[1], x.shape[2])
                    token_selected_slots = token_selected_slots.view(
                        x.shape[0], self.routing_method.top_k)
                    token_final_scales = torch.ones_like(
                        token_selected_slots, dtype=token_final_scales.dtype)
            else:
                raise NotImplementedError(
                    f"Not available alltoall method type: {alltoall_method_type!r}"
                )

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

            elif self.has_deepseek_fp8_block_scales:
                use_deepseek_fp8_block_scale = True
            elif self.has_w4afp8:
                use_w4a8_group_scaling = True
                weight_dtype = torch.quint4x2
            else:
                raise ValueError(
                    f"unsupported quantization mode: {self.quant_config.quant_mode}"
                )

        if self.use_dp and self.parallel_size > 1 and not disable_fp4_allgather(
        ) and not self.enable_alltoall:
            x, x_sf, token_selected_slots, token_final_scales, token_selected_experts_for_statistic = allgather(
                [
                    x, x_sf, token_selected_slots, token_final_scales,
                    token_selected_experts_for_statistic
                ],
                self.mapping,
                dim=0,
                sizes=None if use_dp_padding else all_rank_num_tokens)
            # Fp4 gemm has extra scaling factor
            if x_sf is not None:
                x_sf = reswizzle_sf(x_sf, x_row, x_col,
                                    self.scaling_vector_size)

        if self.layer_load_balancer and not self.layer_load_balancer.is_static_routing(
        ):
            self.layer_load_balancer.statistic(
                token_selected_experts_for_statistic, is_first_call,
                is_last_call)

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
            if self.alltoall_method_type == AlltoallMethodType.MNNVL:
                x, x_sf = self.alltoall_postquant_dispatch(
                    x, x_sf, x_row, x_col, alltoall_info)
            elif self.alltoall_method_type == AlltoallMethodType.DeepEP:
                if x_sf is not None:
                    if self.has_nvfp4:
                        x_sf = unswizzle_sf(x_sf, x_row, x_col,
                                            self.scaling_vector_size)
                    # Adapter between `x_sf` and DeepEP
                    # TODO: remove the adapter by adding dtype support to DeepEP
                    x_sf_dtype = x_sf.dtype
                    x_sf = x_sf.view(torch.float32)
                (x, x_sf), recv_topk_idx, token_final_scales, num_recv_tokens_per_expert_list, deep_ep_handle = \
                    self.deep_ep_buffer.dispatch((x, x_sf), token_selected_slots.to(torch.int64), token_final_scales, self.num_slots)
                if x_sf is not None:
                    x_sf = x_sf.view(x_sf_dtype)
                    if self.has_nvfp4:
                        x_sf = swizzle_sf(x_sf, x.shape[0], x.shape[1] * 2,
                                          self.scaling_vector_size)
            elif self.alltoall_method_type == AlltoallMethodType.DeepEPLowLatency:
                raise NotImplementedError(
                    "Not implemented postquant for DeepEPLowLatency, please set TRTLLM_MOE_POST_QUANT_ALLTOALLV=0"
                )
            else:
                raise NotImplementedError(
                    f"Not available alltoall method type: {alltoall_method_type!r}"
                )

        if self.enable_alltoall:
            # Adapter between `torch.ops.trtllm.fused_moe` and DeepEP
            # TODO: remove the adapter by changing APIs
            if self.alltoall_method_type == AlltoallMethodType.DeepEP:
                token_selected_slots = recv_topk_idx.to(torch.int32)
                mask = token_selected_slots == -1
                token_selected_slots += self.expert_size_per_partition * self.mapping.moe_ep_rank
                token_selected_slots[mask] = self.num_slots
                num_recv_token_is_zero = x.shape[0] == 0
                if x.shape[0] == 0:
                    x = torch.zeros((1, x.shape[1]),
                                    dtype=x.dtype,
                                    device=x.device)
                    token_selected_slots = torch.full(
                        (1, token_selected_slots.shape[1]),
                        self.num_slots,
                        dtype=token_selected_slots.dtype,
                        device=token_selected_slots.device)
                    token_final_scales = torch.ones(
                        (1, token_final_scales.shape[1]),
                        dtype=token_final_scales.dtype,
                        device=token_final_scales.device)

        final_hidden_states = torch.ops.trtllm.fused_moe(
            x,
            token_selected_slots,
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
            enable_alltoall=self.enable_alltoall,
            use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
            use_w4a8_group_scaling=use_w4a8_group_scaling,
            min_latency_mode=cutlass_min_latency_mode,
            tune_max_num_tokens=self.tune_max_num_tokens,
        )

        if self.layer_load_balancer and not self.layer_load_balancer.is_static_routing(
        ) and is_last_call:
            self.layer_load_balancer.set_cpu_stage()

        if cutlass_min_latency_mode:
            assert not self.reduce_results
            assert not self.enable_alltoall
        else:
            # Custom op requires all inputs are in the same type.
            # Only in cutlass_min_latency_mode, the output is a list of tensors.
            # Otherwise, the output should be unpacked as a single tensor.
            final_hidden_states = final_hidden_states[0]

        if self.enable_alltoall:
            if self.alltoall_method_type == AlltoallMethodType.MNNVL:
                final_hidden_states = self.alltoall_combine(
                    final_hidden_states, alltoall_info, token_count)
            elif self.alltoall_method_type == AlltoallMethodType.DeepEP:
                if num_recv_token_is_zero:
                    final_hidden_states = final_hidden_states[:0]
                final_hidden_states = self.deep_ep_buffer.combine(
                    final_hidden_states, deep_ep_handle)
            elif self.alltoall_method_type == AlltoallMethodType.DeepEPLowLatency:
                final_hidden_states = self.deep_ep_buffer.low_latency_combine(
                    final_hidden_states.view(
                        self.expert_size_per_partition,
                        self.deep_ep_max_num_tokens * self.mapping.moe_ep_size,
                        final_hidden_states.shape[1]), deep_ep_topk_idx,
                    deep_ep_topk_weights, deep_ep_handle)
            else:
                raise NotImplementedError(
                    f"Not available alltoall method type: {alltoall_method_type!r}"
                )

        if self.layer_load_balancer and not self.layer_load_balancer.is_static_routing(
        ) and is_last_call:
            self.layer_load_balancer.maybe_cudagraph_done_set_cpu_stage()

        return final_hidden_states

    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        do_finalize: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
    ) -> torch.Tensor:
        if self.use_dp:
            assert all_rank_num_tokens is not None
            assert use_dp_padding is not None
            num_rows = sum(all_rank_num_tokens)
        else:
            num_rows = x.shape[0]

        # in case of num_rows is larger than max_chunk_size, we need to split the input into multiple chunks
        num_chunks = (num_rows + self.moe_max_num_tokens -
                      1) // self.moe_max_num_tokens
        # TODO: remove cutlass_min_latency_mode since it is not used anymore
        cutlass_min_latency_mode = not do_finalize

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
                is_first_call = idx_chunk == 0
                is_last_call = idx_chunk == num_chunks - 1
                if not self.enable_alltoall:
                    if idx_chunk % 2 == 0:
                        with torch.cuda.stream(self.aux_stream):
                            outputs = self.forward_chunk(
                                x,
                                router_logits,
                                all_rank_num_tokens=all_rank_num_tokens_list[
                                    idx_chunk] if self.use_dp else None,
                                use_dp_padding=use_dp_padding,
                                repeating_info=(is_first_call, is_last_call))
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
                            use_dp_padding=use_dp_padding,
                            repeating_info=(is_first_call, is_last_call))
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
                        if self.use_dp else None,
                        repeating_info=(is_first_call, is_last_call))

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

    def alltoall_prepare_maybe_dispatch(
            self, all_rank_num_tokens: list, x: torch.Tensor,
            token_selected_slots: torch.Tensor,
            token_final_scales: torch.Tensor,
            token_selected_experts_for_statistic: Optional[torch.Tensor]):
        top_k = self.routing_method.experts_per_token
        max_num_token = max(all_rank_num_tokens)

        if self.enable_alltoall_without_allgather:
            if token_selected_experts_for_statistic is not None:
                token_selected_experts_for_statistic = torch.nn.functional.pad(
                    token_selected_experts_for_statistic,
                    (0, 0, 0, max_num_token -
                     token_selected_experts_for_statistic.shape[0]), 'constant',
                    self.num_experts)

                gathered_token_selected_experts_for_statistic = allgather(
                    token_selected_experts_for_statistic)

                gathered_token_selected_experts_for_statistic = torch.flatten(
                    gathered_token_selected_experts_for_statistic.contiguous(),
                    start_dim=0,
                    end_dim=-2)
            else:
                gathered_token_selected_experts_for_statistic = None

            alltoall_info, token_selected_slots, token_final_scales = MnnvlMoe.mnnvl_moe_alltoallv_prepare_without_allgather(
                token_selected_slots, token_final_scales,
                self.alltoall_prepare_workspace, max_num_token, self.ep_rank,
                self.ep_size, self.num_slots, top_k)
        else:
            token_selected_slots = torch.nn.functional.pad(
                token_selected_slots,
                (0, 0, 0, max_num_token - token_selected_slots.shape[0]),
                'constant', self.num_slots)
            token_selected_experts_for_statistic = torch.nn.functional.pad(
                token_selected_experts_for_statistic,
                (0, 0, 0,
                 max_num_token - token_selected_experts_for_statistic.shape[0]),
                'constant', self.num_experts
            ) if token_selected_experts_for_statistic is not None else None
            token_final_scales = torch.nn.functional.pad(
                token_final_scales,
                (0, 0, 0, max_num_token - token_final_scales.shape[0]))
            gathered_token_selected_slots, gathered_token_final_scales, gathered_token_selected_experts_for_statistic = allgather(
                [
                    token_selected_slots, token_final_scales,
                    token_selected_experts_for_statistic
                ],
                self.mapping,
                dim=0)
            if gathered_token_selected_experts_for_statistic is not None:
                gathered_token_selected_experts_for_statistic = torch.flatten(
                    gathered_token_selected_experts_for_statistic.contiguous(),
                    start_dim=0,
                    end_dim=-2)
            gathered_token_selected_slots = torch.flatten(
                gathered_token_selected_slots.contiguous(),
                start_dim=0,
                end_dim=-2)
            gathered_token_final_scales = torch.flatten(
                gathered_token_final_scales.contiguous(),
                start_dim=0,
                end_dim=-2)
            gathered_target_rank_ids = MnnvlMoe.compute_target_rank_id(
                gathered_token_selected_slots, self.num_slots, self.ep_size)
            alltoall_info, token_selected_slots, token_final_scales = MnnvlMoe.mnnvl_moe_alltoallv_prepare(
                gathered_target_rank_ids, None, gathered_token_selected_slots,
                gathered_token_final_scales, max_num_token, self.num_slots,
                top_k, self.ep_rank, self.ep_size)

        if not self.use_postquant_alltoall:
            assert not isinstance(
                x, Fp4QuantizedTensor
            ), "pre-quant alltoall doesn't support fp4 tensor"
            x = MnnvlMoe.mnnvl_moe_alltoallv(x, alltoall_info,
                                             self.alltoall_workspace,
                                             self.ep_rank, self.ep_size)

        return x, token_selected_slots, token_final_scales, gathered_token_selected_experts_for_statistic, alltoall_info

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

    def register_parameter_weight_slot_fn(self, weight_name: str,
                                          local_slot_id: int):
        assert hasattr(
            self,
            weight_name), f"FusedMoE doesn't has weight attr: {weight_name}"
        weight_tensor = getattr(self, weight_name).data[local_slot_id]
        self.layer_load_balancer.register_weight_slot(local_slot_id,
                                                      weight_name,
                                                      weight_tensor)

    def register_to_fix_weight_fn(self, weight_name: str):
        assert hasattr(
            self,
            weight_name), f"FusedMoE doesn't has weight attr: {weight_name}"
        param = getattr(self, weight_name)
        weight_tensor = param.detach()
        assert isinstance(
            weight_tensor,
            torch.Tensor), f'weight {weight_name} should be a tensor'
        assert weight_tensor.is_contiguous(
        ), f'weight {weight_name} should be a is_contiguous, shape={weight_tensor.shape}, strides={weight_tensor.is_contiguous()}'
        assert weight_tensor.numel() * weight_tensor.element_size() == weight_tensor.untyped_storage().size(),\
            f'weight {weight_name} shape={weight_tensor.shape} storage_size = {weight_tensor.untyped_storage().size()}, numel={weight_tensor.numel()}, eltsize={weight_tensor.element_size()}, dtype={weight_tensor.dtype}'
        self.layer_load_balancer.fix_tensor(weight_tensor)
        param.data = weight_tensor

    def register_all_parameter_slot_and_to_fix_weight_fns(
            self, weight_and_tensor_dict: Dict[str, torch.Tensor]):
        """
        weight_and_tensor_dict: key is the name of the weight, value is the tensor of loaded shared tensor shard.
            E.g. if num_experts=256 and 4 GPUs per node, then each rank need to load 256 / 4 = 64 expert weights for host sharing.
            By this way, host_tensor_sharer can share the weights and each rank has access to all 256 experts.
        """
        for local_slot_id, expert_id in enumerate(
                self.initial_local_expert_ids):
            for weight_name in weight_and_tensor_dict:
                self.layer_load_balancer.add_register_weight_fn(
                    self.register_parameter_weight_slot_fn,
                    (weight_name, local_slot_id))
        for weight_name in weight_and_tensor_dict:
            self.layer_load_balancer.add_to_fix_weight_fn(
                self.register_to_fix_weight_fn, (weight_name, ))

        local_shared_load_expert_ids = self.layer_load_balancer.get_load_expert_ids(
        )
        for expert_id in range(self.num_experts):
            for weight_name, weight_tensor in weight_and_tensor_dict.items():
                if expert_id in local_shared_load_expert_ids:
                    local_slot_id = local_shared_load_expert_ids.index(
                        expert_id)
                    self.layer_load_balancer.host_tensor_sharer.share_host_tensor_with_shape(
                        expert_id, weight_name, weight_tensor[local_slot_id])
                else:
                    self.layer_load_balancer.host_tensor_sharer.pre_register_host_tensor_with_shape(
                        expert_id, weight_name, weight_tensor.dtype,
                        weight_tensor[0].shape)

    def load_weights(self, weights: List[Dict]):
        assert self._weights_created
        assert len(weights) == 1
        weights = weights[0]

        self.quant_method.load_weights(self, weights, self.weight_loading_mode)
