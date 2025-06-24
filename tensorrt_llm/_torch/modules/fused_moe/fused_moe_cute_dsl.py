import math
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from ...distributed import allgather
from ...model_config import ModelConfig
from ...utils import Fp4QuantizedTensor, disable_fp4_allgather, reswizzle_sf
from .fused_moe_cutlass import CutlassFusedMoE
from .quantization import MoEWeightLoadingMode
from .routing import BaseMoeRoutingMethod


def swiglu_fused_moe(x):
    x, gate = x.chunk(2, dim=-1)
    return F.silu(gate) * x


def cute_dsl_fp8_group_blockwise_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    offset_array: torch.Tensor,
) -> torch.Tensor:
    m, k = a.shape[0], a.shape[1]
    l, n, k = b.shape[0], b.shape[1], b.shape[2]
    num_group, w_n, w_k = b_sf.shape[0], b_sf.shape[1], b_sf.shape[2]

    # Note: view(int8) will cause error.
    a_tmp = a.as_strided((m, k, 1), (k, 1, m * k))
    b_tmp = b.permute(1, 2, 0)

    m_padded = (m + 3) // 4 * 4
    input_scale_tmp = a_sf[0:m_padded * w_k]
    input_scale_tmp = input_scale_tmp.reshape(-1, m_padded)
    input_scale_tmp = input_scale_tmp[:w_k, :m].contiguous().permute(1, 0)
    input_scale_tmp = input_scale_tmp.as_strided((m, w_k, 1), (1, m, m * w_k))

    weight_scale_tmp = b_sf.permute(1, 2, 0)

    def pad_and_multiply(scale, tensor):
        cm, ck, _ = scale.shape
        m, k, _ = tensor.shape
        IsGroupWise = False
        IsBlockWise = False
        if ck == math.ceil(k / 128):
            IsGroupWise = True
        if cm == math.ceil(m / 128):
            IsBlockWise = True
        if not IsBlockWise and not IsGroupWise:
            raise ValueError("Only support granularity = 128")

        k_idx = torch.arange(k, device=scale.device)
        if IsGroupWise:
            k_idx = k_idx // 128
        m_idx = torch.arange(m, device=scale.device)
        if IsBlockWise:
            m_idx = m_idx // 128
        expanded_scale = scale[m_idx[:, None], k_idx, :]

        result = expanded_scale * tensor

        return result

    updated_a = pad_and_multiply(input_scale_tmp, a_tmp.to(torch.float32))
    updated_b = pad_and_multiply(weight_scale_tmp, b_tmp.to(torch.float32))

    ref = torch.zeros((m, n), device="cuda", dtype=torch.float32)

    len_offset_array = offset_array.shape[0]
    for i in range(len_offset_array - 1):
        start = offset_array[i]
        end = offset_array[i + 1]
        # assert start <= end, f"Invalid group boundaries: start={start} > end={end}"
        ref[start:end, :] = torch.einsum("mk,nk->mn", updated_a[start:end, :,
                                                                0],
                                         updated_b[:, :, i])
    ref = ref.to(torch.bfloat16)
    return ref


class CuteDslFusedMoE(CutlassFusedMoE):
    """
    Python Flow of Fused Mixture of Experts (MoE) Layer.

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
            aux_stream=aux_stream,
            weight_loading_mode=weight_loading_mode,
            apply_router_weight_on_input=apply_router_weight_on_input,
            layer_idx=layer_idx,
        )

    # def forward_chunk(
    #         self,
    #         x: Union[torch.Tensor, Fp4QuantizedTensor],
    #         router_logits: torch.Tensor,
    #         cutlass_min_latency_mode: bool = False,
    #         output_dtype: Optional[torch.dtype] = None,
    #         all_rank_num_tokens: Optional[List[int]] = None,
    #         use_dp_padding: Optional[bool] = None,
    #         repeating_info: Tuple = (True, True),
    # ) -> torch.Tensor:
    #     if isinstance(x, Fp4QuantizedTensor):
    #         assert output_dtype is not None
    #         output_dtype = output_dtype
    #     else:
    #         output_dtype = x.dtype

    #     is_first_call, is_last_call = repeating_info

    #     if (self.layer_load_balancer
    #             and not self.layer_load_balancer.is_static_routing()
    #             and is_first_call):
    #         self.layer_load_balancer.wait_for_gpu_stage()

    #     use_deepseek_fp8_block_scale = False
    #     weight_dtype = self.w3_w1_weight.dtype

    #     token_selected_experts, token_final_scales = self.routing_method.apply(
    #         router_logits)

    #     assert token_selected_experts.shape[
    #         1] == self.routing_method.experts_per_token
    #     assert token_selected_experts.shape == token_final_scales.shape
    #     assert token_selected_experts.shape[0] == router_logits.shape[0]
    #     assert token_final_scales.dtype == torch.float32
    #     assert token_selected_experts.dtype == torch.int32

    #     if self.apply_router_weight_on_input:
    #         assert (self.routing_method.top_k == 1
    #                 ), "Current workaround only supports top-1 routing"
    #         assert (
    #             x.dtype != torch.float8_e4m3fn
    #         ), "Current workaround for apply_router_weight_on_input does not support fp8 input"
    #         x = x * token_final_scales.to(x.dtype)
    #         # TODO: remove this once we have correct fusedmoe kernel ready
    #         token_final_scales = None

    #     if (self.layer_load_balancer
    #             and not self.layer_load_balancer.is_static_routing()
    #             and is_first_call):
    #         self.layer_load_balancer.maybe_cudagraph_done_wait()

    #     need_statistic = False
    #     if self.layer_load_balancer is None:
    #         token_selected_slots = token_selected_experts
    #     else:
    #         token_selected_slots = self.layer_load_balancer.route(
    #             token_selected_experts, self.use_dp)
    #         if not self.layer_load_balancer.is_static_routing():
    #             need_statistic = True

    #     # If load balancer is disabled, the statistics are collected from expert IDs.
    #     # If load balancer is enabled, the statistics are collected from expert slot IDs.
    #     ExpertStatistic.set_layer(self.layer_idx)
    #     ExpertStatistic.maybe_add_info(self.num_slots, token_selected_slots)

    #     token_selected_experts_for_statistic = (token_selected_experts
    #                                             if need_statistic else None)

    #     if self.enable_alltoall:
    #         if self.alltoall_method_type == AlltoallMethodType.MNNVL:
    #             token_count = x.shape[0]
    #             alltoall_info = None
    #             (
    #                 x,
    #                 token_selected_slots,
    #                 token_final_scales,
    #                 token_selected_experts_for_statistic,
    #                 alltoall_info,
    #             ) = self.alltoall_prepare_maybe_dispatch(
    #                 all_rank_num_tokens,
    #                 x,
    #                 token_selected_slots,
    #                 token_final_scales,
    #                 token_selected_experts_for_statistic,
    #             )
    #         elif self.alltoall_method_type == AlltoallMethodType.DeepEP:
    #             if not self.use_postquant_alltoall:
    #                 (
    #                     x,
    #                     recv_topk_idx,
    #                     token_final_scales,
    #                     num_recv_tokens_per_expert_list,
    #                     deep_ep_handle,
    #                 ) = self.deep_ep_buffer.dispatch(
    #                     x,
    #                     token_selected_slots.to(torch.int64),
    #                     token_final_scales,
    #                     self.num_slots,
    #                 )
    #         elif self.alltoall_method_type == AlltoallMethodType.DeepEPLowLatency:
    #             if not self.use_postquant_alltoall:
    #                 deep_ep_topk_idx = token_selected_slots.to(torch.int64)
    #                 deep_ep_topk_weights = token_final_scales
    #                 x, recv_expert_count, deep_ep_handle = (
    #                     self.deep_ep_buffer.low_latency_dispatch(
    #                         x,
    #                         deep_ep_topk_idx,
    #                         self.deep_ep_max_num_tokens,
    #                         self.num_slots,
    #                     ))
    #                 # x shape: [#local experts, #max recv tokens, hidden_size]
    #                 # recv_expert_count shape: [#local experts]

    #                 # Adapter between `torch.ops.trtllm.fused_moe` and DeepEP
    #                 # TODO: remove the adapter by changing `torch.ops.trtllm.fused_moe` API
    #                 mask = torch.arange(
    #                     x.shape[1], dtype=torch.int32, device=x.device).expand(
    #                         x.shape[0],
    #                         x.shape[1]) < recv_expert_count.unsqueeze(1)
    #                 token_selected_slots = torch.full(
    #                     (x.shape[0], x.shape[1], self.routing_method.top_k),
    #                     self.num_slots,
    #                     dtype=torch.int32,
    #                     device=x.device,
    #                 )
    #                 token_selected_slots[:, :, 0] = torch.where(
    #                     mask,
    #                     torch.arange(
    #                         x.shape[0] * self.mapping.moe_ep_rank,
    #                         x.shape[0] * (self.mapping.moe_ep_rank + 1),
    #                         dtype=torch.int32,
    #                         device=x.device,
    #                     ).unsqueeze(1),
    #                     self.num_slots,
    #                 )
    #                 x = x.view(x.shape[0] * x.shape[1], x.shape[2])
    #                 token_selected_slots = token_selected_slots.view(
    #                     x.shape[0], self.routing_method.top_k)
    #                 token_final_scales = torch.ones_like(
    #                     token_selected_slots, dtype=token_final_scales.dtype)
    #         else:
    #             raise NotImplementedError(
    #                 f"Not available alltoall method type: {alltoall_method_type!r}"
    #             )

    #     x_sf = None
    #     if self.has_any_quant:
    #         if self.has_fp8_qdq:
    #             x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
    #                 x, self.fc31_input_dequant)
    #         elif self.has_nvfp4:
    #             if not disable_fp4_allgather() or self.use_postquant_alltoall:
    #                 if isinstance(x, Fp4QuantizedTensor):
    #                     x, x_sf = x.fp4_tensor, x.scaling_factor
    #                     x_row = x.shape[0]
    #                     # note: we use uint8 to store 2 fp4 values
    #                     x_col = x.shape[1] * 2
    #                 else:
    #                     x_row = x.shape[0]
    #                     x_col = x.shape[1]
    #                     x, x_sf = torch.ops.trtllm.fp4_quantize(
    #                         x, self.fc31_input_scale, self.scaling_vector_size,
    #                         False)

    #         elif self.has_deepseek_fp8_block_scales:
    #             use_deepseek_fp8_block_scale = True
    #         elif self.has_w4afp8:
    #             weight_dtype = torch.quint4x2
    #         else:
    #             raise ValueError(
    #                 f"unsupported quantization mode: {self.quant_config.quant_mode}"
    #             )

    #     if (self.use_dp and self.parallel_size > 1
    #             and not disable_fp4_allgather() and not self.enable_alltoall):
    #         (
    #             x,
    #             x_sf,
    #             token_selected_slots,
    #             token_final_scales,
    #             token_selected_experts_for_statistic,
    #         ) = allgather(
    #             [
    #                 x,
    #                 x_sf,
    #                 token_selected_slots,
    #                 token_final_scales,
    #                 token_selected_experts_for_statistic,
    #             ],
    #             self.mapping,
    #             dim=0,
    #             sizes=None if use_dp_padding else all_rank_num_tokens,
    #         )
    #         # Fp4 gemm has extra scaling factor
    #         if x_sf is not None:
    #             x_sf = reswizzle_sf(x_sf, x_row, x_col,
    #                                 self.scaling_vector_size)

    #     if (self.layer_load_balancer
    #             and not self.layer_load_balancer.is_static_routing()):
    #         self.layer_load_balancer.statistic(
    #             token_selected_experts_for_statistic, is_first_call,
    #             is_last_call)

    #     if self.smart_router and not cutlass_min_latency_mode:
    #         ep_size = self.cluster_size
    #         ep_rank = self.cluster_rank
    #         expert_start = ep_rank * self.num_experts // ep_size
    #         expert_end = min(self.num_experts,
    #                          (ep_rank + 1) * self.num_experts // ep_size)
    #         w3_w1_weight = self.w3_w1_weight.narrow(0, expert_start,
    #                                                 expert_end - expert_start)
    #         w2_weight = self.w2_weight.narrow(0, expert_start,
    #                                           expert_end - expert_start)
    #         cluster_size = self.ep_size
    #         cluster_rank = self.ep_rank
    #         quant_scales = self.get_quant_scales(expert_start, expert_end)
    #     else:
    #         ep_size = self.ep_size
    #         ep_rank = self.ep_rank
    #         w3_w1_weight = self.w3_w1_weight
    #         w2_weight = self.w2_weight
    #         cluster_size = self.cluster_size
    #         cluster_rank = self.cluster_rank
    #         quant_scales = self.quant_scales

    #     if self.use_postquant_alltoall:
    #         if self.alltoall_method_type == AlltoallMethodType.MNNVL:
    #             x, x_sf = self.alltoall_postquant_dispatch(
    #                 x, x_sf, x_row, x_col, alltoall_info)
    #         elif self.alltoall_method_type == AlltoallMethodType.DeepEP:
    #             if x_sf is not None:
    #                 if self.has_nvfp4:
    #                     x_sf = unswizzle_sf(x_sf, x_row, x_col,
    #                                         self.scaling_vector_size)
    #                 # Adapter between `x_sf` and DeepEP
    #                 # TODO: remove the adapter by adding dtype support to DeepEP
    #                 x_sf_dtype = x_sf.dtype
    #                 x_sf = x_sf.view(torch.float32)
    #             (
    #                 (x, x_sf),
    #                 recv_topk_idx,
    #                 token_final_scales,
    #                 num_recv_tokens_per_expert_list,
    #                 deep_ep_handle,
    #             ) = self.deep_ep_buffer.dispatch(
    #                 (x, x_sf),
    #                 token_selected_slots.to(torch.int64),
    #                 token_final_scales,
    #                 self.num_slots,
    #             )
    #             if x_sf is not None:
    #                 x_sf = x_sf.view(x_sf_dtype)
    #                 if self.has_nvfp4:
    #                     x_sf = swizzle_sf(x_sf, x.shape[0], x.shape[1] * 2,
    #                                       self.scaling_vector_size)
    #         elif self.alltoall_method_type == AlltoallMethodType.DeepEPLowLatency:
    #             raise NotImplementedError(
    #                 "Not implemented postquant for DeepEPLowLatency, please set TRTLLM_MOE_POST_QUANT_ALLTOALLV=0"
    #             )
    #         else:
    #             raise NotImplementedError(
    #                 f"Not available alltoall method type: {self.alltoall_method_type!r}"
    #             )

    #     if self.enable_alltoall:
    #         # Adapter between `torch.ops.trtllm.fused_moe` and DeepEP
    #         # TODO: remove the adapter by changing APIs
    #         if self.alltoall_method_type == AlltoallMethodType.DeepEP:
    #             token_selected_slots = recv_topk_idx.to(torch.int32)
    #             mask = token_selected_slots == -1
    #             token_selected_slots += (self.expert_size_per_partition *
    #                                      self.mapping.moe_ep_rank)
    #             token_selected_slots[mask] = self.num_slots
    #             num_recv_token_is_zero = x.shape[0] == 0
    #             if x.shape[0] == 0:
    #                 x = torch.zeros((1, x.shape[1]),
    #                                 dtype=x.dtype,
    #                                 device=x.device)
    #                 token_selected_slots = torch.full(
    #                     (1, token_selected_slots.shape[1]),
    #                     self.num_slots,
    #                     dtype=token_selected_slots.dtype,
    #                     device=token_selected_slots.device,
    #                 )
    #                 token_final_scales = torch.ones(
    #                     (1, token_final_scales.shape[1]),
    #                     dtype=token_final_scales.dtype,
    #                     device=token_final_scales.device,
    #                 )

    #     (
    #         unpermuted_token_selected_experts_tensor,
    #         unpermuted_source_token_ids_tensor,
    #         permuted_source_token_ids_tensor,
    #         permuted_token_selected_experts_tensor,
    #         permuted_data_tensor,
    #         expert_first_token_offset_tensor,
    #         permuted_token_final_scales_tensor,
    #         src_to_dest_map_tensor,
    #     ) = torch.ops.trtllm.moe_permute_op(
    #         x,
    #         token_selected_slots,
    #         token_final_scales,
    #         None,  # w3_w1_weight.view(weight_dtype),
    #         None,  # w2_weight.view(weight_dtype),
    #         None,  # quant_scales,
    #         input_sf=x_sf,
    #         num_experts_on_rank=self.expert_size_per_partition,
    #         tp_size=self.tp_size,
    #         tp_rank=self.tp_rank,
    #         ep_size=ep_size,
    #         ep_rank=ep_rank,
    #         cluster_size=cluster_size,
    #         cluster_rank=cluster_rank,
    #         min_latency_mode=cutlass_min_latency_mode,
    #         use_fp8_block_scaling=use_deepseek_fp8_block_scale,
    #     )

    #     act_input_fp8, act_input_sf = torch.ops.trtllm.fp8_quantize_1x128(
    #         permuted_data_tensor)
    #     h1 = cute_dsl_fp8_group_blockwise_gemm_ref(
    #         a=act_input_fp8,
    #         b=w3_w1_weight.view(weight_dtype),
    #         a_sf=act_input_sf,
    #         b_sf=quant_scales[0],
    #         offset_array=expert_first_token_offset_tensor,
    #     )
    #     h2 = swiglu_fused_moe(h1)
    #     act_input_fp8, act_input_sf = torch.ops.trtllm.fp8_quantize_1x128(h2)
    #     h3 = cute_dsl_fp8_group_blockwise_gemm_ref(
    #         a=act_input_fp8,
    #         b=w2_weight.view(weight_dtype),
    #         a_sf=act_input_sf,
    #         b_sf=quant_scales[1],
    #         offset_array=expert_first_token_offset_tensor,
    #     )
    #     final_hidden_states = torch.ops.trtllm.moe_finalize_scale_op(
    #         h3,
    #         None,
    #         token_final_scales,
    #         src_to_dest_map_tensor,
    #         unpermuted_token_selected_experts_tensor,
    #         expert_first_token_offset_tensor,
    #         x.shape[0],  # num_rows
    #         x.shape[1],  # hidden_size
    #         self.routing_method.top_k,
    #         self.expert_size_per_partition,  # num_experts_per_node
    #         self.tp_size,
    #         self.tp_rank,
    #         ep_size,
    #         ep_rank,
    #     )

    #     if self.enable_alltoall:
    #         if self.alltoall_method_type == AlltoallMethodType.MNNVL:
    #             final_hidden_states = self.alltoall_combine(
    #                 final_hidden_states, alltoall_info, token_count)
    #         elif self.alltoall_method_type == AlltoallMethodType.DeepEP:
    #             if num_recv_token_is_zero:
    #                 final_hidden_states = final_hidden_states[:0]
    #             final_hidden_states = self.deep_ep_buffer.combine(
    #                 final_hidden_states, deep_ep_handle)
    #         elif self.alltoall_method_type == AlltoallMethodType.DeepEPLowLatency:
    #             final_hidden_states = self.deep_ep_buffer.low_latency_combine(
    #                 final_hidden_states.view(
    #                     self.expert_size_per_partition,
    #                     self.deep_ep_max_num_tokens * self.mapping.moe_ep_size,
    #                     final_hidden_states.shape[1],
    #                 ),
    #                 deep_ep_topk_idx,
    #                 deep_ep_topk_weights,
    #                 deep_ep_handle,
    #             )
    #         else:
    #             raise NotImplementedError(
    #                 f"Not available alltoall method type: {self.alltoall_method_type!r}"
    #             )

    #     if (self.layer_load_balancer
    #             and not self.layer_load_balancer.is_static_routing()
    #             and is_last_call):
    #         self.layer_load_balancer.maybe_cudagraph_done_set_cpu_stage()

    #     return final_hidden_states

    def forward_chunk(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
    ) -> torch.Tensor:
        if isinstance(x, Fp4QuantizedTensor):
            assert output_dtype is not None
            output_dtype = output_dtype
        else:
            output_dtype = x.dtype

        # apply routing
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

        # quantize inputs
        use_deepseek_fp8_block_scale = False
        weight_dtype = self.w3_w1_weight.dtype
        x_sf = None
        if self.has_any_quant:
            if self.has_fp8_qdq:
                x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    x, self.fc31_input_dequant)
            elif self.has_deepseek_fp8_block_scales:
                use_deepseek_fp8_block_scale = True
            elif self.has_w4afp8:
                weight_dtype = torch.quint4x2
            elif self.has_nvfp4 and not disable_fp4_allgather():
                if isinstance(x, Fp4QuantizedTensor):
                    x_row = x.shape[0]
                    # note: we use uint8 to store 2 fp4 values
                    x_col = x.shape[1] * 2
                    x, x_sf = x.fp4_tensor, x.scaling_factor
                else:
                    x_row = x.shape[0]
                    x_col = x.shape[1]
                    x, x_sf = torch.ops.trtllm.fp4_quantize(
                        x, self.fc31_input_scale, self.scaling_vector_size,
                        False)
            else:
                raise ValueError(
                    f"unsupported quantization mode: {self.quant_config.quant_mode}"
                )

        # gather inputs for attention dp
        if self.use_dp and self.parallel_size > 1 and not disable_fp4_allgather(
        ):
            x, x_sf, token_selected_experts, token_final_scales = allgather(
                [x, x_sf, token_selected_experts, token_final_scales],
                self.mapping,
                dim=0,
                sizes=None if use_dp_padding else all_rank_num_tokens)
            # Fp4 gemm has extra scaling factor
            if x_sf is not None:
                x_sf = reswizzle_sf(x_sf, x_row, x_col,
                                    self.scaling_vector_size)

        # final_hidden_states = torch.ops.trtllm.fused_moe(
        #     x,
        #     token_selected_experts,
        #     token_final_scales,
        #     self.w3_w1_weight.view(weight_dtype),
        #     self.w2_weight.view(weight_dtype),
        #     output_dtype,
        #     quant_scales=self.quant_scales,
        #     input_sf=x_sf,
        #     tp_size=self.tp_size,
        #     tp_rank=self.tp_rank,
        #     ep_size=self.ep_size,
        #     ep_rank=self.ep_rank,
        #     cluster_size=self.cluster_size,
        #     cluster_rank=self.cluster_rank,
        #     enable_alltoall=self.enable_alltoall,
        #     use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
        #     use_w4a8_group_scaling=use_w4a8_group_scaling,
        #     min_latency_mode=False,
        #     tune_max_num_tokens=self.tune_max_num_tokens,
        # )

        # # Custom op requires all inputs are in the same type.
        # # Only in cutlass_min_latency_mode, the output is a list of tensors.
        # # Otherwise, the output should be unpacked as a single tensor.
        # final_hidden_states = final_hidden_states[0]

        (
            unpermuted_token_selected_experts_tensor,
            unpermuted_source_token_ids_tensor,
            permuted_source_token_ids_tensor,
            permuted_token_selected_experts_tensor,
            permuted_data_tensor,
            expert_first_token_offset_tensor,
            permuted_token_final_scales_tensor,
            src_to_dest_map_tensor,
        ) = torch.ops.trtllm.moe_permute_op(
            x,
            token_selected_experts,
            token_final_scales,
            None,  # w3_w1_weight.view(weight_dtype),
            None,  # w2_weight.view(weight_dtype),
            None,  # quant_scales,
            input_sf=x_sf,
            num_experts_on_rank=self.expert_size_per_partition,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=self.cluster_size,
            cluster_rank=self.cluster_rank,
            min_latency_mode=False,
            use_fp8_block_scaling=use_deepseek_fp8_block_scale,
        )

        act_input_fp8, act_input_sf = torch.ops.trtllm.fp8_quantize_1x128(
            permuted_data_tensor)
        h1 = cute_dsl_fp8_group_blockwise_gemm_ref(
            a=act_input_fp8,
            b=self.w3_w1_weight.view(weight_dtype),
            a_sf=act_input_sf,
            b_sf=self.quant_scales[0],
            offset_array=expert_first_token_offset_tensor,
        )
        h2 = swiglu_fused_moe(h1)
        act_input_fp8, act_input_sf = torch.ops.trtllm.fp8_quantize_1x128(h2)
        h3 = cute_dsl_fp8_group_blockwise_gemm_ref(
            a=act_input_fp8,
            b=self.w2_weight.view(weight_dtype),
            a_sf=act_input_sf,
            b_sf=self.quant_scales[1],
            offset_array=expert_first_token_offset_tensor,
        )
        final_hidden_states = torch.ops.trtllm.moe_finalize_scale_op(
            h3,
            None,
            token_final_scales,
            src_to_dest_map_tensor,
            unpermuted_token_selected_experts_tensor,
            expert_first_token_offset_tensor,
            x.shape[0],  # num_rows
            x.shape[1],  # hidden_size
            self.routing_method.top_k,
            self.expert_size_per_partition,  # num_experts_per_node
            self.tp_size,
            self.tp_rank,
            self.ep_size,
            self.ep_rank,
        )

        return final_hidden_states
