import os
from typing import Dict, List, Optional, Union

import torch

from tensorrt_llm._mnnvl_utils import MnnvlMoe, MoEAlltoallInfo
from tensorrt_llm._utils import logger

from ...distributed import allgather, reducescatter
from ...expert_statistic import ExpertStatistic
from ...model_config import ModelConfig, MoeLoadBalancerConfig
from ...utils import (EventType, Fp4QuantizedTensor, disable_fp4_allgather,
                      reswizzle_sf, swizzle_sf, unswizzle_sf)
from .interface import MoE
from .moe_load_balancer import MoeLoadBalancer
from .quantization import (FP8BlockScalesFusedMoEMethod, FP8QDQFusedMoEMethod,
                           MoEWeightLoadingMode, NVFP4CutlassFusedMoEMethod,
                           UnquantizedFusedMoEMethod, WInt4AFP8FusedMoEMethod)
from .routing import BaseMoeRoutingMethod


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
        enable_alltoall (bool): whether to enable alltoall instead of allgather/reducescatter

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
        enable_alltoall: bool = False,
        moe_load_balancer: Optional[MoeLoadBalancer] = None,
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
        moe_load_balancer_config = model_config.moe_load_balancer
        if moe_load_balancer_config is None:
            assert moe_load_balancer is None
            # A dummy MoeLoadBalancerConfig to generate default initial_global_assignments
            moe_load_balancer_config = MoeLoadBalancerConfig()
            moe_load_balancer_config.setup(num_experts=num_experts,
                                           ep_rank=self.ep_rank,
                                           ep_size=self.ep_size)
        else:
            assert moe_load_balancer is not None

        self.num_slots = moe_load_balancer_config.num_slots
        if self.smart_router:
            assert self.num_slots == self.num_experts, "Smart router should not have redundant slots"

        self.initial_global_assignments = moe_load_balancer_config.get_layer_initial_global_assignments(
            layer_idx)
        self.expert_size_per_partition = moe_load_balancer_config.num_local_slots
        self.slot_start = moe_load_balancer_config.slot_start
        self.slot_end = moe_load_balancer_config.slot_end
        self.initial_local_expert_ids = self.initial_global_assignments[
            self.slot_start:self.slot_end]
        assert len(
            self.initial_local_expert_ids) == self.expert_size_per_partition

        self.balancer_layer = None
        if moe_load_balancer is not None:
            self.balancer_layer = moe_load_balancer.add_layer(
                expert_count=num_experts,
                top_k=routing_method.experts_per_token,
                slot_count_per_rank=self.expert_size_per_partition,
            )
            self.balancer_layer.set_initial_weight_assignments(
                self.initial_global_assignments)
            logger.info(
                f"MoE load balancer enabled. num_experts = {num_experts}, num_slots = {self.num_slots}, ep_size = {self.ep_size}"
            )
            logger.info(
                f"initial_global_assignments (layer {layer_idx}) = {self.initial_global_assignments}"
            )

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

    @property
    def has_w4afp8(self):
        assert self._weights_created
        return self.quant_config and self.quant_config.quant_mode.is_int4_weight_only_per_group(
        )

    def _get_quant_method(self):
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True):
            if self.quant_config.layer_quant_mode.has_fp8_qdq():
                return FP8QDQFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_fp8_block_scales():
                return FP8BlockScalesFusedMoEMethod()
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
        if self.balancer_layer is None:
            token_selected_slots = token_selected_experts
        else:
            # If attention DP is enabled, token_selected_experts is a local rank tensor,
            # so we need to offset the round robin position by ep_rank
            token_selected_slots = self.balancer_layer.route(
                token_selected_experts, offset_by_ep_rank=self.use_dp)

        # If load balancer is disabled, the statistics are collected from expert IDs.
        # If load balancer is enabled, the statistics are collected from expert slot IDs.
        ExpertStatistic.set_layer(self.layer_idx)
        ExpertStatistic.maybe_add_info(self.num_slots, token_selected_slots)

        assert token_selected_slots.shape[
            1] == self.routing_method.experts_per_token
        assert token_selected_slots.shape == token_final_scales.shape
        assert token_selected_slots.shape[0] == router_logits.shape[0]
        assert token_final_scales.dtype == torch.float32
        assert token_selected_slots.dtype == torch.int32

        if self.apply_router_weight_on_input:
            assert self.routing_method.top_k == 1, "Current workaround only supports top-1 routing"
            assert x.dtype != torch.float8_e4m3fn, "Current workaround for apply_router_weight_on_input does not support fp8 input"
            x = x * token_final_scales.to(x.dtype)
            # TODO: remove this once we have correct fusedmoe kernel ready
            token_final_scales = None

        token_count = x.shape[0]

        alltoall_info = None

        if self.enable_alltoall:
            x, token_selected_slots, token_final_scales, alltoall_info = \
                self.alltoall_prepare_maybe_dispatch(all_rank_num_tokens,
                                                     x,
                                                     token_selected_slots,
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
            x, x_sf, token_selected_slots, token_final_scales = allgather(
                [x, x_sf, token_selected_slots, token_final_scales],
                self.mapping,
                dim=0,
                sizes=None if use_dp_padding else all_rank_num_tokens)
            # Fp4 gemm has extra scaling factor
            if x_sf is not None:
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
            use_fp8_block_scaling=use_fp8_block_scaling,
            use_w4a8_group_scaling=use_w4a8_group_scaling,
            min_latency_mode=cutlass_min_latency_mode,
            tune_max_num_tokens=self.tune_max_num_tokens,
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

    def alltoall_prepare_maybe_dispatch(self, all_rank_num_tokens: list,
                                        x: torch.Tensor,
                                        token_selected_slots: torch.Tensor,
                                        token_final_scales: torch.Tensor):
        top_k = self.routing_method.experts_per_token
        expert_count = self.num_experts
        # gather router info
        max_num_token = max(all_rank_num_tokens)
        token_selected_slots = torch.nn.functional.pad(
            token_selected_slots,
            (0, 0, 0, max_num_token - token_selected_slots.shape[0]),
            'constant', self.num_experts)
        token_final_scales = torch.nn.functional.pad(
            token_final_scales,
            (0, 0, 0, max_num_token - token_final_scales.shape[0]))
        gathered_token_selected_slots, gathered_token_final_scales = allgather(
            [token_selected_slots, token_final_scales], self.mapping, dim=0)
        gathered_token_selected_slots = torch.flatten(
            gathered_token_selected_slots.contiguous(), start_dim=0, end_dim=-2)
        gathered_token_final_scales = torch.flatten(
            gathered_token_final_scales.contiguous(), start_dim=0, end_dim=-2)
        gathered_target_rank_ids = MnnvlMoe.compute_target_rank_id(
            gathered_token_selected_slots, self.num_experts, self.ep_size)
        alltoall_info, token_selected_slots, token_final_scales = MnnvlMoe.mnnvl_moe_alltoallv_prepare(
            gathered_target_rank_ids, None, gathered_token_selected_slots,
            gathered_token_final_scales, max_num_token, expert_count, top_k,
            self.ep_rank, self.ep_size)

        if not self.use_postquant_alltoall:
            assert not isinstance(
                x, Fp4QuantizedTensor
            ), "pre-quant alltoall doesn't support fp4 tensor"
            x = MnnvlMoe.mnnvl_moe_alltoallv(x, alltoall_info,
                                             self.alltoall_workspace,
                                             self.ep_rank, self.ep_size)

        return x, token_selected_slots, token_final_scales, alltoall_info

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

        self.quant_method.load_weights(self, weights, self.weight_loading_mode)
