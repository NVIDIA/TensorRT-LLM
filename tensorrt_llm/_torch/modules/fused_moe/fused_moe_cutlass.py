from typing import Dict, List, Optional, Union

import torch

from ...distributed import allgather, reducescatter
from ...model_config import ModelConfig
from ...utils import (EventType, Fp4QuantizedTensor, ceil_div,
                      disable_fp4_allgather, swizzle_sf)
from .interface import MoE
from .quantization import (DeepSeekFP8BlockScalesFusedMoEMethod,
                           FP8QDQFusedMoEMethod, MoEWeightLoadingMode,
                           NVFP4CutlassFusedMoEMethod,
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

        self.num_slots = self.num_experts
        self.expert_size_per_partition = self.num_experts // self.ep_size
        self.initial_global_assignments = [
            (ep_rank * self.num_experts // self.ep_size + local_slot_id) %
            self.num_experts for ep_rank in range(self.ep_size)
            for local_slot_id in range(self.expert_size_per_partition)
        ]
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
        self.moe_max_num_tokens = model_config.moe_max_num_tokens or max_num_tokens
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

        # If True, the router weight will be multiplied on the input rather than at the end of FC2
        self.apply_router_weight_on_input = apply_router_weight_on_input

        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def _check_configs(self):
        assert self._weights_created

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
        if self.parallel_size > 1:
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

        use_allgather = self.use_dp and self.parallel_size > 1 and not disable_fp4_allgather(
        )

        # quantize inputs
        use_deepseek_fp8_block_scale = False
        use_w4a8_group_scaling = False
        weight_dtype = self.w3_w1_weight.dtype
        x_sf = None
        if self.has_any_quant:
            if self.has_fp8_qdq:
                x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    x, self.fc31_input_dequant)
            elif self.has_deepseek_fp8_block_scales:
                use_deepseek_fp8_block_scale = True
            elif self.has_w4afp8:
                use_w4a8_group_scaling = True
                weight_dtype = torch.quint4x2
            elif self.has_nvfp4:
                if use_allgather:
                    if isinstance(x, Fp4QuantizedTensor):
                        assert not x.is_sf_swizzled, "Fp4QuantizedTensor should not be swizzled before communication"
                        x_row = x.shape[0]
                        # note: we use uint8 to store 2 fp4 values
                        x_col = x.shape[1] * 2
                        x, x_sf = x.fp4_tensor, x.scaling_factor
                    else:
                        x_row = x.shape[0]
                        x_col = x.shape[1]
                        x, x_sf = torch.ops.trtllm.fp4_quantize(
                            x,
                            self.fc31_input_scale,
                            self.scaling_vector_size,
                            sfUseUE8M0=False,
                            swizzedLayout=False)
                        x_sf = x_sf.view(
                            x_row, ceil_div(x_col, self.scaling_vector_size))
                else:
                    if not isinstance(x, Fp4QuantizedTensor):
                        x, x_sf = torch.ops.trtllm.fp4_quantize(
                            x,
                            self.fc31_input_scale,
                            self.scaling_vector_size,
                            sfUseUE8M0=False,
                            swizzedLayout=True)
            else:
                raise ValueError(
                    f"unsupported quantization mode: {self.quant_config.quant_mode}"
                )

        # gather inputs for attention dp
        if use_allgather:
            x, x_sf, token_selected_experts, token_final_scales = allgather(
                [x, x_sf, token_selected_experts, token_final_scales],
                self.mapping,
                dim=0,
                sizes=None if use_dp_padding else all_rank_num_tokens)
            x_row = x.shape[0]
            # Fp4 gemm has extra scaling factor
            if x_sf is not None:
                x_sf = swizzle_sf(x_sf, x_row, x_col, self.scaling_vector_size)

        final_hidden_states = torch.ops.trtllm.fused_moe(
            x,
            token_selected_experts,
            token_final_scales,
            self.w3_w1_weight.view(weight_dtype),
            None,  # fc1_expert_biases
            self.w2_weight.view(weight_dtype),
            None,  # fc2_expert_biases
            output_dtype,
            quant_scales=self.quant_scales,
            input_sf=x_sf,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=self.cluster_size,
            cluster_rank=self.cluster_rank,
            enable_alltoall=self.enable_alltoall,
            use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
            use_w4a8_group_scaling=use_w4a8_group_scaling,
            min_latency_mode=False,
            tune_max_num_tokens=self.tune_max_num_tokens,
        )

        # Custom op requires all inputs are in the same type.
        # Only in cutlass_min_latency_mode, the output is a list of tensors.
        # Otherwise, the output should be unpacked as a single tensor.
        final_hidden_states = final_hidden_states[0]

        return final_hidden_states

    def split_chunk(self, split_token_num: int, split_num_chunks: int):
        val_div = split_token_num // split_num_chunks
        val_mod = split_token_num % split_num_chunks
        split_chunk_size_list = [val_div + 1] * val_mod + [val_div] * (
            split_num_chunks - val_mod)
        return split_chunk_size_list

    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        do_finalize: bool = True,  # used by other MoE backends
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        all_rank_max_num_tokens: Optional[int] = None,
        use_dp_padding: Optional[bool] = None,
    ) -> torch.Tensor:
        assert do_finalize, "CutlassFusedMoE does not support do_finalize=False"
        if self.use_dp:
            assert all_rank_num_tokens is not None
            assert use_dp_padding is not None
            num_rows = sum(all_rank_num_tokens)
        else:
            num_rows = x.shape[0]

        # in case of num_rows is larger than max_chunk_size, we need to split the input into multiple chunks
        num_chunks = (num_rows + self.moe_max_num_tokens -
                      1) // self.moe_max_num_tokens

        if use_dp_padding:
            all_rank_num_tokens_padded = [all_rank_max_num_tokens
                                          ] * len(all_rank_num_tokens)
        else:
            all_rank_num_tokens_padded = all_rank_num_tokens

        if num_chunks == 1:
            outputs = self.forward_chunk(
                x,
                router_logits,
                output_dtype,
                all_rank_num_tokens=all_rank_num_tokens_padded,
                use_dp_padding=use_dp_padding)
            outputs = self.reducescatter_or_allreduce(
                outputs,
                all_rank_num_tokens=all_rank_num_tokens_padded,
                use_dp_padding=use_dp_padding)
        else:
            if self.use_dp:
                all_rank_chunk_size_list = [
                    self.split_chunk(val, num_chunks)
                    for val in all_rank_num_tokens_padded
                ]
                all_rank_num_tokens_list = [[
                    val[idx_chunk] for val in all_rank_chunk_size_list
                ] for idx_chunk in range(num_chunks)]
                chunk_size_list = all_rank_chunk_size_list[self.rank]
            else:
                all_rank_num_tokens_list = [None] * num_chunks
                chunk_size_list = self.split_chunk(x.shape[0], num_chunks)

            x_list = x.split(chunk_size_list)
            router_logits_list = router_logits.split(chunk_size_list)

            self.event_dict[EventType.Main].record()
            with torch.cuda.stream(self.aux_stream):
                self.event_dict[EventType.Main].wait()

            def _forward_chunk(x_, router_logits_, idx):
                return self.forward_chunk(
                    x_,
                    router_logits_,
                    all_rank_num_tokens=all_rank_num_tokens_list[idx]
                    if self.use_dp else None,
                    use_dp_padding=use_dp_padding)

            def _reducescatter_or_allreduce(x_, idx):
                return self.reducescatter_or_allreduce(
                    x_,
                    all_rank_num_tokens=all_rank_num_tokens_list[idx],
                    use_dp_padding=use_dp_padding)

            outputs_list = []
            # Postpone reduce-scatter/all-reduce to the next iteration to achieve better overlap
            for idx_chunk, (x, router_logits) in enumerate(
                    zip(x_list, router_logits_list)):

                if idx_chunk % 2 == 0:
                    with torch.cuda.stream(self.aux_stream):
                        outputs = _forward_chunk(x, router_logits, idx_chunk)
                    if idx_chunk > 0:
                        outputs_list[-1] = _reducescatter_or_allreduce(
                            outputs_list[-1], idx_chunk - 1)
                else:
                    outputs = _forward_chunk(x, router_logits, idx_chunk)
                    with torch.cuda.stream(self.aux_stream):
                        outputs_list[-1] = _reducescatter_or_allreduce(
                            outputs_list[-1], idx_chunk - 1)

                outputs_list.append(outputs)

            if num_chunks % 2 == 0:
                outputs_list[-1] = _reducescatter_or_allreduce(
                    outputs_list[-1], -1)
            else:
                with torch.cuda.stream(self.aux_stream):
                    outputs_list[-1] = _reducescatter_or_allreduce(
                        outputs_list[-1], -1)
            with torch.cuda.stream(self.aux_stream):
                self.event_dict[EventType.MoeChunkingOverlap].record()
            self.event_dict[EventType.MoeChunkingOverlap].wait()

            outputs = torch.cat(outputs_list)

        if self.use_dp:
            rank = self.mapping.tp_rank
            outputs = outputs[:all_rank_num_tokens[rank]]
        return outputs

    def load_weights(self, weights: List[Dict]):
        assert self._weights_created
        assert len(weights) == 1
        weights = weights[0]

        self.quant_method.load_weights(self, weights, self.weight_loading_mode)
