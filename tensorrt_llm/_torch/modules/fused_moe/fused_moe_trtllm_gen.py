import os
from functools import cached_property
from typing import Dict, List, Optional, Union

import torch
from torch import nn

from tensorrt_llm._mnnvl_utils import MnnvlMemory, MnnvlMoe
from tensorrt_llm._torch.distributed.moe_alltoall import MoeAlltoAll
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from ...custom_ops.trtllm_gen_custom_ops import \
    fp4_block_scale_fake_output_without_finalize
from ...distributed import allgather
from ...expert_statistic import ExpertStatistic
from ...model_config import ModelConfig
from ...utils import AuxStreamType, Fp4QuantizedTensor, ceil_div
from .interface import AlltoallMethodType, MoE, MoEWeightLoadingMode
from .quantization import (DeepSeekFP8BlockScalesFusedMoEMethod,
                           NVFP4TRTLLMGenFusedMoEMethod,
                           W4A8MXFP4FP8TRTLLMGenFusedMoEMethod,
                           W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod,
                           W4A8NVFP4FP8TRTLLMGenFusedMoEMethod,
                           W4A16MXFP4TRTLLMGenFusedMoEMethod)
from .routing import BaseMoeRoutingMethod, DeepSeekV3MoeRoutingMethod


class TRTLLMGenFusedMoE(MoE):
    """
    Fused Mixture of Experts (MoE) Layer with performance tuning.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.
        aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary CUDA streams for overlapping.

    MoE torch custom op:
        Only support min-latency mode now (SM100 Blackwell only).
        Quant: fp8 block scales quant and nvfp4 quant and w4a16_mxfp4 quant
            FusedMoE Op: routing(topK, etc.) + scatter + gemm1 + swiglu + gemm2 + finalize MoeRoute

    FusedMoE module:
        min-latency mode:
            dynamic quant + FusedMoe Op
            equals to: dynamic quant + routing(topK, etc.) + scatter + gemm1 + swiglu + gemm2 + finalize MoeRoute

    In min-latency mode, setting `reduce_results=False` disables the AllReduce in the FusedMoE module, so any necessary AllReduce operations must be added explicitly in the model definition.
    AttentionDP should be turned off for min-latency mode.

    When we have redundant expert, we have more weight slots than `num_experts`, in that case, we separate the concepts of expert and slot.
    Expert is the concept from model's perspective while slot is the concept from model engine's perspective.
    There should be at lease `num_experts` slots in the model engine. More than that is OK, in that case, some experts may have multiple replicas.
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
        aux_stream_dict: Optional[Dict[AuxStreamType,
                                       torch.cuda.Stream]] = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
        VANILLA,
        layer_idx: Optional[int] = None,
        bias: bool = False,
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        swiglu_limit: Optional[torch.Tensor] = None,
    ):
        super().__init__(
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
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            layer_idx=layer_idx,
        )

        sm_version = get_sm_version()
        if sm_version >= 120:
            raise NotImplementedError(
                "TRTLLMGenFusedMoE does not support SM120 and above.")

        assert not self.smart_router, "Smart router is not supported in TRTLLMGenFusedMoE."

        # Note: Load balancer initialization is handled by base class _init_load_balancer()
        # If no load balancer is available, the base class will set:
        # - self.num_slots = self.num_experts
        # - self.expert_size_per_partition = self.num_experts // self.ep_size
        # - self.initial_global_assignments, self.slot_start, self.slot_end, etc.

        # TODO: AlltoAll code is largely duplicated with WideEPMoE. Consider refactor and reuse in the future.
        self.alltoall_method_type = self.select_alltoall_method_type()
        logger.info_once(
            f"{self.__class__.__name__} selects alltoall_method_type {self.alltoall_method_type!r}",
            key="alltoall_method_type")
        self.alltoall_workspace = None
        self.alltoall_prepare_workspace = None
        self.use_low_precision_combine = False
        if self.enable_alltoall:
            self.use_low_precision_combine = model_config.use_low_precision_moe_combine

            if self.alltoall_method_type == AlltoallMethodType.MNNVL:
                if self.moe_alltoall_backend == "mnnvllatency":
                    MnnvlMemory.initialize()
                    self.alltoall_workspace = MnnvlMoe.get_moe_workspaces(
                        model_config.mapping)
                    self.alltoall_prepare_workspace = MnnvlMoe.get_moe_prepare_workspace(
                        model_config.mapping)
                elif self.moe_alltoall_backend == "mnnvlthroughput":
                    workspace_mb = int(
                        os.environ.get("TRTLLM_MOE_A2A_WORKSPACE_MB", "2048"))
                    self.moe_a2a = MoeAlltoAll(
                        mapping=self.mapping,
                        max_num_tokens=model_config.max_num_tokens,
                        top_k=self.routing_method.experts_per_token,
                        num_experts=self.num_slots,
                        workspace_size_per_rank=workspace_mb * 1024 * 1024,
                    )
                else:
                    raise ValueError(
                        f"Unsupported moe alltoall backend: {self.moe_alltoall_backend}"
                    )
            elif self.alltoall_method_type == AlltoallMethodType.DeepEP or self.alltoall_method_type == AlltoallMethodType.DeepEPLowLatency:
                raise NotImplementedError(
                    "DeepEP and DeepEPLowLatency are not supported for TRTLLMGenFusedMoE yet"
                )
            else:
                raise NotImplementedError(
                    f"Not available alltoall method type: {self.alltoall_method_type!r}"
                )

        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def select_alltoall_method_type(self) -> AlltoallMethodType:
        # If no attention DP, no need to use AlltoAll.
        if self.mapping.dp_size == 1:
            return AlltoallMethodType.NotEnabled

        # AlltoAll cannot support MoE TP.
        if self.mapping.moe_tp_size != 1:
            return AlltoallMethodType.NotEnabled

        if not MnnvlMemory.supports_mnnvl():
            return AlltoallMethodType.NotEnabled

        all2all_method_type = os.environ.get("TRTLLM_FORCE_ALLTOALL_METHOD")
        if all2all_method_type is not None:
            if AlltoallMethodType[all2all_method_type] in [
                    AlltoallMethodType.DeepEP,
                    AlltoallMethodType.DeepEPLowLatency
            ]:
                raise NotImplementedError(
                    "DeepEP and DeepEPLowLatency are not supported for CutlassFusedMoE yet"
                )
            return AlltoallMethodType[all2all_method_type]

        if os.environ.get("TRTLLM_MOE_DISABLE_ALLTOALLV", "0") == "1":
            return AlltoallMethodType.NotEnabled

        # TODO: We found that MNNVL performs better than NCCL AllGather/ReduceScatter,
        # regardless of the relationship between EP size and topK. We favor AlltoAll for now.
        # if not self.mapping.moe_ep_size > self.routing_method.experts_per_token:
        #     return AlltoallMethodType.NotEnabled

        return AlltoallMethodType.MNNVL

    def _supports_load_balancer(self) -> bool:
        """TRTLLMGenFusedMoE supports load balancer."""
        return True

    @cached_property
    def enable_alltoall(self):
        """ enable_alltoall (bool): whether to enable alltoall instead of allgather/reducescatter
        """
        return self.alltoall_method_type != AlltoallMethodType.NotEnabled

    @cached_property
    def moe_alltoall_backend(self):
        # "mnnvlthroughput" (default) or "mnnvllatency"
        return os.environ.get("TRTLLM_MOE_ALLTOALL_BACKEND",
                              "mnnvlthroughput").strip().lower()

    def _check_configs(self):
        assert self.has_deepseek_fp8_block_scales \
            or self.has_nvfp4 or self.has_w4a16_mxfp4 or self.has_w4a8_nvfp4_fp8 \
            or self.has_w4a8_mxfp4_fp8 or self.has_w4a8_mxfp4_mxfp8, "TRTLLMGenFusedMoE only supports fp8_block_scaling, nvfp4, w4a16_mxfp4, w4a8_mxfp4_fp8 and w4a8_mxfp4_mxfp8 dtypes."

        if self.bias or self.swiglu_alpha is not None or self.swiglu_beta is not None or self.swiglu_limit is not None:
            assert self.has_w4a16_mxfp4 or self.has_w4a8_mxfp4_fp8 or self.has_w4a8_mxfp4_mxfp8, "TRTLLMGenFusedMoE only supports mxfp4 quantization with bias, swiglu_alpha, swiglu_beta and swiglu_limit."

    def _get_quant_method(self):
        if self.quant_config is not None:
            if self.quant_config.layer_quant_mode.has_fp8_block_scales():
                return DeepSeekFP8BlockScalesFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_nvfp4():
                return NVFP4TRTLLMGenFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_w4a16_mxfp4():
                return W4A16MXFP4TRTLLMGenFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_w4a8_nvfp4_fp8():
                return W4A8NVFP4FP8TRTLLMGenFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_w4a8_mxfp4_fp8():
                return W4A8MXFP4FP8TRTLLMGenFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_w4a8_mxfp4_mxfp8():
                return W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod()
            else:
                raise NotImplementedError(
                    f"Unsupported quantization method by TRTLLMGenFusedMoE: {self.quant_config.quant_mode}"
                )
        else:
            raise NotImplementedError(
                "TRTLLMGenFusedMoE doesn't support fp16/bf16/fp32 MoE.")

    def create_weights(self):
        if self._weights_created:
            return

        self.quant_method = self._get_quant_method()
        self.quant_method.create_weights(self)

        self._weights_created = True
        self._check_configs()

        if (self.has_w4a16_mxfp4 or self.has_w4a8_nvfp4_fp8
                or self.has_w4a8_mxfp4_fp8
                or self.has_w4a8_mxfp4_mxfp8) and not self.bias:
            self.w3_w1_bias = nn.Parameter(torch.zeros(
                (self.w3_w1_weight.shape[0], self.w3_w1_weight.shape[1]),
                dtype=torch.float32),
                                           requires_grad=False)
            self.register_parameter("w3_w1_bias", self.w3_w1_bias)
            self.w2_bias = nn.Parameter(torch.zeros(
                (self.w2_weight.shape[0], self.w2_weight.shape[1]),
                dtype=torch.float32),
                                        requires_grad=False)
            self.register_parameter("w2_bias", self.w2_bias)

    def load_weights(self, weights: List[Dict]):
        assert self._weights_created

        assert len(weights) == 1
        weights = weights[0]

        self.quant_method.load_weights(self, weights, self.weight_loading_mode)

    def post_load_weights(self):
        self.quant_method.post_load_weights(self)

    def _quantize_for_post_quant_comm(self, x):
        """Quantize inputs prior to post-communication (alltoall/allgather).
        Returns: (x, x_sf, x_row, x_col)
        """
        x_row = x.shape[0]
        x_col = x.shape[1]
        x_sf = None
        if self.has_w4a8_mxfp4_fp8:
            x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                x, self.fc31_input_dequant[0])
            x_row, x_col = x.shape[0], x.shape[1]
        elif self.has_nvfp4:
            if isinstance(x, Fp4QuantizedTensor):
                assert not x.is_sf_swizzled, "Fp4QuantizedTensor should not be swizzled before communication"
                x_row = x.shape[0]
                x_col = x.shape[1] * 2
                x, x_sf = x.fp4_tensor, x.scaling_factor
            else:
                x_row = x.shape[0]
                x_col = x.shape[1]
                x, x_sf = torch.ops.trtllm.fp4_quantize(
                    x, self.fc31_input_scale, self.scaling_vector_size, False,
                    False)
        elif self.has_w4a8_mxfp4_mxfp8:
            x, x_sf = torch.ops.trtllm.mxfp8_quantize(
                x, False, alignment=self.quant_method.input_hidden_alignment)
            x_row, x_col = x.shape[0], x.shape[1]
        elif self.has_deepseek_fp8_block_scales:
            # No change required before communication
            pass
        elif self.has_w4a16_mxfp4:
            pad_size = self.w3_w1_weight.shape[-1] * 2 - x.shape[-1]
            x = torch.nn.functional.pad(x, (0, pad_size))
        elif self.has_w4a8_nvfp4_fp8:
            x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                x, 1.0 / self.fc31_input_scale)
        else:
            raise ValueError(
                f"unsupported quantization mode for post communication: {self.quant_config.quant_mode}"
            )
        return x, x_sf, x_row, x_col

    def forward_impl(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool = True,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:

        assert x.dtype == torch.bfloat16

        # DeepSeekV3 style routing
        if isinstance(self.routing_method, DeepSeekV3MoeRoutingMethod):
            top_k = self.routing_method.routing_impl.top_k
            routing_bias = self.routing_method.e_score_correction_bias
            n_group = self.routing_method.routing_impl.n_group
            topk_group = self.routing_method.routing_impl.topk_group
            routed_scaling_factor = self.routing_method.routing_impl.routed_scaling_factor
        else:
            top_k = self.routing_method.top_k
            routing_bias = None
            n_group = None
            topk_group = None
            routed_scaling_factor = None

        run_post_quant_allgather = (self.use_dp and self.parallel_size > 1
                                    and not self.enable_alltoall)
        post_quant_comm = run_post_quant_allgather or self.enable_alltoall

        x_sf = None
        token_selected_experts = None
        token_final_scales = None
        x_row = x.shape[0]
        x_col = x.shape[1]
        token_count = x.shape[0]
        alltoall_info = None
        # Determine if this is first/last call (TRTLLMGenFusedMoE doesn't use chunking)
        is_first_call = self.repeat_idx == 0
        is_last_call = self.repeat_idx == self.repeat_count - 1

        if post_quant_comm:
            # Start GPU stage for first call
            self._load_balancer_start_wait_gpu_stage(is_first_call)
            token_selected_experts, token_final_scales = self.routing_method.apply(
                router_logits)
            token_selected_experts = token_selected_experts.to(torch.int32)
            if token_final_scales is not None:
                token_final_scales = token_final_scales.to(torch.bfloat16)

            self._load_balancer_done_wait_gpu_stage(is_first_call)

            ignore_allreduce = self.enable_alltoall and self.alltoall_method_type == AlltoallMethodType.MNNVL and self.moe_alltoall_backend == "mnnvllatency"
            self._load_balancer_update_statistic(
                token_selected_experts,
                is_first_call,
                is_last_call,
                ignore_allreduce=ignore_allreduce)

            # Route tokens to slots
            token_selected_slots = self._load_balancer_route(
                token_selected_experts, self.use_dp)

            # Update expert statistics
            ExpertStatistic.set_layer(self.layer_idx)
            ExpertStatistic.maybe_add_info(self.num_slots, token_selected_slots)

            # Use routed slots for subsequent processing
            token_selected_experts = token_selected_slots

            x, x_sf, x_row, x_col = self._quantize_for_post_quant_comm(x)

        if self.enable_alltoall:
            assert all_rank_num_tokens is not None, "all_rank_num_tokens required for alltoall"

            runtime_max_tokens_per_rank = max(
                all_rank_num_tokens) if all_rank_num_tokens else token_count

            if token_final_scales is None:
                token_final_scales = torch.ones_like(token_selected_experts,
                                                     dtype=torch.float32)
            else:
                token_final_scales = token_final_scales.to(torch.float32)

            if self.moe_alltoall_backend == "mnnvllatency":
                assert self.alltoall_prepare_workspace is not None, "alltoall_prepare_workspace should be initialized"
                if is_last_call:
                    loadbalancer_local_statistic_info = self._load_balancer_get_local_statistic_tensor(
                    )
                else:
                    loadbalancer_local_statistic_info = None
                alltoall_info, gathered_loadbalancer_local_statistic_info = MnnvlMoe.mnnvl_moe_alltoallv_prepare_without_allgather(
                    token_selected_experts,
                    loadbalancer_local_statistic_info,
                    self.alltoall_prepare_workspace,
                    runtime_max_tokens_per_rank,
                    self.ep_rank,
                    self.ep_size,
                    self.num_experts,
                    self.num_slots,
                    top_k,
                )
                if gathered_loadbalancer_local_statistic_info is not None:
                    gathered_loadbalancer_local_statistic_info = gathered_loadbalancer_local_statistic_info.view(
                        (self.mapping.moe_ep_size, self.num_experts))
                    self._load_balancer_update_statistic_with_gathered_statistic(
                        gathered_loadbalancer_local_statistic_info)

                if x_sf is not None:
                    x_sf = x_sf.view(x_row,
                                     ceil_div(x_col, self.scaling_vector_size))

                x, x_sf, token_selected_experts, token_final_scales = MnnvlMoe.mnnvl_moe_alltoallv(
                    [x, x_sf, token_selected_experts, token_final_scales],
                    alltoall_info,
                    self.alltoall_workspace,
                    self.ep_rank,
                    self.ep_size,
                )

                torch.ops.trtllm.memset_expert_ids(
                    token_selected_experts,
                    alltoall_info.recv_rank_count_cumsum,
                    runtime_max_tokens_per_rank,
                    top_k,
                    -1,  # Caution: TRTLLM-Gen uses -1 as invalid token expert id
                    self.ep_size,
                )

                if x_sf is not None:
                    x_sf = x_sf.flatten()

                if token_final_scales is not None:
                    token_final_scales = token_final_scales.to(torch.bfloat16)
            elif self.moe_alltoall_backend == "mnnvlthroughput":
                if x_sf is not None:
                    x_sf = x_sf.view(x_row,
                                     ceil_div(x_col, self.scaling_vector_size))

                payloads = []
                payloads.append(x)
                if x_sf is not None:
                    payloads.append(x_sf)
                    expert_id_payload_index = 2
                else:
                    expert_id_payload_index = 1
                payloads.append(token_selected_experts)
                payloads.append(token_final_scales)

                recv_tensors = self.moe_a2a.dispatch(
                    token_selected_experts,
                    payloads,
                    runtime_max_tokens_per_rank,
                    invalid_token_expert_id=
                    -1,  # Caution: TRTLLM-Gen uses -1 as invalid token expert id
                    expert_id_payload_index=expert_id_payload_index,
                )

                if x_sf is not None:
                    x_recv, x_sf_recv, token_selected_experts_recv, token_final_scales_recv = recv_tensors
                    x_sf = x_sf_recv.view(-1, x_sf_recv.shape[-1])
                else:
                    x_recv, token_selected_experts_recv, token_final_scales_recv = recv_tensors
                x = x_recv.view(-1, x_recv.shape[-1])
                token_selected_experts = token_selected_experts_recv.view(
                    -1, token_selected_experts_recv.shape[-1])
                token_final_scales = token_final_scales_recv.view(
                    -1, token_final_scales_recv.shape[-1])

                if x_sf is not None:
                    x_sf = x_sf.flatten()

                if token_final_scales is not None:
                    token_final_scales = token_final_scales.to(torch.bfloat16)
            else:
                raise ValueError(
                    f"Unsupported moe alltoall backend: {self.moe_alltoall_backend}"
                )

        elif run_post_quant_allgather:
            if x_sf is not None:
                x_sf = x_sf.view(x_row, ceil_div(x_col,
                                                 self.scaling_vector_size))
                assert len(
                    x_sf.shape
                ) == 2, "The hidden states scaling factor should be 2D tensor before allgather"
            x, x_sf, token_selected_experts, token_final_scales = allgather(
                [x, x_sf, token_selected_experts, token_final_scales],
                self.mapping,
                dim=0,
                sizes=None if use_dp_padding else all_rank_num_tokens)
            if x_sf is not None:
                x_sf = x_sf.flatten()

        router_logits_arg = router_logits if not post_quant_comm else None
        routing_bias_arg = routing_bias if not post_quant_comm else None

        moe_output: Optional[torch.Tensor] = None
        use_workspace_output = False
        # TODO: use_workspace_output only supports w4a8_mxfp4_mxfp8 (gpt-oss) for now
        if self.enable_alltoall and self.moe_alltoall_backend == "mnnvlthroughput" and self.has_w4a8_mxfp4_mxfp8:
            moe_output = self.moe_a2a.get_combine_payload_tensor_in_workspace(
                runtime_max_tokens_per_rank, self.hidden_size, torch.bfloat16)
            use_workspace_output = True

        # TODO: since routing kernel is integrated into moe_runner for fp8,
        #       here we just route the I/Os for moe_runner
        if self.has_deepseek_fp8_block_scales:
            assert do_finalize, "fp8_block_scale_moe_runner does not support do_finalize=False"
            x_val, x_scale = torch.ops.trtllm.fp8_quantize_1x128(x)

            final_hidden_states = torch.ops.trtllm.fp8_block_scale_moe_runner(
                router_logits_arg,
                routing_bias_arg,
                x_val,
                x_scale,
                self.w3_w1_weight,
                self.w3_w1_weight_scaling_factor,
                self.w2_weight,
                self.w2_weight_scaling_factor,
                self.num_slots,
                top_k,
                n_group,
                topk_group,
                self.intermediate_size_per_partition,
                self.
                slot_start,  # local_expert_start;  use ep_rank if stride!=1
                self.expert_size_per_partition,  # local_expert_size
                routed_scaling_factor,
                self.routing_method.routing_method_type,
                topk_weights=token_final_scales,
                topk_ids=token_selected_experts,
            )
        elif self.has_nvfp4:
            scale_factor_use_ue8m0 = False
            is_scale_factor_swizzled = False  # use linear layout here

            if not post_quant_comm:
                hidden_states_fp4, hidden_states_scale_linear_fp4 = (
                    torch.ops.trtllm.fp4_quantize(
                        x,
                        self.fc31_input_scale,
                        self.scaling_vector_size,
                        scale_factor_use_ue8m0,
                        is_scale_factor_swizzled,
                    ))
            else:
                hidden_states_fp4, hidden_states_scale_linear_fp4 = x, x_sf

            outputs = torch.ops.trtllm.fp4_block_scale_moe_runner(
                router_logits_arg,
                routing_bias_arg,
                hidden_states_fp4,
                hidden_states_scale_linear_fp4.view(torch.float8_e4m3fn),
                self.w3_w1_weight,
                self.w3_w1_weight_scale.view(torch.float8_e4m3fn),
                self.w2_weight,
                self.w2_weight_scale.view(torch.float8_e4m3fn),
                self.fc31_scale_c.data,
                self.fc31_alpha.data,
                self.fc2_alpha.data,
                self.num_slots,
                top_k,
                n_group,
                topk_group,
                self.intermediate_size_per_partition,
                self.
                slot_start,  # local_expert_start;  use ep_rank if stride!=1
                self.expert_size_per_partition,  # local_expert_size
                routed_scaling_factor,
                self.routing_method.routing_method_type,
                do_finalize=do_finalize,
                topk_weights=token_final_scales,
                topk_ids=token_selected_experts,
            )

            if not do_finalize:
                assert not self.reduce_results, "reduce_results must be False when do_finalize is False"
                return outputs
            else:
                final_hidden_states = outputs[0]
        elif self.has_w4a16_mxfp4:
            assert x.dtype == torch.bfloat16
            if not post_quant_comm:
                pad_size = self.w3_w1_weight.shape[-1] * 2 - x.shape[-1]
                x = torch.nn.functional.pad(x, (0, pad_size))
            else:
                x = x

            intermediate_size_per_partition_padded = self.w3_w1_weight.shape[
                -2] // 2
            final_hidden_states = torch.ops.trtllm.bf16_mxe2m1_block_scale_moe_runner(
                router_logits_arg,
                routing_bias_arg,
                x,
                self.w3_w1_weight,
                self.w3_w1_weight_scale,
                self.w3_w1_bias,
                self.swiglu_alpha,
                self.swiglu_beta,
                self.swiglu_limit,
                self.w2_weight,
                self.w2_weight_scale,
                self.w2_bias,
                self.num_slots,
                top_k,
                n_group,
                topk_group,
                intermediate_size_per_partition_padded,
                self.hidden_size,  # valid_hidden_size
                self.quant_method.
                intermediate_size_per_partition_lean,  # valid_intermediate_size
                self.
                slot_start,  # local_expert_start;  use ep_rank if stride!=1
                self.expert_size_per_partition,  # local_expert_size
                routed_scaling_factor,
                self.routing_method.routing_method_type,
                0,  # act_type
                token_final_scales,
                token_selected_experts,
            )
            final_hidden_states = final_hidden_states[:, :self.
                                                      hidden_size].contiguous()
        elif self.has_w4a8_nvfp4_fp8:

            if not post_quant_comm:
                hidden_states_fp8, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    x, 1.0 / self.fc31_input_scale)
            else:
                hidden_states_fp8 = x

            outputs = torch.ops.trtllm.fp8_fp4_block_scale_moe_runner(
                router_logits_arg,
                routing_bias_arg,
                hidden_states_fp8,
                self.w3_w1_weight,
                self.w3_w1_weight_scale.view(torch.float8_e4m3fn),
                self.w2_weight,
                self.w2_weight_scale.view(torch.float8_e4m3fn),
                self.fc31_scale_c.data,
                self.fc31_alpha.data,
                self.fc2_alpha.data,
                self.num_slots,
                top_k,
                n_group,
                topk_group,
                self.intermediate_size_per_partition,
                self.
                slot_start,  # local_expert_start;  use ep_rank if stride!=1
                self.expert_size_per_partition,  # local_expert_size
                routed_scaling_factor,
                self.routing_method.routing_method_type,
                do_finalize=do_finalize,
                act_type=0,
                topk_ids=token_selected_experts,
                topk_weights=token_final_scales,
            )

            if not do_finalize:
                assert not self.reduce_results, "reduce_results must be False when do_finalize is False"
                return outputs
            else:
                final_hidden_states = outputs[0]
        elif self.has_w4a8_mxfp4_fp8:
            pad_size = self.w3_w1_weight.shape[-1] * 2 - x.shape[-1]
            if not post_quant_comm:
                x = torch.nn.functional.pad(x, (0, pad_size))
                x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    x, self.fc31_input_gate_dequant[0])
            else:
                x = x
            intermediate_size_per_partition_padded = self.w3_w1_weight.shape[
                -2] // 2

            final_hidden_states = torch.ops.trtllm.e4m3_mxe2m1_block_scale_moe_runner(
                router_logits_arg,
                routing_bias_arg,
                x,
                self.w3_w1_weight,
                self.w3_w1_weight_scale,
                self.w3_w1_bias,
                self.swiglu_alpha,
                self.swiglu_beta,
                self.swiglu_limit,
                self.w2_weight,
                self.w2_weight_scale,
                self.w2_bias,
                self.fc31_input_dequant,  # output1_scales_scalar
                self.fc31_input_gate_dequant,  # output1_scales_gate_scalar
                self.fc2_input_dequant,  # output2_scales_scalar
                self.num_slots,
                top_k,
                n_group,
                topk_group,
                intermediate_size_per_partition_padded,
                self.hidden_size,  # valid_hidden_size_per_partition
                self.quant_method.
                intermediate_size_per_partition_lean,  # valid_intermediate_size_per_partition
                self.
                slot_start,  # local_expert_start;  use ep_rank if stride!=1
                self.expert_size_per_partition,  # local_expert_size
                routed_scaling_factor,
                self.routing_method.routing_method_type,
                0,  # act_type
                token_final_scales,
                token_selected_experts,
            )
            final_hidden_states = final_hidden_states[:, :self.
                                                      hidden_size].contiguous()
        elif self.has_w4a8_mxfp4_mxfp8:
            if not post_quant_comm:
                # TRTLLM-Gen uses linear SF layout for the mxfp8 input.
                mxfp8_x, sf = torch.ops.trtllm.mxfp8_quantize(
                    x,
                    False,
                    alignment=self.quant_method.input_hidden_alignment)
            else:
                mxfp8_x, sf = x, x_sf

            intermediate_size_per_partition_padded = self.w3_w1_weight.shape[
                -2] // 2

            final_hidden_states = torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner(
                router_logits_arg,
                routing_bias_arg,
                mxfp8_x,
                sf,
                self.w3_w1_weight,
                self.w3_w1_weight_scale,
                self.w3_w1_bias,
                self.swiglu_alpha,
                self.swiglu_beta,
                self.swiglu_limit,
                self.w2_weight,
                self.w2_weight_scale,
                self.w2_bias,
                self.num_slots,
                top_k,
                n_group,
                topk_group,
                intermediate_size_per_partition_padded,
                self.hidden_size,  # valid_hidden_size
                self.quant_method.
                intermediate_size_per_partition_lean,  # valid_intermediate_size
                self.
                slot_start,  # local_expert_start;  use ep_rank if stride!=1
                self.expert_size_per_partition,  # local_expert_size
                routed_scaling_factor,
                self.routing_method.routing_method_type,
                0,  # act_type
                token_final_scales,
                token_selected_experts,
                output=moe_output,
            )
        else:
            raise NotImplementedError(
                "TRTLLMGenFusedMoE only supports fp8_block_scaling, nvfp4, w4a16_mxfp4, w4a8_mxfp4_mxfp8 and w4a8_mxfp4_fp8 dtypes."
            )

        # Handle load balancer CPU stage if needed
        self._load_balancer_start_set_cpu_stage(is_last_call)

        # Combine results if using alltoall
        if self.enable_alltoall:
            if self.moe_alltoall_backend == "mnnvllatency":
                if alltoall_info is not None:
                    final_hidden_states = MnnvlMoe.mnnvl_moe_alltoallv_combine(
                        final_hidden_states,
                        alltoall_info,
                        self.alltoall_workspace,
                        ep_rank=self.ep_rank,
                        ep_size=self.ep_size,
                        top_k=top_k,
                        use_low_precision_combine=self.
                        use_low_precision_combine,
                        token_count=token_count,
                    )
            elif self.moe_alltoall_backend == "mnnvlthroughput":
                # If use_workspace_output=True, the MoE result is already in workspace
                # Otherwise, we need to reshape and pass it
                if use_workspace_output:
                    # Workspace payload is returned as 2D [ep_size * max_tokens, hidden]; reshape to 3D.
                    hidden = final_hidden_states.shape[-1]
                    payload = moe_output.view(self.ep_size,
                                              runtime_max_tokens_per_rank,
                                              hidden)
                    final_hidden_states = self.moe_a2a.combine(
                        payload,
                        runtime_max_tokens_per_rank,
                        payload_in_workspace=True)
                else:
                    hidden = final_hidden_states.shape[-1]
                    payload = final_hidden_states.view(
                        self.ep_size, runtime_max_tokens_per_rank, hidden)
                    final_hidden_states = self.moe_a2a.combine(
                        payload,
                        runtime_max_tokens_per_rank,
                        payload_in_workspace=False)
            else:
                raise ValueError(
                    f"Unsupported moe alltoall backend: {self.moe_alltoall_backend}"
                )

        final_hidden_states = self.reducescatter_or_allreduce(
            final_hidden_states,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
        )

        self._load_balancer_done_set_cpu_stage(is_last_call)

        if use_dp_padding:
            rank = self.parallel_rank
            final_hidden_states = final_hidden_states[:
                                                      all_rank_num_tokens[rank]]

        # Update repeat index for load balancer
        if self.layer_load_balancer:
            self.repeat_idx = 0 if self.repeat_idx == self.repeat_count - 1 else self.repeat_idx + 1

        return final_hidden_states

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
        if do_finalize:
            # TRTLLMGenFusedMoE only supports bfloat16 output
            return super().forward_fake(x,
                                        router_logits,
                                        do_finalize=do_finalize,
                                        output_dtype=torch.bfloat16,
                                        all_rank_num_tokens=all_rank_num_tokens,
                                        use_dp_padding=use_dp_padding,
                                        **kwargs)
        else:
            is_deepseek_v3_routing = isinstance(self.routing_method,
                                                DeepSeekV3MoeRoutingMethod)
            top_k = self.routing_method.routing_impl.top_k if is_deepseek_v3_routing else self.routing_method.top_k
            routing_bias = self.routing_method.e_score_correction_bias if is_deepseek_v3_routing else None
            return fp4_block_scale_fake_output_without_finalize(
                x,
                self.num_experts,
                top_k,
                routing_bias,
            )
