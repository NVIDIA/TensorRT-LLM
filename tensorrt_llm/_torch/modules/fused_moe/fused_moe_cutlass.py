import os
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union

import torch

from tensorrt_llm._mnnvl_utils import MnnvlMemory, MnnvlMoe
from tensorrt_llm._torch.distributed.moe_alltoall import MoeAlltoAll
from tensorrt_llm.logger import logger

from ...distributed import allgather
from ...expert_statistic import ExpertStatistic
from ...model_config import ModelConfig
from ...utils import (ActivationType, AuxStreamType, EventType,
                      Fp4QuantizedTensor)
from .interface import AlltoallMethodType, MoE
from .quantization import UnquantizedFusedMoEMethod

# isort: off
from .quantization import (
    DeepSeekFP8BlockScalesFusedMoEMethod, FP8QDQFusedMoEMethod,
    MoEWeightLoadingMode, NVFP4CutlassFusedMoEMethod, UnquantizedFusedMoEMethod,
    INT8WoqPerChannelFusedMoEMethod, W4A8MXFP4FP8CutlassFusedMoEMethod,
    W4A8MXFP4MXFP8CutlassFusedMoEMethod, WFP4A16FusedMoEMethod,
    WInt4AFP8FusedMoEMethod)
# isort: on
from .routing import BaseMoeRoutingMethod


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
        init_load_balancer: bool = True,
        without_comm: bool = False,
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

        # When without_comm=True, skip communication initialization (ConfigurableMoE will handle it)
        if not without_comm:
            self.alltoall_method_type = self.select_alltoall_method_type()
            logger.info_once(
                f"{self.__class__.__name__} selects alltoall_method_type {self.alltoall_method_type!r}",
                key="alltoall_method_type")
            self.alltoall_workspace = None
            self.alltoall_prepare_workspace = None
            self.use_low_precision_combine = False
            if self.enable_alltoall:
                self.use_low_precision_combine = model_config.use_low_precision_moe_combine

                if self.alltoall_method_type == AlltoallMethodType.NVLinkTwoSided:
                    MnnvlMemory.initialize()
                    self.alltoall_workspace = MnnvlMoe.get_moe_workspaces(
                        model_config.mapping)
                    self.alltoall_prepare_workspace = MnnvlMoe.get_moe_prepare_workspace(
                        model_config.mapping)
                elif self.alltoall_method_type == AlltoallMethodType.NVLinkOneSided:
                    # Calculate required workspace size
                    ep_size = self.mapping.moe_ep_size
                    max_num_tokens = model_config.max_num_tokens
                    hidden_size = self.hidden_size
                    dtype = self.dtype or torch.float16

                    workspace_size = MoeAlltoAll.calculate_required_workspace_size(
                        ep_size, self.routing_method.experts_per_token,
                        max_num_tokens, hidden_size, dtype)

                    self.moe_a2a = MoeAlltoAll(
                        mapping=self.mapping,
                        max_num_tokens=model_config.max_num_tokens,
                        top_k=self.routing_method.experts_per_token,
                        num_experts=self.num_slots,
                        workspace_size_per_rank=workspace_size,
                    )
                elif self.alltoall_method_type == AlltoallMethodType.DeepEP or self.alltoall_method_type == AlltoallMethodType.DeepEPLowLatency:
                    raise NotImplementedError(
                        "DeepEP and DeepEPLowLatency are not supported for CutlassFusedMoE yet"
                    )
                else:
                    raise NotImplementedError(
                        f"Unsupported alltoall method type: {self.alltoall_method_type!r}"
                    )
        else:
            # When without_comm=True, set minimal attributes
            # Communication will be handled by parent wrapper (e.g., ConfigurableMoE)
            self.alltoall_method_type = AlltoallMethodType.NotEnabled
            self.alltoall_workspace = None
            self.alltoall_prepare_workspace = None
            self.use_low_precision_combine = False
            self.moe_a2a = None

        # If True, the router weight will be multiplied on the input rather than at the end of FC2
        self.apply_router_weight_on_input = apply_router_weight_on_input

        # Finalize fusion should be disabled if Lora is used.
        self.use_fused_finalize = not model_config.moe_disable_finalize_fusion and model_config.lora_config is None

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
                    | self.quant_config.quant_mode.is_weight_only()
                    | self.quant_config.quant_mode.has_w4a8_mxfp4_fp8()
                    | self.quant_config.quant_mode.has_w4a16_mxfp4()
                    | self.quant_config.quant_mode.has_w4a8_mxfp4_mxfp8()):
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

        # TODO: We found that NVLinkOneSided performs better than NCCL AllGather/ReduceScatter,
        # regardless of the relationship between EP size and topK. We favor NVLinkOneSided for now.
        # if not self.mapping.moe_ep_size > self.routing_method.experts_per_token:
        #     return AlltoallMethodType.NotEnabled
        return AlltoallMethodType.NVLinkOneSided

    @cached_property
    def enable_alltoall(self):
        """ enable_alltoall (bool): whether to enable alltoall instead of allgather/reducescatter
        """
        return self.alltoall_method_type != AlltoallMethodType.NotEnabled

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
                pad_size = self.hidden_size - x.shape[1]
                x = torch.nn.functional.pad(x, (0, pad_size))
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
            elif self.has_w4a8_mxfp4_mxfp8:
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

    def run_moe(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        x_sf: Optional[torch.Tensor] = None,
        is_sf_swizzled: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        tuner_num_tokens: Optional[int] = None,
        tuner_top_k: Optional[int] = None,
        moe_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run MoE computation with Cutlass backend.

        This method encapsulates the core MoE computation logic, handling different
        quantization schemes.

        Args:
            x: Input hidden states (may be pre-quantized)
            token_selected_experts: Expert IDs or expert slots [num_tokens, top_k]
                                   If EPLB is enabled, represents expert slots; otherwise expert IDs
            token_final_scales: Final scaling factors for each token
            x_sf: Input scale factors (optional, for certain quantization schemes)
            is_sf_swizzled: Whether scaling factors are swizzled
            output_dtype: Output data type (optional)
            tuner_num_tokens: Number of tokens for profiling tuner (optional)
            tuner_top_k: Top-k value for profiling tuner (optional)
            moe_output: Pre-allocated output buffer (optional)

        Returns:
            final_hidden_states: Output tensor from MoE computation
        """
        # Determine weight dtype based on quantization mode
        weight_dtype = self.w3_w1_weight.dtype
        if self.has_any_quant:
            if self.has_w4afp8:
                weight_dtype = torch.quint4x2
            elif self.has_w4a16_mxfp4:
                weight_dtype = torch.uint8

        final_hidden_states = torch.ops.trtllm.fused_moe(
            x,
            token_selected_experts,
            token_final_scales,
            self.w3_w1_weight.view(weight_dtype),
            self.w3_w1_bias,
            self.w2_weight.view(weight_dtype),
            self.w2_bias,
            output_dtype,
            quant_scales=self.quant_scales,
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
            enable_alltoall=self.enable_alltoall,
            use_deepseek_fp8_block_scale=self.has_deepseek_fp8_block_scales,
            use_w4_group_scaling=self.has_w4afp8 or self.has_w4a16_mxfp4,
            use_int8_woq_per_channel=self.has_int8_woq_per_channel,
            use_mxfp8_act_scaling=self.has_w4a8_mxfp4_mxfp8,
            min_latency_mode=False,
            use_fused_finalize=self.use_fused_finalize,
            tune_max_num_tokens=self.tune_max_num_tokens,
            tuner_num_tokens=tuner_num_tokens,
            tuner_top_k=tuner_top_k,
            activation_type=self.activation_type,
            unpadded_hidden_size=self.unpadded_hidden_size,
            out_tensor=moe_output,
        )
        # Custom op requires all inputs are in the same type.
        # Only in cutlass_min_latency_mode, the output is a list of tensors.
        # Otherwise, the output should be unpacked as a single tensor.
        final_hidden_states = final_hidden_states[0]

        return final_hidden_states

    def forward_chunk(
            self,
            x: Union[torch.Tensor, Fp4QuantizedTensor],
            router_logits: torch.Tensor,
            output_dtype: Optional[torch.dtype] = None,
            all_rank_num_tokens: Optional[List[int]] = None,
            use_dp_padding: Optional[bool] = None,
            repeating_info: tuple = (True, True),
    ) -> torch.Tensor:
        if isinstance(x, Fp4QuantizedTensor):
            assert output_dtype is not None
        else:
            output_dtype = x.dtype

        is_first_call, is_last_call = repeating_info

        self._load_balancer_start_wait_gpu_stage(is_first_call)

        # apply routing
        token_selected_experts, token_final_scales = self.routing_method.apply(
            router_logits)
        assert token_selected_experts.shape[
            1] == self.routing_method.experts_per_token
        assert token_selected_experts.shape == token_final_scales.shape
        assert token_selected_experts.shape[0] == router_logits.shape[0]
        assert token_final_scales.dtype == torch.float32
        assert token_selected_experts.dtype == torch.int32

        if self.layer_load_balancer:
            self._load_balancer_done_wait_gpu_stage(is_first_call)
            ignore_allreduce = self.alltoall_method_type == AlltoallMethodType.NVLinkTwoSided
            self._load_balancer_update_statistic(
                token_selected_experts,
                is_first_call,
                is_last_call,
                ignore_allreduce=ignore_allreduce)
            token_selected_slots = self._load_balancer_route(
                token_selected_experts, self.use_dp)
        else:
            token_selected_slots = token_selected_experts

        # If load balancer is disabled, the statistics are collected from expert IDs.
        # If load balancer is enabled, the statistics are collected from expert slot IDs.
        ExpertStatistic.set_layer(self.layer_idx)
        ExpertStatistic.maybe_add_info(self.num_slots, token_selected_slots)

        if self.apply_router_weight_on_input:
            assert x.dtype != torch.float8_e4m3fn, "Current workaround for apply_router_weight_on_input does not support fp8 input"
            x = x * token_final_scales.to(x.dtype)
            # TODO: remove this once we have correct fusedmoe kernel ready
            token_final_scales = None

        run_post_quant_allgather = self.use_dp and self.parallel_size > 1

        # Quantize inputs using extracted method
        # For post_quant_comm scenarios, x_sf will be reshaped to 2D inside quantize_input
        post_quant_comm = run_post_quant_allgather or self.enable_alltoall
        x, x_sf = self.quantize_input(x, post_quant_comm=post_quant_comm)

        # Prepare additional information for profiling in case padding is applied when using alltoall.
        # Only the non-alltoall case is considered for profiling in the warmup phase.
        # Therefore, to get the correct tactics during the actual inference, the inputs to the tuner should be the same as when not using alltoall.
        if self.enable_alltoall:
            if all_rank_num_tokens is not None:
                tuner_num_tokens = sum(all_rank_num_tokens)
            else:
                tuner_num_tokens = x.shape[0] * self.mapping.tp_size
            tuner_top_k = token_selected_slots.shape[1]
        else:
            tuner_num_tokens = None
            tuner_top_k = None

        # Alltoall or allgather for attention DP
        token_count = x.shape[0]
        alltoall_info = None  # Store for later combine
        is_sf_swizzled = True  # In case of post-quant communication, scaling factors will not be swizzled before communication, and swizzling after communication is merged into MoE.
        if self.enable_alltoall:
            assert all_rank_num_tokens is not None, "all_rank_num_tokens required for alltoall"
            # Prepare alltoall indices
            top_k = self.routing_method.experts_per_token
            runtime_max_tokens_per_rank = max(
                all_rank_num_tokens) if all_rank_num_tokens else token_count

            # Handle case where token_final_scales might be None (when apply_router_weight_on_input=True)
            if token_final_scales is None:
                token_final_scales = torch.ones_like(token_selected_slots,
                                                     dtype=torch.float32)

            if self.alltoall_method_type == AlltoallMethodType.NVLinkTwoSided:
                assert self.alltoall_prepare_workspace is not None, "alltoall_prepare_workspace should be initialized"
                if is_last_call:
                    loadbalancer_local_statistic_info = self._load_balancer_get_local_statistic_tensor(
                    )
                else:
                    loadbalancer_local_statistic_info = None
                alltoall_info, gathered_loadbalancer_local_statistic_info = MnnvlMoe.mnnvl_moe_alltoallv_prepare_without_allgather(
                    token_selected_slots, loadbalancer_local_statistic_info,
                    self.alltoall_prepare_workspace,
                    runtime_max_tokens_per_rank, self.ep_rank, self.ep_size,
                    self.num_experts, self.num_slots, top_k)
                if gathered_loadbalancer_local_statistic_info is not None:
                    gathered_loadbalancer_local_statistic_info = gathered_loadbalancer_local_statistic_info.view(
                        (self.mapping.moe_ep_size, self.num_experts))
                    self._load_balancer_update_statistic_with_gathered_statistic(
                        gathered_loadbalancer_local_statistic_info)

                # Dispatch x, x_sf, token_selected_slots, token_final_scales in one alltoall kernel
                x, x_sf, token_selected_slots, token_final_scales = MnnvlMoe.mnnvl_moe_alltoallv(
                    [x, x_sf, token_selected_slots, token_final_scales],
                    alltoall_info, self.alltoall_workspace, self.ep_rank,
                    self.ep_size)

                torch.ops.trtllm.memset_expert_ids(
                    token_selected_slots, alltoall_info.recv_rank_count_cumsum,
                    runtime_max_tokens_per_rank, top_k, self.num_slots,
                    self.ep_size)
            elif self.alltoall_method_type == AlltoallMethodType.NVLinkOneSided:
                # Python MoeAlltoAll path

                payloads = []
                payloads.append(x)
                if x_sf is not None:
                    payloads.append(x_sf)
                    expert_id_payload_index = 2
                else:
                    expert_id_payload_index = 1
                payloads.append(token_selected_slots)
                payloads.append(token_final_scales)

                recv_tensors = self.moe_a2a.dispatch(
                    token_selected_slots,
                    payloads,
                    runtime_max_tokens_per_rank,
                    invalid_token_expert_id=self.
                    num_slots,  # Caution: Cutlass MoE uses num_slots as invalid token expert id
                    expert_id_payload_index=expert_id_payload_index,
                )

                if x_sf is not None:
                    x_recv, x_sf_recv, token_selected_slots_recv, token_final_scales_recv = recv_tensors
                    x_sf = x_sf_recv.view(-1, x_sf_recv.shape[-1])
                else:
                    x_recv, token_selected_slots_recv, token_final_scales_recv = recv_tensors
                x = x_recv.view(-1, x_recv.shape[-1])
                token_selected_slots = token_selected_slots_recv.view(
                    -1, token_selected_slots_recv.shape[-1])
                token_final_scales = token_final_scales_recv.view(
                    -1, token_final_scales_recv.shape[-1])
            else:
                raise ValueError(
                    f"Unsupported moe alltoall method type: {self.alltoall_method_type}"
                )

        elif run_post_quant_allgather:
            # Original allgather logic
            # x_sf is already 2D after quantize_input with post_quant_comm=True

            x, x_sf, token_selected_slots, token_final_scales = allgather(
                [x, x_sf, token_selected_slots, token_final_scales],
                self.mapping,
                dim=0,
                sizes=None if use_dp_padding else all_rank_num_tokens)

        # Optionally provide an output tensor to fused_moe so it writes directly to our buffer
        moe_output: Optional[torch.Tensor] = None
        if self.alltoall_method_type == AlltoallMethodType.NVLinkOneSided:
            # Retrieve a workspace-backed output tensor sized by runtime tokens
            runtime_max_tokens_per_rank = max(
                all_rank_num_tokens) if all_rank_num_tokens else x.shape[0]
            moe_output = self.moe_a2a.get_combine_payload_tensor_in_workspace(
                runtime_max_tokens_per_rank, self.unpadded_hidden_size,
                output_dtype)

        # Call extracted run_moe method
        final_hidden_states = self.run_moe(
            x=x,
            token_selected_experts=token_selected_slots,
            token_final_scales=token_final_scales,
            x_sf=x_sf,
            is_sf_swizzled=not post_quant_comm,
            output_dtype=output_dtype,
            tuner_num_tokens=tuner_num_tokens,
            tuner_top_k=tuner_top_k,
            moe_output=moe_output,
        )

        self._load_balancer_start_set_cpu_stage(is_last_call)

        # Combine results if using alltoall
        if self.enable_alltoall:
            if self.alltoall_method_type == AlltoallMethodType.NVLinkTwoSided:
                if alltoall_info is not None:
                    top_k = self.routing_method.experts_per_token
                    final_hidden_states = MnnvlMoe.mnnvl_moe_alltoallv_combine(
                        final_hidden_states,
                        alltoall_info,
                        self.alltoall_workspace,
                        ep_rank=self.ep_rank,
                        ep_size=self.ep_size,
                        top_k=top_k,
                        use_low_precision_combine=self.
                        use_low_precision_combine,
                        token_count=token_count)
            elif self.alltoall_method_type == AlltoallMethodType.NVLinkOneSided:
                output_hidden_size = final_hidden_states.shape[-1]
                runtime_max_tokens_per_rank = max(
                    all_rank_num_tokens) if all_rank_num_tokens else token_count
                final_hidden_states = self.moe_a2a.combine(
                    final_hidden_states.view(self.ep_size,
                                             runtime_max_tokens_per_rank,
                                             output_hidden_size),
                    runtime_max_tokens_per_rank,
                    payload_in_workspace=True)
            else:
                raise ValueError(
                    f"Unsupported moe alltoall method type: {self.alltoall_method_type}"
                )

        self._load_balancer_done_set_cpu_stage(is_last_call)

        return final_hidden_states

    def split_chunk(self, split_token_num: int, split_num_chunks: int):
        val_div = split_token_num // split_num_chunks
        val_mod = split_token_num % split_num_chunks
        split_chunk_size_list = [val_div + 1] * val_mod + [val_div] * (
            split_num_chunks - val_mod)
        return split_chunk_size_list

    def forward_impl(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool = True,  # used by other MoE backends
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert do_finalize, "CutlassFusedMoE does not support do_finalize=False"
        if self.use_dp and self.parallel_size > 1:
            assert all_rank_num_tokens is not None
            assert use_dp_padding is not None
            num_rows = sum(all_rank_num_tokens)
        else:
            num_rows = x.shape[0]

        if use_dp_padding:
            all_rank_num_tokens_padded = [max(all_rank_num_tokens)
                                          ] * len(all_rank_num_tokens)
            num_rows = sum(all_rank_num_tokens_padded)
        else:
            all_rank_num_tokens_padded = all_rank_num_tokens

        # in case of num_rows is larger than max_chunk_size, we need to split the input into multiple chunks
        num_chunks = (num_rows + self.moe_max_num_tokens -
                      1) // self.moe_max_num_tokens

        if num_chunks == 1:
            is_first_call = self.repeat_idx == 0
            is_last_call = self.repeat_idx == self.repeat_count - 1
            outputs = self.forward_chunk(
                x,
                router_logits,
                output_dtype,
                all_rank_num_tokens=all_rank_num_tokens_padded,
                use_dp_padding=use_dp_padding,
                repeating_info=(is_first_call, is_last_call))
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
                chunk_size_list = all_rank_chunk_size_list[self.parallel_rank]
            else:
                all_rank_num_tokens_list = [None] * num_chunks
                chunk_size_list = self.split_chunk(x.shape[0], num_chunks)

            x_list = x.split(chunk_size_list)
            router_logits_list = router_logits.split(chunk_size_list)

            self.event_dict[EventType.Main].record()
            with torch.cuda.stream(self.aux_stream):
                self.event_dict[EventType.Main].wait()

            def _forward_chunk(x_, router_logits_, idx):
                is_first_call = idx == 0 and self.repeat_idx == 0
                is_last_call = idx == num_chunks - 1 and self.repeat_idx == self.repeat_count - 1
                return self.forward_chunk(
                    x_,
                    router_logits_,
                    all_rank_num_tokens=all_rank_num_tokens_list[idx]
                    if self.use_dp else None,
                    use_dp_padding=use_dp_padding,
                    repeating_info=(is_first_call, is_last_call))

            def _reducescatter_or_allreduce(x_, idx):
                return self.reducescatter_or_allreduce(
                    x_,
                    all_rank_num_tokens=all_rank_num_tokens_list[idx],
                    use_dp_padding=use_dp_padding)

            outputs_list = []
            # Postpone reduce-scatter/all-reduce to the next iteration to achieve better overlap
            for idx_chunk, (x, router_logits) in enumerate(
                    zip(x_list, router_logits_list)):
                if not (self.alltoall_method_type
                        == AlltoallMethodType.NVLinkOneSided
                        or self.alltoall_method_type
                        == AlltoallMethodType.NVLinkTwoSided):
                    if idx_chunk % 2 == 0:
                        with torch.cuda.stream(self.aux_stream):
                            outputs = _forward_chunk(x, router_logits,
                                                     idx_chunk)
                        if idx_chunk > 0:
                            outputs_list[-1] = _reducescatter_or_allreduce(
                                outputs_list[-1], idx_chunk - 1)
                    else:
                        outputs = _forward_chunk(x, router_logits, idx_chunk)
                        with torch.cuda.stream(self.aux_stream):
                            outputs_list[-1] = _reducescatter_or_allreduce(
                                outputs_list[-1], idx_chunk - 1)
                else:
                    outputs = _forward_chunk(x, router_logits, idx_chunk)

                outputs_list.append(outputs)

            if not (self.alltoall_method_type
                    == AlltoallMethodType.NVLinkOneSided
                    or self.alltoall_method_type
                    == AlltoallMethodType.NVLinkTwoSided):
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

        if self.use_dp and self.parallel_size > 1:
            rank = self.parallel_rank
            outputs = outputs[:all_rank_num_tokens[rank]]
        self.repeat_idx = 0 if self.repeat_idx == self.repeat_count - 1 else self.repeat_idx + 1
        return outputs

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

        if not isinstance(self.quant_method, UnquantizedFusedMoEMethod):
            assert not allow_partial_loading, "Partial loading is not supported for quantized MoE now"
            self.quant_method.load_weights(self, weights,
                                           self.weight_loading_mode)
        else:
            self.quant_method.load_weights(
                self,
                weights,
                self.weight_loading_mode,
                allow_partial_loading=allow_partial_loading)

    def post_load_weights(self):
        self.quant_method.post_load_weights(self)
