import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...autotuner import (AutoTuner, ConstraintSpec, DynamicTensorSpec,
                          OptimizationProfile, TunableRunner, TuningConfig)
from ...custom_ops.cute_dsl_custom_ops import (
    GroupedGemmInputsHelper,
    Sm100BlockScaledContiguousGatherGroupedGemmSwigluFusionRunner,
    Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner,
    Sm100BlockScaledContiguousGroupedGemmRunner,
    Sm100BlockScaledContiguousGroupedGemmSwigluFusionRunner)
from ...distributed import allgather
from ...model_config import ModelConfig
from ...utils import (AuxStreamType, EventType, Fp4QuantizedTensor,
                      get_last_power_of_2_num_tokens_buckets,
                      last_positive_power_of_2)
from .fused_moe_cutlass import CutlassFusedMoE
from .interface import AlltoallMethodType
from .quantization import MoEWeightLoadingMode, NVFP4CuteDslFusedMoEMethod
from .routing import BaseMoeRoutingMethod


@torch.compile(options={"max-autotune": True})
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

    # Note: we have different output scale shape for fp8_quantize_1x128, so we need to handle it differently for sm100 and other archs.
    if is_sm_100f():
        input_scale_tmp = a_sf.permute(1, 0).as_strided((m, w_k, 1),
                                                        (1, m, m * w_k))
    else:
        m_padded = (m + 3) // 4 * 4
        input_scale_tmp = a_sf[0:m_padded * w_k]
        input_scale_tmp = input_scale_tmp.reshape(-1, m_padded)
        input_scale_tmp = input_scale_tmp[:w_k, :m].contiguous().permute(1, 0)
        input_scale_tmp = input_scale_tmp.as_strided((m, w_k, 1),
                                                     (1, m, m * w_k))

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


def cute_dsl_nvfp4_grouped_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    tile_idx_to_group_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    tile_size: int,
    output_dtype: torch.dtype,
    scaling_vector_size: int = 16,
):
    assert a.dtype == torch.float4_e2m1fn_x2
    assert a.dim() == 2
    assert b.dtype == torch.float4_e2m1fn_x2
    assert b.dim() == 3
    assert a_sf.dtype == torch.uint8
    assert a_sf.dim() == 1
    assert b_sf.dtype == torch.uint8
    assert b_sf.dim() == 3
    assert alpha.dtype == torch.float32
    assert alpha.dim() == 1

    m, k = a.size(0), a.size(1) * 2
    l, n = b.size(0), b.size(1)
    scale_k = k // scaling_vector_size
    assert m % tile_size == 0
    assert k % (scaling_vector_size * 4) == 0
    assert b.size(2) * 2 == k
    assert a_sf.size(0) == m * scale_k
    assert b_sf.size(0) == l
    assert b_sf.size(1) == n
    assert b_sf.size(2) == scale_k
    assert alpha.size(0) == l

    num_tiles = m // tile_size
    assert tile_idx_to_group_idx.dtype == torch.int32
    assert tile_idx_to_group_idx.size() == (num_tiles, )
    assert num_non_exiting_tiles.dtype == torch.int32
    assert num_non_exiting_tiles.size() == (1, )

    num_tiles_per_expert = torch.bincount(
        tile_idx_to_group_idx[:num_non_exiting_tiles[0].item()], minlength=l)
    offsets = [0] + num_tiles_per_expert.cumsum(dim=0).tolist()

    ref = torch.empty(m, n, dtype=output_dtype, device="cuda")
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        if end <= start:
            continue
        a_sliced = a[start * tile_size:end * tile_size]
        a_sf_sliced = a_sf[start * tile_size * k // scaling_vector_size:end *
                           tile_size * k // scaling_vector_size]
        ref[start * tile_size:end * tile_size] = torch.ops.trtllm.nvfp4_gemm(
            a_sliced.view(torch.uint8), b[i].view(torch.uint8), a_sf_sliced,
            b_sf[i], alpha[i], output_dtype)

    return ref


class CuteDslFusedMoENvfp4InputsHelper(GroupedGemmInputsHelper):

    def __init__(self, num_experts: int, top_k: int, num_local_experts: int,
                 local_expert_offset: int):
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset

    def infer_shape_num_tokens(self, input_shapes: List[torch.Size]) -> int:
        return input_shapes[0][0]

    def inputs_pre_hook(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        x, token_selected_experts, *others = inputs
        num_tokens = token_selected_experts.size(0)
        num_tokens_per_expert = self.generate_num_tokens_per_expert(
            num_tokens, approx_max_load=True)

        new_token_selected_experts = []
        for i, curr_num_tokens in enumerate(num_tokens_per_expert,
                                            start=self.local_expert_offset):
            new_token_selected_experts.extend([i] * curr_num_tokens)
        new_token_selected_experts = new_token_selected_experts + [-1] * (
            num_tokens * self.top_k - len(new_token_selected_experts))
        new_token_selected_experts = torch.tensor(
            new_token_selected_experts,
            dtype=token_selected_experts.dtype,
            device=token_selected_experts.device)
        new_token_selected_experts = new_token_selected_experts.view(
            self.top_k, num_tokens).transpose(0, 1).contiguous()
        return x, new_token_selected_experts, *others


class CuteDslFusedMoENvfp4Runner(TunableRunner):
    tuning_config_cache = dict()

    def __init__(self,
                 forward_impl: Callable,
                 num_experts: int,
                 top_k: int,
                 num_local_experts: int,
                 local_expert_offset: int,
                 enable_finalize_fusion: bool = True,
                 enable_alltoall: bool = False,
                 output_dtype: torch.dtype = torch.bfloat16,
                 scaling_vector_size: int = 16):
        super().__init__()
        self.forward_impl = forward_impl
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.enable_finalize_fusion = enable_finalize_fusion
        self.enable_alltoall = enable_alltoall

        assert output_dtype == torch.bfloat16
        self.output_dtype = output_dtype
        self.scaling_vector_size = scaling_vector_size

    def unique_id(self):
        return (
            self.num_experts,
            self.top_k,
            self.num_local_experts,
            self.local_expert_offset,
            self.enable_finalize_fusion,
            self.enable_alltoall,
            self.output_dtype,
            self.scaling_vector_size,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> List[int]:
        return [128, 256]

    def get_tuning_config(self) -> TuningConfig:
        key = self.unique_id()
        if key not in self.__class__.tuning_config_cache:
            helper = CuteDslFusedMoENvfp4InputsHelper(self.num_experts,
                                                      self.top_k,
                                                      self.num_local_experts,
                                                      self.local_expert_offset)
            self.__class__.tuning_config_cache[key] = TuningConfig(
                dynamic_tensor_specs=(DynamicTensorSpec(
                    0, 0, get_last_power_of_2_num_tokens_buckets,
                    last_positive_power_of_2), ),
                constraint_specs=(ConstraintSpec(1, 0,
                                                 helper.infer_shape_num_tokens),
                                  ConstraintSpec(2, 0,
                                                 helper.infer_shape_num_tokens),
                                  ConstraintSpec(3, 0,
                                                 helper.infer_shape_num_tokens),
                                  ConstraintSpec(
                                      4, 0, helper.infer_shape_num_tokens)),
                inputs_pre_hook=helper.inputs_pre_hook,
                use_cold_l2_cache=True,
            )
        return self.__class__.tuning_config_cache[key]

    def forward(self, inputs: List[torch.Tensor],
                tactic: Optional[int]) -> torch.Tensor:
        if isinstance(tactic, int) and tactic > 0:
            tile_size = tactic
        else:
            tile_size = 128
        return self.forward_impl(*inputs,
                                 enable_alltoall=self.enable_alltoall,
                                 tile_size=tile_size)

    @AutoTuner.TacticsCapture.register_runner_tactic_comb_checker
    @staticmethod
    def runner_tactic_comb_checker(
            comb: List[Tuple[TunableRunner, Any]]) -> bool:
        tile_size = None
        for runner, tactic in comb:
            if isinstance(runner, CuteDslFusedMoENvfp4Runner):
                tile_size = tactic
        if tile_size is None:
            return True

        for runner, tactic in comb:
            if isinstance(
                    runner,
                (Sm100BlockScaledContiguousGroupedGemmRunner,
                 Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner,
                 Sm100BlockScaledContiguousGroupedGemmSwigluFusionRunner,
                 Sm100BlockScaledContiguousGatherGroupedGemmSwigluFusionRunner
                 )):
                mma_tiler_mn, *_ = tactic
                if mma_tiler_mn[0] != tile_size:
                    return False
        return True


class CuteDslFusedMoE(CutlassFusedMoE):
    """CuteDSL flow of fused mixture of experts (MoE) Layer.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary CUDA streams for overlapping.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.
    """

    @classmethod
    def can_implement(
        cls,
        quant_algo: Optional[QuantAlgo],
        dtype_activation: torch.dtype = torch.bfloat16,
        gptoss_style: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if CuteDslFusedMoE can implement the given quantization algorithm.

        CuteDslFusedMoE supports:
        - NVFP4: SM in {100, 103}

        Does NOT support unquantized mode. Output dtype is hardcoded to bfloat16.
        Does NOT support gptoss_style (bias/swiglu with custom alpha/beta/limit).

        Args:
            quant_algo: The quantization algorithm to check (None for unquantized)
            dtype_activation: The activation input data type. Only bfloat16 is supported
                because output dtype is hardcoded to bfloat16 (input/output dtype must match).
            gptoss_style: Whether gptoss_style (bias/swiglu with custom alpha/beta/limit) is enabled.
                CuteDslFusedMoE does NOT support gptoss_style.

        Returns:
            Tuple[bool, Optional[str]]: (can_implement, skip_reason)
        """
        from .interface import _warn_and_return

        sm_version = get_sm_version()

        # CuteDslFusedMoE requires at least SM90
        if sm_version < 90:
            return _warn_and_return(
                f"CuteDslFusedMoE requires SM >= 90, got SM{sm_version}")

        # Check dtype_activation: output is hardcoded to bfloat16, so input must also be bfloat16
        # to maintain input/output dtype consistency
        if dtype_activation != torch.bfloat16:
            return _warn_and_return(
                f"CuteDslFusedMoE only supports bfloat16 activation (output is hardcoded to bfloat16), "
                f"got {dtype_activation}")

        # CuteDslFusedMoE does NOT support unquantized mode
        if quant_algo is None:
            return _warn_and_return(
                "CuteDslFusedMoE does not support unquantized mode")

        # CuteDslFusedMoE does NOT support gptoss_style
        if gptoss_style:
            return _warn_and_return(
                "CuteDslFusedMoE does not support gptoss_style (bias/swiglu with custom alpha/beta/limit)"
            )

        # NVFP4 - SM in {100, 103}
        if quant_algo == QuantAlgo.NVFP4:
            if sm_version not in {100, 103}:
                return _warn_and_return(
                    f"NVFP4 requires SM100 or SM103, got SM{sm_version}")
            return True, None

        return _warn_and_return(
            f"CuteDslFusedMoE does not support quant_algo={quant_algo}")

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
        apply_router_weight_on_input: bool = False,
        layer_idx: Optional[int] = None,
        init_load_balancer: bool = True,
        without_comm: bool = False,
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
            apply_router_weight_on_input=apply_router_weight_on_input,
            layer_idx=layer_idx,
            init_load_balancer=init_load_balancer,
            without_comm=without_comm,
        )
        if self.aux_stream_dict is None:
            self.aux_stream_dict = aux_stream_dict if aux_stream_dict is not None else {}
        if AuxStreamType.MoeOutputMemset not in self.aux_stream_dict:
            self.aux_stream_dict[
                AuxStreamType.MoeOutputMemset] = torch.cuda.Stream()
        if self.event_dict is None:
            self.event_dict = {}
        for key in [EventType.Main, EventType.MoeOutputMemset]:
            if key not in self.event_dict:
                self.event_dict[key] = torch.cuda.Event()

    def select_alltoall_method_type(self) -> AlltoallMethodType:
        return AlltoallMethodType.NotEnabled

    def _get_quant_method(self):
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True):
            if self.quant_config.layer_quant_mode.has_nvfp4():
                return NVFP4CuteDslFusedMoEMethod()
        return super()._get_quant_method()

    def supports_moe_output_in_alltoall_workspace(self):
        return self.has_nvfp4

    def quantize_input(self,
                       x: Union[torch.Tensor, Fp4QuantizedTensor],
                       post_quant_comm: bool = True):
        """Quantize inputs prior to post-communication (alltoall/allgather) or before MoE computation.

        Args:
            x: Input tensor to quantize
            post_quant_comm:
                If True, quantize for post-quant communication path.
                If False, quantize for non-communication path

        Returns: (x, x_sf) where x_sf is already reshaped to 2D if needed

        For quantization methods that produce scaling factors:
        - x_sf is reshaped from 1D to 2D: [num_elements] -> [batch_size, ceil_div(hidden_size, scaling_vector_size)]
        - The 2D shape is required for proper handling in alltoall/allgather operations
        - scaling_vector_size is typically the group size for block-wise quantization
        """
        x_sf = None
        if self.has_nvfp4:
            if isinstance(x, Fp4QuantizedTensor):
                assert not x.is_sf_swizzled, "Fp4QuantizedTensor should not be swizzled before communication"
                x_row = x.shape[0]
                x, x_sf = x.fp4_tensor, x.scaling_factor
            else:
                x_row = x.shape[0]
                x, x_sf = torch.ops.trtllm.fp4_quantize(
                    x, self.fc31_input_scale, self.scaling_vector_size, False,
                    False)
        elif self.has_deepseek_fp8_block_scales:
            # FP8 block scales doesn't support permutation of quantized inputs.
            # WAR: The quantization is in run_moe_fp8_block_scales.
            pass
        else:
            raise ValueError(
                f"{self.__class__.__name__} doesn't support quantization mode {self.quant_config.quant_mode}."
            )

        if x_sf is not None:
            x_sf = x_sf.view(x_row, -1)
        return x, x_sf

    def run_moe_nvfp4(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        moe_output: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
    ) -> torch.Tensor:
        assert self.has_nvfp4
        output_dtype = torch.bfloat16

        if moe_output is None:
            moe_output = torch.empty(
                (token_final_scales.size(0), self.hidden_size),
                dtype=output_dtype,
                device=x.device)
        else:
            assert moe_output.size() == (token_final_scales.size(0),
                                         self.hidden_size)
            assert moe_output.dtype == output_dtype

        tuner = AutoTuner.get()
        runner = CuteDslFusedMoENvfp4Runner(
            forward_impl=self.run_moe_nvfp4_impl,
            num_experts=self.num_slots,
            top_k=self.routing_method.experts_per_token,
            num_local_experts=self.expert_size_per_partition,
            local_expert_offset=self.slot_start,
            enable_finalize_fusion=self.use_fused_finalize,
            enable_alltoall=enable_alltoall,
        )

        inputs = [
            x, token_selected_experts, token_final_scales, x_sf, moe_output
        ]
        _, best_tactic = tuner.choose_one(
            "CuteDslFusedMoE::run_moe_nvfp4",
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        return runner(inputs, tactic=best_tactic)

    def run_moe_nvfp4_impl(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        x_sf: torch.Tensor,
        moe_output: torch.Tensor,
        enable_alltoall: bool = False,
        tile_size: int = 128,
    ) -> torch.Tensor:
        output_dtype = torch.bfloat16

        tile_idx_to_expert_idx, tile_idx_to_mn_limit, expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx, total_num_padded_tokens, num_non_exiting_tiles = torch.ops.trtllm.moe_sort(
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            num_experts=self.num_slots,
            top_k=self.routing_method.experts_per_token,
            local_expert_offset=self.slot_start,
            local_num_experts=self.expert_size_per_partition,
            tile_tokens_dim=tile_size,
        )

        if self.use_fused_finalize:
            self.event_dict[EventType.Main].record()
            moe_output.record_stream(
                self.aux_stream_dict[AuxStreamType.MoeOutputMemset])

        x, x_sf = torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_swiglu_blackwell(
            input=x.view(torch.float4_e2m1fn_x2),
            weight=self.w3_w1_weight.view(torch.float4_e2m1fn_x2),
            input_scale=x_sf.view(torch.uint8),
            weight_scale=self.quant_scales.fc1_weight_block.view(torch.uint8),
            alpha=self.quant_scales.fc1_global,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            global_sf=self.fc2_input_scale,
            num_experts=self.num_slots,
            top_k=self.routing_method.experts_per_token,
            num_local_experts=self.expert_size_per_partition,
            local_expert_offset=self.slot_start,
            tile_size=tile_size,
        )

        if self.use_fused_finalize:
            with torch.cuda.stream(
                    self.aux_stream_dict[AuxStreamType.MoeOutputMemset]):
                self.event_dict[EventType.Main].wait()
                torch.ops.trtllm.moe_output_memset_inplace(
                    input=moe_output,
                    tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                    num_non_exiting_tiles=num_non_exiting_tiles,
                    tile_tokens_dim=tile_size,
                    top_k=self.routing_method.experts_per_token,
                    ep_size=self.mapping.moe_ep_size,
                    enable_alltoall=enable_alltoall,
                )
                self.event_dict[EventType.MoeOutputMemset].record()
            self.event_dict[EventType.MoeOutputMemset].wait()

            torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell(
                input=x.view(torch.float4_e2m1fn_x2),
                weight=self.w2_weight.view(torch.float4_e2m1fn_x2),
                input_scale=x_sf.view(torch.uint8),
                weight_scale=self.quant_scales.fc2_weight_block.view(
                    torch.uint8),
                alpha=self.quant_scales.fc2_global,
                output=moe_output,
                tile_idx_to_group_idx=tile_idx_to_expert_idx,
                tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                token_final_scales=token_final_scales,
                num_experts=self.num_slots,
                top_k=self.routing_method.experts_per_token,
                num_local_experts=self.expert_size_per_partition,
                local_expert_offset=self.slot_start,
                tile_size=tile_size,
                output_dtype=output_dtype,
            )
        else:
            x = torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_blackwell(
                input=x.view(torch.float4_e2m1fn_x2),
                weight=self.w2_weight.view(torch.float4_e2m1fn_x2),
                input_scale=x_sf.view(torch.uint8),
                weight_scale=self.quant_scales.fc2_weight_block.view(
                    torch.uint8),
                alpha=self.quant_scales.fc2_global,
                tile_idx_to_group_idx=tile_idx_to_expert_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                num_experts=self.num_slots,
                top_k=self.routing_method.experts_per_token,
                num_local_experts=self.expert_size_per_partition,
                local_expert_offset=self.slot_start,
                tile_size=tile_size,
                output_dtype=output_dtype,
            )
            torch.ops.trtllm.moe_unpermute_inplace(
                permuted_input=x,
                output=moe_output,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                topk_scales=token_final_scales,
            )
        return moe_output

    def run_moe_fp8_block_scales(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
    ) -> torch.Tensor:
        assert self.has_deepseek_fp8_block_scales
        assert x_sf is None
        weight_dtype = self.w3_w1_weight.dtype

        (
            permuted_row_to_unpermuted_row,
            permuted_token_selected_experts,
            x,
            expert_first_token_offset,
            permuted_token_final_scales,
            unpermuted_row_to_permuted_row,
        ) = torch.ops.trtllm.moe_permute_op(
            x,
            token_selected_experts,
            token_final_scales,
            None,  # w3_w1_weight.view(weight_dtype),
            None,  # w2_weight.view(weight_dtype),
            None,  # quant_scales,
            input_sf=None,
            num_experts_on_rank=self.expert_size_per_partition,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=self.cluster_size,
            cluster_rank=self.cluster_rank,
            min_latency_mode=False,
            use_fp8_block_scaling=True,
        )
        x, x_sf = torch.ops.trtllm.fp8_quantize_1x128(x)
        x = cute_dsl_fp8_group_blockwise_gemm_ref(
            a=x,
            b=self.w3_w1_weight.view(weight_dtype),
            a_sf=x_sf,
            b_sf=self.quant_scales[0],
            offset_array=expert_first_token_offset,
        )
        x = swiglu_fused_moe(x)
        x, x_sf = torch.ops.trtllm.fp8_quantize_1x128(x)
        x = cute_dsl_fp8_group_blockwise_gemm_ref(
            a=x,
            b=self.w2_weight.view(weight_dtype),
            a_sf=x_sf,
            b_sf=self.quant_scales[1],
            offset_array=expert_first_token_offset,
        )
        x = torch.ops.trtllm.moe_finalize_scale_op(
            x,
            None,  # biases
            token_final_scales,
            unpermuted_row_to_permuted_row,
            permuted_row_to_unpermuted_row,
            token_selected_experts,
            expert_first_token_offset,
            enable_alltoall,
            token_final_scales.size(0),  # num_rows
            self.hidden_size,  # (possibly padded) hidden_size
            self.unpadded_hidden_size,  # original hidden size
            self.routing_method.top_k,
            self.expert_size_per_partition,  # num_experts_per_node
            self.tp_size,
            self.tp_rank,
            self.ep_size,
            self.ep_rank,
        )
        return x

    def run_moe(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        moe_output: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
    ) -> torch.Tensor:
        """
        Run MoE computation with CuteDSL backend.

        This method encapsulates the core MoE computation logic, handling different
        quantization schemes (fp8_block_scales and nvfp4).

        Args:
            # Standard MoE interface parameters:
            x: Input hidden states (may be pre-quantized)
            token_selected_experts: Expert IDs [num_tokens, top_k]. If EPLB is enabled,
                                    this represents expert slots [num_tokens, top_k] instead.
            token_final_scales: Final scaling factors for each token
            x_sf: Input scale factors (optional, for certain quantization schemes)
            moe_output: Pre-allocated MoE output buffer (optional, for NVLINK one-sided backend).
            enable_alltoall: Whether alltoall communication is enabled.

        Returns:
            final_hidden_states tensor.
        """
        if self.has_nvfp4:
            return self.run_moe_nvfp4(
                x=x,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                x_sf=x_sf,
                moe_output=moe_output,
                enable_alltoall=enable_alltoall)
        elif self.has_deepseek_fp8_block_scales:
            return self.run_moe_fp8_block_scales(
                x=x,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                x_sf=x_sf,
                enable_alltoall=enable_alltoall)
        else:
            raise ValueError(
                f"{self.__class__.__name__} doesn't support quantization mode {self.quant_config.quant_mode}."
            )

    def forward_chunk(
            self,
            x: Union[torch.Tensor, Fp4QuantizedTensor],
            router_logits: torch.Tensor,
            output_dtype: Optional[torch.dtype] = None,
            all_rank_num_tokens: Optional[List[int]] = None,
            use_dp_padding: Optional[bool] = None,
            repeating_info: tuple = (True, True),
    ) -> torch.Tensor:
        # Currently, the default path is that ConfigurableMoE calls CuteDslFusedMoE.run_moe.
        # This forward_chunk method is a reference implementation of the legacy path.
        # Apply routing
        token_selected_experts, token_final_scales = self.routing_method.apply(
            router_logits)
        assert token_selected_experts.shape[
            1] == self.routing_method.experts_per_token
        assert token_selected_experts.shape == token_final_scales.shape
        assert token_selected_experts.shape[0] == router_logits.shape[0]
        assert token_final_scales.dtype == torch.float32
        assert token_selected_experts.dtype == torch.int32

        x, x_sf = self.quantize_input(x)

        if self.use_dp and self.parallel_size > 1:
            x, x_sf, token_selected_experts, token_final_scales = allgather(
                [x, x_sf, token_selected_experts, token_final_scales],
                self.mapping,
                dim=0,
                sizes=None if use_dp_padding else all_rank_num_tokens)

        x = self.run_moe(x=x,
                         token_selected_experts=token_selected_experts,
                         token_final_scales=token_final_scales,
                         x_sf=x_sf,
                         enable_alltoall=False)
        return x
